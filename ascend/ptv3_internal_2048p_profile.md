# PTV3 2048 点内部性能拆分

测试设备：`192.168.7.101`，Ascend310P1
测试日期：2026-09-03

## 结论

当前 CPU-PTV3/NPU-head 基线中，不是 Generator 和 Discriminator 整体运行在
CPU，而是只有它们各自的完整 `object_encoder`，即 PTV3，运行在 CPU。PTV3
内部的 serialization、HashSparseConv、attention、MLP、pooling 和 projection
当前也全部运行在 CPU，不会根据算子是否受 NPU 支持而自动迁移。

Generator/Discriminator 的 gripper encoder、diffusion head、scheduler 张量计算、
sample encoder 和 discriminator head 运行在 NPU。SO(3) 路径中的 `sinc` 在当前
torch-npu 环境会回退 CPU。

默认 encoder 配置确实是 5 个 stage、`2/2/2/6/2` 个 block，总计 14 个 block。

2048 点时，单个 PTV3 生成 `[1, 512]` embedding 的典型耗时约为：

| PTV3 | 六段中位数合计 | 含前后传输中位数 |
|---|---:|---:|
| Generator object encoder | 2855.90 ms | 2857.87 ms |
| Discriminator object encoder | 2858.83 ms | 2860.49 ms |

前后 CPU/NPU 传输约 1 ms，不是瓶颈。Stage 2 是主要瓶颈，其中两个 CPU
attention 合计约 1.91-1.92 秒。

## 六段顶层耗时

以下使用 5 次无 hooks 测量的中位数；每段互斥。所有六段当前设备均为 CPU。

| 阶段 | Generator | Discriminator | 当前设备 | 说明 |
|---|---:|---:|---|---|
| 1. 点云格式转换 | 0.279 ms | 0.262 ms | CPU | reshape、`coord/feat/grid_size/offset` 字典 |
| 2. 空间序列化 | 61.448 ms | 60.408 ms | CPU | batch 构造、voxel coord、4 种编码、argsort/inverse |
| 3. Sparse Conv Embedding | 14.517 ms | 15.315 ms | CPU | k=5 HashSparseConv + BatchNorm + GELU |
| 4. Encoder Stage 0 | 78.722 ms | 79.714 ms | CPU | 2048 点，2 blocks，无 down pooling |
| 4. Encoder Stage 1 | 125.566 ms | 122.045 ms | CPU | 1964 点，down pooling + 2 blocks |
| 4. Encoder Stage 2 | **2018.562 ms** | **2019.003 ms** | CPU | 1562 点，down pooling + 2 blocks |
| 4. Encoder Stage 3 | 420.990 ms | 403.481 ms | CPU | 653 点，down pooling + 6 blocks |
| 4. Encoder Stage 4 | 148.151 ms | 144.509 ms | CPU | 172 点，down pooling + 2 blocks |
| 5. Global mean pooling | 2.113 ms | 2.114 ms | CPU | `scatter_reduce_(mean)` |
| 6. Linear projection | 0.090 ms | 0.085 ms | CPU | 当前实际为 `Identity`，因为 512 -> 512 |
| **六段合计** | **2855.897 ms** | **2858.829 ms** | CPU | 直接按每次六段之和统计 |

各行独立中位数相加与“六段合计”的中位数会有少量差异，这是中位数不可加导致，
不是遗漏阶段。

## 设备边界

边界传输不计入上述六段：

| 边界 | Generator | Discriminator | 设备方向 |
|---|---:|---:|---|
| kappa scale + points copy | 0.772 ms | 0.690 ms | NPU -> CPU |
| `[1,512]` embedding copy | 0.220 ms | 0.180 ms | CPU -> NPU |

当前基线的真实放置方式是把整个 PTV3 module 及其参数保留在 CPU；PyTorch 不会
只因为 `matmul/softmax` 支持 NPU 就自动把 attention 分派到 NPU。

## 点数变化

相同输入的有效点数如下：

| 位置 | 点数 | 通道数 | Attention heads | Patch |
|---|---:|---:|---:|---:|
| Embedding / Stage 0 | 2048 | 32 | 2 | 1024 |
| Stage 1 down 后 | 1964 | 64 | 4 | 1024 |
| Stage 2 down 后 | 1562 | 128 | 8 | 1024 |
| Stage 3 down 后 | 653 | 256 | 16 | 653 |
| Stage 4 down 后 | 172 | 512 | 32 | 172 |

Stage 2 的 1562 点会 padding 成两个 1024-token patch。其 8-head attention
logits shape 约为 `[2, 8, 1024, 1024]`，这是 CPU attention 出现约 1 秒/block
的直接原因。

## Encoder Stage 内部拆分

以下是 Generator 和 Discriminator 共 6 次详细 hooks 的平均结果。Hooks 会引入
少量开销，因此顶层 stage 耗时以前表为准；本表用于解释 stage 内组成。所有项目
当前均运行在 CPU。

| Stage | Down pooling | CPE HashSparseConv | Attention | FFN | Norm/residual/其他 | 诊断合计 |
|---:|---:|---:|---:|---:|---:|---:|
| 0, 2 blocks | 0.00 ms | 18.40 ms | 49.59 ms | 5.08 ms | 5.76 ms | 78.83 ms |
| 1, 2 blocks | 8.58 ms | 28.53 ms | 81.80 ms | 5.16 ms | 6.11 ms | 130.18 ms |
| 2, 2 blocks | 8.11 ms | 88.24 ms | **1915.13 ms** | 9.44 ms | 7.54 ms | 2028.46 ms |
| 3, 6 blocks | 8.83 ms | 163.32 ms | 201.67 ms | 36.47 ms | 21.39 ms | 431.68 ms |
| 4, 2 blocks | 7.48 ms | 84.92 ms | 39.74 ms | 12.47 ms | 6.85 ms | 151.46 ms |

这里的 CPE 只列每个 block 的 k=3 HashSparseConv。其后的 CPE linear/norm、
pre-attention norm、pre-FFN norm、残差和索引开销包含在“其他”中。

## Embedding 内部拆分

| 子阶段 | Generator | Discriminator | 当前设备 |
|---|---:|---:|---|
| k=5 HashSparseConv | 15.216 ms | 15.205 ms | CPU |
| BatchNorm | 0.242 ms | 0.241 ms | CPU |
| GELU | 0.294 ms | 0.431 ms | CPU |

HashSparseConv 并不是单个 dense convolution。它包含 int64 hash、sort、邻居坐标、
`searchsorted`、gather/mask 和最终 `einsum`。当前它们全部在 CPU。未来可以让 CPU
预计算 neighbor index、NPU 执行 gather/einsum，但当前代码没有该拆分。

## Attention 内部分析

CPU attention 的 stage 汇总如下：

| Stage | Attention 总耗时 | QKV linear | Softmax | Output projection | QK/AV matmul、索引及其他 |
|---:|---:|---:|---:|---:|---:|
| 0 | 49.59 ms | 1.53 ms | 15.79 ms | 1.49 ms | 30.79 ms |
| 1 | 81.80 ms | 1.91 ms | 33.22 ms | 1.80 ms | 44.87 ms |
| 2 | **1915.13 ms** | 3.00 ms | 65.26 ms | 2.29 ms | **1844.58 ms** |
| 3 | 201.67 ms | 12.33 ms | 71.09 ms | 9.09 ms | 109.17 ms |
| 4 | 39.74 ms | 4.54 ms | 4.69 ms | 2.77 ms | 27.74 ms |
| **14 blocks 合计** | **2287.93 ms** | - | - | - | - |

Stage 2 的主要问题不是 QKV Linear 或 Softmax，而是手写 attention 中的大矩阵
QK/AV 计算及相关 layout/index 操作。

## NPU Attention 可行性实测

将每个 stage 相同 shape 的 QKV、序列 gather、QK matmul、softmax、AV matmul和
projection 单独放到 310P1 FP32 测得：

| Stage | 每 block NPU 中位数 | 该 stage 全部 blocks 估算 | CPU attention |
|---:|---:|---:|---:|
| 0 | 2.266 ms | 4.531 ms | 49.59 ms |
| 1 | 3.140 ms | 6.279 ms | 81.80 ms |
| 2 | 5.189 ms | 10.378 ms | 1915.13 ms |
| 3 | 2.989 ms | 17.937 ms | 201.67 ms |
| 4 | 2.001 ms | 4.002 ms | 39.74 ms |
| **14 blocks 合计** | - | **43.127 ms** | **2287.93 ms** |

这证明 dense attention 数据路径可以在 310P1 上执行，而且 Stage 2 有很大的加速
空间。5 个 stage 的 NPU 输出均为 finite；相对同一 FP32 CPU 计算，最大绝对误差
为 `4.3e-5` 到 `9.9e-5`，最低 cosine 为 `0.99999994`。

但 `43.127 ms` 不是完整混合 PTV3 的预测延迟，因为它不包含：

- CPU serialization、pooling cluster 和 padding metadata 构造；
- 14 个 block 的 CPU/NPU 传输；
- CPE HashSparseConv；
- NPU shape 首次编译；
- 当前 310P 不具备的 fused attention 路径。

直接逐 block 把 feature 搬到 NPU 再搬回 CPU 不一定最优。合理架构是 CPU 一次性
预计算 serialization、每层 pooling map 和每个 CPE neighbor index，让 feature
stream 尽量常驻 NPU，再由 NPU 执行 gather/einsum、attention、norm、MLP 和 dense
projection。Global mean pooling 仍需 NPU 原生 segment reduction 或定制算子。

## 当前及建议设备拆分

| PTV3 子阶段 | 当前设备 | 310P 建议 |
|---|---|---|
| 点云格式化 | CPU | CPU，开销可忽略 |
| Morton/Hilbert serialization | CPU | CPU 预计算；int64 位移在当前 NPU 有硬错误 |
| Pooling topology、unique/sort | CPU | CPU 预计算 |
| Segment max/mean | CPU | 定制 NPU 实现；`scatter_reduce_` 当前会回退 CPU |
| Sparse neighbor hash/search | CPU | CPU 预计算 neighbor index |
| Sparse gather/einsum | CPU | NPU |
| Attention metadata | CPU | CPU 预计算 |
| QKV/QK/softmax/AV/projection | CPU | NPU，已实测可运行 |
| LayerNorm/BatchNorm/GELU/MLP | CPU | NPU |
| Global mean pooling | CPU | 定制或改写为 NPU 原生 reduction |
| Final projection | CPU Identity | 保持 Identity；无需优化 |

## 复现

CPU 内部分解脚本：

```bash
cd /root/Workspace/GraspgenX/GraspGenX
source /root/Workspace/IB_Robot-pi05-npu-core-eval/.shrc_local
source /root/Workspace/IB_Robot-pi05-npu-core-eval/install/setup.sh

python3 ascend/tools/profile_ptv3_stages.py \
  --point-count 2048 \
  --warmup 1 \
  --runs 5 \
  --detail-runs 3 \
  --cpu-threads 16 \
  --output ascend/results/ptv3_internal_2048p_cpu.json
```

文件：

```text
ascend/tools/profile_ptv3_stages.py
ascend/results/ptv3_internal_2048p_cpu.json
ascend/results/ptv3_attention_npu_2048p_fp32.json
```
