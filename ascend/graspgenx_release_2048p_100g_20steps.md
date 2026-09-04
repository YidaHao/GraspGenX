# GraspGenX Release Workload Baseline

测试日期：2026-09-03

## 测试结论

在 `192.168.7.101` 的 Ascend310P1 上，使用 2048 点、100 grasps、20
diffusion steps 时，当前 CPU-PTV3/NPU-head 混合流水线的稳态端到端基线为：

| 指标 | 耗时 |
|---|---:|
| 稳态均值 | 6572.00 ms |
| **稳态中位数** | **6545.53 ms** |
| P95 | 6723.36 ms |
| 最小值 | 6443.78 ms |
| 最大值 | 6746.50 ms |
| 样本数 | 5 |

端到端基线应使用没有细粒度同步的 `6545.53 ms` 中位数。逐阶段 profile
会在每个边界执行 NPU synchronize，因此只用于耗时归因，不作为请求延迟。

每个请求中的模型调用次数为：

| 模块 | 调用次数 |
|---|---:|
| Generator PTV3 | 1 |
| Generator diffusion head | 20 |
| Discriminator PTV3 | 1 |
| Discriminator head | 1 |

输出 shape 为 `[1, 100, 4, 4]` grasps 和 `[1, 100, 1]` confidence，所有
输出均为有限值。

## 测试口径

| 配置 | 值 |
|---|---|
| NPU | Ascend310P1，`npu:0` |
| PyTorch / torch-npu | 2.5.1 / 2.5.1 |
| 输入点数 | 2048 |
| Grasps | 100 |
| Diffusion steps | 20 |
| 精度 | FP32 |
| PTV3 设备 | CPU |
| Dense heads 设备 | NPU |
| CPU intra-op threads | 16 |
| 权重 | 随机初始化，完整 release shape |
| 单个 PTV3 参数量 | 38,660,640 |
| Grid size | 0.01 |
| Kappa | 3.27 |
| PTV3 flash attention | 关闭 |
| Serialization order shuffle | 关闭 |

随机权重不会改变网络 shape、点云拓扑、排序、pooling、attention 或 dense
kernel 的计算量，适合性能基线，但不能用于精度评估。

## 整体拆分

以下是 3 次带同步 profile 的均值：

| 阶段 | 均值 | 占 profile 总耗时 |
|---|---:|---:|
| Generator PTV3，CPU | 2893.08 ms | 41.94% |
| Generator 其他环节 | 718.07 ms | 10.41% |
| Discriminator PTV3，CPU | 3264.10 ms | 47.32% |
| Discriminator 其他环节 | 22.75 ms | 0.33% |
| **两个 PTV3 合计** | **6157.19 ms** | **89.26%** |
| **Profiled end-to-end** | **6898.01 ms** | **100%** |

Discriminator PTV3 的 3 次数据中有一次 `3942.08 ms` 的 CPU 抖动；其中位数
为 `2930.86 ms`。Generator PTV3 中位数为 `2919.48 ms`。按中位数估计，
两个 PTV3 的典型总耗时约为 `5850.34 ms`。

## Generator 明细

重复阶段均按一次完整请求累计。例如 diffusion head 的 `300.27 ms` 是 20 次
调用之和。

| Generator 阶段 | 每请求均值 | 调用数 | 平均单次 |
|---|---:|---:|---:|
| 点云乘 kappa | 0.48 ms | 1 | 0.48 ms |
| 点云 NPU 到 CPU | 0.37 ms | 1 | 0.37 ms |
| PTV3 格式化 | 0.41 ms | 1 | 0.41 ms |
| **PTV3 encoder，CPU** | **2893.08 ms** | 1 | **2893.08 ms** |
| Embedding CPU 到 NPU | 0.23 ms | 1 | 0.23 ms |
| Grasp batch mapping | 1.65 ms | 1 | 1.65 ms |
| Object embedding 扩展到 100 grasps | 0.30 ms | 1 | 0.30 ms |
| Gripper encoder | 2.37 ms | 1 | 2.37 ms |
| Gripper embedding 扩展 | 0.26 ms | 1 | 0.26 ms |
| Conditioning concat | 0.29 ms | 1 | 0.29 ms |
| Scheduler setup | 0.57 ms | 1 | 0.57 ms |
| Noise 初始化 | 0.19 ms | 1 | 0.19 ms |
| Likelihood 初始化 | 0.28 ms | 1 | 0.28 ms |
| Grasp history 初始化 | 0.25 ms | 1 | 0.25 ms |
| **Diffusion head，NPU** | **300.27 ms** | 20 | **15.01 ms** |
| Scheduler step | 99.00 ms | 20 | 4.95 ms |
| Likelihood | 99.41 ms | 19 | 5.23 ms |
| Sample update | 4.84 ms | 20 | 0.24 ms |
| SO(3) pose conversion | 187.93 ms | 20 | 9.40 ms |
| Grasp history store | 4.60 ms | 20 | 0.23 ms |
| **Generator total** | **3611.15 ms** | 1 | - |

20 步 diffusion 循环的六个主要阶段合计约 `695.05 ms`：

```text
diffusion head     300.27 ms
scheduler step      99.00 ms
likelihood           99.41 ms
sample update         4.84 ms
SO(3) conversion    187.93 ms
history store         4.60 ms
```

## Discriminator 明细

| Discriminator 阶段 | 每请求均值 |
|---|---:|
| 点云乘 kappa | 0.36 ms |
| 点云 NPU 到 CPU | 0.33 ms |
| PTV3 格式化 | 0.37 ms |
| **PTV3 encoder，CPU** | **3264.10 ms** |
| Embedding CPU 到 NPU | 0.28 ms |
| Grasp matrix 转 `r3_so3` | 12.26 ms |
| Grasp batch mapping | 1.35 ms |
| Sample encoder | 1.44 ms |
| Object embedding 扩展 | 0.29 ms |
| Gripper encoder | 2.15 ms |
| Gripper embedding 扩展 | 0.25 ms |
| Conditioning concat | 0.29 ms |
| Discriminator head，NPU | 1.89 ms |
| Sigmoid | 0.22 ms |
| **Discriminator total** | **3286.85 ms** |

## 冷启动

| 阶段 | 耗时 |
|---|---:|
| 模型构造、设备放置及首次同步 | 36.78 s |
| 第一个完整 warmup 请求 | 103.48 s |

Warmup 包含当前 batch shape 的 NPU 单算子编译，不属于稳态延迟。服务部署时应在
接收真实请求前使用相同的 2048 点、100 grasps、20 steps shape 完成预热。

## 与单步微基准比较

此前相同 2048 点、但只有 1 grasp 和 1 diffusion step 的端到端中位数为
`5956.25 ms`。本次 release workload 中位数为 `6545.53 ms`，增加
`589.28 ms`，约 `9.89%`。

两个 PTV3 的调用次数没有随 grasp 数和 diffusion steps 增加，额外耗时主要来自
100-grasp batch 上重复 20 次的 diffusion loop。

## 复现方式

目标机：`root@192.168.7.101`。

```bash
cd /root/Workspace/GraspgenX/GraspGenX
source /root/Workspace/IB_Robot-pi05-npu-core-eval/.shrc_local
source /root/Workspace/IB_Robot-pi05-npu-core-eval/install/setup.sh

python3 ascend/tools/benchmark_release_baseline.py \
  --point-count 2048 \
  --num-grasps 100 \
  --diffusion-steps 20 \
  --warmup 1 \
  --profile-runs 3 \
  --runs 5 \
  --output ascend/results/release_2048p_100g_20steps_random.json
```

基准实现：

```text
ascend/tools/benchmark_release_baseline.py
```

原始机器可读结果：

```text
ascend/results/release_2048p_100g_20steps_random.json
```

## 代码路径

当前产品源码的调用关系：

```text
graspgenx/models/grasp_gen.py
  GraspGen.forward
    -> grasp_generator.infer
    -> grasp_discriminator.infer

graspgenx/models/generator.py
  GraspGenGenerator.forward_inference
    -> object_encoder                   1 次
    -> diffusion loop                  20 次
       -> diffusion_head
       -> scheduler step
       -> likelihood
       -> rt_to_matrix

graspgenx/models/discriminator.py
  GraspGenDiscriminator.forward
    -> object_encoder                   1 次
    -> sample_encoder
    -> prediction_head                 1 次

graspgenx/models/ptv3/ptv3_vanilla.py
  PointTransformerV3Vanilla.forward
```

基准脚本使用上述实际 PTV3、diffusion head、scheduler、姿态转换、sample/gripper
encoder 和 discriminator head，仅由脚本显式编排 CPU/NPU 边界。原因是附件中的
`benchmark_device.py` 依赖当前 checkout 不具备的 CPU-offload/stage-timer hooks，
而当前 generator 还会把 `grasps_per_iteration` 默认创建在 CPU，无法直接执行这条
混合设备流水线。

## 后续优化判断

1. PTV3 仍是第一瓶颈，典型约占端到端时间 89%。
2. 如果两个 PTV3 总耗时可以从约 5.85 秒降到 100 ms，端到端预计仍约为
   0.8 秒，而不是 100 ms。
3. PTV3 之后的下一瓶颈是 diffusion loop，尤其是 diffusion head 和 SO(3)
   conversion。
4. 当前 NPU 环境明确提示 `sinc` 不受 NPU 支持并回退 CPU，这与 20 次 SO(3)
   conversion 合计约 188 ms 相符。
