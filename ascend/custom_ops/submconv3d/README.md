# Ascend SubM 算子组：建图与稀疏卷积

这是一组基于 Ascend C 的 **子流形稀疏三维卷积前向算子**，包含邻居建图 `BuildSubmMap` 和卷积计算 `SubmConv3d`。输出仍对应输入的活动点，不生成新的活动体素。算子组提供 PyTorch/Torch-NPU 接口、可复用的邻居索引和 TorchAir converter。当前实现通过官方注册和构建配置支持 Ascend 310P 产品族，平台限制不编码进算子名称。

- `BuildSubmMap`：在 AICore 构建邻居 map；支持唯一坐标精确查询，或使用 CPU 选定的 hash 代表行表进行查询。
- `SubmConv3d`：根据 map 执行 gather 和 Cube 矩阵乘，FP16 特征/权重、FP32 内部累加、FP16 输出；当前为 G4 输出通道分组复用。
- Python 模块 `SubMConv3d` 可选加 bias，并支持 `torch.compile(..., fullgraph=True)`。

这是前向验证实现，不是完整的 spconv 替代库。以下命令在目标设备的源码目录执行，不在本机文档库中执行。

## 1. 支持范围

| 项目 | 约定 |
| --- | --- |
| 已验证设备 | Ascend310P1/aarch64、Ascend310P3/x86_64 |
| 已验证算子工具链 | P1 CANN 8.1.RC1、P3 CANN 8.3（历史验证为 8.1.RC1）；Torch/Torch-NPU 2.5.1；P1 Python 3.11，P3 Python 3.10 |
| 坐标 | 连续 NPU int32 `[N,4]`，列顺序为 `[batch,x,y,z]`；允许负 xyz |
| 点数 | `1 <= N <= 4096` |
| 可选代表行表 | 连续 NPU `sorted_keys` int64 `[M]`、`source_rows` int32 `[M]`，`1 <= M <= N`；必须同时提供或同时省略 |
| 特征 / 权重 | 连续 NPU FP16 `[N,Cin]` / `[K^3,Cin,Cout]` |
| 通道 / 卷积核 | Cin、Cout 均为 16 的倍数，范围 16..512；K 为 1、3、5 |
| map | 连续 NPU int32 `[N,ceil(K^3/8)*8]`，缺失或补齐位置为 -1 |
| bias / 输出 | 可选 NPU FP16 `[Cout]` / NPU FP16 `[N,Cout]` |

所有输入须位于同一 NPU。独立算子不提供 CPU fallback，不支持训练反向、空点云、Cin=3、非对齐通道、stride/dilation/groups 扩展或一般 `SparseConvTensor` 接口。

`build_subm_map(indices, kernel_size=3, sorted_keys=None, source_rows=None)` 有两种契约：

- **不传代表行表：坐标行必须唯一，代码不执行去重检查。** 查询按完整坐标精确匹配；K=1 为 identity，K=3 且 N<=128、K=5 且 N<=256 使用扫描，更大 N 使用每核私有 hash 表。私有 hash 探测用完整 `[batch,x,y,z]` 校验消解碰撞，此校验仅属于唯一坐标分支。
- **传代表行表：key 必须严格升序且唯一，source row 必须在 `[0,N)`。** 内容是调用者前置条件，不执行内容扫描校验。查询始终对有符号 int64 hash `batch*334214467 + x*73856093 + y*19349669 + z*83492791` 做二分查找，包括 K=1；返回表中指定的代表行，不做坐标复核，刻意保留 reference hash 碰撞语义。PTV3 使用原始 CPU int64 hash `.sort()` 后每个 key 的首项，不能改为最小行号或 stable sort。

复用 map 时，坐标值、batch、行顺序、K 和可选代表行表必须完全不变；不能仅凭点数或 stage 名缓存。map 内容及其与输入的一致性由调用者保证。

G4 每组最多包含 4 个 Cout tile（每 tile 16 通道）；每个输入 tile 仅 gather 一次供组内共享，各输出 tile 在 FP32 中累加，独占输出、无 atomics。显式反向事件保护缓冲区复用，修复 L0/L1 读写 hazard；未采用 double buffer 或有效行压缩。

## 2. 编译与安装

需要已有可用的 CANN 开发工具链、GCC/G++、CMake、Python 开发头文件、Torch 和 Torch-NPU。脚本不安装依赖、不修改系统 CANN 配置。

在 `7.101` 的 `ascend` 工作区：

```bash
cd /root/Workspace/GraspgenX/GraspGenX/ascend/custom_ops/submconv3d
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1
export PYTHON_BIN=/usr/bin/python3
bash build.sh
source env.sh
```

`build.sh` 完成工程生成、Host/Ascend C 编译、OPP 打包安装和 PyTorch bridge 编译：

| 产物 | 位置 |
| --- | --- |
| 生成的工程、编译中间文件 | `build/` |
| 私有 OPP 安装前缀 | `opp/vendors/graspgenx_subm/` |
| PyTorch 扩展 | `build/torch_bridge.so` |

安装仅写入当前算子目录，不覆盖系统 OPP。每个新的运行 shell 都要 `source env.sh`，以设置私有 OPP、动态库和仓库 Python 路径。请统一使用下方的完整 Python 包路径，避免与同文件的另一种导入名称重复注册算子。

在 `136.109` 使用本次已验证的 CANN 8.3 和已有 Python 环境：

```bash
export ASCEND_HOME_PATH=/home/huawei/Ascend/ascend-toolkit/8.3.RC1
export PYTHON_BIN=/home/huawei/hyd/Workspace/IB_Robot/venv/bin/python
bash build.sh
source env.sh
```

P1/P3 可复用 kernel 源码，但 Host `.so`、PyTorch bridge 必须匹配目标架构、Python 和 ABI，不能直接复制 x86 二进制到 aarch64。切换架构或工具链版本时，应在干净的独立构建目录重建，不混用旧 `build/`、`opp/` 产物。`env.sh` 自动选择架构及 openEuler/Ubuntu GCC 头文件布局。

此次中性更名不提供旧名称别名。旧 `submconv3d310p` 目录中的构建、安装和日志仅为历史产物，不参与新构建；不要把它们复制到新目录。使用新的 shell/process，只加载当前 `env.sh`，避免混入旧 OPP 环境。恢复旧源码/OPP 的对照曾遇到生成的 `set_env.bash` 硬编码安装路径问题，已通过绑定实际 OPP 路径解决；加载后核对实际 `ASCEND_CUSTOM_OPP_PATH` 与 bridge/Host/kernel 配套，绝不混用新旧可选输入 ABI。

## 3. Torch-NPU 调用

加载环境后，以下是可直接运行的最小示例：

```python
import torch
from ascend.custom_ops.submconv3d.submconv3d import (
    SubMConv3d, build_subm_map,
)

torch.npu.set_device(0)
torch.manual_seed(0)
indices = torch.zeros((17, 4), dtype=torch.int32)
indices[:, 1] = torch.arange(17, dtype=torch.int32)
indices = indices.npu()
features = torch.randn(17, 32, dtype=torch.float16).npu()
layer = SubMConv3d(32, 64, kernel_size=3, bias=True).npu().eval()

with torch.no_grad():
    neighbors = build_subm_map(indices, 3)
    output = layer(features, indices, neighbor_map=neighbors)
torch.npu.synchronize()
print(output.shape, output.dtype, output.device)
# torch.Size([17, 64]) torch.float16 npu:0
```

省略 `neighbor_map` 会按唯一坐标契约重新建图。已有满足上文契约的 NPU 代表行表时，显式建图后传给模块即可，无需修改卷积接口：

```python
with torch.no_grad():
    neighbors = build_subm_map(indices, 3, sorted_keys=sorted_keys,
                               source_rows=source_rows)
    output = layer(features, indices, neighbor_map=neighbors)
```

`sorted_keys`/`source_rows` 由调用者准备；PTV3 的 CPU 代表行选择见 `graspgenx/models/ptv3/ptv3_ascend.py` 中 `SubMCPEConv._get_npu_map`。模块参数初始为 CPU FP16；`eval()` 不关闭 autograd，调用时仍须使用 `torch.no_grad()` 或 inference mode。

需要管理权重或提供已有 map 时，可使用函数或底层接口：

```python
from ascend.custom_ops.submconv3d.submconv3d import subm_conv3d

with torch.no_grad():
    output = subm_conv3d(features, indices, layer.weight, layer.bias,
                         kernel_size=3, neighbor_map=neighbors)
    raw = torch.ops.graspgenx_subm.subm_conv3d(features, neighbors, layer.weight)
```

底层 `raw` 不含 bias。map 的前 K³ 列依次对应 dx、dy、dz 从 `-(K//2)` 到 `K//2` 的字典序偏移，查询位置为输出坐标加该偏移。

## 4. TorchAir 编译

沿用上例对象：

```python
import torchair

compiled = torch.compile(layer, backend=torchair.get_npu_backend(),
                         fullgraph=True, dynamic=False)
with torch.no_grad():
    actual = compiled(features, indices, neighbor_map=neighbors)
torch.npu.synchronize()
torch.testing.assert_close(actual.cpu(), output.cpu(), rtol=0, atol=0)
```

不传 map 时，图中包含建图和卷积两个自定义算子；传入 map 时只包含卷积及可选 Add。`build_subm_map` 的可选代表行输入同时支持 eager、fake/meta 和 TorchAir converter，可在被编译函数内调用，或用同一 backend 单独编译 builder。首次调用包含编译开销；更改形状可能重新编译，不能与预热后的执行时间混算。这里保证的是单算子/模块全图，不代表整个 PTV3 可全图编译。

Torch-NPU 单算子 `jit_compile` 与 TorchAir 图编译是不同开关。库本身不修改全局 JIT 设置，PTV3 默认 `jit_compile=False`。`BuildSubmMap` 的 GE shape inference 已兼容 JIT 关闭时的零尺寸探测；真实 N/M 限制仍由 bridge 和运行时 tiling 检查，不表示支持空点云。P3 CANN 8.3 已覆盖此路径；历史 CANN 8.1 环境关闭单算子 JIT 时内置 Add 也曾失败，不加载自定义 OPP 的对照同样失败，不能据此判断卷积 kernel 不兼容。

## 5. 测试与 PTV3 接入

```bash
# 仓库根目录；包含 eager、fake/meta、TorchAir、精度和非法输入检查
SUBM_SMOKE=1 bash ascend/tests/run_tests.sh -v
```

Ascend 测试统一位于 `ascend/tests/`；入口自动加载 CPE 和 GridEncode 两个已构建的私有算子环境，在独立 Python 进程中运行四份 CPE 测试及两份 GridEncode 测试，并启用 NPU 用例。新增 `test_cpe_postops.py` 11 项后，P1 六套共 71 项已全部通过，无跳过。根目录 `tests/` 的通用/CUDA 测试不变。仅测试 CPE 或指定 unittest 用例时，仍可单独加载 CPE 环境：

```bash
source ascend/custom_ops/submconv3d/env.sh
"$PYTHON_BIN" -B ascend/tests/test_build_subm_map.py -v
"$PYTHON_BIN" -B ascend/tests/test_submconv3d.py SubMConvTests.test_platform_registration_artifacts -v
PTV3_CPE_NPU=1 "$PYTHON_BIN" -B ascend/tests/test_ptv3_cpe.py -v
ASCEND_TEST_NPU=1 "$PYTHON_BIN" -B ascend/tests/test_cpe_postops.py -v
```

`SUBM_SMOKE=1` 额外覆盖 2048/3500/4096 点。builder 测试覆盖唯一坐标、代表行/碰撞、可选输入校验及 eager/TorchAir；CPE 测试覆盖缓存、checkpoint 和回退语义。map 须精确匹配；浮点数值门槛仅为 cosine >= 0.9999，形状错误、非有限输出或运行失败不能通过。其他误差、repeat drift 和延迟仅报告。测试也检查生成的 ACLNN SoC 支持表、op-info 和 kernel 目录，确认只包含当前声明的平台。

新增 post-ops 测试覆盖显式 FP16 算式、packed 权重/bias、CPU 原参数与 checkpoint keys/strict reload、CPU 控制替换、attention 两种入口、回退和特征复制边界。P1 六套统一回归已通过（12 + 10 + 11 + 11 + 11 + 16），日志为 `build/cpe_postops_unified.log`；P3 新套件和原 CPE 套件各 11/11（各约 13 s）。默认 validator 六组过线且 repeat=0，2048 点 G/D 中位数为 112.202/110.887 ms，3500 点为 122.122/122.863 ms，六组中位数均达到 150 ms 的报告目标，但不承诺尾延迟。默认 2048 点 profiler 也通过；这是单进程回归，不替代正式 A/B 数据。

默认 profiler 的额外 dtype 审计确认每 encoder 仅 14 次浮点 H2D 和 14 次 D2H，CPE/attention/FFN 可见特征均为 NPU FP16；NPU FP32 tensor 仅见于 14 个 AddLayerNorm 调用的 mean/rstd 统计量。Cube 的内部 FP32 累加属于 kernel 内实现，不在此 tensor 审计范围内。新结果分别为 `ascend/results/ptv3_ascend_cpe_postops_resident_fp16_validation.json` 和 `ascend/results/ptv3_ascend_cpe_postops_fp16_audit_20260912_175105_profile_n2048.json`。

历史记录（2026-09-10，CPU post-ops 时代）：P1 当轮原生构建及完整套件已完成：conv 12/12（532.893 s）、map 10/10（438.709 s）、CPE 11/11（78.016 s），日志为本算子包的 `build/final_{conv,map,cpe}_tests.log`。最终 K=5 扫描至 N=256 的版本已安装并通过测试。P3 CANN 8.3 的阈值调整前快照也已通过 conv 12/12、map 10/10、CPE 11/11；不代表 P3 已验证最终阈值。

同一历史轮次的 P1 真实模型 3 进程 x 20 样本比较及合并后的默认模型 validator 均已完成，G/D x 三种点数共六组全部通过 golden 精度门槛，最低 cosine 为 0.9999591909124736，repeat drift 为 0；合并后六组输出与当轮旧 SubM 控制逐位一致。默认 stage profiler 的 2048 点 G/D 检查也通过。P3 最低 cosine 为 0.9999688915。这些是旧 CPU post-ops 路径的 encoder 精度验证，不是当前 FP16 post-ops 的逐位一致性或完整抓取质量验证。

PTV3 的 SubM 接入统一为 `graspgenx/models/ptv3/ptv3_ascend.py` 中的 `PointTransformerV3Ascend`；`PointTransformerV3Subm` 已移除且无别名。用户已批准支持形状的 CPE post-ops 默认采用 NPU FP16，无新增模型类或用户开关。**CPU 原 `.sort()` hash 代表行选择 + NPU `BuildSubmMap` 查询 + G4 NPU FP16 卷积保持不变**；原始 SubM 算子包 ABI/kernel 未改变，仅模型适配器扩展驻留特征路径。

每个支持的 block 数据流为：`CPU FP32 input -> 一次 H2D FP16 -> SubMCPEConv(NPU Half): raw conv + npu_bias -> F.linear(packed Half) -> npu_layer_norm_eval(packed Half) -> FP16 residual -> attention 直接使用 Half -> FP16 residual/LayerNorm/FFN -> CPU FP32 exit`。attention 的 CPU FP32 控制入口保留先 `prepare_indices`、后 H2D 的旧顺序；新 NPU FP16 入口不再复制特征。FFN 计算、返回 CPU FP32 的边界和 downsampling 均不变。

`AscendBlock` 的四个 nonpersistent buffer 为 `cpe_linear_weight_npu`、`cpe_linear_bias_npu`、`cpe_norm_weight_npu`、`cpe_norm_bias_npu`；卷积另缓存 `npu_bias`，沿用 `npu_weight`。CPE 原参数保留 CPU FP32 对象和 checkpoint keys，构造及 load-state-dict post-hook 打包，forward 不临时重打包。`_pack_cpe_weights` 用 `getattr` 安全检查支持形状，允许 benchmark 替换 `CachedCPEConv` 后 strict reload。

内部 `SubMCPEConv.forward` 支持 CPU FP32 -> CPU FP32 的旧语义（NPU raw conv 返回 CPU 后加原 bias），以及与卷积权重同设备的 NPU FP16 -> FP16（加 packed `npu_bias`）。前者供 CPU post-ops 控制使用，并非原始算子新增 CPU ABI。NPU 特征连续保持 FP16，不引入 FP32 feature/upcast；Cube 内部 FP32 累加及未使用的 LayerNorm FP32 mean/rstd 统计量不在此限制内。

初始四路空间编码仍由 [GridEncode](../grid_encode/README.md) 执行；网格化与 downsampling 保持 CPU FP32，depth、batch 拼接、排序/inverse 和 Cin=3 stem 仍在 CPU，**不是全 NPU encoder**。`ptv3_vanilla.py` 保持 reference 不变。

真实点云可有重复体素或 hash 碰撞，默认使用代表行分支而非独立唯一坐标分支。坐标/batch 超出 int32 时，模型先用 CPU reference map，再执行 NPU 卷积和 FP16 post-ops；不支持的卷积形状/N 则使用 CPU FP32 输入模式、CPU map+卷积及原 `feat + cpe_norm(cpe_linear(cpe))` 后处理公式。这是适配器显式回退，不是 CANN 自动提供 CPU kernel；NPU 运行异常不会静默重试 CPU。

map 只在当前点云、当前 stage、单次前向内复用。缓存是内部行为，无构造参数、环境变量或 CLI 开关；默认支持形状依赖本自定义算子包。`validate_ptv3.py` 和 `profile_ptv3_stages.py` 仅通过注释/取消注释顶部直接 import 选择 Ascend 或 vanilla，配置保持在脚本顶部；构造混合模型后不要对整个模型调用 `.npu()`、`.half()` 或 `.float()`。

以下标签仅用于 `ascend/tools/benchmark_ptv3_subm.py` 内部对照，不是模型 API 或独立模型类。脚本对全部三方案显式绑定旧 CPU FP32 post-ops，metadata 为 `cpe_post_ops=cpu_fp32_benchmark_control`，仅比较卷积/map 消融，不测当前默认 FP16 post-ops：

| 标签 | CPE map / 卷积 |
| --- | --- |
| `current` | 缓存 CPU map + CPU FP32 卷积（控制，不是现行默认） |
| `npu_subm` | CPU map + G4 NPU FP16 卷积 |
| `mixed_subm` | 与默认相同的 CPU 代表行 + NPU 查询 + G4 NPU FP16 卷积，但 post-ops 固定 CPU |

从仓库根目录运行三方案比较：

```bash
source ascend/custom_ops/submconv3d/env.sh
source ascend/custom_ops/grid_encode/env.sh
"$PYTHON_BIN" ascend/tools/benchmark_ptv3_subm.py
```

需要原有 `ascend/baselines/ptv3-cuda-fp32-eager/` 中两套权重和三份 reference NPZ。配置在脚本顶部，结果写入新的 `ascend/results/subm_compare_*` 目录，不覆盖 golden。三方案逐样本轮换顺序；`current` 也不是历史未缓存控制。测量是单对象、混合 eager、encoder-only，含每次前向的建图和传输，不是完整抓取流程。默认采用仍以 P1 为先；历史 P3 大点数相对 CPU CPE 的回退不能用本轮不同控制的改善抵消。

`ascend/tools/profile_cpe_ops.py` 同样仅用于 **CPU FP32 post-ops 控制**的 snapshot/replay/分项及 in-model 调试，不代表当前 FP16 candidate，分项和也不是整体延迟。测量当前 post-ops 改动应在上述环境中运行 `"$PYTHON_BIN" ascend/tools/benchmark_cpe_postops.py`：同一驻留模型仅切换 `forward_cpe` 的 CPU 控制/默认 NPU 路径，输出 `ascend/results/cpe_postops_compare_*/`，不覆盖 golden。

## 6. 手段与结果

<a id="cpe-postops"></a>

### CPE post-ops 手段与结果

P1 G+D、N=2048：旧 CPU FP32 post-ops 控制 280.862 -> 默认 NPU FP16 post-ops 223.002 ms，-57.859 ms（-20.60%）；P3：77.977 -> 70.379 ms，-7.597 ms（-9.74%）。范围为单对象 encoder-only、同步 host wall time 含传输，非完整请求；结论：本轮中位数改善，但仅各 3 进程、每 case/variant 60 个计时样本，不主张统计显著性或尾延迟全面改善。

手段是把卷积 bias、CPE linear、LayerNorm 和 residual 放入驻留 NPU FP16 区域，让 attention 直接使用结果，消除中途特征回传/再次上传。CPU reference map 语义、NPU 查询/卷积 kernel、GridEncode、dense 计算和 FFN exit/downsampling 固定。
控制是旧 **NPU 卷积 + CPU FP32 post-ops**，不是纯 CPU CPE；本轮不能回答相对纯 CPU CPE 的收益。

| 设备 / N | CPU post-ops ms | NPU post-ops ms | 变化 ms | 变化 % |
| --- | ---: | ---: | ---: | ---: |
| P1 / 64 | 215.756 | 173.091 | -42.665 | -19.77% |
| P1 / 2048 | 280.862 | 223.002 | -57.859 | -20.60% |
| P1 / 3500 | 313.121 | 246.439 | -66.682 | -21.30% |
| P3 / 64 | 46.841 | 41.692 | -5.149 | -10.99% |
| P3 / 2048 | 77.977 | 70.379 | -7.597 | -9.74% |
| P3 / 3500 | 99.361 | 90.760 | -8.601 | -8.66% |

每个 encoder 使用同一驻留模型，只重绑定 `forward_cpe`；每设备 3 独立进程、CPU 16 线程，warmup 3 / 测量 20 / repeat 3，按 run/process 交替 AB/BA。
G+D 为**每进程 G/D encoder 中位数之和再取中位数**，独立聚合的 G、D 中位数不必相加等于此表。差值/百分比由未舍入 JSON 计算；表内时间保留 3 位小数。
N=2048 另有每进程、每 encoder/variant 3 次分阶段 profile；分项计时仅作诊断，不能相加替代完整 encoder wall time。

每设备 972 个输出（936 direct，含 warmup/测量/repeat；36 profile）均通过 CUDA golden gate；另 36 次 profile/direct 比较也通过且 max abs 为 0。要求形状 `[1,512]`、有限值、cosine >= 0.9999，运行失败不能通过。

| 设备 | CPU 控制最低 CUDA cosine | NPU candidate 最低 CUDA cosine | Candidate 最大 CUDA abs error | Repeat max abs |
| --- | ---: | ---: | ---: | ---: |
| P1 | 0.9999591909124736 | 0.9999510227078675 | 0.0369681 | 0 |
| P3 | 0.9999688914724082 | 0.9999525076094965 | 0.0367813 | 0 |

FP16 post-ops 会改变误差，candidate 不与旧控制逐位一致；两者最大 abs 差分别为 P1 0.0138216、P3 0.0243397。除约定 cosine gate 外的误差、repeat drift 和延迟均仅报告。
所有长尾保留，合并每 case/variant 的 60 样本后，两设备各组 P95 均改善，但 P3 G/64 的 P99/max 为 24.368/24.454 -> 25.338/26.533 ms，P3 D/3500 为 50.693/50.790 -> 51.819/59.255 ms，均回退。
P1 G/2048 的 max 也从 154.889 -> 167.313 ms。60 样本不足以稳定估计尾部；P3 原有约占一个 CPU 核的 glob 搜索及桌面负载未被终止，视为用户活动和潜在噪声，不剔除相应样本。

复制结构测试显示每 encoder 14 个支持 block：控制 28 H2D + 28 D2H，candidate 14 + 14，且无 forward 参数重打包。这是浮点特征的 Python dispatch 计数，**不是 DMA/kernel 次数**，整数 metadata 复制不在此计数内；不得以此或 kernel/TransData 数量代替整体性能测量。

- [P1 三进程原始汇总](../../results/cpe_postops_compare_20260912_170605_032812/summary.json)
- [P3 私有三进程原始汇总（SSH）](ssh://huawei@192.168.136.109/home/huawei/hyd/Workspace/GraspgenX/cpe_postops_validation_20260912_loQWePSS/ascend/results/cpe_postops_compare_20260912_171258_329608/summary.json)

以下各表及 G2/G4/唯一坐标建图记录均为加入 GridEncode 前、CPU post-ops 时代的历史 CPE 数据，原值保留；其中“默认”指当时的默认。当前卷积/map benchmark 对所有方案固定 CPU post-ops、统一使用 GridEncode，不能将不同批次的绝对延迟差归因于本轮 post-ops。

### 默认路径三方案比较

P1 G+D、N=2048：控制 `current` 472.260 -> 默认 `mixed_subm` 381.349 ms，-90.911 ms（-19.25%），结论：改善。P3 同规模：90.182 -> 97.583 ms，+7.401 ms（+8.21%），结论：回退。均为单对象 encoder-only；3 进程、每进程 warmup 3 / 测量 20、CPU 16 线程，非完整抓取质量或端到端延迟。

| 设备 / N | `current` ms | `npu_subm` ms | `mixed_subm` ms | 默认相对 `current` | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| P1 / 64 | 319.676 | 266.525 | 262.608 | -57.068 ms（-17.85%） | 改善 |
| P1 / 2048 | 472.260 | 392.255 | 381.349 | -90.911 ms（-19.25%） | 改善 |
| P1 / 3500 | 548.071 | 462.530 | 450.790 | -97.281 ms（-17.75%） | 改善 |
| P3 / 64 | 59.317 | 58.754 | 58.327 | -0.990 ms（-1.67%） | 噪声内，未定 |
| P3 / 2048 | 90.182 | 94.976 | 97.583 | +7.401 ms（+8.21%） | 回退 |
| P3 / 3500 | 109.534 | 119.600 | 124.859 | +15.325 ms（+13.99%） | 回退 |

数值为 G、D 各自的进程中位数再取中位数之和。代表行方案把邻居查询移到 NPU；单独对比 G4 CPU-map 控制 `npu_subm`，N=2048 时 P1 为 392.255 -> 381.349 ms，-10.906 ms（-2.7802%），P3 为 94.976 -> 97.583 ms，+2.607 ms（+2.74%）。上述三方案数据在模型类合并前采集。合并后默认 validator 的 G+D 中位数之和为 64/2048/3500 点 261.588/379.717/450.004 ms（1 进程、warmup 3、20 样本，无同期控制，只作回归），见 `ascend/results/ptv3_ascend_cpe_g4_mixed_validation.json`；脚本中每 encoder 150 ms 的报告目标仍未全部达到。

P1 数据：`ascend/results/subm_compare_20260910_135612/summary.json`。P3 CANN 8.3 数据位于私有工作区 `cpe_validation_20260910_14n8IIj9` 的 `ascend/results/subm_compare_20260910_144355/summary.json`；测试期间有约占用一个 CPU 核的无关文件搜索，未干预，因此小幅差异和尾延迟需谨慎解释。

### G2 分组复用

P1 G+D、N=2048：旧 SubM 控制 407.277 -> G2 395.687 ms，-11.590 ms（-2.85%）；encoder-only，独立顺序执行的两组各 3 进程，结论：该测量改善，非普遍加速证据。G2 通过两个输出通道 tile 共享 gather；两组 CPU 控制 459.740 / 460.075 ms 基本稳定，但旧/新 OPP 未交错测量。

### G4 分组复用

P1 冻结 N=2048 输入、每个 encoder 14 个 block，G+D 共 28 次卷积调用的延迟之和：旧 kernel 约 45.187 -> G4 29.377 ms，约 -15.810 ms（-34.99%）；仅驻留输入的隔离、预热后 host 延迟，1 进程 / 10 样本，结论：隔离测量改善，不是 encoder 加速幅度。手段为最多 4 个 Cout tile 共享 gather，独占输出和 FP32 累加不变。

### 唯一坐标建图

P1 dense K=3：旧扫描控制 N=2048 为 16.410 -> 最终版 1.074 ms，-15.336 ms（-93.46%）；N=4096 为 61.517 -> 1.783 ms，-59.734 ms（-97.10%）。仅独立 coordinate-only builder、驻留输入、host 同步计时，各 1 进程 / 20 样本 / warmup 3；结论：隔离测量改善，不是默认模型收益。大 N 改为每核私有 hash，较小 N 保留扫描；最终 60 个唯一坐标用例全部通过，包括 K=5 扫描至 N=256 的阈值。控制与最终版数据分别位于 `ascend/results/subm_map_unique_coordinate_20260910_082549_852667/summary.json` 和 `ascend/results/subm_map_unique_coordinate_20260910_091716_676283/summary.json`。小 N 的差异接近启动开销与噪声，不主张全部形状提速。

## 7. 官方平台注册与不支持处理

两个 Host 算子通过 CANN `OpDef` 注册：

```cpp
AICore().SetTiling(optiling::Tiling).AddConfig("ascend310p");
```

构建目标保持 `ai_core-ascend310p` / `ASCEND_COMPUTE_UNIT=ascend310p`，生成对应 op-info、ACLNN `socSupportList` 和 kernel 目录。P1、P3 属于此支持族；P3 性能回退不等于不能执行。

`AddConfig` 是支持配置，不应脱离生成器和显式编译目标视作跨版本的万能运行时拦截器。升级 CANN 或增加目标时须重新核对生成产物，不能只凭算子类型存在或 Meta 推断成功认定支持当前设备。

- 框架按当前平台、dtype、format 等匹配已注册的实现。不存在适配实现时，可能在匹配、编译、tiling 或加载阶段失败；不同调用路径和 CANN 版本的错误码不保证相同。
- 有另行注册且适配的 AICPU/HostCPU 实现时，相关选择流程才能使用它；框架不会自动把 Ascend C 数学计算转换为 CPU 代码。本包只提供 AICore 计算实现。
- Torch-NPU 的 dispatcher CPU fallback 不是已注册 NPU kernel 抛异常后的自动重试。本 bridge 的 `OpCommand` 错误按原样传播，没有宽泛异常捕获和静默 CPU 回退。
- TorchAir 的 converter 只生成 GE 节点，实际平台匹配仍由后续执行流程完成；不能把 `EnableFallBack()` 生成的 GE-to-ACLNN 调用误认为 CPU 计算。

当前验证包含支持平台上的运行和生成产物检查，不冒充非 310P 硬件上的负向运行验证。官方依据：[AddConfig](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0986.html)、[构建配置生成](https://raw.gitcode.com/cann/asc-devkit/files/master/tools/build/opbuild/op_cfg_generator.cpp)、[Torch-NPU CPU fallback](https://raw.githubusercontent.com/Ascend/pytorch/v2.5.1-7.0.0/torch_npu/csrc/aten/VariableFallbackKernel.cpp)。
