# SubMConv3d 使用说明

基于 Ascend C 的 **子流形稀疏三维卷积前向算子**：输出仍对应输入的活动点，不生成新的活动体素。提供 PyTorch/Torch-NPU 接口、可复用的邻居索引和 TorchAir converter。当前实现通过官方注册和构建配置支持 Ascend 310P 产品族，平台限制不编码进算子名称。

- `BuildSubmMap`：在 AICore 构建邻居 map；首版为 O(N²) 坐标扫描。
- `SubmConv3d`：根据 map 执行 gather 和 Cube 矩阵乘，FP16 特征/权重、FP32 内部累加、FP16 输出。
- Python 模块 `SubMConv3d` 可选加 bias，并支持 `torch.compile(..., fullgraph=True)`。

这是前向验证实现，不是完整的 spconv 替代库。以下命令在目标设备的源码目录执行，不在本机文档库中执行。

## 1. 支持范围

| 项目 | 约定 |
| --- | --- |
| 已验证设备 | Ascend310P1/aarch64、Ascend310P3/x86_64 |
| 已验证算子工具链 | CANN 8.1.RC1，Torch/Torch-NPU 2.5.1；P1 Python 3.11，P3 Python 3.10 |
| 坐标 | 连续 NPU int32 `[N,4]`，列顺序为 `[batch,x,y,z]`；允许负 xyz |
| 点数 | `1 <= N <= 4096` |
| 特征 / 权重 | 连续 NPU FP16 `[N,Cin]` / `[K^3,Cin,Cout]` |
| 通道 / 卷积核 | Cin、Cout 均为 16 的倍数，范围 16..512；K 为 1、3、5 |
| map | 连续 NPU int32 `[N,ceil(K^3/8)*8]`，缺失或补齐位置为 -1 |
| bias / 输出 | 可选 NPU FP16 `[Cout]` / NPU FP16 `[N,Cout]` |

所有输入须位于同一 NPU。独立算子不提供 CPU fallback，不支持训练反向、空点云、Cin=3、非对齐通道、stride/dilation/groups 扩展或一般 `SparseConvTensor` 接口。

**独立 NPU 建图器要求坐标行唯一，代码不执行去重检查。** 复用 map 时，坐标值、batch、行顺序和 K 必须完全不变；不能仅凭点数或 stage 名缓存。map 内容及其与坐标的一致性由调用者保证。

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

在 `136.109` 使用其已有 Python 环境时，先修改两个环境变量：

```bash
export ASCEND_HOME_PATH=/home/huawei/Ascend/ascend-toolkit/8.1.RC1
export PYTHON_BIN=/home/huawei/hyd/Workspace/IB_Robot/venv/bin/python
bash build.sh
source env.sh
```

P1/P3 可复用 kernel 源码，但 Host `.so`、PyTorch bridge 必须匹配目标架构、Python 和 ABI，不能直接复制 x86 二进制到 aarch64。切换架构或工具链版本时，应在干净的独立构建目录重建，不混用旧 `build/`、`opp/` 产物。`env.sh` 自动选择架构及 openEuler/Ubuntu GCC 头文件布局。

此次中性更名不提供旧名称别名。旧 `submconv3d310p` 目录中的构建、安装和日志仅为历史产物，不参与新构建；不要把它们复制到新目录。使用新的 shell/process，只加载当前 `env.sh`，避免混入旧 OPP 环境。

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

省略 `neighbor_map` 会重新建图。模型参数初始为 CPU FP16；`eval()` 不关闭 autograd，调用时仍须使用 `torch.no_grad()` 或 inference mode。

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

不传 map 时，图中包含建图和卷积两个自定义算子；传入 map 时只包含卷积及可选 Add。首次调用包含编译开销；更改形状可能重新编译，不能与预热后的执行时间混算。这里保证的是单算子/模块全图，不代表整个 PTV3 可全图编译。

Torch-NPU 单算子 `jit_compile` 与 TorchAir 图编译是不同开关。库本身不修改全局 JIT 设置。P1 已验证 `jit_compile=False` 的含 bias 小例；P3 的 CANN 8.1 环境曾在关闭单算子 JIT 后连内置 Add 也失败，不加载自定义 OPP 的对照同样失败，不能据此判断卷积 kernel 不兼容。

## 5. 测试与 PTV3 接入

```bash
# 当前算子目录；包含 eager、fake/meta、TorchAir、精度和非法输入检查
SUBM_SMOKE=1 bash run_tests.sh -v
```

`SUBM_SMOKE=1` 额外覆盖 2048/3500/4096 点。数值门槛为 cosine >= 0.9999；形状错误、非有限输出或运行失败不能通过。其他误差仅报告。测试也检查生成的 ACLNN SoC 支持表、op-info 和 kernel 目录，确认只包含当前声明的平台。

中性更名后的 P1 原生重建已通过 9 项测试（含 22 个函数级 TorchAir 全图），以及 `jit_compile=False` 的模块和 PTv3 接入回归。日志为 `build/validation.log`、`build/integration.log`；这轮是功能回归，不是新增性能测量。

PTV3 适配类位于 `graspgenx/models/ptv3/ptv3_ascend.py`：

| 类 | CPE 路径 |
| --- | --- |
| `PointTransformerV3Ascend` | 默认：CPU 建图按 stage 复用，CPU FP32 卷积 |
| `PointTransformerV3Subm` | 相同 CPU map，NPU FP16 卷积，CPU FP32 bias/Linear/LayerNorm/残差 |

两者保留连续 NPU FP16 attention/norm/FFN；几何、serialization、downsample、Cin=3 stem 仍在 CPU。真实点云有重复体素，适配器刻意保留原 CPU hash/sort/searchsorted 语义，不使用独立 NPU 建图器。map 只在当前点云、当前 stage、单次前向内复用；未适配的卷积形状继续使用 CPU。这个形状回退由 PTV3 适配器显式实现，不代表 CANN 自动提供了 CPU kernel。

CPE 索引缓存已是默认内部行为，没有构造参数、环境变量、CLI 或单独模型类可关闭它。默认缓存不依赖本自定义算子包。SubM 仍为独立候选，可在两个验证脚本顶部切换直接 import；构造混合模型后不要对整个模型调用 `.npu()`、`.half()` 或 `.float()`。

从仓库根目录比较默认缓存 CPU CPE 与 SubM：

```bash
source ascend/custom_ops/submconv3d/env.sh
"$PYTHON_BIN" ascend/tools/benchmark_ptv3_subm.py
```

需要原有 `ascend/baselines/ptv3-cuda-fp32-eager/` 中两套权重和三份 reference NPZ。配置在脚本顶部，结果写入新的 `ascend/results/subm_compare_*` 目录，不覆盖 golden。现在是 AB/BA 两方案比较，`current` 表示默认缓存 CPU CPE，不是历史三方案中的未缓存控制。该比较是混合 eager 的 encoder-only 测量。

## 6. 官方平台注册与不支持处理

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
