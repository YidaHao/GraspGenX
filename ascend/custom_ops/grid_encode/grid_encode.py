"""Integer spatial serialization on Ascend 310P, independent of the PTV3 model.

Import after sourcing this package's env.sh and building its private OPP/bridge.
The raw operator is torch.ops.graspgenx_grid.grid_encode(grid_coord, depth).
No CPU execution, implicit casts, coordinate scans or global JIT settings.
"""

from pathlib import Path

import torch
import torch_npu  # noqa: F401 -- registers NPU and bundled TorchAir
import torchair
from torchair import ge


__all__ = ["grid_encode"]

_LIBRARY = Path(__file__).resolve().parent / "build" / "torch_bridge.so"
if not _LIBRARY.is_file():
    raise FileNotFoundError(f"Build the GridEncode bridge first: {_LIBRARY}")
torch.ops.load_library(str(_LIBRARY))


def _check_grid_coord(grid_coord, depth):
    # Meta/fake has no NPU storage format; runtime tiling checks ND.
    if grid_coord.device.type not in ("npu", "meta"):
        raise RuntimeError("grid_coord must be an NPU tensor")
    if grid_coord.layout != torch.strided:
        raise RuntimeError("grid_coord must have strided layout")
    if grid_coord.dtype != torch.int32:
        raise RuntimeError("grid_coord must have dtype int32")
    if grid_coord.ndim != 2 or grid_coord.shape[1] != 3:
        raise RuntimeError("grid_coord must have shape [N, 3]")
    if not 1 <= grid_coord.shape[0] <= 4096:
        raise RuntimeError("N must be in [1, 4096]")
    if not grid_coord.is_contiguous():
        raise RuntimeError("grid_coord must be contiguous")
    if not isinstance(depth, int) or not 1 <= depth <= 16:
        raise RuntimeError("depth must be in [1, 16]")


@torch.library.register_fake("graspgenx_grid::grid_encode")
def _grid_encode_fake(grid_coord, depth):
    _check_grid_coord(grid_coord, depth)
    return grid_coord.new_empty((grid_coord.shape[0], 4), dtype=torch.int64)


@torchair.register_fx_node_ge_converter(torch.ops.graspgenx_grid.grid_encode.default)
def _grid_encode_ge(grid_coord, depth, meta_outputs=None):
    return ge.custom_op(
        "GridEncode",
        inputs={"grid_coord": grid_coord},
        outputs=["spatial_codes"],
        attrs={"depth": ge.attr.Int(depth)},
    )


def grid_encode(grid_coord, depth):
    """Return contiguous ND NPU int64 [N, 4] spatial codes on the input device.

    Input is contiguous ND NPU int32 [N, 3] in (x, y, z) order, 1 <= N <= 4096,
    with integer depth in [1, 16]. Each coordinate must satisfy
    0 <= coordinate < 2**depth. Contents are a caller precondition: there is no
    min/max scan, device-to-host value check or clamping. Depth 0 is rejected.

    Columns are (z, z-trans, hilbert, hilbert-trans), exactly matching
    ptv3_vanilla.encode(grid_coord, batch=zeros(N, int64), depth, order).
    Both trans orders swap x/y only. There are no batch bits, origin offsets,
    floating-point quantization, sorting, permutations or feature operations.
    N=1 still returns [1, 4]. This integer-only operation has no backward.
    """
    _check_grid_coord(grid_coord, depth)
    return torch.ops.graspgenx_grid.grid_encode(grid_coord, depth)
