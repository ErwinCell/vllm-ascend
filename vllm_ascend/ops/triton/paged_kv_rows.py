# SPDX-License-Identifier: Apache-2.0
"""Write small batches of token rows directly into a paged KV cache."""

import torch
from vllm.triton_utils import tl, triton

MAX_TOKEN_ROWS = 32
MAX_ROW_WIDTH = 2048
ROW_ALIGNMENT_BYTES = 32
MAX_CACHE_ELEMENTS = 1 << 31


@triton.jit
def write_rows_kernel(
    cache,
    updates,
    slots,
    CACHE_ROWS: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    UPDATE_STRIDE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots + row)
    if (slot >= 0) & (slot < CACHE_ROWS):
        cols = tl.arange(0, BLOCK_COLS)
        values = tl.load(updates + row * UPDATE_STRIDE + cols, cols < ROW_WIDTH, other=0)
        offset = slot.to(tl.int32) * ROW_WIDTH
        tl.store(cache + offset + cols, values, cols < ROW_WIDTH)


def supported_layout(cache, updates, slots):
    """Check tensor metadata without reading device slots."""
    return (
        cache.ndim == updates.ndim == 2
        and slots.ndim == 1
        and cache.shape[0] > 0
        and 0 < updates.shape[0] == slots.numel() <= MAX_TOKEN_ROWS
        and 0 < cache.shape[1] == updates.shape[1] <= MAX_ROW_WIDTH
        and cache.numel() <= MAX_CACHE_ELEMENTS
        and cache.dtype == updates.dtype
        and cache.dtype in (torch.bfloat16, torch.float16)
        and slots.dtype in (torch.int32, torch.int64)
        and cache.device == updates.device == slots.device
        and cache.is_contiguous()
        and slots.is_contiguous()
        and all(t.storage_offset() == 0 for t in (cache, updates, slots))
        and updates.stride(1) == 1
        and updates.stride(0) >= updates.shape[1]
        and (updates.shape[0] - 1) * updates.stride(0) + updates.shape[1] <= MAX_CACHE_ELEMENTS
        and (cache.shape[1] * cache.element_size()) % ROW_ALIGNMENT_BYTES == 0
    )


def write_paged_rows(cache: torch.Tensor, updates: torch.Tensor, slots: torch.Tensor) -> None:
    """Mutate cache on the current stream. Callers guarantee distinct live slots.

    Each in-range slot identifies one row. Padding is ignored and the kernel
    bounds checks addresses. Updates and slots must not alias the cache writes.
    The caller retains its existing path for unsupported layouts or callers
    without the distinct-slot guarantee.
    """
    if cache.device.type != "npu" or not supported_layout(cache, updates, slots):
        raise ValueError("unsupported paged KV row layout")
    width = cache.shape[1]
    write_rows_kernel[(slots.numel(),)](
        cache, updates, slots, cache.shape[0], width, updates.stride(0), triton.next_power_of_2(width)
    )
