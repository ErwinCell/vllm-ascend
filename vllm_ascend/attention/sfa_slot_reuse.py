# SPDX-License-Identifier: Apache-2.0
"""Reuse a slot conversion only within one model-forward invocation.

The entry cache belongs to vLLM's forward context, never to an attention layer
or a long-lived metadata object. Both context entry and the FULL graph wrapper
reset it. A captured graph therefore contains its own conversion operation and
replays that operation when the original input buffer is updated in place.
"""

import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import logger

STREAM_INFO_FIELDS = 3


def int64_slots_for_forward(slots: torch.Tensor, metadata) -> torch.Tensor:
    if slots.dtype == torch.int64:
        return slots
    if (
        slots.dtype != torch.int32
        or slots.ndim != 1
        or not slots.is_contiguous()
        or slots.numel() == 0
        or slots.device.type not in ("cpu", "npu")
        or slots is not getattr(metadata, "slot_mapping", None)
        or getattr(metadata, "dsa_cp_context", None) is not None
        or getattr(metadata, "dcp_context", None) is not None
    ):
        return slots.to(torch.int64)

    try:
        context = get_forward_context()
    except AssertionError:
        return slots.to(torch.int64)
    if context.cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
        return slots.to(torch.int64)

    # Never share a conversion across producer streams or from eager warmup
    # into capture. The full graph wrapper resets the cache at each invocation,
    # including a second capture using the same outer forward context.
    if slots.device.type == "npu":
        if slots.device.index != torch.npu.current_device():
            return slots.to(torch.int64)
        # current_stream() constructs a Python Stream wrapper on every call.
        # Use the same native identity accessor directly; no new extension is
        # required. Builds without this interface retain the original path.
        get_stream = getattr(torch_npu._C, "_npu_getCurrentStream", None)
        if not callable(get_stream):
            return slots.to(torch.int64)
        stream = get_stream(slots.device.index)
        if not isinstance(stream, tuple) or len(stream) != STREAM_INFO_FIELDS:
            return slots.to(torch.int64)
        capturing = torch.npu.is_current_stream_capturing()
    else:
        stream, capturing = None, False
    key = (id(metadata), id(slots), stream, capturing)
    cache = getattr(context, "_sfa_int64_slot_cache", None)
    if cache is None:
        cache = {}
        context._sfa_int64_slot_cache = cache
    entry = cache.get(key)
    if entry is not None and entry[0] is metadata and entry[1] is slots:
        if not getattr(context, "_sfa_slot_reuse_log_attempted", False):
            logger.info_once("[sfa_reuse] slot int64 conversion reuse active")
            context._sfa_slot_reuse_log_attempted = True
        return entry[2]

    converted = slots.to(torch.int64)
    # Strong references prevent id reuse; the whole cache dies with this
    # forward context. The graph's allocator owns captured device storage.
    cache[key] = (metadata, slots, converted)
    return converted
