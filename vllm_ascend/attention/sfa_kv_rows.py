# SPDX-License-Identifier: Apache-2.0
"""Route capture-time SFA indexer cache writes to a bounded last-dimension copy."""

from typing import TYPE_CHECKING

import torch
import torch_npu
import vllm.envs as vllm_envs
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

if HAS_TRITON:
    from vllm_ascend.ops.triton.paged_kv_rows import supported_layout, write_paged_rows

if TYPE_CHECKING:
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata


def _write_kv_cache(cache: torch.Tensor, updates: torch.Tensor, slots: torch.Tensor) -> None:
    write_paged_rows(cache, updates, slots)


def _fake_write_kv_cache(cache: torch.Tensor, updates: torch.Tensor, slots: torch.Tensor) -> None:
    return None


if HAS_TRITON:
    direct_register_custom_op(
        op_name="ascend_write_small_paged_kv_rows",
        op_func=_write_kv_cache,
        mutates_args=["cache"],
        fake_impl=_fake_write_kv_cache,
        dispatch_key="PrivateUse1",
    )


def _use_paged_kv_rows(
    layer: "AscendSFAImpl",
    metadata: "AscendSFAMetadata",
    cache: torch.Tensor,
    slots: torch.Tensor,
    updates: torch.Tensor,
) -> bool:
    if (
        not HAS_TRITON
        or vllm_envs.VLLM_USE_V2_MODEL_RUNNER
        or updates.device.type != "npu"
        or layer.enable_dsa_cp
        or layer.enable_sparse_sfa_c8
        or layer.enable_sparse_li_c8
        or layer.is_kv_producer
        or metadata.dsa_cp_context is not None
        or metadata.dcp_context is not None
        or metadata.sfa_cp_metadata is not None
        or metadata.attn_state not in (AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding)
        or get_ascend_device_type() is not AscendDeviceType.A3
        or not supported_layout(cache, updates, slots)
    ):
        return False
    try:
        context = get_forward_context()
    except AssertionError:
        return False
    descriptor = context.batch_descriptor
    return (
        context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        and descriptor is not None
        and descriptor.uniform
        and getattr(context, "capturing", False)
        and getattr(context, "is_draft_model", None) is False
        and torch.npu.is_current_stream_capturing()
    )


def write_indexer_cache(
    layer: "AscendSFAImpl",
    metadata: "AscendSFAMetadata",
    cache: torch.Tensor,
    slots: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    """SFA decode metadata maps current writable tokens to distinct live slots."""
    if _use_paged_kv_rows(layer, metadata, cache, slots, updates):
        torch.ops.vllm.ascend_write_small_paged_kv_rows(cache, updates, slots)
        logger.info_once("[decode_kv_rows] small paged KV writer captured")
        return
    torch_npu.npu_scatter_nd_update_(cache, slots.view(-1, 1), updates)
