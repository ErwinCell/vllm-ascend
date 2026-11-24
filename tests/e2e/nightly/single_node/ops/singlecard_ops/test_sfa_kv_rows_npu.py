# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, ForwardContext, override_forward_context
from vllm.triton_utils import HAS_TRITON

from tests.e2e.nightly.single_node.ops.graph_timing import graph_replay_us
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_kv_rows import write_indexer_cache
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


@pytest.mark.skipif(not HAS_TRITON or get_ascend_device_type() != AscendDeviceType.A3, reason="requires A3 with Triton")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_indexer_cache_cold_capture_and_dynamic_rows(dtype, slot_dtype, record_property):
    rows, width, capacity = 16, 128, 257
    layer = SimpleNamespace(
        enable_dsa_cp=False, enable_sparse_sfa_c8=False, enable_sparse_li_c8=False, is_kv_producer=False
    )
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.SpecDecoding, dsa_cp_context=None, dcp_context=None, sfa_cp_metadata=None
    )
    ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=BatchDescriptor(rows, uniform=True),
    )
    ctx.capturing = True
    ctx.is_draft_model = False
    cache = torch.zeros(capacity, width, dtype=dtype, device="npu")
    slots = torch.full((rows,), -1, dtype=slot_dtype, device="npu")
    # A sliced update can retain padding between rows.
    updates = torch.zeros(rows, width + 16, dtype=dtype, device="npu")[:, :width]
    graph = torch.npu.NPUGraph()
    with (
        override_forward_context(ctx),
        patch.object(torch_npu, "npu_scatter_nd_update_", side_effect=AssertionError("unexpected scatter fallback")),
        torch.npu.graph(graph),
    ):
        write_indexer_cache(layer, metadata, cache, slots, updates)

    for step, live in enumerate(((0, 128), (capacity - 1, 1), ())):
        values = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width).add(step).to(dtype)
        # Verify that a signed zero and a NaN payload survive the copy.
        values.view(torch.int16)[0, :2] = torch.tensor([-32768, 0x7FC1], dtype=torch.int16)
        updates.copy_(values)
        slots.fill_(-1)
        if live:
            slots[: len(live)].copy_(torch.tensor(live, dtype=slot_dtype))
        cache.fill_(1)
        graph.replay()
        expected = torch.ones(cache.shape, dtype=dtype)
        if live:
            expected[list(live)] = values[: len(live)]
        assert torch.equal(cache.cpu().view(torch.int16), expected.view(torch.int16))

    # Compare device replay times using the same valid, unique slots and update values.
    slots.copy_(torch.arange(rows, dtype=slot_dtype))
    reference_cache = torch.zeros_like(cache)
    reference = torch.npu.NPUGraph()
    with torch.npu.graph(reference):
        torch_npu.npu_scatter_nd_update_(reference_cache, slots.view(-1, 1), updates)
    record_property("scatter_us", graph_replay_us(reference))
    record_property("row_write_us", graph_replay_us(graph))
