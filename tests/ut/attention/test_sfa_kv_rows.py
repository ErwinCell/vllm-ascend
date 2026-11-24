# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.attention import sfa_kv_rows
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.triton.paged_kv_rows import supported_layout
from vllm_ascend.utils import AscendDeviceType


def _meta_tensors(rows=16, width=128, capacity=64, update_stride=None):
    return (
        torch.empty(capacity, width, device="meta", dtype=torch.bfloat16),
        torch.empty_strided((rows, width), (update_stride or width, 1), device="meta", dtype=torch.bfloat16),
        torch.empty(rows, device="meta", dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "rows,width,capacity,stride,allowed",
    [
        (1, 16, 1, 16, True),
        (32, 2048, 64, 2056, True),
        (0, 128, 64, 128, False),
        (33, 128, 64, 128, False),
        (2, 129, 64, 136, False),
        (1, 2064, 64, 2064, False),
        (1, 128, 0, 128, False),
        (2, 128, 64, 64, False),
        (1, 16, 1 << 27, 16, True),
        (1, 16, (1 << 27) + 1, 16, False),
        (2, 128, 64, (1 << 31) - 127, False),
    ],
)
def test_layout_bounds(rows, width, capacity, stride, allowed):
    assert supported_layout(*_meta_tensors(rows, width, capacity, stride)) is allowed


def test_layout_requires_matching_flat_rows_and_contiguous_slots():
    cache, updates, slots = _meta_tensors()
    assert not supported_layout(cache[1:], updates, slots)
    assert not supported_layout(cache.t(), updates, slots)
    assert not supported_layout(cache, updates[:, ::2], slots)
    assert not supported_layout(cache, updates, slots.unsqueeze(1))
    assert not supported_layout(cache, updates, slots[::2])
    assert not supported_layout(cache, updates.float(), slots)
    assert not supported_layout(cache, updates, slots.float())
    assert not supported_layout(cache, updates, torch.zeros(slots.shape, dtype=torch.int32))


@pytest.fixture
def capture(monkeypatch):
    layer = SimpleNamespace(
        enable_dsa_cp=False, enable_sparse_sfa_c8=False, enable_sparse_li_c8=False, is_kv_producer=False
    )
    metadata = SimpleNamespace(
        dsa_cp_context=None, dcp_context=None, sfa_cp_metadata=None, attn_state=AscendAttentionState.DecodeOnly
    )
    context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=SimpleNamespace(uniform=True),
        capturing=True,
        is_draft_model=False,
    )
    monkeypatch.setattr(sfa_kv_rows, "HAS_TRITON", True)
    monkeypatch.setattr(sfa_kv_rows, "supported_layout", lambda *_: True, raising=False)
    monkeypatch.setattr(sfa_kv_rows, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(sfa_kv_rows.vllm_envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    monkeypatch.setattr(sfa_kv_rows, "get_forward_context", lambda: context)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True, raising=False)
    return layer, metadata, context


@pytest.mark.parametrize("state", [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding])
def test_accepts_uniform_target_decode_capture(capture, state):
    layer, metadata, _ = capture
    metadata.attn_state = state
    updates = SimpleNamespace(device=SimpleNamespace(type="npu"))
    assert sfa_kv_rows._use_paged_kv_rows(layer, metadata, None, None, updates)


@pytest.mark.parametrize(
    "target,attribute,value",
    [
        ("layer", "enable_dsa_cp", True),
        ("layer", "enable_sparse_sfa_c8", True),
        ("layer", "enable_sparse_li_c8", True),
        ("layer", "is_kv_producer", True),
        ("metadata", "dsa_cp_context", object()),
        ("metadata", "dcp_context", object()),
        ("metadata", "sfa_cp_metadata", object()),
        ("metadata", "attn_state", AscendAttentionState.ChunkedPrefill),
        ("context", "cudagraph_runtime_mode", CUDAGraphMode.NONE),
        ("context", "capturing", False),
        ("context", "is_draft_model", True),
        ("context", "batch_descriptor", None),
        ("context", "batch_descriptor", SimpleNamespace(uniform=False)),
        ("module", "HAS_TRITON", False),
        ("module", "get_ascend_device_type", lambda: AscendDeviceType.A2),
        ("module", "supported_layout", lambda *_: False),
        ("module", "get_forward_context", Mock(side_effect=AssertionError)),
        ("envs", "VLLM_USE_V2_MODEL_RUNNER", True),
        ("npu", "is_current_stream_capturing", lambda: False),
    ],
)
def test_unsupported_contexts_use_scatter(capture, monkeypatch, target, attribute, value):
    layer, metadata, context = capture
    owners = {
        "layer": layer,
        "metadata": metadata,
        "context": context,
        "module": sfa_kv_rows,
        "envs": sfa_kv_rows.vllm_envs,
        "npu": torch.npu,
    }
    monkeypatch.setattr(owners[target], attribute, value)
    scatter = Mock()
    monkeypatch.setattr(sfa_kv_rows.torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    cache = torch.zeros(8, 16)
    slots = torch.tensor([0, 7], dtype=torch.int32)
    updates = SimpleNamespace(device=SimpleNamespace(type="npu"))
    sfa_kv_rows.write_indexer_cache(layer, metadata, cache, slots, updates)
    scatter.assert_called_once()
    destination, indices, source = scatter.call_args.args
    assert destination is cache and source is updates
    torch.testing.assert_close(indices, slots.view(-1, 1))
