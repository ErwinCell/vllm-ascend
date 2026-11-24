# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_ascend.attention import sfa_slot_reuse
from vllm_ascend.attention.sfa_slot_reuse import int64_slots_for_forward
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


def _context(mode):
    return ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={}, cudagraph_runtime_mode=mode)


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_exec_kv_shares_converted_slots_only_within_forward(monkeypatch, mode):
    slots = torch.tensor([-1, 7], dtype=torch.int32)
    metadata = SimpleNamespace(slot_mapping=slots)
    consume = Mock()
    monkeypatch.setattr(torch_npu, "npu_kv_rmsnorm_rope_cache", consume, raising=False)
    layer = SimpleNamespace(
        num_kv_heads=1,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        enable_sparse_sfa_c8=False,
        enable_dsa_cp=False,
        kv_a_layernorm=SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-5),
    )
    x = torch.zeros(2, 6)
    with override_forward_context(_context(mode)):
        for _ in range(2):
            AscendSFAImpl.exec_kv(layer, x, None, None, (None, None), slots, metadata)
        first, second = [call.args[4] for call in consume.call_args_list]
        assert first is second
        assert first.dtype == torch.int64
        assert torch.equal(first, slots.long())
        assert int64_slots_for_forward(slots, SimpleNamespace(slot_mapping=slots)) is not first

    slots.add_(10)
    with override_forward_context(_context(mode)):
        fresh = int64_slots_for_forward(slots, metadata)
        assert fresh is not first
        assert torch.equal(fresh, slots.long())


@pytest.mark.parametrize("case", ["piecewise", "foreign", "strided", "rank2", "int16", "dsa_cp", "dcp", "empty"])
def test_unsupported_slots_use_original_conversion(case):
    slots = torch.arange(8, dtype=torch.int32)
    if case == "strided":
        slots = slots[::2]
    elif case == "rank2":
        slots = slots.view(2, 4)
    elif case == "int16":
        slots = slots.short()
    elif case == "empty":
        slots = slots[:0]
    metadata = SimpleNamespace(slot_mapping=slots.clone() if case == "foreign" else slots)
    if case in ("dsa_cp", "dcp"):
        setattr(metadata, case + "_context", object())
    mode = CUDAGraphMode.PIECEWISE if case == "piecewise" else CUDAGraphMode.FULL
    with override_forward_context(_context(mode)):
        first = int64_slots_for_forward(slots, metadata)
        assert int64_slots_for_forward(slots, metadata) is not first
        torch.testing.assert_close(first, slots.long(), rtol=0, atol=0)


def test_missing_context_and_already_int64(monkeypatch):
    monkeypatch.setattr(sfa_slot_reuse, "get_forward_context", Mock(side_effect=AssertionError))
    slots = torch.tensor([0, -1], dtype=torch.int32)
    metadata = SimpleNamespace(slot_mapping=slots)
    first = int64_slots_for_forward(slots, metadata)
    assert int64_slots_for_forward(slots, metadata) is not first
    torch.testing.assert_close(first, slots.long())
    assert int64_slots_for_forward(first, metadata) is first
