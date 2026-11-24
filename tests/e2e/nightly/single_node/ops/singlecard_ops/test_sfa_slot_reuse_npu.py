# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, ForwardContext, override_forward_context

from tests.e2e.nightly.single_node.ops.graph_timing import graph_replay_us
from vllm_ascend.attention.sfa_slot_reuse import int64_slots_for_forward
from vllm_ascend.compilation import acl_graph


def _context(mode=CUDAGraphMode.NONE):
    ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=mode,
        batch_descriptor=BatchDescriptor(2),
    )
    ctx.is_draft_model = False
    return ctx


def test_each_full_capture_owns_its_slot_conversion(monkeypatch):
    slots = torch.tensor([-1, 17], dtype=torch.int32, device="npu")
    metadata = SimpleNamespace(slot_mapping=slots)

    def consume(value):
        result = int64_slots_for_forward(value, metadata)
        assert int64_slots_for_forward(value, metadata) is result
        return result.clone()

    options = SimpleNamespace(debug_log_enable=False, gc_disable=False, weak_ref_output=False)
    config = SimpleNamespace(compilation_config=None)
    first, second = [acl_graph.ACLGraphWrapper(consume, config, CUDAGraphMode.FULL, options) for _ in range(2)]
    monkeypatch.setattr(acl_graph, "validate_cudagraph_capturing_enabled", lambda: None)
    ctx = _context()
    with override_forward_context(ctx):
        torch.testing.assert_close(first(slots), slots.long())
        slots.add_(5)
        torch.testing.assert_close(first(slots), slots.long())
        ctx.cudagraph_runtime_mode = CUDAGraphMode.FULL
        # Retain the capture outputs; the graph wrappers store weak references.
        capture_outputs = (first(slots), second(slots))
        for wrapper, output in zip((first, second), capture_outputs, strict=True):
            slots.add_(11)
            result = wrapper(slots)
            torch.testing.assert_close(result, slots.long(), rtol=0, atol=0)
            torch.testing.assert_close(output, result, rtol=0, atol=0)


def test_slot_conversions_are_stream_local():
    slots = torch.tensor([23, -1], dtype=torch.int32, device="npu")
    metadata = SimpleNamespace(slot_mapping=slots)
    with override_forward_context(_context()):
        original = int64_slots_for_forward(slots, metadata)
        other_stream = torch.npu.Stream()
        other_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(other_stream):
            separate = int64_slots_for_forward(slots, metadata)
            assert separate is not original
            assert int64_slots_for_forward(slots, metadata) is separate
        other_stream.synchronize()
        torch.testing.assert_close(separate, original, rtol=0, atol=0)


def test_repeated_layer_conversions(record_property):
    slots = torch.arange(16, device="npu", dtype=torch.int32)
    metadata = SimpleNamespace(slot_mapping=slots)
    baseline, reused = torch.npu.NPUGraph(), torch.npu.NPUGraph()
    with torch.npu.graph(baseline):
        per_layer = [slots.long() for _ in range(78)]
    with override_forward_context(_context(CUDAGraphMode.FULL)), torch.npu.graph(reused):
        shared = [int64_slots_for_forward(slots, metadata) for _ in range(78)]
    slots.add_(5)
    baseline.replay()
    reused.replay()
    assert all(tensor is shared[0] for tensor in shared)
    torch.testing.assert_close(shared[0], per_layer[-1], rtol=0, atol=0)
    record_property("per_layer_us", graph_replay_us(baseline))
    record_property("shared_us", graph_replay_us(reused))
