# SPDX-License-Identifier: Apache-2.0
import torch


def graph_replay_us(graph, repeats=32):
    """Measure device time per replay after one warmup."""
    graph.replay()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats
