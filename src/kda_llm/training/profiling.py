"""On-demand CUDA trace capture for real training steps."""

from __future__ import annotations

from pathlib import Path

import torch


class GPUProfiler:
    def __init__(self, start_step: int, warmup_steps: int, active_steps: int, output_dir: str) -> None:
        self.start_step, self.end_step = start_step, start_step + warmup_steps + active_steps
        self.output_dir, self.warmup_steps, self.active_steps = Path(output_dir), warmup_steps, active_steps
        self.profiler: torch.profiler.profile | None = None

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=self.warmup_steps, active=self.active_steps),
            record_shapes=True,
            profile_memory=True,
        )
        self.profiler.__enter__()

    def step(self, step: int) -> tuple[Path, Path] | None:
        if self.profiler is None:
            return None
        self.profiler.step()
        if step + 1 != self.end_step:
            return None
        self.profiler.__exit__(None, None, None)
        trace = self.output_dir / f"trace-step-{self.start_step}-{self.end_step}.json"
        summary = self.output_dir / f"summary-step-{self.start_step}-{self.end_step}.txt"
        self.profiler.export_chrome_trace(str(trace))
        summary.write_text(self.profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=30), encoding="utf-8")
        self.profiler = None
        return trace, summary
