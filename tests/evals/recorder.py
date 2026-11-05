"""Structured persistence and history management for evaluation benchmark runs."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import typing
from pydantic import BaseModel, ConfigDict, Field


class EvalRecord(BaseModel):
    """Execution telemetry and validation outcome for a single evaluation fixture."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    eval_id: str
    topic: str
    timestamp: str
    passed: bool
    iterations_used: int
    duration_ms: float
    legacy_input_code: str
    modern_output_code: str
    critic_verdict: dict[str, typing.Any]
    gates_verified: dict[str, bool]


class RunSummary(BaseModel):
    """Aggregate statistics for an evaluation benchmark run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    tag: str
    timestamp: str
    total_evals: int
    passed_evals: int
    failed_evals: int
    pass_rate_percent: float
    avg_iterations: float
    total_duration_ms: float
    evals: list[dict[str, typing.Any]] = Field(default_factory=list)


class EvalRunRecorder:
    """Manages creation of run directories, per-eval JSON logs, summaries, and manifest."""

    def __init__(self, base_dir: typing.Optional[Path] = None) -> None:
        if base_dir is None:
            # Default to tests/evals/results relative to repo structure
            self.base_dir = Path(__file__).resolve().parent / "results"
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.active_run_id: typing.Optional[str] = None
        self.active_tag: str = "baseline"
        self.run_dir: typing.Optional[Path] = None
        self.run_start_time: typing.Optional[datetime.datetime] = None
        self.records: list[EvalRecord] = []

    def start_run(self, run_id: typing.Optional[str] = None, tag: str = "baseline") -> str:
        """Initialize a new evaluation benchmark run directory."""
        now = datetime.datetime.now(datetime.timezone.utc)
        self.run_start_time = now
        self.active_tag = tag

        if run_id:
            self.active_run_id = run_id
        else:
            timestamp_slug = now.strftime("%Y%m%d_%H%M%S")
            self.active_run_id = f"run_{timestamp_slug}_{tag}"

        self.run_dir = self.base_dir / self.active_run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.records = []
        return self.active_run_id

    def record_eval(
        self,
        eval_id: str,
        topic: str,
        passed: bool,
        iterations_used: int,
        duration_ms: float,
        legacy_input_code: str,
        modern_output_code: str,
        critic_verdict: dict[str, typing.Any],
        gates_verified: dict[str, bool],
    ) -> EvalRecord:
        """Write an individual JSON record for an evaluation test case."""
        if not self.run_dir or not self.active_run_id:
            raise RuntimeError("Run must be started with start_run() before recording evals.")

        record = EvalRecord(
            run_id=self.active_run_id,
            eval_id=eval_id,
            topic=topic,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            passed=passed,
            iterations_used=iterations_used,
            duration_ms=round(duration_ms, 2),
            legacy_input_code=legacy_input_code,
            modern_output_code=modern_output_code,
            critic_verdict=critic_verdict,
            gates_verified=gates_verified,
        )

        record_file = self.run_dir / f"{eval_id}.json"
        with open(record_file, "w", encoding="utf-8") as f:
            f.write(record.model_dump_json(indent=2))

        self.records.append(record)
        return record

    def finish_run(self) -> RunSummary:
        """Generate run summary, write summary.json, and update history_manifest.json."""
        if not self.run_dir or not self.active_run_id or not self.run_start_time:
            raise RuntimeError("Cannot finish a run that was not started.")

        total = len(self.records)
        passed = sum(1 for r in self.records if r.passed)
        failed = total - passed
        pass_rate = round((passed / total * 100.0) if total > 0 else 0.0, 2)
        avg_iter = round((sum(r.iterations_used for r in self.records) / total) if total > 0 else 0.0, 2)
        total_duration = round(sum(r.duration_ms for r in self.records), 2)

        summary = RunSummary(
            run_id=self.active_run_id,
            tag=self.active_tag,
            timestamp=self.run_start_time.isoformat(),
            total_evals=total,
            passed_evals=passed,
            failed_evals=failed,
            pass_rate_percent=pass_rate,
            avg_iterations=avg_iter,
            total_duration_ms=total_duration,
            evals=[
                {
                    "eval_id": r.eval_id,
                    "topic": r.topic,
                    "passed": r.passed,
                    "iterations": r.iterations_used,
                    "duration_ms": r.duration_ms,
                }
                for r in self.records
            ],
        )

        # 1. Write summary.json in run directory
        summary_file = self.run_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))

        # 2. Update global history_manifest.json
        manifest_file = self.base_dir / "history_manifest.json"
        manifest_data: list[dict[str, typing.Any]] = []
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception:
                manifest_data = []

        manifest_entry = {
            "run_id": summary.run_id,
            "tag": summary.tag,
            "timestamp": summary.timestamp,
            "total_evals": summary.total_evals,
            "passed_evals": summary.passed_evals,
            "pass_rate_percent": summary.pass_rate_percent,
            "avg_iterations": summary.avg_iterations,
            "total_duration_ms": summary.total_duration_ms,
        }
        manifest_data.append(manifest_entry)

        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return summary
