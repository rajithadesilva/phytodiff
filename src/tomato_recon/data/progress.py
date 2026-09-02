"""Shared command-line progress rendering for dataset conversion scripts."""

from __future__ import annotations

import sys
from typing import Any


class ProgressPrinter:
    """Render compact live progress in a terminal and readable CI logs otherwise."""

    def __init__(self) -> None:
        self.stream = sys.stderr
        self.interactive = self.stream.isatty()
        self.line_width = 0

    def _write_instance_line(self, line: str, *, final: bool = False) -> None:
        if self.interactive:
            width = max(self.line_width, len(line))
            print(
                f"\r{line:<{width}}",
                end="\n" if final else "",
                file=self.stream,
                flush=True,
            )
            self.line_width = 0 if final else width
        else:
            print(line, file=self.stream, flush=True)

    def __call__(self, update: dict[str, Any]) -> None:
        phase = str(update["phase"])
        if phase == "loading_splits":
            splits = ", ".join(update["splits"])
            print(
                f"Loading {update['dataset']} splits ({splits}) from {update['raw_root']}",
                file=self.stream,
                flush=True,
            )
            print(f"Output dataset: {update['output_root']}", file=self.stream, flush=True)
            return
        if phase == "discovered":
            counts = ", ".join(
                f"{name}={count}" for name, count in update["split_counts"].items()
            )
            ignored = int(update.get("ignored_count", 0))
            suffix = f"; ignored incomplete={ignored}" if ignored else ""
            print(
                f"Discovered {update['total']} complete point-cloud instances ({counts}){suffix}",
                file=self.stream,
                flush=True,
            )
            return
        if phase == "finished":
            ignored = (
                f"{int(update['ignored_count'])} incomplete ignored, "
                if "ignored_count" in update
                else ""
            )
            print(
                "Preprocessing finished: "
                f"{update['new_instance_count']} new, "
                f"{update['resumed_instance_count']} resumed, "
                f"{update['skipped_instance_count']} unchanged, "
                f"{ignored}"
                f"next folder is plant_{update['next_plant_number']:06d}",
                file=self.stream,
                flush=True,
            )
            return

        prefix = (
            f"[{update['current']:>{len(str(update['total']))}}/{update['total']}] "
            f"{update['source_instance_id']} ({update['split']})"
        )
        target = update.get("instance_id")
        if phase == "checking":
            self._write_instance_line(f"{prefix}: checking inputs")
        elif phase == "processing":
            self._write_instance_line(f"{prefix} -> {target}: {update['action']}")
        elif phase == "writing":
            self._write_instance_line(f"{prefix} -> {target}: writing cache")
        elif phase == "skipped":
            elapsed = float(update["elapsed_seconds"])
            self._write_instance_line(
                f"{prefix} -> {target}: unchanged, skipped ({elapsed:.1f}s)", final=True
            )
        elif phase == "completed":
            elapsed = float(update["elapsed_seconds"])
            self._write_instance_line(
                f"{prefix} -> {target}: complete "
                f"({update['point_count']} points, {elapsed:.1f}s)",
                final=True,
            )
