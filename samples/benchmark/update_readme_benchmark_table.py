#!/usr/bin/env python3

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
README_PATH = ROOT / "readme.md"


@dataclass(frozen=True)
class BenchmarkTarget:
    framework: str
    script: str


TARGETS = [
    BenchmarkTarget(framework="Backtrader", script="main_backtrader.py"),
    BenchmarkTarget(framework="Backtrader-next", script="main_backtrader_next.py"),
    BenchmarkTarget(framework="Backtesting", script="main_backtesting.py"),
    BenchmarkTarget(framework="Aptrade", script="main_aptrade.py"),
]


TIME_PATTERN = re.compile(r"Execution time .*?:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds", re.IGNORECASE)


def run_target(target: BenchmarkTarget) -> float:
    cmd = ["uv", "run", target.script]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        details = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(
            f"Failed running {target.script} (exit code {result.returncode}).\n{details}"
        )

    output = (result.stdout + "\n" + result.stderr).strip()
    match = TIME_PATTERN.search(output)
    if not match:
        raise RuntimeError(
            f"Could not find execution time in output from {target.script}.\n{output}"
        )

    return float(match.group(1))


def relative_speed_label(framework: str, execution_time: float, baseline_time: float) -> str:
    if framework == "Backtrader":
        return "Baseline"

    return f"{baseline_time / execution_time:.1f}x faster than Backtrader"
    

def build_table(times: dict[str, float]) -> str:
    baseline = times["Backtrader"]
    lines = [
        "| Framework | Execution Time | Relative Speed |",
        "|---|---|---|",
    ]

    for target in TARGETS:
        value = times[target.framework]
        relative = relative_speed_label(target.framework, value, baseline)
        lines.append(f"| {target.framework} | {value:.2f} sec | {relative} |")

    return "\n".join(lines)


def update_readme(table: str) -> None:
    text = README_PATH.read_text(encoding="utf-8")
    marker = "### 3. Performance Results"
    marker_index = text.rfind(marker)
    if marker_index == -1:
        raise RuntimeError(f"Could not find section marker '{marker}' in {README_PATH}")

    prefix = text[:marker_index]
    suffix = f"{marker}\n\n{table}\n"
    README_PATH.write_text(prefix + suffix, encoding="utf-8")


def main() -> None:
    times: dict[str, float] = {}
    for target in TARGETS:
        elapsed = run_target(target)
        times[target.framework] = elapsed
        print(f"{target.framework}: {elapsed:.4f} seconds")

    table = build_table(times)
    update_readme(table)
    print(f"\nUpdated table in {README_PATH}")


if __name__ == "__main__":
    main()