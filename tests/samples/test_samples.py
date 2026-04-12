"""
Run each sample script that has a reference.txt and compare stdout to it.
A new sample is picked up automatically once its reference.txt is added.
"""
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
SAMPLES_DIR = ROOT_DIR / "samples"


def _cases():
    for ref in sorted(SAMPLES_DIR.glob("*/reference.txt")):
        folder = ref.parent
        script = folder / f"{folder.name}.py"
        if script.exists():
            yield pytest.param(script, ref, id=folder.name)


import pytest


@pytest.mark.parametrize("script,reference", _cases())
def test_sample_output(script, reference):
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == reference.read_text()
