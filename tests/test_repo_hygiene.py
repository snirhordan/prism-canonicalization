"""The repository ships no machine-specific paths and no tool attribution."""

import os
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FORBIDDEN = ["/home/" + "snirhordan", "Co-Authored" + "-By", "Generated " + "with"]
TEXT_SUFFIXES = (".py", ".md", ".toml", ".txt", ".cfg", ".yml", ".yaml", ".json", ".sh", ".tex",
                 ".gitignore", "LICENSE")


def _tracked_or_present_text_files():
    try:
        out = subprocess.check_output(["git", "-C", ROOT, "ls-files", "-co", "--exclude-standard"],
                                      stderr=subprocess.DEVNULL).decode().split("\n")
    except Exception:
        pytest.skip("not a git checkout")
    return [f for f in out if f and f.endswith(TEXT_SUFFIXES)]


def test_no_absolute_home_paths_or_attribution():
    bad = []
    for rel in _tracked_or_present_text_files():
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            continue
        text = open(path, encoding="utf-8", errors="ignore").read()
        bad += [f"{rel}: {pat}" for pat in FORBIDDEN if pat in text]
    assert not bad, "\n".join(bad)
