"""Offline test of the export pipeline (no GPU, no SAM 3).

A fake runner answers every text prompt with the masks of that class taken from an
existing CVAT export folder (default: submissions/). Running the full pipeline on it
must give: (1) archives that pass scripts/inspect_submissions.py, and (2) ~100%
agreement with the source export in bonus/compare.py.

usage: python bonus/test_export.py [--src submissions] [--out bonus_test_out]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bonus"))
import lab_utils as L  # noqa: E402
import sam3_autolabel as S  # noqa: E402


class FakeRunner:
    """Returns ground masks from a CVAT export as if SAM 3 had predicted them."""

    def __init__(self, src: Path):
        self.src = src; self.task = None; self.current_image = None; self.cache = {}

    def load(self, task, info):
        self.task = task
        classes = json.load(open(ROOT / "data" / info["path"] / "classes.json"))
        z = self.src / f"{task}.zip"
        per_image = {}
        if info["type"] == "semantic":
            name2id = classes["trainid"]
            for stem, lab in L.load_semantic_submission(z, name2id).items():
                per_image[stem] = [(n, lab == i) for n, i in name2id.items()]
        else:
            coco = L._find_coco_json(L._read_members(z))
            for stem, items in L._instances_by_image(coco, set(classes["classes"])).items():
                per_image[stem] = [(c, m) for c, m, _ in items]
        self.cache = per_image

    def segment(self, image, prompts, threshold=0.25):
        stem = Path(self.current_image).stem
        out = []
        for p in prompts:
            for lab, m in self.cache.get(stem, []):
                if p in S.PROMPTS.get(lab, [lab])[:1] and m.any():     # first prompt of the class only
                    out.append((p, m.astype(bool), 0.9))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "submissions"))
    ap.add_argument("--out", default=str(ROOT / "bonus_test_out"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(ROOT / "data/manifest.json"))["tasks"]
    runner = FakeRunner(Path(a.src))
    for task, info in manifest.items():
        if not (Path(a.src) / f"{task}.zip").is_file():
            continue
        runner.load(task, info)
        r = S.run_task(runner, ROOT, task, info, out)
        print(task, r["per_image"])
    print("\n== inspect_submissions on tool output ==")
    subprocess.run([sys.executable, str(ROOT / "scripts/inspect_submissions.py"), "--dir", str(out)], check=False)
    print("\n== agreement tool vs source (expect ~100%) ==")
    subprocess.run([sys.executable, str(ROOT / "bonus/compare.py"), "--tool", str(out), "--final", a.src], check=False)


if __name__ == "__main__":
    main()
