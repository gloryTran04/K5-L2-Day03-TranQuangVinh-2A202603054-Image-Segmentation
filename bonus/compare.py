"""Measure how much of the final (human-checked) annotation the SAM 3 tool produced.

usage:
  python bonus/compare.py --tool bonus_out --final submissions [--json bonus/compare.json]

For every task present in both folders it scores the TOOL export against the FINAL
export with the lab's own metric code (lab_utils.py):
  semantic  -> mIoU          instance -> mean matched IoU x recall@0.5     panoptic -> PQ
The number is AGREEMENT with the final annotation ("the tool already did X% of it"),
not correctness. When the official reference is installed, run scoring/score.py on
both folders as well to compare tool-vs-reference and final-vs-reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lab_utils as L  # noqa: E402


def agreement(task: str, info: dict, tool_zip: Path, final_zip: Path) -> dict:
    classes = json.load(open(ROOT / "data" / info["path"] / "classes.json"))
    if info["type"] == "semantic":
        name2id = classes["trainid"]; id2name = {v: k for k, v in name2id.items()}
        tool = L.load_semantic_submission(tool_zip, name2id)
        final = L.load_semantic_submission(final_zip, name2id)
        r = L.score_semantic(tool, final, id2name)
        return {"metric": "mIoU", "value": r["value"],
                "per_class": {k: (None if v is None else round(v, 3)) for k, v in r["per_class_iou"].items()}}
    cn = set(classes["classes"])
    tool = L._find_coco_json(L._read_members(tool_zip))
    final = L._find_coco_json(L._read_members(final_zip))
    if info["type"] == "panoptic":
        stuff = set(classes["stuff"])
        r = L.score_panoptic(L.build_pred_panoptic(tool, cn, stuff), L.build_pred_panoptic(final, cn, stuff), cn)
        return {"metric": "PQ", "value": r["value"], "SQ": round(r["SQ"], 3), "RQ": round(r["RQ"], 3)}
    r = L.score_instance(tool, final, cn)
    return {"metric": "matchedIoU x recall", "value": r["value"], "recall@0.5": round(r["recall@0.5"], 3),
            "precision@0.5": round(r["precision@0.5"], 3), "tool_objects": r["submitted_instances"],
            "final_objects": r["gt_instances"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="bonus_out")
    ap.add_argument("--final", default="submissions")
    ap.add_argument("--json")
    a = ap.parse_args()
    manifest = json.load(open(ROOT / "data/manifest.json"))["tasks"]
    rows = {}
    for task, info in manifest.items():
        t, f = Path(a.tool) / f"{task}.zip", Path(a.final) / f"{task}.zip"
        if t.is_file() and f.is_file():
            rows[task] = agreement(task, info, t, f)
    print(f"{'task':16s} {'metric':22s} {'agreement':>9s}")
    for task, r in rows.items():
        print(f"{task:16s} {r['metric']:22s} {r['value'] * 100:8.1f}%")
    if rows:
        w = {t: manifest[t]["weight"] for t in rows}
        avg = sum(rows[t]["value"] * w[t] for t in rows) / sum(w.values())
        print(f"{'weighted avg':16s} {'':22s} {avg * 100:8.1f}%")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
