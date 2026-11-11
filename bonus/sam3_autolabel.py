"""Day 3 bonus: SAM 3 auto-labelling tool -> CVAT-importable pre-annotations.

Pipeline (per task in data/manifest.json):
  1. For every class in the task's classes.json, prompt SAM 3 with one or more text
     prompts (PROMPTS below) and collect instance masks + confidence scores.
  2. Post-process by task type:
       semantic  -> per-pixel argmax over class score maps; "blocker" prompts for
                    objects outside the task schema (cars, grass, walls...) keep those
                    pixels unlabeled instead of leaking into road/vegetation.
       instance  -> one mask per detected object, duplicate removal (same object found
                    by two prompts or two classes).
       panoptic  -> stuff = one merged region per class, things = one mask per object,
                    things painted over stuff so the result has no overlaps.
  3. Write exactly the archive layout CVAT itself exports, so the result can be
     imported with Actions -> Upload annotations:
       semantic            -> "Segmentation mask 1.1" ZIP
       instance / panoptic -> "COCO 1.0" ZIP (uncompressed RLE, iscrowd=1, like CVAT)

The SAM 3 part needs a GPU runtime (Google Colab) and access to the gated
facebook/sam3 checkpoint. The post-processing / export part only needs numpy + PIL,
so it can be tested without a GPU (see bonus/test_export.py).
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Prompt configuration: task class name -> SAM 3 text prompts
# ---------------------------------------------------------------------------
PROMPTS = {
    "road": ["road", "asphalt street"],
    "sidewalk": ["sidewalk", "pavement"],
    "building": ["building", "house"],
    "vegetation": ["tree", "bush", "hedge"],
    "sky": ["sky"],
    "pole": ["pole", "street light pole", "traffic signal pole"],
    "traffic sign": ["traffic sign", "road sign"],
    "traffic light": ["traffic light"],
    "person": ["person"],
    "bicycle": ["bicycle"],
    "car": ["car"],
    "motorcycle": ["motorcycle", "scooter"],
    "bus": ["bus"],
    "truck": ["truck"],
}
# Objects that are NOT in a semantic task's schema. If SAM 3 is more confident a pixel
# is one of these, the pixel stays unlabeled (Cityscapes/BDD convention: ignore).
BLOCKERS = ["car", "bus", "truck", "person", "motorcycle", "grass", "lawn",
            "wall", "fence", "traffic light", "dirt"]
STUFF = {"road", "sidewalk", "building", "vegetation", "sky"}

SCORE_THR = {"default": 0.40, "stuff": 0.30, "semantic": 0.30}
MIN_AREA = 30            # pixels; drop specks
DUP_IOU_SAME = 0.70      # same class, two prompts found the same object
DUP_IOU_CROSS = 0.85     # two classes claim the same object -> keep higher score


@dataclass
class Det:
    label: str
    mask: np.ndarray      # bool HxW
    score: float
    prompt: str = ""


@dataclass
class TaskResult:
    task: str
    type: str
    frames: dict = field(default_factory=dict)   # file_name -> payload (see builders)
    seconds: float = 0.0


# ---------------------------------------------------------------------------
# SAM 3 wrapper (Hugging Face transformers)
# ---------------------------------------------------------------------------
class Sam3Runner:
    def __init__(self, model_id: str = "facebook/sam3", device: str | None = None):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32          # fp32 is the safest choice on a Colab T4 (~3.5 GB VRAM)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.dtype = dtype

    def segment(self, image: Image.Image, prompts: list[str], threshold: float = 0.25) -> list[tuple[str, np.ndarray, float]]:
        """Return [(prompt, bool mask HxW, score)] for every prompt."""
        torch = self.torch
        out = []
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        sizes = img_inputs.get("original_sizes").tolist()
        vision_embeds = None
        try:
            with torch.no_grad():
                vision_embeds = self.model.get_vision_features(
                    pixel_values=img_inputs.pixel_values.to(self.dtype))
        except Exception:
            vision_embeds = None          # fall back to a full forward pass per prompt
        for prompt in prompts:
            with torch.no_grad():
                if vision_embeds is not None:
                    text_inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
                    outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
                else:
                    inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
                    outputs = self.model(**inputs)
            res = self.processor.post_process_instance_segmentation(
                outputs, threshold=threshold, mask_threshold=0.5, target_sizes=sizes)[0]
            for m, s in zip(res["masks"], res["scores"]):
                out.append((prompt, m.detach().cpu().numpy().astype(bool), float(s)))
        return out


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------
def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter / np.logical_or(a, b).sum())


def dedupe(dets: list[Det]) -> list[Det]:
    """Drop duplicate detections of the same physical object."""
    keep: list[Det] = []
    for d in sorted(dets, key=lambda d: d.score, reverse=True):
        dup = False
        for k in keep:
            thr = DUP_IOU_SAME if k.label == d.label else DUP_IOU_CROSS
            if iou(k.mask, d.mask) >= thr:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


def collect(runner, image: Image.Image, labels: list[str], extra: list[str] = ()) -> list[Det]:
    """Run every prompt of every label (+ extra blocker prompts) on one image."""
    prompt_to_label = {}
    for lab in labels:
        for p in PROMPTS.get(lab, [lab]):
            prompt_to_label[p] = lab
    for p in extra:
        prompt_to_label.setdefault(p, "__blocker__")
    raw = runner.segment(image, list(prompt_to_label))
    return [Det(prompt_to_label[p], m, s, p) for p, m, s in raw if m.sum() >= MIN_AREA]


def build_semantic(dets: list[Det], labels: list[str], shape) -> np.ndarray:
    """-> HxW int map: index into labels, -1 = unlabeled."""
    names = labels + ["__blocker__"]
    best = np.zeros((len(names),) + tuple(shape), np.float32)
    for d in dets:
        if d.score < SCORE_THR["semantic"]:
            continue
        i = names.index(d.label)
        best[i] = np.where(d.mask, np.maximum(best[i], d.score), best[i])
    arg = best.argmax(0)
    has = best.max(0) > 0
    out = np.where(has, arg, -1)
    out[out == len(labels)] = -1          # blocker wins -> unlabeled
    return out


def build_instances(dets: list[Det], labels: list[str]) -> list[Det]:
    dets = [d for d in dets if d.label in labels and d.score >= SCORE_THR["default"]]
    return dedupe(dets)


def build_panoptic(dets: list[Det], labels: list[str], shape) -> list[Det]:
    """Non-overlapping list: one Det per stuff class + one per thing instance."""
    stuff_labels = [l for l in labels if l in STUFF]
    stuff_dets = [d for d in dets if d.label in stuff_labels and d.score >= SCORE_THR["stuff"]]
    sem = build_semantic(stuff_dets, stuff_labels, shape)
    things = build_instances([d for d in dets if d.label not in STUFF], labels)
    owner = np.full(shape, -1, np.int32)
    for i, _ in enumerate(stuff_labels):
        owner[sem == i] = i
    out = [Det(l, sem == i, 1.0, "stuff") for i, l in enumerate(stuff_labels)]
    for j, t in enumerate(sorted(things, key=lambda d: d.score)):   # higher score painted last
        owner[t.mask] = len(stuff_labels) + j
    things_sorted = sorted(things, key=lambda d: d.score)
    result = []
    for k, d in enumerate(out + things_sorted):
        m = owner == k
        if m.sum() >= MIN_AREA:
            result.append(Det(d.label, m, d.score, d.prompt))
    return result


# ---------------------------------------------------------------------------
# CVAT-format writers
# ---------------------------------------------------------------------------
def coco_rle(mask: np.ndarray) -> dict:
    """Uncompressed COCO RLE (column-major, starts with a run of zeros) - as CVAT exports."""
    flat = mask.flatten(order="F").astype(np.uint8)
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat[0] == 1:
        runs = [0] + runs
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": [int(r) for r in runs]}


def write_coco_zip(path: Path, labels: list[str], frames: dict[str, tuple[int, int, list[Det]]]):
    cats = [{"id": i + 1, "name": n, "supercategory": ""} for i, n in enumerate(labels)]
    images, anns, aid = [], [], 1
    for img_id, (fname, (h, w, dets)) in enumerate(sorted(frames.items()), start=1):
        images.append({"id": img_id, "width": w, "height": h, "file_name": fname, "license": 0,
                       "flickr_url": "", "coco_url": "", "date_captured": 0})
        for d in dets:
            ys, xs = np.nonzero(d.mask)
            if xs.size == 0:
                continue
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            anns.append({"id": aid, "image_id": img_id, "category_id": labels.index(d.label) + 1,
                         "segmentation": coco_rle(d.mask), "area": float(d.mask.sum()),
                         "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                         "iscrowd": 1, "attributes": {"occluded": False, "score": round(d.score, 4)}})
            aid += 1
    data = {"licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {"contributor": "", "date_created": "", "description": "SAM3 auto-label",
                     "url": "", "version": "", "year": ""},
            "categories": cats, "images": images, "annotations": anns}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations/instances_default.json", json.dumps(data))
    return len(anns)


def voc_color(i: int) -> tuple[int, int, int]:
    """Standard VOC palette colour for index i (what CVAT uses in SegmentationObject)."""
    r = g = b = 0
    c = i
    for j in range(8):
        r |= ((c >> 0) & 1) << (7 - j)
        g |= ((c >> 1) & 1) << (7 - j)
        b |= ((c >> 2) & 1) << (7 - j)
        c >>= 3
    return r, g, b


def write_segmask_zip(path: Path, labels: list[str], colors: dict, frames: dict[str, np.ndarray]):
    lm = ["# label:color_rgb:parts:actions", "background:0,0,0::"]
    lm += [f"{n}:{','.join(map(str, colors[n]))}::" for n in sorted(labels)]
    stems = sorted(Path(f).stem for f in frames)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("labelmap.txt", "\n".join(lm) + "\n")
        z.writestr("ImageSets/Segmentation/default.txt", "\n".join(stems) + "\n")
        for fname, sem in frames.items():
            stem = Path(fname).stem
            cls = np.zeros(sem.shape + (3,), np.uint8)
            obj = np.zeros(sem.shape + (3,), np.uint8)
            for i, n in enumerate(labels):
                m = sem == i
                if m.any():
                    cls[m] = colors[n]
                    obj[m] = voc_color(i + 1)
            for folder, arr in (("SegmentationClass", cls), ("SegmentationObject", obj)):
                buf = io.BytesIO(); Image.fromarray(arr).save(buf, "PNG")
                z.writestr(f"{folder}/{stem}.png", buf.getvalue())


def overlay(image: Image.Image, dets: list[Det], colors: dict) -> Image.Image:
    arr = np.array(image.convert("RGB")).astype(np.float32)
    rng = np.random.default_rng(0)
    for d in dets:
        col = np.array(colors.get(d.label, rng.integers(40, 255, 3)), np.float32)
        if d.label not in STUFF and d.prompt != "stuff":
            col = rng.integers(40, 255, 3).astype(np.float32)
        arr[d.mask] = 0.45 * arr[d.mask] + 0.55 * col
    return Image.fromarray(arr.astype(np.uint8))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_task(runner, repo: Path, task: str, info: dict, out_dir: Path) -> dict:
    base = repo / "data" / info["path"]
    classes = json.load(open(base / "classes.json"))
    labels, colors = classes["classes"], {k: tuple(v) for k, v in classes["colors"].items()}
    t0 = time.time()
    frames, summary = {}, {}
    (out_dir / "preview").mkdir(parents=True, exist_ok=True)
    for img_path in sorted((base / "images").glob("*.jpg")):
        image = Image.open(img_path).convert("RGB")
        shape = (image.height, image.width)
        runner.current_image = img_path.name      # used only by the offline test runner
        if info["type"] == "semantic":
            blockers = [b for b in BLOCKERS if b not in labels]
            dets = collect(runner, image, labels, blockers)
            sem = build_semantic(dets, labels, shape)
            frames[img_path.name] = sem
            vis = [Det(l, sem == i, 1.0, "stuff") for i, l in enumerate(labels)]
            summary[img_path.name] = {l: round(float((sem == i).mean() * 100), 1) for i, l in enumerate(labels)}
            summary[img_path.name]["unlabeled"] = round(float((sem < 0).mean() * 100), 1)
        else:
            dets = collect(runner, image, labels)
            dets = build_panoptic(dets, labels, shape) if info["type"] == "panoptic" else build_instances(dets, labels)
            frames[img_path.name] = (image.height, image.width, dets)
            vis = dets
            counts = {}
            for d in dets:
                counts[d.label] = counts.get(d.label, 0) + 1
            summary[img_path.name] = counts
        overlay(image, vis, colors).save(out_dir / "preview" / f"{task}__{img_path.stem}.jpg", quality=85)
    zpath = out_dir / f"{task}.zip"
    if info["type"] == "semantic":
        write_segmask_zip(zpath, labels, colors, frames)
    else:
        write_coco_zip(zpath, labels, frames)
    return {"task": task, "type": info["type"], "seconds": round(time.time() - t0, 1), "per_image": summary}


def run_all(repo: str | Path, out_dir: str | Path, tasks: list[str] | None = None, runner=None) -> list[dict]:
    repo, out_dir = Path(repo), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(repo / "data/manifest.json"))["tasks"]
    runner = runner or Sam3Runner()
    report = []
    for task, info in manifest.items():
        if tasks and task not in tasks:
            continue
        r = run_task(runner, repo, task, info, out_dir)
        print(f"{task:16s} {r['type']:9s} {r['seconds']:6.1f}s  {r['per_image']}", flush=True)
        report.append(r)
    (out_dir / "run_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
