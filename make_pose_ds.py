import os
import json
import random
import shutil
from pathlib import Path

def pick_filename(s):
    if "?d=" in s:
        s = s.split("?d=")[-1]
    return os.path.basename(s)

def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def main():
    ls_json = Path("waga-analogowa-labele.json")
    images_dir = Path("WagaAnalogowaDataSet")
    out_dir = Path("waga_pose_ds")
    val_ratio = 0.2
    seed = 42
    rect_label = "Tarcza"
    kp_order = ("zero", "center", "tip")

    if out_dir.exists():
        shutil.rmtree(out_dir)

    (out_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (out_dir / "images/val").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels/train").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels/val").mkdir(parents=True, exist_ok=True)

    tasks = json.loads(ls_json.read_text(encoding="utf-8"))
    items = []

    for t in tasks:
        img_field = (t.get("data") or {}).get("image")
        if not img_field:
            continue
        filename = pick_filename(img_field)
        src_img = images_dir / filename
        if not src_img.exists():
            continue

        ann_list = t.get("annotations") or t.get("completions") or []
        if not ann_list:
            continue
        results = (ann_list[0].get("result") or [])
        if not results:
            continue

        bbox = None
        kps = {}

        for r in results:
            rtype = r.get("type")
            val = r.get("value") or {}

            if rtype == "rectanglelabels":
                labs = val.get("rectanglelabels") or []
                if labs and labs[0].strip().lower() == rect_label.lower():
                    x = float(val.get("x", 0.0)) / 100.0
                    y = float(val.get("y", 0.0)) / 100.0
                    w = float(val.get("width", 0.0)) / 100.0
                    h = float(val.get("height", 0.0)) / 100.0
                    xc = x + w / 2.0
                    yc = y + h / 2.0
                    bbox = (clamp01(xc), clamp01(yc), clamp01(w), clamp01(h))

            if rtype == "keypointlabels":
                labs = val.get("keypointlabels") or []
                if not labs:
                    continue
                lab = labs[0].strip().lower()
                x = float(val.get("x", 0.0)) / 100.0
                y = float(val.get("y", 0.0)) / 100.0
                kps[lab] = (clamp01(x), clamp01(y))

        if bbox is None:
            bbox = (0.5, 0.5, 1.0, 1.0)

        items.append((src_img, filename, bbox, kps))

    if not items:
        raise SystemExit("0 elementów: JSON nie pasuje albo images_dir nie zawiera plików z JSON.")

    random.seed(seed)
    random.shuffle(items)
    split = int(len(items) * (1.0 - val_ratio))
    train_items = items[:split]
    val_items = items[split:]

    def write_split(name, split_items):
        for src_img, filename, bbox, kps in split_items:
            dst_img = out_dir / f"images/{name}/{filename}"
            dst_lbl = out_dir / f"labels/{name}/{Path(filename).stem}.txt"

            shutil.copy2(src_img, dst_img)

            xc, yc, w, h = bbox
            parts = ["0", f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}"]

            for k in kp_order:
                if k in kps:
                    x, y = kps[k]
                    v = 2
                else:
                    x, y, v = 0.0, 0.0, 0
                parts += [f"{x:.6f}", f"{y:.6f}", str(v)]

            dst_lbl.write_text(" ".join(parts) + "\n", encoding="utf-8")

    write_split("train", train_items)
    write_split("val", val_items)

    (out_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "names: [tarcza]",
                "kpt_shape: [3,2]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("train", len(train_items), "val", len(val_items))

if __name__ == "__main__":
    main()
