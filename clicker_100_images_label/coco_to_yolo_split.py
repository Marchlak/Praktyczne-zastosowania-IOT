import argparse
import json
import os
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def stem_from_filename(name: str) -> str:
    p = Path(name)
    return p.stem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="result.json")
    ap.add_argument("--images", default="images")
    ap.add_argument("--out", default="dataset_yolo")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {}
    for im in coco.get("images", []):
        images_by_id[im["id"]] = {
            "file_name": im["file_name"],
            "width": im["width"],
            "height": im["height"],
        }

    cats = coco.get("categories", [])
    cat_id_to_cls = {}
    for c in cats:
        name = str(c.get("name", "")).strip()
        if name.isdigit():
            cat_id_to_cls[c["id"]] = int(name)
        else:
            cat_id_to_cls[c["id"]] = c["id"]

    labels = {im_id: [] for im_id in images_by_id.keys()}

    for ann in coco.get("annotations", []):
        im_id = ann["image_id"]
        if im_id not in images_by_id:
            continue
        bbox = ann.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            continue

        iw = images_by_id[im_id]["width"]
        ih = images_by_id[im_id]["height"]

        xc = (x + w / 2.0) / iw
        yc = (y + h / 2.0) / ih
        wn = w / iw
        hn = h / ih

        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < wn <= 1 and 0 < hn <= 1):
            continue

        cls = cat_id_to_cls.get(ann["category_id"], ann["category_id"])
        labels[im_id].append((cls, xc, yc, wn, hn))

    out = Path(args.out)
    for p in [
        out / "images" / "train",
        out / "images" / "val",
        out / "labels" / "train",
        out / "labels" / "val",
    ]:
        p.mkdir(parents=True, exist_ok=True)

    im_ids = list(images_by_id.keys())
    random.Random(args.seed).shuffle(im_ids)
    n_val = max(1, int(round(len(im_ids) * args.val))) if len(im_ids) > 1 else 0
    val_ids = set(im_ids[:n_val])
    train_ids = [i for i in im_ids if i not in val_ids]

    def find_image_path(file_name: str) -> Path:
        p = Path(args.images) / file_name
        if p.exists():
            return p
        cand = list(Path(args.images).glob(stem_from_filename(file_name) + ".*"))
        cand = [c for c in cand if c.suffix.lower() in IMG_EXTS]
        if cand:
            return cand[0]
        return p

    def write_split(split: str, ids: list[int]):
        for im_id in ids:
            meta = images_by_id[im_id]
            src = find_image_path(meta["file_name"])
            if not src.exists():
                continue
            dst_img = out / "images" / split / src.name
            shutil.copy2(src, dst_img)

            dst_lbl = out / "labels" / split / (dst_img.stem + ".txt")
            lines = []
            for cls, xc, yc, wn, hn in labels.get(im_id, []):
                lines.append(f"{int(cls)} {xc:.8f} {yc:.8f} {wn:.8f} {hn:.8f}")
            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    write_split("train", train_ids)
    write_split("val", list(val_ids))

    names = [str(i) for i in range(10)]
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {names}\n",
        encoding="utf-8",
    )

    print(f"OK: {out.resolve()}")
    print(f"train images: {len(list((out/'images'/'train').glob('*')))}")
    print(f"val images: {len(list((out/'images'/'val').glob('*')))}")
    print(f"data.yaml: {data_yaml.resolve()}")

if __name__ == "__main__":
    main()
