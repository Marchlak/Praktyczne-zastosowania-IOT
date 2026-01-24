import argparse
import json
import os
from pathlib import Path
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def iter_images(src: Path):
    if src.is_file():
        if src.suffix.lower() in IMG_EXTS:
            yield src
        return
    for p in sorted(src.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def decode_4digits(result, conf_min: float, force4: bool):
    items = []
    if result.boxes is None or len(result.boxes) == 0:
        return "", []

    for b in result.boxes:
        conf = float(b.conf[0].item())
        if conf < conf_min:
            continue
        cls = int(b.cls[0].item())
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        xc = (x1 + x2) / 2.0
        items.append({"cls": cls, "conf": conf, "xyxy": [x1, y1, x2, y2], "xc": xc})

    if not items:
        return "", []

    items.sort(key=lambda d: d["xc"])

    if force4:
        if len(items) != 4:
            return "", items
        code = "".join(str(d["cls"]) for d in items)
        return code, items

    if len(items) > 4:
        items = sorted(items, key=lambda d: d["conf"], reverse=True)[:4]
        items.sort(key=lambda d: d["xc"])

    code = "".join(str(d["cls"]) for d in items) if len(items) == 4 else ""
    return code, items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/train2/weights/best.pt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="pred_out")
    ap.add_argument("--force4", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.source)
    imgs = list(iter_images(src))
    if not imgs:
        raise SystemExit("No images found")

    summary = []

    for img_path in imgs:
        res = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            save=True,
            project=str(out_dir),
            name="vis",
            exist_ok=True,
        )[0]

        code, dets = decode_4digits(res, conf_min=args.conf, force4=args.force4)

        item = {
            "image": str(img_path),
            "code": code,
            "detections": [
                {"cls": d["cls"], "conf": d["conf"], "xyxy": d["xyxy"]}
                for d in dets
            ],
        }
        summary.append(item)

        print(f"{img_path.name}: {code if code else '[no-4digits]'} (n={len(dets)})")

    (out_dir / "predictions.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved visuals to: {out_dir / 'vis'}")
    print(f"Saved json to: {out_dir / 'predictions.json'}")

if __name__ == "__main__":
    main()
