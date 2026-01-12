import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def read_label_kps_norm(label_path: Path):
    if not label_path.exists():
        return None, None
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None, None
    parts = lines[0].split()
    vals = [float(x) for x in parts]
    if len(vals) != 11:
        return None, None
    xc, yc, w, h = vals[1], vals[2], vals[3], vals[4]
    kps = np.array(vals[5:], dtype=np.float32).reshape(3, 2)
    bbox = np.array([xc, yc, w, h], dtype=np.float32)
    return bbox, kps


def norm_to_px_kps(kps_norm, img_w, img_h):
    k = kps_norm.copy()
    k[:, 0] *= float(img_w)
    k[:, 1] *= float(img_h)
    return k


def norm_to_px_bbox(bbox_norm, img_w, img_h):
    xc, yc, w, h = bbox_norm.tolist()
    x1 = (xc - w / 2.0) * float(img_w)
    y1 = (yc - h / 2.0) * float(img_h)
    x2 = (xc + w / 2.0) * float(img_w)
    y2 = (yc + h / 2.0) * float(img_h)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def infer_pred_kps_px(model, img_path: Path, conf: float):
    r = model.predict(str(img_path), conf=conf, verbose=False)[0]
    if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
        return None
    idx = 0
    if r.boxes is not None and r.boxes.conf is not None and len(r.boxes.conf) > 0:
        idx = int(torch.argmax(r.boxes.conf).cpu().item())
    k = r.keypoints.xy[idx].cpu().numpy()
    return k.astype(np.float32)


def load_eval_csv(csv_path: Path, img_col: str | None, pred_col: str | None, gt_col: str | None):
    if csv_path is None or not csv_path.exists():
        return {}, None, None, None

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return {}, None, None, None

        def pick_col(cands):
            for c in cands:
                if c in fieldnames:
                    return c
            return None

        if img_col is None:
            img_col = pick_col(["image", "img", "path", "file", "filename"])
            if img_col is None:
                for c in fieldnames:
                    lc = c.lower()
                    if "img" in lc or "file" in lc or "path" in lc:
                        img_col = c
                        break
            if img_col is None:
                img_col = fieldnames[0]

        if pred_col is None:
            for c in fieldnames:
                lc = c.lower()
                if "pred" in lc and ("g" in lc or "gram" in lc or "waga" in lc or "weight" in lc):
                    pred_col = c
                    break
            if pred_col is None:
                for c in fieldnames:
                    if "pred" in c.lower():
                        pred_col = c
                        break

        if gt_col is None:
            for c in fieldnames:
                lc = c.lower()
                if ("gt" in lc or "true" in lc) and ("g" in lc or "gram" in lc or "waga" in lc or "weight" in lc):
                    gt_col = c
                    break
            if gt_col is None:
                for c in fieldnames:
                    lc = c.lower()
                    if lc.startswith("gt") or "gt" in lc or "true" in lc:
                        gt_col = c
                        break

        m = {}
        for row in reader:
            p = row.get(img_col, "") or ""
            base = os.path.basename(p)
            stem = os.path.splitext(base)[0]

            item = {"row": row}
            if pred_col and row.get(pred_col, "") != "":
                try:
                    item["pred_g"] = float(row[pred_col])
                except:
                    pass
            if gt_col and row.get(gt_col, "") != "":
                try:
                    item["gt_g"] = float(row[gt_col])
                except:
                    pass
            if "pred_g" in item and "gt_g" in item:
                item["err_g"] = item["pred_g"] - item["gt_g"]
                item["abs_err_g"] = abs(item["err_g"])

            if base:
                m[base] = item
            if stem:
                m[stem] = item

        return m, img_col, pred_col, gt_col


def draw_kps(img, kps_px, color, radius=5, thickness=-1, line_thickness=2):
    k = [(int(round(x)), int(round(y))) for x, y in kps_px.tolist()]
    for (x, y) in k:
        cv2.circle(img, (x, y), radius, color, thickness)
    if len(k) >= 2:
        cv2.line(img, k[0], k[1], color, line_thickness)
    if len(k) >= 3:
        cv2.line(img, k[0], k[2], color, line_thickness)
    return img


def put_text_block(img, lines, org=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.8, thickness=2):
    x, y = org
    for ln in lines:
        cv2.putText(img, ln, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, ln, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += int(round(28 * scale + 12))
    return img


def write_index_html(out_dir: Path, rel_paths):
    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'><title>viz</title>")
    html.append("<style>body{font-family:system-ui;margin:20px} .g{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px} img{max-width:100%;height:auto;border-radius:10px;border:1px solid #ddd}</style>")
    html.append("</head><body><h1>viz</h1><div class='g'>")
    for rp in rel_paths:
        html.append(f"<a href='{rp}'><img src='{rp}' loading='lazy'></a>")
    html.append("</div></body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--img_col", default=None)
    ap.add_argument("--pred_col", default=None)
    ap.add_argument("--gt_col", default=None)
    ap.add_argument("--out_dir", default="viz_out")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--sort", default="abs_err", choices=["abs_err", "name"])
    args = ap.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels) if args.labels else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_map = {}
    img_col = pred_col = gt_col = None
    if args.csv:
        eval_map, img_col, pred_col, gt_col = load_eval_csv(Path(args.csv), args.img_col, args.pred_col, args.gt_col)

    model = YOLO(args.model)

    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
        imgs.extend(sorted(img_dir.glob(ext)))
    imgs = sorted(imgs)

    matched = 0
    for p in imgs:
        if p.name in eval_map or p.stem in eval_map:
            matched += 1

    if args.csv:
        print(f"CSV cols: img_col={img_col} pred_col={pred_col} gt_col={gt_col}")
        print(f"CSV match: {matched}/{len(imgs)}")

    items = []
    for p in imgs:
        e = eval_map.get(p.name) or eval_map.get(p.stem) or {}
        abs_err = e.get("abs_err_g", None)
        items.append((p, abs_err))

    if args.sort == "abs_err":
        items.sort(key=lambda t: (-1 if t[1] is None else t[1]), reverse=True)
    else:
        items.sort(key=lambda t: t[0].name)

    if args.topk and args.topk > 0:
        items = items[: args.topk]

    rels = []

    for img_path, _ in items:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        e = eval_map.get(img_path.name) or eval_map.get(img_path.stem) or {}
        pred_g = e.get("pred_g", None)
        gt_g = e.get("gt_g", None)
        err_g = e.get("err_g", None)
        abs_err_g = e.get("abs_err_g", None)

        pred_kps = infer_pred_kps_px(model, img_path, args.conf)

        gt_kps_px = None
        gt_bbox_px = None
        if lbl_dir is not None:
            bbox_norm, kps_norm = read_label_kps_norm(lbl_dir / (img_path.stem + ".txt"))
            if kps_norm is not None:
                gt_kps_px = norm_to_px_kps(kps_norm, w, h)
            if bbox_norm is not None:
                gt_bbox_px = norm_to_px_bbox(bbox_norm, w, h)

        vis = img.copy()

        if gt_bbox_px is not None:
            x1, y1, x2, y2 = gt_bbox_px
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if gt_kps_px is not None:
            vis = draw_kps(vis, gt_kps_px, (255, 0, 0), radius=6)

        if pred_kps is not None:
            vis = draw_kps(vis, pred_kps, (0, 255, 0), radius=6)

        lines = [img_path.name]
        lines.append("pred: " + ("N/A" if pred_g is None else f"{pred_g:.1f} g"))
        lines.append("gt:   " + ("N/A" if gt_g is None else f"{gt_g:.1f} g"))
        if err_g is not None and abs_err_g is not None:
            lines.append(f"err:  {err_g:+.1f} g | abs: {abs_err_g:.1f} g")

        vis = put_text_block(vis, lines, org=(10, 30), scale=0.8)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), vis)
        rels.append(img_path.name)

    write_index_html(out_dir, rels)


if __name__ == "__main__":
    main()
