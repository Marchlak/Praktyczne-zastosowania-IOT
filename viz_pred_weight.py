import argparse
import csv
import glob
import math
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def list_images(p: str) -> list[str]:
    if os.path.isdir(p):
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
        out = []
        for e in exts:
            out.extend(glob.glob(os.path.join(p, e)))
        return sorted(out)
    return [p]


def load_label_kpts(label_file: str, img_w: int, img_h: int, kpt_count: int) -> np.ndarray | None:
    if not label_file or not os.path.isfile(label_file):
        return None
    with open(label_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if not lines:
        return None
    parts = lines[0].split()
    vals = [float(x) for x in parts]
    need = 5 + kpt_count * 2
    if len(vals) < need:
        return None
    kpts = []
    off = 5
    for i in range(kpt_count):
        x = vals[off + 2 * i] * img_w
        y = vals[off + 2 * i + 1] * img_h
        kpts.append([x, y])
    return np.array(kpts, dtype=np.float32)


def cw_angle_deg(center: np.ndarray, zero: np.ndarray, needle: np.ndarray) -> float | None:
    v0 = zero - center
    v1 = needle - center
    if float(np.hypot(v0[0], v0[1])) < 1e-6 or float(np.hypot(v1[0], v1[1])) < 1e-6:
        return None
    v0m = np.array([v0[0], -v0[1]], dtype=np.float32)
    v1m = np.array([v1[0], -v1[1]], dtype=np.float32)
    a0 = math.atan2(float(v0m[1]), float(v0m[0]))
    a1 = math.atan2(float(v1m[1]), float(v1m[0]))
    return (math.degrees(a0 - a1) % 360.0)


def weight_from_angle(angle_deg: float, cap_g: float, sweep_deg: float, wrap: bool) -> float:
    if sweep_deg <= 0:
        sweep_deg = 360.0
    if sweep_deg >= 360.0:
        a = angle_deg
    else:
        a = (angle_deg % sweep_deg) if wrap else min(max(angle_deg, 0.0), sweep_deg)
    return cap_g * (a / sweep_deg)


def pick_best_detection(res) -> int | None:
    if res is None or res.keypoints is None:
        return None
    n = int(res.keypoints.xy.shape[0]) if hasattr(res.keypoints, "xy") else 0
    if n <= 0:
        return None
    if res.boxes is not None and len(res.boxes) == n and hasattr(res.boxes, "conf"):
        s = res.boxes.conf.detach().cpu().numpy().reshape(-1)
        return int(np.argmax(s))
    if hasattr(res.keypoints, "conf") and res.keypoints.conf is not None:
        s = res.keypoints.conf.detach().cpu().numpy()
        s = s.mean(axis=1).reshape(-1)
        return int(np.argmax(s))
    return 0


def draw_text_block(img: np.ndarray, lines: list[str], x: int, y: int, font_scale: float, thickness: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = max(6, int(6 * font_scale))
    line_gap = max(6, int(10 * font_scale))
    cur_y = y
    for line in lines:
        (tw, th), base = cv2.getTextSize(line, font, font_scale, thickness)
        rect_x1 = max(0, x - pad)
        rect_y1 = max(0, cur_y - th - pad)
        rect_x2 = min(img.shape[1] - 1, x + tw + pad)
        rect_y2 = min(img.shape[0] - 1, cur_y + base + pad)
        overlay = img.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0.0, img)
        cv2.putText(img, line, (x, cur_y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, cur_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cur_y += th + base + line_gap


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def write_index(out_dir: Path, items: list[dict]) -> None:
    items_sorted = sorted(items, key=lambda r: (1e18 if r["abs_err_g"] == "" else float(r["abs_err_g"])), reverse=True)
    rows = []
    for r in items_sorted:
        cap = []
        cap.append(r["image"])
        if r["pred_g"] != "":
            cap.append(f"pred: {int(round(float(r['pred_g'])))} g ({float(r['pred_angle_deg']):.1f}°)")
        else:
            cap.append("pred: -")
        if r["gt_g"] != "":
            cap.append(f"gt: {int(round(float(r['gt_g'])))} g ({float(r['gt_angle_deg']):.1f}°)")
            if r["abs_err_g"] != "":
                cap.append(f"abs err: {float(r['abs_err_g']):.1f} g")
        caption = html_escape(" | ".join(cap))
        fn = html_escape(Path(r["out_image"]).name)
        rows.append(
            f"""
            <div class="card">
              <a href="{fn}" target="_blank" rel="noopener">
                <img src="{fn}" loading="lazy" />
              </a>
              <div class="cap">{caption}</div>
            </div>
            """
        )

    html = f"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>viz</title>
<style>
  body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial; background:#0b0b0c; color:#eaeaea; }}
  .top {{ position: sticky; top: 0; background: rgba(11,11,12,0.92); backdrop-filter: blur(6px); padding: 12px 16px; border-bottom: 1px solid #222; }}
  .top h1 {{ margin: 0; font-size: 16px; font-weight: 650; }}
  .top .sub {{ opacity: 0.8; font-size: 13px; margin-top: 2px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; padding: 12px; }}
  .card {{ background:#111115; border: 1px solid #222; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 22px rgba(0,0,0,0.35); }}
  .card img {{ width: 100%; display: block; }}
  .cap {{ padding: 10px 12px; font-size: 13px; line-height: 1.35; word-break: break-word; }}
  a {{ color: inherit; text-decoration: none; }}
</style>
</head>
<body>
  <div class="top">
    <h1>Wizualizacja predykcji</h1>
    <div class="sub">posortowane malejąco po błędzie (jeśli jest GT)</div>
  </div>
  <div class="grid">
    {''.join(rows)}
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--out_dir", default="viz_val")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--cap_g", type=float, default=5000.0)
    ap.add_argument("--sweep_deg", type=float, default=360.0)
    ap.add_argument("--wrap", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default=None)

    ap.add_argument("--kpt_center", type=int, default=1)
    ap.add_argument("--kpt_zero", type=int, default=0)
    ap.add_argument("--kpt_needle", type=int, default=2)
    ap.add_argument("--kpt_count", type=int, default=3)

    ap.add_argument("--line_thick", type=int, default=3)
    ap.add_argument("--pt_radius", type=int, default=6)
    ap.add_argument("--font_scale", type=float, default=1.8)
    ap.add_argument("--font_thick", type=int, default=4)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = list_images(args.images)
    if not imgs:
        raise SystemExit(f"no images in: {args.images}")

    model = YOLO(args.model)

    rows = []
    for img_path in imgs:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        pred = model.predict(
            source=img,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]

        bi = pick_best_detection(pred)
        pred_kpts = None
        if bi is not None and pred.keypoints is not None:
            pred_kpts = pred.keypoints.xy[bi].detach().cpu().numpy().astype(np.float32)

        h, w = img.shape[:2]
        label_kpts = None
        gt_g = None
        gt_deg = None
        if args.labels:
            lf = os.path.join(args.labels, Path(img_path).with_suffix(".txt").name)
            label_kpts = load_label_kpts(lf, w, h, args.kpt_count)
            if label_kpts is not None:
                c = label_kpts[args.kpt_center]
                z = label_kpts[args.kpt_zero]
                n = label_kpts[args.kpt_needle]
                gt_deg = cw_angle_deg(c, z, n)
                if gt_deg is not None:
                    gt_g = weight_from_angle(gt_deg, args.cap_g, args.sweep_deg, args.wrap)

        pred_g = None
        pred_deg = None
        if pred_kpts is not None and pred_kpts.shape[0] >= args.kpt_count:
            c = pred_kpts[args.kpt_center]
            z = pred_kpts[args.kpt_zero]
            n = pred_kpts[args.kpt_needle]
            pred_deg = cw_angle_deg(c, z, n)
            if pred_deg is not None:
                pred_g = weight_from_angle(pred_deg, args.cap_g, args.sweep_deg, args.wrap)

            cc = tuple(int(x) for x in c)
            zz = tuple(int(x) for x in z)
            nn = tuple(int(x) for x in n)

            cv2.line(img, cc, zz, (255, 0, 0), args.line_thick, cv2.LINE_AA)
            cv2.line(img, cc, nn, (0, 255, 0), args.line_thick, cv2.LINE_AA)

            cv2.circle(img, cc, args.pt_radius, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, zz, args.pt_radius, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(img, nn, args.pt_radius, (0, 255, 0), -1, cv2.LINE_AA)

        if label_kpts is not None:
            c = label_kpts[args.kpt_center]
            z = label_kpts[args.kpt_zero]
            n = label_kpts[args.kpt_needle]
            cc = tuple(int(x) for x in c)
            zz = tuple(int(x) for x in z)
            nn = tuple(int(x) for x in n)
            r = max(4, args.pt_radius - 2)
            cv2.circle(img, cc, r, (0, 128, 255), 2, cv2.LINE_AA)
            cv2.circle(img, zz, r, (0, 128, 255), 2, cv2.LINE_AA)
            cv2.circle(img, nn, r, (0, 128, 255), 2, cv2.LINE_AA)

        name = Path(img_path).name
        lines = [name]
        if pred_g is not None:
            lines.append(f"pred: {int(round(pred_g))} g  ({pred_deg:.1f}°)")
        else:
            lines.append("pred: -")
        if gt_g is not None:
            lines.append(f"gt:   {int(round(gt_g))} g  ({gt_deg:.1f}°)")
            if pred_g is not None:
                lines.append(f"abs err: {abs(pred_g - gt_g):.1f} g")

        fs = args.font_scale * max(1.0, w / 1200.0)
        draw_text_block(img, lines, 20, 70, fs, args.font_thick)

        out_path = out_dir / (Path(img_path).stem + "_vis.jpg")
        cv2.imwrite(str(out_path), img)

        rows.append(
            {
                "image": name,
                "pred_g": "" if pred_g is None else float(pred_g),
                "pred_angle_deg": "" if pred_deg is None else float(pred_deg),
                "gt_g": "" if gt_g is None else float(gt_g),
                "gt_angle_deg": "" if gt_deg is None else float(gt_deg),
                "abs_err_g": "" if (pred_g is None or gt_g is None) else float(abs(pred_g - gt_g)),
                "out_image": str(out_path),
            }
        )

    write_index(out_dir, rows)

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
                    "pred_g",
                    "pred_angle_deg",
                    "gt_g",
                    "gt_angle_deg",
                    "abs_err_g",
                    "out_image",
                ],
            )
            wri.writeheader()
            wri.writerows(rows)


if __name__ == "__main__":
    main()
