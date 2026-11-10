import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def decode_4digits_from_result(result, conf_min: float, force4: bool):
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


def draw_text_block(img, lines, x, y, font_scale, thickness):
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


def apply_rotate(frame, rotate):
    r = rotate % 360
    if r == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if r == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if r == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--rotate", type=int, default=0)

    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")

    ap.add_argument("--force4", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--ema", type=float, default=0.25)

    ap.add_argument("--save_dir", default="")
    args = ap.parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera: {args.cam}")

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    last_code = ""
    last_items = []
    last_ts = 0.0

    t0 = time.time()
    fps = 0.0
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)
        frame = apply_rotate(frame, args.rotate)

        vis = frame.copy()
        h, w = vis.shape[:2]

        do_infer = (args.every <= 1) or (frame_i % args.every == 0)
        if do_infer:
            res = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]

            code, items = decode_4digits_from_result(res, conf_min=args.conf, force4=args.force4)

            if code:
                if last_code == "":
                    last_code = code
                else:
                    a = float(args.ema)
                    a = min(max(a, 0.0), 1.0)
                    last_code = code if (np.random.rand() < a) else last_code
                last_items = items
                last_ts = time.time()
            else:
                last_items = items
                last_ts = time.time()

        for d in last_items:
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
            label = f"{d['cls']} {d['conf']:.2f}"
            cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        dt = time.time() - t0
        if dt >= 0.5:
            fps = (frame_i + 1) / dt

        age_ms = int((time.time() - last_ts) * 1000) if last_ts > 0 else -1
        code_show = last_code if last_code else "-"
        lines = [
            f"code: {code_show}",
            f"n: {len(last_items)}  conf>={args.conf:.2f}  imgsz:{args.imgsz}",
            f"fps: {fps:.1f}  age: {age_ms} ms",
            "q: quit   s: save",
        ]

        fs = 1.2 * max(1.0, w / 1200.0)
        draw_text_block(vis, lines, 20, 60, fs, 3)

        cv2.imshow("clicker_live", vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s") and save_dir is not None:
            out = save_dir / f"frame_{int(time.time()*1000)}_{code_show}.jpg"
            cv2.imwrite(str(out), vis)

        frame_i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
