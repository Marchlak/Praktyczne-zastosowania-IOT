import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


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


def apply_rotate(frame: np.ndarray, rotate: int) -> np.ndarray:
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
    ap.add_argument("--font_scale", type=float, default=1.2)
    ap.add_argument("--font_thick", type=int, default=3)

    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--ema", type=float, default=0.2)

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

    last_g = None
    last_deg = None

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

        h, w = frame.shape[:2]
        vis = frame.copy()

        do_infer = (args.every <= 1) or (frame_i % args.every == 0)
        if do_infer:
            pred = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]

            bi = pick_best_detection(pred)
            if bi is not None and pred.keypoints is not None:
                kpts = pred.keypoints.xy[bi].detach().cpu().numpy().astype(np.float32)
                if kpts.shape[0] >= args.kpt_count:
                    c = kpts[args.kpt_center]
                    z = kpts[args.kpt_zero]
                    n = kpts[args.kpt_needle]

                    deg = cw_angle_deg(c, z, n)
                    if deg is not None:
                        grams = weight_from_angle(deg, args.cap_g, args.sweep_deg, args.wrap)

                        if last_g is None:
                            last_g = float(grams)
                            last_deg = float(deg)
                        else:
                            a = float(args.ema)
                            a = min(max(a, 0.0), 1.0)
                            last_g = (1.0 - a) * float(last_g) + a * float(grams)
                            last_deg = float(deg)

        if last_deg is not None:
            pred = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]
            bi = pick_best_detection(pred)
            if bi is not None and pred.keypoints is not None:
                kpts = pred.keypoints.xy[bi].detach().cpu().numpy().astype(np.float32)
                if kpts.shape[0] >= args.kpt_count:
                    c = kpts[args.kpt_center]
                    z = kpts[args.kpt_zero]
                    n = kpts[args.kpt_needle]
                    cc = tuple(int(x) for x in c)
                    zz = tuple(int(x) for x in z)
                    nn = tuple(int(x) for x in n)

                    cv2.line(vis, cc, zz, (255, 0, 0), args.line_thick, cv2.LINE_AA)
                    cv2.line(vis, cc, nn, (0, 255, 0), args.line_thick, cv2.LINE_AA)

                    cv2.circle(vis, cc, args.pt_radius, (255, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(vis, zz, args.pt_radius, (255, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(vis, nn, args.pt_radius, (0, 255, 0), -1, cv2.LINE_AA)

        dt = time.time() - t0
        if dt >= 0.5:
            fps = (frame_i + 1) / dt

        lines = []
        if last_g is None or last_deg is None:
            lines.append("pred: -")
        else:
            lines.append(f"pred: {int(round(last_g))} g  ({last_deg:.1f}°)")
            lines.append(f"max: {int(args.cap_g)} g, sweep: {args.sweep_deg:.0f}°")
        lines.append(f"fps: {fps:.1f}")

        fs = args.font_scale * max(1.0, w / 1200.0)
        draw_text_block(vis, lines, 20, 60, fs, args.font_thick)

        cv2.imshow("meter_live", vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s") and save_dir is not None:
            out = save_dir / f"frame_{int(time.time()*1000)}.jpg"
            cv2.imwrite(str(out), vis)

        frame_i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
