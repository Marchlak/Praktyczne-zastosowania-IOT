import argparse
import math
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
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

    args = ap.parse_args()

    if not os.path.isfile(args.image):
        raise SystemExit(f"no such image: {args.image}")

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("failed to read image")

    model = YOLO(args.model)

    pred = model.predict(
        source=img,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )[0]

    bi = pick_best_detection(pred)
    if bi is None or pred.keypoints is None:
        print("-")
        return

    kpts = pred.keypoints.xy[bi].detach().cpu().numpy().astype(np.float32)
    if kpts.shape[0] < args.kpt_count:
        print("-")
        return

    c = kpts[args.kpt_center]
    z = kpts[args.kpt_zero]
    n = kpts[args.kpt_needle]

    deg = cw_angle_deg(c, z, n)
    if deg is None:
        print("-")
        return

    grams = weight_from_angle(deg, args.cap_g, args.sweep_deg, args.wrap)

    print(f"{grams:.2f}")


if __name__ == "__main__":
    main()
