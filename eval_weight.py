import os
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def wrap_0_2pi(a):
    return a % (2 * math.pi)

def weight_from_points(z, c, t, cap_g, invert=False):
    az = math.atan2(z[1] - c[1], z[0] - c[0])
    at = math.atan2(t[1] - c[1], t[0] - c[0])
    diff = wrap_0_2pi((az - at) if invert else (at - az))
    return (diff / (2 * math.pi)) * cap_g

def read_gt_label(label_path):
    parts = label_path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 5 + 3 * 3:
        return None
    nums = list(map(float, parts[1:]))
    kps = []
    base = 4
    for i in range(3):
        x = nums[base + i * 3 + 0]
        y = nums[base + i * 3 + 1]
        v = nums[base + i * 3 + 2]
        kps.append((x, y, v))
    return kps

def infer_pred_kps_norm(model, img_path, conf):
    r = model.predict(str(img_path), conf=conf, verbose=False)[0]
    if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
        return None
    k = r.keypoints.xy[0].cpu().numpy()
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    k[:, 0] = k[:, 0] / max(w, 1)
    k[:, 1] = k[:, 1] / max(h, 1)
    return k

def metrics(errors):
    e = np.array(errors, dtype=float)
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e * e)))
    med = float(np.median(np.abs(e)))
    return mae, rmse, med

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cap_g", type=float, default=5000.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out_csv", default="eval_report.csv")
    args = ap.parse_args()

    model = YOLO(args.model)
    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)

    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]])
    rows = []

    for img_path in img_paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        gt_kps = read_gt_label(lbl_path)
        if gt_kps is None:
            continue

        pred_kps = infer_pred_kps_norm(model, img_path, args.conf)
        if pred_kps is None or pred_kps.shape[0] < 3:
            rows.append(
                dict(
                    image=img_path.name,
                    gt_g=None,
                    pred_g=None,
                    abs_err_g=None,
                    ok=False,
                )
            )
            continue

        z_gt = (gt_kps[0][0], gt_kps[0][1])
        c_gt = (gt_kps[1][0], gt_kps[1][1])
        t_gt = (gt_kps[2][0], gt_kps[2][1])

        z_pr = (float(pred_kps[0][0]), float(pred_kps[0][1]))
        c_pr = (float(pred_kps[1][0]), float(pred_kps[1][1]))
        t_pr = (float(pred_kps[2][0]), float(pred_kps[2][1]))

        gt_g = weight_from_points(z_gt, c_gt, t_gt, args.cap_g, invert=False)
        pred_g_a = weight_from_points(z_pr, c_pr, t_pr, args.cap_g, invert=False)
        pred_g_b = weight_from_points(z_pr, c_pr, t_pr, args.cap_g, invert=True)

        rows.append(
            dict(
                image=img_path.name,
                gt_g=float(gt_g),
                pred_g=float(pred_g_a),
                pred_g_invert=float(pred_g_b),
                ok=True,
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    ok = df[df["ok"] == True].copy()
    if len(ok) == 0:
        print("0 poprawnych predykcji keypointów")
        print("CSV:", args.out_csv)
        return

    ok["err_a"] = ok["pred_g"] - ok["gt_g"]
    ok["err_b"] = ok["pred_g_invert"] - ok["gt_g"]

    mae_a, rmse_a, med_a = metrics(ok["err_a"].tolist())
    mae_b, rmse_b, med_b = metrics(ok["err_b"].tolist())

    use_invert = mae_b < mae_a
    err = ok["err_b"] if use_invert else ok["err_a"]
    mae, rmse, med = metrics(err.tolist())

    abs_err = np.abs(err.to_numpy())
    within_20 = float(np.mean(abs_err <= 20.0) * 100.0)
    within_40 = float(np.mean(abs_err <= 40.0) * 100.0)
    within_100 = float(np.mean(abs_err <= 100.0) * 100.0)

    total = len(df)
    good = len(ok)
    miss = total - good

    print("images_total:", total)
    print("ok_keypoints:", good)
    print("missing_keypoints:", miss)
    print("direction:", "invert" if use_invert else "normal")
    print("MAE_g:", round(mae, 2))
    print("RMSE_g:", round(rmse, 2))
    print("MedianAbs_g:", round(med, 2))
    print("<=20g:", round(within_20, 2), "%")
    print("<=40g:", round(within_40, 2), "%")
    print("<=100g:", round(within_100, 2), "%")
    print("CSV:", args.out_csv)

if __name__ == "__main__":
    main()
