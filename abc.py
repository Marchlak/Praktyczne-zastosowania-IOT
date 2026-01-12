from pathlib import Path

def check(split):
    img_dir = Path("waga_pose_ds") / "images" / split
    lbl_dir = Path("waga_pose_ds") / "labels" / split
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]])
    missing = 0
    bad = 0
    for img in imgs:
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            missing += 1
            continue
        line = lbl.read_text(encoding="utf-8").strip().split()
        if len(line) != 14:
            bad += 1
    print(split, "images", len(imgs), "missing_labels", missing, "bad_format", bad)

check("train")
check("val")
