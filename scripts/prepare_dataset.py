import shutil
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "dataset" / "bird"
OUT_DIR = ROOT / "data" / "bird"
VAL_SIZE = 0.2
SEED = 42

def label_path(image):
    return RAW_DIR / "labels" / f"{image.stem}.txt"

def is_valid_box(xc, yc, w, h):
    inside_image = 0 <= xc - w / 2 and xc + w / 2 <= 1 and 0 <= yc - h / 2 and yc + h / 2 <= 1 and w > 0 and h > 0
    return inside_image

def clean_label(text):
    lines = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        box = [float(value) for value in parts[1:]]
        if is_valid_box(*box):
            lines.append("0 " + " ".join(parts[1:]) + "\n")
    return "".join(lines)

def copy_split(images, split):
    images_dir = OUT_DIR / "images" / split
    labels_dir = OUT_DIR / "labels" / split
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for image in images:
        shutil.copy(image, images_dir / image.name)
        label = clean_label(label_path(image).read_text())
        (labels_dir / f"{image.stem}.txt").write_text(label)

def main():
    images = sorted(RAW_DIR.glob("*.jpg"))
    images = [image for image in images if label_path(image).exists()]
    train, val = train_test_split(images, test_size=VAL_SIZE, random_state=SEED)

    shutil.rmtree(OUT_DIR, ignore_errors=True)
    copy_split(train, "train")
    copy_split(val, "val")

    data_config = {
        "path": OUT_DIR.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "bird"},
    }
    with open(OUT_DIR / "data.yaml", "w") as f:
        yaml.safe_dump(data_config, f, sort_keys=False)

    print(f"train: {len(train)}, val: {len(val)}")


if __name__ == "__main__":
    main()
