import shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "bird" / "data.yaml"
BEST_WEIGHTS = ROOT / "weights" / "best.pt"
BEST_CONF = ROOT / "weights" / "best_conf.txt"
PRETRAINED = "yolo26n.pt"
SEED = 42

def evaluate(model, name, single_cls=False):
    box = model.val(data=str(DATA), single_cls=single_cls, name=name, exist_ok=True, plots=False).box
    return {
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "best_conf": float(box.px[box.f1_curve.mean(0).argmax()]),
    }

def main():
    model = YOLO(PRETRAINED)
    model.train(
        data=str(DATA),
        epochs=100,
        patience=20,
        freeze=10,
        imgsz=640,
        workers=2,
        batch=16,
        seed=SEED,
        name="finetune",
        exist_ok=True,
    )

    BEST_WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, BEST_WEIGHTS)
    pretrained = evaluate(YOLO(PRETRAINED), "val_pretrained", single_cls=True)
    finetuned = evaluate(YOLO(BEST_WEIGHTS), "val_finetune")
    print("pretrained:", pretrained)
    print("finetuned:", finetuned)
    BEST_CONF.write_text(str(finetuned["best_conf"]))

if __name__ == "__main__":
    main()
