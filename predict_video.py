import argparse
import cv2
from ultralytics import YOLO

WEIGHTS = "weights/best.pt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = YOLO(WEIGHTS)
    video = cv2.VideoCapture(args.input)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    while True:
        ok, frame = video.read()
        if not ok:
            break
        result = model(frame, verbose=False)[0]
        writer.write(result.plot())  

    video.release()
    writer.release()

if __name__ == "__main__":
    main()
