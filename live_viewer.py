import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True, help="directory with *_original.png and *_deblurred.png")
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def latest_pair(directory: Path):
    deblurred = sorted(directory.glob("*_deblurred.png"))
    if not deblurred:
        return None
    idx = deblurred[-1].name.split("_", 1)[0]
    original = directory / f"{idx}_original.png"
    if not original.exists():
        return None
    return original, deblurred[-1], idx


def add_label(frame: np.ndarray, text: str):
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (220, 36), (0, 0, 0), thickness=-1)
    cv2.putText(labeled, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def maybe_resize(frame: np.ndarray, scale: float):
    if scale == 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def main():
    args = parse_args()
    last_idx = None

    while True:
        pair = latest_pair(args.dir)
        if pair is not None:
            original_path, deblurred_path, idx = pair
            if idx != last_idx:
                original = cv2.imread(str(original_path))
                deblurred = cv2.imread(str(deblurred_path))
                if original is not None and deblurred is not None:
                    original = add_label(original, f"original {idx}")
                    deblurred = add_label(deblurred, f"deblurred {idx}")
                    canvas = np.hstack([original, deblurred])
                    canvas = maybe_resize(canvas, args.display_scale)
                    cv2.imshow("ASPDC Live Viewer", canvas)
                    last_idx = idx

        key = cv2.waitKey(int(args.poll_interval * 1000)) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
