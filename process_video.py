import argparse
import functools
import os
from pathlib import Path
import re
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from networks.sub_networks import DeblurringNet
from tqdm import tqdm
from utils.data_processing import get_normalize, toTensor

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def print_timing_summary(label, times):
    values = np.asarray(times, dtype=np.float64)
    if values.size == 0:
        print(f"\n--- {label} times (s) ---")
        print("No samples collected.")
        return

    print(f"\n--- {label} times (s) ---")
    print(f"count    {values.size}")
    print(f"mean     {values.mean():.6f}")
    print(f"std      {values.std(ddof=1):.6f}" if values.size > 1 else "std      0.000000")
    print(f"min      {values.min():.6f}")
    print(f"25%      {np.percentile(values, 25):.6f}")
    print(f"50%      {np.percentile(values, 50):.6f}")
    print(f"75%      {np.percentile(values, 75):.6f}")
    print(f"max      {values.max():.6f}")


def natural_sort_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', path.name)
    ]


def process_video_frames(video_path):
    source = int(video_path) if str(video_path).isdigit() else str(video_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video source: {video_path}')

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            yield frame, None

    finally:
        cap.release()


def process_image_sequence_frames(directory: Path):
    image_paths = sorted(
        (
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )

    if not image_paths:
        supported = ', '.join(sorted(IMAGE_EXTENSIONS))
        raise RuntimeError(
            f'No supported image files found in directory: {directory}. '
            f'Supported extensions: {supported}'
        )

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f'Failed to read image file: {image_path}')
        yield frame, image_path.name


def process_source_frames(src: str):
    src_path = Path(src)
    if src_path.is_dir():
        yield from process_image_sequence_frames(src_path)
        return

    yield from process_video_frames(src)


def preprocess_to_tensor(img):
    normalize = get_normalize()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, _ = normalize(img, img)
    return toTensor(img).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dst', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, default=Path('final_model/DeblurringNet_FT.pth'))
    parser.add_argument(
        '--src',
        type=str,
        required=True,
        help='video file, camera index, or directory with image sequence',
    )
    parser.add_argument('--max-frames', type=int, default=None)

    return parser.parse_args()


def save_array_as_image(filepath: Path, img: np.ndarray):
    img = (img * 255).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filepath, img)


def save_inference_outputs(dst: Path, idx: int, frame: np.ndarray, result: np.ndarray, source_name: str | None):
    if source_name is None:
        cv2.imwrite(dst / f'frame_original_{idx:06d}.jpg', frame)
        save_array_as_image(dst / f'frame_deblurred_{idx:06d}.jpg', result)
        return

    save_array_as_image(dst / source_name, result)


def video_inference(cfg):
    net = DeblurringNet(norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)).to(
        cfg.device)
    pretrained_dict = torch.load(cfg.model_path, map_location=cfg.device)
    net.load_state_dict(pretrained_dict['deblurring_state_dict'])
    os.makedirs(cfg.dst, exist_ok=True)
    preprocess_times = []
    nn_eval_times = []
    postprocess_times = []
    with torch.no_grad():
        for idx, (frame, source_name) in tqdm(enumerate(process_source_frames(cfg.src))):
            if cfg.max_frames is not None and idx >= cfg.max_frames:
                break

            t0 = time.perf_counter()
            img_tensor = preprocess_to_tensor(np.copy(frame))
            img_tensor = img_tensor.to(cfg.device)
            preprocess_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            result, _ = net(img_tensor)
            if cfg.device == 'cuda':
                torch.cuda.synchronize()
            nn_eval_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            result = torch.clamp(result, -1, 1)
            result = (result + 1) / 2
            result = result[0, ...].detach().permute(1, 2, 0).cpu().numpy()
            postprocess_times.append(time.perf_counter() - t0)

            save_inference_outputs(cfg.dst, idx, frame, result, source_name)

    for label, times in [('Preprocess', preprocess_times), ('Forward pass', nn_eval_times), ('Postprocess', postprocess_times)]:
        print_timing_summary(label, times)


if __name__ == '__main__':
    cfg = parse_args()
    video_inference(cfg)
