import functools
import time
from argparse import ArgumentParser
from pathlib import Path
from select import select
from socket import create_connection
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn as nn

from networks.sub_networks import DeblurringNet
from process_video import preprocess_to_tensor, save_array_as_image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="video source path, URL, camera index, or tcp://host:port")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--drop-old-frames", action="store_true")
    parser.add_argument("--grayscale-input", action="store_true")
    return parser.parse_args()


def iter_frames_from_capture(src: str):
    source = int(src) if src.isdigit() else src
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {src}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def decode_jpegs_from_buffer(buffer: bytearray):
    frames = []
    while True:
        start = buffer.find(b"\xff\xd8")
        if start < 0:
            break
        end = buffer.find(b"\xff\xd9", start + 2)
        if end < 0:
            if start > 0:
                del buffer[:start]
            break
        jpeg = np.frombuffer(buffer[start:end + 2], dtype=np.uint8)
        frame = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        del buffer[:end + 2]
        if frame is not None:
            frames.append(frame)
    return frames


def iter_frames_from_tcp_mjpeg(src: str, drop_old_frames: bool):
    parsed = urlparse(src)
    if not parsed.hostname or not parsed.port:
        raise RuntimeError(f"Invalid tcp source: {src}")

    sock = create_connection((parsed.hostname, parsed.port), timeout=5.0)
    sock.setblocking(False)
    buffer = bytearray()
    try:
        while True:
            ready, _, _ = select([sock], [], [], 5.0)
            if not ready:
                continue
            while True:
                try:
                    chunk = sock.recv(65536)
                except BlockingIOError:
                    break
                if not chunk:
                    return
                buffer.extend(chunk)
                if not drop_old_frames:
                    frames = decode_jpegs_from_buffer(buffer)
                    for frame in frames:
                        yield frame
            if drop_old_frames:
                frames = decode_jpegs_from_buffer(buffer)
                if frames:
                    yield frames[-1]
    finally:
        sock.close()


def iter_frames(src: str, drop_old_frames: bool):
    if src.startswith("tcp://"):
        yield from iter_frames_from_tcp_mjpeg(src, drop_old_frames)
    else:
        yield from iter_frames_from_capture(src)


def load_model(device: str):
    net = DeblurringNet(
        norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    ).to(device)
    pretrained_dict = torch.load("final_model/DeblurringNet_FT.pth", map_location=device)
    net.load_state_dict(pretrained_dict["deblurring_state_dict"])
    net.eval()
    return net


def maybe_resize(frame: np.ndarray, scale: float):
    if scale == 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def maybe_convert_to_grayscale(frame: np.ndarray, grayscale_input: bool):
    if not grayscale_input:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main():
    args = parse_args()
    net = load_model(args.device)
    preprocess_times = []
    nn_times = []
    postprocess_times = []
    processed = 0

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    try:
        with torch.no_grad():
            for frame in iter_frames(args.src, args.drop_old_frames):
                frame = maybe_convert_to_grayscale(frame, args.grayscale_input)

                t0 = time.perf_counter()
                img_tensor = preprocess_to_tensor(np.copy(frame)).to(args.device)
                preprocess_times.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                result, _ = net(img_tensor)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                nn_times.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                result = torch.clamp(result, -1, 1)
                result = (result + 1) / 2
                result = result[0, ...].detach().permute(1, 2, 0).cpu().numpy()
                postprocess_times.append(time.perf_counter() - t0)

                if args.save_dir is not None:
                    cv2.imwrite(str(args.save_dir / f"{processed:06d}_original.png"), frame)
                    save_array_as_image(args.save_dir / f"{processed:06d}_deblurred.png", result)

                original_show = maybe_resize(frame, args.display_scale)
                deblurred_show = maybe_resize(
                    cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    args.display_scale,
                )
                if not args.no_display:
                    cv2.imshow("original", original_show)
                    cv2.imshow("deblurred", deblurred_show)

                processed += 1
                key = (cv2.waitKey(1) & 0xFF) if not args.no_display else 255
                if key in (27, ord("q")):
                    break
                if args.max_frames is not None and processed >= args.max_frames:
                    break
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()

    if nn_times:
        arr = np.asarray(nn_times, dtype=np.float64)
        print(
            f"Processed {processed} frames. "
            f"NN mean={arr.mean():.4f}s p50={np.percentile(arr, 50):.4f}s "
            f"p90={np.percentile(arr, 90):.4f}s max={arr.max():.4f}s"
        )


if __name__ == "__main__":
    main()
