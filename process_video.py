import argparse
import functools
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from networks.sub_networks import DeblurringNet
from tqdm import tqdm
from utils.data_processing import get_normalize, toTensor

def process_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    try:
        while True:
            ret, frame = cap.read()
        
            if not ret:
                break
    
            yield frame
         
    finally:
        cap.release()

def preprocess_to_tensor(img):
    normalize = get_normalize()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, _ = normalize(img, img)
    return toTensor(img).unsqueeze(0) 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dst', type=Path, required=True)
    parser.add_argument('--src', type=Path, required=True)

    return parser.parse_args()

def save_array_as_image(filepath: Path, img: np.ndarray):
        
    img = (img* 255).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filepath, img)

def video_inference(cfg):
    net = DeblurringNet(norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)).to(
        cfg.device)
    pretrained_dict = torch.load('final_model/DeblurringNet_FT.pth')
    net.load_state_dict(pretrained_dict['deblurring_state_dict'])
    os.makedirs(cfg.dst, exist_ok=True)
    preprocess_times = []
    nn_eval_times = []
    postprocess_times = []
    with torch.no_grad():
        for idx, frame in tqdm(enumerate(process_video_frames(cfg.src))):
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

            cv2.imwrite(cfg.dst / f'frame_original_{idx:06d}.jpg', frame)
            save_array_as_image(cfg.dst / f'frame_deblurred_{idx:06d}.jpg', result)

    for label, times in [('Preprocess', preprocess_times), ('Forward pass', nn_eval_times), ('Postprocess', postprocess_times)]:
        print(f"\n--- {label} times (s) ---")
        print(pd.Series(times).describe())

if __name__ == '__main__':
    cfg = parse_args()
    video_inference(cfg)

