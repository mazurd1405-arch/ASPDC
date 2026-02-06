import argparse
import functools

import cv2
import numpy as np
import torch
import torch.nn as nn
from networks.sub_networks import DeblurringNet
from tqdm import tqdm
from utils.data_processing import get_normalize, toTensor

def process_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    parser.add_argument('--testset_dir', type=str, default='/app/repo/GOPRO')
    parser.add_argument('--dst', type=str, default='/app/output/Capture0010/motion06350/')
    parser.add_argument('--src', type=str, default='/app/data/Capture0010/motion_06350.mov')

    return parser.parse_args()

def numpy_to_cv2(filepath, img):
        
    img = (img* 255).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filepath, img)

def video_inference(cfg):
    net = DeblurringNet(norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)).to(
        cfg.device)

    pretrained_dict = torch.load('final_model/DeblurringNet_FT.pth')
    net.load_state_dict(pretrained_dict['deblurring_state_dict'])
    
    with torch.no_grad():
        for idx, frame in tqdm(enumerate(process_video_frames(cfg.src))):
            img_tensor = preprocess_to_tensor(np.copy(frame))
            img_tensor = img_tensor.to(cfg.device)

            result, _ = net(img_tensor)

            result = torch.clamp(result, -1, 1)
            result = (result + 1) /2

            result = result[0, ...].detach().permute(1, 2, 0).cpu().numpy()

            cv2.imwrite(f'{cfg.dst}frame_original{idx}.jpg', frame)
            numpy_to_cv2(f'{cfg.dst}frame_deblurred_{idx}.jpg', result)


if __name__ == '__main__':
    cfg = parse_args()
    video_inference(cfg)

