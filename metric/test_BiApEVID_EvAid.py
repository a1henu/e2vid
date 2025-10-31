import os, sys

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import argparse
import torch
import cv2
import os
import numpy as np

from model.model import E2VIDRecurrent
from model.BiApEVID import BiApEVID
from options.inference_options import set_inference_options
from utils.inference_utils import get_device, events_to_voxel_grid_pytorch, CropParameters
from utils.recon_utils import AdaE2VIDReconstructor

def load_img(path, device):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
    return img.unsqueeze(0).to(device)

def save_img(path, img_tensor):
    img_np = (img_tensor.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    cv2.imwrite(path, img_np)

def load_evs(path, start_idx, end_idx, num_bins, height, width, crop, device):
    evs = [] # [1, T, B, H, W]
    timestamps = []
    for idx in range(start_idx, end_idx):
        ev = np.loadtxt(os.path.join(path, f'{idx:06d}.txt'), dtype=np.float32)
        voxel_grid = events_to_voxel_grid_pytorch(ev, num_bins, width, height, device)
        timestamps.append(ev[-1, 0])
        evs.append(crop.pad(voxel_grid))
    return torch.stack(evs, dim=0).unsqueeze(0).to(device), timestamps

width = 954
height = 636
options = {
    'height': height,
    'width': width,
    'num_bins': 5,
    'num_encoders': 3,
    'base_num_channels': 32,
    'norm': 'BN',
    'use_upsample_conv': False,
}

dataset_root = '/mnt/nas-cp/liboyu/EvAid/data_align-n/HFR/HFR-aligned'
# dataset_root = '/mnt/nas-cp/liboyu/EvAid/data_align-n/recons/recons-aligned'
BiApEVID_ckpt = '/mnt/D/baichenxu/code/rpg_e2vid/logs/20251030/biape2vid_model/biape2vid_best.pth.tar'

device = get_device(use_gpu=True)
model = BiApEVID(E2VIDRecurrent(options), E2VIDRecurrent(options)).to(device)
model_state = torch.load(BiApEVID_ckpt, map_location=device)['state_dict']
model.load_state_dict(model_state)
model.eval()
crop = CropParameters(width, height, options['num_encoders'])

parser = argparse.ArgumentParser(description='Testing AdaptE2VID model')
parser.add_argument('--data', type=str, default='playball', help='Name of the dataset to test on')
set_inference_options(parser)
args = parser.parse_args()

dataset_path = os.path.join(dataset_root, args.data)
frames_path = os.path.join(dataset_path, 'gt')
events_path = os.path.join(dataset_path, 'event')
save_dir = args.output_folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

length = 1500
delta_f = 50

idx = 1
while idx <= length:
    end_idx = min(idx + delta_f, length)
    f0 = load_img(os.path.join(frames_path, f'{idx:06d}_img.jpg'), device)
    f1 = load_img(os.path.join(frames_path, f'{end_idx:06d}_img.jpg'), device)
    
    evs_filelist = []
    for ev_idx in range(idx, end_idx):
        evs_filelist.append(os.path.join(events_path, f'{ev_idx:06d}.txt'))
    model.inference(f0, f1, evs_filelist, idx, save_dir)
    
    idx += delta_f