import os, sys

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import argparse
import torch
import cv2
import numpy as np

from model.model import E2VIDRecurrent
from options.inference_options import set_inference_options
from utils.inference_utils import get_device, events_to_voxel_grid_pytorch
from utils.recon_utils import AdaE2VIDReconstructor

# dataset_root = '/mnt/nas-cp/liboyu/EvAid/data_align-n/HFR/HFR-aligned'
dataset_root = '/mnt/nas-cp/liboyu/EvAid/data_align-n/recons/recons-aligned'
adapter_ckpt = '/mnt/D/baichenxu/code/rpg_e2vid/logs/20251030/ada_e2vid_model/adapter_best.pth.tar'
e2vid_ckpt = '/mnt/D/baichenxu/code/rpg_e2vid/logs/20251030/ada_e2vid_model/e2vid_best.pth.tar'

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

parser = argparse.ArgumentParser(description='Testing AdaptE2VID model')
parser.add_argument('--data', type=str, default='playball', help='Name of the dataset to test on')
set_inference_options(parser)
args = parser.parse_args()

device = get_device(use_gpu=True)
adapter = E2VIDRecurrent(options).to(device)
e2vid = E2VIDRecurrent(options).to(device)

adapter_state = torch.load(adapter_ckpt, map_location=device)['state_dict']
e2vid_state = torch.load(e2vid_ckpt, map_location=device)['state_dict']
adapter.load_state_dict(adapter_state)
e2vid.load_state_dict(e2vid_state)
adapter.eval()
e2vid.eval()

reconstructor = AdaE2VIDReconstructor(e2vid, adapter, args, options['height'], options['width'])

dataset_path = os.path.join(dataset_root, args.data)
frames_path = os.path.join(dataset_path, 'gt')
events_path = os.path.join(dataset_path, 'event')

for idx in range(len(os.listdir(events_path))):
    event_name = f'{idx+1:06d}.txt'
    if idx % 500 == 0:
        init_frame = cv2.imread(os.path.join(frames_path, f'{idx+1:06d}_img.jpg'), cv2.IMREAD_GRAYSCALE)
        init_frame = torch.from_numpy(init_frame).unsqueeze(0).float() / 255.0
        reconstructor.initialize(init_frame.unsqueeze(0).to(device))
        
    event_file = os.path.join(events_path, event_name)
    event = np.loadtxt(event_file)
    last_timestamp = event[-1, 0]
    vox = events_to_voxel_grid_pytorch(
        event,
        num_bins=reconstructor.num_bins,
        width=width,
        height=height,
        device=device
    )
    reconstructor.update_reconstruction(vox, idx + 1, last_timestamp)
    