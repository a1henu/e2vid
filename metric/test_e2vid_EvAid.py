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
from utils.loading_utils import load_model
from image_reconstructor import ImageReconstructor

dataset_root = '/mnt/nas-cp/liboyu/EvAid/data_align-n/HFR/HFR-aligned'

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

device = get_device(use_gpu=True)
# e2vid_ckpt = '/mnt/D/baichenxu/code/rpg_e2vid/pretrained/pretrained_e2vid.pth.tar'
# e2vid = load_model(e2vid_ckpt).to(device)
e2vid_ckpt = '/mnt/D/baichenxu/code/rpg_e2vid/logs/20251026/e2vid_model/e2vid_epoch_0200.pth.tar'
e2vid = E2VIDRecurrent(options).to(device)

parser = argparse.ArgumentParser(description='Testing AdaptE2VID model')
parser.add_argument('--data', type=str, default='wall', help='Name of the dataset to test on')
set_inference_options(parser)
args = parser.parse_args()

e2vid.eval()

reconstructor = ImageReconstructor(e2vid, height, width, e2vid.num_bins, args)

dataset_path = os.path.join(dataset_root, args.data)
frames_path = os.path.join(dataset_path, 'gt')
events_path = os.path.join(dataset_path, 'event')

for idx, event_name in enumerate(sorted(os.listdir(events_path))):
    if not event_name.endswith('.txt'):
        continue
        
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
    reconstructor.update_reconstruction(vox, idx + 2, last_timestamp)
    