from options.inference_options import set_inference_options
from utils.loading_utils import get_device
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.recon_utils import AdaE2VIDReconstructor
from utils.event_readers import FixedSizeEventNpyReader
from utils.timers import CudaTimer
from dataset.dataset import load_image, load_events
from model.model import E2VIDRecurrent

import os
import torch
import numpy as np

import argparse
import cv2
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing AdaptE2VID model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('-iv', '--input_voxels', required=True, type=str, help='Path to input voxel file')
    parser.add_argument('-if', '--input_frame', required=True, type=str, help='Path to input initial frame file')
    set_inference_options(parser)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    adapter_config = config['adapter']
    e2vid_config = config['model']
    inference_config = config['inference']
    
    frame = load_image(args.input_frame)
    height, width = frame.shape[1], frame.shape[2]

    num_events_per_pixel = inference_config.get('num_events_per_pixel', 0.35)
    N = inference_config.get('window_events', width * height * num_events_per_pixel)

    # event_iter = FixedSizeEventNpyReader(
    #     path=args.input_event,
    #     num_events=N,
    #     start_index=inference_config.get('skip_events', 0),
    #     copy_out=True
    # )
    
    device = get_device(args.use_gpu)
    adapter = E2VIDRecurrent(adapter_config).to(device)
    e2vid = E2VIDRecurrent(e2vid_config).to(device)
    
    ckpt = {
        'adapter': adapter_config['ckpt_path'],
        'e2vid': e2vid_config['ckpt_path']
    }
    adapter_state = torch.load(ckpt['adapter'], map_location=device)['state_dict']
    e2vid_state = torch.load(ckpt['e2vid'], map_location=device)['state_dict']
    adapter.load_state_dict(adapter_state)
    e2vid.load_state_dict(e2vid_state)
    adapter.eval()
    e2vid.eval()
    
    reconstructor = AdaE2VIDReconstructor(e2vid, adapter, args, height, width)
    reconstructor.initialize(frame.unsqueeze(0).to(device))
    
    initial_offset = inference_config.get('skip_events', 0)
    sub_offset = inference_config.get('suboffset', 0)
    start_idx = initial_offset + sub_offset

    event_iter = sorted(os.listdir(args.input_voxels))
    with CudaTimer('Processing entire dataset'):
        for i, event_window in enumerate(event_iter):
            
            if event_window.endswith('.npy'):
                voxel_path = os.path.join(args.input_voxels, event_window)
                event_voxels = np.load(voxel_path)
                event_tensor = torch.from_numpy(event_voxels).to(device)
                print(event_tensor.shape)
                last_timestamp = i
                
                num_events = 1
                reconstructor.update_reconstruction(event_tensor, start_idx + num_events, last_timestamp)
                
                start_idx += num_events