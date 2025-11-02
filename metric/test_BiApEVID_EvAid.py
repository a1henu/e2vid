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
from model.swinir import SwinIR
from options.inference_options import set_inference_options
from utils.inference_utils import get_device, events_to_voxel_grid_pytorch, CropParameters
from utils.aperture_utils import degrade_img, denoise_img
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

def get_shape(dataset_path):
    shape_file = os.path.join(dataset_path, 'shape.txt')
    shape = np.loadtxt(shape_file, dtype=np.int32)
    width, height = shape[0], shape[1]
    return width, height

def get_event_list(dataset_path):
    evs_path = os.path.join(dataset_path, 'event')
    event_files = os.listdir(evs_path)
    return sorted(event_files), len(event_files)

def get_frame_list(dataset_path):
    frames_path = os.path.join(dataset_path, 'gt')
    frame_files = os.listdir(frames_path)
    return sorted(frame_files)

def get_options(width, height):
    options = {
        'height': height,
        'width': width,
        'num_bins': 5,
        'num_encoders': 3,
        'base_num_channels': 32,
        'norm': 'BN',
        'use_upsample_conv': False,
    }
    return options

def load_recons_model(options, ckpt_path, device):
    model = BiApEVID(E2VIDRecurrent(options), E2VIDRecurrent(options)).to(device)
    model_state = torch.load(ckpt_path, map_location=device)['state_dict']
    model.load_state_dict(model_state)
    model.eval()
    
    crop = CropParameters(options['width'], options['height'], options['num_encoders'])
    return model, crop

def load_denoise_model(ckpt_path, device):
    model = SwinIR(
        img_size=72,
        patch_size=1,
        in_chans=1,  # Grayscale input
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=8,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        upscale=1,  # No upscaling for denoising
        img_range=1.,
        upsampler='',  # No upsampler for denoising
        resi_connection='1conv'
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model

def inference_sequence(reconstructor, denoiser, f0_path, f1_path, evs_filelist, start_idx, savedir, device):
    f0_img = cv2.imread(f0_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    f1_img = cv2.imread(f1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    f0_noised = degrade_img(f0_img)
    f1_noised = degrade_img(f1_img)
    f0_denoised = denoise_img(f0_noised, denoiser, device)
    f1_denoised = denoise_img(f1_noised, denoiser, device)
    
    reconstructor.inference(f0_denoised, f1_denoised, evs_filelist, start_idx, savedir)

def main():
    parser = argparse.ArgumentParser(description='Testing AdaptE2VID model')
    parser.add_argument('--dataset_root', type=str, default='/mnt/D/baichenxu/dataset/EventHDR', help='Root path of the dataset')
    parser.add_argument('--data', type=str, default='bear', help='Name of the dataset to test on')
    parser.add_argument('--delta_frame', type=int, default=50, help='Frame delta for event sequences')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment')
    parser.add_argument('--recons_ckpt', type=str, help='Path to the reconstruction model checkpoint')
    set_inference_options(parser)
    args = parser.parse_args()

    save_dir = args.output_folder
    os.makedirs(save_dir, exist_ok=True)
    recons_ckpt = args.recons_ckpt
    denoiser_ckpt = '/home/baichenxu/projects/EvHDR/code/EventHDR/experiments/swinir_denoise_ds_20250901_203007/latest_checkpoint.pth'
    
    device = get_device(use_gpu=True)
    dataset_path = os.path.join(args.dataset_root, args.data)
    width, height = get_shape(dataset_path)
    options = get_options(width, height)
    event_files, data_length = get_event_list(dataset_path)
    frame_files = get_frame_list(dataset_path)
    
    reconstructor, crop = load_recons_model(options, recons_ckpt, device)
    denoiser = load_denoise_model(denoiser_ckpt, device)
    start_idx = 0
    while start_idx < data_length:
        end_idx = min(start_idx + args.delta_frame, data_length - 1)
        f0_path = os.path.join(dataset_path, 'gt', frame_files[start_idx])
        f1_path = os.path.join(dataset_path, 'gt', frame_files[end_idx])
        evs_filelist = [os.path.join(dataset_path, 'event', event_files[i]) for i in range(start_idx, end_idx)]
        inference_sequence(reconstructor, denoiser, f0_path, f1_path, evs_filelist, start_idx+1, save_dir, device)
        start_idx += args.delta_frame

if __name__ == '__main__':
    main()