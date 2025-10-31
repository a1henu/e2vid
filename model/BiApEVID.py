import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset.dataset import SeqDataset
from utils.event_utils import event_density, reverse_voxels
from utils.data_utils import SeqCrop128
from utils.loading_utils import get_device
from utils.inference_utils import CropParameters, events_to_voxel_grid_pytorch

class Mixer(nn.Module):
    def __init__(self, feat_ch: int, mid: int = 32):
        super().__init__()
        # in: [s0, s1] -> 2ch； feats -> feat_ch
        self.net = nn.Sequential(
            nn.Conv2d(2 + feat_ch, mid, 3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(mid, 2, 1) 
        )

    def forward(self, s0, s1, feats):
        # s0,s1: [B,1,H,W]; feats: [B,feat_ch,H,W]
        x = torch.cat([s0, s1, feats], dim=1)  # [B, 2+feat_ch, H, W]
        logits = self.net(x)                   # [B, 2, H, W]
        w = torch.softmax(logits, dim=1)       # [B, 2, H, W]
        fused = (w[:, 0] * s0) + (w[:, 1] * s1)
        return fused, w
    
class BiApEVID(nn.Module):
    def __init__(self, adapter, e2vid):
        super().__init__()
        self.feat_ch = 2
        
        self.adapter = adapter
        self.e2vid   = e2vid
        self.mixer   = Mixer(feat_ch=self.feat_ch)

        self.device = get_device(use_gpu=True)
        self.height = e2vid.config['height']
        self.width  = e2vid.config['width']
        self.num_bins = e2vid.config['num_bins']
        self.crop = CropParameters(self.width, self.height, e2vid.config['num_encoders'])

    def forward(self, f0, f1, evs):
        # f0,f1: [B,1,H,W]; ev: [B,T,Bin,H,W]
        bsz, t, bin, h, w = evs.shape

        # forward process
        f0_vox = f0.repeat(1, bin, 1, 1)
        adapter_i0, st_f = self.adapter(f0_vox, None)
        fwd = [f0]
        for i in range(t):
            recon, st_f = self.e2vid(evs[:,i], st_f)
            fwd.append(recon)
        
        # backward process
        f1_vox = f1.repeat(1, bin, 1, 1)
        adapter_i1, st_b = self.adapter(f1_vox, None)
        bwd = [f1]
        for i in range(t-1, -1, -1):
            rev_ev = reverse_voxels(evs[:,i])
            recon, st_b = self.e2vid(rev_ev, st_b)
            bwd.append(recon)
        bwd = bwd[::-1]  # reverse to time order
        
        outs, ws = [f0], []
        for i in range(1, t):
            s0 = fwd[i]
            s1 = bwd[i]
            
            ev_density = event_density(evs[:,i])
            feats = torch.cat([ev_density, (s0 - s1).abs()], dim=1)  # [B,2,H,W]
            fused, w = self.mixer(s0, s1, feats)
            outs.append(fused)
            ws.append(w)
        outs.append(f1)
        
        fused = torch.stack(outs, dim=1)  # [B,T+1,1,H,W]
        ws    = torch.stack(ws, dim=1)    # [B,T-1,2,H,W]
        return fused, ws, adapter_i0, adapter_i1, fwd, bwd
    
    def initialize_inference(self, f0, f1, evs_list, start_idx, save_dir):
        end_idx = start_idx + len(evs_list)
        forward_dir = os.path.join(save_dir, 'forward')
        backward_dir = os.path.join(save_dir, 'backward')
        os.makedirs(forward_dir, exist_ok=True)
        os.makedirs(backward_dir, exist_ok=True)
        
        fwd_filelist = self.inference_forward(f0, evs_list, start_idx, forward_dir)
        bwd_filelist = self.inference_backward(f1, evs_list, end_idx, backward_dir)
        return fwd_filelist, bwd_filelist
        
    def inference_forward(self, f0, evs_filelist, start_idx, save_dir):
        t = len(evs_filelist)
        self.save_img(os.path.join(save_dir, f'{start_idx:06d}.png'), f0)
        f0_vox = f0.repeat(1, self.num_bins, 1, 1)
        f0_vox = self.crop.pad(f0_vox)
        with torch.no_grad():
            _, st_f = self.adapter(f0_vox, None)
        
        fwd_filelist = [os.path.join(save_dir, f'{start_idx:06d}.png')]
        for i in range(t):
            ev = np.loadtxt(evs_filelist[i], dtype=np.float32)
            ev = events_to_voxel_grid_pytorch(ev, self.num_bins, self.width, self.height, self.device)
            ev = self.crop.pad(ev).unsqueeze(0)
            with torch.no_grad():
                recon, st_f = self.e2vid(ev, st_f)
            self.save_img(os.path.join(save_dir, f'{start_idx + i + 1:06d}.png'), recon[:, :, self.crop.iy0:self.crop.iy1, self.crop.ix0:self.crop.ix1])
            fwd_filelist.append(os.path.join(save_dir, f'{start_idx + i + 1:06d}.png'))
        return fwd_filelist
    
    def inference_backward(self, f1, evs_filelist, end_idx, save_dir):
        t = len(evs_filelist)
        start_idx = end_idx - t
        self.save_img(os.path.join(save_dir, f'{end_idx:06d}.png'), f1)
        f1_vox = f1.repeat(1, self.num_bins, 1, 1)
        f1_vox = self.crop.pad(f1_vox)
        with torch.no_grad():
            _, st_b = self.adapter(f1_vox, None)
        
        bwd_filelist = [os.path.join(save_dir, f'{end_idx:06d}.png')]
        for i in range(t-1, -1, -1):
            ev = np.loadtxt(evs_filelist[i], dtype=np.float32)
            ev = events_to_voxel_grid_pytorch(ev, self.num_bins, self.width, self.height, self.device)
            ev = self.crop.pad(ev).unsqueeze(0)
            rev_ev = reverse_voxels(ev)
            with torch.no_grad():
                recon, st_b = self.e2vid(rev_ev, st_b)
            self.save_img(os.path.join(save_dir, f'{start_idx + i:06d}.png'), recon[:, :, self.crop.iy0:self.crop.iy1, self.crop.ix0:self.crop.ix1])
            bwd_filelist.append(os.path.join(save_dir, f'{start_idx + i:06d}.png'))
        bwd_filelist = bwd_filelist[::-1]
        return bwd_filelist
    
    def fuse_frames(self, evs_filelist, forward_filelist, backward_filelist, start_idx, save_dir):
        for i, (evs, fwd, bwd) in enumerate(zip(evs_filelist, forward_filelist, backward_filelist)):
            ev = np.loadtxt(evs, dtype=np.float32)
            ev = events_to_voxel_grid_pytorch(ev, self.num_bins, self.width, self.height, self.device)
            s0 = self.load_img(fwd, self.device)
            s1 = self.load_img(bwd, self.device)
            
            ev_density = event_density(ev.unsqueeze(0))
            feats = torch.cat([ev_density, (s0 - s1).abs()], dim=1)  # [B,2,H,W]
            fused, _ = self.mixer(s0, s1, feats)
            self.save_img(os.path.join(save_dir, f'{start_idx + i:06d}.png'), fused)

    def inference(self, f0, f1, evs_filelist, start_idx, save_dir):
        fwd_filelist, bwd_filelist = self.initialize_inference(f0, f1, evs_filelist, start_idx, save_dir)

        fuse_dir = os.path.join(save_dir, 'fused')
        os.makedirs(fuse_dir, exist_ok=True)
        self.fuse_frames(evs_filelist, fwd_filelist[1:], bwd_filelist[1:], start_idx, fuse_dir)

    @staticmethod
    def load_img(img_path, device):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        return img_tensor

    @staticmethod
    def save_img(save_path, img_tensor):
        img_np = img_tensor.squeeze().cpu().detach().numpy()
        img_np = (img_np * 255.0).astype(np.uint8)
        cv2.imwrite(save_path, img_np)