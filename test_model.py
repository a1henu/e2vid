import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np

from torch.utils.data import DataLoader
from utils.event_utils import event_density, reverse_voxels
from dataset.dataset import SeqDataset
from utils.data_utils import SeqCrop128
from model.BiApEVID import BiApEVID
from model.model import E2VIDRecurrent
    
if __name__ == "__main__":
    dataroot = "/mnt/D/baichenxu/datasets/ApEvid_dataset"
    # adapter_ckpt = torch.load("./logs/20251030/ada_e2vid_model/adapter_best.pth.tar")['state_dict']
    # e2vid_ckpt   = torch.load("./logs/20251030/ada_e2vid_model/e2vid_best.pth.tar")['state_dict']
    biape2vid_ckpt = torch.load("./logs/20251030/biape2vid_model/biape2vid_best.pth.tar")['state_dict']
    
    width = 128
    height = 128
    options = {
        'height': height,
        'width': width,
        'num_bins': 5,
        'num_encoders': 3,
        'base_num_channels': 32,
        'norm': 'BN',
        'use_upsample_conv': False,
    }
    
    dataset = SeqDataset(
        root=dataroot, 
        split="val", 
        transform=SeqCrop128(mode="random", pad_if_small=True),
        num_iter=30
    )
    
    # adapter = E2VIDRecurrent(options).to('cuda')
    # e2vid   = E2VIDRecurrent(options).to('cuda')
    # adapter.load_state_dict(adapter_ckpt)
    # e2vid.load_state_dict(e2vid_ckpt)
    model = BiApEVID(E2VIDRecurrent(options), E2VIDRecurrent(options)).to('cuda')
    model.load_state_dict(biape2vid_ckpt)
    model.eval()
    
    sample = dataset[59]
    frames = sample['frames'].unsqueeze(0).to('cuda')  # [1,T+1,1,H,W]
    events = sample['events'].unsqueeze(0).to('cuda')  # [1,T,Bin,H,W]
    f0, f1 = frames[:,0], frames[:,-1]
    with torch.no_grad():
        fused, ws, adapter_i0, adapter_i1, fwd, bwd = model(f0, f1, events)
    
    fwd_imgs = [img.squeeze().cpu().numpy() for img in fwd]
    bwd_imgs = [img.squeeze().cpu().numpy() for img in bwd]
    fused_imgs = [img.squeeze().cpu().numpy() for img in fused.squeeze(0)]
    gt_imgs = [img.squeeze().cpu().numpy() for img in frames.squeeze(0)[1:-1]]
    ws = [w.squeeze().cpu().numpy() for w in ws.squeeze(0)]

    combined_imgs = [np.hstack((fwd_img, bwd_img, fused_img, gt_img)) for fwd_img, bwd_img, fused_img, gt_img in zip(fwd_imgs, bwd_imgs, fused_imgs, gt_imgs)]
    long_img = np.vstack(combined_imgs)
    long_img = (long_img * 255).astype(np.uint8)
    cv2.imwrite("test_biape2vid_long_image.png", long_img)
    
    ws_img = np.vstack([np.hstack((w[0], w[1])) for w in ws])
    ws_img = (ws_img * 255).astype(np.uint8)
    cv2.imwrite("test_biape2vid_attention_weights.png", ws_img)