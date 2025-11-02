import argparse
import cv2
import os
import torch
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import lpips

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='Path to the predicted video file.')
    parser.add_argument('--gt', type=str, required=True, help='Path to the ground truth video file.')
    parser.add_argument('--start', type=int, default=1, help='Starting frame index.')
    args = parser.parse_args()

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    loss_fn = lpips.LPIPS(net='vgg').to('cuda')
    
    pred_filelists = sorted(os.listdir(args.pred))
    gt_filelists = sorted(os.listdir(args.gt))[args.start:]

    with open(os.path.join(args.pred, '../files.txt'), 'w') as f:
        for pred, gt in zip(pred_filelists, gt_filelists):
            f.write(f'Predicted: {pred}, Ground Truth: {gt}\n')

    length = len(pred_filelists)
    
    for pred, gt in zip(pred_filelists, gt_filelists):
        pred_path = os.path.join(args.pred, pred)
        gt_path = os.path.join(args.gt, gt)
        
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        total_psnr += psnr(gt_img, pred_img, data_range=1.0)
        total_ssim += ssim(gt_img, pred_img, data_range=1.0)

        pred_tensor = torch.from_numpy(pred_img).unsqueeze(0).unsqueeze(0).to('cuda')
        gt_tensor = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0).to('cuda')
        lpips_value = loss_fn(pred_tensor, gt_tensor, normalize=True).item()
        total_lpips += lpips_value

    print(f'Average PSNR: {total_psnr / length:.2f} dB')
    print(f'Average SSIM: {total_ssim / length:.4f}')
    print(f'Average LPIPS: {total_lpips / length:.4f}')

    with open(os.path.join(args.pred, '../metrics.txt'), 'w') as f:
        f.write(f'Average PSNR: {total_psnr / length:.2f} dB\n')
        f.write(f'Average SSIM: {total_ssim / length:.4f}\n')
        f.write(f'Average LPIPS: {total_lpips / length:.4f}\n')