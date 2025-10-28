import argparse
import cv2
import os

from skimage.metrics import peak_signal_noise_ratio as psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='Path to the predicted video file.')
    parser.add_argument('--gt', type=str, required=True, help='Path to the ground truth video file.')
    parser.add_argument('--start', type=int, default=1, help='Starting frame index.')
    parser.add_argument('--end', type=int, default=100, help='Ending frame index.')
    args = parser.parse_args()

    total_psnr = 0.0
    for idx in range(args.start, args.end + 1):
        pred_file = os.path.join(args.pred, f'frame_{idx:10d}.png')
        gt_file = os.path.join(args.gt, f'{idx:06d}_img.jpg')

        pred_frame = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
        gt_frame = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)

        total_psnr += psnr(gt_frame, pred_frame, data_range=255)

    print('Average PSNR: {:.2f} dB'.format(total_psnr / (args.end - args.start + 1)))