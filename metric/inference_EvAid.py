import pandas as pd

import argparse
import subprocess
import os

def main(experiment_name, recons_ckpt):
    output_root = '/mnt/D/baichenxu/code/rpg_e2vid/outputs/EvAid/metric_recons'
    print(f'Starting batch processing for experiment: {experiment_name}')
    print(f'Reconstruction checkpoint: {recons_ckpt}')
    
    df = pd.read_csv('EvAid_fps.csv')
    for index, row in df.iterrows():
        dataset_root = row['dataset_root']
        dataset_name = row['dataset_name']
        fps = row['fps']
        delta_time = row['delta_time']
        delta_frame = row['delta_frame']
        output_folder = os.path.join(output_root, dataset_name + '_' + experiment_name)
        pred_folder = os.path.join(output_folder, 'fused')
        gt_folder = os.path.join(dataset_root, dataset_name, 'gt')
        
        print(f'Processing dataset: {dataset_name}')
        print(f'Root: {dataset_root}, FPS: {fps}, Delta Time: {delta_time}, Delta Frame: {delta_frame}')
        # Add your processing logic here
        subprocess.run([
            'python', 'test_BiApEVID_EvAid.py',
            '--dataset_root', dataset_root,
            '--data', dataset_name,
            '--output_folder', output_folder,
            '--delta_frame', str(delta_frame),
            '--experiment_name', experiment_name,
            '--recons_ckpt', recons_ckpt
        ], check=True)
        
        subprocess.run([
            'python', 'metric_EvAid.py',
            '--pred', pred_folder,
            '--gt', gt_folder,
            '--start', '1'
        ], check=True)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch process EvAid datasets')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--recons_ckpt', type=str, help='Path to the reconstruction model checkpoint')
    args = parser.parse_args()
    main(args.experiment_name, args.recons_ckpt)