import argparse
import glob
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys
import os
from typing import Union
import pandas as pd

import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)
sys.path.append(current_dir)

from data_loader import DataLoader
from OutlierDetection import OutlierDetection, ENDINGS, print_and_write

needed_frames_dict = {
    '011_Chicago_Billionaires_Millionaires_Lake_Shore_Mansions_The_North_Shore_15': [3465, 3466, 3467, 3468, 3469],
    'video_0284_15': [190],
    'video_0320_15': [404],
    'video_0869_15': [222],
    'video_0171_15': [252]
}
def main():
    parser = argparse.ArgumentParser(description="Used to identify which scenarios pass and which fail")
    parser.add_argument('--dataset',
                        type=str,
                        choices=['External_Jutah', 'OpenPilot_2k19', 'OpenPilot_2016'],
                        default=['OpenPilot_2016', 'OpenPilot_2k19', 'External_Jutah'],
                        nargs='+',
                        help="The dataset to use. Choose between 'External_Jutah', 'OpenPilot_2k19', or 'OpenPilot_2016'.")
    parser.add_argument('--dataset_directory',
                        type=str,
                        help="The location of the dataset")
    parser.add_argument("--cache_only",
                        action='store_true',
                        help="If set, bypass searching for videos")
    args = parser.parse_args()
    versions = []
    if args.cache_only:
        versions = ['2022_04', '2022_07', '2022_11', '2023_03', '2023_06']
    datasets = args.dataset
    for dataset in datasets:
        DATASET_DIRECTORY = f"{args.dataset_directory}/{dataset}"
        # Get all video files
        if args.cache_only:
            print('Checking cache')
            video_file_paths = glob.glob(f"./cache/*.pkl")
            video_filenames = [os.path.basename(v)[:-4] for v in video_file_paths]
            if dataset == 'OpenPilot_2016':
                video_filenames = [v for v in video_filenames if os.path.basename(v).startswith('2016-')]
            elif dataset == 'OpenPilot_2k19':
                video_filenames = [v for v in video_filenames if os.path.basename(v).startswith('video_')]
            elif dataset == 'External_Jutah':
                video_filenames = [v for v in video_filenames if os.path.basename(v)[3] == '_']
        else:
            print(f'Searching in {DATASET_DIRECTORY}')
            video_file_paths = glob.glob(f"{DATASET_DIRECTORY}/1_ProcessedData/*.mp4")
            video_filenames = [os.path.basename(v)[:-4] for v in video_file_paths]
        print(f'Found: {len(video_filenames)}')
        video_filenames = sorted(video_filenames)
        for video_filename in tqdm(video_filenames, desc="Processing Video", leave=False, position=0,
                                   total=len(video_filenames)):
            # print(video_filename)
            if video_filename not in needed_frames_dict:
                continue
            dl = DataLoader(filename=video_filename, data_path=args.dataset_directory, bypass_checks=args.cache_only)
            if not args.cache_only:
                dl.validate_h5_files()
            dl.load_data(terminal_print=True, refresh_cache=False)
            if len(versions) == 0:
                versions.extend(dl.versions)
            needed_frames = needed_frames_dict[video_filename]
            outer_save_dir = f'{args.dataset_directory}/referenced_examples/sut4/{dataset}/cache/{video_filename}/'
            od = OutlierDetection(dl.readings, versions)
            dl.save_frames(needed_frames, outer_save_dir,
                           steering={frame: dl.readings[:, frame] for frame in needed_frames},
                           labels=['AV1', 'AV2', 'AV3', 'AV4', 'SUT'])
            for frame in needed_frames:
                with open(f'{outer_save_dir}/output_{frame}.txt', 'w') as f:
                    start_index = frame
                    end_index = frame + 1
                    np.set_printoptions(precision=2, suppress=True)
                    print_and_write('readings', f)
                    print_and_write(od.readings[:, start_index - 2:end_index + 2], f)
                    print_and_write('potential_outliers', f)
                    print_and_write(od.potential_outliers[4, start_index - 2:end_index + 2], f)
                    print_and_write('outlier_probs', f)
                    print_and_write(100 * od.outlier_probs[start_index - 2:end_index + 2], f)
                    print_and_write('gap_arr', f)
                    print_and_write(od.gap_arr[start_index - 2:end_index + 2], f)
                    print_and_write('range_arr', f)
                    print_and_write(od.range_arr[start_index - 2:end_index + 2], f)
                    np.set_printoptions(precision=3, suppress=True)
                    print_and_write('outlier_q', f)
                    print_and_write(od.outlier_q[start_index - 2:end_index + 2], f)


if __name__ == "__main__":
    main()
