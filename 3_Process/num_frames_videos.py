import argparse
import glob
import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
import sys
import os
from typing import Union

import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from data_loader import DataLoader
from OutlierDetection import OutlierDetection, ENDINGS, print_and_write

def main():
    parser = argparse.ArgumentParser(description="Used to identify which scenarios pass and which fail")
    parser.add_argument('--dataset',
                        type=str,
                        choices=['External_Jutah', 'OpenPilot_2k19', 'OpenPilot_2016'],
                        default=['OpenPilot_2016', 'External_Jutah', 'OpenPilot_2k19'],
                        nargs='+',
                        help="The dataset to use. Choose between 'External_Jutah', 'OpenPilot_2k19', or 'OpenPilot_2016'.")
    parser.add_argument('--dataset_directory',
                        type=str,
                        help="The location of the dataset")
    parser.add_argument("--cache_only",
                        action='store_true',
                        help="If set, bypass searching for videos")
    args = parser.parse_args()
    severity = 20
    conf = 0.90
    confs = [0.5, 0.75, 0.9, 0.95, 0.99]
    sevs = [10, 20, 30, 40, 50]
    data = {dataset: {(conf, sev): {'conf': [0, 0], 'sev': [0, 0], 'both': [0, 0]} for (conf, sev) in
                      itertools.product(confs, sevs)} for dataset in args.dataset}
    for dataset in args.dataset:
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
        versions = []
        if args.cache_only:
            versions = ['2022_04', '2022_07', '2022_11', '2023_03', '2023_06']
        num_videos = 0
        total_failures = 0
        for video_filename in tqdm(video_filenames, desc="Processing Video", leave=False, position=0,
                                   total=len(video_filenames)):
            dl = DataLoader(filename=video_filename, data_path=args.dataset_directory, bypass_checks=args.cache_only)
            if not args.cache_only:
                dl.validate_h5_files()
            dl.load_data(terminal_print=False, refresh_cache=False)
            if len(versions) == 0:
                versions.extend(dl.versions)
            readings = dl.readings
            od = OutlierDetection(readings, versions, f'./figures/{dataset}')
            for (conf, sev) in itertools.product(confs, sevs):
                num_confs = np.count_nonzero((od.outlier_probs >= conf) & od.potential_outliers[4])
                num_sevs = np.count_nonzero((od.gap_arr >= sev) & od.potential_outliers[4])
                num_both = np.count_nonzero((od.gap_arr >= sev) & (od.outlier_probs >= conf) & od.potential_outliers[4])
                if num_confs > 0:
                    data[dataset][(conf, sev)]['conf'][0] += num_confs
                    data[dataset][(conf, sev)]['conf'][1] += 1
                if num_sevs > 0:
                    data[dataset][(conf, sev)]['sev'][0] += num_sevs
                    data[dataset][(conf, sev)]['sev'][1] += 1
                if num_both > 0:
                    data[dataset][(conf, sev)]['both'][0] += num_both
                    data[dataset][(conf, sev)]['both'][1] += 1
    conf = 0.90
    choices = [(conf, 10), (conf, 20), (conf, 30), (conf, 40), (conf, 50)]
    with open(f'./gen_figures/table5.txt', 'w') as f:
        for index, (conf, sev) in enumerate(choices):
            row = ''
            if index == 0:
                row += f'\\multirow{{{len(choices)}}}{{*}}{{{int(100 * conf)}\\%}}'
            row += f' & {sev}\\textdegree '
            for dataset in ['OpenPilot_2016', 'OpenPilot_2k19', 'External_Jutah']:
                for dtype in ['conf', 'sev', 'both']:
                    if dtype == 'conf':
                        if index > 0:
                            row += ' & & '
                        else:
                            row += f' & \\multirow{{{len(choices)}}}{{*}}{{{data[dataset][(conf, sev)][dtype][0]}}} & \\multirow{{{len(choices)}}}{{*}}{{{data[dataset][(conf, sev)][dtype][1]}}}'
                    else:
                        row += f' & {data[dataset][(conf, sev)][dtype][0]} '
                        row += f' & {data[dataset][(conf, sev)][dtype][1]} '
            row += ' \\\\'
            print_and_write(row, f)


if __name__ == "__main__":
    main()
