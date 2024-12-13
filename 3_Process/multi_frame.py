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

from data_loader import DataLoader
from OutlierDetection import OutlierDetection, ENDINGS, print_and_write


def find_maximal_failure(od, sut, sev, conf):
    return od.find_maximal_system_failure(sut, sev, conf/100)

def handle_video(video_filename, dataset_directory, dataset, severities, confidences, cache_only, threaded=True):
    dl = DataLoader(filename=video_filename, data_path=dataset_directory, bypass_checks=cache_only)
    if not cache_only:
        dl.validate_h5_files()
    dl.load_data(terminal_print=False, refresh_cache=False)
    readings = dl.readings
    if cache_only:
        versions = ['2022_04', '2022_07', '2022_11', '2023_03', '2023_06']
    else:
        versions = dl.versions
    od = OutlierDetection(readings, versions, f'./figures/{dataset}')
    maximal = {}
    if threaded:
        with Pool() as p:
            jobs = {}
            for sev in severities:
                for conf in confidences:
                    jobs[(sev, conf)] = p.apply_async(find_maximal_failure, (od, 4, sev, conf))
            for key, job in jobs.items():
                res = job.get()
                if len(res.keys()) == 0:
                    val = 0
                else:
                    val = max(res.keys())
                maximal[key] = val
    else:
        for sev in severities:
            for conf in confidences:
                res = find_maximal_failure(od, 4, sev, conf)
                if len(res.keys()) == 0:
                    val = 0
                else:
                    val = max(res.keys())
                maximal[(sev, conf)] = val
    return maximal

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
    parser.add_argument('--severities',
                    type=float,
                    # default=[10, 20, 45, 90, 180],
                    default=[10, 20, 30, 40, 50],
                    nargs='+',
                    help="The severities to check for")
    parser.add_argument('--confidences',
                    type=float,
                    default=[50, 75, 90, 95, 99],
                    nargs='+',
                    help="The confidences to check for as a percent")
    parser.add_argument("--cache_only",
                        action='store_true',
                        help="If set, bypass searching for videos")
    args = parser.parse_args()
    severities = args.severities
    confidences = args.confidences
    dataframes = {}
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
        versions = []
        if args.cache_only:
            versions = ['2022_04', '2022_07', '2022_11', '2023_03', '2023_06']
        maximal = None
        with Pool() as p:
            jobs = {}
            for video_filename in video_filenames:
                jobs[video_filename] = p.apply_async(handle_video, (video_filename, args.dataset_directory, dataset, severities, confidences, args.cache_only, False))
            for video_filename in tqdm(video_filenames, desc="Processing Video", leave=False, position=0,
                                   total=len(video_filenames)):
                res = jobs[video_filename].get()
                if maximal is None:
                    maximal = res
                else:
                    for key, val in maximal.items():
                        maximal[key] = max(res[key], val)
        data_dict = {f'${sev}^\circ$': [maximal[(sev, conf)] for conf in confidences] for sev in severities}
        data = pd.DataFrame.from_dict(data_dict, columns=[f'{conf}%' for conf in confidences], orient='index')
        dataframes[dataset] = data
        data.to_csv(f'{dataset}.csv')
    for dataset, data in dataframes.items():
        print(dataset)
        print(data.to_latex())
    print('concat', datasets)
    data = pd.concat([dataframes[x] for x in datasets], axis=1)
    data.to_csv('all.csv')
    with open(f'./gen_figures/table6.txt', 'w') as f:
        print_and_write(data.to_latex(), f)

if __name__ == "__main__":
    main()