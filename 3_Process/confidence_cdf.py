import argparse
import glob
import pickle

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
from OutlierDetection import OutlierDetection, ENDINGS

conf_color = 'red'
severity_color = 'green'

dataset_labels = {
    'OpenPilot_2016': 'comma.ai 2016',
    'OpenPilot_2k19': 'comma.ai 2k19',
    'External_Jutah': 'External JUtah',
}
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
    confidences = {}
    x = np.linspace(1, 0, 1001)
    if not os.path.exists('conf.pkl'):
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
            versions = []
            if args.cache_only:
                versions = ['2022_04', '2022_07', '2022_11', '2023_03', '2023_06']
            all_data = []
            for video_filename in tqdm(video_filenames, desc="Processing Video", leave=False, position=0,
                                       total=len(video_filenames)):
                dl = DataLoader(filename=video_filename, data_path=args.dataset_directory, bypass_checks=args.cache_only)
                if not args.cache_only:
                    dl.validate_h5_files()
                dl.load_data(terminal_print=False, refresh_cache=False)
                if len(versions) == 0:
                    versions.extend(dl.versions)
                readings = dl.readings
                all_data.append(readings)
            all_data = np.concatenate(all_data, axis=1)
            print(dataset)
            od = OutlierDetection(all_data, versions, f'./gen_figures/{dataset}')
            conf = np.zeros(od.gap_arr.shape)
            conf[od.potential_outliers[4]] = od.outlier_probs[od.potential_outliers[4]]
            confidences[dataset] = conf
        with open('conf.pkl', 'wb') as f:
            pickle.dump(confidences, f)
    else:
        with open('conf.pkl', 'rb') as f:
            confidences = pickle.load(f)
    max_y = 0
    yticks = [0, 2, 4, 6, 8]
    new_yticks = []
    n90 = 0
    n99 = 0
    total = 0
    for index, dataset in enumerate(dataset_labels):
        conf = confidences[dataset]
        y = [100 * np.count_nonzero(conf >= i) / len(conf) for i in x]
        max_y = max(max_y, max(y[:-1]))
        plt.plot(x[:-1] * 100, y[:-1], label=dataset_labels[dataset], color=f'C{index}')
        perc_90 = 100 * np.count_nonzero(conf >= 0.9) / len(conf)
        print(f'Dataset: {dataset}')
        n90 += np.count_nonzero(conf >= 0.9)
        n99 += np.count_nonzero(conf >= 0.99)
        total += len(conf)
        print(f'>90% {100 * np.count_nonzero(conf >= 0.9) / len(conf)}%, {np.count_nonzero(conf >= 0.9)}/{len(conf)}')
        print(f'>99% {100 * np.count_nonzero(conf >= 0.99) / len(conf)}%, {np.count_nonzero(conf >= 0.99)}/{len(conf)}')
        new_yticks.append(perc_90)
        plt.hlines(perc_90, 90, 100, color=f'C{index}', linestyles='dashed')
    print(f'All')
    print(f'>90% {n90/total:%}, {n90}/{total}')
    print(f'>99% {n99 / total:%}, {n99}/{total}')
    plt.vlines(90, 0, max_y, color='r', linestyles='dashed', label='90% Confidence')
        # plt.scatter(x[-1], y[-1], color=f'C{index}', label='_')
    ytick_labels = [str(i) for i in yticks]
    yticks.extend(new_yticks)
    ytick_labels.extend([f'{i:0.3f}' + ('\n' if index == 1 else '')  for index, i in enumerate(new_yticks)])
    print(yticks)
    plt.gca().set_yticks(yticks, labels=ytick_labels)
    # plt.gca().tick_params(axis='y', rotation=45)
    # plt.xlim((0, 100))
    # plt.ylim((0, 9))
    plt.gca().invert_xaxis()
    plt.ylabel('Percentage of Dataset $> X$ Confidence')
    plt.xlabel('Confidence (%)')
    plt.legend()
    for yl in [0, 2, 4, 6, 8]:
        plt.gca().axhline(yl, color='gray', linewidth=0.5, linestyle='dotted')
    for xl in [0, 20, 40, 60, 80, 100]:
        plt.gca().axvline(xl, color='gray', linewidth=0.5, linestyle='dotted')
    for ending in ENDINGS:
        plt.savefig(f'./gen_figures/conf_cdf.{ending}')
    plt.figure()
    x = x[::-1]
    for index, dataset in enumerate(dataset_labels):
        conf = confidences[dataset]
        y = [100*np.count_nonzero(conf < i) / len(conf) for i in x]
        plt.plot(x[1:]*100, y[1:], label=dataset_labels[dataset], color=f'C{index}')
        # plt.scatter(x[-1], y[-1], color=f'C{index}', label='_')

    plt.ylabel('Percentage of Dataset $\leq X$ Confidence')
    plt.xlabel('Confidence (%)')
    plt.legend()
    for ending in ENDINGS:
        plt.savefig(f'./gen_figures/conf_cdf_reversed.{ending}')



if __name__ == "__main__":
    main()