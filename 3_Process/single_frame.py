import argparse
import glob
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
# from matplotlib_venn import venn2

conf_color = 'red'
severity_color = 'green'


def single_frame_failures_hist(od, sut: Union[str, int, None], clip=None, confline=None, severityline=None, vmax=None):
    if type(sut) is str:
        sut = od.versions.index(sut)
    if sut is None:
        prob = np.copy(od.outlier_probs)
        gap_arr = np.copy(od.gap_arr)
        finite = np.where(np.isfinite(gap_arr) & np.isfinite(prob))
        prob = prob[finite]
        gap_arr = gap_arr[finite]
    else:
        sut_failures = np.where(od.potential_outliers[sut])
        prob = od.outlier_probs[sut_failures]
        gap_arr = od.gap_arr[sut_failures]
    if clip is not None:
        gap_arr = np.clip(gap_arr, 0, clip)
    prob = 100 * prob
    heatmap, xedges, yedges = np.histogram2d(prob, gap_arr, range=[[0, 100], [0, clip if clip is not None else max(gap_arr)]], bins=[100, 90])
    print('Max', np.amax(heatmap))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    print(extent)
    plt.figure()
    heatmap[heatmap == 0] = np.nan  # mask out values that are exactly 0
    im = plt.imshow(heatmap.T, extent=extent, origin='lower', vmax=vmax)
    plt.colorbar(im)
    if confline is not None:
        # plt.vlines(xedges[0] + (xedges[-1]-xedges[0]) * confline / clip, yedges[0], yedges[-1], label=f'{confline}% Confident')
        plt.vlines(confline, 0, clip, colors=conf_color, linestyles='--', label=f'{confline}% Confident')
    if severityline is not None:
        # ls = ['dashdot', 'dotted', (0, (1, 10)), (5, (10, 3)), (0, (3, 10, 1, 10))]
        for index, sevline in enumerate(severityline):
            plt.hlines(sevline, 0, 100, colors=severity_color, linestyles='-.',
                        label=f'Severity' if index == 0 else '_')
                       # label=f'{sevline}$^\circ$ Severity')
    plt.ylabel('Severity (degrees)')
    sut_label = od.versions[sut] if sut is not None else 'any'
    plt.xlabel(f'Confidence (%)')
    # plt.tight_layout()
    plt.legend()
    clip_ending = '' if clip is None else f'_{clip}'
    vmax_string = f'_{vmax}' if vmax is not None else ''
    for ending in ENDINGS:
        plt.savefig(f'{od.label}_conf_vs_value_hist_sut_{sut_label}{clip_ending}_{vmax_string}_5line.{ending}')
    # plt.show()


# def single_frame_failures_venn(od, sut: Union[str, int, None], clip=None, confline=None, severityline=None):
#     # plt.figure(figsize=(10, 10))
#     plt.figure()
#     if type(sut) is str:
#         sut = od.versions.index(sut)
#     if sut is None:
#         prob = np.copy(od.outlier_probs)
#         gap_arr = np.copy(od.gap_arr)
#         finite = np.where(np.isfinite(gap_arr) & np.isfinite(prob))
#         prob = prob[finite]
#         gap_arr = gap_arr[finite]
#     else:
#         sut_failures = np.where(od.potential_outliers[sut])
#         prob = od.outlier_probs[sut_failures]
#         gap_arr = od.gap_arr[sut_failures]
#     if clip is not None:
#         gap_arr = np.clip(gap_arr, 0, clip)
#     prob = 100 * prob
#     conf_match = set(np.where(prob >= confline)[0])
#     sev_match = set(np.where(gap_arr >= severityline)[0])
#     total = len(conf_match.union(sev_match))
#     venns = venn2([conf_match, sev_match],
#                    set_labels=(f'Confidence $\geq{confline}$%', f'Severity $\geq{severityline}^\circ$'),
#                    set_colors=(conf_color, severity_color)
#                    , subset_label_formatter=lambda x: f"{x}\n({(x/total):0.2%})"
#                    )
#     plt.tight_layout()
#     for ending in ENDINGS:
#         plt.savefig(f'{od.label}_venn.{ending}')

# def single_frame_failures_venn_max(od, sut: Union[str, int, None], clip=None, confline=None, severityline=None):
#     # plt.figure(figsize=(10, 10))
#     plt.figure()
#     if type(sut) is str:
#         sut = od.versions.index(sut)
#     if sut is None:
#         prob = np.copy(od.outlier_probs)
#         gap_arr = np.copy(od.gap_arr)
#         finite = np.where(np.isfinite(gap_arr) & np.isfinite(prob))
#         prob = prob[finite]
#         gap_arr = gap_arr[finite]
#     else:
#         sut_failures = np.where(od.potential_outliers[sut])
#         prob = od.outlier_probs[sut_failures]
#         gap_arr = od.gap_arr[sut_failures]
#     if clip is not None:
#         gap_arr = np.clip(gap_arr, 0, clip)
#     prob = 100 * prob
#     conf_match = set(np.where(prob < confline)[0])
#     sev_match = set(np.where(gap_arr >= severityline)[0])
#     total = len(conf_match.union(sev_match))
#     venns = venn2([conf_match, sev_match],
#                    set_labels=(f'Confidence $\leq{confline}$%', f'Severity $\geq{severityline}^\circ$'),
#                    set_colors=(conf_color, severity_color)
#                    , subset_label_formatter=lambda x: f"{x}\n({(x/total):0.2%})"
#                    )
#     plt.tight_layout()
#     for ending in ENDINGS:
#         plt.savefig(f'{od.label}_venn_{severityline}_{confline}_max.{ending}')
    

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
    all_merged = []
    os.makedirs('./figures', exist_ok=True)
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
        all_merged.extend(all_data)
        all_data = np.concatenate(all_data, axis=1)
        print(dataset)
        od = OutlierDetection(all_data, versions, f'./gen_figures/{dataset}')
        single_frame_failures_hist(od, 4, clip=90, confline=90, severityline=[10, 20, 30, 40, 50], vmax=None)
        single_frame_failures_hist(od, 4, clip=90, confline=90, severityline=[10, 20, 30, 40, 50], vmax=2750)
        # single_frame_failures_venn(od, 4, confline=99, severityline=20)
        # for conf in [10, 20, 50, 75, 90, 99]:
        #     single_frame_failures_venn_max(od, 4, clip=90, confline=conf, severityline=20)
    all_merged = np.concatenate(all_merged, axis=1)
    od = OutlierDetection(all_merged, versions, f'./gen_figures/all')
    single_frame_failures_hist(od, 4, clip=90, confline=90, severityline=[10, 20, 30, 40, 50], vmax=2750)
    # single_frame_failures_venn(od, 4, confline=99, severityline=20)
    # for conf in [10, 20, 50, 75, 90, 99]:
    #     single_frame_failures_venn_max(od, 4, clip=90, confline=conf, severityline=20)
        

if __name__ == "__main__":
    main()