import itertools
import math
import os
import sys
import glob
import operator
import argparse
from multiprocessing import Pool
from numbers import Number
from random import shuffle
from typing import Union, List

import numpy as np
import pandas as pd
from statsmodels.compat import scipy

from tqdm import tqdm
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy.integrate as integrate

import scipy.stats as stats

current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from data_loader import DataLoader
from constants import CLIPPING_DEGREE

# from constants import CLIPPING_DEGREE
ENDINGS = ['png', 'pdf']


def dixon(all_data):
    # sort each column, taken from https://stackoverflow.com/a/43218732
    sidx = all_data.argsort(axis=0)
    all_data = all_data[sidx, np.arange(sidx.shape[1])]
    range_arr = np.max(all_data, axis=0) - np.min(all_data, axis=0)
    gap_arr = np.max(np.array([np.abs(all_data[1, :] - all_data[0, :]),
                               np.abs(all_data[-1, :] - all_data[-2, :])]), axis=0)
    q = gap_arr / range_arr
    prob = direct_dixon(q, all_data.shape[0])
    return prob, q, range_arr, gap_arr


def direct_dixon(q, sample_count):
    if sample_count == 3:
        prob = dixon_three(q)
    elif sample_count == 4:
        prob = dixon_four(q)
    elif sample_count == 5:
        prob = dixon_five(q)
    else:
        raise NotImplementedError(f'Dixon not implemented for {sample_count} samples')
    # return -1 + 2*prob
    return prob


def dixon_four(q):
    return (5 - (6 / np.pi) * (
            np.arctan(np.sqrt(4 * np.power(q, 2) - 4 * q + 3)) + np.arctan(np.sqrt(3 * np.power(q, 2) - 4 * q + 4) / q)
    ))


def h(r):
    return ((2-r)/(np.sqrt(3*r*r-4*r+4)))*np.arctan(((1-r)*np.sqrt(5*(3*r*r-4*r+4)))/(3*r*r-3*r+4))


def dixon_five_cdf(r):
    if r == 0:
        return 0
    return 15*(h(r)+h(1/r))/(np.pi*np.pi*(r*r-r+1))


def __integrate_dixon_five(R):
    return integrate.quad(dixon_five_cdf, 0, R)


def dixon_five(q):
    try:
        with Pool(28) as p:
            vals = p.map(__integrate_dixon_five, q)
    except AssertionError:
        # The caller is using multiprocessing, so we can't use it here.
        vals = [__integrate_dixon_five(x) for x in q]
    # vals = [integrate.quad(dixon_five_cdf, 0, R) for R in q]
    print(f'maximal intregration error {max([val[1] for val in vals])}')
    return np.array([val[0] for val in vals])


def dixon_three(q):
    one_side_prob = (3 / np.pi) * (np.arctan((2 / np.sqrt(3)) * (q - 0.5))) + 0.5
    return one_side_prob


def has_contiguous(arr: List[bool], true_count: int):
    """
    Returns true iff arr has at least true_count consecutive True values
    """
    longest = 0
    current = 0
    for i in arr:
        if i:
            current += 1
        else:
            longest = max(longest, current)
    longest = max(longest, current)
    return longest >= true_count


def contiguous_outlier_prob(possible_seqs, sut_not_outlier_prob, sut_outlier_prob, test_length):
    if np.product(sut_outlier_prob) == 0:
        return 0
    cumulative_prob = 0
    for possible_seq in possible_seqs:
        frame_probs = np.zeros(test_length)
        frame_probs[possible_seq] = sut_outlier_prob[possible_seq]
        frame_probs[~possible_seq] = sut_not_outlier_prob[~possible_seq]
        prob = np.product(frame_probs)
        cumulative_prob += prob
    return cumulative_prob


def contiguous_outlier_prob_range(range_start, num, possible_seqs, sut_not_outlier_prob, sut_outlier_prob, test_length):
    probs = []
    for starting_index in range(range_start, range_start+num):
        prob = contiguous_outlier_prob(possible_seqs,
                                       sut_not_outlier_prob[starting_index:starting_index + test_length],
                                       sut_outlier_prob[starting_index:starting_index + test_length],
                                       test_length)
        probs.append(prob)
    return probs


class OutlierDetection:
    def __init__(self, readings, versions, label=''):
        self.readings = readings
        self.label = label
        self.num_readings = self.readings.shape[1]
        self.versions = versions
        self.last_version = self.versions[-1]
        self.outlier_probs, self.outlier_q, self.range_arr, self.gap_arr = dixon(self.readings)
        self.potential_outliers = []
        self.mean = self.readings.mean(axis=0)
        for version_index in range(len(self.versions)):
            sut_readings = self.readings[version_index, :]
            other_readings = self.readings[[i for i in range(len(self.versions)) if i != version_index], :]
            potential_outlier = np.all(np.abs(sut_readings - self.mean) > np.abs(other_readings - self.mean), axis=0)
            # judge potential outlier based on if SUT is farthest from the mean
            # potential_outlier = (np.all(sut_readings > other_readings, axis=0)
            #                      | np.all(sut_readings < other_readings, axis=0))
            # po1 = np.all(np.abs(sut_readings - self.mean) > np.abs(other_readings - self.mean), axis=0)
            # po2 = (np.all(sut_readings > other_readings, axis=0)
            #                      | np.all(sut_readings < other_readings, axis=0))
            # np.where(po1 & ~po2)
            self.potential_outliers.append(potential_outlier)
            print(f'version {version_index} potential outlier in {np.count_nonzero(potential_outlier)} readings of {self.num_readings} ({100*np.count_nonzero(potential_outlier)/self.num_readings}%)')
        self.potential_outliers = np.array(self.potential_outliers)
        print(self.versions)

    def single_frame_failures(self, sut: Union[str, int, None], failure_prob_thresh: float, severity_thresh: float):
        if type(sut) is str:
            sut = self.versions.index(sut)
        failure_criteria = (self.outlier_probs >= failure_prob_thresh) & (self.gap_arr >= severity_thresh)
        if sut is not None:
            found = (self.potential_outliers[sut]) & failure_criteria
        else:
            found = failure_criteria
        print(f'Found {found.sum()} values of {self.num_readings} ({100*found.sum()/self.num_readings:0.4f}%)')
        return np.where(found)

    def single_frame_binned_value_hist(self, sut, value_bins=None):
        if value_bins is None:
            value_bins = [10, 45, 90]
        if type(sut) is str:
            sut = self.versions.index(sut)
        if sut is None:
            prob = np.copy(self.outlier_probs)
            gap_arr = np.copy(self.gap_arr)
            finite = np.where(np.isfinite(gap_arr) & np.isfinite(prob))
            prob = prob[finite]
            gap_arr = gap_arr[finite]
        else:
            sut_failures = np.where(self.potential_outliers[sut])
            prob = self.outlier_probs[sut_failures]
            gap_arr = self.gap_arr[sut_failures]
        for index in range(len(value_bins)):
            # keep_indices = np.where((value_bins[index] < gap_arr) & (gap_arr <= value_bins[index+1])) \
            #     if index < len(value_bins) - 1 else np.where((value_bins[index] < gap_arr))
            keep_indices = np.where(value_bins[index] <= gap_arr)
            keep_prob = prob[keep_indices]
            # keep_gap = gap_arr[keep_indices]
            # sev_string = f'{value_bins[index]} < S' + (f' <= {value_bins[index + 1]}' if index < len(value_bins) - 1 else '')
            sev_string = f'{value_bins[index]} <= S'
            # plt.hist(keep_prob, bins=np.linspace(0.8, 1, 21), label=f'Severity {sev_string}', log=True)
            plt.hist(keep_prob, bins=np.linspace(0, 1, 101), label=f'Severity {sev_string}')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.show()


    def find_maximal_system_failure(self, sut, severity_thresh, failure_prob_thresh, show=False, severity_thresh_max=None, conf_thresh_max=None):
        if type(sut) is str:
            sut = self.versions.index(sut)
        satisfying_outliers = (self.gap_arr >= severity_thresh) & self.potential_outliers[sut]
        if severity_thresh_max is not None:
            satisfying_outliers = (self.gap_arr <= severity_thresh_max) & satisfying_outliers
        if np.all(~satisfying_outliers):
            # found nothing, return immediately
            print(f'Found no system failures at {severity_thresh} severity and {failure_prob_thresh} confidence.')
            return {}
        sut_outlier_prob = np.copy(self.outlier_probs)
        sut_outlier_prob[~satisfying_outliers] = 0  # if it didn't meet the criteria then it has 0% chance
        # sut_not_outlier_prob = 1 - sut_outlier_prob
        maximal_length = []
        maximal_dict = defaultdict(list)
        for starting_index in tqdm(range(self.num_readings), disable=True):
            end_frame = None
            for ending_index in range(starting_index + 1, self.num_readings):
                cur_prob = np.prod(sut_outlier_prob[starting_index:ending_index])
                if cur_prob < failure_prob_thresh:
                    end_frame = ending_index - 1
                    break
            if end_frame is None:
                continue
            cur_prob = np.prod(sut_outlier_prob[starting_index:end_frame])
            if cur_prob > failure_prob_thresh:
                duration = end_frame - starting_index
                if duration > 0:
                    maximal_length.append(duration)
                    maximal_dict[duration].append((starting_index, end_frame))
        if show:
            print(f'Severity {severity_thresh}, failure thresh: {failure_prob_thresh}')
            print(f'Maximal System Failure: {max(maximal_length)} at {maximal_dict[max(maximal_length)]}')
            plt.hist(maximal_length, log=True, bins=max(maximal_length) + 1)
            plt.xlabel('Number of frames')
            plt.ylabel('Count')
            plt.title(f'Maximal duration where >{failure_prob_thresh} conf all frames are >{severity_thresh} off for {self.versions[sut]}')
            plt.show()
        return maximal_dict

    def system_failure(self, sut, test_length: int, min_frame_count: int, severity_thresh: float,
                       failure_prob_thresh: Union[float, None] = None,
                       threads: int=1):
        """
        Given an SUT, find all subsequences of tests of length test_length that have a
        probability > failure_prob_thresh to yield min_frame_count consecutive frames of > severity_thresh
        """
        # https://stackoverflow.com/a/14931808
        possible_seqs = [np.array(i) for i in itertools.product([False, True], repeat=test_length)
                         if has_contiguous(i, min_frame_count)]
        satisfying_outliers = (self.gap_arr >= severity_thresh) & self.potential_outliers[sut]
        sut_outlier_prob = np.copy(self.outlier_probs)
        sut_outlier_prob[~satisfying_outliers] = 0  # if it didn't meet the criteria then it has 0% chance
        sut_not_outlier_prob = 1 - sut_outlier_prob
        probs_jobs = []
        found_probs = []
        num_tests = self.num_readings - test_length
        if threads > 1:
            num_per_thread = math.ceil(self.num_readings / threads)
            with Pool(threads) as p:
                starting_index = 0
                while starting_index < num_tests:
                    cumulative_prob = p.apply_async(contiguous_outlier_prob_range,
                                                    (starting_index,
                                                     min(num_per_thread, num_tests - starting_index),
                                                     possible_seqs,
                                                     sut_not_outlier_prob,
                                                     sut_outlier_prob,
                                                     test_length))
                    probs_jobs.append(cumulative_prob)
                    starting_index += num_per_thread
                for job in tqdm(probs_jobs):
                    found_probs.extend(job.get())
        else:
            for starting_index in tqdm(range(self.num_readings - test_length)):
                prob = contiguous_outlier_prob(possible_seqs,
                                               sut_not_outlier_prob[starting_index:starting_index + test_length],
                                               sut_outlier_prob[starting_index:starting_index + test_length],
                                               test_length)
                found_probs.append(prob)
        found_probs = np.array(found_probs)
        n, bins, patches = plt.hist(found_probs, log=True, bins=100)
        if failure_prob_thresh is not None:
            num_greater_than = (found_probs > failure_prob_thresh).sum()
            plt.vlines(failure_prob_thresh, 0, max(n), colors='r',
                       label=f'Number >{failure_prob_thresh:0.2}={num_greater_than} of '
                             f'{num_tests} ({100*num_greater_than / num_tests:0.1f}%)')
        plt.xlabel(f'Given test of length {test_length}, probability of finding {min_frame_count} '
                   f'contiguous failures >{severity_thresh} for {self.versions[sut]}')
        plt.ylabel(f'Count')
        plt.legend()
        failure_prob_thresh_str = '_fail_' + str(failure_prob_thresh).replace('\\.', '_') \
            if failure_prob_thresh is not None else ''
        for ending in ENDINGS:
            plt.savefig(f'length_{test_length}_min_{min_frame_count}_'
                        f'sev_{severity_thresh}{failure_prob_thresh_str}_sut_{self.versions[sut]}.{ending}')
        plt.show()
        return found_probs

    def satisfying_frames(self, sut: Union[str, int], conf=None, severity=None, max_conf=None, max_severity=None):
        if conf is None:
            conf = -np.inf
        if severity is None:
            severity = -np.inf
        if max_conf is None:
            max_conf = np.inf
        if max_severity is None:
            max_severity = np.inf
        if type(sut) is str:
            sut = self.versions.index(sut)
        return np.where(self.potential_outliers[sut] & (self.gap_arr >= severity) & (self.gap_arr <= max_severity) & (self.outlier_probs >= conf) & (self.outlier_probs <= max_conf))


    def single_frame_failures_hist(self, sut: Union[str, int, None], clip=None, confline=None, severityline=None):
        if type(sut) is str:
            sut = self.versions.index(sut)
        if sut is None:
            prob = np.copy(self.outlier_probs)
            gap_arr = np.copy(self.gap_arr)
            finite = np.where(np.isfinite(gap_arr) & np.isfinite(prob))
            prob = prob[finite]
            gap_arr = gap_arr[finite]
        else:
            sut_failures = np.where(self.potential_outliers[sut])
            prob = self.outlier_probs[sut_failures]
            gap_arr = self.gap_arr[sut_failures]
        if clip is not None:
            gap_arr = np.clip(gap_arr, 0, clip)
        prob = 100 * prob
        heatmap, xedges, yedges = np.histogram2d(prob, gap_arr, range=[[0, 100], [0, clip if clip is not None else max(gap_arr)]], bins=[100, 90])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        print(extent)
        plt.clf()
        heatmap[heatmap == 0] = np.nan  # mask out values that are exactly 0
        im = plt.imshow(heatmap.T, extent=extent, origin='lower',
                        # norm=mpl.colors.LogNorm(),
                        # aspect=max(prob) / max(gap_arr)
                        )
        plt.colorbar(im)
        if confline is not None:
            # plt.vlines(xedges[0] + (xedges[-1]-xedges[0]) * confline / clip, yedges[0], yedges[-1], label=f'{confline}% Confident')
            plt.vlines(confline, 0, clip, colors='red', linestyles='--', label=f'{confline}% Confident')
        if severityline is not None:
            plt.hlines(severityline, 0, 100, colors='magenta', linestyles='-.', label=f'{severityline}$^\circ$ Severity')
        plt.ylabel('Severity (degrees)')
        sut_label = self.versions[sut] if sut is not None else 'any'
        plt.xlabel(f'Confidence (%)')
        # plt.tight_layout()
        plt.legend()
        clip_ending = '' if clip is None else f'_{clip}'
        for ending in ENDINGS:
            plt.savefig(f'{self.label}_conf_vs_value_hist_sut_{sut_label}{clip_ending}.{ending}')
        plt.show()


def print_and_write(string, file):
    file.write(f'{string}\n')
    print(string)

def main():
    # x = np.linspace(0, 1, 10000)
    # for i in [3, 4, 5]:
    #     plt.plot(x, direct_dixon(x, i), label=f'{i}')
    # plt.xlabel('Gap/Range')
    # plt.ylabel('Confidence in error')
    # plt.legend()
    # plt.show()
    # quit()
    print('Running')
    parser = argparse.ArgumentParser(description="Used to identify which scenarios pass and which fail")
    parser.add_argument('--dataset',
                        type=str,
                        choices=['External_Jutah', 'OpenPilot_2k19', 'OpenPilot_2016'],
                        required=True,
                        help="The dataset to use. Choose between 'External_Jutah', 'OpenPilot_2k19', or 'OpenPilot_2016'.")
    parser.add_argument('--dataset_directory',
                        type=str,
                        default="../1_Datasets/Data",
                        help="The location of the dataset")
    args = parser.parse_args()
    DATASET_DIRECTORY = f"{args.dataset_directory}/{args.dataset}"
    # Get all video files
    print(f'Searching in {DATASET_DIRECTORY}')
    video_file_paths = glob.glob(f"{DATASET_DIRECTORY}/1_ProcessedData/*.mp4")
    video_filenames = [os.path.basename(v)[:-4] for v in video_file_paths]
    print(f'Found: {len(video_filenames)}')
    video_filenames = sorted(video_filenames)
    all_data = []
    versions = []
    cur_biggest = 0
    cur_biggest_index = 0
    for video_filename in tqdm(video_filenames, desc="Processing Video", leave=False, position=0,
                               total=len(video_filenames)):
        dl = DataLoader(filename=video_filename, data_path=args.dataset_directory)
        dl.validate_h5_files()
        dl.load_data(terminal_print=True, refresh_cache=False)
        if len(versions) == 0:
            versions.extend(dl.versions)
        # Get the readings
        readings = dl.readings
        od = OutlierDetection(dl.readings, versions)
        maximal = od.find_maximal_system_failure(4, 20, 0.9)
        print(maximal)
        maximal_decon = []
        needed_frames = set()
        for duration in maximal:
            for (start_index, end_index) in maximal[duration]:
                maximal_decon.append((duration, start_index, end_index))
                for i in range(start_index-2, end_index+2):
                    needed_frames.add(i)
        needed_frames = list(needed_frames)
        outer_save_dir = f'/p/difftest/highsev_lowconf_examples/sut4/{args.dataset}/cache/{video_filename}/'
        dl.save_frames(needed_frames, outer_save_dir,
                       steering={frame: od.readings[:, frame] for frame in needed_frames},
                       labels=[' S1', ' S2', ' S3', ' S4', 'SUT'])
        for duration, start_index, end_index in maximal_decon:
            save_dir = f'/p/difftest/highsev_lowconf_examples/sut4/{args.dataset}/{duration}/{video_filename}/{start_index}/'
            dl.save_frames([i for i in range(start_index-2, end_index+2)], save_dir,
                           steering={frame: od.readings[:, frame] for frame in range(start_index-2, end_index+2)},
                           labels=[' S1', ' S2', ' S3', ' S4', 'SUT'])
            with open(f'{save_dir}/output.txt', 'w') as f:
                np.set_printoptions(precision=2, suppress=True)
                print_and_write(f'Duration {duration}', f)
                print_and_write('readings', f)
                print_and_write(od.readings[:, start_index - 2:end_index + 2], f)
                print_and_write('potential_outliers', f)
                print_and_write(od.potential_outliers[4, start_index - 2:end_index + 2], f)
                print_and_write('outlier_probs', f)
                print_and_write(100*od.outlier_probs[start_index - 2:end_index + 2], f)
                print_and_write('gap_arr', f)
                print_and_write(od.gap_arr[start_index - 2:end_index + 2], f)
                print_and_write('range_arr', f)
                print_and_write(od.range_arr[start_index - 2:end_index + 2], f)
                np.set_printoptions(precision=3, suppress=True)
                print_and_write('outlier_q', f)
                print_and_write(od.outlier_q[start_index - 2:end_index + 2], f)
                impact = max(od.gap_arr[start_index:end_index]*od.outlier_probs[start_index:end_index])
                if impact > cur_biggest:
                    cur_biggest = impact
                    cur_biggest_index = (video_filename, start_index)
        print('Biggest impact', cur_biggest_index, cur_biggest)
        # remove NaNs
        # readings = readings[:, ~np.isnan(readings).any(axis=0)]
        all_data.append(readings)
    all_data = np.concatenate(all_data, axis=1)
    od = OutlierDetection(all_data, versions)
    # maximal = od.find_maximal_system_failure(4, 20, 0.9)
    # print(maximal)
    # od.single_frame_binned_value_hist(4)
    # od.single_frame_failures_hist(4)

    # SEVERITIES = [10, 20, 45, 90, 180, 270]
    # CONFIDENCES = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999]
    # maximal_length_df = pd.DataFrame(columns=CONFIDENCES)
    # num_maximal_df = pd.DataFrame(columns=CONFIDENCES)
    # with tqdm(total=len(SEVERITIES)*len(CONFIDENCES)) as pbar:
    #     for severity in SEVERITIES:
    #         max_length_arr = []
    #         num_maximal_arr = []
    #         for failure_prob_thresh in CONFIDENCES:
    #             output = od.find_maximal_system_failure(od.last_version, severity, failure_prob_thresh)
    #             maximal_length = max(output.keys())
    #             num_maximal = len(output[maximal_length]) if maximal_length > 0 else 0
    #             max_length_arr.append(maximal_length)
    #             num_maximal_arr.append(num_maximal)
    #             pbar.update(1)
    #         maximal_length_df.loc[severity] = max_length_arr
    #         num_maximal_df.loc[severity] = num_maximal_arr
    # print('maximal length')
    # print(maximal_length_df.to_latex())
    # print('num maximal')
    # print(num_maximal_df.to_latex())

    # od.single_frame_failures(4, 0.99, 90)
    # od.single_frame_failures_hist(3)
    # od.single_frame_failures_hist(None)
    # od.single_frame_failures_hist(4, 90)
    # od.single_frame_failures_hist(None, 90)
    # od.system_failure(3, 5, 2, 90, 0.5, threads=32)
    # od.system_failure(3, 5, 5, 90, 0.5, threads=32)
    # od.system_failure(3, 10, 5, 90, 0.5, threads=32)
    # for i in range(6):
    #     od.system_failure(4, 10, 5+i, 90, 0.5, threads=32)
    # od.system_failure(3, 20, 15, 90, 0.5, threads=32)


if __name__ == '__main__':
    main()
