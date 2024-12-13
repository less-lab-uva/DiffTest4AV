import os
import cv2
import math
import h5py
import glob
import shutil
import pickle
from typing import List, Dict

from tqdm import tqdm

from visualization_functions import show_steering
from constants import OPENPILOT_CONTROL_RATE

import numpy as np

COLORS = [
    [255, 127, 14],
    [31, 119, 180],
    [44, 160, 44],
    [255, 127, 255],
    [0, 0, 255]
]

class DataLoader:
    def __init__(self, filename: str, data_path: List[str] = "../1_Datasets/Data", force_reload: bool = False, bypass_checks=False):
        self.filename = filename
        self.data_path = data_path

        # Define the cache directory
        self.cache_dir = "./cache"

        self.frame_cache = {}

        # Check if force_reload is True and delete cache directory if it exists
        if force_reload and os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            print(f"Cache directory {self.cache_dir} has been deleted due to force reload.")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Init variables

        # This holds variables about the filenames
        self.dataset = ""
        self.video_filepath = ""
        
        # Holds variables about the steering data
        self.h5_filepaths = []

        # Holds variables about the frames in the video
        self.total_video_frames = -1
        self.total_h5_readings = -1

        # Holds the actual steering data
        self.readings = None
        
        # Determine the filepath
        self._determine_video_filename_path()
        if not bypass_checks:
            if self.video_filepath == "":
                print(f"WARNING: {filename} could not be located in {data_path}")
                return
        
            # Get the dataset the versions filepath and load how long they are
            self._determine_dataset()
            self._determine_available_versions()
            self._determine_h5_filename_path()
            self.load_all_h5_lengths()

            # Load the video frames
            self.total_video_frames = DataLoader.get_video_length(self.video_filepath)

    def _determine_video_filename_path(self):
        # Get the filepath
        available_filepaths = glob.glob(f"{self.data_path}/*/1_ProcessedData/*.mp4")
        avilable_filenames = [os.path.basename(filepath)[:-4] for filepath in available_filepaths]
        for current_index, current_filename in enumerate(avilable_filenames):
            if current_filename == self.filename:
                self.video_filepath = available_filepaths[current_index]
                break
        
    def _determine_dataset(self):
        # Update the dataset
        if self.video_filepath != "":
            self.dataset = self.video_filepath.split("/")[3]

    def _determine_available_versions(self):
        # Get the versions with an h5 file
        version_path = f"{self.data_path}/{self.dataset}/2_SteeringData"
        available_versions_paths = glob.glob(f"{version_path}/*/{self.filename}.h5")
        available_versions = [version_path.split("/")[-2] for version_path in available_versions_paths]
        self.versions = sorted(available_versions)

    def _determine_h5_filename_path(self):
        # Create the filepaths
        if len(self.versions) == 0:
            return
        for version in self.versions:
            self.h5_filepaths.append(f"{self.data_path}/{self.dataset}/2_SteeringData/{version}/{self.filename}.h5")

    @staticmethod
    def get_video_length(video_filepath):
        # Load the video and determine how many frames there are
        cap = cv2.VideoCapture(video_filepath)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return -1
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count > 100, "Cant get the video frame count"
        cap.release()
        return frame_count

    @staticmethod
    def get_h5_length(h5_filepath):
        key_length = -1
        try:
            f = h5py.File(h5_filepath, "r")
            keys = f.keys()
            key_length = len(keys)
            f.close()
        except Exception as e:
            pass

        return key_length
    
    def load_all_h5_lengths(self):
        h5_readings = []
        for h5_file in self.h5_filepaths:
            h5_filelength = DataLoader.get_h5_length(h5_file)
            h5_readings.append(h5_filelength)
        self.total_h5_readings = np.min(h5_readings)
        return h5_readings

    def validate_h5_files(self):
        h5_readings = self.load_all_h5_lengths()
        for i ,h5_filelength in enumerate(h5_readings):
            if math.floor(h5_filelength/OPENPILOT_CONTROL_RATE) != self.total_video_frames:
                print(f"WARNING: h5 file length not expected length {math.floor(h5_filelength/OPENPILOT_CONTROL_RATE)}/{self.total_video_frames} - {self.filename} - {self.versions[i]}")
                return False
        return True

    @staticmethod
    def read_index_from_h5(f, index):
        try:
            data = f.get(index)
            steer          = float(data[0][0])
            frame_number   = int(data[0][1])
            message        = str(data[0][2])
        except TypeError as e:
            steer          = np.nan
            frame_number   = np.nan
            message        = None
        
        return steer, frame_number, message

    def load_data(self, terminal_print=True, refresh_cache=False):
        # Check if cached data exists
        cache_filepath = os.path.join(self.cache_dir, f"{self.filename}.pkl")
        if os.path.exists(cache_filepath) and not refresh_cache:
            if terminal_print:
                print("\nLoading data from cache...")
            with open(cache_filepath, 'rb') as f:
                self.readings = pickle.load(f)
            return

        if len(self.h5_filepaths) == 0:
            print("No filepaths found.")
        # HERE
        self.readings = np.full((len(self.h5_filepaths), self.total_video_frames), np.nan)

        for h5_file_index, h5_file in tqdm(enumerate(self.h5_filepaths), desc="Processing version", leave=False, total=len(self.h5_filepaths), position=1):
            # Read the file
            h5_f = h5py.File(h5_file, 'r')

            current_frame = 0
            current_frame_readings = []

            # For each key in the common set of keys
            for key_index in tqdm(range(self.total_h5_readings), desc="Loading data from file", leave=False, total=self.total_h5_readings, position=2):

                # Generate a key
                index = "{0:09d}_data".format(key_index)

                # Get the steering angles and frame numbers
                data = DataLoader.read_index_from_h5(h5_f, index)
                steering_angle = data[0]
                frame_number = data[1]
                message = data[2]

                if frame_number == current_frame:
                    current_frame_readings.append(steering_angle)
                else:
                    frame_reading = np.median(current_frame_readings)
                    self.readings[h5_file_index][current_frame] = frame_reading
                    current_frame = frame_number
                    current_frame_readings = [steering_angle]


            # Close the h5 file
            h5_f.close()

        # Save loaded data to cache
        with open(cache_filepath, 'wb') as f:
            pickle.dump(self.readings, f)

    def save_frames(self, frames, save_directory=".", steering: Dict[int, List[float]] = None, labels=None):
        if len(frames) == 0:
            return
        print(f'Saving to {os.path.abspath(save_directory)}')
        os.makedirs(save_directory, exist_ok=True)
        # sort the list of frames
        frames = sorted(frames)
        still_need = []
        for frame in frames:
            if frame in self.frame_cache:
                save_path = f'{save_directory}/{self.filename}_frame{frame}.png'
                # shutil.copy(self.frame_cache[frame], save_path)
                try:
                    os.symlink(self.frame_cache[frame], save_path)
                except FileExistsError:
                    pass
            else:
                still_need.append(frame)
        if len(still_need) == 0:
            return
        else:
            frames = still_need
        # Load the video and determine how many frames there are
        cap = cv2.VideoCapture(self.video_filepath)
        if not cap.isOpened():
            print("Error: Could not open video.")
            for current_frame_id in frames:
                frame = np.zeros((1208, 1928, 3), dtype=np.uint8)
                save_path = f'{save_directory}/{self.filename}_frame{current_frame_id}.png'
                if steering is not None:
                    frame = show_steering(frame, steering[current_frame_id], COLORS, labels=labels, CLIPPING_DEGREE=90,
                                          frame_id=current_frame_id)
                cv2.imwrite(save_path, frame)
                self.frame_cache[current_frame_id] = save_path
        # Save the frames
        max_index = min(np.max(frames), self.total_video_frames)
        for current_frame_id in tqdm(range(max_index+1), desc="Parsing Video", total=max_index, leave=False, position=1):

            # Read the next frame from the video
            ret, frame = cap.read()
            # If the frame was not retrieved, we've reached the end of the video
            if not ret:
                print("Error: Not all images were saved")
                break

            if current_frame_id in frames:
                save_path = f'{save_directory}/{self.filename}_frame{current_frame_id}.png'
                if steering is not None:
                    frame = show_steering(frame, steering[current_frame_id], COLORS, labels=labels, CLIPPING_DEGREE=90, frame_id=current_frame_id)
                cv2.imwrite(save_path, frame)
                self.frame_cache[current_frame_id] = save_path

        return