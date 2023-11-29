import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import soundfile as sf

from pydub import AudioSegment

from concurrent.futures import ThreadPoolExecutor, as_completed
import tools

import warnings

warnings.filterwarnings('ignore')

# change style
plt.style.use('ggplot')
import librosa
import librosa.display

from tqdm import tqdm
from datetime import datetime

tqdm.pandas()
import os
from glob import glob
import random
from typing import List, Tuple, Dict
from time import sleep

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import librosa

from extract_audio_features import extract_features, N_FFT

from joblib import Parallel, delayed

from params import SOUNDS_DATASET_PATH, SAMPLE_RATE, CLASS_COLORS, AIRS_PATH, LOW_PITCHED_DRUMS, HIGH_PITCHED_DRUMS, \
    CLASS_COLORS
from tools import *

# %%

from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift, PolarityInversion,
    Gain, ApplyImpulseResponse, Shift, LowPassFilter, HighPassFilter,
    BandPassFilter, BandStopFilter
)

# AUGMENTATION PARAMETERS

APPLY_NOISE = True
P_NOISE = 5 / 10
MIN_NOISE_AMPLITUDE, MAX_NOISE_AMPLITUDE = 0.001, 0.008

APPLY_TIME_STRETCH = True
P_TIME_STRETCH = 8 / 10
MIN_STRETCH_RATE, MAX_STRETCH_RATE = 0.5, 2.0

APPLY_PITCH_SCALING = True
P_PITCH_SCALING = 9 / 10
MIN_SEMITONES, MAX_SEMITONES = -4, 4

APPLY_POLARITY_INVERSION = True
P_POLARITY_INVERSION = 8 / 10

APPLY_RANDOM_GAIN = True
P_RANDOM_GAIN = 7 / 10
MIN_GAIN_DB, MAX_GAIN_DB = -12, 12

APPLY_TIME_SHIFT = True
P_TIME_SHIFT = 5 / 10
MIN_SHIFT_FRACTION, MAX_SHIFT_FRACTION = 0.05, 0.3

APPLY_IMPULSE_RESPONSE = True
P_IMPULSE_RESPONSE = 1 / 10

APPLY_FILTERING = False  # todo
P_FILTERING = 1 / 10
FILTER_TYPE = random.choice(["low_pass", "high_pass", "band_pass", "band_reject"])
MIN_FILTER_FREQ, MAX_FILTER_FREQ = 40, 1000

# DATASET AUGMENTATIONS PARAMETERS

TARGET_NUMBER_PER_CLASS = 2000


def augment_audio(y, sample_rate,
                  apply_noise=APPLY_NOISE, min_noise_amplitude=MIN_NOISE_AMPLITUDE,
                  max_noise_amplitude=MAX_NOISE_AMPLITUDE, p_noise=P_NOISE,
                  apply_time_stretch=APPLY_TIME_STRETCH, min_stretch_rate=MIN_STRETCH_RATE,
                  max_stretch_rate=MAX_STRETCH_RATE, p_time_stretch=P_TIME_STRETCH,
                  apply_pitch_scaling=APPLY_PITCH_SCALING, min_semitones=MIN_SEMITONES, max_semitones=MAX_SEMITONES,
                  p_pitch_scaling=P_PITCH_SCALING,
                  apply_polarity_inversion=APPLY_POLARITY_INVERSION, p_polarity_inversion=P_POLARITY_INVERSION,
                  apply_random_gain=APPLY_RANDOM_GAIN, min_gain_db=MIN_GAIN_DB, max_gain_db=MAX_GAIN_DB,
                  p_random_gain=P_RANDOM_GAIN,
                  apply_impulse_response=APPLY_IMPULSE_RESPONSE, ir_path=AIRS_PATH,
                  p_impulse_response=P_IMPULSE_RESPONSE,
                  apply_time_shift=APPLY_TIME_SHIFT, min_shift_fraction=MIN_SHIFT_FRACTION,
                  max_shift_fraction=MAX_SHIFT_FRACTION, fade_shift=True,
                  fade_shift_duration=0.01, p_time_shift=P_TIME_SHIFT,
                  apply_filtering=APPLY_FILTERING, filter_type=FILTER_TYPE, min_filter_freq=MIN_FILTER_FREQ,
                  max_filter_freq=MAX_FILTER_FREQ,
                  p_filtering=P_FILTERING):
    """
    Augment an audio signal with the given parameters.

    Args:
        y (np.ndarray): Audio signal
        sample_rate (int): Sample rate of the audio signal
        apply_noise (bool): Whether to apply noise augmentation
        max_noise_amplitude (float): Maximum amplitude of the noise to add
        apply_time_stretch (bool): Whether to apply time stretch augmentation
        min_stretch_rate (float): Minimum stretch rate
        max_stretch_rate (float): Maximum stretch rate
        apply_pitch_scaling (bool): Whether to apply pitch scaling augmentation
        min_semitones (int): Minimum pitch scaling in semitones
        max_semitones (int): Maximum pitch scaling in semitones
        apply_polarity_inversion (bool): Whether to apply polarity inversion augmentation
        apply_random_gain (bool): Whether to apply random gain augmentation
        min_gain_db (float): Minimum gain in dB
        max_gain_db (float): Maximum gain in dB
        apply_impulse_response (bool): Whether to apply impulse response augmentation
        ir_path (str): Path to the impulse response file
        max_ir_gain_db (float): Maximum impulse response gain in dB
        apply_time_shift (bool): Whether to apply time shift augmentation
        max_shift_fraction (int): Maximum number of samples to shift
        apply_filtering (bool): Whether to apply filtering augmentation
        filter_type (str): Type of filter to apply.
        min_filter_freq (int): Minimum filter frequency
        max_filter_freq (int): Maximum filter frequency

    """
    augmentations = []
    # parameters strings with abbreviated names and values (with sub parameters)
    params_str_list = []
    if apply_noise and random.random() < p_noise:
        params_str_list.append(f"noise_{int(apply_noise)}")
        augmentations.append(
            AddGaussianNoise(max_amplitude=max_noise_amplitude, min_amplitude=min_noise_amplitude, p=1.0))
    if apply_time_stretch and random.random() < p_time_stretch:
        params_str_list.append(f"ts_{int(apply_time_stretch)}")
        augmentations.append(TimeStretch(min_rate=min_stretch_rate, max_rate=max_stretch_rate, p=1.0))
    if apply_pitch_scaling and random.random() < p_pitch_scaling:
        params_str_list.append(f"ps_{int(apply_pitch_scaling)}")
        augmentations.append(PitchShift(min_semitones=min_semitones, max_semitones=max_semitones, p=1.0))
    if apply_polarity_inversion and random.random() < p_polarity_inversion:
        params_str_list.append(f"pi_{int(apply_polarity_inversion)}")
        augmentations.append(PolarityInversion(p=1.0))
    if apply_random_gain and random.random() < p_random_gain:
        params_str_list.append(f"rg_{int(apply_random_gain)}")
        augmentations.append(Gain(min_gain_in_db=min_gain_db, max_gain_in_db=max_gain_db, p=1.0))
    if apply_impulse_response and ir_path is not None and random.random() < p_impulse_response:
        params_str_list.append(f"ir_{int(apply_impulse_response)}")
        augmentations.append(ApplyImpulseResponse(ir_path=ir_path, p=1.0))
    if apply_time_shift and random.random() < p_time_shift:
        params_str_list.append(f"ts_{int(apply_time_shift)}")
        augmentations.append(Shift(min_fraction=min_shift_fraction, max_fraction=max_shift_fraction, fade=fade_shift,
                                   fade_duration=fade_shift_duration, p=1.0))

    if apply_filtering and random.random() < p_filtering:
        params_str_list.append(
            f"fl_{int(apply_filtering)}_ft_{filter_type}_min_f_{min_filter_freq}_max_f_{max_filter_freq}")

        if filter_type == 'low_pass':

            augmentations.append(
                LowPassFilter(min_cutoff_freq=min_filter_freq, max_cutoff_freq=max_filter_freq, p=1.0))
        elif filter_type == 'high_pass':
            augmentations.append(
                HighPassFilter(min_cutoff_freq=min_filter_freq, max_cutoff_freq=max_filter_freq, p=1.0))
        elif filter_type == 'band_pass':
            augmentations.append(
                BandPassFilter(min_center_freq=min_filter_freq, max_center_freq=max_filter_freq, p=1.0))
        elif filter_type == 'band_reject':
            augmentations.append(
                BandStopFilter(min_center_freq=min_filter_freq, max_center_freq=max_filter_freq, p=1.0))

    if not augmentations:
        return augment_audio(y, sample_rate)

    augmenter = Compose(augmentations)
    return augmenter(y, sample_rate), "__".join(params_str_list)


# %%


def save_audio_file(file_path: str, y: np.ndarray, sr: int) -> str:
    """
    Save audio file
    """
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()[1:]  # convert to lowercase and remove leading "."

    # Mapping from file extensions to formats recognized by soundfile library
    format_map = {
        "wav": "WAV",
        "flac": "FLAC",
        "aif": "AIFF",
        "aiff": "AIFF",
        "mp3": "MP3",
        "ogg": "OGG",
    }

    if file_extension in format_map:
        if file_extension in ["mp3", "ogg", "aiff", "aif"]:
            # Convert numpy array to pydub.AudioSegment
            """
            audio_segment = AudioSegment(
                y.tobytes(),
                frame_rate=sr,
                sample_width=y.dtype.itemsize,
                channels=1
            )
            # Save as MP3
            audio_segment.export(file_path, format="mp3")
            """
            # rename to flac
            file_path = file_path.replace(f".{file_extension}", '.flac')
            # save as flac
            sf.write(file_path, y, sr, format="FLAC")

        else:
            sf.write(file_path, y, sr, format=format_map[file_extension])
    else:
        raise ValueError(f"Unsupported audio file extension: {file_extension}")

    return file_path


def generate_augmented_audio(orig_file_path: str, n_augmentations: int = 1, **kwargs) -> Dict[str, np.ndarray]:
    """
    Generate n augmented audio files from the original file
    """
    # get file name without extension, extension
    _, file_extension__ = os.path.splitext(orig_file_path)
    augmented_files = {}  # path: signal

    y, sr = load_audio_file(orig_file_path)

    for _ in range(n_augmentations):
        # get augmented audio
        augmented_y, params_str = augment_audio(y=y, sample_rate=sr, **kwargs)
        # save augmented file
        now_str_time = datetime.now().strftime("%Y%m%d%H%M%S")
        augmented_file_path = orig_file_path.replace(file_extension__,
                                                     f"@augmented__{params_str}__{now_str_time}{file_extension__}")

        augmented_file_path = save_audio_file(augmented_file_path, augmented_y, sr)

        # append to list of augmented files
        augmented_files[augmented_file_path] = augmented_y
    return augmented_files


# augmented_files = generate_augmented_audio(orig_file_path = "Classic Clean (Snare).wav", n_augmentations = 20)

# %%

# delete all augmented files (file name contains "@augmented__") in a folder and subfolders (recursive)
def delete_augmented_files(folder_path: str):
    count = 0
    # get files containes "@augmented__" in folder (use glob)
    files_augmented = glob(os.path.join(folder_path, "**", "*@augmented__*"), recursive=True)
    if not files_augmented:
        print(f"> No augmented files found in {folder_path}")
        return count

    print(f"> Found {len(files_augmented)} augmented files, deleting...",
          files_augmented[:3] + ["..."] + files_augmented[-3:])
    # delete files
    for file_path in tqdm(files_augmented, total=len(files_augmented), desc="Deleting all augmented files"):
        os.remove(file_path)
        count += 1

    return count


def ask_confirmation():
    """
    Ask confirmation to continue
    """
    while True:
        answer = input("... Do you want to continue? (y/n): ")

        if answer.lower() == 'y':
            return True
        elif answer.lower() == 'n':
            return False
        else:
            print("Invalid answer. Try again.")
            continue


# task : get augmented files and add them to the dataset
def task_augment_file_row(orig_file_path):
    if not os.path.exists(orig_file_path):
        print(f"! Orig File {orig_file_path} does not exist")
        return None
    class_name = os.path.basename(os.path.dirname(orig_file_path))

    try:
        # get augmented files
        augmented_files_dict = generate_augmented_audio(orig_file_path, n_augmentations=1)

        # extract features from augmented files
        augmented_file_path, augmented_y = list(augmented_files_dict.items())[0]
        augmented_y = pad_signal(augmented_y, target_length=N_FFT)
        features = extract_features(augmented_y, sr=SAMPLE_RATE)  # todo sr associated to file augmentation

        return {
            "orig_file_path": orig_file_path,
            "file_path": augmented_file_path,
            "split": "train",
            "class": class_name,
            "is_augmented": 1,
            **features,
        }
    except Exception as e:
        print(f"! Error augmenting file {orig_file_path}: {e}")
        return None


# %%
def main():
    # LOAD DATASET (with selected features) -------------------------
    now_day_str = "20230511"
    dataset_csv_path = os.path.join(SOUNDS_DATASET_PATH, f'dataset_features_cleaned_{now_day_str}.csv')
    if not os.path.exists(dataset_csv_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_csv_path}")
    dataset = pd.read_csv(dataset_csv_path)
    dataset.set_index('file_path', inplace=True)
    print(f"> Dataset shape: {dataset.shape}")

    # SELECT TRAINING DATASET ---------------------------------------
    train_dataset = dataset.query("split == 'train'")

    # SELECT N EXAMPLES TO AUGMENT ----------------------------------
    table_train_class_counts = train_dataset["class"].value_counts()

    table_n_examples_to_add = TARGET_NUMBER_PER_CLASS - table_train_class_counts
    table_n_examples_to_add = table_n_examples_to_add[table_n_examples_to_add > 0]

    # GET FILE PATHS TO AUGMENT -------------------------------------
    original_file_paths_to_augment = []
    for class_name, n_examples_to_add in tqdm(table_n_examples_to_add.items(), total=len(table_n_examples_to_add),
                                              desc="Get file paths to augment"):
        print(f"> Class {class_name} : {n_examples_to_add} examples to add")

        # get n_examples_to_add random examples from the class
        class_samples_df = train_dataset[train_dataset["class"] == class_name]
        class_examples_file_paths = class_samples_df.sample(min(n_examples_to_add, len(class_samples_df))).index.values
        print(f"  - {len(class_examples_file_paths)} different examples")

        # duplicate the file paths to reach the target number of examples
        n_duplicates = int(np.ceil(n_examples_to_add / len(class_examples_file_paths)))
        class_examples_file_paths = np.tile(class_examples_file_paths, n_duplicates)[:n_examples_to_add]
        print(f"  - {len(class_examples_file_paths)} examples final (after duplication)")

        # add them to the list of files to augment
        original_file_paths_to_augment.extend(class_examples_file_paths)

    print(f"\n>>> {len(original_file_paths_to_augment)} new (augmented) files will be created")

    # ---------------------------------------------

    if original_file_paths_to_augment:
        print()

        if ask_confirmation():
            # DELETE ALL AUGMENTED FILES
            if delete_augmented_files(SOUNDS_DATASET_PATH) > 0:
                sleep(45)
                delete_augmented_files(SOUNDS_DATASET_PATH)
                sleep(15)
                delete_augmented_files(SOUNDS_DATASET_PATH)

            if not ask_confirmation():
                print("Aborting...")
                return

            # AUGMENT FILES
            augmented_row_dicts = []

            # get augmented files from original_file_paths_to_augment
            # sequential version
            # for orig_file_path in tqdm(original_file_paths_to_augment, total=len(original_file_paths_to_augment),
            #                           desc="Augmenting files"):
            #    augmented_row_dicts.append(task_augment_file_row(orig_file_path))

            # parallel version (concurrent.futures)
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(task_augment_file_row, orig_file_path)
                           for orig_file_path in original_file_paths_to_augment]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting files"):
                    row_dict = future.result()
                    if row_dict is not None:
                        augmented_row_dicts.append(row_dict)

            # Convert the list of feature dictionaries into a DataFrame
            augmented_features_df = pd.DataFrame(augmented_row_dicts)
            # set file_path as index
            augmented_features_df.set_index('file_path', inplace=True)
            # Combine the original dataset and the augmented DataFrame
            df_augmented_final = pd.concat([dataset, augmented_features_df])
            # fill NaN values in "augmented" column with 0
            df_augmented_final["is_augmented"].fillna(0, inplace=True)
            # convert "augmented" column to int
            df_augmented_final["is_augmented"] = df_augmented_final["is_augmented"].astype(int)
            print(f"> Augmented final dataset shape: {df_augmented_final.shape}")

            # save augmented dataset
            output_augmented_dataset_csv_path = os.path.join(SOUNDS_DATASET_PATH,
                                                             f'dataset_features_cleaned_augmented_{TARGET_NUMBER_PER_CLASS}_{now_day_str}.csv')
            df_augmented_final.to_csv(output_augmented_dataset_csv_path, index=True)
    return 1


if __name__ == "__main__":
    main()
