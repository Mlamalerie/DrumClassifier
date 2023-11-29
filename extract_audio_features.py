
from scipy.signal import hilbert
import numpy as np
import streamlit as st
import scipy
from scipy.stats import skew
import pandas as pd

import librosa
import librosa.display
from tqdm import tqdm
from datetime import datetime

tqdm.pandas()
import os
from typing import List, Tuple, Dict

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from params import SOUNDS_DATASET_PATH, SAMPLE_RATE
from tools import load_audio_file, pad_signal

## Set Parameters

N_MFCC = 12  # 10-20
N_FFT = 2048
RMS_FRAME_LENGTH = ZCR_FRAME_LENGTH = 1024
HOP_LENGTH = 512


## Time-domain Features


### Root Mean Square (RMS)

def get_rms_log_stats(y, frame_length=RMS_FRAME_LENGTH, hop_length=HOP_LENGTH) -> Dict[str, float]:
    """
    Calcule les statistiques du RMS d'un signal audio.

    Args:
        y (np.ndarray): Le signal audio.
        frame_length (int, optional): La taille de la fenêtre. Defaults to 1024. Dans cette fenêtre, on calcule le RMS, c'est à dire la racine carrée de la moyenne des carrés des échantillons.
        hop_length (int, optional): Le pas de la fenêtre (la distance entre deux fenêtres consécutives). Defaults to 512.
    """
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    rms_log = np.log10(rms) if np.min(rms) > 0 else np.log10(rms + 1e-6)

    rms_log_sum = np.sum(rms_log)  # todo rename to rms_log_sum
    rms_log_mean = np.mean(rms_log)
    rms_log_max = np.max(rms_log)
    rms_log_min = np.min(rms_log)
    rms_log_std = np.std(rms_log)

    rms_log_diff = np.diff(rms_log)  # difference between consecutive RMS values

    rms_log_crest_factor = rms_log_max - rms_log_mean  # crest factor = peak RMS / average RMS ;

    return {
        'rms_log_sum': rms_log_sum,
        'rms_log_mean': rms_log_mean,
        'rms_log_max': rms_log_max,
        # 'rms_log_min': rms_log_min, pas pertinent car toujours faible pour les sons
        'rms_log_std': rms_log_std,
        'rms_log_diff_abs_mean': np.mean(np.abs(rms_log_diff)),
        'rms_log_crest_factor': rms_log_crest_factor
    }, rms_log


### Zero crossing rate


def get_zero_crossing_rate_stats(y, frame_length=ZCR_FRAME_LENGTH, hop_length=HOP_LENGTH) -> Dict[str, float]:
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)
    loudest_frame = np.argmax(zcr)
    return {
        "zcr_sum": np.sum(zcr),
        'zcr_mean': np.mean(zcr),
        'zcr_std': np.std(zcr),
        'zcr_loudest_frame': zcr[0, loudest_frame]
    }, zcr


### Temporal centroid

def get_temporal_centroid(y, sr) -> float:
    energy = np.sum(y ** 2)
    energy_cumulative = np.cumsum(y ** 2)
    half_energy_time = np.argmax(energy_cumulative >= 0.5 * energy)
    return half_energy_time / sr


### Sound Envelope
#### Attack time


def get_attack_time(y, sr):
    """
    Calcule le temps d'attaque logarithmique d'un signal audio.

    Args:
        y (np.ndarray): Signal audio.
        sr (int): Fréquence d'échantillonnage du signal audio.

    Returns:
        float: Temps d'attaque logarithmique en secondes.
    """
    # Calcul de l'enveloppe du signal
    envelope = np.abs(hilbert(y))
    # Recherche de l'indice du pic d'enveloppe
    peak_idx = np.argmax(envelope)
    return peak_idx / float(sr)


def get_log_attack_time(y, sr):
    """
    Calcule le temps d'attaque logarithmique d'un signal audio.

    Args:
        y (np.ndarray): Signal audio.
        sr (int): Fréquence d'échantillonnage du signal audio.

    Returns:
        float: Temps d'attaque logarithmique en secondes.
    """
    attack_time = get_attack_time(y, sr)
    # Conversion du temps d'attaque en échelle logarithmique
    log_attack_time = np.log(attack_time) if attack_time > 0 else np.log(1e-5)
    """
      Retourner 0 :

      Avantage : Cette valeur est simple et facile à interpréter.
      Inconvénient : Si le temps d'attaque réel est très proche de 0, la différence entre les valeurs réelles et 0 peut être importante en termes de log. Cela pourrait affecter l'analyse ou l'apprentissage automatique ultérieur.

      Retourner np.log10(1e-5) :

      Avantage : Cette valeur préserve l'échelle logarithmique et évite de créer une différence artificielle importante entre les valeurs réelles et la valeur par défaut.
      Inconvénient : La valeur np.log10(1e-5) est moins intuitive et peut nécessiter une explication supplémentaire pour être interprétée correctement.
      """
    return log_attack_time


#### Release time

def get_release_time(y, sr, threshold=0.9):
    """
    Calcule le log du temps de relâchement d'un signal audio.

    Args:
        y (np.ndarray): Le signal audio.
        sr (int): Le taux d'échantillonnage du signal audio.
        threshold (float): Le seuil pour détecter le relâchement du signal audio.

    Returns:
        float: Le log du temps de relâchement.
    """
    # Calcul de l'enveloppe analytique
    y_analytic = hilbert(y)
    y_env = np.abs(y_analytic)

    if len(y_env) == 0:
        return 0, 0

    # Recherche des indices où l'enveloppe tombe en dessous du seuil
    below_threshold_indices = np.where(y_env < threshold * np.max(y_env))[0]

    return below_threshold_indices[0] / float(sr)


def get_log_release_time(y, sr, threshold=0.9):
    """
    Calcule le log du temps de relâchement d'un signal audio.

    Args:
        y (np.ndarray): Le signal audio.
        sr (int): Le taux d'échantillonnage du signal audio.
        threshold (float): Le seuil pour détecter le relâchement du signal audio.

    Returns:
        float: Le log du temps de relâchement.
    """
    release_time = get_release_time(y, sr, threshold)
    return np.log(release_time) if release_time > 0 else np.log(1e-5)


## Frequency-domain Features

### Pitch

def get_pitch(y, sr, fmin=20.0, fmax=2000):
    """
    Estime le pitch d'un signal audio en utilisant la méthode de l'autocorrélation.

    Args:
        y (np.ndarray): Le signal audio.
        sr (int): La fréquence d'échantillonnage.
        fmin (int, optional): La fréquence minimale pour estimer le pitch. Defaults to 50.
        fmax (int, optional): La fréquence maximale pour estimer le pitch. Defaults to 2000.

    Returns:
        float: Le pitch estimé en Hz.
    """
    # Calculer la fonction d'autocorrélation à court terme
    # l'auto-corrélation est calculée sur une fenêtre de 3 secondes
    autocorr = librosa.autocorrelate(y, max_size=3 * sr)

    # Trouver les limites de la région d'intérêt dans l'autocorrélation
    i_min = int(sr // fmax)
    i_max = int(sr // fmin)

    # Trouver l'index du maximum local dans la région d'intérêt
    i_peak = np.argmax(autocorr[i_min:i_max]) + i_min

    return float(sr) / i_peak


def pitch_to_midi(pitch):
    """
    Convertit le pitch en Hz en midi.

    Args:
        pitch (float): Le pitch en Hz.

    Returns:
        float: Le pitch en midi.
    """
    return 69 + 12 * np.log2(pitch / 440.0)


def midi_to_note_name(midi: float):
    """
    Convertit le pitch en midi en nom de note.

    Args:
        midi (float): Le pitch en midi.

    Returns:
        str: Le nom de la note.
    """
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return notes[int(midi) % 12] + str(int(midi) // 12 - 1)


# get pitch each segment of a signal
def get_pitch_segments(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, fmin=25.0, fmax=4000.0):
    """
    Estime le pitch de chaque segment d'un signal audio.

    Args:
        y (np.ndarray): Le signal audio.
        sr (int): La fréquence d'échantillonnage.
        n_fft (int): La taille de la fenêtre de Fourier.
        hop_length (int): Le nombre d'échantillons entre les fenêtres consécutives.
        fmin (int, optional): La fréquence minimale pour estimer le pitch. Defaults to 50.
        fmax (int, optional): La fréquence maximale pour estimer le pitch. Defaults to 2000.

    Returns:
        np.ndarray: Le pitch estimé en Hz pour chaque segment.
    """
    pitches = []
    for y_segment in librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length):
        pitch = get_pitch(y=y_segment, sr=sr, fmin=fmin, fmax=fmax)
        pitches.append(pitch)

    return np.array(pitches)


### MFCC

# transform a Dict[str, np.ndarray] into a Dict[str, float]
def flatten_dict(d: Dict[str, np.ndarray], start: int = 0) -> Dict[str, float]:
    # exemple {"mfcc_mean" : np.ndarray } -> [mfcc_mean_0, mfcc_mean_1, ...]
    result = {}
    for k, tab in d.items():
        for i, v in enumerate(tab, start=start):
            result[f'{k}_{i}'] = v
    return result


def get_mfcc_stats(y, sr, n_mfcc: int = N_MFCC, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> Dict[
    str, np.ndarray]:
    """
    Calcule les statistiques du MFCC d'un signal audio.

    Args:
        y (np.ndarray): Le signal audio.
        sr (int): La fréquence d'échantillonnage.
        n_mfcc (int, optional): Le nombre de MFCC. Defaults to 20. Un nombre plus élevé de MFCC peut fournir une représentation plus détaillée du signal audio, mais peut également augmenter le temps de calcul et la complexité du modèle de classification.
        n_fft (int, optional): La taille de la fenêtre pour la STFT. Defaults to 2048.
        hop_length (int, optional): Le pas de la fenêtre. Defaults to 512.

    Returns:
        dict: Les statistiques du MFCC.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    return {
        'mfcc_mean': mfcc.mean(axis=1),
        'mfcc_std': mfcc.std(axis=1),
        'mfcc_max': mfcc.max(axis=1),
        'mfcc_min': mfcc.min(axis=1),
        'mfcc_skew': scipy.stats.skew(mfcc, axis=1),
        'mfcc_kurtosis': scipy.stats.kurtosis(mfcc, axis=1),
    }


### Chroma Features

### Spectral Features

#### Spectral centroid

def get_spectral_centroid(y, sr, n_fft=N_FFT) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute the spectral centroid of the signal y

    Args:
        y (np.ndarray): audio signal
        sr (int): sampling rate
        n_fft (int, optional): FFT window size. Defaults to N_FFT.

    Returns:
        Tuple[Dict[str, float], np.ndarray]: a tuple containing a dictionary with the mean and std of the spectral centroid and the spectral centroid
    """
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)
    return {"spec_cent_mean": np.mean(spec_cent), "spec_cent_std": np.std(spec_cent)}, spec_cent


"""#### Spectral bandwidth

Le "spectral bandwidth" (ou largeur de bande spectrale) est une mesure qui décrit la répartition des fréquences dans un signal audio. Plus précisément, il s'agit de la largeur de la distribution spectrale qui contient une certaine quantité d'énergie, généralement définie comme 50% de la puissance spectrale totale (ce qui correspond à la largeur de bande spectrale à -3 dB).

Le spectral bandwidth peut être utile pour décrire un audio, car il fournit des informations sur la répartition des fréquences et la complexité spectrale du signal. Par exemple, un signal audio avec une bande passante spectrale étroite aura une tonalité plus pure (comme une note de flûte), tandis qu'un signal avec une large bande passante spectrale aura un spectre de fréquences plus large et un son plus complexe (comme le bruit d'un tambour).

"""


def get_spectral_bandwidth(y, sr, n_fft=N_FFT):
    """Compute the spectral bandwidth of the signal y

    Args:
        y (np.ndarray): audio signal
        sr (int): sampling rate
        n_fft (int, optional): FFT window size. Defaults to N_FFT.

    Returns:
        Tuple[Dict[str, float], np.ndarray]: a tuple containing a dictionary with the mean and std of the spectral bandwidth and the spectral bandwidth
    """
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft)
    return {"spec_bw_mean": np.mean(spec_bw), "spec_bw_std": np.std(spec_bw)}, spec_bw


#### Spectral flatness

def get_spectral_flatness(y, sr, n_fft=N_FFT) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute the spectral flatness of the signal y

    Args:
        y (np.ndarray): audio signal
        sr (int): sampling rate
        n_fft (int, optional): FFT window size. Defaults to N_FFT.

    Returns:
        Tuple[Dict[str, float], np.ndarray]: a tuple containing a dictionary with the mean and std of the spectral flatness and the spectral flatness
    """
    spec_flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft)
    return {"spec_flatness_mean": np.mean(spec_flatness), "spec_flatness_std": np.std(spec_flatness)}, spec_flatness


#### Spectral rolloff


def get_spectral_rolloff(y, sr, roll_percent=0.85, n_fft=N_FFT) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute the spectral rolloff of the signal y

    Args:
        y (np.ndarray): audio signal
        sr (int): sampling rate
        roll_percent (float, optional): percentage of the total energy. Defaults to 0.85.
        n_fft (int, optional): FFT window size. Defaults to N_FFT.

    Returns:
        Tuple[Dict[str, float], np.ndarray]: a tuple containing a dictionary with the mean and std of the spectral rolloff and the spectral rolloff
    """
    percent_str = str(roll_percent).split(".")[-1]
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent, n_fft=n_fft)
    return {f"spec_rolloff_{percent_str}_mean": np.mean(spec_rolloff),
            f"spec_rolloff_{percent_str}_std": np.std(spec_rolloff)}, spec_rolloff


#### Spectral contrast


def get_spectral_contrast(y, sr, n_fft=N_FFT) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute the spectral contrast of the signal y

    Args:
        y (np.ndarray): audio signal
        sr (int): sampling rate
        n_fft (int, optional): FFT window size. Defaults to N_FFT.

    Returns:
        Tuple[Dict[str, float], np.ndarray]: a tuple containing a dictionary with the mean and std of the spectral contrast and the spectral contrast
    """
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
    return {"spec_contrast_mean": np.mean(spec_contrast), "spec_contrast_std": np.std(spec_contrast)}, spec_contrast


## Extract features"""

def extract_features(y: np.ndarray, sr: int,
                     extract_duration : bool = True,
                        extract_rms_log : bool = True,
                        extract_zcr : bool = True,
                        extract_temp_cent : bool = True,
                        extract_attack_time : bool = True,
                        extract_pitch : bool = True,
                        extract_mfcc : bool = True,
                        extract_spec_cent : bool = True,
                        extract_spec_bw : bool = True,
                        extract_spec_flatness : bool = True,
                        extract_spec_rolloff : bool = True,
                        extract_spec_contrast : bool = True) -> Dict[str, float]:
    """
    Extract features from a sound.

    Args:
        y (np.ndarray): The sound.
        sr (int): The sampling rate of the sound.
        extract_duration (bool, optional): Whether to extract the duration of the sound. Defaults to True.
        extract_rms_log (bool, optional): Whether to extract the log of the RMS. Defaults to True.
        extract_zcr (bool, optional): Whether to extract the zero crossing rate. Defaults to True.
        extract_temp_cent (bool, optional): Whether to extract the temporal centroid. Defaults to True.
        extract_attack_time (bool, optional): Whether to extract the attack time. Defaults to True.
        extract_pitch (bool, optional): Whether to extract the pitch. Defaults to True.
        extract_mfcc (bool, optional): Whether to extract the MFCC. Defaults to True.
        extract_spec_cent (bool, optional): Whether to extract the spectral centroid. Defaults to True.
        extract_spec_bw (bool, optional): Whether to extract the spectral bandwidth. Defaults to True.
        extract_spec_flatness (bool, optional): Whether to extract the spectral flatness. Defaults to True.
        extract_spec_rolloff (bool, optional): Whether to extract the spectral rolloff. Defaults to True.
        extract_spec_contrast (bool, optional): Whether to extract the spectral contrast. Defaults to True.


    Returns:
        dict: A dictionary containing the extracted features.
    """

    features = {}

    # TIME DOMAIN FEATURES
    if extract_duration:
        features["duration"] = librosa.get_duration(y=y, sr=sr)
    if extract_rms_log:
        features.update(get_rms_log_stats(y=y)[0])
    if extract_zcr:
        features.update(get_zero_crossing_rate_stats(y=y)[0])
    if extract_temp_cent:
        features["temp_cent"] = get_temporal_centroid(y=y, sr=sr)
    if extract_attack_time:
        features["attack_time"] = get_attack_time(y=y, sr=sr)

    # FREQUENCY DOMAIN FEATURES
    if extract_pitch:
        features["pitch"] = get_pitch(y=y, sr=sr)
    if extract_mfcc:
        features.update(flatten_dict(get_mfcc_stats(y=y, sr=sr), start=1))
    if extract_spec_cent:
        features.update(get_spectral_centroid(y=y, sr=sr)[0])
    if extract_spec_bw:
        features.update(get_spectral_bandwidth(y=y, sr=sr)[0])
    if extract_spec_flatness:
        features.update(get_spectral_flatness(y=y, sr=sr)[0])
    if extract_spec_rolloff:
        features.update(get_spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0])
        features.update(get_spectral_rolloff(y=y, sr=sr, roll_percent=0.15)[0])
    if extract_spec_contrast:
        features.update(get_spectral_contrast(y=y, sr=sr)[0])

    return features



### • Extract features from dataset"""

def task_extract_features(sound):
    if not os.path.exists(sound.file_path):
        print(f"File {sound.file_path} does not exist")
        return None

    try:
        y, sr = load_audio_file(sound.file_path, sr=SAMPLE_RATE)
        y = pad_signal(y=y, target_length=N_FFT)
        features = extract_features(y, sr)
        features["file_path"] = sound.file_path
        return features
    except Exception as e:
        print(e)
        return None


def extract_features_from_dataset(dataset: pd.DataFrame, parallel_mode=True, n_workers=4, bool_streamlit = True) -> pd.DataFrame:
    lock = threading.Lock()

    if bool_streamlit:
        progress_text = "Operation in progress. Please wait."
        my_bar_st = st.progress(0, text=progress_text)

    feature_dicts = []

    if parallel_mode:
        print("> Start processing sounds (parallel mode)")
        sounds = dataset.itertuples()
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(task_extract_features, sound) for sound in sounds]
            for future in tqdm(as_completed(futures), total=len(futures)):
                features = future.result()
                if features is not None:
                    feature_dicts.append(features)
                if bool_streamlit:
                    with lock:
                        my_bar_st.progress(len(feature_dicts) / len(dataset), text=progress_text)


        print("> End processing sounds (parallel mode).")
        print(f"> {len(feature_dicts)}/{len(dataset)} sounds processed.")

        # Concatenate all the feature dictionaries into a new DataFrame
        feature_df = pd.DataFrame(feature_dicts)

        # Combine the original dataset and the feature DataFrame (merge on the file_path column)
        dataset = pd.merge(dataset, feature_df, on="file_path", how="left")
        print("> Dataset updated with the extracted features.")
        print(f"  > Dataset shape: {dataset.shape}")


    else:
        print("> Start processing sounds (sequential mode)")
        # Iterate over the dataset and extract features
        for sound in tqdm(dataset.itertuples(), total=len(dataset)):
            y, sr = load_audio_file(sound.file_path, sr=SAMPLE_RATE)
            y = pad_signal(y=y, target_length=N_FFT)
            features = extract_features(y, sr)

            # Add an 'Index' key to the features dictionary to keep track of the original index
            features['Index'] = sound.Index
            feature_dicts.append(features)

        print("> End processing sounds (sequential mode).")

        # Convert the list of feature dictionaries into a DataFrame
        feature_df = pd.DataFrame(feature_dicts)

        # Set the 'Index' column as the index of the feature_df DataFrame
        feature_df.set_index('Index', inplace=True)

        # Combine the original dataset and the feature DataFrame
        dataset = pd.concat([dataset, feature_df], axis=1)
        print("> Dataset updated with the extracted features.")
        print(f" > Dataset shape: {dataset.shape}")

    return dataset


def main():
    # Load the dataset
    dataset_csv_path = os.path.join(SOUNDS_DATASET_PATH, 'dataset.csv')
    dataset = pd.read_csv(dataset_csv_path)
    print(f"Dataset shape: {dataset.shape}")

    # Extract features from the dataset
    dataset = extract_features_from_dataset(dataset)

    # Save the dataset
    now_str = datetime.now().strftime("%Y%m%d")
    dataset.to_csv(os.path.join(SOUNDS_DATASET_PATH, f"dataset_features_{now_str}.csv"), index=False)

if __name__ == "__main__":
    main()