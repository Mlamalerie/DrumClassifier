import os
import IPython.display as ipd
import librosa
from params import SAMPLE_RATE
from typing import List, Tuple
import pandas as pd
import numpy as np
import os
import librosa
import soundfile as sf
from glob import glob

# get all audios files in the dataset (wav, mp3, ...) use glob
def get_all_audios_files(dir_path, audio_extensions=[".wav", ".mp3", ".ogg", ".flac"]):
    files = []
    for extension in audio_extensions:
        files.extend(glob(os.path.join(dir_path, "**", "*" + extension), recursive=True))
    return files

def get_columns_by_prefix_(columns_list: list) -> dict:
    """
    Récupère les colonnes par préfixe à partir d'une liste de colonnes.

    Args:
        columns_list (list): Liste de colonnes.

    Returns:
        dict: Dictionnaire contenant les préfixes comme clés et les colonnes associées comme valeurs.
    """
    prefixes = set([col.split('_')[0] for col in columns_list])
    columns_by_prefix = {}
    for prefix in prefixes:
        columns_by_prefix[prefix] = [col for col in columns_list if col.startswith(prefix)]
    return columns_by_prefix


def get_columns_starts_with(columns_list: list, prefix) -> dict:
    """
    Récupère les colonnes par préfixe à partir d'une liste de colonnes.

    Args:
        columns_list (list): Liste de colonnes.

    Returns:
        dict: Dictionnaire contenant les préfixes comme clés et les colonnes associées comme valeurs.
    """

    return [col for col in columns_list if col.startswith(prefix)]

def generate_chunks(lst, chunk_size):
    """
    Génère des chunks (morceaux) d'une liste en fonction d'une taille de chunk donnée.

    Args:
        lst (list): Liste d'éléments à diviser en chunks.
        chunk_size (int): Taille de chaque chunk.

    Yields:
        list: Chunk (morceau) de la liste d'éléments.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def pad_signal(y: np.ndarray, target_length: int) -> np.ndarray:
    """
    Ajoute du padding à un signal pour atteindre une longueur spécifiée.

    Args:
        y (np.ndarray): Le signal audio.
        target_length (int): La longueur souhaitée du signal après padding.

    Returns:
        np.ndarray: Le signal audio avec padding.
    """
    len_y = len(y)

    if len_y < target_length:
        # Ajoute du padding pour que la longueur du signal soit égale à target_length
        y_padded = np.pad(y, (0, target_length - len_y), 'constant')
    else:
        # Si la longueur du signal est déjà supérieure ou égale à target_length, on le retourne tel quel
        y_padded = y

    return y_padded

def load_audio_file(file_path: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    ok = False
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        #NoBackendError
        if 'NoBackendError' in str(e) or "LibsndfileError" in str(e):
            try:
                y, sr = sf.read(file_path)
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier {file_path}: {e}")
            else:
                ok = True
        else:
            print(f"Erreur innatendue lors de la lecture du fichier {file_path}: {e}")
    else:
        ok = True
    return (y, sr) if ok else (None, None)

def play_audio(file_path: str = None, sr=SAMPLE_RATE, y=None, title=None):

    if file_path is None and y is None:
        raise ValueError("file_path ou y doit être spécifié.")
    if file_path is not None and y is not None:
        raise ValueError("file_path et y ne peuvent pas être spécifiés en même temps.")

    if file_path is not None and not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    title = os.path.join(os.path.basename(os.path.dirname(file_path)), os.path.basename(file_path)) if title is None else title

    print(">", title)
    if file_path is not None:
        # get extension

        ext = file_path.split('.')[-1]
        # change file_path : keep only parent folder / file name

        if ext in ['flac',"aif","aiff"]:
            y, sr = load_audio_file(file_path, sr=sr)
            return ipd.display(ipd.Audio(y, rate=sr))
        else:
            return ipd.display(ipd.Audio(file_path, rate=sr))
    elif y is not None:
        return ipd.display(ipd.Audio(y, rate=sr))


