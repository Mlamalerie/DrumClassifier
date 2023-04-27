from params import AIRS_PATH, SAMPLE_RATE
import os
from glob import glob
from tools import get_all_audios_files, load_audio_file
import librosa
from tqdm import tqdm
import soundfile as sf
import re


def save_audio(y, sr, output_path):
    """
    Sauvegarde un signal audio au format WAV.

    """
    # Enregistrer l'audio au format WAV
    with sf.SoundFile(output_path, mode='w', samplerate=sr, channels=y.shape[0], subtype='PCM_16') as outfile:
        outfile.write(y.T)

def change_all_sample_rate(target_sample_rate) -> list:
    # Changer le taux d'échantillonnage

    air_files = get_all_audios_files(AIRS_PATH)
    # exclude files that are already resampled (with _resampled in the name)
    air_files = [file for file in air_files if not re.search("_resampled", file)]

    for file_path in tqdm(air_files):
        # Charger l'audio avec librosa
        y, original_sample_rate = librosa.load(file_path, sr=None, mono=False)

        # Changer le taux d'échantillonnage
        target_sample_rate = SAMPLE_RATE  # Définir le nouveau taux d'échantillonnage
        y_resampled = librosa.resample(y, orig_sr=original_sample_rate, target_sr=target_sample_rate)

        # Enregistrer l'audio au même format
        file_extension = file_path.split('.')[-1]

        # Changer le chemin de sortie
        output_path = file_path.replace(f".{file_extension}", f"_resampled.wav")

        # Enregistrer l'audio au format WAV
        save_audio(y_resampled, target_sample_rate, output_path)

    return air_files # old files

def ask_confirmation(text: str) -> bool:

    answer = input(f"{text} (y/n) ")
    while answer not in ['y', 'n']:
        answer = input(f"{text} (y/n) ")
    return answer == 'y'


def delete_old_files(files : list) -> None:
    # ask confirmation to delete old files

    if ask_confirmation(f"Do you want to delete {len(files)} old files ?"):
        for file in tqdm(files):
            os.remove(file)

def main() -> None:

    print(f" > This script will change the sample rate of all files in the AIRS dataset to {SAMPLE_RATE} Hz.")

    old_files = change_all_sample_rate(target_sample_rate=SAMPLE_RATE)
    delete_old_files(old_files)





if __name__ == "__main__":
    main()