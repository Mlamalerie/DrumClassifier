import os
import concurrent.futures
from tqdm import tqdm
from pydub import AudioSegment

def convert_to_wav(audio_files):
    """
    Convertit une liste de fichiers audio en format WAV en utilisant le multithreading.

    Args:
        audio_files (list): Liste des chemins des fichiers audio à convertir.

    Returns:
        None
    """
    # Définir une fonction pour la conversion d'un fichier audio
    def convert_file(audio_file):
        # Récupérer le nom du fichier sans l'extension pour le nom du fichier de sortie
        output_file = f'{os.path.splitext(audio_file)[0]}.wav'

        # Charger le fichier audio avec pydub
        audio = AudioSegment.from_file(audio_file)

        # Convertir en format WAV
        audio.export(output_file, format='wav')

        return output_file

    # Utiliser concurrent.futures pour effectuer la conversion en multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Lancer la conversion pour chaque fichier audio
        converted_files = list(tqdm(executor.map(convert_file, audio_files), total=len(audio_files),
                                    desc='Conversion en cours', unit=' fichier'))

    return converted_files