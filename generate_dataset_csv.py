from params import SOUNDS_DATASET_PATH
import os
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.15
RANDOM_STATE = 123


# get file : path, name, extension and class (folder name)
def get_file_info(file_path):
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)
    file_label = os.path.basename(os.path.dirname(file_path))
    return file_path, file_name, file_extension, file_label


test_audio_path = "G:\Shared drives\PFE - ING3 Mlamali\DrumClassifier - Sounds Dataset\Kick\Kick 0000.flac"
print(get_file_info(test_audio_path))


# %%

# get all audios files in the dataset (wav, mp3, ...) use glob
def get_all_audios_files(dir_path, audio_extensions=[".wav", ".mp3", ".ogg", ".flac"]):
    files = []
    for extension in audio_extensions:
        files.extend(glob(os.path.join(dir_path, "**", "*" + extension), recursive=True))
    return files


def get_df_audio_files(dir_path, audio_extensions=[".wav", ".mp3", ".ogg", ".flac", ".aiff", ".aif"]):
    # get all audios files
    files = get_all_audios_files(dir_path, audio_extensions)
    # get file : path, name, extension and class (folder name)
    files_info = [get_file_info(file) for file in files]
    df = pd.DataFrame(files_info, columns=["file_path", "file_name", "file_extension", "class"])
    return df


def add_split_column(df, test_size=0.2, random_state=123, grouping_column="class"):
    # add split column into train and test
    # each class should be represented in both train and test
    train_index, test_index = train_test_split(df.index, test_size=test_size, random_state=random_state,
                                               stratify=df[grouping_column])
    # note : le paramètre stratify permet de s'assurer que la répartition des classes dans les ensembles d'entraînement et de test est équilibrée et reflète la répartition des classes dans l'ensemble de données global.
    df["split"] = "train"
    df.loc[test_index, "split"] = "test"
    return df



def generate_dataset_csv(dir_path: str, output_csv_path: str, test_size: float = TEST_SIZE,
                         random_state: int = RANDOM_STATE):

    print("Generate dataset csv file ...")
    # create a dataframe
    df = get_df_audio_files(dir_path)

    # add split column into train and test
    df = add_split_column(df, test_size, random_state)

    # print info df

    print("> Dataset shape: {}".format(df.shape))
    print("> Dataset columns: {}".format(df.columns))
    print("> Dataset classes: {}".format(df["class"].unique()))
    print("> Dataset info:")
    # count the number of files per class and split
    print(df.groupby(["class", "split"]).count()["file_path"])

    # save the dataframe to csv
    df.to_csv(output_csv_path, index=False)

    # print output
    print("> Saved to: {}".format(output_csv_path))

    return df


def main():

    # generate the dataset csv file
    output_csv_path = os.path.join(SOUNDS_DATASET_PATH, "dataset.csv")
    _ = generate_dataset_csv(SOUNDS_DATASET_PATH, output_csv_path)

if __name__ == "__main__":
    main()
