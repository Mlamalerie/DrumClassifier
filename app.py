import base64
import os
import zipfile
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
from pydub import AudioSegment
import glob
from IPython.display import Audio

import shutil

import time
import pickle
from tools import get_all_audios_files, load_audio_file
from extract_audio_features import extract_features, extract_features_from_dataset

st.set_page_config(
    page_title="Drum Kit Classifier",
    page_icon="🥁",
    layout="centered",
    initial_sidebar_state="expanded",
)

# st.image("logo.jpg")
st.sidebar.title("Drum Kit Classifier 🥁")
st.sidebar.caption("Use machine learning to classify drum kits")
st.sidebar.divider()
st.sidebar.markdown("""
**📝 Instructions:**
1. Upload your drum kit zip file.
2. Select a model.
3. Click on the `Classify` button.
4. Wait for the classification results to appear.
""")
st.sidebar.divider()
st.sidebar.success("This is a **beta** version of the app.")

# =======
#   Session state
# =======
# We need to set up session state via st.session_state so that app interactions don't reset the app.

if "ok_submit" not in st.session_state:
    st.session_state.ok_submit = False


# =======
#   Utils
# =======

# Load your trained models here (example models: model1, model2)
# You may need to adjust this part to suit your actual models
@st.cache_resource(show_spinner=True)
def load_ml_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_path}' not found")

    items = []
    with open(model_path, 'rb') as f:
        while True:
            try:
                items.append(pickle.load(f))
            except EOFError:
                break

    model = items[0]
    print(model, "#######################", type(model))
    scaler = items[1]
    features_columns = items[2]
    print(f"> '{model_path}' loaded successfully")
    print(f">>> Model: {model}")
    print(f">>> Scaler: {scaler}")
    print(f">>> {len(features_columns)} Features columns: {features_columns[:5]} ... {features_columns[-5:]}")

    return model, scaler, features_columns


RF_MODEL_PATH = './models/RandomForest data[aug=1, s=10291, s_per_class=[596,1283], n_feats=77, feat_select=1] 20230514_04.pkl'
SVC_MODEL_PATH = './models/SVC data[aug=1, s=10291, s_per_class=[596,1283], n_feats=77, feat_select=1] 20230514_15.pkl'
LGBM_MODEL_PATH = './models/LGBM data[aug=1, s=10291, s_per_class=[596,1283], n_feats=77, feat_select=1] 20230515_02.pkl'
model_rf, scaler_rf, features_columns_rf = load_ml_model(RF_MODEL_PATH)
model_svc, scaler_svc, features_columns_svc = load_ml_model(SVC_MODEL_PATH)
model_lgbm, scaler_lgbm, features_columns_lgbm = load_ml_model(LGBM_MODEL_PATH)

MODEL_MAPPING = {
    'RandomForest': model_rf,
    'SVC': model_svc,
    'LGBM': model_lgbm,
}

SCALER_MAPPING = {
    'RandomForest': scaler_rf,
    'SVC': scaler_svc,
    'LGBM': scaler_lgbm,
}

FEATURES_COLUMNS_MAPPING = {
    'RandomForest': features_columns_rf,
    'SVC': features_columns_svc,
    'LGBM': features_columns_lgbm,
}


@st.cache(allow_output_mutation=True)
def load_audio(file_path):
    y, sr = load_audio_file(file_path=file_path)
    return y, sr


# Créer dataset à partir d'une liste de fichiers audio
@st.cache_resource(show_spinner=True)
def create_dataset_from_audio_files(audio_files: list):
    # result dataset : columns = file_path, file_name
    file_path_series = pd.Series(audio_files)
    file_name_series = file_path_series.apply(lambda x: os.path.basename(x).split('.')[0])

    return pd.DataFrame(
        {
            'file_path': file_path_series,
            'file_name': file_name_series,
        }
    )


def make_archive(source, destination):
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move(f'{name}.{format}', destination)

    return destination


def main():
    uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # get the name of the uploaded file
            uploaded_file_name = uploaded_file.name

            # extract zip file
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # create columns 20% - 80%
            col1, col2 = st.columns([0.2, 0.8])
            if temp_dir is not None:
                # get all audios files in the dataset (wav, mp3, ...) use glob
                audios = get_all_audios_files(temp_dir)
                # display count of audios files
                if len(audios) == 0:
                    st.error("No audio files found", icon='📂')
                else:
                    col1.metric(label="📂 Audio files found.", value=len(audios))

                    model_name = col2.selectbox(
                        "Select the model 🤖",
                        MODEL_MAPPING.keys(),
                    )
                    # submit button
                    ok_button = col1.button("Predict", key="predict")
                    st.session_state.ok_submit = ok_button

            if st.session_state.ok_submit and len(audios) > 0:
                model = MODEL_MAPPING[model_name]
                scaler = SCALER_MAPPING[model_name]
                # get features from model
                if model_name == 'RandomForest':
                    features_columns = model.feature_importances_
                    print(features_columns)

                # create dataset from audio files
                dataset = create_dataset_from_audio_files(audios)
                # add features to dataset
                with st.spinner("Extracting features..."):
                    dataset = extract_features_from_dataset(dataset, bool_streamlit=True)

                with st.expander("Show dataset"):
                    # display dataset
                    # features selection : keep only features columns which are in the model
                    st.write("dataset", dataset.shape, dataset.columns.tolist())
                    st.dataframe(dataset.head())
                    X = dataset[FEATURES_COLUMNS_MAPPING[model_name]]
                    # scale dataset
                    X = scaler.transform(X)
                    st.write("X", X.shape)
                    st.write(X)

                # make predictions and probabilities
                with st.spinner("Making predictions..."):
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)  # get probabilities

                # new code...
                confidence_threshold = 0.75  # set your confidence threshold here
                top_n = 2  # set the number of top predictions you want to show

                # for each prediction, get the top 2 classes and their probabilities
                top_predictions = []
                for proba in probabilities:
                    # get top 2 predictions
                    top_indices = np.argsort(proba)[-top_n:]
                    top_probs = proba[top_indices]
                    top_classes = model.classes_[top_indices]
                    # if the highest probability is less than the threshold, show top 2 predictions
                    if top_probs[-1] < confidence_threshold:
                        top_prediction = f"{top_classes[-1]} ({top_probs[-1] * 100:.0f}%), {top_classes[-2]} ({top_probs[-2] * 100:.0f}%)"
                    else:
                        top_prediction = f"{top_classes[-1]} ({top_probs[-1] * 100:.0f}%)"
                    top_predictions.append(top_prediction)

                # concatenate top_predictions to dataset
                dataset['top_predictions'] = top_predictions
                dataset['predictions'] = predictions

                # display predictions in a grid
                st.subheader("Predictions 🔮")

                with st.expander("Show predictions"):
                    st.caption(f"Showing top {top_n} predictions with confidence > {confidence_threshold}")
                    st.caption("*file_name* ⇢ **top_prediction**")

                    n_cols = 2
                    cols = st.columns(n_cols)
                    for e, (idx_row, row) in enumerate(dataset.iterrows()):
                        audio_file = row['file_path']
                        audio_name = row['file_name']
                        top_prediction = row['top_predictions']
                        cols[e % n_cols].audio(audio_file)
                        cols[e % n_cols].markdown(f'"*{audio_name}*" ⇢ **{top_prediction}**')

                st.subheader("Download 📥")
                # buttons

                # download dataset with predictions
                with st.spinner("Downloading dataset with predictions..."):
                    name_download = f"predictions - {uploaded_file_name}.csv"
                    st.download_button(
                        label="Download dataset with predictions",
                        data=dataset.loc[:, ['file_name', 'predictions', 'top_predictions']].to_csv(index=False),
                        file_name=name_download,
                        mime="text/csv",
                    )




# Function to create classified directories
def create_classified_directories(temp_dir, dataset):
    # verify if temp_dir exists
    if not os.path.exists(temp_dir):
        raise ValueError(f"temp_dir {temp_dir} does not exist")
    # create directories
    class_dirs = {class_name: os.path.join(temp_dir, class_name) for class_name in dataset['predictions'].unique()}
    for class_dir in class_dirs.values():
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    st.write(class_dirs)
    for idx, row in dataset.iterrows():
        shutil.copy2(row['file_path'], os.path.join(class_dirs[row['predictions']], os.path.basename(row['file_path'])))

    return temp_dir


# Functions to download files
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'


if __name__ == "__main__":
    main()
