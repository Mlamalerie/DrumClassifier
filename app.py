import os
import zipfile
import tempfile
import pandas as pd
import streamlit as st
from pydub import AudioSegment
from IPython.display import Audio
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

if not "ok_submit" in st.session_state:
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
    print(model,"#######################",type(model))
    scaler = items[1]
    features_columns = items[2]
    print(f"> '{model_path}' loaded successfully")
    print(f">>> Model: {model}")
    print(f">>> Scaler: {scaler}")
    print(f">>> {len(features_columns)} Features columns: {features_columns[:5]} ... {features_columns[-5:]}")

    return model, scaler, features_columns

RF_MODEL_PATH = './models/RandomForest data[aug=1, s=10291, s_per_class=[596,1283], n_feats=98, feat_select=0] 20230513_10.pkl'
SVC_MODEL_PATH = './models/SVC data[aug=1, s=10291, s_per_class=[596,1283], n_feats=98, feat_select=0] 20230513_05.pkl'
LGBM_MODEL_PATH = './models/LGBM data[aug=1, s=10291, s_per_class=[596,1283], n_feats=98, feat_select=0] 20230513_15.pkl'
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
def create_dataset_from_audio_files(audio_files : list):
    # result dataset : columns = file_path, file_name
    file_path_series = pd.Series(audio_files)
    file_name_series = file_path_series.apply(lambda x: os.path.basename(x).split('.')[0])

    dataset = pd.DataFrame({
        'file_path': file_path_series,
        'file_name': file_name_series,
    })

    return dataset

def main():
    uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # create columns 20% - 80%
            col1,col2 = st.columns([0.2,0.8])
            if temp_dir is not None:
                # get all audios files in the dataset (wav, mp3, ...) use glob
                audios = get_all_audios_files(temp_dir)
                # display count of audios files
                if len(audios) == 0:
                    st.error("No audio files found",icon='📂')
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
                    dataset = extract_features_from_dataset(dataset)

                with st.expander("Show dataset"):
                    # display dataset
                    # features selection : keep only features columns which are in the model
                    st.write("dataset",dataset.shape, dataset.columns.tolist())
                    st.dataframe(dataset.head())
                    X = dataset[FEATURES_COLUMNS_MAPPING[model_name]]
                    # scale dataset
                    X = scaler.transform(X)
                    st.write("X",X.shape)
                    st.write(X)

                # make predictions
                with st.spinner("Making predictions..."):
                    predictions = model.predict(X)
                
                # concatenate predictions to dataset
                dataset['predictions'] = predictions

                # display predictions in a grid
                st.subheader("Predictions 🔮")
                st.caption(f"*file_name* ⇢ **prediction**")
                n_cols = 3
                cols = st.columns(n_cols)
                for e,(idx_row, row) in enumerate(dataset.iterrows()):

                    audio_file = row['file_path']
                    audio_name = row['file_name']
                    class_pred = row['predictions']
                    cols[e%n_cols].audio(audio_file)
                    cols[e%n_cols].markdown(f"*{audio_name}* ⇢ **{class_pred}**")

                    # sleep 1s every row
                    if e%n_cols == 0:
                        time.sleep(1)




if __name__ == "__main__":
    main()
