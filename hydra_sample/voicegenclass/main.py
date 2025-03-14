import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import the model factory function from model.py
from model import get_model

def extract_features_from_audio(file_name, config):
    """
    Extracts MFCC features along with additional audio features from a raw audio file.
    Returns a concatenated feature vector.
    """
    # Load audio using soundfile
    audio, sample_rate = sf.read(file_name)
    
    # Convert stereo to mono if necessary
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Extract MFCC features using librosa
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=config.features.n_mfcc)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    
    # If necessary, slice the MFCC features to match training data (e.g., use first 18 coefficients)
    if len(mfccs_processed) > 18:
        mfccs_processed = mfccs_processed[:18]
    
    # Extract additional features: zero crossing count and spectral centroid
    zero_crossings = np.array([librosa.zero_crossings(audio, pad=False).sum()])
    spectral_centroid = np.array([np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))])
    
    # Concatenate features into a single vector
    features = np.concatenate((mfccs_processed, zero_crossings, spectral_centroid))
    return features

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Get original working directory (where main.py is located)
    BASE_DIR = get_original_cwd()
    
    # Build the absolute path for the training CSV file
    train_csv_path = os.path.join(BASE_DIR, cfg.dataset.base_path, cfg.dataset.train_csv)
    print(f"Loading training data from: {train_csv_path}")
    
    # Load training data from CSV using pandas
    df = pd.read_csv(train_csv_path)
    
    # Separate features and labels
    X = df.drop(columns=[cfg.dataset.label_column]).values
    y = df[cfg.dataset.label_column].values
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.model.test_size, random_state=cfg.model.random_state
    )
    
    # Instantiate the model using the generic factory call
    model = get_model(cfg)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
    
    # Build the absolute path for the new audio file for inference
    new_audio_path = os.path.join(BASE_DIR, cfg.dataset.base_path, cfg.dataset.new_audio_file)
    print(f"Extracting features from new audio file: {new_audio_path}")
    
    # Extract features from the new audio file and predict its label
    new_features = extract_features_from_audio(new_audio_path, cfg)
    new_prediction = model.predict(new_features.reshape(1, -1))
    print(f"Prediction for new audio file '{cfg.dataset.new_audio_file}': {new_prediction[0]}")

if __name__ == "__main__":
    main()
