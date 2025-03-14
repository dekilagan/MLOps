# Voice Classification with Hydra

This repository demonstrates a machine learning pipeline for voice classification using Hydra as the configuration manager. The pipeline includes:

- **Data ingestion:**  
  Loading training data from a CSV file containing pre-extracted audio features.
- **Model training:**  
  Training a classification model—currently supporting multiple model types:
  - **SVC**
  - **Random Forest**
  - **Logistic Regression**
  - **K-Nearest Neighbors**
- **Feature extraction:**  
  Extracting features from new raw audio files using SoundFile and librosa.
- **Prediction:**  
  Using the trained model to predict labels for new audio data.

Hydra externalizes configuration details (data paths, model parameters, feature extraction parameters, etc.), so you can easily switch between models and modify hyperparameters without changing the source code.

## Repository Structure

├── main.py # Main script to run the pipeline 
├── config 
  │ 
  └── config.yaml # Hydra configuration file 
├── data 
  |
  ├── voice.csv # CSV file with pre-extracted audio features and labels 
  └── 1001_DFA_ANG_XX.wav # New raw audio file for prediction 
└── README.md # This file

## Components Overview

- **main.py**  
  This script:
  - Loads configuration settings from `config/config.yaml` using Hydra.
  - Loads training data from `data/voice.csv` via pandas.
  - Splits the data into training and testing sets.
  - Instantiates a model based on the configuration (selectable among SVC, Random Forest, Logistic Regression, or K-Nearest Neighbors).
  - Trains the selected model and evaluates its performance.
  - Extracts features from a new raw audio file (using SoundFile for loading audio and librosa for feature extraction).
  - Uses the trained model to predict the label of the new audio file.

- **config/config.yaml**  
  This YAML file contains configuration parameters such as:
  - **Dataset details:**  
    File paths for the CSV and new audio files.
  - **Feature extraction settings:**  
    The number of MFCCs to extract.
  - **Model configuration:**  
    A parameter to select the model type (e.g., `svc`, `random_forest`, `logistic_regression`, or `knn`) along with specific hyperparameters for each model.
  - **General training parameters:**  
    Such as test split size and random seed.

- **data/**  
  Contains:
  - The CSV file (`voice.csv`) with pre-extracted features.
  - New raw audio files (e.g., `1001_DFA_ANG_XX.wav`) for testing or deployment.

## Prerequisites

Ensure you have Python installed. It is recommended to use a virtual environment. Then install the required packages:

```bash
pip install hydra-core omegaconf pandas librosa numpy scikit-learn soundfile
