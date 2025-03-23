import pandas as pd
import numpy as np
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import nltk
import cv2
from PIL import Image
import wave
import struct
import librosa
import io
import joblib

# Download NLTK resources
def download_nltk_resources():
    """Download NLTK resources if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

# Make sure NLTK resources are available
download_nltk_resources()

# Functions for processing different types of data
def process_tabular_data(X, y, test_size=0.2, handle_missing=True, missing_strategy='mean', 
                        encode_categorical=True, categorical_encoding='one-hot',
                        scale_features=True, scaling_method='standard', random_state=42):
    # Function implementation...

def process_text_data(X, y=None, preprocessing_options=None, vectorizer_type='TF-IDF', max_features=5000, 
                      test_size=0.2, random_state=42, for_topic_modeling=False):
    # Function implementation...

def process_image_data(images, labels=None, resize_dim=128, augmentation=None, test_size=0.2, random_state=42):
    # Function implementation...

def process_audio_data(audio_files, labels=None, sample_rate=44100, duration=3, test_size=0.2, random_state=42):
    # Function implementation...

# Functions for saving and loading processed data
def save_processed_data(data, data_type, directory='Data/processed'):
    """
    Save processed data to files.
    
    Parameters:
    -----------
    data : various types
        Processed data to save
    data_type : str
        Type of data ('tabular', 'text', 'images', 'audio')
    directory : str, default='Data/processed'
        Directory to save the processed data
    """
    if data_type == 'tabular':
        X_train, X_test, y_train, y_test, pipeline = data
        if not os.path.exists(f"{directory}/tabular"):
            os.makedirs(f"{directory}/tabular")
        np.save(f"{directory}/tabular/X_train.npy", X_train)
        np.save(f"{directory}/tabular/X_test.npy", X_test)
        np.save(f"{directory}/tabular/y_train.npy", y_train)
        np.save(f"{directory}/tabular/y_test.npy", y_test)
        joblib.dump(pipeline, f"{directory}/tabular/pipeline.pkl")
    
    elif data_type == 'text':
        X_train_vec, X_test_vec, y_train, y_test, vectorizer = data
        if not os.path.exists(f"{directory}/text"):
            os.makedirs(f"{directory}/text")
        joblib.dump(X_train_vec, f"{directory}/text/X_train_vec.pkl")
        joblib.dump(X_test_vec, f"{directory}/text/X_test_vec.pkl")
        np.save(f"{directory}/text/y_train.npy", y_train)
        np.save(f"{directory}/text/y_test.npy", y_test)
        joblib.dump(vectorizer, f"{directory}/text/vectorizer.pkl")
    
    elif data_type == 'images':
        X_train, X_test, y_train, y_test = data
        if not os.path.exists(f"{directory}/images"):
            os.makedirs(f"{directory}/images")
        np.save(f"{directory}/images/X_train.npy", X_train)
        np.save(f"{directory}/images/X_test.npy", X_test)
        np.save(f"{directory}/images/y_train.npy", y_train)
        np.save(f"{directory}/images/y_test.npy", y_test)
    
    elif data_type == 'audio':
        X_train, X_test, y_train, y_test = data
        if not os.path.exists(f"{directory}/audio"):
            os.makedirs(f"{directory}/audio")
        np.save(f"{directory}/audio/X_train.npy", X_train)
        np.save(f"{directory}/audio/X_test.npy", X_test)
        np.save(f"{directory}/audio/y_train.npy", y_train)
        np.save(f"{directory}/audio/y_test.npy", y_test)

def load_processed_data(data_type, directory='Data/processed'):
    """
    Load processed data from files.
    
    Parameters:
    -----------
    data_type : str
        Type of data ('tabular', 'text', 'images', 'audio')
    directory : str, default='Data/processed'
        Directory to load the processed data from
    
    Returns:
    --------
    Various types depending on data_type
        Loaded processed data
    """
    if data_type == 'tabular':
        X_train = np.load(f"{directory}/tabular/X_train.npy")
        X_test = np.load(f"{directory}/tabular/X_test.npy")
        y_train = np.load(f"{directory}/tabular/y_train.npy")
        y_test = np.load(f"{directory}/tabular/y_test.npy")
        pipeline = joblib.load(f"{directory}/tabular/pipeline.pkl")
        return X_train, X_test, y_train, y_test, pipeline
    
    elif data_type == 'text':
        X_train_vec = joblib.load(f"{directory}/text/X_train_vec.pkl")
        X_test_vec = joblib.load(f"{directory}/text/X_test_vec.pkl")
        y_train = np.load(f"{directory}/text/y_train.npy")
        y_test = np.load(f"{directory}/text/y_test.npy")
        vectorizer = joblib.load(f"{directory}/text/vectorizer.pkl")
        return X_train_vec, X_test_vec, y_train, y_test, vectorizer
    
    elif data_type == 'images':
        X_train = np.load(f"{directory}/images/X_train.npy")
        X_test = np.load(f"{directory}/images/X_test.npy")
        y_train = np.load(f"{directory}/images/y_train.npy")
        y_test = np.load(f"{directory}/images/y_test.npy")
        return X_train, X_test, y_train, y_test
    
    elif data_type == 'audio':
        X_train = np.load(f"{directory}/audio/X_train.npy")
        X_test = np.load(f"{directory}/audio/X_test.npy")
        y_train = np.load(f"{directory}/audio/y_train.npy")
        y_test = np.load(f"{directory}/audio/y_test.npy")
        return X_train, X_test, y_train, y_test
