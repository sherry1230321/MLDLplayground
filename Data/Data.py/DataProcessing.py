import pandas as pd
import numpy as np
import os
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

def process_tabular_data(X, y, test_size=0.2, handle_missing=True, missing_strategy='mean', 
                        encode_categorical=True, categorical_encoding='one-hot',
                        scale_features=True, scaling_method='standard', random_state=42):
    """
    Process tabular data for machine learning.
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature data
    y : pandas Series
        Target variable
    test_size : float, default=0.2
        Proportion of data to use for testing
    handle_missing : bool, default=True
        Whether to handle missing values
    missing_strategy : str, default='mean'
        Strategy for imputing missing values ('mean', 'median', 'most_frequent', 'constant')
    encode_categorical : bool, default=True
        Whether to encode categorical features
    categorical_encoding : str, default='one-hot'
        Method for encoding categorical features ('one-hot', 'label')
    scale_features : bool, default=True
        Whether to scale numerical features
    scaling_method : str, default='standard'
        Method for scaling numerical features ('standard', 'minmax', 'robust', 'none')
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    X_train : numpy array
        Training features
    X_test : numpy array
        Testing features
    y_train : numpy array
        Training targets
    y_test : numpy array
        Testing targets
    preprocessing_pipeline : sklearn Pipeline
        The preprocessing pipeline used for transformations
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    transformers = []
    
    # Handle numerical features
    if numerical_cols:
        num_pipeline_steps = []
        
        # Add imputer if required
        if handle_missing:
            num_pipeline_steps.append(('imputer', SimpleImputer(strategy=missing_strategy)))
        
        # Add scaler if required
        if scale_features and scaling_method != 'none':
            if scaling_method == 'standard':
                num_pipeline_steps.append(('scaler', StandardScaler()))
            elif scaling_method == 'minmax':
                num_pipeline_steps.append(('scaler', MinMaxScaler()))
            elif scaling_method == 'robust':
                num_pipeline_steps.append(('scaler', RobustScaler()))
        
        if num_pipeline_steps:
            transformers.append(('num', Pipeline(steps=num_pipeline_steps), numerical_cols))
    
    # Handle categorical features
    if categorical_cols and encode_categorical:
        cat_pipeline_steps = []
        
        # Add imputer for categorical features if required
        if handle_missing:
            cat_pipeline_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        
        # Add encoder based on the selected method
        if categorical_encoding == 'one-hot':
            cat_pipeline_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
        elif categorical_encoding == 'label':
            cat_pipeline_steps.append(('encoder', LabelEncoder()))
        
        if cat_pipeline_steps:
            transformers.append(('cat', Pipeline(steps=cat_pipeline_steps), categorical_cols))
    
    # Create and fit the preprocessing pipeline
    if transformers:
        preprocessing_pipeline = ColumnTransformer(transformers=transformers, remainder='passthrough')
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)
    else:
        preprocessing_pipeline = None
        X_train_processed = X_train.values
        X_test_processed = X_test.values
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessing_pipeline

def process_text_data(X, y=None, preprocessing_options=None, vectorizer_type='TF-IDF', max_features=5000, 
                      test_size=0.2, random_state=42, for_topic_modeling=False):
    """
    Process text data for NLP tasks.
    
    Parameters:
    -----------
    X : pandas Series or list
        Text data
    y : pandas Series, optional
        Target variable for supervised learning tasks
    preprocessing_options : list, default=None
        List of preprocessing steps to apply
    vectorizer_type : str, default='TF-IDF'
        Type of vectorizer to use ('CountVectorizer', 'TF-IDF', 'Word Embeddings (Word2Vec)')
    max_features : int, default=5000
        Maximum number of features for vectorization
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random state for reproducibility
    for_topic_modeling : bool, default=False
        If True, returns the document-term matrix instead of train-test split
    
    Returns:
    --------
    Various outputs depending on parameters and use case
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    
    # Apply text preprocessing
    processed_texts = []
    
    for text in X:
        if pd.isna(text):
            processed_texts.append("")
            continue
            
        processed_text = text
        
        if preprocessing_options is not None:
            # Convert to lowercase
            if "Lowercase" in preprocessing_options:
                processed_text = processed_text.lower()
            
            # Remove punctuation
            if "Remove Punctuation" in preprocessing_options:
                processed_text = re.sub(r'[^\w\s]', '', processed_text)
            
            # Tokenize
            tokens = word_tokenize(processed_text)
            
            # Remove stopwords
            if "Remove Stopwords" in preprocessing_options:
                stop_words = set(stopwords.words('english'))
                tokens = [word for word in tokens if word not in stop_words]
            
            # Apply stemming
            if "Stemming" in preprocessing_options:
                stemmer = PorterStemmer()
                tokens = [stemmer.stem(word) for word in tokens]
            
            # Apply lemmatization
            if "Lemmatization" in preprocessing_options:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            # Join tokens back into text
            processed_text = ' '.join(tokens)
        
        processed_texts.append(processed_text)
    
    # Vectorize the text
    if vectorizer_type == 'CountVectorizer':
        vectorizer = CountVectorizer(max_features=max_features)
    elif vectorizer_type == 'TF-IDF':
        vectorizer = TfidfVectorizer(max_features=max_features)
    else:  # Word embeddings - simplified implementation for demonstration
        # In a real implementation, you would use gensim's Word2Vec or similar
        vectorizer = TfidfVectorizer(max_features=max_features)
    
    # For topic modeling, return the document-term matrix directly
    if for_topic_modeling:
        dtm = vectorizer.fit_transform(processed_texts)
        return vectorizer, dtm
    
    # For supervised learning, perform train-test split
    if y is not None:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, y, test_size=test_size, random_state=random_state
        )
        
        # Vectorize
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        return X_train_vec, X_test_vec, y_train, y_test, vectorizer
    else:
        # Just vectorize without splitting
        X_vec = vectorizer.fit_transform(processed_texts)
        return X_vec, vectorizer

def process_image_data(images, labels=None, resize_dim=128, augmentation=None, test_size=0.2, random_state=42):
    """
    Process image data for image classification tasks.
    
    Parameters:
    -----------
    images : list of PIL.Image or numpy arrays
        List of images to process
    labels : array-like, optional
        Labels for the images
    resize_dim : int, default=128
        Dimension to resize images to (square)
    augmentation : list, default=None
        List of augmentation techniques to apply
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    X_train : numpy array
        Training images
    X_test : numpy array
        Testing images
    y_train : numpy array
        Training labels
    y_test : numpy array
        Testing labels
    """
    processed_images = []
    
    for image in images:
        # Resize image
        if isinstance(image, Image.Image):
            resized_img = image.resize((resize_dim, resize_dim))
        else:  # Assume numpy array
            resized_img = cv2.resize(image, (resize_dim, resize_dim))
        
        # Convert to numpy array if PIL Image
        if isinstance(resized_img, Image.Image):
            img_array = np.array(resized_img)
        else:
            img_array = resized_img
        
        # Normalize pixel values
        if img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0
        
        processed_images.append(img_array)
    
    # Convert to numpy array
    processed_images = np.array(processed_images)
    
    # Data augmentation would be implemented here in a real application
    
    # Split data if labels are provided
    if labels is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            processed_images, labels, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    else:
        return processed_images

def process_audio_data(audio_files, labels=None, sample_rate=44100, duration=3, test_size=0.2, random_state=42):
    """
    Process audio data for audio classification tasks.
    
    Parameters:
    -----------
    audio_files : list of str
        List of paths to audio files
    labels : array-like, optional
        Labels for the audio files
    sample_rate : int, default=44100
        Sample rate for audio files
    duration : int, default=3
        Duration of each audio file in seconds
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    X_train : numpy array
        Training audio data
    X_test : numpy array
        Testing audio data
    y_train : numpy array
        Training labels
    y_test : numpy array
        Testing labels
    """
    processed_audios = []
    
    for file in audio_files:
        # Load audio file
        audio, sr = librosa.load(file, sr=sample_rate, duration=duration)
        
        # Ensure the audio is the correct length
        if len(audio) < sample_rate * duration:
            padding = sample_rate * duration - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        else:
            audio = audio[:sample_rate * duration]
        
        processed_audios.append(audio)
    
    # Convert to numpy array
    processed_audios = np.array(processed_audios)
    
    # Split data if labels are provided
    if labels is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            processed_audios, labels, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    else:
        return processed_audios

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
