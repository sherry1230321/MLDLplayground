import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def extract_features(file_name):
    """Extract audio features from an audio file."""
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
        
        # Aggregate features
        features = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_contrast, axis=1)
        ])
        
        return features
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None

def load_data(directory):
    """Load data from a directory and extract features."""
    features_list = []
    labels = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)  # Use the folder name as the label
                features = extract_features(file_path)
                if features is not None:
                    features_list.append(features)
                    labels.append(label)

    return np.array(features_list), np.array(labels)

def main():
    directory = 'path_to_your_audio_files'  # Replace with your directory containing .wav files

    # Load data
    X, y = load_data(directory)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Random Forest classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Evaluate the classifier
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Save the model and scaler
    import joblib
    joblib.dump(classifier, 'audio_classifier.pkl')
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == '__main__':
    main()
