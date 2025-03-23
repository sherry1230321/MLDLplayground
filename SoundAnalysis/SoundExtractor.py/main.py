import librosa
import numpy as np
import pandas as pd
import os

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

def process_directory(directory):
    """Process all files in a given directory and extract audio features."""
    features_list = []
    file_names = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                features = extract_features(file_path)
                if features is not None:
                    features_list.append(features)
                    file_names.append(file)

    # Create a DataFrame and save to CSV
    feature_columns = [f'mfcc_{i}' for i in range(40)] + [f'chroma_{i}' for i in range(12)] + [f'spectral_contrast_{i}' for i in range(7)]
    df = pd.DataFrame(features_list, columns=feature_columns)
    df['file_name'] = file_names

    output_csv = os.path.join(directory, 'audio_features.csv')
    df.to_csv(output_csv, index=False)

    print(f"Features extracted and saved to {output_csv}")

if __name__ == '__main__':
    directory = 'path_to_your_audio_files'  # Replace with your directory containing .wav files
    process_directory(directory)
