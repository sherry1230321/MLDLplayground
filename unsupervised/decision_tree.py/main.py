import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from extract_features import extract_features  # Make sure this is in the same directory or adjust the import path

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

    # Train a Decision Tree classifier
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Evaluate the classifier
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Save the model and scaler
    import joblib
    joblib.dump(classifier, 'decision_tree_classifier.pkl')
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == '__main__':
    main()
