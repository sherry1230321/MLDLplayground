import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from fcmeans import FCM
import joblib

def load_data(csv_file):
    """Load network traffic data from a CSV file."""
    df = pd.read_csv(csv_file)
    # Assume the label column is named 'label'
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y

def main():
    csv_file = 'path_to_your_network_traffic_data.csv'  # Replace with your CSV file path

    # Load data
    X, y = load_data(csv_file)

    # Encode labels if they are not numerical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply FCM clustering
    fcm = FCM(n_clusters=2, random_state=42)
    fcm.fit(X_scaled)
    y_fcm = fcm.predict(X_scaled)

    # Evaluate clustering
    print("FCM Clustering Results:")
    print(pd.crosstab(y, y_fcm, rownames=['Actual'], colnames=['Predicted']))

    # Save the model and scaler
    joblib.dump(fcm, 'fcm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == '__main__':
    main()
