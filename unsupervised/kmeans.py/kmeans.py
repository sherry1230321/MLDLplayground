import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import joblib
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load the network traffic data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data by scaling it."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def save_model(model, file_path):
    """Save the K-means model to a file."""
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """Load the K-means model from a file."""
    model = joblib.load(file_path)
    print(f"Model loaded from {file_path}")
    return model

# Load the processed network traffic data
data = load_data('network_traffic.csv')

# Preprocess the data
data_scaled, scaler = preprocess_data(data)

# Fit K-means model
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data_scaled)

# Assign clusters
data['cluster'] = kmeans.labels_

# Calculate the distance from each point to its cluster centroid
distances = kmeans.transform(data_scaled)
data['distance_to_centroid'] = np.min(distances, axis=1)

# Define a threshold for outliers (e.g., 95th percentile)
threshold = np.percentile(data['distance_to_centroid'], 95)
data['anomaly'] = data['distance_to_centroid'] > threshold

# Save the model and cluster assignments
save_model(kmeans, 'models/kmeans_model.pkl')
data.to_csv('clustered_data.csv', index=False)
data[data['anomaly']].to_csv('anomalies.csv', index=False)

# Analyze the clusters
for cluster in range(4):
    cluster_data = data[data['cluster'] == cluster]
    print(f'Cluster {cluster} analysis:')
    print(cluster_data.describe())

# Example: Network Intrusion Detection
normal_traffic = data[data['cluster'] == 0]
potential_intrusions = data[data['anomaly']]

print('Normal Traffic:')
print(normal_traffic.head())

print('Potential Intrusions:')
print(potential_intrusions.head())

# Plot the clusters
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=data['cluster'])
plt.title('K-means Clustering of Network Traffic')
plt.show()

# Plot potential intrusions
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=data['anomaly'])
plt.title('Potential Intrusions Detected')
plt.show()
