# kmeans.py
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
data = datasets.load_iris()
X = data.data

# Train K-means model
model = KMeans(n_clusters=3)
y_pred = model.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title('K-means Clustering')
plt.show()
