# fuzzy_cmeans.py
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Generate sample data
n_samples = 150
x = np.linspace(0, 10, n_samples)
y = np.sin(x) + 0.1 * np.random.randn(n_samples)
data = np.vstack((x, y))

# Train Fuzzy C-means model
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data, 3, 2, error=0.005, maxiter=1000)

# Plot results
cluster_membership = np.argmax(u, axis=0)
for j in range(3):
    plt.scatter(data[0, cluster_membership == j], data[1, cluster_membership == j], label=f'Cluster {j+1}')
plt.legend()
plt.title('Fuzzy C-means Clustering')
plt.show()
