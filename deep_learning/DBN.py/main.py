# dbn.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dbn.tensorflow import SupervisedDBNClassification

# Generate synthetic data
def generate_data(n_samples=1000, n_features=20):
    X = np.random.rand(n_samples, n_features)
    y = (np.sum(X, axis=1) > n_features / 2).astype(int)
    return X, y

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train DBN model
model = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                    learning_rate_rbm=0.05,
                                    learning_rate=0.1,
                                    n_epochs_rbm=10,
                                    n_iter_backprop=100,
                                    batch_size=32,
                                    activation_function='relu',
                                    dropout_p=0.2)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy}')
