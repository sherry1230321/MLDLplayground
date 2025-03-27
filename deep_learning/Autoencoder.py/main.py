import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load dataset
data = load_iris()
X_train, X_test, _, _ = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define model path
model_path = "autoencoder_model.h5"

# Check if a saved model exists
if os.path.exists(model_path):
    print("Loading existing model...")
    autoencoder = load_model(model_path)
else:
    # Build AE model
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)

    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    
    # Compile model
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train model
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=8, validation_split=0.2)
    
    # Save model
    autoencoder.save(model_path)
    print("Model saved.")

# Evaluate model
loss = autoencoder.evaluate(X_test, X_test)
print(f'Loss: {loss}')
