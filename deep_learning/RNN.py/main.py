import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Generate synthetic sequential data
def generate_data(n_samples=1000, seq_len=10):
    X = np.random.rand(n_samples, seq_len, 1)
    y = np.sum(X, axis=1)
    return X, y

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define model path
model_path = "rnn_model.h5"

# Check if a saved model exists
if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
else:
    # Build RNN model
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mse')

    # Train model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
    
    # Save model
    model.save(model_path)
    print("Model saved.")

# Evaluate model
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')
