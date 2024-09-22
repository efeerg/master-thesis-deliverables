import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

class LSTMModel:
    def __init__(self, sequence_length, num_features, hidden_units, learning_rate=0.001):
        # The model is initialized with parameters like sequence_length, num_features, hidden_units, and learning_rate.
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        # LSTM model with two layers, dropout regularization, and dense output
        model = Sequential()
        
        # First LSTM Layer with return_sequences=True for stacking
        model.add(LSTM(units=self.hidden_units, return_sequences=True, input_shape=(self.sequence_length, self.num_features)))
        model.add(Dropout(0.3))  # Dropout for regularization

        # Second LSTM Layer without return_sequences, as it's the final LSTM layer
        model.add(LSTM(units=self.hidden_units, return_sequences=False))
        model.add(Dropout(0.3))  # Dropout for regularization

        # Dense layer for feature extraction
        model.add(Dense(64, activation='relu'))

        # Output layer with linear activation for unrestricted values
        model.add(Dense(1, activation='linear'))

        # Compile the model with Adam optimizer and Mean Squared Error (MSE) loss
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

        return model

    def train(self, X_train, y_train, epochs=100, batch_size=32):
        # Scales target values to be between 0 and 1, then trains the LSTM model with X_train and y_train
        print(f"Training LSTMModel with X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        y_train_scaled = y_train / 10  # Scale target values to the range [0, 1]
        self.model.fit(X_train, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1)

    def evaluate(self, X_test, y_test):
        # Evaluates the LSTM model with X_test and y_test
        y_test_scaled = y_test / 10  # Scale test targets to the range [0, 1]
        loss = self.model.evaluate(X_test, y_test_scaled, verbose=1)
        print(f"Test loss: {loss}")
        return loss

    def predict(self, X):
        # Makes predictions with the LSTM model, then scales predictions to the range [0, 10]
        predictions = self.model.predict(X)
        scaled_predictions = predictions * 10  # Scale back to [0, 10]
        return np.clip(np.round(scaled_predictions), 0, 10)  # Ensure predictions are rounded and clipped to [0, 10]
