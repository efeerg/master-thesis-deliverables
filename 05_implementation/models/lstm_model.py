import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMModel:
    def __init__(self, sequence_length, num_features, hidden_units, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        """
        Build and compile the LSTM model.
        """
        model = Sequential()
        model.add(LSTM(units=self.hidden_units, input_shape=(self.sequence_length, self.num_features)))
        model.add(Dense(1))  # Single output for each timestep
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        print("LSTM model built with the following configuration:")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Number of features: {self.num_features}")
        print(f"Hidden units: {self.hidden_units}")
        print(f"Learning rate: {self.learning_rate}")
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Train the LSTM model.
        """
        print("\nStarting LSTM model training...")
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        print("\nLSTM model training complete.")

    def evaluate(self, X_test, y_test):
        """
        Evaluate the LSTM model on the test data.
        """
        print("\nEvaluating LSTM model...")
        loss = self.model.evaluate(X_test, y_test, verbose=1)
        print(f"Test loss: {loss}")
        return loss

    def predict(self, X):
        """
        Make predictions using the trained LSTM model.
        """
        print("\nMaking predictions with LSTM model...")
        predictions = self.model.predict(X)
        rounded_predictions = tf.round(predictions)  # Round predictions to the nearest integer
        return rounded_predictions.numpy()  # Convert to NumPy array

# Example usage:
# lstm = LSTMModel(sequence_length=3, num_features=10, hidden_units=64)
# lstm.train(X_train, y_train, epochs=100, batch_size=32)
# predictions = lstm.predict(X_test)
