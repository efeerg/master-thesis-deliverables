from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

class RFModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    def train(self, X_train, y_train):
        """
        Train the Random Forest model. Reshape data to fit scikit-learn's requirements.
        """
        print("\nStarting Random Forest model training...")
        # Reshape X_train from (num_samples, sequence_length, num_features) to (num_samples, sequence_length * num_features)
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train_reshaped, y_train)
        print("\nRandom Forest model training complete.")

    def evaluate(self, X_test, y_test):
        """
        Evaluate the Random Forest model on the test data.
        """
        print("\nEvaluating Random Forest model...")
        # Reshape X_test from (num_samples, sequence_length, num_features) to (num_samples, sequence_length * num_features)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        predictions = self.model.predict(X_test_reshaped)
        rounded_predictions = np.round(predictions)  # Round predictions to the nearest integer
        mse = mean_squared_error(y_test, rounded_predictions)
        print(f"Test MSE: {mse}")
        return mse

    def predict(self, X):
        """
        Make predictions using the trained Random Forest model.
        """
        print("\nMaking predictions with Random Forest model...")
        # Reshape X from (num_samples, sequence_length, num_features) to (num_samples, sequence_length * num_features)
        X_reshaped = X.reshape(X.shape[0], -1)
        predictions = self.model.predict(X_reshaped)
        rounded_predictions = np.round(predictions)  # Round predictions to the nearest integer
        return rounded_predictions

# Example usage:
# rf = RFModel(n_estimators=100, max_depth=10)
# rf.train(X_train, y_train)
# predictions = rf.predict(X_test)
