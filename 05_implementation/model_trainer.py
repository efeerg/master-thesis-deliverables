import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from models.lstm_model import LSTMModel
from models.rf_model import RFModel

class ModelTrainer:
    def __init__(self, model_type, sequence_length, num_features, hidden_units=None, epochs=None, batch_size=None, learning_rate=None):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None

    def build_model(self):
        if self.model_type == "LSTM":
            self.model = LSTMModel(
                sequence_length=self.sequence_length,
                num_features=self.num_features,
                hidden_units=self.hidden_units,
                learning_rate=self.learning_rate
            )
            print("LSTM model instance created.")
        elif self.model_type == "RandomForest":
            self.model = RFModel()  # Customize parameters if needed
            print("Random Forest model instance created.")

    # Function to calculate regression metrics
    def calculate_regression_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return mse, mae, rmse, r2

    # Function to calculate classification metrics
    def calculate_classification_metrics(self, y_true, y_pred):
        # Clip predictions and true values to be between 0 and 10
        y_pred_clipped = np.round(y_pred)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_clipped)
        precision = precision_score(y_true, y_pred_clipped, average='weighted')
        recall = recall_score(y_true, y_pred_clipped, average='weighted')
        f1 = f1_score(y_true, y_pred_clipped, average='weighted')
        labels = np.arange(0, 11)  # This generates an array of labels [0, 1, 2, ..., 10]
        confusion = confusion_matrix(y_true, y_pred_clipped, labels=labels)

        return accuracy, precision, recall, f1, confusion

    def train(self, X_train, y_train):
        print("="*50)
        print("Starting training process...")
        print("="*50)
        
        for month in range(y_train.shape[1]):
            X_batch = X_train[:, month:month+self.sequence_length, :]
            y_batch = y_train[:, month]
            
            print(f"\nTraining for month {month + 1} using X[:, {month}:{month + self.sequence_length}, :]")
            print(f"X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")
            
            self.model.train(X_batch, y_batch)
        
        print("\nTraining complete.")
        print("="*50)

    def test_single_step_ahead(self, X_test, y_test):
        print("="*50)
        print("Starting testing process with Single-step-ahead Forecasting...")
        print("="*50)

        mse_list = []
        mae_list = []
        rmse_list = []
        r2_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        confusion_matrices = []

        # Only predict the first step
        X_batch = X_test[:, :self.sequence_length, :]
        y_batch = y_test[:, 0]  # Only predict the first step

        print(f"\nTesting using X[:, :{self.sequence_length}, :]")
        print(f"X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")

        y_pred = self.model.predict(X_batch)

        # Calculate regression metrics
        mse, mae, rmse, r2 = self.calculate_regression_metrics(y_batch, y_pred)
        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)

        # Calculate classification metrics
        accuracy, precision, recall, f1, confusion = self.calculate_classification_metrics(y_batch, y_pred)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        confusion_matrices.append(confusion)

        print(f"MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R²: {r2}")
        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

        print("\nTesting complete.")
        print("="*50)

        # Return metrics
        return {
            "mse": mse_list,
            "mae": mae_list,
            "rmse": rmse_list,
            "r2": r2_list,
            "accuracy": accuracy_list,
            "precision": precision_list,
            "recall": recall_list,
            "f1": f1_list,
            "confusion_matrices": confusion_matrices
        }

    def test_multi_step_ahead_iterative(self, X_test, y_test):
        print("="*50)
        print("Starting testing process with Multi-step-ahead Iterative Forecasting...")
        print("="*50)
        
        mse_list = []
        mae_list = []
        rmse_list = []
        r2_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        confusion_matrices = []

        # Start with the initial input sequence from X_test
        input_sequence = X_test[:, :self.sequence_length, :]

        for month in range(y_test.shape[1]):
            print(f"\nTesting for month {month + 1} using input sequence of shape {input_sequence.shape}")
            
            # Predict the next month
            y_pred = self.model.predict(input_sequence)
            
            # Reshape y_pred to ensure it matches the expected dimensions for input_sequence
            if len(y_pred.shape) == 1:
                y_pred = y_pred[:, np.newaxis]  # Make it 2D: (batch_size, 1)
            
            # Expand y_pred to match the number of features in input_sequence (for RandomForest, replicate across features)
            y_pred_expanded = np.repeat(y_pred, self.num_features, axis=1)
            y_pred_expanded = y_pred_expanded[:, np.newaxis, :]  # Add a time dimension

            # Update the input sequence: remove the oldest month and add the predicted month
            input_sequence = np.concatenate([input_sequence[:, 1:, :], y_pred_expanded], axis=1)
            
            # Get the actual target values for comparison
            y_batch = y_test[:, month]
            
            # Calculate regression metrics
            mse, mae, rmse, r2 = self.calculate_regression_metrics(y_batch, y_pred)
            mse_list.append(mse)
            mae_list.append(mae)
            rmse_list.append(rmse)
            r2_list.append(r2)

            # Calculate classification metrics
            accuracy, precision, recall, f1, confusion = self.calculate_classification_metrics(y_batch, y_pred)
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            confusion_matrices.append(confusion)
            
            print(f"Predictions for month {month + 1}: {y_pred.flatten()}")
            print(f"Month {month + 1} - MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R²: {r2}")
            print(f"Month {month + 1} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
        
        print("\nTesting complete.")
        print("="*50)

        # Return metrics for all months
        return {
            "mse": mse_list,
            "mae": mae_list,
            "rmse": rmse_list,
            "r2": r2_list,
            "accuracy": accuracy_list,
            "precision": precision_list,
            "recall": recall_list,
            "f1": f1_list,
            "confusion_matrices": confusion_matrices
        }

    def test_multi_step_ahead_direct(self, X_test, y_test):
        print("="*50)
        print("Starting testing process with Multi-step-ahead Direct Forecasting...")
        print("="*50)
    
        mse_list = []
        mae_list = []
        rmse_list = []
        r2_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        confusion_matrices = []
    
        # Set the expected sequence length (should match training sequence length)
        expected_sequence_length = self.sequence_length
    
        for month in range(y_test.shape[1]):
            # Extract batch for current month
            X_batch = X_test[:, month:month+self.sequence_length, :]
            y_batch = y_test[:, month]
    
            # If the sequence is shorter than expected, pad it
            if X_batch.shape[1] < expected_sequence_length:
                padding_length = expected_sequence_length - X_batch.shape[1]
                X_batch = np.pad(X_batch, ((0, 0), (0, padding_length), (0, 0)), mode='constant')
    
            print(f"\nTesting for month {month + 1} using X[:, {month}:{month + self.sequence_length}, :]")
            print(f"X_batch shape after padding (if needed): {X_batch.shape}, y_batch shape: {y_batch.shape}")
    
            # Make predictions using the model
            y_pred = self.model.predict(X_batch)
    
            # Calculate regression metrics
            mse, mae, rmse, r2 = self.calculate_regression_metrics(y_batch, y_pred)
            mse_list.append(mse)
            mae_list.append(mae)
            rmse_list.append(rmse)
            r2_list.append(r2)
    
            # Calculate classification metrics
            accuracy, precision, recall, f1, confusion = self.calculate_classification_metrics(y_batch, y_pred)
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            confusion_matrices.append(confusion)
    
            print(f"Month {month + 1} - MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R²: {r2}")
            print(f"Month {month + 1} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    
        print("\nTesting complete.")
        print("="*50)
    
        return {
            "mse": mse_list,
            "mae": mae_list,
            "rmse": rmse_list,
            "r2": r2_list,
            "accuracy": accuracy_list,
            "precision": precision_list,
            "recall": recall_list,
            "f1": f1_list,
            "confusion_matrices": confusion_matrices
        }
