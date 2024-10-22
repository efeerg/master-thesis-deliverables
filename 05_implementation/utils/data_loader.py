import numpy as np
import pandas as pd
import os

class DataLoader:
    def __init__(self, input_folder, sequence_length, included_metrics, n):
        self.input_folder = input_folder
        self.sequence_length = sequence_length
        self.included_metrics = included_metrics
        self.n = n

    def load_data(self):
        """
        Load all Parquet files from the input folder (excluding 'maintenance_score_experiment.parquet')
        and combine them into a 3D NumPy array (X). Also load the target variable (y) from
        'maintenance_score_experiment.parquet'. Split the data into training and testing sets.
        """
        print("="*50)
        print("STEP 1: Loading Feature Data (Excluding Target Variable)")
        print("="*50)
        
        data_list = []
        target_df = None
        
        for filename in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, filename)
            if filename.endswith(".parquet"):
                if filename == "maintenance_score_experiment.parquet":
                    print("\nFound target variable file: maintenance_score_experiment.parquet")
                elif filename.replace(".parquet", "") in self.included_metrics:
                    # Load feature data
                    df = pd.read_parquet(file_path)
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaNs
                    df.fillna(0, inplace=True)  # Replace NaNs with 0 or any other strategy you prefer
                    data_list.append(df.values)
                    print(f"Loaded {filename}: Shape {df.shape}")
                else:
                    print(f"\nSkipping {filename} as it's not in the included metrics.")

        print("\nFinished loading feature data.")
        print("="*50)
        
        print("\nSTEP 2: Loading Target Variable")
        print("="*50)

        # Load target variable data
        target_df = pd.read_parquet(os.path.join(self.input_folder, "maintenance_score_experiment.parquet"))
        target_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaNs
        target_df.fillna(0, inplace=True)  # Replace NaNs with 0 or any other strategy you prefer
        y = target_df.values
        print(f"Loaded target variable: Shape {y.shape}")
        
        print("\nFinished loading target variable.")
        print("="*50)

        print("\nSTEP 3: Stacking Feature Data into 3D Array")
        print("="*50)

        # Stack the list of 2D arrays into a single 3D array
        X = np.stack(data_list, axis=-1)
        print(f"Feature data stacked into 3D array: Shape {X.shape}")
        
        print("\nSTEP 4: Splitting Data into Training and Testing Sets")
        print("="*50)

        # Determine the training and testing ranges
        X_train = X[:, :self.sequence_length + (self.n - 1), :]
        X_test = X[:, -(self.sequence_length + 2):, :]

        y_train = y[:, :self.n]
        y_test = y[:, -self.sequence_length:]

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        print("\nData splitting complete.")
        print("="*50)
        
        return X_train, X_test, y_train, y_test

# Example usage:
# loader = DataLoader(input_folder="../../01_input/input/metrics/", sequence_length=3, n=9)
# X_train, X_test, y_train, y_test = loader.load_data()
