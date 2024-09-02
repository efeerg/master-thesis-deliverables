# Implementation Documentation

This document provides an overview of the implementation contained within the `05_implementation` directory. The directory includes the main implementation files necessary for training, testing, and evaluating the machine learning models used in this project.

## Directory Structure

The `05_implementation` directory is organized into several key files and subdirectories:

- **`main.py`**: The entry point for running experiments, which orchestrates data loading, model training, testing, and saving results.
- **`utils/`**: A subdirectory containing utility scripts, including the data loading functionality.
  - **`data_loader.py`**: Script responsible for loading and preprocessing the data.
- **`models/`**: A subdirectory containing the model definitions.
  - **`lstm_model.py`**: Defines the LSTM model architecture and its associated training and prediction methods.
  - **`rf_model.py`**: Defines the Random Forest model and its associated training and prediction methods.
- **`model_trainer.py`**: Manages the training and testing processes for the different models and forecasting methods.

## Key Files and Their Purpose

### `main.py`

The `main.py` script is the central file used to execute experiments. It loads the experiment configuration, prepares the data, builds the specified model, trains it, and evaluates its performance.

**Main Functions:**
- `load_config(config_path)`: Loads experiment settings from a JSON configuration file.
- `save_results(output_folder, file_name, results)`: Saves the experiment results to a JSON file.
- `main(config_path)`: Executes the overall process, including data loading, model training, and evaluation.

### `utils/data_loader.py`

This script provides the `DataLoader` class, which is responsible for loading and organizing the data into appropriate formats for training and testing.

**Main Functions:**
- `__init__(input_folder, sequence_length, n)`: Initializes the data loader with the specified input folder, sequence length, and the number of months for splitting the data into training and testing sets.
- `load_data()`: Loads the dataset and splits it into training and testing sets, returning the prepared data.

### `model_trainer.py`

The `ModelTrainer` class in this script handles the lifecycle of model training and evaluation. It supports various models and forecasting methods.

**Main Functions:**
- `__init__(model_type, sequence_length, num_features, hidden_units=None, epochs=None, batch_size=None, learning_rate=None)`: Initializes the trainer with parameters for the model and training process.
- `build_model()`: Constructs the model specified by the configuration (either LSTM or Random Forest).
- `train(X_train, y_train)`: Trains the model on the provided training data.
- `test_single_step_ahead(X_test, y_test)`: Performs single-step-ahead forecasting and evaluation.
- `test_multi_step_ahead_iterative(X_test, y_test)`: Performs multi-step-ahead iterative forecasting and evaluation.
- `test_multi_step_ahead_direct(X_test, y_test)`: Performs multi-step-ahead direct forecasting and evaluation.

### `models/lstm_model.py`

This script contains the `LSTMModel` class, which defines the architecture and methods for an LSTM-based model.

**Main Functions:**
- `__init__(sequence_length, num_features, hidden_units, learning_rate)`: Initializes the LSTM model with the specified architecture and training parameters.
- `build_model()`: Constructs the LSTM network architecture.
- `train(X_train, y_train, epochs, batch_size)`: Trains the LSTM model on the provided data.
- `predict(X)`: Makes predictions using the trained LSTM model.

### `models/rf_model.py`

This script defines the `RFModel` class, implementing the Random Forest model used for forecasting.

**Main Functions:**
- `__init__(n_estimators, max_depth, random_state)`: Initializes the Random Forest model with specified hyperparameters.
- `train(X_train, y_train)`: Trains the Random Forest model.
- `predict(X)`: Makes predictions using the trained Random Forest model.

## Running an Experiment

To execute an experiment, use the `main.py` script with the appropriate configuration file:

```bash
python main.py --config path/to/config.json
```

Replace `path/to/config.json` with the path to your specific configuration file (e.g., `lstm_config.json` or `rf_config.json`).

## Extending the Implementation

The `05_implementation` directory is designed to be modular and easily extendable. You can add new models, forecasting methods, or data preprocessing steps by following the structure outlined in this document.

---

This `README.md` provides a comprehensive guide to the contents of the `05_implementation` directory, enabling users to understand and extend the functionality of the project.
