import json
import os
import numpy as np
import argparse
from utils.data_loader import DataLoader
from model_trainer import ModelTrainer

def load_config(config_path):
    config = json.loads(config_path)
    return config

def save_results(output_folder, file_name, results):
    # Convert numpy arrays to lists for JSON serialization
    for key in results:
        if isinstance(results[key], np.ndarray):
            results[key] = results[key].tolist()
        elif isinstance(results[key], list):
            results[key] = [item.tolist() if isinstance(item, np.ndarray) else item for item in results[key]]
    
    output_path = os.path.join(output_folder, file_name)
    with open(output_path, 'w') as outfile:
        json.dump(results, outfile, indent=4)
    print(f"Results saved to {output_path}")

def main(config_path):
    print("="*50)
    print("EXPERIMENT START")
    print("="*50)
    
    # Load the configuration from the JSON file
    config = load_config(config_path)
    
    print("\nConfiguration Loaded:")
    print("="*50)
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Initialize DataLoader with the input folder path, sequence length, and n
    loader = DataLoader(
        input_folder=config['input_folder'],
        sequence_length=config['sequence_length'],
        included_metrics=config['included_metrics'],
        n=config.get('n', 9)
    )
    
    # Load the data into training and testing sets
    X_train, X_test, y_train, y_test = loader.load_data()
    
    # Initialize the model trainer
    trainer = ModelTrainer(
        model_type=config['model'],
        sequence_length=config['sequence_length'],
        num_features=X_train.shape[2],
        hidden_units=config.get('hidden_units'),  # Not needed for RandomForest
        epochs=config.get('epochs'),  # Not needed for RandomForest
        batch_size=config.get('batch_size'),  # Not needed for RandomForest
        learning_rate=config.get('learning_rate')  # Not needed for RandomForest
    )
    
    # Build the model
    trainer.build_model()
    
    # Train the model
    trainer.train(X_train, y_train)
    
    # Select the forecasting method
    forecasting_type = config['forecasting_type']

    if forecasting_type == 'single-step-ahead':
        metrics = trainer.test_single_step_ahead(X_test, y_test)
    elif forecasting_type == 'multi-step-ahead-iterative':
        metrics = trainer.test_multi_step_ahead_iterative(X_test, y_test)
    elif forecasting_type == 'multi-step-ahead-direct':
        metrics = trainer.test_multi_step_ahead_direct(X_test, y_test)
    else:
        raise ValueError(f"Unsupported forecasting type: {forecasting_type}")
    
    # Create a standardized file name for the output based on the model type
    if config['model'] == 'LSTM':
        file_name = f"LSTM_{forecasting_type}_seq{config['sequence_length']}_n{config['n']}_ep{config['epochs']}_bs{config['batch_size']}_lr{config['learning_rate']}_hu{config['hidden_units']}_st{config['sample_type']}.json"
    elif config['model'] == 'RandomForest':
        file_name = f"RandomForest_{forecasting_type}_seq{config['sequence_length']}_n{config['n']}_ne{config['n_estimators']}_md{config['max_depth']}_rs{config['random_state']}_st{config['sample_type']}.json"
    
    # Output the metrics
    print("\nFinal Metrics:")
    print("="*50)
    for key, value in metrics.items():
        if key == 'confusion_matrices' or key == 'predictions':
            continue    
        print(f"{key}: {value}")
    
    # Save the results as a JSON file
    save_results(config['output_folder'], file_name, metrics)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment based on JSON configuration.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()
    
    # Run the main function with the provided configuration file path
    main(args.config)
