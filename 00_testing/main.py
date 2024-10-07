from maintenance_score.maintenance_score_pipeline import maintenance_score_calculation
from data_sampling.data_sampling import data_sampling
from data_processing.data_processing import data_processing

import argparse
import json
import os
import subprocess
import pandas as pd


def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def main(config_path):
    print("="*50)
    print("EXPERIMENT START")
    print("="*50)
    
    # Load the configuration from the JSON file
    config = load_config(config_path)
    
    print("Configuration Loaded:")
    print("-"*50)
    for key, value in config.items():
        print(f"{key}: {value}")


    begin_time = (config['begin_time']['year'], config['begin_time']['month'])
    end_time = (config['end_time']['year'], config['end_time']['month'])
    json_file_path = config['maintained_pipeline']['json_file']

    # MAINTENANCE SCORE PIPELINE
    if config['execute_maintained']:
        print("="*50)
        print("Maintenance Score Calculation")
        print("Parameters for the maintenance score calculation:")
        print(f"Begin time: {begin_time}")
        print(f"End time: {end_time}")
        print("JSON file path: ", json_file_path)
        print("-"*50)
        
        # Dataframe with the maintenance score
        df = maintenance_score_calculation(json_file_path, begin_time, end_time, config['maintained_pipeline']['selected_month'])
        df.to_parquet(config['ds_pipeline']['parquet_file'])
        print("Dataframe shape (After Maintenance score calculation): {}".format(df.shape))
        print(df['maintenance_score'].value_counts())
    
    # DATA PREPARATION PIPELINE
    if config['execute_ds']:
        print("="*50)
        print("Sampling Data")
        print("Parameters for data sampling:")
        print("Sample type: ", config['ds_pipeline']['sample_type'])
        print("Sample percentage: {}".format(config['ds_pipeline']['sample_perc'] if config['ds_pipeline']['sample_type'] == "unbalanced" else "None"))
        print("Sample size: {}".format(config['ds_pipeline']['sample_size'] if config['ds_pipeline']['sample_type'] == "balanced" else "None"))
        print("Sample dictionary: {}".format(config['ds_pipeline']['sample_dict'] if config['ds_pipeline']['sample_type'] == "hybrid" else "None"))
        print("-"*50)

        prep_df = pd.read_parquet(config['ds_pipeline']['parquet_file'])
        print("Dataframe shape (Before sampling): {}".format(prep_df.shape))
        print(prep_df['maintenance_score'].value_counts())
    
        # Sampling types
        if config['ds_pipeline']['sample_type'] == 'unbalanced':
            df = data_sampling(prep_df, config['ds_pipeline']['sample_type'], 
                        sample_perc=config['ds_pipeline']['sample_perc'],
                        sample_size=None,
                        sample_dict=None)
        elif config['ds_pipeline']['sample_type'] == 'balanced':
            df = data_sampling(prep_df, config['ds_pipeline']['sample_type'],
                        sample_perc=None,
                        sample_size=config['ds_pipeline']['sample_size'],
                        sample_dict=None)
        else:
            df = data_sampling(prep_df, config['ds_pipeline']['sample_type'],
                        sample_perc=None,
                        sample_size=config['ds_pipeline']['sample_size'],
                        sample_dict=config['ds_pipeline']['sample_dict'])
    
        print("Dataframe shape (After sampling): {}".format(df.shape))
        print(df['maintenance_score'].value_counts())
        # df.to_parquet(config['ds_pipeline']['parquet_file'])

        print("-"*50)
        print("Data Processing")
        print("-"*50)
        data_processing(df, begin_time, end_time)
    
    # TESTING PIPELINE
    if config['execute_test']:
        print("="*50)
        print("Model Trainer")
        print("-"*50)
        config['test_pipeline'].update({'sample_type': config['ds_pipeline']['sample_type']})
        print(f"Updated configuration for the model trainer:{config['test_pipeline']}")
        command = ['python', '../05_implementation/main.py', '--config', json.dumps(config['test_pipeline'])]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print("Error:", e.stderr)

    print("="*50)
    print("EXPERIMENT COMPLETED")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment based on JSON configuration.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()
    
    # Run the main function with the provided configuration file path
    main(args.config)