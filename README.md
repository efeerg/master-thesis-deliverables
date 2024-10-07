# Master Thesis: GitHub Library Maintenance Prediction

This repository contains the code deliverables for the Master Thesis entitled **"GitHub Library Maintenance Prediction: A Comparison of Machine Learning and Deep Learning Methods"**. The folders are structured matching with the chapters in the thesis, each folder corresponding to the respective chapters and components of the study. All scripts and data required to reproduce the results and experiments described in the thesis are provided.

## Repository Structure

- `00_testing/`: This folder contains the scripts and configurations for running the pipelines. It serves as the testing environment for all implemented models.
- `01_input/`: This directory holds the input datasets, including JSON and Parquet files, which are utilized during the execution of the experiment flow.
- `02_methodology/`: This folder includes Jupyter notebooks and other relevant files that document the methodology used in the literature review and feature extraction.
- `03_lit_review/`: This directory contains the results of the literature review, stored in Excel format. These files provide a summary of the relevant research articles and their key findings, which informed the choice of models and features in the thesis.
- `04_experiment/`: This folder contains the core experimental setup for training and testing the machine learning and deep learning models. It includes scripts for data sampling, training configuration, and model evaluation.
- `05_implementation/`: This directory includes the implementation of the Random Forest and LSTM models, as well as utility scripts for loading data and preparing the environment for training.
- `06_results/`: This folder contains the outputs generated by the models, including evaluation metrics, confusion matrices, and plots that illustrate the comparative performance of the models.

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repository-link.git
   ```

2. **Install dependencies**:
   The required Python packages can be installed using the following command:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Configuration**: Ensure the configuration files located in `00_testing/config/` are correctly set:
   - `complete_lstm.json`: Configuration for the LSTM model.
   - `complete_rf.json`: Configuration for the Random Forest model.

2. **Execution**: To run the project and execute the model training and evaluation process, enter in `00_testing/` and run the following command:
   ```bash
   python main.py --config config/complete_lstm.json
   python main.py --config config/complete_rf.json
   ```

3. **Testing**: We have three different sampling types for `sample_type`: `unbalanced`, `balanced`, and `hybrid`. In addition to that, we have three different types for `forecasting_type`: `single-step-ahead`, `multi-step-ahead-iterative`, and `multi-step-ahead-direct`. These configurations can be tried out by writing them to the configurations file to perform different tests.

4. **Output**: Results, including metrics and visualizations, will be saved in the `06_results/` directory.

## Methodology

### Data Collection
The data is sourced from GitHub repositories using their API, focusing on commit history, issue activities, and pull requests. The data is stored in a Neo4j graph database and transformed into JSON format for further processing.

### Modeling
- **Random Forest (RF)**: A machine learning model that uses an ensemble of decision trees.
- **Long Short-Term Memory (LSTM)**: A deep learning model optimized for time-series data. 

Both models aim to predict the "maintenance score," a metric representing the ongoing maintenance activity of a repository, based on the OpenSSF Scorecard.

### Evaluation
The models are evaluated using both regression metrics (MSE, RMSE) and classification metrics (accuracy, precision, recall, confusion matrix). The results from different sampling methods and configurations are compared and analyzed.

## Results

The results, saved in the `06_results/` folder, include:
- **JSON files** storing the evaluation metrics for each experiment.
- **Evaluation plots** depicting the performance of each model across different configurations.
