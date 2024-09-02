import subprocess
import re
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from urllib.parse import urlparse

def find_x_over_10(text):
    pattern = r'\b(\d{1,2}) / 10\b'
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    else:
        return None

def execute_cli_command(command):
    try:
        reslt = 0
        reslt_exp = ""
        result = subprocess.check_output(command, shell=True, text=True)
        result = json.loads(result)
        result = result['checks'][0]
        reslt = result['score']
        reslt_exp = result['reason']
        return reslt, reslt_exp
    except Exception as e:
        print(f"Error executing CLI command: {e}")
        return None, None

# Load the JSON file into a pandas dataframe
df = pd.read_json('Data/Raw/pypi_metrics_file.json')
data_df = df.transpose()

data_df.reset_index(inplace=True)
data_df.rename(columns={'index': 'github_link'}, inplace=True)
data_df = data_df.reindex(columns=['project_name', 'github_link', 'project_url', 'project_id', 'metric_results'])

# Extract parameters from metric_results column
df = pd.json_normalize(data_df['metric_results'])

# Merge the two dataframes
data_df = pd.concat([data_df, df], axis=1)

# Assuming your DataFrame is named data_df
data_df = data_df.applymap(lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x)

new_df = data_df[['github_link', 'project_name', 'project_url']].copy()
new_df['link'] = data_df['github_link'].apply(lambda x: urlparse(x).path)
new_df['maintenance_score'] = None
new_df['explanation'] = None

for index, row in new_df.iterrows():
    # if index > 1:
    #     break
    # Print a fancy box displaying the index
    print("+" + "-" * (len(str(index)) + 2) + "+")
    print("| " + str(index) + " |")
    print("+" + "-" * (len(str(index)) + 2) + "+")    
    # Execute the CLI command and save the output to a string
    command = "scorecard --repo=github.com{} --checks Maintained --format json".format(row['link'])
    score, explanation = execute_cli_command(command)
    new_df.at[index, 'maintenance_score'] = score
    new_df.at[index, 'explanation'] = explanation
    print("result = ", score)
    print("explanation = ", explanation)
#     if output_string is None or len(output_string.splitlines()) == 0:
#         new_df.at[index, 'maintenance_score'] = 0
#         continue
#     for line in output_string.splitlines():
#         if "| Maintained " in line:
#             new_df.at[index, 'explanation'] = line.split("|")[3].strip()
#             matches = find_x_over_10(line)
#             new_df.at[index, 'maintenance_score'] = matches
#             break

new_df.to_parquet('Data/Processed/maintenance_score.parquet')