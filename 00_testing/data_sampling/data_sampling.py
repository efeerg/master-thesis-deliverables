#!/usr/bin/env python
# coding: utf-8

import pandas as pd


def sample_percentage(df, perc):
    # Take a random amount sample of the DataFrame
    sampled_df = df.sample(frac=perc/100, random_state=42)
    return sampled_df


def sample_balanced(df, size):
    # Take Y samples for each unique value in the maintenance_score column
    sampled_df = df.groupby('maintenance_score').apply(lambda x: x.sample(min(len(x), size), random_state=42)).reset_index(drop=True)
    sampled_df


def sample_hybrid(df, perc_dict = {
        0: 0.35,   # 35% for maintenance_score = 0
        1: 0.05,   # 5% for maintenance_score = 1
        2: 0.05,   # 5% for maintenance_score = 2
        3: 0.05,   # 5% for maintenance_score = 3
        4: 0.05,   # 5% for maintenance_score = 4
        5: 0.05,   # 5% for maintenance_score = 5
        6: 0.05,   # 5% for maintenance_score = 6
        7: 0.05,   # 5% for maintenance_score = 7
        8: 0.05,   # 5% for maintenance_score = 8
        9: 0.05,   # 5% for maintenance_score = 9
        10: 0.20   # 20% for maintenance_score = 10
    }, size=20):

    # Calculate the number of samples to take for each maintenance_score value
    sample_sizes = {
        key: int(size * perc_dict[key]) for key in perc_dict
    }

    # Sample the DataFrame based on the calculated sample sizes
    sampled_dfs = []
    for score, size in sample_sizes.items():
        sampled_df = df[df['maintenance_score'] == score].sample(
            n=min(size, len(df[df['maintenance_score'] == score])), 
            random_state=42
        )
        sampled_dfs.append(sampled_df)

    # Combine the sampled DataFrames
    final_sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)

    # Display the final sampled DataFrame
    return final_sampled_df


def data_sampling(df, sample_type, sample_perc, sample_size, sample_dict):
    if sample_type == 'unbalanced':
        df = sample_percentage(df, sample_perc)
    elif sample_type == 'balanced':
        df = sample_balanced(df, sample_size)
    else:
        df = sample_hybrid(df, sample_perc, sample_dict)
    return df




