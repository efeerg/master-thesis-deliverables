#!/usr/bin/env python
# coding: utf-8

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import math

# ## Supporting Functions
def implement_months(repository):
    if repository.empty:
        return None  # or handle the empty case appropriately
    
    if 'date_month' in repository.columns:
        # Convert 'date_month' column to datetime format
        repository['date_month'] = pd.to_datetime(repository['date_month'])

        # Extract year and month from the 'date_month' column
        repository['year'] = repository['date_month'].dt.year
        repository['month'] = repository['date_month'].dt.month

        repository.drop(columns=['date_month'], inplace=True)
    
    repository = repository.sort_values(by=['year', 'month'], ascending=True)
    repository.reset_index(inplace=True, drop=True)
    repository['month'] = repository['month'].astype(str).str.zfill(2)
    repository['date'] = repository['month'].astype(str) + '-' + repository['year'].astype(str)

    # Create a complete date range from the minimum to maximum dates in the original data
    # max and min values likely to be a fixed value for all repositories
    min_year, min_month = repository['year'].iloc[0], repository['month'].iloc[0]
    max_year, max_month = repository['year'].iloc[-1], repository['month'].iloc[-1]
    min_date = f"{min_year}-{min_month}"
    max_date = f"{max_year}-{max_month}"
    date_range = pd.date_range(start=min_date, end=max_date, freq='MS')

    # Create a DataFrame from the date range
    date_df = pd.DataFrame({'date': date_range})

    # Extract year and month from the date range
    date_df['year'] = date_df['date'].dt.year
    date_df['month'] = date_df['date'].dt.month.astype(str).str.zfill(2)

    # Convert the date column to the same format as in your original DataFrame
    date_df['date'] = date_df['date'].dt.strftime('%m-%Y')

    # Merge the original DataFrame with the date DataFrame to fill in missing values
    repository = pd.merge(date_df, repository, on=['year', 'month', 'date'], how='left')

    repository = repository.fillna(0)

    return repository

def array_to_duration(repository, column):
    repository["duration"] = repository[column].apply(lambda x: x[0] * 30 + x[1] + x[2] / (24 * 3600) + x[3] / (24 * 3600 * 10 ** 9) if isinstance(x, (list, tuple)) and len(x) >= 4 else np.inf)
    repository["duration"] = repository["duration"].replace(np.inf, repository["duration"].median())
    repository.drop(columns=[column], inplace=True)
    return repository

# Function to fill the new dataframe with commit counts
def fill_counts(row, row_index, df, information):
    for entry in row:
        year_month = f"{entry['month']:02d}-{entry['year']}"
        if year_month in df.columns:
            df.at[row_index, year_month] = entry[information]

def extract_comments_and_issues(json_data):
    issue_df = json_data[['issue.createdAt', 'issue.creatorRole', 'comments']].copy()
    issue_df['issue.createdAt'] = issue_df['issue.createdAt'].apply(lambda x: pd.to_datetime(x))
    issue_df['month'] = issue_df['issue.createdAt'].dt.month
    issue_df['year'] = issue_df['issue.createdAt'].dt.year
    issue_df['date'] = issue_df['month'].astype(str).str.zfill(2) + '-' + issue_df['year'].astype(str)
    issue_df = issue_df.rename(columns={'issue.creatorRole': 'creatorRole'})
    issue_df = issue_df.drop(columns=['issue.createdAt'])
    issue_df

    comments_list = []
    for comments in issue_df['comments']:
        comments_list.extend(comments)
    issue_df = issue_df.drop(columns=['comments'])

    if comments_list != []:    
        comments_df = pd.json_normalize(comments_list)
        comments_df['createdAt'] = comments_df['createdAt'].apply(lambda x: pd.to_datetime(x))
        comments_df['month'] = comments_df['createdAt'].dt.month
        comments_df['year'] = comments_df['createdAt'].dt.year
        comments_df = comments_df.drop(columns=['createdAt', 'creator'])

        issue_df = pd.concat([issue_df, comments_df]).reset_index(drop=True)

    # Filtering valid roles
    valid_roles = ['COLLABORATOR', 'MEMBER', 'OWNER']
    issue_df = issue_df[issue_df['creatorRole'].isin(valid_roles)]

    grouped_counts = issue_df.groupby(['month', 'year']).size().reset_index(name='sum')
    # grouped_counts['month'] = grouped_counts['month'].astype('Int64')
    # grouped_counts['year'] = grouped_counts['year'].astype('Int64')
    grouped_counts['sum'] = grouped_counts['sum'].astype('Int64')
    grouped_counts = implement_months(grouped_counts)
    return grouped_counts

def calculate_three_month_score(df):
    score = df.sum(axis=1)
    return score

def check_all_true(df):
    return df.apply(lambda row: row[df.columns[0]] and row[df.columns[1]] and row[df.columns[2]], axis=1)

def data_processing(df, begin_time, end_time):
    # Define the start and end dates (we are getting three months before of the starting date, because each month should consider the activities based on the last 90 days)
    start_year, start_month = begin_time[1], begin_time[0]
    end_year, end_month = end_time[1], end_time[0]

    # Generate the list of months between start and end dates
    months = pd.date_range(start=f"{start_month}-{start_year}", end=f"{end_month}-{end_year}", freq='MS').strftime("%m-%Y").tolist()

    # ## Commits per Month
    commit_per_month = df['get_commits_per_month']

    # Create a new dataframe with months as columns
    commit_per_month_structured = pd.DataFrame(index=commit_per_month.index, columns=months)

    # Apply the function to each row
    for i in range(len(commit_per_month)):
        fill_counts(commit_per_month.iloc[i], i, commit_per_month_structured, 'COUNT(c)')

    commit_per_month_structured.fillna(0, inplace=True)

    # ### Saving the data
    commit_per_month_structured.fillna(0, inplace=True)
    commit_per_month_structured.to_parquet('../01_input/input/metrics/commit_per_month.parquet')


    # ## Average Issue Close Time per Month
    avg_issue_close_time_per_month = df['get_avg_issue_close_time_per_month']

    # Create a new dataframe with months as columns
    avg_issue_close_time_per_month_structured_duration = pd.DataFrame(index=avg_issue_close_time_per_month.index, columns=months)
    avg_issue_close_time_per_month_structured_count = pd.DataFrame(index=avg_issue_close_time_per_month.index, columns=months)

    # Apply the function to each row
    for i in range(len(avg_issue_close_time_per_month)):
        if avg_issue_close_time_per_month.iloc[i] is None:
            continue
        x = pd.json_normalize(avg_issue_close_time_per_month.iloc[i])
        df_entry = array_to_duration(x, 'AVG(open_duration)')
        df_entry = implement_months(df_entry)
        for j in df_entry['date']:
            if j in avg_issue_close_time_per_month_structured_duration.columns:
                avg_issue_close_time_per_month_structured_duration.at[i, j] = df_entry[df_entry['date'] == j]['duration'].values[0]
                avg_issue_close_time_per_month_structured_count.at[i, j] = df_entry[df_entry['date'] == j]['COUNT(open_duration)'].values[0]

    avg_issue_close_time_per_month_structured_duration.replace(0, None, inplace=True)
    avg_issue_close_time_per_month_structured_count.replace(0, None, inplace=True)

    # ### Saving the data
    avg_issue_close_time_per_month_structured_count.fillna(0, inplace=True)
    avg_issue_close_time_per_month_structured_duration.fillna(0, inplace=True)
    avg_issue_close_time_per_month_structured_count.to_parquet('../01_input/input/metrics/avg_issue_close_time_per_month_count.parquet')
    avg_issue_close_time_per_month_structured_duration.to_parquet('../01_input/input/metrics/avg_issue_close_time_per_month_duration.parquet')


    # ## Average PR Close Time Per Month
    avg_pull_request_close_time_per_month = df['get_avg_pull_request_close_time_per_month']
    # Create a new dataframe with months as columns
    avg_pull_request_close_time_per_month_structured = pd.DataFrame(index=avg_pull_request_close_time_per_month.index, columns=months)

    # Apply the function to each row
    for i in range(len(avg_pull_request_close_time_per_month)):
        if avg_pull_request_close_time_per_month.iloc[i] is None:
            continue
        x = pd.json_normalize(avg_pull_request_close_time_per_month.iloc[i])
        df_entry = array_to_duration(x, 'AVG(open_duration)')
        df_entry = implement_months(df_entry)
        for j in df_entry['date']:
            if j in avg_pull_request_close_time_per_month_structured.columns:
                avg_pull_request_close_time_per_month_structured.at[i, j] = df_entry[df_entry['date'] == j]['duration'].values[0]


    avg_pull_request_close_time_per_month_structured.replace(0, None, inplace=True)

    # ### Saving the data
    avg_pull_request_close_time_per_month_structured.fillna(0, inplace=True)
    avg_pull_request_close_time_per_month_structured.to_parquet('../01_input/input/metrics/avg_pull_request_close_time_per_month.parquet')


    # ## New Issue Author Count per Month
    new_issue_author_count_per_month = df['get_new_issue_author_count_per_month']

    # Create a new dataframe with months as columns
    new_issue_author_count_per_month_structured = pd.DataFrame(index=new_issue_author_count_per_month.index, columns=months)

    # Apply the function to each row
    for i in range(len(new_issue_author_count_per_month)):
        if new_issue_author_count_per_month.iloc[i] is None:
            continue
        df_entry = pd.json_normalize(new_issue_author_count_per_month.iloc[i])
        df_entry = implement_months(df_entry)
        for j in df_entry['date']:
            if j in new_issue_author_count_per_month_structured.columns:
                new_issue_author_count_per_month_structured.at[i, j] = df_entry[df_entry['date'] == j]['new_authors_count'].values[0]

    # ### Saving the data
    new_issue_author_count_per_month_structured.fillna(0, inplace=True)
    new_issue_author_count_per_month_structured.to_parquet('../01_input/input/metrics/new_issue_author_count_per_month.parquet')


    # ## New PR Author Count per Month
    new_pull_request_author_count_per_month = df['get_new_pull_request_author_count_per_month']

    # Create a new dataframe with months as columns
    new_pull_request_author_count_per_month_structured = pd.DataFrame(index=new_pull_request_author_count_per_month.index, columns=months)

    # Apply the function to each row
    for i in range(len(new_pull_request_author_count_per_month)):
        if new_pull_request_author_count_per_month.iloc[i] is None:
            continue
        df_entry = pd.json_normalize(new_pull_request_author_count_per_month.iloc[i])
        df_entry = implement_months(df_entry)
        for j in df_entry['date']:
            if j in new_pull_request_author_count_per_month_structured.columns:
                new_pull_request_author_count_per_month_structured.at[i, j] = df_entry[df_entry['date'] == j]['new_authors_count'].values[0]

    # ### Saving the data
    new_pull_request_author_count_per_month_structured.fillna(0, inplace=True)
    new_pull_request_author_count_per_month_structured.to_parquet('../01_input/input/metrics/new_pull_request_author_count_per_month.parquet')


    # ## Average Issue Response Time per Month
    avg_issue_response_time_per_month = df['get_avg_issue_response_time_per_month']

    # Create a new dataframe with months as columns
    avg_issue_response_time_per_month_structured = pd.DataFrame(index=avg_issue_response_time_per_month.index, columns=months)

    # Apply the function to each row
    for i in range(len(avg_issue_response_time_per_month)):
        if avg_issue_response_time_per_month.iloc[i] is None:
            continue
        df_entry = pd.json_normalize(avg_issue_response_time_per_month.iloc[i])
        df_entry = array_to_duration(df_entry, 'avg_response_time')
        df_entry = implement_months(df_entry)
        for j in df_entry['date']:
            if j in avg_issue_response_time_per_month_structured.columns:
                avg_issue_response_time_per_month_structured.at[i, j] = df_entry[df_entry['date'] == j]['duration'].values[0]

    # ### Saving the data
    avg_issue_response_time_per_month_structured.fillna(0, inplace=True)
    avg_issue_response_time_per_month_structured.to_parquet('../01_input/input/metrics/avg_issue_response_time_per_month.parquet')


    # ## Average PR Merge Time per Month
    avg_pull_request_merge_time_per_month = df['get_avg_pull_request_merge_time_per_month']

    # Create a new dataframe with months as columns
    avg_pull_request_merge_time_per_month_structured = pd.DataFrame(index=avg_pull_request_merge_time_per_month.index, columns=months)

    # Apply the function to each row
    for i in range(len(avg_pull_request_merge_time_per_month)):
        if avg_pull_request_merge_time_per_month.iloc[i] is None:
            continue
        df_entry = pd.json_normalize(avg_pull_request_merge_time_per_month.iloc[i])
        df_entry = array_to_duration(df_entry, 'avg_merge_duration')
        df_entry = implement_months(df_entry)
        for j in df_entry['date']:
            if j in avg_pull_request_merge_time_per_month_structured.columns:
                avg_pull_request_merge_time_per_month_structured.at[i, j] = df_entry[df_entry['date'] == j]['duration'].values[0]

    # ### Saving the data
    avg_pull_request_merge_time_per_month_structured.fillna(0, inplace=True)
    avg_pull_request_merge_time_per_month_structured.to_parquet('../01_input/input/metrics/avg_pull_request_merge_time_per_month.parquet')


    # ## Closed Issues per Month
    closed_issues_per_month = df['get_closed_issues_per_month']

    # Create a new dataframe with months as columns
    closed_issues_per_month_opened_issues_structured = pd.DataFrame(index=closed_issues_per_month.index, columns=months)
    closed_issues_per_month_closed_issues_structured = pd.DataFrame(index=closed_issues_per_month.index, columns=months)

    # Apply the function to each row
    for i in range(len(closed_issues_per_month)):
        if closed_issues_per_month.iloc[i] is None:
            continue
        df_entry = pd.json_normalize(closed_issues_per_month.iloc[i])
        df_entry = implement_months(df_entry)
        for j in df_entry['date']:
            if j in closed_issues_per_month_opened_issues_structured.columns:
                closed_issues_per_month_opened_issues_structured.at[i, j] = df_entry[df_entry['date'] == j]['opened_issues'].values[0]
                closed_issues_per_month_closed_issues_structured.at[i, j] = df_entry[df_entry['date'] == j]['closed_issues'].values[0]

    # ### Saving the data
    closed_issues_per_month_closed_issues_structured.fillna(0, inplace=True)
    closed_issues_per_month_closed_issues_structured.to_parquet('../01_input/input/metrics/closed_issues_per_month_closed_issues.parquet')
    closed_issues_per_month_closed_issues_structured.fillna(0, inplace=True)
    closed_issues_per_month_closed_issues_structured.to_parquet('../01_input/input/metrics/closed_issues_per_month_closed_issues.parquet')


    # ## Closed PR per Month
    closed_pull_requests_per_month = df['get_closed_pull_requests_per_month']

    # Create a new dataframe with months as columns
    closed_pull_requests_per_month_open_pull_requests_structured = pd.DataFrame(index=closed_pull_requests_per_month.index, columns=months)
    closed_pull_requests_per_month_closed_pull_requests_structured = pd.DataFrame(index=closed_pull_requests_per_month.index, columns=months)

    # Apply the function to each row
    for i in range(len(closed_pull_requests_per_month)):
        if closed_pull_requests_per_month.iloc[i] is None:
            continue
        df_entry = pd.json_normalize(closed_pull_requests_per_month.iloc[i])
        df_entry = implement_months(df_entry)
        for j in df_entry['date']:
            if j in closed_pull_requests_per_month_open_pull_requests_structured.columns:
                closed_pull_requests_per_month_open_pull_requests_structured.at[i, j] = df_entry[df_entry['date'] == j]['open_pull_requests'].values[0]
                closed_pull_requests_per_month_closed_pull_requests_structured.at[i, j] = df_entry[df_entry['date'] == j]['closed_pull_requests'].values[0]

    # ### Saving the data
    closed_pull_requests_per_month_open_pull_requests_structured.fillna(0, inplace=True)
    closed_pull_requests_per_month_open_pull_requests_structured.to_parquet('../01_input/input/metrics/closed_pull_requests_per_month_open_pull_requests.parquet')
    closed_pull_requests_per_month_closed_pull_requests_structured.fillna(0, inplace=True)
    closed_pull_requests_per_month_closed_pull_requests_structured.to_parquet('../01_input/input/metrics/closed_pull_requests_per_month_closed_pull_requests.parquet')


    # ## Get Project Information
    project_information = pd.json_normalize(df['get_project_information'].apply(lambda x: x[0] if x is not None else None))
    # Convert "archivedAt" and "createdAt" columns to datetime type
    project_information["archivedAt"] = project_information["archivedAt"].apply(lambda x: pd.to_datetime(x) if x != "0001-01-01T01:01:01+00:00" else pd.to_datetime("1970-01-01T00:00:00+00:00"))
    project_information["createdAt"] = pd.to_datetime(project_information["createdAt"])

    # Extract year and month
    project_information["create_year"] = project_information["createdAt"].dt.year.astype('Int64')
    project_information["create_month"] = project_information["createdAt"].dt.month.astype('Int64')
    project_information["archive_year"] = project_information["archivedAt"].dt.year.astype('Int64')
    project_information["archive_month"] = project_information["archivedAt"].dt.month.astype('Int64')
    isArchived = project_information["isArchived"].astype('Int64')
    project_information.drop(columns=['archivedAt', 'createdAt'], inplace=True)

    # Create a new dataframe with months as columns
    project_information_structured = pd.DataFrame(index=project_information.index, columns=months)

    def fill_dataframe(df1, start_year, start_month, end_year, end_month):
        # Generate the months for the second table
        months = pd.date_range(start=f"{start_month}-{start_year}", end=f"{end_month}-{end_year}", freq='MS').strftime("%m-%Y").tolist()

        # Initialize the second dataframe with NaN values
        df2 = pd.DataFrame(None, index=df1.index, columns=months)

        for idx, row in df1.iterrows():
            # Check for NAType or missing values
            if pd.isna(row['create_year']) or pd.isna(row['create_month']) or (row['isArchived'] and (pd.isna(row['archive_year']) or pd.isna(row['archive_month']))):
                print(f"Row {idx} contains missing data. Filling row with False values.")
                df2.loc[idx, months] = False
                continue  # Skip to the next iteration

            # Convert to integers
            create_year = int(row['create_year'])
            create_month = int(row['create_month'])
            create_date = pd.Period(year=create_year, month=create_month, freq='M')

            if row['isArchived']:
                archive_year = int(row['archive_year'])
                archive_month = int(row['archive_month'])
                archive_date = pd.Period(year=archive_year, month=archive_month, freq='M')
            else:
                archive_date = pd.Period(year=end_year, month=end_month, freq='M')

            start_date = pd.Period(year=start_year, month=start_month, freq='M')

            start_fill = max(create_date, start_date)
            end_fill = archive_date

            for month in months:
                period = pd.Period(month, freq='M')
                if start_fill <= period <= end_fill:
                    df2.at[idx, month] = True
                else:
                    df2.at[idx, month] = False

        return df2

    # Apply the function to each row in the original dataframe
    project_information_structured = fill_dataframe(project_information, start_year, start_month, end_year, end_month)
    project_information_structured.to_parquet('../01_input/input/metrics/project_information.parquet')


    # ## Issues
    issues = df["get_issues_and_issue_comments"]
    # Create a new dataframe with months as columns
    issues_structured = pd.DataFrame(index=issues.index, columns=months)

    # # Apply the function to each row
    for i in range(len(issues)):
        if issues[i] is None:
            continue
        inp = pd.json_normalize(issues[i])
        if inp.empty:
            continue
        df_entry = extract_comments_and_issues(inp)
        if df_entry is None:
            continue
        for j in df_entry['date']:
            if j in issues_structured.columns:
                issues_structured.at[i, j] = df_entry[df_entry['date'] == j]['sum'].values[0]

    issues_structured.fillna(0, inplace=True)

    # Save to Parquet
    issues_structured.to_parquet('../01_input/input/metrics/issues.parquet')

    # ## Maintenance Score
    # Generate the list of months between start and end dates
    months = pd.date_range(start=f"{start_month}-{start_year}", end=f"{end_month}-{end_year}", freq='MS').strftime("%m-%Y").tolist()[2:]

    pi_activity_score = pd.DataFrame(index=project_information_structured.index, columns=months)
    commit_activity_score = pd.DataFrame(index=commit_per_month_structured.index, columns=months)
    issue_activity_score = pd.DataFrame(index=issues_structured.index, columns=months)

    for i in range(len(pi_activity_score.columns)):
        pi_activity_score.iloc[:, i] = check_all_true(project_information_structured.iloc[:, i:i+3])
        commit_activity_score.iloc[:, i] = calculate_three_month_score(commit_per_month_structured.iloc[:, i:i+3])
        issue_activity_score.iloc[:, i] = calculate_three_month_score(issues_structured.iloc[:, i:i+3])

    maintained_score = commit_activity_score + issue_activity_score
    t = 4 * 90 / 30
    maintained_score = maintained_score.map(lambda x: min(math.floor(10 * x  / t), 10))
    maintained_score = maintained_score.where(pi_activity_score, 0.0)
    maintained_score = maintained_score.astype(int)
    maintained_score.to_parquet('../01_input/input/metrics/maintenance_score_experiment.parquet')


