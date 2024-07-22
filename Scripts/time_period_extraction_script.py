# import packages
import pandas as pd
import auxiliary

# Read merged dataset
df = pd.read_csv('merged_dataset.csv')
print("Number of instances in the dataset: ", len(df))
print("Unique UE ids in the datasets: ", df['imeisv_ue'].unique())

# Print the NaN counts for each column
nan_counts = df.isnull().sum()
for column, count in nan_counts.items():
    print(f"Column '{column}': {count} NaN values")

# Merge bearer uplink and downlink bytes
df['total_dl_bytes_ue'] = df['bearer_0_dl_total_bytes_ue'] + df['bearer_1_dl_total_bytes_ue']
df['total_ul_bytes_ue'] = df['bearer_0_ul_total_bytes_ue'] + df['bearer_1_ul_total_bytes_ue']
remove_cols = ['bearer_0_dl_total_bytes_ue', 'bearer_1_dl_total_bytes_ue', 'bearer_0_ul_total_bytes_ue',
               'bearer_1_ul_total_bytes_ue']
df = df.drop(columns=remove_cols)

# Convert timestamp to date time
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort and reset index before operations
df.sort_values(by=['imeisv_ue', 'timestamp'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Find period boundaries
pb = auxiliary.find_period_boundaries_with_first_record_start(df)

gap_value = pd.Timedelta(seconds=5)
id_column_name = 'imeisv_ue'
time_column_name = 'timestamp'

# Append time related features for sequence extraction
chunk_size = 1  # Define the desired number of instances per chunk
statistics_df = auxiliary.calculate_fixed_window_statistics(df, pb, chunk_size)


# Define attack periods
attack_periods = [
    (pd.to_datetime('2024-01-24 14:48:30', utc=True), pd.to_datetime('2024-01-24 14:58:30', utc=True)),
    (pd.to_datetime('2024-01-25 14:05:00', utc=True), pd.to_datetime('2024-01-25 14:10:00', utc=True))
]

# Define the benign user's IMEISV
benign_imeisv = 8609960480666910


def label_record(row, attack_periods, benign_imeisv):
    # Convert string to datetime for comparison, if not already done
    row_timestamp = pd.to_datetime(row['chunk_start'], utc=True)
    
    # Always label the benign user as 0
    if row['imeisv_ue'] == benign_imeisv:
        return 0
    
    # Check if the record falls within any of the attack periods
    for start, end in attack_periods:
        if start <= row_timestamp <= end:
            return 1  # Record falls within an attack period
    
    # Default label for records outside attack periods
    return 0


# Example usage (assuming 'statistics_df' is your DataFrame)
statistics_df['label'] = statistics_df.apply(label_record, axis=1, args=(attack_periods, benign_imeisv))
statistics_df = statistics_df.drop(columns=['timestamp'])
print(len(statistics_df))
statistics_df.to_csv('merged_dataset_final.csv', index=False)
