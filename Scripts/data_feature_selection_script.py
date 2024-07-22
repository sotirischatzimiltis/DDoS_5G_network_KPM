import pandas as pd

"""Load Cell based dataset"""
df_cell = pd.read_csv('../Datasets/enb_counters.csv')

# Remove columns with singular values
singular_columns = df_cell.columns[df_cell.nunique() == 1]
df_cell = df_cell.drop(columns=singular_columns)

# Remove instance id since we have cell_id to combine datasets
drop_col = 'instance_id'
df_cell = df_cell.drop(columns=drop_col)

# Convert timestamp feature from object to datetime
df_cell['timestamp'] = pd.to_datetime(df_cell['timestamp'])
df_cell.head()

# Remove rows where only the first two features have values
df_cell = df_cell[df_cell.iloc[:, 1:].notna().any(axis=1)].reset_index(drop=True)
print(len(df_cell))

# Analysis of cell features for removal
columns = df_cell.columns
for col in columns:
    print(col)

# Print how many Nan values exist in each column
nan_counts = df_cell.isnull().sum()
for column, count in nan_counts.items():
    print(f"Column '{column}': {count} NaN values")

features_to_remove = ['cell_X_dl_err','cell_X_dl_use_avg','cell_X_dl_use_max',
                      'cell_X_drb_count_avg','cell_X_drb_count_max',
                      'cell_X_drb_count_min', 'cell_X_ul_err',
                      'cell_X_ul_use_avg', 'cell_X_ul_use_max', 'msg_ng_paging',
                      'msg_ng_path_switch_request','msg_ng_path_switch_request_acknowledge',
                      'msg_ng_pdu_session_resource_notify','msg_xn_handover_request_acknowledge_recv',
                      'msg_xn_handover_request_acknowledge_sent', 'msg_xn_handover_request_recv',
                      'msg_xn_handover_request_sent','msg_xn_ng_ran_node_configuration_update_acknowledge_sent',
                      'msg_xn_ng_ran_node_configuration_update_recv', 'msg_xn_sn_status_transfer_recv',
                      'msg_xn_sn_status_transfer_sent', 'msg_xn_ue_context_release_recv',
                      'msg_xn_ue_context_release_sent', 'rf_samples_tx2_count', 'rf_samples_tx2_max',
                      'rf_samples_tx2_rms', 'cell_X_erab_count_avg','cell_X_erab_count_max',
                      'cell_X_erab_count_min', 'msg_ng_error_indication' ]
df_cleaned_cell = df_cell.drop(columns=features_to_remove)
df_cleaned_cell.head()


"""Load UE dataset"""
# Read ue dataset and remove features
df_ue = pd.read_csv('../Datasets/amari_ue_data.csv')
features_to_remove = ['5g_tmsi', 'amf_ue_id', 'bearer_0_apn', 'bearer_0_ip',
                      'bearer_0_ipv6', 'bearer_0_pdu_session_id',
                      'bearer_0_qos_flow_id', 'bearer_0_sst', 'bearer_1_apn',
                      'bearer_1_ip', 'bearer_1_ipv6', 'bearer_1_pdu_session_id',
                      'bearer_1_qos_flow_id', 'bearer_1_sst', 'cell_X_cell_id',
                      'ran_plmn', 'registered', 'rnti', 't3512', 'tac',
                      'tac_plmn', 'ue_aggregate_max_bitrate_dl',
                      'ue_aggregate_max_bitrate_ul', 'dl_bitrate','ul_bitrate',
                      'ran_id', 'ran_ue_id', 'cell_X_ri']
df_ue = df_ue.drop(columns=features_to_remove)
df_ue['timestamp'] = pd.to_datetime(df_ue['timestamp'])
df_ue = df_ue.sort_values(by=['imeisv', 'timestamp'])
df_ue.head()

# print how many Nan values exist in each column
nan_counts = df_ue.isnull().sum()
for column, count in nan_counts.items():
    print(f"Column '{column}': {count} NaN values")

# Merge the DataFrames based on cell_id and timestamp
df_ue_suffixed = df_ue.add_suffix('_ue')
df_cleaned_cell_suffixed = df_cleaned_cell.add_suffix('_cell')

# Remove the suffix from the merge key columns
df_ue_suffixed.rename(columns={'cell_id_ue': 'cell_id', 'timestamp_ue': 'timestamp'}, inplace=True)
df_cleaned_cell_suffixed.rename(columns={'cell_id_cell': 'cell_id', 'timestamp_cell': 'timestamp'}, inplace=True)

# Merge the DataFrames on 'cell_id' and 'timestamp'
merged_df = pd.merge(df_ue_suffixed, df_cleaned_cell_suffixed, on=['cell_id', 'timestamp'], how='inner')

# Check the merged DataFrame
print("\nMerged DataFrame:")
print(merged_df)
print(merged_df.columns)

# Print how many Nan values exist in each column
nan_counts = merged_df.isnull().sum()
for column, count in nan_counts.items():
    print(f"Column '{column}': {count} NaN values")

# Drop NaN values of merged df
merged_df = merged_df.dropna()

# Save merged df
merged_df.to_csv('merged_dataset.csv', index=False)
