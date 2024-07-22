"""Import Packages"""
import os
import random
import statistics

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import auxiliary
from auxiliary import form_sequences_dynamic

'''LOAD DATASET'''
print("Loading Dataset")
df = pd.read_csv('merged_dataset_final.csv')
print(df['imeisv_ue'].unique())
# imeisv_to_drop = ['8609960468879056', '8628490443809956']
# df = df[df['imeisv_ue'] != 8609960468879056]
# df = df[df['imeisv_ue'] != 8628490443809956]
print(df['imeisv_ue'].unique())


# feature_columns = ['total_dl_bytes_ue', 'total_ul_bytes_ue', 'cell_X_cqi_ue', 'cell_X_dl_bitrate_ue',
#                    'cell_X_dl_mcs_ue', 'cell_X_dl_retx_ue', 'cell_X_dl_tx_ue', 'cell_X_epre_ue', 'cell_X_initial_ta_ue',
#                    'cell_X_p_ue_ue', 'cell_X_pusch_snr_ue', 'cell_X_turbo_decoder_avg_ue',
#                    'cell_X_turbo_decoder_max_ue', 'cell_X_turbo_decoder_min_ue', 'cell_X_ul_bitrate_ue',
#                    'cell_X_ul_mcs_ue', 'cell_X_ul_path_loss_ue', 'cell_X_ul_phr_ue', 'cell_X_ul_retx_ue',
#                    'cell_X_ul_tx_ue', 'cell_X_dl_bitrate_cell', 'cell_X_dl_retx_cell',
#                    'cell_X_dl_sched_users_avg_cell', 'cell_X_dl_sched_users_max_cell', 'cell_X_dl_tx_cell',
#                    'cell_X_ue_active_count_avg_cell', 'cell_X_ue_active_count_max_cell',
#                    'cell_X_ue_active_count_min_cell', 'cell_X_ue_count_avg_cell', 'cell_X_ue_count_max_cell',
#                    'cell_X_ue_count_min_cell', 'cell_X_ul_bitrate_cell', 'cell_X_ul_retx_cell',
#                    'cell_X_ul_sched_users_avg_cell', 'cell_X_ul_sched_users_max_cell', 'cell_X_ul_tx_cell',
#                    'cpu_cell', 'duration_cell', 'msg_ng_downlink_nas_transport_cell', 'rf_rx_count_cell',
#                    'rf_rx_cpu_time_cell', 'rf_rx_sample_rate_cell', 'rf_rxtx_delay_avg_cell', 'rf_rxtx_delay_max_cell',
#                    'rf_rxtx_delay_min_cell', 'rf_rxtx_delay_sd_cell', 'rf_samples_rx1_count_cell',
#                    'rf_samples_rx1_max_cell', 'rf_samples_rx1_rms_cell', 'rf_samples_tx1_count_cell',
#                    'rf_samples_tx1_max_cell', 'rf_samples_tx1_rms_cell', 'rf_tx_count_cell', 'rf_tx_cpu_time_cell',
#                    'rf_tx_sample_rate_cell', 'msg_ng_initial_context_setup_request_cell',
#                    'msg_ng_initial_context_setup_request_cell', 'msg_ng_initial_context_setup_response_cell',
#                    'msg_ng_initial_ue_message_cell', 'msg_ng_pdu_session_resource_release_command_cell',
#                    'msg_ng_pdu_session_resource_release_response_cell', 'msg_ng_pdu_session_resource_setup_request_cell',
#                    'msg_ng_pdu_session_resource_setup_response_cell', 'msg_ng_setup_request_cell',
#                    'msg_ng_setup_response_cell', 'msg_ng_ue_context_release_command_cell',
#                    'msg_ng_ue_context_release_complete_cell', 'msg_ng_ue_context_release_request_cell',
#                    'msg_ng_ue_radio_capability_info_indication_cell', 'msg_ng_uplink_nas_transport_cell',
#                    ]
feature_columns = ['total_dl_bytes_ue', 'total_ul_bytes_ue', 'cell_X_cqi_ue', 'cell_X_dl_bitrate_ue',
                    'cell_X_dl_retx_ue',  'cell_X_initial_ta_ue',
                   'cell_X_p_ue_ue', 'cell_X_pusch_snr_ue',
                   'cell_X_turbo_decoder_max_ue', 'cell_X_turbo_decoder_min_ue', 'cell_X_ul_bitrate_ue',
                   'cell_X_ul_mcs_ue',  'cell_X_ul_phr_ue', 'cell_X_ul_retx_ue',
                   'cell_X_ul_tx_ue', 'cell_X_dl_bitrate_cell', 'cell_X_dl_retx_cell',
                   'cell_X_dl_sched_users_avg_cell', 'cell_X_dl_sched_users_max_cell', 'cell_X_dl_tx_cell',
                   'cell_X_ue_active_count_avg_cell', 'cell_X_ue_active_count_max_cell',
                    'cell_X_ue_count_avg_cell', 'cell_X_ue_count_max_cell',
                   'cell_X_ue_count_min_cell', 'cell_X_ul_bitrate_cell', 'cell_X_ul_retx_cell',
                   'cell_X_ul_sched_users_avg_cell', 'cell_X_ul_sched_users_max_cell', 'cell_X_ul_tx_cell',
                   'cpu_cell', 'msg_ng_downlink_nas_transport_cell',
                   'rf_rx_cpu_time_cell', 'rf_rxtx_delay_avg_cell', 'rf_rxtx_delay_max_cell',
                   'rf_rxtx_delay_min_cell', 'rf_rxtx_delay_sd_cell', 'rf_samples_rx1_count_cell',
                   'rf_samples_rx1_max_cell', 'rf_samples_rx1_rms_cell', 'rf_samples_tx1_count_cell',
                    'rf_samples_tx1_rms_cell', 'rf_tx_count_cell', 'rf_tx_cpu_time_cell',
                   'rf_tx_sample_rate_cell'
                   ]
# , 'msg_ng_initial_context_setup_request_cell',
#                    'msg_ng_initial_context_setup_request_cell', 'msg_ng_initial_context_setup_response_cell',
#                    'msg_ng_initial_ue_message_cell', 'msg_ng_pdu_session_resource_release_command_cell',
#                    'msg_ng_pdu_session_resource_release_response_cell', 'msg_ng_pdu_session_resource_setup_request_cell',
#                    'msg_ng_pdu_session_resource_setup_response_cell', 'msg_ng_setup_request_cell',
#                    'msg_ng_setup_response_cell', 'msg_ng_ue_context_release_command_cell',
#                    'msg_ng_ue_context_release_complete_cell', 'msg_ng_ue_context_release_request_cell',
#                    'msg_ng_ue_radio_capability_info_indication_cell', 'msg_ng_uplink_nas_transport_cell'
# [3557821101183501 8609960468879056 8609960480666910 8609960480859056
#  8628490433231156 8628490443809956 8642840401594200 8642840401612300
#  8642840401624200]

#  yes      exclude            benign          yes
#  ony-second    exclude     only second     yes
#  yes

# [3557821101183501  8609960480859056 8628490433231156
#  8642840401594200 8642840401612300 8642840401624200]

#  yes  yes yes
#  only second     yes  yes
"""CREATE SEQUENCES OF CONSECUTIVE INSTANCES"""
seq_len = 12
print("Create Sequences with length: ", seq_len)

sequences_all = []
labels_all = []
# data = df[df['imeisv_ue'] != 8642840401624200]  # drop data from malicious UE
data = df.copy()
for _, group in data.groupby(['imeisv_ue', 'period_start', 'period_end']):
    sequences, labels, timestamps = form_sequences_dynamic(group, seq_len, feature_columns)
    sequences_all.extend(sequences)
    labels_all.extend(labels)
print("Number of sequence created for training and test data: ", len(sequences_all))


# Form malicious sequences
# sequences_malicious = []
# labels_malicious = []
# timestamps_malicious = []
# df_8642840401612300 = df[df['imeisv_ue'] == 8642840401624200]
# for _, group in df_8642840401612300.groupby(['imeisv_ue', 'period_start', 'period_end']):
#     sequences, labels, timestamps_individual = form_sequences_dynamic(group, seq_len, feature_columns)
#     sequences_malicious.extend(sequences)
#     labels_malicious.extend(labels)
#     timestamps_malicious.extend(timestamps_individual)
# print("Number of sequence created for individual malicious UE testing : ", len(sequences_malicious))


# calculate rate of change in the sequences
if seq_len == 1:
    data_complete = np.mean(sequences_all, axis=1)
    #data_malicious = np.mean(sequences_malicious, axis=1)
else:
    roc_sequences = auxiliary.calculate_rate_of_change(sequences_all, percentage=False)
    #roc_malicious = auxiliary.calculate_rate_of_change(sequences_malicious, percentage=False)
    # average rate of change
    roc_avg = np.mean(roc_sequences, axis=1)
    #roc_avg_malicious = np.mean(roc_malicious, axis=1)
    # Sequence average
    sequence_avg = np.mean(sequences_all, axis=1)
    #sequence_avg_malicious = np.mean(sequences_malicious, axis=1)
    # Combine arrays horizontally
    data_complete = np.hstack((sequence_avg[:, 2:],  roc_avg[:, :2]))
    #data_malicious = np.hstack((sequence_avg_malicious[:, 2:], roc_avg_malicious[:, :2]))

# Train val test split
X_train, X_test, y_train, y_test = train_test_split(data_complete, labels_all, test_size=0.2, random_state=42, stratify=labels_all)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''Class imbalance/resampling'''
smote = SMOTE(sampling_strategy={1: 5000}, random_state=42)
smote_tomek = SMOTETomek(smote=smote, random_state=0)
print(sorted(Counter(y_train).items()))
X_train_smotetomek, y_train_smotetomek = smote_tomek.fit_resample(X_train_scaled, y_train)
print(sorted(Counter(y_train_smotetomek).items()))


'''DECISION TREE'''
#None
DT_clf = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1, class_weight='balanced') #,random_state=42
DT_clf.fit(X_train_smotetomek, y_train_smotetomek)
# DT_clf.fit(X_train_scaled, y_train)
y_pred_DT = DT_clf.predict(X_test_scaled)
auxiliary.perf_eval(y_test, y_pred_DT, 'Decision Tree')

'''RANDOM FOREST'''
# 20-8 the new results
#20-10 the old results
RF_clf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=8, min_samples_split=2,
                                min_samples_leaf=1, class_weight='balanced', random_state=42) #
RF_clf.fit(X_train_smotetomek, y_train_smotetomek)
# RF_clf.fit(X_train_scaled, y_train)
y_pred_RF = RF_clf.predict(X_test_scaled)
auxiliary.perf_eval(y_test, y_pred_RF, 'Random Forest')

'''XGBOOST'''
# 12
# XGB_clf = XGBClassifier(objective='reg:squarederror', reg_alpha=0.3) #scale_pos_weight=]#
XGB_clf = XGBClassifier(
    objective='reg:squarederror',  # Keep the same objective
    n_estimators=10,               # Reduce the number of boosting rounds
    learning_rate=0.3,             # Increase the learning rate
    max_depth=8,                   # Reduce the maximum depth of trees
    subsample=0.5,                 # Use a smaller fraction of the data
    reg_alpha=1.0,                 # Increase L1 regularization term
    reg_lambda=1.0,                # Increase L2 regularization term
    random_state=42                # Set a random seed for reproducibility
)
XGB_clf.fit(X_train_smotetomek, y_train_smotetomek)
# XGB_clf.fit(X_train_scaled, y_train)
y_pred_XGB = XGB_clf.predict(X_test_scaled)
auxiliary.perf_eval(y_test, y_pred_XGB, 'XGBoost')

''' MAJORITY VOTING FOR CLASSIFIERS'''
# Combine predictions using majority voting
final_predictions = []
for i in range(len(y_pred_DT)):
    # Collect predictions for each instance across classifiers
    instance_pred = [y_pred_DT[i], y_pred_RF[i], y_pred_XGB[i]]
    # Determine the majority vote using mode
    try:
        majority_vote = statistics.mode(instance_pred)
    except statistics.StatisticsError:  # Handle the rare case where there is no mode
        majority_vote = instance_pred[0]  # You could also choose randomly or based on classifier priority
    final_predictions.append(majority_vote)

auxiliary.perf_eval(y_test, final_predictions, 'Ensemble predictions')


# '''INDIVIDUAL TESTING '''
# X_test_malicious = scaler.transform(data_malicious)
# y_test_m = labels_malicious
#
# y_pred_DT = DT_clf.predict(X_test_malicious)
# auxiliary.perf_eval(y_test_m, y_pred_DT, 'Malicious UE  Decision Tree')
#
# y_pred_RF = RF_clf.predict(X_test_malicious)
# auxiliary.perf_eval(y_test_m, y_pred_RF, 'Malicious UE Test Random Forest')
#
# y_pred_XGB = XGB_clf.predict(X_test_malicious)
# auxiliary.perf_eval(y_test_m, y_pred_XGB, 'Malicious UE Test XGBoost')
#
#
# ''' STACKING ENSEMBLE CLASSIFIER'''
# # Combine predictions using majority voting
# final_predictions = []
# for i in range(len(y_pred_DT)):
#     # Collect predictions for each instance across classifiers
#     instance_pred = [y_pred_DT[i], y_pred_RF[i], y_pred_XGB[i]]
#     # Determine the majority vote using mode
#     try:
#         majority_vote = statistics.mode(instance_pred)
#     except statistics.StatisticsError:  # Handle the rare case where there is no mode
#         majority_vote = instance_pred[0]  # You could also choose randomly or based on classifier priority
#     final_predictions.append(majority_vote)
#
# auxiliary.perf_eval(y_test_m, final_predictions, 'Ensemble predictions')
#
#
# '''Meta classifier majority voting predictions'''
# # Dataframe that stores timestamps of predictions
# meta_clf_df = pd.DataFrame({
#     'Timestamp': timestamps_malicious,
#     'Prediction': final_predictions
# })
# meta_clf_df['Timestamp'] = pd.to_datetime(meta_clf_df['Timestamp'])
# print(meta_clf_df)
#
#
# def majority_vote(predictions):
#     data_count = Counter(predictions)
#     most_common = data_count.most_common(1)[0][0]
#     return most_common
#
#
# def process_sequences(df, max_diff_seconds=5, seq_length=1):
#     majority_votes = []
#     final_timestamps = []  # List to store the timestamp of the final element in each sequence
#     # Iterate over DataFrame rows ensuring we do not go out of bounds when forming sequences
#     for start_index in range(len(df) - seq_length + 1):
#         valid_sequence = True
#         current_sequence = []  # Initialize an empty sequence
#
#         # Form the sequence by checking time difference criteria
#         for i in range(seq_length):
#             current_index = start_index + i
#             if i > 0 and (df.iloc[current_index]['Timestamp'] - df.iloc[current_index - 1]['Timestamp']).total_seconds() > max_diff_seconds:
#                 valid_sequence = False
#                 break
#             current_sequence.append(current_index)  # Store indices instead of direct row access
#
#         # If the sequence is valid and complete, compute the majority vote
#         if valid_sequence:
#             predictions = [df.iloc[idx]['Prediction'] for idx in current_sequence]
#             majority_votes.append(majority_vote(predictions))
#             final_timestamps.append(df.iloc[current_sequence[-1]]['Timestamp'])  # Correctly access the last timestamp
#
#     return majority_votes, final_timestamps
#
#
# # Function call to process the sequences
# majority_results, timestamps = process_sequences(meta_clf_df)
# print("Majority Voting Results:", len(majority_results))
# print(len(timestamps))
#
# final_predictions_df = pd.DataFrame({
#     'Timestamp': timestamps,
#     'Majority Vote': y_pred_RF
# })
# final_predictions_df['Timestamp'] = pd.to_datetime(final_predictions_df['Timestamp'])
#
# # Plotting
# plt.figure(figsize=(10, 6))  # Set the figure size
# plt.plot(final_predictions_df['Timestamp'], final_predictions_df['Majority Vote'], marker='o', linestyle='-')  # Line plot with markers
# plt.title('Majority Votes Over Time')  # Title of the plot
# plt.xlabel('Timestamp')  # Label for the x-axis
# plt.ylabel('Majority Vote')  # Label for the y-axis
# plt.grid(True)  # Enable grid for easier readability
# plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
# plt.tight_layout()  # Adjust layout to fit all elements
# # Example usage
# attack_periods = [
#     {'start': '2024-01-24 14:48:30', 'end': '2024-01-24 14:58:30'},
#     {'start': '2024-01-25 14:05:00', 'end': '2024-01-25 14:10:00'}
# ]
# # Highlighting the attack periods
# ax = plt.gca()  # Get the current Axes instance on the current figure matching the given keyword args, or create one.
# for period in attack_periods:
#     ax.axvspan(pd.to_datetime(period['start']), pd.to_datetime(period['end']), color='red', alpha=0.3)
# plt.show()
