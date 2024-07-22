""" Script that holds all auxiliary functions needed"""
# Import packages
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


""" PERFORMANCE EVALUATION METHOD"""


def perf_eval(y_test, y_pred, classifier):
    # print('Performance of classifier: ', classifier)
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    # print(cm)
    # Compute other statistics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Print the statistics
    # print("Accuracy: {:.4f}".format(accuracy))
    # print("Precision: {:.4f}".format(precision))
    # print("Recall: {:.4f}".format(recall))
    # print("F1 Score: {:.4f}".format(f1))
    # print("ROC-AUC Score: {:.4f}".format(roc_auc))
    # Assuming accuracy, precision, recall, f1, roc_auc, TN, FP, FN, TP are already defined
    print("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{},{},{},{}".format(accuracy, precision, recall, f1, roc_auc, TN, FP, FN,
                                                                  TP))

    classes = ['Benign', 'DDoS']

    report = classification_report(y_test, y_pred)
    #print(report)

    # Plotting using Seaborn's heatmap
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()


"""Compute rate of change"""


def calculate_rate_of_change(data, percentage):
    epsilon = 1e-10
    data = np.array(data)
    if percentage:
        diff = np.diff(data, axis=1)  # Calculate the difference along the time axis
        denominator = data[:, :-1, :] + epsilon  # To avoid division by zero
        roc = diff / denominator * 100
    else:
        roc = np.diff(data, n=1, axis=1)  # Calculate simple difference
    return roc


"""EXTRACT SEQUENCES METHOD"""

#
# def form_sequences_dynamic(group, sequence_length, feature_columns):
#     """
#     Generates sequences of a specified length with a stride of 1, allowing for overlapping sequences.
#     Each sequence's data is based on the specified feature columns, and sequences are grouped by a label.
#     Additionally, the timestamp of the last element in each sequence is recorded.
#
#     Parameters:
#     - group (pd.DataFrame): A pandas DataFrame containing the time series data from which sequences will be generated.
#                              This DataFrame should include the feature columns specified, as well as a 'label' column.
#     - sequence_length (int): The desired length of each generated sequence. Sequences will contain this many consecutive
#                              rows from the input DataFrame, subject to label continuity.
#     - feature_columns (list of str): A list of column names from the DataFrame that should be included in each sequence.
#                              These columns contain the features based on which the sequences are formed.
#     - timestamp_column (str): The name of the column in the DataFrame that contains the timestamp of each row.
#
#     Returns:
#     - sequences (list of lists): A nested list where each sublist represents a sequence of data points.
#                                           Each sequence is a list of rows, and each row is a list of feature values.
#     - labels (list): A list of labels corresponding to each sequence. The label denotes the common label of all rows
#                      within a sequence, based on the 'label' column in the input DataFrame.
#     - timestamps (list): A list of timestamps corresponding to the last element in each sequence.
#     """
#     sequences = []
#     labels = []
#     timestamps = []
#     current_sequence = []
#     current_label = None
#
#     for _, row in group.iterrows():
#         # Add the current row to the sequence if the label matches or if there's no current sequence
#         if not current_sequence or row['label'] == current_label:
#             current_sequence.append(row[feature_columns].values.tolist())
#             current_label = row['label']
#         else:
#             # If the label changes, start a new sequence
#             current_sequence = [row[feature_columns].values.tolist()]
#             current_label = row['label']
#
#         # Once the current sequence reaches the desired length
#         if len(current_sequence) == sequence_length:
#             sequences.append(current_sequence)
#             labels.append(current_label)
#             timestamps.append(row['chunk_end'])  # Save the timestamp of the last element
#             # Remove the first element for the next sequence (stride of 1)
#             current_sequence = current_sequence[1:]
#     return sequences, labels, timestamps


def form_sequences_dynamic(group, sequence_length, feature_columns):
    sequences = []
    labels = []
    timestamps = []
    current_sequence = []

    for _, row in group.iterrows():
        current_sequence.append(row[feature_columns].values.tolist())

        # Once the current sequence reaches the desired length
        if len(current_sequence) == sequence_length:
            sequences.append(current_sequence)
            labels.append(row['label'])  # Assign the label of the last element in the sequence
            timestamps.append(row['chunk_end'])  # Save the timestamp of the last element
            # Remove the first element for the next sequence (stride of 1)
            current_sequence = current_sequence[1:]

    return sequences, labels, timestamps

""" Method that extracts periods of data """


def find_period_boundaries_with_first_record_start(df, id_column='imeisv_ue', time_column='timestamp',
                                                   gap=pd.Timedelta(seconds=5)):
    period_boundaries = {}
    for imeisv_id, group in df.groupby(id_column):
        # Ensure the DataFrame is sorted by the timestamp to perform correct operations
        group = group.sort_values(by=time_column).copy()
        # Calculate the time difference between consecutive timestamps
        group['time_diff'] = group[time_column].diff()
        # Identify rows that start a new period with a gap larger than specified
        group['new_period'] = group['time_diff'] > gap
        # Explicitly mark the first row of each group as the start of a new period
        if not group.empty:
            group.iloc[0, group.columns.get_loc('new_period')] = True
            # Extract indexes where `new_period` is True, to identify the starts of new periods
            new_period_indexes = group[group['new_period']].index.tolist()
            # Prepare to capture start and end times of periods
            imeisv_periods = []
            # Iterate over the indexes to determine the start and end times for each period
            for i, start_idx in enumerate(new_period_indexes):
                # Start time is the timestamp at the current `True` index
                start_time = group.loc[start_idx, time_column]
                # End time is the timestamp just before the next `True` index, if not the last one
                if i + 1 < len(new_period_indexes):
                    next_start_idx = new_period_indexes[i + 1] - 1
                    # Index right before the next `True`
                    end_time = group.loc[next_start_idx, time_column]
                else:
                    # For the last period, end time is the last timestamp in the group
                    end_time = group[time_column].iloc[-1]
                imeisv_periods.append((start_time, end_time))
            period_boundaries[imeisv_id] = imeisv_periods
    return period_boundaries


"""Method that appends certain new information about each row"""


def calculate_fixed_window_statistics(df, periods, chunk_size):
    all_stats = []  # List to hold data from all periods

    for imeisv, imeisv_periods in periods.items():
        for period_idx, (start, end) in enumerate(imeisv_periods):
            # Filter the dataframe for the current period
            period_df = df[(df['imeisv_ue'] == imeisv) & (df['timestamp'] >= start) &
                           (df['timestamp'] <= end)].sort_values('timestamp')

            # Determine the number of chunks
            total_rows = len(period_df)
            chunks = [period_df[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]

            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk) > 1:
                    # Calculate the mean for each column if chunk size > 1
                    chunk_stats = chunk.mean(numeric_only=True)
                else:
                    # Use the original chunk data if chunk size = 1
                    chunk_stats = chunk.iloc[0]  # Assuming there's at least one row

                # Convert chunk_stats to a DataFrame
                if isinstance(chunk_stats, pd.Series):
                    chunk_stats = pd.DataFrame([chunk_stats])

                # Add identifiers and timestamps
                chunk_stats['imeisv_ue'] = imeisv
                chunk_stats['period_index'] = period_idx
                chunk_stats['chunk_index'] = chunk_idx
                chunk_stats['chunk_start'] = chunk['timestamp'].iloc[0]
                chunk_stats['chunk_end'] = chunk['timestamp'].iloc[-1] if len(chunk) > 1 else chunk['timestamp']
                chunk_stats['period_start'] = start
                chunk_stats['period_end'] = end

                # Append the chunk or mean statistics to the list
                all_stats.append(chunk_stats)

    # Concatenate all results into a single DataFrame outside the loops
    statistics_df = pd.concat(all_stats, ignore_index=True)

    return statistics_df
