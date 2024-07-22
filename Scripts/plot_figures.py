import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#
# # F1-SCORE FIGURE TEST SET

# Data
data = {
    "Duration": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    "DT": [0.448, 0.906, 0.886, 0.774, 0.763, 0.911, 0.902, 0.925, 0.946, 0.973, 0.988, 0.987],
    "RF": [0.681, 0.707, 0.723, 0.693, 0.750, 0.758, 0.723, 0.844, 0.826, 0.864, 0.917, 0.891],
    "XGBoost": [0.924, 0.954, 0.945, 0.968, 0.969, 0.960, 0.928, 0.971, 0.978, 0.976, 0.968, 0.979],
    "Stacking Ens.": [0.699, 0.915, 0.911, 0.905, 0.877, 0.925, 0.900, 0.932, 0.946, 0.980, 0.988, 0.983],
    "LSTM": [0.490, 0.645, 0.743, 0.817, 0.830, 0.812, 0.890, 0.797, 0.877, 0.896, 0.890, 0.936]
}

df = pd.DataFrame(data)

# Plotting the data
plt.figure(figsize=(7, 4))
models = ["DT", "RF", "XGBoost", "Stacking Ens.", "LSTM"]
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628', 'gray']  # Blue, Orange, Green, Teal, Purple

for model, color in zip(models, colors):
    plt.plot(df["Duration"], df[model], marker='o', label=model, color=color)

plt.xlabel('Sequence Length (secs)', fontsize=13)
plt.ylabel('F1-Score', fontsize=13)
plt.xticks(df["Duration"], fontsize=13)  # Ensure all durations are marked
plt.yticks(fontsize=13)

# Add legend below the plot
plt.legend(title="Models", fontsize=12, title_fontsize=12, ncol=2) # bbox_to_anchor=(0.5, -0.15), loc='upper center',

plt.grid(True)
plt.tight_layout()  # Adjust layout to fit everything neatly

# Save the figure with 400 DPI
plt.savefig('F1_Score_Performance_Plot_Test_1_12.png', dpi=400)
plt.show()

#
#
# DETECTION RATE FIGURE
# data = {
#     "Duration": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
#     "DT": [0.962633333, 0.9634, 0.9723, 0.971683333, 0.974433333, 0.984883333, 0.98645, 0.988383333, 0.991533333, 0.984566667, 0.977116667, 0.987866667],
#     "RF": [0.986033333, 0.9699, 0.992216667, 0.991883333, 0.984183333, 0.992516667, 0.989316667, 0.992616667, 0.998583333, 1, 1, 0.99795],
#     "XGBoost": [0.975666667, 0.971916667, 0.976616667, 0.9871, 0.984116667, 0.977633333, 0.98495, 0.9841, 0.9934, 0.990383333, 0.970766667, 0.983733333],
#     "Stacking Ens.": [0.981966667, 0.972966667, 0.980083333, 0.988333333, 0.980466667, 0.9834, 0.98645, 0.987116667, 0.9934, 0.990383333, 0.977116667, 0.989983333]
# }
#
# df = pd.DataFrame(data)
#
# # Colors for accessibility
# colors = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628']  # Blue, Orange, Green, Teal
#
# # Plotting the data
# plt.figure(figsize=(8, 4))
# models = ["DT", "RF", "XGBoost", "Stacking Ens."]
# colors = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628']  # Blue, Orange, Green, Teal
# for model,color in zip(models, colors):
#     plt.plot(df["Duration"], df[model], marker='o', label=model, color=color)
#
# plt.xlabel('Sequence Length (secs)', fontsize=14)
# plt.ylabel('Detection Rate', fontsize=14)
# plt.xticks(df["Duration"], fontsize=14)  # Ensure all durations are marked
# plt.yticks(fontsize=14)
# plt.legend(title="Models", fontsize=14, title_fontsize=14)
# plt.grid(True)
# plt.tight_layout()  # Adjust layout to fit everything neatly
#
# # Save the figure with 400 DPI for high-quality publication
# plt.savefig('Detection_Rate_Performance_Plot.png', dpi=400)
# plt.show()
#
# AVERAGE DETECTION RATE OF TEST SET
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Data for average detection rates and corresponding sequence durations
# durations = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
# averages = np.array([0.976575, 0.969546, 0.980304, 0.98475, 0.9808, 0.984608, 0.986792, 0.988054, 0.994229, 0.991333, 0.98125, 0.989883])
#
# plt.figure(figsize=(6, 4))
# plt.plot(durations, averages, marker='o', linestyle='-', color='blue')
#
# # Adding title and labels
# plt.xlabel('Sequence Length (secs)', fontsize=13)
# plt.ylabel('Average Detection Rate', fontsize=13)
# plt.grid(True)
# plt.xticks(durations, fontsize=13)
# plt.yticks(np.linspace(0.96, 1.0, 9), fontsize=13)  # Setting y-ticks to show detailed range near the data values
# plt.tight_layout()  # Adjust layout to fit everything neatly
# # Save the figure with 300 DPI
# plt.savefig('Average_Detection_Rate_Plot.png', dpi=400)
# plt.show()

#
#
# # INDIVIDUAL TESTING
# # F1-score per model
# # Data for F1-scores
# data = {
#     "Duration": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
#     "DT": [0.4373, 0.765983333, 0.695166667, 0.695466667, 0.690633333, 0.783583333, 0.732666667, 0.7198, 0.760733333, 0.68145, 0.735783333, 0.642616667],
#     "RF": [0.58725, 0.77535, 0.692683333, 0.700766667, 0.646466667, 0.81565, 0.835616667, 0.855233333, 0.861933333, 0.892933333, 0.885416667, 0.857083333],
#     "XGBoost": [0.503483333, 0.61765, 0.555166667, 0.554466667, 0.565166667, 0.67985, 0.7112, 0.7125, 0.714033333, 0.6755, 0.636133333, 0.540916667],
#     "Stacking Ens.": [0.558633333, 0.8355, 0.65265, 0.642733333, 0.67765, 0.760133333, 0.78425, 0.778833333, 0.777566667, 0.72565, 0.739633333, 0.70335]
# }
# df = pd.DataFrame(data)
#
# # Plotting
# plt.figure(figsize=(6, 4))
# colors = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628']  # Blue, Orange, Green, Teal
# for model, color in zip(df.columns[1:], colors):
#     plt.plot(df["Duration"], df[model], marker='o', label=model, color=color)
#
# plt.xlabel('Sequence Length (secs)', fontsize=13)
# plt.ylabel('F1-Score', fontsize=13)
# plt.xticks(df["Duration"], fontsize=13)
# plt.yticks(fontsize=13)
# plt.grid(True)
# plt.ylim([0.3, 0.9])  # Adjust the max value if needed
# # Adjust legend location with horizontal layout
# plt.legend(title="Models", fontsize=11, title_fontsize=12, loc='lower center',  ncol=4)
#
# plt.tight_layout()
# plt.savefig('F1_Score_Performance_Plot_individual_testing.png', dpi=400)
# plt.show()
#
# # Average Detection rate and FPR%
# data = {
#     'Sequence Length': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
#     'DR': [0.547704167, 0.73635, 0.719475, 0.733466667, 0.734304167, 0.704720833, 0.739008333, 0.742270833, 0.742675, 0.709254167, 0.6956625, 0.625283333],
#     'FPR %': [0.438299371, 0.289749413, 2.223794529, 2.695909586, 5.473487392, 0.04196694, 0.068408925, 0.074338986, 0.05060066, 0.061193458, 0.044667532, 0.047851693]
# }
# df = pd.DataFrame(data)
#
# # Plotting
# fig, ax = plt.subplots(figsize=(6, 4))
#
# # Plotting the Detection Rate
# ax.plot(df['Sequence Length'], df['DR'], marker='o', label='DR', color='blue')
# ax.set_xlabel('Sequence Length (secs)', fontsize=13)
# ax.set_ylabel('Average Detection Rate', fontsize=13)
# ax.tick_params(axis='both', labelsize=13)
# ax.grid(True)
#
# # Creating secondary y-axis
# ax2 = ax.twinx()
# ax2.plot(df['Sequence Length'], df['FPR %'], marker='x', label='FPR (%)', color='red')
# ax2.set_ylabel('False Positive Rate (FPR%)', fontsize=13)
# ax2.tick_params(axis='y', labelsize=13)
#
# # Adding legends
# handles1, labels1 = ax.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# ax.legend(handles1 + handles2, labels1 + labels2, loc='center left', bbox_to_anchor=(0.65, 0.2), fontsize=13)
#
# # Title and layout
# fig.tight_layout()
# plt.savefig('DR_FPR_Performance_Plot_individual_testing.png', dpi=400)
# plt.show()



# import matplotlib.pyplot as plt
# import pandas as pd
#
# # Data preparation
# data = {
#     'Model': ['DT', 'RF', 'Xgboost', 'Stacking']*3,  # repeating for each sequence length
#     'Sequence Length': [35]*4 + [40]*4 + [45]*4,
#     'DR': [0.69105, 0.876333333, 0.650883333, 0.737766667,
#            0.698483333, 0.868116667, 0.656583333, 0.7459,
#            0.749966667, 0.8362, 0.658333333, 0.7262],
#     'FPR (%)': [0.080872089, 0.115215031, 0.0420978, 0.035450779,
#                 0.119401614, 0.072329824, 0.052812252, 0.052812252,
#                 0.0952483, 0.032146301, 0.041671131, 0.033336905]
# }
#
# df = pd.DataFrame(data)
#
# # Pivot data for easy plotting
# dr_pivot = df.pivot("Model", "Sequence Length", "DR")
# fpr_pivot = df.pivot("Model", "Sequence Length", "FPR (%)")
#
# # Define colors for each sequence length
# colors = ['red', 'blue', 'gold']
#
# # Plotting Detection Rate
# plt.figure(figsize=(6, 4))
# ax = dr_pivot.plot(kind='bar', color=colors, fontsize=13)
# ax.set_ylim([0.6, 1])  # Setting the y-axis to start from 0.6
# plt.ylabel('Detection Rate', fontsize=13)
# plt.xlabel('Model', fontsize=13)
# plt.xticks(rotation=0)
# plt.grid(True)  # Adding grid
# plt.legend(title='Sequence Length', title_fontsize=13, fontsize=13)
# plt.tight_layout()
# plt.savefig('detection_rate_plot_35_45_rgb.png')
# plt.show()
#
# # Plotting False Positive Rate Percentage
# plt.figure(figsize=(6, 4))
# ax = fpr_pivot.plot(kind='bar', color=colors, fontsize=13)
# plt.ylabel('FPR (%)', fontsize=13)
# plt.xlabel('Model', fontsize=13)
# plt.xticks(rotation=0)
# plt.grid(True)  # Adding grid
# plt.legend(title='Sequence Length', title_fontsize=13, fontsize=13)
# plt.tight_layout()
# plt.savefig('fpr_percentage_plot_35_45_rgb.png')
# plt.show()
#
#
# # grayscale bw plots for bar charts
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # Data preparation
# data = {
#     'Model': ['DT', 'RF', 'Xgboost', 'Stacking']*3,  # repeating for each sequence length
#     'Sequence Length': [35]*4 + [40]*4 + [45]*4,
#     'DR': [0.69105, 0.876333333, 0.650883333, 0.737766667,
#            0.698483333, 0.868116667, 0.656583333, 0.7459,
#            0.749966667, 0.8362, 0.658333333, 0.7262],
#     'FPR (%)': [0.080872089, 0.115215031, 0.0420978, 0.035450779,
#                 0.119401614, 0.072329824, 0.052812252, 0.052812252,
#                 0.0952483, 0.032146301, 0.041671131, 0.033336905]
# }
#
# df = pd.DataFrame(data)
#
# # Pivot data for easy plotting
# dr_pivot = df.pivot("Model", "Sequence Length", "DR")
# fpr_pivot = df.pivot("Model", "Sequence Length", "FPR (%)")
#
# # Define patterns for each sequence length (matching the number of sequence lengths)
# patterns = ['/', 'o', '\\']  # Three patterns for three sequence lengths
# colors = ['black', 'gray', 'lightgray']  # Grayscale colors
#
# # Plotting Detection Rate
# plt.figure(figsize=(6, 4))
# ax = dr_pivot.plot(kind='bar', color=colors, edgecolor='black', linewidth=1.2)
# for i, bar in enumerate(ax.patches):  # Apply pattern to each bar
#     bar.set_hatch(patterns[i % len(patterns)])
# ax.set_ylim([0.6, 1])  # Set the y-axis to start from 0.6
# plt.title('Detection Rate by Model and Sequence Length', fontsize=13)
# plt.ylabel('Detection Rate', fontsize=13)
# plt.xlabel('Model', fontsize=13)
# plt.xticks(rotation=0)
# plt.grid(True, linestyle='--')
# plt.legend(title='Sequence Length', title_fontsize=13, fontsize=13)
# plt.tight_layout()
# plt.savefig('detection_rate_plot_35_45_bw.png')
# plt.show()
#
# # Plotting False Positive Rate Percentage
# plt.figure(figsize=(6, 4))
# ax = fpr_pivot.plot(kind='bar', color=colors, edgecolor='black', linewidth=1.2)
# for i, bar in enumerate(ax.patches):  # Apply pattern to each bar
#     bar.set_hatch(patterns[i % len(patterns)])
# plt.title('False Positive Rate Percentage by Model and Sequence Length', fontsize=13)
# plt.ylabel('FPR (%)', fontsize=13)
# plt.xlabel('Model', fontsize=13)
# plt.xticks(rotation=0)
# plt.grid(True, linestyle='--')
# plt.legend(title='Sequence Length', title_fontsize=13, fontsize=13)
# plt.tight_layout()
# plt.savefig('fpr_percentage_plot_35_45_bw.png')
# plt.show()


# per UE DR and FPR (%)
# data = {
#     'UE ID': ['UE1', 'UE2', 'UE3', 'UE4', 'UE5', 'UE6'],
#     'DR': [1, 1, 1, 0.3254, 0.9663, 0.9663],
#     'FPR(%)': [0, 0.073284477, 0, 0, 0.427519074, 0.166389351]
# }
#
# df = pd.DataFrame(data)
#
# # Setting the index to UE ID for better plotting
# df.set_index('UE ID', inplace=True)
#
# # Plotting
# fig, ax = plt.subplots(figsize=(6, 4))
# df['DR'].plot(marker='o', label='DR', color='blue', ax=ax)
# ax.set_ylabel('Detection Rate (DR)', fontsize=13)
# ax.set_xlabel('UE ID', fontsize=13)
# ax.tick_params(axis='both', labelsize=13)  # Set fontsize for ticks
# ax.grid(True)
#
# # Creating secondary y-axis
# ax2 = ax.twinx()
# df['FPR(%)'].plot(marker='x', label='FPR (%)', color='red', ax=ax2)
# ax2.set_ylabel('False Positive Rate (FPR%)', fontsize=13)
# ax2.tick_params(axis='y', labelsize=13)  # Set fontsize for ticks on the secondary y-axis
#
# # Adding legends
# legend1 = ax.legend(loc='center left', bbox_to_anchor=(0, 0.4), fontsize=13)
# legend2 = ax2.legend(loc='center left', bbox_to_anchor=(0, 0.5), fontsize=13)
#
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('DR_FPR_Line_Graph_per_UE.png', dpi=400)  # Saving the plot
# plt.show()
