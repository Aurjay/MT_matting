import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the directories for each algorithm
dirs = [
    'I:\\Werkstudenten\\Deepak_Raj\\DATASETS\\Results_all_models_final\\public\\MattingV2\\metrics',
    'I:\\Werkstudenten\\Deepak_Raj\\DATASETS\\Results_all_models_final\\public\\DeepFTSG\\metrics',
    'I:\\Werkstudenten\\Deepak_Raj\\DATASETS\\Results_all_models_final\\public\\BSUVnet\\metrics'
]
metrics = ['IoU', 'Precision', 'Recall', 'F1 Score', 'Specificity']
algorithm_names = ['MattingV2', 'DeepFTSG', 'BSUVnet']

# Initialize an empty dictionary to store the data
data = {alg: [] for alg in dirs}
dataset_names = {alg: [] for alg in dirs}

# Load and parse the data from the text files
for alg in dirs:
    for file in sorted(os.listdir(alg)):  # Ensure files are read in the same order
        if file.endswith('.txt'):
            dataset_results = {}
            with open(os.path.join(alg, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    metric, value = line.split(':')
                    metric = metric.strip()
                    value = float(value.strip())
                    dataset_results[metric] = value
            data[alg].append(dataset_results)
            dataset_names[alg].append(os.path.splitext(file)[0])  # Use filename without extension as dataset name

# Convert the data to a DataFrame
df_list = []
for alg, alg_name in zip(dirs, algorithm_names):
    df = pd.DataFrame(data[alg])
    df['Algorithm'] = alg_name
    df['Dataset'] = dataset_names[alg]
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

# Debug print to check the DataFrame structure
print(df.head())

# Plot bar charts for each dataset
datasets = df['Dataset'].unique()

for dataset in datasets:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Select data for the current dataset
    dataset_df = df[df['Dataset'] == dataset]

    # Extract the metric values for each algorithm
    values_alg1 = dataset_df[dataset_df['Algorithm'] == 'MattingV2'][metrics].values.flatten()
    values_alg2 = dataset_df[dataset_df['Algorithm'] == 'DeepFTSG'][metrics].values.flatten()
    values_alg3 = dataset_df[dataset_df['Algorithm'] == 'BSUVnet'][metrics].values.flatten()

    # Ensure there are values to plot
    if len(values_alg1) == 0 or len(values_alg2) == 0 or len(values_alg3) == 0:
        print(f"Skipping dataset {dataset} due to missing data.")
        continue

    # Plot bars
    bar_width = 0.2
    r1 = range(len(metrics))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    bars1 = ax.bar(r1, values_alg1, color=(31/255, 119/255, 180/255, 0.6), width=bar_width, edgecolor='grey', label='MattingV2')
    bars2 = ax.bar(r2, values_alg2, color=(44/255, 160/255, 44/255, 0.6), width=bar_width, edgecolor='grey', label='DeepFTSG')
    bars3 = ax.bar(r3, values_alg3, color=(214/255, 39/255, 40/255, 0.6), width=bar_width, edgecolor='grey', label='BSUVnet')

    # Add labels
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Values', fontweight='bold')
    ax.set_title(f'Comparison of Algorithms for {dataset}')
    ax.set_xticks([r + bar_width for r in range(len(metrics))])
    ax.set_xticklabels(metrics)

    # Add legend
    ax.legend()

    # Display the plot
    plt.show()
