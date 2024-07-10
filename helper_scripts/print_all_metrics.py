import os
import pandas as pd

# Define the directories for each algorithm
dirs = [
    'I:\\Werkstudenten\\Deepak_Raj\\DATASETS\\Results_all_models_final\\public\\MattingV2\\metrics',
    'I:\\Werkstudenten\\Deepak_Raj\\DATASETS\\Results_all_models_final\\public\\DeepFTSG\\metrics',
    'I:\\Werkstudenten\\Deepak_Raj\\DATASETS\\Results_all_models_final\\public\\BSUVnet\\metrics'
]
algorithm_names = ['MattingV2', 'DeepFTSG', 'BSUVnet']

# Initialize empty dictionary to store all metrics
metrics = {alg: [] for alg in algorithm_names}

# Load and parse the data from the text files
for alg, alg_dir in zip(algorithm_names, dirs):
    alg_metrics = []
    for file in sorted(os.listdir(alg_dir)):
        if file.endswith('.txt'):
            dataset_name = os.path.splitext(file)[0]  # Get dataset name from filename (remove .txt extension)
            with open(os.path.join(alg_dir, file), 'r') as f:
                lines = f.readlines()
                metric_dict = {'Dataset': dataset_name}
                for line in lines:
                    metric, value = line.split(':')
                    metric = metric.strip()
                    value = value.strip()
                    metric_dict[metric] = value
                alg_metrics.append(metric_dict)
    metrics[alg] = alg_metrics

# Print all metrics for each algorithm
for alg, alg_metrics in metrics.items():
    print(f"Metrics for {alg}:")
    for metric_dict in alg_metrics:
        print(f"Dataset: {metric_dict['Dataset']}")
        for metric, value in metric_dict.items():
            if metric != 'Dataset':  # Skip printing the 'Dataset' key again
                print(f"{metric}: {value}")
        print()
