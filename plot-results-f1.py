import os
import pandas as pd
import plotly.graph_objs as go

# Define the directories for each algorithm
dirs = [
    'I:\\Werkstudenten\\Deepak_Raj\\DATASETS\\Results_all_models_final\\public\\MattingV2\\metrics',
    'I:\\Werkstudenten\\Deepak_Raj\\DATASETS\\Results_all_models_final\\public\\DeepFTSG\\metrics',
    'I:\\Werkstudenten\\Deepak_Raj\\DATASETS\\Results_all_models_final\\public\\BSUVnet\\metrics'
]
algorithm_names = ['MattingV2', 'DeepFTSG', 'BSUVnet']

# Initialize empty lists to store F1 scores and dataset names
f1_scores = {alg: [] for alg in algorithm_names}
dataset_names = []

# Load and parse the data from the text files
for alg, alg_dir in zip(algorithm_names, dirs):
    f1_scores_list = []
    dataset_names_list = []
    for file in sorted(os.listdir(alg_dir)):
        if file.endswith('.txt'):
            with open(os.path.join(alg_dir, file), 'r') as f:
                lines = f.readlines()
                f1_score_found = False
                for line in lines:
                    metric, value = line.split(':')
                    metric = metric.strip()
                    value = float(value.strip())
                    if metric == 'F1 Score':
                        f1_scores_list.append(value)
                        dataset_names_list.append(os.path.splitext(file)[0])  # Use filename without extension as dataset name
                        f1_score_found = True
                        break  # Exit the loop after finding F1 Score
                if not f1_score_found:
                    # If F1 Score not found in file, append None
                    f1_scores_list.append(None)
                    dataset_names_list.append(os.path.splitext(file)[0])  # Use filename without extension as dataset name
    f1_scores[alg] = f1_scores_list
    if not dataset_names:  # Only set dataset_names once
        dataset_names = dataset_names_list

# Ensure all F1 scores lists are of the same length
max_len = max(len(scores) for scores in f1_scores.values())
for alg in f1_scores:
    if len(f1_scores[alg]) < max_len:
        f1_scores[alg].extend([None] * (max_len - len(f1_scores[alg])))

# Create a DataFrame for F1 scores
df = pd.DataFrame({
    'Algorithm': sum([[alg] * max_len for alg in algorithm_names], []),
    'Dataset': dataset_names * len(algorithm_names),
    'F1 Score': sum(f1_scores.values(), []),
})

# Ensure Dataset column is treated as categorical for correct axis ordering
df['Dataset'] = pd.Categorical(df['Dataset'], categories=dataset_names, ordered=True)

# Define colors, markers, line styles, line widths, and marker sizes for each algorithm
color_map = {
    'MattingV2': 'blue',
    'DeepFTSG': 'green',
    'BSUVnet': 'red'
}
marker_map = {
    'MattingV2': 'circle',
    'DeepFTSG': 'square',
    'BSUVnet': 'diamond'
}
line_style_map = {
    'MattingV2': 'solid',
    'DeepFTSG': 'dot',
    'BSUVnet': 'dash'
}
line_width_map = {
    'MattingV2': 2,
    'DeepFTSG': 2,
    'BSUVnet': 2
}
marker_size_map = {
    'MattingV2': 10,
    'DeepFTSG': 10,
    'BSUVnet': 10
}

# Create traces for each algorithm
traces = []
for alg in algorithm_names:
    alg_df = df[df['Algorithm'] == alg]
    trace = go.Scatter(
        x=alg_df['Dataset'],
        y=alg_df['F1 Score'],
        mode='lines+markers+text',  # Include text annotations
        name=alg,
        line=dict(color=color_map[alg], dash=line_style_map[alg], width=line_width_map[alg]),
        marker=dict(symbol=marker_map[alg], size=marker_size_map[alg]),
        text=[f'F1 Score: {score:.4f}' if pd.notna(score) else 'No Data' for score in alg_df['F1 Score']],
        textposition='top center',  # Position text above markers
        hoverinfo='none',  # Disable hover info (optional)
        textfont=dict(size=14)  # Increase text font size
    )
    traces.append(trace)

# Define layout for the plot with increased font sizes for axis ticks
layout = go.Layout(
    title='F1 Scores Comparison Across Algorithms and Datasets',
    title_font=dict(size=24),  # Title font size
    xaxis=dict(
        title='Dataset',
        tickangle=-45,
        title_font=dict(size=18),  # X-axis label font size
        tickfont=dict(size=16)  # X-axis tick font size
    ),
    yaxis=dict(
        title='F1 Score',
        tickformat='.4f',
        title_font=dict(size=18),  # Y-axis label font size
        tickfont=dict(size=14)  # Y-axis tick font size
    ),
    showlegend=True,
    legend=dict(x=0, y=1.0, font=dict(size=16)),  # Legend font size
    margin=dict(l=80, r=80, t=80, b=80),
    hovermode='closest'
)

# Create figure object and plot
fig = go.Figure(data=traces, layout=layout)
fig.show()
