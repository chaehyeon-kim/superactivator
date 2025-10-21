import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import pandas as pd
import torch
import math
from collections import defaultdict
from sklearn.metrics import auc
import sys
sys.path.append(os.path.abspath(".."))
from utils import repo_file, repo_path

experiments_dir = repo_path('Experiments')
if experiments_dir.exists():
    os.chdir(experiments_dir)

from utils.filter_datasets_utils import filter_concept_dict
from utils.general_utils import get_paper_plotting_style
from utils.default_percentthrumodels import (
    CLIP_PERCENTTHRUMODELS, LLAMA_VISION_PERCENTTHRUMODELS,
    LLAMA_TEXT_PERCENTTHRUMODELS, GEMMA_TEXT_PERCENTTHRUMODELS,
    QWEN_TEXT_PERCENTTHRUMODELS, get_model_default_percentthrumodels
)


def filter_unsupervised_detection_metrics(detection_metrics, best_clusters_per_concept):
    """
    Filter unsupervised detection metrics to only include best matching clusters.
    
    Args:
        detection_metrics: DataFrame with concept column in format "('concept_name', 'cluster_id')"
        best_clusters_per_concept: Dict mapping concept names to best cluster info
        
    Returns:
        Filtered DataFrame with concept column containing just the concept names
    """
    filtered_rows = []
    for _, row in detection_metrics.iterrows():
        try:
            # Parse concept and cluster from the tuple format
            concept_tuple = eval(row['concept'])  # Convert string tuple to actual tuple
            concept_name = concept_tuple[0]
            cluster_id = concept_tuple[1]
            
            if concept_name in best_clusters_per_concept:
                best_cluster = best_clusters_per_concept[concept_name]['best_cluster']
                if str(cluster_id) == str(best_cluster):
                    # Create a new row with just the concept name
                    new_row = row.copy()
                    new_row['concept'] = concept_name
                    filtered_rows.append(new_row)
        except:
            # Skip rows that can't be parsed
            continue
    
    if filtered_rows:
        return pd.DataFrame(filtered_rows)
    else:
        return pd.DataFrame()  # Empty dataframe if no matches
# Verify imports
# print(f"Imported LLAMA_TEXT_PERCENTTHRUMODELS: {LLAMA_TEXT_PERCENTTHRUMODELS}")

"""""""""""

General Functions

"""""""""""
def plot_concept_metrics(metric_dfs, metric_name, title, xmin=None, xmax=None):
    """
    Plots a horizontal bar chart comparing a chosen metric across multiple metrics DataFrames.

    Args:
        metric_dfs (dict): Dictionary mapping labels (str) to pd.DataFrame of concept metrics.
        metric_name (str): The metric to plot (e.g., 'accuracy', 'f1').
        title (str): Title of the plot.

    Returns:
        None (Displays the plot).
    """
    labels = list(metric_dfs.keys())  # Extract labels
    metric_dfs = list(metric_dfs.values())  # Extract corresponding DataFrames

    colors = sns.color_palette("husl", len(metric_dfs))  # Generate distinct colors
    
    # Extract all unique concepts from the first DataFrame (assume all have the same concepts)
    concepts = metric_dfs[0]["concept"].tolist()
    
    # Increase spacing by modifying the y positions
    spacing = 0.3  # Adjust spacing factor
    y = np.arange(len(concepts)) * (len(metric_dfs) * 0.2 + spacing)  # Space out concepts
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 20))
    
    # Bar width
    bar_width = 0.2

    # Plot each metric_df's values
    for i, (label, df, color) in enumerate(zip(labels, metric_dfs, colors)):
        values = df.set_index("concept")[metric_name].reindex(concepts).values
        ax.barh(y + i * bar_width, values, height=bar_width, label=label, color=color)
    
    # Formatting
    ax.set_yticks(y + (len(metric_dfs) - 1) * bar_width / 2)
    ax.set_yticklabels(concepts)
    ax.set_xlabel(metric_name.capitalize())
    ax.set_title(title)
    
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    
    # Move legend outside the plot
    ax.legend(title="Source", loc="upper left", bbox_to_anchor=(1, 1))
    
    # Show the plot
    plt.tight_layout()
    plt.show()

    

def plot_average_metrics(metric_dfs, metric_name, title=None, xmin=None, xmax=None):
    """
    Plots a horizontal bar chart comparing average metrics across multiple methods.

    Args:
        metric_dfs (dict): Mapping from label -> pd.DataFrame (with 'concept' and metric_name columns).
        metric_name (str): Metric to plot ('f1', 'accuracy', etc).
        title (str, optional): Title for the plot.
        xmin (float, optional): Minimum x-axis value.
        xmax (float, optional): Maximum x-axis value.
    """
    labels = list(metric_dfs.keys())
    avg_metrics = [df[metric_name].mean() for df in metric_dfs.values()]

    fig, ax = plt.subplots(figsize=(8, len(labels) * 0.7))

    colors = sns.color_palette("husl", len(labels))
    bars = ax.barh(labels, avg_metrics, color=colors)

    # Add text annotations at the end of each bar
    for bar, value in zip(bars, avg_metrics):
        ax.text(
            bar.get_width() + 0.01,  # slightly offset from the end of bar
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel(metric_name.capitalize())
    ax.set_ylabel("Concept Discovery Method")
    if title:
        ax.set_title(title)

    if xmin is not None or xmax is not None:
        ax.set_xlim(left=xmin if xmin is not None else 0,
                    right=xmax if xmax is not None else 1)

    plt.grid(axis='x', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    

def plot_grouped_metrics(data_dict, baseline_dfs, metric, baseline_type, curr_concepts=None, title=None, xmin=0, xmax=1):
    """
    Plots a horizontal grouped bar chart for a chosen metric for each concept,
    for each model and scheme.

    Args:
        data_dict (dict): Nested dictionary of {model: {scheme: DataFrame}}.
        metric (str): The name of the metric column to plot.
        title (str, optional): The plot title.
        xmin (float, optional): Minimum x-axis value.
        xmax (float, optional): Maximum x-axis value.
    """
    models = list(data_dict.keys())
    # Get the set of concepts for each model (assume all scheme dfs have the same concepts)
    model_concepts = {}
    for model in models:
        schemes = list(data_dict[model].keys())
        if schemes:
            df = data_dict[model][schemes[0]]
            if curr_concepts is not None:
                model_concepts[model] = [concept for concept in df['concept'].unique() if concept in curr_concepts]
            else:
                model_concepts[model] = df['concept'].unique().tolist()
        else:
            model_concepts[model] = []
    
    # Gather all schemes across models and create color mapping
    all_schemes = [scheme for scheme in data_dict[models[0]].keys()]
    cmap = plt.get_cmap("tab10")
    scheme_colors = {scheme: cmap(i) for i, scheme in enumerate(all_schemes)}
    baseline_colors = {model: cmap(i+len(all_schemes)) for i, model in enumerate(models)}

    # Determine vertical positions
    y_positions = {}  # mapping (model, concept) -> y coordinate
    model_centers = {}  # mapping model -> vertical center (for model label)
    current_y = 0
    model_gap = 0.7  # gap between models

    for model in models:
        concepts = model_concepts[model]
        for concept in concepts:
            y_positions[(model, concept)] = current_y
            current_y += 1
        model_centers[model] = np.mean([y_positions[(model, c)] for c in concepts]) if concepts else current_y
        current_y += model_gap  # add gap after each model

    # Plotting setup
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot bars for each (model, concept)
    for model in models:
        baseline_model_df = baseline_dfs[model]
        for concept in model_concepts[model]:
                
            applicable_schemes = []
            values = {}

            # Collect metric values for each scheme
            for scheme in all_schemes:
                if scheme in data_dict[model]:
                    df = data_dict[model][scheme]
                    row = df[df['concept'] == concept]
                    if not row.empty:
                        values[scheme] = row.iloc[0][metric]
                        applicable_schemes.append(scheme)

            if not applicable_schemes:
                continue

            # Bar height calculation and placement
            bar_height = 0.6 / len(applicable_schemes)
            center_y = y_positions[(model, concept)]
            for i, scheme in enumerate(applicable_schemes):
                offset = (i - (len(applicable_schemes) - 1) / 2) * bar_height
                ax.barh(center_y + offset, values[scheme], height=bar_height, color=scheme_colors[scheme],
                        label=scheme if (model == models[0] and concept == model_concepts[model][0]) else None)
                
            offset = (len(applicable_schemes) - (len(applicable_schemes) - 1) / 2) * bar_height
            baseline_val = baseline_model_df[baseline_model_df['concept'] == concept].iloc[0][metric]
            ax.barh(center_y + offset, baseline_val, height=bar_height, color=baseline_colors[model], label=f"{model} {baseline_type}")

            # Align concept labels with the center of the bars
            # Get the position for the middle of the bars for this concept
            bar_positions = []
            for i, scheme in enumerate(applicable_schemes):
                bar_positions.append(center_y + (i - (len(applicable_schemes) - 1) / 2) * bar_height)

            # The position of the concept label is aligned with the center of the bars
            concept_label_x = xmin - 0.01 * (xmax - xmin)
            concept_label_y = np.mean(bar_positions)  # The average position of all the bars for this concept
            ax.text(concept_label_x, concept_label_y, concept, va='center', ha='right', fontsize=10)

    # Remove y-ticks
    ax.set_yticks([])

    ax.set_xlabel(metric, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)

    # Position model labels
    current_xlim = ax.get_xlim()
    model_label_x = xmin - 0.1 * (xmax - xmin)
    for model in models:
        ax.text(model_label_x, model_centers[model], model, va='center', ha='right', fontsize=12, fontweight='bold')

    # Create legend
     # Retrieve the handles and labels created by the plotting calls.
    handles, labels = ax.get_legend_handles_labels()
    
    # Separate sub-label entries from baseline entries.
    sub_label_handles = []
    sub_label_labels = []
    baseline_handles = []
    baseline_labels = []
    
    for h, l in zip(handles, labels):
        # We assume baseline entries contain the baseline_type string.
        if baseline_type in l:
            if l not in baseline_labels:
                baseline_handles.append(h)
                baseline_labels.append(l)
        else:
            if l not in sub_label_labels:
                sub_label_handles.append(h)
                sub_label_labels.append(l)
    
    # Reverse the order of sub-label entries and baseline entries individually.
    sub_label_handles = list(sub_label_handles)[::-1]
    sub_label_labels = list(sub_label_labels)[::-1]
    baseline_handles = list(baseline_handles)[::-1]
    baseline_labels = list(baseline_labels)[::-1]
    
    # Combine: first the reversed sub-labels, then the reversed baseline entries.
    ordered_handles = baseline_handles + sub_label_handles
    ordered_labels = baseline_labels + sub_label_labels 
    
    # Create the legend with the reordered entries.
    ax.legend(ordered_handles, ordered_labels, title="Legend",
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    

    plt.show()



def plot_average_grouped_metrics(data_dict, baseline_dfs, metric, baseline_type, title=None, xmin=None, xmax=None):
    """
    Plots a horizontal grouped bar chart for a chosen metric averaged across concepts,
    including baseline values computed from a dictionary of baseline dataframes.

    Args:
        data_dict (dict): A dictionary where keys are overall labels and values are dictionaries.
                          The inner dictionary has sub-labels as keys and dataframes as values.
                          Each dataframe must have a column corresponding to the chosen metric.
        baseline_dfs (dict): A dictionary mapping each overall label to a dataframe.
                             Each dataframe must have a column corresponding to the chosen metric.
        metric (str): The column name for which the average is computed.
        title (str, optional): The title of the plot.
        xmin (float, optional): Minimum x-axis value.
        xmax (float, optional): Maximum x-axis value.
    """
    # Get overall labels and assume that each inner dict has the same sub-labels
    # overall_labels = list(data_dict.keys())[::-1]
    overall_labels = list(data_dict.keys())
    sub_labels = list(next(iter(data_dict.values())).keys())
    
    # Compute the average of the chosen metric for each (overall, sub) pair from data_dict
    averages = {
        ov_label: {
            sub_label: df[metric].mean() 
            for sub_label, df in sub_dict.items()
        } 
        for ov_label, sub_dict in data_dict.items()
    }
    
    # Compute baseline average metric values over concepts for each overall label.
    baseline_values = {
        ov_label: baseline_dfs[ov_label][metric].mean() 
        for ov_label in overall_labels if ov_label in baseline_dfs
    }
    
    # Set up plotting parameters for grouped horizontal bars.
    n_groups = len(overall_labels)
    n_sub = len(sub_labels)
    # Allocate space for sub-label bars and one extra for baseline within each group.
    bar_height = 0.8 / (n_sub + 1)
    
    # Create a color map for the sub-labels.
    cmap = plt.get_cmap("tab10")
    colors = {sub_label: cmap(i) for i, sub_label in enumerate(sub_labels)}
    baseline_colors = {ov_label: cmap(i+len(sub_labels)) for i, ov_label in enumerate(overall_labels)}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine y positions for each overall group.
    y_group_positions = np.arange(n_groups)
    
    # Plot bars for each overall label.
    for i, ov_label in enumerate(overall_labels):
        group_center = y_group_positions[i]
        start_y = group_center - 0.2  # Center the group vertically around the integer position.
        
        # Plot bars for each sub-label
        for j, sub_label in enumerate(sub_labels):
            if sub_label in data_dict[ov_label].keys():
                y = start_y + j * bar_height
                value = averages[ov_label][sub_label]
                ax.barh(y, value, height=bar_height, color=colors[sub_label],
                        label=sub_label if i == 0 else None)
                           
        # Plot baseline bar (placed as the last bar in the group)
        baseline_y = start_y + n_sub * bar_height
        baseline_val = baseline_values.get(ov_label, 0)
        ax.barh(baseline_y, baseline_val, height=bar_height,
                alpha=0.6, color=baseline_colors[ov_label], label=f'{ov_label} {baseline_type}')
    
    # Set y-axis ticks to show overall labels (center of each group)
    ax.set_yticks(y_group_positions)
    ax.set_yticklabels(overall_labels, fontsize=12)
    
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    
    ax.set_xlabel(metric, fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    
     # Retrieve the handles and labels created by the plotting calls.
    handles, labels = ax.get_legend_handles_labels()
    
    # Separate sub-label entries from baseline entries.
    sub_label_handles = set()
    sub_label_labels = set()
    baseline_handles = set()
    baseline_labels = set()
    
    for h, l in zip(handles, labels):
        # We assume baseline entries contain the baseline_type string.
        if baseline_type in l:
            baseline_handles.add(h)
            baseline_labels.add(l)
        else:
            sub_label_handles.add(h)
            sub_label_labels.add(l)
    
    # Reverse the order of sub-label entries and baseline entries individually.
    sub_label_handles = list(sub_label_handles)[::-1]
    sub_label_labels = list(sub_label_labels)[::-1]
    baseline_handles = list(baseline_handles)[::-1]
    baseline_labels = list(baseline_labels)[::-1]
    
    # Combine: first the reversed sub-labels, then the reversed baseline entries.
    ordered_handles = baseline_handles + sub_label_handles
    ordered_labels = baseline_labels + sub_label_labels 
    
    # Create the legend with the reordered entries.
    ax.legend(ordered_handles, ordered_labels, title="Legend",
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to leave space for the legend
    plt.show()



"""""""""""

F1 Detection 

"""""""""""
def get_per_concept_prompt_scores(dataset_name, model_name, metric, split='test'):
    """
    Load prompt detection scores from a saved CSV file for each concept.

    Args:
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model (for consistency with get_weighted_prompt_score).
        metric (str): 'f1', 'tpr', 'fpr', 'tnr', 'fnr', 'accuracy', 'fp', 'fn', 'tp', 'tn', 'precision', 'recall'.
        split (str, optional): Dataset split. Defaults to 'test'.

    Returns:
        dict: A dictionary mapping concept to its score.
    """
    # Use relative path that works in both environments
    if os.path.exists('./Experiments'):
        prompt_results_dir = f'./Experiments/prompt_results/{dataset_name}'
    else:
    prompt_results_dir = repo_path('Experiments', 'prompt_results', dataset_name)
    csv_files = [f for f in os.listdir(prompt_results_dir) if f.endswith('_f1_scores.csv') and dataset_name in f]

    if not csv_files:
        n_concepts = {
            'Broden-Pascal': 94,
            'Broden-OpenSurfaces': 48
        }.get(dataset_name, 46)
        return {f'concept_{i}': 0.5 for i in range(n_concepts)}

    csv_file = os.path.join(prompt_results_dir, csv_files[0])
    df = pd.read_csv(csv_file)

    # Calculate precision and recall if they're not in the dataframe but tp, fp, fn are
    if metric == 'precision' and metric not in df.columns and all(col in df.columns for col in ['tp', 'fp']):
        # Precision = tp / (tp + fp), handling division by zero
        df['precision'] = df.apply(lambda row: row['tp'] / (row['tp'] + row['fp']) if (row['tp'] + row['fp']) > 0 else 0.0, axis=1)
    
    if metric == 'recall' and metric not in df.columns and all(col in df.columns for col in ['tp', 'fn']):
        # Recall = tp / (tp + fn), handling division by zero  
        df['recall'] = df.apply(lambda row: row['tp'] / (row['tp'] + row['fn']) if (row['tp'] + row['fn']) > 0 else 0.0, axis=1)

    # Check if the metric exists in the dataframe
    if metric not in df.columns:
        print(f"Warning: Metric '{metric}' not found in {csv_file}. Available metrics: {list(df.columns)}")
        return {}

    return filter_concept_dict(dict(zip(df['concept'], df[metric])), dataset_name)

def get_weighted_prompt_score(dataset_name, model_name, metric, split='test'):
    """
    Calculates a single weighted prompt score for a given metric.

    Args:
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
        metric (str): 'f1', 'tpr', 'fpr', 'tnr', 'fnr', 'accuracy', 'fp', 'fn', 'tp', 'tn', 'precision', 'recall'.
        split (str, optional): Dataset split. Defaults to 'test'.

    Returns:
        float: The weighted prompt score.
    """
    per_concept_scores = get_per_concept_prompt_scores(dataset_name, model_name, metric, split)

    # === Load ground-truth counts for weighted average
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
    elif model_name == 'CLIP':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt'
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt'
    else:
        raise ValueError("Unknown model_name")

    gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    gt_concepts = set(gt_samples_per_concept.keys())

    # For count-based metrics (fp, fn, tp, tn), sum them directly
    if metric in ['fp', 'fn', 'tp', 'tn']:
        total_count = 0
        for concept, count in per_concept_scores.items():
            if concept in gt_concepts:
                total_count += count
        return total_count
    else:
        # For rate-based metrics, calculate weighted average
        weighted_sum = 0
        total_samples = 0
        for concept, score in per_concept_scores.items():
            if concept in gt_concepts:
                count = len(gt_samples_per_concept[concept])
                weighted_sum += score * count
                total_samples += count

        return weighted_sum / total_samples if total_samples > 0 else 0.0


def get_prompt_scores(dataset_name, model_name, metric,
                      weighted_avg=True, split='test'):
    """
    Load prompt detection scores from saved CSV files and optionally compute weighted average.

    Args:
        dataset_name: Name of the dataset
        metric: 'f1', 'tpr', 'fpr', 'tnr', 'fnr', 'accuracy', 'fp', 'fn', 'tp', 'tn', 'precision', 'recall'
        weighted_avg: Whether to compute weighted average based on GT sample counts
        split: Dataset split (e.g., 'test')
        model_name: Needed to determine correct input size for gt_samples_per_concept

    Returns:
        Dict of concept -> score (if f1), or scalar for other metrics
    """
    if weighted_avg:
        return get_weighted_prompt_score(dataset_name, model_name, metric, split)
    else:
        scores = get_per_concept_prompt_scores(dataset_name, metric)
        if metric == 'f1':
            return scores
        else:
            return sum(scores.values()) / len(scores) if scores else 0.0


def plot_predictions_vs_percentiles(dataset_name, model_name, sample_type, concept, scheme='avg', split='test'):
    """
    For a given concept and scheme, plots:
    - Predicted Positives and Predicted Negatives over percentiles
    - Horizontal lines for Ground Truth Positives and Negatives
    """
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # === Construct con_label based on scheme
    n_clusters = 1000 if sample_type == 'patch' else 50

    if scheme == 'avg':
        con_labels = {
            f'labeled {sample_type} avg': f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_100'
        }
    elif scheme == 'linsep':
        con_labels = {
            f'labeled {sample_type} linsep': f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_100'
        }
    elif scheme == 'kmeans':
        con_labels = {
            f'unsupervised {sample_type} kmeans': f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_100'
        }
    elif scheme == 'kmeans linsep':
        con_labels = {
            f'unsupervised {sample_type} linsep kmeans': f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_100'
        }
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

    # === Plotting
    plt.figure(figsize=(10, 6))

    for method_name, con_label in con_labels.items():
        pred_pos_counts = []
        pred_neg_counts = []
        gt_pos = None
        gt_neg = None
        valid_percentiles = []

        for pct in percentiles:
            try:
                df = torch.load(
                    f'Quant_Results/{dataset_name}/detectionmetrics_per_{pct}_{con_label}.pt',
                    weights_only=False
                )
            except FileNotFoundError:
                continue

            df = df[df['concept'] == concept]
            if df.empty:
                continue

            row = df.iloc[0]
            tp, tn, fp, fn = int(row['tp']), int(row['tn']), int(row['fp']), int(row['fn'])

            pred_pos_counts.append(tp + fp)
            pred_neg_counts.append(tn + fn)
            valid_percentiles.append(pct)

            if gt_pos is None:
                gt_pos = tp + fn
                gt_neg = tn + fp

        if pred_pos_counts:
            plt.plot(valid_percentiles, pred_pos_counts, marker='o', label=f"{method_name} - Pred Pos")
            plt.plot(valid_percentiles, pred_neg_counts, marker='s', label=f"{method_name} - Pred Neg")

    if gt_pos is not None:
        plt.axhline(y=gt_pos, color='green', linestyle='--', label=f"GT Positives = {gt_pos}")
    if gt_neg is not None:
        plt.axhline(y=gt_neg, color='purple', linestyle='--', label=f"GT Negatives = {gt_neg}")

    plt.title(f"Predicted Counts vs Percentile for Concept: '{concept}' ({scheme})")
    plt.xlabel("Percentile Threshold")
    plt.ylabel("Sample Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_detection_scores(dataset_name, split, model_name, sample_types, metric='f1', weighted_avg=True,
                            plot_type='both', concept_types=None, baseline_types=None, percentthrumodel=100):
    """
    Compute detection scores for different concept learning methods across percentile thresholds.
    
    Args:
        dataset_name: Name of the dataset
        split: Data split to use ('test', 'val', etc.)
        model_name: Model name (e.g., 'CLIP', 'Llama')
        sample_types: List of sample types (e.g., ['cls', 'patch'])
        metric: Metric to compute (default: 'f1')
        weighted_avg: Whether to use weighted average (default: True)
        plot_type: 'supervised', 'unsupervised', or 'both' (default: 'both')
        concept_types: List of concept types to include. Options: ['avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae']
        baseline_types: List of baseline types to include. Options: ['random', 'always_yes', 'always_no', 'prompt']
        percentthrumodel: Percentage through model for embeddings (default: 100)
    
    Returns:
        dict: Dictionary containing:
            - 'percentiles': List of percentile thresholds
            - 'concept_data': Dict mapping method names to their score lists
            - 'baseline_data': Dict mapping baseline names to their constant scores
            - 'gt_samples_per_concept': Ground truth samples dict (for weighted averaging)
            - 'style_map': Style information for plotting
            - 'con_labels': Concept labels used
    """
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95]

    # Validate concept_types if provided
    if concept_types is not None:
        valid_types = {'avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae'}
        provided_types = set(concept_types)
        invalid_types = provided_types - valid_types
        if invalid_types:
            raise ValueError(f"Invalid concept types: {invalid_types}. Valid options: {valid_types}")

    # Build concept labels
    con_labels = {}
    for sample_type in sample_types:
        n_clusters = 1000 if sample_type == 'patch' else 50
        if plot_type in ('supervised', 'both'):
            if concept_types is None or 'avg' in concept_types:
                con_labels[f'labeled {sample_type} avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'linsep' in concept_types:
                con_labels[f'labeled {sample_type} linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
        if plot_type in ('unsupervised', 'both'):
            if concept_types is None or 'kmeans' in concept_types:
                con_labels[f'unsupervised {sample_type} kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'linsep kmeans' in concept_types:
                con_labels[f'unsupervised {sample_type} linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
        # Add SAE for supported models and datasets
        if (concept_types is None or 'sae' in concept_types):
            # CLIP SAE for vision datasets (only available at percentthrumodel=92)
            if model_name == 'CLIP' and sample_type == 'patch':
                if percentthrumodel == 92:
                    con_labels[f'{sample_type} sae'] = f'{model_name}_sae_{sample_type}_dense'
                else:
                    print(f"Warning: SAE concepts for CLIP are only available at percentthrumodel=92, not {percentthrumodel}")
            # Gemma SAE for text datasets (only available at percentthrumodel=81)
            elif model_name == 'Gemma' and sample_type == 'patch':
                if percentthrumodel == 81:
                    con_labels[f'{sample_type} sae'] = f'{model_name}_sae_{sample_type}_dense'
                else:
                    print(f"Warning: SAE concepts for Gemma are only available at percentthrumodel=81, not {percentthrumodel}")

    style_map = {
        'labeled patch avg': {'color': 'orchid', 'type': 'supervised', 'label': 'Token: Supervised'},
        'labeled patch linsep': {'color': 'indigo', 'type': 'supervised', 'label': 'Token: Supervised'},
        'labeled cls avg': {'color': 'goldenrod', 'type': 'supervised', 'label': 'CLS: Supervised'},
        'labeled cls linsep': {'color': 'orangered', 'type': 'supervised', 'label': 'CLS: Supervised'},
        'unsupervised patch kmeans': {'color': 'orchid', 'type': 'unsupervised', 'label': 'Token: Unsupervised'},
        'unsupervised patch linsep kmeans': {'color': 'indigo', 'type': 'unsupervised', 'label': 'Token: Unsupervised'},
        'unsupervised cls kmeans': {'color': 'goldenrod', 'type': 'unsupervised', 'label': 'CLS: Unsupervised'},
        'unsupervised cls linsep kmeans': {'color': 'orangered', 'type': 'unsupervised', 'label': 'CLS: Unsupervised'},
        'patch sae': {'color': 'darkgreen', 'type': 'sae', 'label': 'Token: SAE Unsupervised'},
    }

    # Load ground truth samples
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
        gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    elif model_name == 'CLIP':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
    elif model_name == 'Llama':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    # Validate baseline_types if provided
    if baseline_types is not None:
        valid_baselines = {'random', 'always_yes', 'always_no', 'prompt'}
        provided_baselines = set(baseline_types)
        invalid_baselines = provided_baselines - valid_baselines
        if invalid_baselines:
            raise ValueError(f"Invalid baseline types: {invalid_baselines}. Valid options: {valid_baselines}")

    # Compute baseline scores
    baseline_data = {}
    
    # Compute prompt scores (only for rate-based metrics, not counts)
    if metric not in ['fp', 'fn', 'tp', 'tn'] and (baseline_types is None or 'prompt' in baseline_types):
        try:
            prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
            baseline_data['prompt'] = prompt_score
        except Exception:
            pass

    # Baseline CSVs: random, always_yes, always_no
    for baseline_type in ['random', 'always_yes', 'always_no']:
        # Skip if baseline_types is specified and this baseline is not included
        if baseline_types is not None and baseline_type not in baseline_types:
            continue
            
        baseline_path = f'Quant_Results/{dataset_name}/{baseline_type}_{model_name}_cls_baseline.csv'
        if not os.path.exists(baseline_path):
            continue
        df = pd.read_csv(baseline_path)
        df = df[df['concept'].isin(gt_samples_per_concept)]
        if weighted_avg:
            total = sum(len(gt_samples_per_concept[c]) for c in df['concept'])
            score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) for _, row in df.iterrows()) / total
        else:
            score = df[metric].mean()
        baseline_data[baseline_type] = score

    # Compute concept method scores across percentiles
    concept_data = {}
    for name, con_label in con_labels.items():
        scores = []
        for percentile in percentiles:
            # For calibration data, append _cal to con_label
            if split == 'cal':
                file_con_label = con_label + '_cal'
            else:
                file_con_label = con_label
                
            # Check if this is an unsupervised method (kmeans or sae)
            if 'kmeans' in con_label or 'sae' in con_label:
                # Unsupervised: use allpairs pattern and CSV format
                file_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{file_con_label}.csv'
                
                if not os.path.exists(file_path):
                    print(f"Warning: File not found - {file_path}")
                    scores.append(0)  # Default to 0 if file not found
                    continue
                    
                # Load CSV file for unsupervised
                detection_metrics = pd.read_csv(file_path)
                
                # For unsupervised, we need to filter to best matching clusters
                # Load the best cluster mapping for the appropriate split
                if split == 'cal':
                    # For calibration, use the calibration-specific best clusters
                    best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}_cal.pt'
                else:
                    # For test/train, use the regular best clusters
                    best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                if os.path.exists(best_clusters_path):
                    best_clusters = torch.load(best_clusters_path, weights_only=False)
                    # Filter to only the best matching (concept, cluster) pairs
                    filtered_rows = []
                    for concept in gt_samples_per_concept:
                        if concept in best_clusters:
                            cluster_id = best_clusters[concept]['best_cluster']
                            # Find the row for this (concept, cluster) pair
                            row = detection_metrics[detection_metrics['concept'] == f"('{concept}', '{cluster_id}')"]
                            if not row.empty:
                                # Create a simplified row with just the concept name
                                simplified_row = row.iloc[0].copy()
                                simplified_row['concept'] = concept
                                filtered_rows.append(simplified_row)
                    
                    if filtered_rows:
                        detection_metrics = pd.DataFrame(filtered_rows)
                    else:
                        detection_metrics = pd.DataFrame()  # Empty dataframe if no matches
                else:
                    print(f"Warning: Best clusters file not found - {best_clusters_path}")
                    detection_metrics = pd.DataFrame()
            else:
                # Supervised: use original pattern and PT format
                file_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{file_con_label}.pt'
                
                if not os.path.exists(file_path):
                    print(f"Warning: File not found - {file_path}")
                    scores.append(0)  # Default to 0 if file not found
                    continue
                    
                detection_metrics = torch.load(file_path, weights_only=False)
            detection_metrics = detection_metrics[detection_metrics['concept'].isin(gt_samples_per_concept)]
            if weighted_avg:
                total = sum(len(gt_samples_per_concept[c]) for c in detection_metrics['concept'])
                if total > 0:
                    score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) for _, row in detection_metrics.iterrows()) / total
                else:
                    score = 0
            else:
                score = detection_metrics[metric].mean() if len(detection_metrics) > 0 else 0
            scores.append(score)
        
        concept_data[name] = scores

    return {
        'percentiles': percentiles,
        'concept_data': concept_data,
        'baseline_data': baseline_data,
        'gt_samples_per_concept': gt_samples_per_concept,
        'style_map': style_map,
        'con_labels': con_labels
    }


def plot_detection_scores(dataset_name, split, model_name, sample_types, metric='f1', weighted_avg=True,
                          plot_type='both', concept_types=None, baseline_types=None, save_filename=None, title=None, 
                          legend_on_plot=False, ylim=None, figsize=(10, 7), percentthrumodel=100,
                          label_font_size=12, legend_font=None, xlabel=None):
    """
    Plot detection scores for different concept learning methods.
    
    Args:
        dataset_name: Name of the dataset
        split: Data split to use ('test', 'val', etc.)
        model_name: Model name (e.g., 'CLIP', 'Llama')
        sample_types: List of sample types (e.g., ['cls', 'patch'])
        metric: Metric to plot (default: 'f1')
        weighted_avg: Whether to use weighted average (default: True)
        plot_type: 'supervised', 'unsupervised', or 'both' (default: 'both')
        concept_types: List of concept types to include. Options: ['avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae']
                      If None, all available types are plotted
        baseline_types: List of baseline types to include. Options: ['random', 'always_yes', 'always_no', 'prompt']
                       If None, all available baselines are plotted
        save_filename: Optional filename to save the plot. If None, uses default naming
        title: Optional title for the plot. If None, uses default title based on dataset type
        legend_on_plot: If True, includes legend on the main plot. If False, creates separate legend figure (default: False)
        ylim: Optional tuple (ymin, ymax) to set y-axis limits. If None, uses default (0, 1.05)
        figsize: Tuple (width, height) for figure size in inches (default: (10, 7))
        percentthrumodel: Percentage through model for embeddings (default: 100)
        label_font_size: Font size for axis labels (default: 12)
        legend_font: Font size for legend. If None, uses label_font_size (default: None)
        xlabel: Custom x-axis label. If None, uses default "Ground Truth Concept Recall Percentage" (default: None)
    """
    # Set save path
    if save_filename is None:
        # Build filename with sample types, concept types, and baseline types
        sample_types_str = '_'.join(sample_types)
        concept_types_str = ''
        if concept_types is not None:
            concept_types_str = '_' + '_'.join([ct.replace(' ', '_') for ct in concept_types])
        baseline_types_str = ''
        if baseline_types is not None:
            baseline_types_str = '_' + '_'.join(baseline_types)
        save_path = f'../Figs/Paper_Figs/{model_name}_{dataset_name}_{sample_types_str}{concept_types_str}{baseline_types_str}_detectplot.pdf'
    else:
        save_path = save_filename
    # Use legend_font if provided, otherwise use label_font_size
    if legend_font is None:
        legend_font = label_font_size
        
    plt.figure(figsize=figsize)
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95]

    # Validate concept_types if provided
    if concept_types is not None:
        valid_types = {'avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae'}
        provided_types = set(concept_types)
        invalid_types = provided_types - valid_types
        if invalid_types:
            raise ValueError(f"Invalid concept types: {invalid_types}. Valid options: {valid_types}")

    con_labels = {}
    for sample_type in sample_types:
        n_clusters = 1000 if sample_type == 'patch' else 50
        if plot_type in ('supervised', 'both'):
            if concept_types is None or 'avg' in concept_types:
                con_labels[f'labeled {sample_type} avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'linsep' in concept_types:
                con_labels[f'labeled {sample_type} linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
        if plot_type in ('unsupervised', 'both'):
            if concept_types is None or 'kmeans' in concept_types:
                con_labels[f'unsupervised {sample_type} kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'linsep kmeans' in concept_types:
                con_labels[f'unsupervised {sample_type} linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
        # Add SAE for supported models and datasets
        if (concept_types is None or 'sae' in concept_types):
            # CLIP SAE for vision datasets (only available at percentthrumodel=92)
            if model_name == 'CLIP' and sample_type == 'patch':
                if percentthrumodel == 92:
                    con_labels[f'{sample_type} sae'] = f'{model_name}_sae_{sample_type}_dense'
                else:
                    print(f"Warning: SAE concepts for CLIP are only available at percentthrumodel=92, not {percentthrumodel}")
            # Gemma SAE for text datasets (only available at percentthrumodel=81)
            elif model_name == 'Gemma' and sample_type == 'patch':
                if percentthrumodel == 81:
                    con_labels[f'{sample_type} sae'] = f'{model_name}_sae_{sample_type}_dense'
                else:
                    print(f"Warning: SAE concepts for Gemma are only available at percentthrumodel=81, not {percentthrumodel}")

    style_map = {
        'labeled patch avg': {'color': 'orchid', 'type': 'supervised', 'label': 'Token: Supervised'},
        'labeled patch linsep': {'color': 'indigo', 'type': 'supervised', 'label': 'Token: Supervised'},
        'labeled cls avg': {'color': 'goldenrod', 'type': 'supervised', 'label': 'CLS: Supervised'},
        'labeled cls linsep': {'color': 'orangered', 'type': 'supervised', 'label': 'CLS: Supervised'},
        'unsupervised patch kmeans': {'color': 'orchid', 'type': 'unsupervised', 'label': 'Token: Unsupervised'},
        'unsupervised patch linsep kmeans': {'color': 'indigo', 'type': 'unsupervised', 'label': 'Token: Unsupervised'},
        'unsupervised cls kmeans': {'color': 'goldenrod', 'type': 'unsupervised', 'label': 'CLS: Unsupervised'},
        'unsupervised cls linsep kmeans': {'color': 'orangered', 'type': 'unsupervised', 'label': 'CLS: Unsupervised'},
        'patch sae': {'color': 'darkgreen', 'type': 'sae', 'label': 'Token: SAE Unsupervised'},
    }

    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
        gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    elif model_name == 'CLIP':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
    elif model_name == 'Llama':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    # Validate baseline_types if provided
    if baseline_types is not None:
        valid_baselines = {'random', 'always_yes', 'always_no', 'prompt'}
        provided_baselines = set(baseline_types)
        invalid_baselines = provided_baselines - valid_baselines
        if invalid_baselines:
            raise ValueError(f"Invalid baseline types: {invalid_baselines}. Valid options: {valid_baselines}")

    # Initialize lists to collect plot lines for legend reordering
    lines_and_labels = []
    baseline_lines_and_labels = []

    # Plot prompt scores (only for rate-based metrics, not counts)
    if metric not in ['fp', 'fn', 'tp', 'tn'] and (baseline_types is None or 'prompt' in baseline_types):
        try:
            prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
            prompt_line = plt.axhline(prompt_score, color='#8B4513', linestyle='-.', linewidth=1.75, label="Prompt")
            baseline_lines_and_labels.append((prompt_line, "Prompt"))
        except Exception:
            pass

    # Baseline CSVs: random, always_yes, always_no
    baseline_style_map = {
        'random':     {'color': '#888888', 'label': 'Random'},
        'always_yes': {'color': '#bbbbbb', 'label': 'Always Pos'},
        'always_no':  {'color': '#dddddd', 'label': 'Always Neg'}
    }

    for baseline_type in ['random', 'always_yes', 'always_no']:
        # Skip if baseline_types is specified and this baseline is not included
        if baseline_types is not None and baseline_type not in baseline_types:
            continue
            
        style = baseline_style_map[baseline_type]
        baseline_path = f'Quant_Results/{dataset_name}/{baseline_type}_{model_name}_cls_baseline.csv'
        if not os.path.exists(baseline_path):
            continue
        df = pd.read_csv(baseline_path)
        df = df[df['concept'].isin(gt_samples_per_concept)]
        if weighted_avg:
            total = sum(len(gt_samples_per_concept[c]) for c in df['concept'])
            score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) for _, row in df.iterrows()) / total
        else:
            score = df[metric].mean()
        baseline_line = plt.axhline(score, color=style['color'], linestyle='-.', linewidth=1.75, label=style['label'])
        baseline_lines_and_labels.append((baseline_line, style['label']))

    # Check if all 4 concept types are present
    all_four_types = False
    if concept_types is not None:
        all_four_types = set(concept_types) == {'avg', 'linsep', 'kmeans', 'linsep kmeans'}
    elif plot_type == 'both':
        # If concept_types is None and plot_type is 'both', all 4 types will be plotted
        all_four_types = True
    
    # Define markers for each concept type when all 4 are present
    concept_markers = {
        'avg': 'o',
        'linsep': 's',
        'kmeans': '^',
        'linsep kmeans': 'D'
    }
    
    seen_labels = set()
    for name, con_label in con_labels.items():
        scores = []
        for percentile in percentiles:
            # For calibration data, append _cal to con_label
            if split == 'cal':
                file_con_label = con_label + '_cal'
            else:
                file_con_label = con_label
                
            # Check if this is an unsupervised method (kmeans or sae)
            if 'kmeans' in con_label or 'sae' in con_label:
                # Unsupervised: use allpairs pattern and CSV format
                file_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{file_con_label}.csv'
                
                if not os.path.exists(file_path):
                    print(f"Warning: File not found - {file_path}")
                    scores.append(0)  # Default to 0 if file not found
                    continue
                    
                # Load CSV file for unsupervised
                detection_metrics = pd.read_csv(file_path)
                
                # For unsupervised, we need to filter to best matching clusters
                # Load the best cluster mapping for the appropriate split
                if split == 'cal':
                    # For calibration, use the calibration-specific best clusters
                    best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}_cal.pt'
                else:
                    # For test/train, use the regular best clusters
                    best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                if os.path.exists(best_clusters_path):
                    best_clusters = torch.load(best_clusters_path, weights_only=False)
                    # Filter to only the best matching (concept, cluster) pairs
                    filtered_rows = []
                    for concept in gt_samples_per_concept:
                        if concept in best_clusters:
                            cluster_id = best_clusters[concept]['best_cluster']
                            # Find the row for this (concept, cluster) pair
                            row = detection_metrics[detection_metrics['concept'] == f"('{concept}', '{cluster_id}')"]
                            if not row.empty:
                                # Create a simplified row with just the concept name
                                simplified_row = row.iloc[0].copy()
                                simplified_row['concept'] = concept
                                filtered_rows.append(simplified_row)
                    
                    if filtered_rows:
                        detection_metrics = pd.DataFrame(filtered_rows)
                    else:
                        detection_metrics = pd.DataFrame()  # Empty dataframe if no matches
                else:
                    print(f"Warning: Best clusters file not found - {best_clusters_path}")
                    detection_metrics = pd.DataFrame()
            else:
                # Supervised: use original pattern and PT format
                file_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{file_con_label}.pt'
                
                if not os.path.exists(file_path):
                    print(f"Warning: File not found - {file_path}")
                    scores.append(0)  # Default to 0 if file not found
                    continue
                    
                detection_metrics = torch.load(file_path, weights_only=False)
            detection_metrics = detection_metrics[detection_metrics['concept'].isin(gt_samples_per_concept)]
            if weighted_avg:
                total = sum(len(gt_samples_per_concept[c]) for c in detection_metrics['concept'])
                if total > 0:
                    score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) for _, row in detection_metrics.iterrows()) / total
                else:
                    score = 0
            else:
                score = detection_metrics[metric].mean() if len(detection_metrics) > 0 else 0
            scores.append(score)

        style = style_map[name]
        color = style['color']
        kind = style['type']
        label = style['label']
        
        # Determine the label and marker to use
        if all_four_types:
            # When all 4 types are present, use sample type + concept type as label
            sample_type_label = 'Token' if 'patch' in name else 'CLS'
            
            if 'avg' in name:
                plot_label = f'{sample_type_label}: avg'
                marker = concept_markers['avg']
            elif 'linsep kmeans' in name:
                plot_label = f'{sample_type_label}: kmeans linsep'
                marker = concept_markers['linsep kmeans']
            elif 'linsep' in name:
                plot_label = f'{sample_type_label}: linsep'
                marker = concept_markers['linsep']
            elif 'kmeans' in name:
                plot_label = f'{sample_type_label}: kmeans'
                marker = concept_markers['kmeans']
            else:
                plot_label = label
                marker = 'o'
        else:
            # Use original labeling scheme
            plot_label = label if label not in seen_labels else None
            seen_labels.add(label)
            marker = 'o'
        
        # Use solid lines for supervised, dotted for unsupervised/sae
        linestyle = '-' if kind == 'supervised' else ':'
        
        # Store plot data for later reordering in legend
        lines_and_labels.append((plt.plot(percentiles, scores, color=color, linestyle=linestyle, marker=marker, markersize=3, linewidth=1.75, label=plot_label)[0], plot_label))

    # Use custom xlabel if provided, otherwise use default
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=label_font_size)
    else:
        plt.xlabel("Ground Truth Concept Recall Percentage", fontsize=label_font_size)
    plt.ylabel("F1" if metric == 'f1' else f"{metric.upper()} Score", fontsize=label_font_size, rotation=0, ha='right')
    if ylim is not None:
        plt.ylim(ylim)
    else:
        plt.ylim(0, 1.05)
    plt.xlim(0, 1)
    # Set ticks every 10%
    tick_positions = np.arange(0, 1.1, 0.1)  # Every 10%
    tick_labels = []
    for pos in tick_positions:
        # Label only at 10%, 30%, 50%, 70%, 90%
        if int(pos * 100) in [10, 30, 50, 70, 90]:
            tick_labels.append(f"{int(pos*100)}%")
        else:
            tick_labels.append("")  # Empty label for other ticks
    plt.xticks(tick_positions, tick_labels, rotation=45)
    # Set tick label font size to match legend font and move x labels closer
    plt.tick_params(axis='both', which='major', labelsize=legend_font)
    plt.tick_params(axis='x', pad=2)  # Reduce padding for x-axis labels
    
    # Hide 0.0 on y-axis
    yticks = plt.yticks()[0]
    ylabels = []
    for tick in yticks:
        if tick == 0:
            ylabels.append("")
        else:
            ylabels.append(f"{tick:.1f}")
    plt.yticks(yticks, ylabels)
    # Use custom title if provided, otherwise use default based on dataset
    if title is not None and title != "":
        plt.title(title, fontweight='bold', fontsize=label_font_size)
    elif title is None:  # Only show default titles if title is None, not empty string
        if dataset_name == 'Stanford-Tree-Bank':
            plt.title("Stanford-Tree-Bank:\nSentence-Level Detection", fontweight='bold', fontsize=label_font_size)
        elif dataset_name == 'iSarcasm':
            plt.title("Augmented iSarcasm:\nTweet-Level Detection", fontweight='bold', fontsize=label_font_size)
        elif dataset_name == 'Sarcasm':
            plt.title("Sarcasm:\nParagraph-Level Detection", fontweight='bold', fontsize=label_font_size)
        elif dataset_name == 'GoEmotions':
            plt.title("Augmented GoEmotions:\nComment-Level Detection", fontweight='bold', fontsize=label_font_size)
        else:
            # For image datasets: Coco, CLEVR, Broden-Pascal, Broden-OpenSurfaces
            # Remove "Broden-" prefix for cleaner titles
            display_name = dataset_name
            if dataset_name == 'Broden-OpenSurfaces':
                display_name = 'OpenSurfaces'
            elif dataset_name == 'Broden-Pascal':
                display_name = 'Pascal'
            plt.title(f"{display_name}:\nImage-Level Detection", fontweight='bold', fontsize=label_font_size)
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Add legend to plot if requested
    if legend_on_plot:
        # Create a compact legend on the right side of the plot
        plt.legend(title="Concept Type", loc='center left', bbox_to_anchor=(1.02, 0.5), 
                  frameon=True, fontsize=legend_font, title_fontsize=legend_font, ncol=1)
    
    # Collect all lines and labels with concept methods first, then baselines (for separate legend if needed)
    all_lines = []
    all_labels = []
    
    # Add concept lines first
    for line, label in lines_and_labels:
        if label is not None:
            all_lines.append(line)
            all_labels.append(label)
    
    # Add baseline lines second
    for line, label in baseline_lines_and_labels:
        all_lines.append(line)
        all_labels.append(label)
    
    # Adjust layout based on whether legend is on plot
    if legend_on_plot:
        plt.tight_layout()
        plt.subplots_adjust(right=0.65)  # Make room for legend on the right
    else:
        plt.tight_layout()
    
    # Save the plot (with or without legend depending on option)
    plt.savefig(save_path, dpi=500, format='pdf', bbox_inches='tight')
    plt.show()
    
    # Create a separate figure for the legend only if not already on plot
    if not legend_on_plot:
        # Calculate height based on number of entries
        n_entries = len(all_lines)
        fig_height = max(6, n_entries * 0.8)  # At least 6 inches, 0.8 inch per entry
        fig_legend = plt.figure(figsize=(4, fig_height))
        
        # Create legend on the new figure with proper spacing
        fig_legend.legend(all_lines, all_labels, title="Concept Type", loc='center', 
                         frameon=True, fontsize=legend_font, title_fontsize=legend_font,
                         labelspacing=2.0, handlelength=3.0, handletextpad=1.0)
        
        # Save the legend as a separate file
        legend_path = save_path.replace('.pdf', '_legend.pdf')
        fig_legend.savefig(legend_path, dpi=500, format='pdf', bbox_inches='tight')
        plt.close(fig_legend)


def _plot_detection_scores_on_axis(ax, dataset_name, split, model_name, sample_types,
                                  metric='f1', weighted_avg=True, plot_type='both',
                                  concept_types=None, baseline_types=None, 
                                  percentthrumodel=100, label_font_size=12,
                                  ylim=None, xlabel=None, ylabel=None, show_legend=False):
    """
    Helper function to plot detection scores on a specific axis.
    Returns lines and labels for legend creation.
    """
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95]
    
    # Validate concept_types if provided
    if concept_types is not None:
        valid_types = {'avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae'}
        provided_types = set(concept_types)
        invalid_types = provided_types - valid_types
        if invalid_types:
            raise ValueError(f"Invalid concept types: {invalid_types}. Valid options: {valid_types}")
    
    con_labels = {}
    for sample_type in sample_types:
        n_clusters = 1000 if sample_type == 'patch' else 50
        if plot_type in ('supervised', 'both'):
            if concept_types is None or 'avg' in concept_types:
                con_labels[f'labeled {sample_type} avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'linsep' in concept_types:
                con_labels[f'labeled {sample_type} linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
        if plot_type in ('unsupervised', 'both'):
            if concept_types is None or 'kmeans' in concept_types:
                con_labels[f'unsupervised {sample_type} kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'linsep kmeans' in concept_types:
                con_labels[f'unsupervised {sample_type} linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
        # Add SAE for supported models and datasets
        if (concept_types is None or 'sae' in concept_types):
            # CLIP SAE for vision datasets (only available at percentthrumodel=92)
            if model_name == 'CLIP' and sample_type == 'patch':
                    if percentthrumodel == 92:
                        con_labels[f'{sample_type} sae'] = f'{model_name}_sae_{sample_type}_dense'
                    else:
                        print(f"Warning: SAE concepts for CLIP are only available at percentthrumodel=92, not {percentthrumodel}")
            # Gemma SAE for text datasets (only available at percentthrumodel=81)
            elif model_name == 'Gemma' and sample_type == 'patch':
                    if percentthrumodel == 81:
                        con_labels[f'{sample_type} sae'] = f'{model_name}_sae_{sample_type}_dense'
                    else:
                        print(f"Warning: SAE concepts for Gemma are only available at percentthrumodel=81, not {percentthrumodel}")
    
    # Define colors for concept types (not sample types)
    concept_colors = {
        'avg': '#1f77b4',  # blue
        'linsep': '#ff7f0e',  # orange  
        'kmeans': '#2ca02c',  # green
        'linsep kmeans': '#d62728',  # red
        'sae': '#9467bd',  # purple
    }
    
    style_map = {
        'labeled patch avg': {'color': concept_colors['avg'], 'type': 'supervised', 'label': 'Avg', 'concept': 'avg'},
        'labeled patch linsep': {'color': concept_colors['linsep'], 'type': 'supervised', 'label': 'Linsep', 'concept': 'linsep'},
        'labeled cls avg': {'color': concept_colors['avg'], 'type': 'supervised', 'label': 'Avg', 'concept': 'avg'},
        'labeled cls linsep': {'color': concept_colors['linsep'], 'type': 'supervised', 'label': 'Linsep', 'concept': 'linsep'},
        'unsupervised patch kmeans': {'color': concept_colors['kmeans'], 'type': 'unsupervised', 'label': 'Kmeans', 'concept': 'kmeans'},
        'unsupervised patch linsep kmeans': {'color': concept_colors['linsep kmeans'], 'type': 'unsupervised', 'label': 'Linsep Kmeans', 'concept': 'linsep kmeans'},
        'unsupervised cls kmeans': {'color': concept_colors['kmeans'], 'type': 'unsupervised', 'label': 'Kmeans', 'concept': 'kmeans'},
        'unsupervised cls linsep kmeans': {'color': concept_colors['linsep kmeans'], 'type': 'unsupervised', 'label': 'Linsep Kmeans', 'concept': 'linsep kmeans'},
        'patch sae': {'color': concept_colors['sae'], 'type': 'sae', 'label': 'SAE', 'concept': 'sae'},
    }
    
    lines_and_labels = []
    
    # Load ground truth samples
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
        gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    elif model_name == 'CLIP':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
    elif model_name == 'Llama':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
    
    # Import filter function
    from utils.filter_datasets_utils import filter_concept_dict
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    # Plot concept detection scores
    for label, con_label in con_labels.items():
        scores = []
        for percentile in percentiles:
            try:
                # For calibration data, append _cal to con_label
                if split == 'cal':
                    file_con_label = con_label + '_cal'
                else:
                    file_con_label = con_label
                
                # Check if this is an unsupervised method (kmeans or sae)
                if 'kmeans' in con_label or 'sae' in con_label:
                    # Unsupervised: use allpairs pattern and CSV format
                    file_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{file_con_label}.csv'
                    
                    if not os.path.exists(file_path):
                        scores.append(0)
                        continue
                    
                    # Load CSV file for unsupervised
                    detection_metrics = pd.read_csv(file_path)
                    
                    # For unsupervised, we need to filter to best matching clusters
                    if split == 'cal':
                        best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}_cal.pt'
                    else:
                        best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                    
                    if os.path.exists(best_clusters_path):
                        best_clusters_per_concept = torch.load(best_clusters_path, weights_only=False)
                        detection_metrics = filter_unsupervised_detection_metrics(detection_metrics, best_clusters_per_concept)
                else:
                    # Supervised: use original pattern and PT format
                    file_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{file_con_label}.pt'
                    
                    if not os.path.exists(file_path):
                        scores.append(0)
                        continue
                    
                    detection_metrics = torch.load(file_path, weights_only=False)
                
                # Filter to valid concepts
                detection_metrics = detection_metrics[detection_metrics['concept'].isin(gt_samples_per_concept)]
                
                # Calculate weighted average if requested
                if weighted_avg:
                    total = sum(len(gt_samples_per_concept[c]) for c in detection_metrics['concept'])
                    if total > 0:
                        score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) 
                                  for _, row in detection_metrics.iterrows()) / total
                    else:
                        score = 0
                else:
                    score = detection_metrics[metric].mean() if len(detection_metrics) > 0 else 0
                
                scores.append(score)
            
            except FileNotFoundError:
                print(f"  Skipping {label} - file not found: {file_path}")
                scores.append(0)
            except Exception as e:
                print(f"  Error loading {label}: {e}")
                scores.append(0)
        
        # Plot the scores
        if any(s > 0 for s in scores):
            # Convert percentiles to percentage for x-axis
            x_vals = [p * 100 for p in percentiles]
            
            # Get style
            style = style_map.get(label, {'color': 'gray', 'type': 'unknown', 'label': label})
            linestyle = '-'  # Always use solid lines
            
            line = ax.plot(x_vals, scores, color=style['color'], linewidth=2, linestyle=linestyle)[0]
            lines_and_labels.append((line, style['label']))
    
    # Set labels and limits
    if xlabel:
        ax.set_xlabel(xlabel if xlabel != 'default' else "Ground Truth Concept Recall Percentage", fontsize=label_font_size)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_font_size)
    
    ax.set_xlim(0, 100)
    if ylim is None:
        ax.set_ylim(0, 1.05)
    else:
        ax.set_ylim(ylim)
    
    # Add grid
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    # Remove duplicates from lines_and_labels
    unique_lines_labels = []
    seen_labels = set()
    for line, label in lines_and_labels:
        if label not in seen_labels:
            unique_lines_labels.append((line, label))
            seen_labels.add(label)
    
    return [l[0] for l in unique_lines_labels], [l[1] for l in unique_lines_labels]


def plot_detection_scores_multiple_percentthru(dataset_name, split, model_name, sample_types, 
                                              metric='f1', weighted_avg=True, plot_type='both', 
                                              concept_types=None, baseline_types=None, 
                                              percentthrumodels=None, save_dir='../Figs/Paper_Figs/',
                                              title_prefix=None, legend_on_plot=False, ylim=None, 
                                              figsize=None, label_font_size=12, legend_font=None, 
                                              xlabel=None, input_size=None, single_figure=True):
    """
    Plot detection scores for multiple percentthrumodel values.
    
    Args:
        dataset_name: Name of the dataset
        split: Data split to use ('test', 'val', etc.)
        model_name: Model name (e.g., 'CLIP', 'Llama', 'Gemma', 'Qwen')
        sample_types: List of sample types (e.g., ['cls', 'patch'])
        metric: Metric to plot (default: 'f1')
        weighted_avg: Whether to use weighted average (default: True)
        plot_type: 'supervised', 'unsupervised', or 'both' (default: 'both')
        concept_types: List of concept types to include. If None, all available types are plotted
        baseline_types: List of baseline types to include. If None, all available baselines are plotted
        percentthrumodels: List of percentthrumodel values to plot. If None, uses model defaults
        save_dir: Directory to save the plots. If None, plots are displayed but not saved (default: '../Figs/Paper_Figs/')
        title_prefix: Optional prefix for plot titles. If None, uses default titles
        legend_on_plot: If True, includes legend on each plot. If False, creates separate legend figures
        ylim: Optional tuple (ymin, ymax) to set y-axis limits. If None, uses default (0, 1.05)
        figsize: Tuple (width, height) for figure size. If None, calculated based on number of subplots
        label_font_size: Font size for axis labels (default: 12)
        legend_font: Font size for legend. If None, uses label_font_size
        xlabel: Custom x-axis label. If None, uses default
        input_size: Input size for the model (used to determine default percentthrumodels)
        single_figure: If True, creates subplots in a single figure. If False, creates separate plots (default: True)
    """
    # Determine percentthrumodels to use
    if percentthrumodels is None:
        if input_size is None:
            # Try to infer input_size based on model and dataset
            if model_name == 'CLIP':
                input_size = (224, 224)
            elif model_name in ['Llama', 'Gemma', 'Qwen']:
                # Check if it's a text dataset
                text_datasets = ['Stanford-Tree-Bank', 'iSarcasm', 'Sarcasm', 'GoEmotions']
                if dataset_name in text_datasets or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                    input_size = ('text', 'text')
                else:
                    input_size = (560, 560)  # Vision Llama
        
        percentthrumodels = get_model_default_percentthrumodels(model_name, input_size)
    
    print(f"Creating detection score plots for {len(percentthrumodels)} percentthrumodel values: {percentthrumodels}")
    
    if single_figure:
        # Create a single figure with subplots
        n_cols = len(percentthrumodels)
        n_rows = 1
        
        # Calculate figure size if not provided
        if figsize is None:
            subplot_width = 5  # Width per subplot
            subplot_height = 4  # Height per subplot
            figsize = (subplot_width * n_cols, subplot_height * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]  # Make it a list for consistency
        
        # Store lines and labels for legend
        all_lines = []
        all_labels = []
        
        for idx, percentthru in enumerate(percentthrumodels):
            ax = axes[idx]
            
            # Plot on this axis
            try:
                lines, labels = _plot_detection_scores_on_axis(
                    ax=ax,
                    dataset_name=dataset_name,
                    split=split,
                    model_name=model_name,
                    sample_types=sample_types,
                    metric=metric,
                    weighted_avg=weighted_avg,
                    plot_type=plot_type,
                    concept_types=concept_types,
                    baseline_types=baseline_types,
                    percentthrumodel=percentthru,
                    label_font_size=label_font_size,
                    ylim=ylim,
                    xlabel='default' if idx == 0 else None,  # Only show xlabel on first subplot
                    ylabel=metric.capitalize() if idx == 0 else None,  # Only show ylabel on first subplot
                    show_legend=False  # We'll create a unified legend
                )
                
                # Collect lines and labels from the first subplot only (they're the same for all)
                if idx == 0:
                    all_lines = lines
                    all_labels = labels
                
                # Set subplot title
                ax.set_title(f"Layer {percentthru}%", fontsize=label_font_size)
                
            except Exception as e:
                print(f"Error creating subplot for percentthrumodel={percentthru}: {e}")
                ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center', transform=ax.transAxes)
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 1)
        
        # Add overall title
        if title_prefix is not None:
            fig.suptitle(title_prefix, fontsize=label_font_size + 2, fontweight='bold')
        
        plt.tight_layout()
        
        # Add legend
        if not legend_on_plot:
            # Create separate legend figure
            fig_legend = plt.figure(figsize=(4, max(6, len(all_lines) * 0.8)))
            fig_legend.legend(all_lines, all_labels, title="Concept Type", loc='center', 
                            frameon=True, fontsize=legend_font if legend_font else label_font_size,
                            title_fontsize=legend_font if legend_font else label_font_size,
                            labelspacing=2.0, handlelength=3.0, handletextpad=1.0)
        else:
            # Add legend to the main figure
            fig.legend(all_lines, all_labels, title="Concept Type", 
                      loc='center left', bbox_to_anchor=(1.02, 0.5),
                      frameon=True, fontsize=legend_font if legend_font else label_font_size,
                      title_fontsize=legend_font if legend_font else label_font_size)
            plt.subplots_adjust(right=0.85)  # Make room for legend
        
        # Save or display
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            sample_types_str = '_'.join(sample_types)
            concept_types_str = ''
            if concept_types is not None:
                concept_types_str = '_' + '_'.join([ct.replace(' ', '_') for ct in concept_types])
            baseline_types_str = ''
            if baseline_types is not None:
                baseline_types_str = '_' + '_'.join(baseline_types)
            
            save_filename = os.path.join(save_dir, 
                f'{model_name}_{dataset_name}_{sample_types_str}{concept_types_str}{baseline_types_str}_all_percentthru_detectplot.pdf')
            
            fig.savefig(save_filename, dpi=500, format='pdf', bbox_inches='tight')
            print(f"Saved combined plot to: {save_filename}")
            
            # Save legend separately if not on plot
            if not legend_on_plot:
                legend_filename = save_filename.replace('.pdf', '_legend.pdf')
                fig_legend.savefig(legend_filename, dpi=500, format='pdf', bbox_inches='tight')
                print(f"Saved legend to: {legend_filename}")
                plt.close(fig_legend)
            
            plt.close(fig)
            return [save_filename]
        else:
            plt.show()
            if not legend_on_plot and 'fig_legend' in locals():
                plt.show()
            return []
    
    else:
        # Original behavior - create separate plots
        # Create save directory if it doesn't exist and save_dir is provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        # Store all generated plot filenames
        plot_files = []
        
        for percentthru in percentthrumodels:
            # Generate filename for this percentthrumodel
            if save_dir is not None:
                sample_types_str = '_'.join(sample_types)
                concept_types_str = ''
                if concept_types is not None:
                    concept_types_str = '_' + '_'.join([ct.replace(' ', '_') for ct in concept_types])
                baseline_types_str = ''
                if baseline_types is not None:
                    baseline_types_str = '_' + '_'.join(baseline_types)
                
                save_filename = os.path.join(save_dir, 
                    f'{model_name}_{dataset_name}_{sample_types_str}{concept_types_str}{baseline_types_str}_percentthru{percentthru}_detectplot.pdf')
            else:
                save_filename = None
        
            # Generate title for this plot
            if title_prefix is not None:
                title = f"{title_prefix} (Layer {percentthru}%)"
            else:
                title = None  # Let plot_detection_scores use its default
            
            # Call the original plot_detection_scores function
            try:
                plot_detection_scores(
                    dataset_name=dataset_name,
                    split=split,
                    model_name=model_name,
                    sample_types=sample_types,
                    metric=metric,
                    weighted_avg=weighted_avg,
                    plot_type=plot_type,
                    concept_types=concept_types,
                    baseline_types=baseline_types,
                    save_filename=save_filename,
                    title=title,
                    legend_on_plot=legend_on_plot,
                    ylim=ylim,
                    figsize=figsize,
                    percentthrumodel=percentthru,
                    label_font_size=label_font_size,
                    legend_font=legend_font,
                    xlabel=xlabel
                )
                if save_filename is not None:
                    plot_files.append(save_filename)
                    print(f"Created plot for percentthrumodel={percentthru}: {save_filename}")
                else:
                    print(f"Displayed plot for percentthrumodel={percentthru} (not saved)")
            except Exception as e:
                print(f"Error creating plot for percentthrumodel={percentthru}: {e}")
                continue
    
        if save_dir is not None:
            print(f"\nCreated {len(plot_files)} plots in {save_dir}")
        else:
            print(f"\nDisplayed {len(percentthrumodels)} plots (not saved)")
        return plot_files


def plot_detection_scores_per_concept_lines(dataset_name, split, model_name, sample_type,
                                           concepts_to_plot=None, concept_type='avg',
                                           metric='f1', percentthrumodels=None,
                                           figsize=None, save_path=None,
                                           max_cols=5, label_font_size=10, 
                                           input_size=None, ylim=(0, 1),
                                           show_cls=True, show_baselines=True,
                                           baseline_types=['random', 'prompt']):
    """
    Plot detection scores for specific concepts across multiple percentthrumodel values.
    Creates individual line plots for each concept arranged in a grid.
    
    Args:
        dataset_name: Name of the dataset
        split: Data split to use ('test', 'cal', etc.)
        model_name: Model name (e.g., 'CLIP', 'Llama', 'Gemma', 'Qwen')
        sample_type: Single sample type ('cls' or 'patch')
        concepts_to_plot: List of concept names to plot. If None, plots all available concepts
        concept_type: Type of concept ('avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae')
        metric: Metric to plot (default: 'f1')
        percentthrumodels: List of percentthrumodel values. If None, uses model defaults
        figsize: Figure size. If None, calculated based on grid dimensions
        save_path: Path to save the figure. If None, displays only
        max_cols: Maximum number of columns (default: 5)
        label_font_size: Font size for labels (default: 10)
        input_size: Input size for determining default percentthrumodels
        ylim: Y-axis limits tuple (default: (0, 1))
        show_cls: Whether to show CLS performance alongside patch/token (default: True)
        show_baselines: Whether to show baseline performances (default: True)
        baseline_types: List of baseline types to show (default: ['random', 'prompt'])
    """
    # Import at function level to avoid circular imports
    import matplotlib.pyplot as plt
    import numpy as np
    from utils.filter_datasets_utils import filter_concept_dict
    
    # Determine percentthrumodels to use
    if percentthrumodels is None:
        if input_size is None:
            # Try to infer input_size based on model and dataset
            if model_name == 'CLIP':
                input_size = (224, 224)
            elif model_name in ['Llama', 'Gemma', 'Qwen']:
                # Check if it's a text dataset
                text_datasets = ['Stanford-Tree-Bank', 'iSarcasm', 'Sarcasm', 'GoEmotions']
                if dataset_name in text_datasets or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                    input_size = ('text', 'text')
                else:
                    input_size = (560, 560)  # Vision Llama
        
        percentthrumodels = get_model_default_percentthrumodels(model_name, input_size)
    
    # Load ground truth samples
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
        else:
            gt_path = None
    elif model_name == 'CLIP':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt'
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt'
    else:
        gt_path = None
    
    if gt_path and os.path.exists(gt_path):
        gt_samples_per_concept = torch.load(gt_path, weights_only=False)
        gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    else:
        print(f"Warning: GT samples file not found: {gt_path}")
        if concepts_to_plot is None:
            print("Error: Cannot determine concepts without GT samples file")
            return
        gt_samples_per_concept = {c: [1] for c in concepts_to_plot}  # Dummy weights
    
    # If concepts_to_plot is None, use all concepts from GT samples
    if concepts_to_plot is None:
        concepts_to_plot = sorted(list(gt_samples_per_concept.keys()))
        print(f"Plotting all {len(concepts_to_plot)} available concepts")
    
    # Validate concepts exist
    valid_concepts = [c for c in concepts_to_plot if c in gt_samples_per_concept]
    if not valid_concepts:
        print(f"Error: None of the requested concepts found in GT samples")
        return
    
    if len(valid_concepts) < len(concepts_to_plot):
        missing = set(concepts_to_plot) - set(valid_concepts)
        print(f"Warning: Some concepts not found: {missing}")
    
    # Get baseline scores if requested
    baseline_scores = {}
    if show_baselines:
        # Random baseline
        if 'random' in baseline_types:
            random_path = f'Quant_Results/{dataset_name}/random_{model_name}_cls_baseline.csv'
            if os.path.exists(random_path):
                df = pd.read_csv(random_path)
                df = df[df['concept'].isin(gt_samples_per_concept)]
                # Per-concept random scores
                baseline_scores['random'] = {row['concept']: row[metric] for _, row in df.iterrows()}
                # Overall average
                total = sum(len(gt_samples_per_concept[c]) for c in df['concept'])
                if total > 0:
                    baseline_scores['random_avg'] = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) 
                                                       for _, row in df.iterrows()) / total
        
        # Prompt baseline
        if 'prompt' in baseline_types:
            # Try to get per-concept prompt scores first
            per_concept_prompt = get_per_concept_prompt_scores(dataset_name, model_name, metric, split)
            if per_concept_prompt:
                baseline_scores['prompt'] = per_concept_prompt
                # Calculate weighted average
                valid_concepts = [c for c in per_concept_prompt if c in gt_samples_per_concept]
                if valid_concepts:
                    total = sum(len(gt_samples_per_concept[c]) for c in valid_concepts)
                    if total > 0:
                        baseline_scores['prompt_avg'] = sum(per_concept_prompt[c] * len(gt_samples_per_concept[c]) 
                                                          for c in valid_concepts) / total
            else:
                # Fallback to get_weighted_prompt_score if per-concept not available
                prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
                if prompt_score is not None:
                    baseline_scores['prompt_avg'] = prompt_score
    
    # Calculate grid layout (including space for average plot)
    n_concepts = len(valid_concepts)
    n_total_plots = n_concepts + 1  # +1 for average
    n_cols = min(max_cols, n_total_plots)
    n_rows = (n_total_plots + n_cols - 1) // n_cols
    
    # Calculate figure size if not provided
    if figsize is None:
        subplot_width = 4.5 if show_cls else 4  # Increased width for legend
        subplot_height = 3 if show_baselines else 2.5
        figsize = (subplot_width * n_cols, subplot_height * n_rows)
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    # Colors for different lines
    colors = {
        'patch': 'orchid' if sample_type == 'patch' else 'goldenrod',
        'cls': 'goldenrod' if sample_type == 'patch' else 'orchid',
        'random': '#888888',
        'prompt': '#8B4513'
    }
    
    # Determine concept label based on type
    n_clusters = 1000 if sample_type == 'patch' else 50
    n_clusters_cls = 50  # Always 50 for CLS
    
    if concept_type == 'avg':
        con_label_base = f'{model_name}_avg_{sample_type}_embeddings'
        con_label_base_cls = f'{model_name}_avg_cls_embeddings' if show_cls else None
    elif concept_type == 'linsep':
        con_label_base = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False'
        con_label_base_cls = f'{model_name}_linsep_cls_embeddings_BD_True_BN_False' if show_cls else None
    elif concept_type == 'kmeans':
        con_label_base = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans'
        con_label_base_cls = f'{model_name}_kmeans_{n_clusters_cls}_cls_embeddings_kmeans' if show_cls else None
    elif concept_type == 'linsep kmeans':
        con_label_base = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans'
        con_label_base_cls = f'{model_name}_kmeans_{n_clusters_cls}_linsep_cls_embeddings_kmeans' if show_cls else None
    elif concept_type == 'sae':
        con_label_base = f'{model_name}_sae_{sample_type}_dense'
        con_label_base_cls = None  # SAE not available for CLS
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    # Percentiles to evaluate
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    # Store all concept scores for averaging
    all_concept_scores = []
    all_concept_scores_cls = []
    
    # Plot each concept
    for idx, concept in enumerate(valid_concepts):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Store scores for this concept across percentthrumodels
        scores = []
        scores_cls = []
        
        for percentthru in percentthrumodels:
            if 'sae' not in concept_type:
                con_label = f'{con_label_base}_percentthrumodel_{percentthru}'
            else:
                con_label = con_label_base  # SAE doesn't have percentthrumodel in name
            
            # Load best percentile for this concept if available
            best_percentile = None
            if 'sae' not in concept_type:
                best_percentile_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
                if os.path.exists(best_percentile_path):
                    try:
                        best_percentiles_data = torch.load(best_percentile_path, weights_only=False)
                        if concept in best_percentiles_data:
                            best_percentile = best_percentiles_data[concept]['best_percentile']
                    except Exception:
                        pass
            
            # If no best percentile found, use default of 0.1
            if best_percentile is None:
                best_percentile = 0.1
            
            # Get score at best percentile
            score_found = False
            
            # For calibration data, append _cal
            if split == 'cal':
                file_con_label = con_label + '_cal'
            else:
                file_con_label = con_label
            
            # Determine file path based on method type
            if 'kmeans' in concept_type or 'sae' in concept_type:
                # Unsupervised methods
                file_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{best_percentile}_{file_con_label}.csv'
                
                if os.path.exists(file_path):
                    try:
                        # Load CSV file
                        detection_metrics = pd.read_csv(file_path)
                        
                        # For unsupervised, filter to best matching clusters if available
                        if split == 'cal':
                            best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}_cal.pt'
                        else:
                            best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                        
                        if os.path.exists(best_clusters_path):
                            best_clusters_per_concept = torch.load(best_clusters_path, weights_only=False)
                            detection_metrics = detection_metrics[detection_metrics.apply(
                                lambda row: row['cluster'] == best_clusters_per_concept.get(row['concept'], -1),
                                axis=1
                            )]
                        
                        # Extract score for this concept
                        if concept in detection_metrics['concept'].values:
                            concept_data = detection_metrics[detection_metrics['concept'] == concept]
                            if len(concept_data) > 0:
                                score = concept_data.iloc[0][metric]
                                scores.append(score)
                                score_found = True
                    except Exception as e:
                        pass
            else:
                # Supervised methods
                file_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{best_percentile}_{file_con_label}.pt'
                
                if os.path.exists(file_path):
                    try:
                        detection_metrics = torch.load(file_path, weights_only=False)
                        
                        # Extract score for this concept
                        if concept in detection_metrics['concept'].values:
                            concept_data = detection_metrics[detection_metrics['concept'] == concept]
                            if len(concept_data) > 0:
                                score = concept_data.iloc[0][metric]
                                scores.append(score)
                                score_found = True
                    except Exception as e:
                        pass
            
            if not score_found:
                scores.append(np.nan)
            
            # CLS scores if requested
            if show_cls and con_label_base_cls:
                con_label_cls = f'{con_label_base_cls}_percentthrumodel_{percentthru}'
                
                # Load best percentile for CLS if available
                best_percentile_cls = None
                best_percentile_path_cls = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label_cls}.pt'
                if os.path.exists(best_percentile_path_cls):
                    try:
                        best_percentiles_data_cls = torch.load(best_percentile_path_cls, weights_only=False)
                        if concept in best_percentiles_data_cls:
                            best_percentile_cls = best_percentiles_data_cls[concept]['best_percentile']
                    except Exception:
                        pass
                
                # If no best percentile found, use default of 0.1
                if best_percentile_cls is None:
                    best_percentile_cls = 0.1
                
                # Get score at best percentile
                score_found_cls = False
                
                if split == 'cal':
                    file_con_label_cls = con_label_cls + '_cal'
                else:
                    file_con_label_cls = con_label_cls
                
                # Determine file path based on method type
                if 'kmeans' in concept_type:
                    file_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{best_percentile_cls}_{file_con_label_cls}.csv'
                else:
                    file_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{best_percentile_cls}_{file_con_label_cls}.pt'
                
                if os.path.exists(file_path):
                    try:
                        if 'kmeans' in concept_type:
                            detection_metrics = pd.read_csv(file_path)
                            # Filter to best matching clusters for CLS
                            if split == 'cal':
                                best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label_cls}_cal.pt'
                            else:
                                best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label_cls}.pt'
                            
                            if os.path.exists(best_clusters_path):
                                best_clusters_per_concept = torch.load(best_clusters_path, weights_only=False)
                                detection_metrics = detection_metrics[detection_metrics.apply(
                                    lambda row: row['cluster'] == best_clusters_per_concept.get(row['concept'], -1),
                                    axis=1
                                )]
                        else:
                            detection_metrics = torch.load(file_path, weights_only=False)
                        
                        # Extract score for this concept
                        if concept in detection_metrics['concept'].values:
                            concept_data = detection_metrics[detection_metrics['concept'] == concept]
                            if len(concept_data) > 0:
                                score = concept_data.iloc[0][metric]
                                scores_cls.append(score)
                                score_found_cls = True
                    except Exception as e:
                        pass
                
                if not score_found_cls:
                    scores_cls.append(np.nan)
        
        # Store for overall average
        all_concept_scores.append(scores)
        if show_cls:
            all_concept_scores_cls.append(scores_cls)
        
        # Plot line for this concept
        valid_scores = [(p, s) for p, s in zip(percentthrumodels, scores) if not np.isnan(s)]
        if valid_scores:
            x_vals, y_vals = zip(*valid_scores)
            label = 'Token' if sample_type == 'patch' else 'CLS'
            ax.plot(x_vals, y_vals, 'o-', linewidth=2, markersize=6, color=colors['patch'], label=label)
        
        # Plot CLS line if requested
        if show_cls and scores_cls:
            valid_scores_cls = [(p, s) for p, s in zip(percentthrumodels, scores_cls) if not np.isnan(s)]
            if valid_scores_cls:
                x_vals, y_vals = zip(*valid_scores_cls)
                ax.plot(x_vals, y_vals, 's--', linewidth=2, markersize=6, color=colors['cls'], label='CLS', alpha=0.8)
        
        # Plot baselines if requested
        if show_baselines:
            # Random baseline for this concept
            if 'random' in baseline_scores and concept in baseline_scores['random']:
                ax.axhline(baseline_scores['random'][concept], color=colors['random'], 
                          linestyle='-.', linewidth=1.5, label='Random', alpha=0.7)
            
            # Prompt baseline for this concept
            if 'prompt' in baseline_scores and concept in baseline_scores['prompt']:
                ax.axhline(baseline_scores['prompt'][concept], color=colors['prompt'], 
                          linestyle='-.', linewidth=1.5, label='Prompt', alpha=0.7)
        
        # Styling
        ax.set_title(concept, fontsize=label_font_size)
        ax.set_ylabel(metric.capitalize(), fontsize=label_font_size-2)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=label_font_size-2)
        
        # Set x-ticks at every 20, always including 0
        if percentthrumodels:
            min_ptm = 0  # Always start from 0
            max_ptm = max(percentthrumodels)
            xticks = list(range(0, int(max_ptm) + 1, 20))
            if max_ptm not in xticks and max_ptm > xticks[-1]:
                xticks.append(int(max_ptm))
            ax.set_xticks(xticks)
            ax.set_xlim(-5, max_ptm + 5)  # Add some padding
        
        # Only show x-axis labels on bottom row
        if row < n_rows - 1:  # Not bottom row
            ax.set_xlabel('')  # Remove x-label
            ax.tick_params(axis='x', labelbottom=False)  # Hide x-tick labels
        else:  # Bottom row
            ax.set_xlabel('% Through Model', fontsize=label_font_size-2)
        
        # Legend for first subplot only
        if idx == 0:
            ax.legend(fontsize=label_font_size-2, loc='upper left', bbox_to_anchor=(1.05, 1), 
                     borderaxespad=0., frameon=True, fancybox=True, shadow=True)
    
    # Add average plot at the end
    avg_idx = n_concepts
    if avg_idx < n_rows * n_cols:
        row = avg_idx // n_cols
        col = avg_idx % n_cols
        ax = axes[row, col]
        
        # Calculate average across all concepts for each percentthrumodel
        avg_scores = []
        for i in range(len(percentthrumodels)):
            valid_scores = [scores[i] for scores in all_concept_scores if i < len(scores) and not np.isnan(scores[i])]
            if valid_scores:
                avg_scores.append(np.mean(valid_scores))
            else:
                avg_scores.append(np.nan)
        
        # Plot average line
        valid_avg = [(p, s) for p, s in zip(percentthrumodels, avg_scores) if not np.isnan(s)]
        if valid_avg:
            x_vals, y_vals = zip(*valid_avg)
            label = 'Token' if sample_type == 'patch' else 'CLS'
            ax.plot(x_vals, y_vals, 'o-', linewidth=3, markersize=8, color=colors['patch'], label=label)
        
        # CLS average
        if show_cls and all_concept_scores_cls:
            avg_scores_cls = []
            for i in range(len(percentthrumodels)):
                valid_scores = [scores[i] for scores in all_concept_scores_cls if i < len(scores) and not np.isnan(scores[i])]
                if valid_scores:
                    avg_scores_cls.append(np.mean(valid_scores))
                else:
                    avg_scores_cls.append(np.nan)
            
            valid_avg_cls = [(p, s) for p, s in zip(percentthrumodels, avg_scores_cls) if not np.isnan(s)]
            if valid_avg_cls:
                x_vals, y_vals = zip(*valid_avg_cls)
                ax.plot(x_vals, y_vals, 's--', linewidth=3, markersize=8, color=colors['cls'], label='CLS', alpha=0.8)
        
        # Average baselines
        if show_baselines:
            if 'random_avg' in baseline_scores:
                ax.axhline(baseline_scores['random_avg'], color=colors['random'], 
                          linestyle='-.', linewidth=2, label='Random', alpha=0.7)
            if 'prompt_avg' in baseline_scores:
                ax.axhline(baseline_scores['prompt_avg'], color=colors['prompt'], 
                          linestyle='-.', linewidth=2, label='Prompt', alpha=0.7)
        
        # Styling for average plot
        ax.set_title('AVERAGE', fontsize=label_font_size+2, fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontsize=label_font_size-2)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=label_font_size-2)
        
        # Set x-ticks at every 20, always including 0
        if percentthrumodels:
            min_ptm = 0  # Always start from 0
            max_ptm = max(percentthrumodels)
            xticks = list(range(0, int(max_ptm) + 1, 20))
            if max_ptm not in xticks and max_ptm > xticks[-1]:
                xticks.append(int(max_ptm))
            ax.set_xticks(xticks)
            ax.set_xlim(-5, max_ptm + 5)  # Add some padding
        
        # Only show x-axis labels if on bottom row
        if row < n_rows - 1:  # Not bottom row
            ax.set_xlabel('')  # Remove x-label
            ax.tick_params(axis='x', labelbottom=False)  # Hide x-tick labels
        else:  # Bottom row
            ax.set_xlabel('% Through Model', fontsize=label_font_size-2)
            
        ax.legend(fontsize=label_font_size-2, loc='upper left', bbox_to_anchor=(1.05, 1), 
                 borderaxespad=0., frameon=True, fancybox=True, shadow=True)
        
        # Hide remaining empty subplots
        for idx in range(avg_idx + 1, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
    else:
        # Hide all remaining empty subplots
        for idx in range(n_concepts, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
    
    # Overall title
    title_parts = [f'{dataset_name} - {model_name} {concept_type}']
    title_parts.append(f'{metric.capitalize()} Detection Scores by Concept')
    if show_cls:
        title_parts[0] += ' (Token vs CLS)'
    
    fig.suptitle('\n'.join(title_parts), fontsize=label_font_size + 4, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])  # Leave space for legend on right
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def plot_average_detection_scores(dataset_name, split, model_name, sample_type,
                                  concept_type='avg', metric='f1', percentthrumodels=None,
                                  figsize=(10, 6), save_path=None, label_font_size=12,
                                  input_size=None, ylim=(0, 1), show_cls=True,
                                  show_baselines=True, baseline_types=['random', 'prompt']):
    """
    Plot the weighted average detection scores across all concepts for multiple percentthrumodel values.
    Weights are based on concept frequency in test samples.
    
    Args:
        dataset_name: Name of the dataset
        split: Data split to use ('test', 'cal', etc.)
        model_name: Model name (e.g., 'CLIP', 'Llama', 'Gemma', 'Qwen')
        sample_type: Primary sample type ('cls' or 'patch')
        concept_type: Type of concept ('avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae')
        metric: Metric to plot (default: 'f1')
        percentthrumodels: List of percentthrumodel values. If None, uses model defaults
        figsize: Figure size (default: (10, 6))
        save_path: Path to save the figure. If None, displays only
        label_font_size: Font size for labels (default: 12)
        input_size: Input size for determining default percentthrumodels
        ylim: Y-axis limits tuple (default: (0, 1))
        show_cls: Whether to show CLS performance alongside patch/token (default: True)
        show_baselines: Whether to show baseline performances (default: True)
        baseline_types: List of baseline types to show (default: ['random', 'prompt'])
    """
    # Import at function level to avoid circular imports
    import matplotlib.pyplot as plt
    import numpy as np
    from utils.filter_datasets_utils import filter_concept_dict
    
    # Determine percentthrumodels to use
    if percentthrumodels is None:
        if input_size is None:
            # Try to infer input_size based on model and dataset
            if model_name == 'CLIP':
                input_size = (224, 224)
            elif model_name in ['Llama', 'Gemma', 'Qwen']:
                # Check if it's a text dataset
                text_datasets = ['Stanford-Tree-Bank', 'iSarcasm', 'Sarcasm', 'GoEmotions']
                if dataset_name in text_datasets or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                    input_size = ('text', 'text')
                else:
                    input_size = (560, 560)  # Vision Llama
        
        percentthrumodels = get_model_default_percentthrumodels(model_name, input_size)
    
    # Load ground truth samples
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
        else:
            gt_path = None
    elif model_name == 'CLIP':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt'
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt'
    else:
        gt_path = None
    
    if gt_path and os.path.exists(gt_path):
        gt_samples_per_concept = torch.load(gt_path, weights_only=False)
        gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    else:
        print(f"Warning: GT samples file not found: {gt_path}")
        return
    
    # Get all concepts
    all_concepts = sorted(list(gt_samples_per_concept.keys()))
    
    # Calculate concept weights for weighted average
    total_samples = sum(len(gt_samples_per_concept[c]) for c in all_concepts)
    concept_weights = {c: len(gt_samples_per_concept[c]) / total_samples for c in all_concepts}
    
    # Get baseline scores if requested
    baseline_scores = {}
    if show_baselines:
        # Random baseline
        if 'random' in baseline_types:
            random_path = f'Quant_Results/{dataset_name}/random_{model_name}_cls_baseline.csv'
            if os.path.exists(random_path):
                df = pd.read_csv(random_path)
                df = df[df['concept'].isin(gt_samples_per_concept)]
                # Calculate both regular and weighted averages
                baseline_scores['random'] = df[metric].mean()
                baseline_scores['random_weighted'] = sum(
                    row[metric] * concept_weights.get(row['concept'], 0) 
                    for _, row in df.iterrows()
                )
        
        # Prompt baseline
        if 'prompt' in baseline_types:
            prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
            if prompt_score is not None:
                baseline_scores['prompt_weighted'] = prompt_score
                # For regular average, calculate from per-concept scores
                per_concept_prompt = get_per_concept_prompt_scores(dataset_name, model_name, metric, split)
                if per_concept_prompt:
                    valid_concepts = [c for c in per_concept_prompt if c in gt_samples_per_concept]
                    if valid_concepts:
                        baseline_scores['prompt'] = np.mean([per_concept_prompt[c] for c in valid_concepts])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors for different lines - matching plot_detection_scores
    if concept_type == 'avg':
        patch_color = 'orchid'
        cls_color = 'goldenrod'
    elif concept_type == 'linsep':
        patch_color = 'indigo'
        cls_color = 'orangered'
    elif concept_type == 'kmeans':
        patch_color = 'orchid'
        cls_color = 'goldenrod'
    elif concept_type == 'linsep kmeans':
        patch_color = 'indigo'
        cls_color = 'orangered'
    elif concept_type == 'sae':
        patch_color = 'darkgreen'
        cls_color = 'goldenrod'  # SAE not available for CLS, but keep consistent
    else:
        patch_color = 'orchid'
        cls_color = 'goldenrod'
    
    colors = {
        'patch': patch_color if sample_type == 'patch' else cls_color,
        'cls': cls_color if sample_type == 'patch' else patch_color,
        'random': '#888888',
        'prompt': '#8B4513'
    }
    
    # Determine concept label based on type
    n_clusters = 1000 if sample_type == 'patch' else 50
    n_clusters_cls = 50  # Always 50 for CLS
    
    if concept_type == 'avg':
        con_label_base = f'{model_name}_avg_{sample_type}_embeddings'
        con_label_base_cls = f'{model_name}_avg_cls_embeddings' if show_cls else None
    elif concept_type == 'linsep':
        con_label_base = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False'
        con_label_base_cls = f'{model_name}_linsep_cls_embeddings_BD_True_BN_False' if show_cls else None
    elif concept_type == 'kmeans':
        con_label_base = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans'
        con_label_base_cls = f'{model_name}_kmeans_{n_clusters_cls}_cls_embeddings_kmeans' if show_cls else None
    elif concept_type == 'linsep kmeans':
        con_label_base = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans'
        con_label_base_cls = f'{model_name}_kmeans_{n_clusters_cls}_linsep_cls_embeddings_kmeans' if show_cls else None
    elif concept_type == 'sae':
        con_label_base = f'{model_name}_sae_{sample_type}_dense'
        con_label_base_cls = None  # SAE not available for CLS
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    # Collect scores across percentthrumodels
    avg_scores = []
    avg_scores_weighted = []
    avg_scores_cls = []
    avg_scores_cls_weighted = []
    
    for percentthru in percentthrumodels:
        # Primary sample type scores
        concept_scores = {}
        
        if 'sae' not in concept_type:
            con_label = f'{con_label_base}_percentthrumodel_{percentthru}'
        else:
            con_label = con_label_base  # SAE doesn't have percentthrumodel in name
        
        for concept in all_concepts:
            # Load best percentile for this concept if available
            best_percentile = None
            if 'sae' not in concept_type:
                best_percentile_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
                if os.path.exists(best_percentile_path):
                    try:
                        best_percentiles_data = torch.load(best_percentile_path, weights_only=False)
                        if concept in best_percentiles_data:
                            best_percentile = best_percentiles_data[concept]['best_percentile']
                    except Exception:
                        pass
            
            # If no best percentile found, use default of 0.1
            if best_percentile is None:
                best_percentile = 0.1
            
            # Get score at best percentile
            if split == 'cal':
                file_con_label = con_label + '_cal'
            else:
                file_con_label = con_label
            
            # Determine file path based on method type
            if 'kmeans' in concept_type or 'sae' in concept_type:
                file_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{best_percentile}_{file_con_label}.csv'
            else:
                file_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{best_percentile}_{file_con_label}.pt'
            
            if os.path.exists(file_path):
                try:
                    if 'kmeans' in concept_type or 'sae' in concept_type:
                        detection_metrics = pd.read_csv(file_path)
                        # For unsupervised, filter to best matching clusters
                        if 'sae' not in concept_type:
                            if split == 'cal':
                                best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}_cal.pt'
                            else:
                                best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                            
                            if os.path.exists(best_clusters_path):
                                best_clusters_per_concept = torch.load(best_clusters_path, weights_only=False)
                                detection_metrics = detection_metrics[detection_metrics.apply(
                                    lambda row: row['cluster'] == best_clusters_per_concept.get(row['concept'], -1),
                                    axis=1
                                )]
                    else:
                        detection_metrics = torch.load(file_path, weights_only=False)
                    
                    # Extract score for this concept
                    if concept in detection_metrics['concept'].values:
                        concept_data = detection_metrics[detection_metrics['concept'] == concept]
                        if len(concept_data) > 0:
                            concept_scores[concept] = concept_data.iloc[0][metric]
                except Exception:
                    pass
        
        # Calculate averages
        if concept_scores:
            # Regular average
            avg_scores.append(np.mean(list(concept_scores.values())))
            # Weighted average
            weighted_sum = sum(concept_scores.get(c, 0) * concept_weights[c] for c in all_concepts if c in concept_scores)
            weighted_total = sum(concept_weights[c] for c in all_concepts if c in concept_scores)
            avg_scores_weighted.append(weighted_sum / weighted_total if weighted_total > 0 else np.nan)
        else:
            avg_scores.append(np.nan)
            avg_scores_weighted.append(np.nan)
        
        # CLS scores if requested
        if show_cls and con_label_base_cls:
            con_label_cls = f'{con_label_base_cls}_percentthrumodel_{percentthru}'
            concept_scores_cls = {}
            
            for concept in all_concepts:
                # Load best percentile for CLS
                best_percentile_cls = None
                best_percentile_path_cls = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label_cls}.pt'
                if os.path.exists(best_percentile_path_cls):
                    try:
                        best_percentiles_data_cls = torch.load(best_percentile_path_cls, weights_only=False)
                        if concept in best_percentiles_data_cls:
                            best_percentile_cls = best_percentiles_data_cls[concept]['best_percentile']
                    except Exception:
                        pass
                
                # If no best percentile found, use default of 0.1
                if best_percentile_cls is None:
                    best_percentile_cls = 0.1
                
                # Get score at best percentile
                if split == 'cal':
                    file_con_label_cls = con_label_cls + '_cal'
                else:
                    file_con_label_cls = con_label_cls
                
                # Determine file path based on method type
                if 'kmeans' in concept_type:
                    file_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{best_percentile_cls}_{file_con_label_cls}.csv'
                else:
                    file_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{best_percentile_cls}_{file_con_label_cls}.pt'
                
                if os.path.exists(file_path):
                    try:
                        if 'kmeans' in concept_type:
                            detection_metrics = pd.read_csv(file_path)
                            # Filter to best matching clusters for CLS
                            if split == 'cal':
                                best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label_cls}_cal.pt'
                            else:
                                best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label_cls}.pt'
                            
                            if os.path.exists(best_clusters_path):
                                best_clusters_per_concept = torch.load(best_clusters_path, weights_only=False)
                                detection_metrics = detection_metrics[detection_metrics.apply(
                                    lambda row: row['cluster'] == best_clusters_per_concept.get(row['concept'], -1),
                                    axis=1
                                )]
                        else:
                            detection_metrics = torch.load(file_path, weights_only=False)
                        
                        # Extract score for this concept
                        if concept in detection_metrics['concept'].values:
                            concept_data = detection_metrics[detection_metrics['concept'] == concept]
                            if len(concept_data) > 0:
                                concept_scores_cls[concept] = concept_data.iloc[0][metric]
                    except Exception:
                        pass
            
            # Calculate CLS averages
            if concept_scores_cls:
                # Regular average
                avg_scores_cls.append(np.mean(list(concept_scores_cls.values())))
                # Weighted average
                weighted_sum = sum(concept_scores_cls.get(c, 0) * concept_weights[c] for c in all_concepts if c in concept_scores_cls)
                weighted_total = sum(concept_weights[c] for c in all_concepts if c in concept_scores_cls)
                avg_scores_cls_weighted.append(weighted_sum / weighted_total if weighted_total > 0 else np.nan)
            else:
                avg_scores_cls.append(np.nan)
                avg_scores_cls_weighted.append(np.nan)
    
    # Plot primary sample type - weighted average only
    label = 'Token' if sample_type == 'patch' else 'CLS'
    
    # Weighted average
    valid_avg_weighted = [(p, s) for p, s in zip(percentthrumodels, avg_scores_weighted) if not np.isnan(s)]
    if valid_avg_weighted:
        x_vals, y_vals = zip(*valid_avg_weighted)
        ax.plot(x_vals, y_vals, 'o-', linewidth=2.5, markersize=8,
                color=colors['patch'], label=label)
    
    # Plot CLS if requested - weighted average only
    if show_cls and avg_scores_cls_weighted:
        valid_avg_cls_weighted = [(p, s) for p, s in zip(percentthrumodels, avg_scores_cls_weighted) if not np.isnan(s)]
        if valid_avg_cls_weighted:
            x_vals, y_vals = zip(*valid_avg_cls_weighted)
            ax.plot(x_vals, y_vals, 's--', linewidth=2.5, markersize=8,
                    color=colors['cls'], label='CLS')
    
    # Plot baselines if requested - weighted versions only
    if show_baselines:
        if 'random_weighted' in baseline_scores:
            ax.axhline(baseline_scores['random_weighted'], color=colors['random'], 
                      linestyle='-.', linewidth=2, label='Random', alpha=0.7)
        
        if 'prompt_weighted' in baseline_scores:
            ax.axhline(baseline_scores['prompt_weighted'], color=colors['prompt'], 
                      linestyle='-.', linewidth=2, label='Prompt', alpha=0.7)
    
    # Styling
    ax.set_xlabel('Layer %', fontsize=label_font_size)
    ax.set_ylabel(metric.capitalize(), fontsize=label_font_size)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=label_font_size-2)
    ax.legend(fontsize=label_font_size-2, loc='best')
    
    # Title
    title_parts = [f'{dataset_name} - {model_name} {concept_type}']
    title_parts.append(f'Weighted Average {metric.capitalize()} Detection Scores')
    if show_cls:
        title_parts[0] += ' (Token vs CLS)'
    
    ax.set_title('\n'.join(title_parts), fontsize=label_font_size + 4, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def plot_average_detection_scores_heatmap(dataset_names, split, model_names, sample_types,
                                         concept_types=['linsep', 'avg'], metric='f1', 
                                         percentthrumodels=None, figsize=None, save_path=None, 
                                         label_font_size=12, input_size=None, cmap='viridis',
                                         show_values=True, vmin=0, vmax=1, show_baselines=True,
                                         baseline_types=['random', 'prompt'], separate_datasets=True,
                                         number_size=9, model_size=14, dataset_size=9, font_size=9,
                                         save_file=None, highlight_max_per_row=False):
    """
    Plot weighted average detection scores as a heatmap for multiple datasets, models, and concept types.
    Creates separate subplots for each dataset by default.
    
    Args:
        dataset_names: Single dataset name or list of dataset names
        split: Data split to use ('test', 'cal', etc.)
        model_names: Single model name or list of model names
        sample_types: Single sample type or list of sample types ('cls' or 'patch')
        concept_types: List of concept types to plot (default: ['linsep', 'avg'])
        metric: Metric to plot (default: 'f1')
        percentthrumodels: List of percentthrumodel values. If None, uses model defaults
        figsize: Figure size. If None, automatically determined based on number of datasets
        save_path: Path to save the figure. If None, displays only
        label_font_size: Font size for labels (default: 12)
        input_size: Input size for determining default percentthrumodels
        cmap: Colormap for heatmap (default: 'viridis')
        show_values: Whether to show values in cells (default: True)
        vmin: Minimum value for colormap (default: 0)
        vmax: Maximum value for colormap (default: 1)
        show_baselines: Whether to include baseline rows (default: True)
        baseline_types: List of baseline types to show (default: ['random', 'prompt'])
                       Options: 'random', 'prompt', 'maxtoken', 'meantoken', 'lasttoken', 'randomtoken'
        separate_datasets: Whether to create separate subplots for each dataset (default: True)
        number_size: Font size for the numbers in heatmap cells (default: 9)
        model_size: Font size for model names at the top (default: 14)
        dataset_size: Font size for dataset names on y-axis (default: 9)
        font_size: Font size for all other text elements (default: 9)
        save_file: Alias for save_path, path to save the figure (default: None)
        highlight_max_per_row: Whether to highlight and print the highest value in each row (default: False)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from utils.filter_datasets_utils import filter_concept_dict
    from utils.general_utils import apply_paper_plotting_style
    import pandas as pd
    import seaborn as sns
    
    # Apply paper plotting style
    apply_paper_plotting_style()
    # Override with custom font sizes
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size
    })
    
    # Ensure inputs are lists
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    if isinstance(model_names, str):
        model_names = [model_names]
    if isinstance(sample_types, str):
        sample_types = [sample_types]
    if isinstance(concept_types, str):
        concept_types = [concept_types]
    
    # Determine figure size and layout
    n_datasets = len(dataset_names)
    if figsize is None:
        if separate_datasets and n_datasets > 1:
            # Calculate figure size based on number of datasets
            fig_width = 14
            fig_height = 5 * n_datasets + 1
            figsize = (fig_width, fig_height)
        else:
            figsize = (12, 8)
    
    # First pass: count max models per dataset
    max_models_per_dataset = 0
    for dataset_name in dataset_names:
        model_count = 0
        for model_name in model_names:
            text_datasets = ['Stanford-Tree-Bank', 'iSarcasm', 'Sarcasm', 'GoEmotions']
            is_text_dataset = dataset_name in text_datasets or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name
            if model_name == 'CLIP' and is_text_dataset:
                continue
            model_count += 1
        max_models_per_dataset = max(max_models_per_dataset, model_count)
    
    # Create figure with subplots grid: rows for datasets, columns for models
    if figsize is None:
        fig_width = 6 * max_models_per_dataset + 2
        fig_height = 4 * n_datasets + 1
        figsize = (fig_width, fig_height)
    
    if separate_datasets:
        fig = plt.figure(figsize=figsize)
        
        # First, we need to determine the number of columns for each model to set proper width ratios
        # This requires a preliminary pass through the data
        model_column_counts = {}
        for dataset_name in dataset_names:
            for model_name in model_names:
                # Skip invalid combinations
                text_datasets = ['Stanford-Tree-Bank', 'iSarcasm', 'Sarcasm', 'GoEmotions']
                is_text_dataset = dataset_name in text_datasets or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name
                if model_name == 'CLIP' and is_text_dataset:
                    continue
                
                # Determine model key
                if model_name == 'Llama':
                    model_key = 'Llama-Text' if is_text_dataset else 'Llama-Vision'
                else:
                    model_key = model_name
                
                # Count columns based on percentthrumodels
                if percentthrumodels is None:
                    if input_size is None:
                        if model_name == 'CLIP':
                            input_size_temp = (224, 224)
                        elif model_name in ['Llama', 'Gemma', 'Qwen']:
                            input_size_temp = ('text', 'text') if is_text_dataset else (560, 560)
                    else:
                        input_size_temp = input_size
                    ptm_count = len(get_model_default_percentthrumodels(model_name, input_size_temp))
                else:
                    ptm_count = len(percentthrumodels)
                
                # Store the column count for this model
                if model_key not in model_column_counts:
                    model_column_counts[model_key] = ptm_count * len(concept_types)
        
        # Calculate width ratios based on column counts
        if max_models_per_dataset == 2:
            model_keys = list(model_column_counts.keys())
            if len(model_keys) >= 2:
                # Ratio of columns determines ratio of widths for square cells
                ratio1 = model_column_counts[model_keys[0]]
                ratio2 = model_column_counts[model_keys[1]]
                # Normalize so the smaller one is 1
                min_ratio = min(ratio1, ratio2)
                ratio1 = ratio1 / min_ratio
                ratio2 = ratio2 / min_ratio
                
                # For 2 models: [model1, space, model2, space, colorbar]
                gs = fig.add_gridspec(n_datasets, 5,
                                      hspace=0.15,
                                      wspace=0.0,
                                      left=0.08, right=0.98,
                                      top=0.93, bottom=0.1,
                                      width_ratios=[ratio1, 0.3, ratio2, 0.01, 0.06])
            else:
                # Fallback if we couldn't determine ratios
                gs = fig.add_gridspec(n_datasets, 5,
                                      hspace=0.15,
                                      wspace=0.0,
                                      left=0.08, right=0.98,
                                      top=0.93, bottom=0.1,
                                      width_ratios=[1, 0.3, 1, 0.01, 0.06])
        else:
            # For more than 2 models, use equal widths for now
            width_ratios = [1] * max_models_per_dataset + [0.06]
            gs = fig.add_gridspec(n_datasets, max_models_per_dataset + 1,
                                  hspace=0.15, wspace=0.01,
                                  left=0.08, right=0.98,
                                  top=0.93, bottom=0.1,
                                  width_ratios=width_ratios)
    
    # Process each dataset
    for dataset_idx, dataset_name in enumerate(dataset_names):
        # Variables to track min/max for entire row
        row_vmin = float('inf')
        row_vmax = float('-inf')
        # Store model data for this dataset
        model_data = {}  # Store data for each model separately
        model_col_idx = 0  # Track column position for this dataset
        
        # Collect data for each model separately
        for model_name in model_names:
            # Skip invalid model-dataset combinations
            text_datasets = ['Stanford-Tree-Bank', 'iSarcasm', 'Sarcasm', 'GoEmotions']
            is_text_dataset = dataset_name in text_datasets or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name
            if model_name == 'CLIP' and is_text_dataset:
                continue  # CLIP doesn't work with text datasets
            
            # Determine percentthrumodels for this model
            if percentthrumodels is None:
                if input_size is None:
                    # Try to infer input_size based on model and dataset
                    if model_name == 'CLIP':
                        input_size = (224, 224)
                    elif model_name in ['Llama', 'Gemma', 'Qwen']:
                        if is_text_dataset:
                            input_size = ('text', 'text')
                        else:
                            input_size = (560, 560)  # Vision Llama
                
                current_percentthrumodels = get_model_default_percentthrumodels(model_name, input_size)
            else:
                current_percentthrumodels = percentthrumodels
            
            # Create unique model key to separate Llama vision from text
            if model_name == 'Llama':
                if is_text_dataset:
                    model_key = 'Llama-Text'
                else:
                    model_key = 'Llama-Vision'
            else:
                model_key = model_name
            
            # Store data for this model
            model_data[model_key] = {
                'percentthrumodels': current_percentthrumodels,
                'rows': [],
                'row_labels': []
            }
            
            # Load ground truth samples for this dataset
            if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                if model_name == 'Llama':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
                elif model_name == 'Gemma':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
                elif model_name == 'Qwen':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
                else:
                    gt_path = None
            elif model_name == 'CLIP':
                gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt'
            elif model_name == 'Llama':
                gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt'
            else:
                gt_path = None
            
            if gt_path and os.path.exists(gt_path):
                gt_samples_per_concept = torch.load(gt_path, weights_only=False)
                gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
            else:
                print(f"Warning: GT samples file not found: {gt_path}")
                continue
            
            # Calculate concept weights
            all_concepts = sorted(list(gt_samples_per_concept.keys()))
            total_samples = sum(len(gt_samples_per_concept[c]) for c in all_concepts)
            concept_weights = {c: len(gt_samples_per_concept[c]) / total_samples for c in all_concepts}
            
            # Define detection types that need concept types
            concept_detection_types = [
                ('patch', 'regular'),   # SuperTok
                ('patch', 'maxtoken'),  # MaxTok
                ('cls', 'regular'),     # CLS
                ('patch', 'meantoken'), # MeanTok
                ('patch', 'lasttoken'), # LastTok
                ('patch', 'randomtoken'), # RandTok
            ]
            
            # Process concept-based detection types first
            for concept_type in concept_types:
                for sample_type, detection_variant in concept_detection_types:
                    # Skip invalid combinations
                    if concept_type == 'sae' and sample_type == 'cls':
                        continue
                    
                    # Skip baseline types if not enabled
                    if detection_variant != 'regular' and not show_baselines:
                        continue
                    
                    # Skip specific baselines if not in baseline_types
                    if detection_variant == 'prompt' and 'prompt' not in baseline_types:
                        continue
                    if detection_variant in ['maxtoken', 'meantoken', 'lasttoken', 'randomtoken'] and detection_variant not in baseline_types:
                        continue
                        
                    # Handle regular detection types
                    if detection_variant == 'regular':
                        # Skip if sample_type not in requested sample_types
                        if sample_type not in sample_types:
                            continue
                            
                        # Determine concept label based on type
                        n_clusters = 1000 if sample_type == 'patch' else 50
                        
                        if concept_type == 'avg':
                            con_label_base = f'{model_name}_avg_{sample_type}_embeddings'
                        elif concept_type == 'linsep':
                            con_label_base = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False'
                        elif concept_type == 'kmeans':
                            con_label_base = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans'
                        elif concept_type == 'linsep kmeans':
                            con_label_base = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans'
                        elif concept_type == 'sae':
                            con_label_base = f'{model_name}_sae_{sample_type}_dense'
                        else:
                            continue
                        
                        # Collect scores across this model's percentthrumodels
                        row_scores = []
                        
                        for percentthru in current_percentthrumodels:
                            concept_scores = {}
                            
                            if 'sae' not in concept_type:
                                con_label = f'{con_label_base}_percentthrumodel_{percentthru}'
                            else:
                                con_label = con_label_base
                            
                            # Get scores for all concepts
                            for concept in all_concepts:
                                # Load best percentile
                                best_percentile = None
                                if 'sae' not in concept_type:
                                    best_percentile_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
                                    if os.path.exists(best_percentile_path):
                                        try:
                                            best_percentiles_data = torch.load(best_percentile_path, weights_only=False)
                                            if concept in best_percentiles_data:
                                                best_percentile = best_percentiles_data[concept]['best_percentile']
                                        except Exception:
                                            pass
                                
                                if best_percentile is None:
                                    best_percentile = 0.1
                                
                                # Get score
                                if split == 'cal':
                                    file_con_label = con_label + '_cal'
                                else:
                                    file_con_label = con_label
                                
                                # Determine file path
                                if 'kmeans' in concept_type or 'sae' in concept_type:
                                    file_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{best_percentile}_{file_con_label}.csv'
                                else:
                                    file_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{best_percentile}_{file_con_label}.pt'
                                
                                if os.path.exists(file_path):
                                    try:
                                        if 'kmeans' in concept_type or 'sae' in concept_type:
                                            detection_metrics = pd.read_csv(file_path)
                                            # Filter to best matching clusters
                                            if 'kmeans' in concept_type and split == 'cal':
                                                best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}_cal.pt'
                                            elif 'kmeans' in concept_type:
                                                best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                                            else:  # SAE
                                                best_clusters_path = f'Unsupervised_Matches/{dataset_name}/SAE_bestdetects_{con_label}.pt'
                                            
                                            if os.path.exists(best_clusters_path):
                                                best_clusters_per_concept = torch.load(best_clusters_path, weights_only=False)
                                                detection_metrics = detection_metrics[detection_metrics.apply(
                                                    lambda row: row['cluster'] == best_clusters_per_concept.get(row['concept'], -1),
                                                    axis=1
                                                )]
                                        else:
                                            detection_metrics = torch.load(file_path, weights_only=False)
                                        
                                        # Extract score
                                        if concept in detection_metrics['concept'].values:
                                            concept_data = detection_metrics[detection_metrics['concept'] == concept]
                                            if len(concept_data) > 0:
                                                concept_scores[concept] = concept_data.iloc[0][metric]
                                    except Exception:
                                        pass
                            
                            # Calculate weighted average
                            if concept_scores:
                                weighted_sum = sum(concept_scores.get(c, 0) * concept_weights[c] 
                                                 for c in all_concepts if c in concept_scores)
                                weighted_total = sum(concept_weights[c] for c in all_concepts if c in concept_scores)
                                row_scores.append(weighted_sum / weighted_total if weighted_total > 0 else np.nan)
                            else:
                                row_scores.append(np.nan)
                        
                        # Add row to model's data
                        model_data[model_key]['rows'].append(row_scores)
                        
                        # Create row label with sample type first
                        sample_label = 'SuperTok' if sample_type == 'patch' else 'CLS'
                        # Capitalize concept type (title case)
                        if concept_type == 'avg':
                            concept_type_display = 'Avg'
                        elif concept_type == 'linsep':
                            concept_type_display = 'Linsep'
                        elif concept_type == 'kmeans':
                            concept_type_display = 'Kmeans'
                        elif concept_type == 'linsep kmeans':
                            concept_type_display = 'Linsep Kmeans'
                        elif concept_type == 'sae':
                            concept_type_display = 'Sae'
                        else:
                            concept_type_display = concept_type.title().replace('_', ' ')
                        
                        # Put detection type first in the label
                        if separate_datasets and n_datasets > 1:
                            model_data[model_key]['row_labels'].append(f'{sample_label} {concept_type_display}')
                        else:
                            dataset_short = dataset_name.replace('Stanford-Tree-Bank', 'STB').replace('GoEmotions', 'GoEmo')
                            model_data[model_key]['row_labels'].append(f'{dataset_short} {sample_label} {concept_type_display}')
                    
                    
                    # Handle token baselines
                    elif detection_variant in ['maxtoken', 'meantoken', 'lasttoken', 'randomtoken']:
                        # Token baselines only work with patch sample type
                        if 'patch' not in sample_types:
                            continue
                        
                        baseline_name = detection_variant
                        baseline_label_mapping = {
                            'maxtoken': 'MaxTok',
                            'meantoken': 'MeanTok',
                            'lasttoken': 'LastTok',
                            'randomtoken': 'RandTok'
                        }
                        baseline_label = baseline_label_mapping[baseline_name]
                        
                        if concept_type == 'sae':
                            continue  # Token baselines don't apply to SAE
                        
                        # Collect scores across percentthrumodels for this concept type
                        baseline_row_scores = []
                        baseline_scores_found = False
                        
                        for percentthru in current_percentthrumodels:
                            # Build concept label
                            if concept_type == 'avg':
                                con_label = f'{model_name}_avg_patch_embeddings_percentthrumodel_{percentthru}'
                            elif concept_type == 'linsep':
                                con_label = f'{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percentthru}'
                            elif concept_type == 'kmeans':
                                con_label = f'{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_{percentthru}'
                            elif concept_type == 'linsep kmeans':
                                con_label = f'{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_{percentthru}'
                            else:
                                continue
                            
                            # Check for best percentiles file
                            best_percentiles_path = f'Quant_Results/{dataset_name}/{baseline_name}_best_percentiles_{con_label}.pt'
                            if not os.path.exists(best_percentiles_path):
                                baseline_row_scores.append(np.nan)
                                continue
                            
                            # Load best percentiles
                            best_percentiles_data = torch.load(best_percentiles_path, weights_only=False)
                            best_percentiles = best_percentiles_data['best_percentiles']
                            
                            # Collect scores for this concept type
                            concept_scores = {}
                            
                            for concept in all_concepts:
                                if concept not in best_percentiles:
                                    continue
                                
                                percentile = best_percentiles[concept]
                                
                                # Try to load detection metrics
                                if 'kmeans' in concept_type:
                                    # Unsupervised methods
                                    detection_path = f'Quant_Results/{dataset_name}/detectionmetrics_{baseline_name}_allpairs_per_{percentile}_{con_label}.csv'
                                    best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{baseline_name}_{con_label}.pt'
                                    
                                    if os.path.exists(detection_path) and os.path.exists(best_clusters_path):
                                        try:
                                            detection_metrics = pd.read_csv(detection_path)
                                            best_clusters = torch.load(best_clusters_path, weights_only=False)
                                            
                                            if concept in best_clusters:
                                                cluster_id = best_clusters[concept]['best_cluster']
                                                row = detection_metrics[detection_metrics['concept'] == f"('{concept}', '{cluster_id}')"]
                                                if not row.empty:
                                                    concept_scores[concept] = row.iloc[0][metric]
                                                    baseline_scores_found = True
                                        except:
                                            pass
                                else:
                                    # Supervised methods
                                    detection_path = f'Quant_Results/{dataset_name}/detectfirst_{baseline_name}_test_{con_label}.csv'
                                    if not os.path.exists(detection_path):
                                        detection_path = f'Quant_Results/{dataset_name}/detectfirst_{baseline_name}_{con_label}.csv'
                                    
                                    if os.path.exists(detection_path):
                                        try:
                                            detection_metrics = pd.read_csv(detection_path)
                                            concept_row = detection_metrics[(detection_metrics['concept'] == concept) & 
                                                                          (detection_metrics['percentile'] == percentile)]
                                            if not concept_row.empty:
                                                concept_scores[concept] = concept_row.iloc[0][metric]
                                                baseline_scores_found = True
                                        except:
                                            pass
                            
                            # Calculate weighted average for this percentthrumodel
                            if concept_scores:
                                weighted_sum = sum(concept_scores.get(c, 0) * concept_weights[c] 
                                                 for c in all_concepts if c in concept_scores)
                                weighted_total = sum(concept_weights[c] for c in all_concepts if c in concept_scores)
                                baseline_row_scores.append(weighted_sum / weighted_total if weighted_total > 0 else np.nan)
                            else:
                                baseline_row_scores.append(np.nan)
                        
                        # Add baseline row for this concept type if we found any scores
                        if baseline_scores_found and baseline_row_scores:
                            model_data[model_key]['rows'].append(baseline_row_scores)
                            
                            # Create label with concept type
                            # Capitalize concept type (title case)
                            if concept_type == 'avg':
                                concept_type_display = 'Avg'
                            elif concept_type == 'linsep':
                                concept_type_display = 'Linsep'
                            elif concept_type == 'kmeans':
                                concept_type_display = 'Kmeans'
                            elif concept_type == 'linsep kmeans':
                                concept_type_display = 'Linsep Kmeans'
                            else:
                                concept_type_display = concept_type.title().replace('_', ' ')
                            
                            if separate_datasets and n_datasets > 1:
                                model_data[model_key]['row_labels'].append(f'{baseline_label} {concept_type_display}')
                            else:
                                dataset_short = dataset_name.replace('Stanford-Tree-Bank', 'STB').replace('GoEmotions', 'GoEmo')
                                model_data[model_key]['row_labels'].append(f'{dataset_short} {baseline_label} {concept_type_display}')
            
            # Add prompt and random baselines at the end (not concept-specific)
            if show_baselines:
                # Prompt baseline
                if 'prompt' in baseline_types:
                    prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
                    if prompt_score is not None:
                        # Create row with same score for all percentthrumodels
                        model_data[model_key]['rows'].append([prompt_score] * len(current_percentthrumodels))
                        if separate_datasets and n_datasets > 1:
                            model_data[model_key]['row_labels'].append('Prompt')
                        else:
                            dataset_short = dataset_name.replace('Stanford-Tree-Bank', 'STB').replace('GoEmotions', 'GoEmo')
                            model_data[model_key]['row_labels'].append(f'{dataset_short} Prompt')
                
                # Random baseline
                if 'random' in baseline_types:
                    random_path = f'Quant_Results/{dataset_name}/random_{model_name}_cls_baseline.csv'
                    if os.path.exists(random_path):
                        df = pd.read_csv(random_path)
                        df = df[df['concept'].isin(gt_samples_per_concept)]
                        # Calculate weighted average
                        random_score = sum(
                            row[metric] * concept_weights.get(row['concept'], 0) 
                            for _, row in df.iterrows()
                        )
                        # Create row with same score for all percentthrumodels
                        model_data[model_key]['rows'].append([random_score] * len(current_percentthrumodels))
                        if separate_datasets and n_datasets > 1:
                            model_data[model_key]['row_labels'].append('Random')
                        else:
                            dataset_short = dataset_name.replace('Stanford-Tree-Bank', 'STB').replace('GoEmotions', 'GoEmo')
                            model_data[model_key]['row_labels'].append(f'{dataset_short} Random')
        
        # First pass: calculate min/max for entire row
        if len(model_data) > 0:
            for model_key, mdata in model_data.items():
                if len(mdata['rows']) > 0:
                    heatmap_data = np.array(mdata['rows'])
                    valid_data = heatmap_data[~np.isnan(heatmap_data)]
                    if len(valid_data) > 0:
                        row_vmin = min(row_vmin, np.min(valid_data))
                        row_vmax = max(row_vmax, np.max(valid_data))
            
            # If no valid data found, use defaults
            if row_vmin == float('inf'):
                row_vmin = vmin
                row_vmax = vmax
            
            # Create custom colormap that makes NaN values gray
            import matplotlib.colors as mcolors
            cmap_custom = plt.cm.get_cmap(cmap).copy()
            cmap_custom.set_bad(color='lightgray')
            
            # Calculate maximum number of data columns and rows across all models
            max_data_cols = 0
            num_rows_per_model = []
            for mdata in model_data.values():
                if len(mdata['rows']) > 0:
                    max_data_cols = max(max_data_cols, len(mdata['rows'][0]))
                    num_rows_per_model.append(len(mdata['rows']))
            
            # Check if all models have the same number of rows
            all_same_rows = len(set(num_rows_per_model)) <= 1
            
            # Second pass: create heatmaps with shared scale
            model_col_idx = 0
            last_heatmap = None  # Track last heatmap for colorbar
            
            for model_key, mdata in model_data.items():
                if len(mdata['rows']) == 0:
                    continue
                
                # Create subplot for this model
                if max_models_per_dataset == 2:
                    # Skip spacing columns: use columns 0 and 2 for models
                    actual_col = 0 if model_col_idx == 0 else 2
                    ax = fig.add_subplot(gs[dataset_idx, actual_col])
                else:
                    ax = fig.add_subplot(gs[dataset_idx, model_col_idx])
                
                # Prepare data for this model's heatmap
                heatmap_data = np.array(mdata['rows'])
                x_labels = [f'{ptm}%' for ptm in mdata['percentthrumodels']]
                y_labels = mdata['row_labels']
                
                # Create heatmap with row-wide scale
                # Don't show numbers if number_size is 0
                show_numbers = show_values and number_size > 0
                last_heatmap = sns.heatmap(heatmap_data, 
                            xticklabels=x_labels,
                            yticklabels=y_labels,
                            cmap=cmap_custom, 
                            vmin=row_vmin,  # Use row-wide min
                            vmax=row_vmax,  # Use row-wide max
                            annot=show_numbers,
                            fmt='.2f',
                            cbar=False,  # No individual colorbar
                            ax=ax,
                            annot_kws={'fontsize': number_size, 'color': 'black'},
                            mask=np.isnan(heatmap_data))
                
                # Highlight maximum value per row if requested
                if highlight_max_per_row:
                    for row_idx in range(heatmap_data.shape[0]):
                        row_data = heatmap_data[row_idx]
                        # Skip rows that are all NaN
                        if not np.all(np.isnan(row_data)):
                            # Find the maximum value in the row
                            max_col_idx = np.nanargmax(row_data)
                            max_value = row_data[max_col_idx]
                            
                            # Add a border around the maximum cell
                            # Rectangle coordinates are (x, y) from bottom-left
                            rect = plt.Rectangle((max_col_idx, row_idx), 1, 1, 
                                               fill=False, edgecolor='red', linewidth=3)
                            ax.add_patch(rect)
                            
                            # Add text annotation with the maximum value
                            # Position text at the center of the cell
                            text_x = max_col_idx + 0.5
                            text_y = row_idx + 0.5
                            
                            # Format the value with 3 decimal places
                            text_value = f'{max_value:.3f}'
                            
                            # Add white background to make text more visible
                            ax.text(text_x, text_y, text_value, 
                                   ha='center', va='center',
                                   fontsize=number_size + 2,  # Slightly larger than regular numbers
                                   fontweight='bold',
                                   color='red',
                                   bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor='white', 
                                           edgecolor='red',
                                           alpha=0.9))
                
                # Styling for this subplot
                if dataset_idx == n_datasets - 1:  # Only label x-axis on bottom row
                    ax.set_xlabel('% Through Model', fontsize=font_size, color='black', labelpad=2)
                
                # Add dataset name as y-label only on leftmost heatmap
                if model_col_idx == 0:
                    # Clean up dataset name
                    display_name = dataset_name
                    if display_name.startswith('Broden-'):
                        display_name = display_name.replace('Broden-', '')
                    if display_name.lower() == 'coco':
                        display_name = 'COCO'
                    ax.set_ylabel(display_name, fontsize=dataset_size, color='black', fontweight='bold')
                
                ax.tick_params(axis='both', labelsize=font_size, colors='black')
                ax.tick_params(axis='y', rotation=0, colors='black')
                ax.tick_params(axis='x', rotation=45, colors='black', pad=2)
                
                # Make the heatmap cells square
                ax.set_aspect('equal')
                
                model_col_idx += 1
            
            # Add colorbar at the far right
            if last_heatmap is not None:
                # Get the position of the last heatmap axes to match colorbar height
                last_ax_pos = ax.get_position()
                
                # Create colorbar axes with matched height
                if max_models_per_dataset == 2:
                    # Colorbar goes in column 4 (after spacing)
                    cbar_ax = fig.add_subplot(gs[dataset_idx, 4])
                else:
                    cbar_ax = fig.add_subplot(gs[dataset_idx, max_models_per_dataset])
                cbar_ax_pos = cbar_ax.get_position()
                
                # Adjust colorbar height to match heatmap height and move it closer
                # Move colorbar left by reducing x0 position
                x_shift = -0.015  # Move left to get much closer to plot
                new_cbar_pos = [cbar_ax_pos.x0 + x_shift, last_ax_pos.y0, 
                               cbar_ax_pos.width, last_ax_pos.height]
                cbar_ax.set_position(new_cbar_pos)
                
                cbar = plt.colorbar(last_heatmap.collections[0], cax=cbar_ax)
                cbar.ax.tick_params(colors='black', labelsize=font_size)
                metric_label = metric.upper() if metric in ['f1', 'iou'] else metric.title()
                cbar.set_label(f'Detection {metric_label}', color='black', fontsize=font_size)
                # Format colorbar tick labels to 1 decimal place
                cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # Add model names as big titles at the top
    if separate_datasets:
        # Store axes for each column to get their positions
        column_axes = {}
        
        # Find the axes for the first row to determine column positions
        for ax in fig.axes:
            if hasattr(ax, 'get_gridspec'):
                try:
                    # Get subplot spec to find position
                    ss = ax.get_subplotspec()
                    if ss is not None:
                        row, col = ss.rowspan.start, ss.colspan.start
                        if row == 0:  # First row
                            if max_models_per_dataset == 2:
                                # Map grid columns to model indices
                                if col == 0:
                                    column_axes[0] = ax  # First model
                                elif col == 2:
                                    column_axes[1] = ax  # Second model
                            elif col < max_models_per_dataset:
                                column_axes[col] = ax
                except:
                    pass
        
        # Get unique model keys that were actually used
        all_model_keys = []
        for dataset_name in dataset_names:
            # Recreate model keys based on what was processed
            for model_name in model_names:
                text_datasets = ['Stanford-Tree-Bank', 'iSarcasm', 'Sarcasm', 'GoEmotions']
                is_text_dataset = dataset_name in text_datasets or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name
                if model_name == 'CLIP' and is_text_dataset:
                    continue
                
                if model_name == 'Llama':
                    model_key = 'Llama-Text' if is_text_dataset else 'Llama-Vision'
                else:
                    model_key = model_name
                
                if model_key not in all_model_keys:
                    all_model_keys.append(model_key)
        
        # Add titles for each model column using actual axes positions
        for col_idx, model_key in enumerate(all_model_keys):
            if col_idx in column_axes:
                ax = column_axes[col_idx]
                # Get the position of this axes in figure coordinates
                bbox = ax.get_position()
                x_center = (bbox.x0 + bbox.x1) / 2
                
                # Position title above the axes (closer to plots)
                fig.text(x_center, 0.95, model_key, 
                        ha='center', va='top',
                        fontsize=model_size, fontweight='bold', color='black')
    
    # Save or display
    # save_file is an alias for save_path
    output_path = save_file if save_file is not None else save_path
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")
    else:
        plt.show()


def plot_detection_scores_per_concept_multiple_percentthru(dataset_name, split, model_name, sample_type,
                                                          concepts_to_plot=None, concept_type='avg',
                                                          metric='f1', percentthrumodels=None,
                                                          figsize=None, save_path=None,
                                                          cmap='viridis', show_values=True,
                                                          label_font_size=12, input_size=None):
    """
    Plot detection scores for specific concepts across multiple percentthrumodel values.
    Each concept is a row, each percentthrumodel is a column.
    
    Args:
        dataset_name: Name of the dataset
        split: Data split to use ('test', 'cal', etc.)
        model_name: Model name (e.g., 'CLIP', 'Llama', 'Gemma', 'Qwen')
        sample_type: Single sample type ('cls' or 'patch')
        concepts_to_plot: List of concept names to plot. If None, plots all available concepts
        concept_type: Type of concept ('avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae')
        metric: Metric to plot (default: 'f1')
        percentthrumodels: List of percentthrumodel values. If None, uses model defaults
        figsize: Figure size. If None, calculated based on grid dimensions
        save_path: Path to save the figure. If None, displays only
        cmap: Colormap for heatmap (default: 'viridis')
        show_values: Whether to show values in cells (default: True)
        label_font_size: Font size for labels (default: 12)
        input_size: Input size for determining default percentthrumodels
    """
    # Import at function level to avoid circular imports
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from utils.filter_datasets_utils import filter_concept_dict
    
    # Determine percentthrumodels to use
    if percentthrumodels is None:
        if input_size is None:
            # Try to infer input_size based on model and dataset
            if model_name == 'CLIP':
                input_size = (224, 224)
            elif model_name in ['Llama', 'Gemma', 'Qwen']:
                # Check if it's a text dataset
                text_datasets = ['Stanford-Tree-Bank', 'iSarcasm', 'Sarcasm', 'GoEmotions']
                if dataset_name in text_datasets or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                    input_size = ('text', 'text')
                else:
                    input_size = (560, 560)  # Vision Llama
        
        percentthrumodels = get_model_default_percentthrumodels(model_name, input_size)
    
    # Load ground truth samples
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
        else:
            gt_path = None
    elif model_name == 'CLIP':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt'
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt'
    else:
        gt_path = None
    
    if gt_path and os.path.exists(gt_path):
        gt_samples_per_concept = torch.load(gt_path, weights_only=False)
        gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    else:
        print(f"Warning: GT samples file not found: {gt_path}")
        if concepts_to_plot is None:
            print("Error: Cannot determine concepts without GT samples file")
            return
        gt_samples_per_concept = {c: [1] for c in concepts_to_plot}  # Dummy weights
    
    # If concepts_to_plot is None, use all concepts from GT samples
    if concepts_to_plot is None:
        concepts_to_plot = sorted(list(gt_samples_per_concept.keys()))
        print(f"Plotting all {len(concepts_to_plot)} available concepts")
    
    # Validate concepts exist
    valid_concepts = [c for c in concepts_to_plot if c in gt_samples_per_concept]
    if not valid_concepts:
        print(f"Error: None of the requested concepts found in GT samples")
        return
    
    if len(valid_concepts) < len(concepts_to_plot):
        missing = set(concepts_to_plot) - set(valid_concepts)
        print(f"Warning: Some concepts not found: {missing}")
    
    # Determine concept label based on type
    n_clusters = 1000 if sample_type == 'patch' else 50
    if concept_type == 'avg':
        con_label_base = f'{model_name}_avg_{sample_type}_embeddings'
    elif concept_type == 'linsep':
        con_label_base = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False'
    elif concept_type == 'kmeans':
        con_label_base = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans'
    elif concept_type == 'linsep kmeans':
        con_label_base = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans'
    elif concept_type == 'sae':
        con_label_base = f'{model_name}_sae_{sample_type}_dense'
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    # Create matrix to store scores
    scores_matrix = np.zeros((len(valid_concepts), len(percentthrumodels)))
    
    # Percentiles to evaluate
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    # Load scores for each concept and percentthrumodel
    for j, percentthru in enumerate(percentthrumodels):
        if 'sae' not in concept_type:
            con_label = f'{con_label_base}_percentthrumodel_{percentthru}'
        else:
            con_label = con_label_base  # SAE doesn't have percentthrumodel in name
        
        # For each percentile, load the data
        percentile_scores = {concept: [] for concept in valid_concepts}
        
        for percentile in percentiles:
            # For calibration data, append _cal
            if split == 'cal':
                file_con_label = con_label + '_cal'
            else:
                file_con_label = con_label
            
            # Determine file path based on method type
            if 'kmeans' in concept_type or 'sae' in concept_type:
                # Unsupervised methods
                file_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{file_con_label}.csv'
                
                if not os.path.exists(file_path):
                    continue
                
                try:
                    # Load CSV file
                    detection_metrics = pd.read_csv(file_path)
                    
                    # For unsupervised, filter to best matching clusters if available
                    if split == 'cal':
                        best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}_cal.pt'
                    else:
                        best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                    
                    if os.path.exists(best_clusters_path):
                        best_clusters_per_concept = torch.load(best_clusters_path, weights_only=False)
                        detection_metrics = filter_unsupervised_detection_metrics(detection_metrics, best_clusters_per_concept)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
            else:
                # Supervised methods
                file_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{file_con_label}.pt'
                
                if not os.path.exists(file_path):
                    continue
                
                try:
                    detection_metrics = torch.load(file_path, weights_only=False)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
            
            # Extract scores for each concept
            for concept in valid_concepts:
                if concept in detection_metrics['concept'].values:
                    concept_data = detection_metrics[detection_metrics['concept'] == concept]
                    if len(concept_data) > 0:
                        score = concept_data.iloc[0][metric]
                        percentile_scores[concept].append(score)
        
        # Average across percentiles for each concept
        for i, concept in enumerate(valid_concepts):
            if percentile_scores[concept]:
                scores_matrix[i, j] = np.mean(percentile_scores[concept])
            else:
                scores_matrix[i, j] = np.nan
    
    # Create figure
    if figsize is None:
        figsize = (len(percentthrumodels) * 1.5 + 2, len(valid_concepts) * 0.8 + 2)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.isnan(scores_matrix)
    sns.heatmap(scores_matrix, 
                xticklabels=[f"{p}%" for p in percentthrumodels],
                yticklabels=valid_concepts,
                cmap=cmap,
                vmin=0,
                vmax=1,
                mask=mask,
                annot=show_values,
                fmt='.3f',
                cbar_kws={'label': metric.capitalize()},
                ax=ax)
    
    # Labels and title
    ax.set_xlabel('Layer (% through model)', fontsize=label_font_size)
    ax.set_ylabel('Concept', fontsize=label_font_size)
    ax.set_title(f'{dataset_name} - {model_name} {sample_type} {concept_type}\n{metric.capitalize()} Detection Scores',
                 fontsize=label_font_size + 2)
    
    # Rotate x labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()
    
    return scores_matrix


def plot_detection_scores_per_concept(dataset_name, split, model_name, sample_types, metric='f1',
                                      concepts_to_plot=None, plot_type='both', n_cols=3, percentthrumodel=100):
    """
    Creates a separate subplot for each concept showing detection performance across percentiles.
    Updated to match the style and functionality of plot_detection_scores.

    Args:
        dataset_name (str): Name of the dataset
        split (str): 'train', 'test', or 'cal' split
        model_name (str): Model name (e.g., 'CLIP', 'Llama')
        sample_types (list): List of sample types (e.g., ['patch', 'cls'])
        metric (str): Metric to plot (default: 'f1')
        concepts_to_plot (list, optional): List of specific concepts to plot. If None, plots all concepts.
        plot_type (str): 'supervised', 'unsupervised', or 'both'
        n_cols (int): Number of columns in the subplot grid
        percentthrumodel (int): Percentage through model for embeddings (default: 100)
    """
    save_path = f'../Figs/Paper_Figs/{model_name}_{dataset_name}_detectplot_per_concept.pdf'
    plt.rcParams.update({'font.size': 8})
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95]

    # Load ground truth samples
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
        gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    elif model_name == 'CLIP':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
    elif model_name == 'Llama':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)

    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    # Determine which concepts to plot
    if concepts_to_plot is None:
        concepts_to_plot = sorted(filter_concept_dict(gt_samples_per_concept, dataset_name).keys())
    else:
        concepts_to_plot = [c for c in concepts_to_plot if c in gt_samples_per_concept]

    if not concepts_to_plot:
        print("No valid concepts to plot.")
        return

    # Setup concept labels
    con_labels = {}
    for sample_type in sample_types:
        n_clusters = 1000 if sample_type == 'patch' else 50
        if plot_type in ('supervised', 'both'):
            con_labels[f'labeled {sample_type} avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
            con_labels[f'labeled {sample_type} linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
        if plot_type in ('unsupervised', 'both'):
            con_labels[f'unsupervised {sample_type} kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
            con_labels[f'unsupervised {sample_type} linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
        # Add SAE for supported models and datasets
        if plot_type in ('unsupervised', 'both'):
            # CLIP SAE for vision datasets (only available at percentthrumodel=92)
            if model_name == 'CLIP' and sample_type == 'patch':
                if percentthrumodel == 92:
                    con_labels[f'{sample_type} sae'] = f'{model_name}_sae_{sample_type}_dense'
                else:
                    print(f"Warning: SAE concepts for CLIP are only available at percentthrumodel=92, not {percentthrumodel}")
            # Gemma SAE for text datasets (only available at percentthrumodel=81)
            elif model_name == 'Gemma' and sample_type == 'patch':
                if percentthrumodel == 81:
                    con_labels[f'{sample_type} sae'] = f'{model_name}_sae_{sample_type}_dense'
                else:
                    print(f"Warning: SAE concepts for Gemma are only available at percentthrumodel=81, not {percentthrumodel}")

    # Style mapping
    style_map = {
        'labeled patch avg': {'color': 'orchid', 'type': 'supervised', 'label': 'patch avg'},
        'labeled patch linsep': {'color': 'indigo', 'type': 'supervised', 'label': 'patch linsep'},
        'labeled cls avg': {'color': 'goldenrod', 'type': 'supervised', 'label': 'cls avg'},
        'labeled cls linsep': {'color': 'orangered', 'type': 'supervised', 'label': 'cls linsep'},
        'unsupervised patch kmeans': {'color': 'orchid', 'type': 'unsupervised', 'label': 'patch avg'},
        'unsupervised patch linsep kmeans': {'color': 'indigo', 'type': 'unsupervised', 'label': 'patch linsep'},
        'unsupervised cls kmeans': {'color': 'goldenrod', 'type': 'unsupervised', 'label': 'cls avg'},
        'unsupervised cls linsep kmeans': {'color': 'orangered', 'type': 'unsupervised', 'label': 'cls linsep'},
        'patch sae': {'color': 'darkgreen', 'type': 'sae', 'label': 'patch sae'},
    }

    # Calculate subplot layout
    n_concepts = len(concepts_to_plot)
    n_rows = math.ceil(n_concepts / n_cols)

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    # Handle different cases of axes returned by subplots
    if n_rows == 1 and n_cols == 1:
        axes = [axes]  # Single subplot, wrap in list
    elif n_rows == 1 or n_cols == 1:
        axes = axes.tolist()  # 1D array of subplots
    else:
        axes = axes.flatten().tolist()  # 2D array of subplots

    # Get prompt scores if available (only for rate-based metrics)
    prompt_scores = {}
    if metric not in ['fp', 'fn', 'tp', 'tn']:
        try:
            # Use per-concept prompt scores if available, otherwise compute from weighted average
            prompt_scores = get_per_concept_prompt_scores(dataset_name, model_name, metric, split)
        except Exception:
            # Fallback: use single prompt score for all concepts  
            try:
                single_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
                prompt_scores = {c: single_score for c in concepts_to_plot}
            except Exception:
                prompt_scores = {}

    # Baseline style map
    baseline_style_map = {
        'random':     {'color': '#888888', 'label': 'Random'},
        'always_yes': {'color': '#bbbbbb', 'label': 'Always Pos'},
        'always_no':  {'color': '#dddddd', 'label': 'Always Neg'}
    }
    
    # Pre-load all baseline data
    baseline_data = {}
    for baseline_type in ['random', 'always_yes', 'always_no']:
        baseline_path = f'Quant_Results/{dataset_name}/{baseline_type}_{model_name}_cls_baseline.csv'
        if not os.path.exists(baseline_path):
            continue
        try:
            df = pd.read_csv(baseline_path)
            df = df[df['concept'].isin(gt_samples_per_concept)]
            baseline_data[baseline_type] = df
        except Exception:
            continue

    # Plot for each concept
    for idx, concept in enumerate(concepts_to_plot):
        ax = axes[idx]

        # Plot prompt score if available
        if concept in prompt_scores:
            ax.axhline(prompt_scores[concept], color='#8B4513', linestyle='-.', linewidth=2,
                       label="Prompt" if idx == 0 else None)

        # Plot baselines
        for baseline_type, df in baseline_data.items():
            concept_row = df[df['concept'] == concept]
            if not concept_row.empty:
                score = concept_row.iloc[0][metric]
                style = baseline_style_map[baseline_type]
                ax.axhline(score, color=style['color'], linestyle='-.', linewidth=1.5,
                           label=style['label'] if idx == 0 else None)

        # Track seen labels to avoid duplicates in legend
        seen_labels = set() if idx > 0 else set()
        
        # Plot concept discovery methods
        for name, con_label in con_labels.items():
            scores = []
            
            for percentile in percentiles:
                # For calibration data, append _cal to con_label
                if split == 'cal':
                    file_con_label = con_label + '_cal'
                else:
                    file_con_label = con_label
                    
                # Check if this is an unsupervised method (kmeans or sae)
                if 'kmeans' in con_label or 'sae' in con_label:
                    # Unsupervised: use allpairs pattern and CSV format
                    file_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{file_con_label}.csv'
                    
                    if not os.path.exists(file_path):
                        scores.append(0)  # Default to 0 if file not found
                        continue
                        
                    try:
                        # Load CSV file for unsupervised
                        detection_metrics = pd.read_csv(file_path)
                        
                        # For unsupervised, need to find the best matching cluster for this concept
                        # Load the best cluster mapping for the appropriate split
                        if split == 'cal':
                            # For calibration, use the calibration-specific best clusters
                            best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}_cal.pt'
                        else:
                            # For test/train, use the regular best clusters
                            best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                        if os.path.exists(best_clusters_path):
                            best_clusters = torch.load(best_clusters_path, weights_only=False)
                            if concept in best_clusters:
                                cluster_id = best_clusters[concept]['best_cluster']
                                # Find the row for this (concept, cluster) pair
                                concept_metrics = detection_metrics[detection_metrics['concept'] == f"('{concept}', '{cluster_id}')"]
                                if not concept_metrics.empty:
                                    scores.append(concept_metrics.iloc[0][metric])
                                else:
                                    scores.append(0)
                            else:
                                scores.append(0)
                        else:
                            scores.append(0)
                    except Exception:
                        scores.append(0)
                else:
                    # Supervised: use original pattern and PT format
                    file_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{file_con_label}.pt'
                    
                    if not os.path.exists(file_path):
                        scores.append(0)  # Default to 0 if file not found
                        continue
                        
                    try:
                        detection_metrics = torch.load(file_path, weights_only=False)
                        concept_metrics = detection_metrics[detection_metrics['concept'] == concept]
                        if not concept_metrics.empty:
                            scores.append(concept_metrics.iloc[0][metric])
                        else:
                            scores.append(0)
                    except Exception:
                        scores.append(0)

            if any(s > 0 for s in scores):  # Only plot if we have some valid scores
                style = style_map[name]
                color = style['color']
                kind = style['type']
                label = style['label']
                linestyle = ':' if plot_type == 'both' and (kind == 'unsupervised' or kind == 'sae') else '-'
                
                # Only show label in first subplot to avoid legend duplication
                plot_label = label if idx == 0 and label not in seen_labels else None
                if idx == 0:
                    seen_labels.add(label)
                    
                ax.plot(percentiles, scores, color=color, linestyle=linestyle,
                        marker='o', markersize=4, label=plot_label)

        # Formatting for each subplot
        ax.set_xlabel("Concept Recall Percentage", fontsize=12)
        ax.set_ylabel(f"{metric.upper()} Score", fontsize=12)
        ax.set_title(f"{concept}", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1.0, 11))
        ax.set_xticklabels([f"{int(x*100)}%" for x in np.linspace(0, 1.0, 11)])
        ax.grid(True, linestyle='--', linewidth=0.5)

    # Remove empty subplots
    for idx in range(n_concepts, len(axes)):
        fig.delaxes(axes[idx])

    # Add overall title matching the style of plot_detection_scores
    if dataset_name == 'Stanford-Tree-Bank':
        title = "Per-Concept Sentence-Level Detection Performance"
    elif 'Sarcasm' in dataset_name or dataset_name == 'GoEmotions':
        title = "Per-Concept Paragraph-Level Detection Performance"
    else:
        title = "Per-Concept Image-Level Detection Performance"
    
    fig.suptitle(title, fontweight='bold', fontsize=16)

    # Add legend to the figure
    if n_concepts > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title="Concept Type", bbox_to_anchor=(1.05, 0.5),
                       loc='center left', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, format='pdf', bbox_inches='tight')
    plt.show()


def summarize_best_detection_scores(dataset_name, split, model_name, sample_types, metric='f1', weighted_avg=True, concept_types=None, percentthrumodel=100):
    """
    Returns a DataFrame summarizing the best detection score (and the percentile it occurs at)
    for each concept discovery method.
    
    Args:
        dataset_name: Name of the dataset
        split: Data split to use ('test', 'val', etc.)
        model_name: Model name (e.g., 'CLIP', 'Llama')
        sample_types: List of sample types (e.g., ['cls', 'patch'])
        metric: Metric to evaluate (default: 'f1')
        weighted_avg: Whether to use weighted average (default: True)
        concept_types: List of concept types to include. Options: ['avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae']
                      If None, all available types are included
        percentthrumodel: Percentage through model for embeddings (default: 100)
    """
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # Validate concept_types if provided
    if concept_types is not None:
        valid_types = {'avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae'}
        provided_types = set(concept_types)
        invalid_types = provided_types - valid_types
        if invalid_types:
            raise ValueError(f"Invalid concept types: {invalid_types}. Valid options: {valid_types}")

    # === Construct concept label mappings
    con_labels = {}
    for sample_type in sample_types:
        n_clusters = 1000 if sample_type == 'patch' else 50
        # Supervised methods
        if concept_types is None or 'avg' in concept_types:
            con_labels[f'labeled {sample_type} avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
        if concept_types is None or 'linsep' in concept_types:
            con_labels[f'labeled {sample_type} linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
        # Unsupervised methods
        if concept_types is None or 'kmeans' in concept_types:
            con_labels[f'unsupervised {sample_type} kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
        if concept_types is None or 'linsep kmeans' in concept_types:
            con_labels[f'unsupervised {sample_type} linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
        # Add SAE for supported models and datasets
        if concept_types is None or 'sae' in concept_types:
            # CLIP SAE for vision datasets (only available at percentthrumodel=92)
            if model_name == 'CLIP' and sample_type == 'patch':
                    if percentthrumodel == 92:
                        con_labels[f'{sample_type} sae'] = f'{model_name}_sae_{sample_type}_dense'
                    else:
                        print(f"Warning: SAE concepts for CLIP are only available at percentthrumodel=92, not {percentthrumodel}")
            # Gemma SAE for text datasets (only available at percentthrumodel=81)
            elif model_name == 'Gemma' and sample_type == 'patch':
                    if percentthrumodel == 81:
                        con_labels[f'{sample_type} sae'] = f'{model_name}_sae_{sample_type}_dense'
                    else:
                        print(f"Warning: SAE concepts for Gemma are only available at percentthrumodel=81, not {percentthrumodel}")

    # === Load ground-truth labels
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
        gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    elif model_name == 'CLIP':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
    elif model_name == 'Llama':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    results = []

    for name, con_label in con_labels.items():
        best_score = -1
        best_percentile = None
        
        # Check if this is an unsupervised method (kmeans or sae)
        if 'kmeans' in con_label or 'sae' in con_label:
            # Load best clusters mapping for unsupervised methods
            best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
            if not os.path.exists(best_clusters_path):
                print(f"Warning: Best clusters file not found - {best_clusters_path}")
                continue
            best_clusters = torch.load(best_clusters_path, weights_only=False)
            
            for percentile in percentiles:
                try:
                    # Unsupervised methods use CSV format with allpairs
                    detection_metrics = pd.read_csv(f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{con_label}.csv')
                except FileNotFoundError:
                    continue
                
                # Filter to only the best matching (concept, cluster) pairs
                filtered_rows = []
                for concept in gt_samples_per_concept:
                    if concept in best_clusters:
                        cluster_id = best_clusters[concept]['best_cluster']
                        # Find the row for this (concept, cluster) pair
                        row = detection_metrics[detection_metrics['concept'] == f"('{concept}', '{cluster_id}')"]
                        if not row.empty:
                            # Create a simplified row with just the concept name
                            simplified_row = row.iloc[0].copy()
                            simplified_row['concept'] = concept
                            filtered_rows.append(simplified_row)
                
                if filtered_rows:
                    detection_metrics = pd.DataFrame(filtered_rows)
                else:
                    continue
                
                if weighted_avg:
                    total_samples = sum(len(gt_samples_per_concept[c]) for c in detection_metrics['concept'])
                    weighted_sum = 0

                    for _, row in detection_metrics.iterrows():
                        n_samples = len(gt_samples_per_concept[row['concept']])
                        weighted_sum += row[metric] * n_samples

                    score = weighted_sum / total_samples if total_samples > 0 else 0
                else:
                    score = detection_metrics[metric].mean()

                if score > best_score:
                    best_score = score
                    best_percentile = percentile
        else:
            # Supervised methods use PT format
            for percentile in percentiles:
                try:
                    detection_metrics = torch.load(f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.pt', weights_only=False)
                except FileNotFoundError:
                    continue

                detection_metrics = detection_metrics[detection_metrics['concept'].isin(gt_samples_per_concept.keys())]

                if weighted_avg:
                    total_samples = sum(len(gt_samples_per_concept[c]) for c in detection_metrics['concept'])
                    weighted_sum = 0

                    for _, row in detection_metrics.iterrows():
                        n_samples = len(gt_samples_per_concept[row['concept']])
                        weighted_sum += row[metric] * n_samples

                    score = weighted_sum / total_samples if total_samples > 0 else 0
                else:
                    score = detection_metrics[metric].mean()

                if score > best_score:
                    best_score = score
                    best_percentile = percentile

        results.append({
            'Method': name,
            f'Best {metric.upper()}': round(best_score, 4),
            'Percentile': best_percentile
        })

    return pd.DataFrame(results)


def summarize_best_detection_scores_using_per_concept_percentiles(dataset_name, split, model_names, sample_types, metric='f1', weighted_avg=True, concept_types=None, percentthrumodel=100, baselines=None, sorted=False, include_errors=True):
    """
    Displays a table summarizing detection scores using per-concept optimal percentiles
    from calibration data, rather than a single percentile for all concepts.
    
    Args:
        dataset_name: Name of the dataset
        split: Data split to use ('test', 'val', etc.)
        model_names: List of model names (e.g., ['CLIP', 'Llama']) or single model name string
        sample_types: List of sample types (e.g., ['cls', 'patch'])
        metric: Metric to evaluate (default: 'f1')
        weighted_avg: Whether to use weighted average (default: True)
        concept_types: List of concept types to include. Options: ['avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae']
                      If None, all available types are included
        percentthrumodel: Percentage through model for embeddings (default: 100)
        baselines: List of baseline methods to include. Options: ['random', 'prompt', 'maxtoken', 'meantoken', 'lasttoken', 'randomtoken']
                   If None, no baselines are included
        sorted: Whether to sort results by metric value from highest to lowest (default: False)
        include_errors: Whether to include ± error bars from bootstrap CI (default: True)
    """
    # Handle single model name as string
    if isinstance(model_names, str):
        model_names = [model_names]
        
    # Validate concept_types if provided
    if concept_types is not None:
        valid_types = {'avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae'}
        provided_types = set(concept_types)
        invalid_types = provided_types - valid_types
        if invalid_types:
            raise ValueError(f"Invalid concept types: {invalid_types}. Valid options: {valid_types}")

    results = []
    
    # Loop through each model
    for model_name in model_names:
        # === Load ground-truth labels
        if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
            if model_name == 'Llama':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
            elif model_name == 'Gemma':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
            elif model_name == 'Qwen':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
            gt_samples_per_concept = torch.load(gt_path, weights_only=False)
        elif model_name == 'CLIP':
            gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
        elif model_name == 'Llama':
            gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
        else:
            continue
        gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
        
        # === Construct concept label mappings
        for sample_type in sample_types:
            n_clusters = 1000 if sample_type == 'patch' else 50
            con_labels = {}
            
            # Supervised methods
            if concept_types is None or 'avg' in concept_types:
                con_labels['avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'linsep' in concept_types:
                con_labels['linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
            # Unsupervised methods
            if concept_types is None or 'kmeans' in concept_types:
                con_labels['kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'linsep kmeans' in concept_types:
                con_labels['linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
            # Add SAE for supported models and datasets
            if concept_types is None or 'sae' in concept_types:
                # CLIP SAE for vision datasets (only available at percentthrumodel=92)
                if model_name == 'CLIP':
                    if percentthrumodel == 92:
                        con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                    else:
                        print(f"Warning: SAE concepts for CLIP are only available at percentthrumodel=92, not {percentthrumodel}")
                # Gemma SAE for text datasets (only available at percentthrumodel=81)
                elif model_name == 'Gemma':
                    if percentthrumodel == 81:
                        con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                    else:
                        print(f"Warning: SAE concepts for Gemma are only available at percentthrumodel=81, not {percentthrumodel}")

            for name, con_label in con_labels.items():
                # Load best percentiles per concept from calibration
                best_percentiles_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
                
                if not os.path.exists(best_percentiles_path):
                    print(f"Warning: Best percentiles file not found - {best_percentiles_path}")
                    continue
                    
                best_percentiles = torch.load(best_percentiles_path, weights_only=False)
                
                # Collect per-concept scores using their optimal percentiles
                concept_scores = []
                concept_weights = []
                concept_errors = []  # For storing per-concept errors
                
                # Check if this is an unsupervised method (kmeans or sae)
                is_unsupervised = 'kmeans' in con_label or 'sae' in con_label
                
                if is_unsupervised:
                    # Load best clusters mapping for unsupervised methods
                    best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                    if not os.path.exists(best_clusters_path):
                        print(f"Warning: Best clusters file not found - {best_clusters_path}")
                        continue
                    best_clusters = torch.load(best_clusters_path, weights_only=False)
                
                for concept in gt_samples_per_concept:
                    if concept not in best_percentiles:
                        continue
                        
                    # Get the best percentile for this specific concept
                    percentile = best_percentiles[concept]['best_percentile']
                    
                    try:
                        if is_unsupervised:
                            # Unsupervised methods use CSV format with allpairs
                            detection_metrics = pd.read_csv(f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{con_label}.csv')
                            
                            if concept in best_clusters:
                                cluster_id = best_clusters[concept]['best_cluster']
                                # Find the row for this (concept, cluster) pair
                                row = detection_metrics[detection_metrics['concept'] == f"('{concept}', '{cluster_id}')"]
                                if not row.empty:
                                    score = row.iloc[0][metric]
                                    concept_scores.append(score)
                                    concept_weights.append(len(gt_samples_per_concept[concept]))
                                    
                                    # Try to load confidence interval if include_errors is True
                                    if include_errors:
                                        ci_file = f'Quant_Results_with_CI/{dataset_name}/per_concept_ci_optimal_{con_label}.csv'
                                        if os.path.exists(ci_file):
                                            try:
                                                ci_df = pd.read_csv(ci_file)
                                                concept_row = ci_df[ci_df['concept'] == concept]
                                                if not concept_row.empty:
                                                    error_col = f'{metric}_error'
                                                    if error_col in concept_row.columns:
                                                        error = concept_row.iloc[0][error_col]
                                                        concept_errors.append(error)
                                                    else:
                                                        concept_errors.append(0)
                                                else:
                                                    concept_errors.append(0)
                                            except:
                                                concept_errors.append(0)
                                        else:
                                            concept_errors.append(0)
                        else:
                            # Supervised methods use PT format
                            detection_metrics = torch.load(f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.pt', weights_only=False)
                            
                            # Find the row for this concept
                            concept_row = detection_metrics[detection_metrics['concept'] == concept]
                            if not concept_row.empty:
                                score = concept_row.iloc[0][metric]
                                concept_scores.append(score)
                                concept_weights.append(len(gt_samples_per_concept[concept]))
                                
                                # Try to load confidence interval if include_errors is True
                                if include_errors:
                                    ci_file = f'Quant_Results_with_CI/{dataset_name}/per_concept_ci_optimal_{con_label}.csv'
                                    if os.path.exists(ci_file):
                                        try:
                                            ci_df = pd.read_csv(ci_file)
                                            concept_row = ci_df[ci_df['concept'] == concept]
                                            if not concept_row.empty:
                                                error_col = f'{metric}_error'
                                                if error_col in concept_row.columns:
                                                    error = concept_row.iloc[0][error_col]
                                                    concept_errors.append(error)
                                                else:
                                                    concept_errors.append(0)
                                            else:
                                                concept_errors.append(0)
                                        except:
                                            concept_errors.append(0)
                                    else:
                                        concept_errors.append(0)
                                
                    except FileNotFoundError:
                        print(f"Warning: Detection metrics not found for percentile {percentile}")
                        continue
                
                # Calculate final score and error
                if concept_scores:
                    if weighted_avg:
                        total_weight = sum(concept_weights)
                        weighted_score = sum(s * w for s, w in zip(concept_scores, concept_weights)) / total_weight
                        final_score = weighted_score
                        # Calculate weighted error if available
                        if include_errors and concept_errors and len(concept_errors) == len(concept_weights):
                            weighted_error = sum(e * w for e, w in zip(concept_errors, concept_weights)) / total_weight
                        else:
                            weighted_error = 0
                    else:
                        final_score = sum(concept_scores) / len(concept_scores)
                        # Calculate mean error if available
                        if include_errors and concept_errors:
                            weighted_error = sum(concept_errors) / len(concept_errors)
                        else:
                            weighted_error = 0
                else:
                    final_score = 0
                    weighted_error = 0
                    
                # Try to load dataset-level CI if available
                if include_errors:
                    dataset_ci_file = f'Quant_Results_with_CI/{dataset_name}/dataset_ci_optimal_{con_label}.json'
                    if os.path.exists(dataset_ci_file):
                        try:
                            import json
                            with open(dataset_ci_file, 'r') as f:
                                dataset_ci = json.load(f)
                            # Use weighted or macro based on weighted_avg setting
                            metric_key = f"{'weighted' if weighted_avg else 'macro'}_{metric}"
                            if metric_key in dataset_ci:
                                weighted_error = dataset_ci[metric_key].get('error', weighted_error)
                        except:
                            pass
                    
                # SAE is a concept method, not a detection method
                method_name = name
                detection_method = 'CLS' if sample_type == 'cls' else 'SuperTokens'
                    
                result_entry = {
                    'Model': model_name,
                    'Sample Type': sample_type,
                    'Concept Method': method_name,
                    'Detection Method': detection_method,
                    f'Best {metric.upper()}': round(final_score, 3)
                }
                
                # Add error column if requested
                if include_errors:
                    if weighted_error > 0:
                        result_entry[f'Best {metric.upper()}'] = f"{final_score:.3f} ± {weighted_error:.3f}"
                    else:
                        result_entry[f'Best {metric.upper()}'] = f"{final_score:.3f}"
                        
                results.append(result_entry)
    
    # Add baselines if requested
    if baselines and model_names:
        model_name = model_names[0]
        # Load ground truth for baseline calculation
        if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
            if model_name == 'Llama':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
            elif model_name == 'Gemma':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
            elif model_name == 'Qwen':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
            gt_samples_per_concept = torch.load(gt_path, weights_only=False)
        elif model_name == 'CLIP':
            gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
        elif model_name == 'Llama':
            gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
        gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
        
        # Add prompt baseline if requested (only for rate-based metrics, not counts)
        if 'prompt' in baselines and metric not in ['fp', 'fn', 'tp', 'tn']:
            try:
                prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
                prompt_error = 0
                
                # Try to load prompt CI if available
                if include_errors:
                    # First try the per-concept CSV to compute weighted error
                    prompt_ci_file = f'Quant_Results_with_CI/{dataset_name}/per_concept_ci_direct_prompt_Llama-3.2-11B-Vision.csv'
                    if os.path.exists(prompt_ci_file):
                        try:
                            ci_df = pd.read_csv(prompt_ci_file)
                            # Filter to concepts that are in ground truth
                            ci_df = ci_df[ci_df['concept'].isin(gt_samples_per_concept.keys())]
                            
                            error_col = f'{metric}_error'
                            if error_col in ci_df.columns:
                                if weighted_avg:
                                    # Compute weighted average of errors
                                    total_weight = sum(len(gt_samples_per_concept[c]) for c in ci_df['concept'])
                                    weighted_error_sum = 0
                                    for _, row in ci_df.iterrows():
                                        concept = row['concept']
                                        if concept in gt_samples_per_concept:
                                            weight = len(gt_samples_per_concept[concept])
                                            weighted_error_sum += row[error_col] * weight
                                    prompt_error = weighted_error_sum / total_weight if total_weight > 0 else 0
                                else:
                                    # Simple average of errors
                                    prompt_error = ci_df[error_col].mean()
                        except Exception as e:
                            # Fallback to dataset-level CI if available
                            dataset_ci_file = f'Quant_Results_with_CI/{dataset_name}/dataset_ci_direct_prompt_Llama-3.2-11B-Vision.json'
                            if os.path.exists(dataset_ci_file):
                                try:
                                    import json
                                    with open(dataset_ci_file, 'r') as f:
                                        dataset_ci = json.load(f)
                                    metric_key = f"{'weighted' if weighted_avg else 'macro'}_{metric}"
                                    if metric_key in dataset_ci:
                                        prompt_error = dataset_ci[metric_key].get('error', 0)
                                except:
                                    pass
                
                result_entry = {
                    'Model': 'Llama',
                    'Sample Type': 'N/A',
                    'Concept Method': 'N/A',
                    'Detection Method': 'Prompt'
                }
                
                if include_errors and prompt_error > 0:
                    result_entry[f'Best {metric.upper()}'] = f"{prompt_score:.3f} ± {prompt_error:.3f}"
                else:
                    result_entry[f'Best {metric.upper()}'] = f"{prompt_score:.3f}"
                    
                results.append(result_entry)
            except Exception as e:
                print(f"Warning: Could not compute prompt baseline: {e}")
        
        # Add random baseline if requested
        if 'random' in baselines:
            baseline_path = f'Quant_Results/{dataset_name}/random_{model_name}_cls_baseline.csv'
            if os.path.exists(baseline_path):
                try:
                    df = pd.read_csv(baseline_path)
                    df = df[df['concept'].isin(gt_samples_per_concept)]
                    if weighted_avg:
                        total = sum(len(gt_samples_per_concept[c]) for c in df['concept'])
                        score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) for _, row in df.iterrows()) / total
                    else:
                        score = df[metric].mean()
                    
                    result_entry = {
                        'Model': 'N/A',
                        'Sample Type': 'N/A',
                        'Concept Method': 'N/A',
                        'Detection Method': 'Random'
                    }
                    
                    # Random baseline typically doesn't have CI
                    if include_errors:
                        result_entry[f'Best {metric.upper()}'] = f"{score:.3f}"
                    else:
                        result_entry[f'Best {metric.upper()}'] = f"{score:.3f}"
                        
                    results.append(result_entry)
                except Exception as e:
                    print(f"Warning: Could not compute random baseline: {e}")
            else:
                print(f"Warning: Random baseline file not found - {baseline_path}")
        
        # Add token-based baselines if requested
        token_baseline_mapping = {
            'maxtoken': 'max',
            'meantoken': 'mean', 
            'lasttoken': 'last',
            'randomtoken': 'random'
        }
        
        for baseline_name, aggregation_method in token_baseline_mapping.items():
            # Check both the current name and 'avgtoken' as an alias for 'meantoken'
            if baseline_name in baselines or (baseline_name == 'meantoken' and 'avgtoken' in baselines):
                # For each model and sample type combination
                for model_name in model_names:
                    for sample_type in sample_types:
                        if sample_type == 'cls':
                            continue  # Token baselines only make sense for patch/token level
                        
                        # Check for baselines across all concept types
                        n_clusters = 1000 if sample_type == 'patch' else 50
                        
                        # Build con_labels for all concept types we're checking
                        concept_type_labels = []
                        if concept_types is None or 'avg' in concept_types:
                            concept_type_labels.append(('Average', f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'))
                        if concept_types is None or 'linsep' in concept_types:
                            concept_type_labels.append(('LinSep', f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'))
                        if concept_types is None or 'kmeans' in concept_types:
                            concept_type_labels.append(('KMeans', f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'))
                        if concept_types is None or 'linsep kmeans' in concept_types:
                            concept_type_labels.append(('KMeans LinSep', f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'))
                        if concept_types is None or 'sae' in concept_types:
                            # Add SAE for supported models and datasets
                            # CLIP SAE for vision datasets (only available at percentthrumodel=92)
                            if model_name == 'CLIP' and percentthrumodel == 92:
                                concept_type_labels.append(('SAE', f'{model_name}_sae_{sample_type}_dense'))
                            # Gemma SAE for text datasets (only available at percentthrumodel=81)
                            elif model_name == 'Gemma' and percentthrumodel == 81:
                                concept_type_labels.append(('SAE', f'{model_name}_sae_{sample_type}_dense'))
                        
                        # Check each concept type
                        for concept_type_name, con_label in concept_type_labels:
                            # Handle unsupervised methods differently
                            is_unsupervised = 'kmeans' in con_label or 'sae' in con_label
                            
                            if is_unsupervised:
                                # For unsupervised methods, best percentiles are stored in bestdetects files
                                best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{baseline_name}_{con_label}.pt'
                                if not os.path.exists(best_clusters_path):
                                    # Only warn once per baseline/model/sample_type combination
                                    if concept_type_name == concept_type_labels[0][0]:  # First concept type
                                        print(f"Warning: No {baseline_name} baseline files found for {model_name} {sample_type}")
                                        print(f"  Looking for: {best_clusters_path}")
                                    continue
                                best_clusters = torch.load(best_clusters_path, weights_only=False)
                                # Extract best_percentiles from best_clusters data
                                best_percentiles = {concept: info['best_percentile'] for concept, info in best_clusters.items() if 'best_percentile' in info}
                            else:
                                # For supervised methods, load regular best_percentiles file
                                best_percentiles_path = f'Quant_Results/{dataset_name}/{baseline_name}_best_percentiles_{con_label}.pt'
                                
                                if not os.path.exists(best_percentiles_path):
                                    # Only warn once per baseline/model/sample_type combination
                                    if concept_type_name == concept_type_labels[0][0]:  # First concept type
                                        print(f"Warning: No {baseline_name} baseline files found for {model_name} {sample_type}")
                                        print(f"  Looking for: {best_percentiles_path}")
                                    continue
                                    
                                best_percentiles_data = torch.load(best_percentiles_path, weights_only=False)
                                # Extract the best_percentiles dict from the saved data
                                best_percentiles = best_percentiles_data['best_percentiles']
                            
                            # Collect per-concept scores using their optimal percentiles
                            concept_scores = []
                            concept_weights = []
                            concept_errors = []  # For storing per-concept errors
                            
                            for concept in gt_samples_per_concept:
                                if concept not in best_percentiles:
                                    continue
                                    
                                # Get the best percentile for this specific concept
                                percentile = best_percentiles[concept]
                                
                                try:
                                    if is_unsupervised:
                                        # Unsupervised methods use CSV format with allpairs
                                        detection_metrics_path = f'Quant_Results/{dataset_name}/detectionmetrics_{baseline_name}_allpairs_per_{percentile}_{con_label}.csv'
                                        detection_metrics = pd.read_csv(detection_metrics_path)
                                        
                                        if concept in best_clusters:
                                            cluster_id = best_clusters[concept]['best_cluster']
                                            # Find the row for this (concept, cluster) pair
                                            row = detection_metrics[detection_metrics['concept'] == f"('{concept}', '{cluster_id}')"]
                                            if not row.empty:
                                                score = row.iloc[0][metric]
                                                concept_scores.append(score)
                                                concept_weights.append(len(gt_samples_per_concept[concept]))
                                                
                                                # Try to load confidence interval for baselines
                                                if include_errors:
                                                    baseline_ci_file = f'Quant_Results_with_CI/{dataset_name}/per_concept_ci_optimal_{baseline_name}_{con_label}.csv'
                                                    if os.path.exists(baseline_ci_file):
                                                        try:
                                                            ci_df = pd.read_csv(baseline_ci_file)
                                                            concept_row = ci_df[ci_df['concept'] == concept]
                                                            if not concept_row.empty:
                                                                error_col = f'{metric}_error'
                                                                if error_col in concept_row.columns:
                                                                    error = concept_row.iloc[0][error_col]
                                                                    concept_errors.append(error)
                                                                else:
                                                                    concept_errors.append(0)
                                                            else:
                                                                concept_errors.append(0)
                                                        except:
                                                            concept_errors.append(0)
                                                    else:
                                                        concept_errors.append(0)
                                    else:
                                        # Supervised methods use CSV format 
                                        # Load from the combined test CSV file
                                        detection_metrics_path = f'Quant_Results/{dataset_name}/detectfirst_{baseline_name}_test_{con_label}.csv'
                                        if not os.path.exists(detection_metrics_path):
                                            # Try without 'test' in the name
                                            detection_metrics_path = f'Quant_Results/{dataset_name}/detectfirst_{baseline_name}_{con_label}.csv'
                                        
                                        detection_metrics = pd.read_csv(detection_metrics_path)
                                        
                                        # Find the row for this concept and percentile
                                        concept_row = detection_metrics[(detection_metrics['concept'] == concept) & 
                                                                      (detection_metrics['percentile'] == percentile)]
                                        if not concept_row.empty:
                                            score = concept_row.iloc[0][metric]
                                            concept_scores.append(score)
                                            concept_weights.append(len(gt_samples_per_concept[concept]))
                                            
                                            # Try to load confidence interval for baselines
                                            if include_errors:
                                                baseline_ci_file = f'Quant_Results_with_CI/{dataset_name}/per_concept_ci_optimal_{baseline_name}_{con_label}.csv'
                                                if os.path.exists(baseline_ci_file):
                                                    try:
                                                        ci_df = pd.read_csv(baseline_ci_file)
                                                        concept_row = ci_df[ci_df['concept'] == concept]
                                                        if not concept_row.empty:
                                                            error_col = f'{metric}_error'
                                                            if error_col in concept_row.columns:
                                                                error = concept_row.iloc[0][error_col]
                                                                concept_errors.append(error)
                                                            else:
                                                                concept_errors.append(0)
                                                        else:
                                                            concept_errors.append(0)
                                                    except:
                                                        concept_errors.append(0)
                                                else:
                                                    concept_errors.append(0)
                                            
                                except FileNotFoundError:
                                    continue
                            
                            # Calculate final score and error
                            if concept_scores:
                                if weighted_avg:
                                    total_weight = sum(concept_weights)
                                    weighted_score = sum(s * w for s, w in zip(concept_scores, concept_weights)) / total_weight
                                    final_score = weighted_score
                                    # Calculate weighted error if available
                                    if include_errors and concept_errors and len(concept_errors) == len(concept_weights):
                                        weighted_error = sum(e * w for e, w in zip(concept_errors, concept_weights)) / total_weight
                                    else:
                                        weighted_error = 0
                                else:
                                    final_score = sum(concept_scores) / len(concept_scores)
                                    # Calculate mean error if available
                                    if include_errors and concept_errors:
                                        weighted_error = sum(concept_errors) / len(concept_errors)
                                    else:
                                        weighted_error = 0
                                
                                # Try to load dataset-level CI for baseline if available
                                if include_errors:
                                    baseline_dataset_ci_file = f'Quant_Results_with_CI/{dataset_name}/{baseline_name}_dataset_ci_optimal_{con_label}.json'
                                    if os.path.exists(baseline_dataset_ci_file):
                                        try:
                                            import json
                                            with open(baseline_dataset_ci_file, 'r') as f:
                                                dataset_ci = json.load(f)
                                            metric_key = f"{'weighted' if weighted_avg else 'macro'}_{metric}"
                                            if metric_key in dataset_ci:
                                                weighted_error = dataset_ci[metric_key].get('error', weighted_error)
                                        except:
                                            pass
                                
                                result_entry = {
                                    'Model': model_name,
                                    'Sample Type': sample_type,
                                    'Concept Method': concept_type_name,
                                    'Detection Method': baseline_name.capitalize()
                                }
                                
                                if include_errors and weighted_error > 0:
                                    result_entry[f'Best {metric.upper()}'] = f"{final_score:.3f} ± {weighted_error:.3f}"
                                else:
                                    result_entry[f'Best {metric.upper()}'] = f"{final_score:.3f}"
                                
                                results.append(result_entry)

    df = pd.DataFrame(results)
    
    # Replace sample types and method names for better presentation
    df['Sample Type'] = df['Sample Type'].replace({'patch': 'Token', 'cls': 'CLS'})
    df['Concept Method'] = df['Concept Method'].replace({
        'avg': 'Average',
        'linsep': 'LinSep', 
        'kmeans': 'KMeans',
        'linsep kmeans': 'KMeans LinSep',
        'sae': 'SAE'
    })
    # Update Detection Method names for better presentation
    if 'Detection Method' in df.columns:
        df['Detection Method'] = df['Detection Method'].replace({
            'Maxtoken': 'MaxToken',
            'Meantoken': 'MeanToken', 
            'Lasttoken': 'LastToken',
            'Randomtoken': 'RandomToken'
        })
    
    # Reorder columns for better presentation
    metric_col = f'Best {metric.upper()}'
    if 'Detection Method' in df.columns:
        column_order = ['Model', 'Sample Type', 'Concept Method', 'Detection Method', metric_col]
    else:
        column_order = ['Model', 'Sample Type', 'Concept Method', metric_col]
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # Sort by metric value if requested
    if sorted and not df.empty:
        if metric_col in df.columns:
            # Create a sorting key that extracts the numeric value
            def extract_value(x):
                if isinstance(x, str) and ' ± ' in x:
                    return float(x.split(' ± ')[0])
                elif isinstance(x, str):
                    try:
                        return float(x)
                    except:
                        return 0
                else:
                    return float(x)
            
            df['_sort_key'] = df[metric_col].apply(extract_value)
            df = df.sort_values(by='_sort_key', ascending=False)
            df = df.drop('_sort_key', axis=1)
            # Reset index for cleaner display
            df = df.reset_index(drop=True)
    
    # Save as LaTeX table (tabular only)
    models_str = '_'.join(model_names)
    latex_filename = f"../Figs/Paper_Tables/{dataset_name}_{models_str}_detection_scores_{metric}_ptm{percentthrumodel}.tex"
    with open(latex_filename, 'w') as f:
        # Create a copy of df for LaTeX output
        df_latex = df.copy()
        
        # Format all float values in the metric column to strings with exactly 3 decimals
        metric_col = f'Best {metric.upper()}'
        if metric_col in df_latex.columns:
            # Convert all metric values to formatted strings
            for idx in df_latex.index:
                val = df_latex.loc[idx, metric_col]
                if isinstance(val, (int, float)):
                    df_latex.loc[idx, metric_col] = f'{val:.3f}'
            
            # Now bold the max value row
            if not df.empty:
                # Find the max value by extracting numeric values
                def extract_value(x):
                    if isinstance(x, str) and ' ± ' in x:
                        return float(x.split(' ± ')[0])
                    elif isinstance(x, str):
                        try:
                            return float(x)
                        except:
                            return 0
                    else:
                        return float(x)
                
                max_val = -float('inf')
                max_idx = 0
                for idx in df_latex.index:
                    val = extract_value(df_latex.loc[idx, metric_col])
                    if val > max_val:
                        max_val = val
                        max_idx = idx
                
                # Bold the entire row with highest score
                for col in df_latex.columns:
                    current_val = df_latex.loc[max_idx, col]
                    df_latex.loc[max_idx, col] = f"\\textbf{{{current_val}}}"
            
        # Convert to LaTeX tabular only (no table environment)
        latex_str = df_latex.to_latex(index=False,
                                     column_format='l' * len(df.columns),
                                     escape=False)
        
        # Extract just the tabular part and add title
        lines = latex_str.split('\n')
        tabular_lines = []
        in_tabular = False
        
        # Add table title as a multicolumn
        title_added = False
        
        for line in lines:
            if '\\begin{tabular}' in line:
                in_tabular = True
                tabular_lines.append(line)
                # Add title after begin{tabular}
                tabular_lines.append(f"\\multicolumn{{{len(df.columns)}}}{{c}}{{\\textbf{{{dataset_name}: Detection Performance}}}} \\\\")
                tabular_lines.append("\\hline")
                title_added = True
            elif in_tabular:
                tabular_lines.append(line)
            if '\\end{tabular}' in line:
                break
                
        f.write('\n'.join(tabular_lines))
    print(f"LaTeX table saved to: {latex_filename}")
    
    # Set pandas display options to show full table
    with pd.option_context('display.max_rows', None, 
                          'display.max_columns', None,
                          'display.width', None,
                          'display.max_colwidth', None):
        # If DataFrame is not empty, highlight the row with the highest score
        if not df.empty:
            metric_col = f'Best {metric.upper()}'
            if metric_col in df.columns:
                # Find the index of the row with the highest score
                # Extract numeric values for comparison
                def extract_value(x):
                    if isinstance(x, str) and ' ± ' in x:
                        return float(x.split(' ± ')[0])
                    elif isinstance(x, str):
                        try:
                            return float(x)
                        except:
                            return 0
                    else:
                        return float(x)
                
                max_val = -float('inf')
                max_idx = 0
                for idx in df.index:
                    val = extract_value(df.loc[idx, metric_col])
                    if val > max_val:
                        max_val = val
                        max_idx = idx
                
                # Create a styler function to bold the best row
                def highlight_best_row(row):
                    if row.name == max_idx:
                        return ['font-weight: bold'] * len(row)
                    else:
                        return [''] * len(row)
                
                # Apply the styling and display
                # In non-Jupyter environments, just print the DataFrame
                try:
                    from IPython.display import display
                    import IPython
                    # Check if we're in a Jupyter environment
                    if hasattr(IPython, 'get_ipython') and IPython.get_ipython() is not None:
                        styled_df = df.style.apply(highlight_best_row, axis=1)
                        display(styled_df)
                    else:
                        print(df.to_string())
                except (ImportError, NameError):
                    # Fallback to simple print
                    print(df.to_string())
            else:
                # If metric column doesn't exist, just display without styling
                print(df.to_string())

def summarize_best_detection_scores_per_concept(dataset_name, split, model_name, sample_types, metric='f1', concept_types=None, display=False, percentthrumodel=100):
    """
    Returns a DataFrame showing, for each concept and discovery method,
    the best detection score and the corresponding percentile.

    Args:
        dataset_name: Name of the dataset
        split: Data split to use ('test', 'val', etc.)
        model_name: Model name (e.g., 'CLIP', 'Llama')
        sample_types: List of sample types (e.g., ['cls', 'patch'])
        metric: Metric to evaluate (default: 'f1')
        concept_types: List of concept types to include. Options: ['avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae']
                      If None, all available types are included
        display: Whether to display the full DataFrame (default: False)
        percentthrumodel: Percentage through model for embeddings (default: 100)

    Returns:
        DataFrame with columns: [concept, method, best_<metric>, percentile]
    """
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # Validate concept_types if provided
    if concept_types is not None:
        valid_types = {'avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae'}
        provided_types = set(concept_types)
        invalid_types = provided_types - valid_types
        if invalid_types:
            raise ValueError(f"Invalid concept types: {invalid_types}. Valid options: {valid_types}")

    # === Build concept discovery labels
    con_labels = {}
    for sample_type in sample_types:
        n_clusters = 1000 if sample_type == 'patch' else 50
        # Supervised methods
        if concept_types is None or 'avg' in concept_types:
            con_labels[f'labeled {sample_type} avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
        if concept_types is None or 'linsep' in concept_types:
            con_labels[f'labeled {sample_type} linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
        # Unsupervised methods
        if concept_types is None or 'kmeans' in concept_types:
            con_labels[f'unsupervised {sample_type} kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
        if concept_types is None or 'linsep kmeans' in concept_types:
            con_labels[f'unsupervised {sample_type} linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
        # Add SAE for supported models and datasets
        if concept_types is None or 'sae' in concept_types:
            # CLIP SAE for vision datasets (only available at percentthrumodel=92)
            if model_name == 'CLIP' and sample_type == 'patch':
                    if percentthrumodel == 92:
                        con_labels[f'{sample_type} sae'] = f'{model_name}_sae_{sample_type}_dense'
                    else:
                        print(f"Warning: SAE concepts for CLIP are only available at percentthrumodel=92, not {percentthrumodel}")
            # Gemma SAE for text datasets (only available at percentthrumodel=81)
            elif model_name == 'Gemma' and sample_type == 'patch':
                    if percentthrumodel == 81:
                        con_labels[f'{sample_type} sae'] = f'{model_name}_sae_{sample_type}_dense'
                    else:
                        print(f"Warning: SAE concepts for Gemma are only available at percentthrumodel=81, not {percentthrumodel}")

    # === Load GT concepts
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
    elif model_name == 'CLIP':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt'
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt'
    gt_concepts = torch.load(gt_path, weights_only=False)
    gt_concepts = filter_concept_dict(gt_concepts, dataset_name)
    results = []

    for method_name, con_label in con_labels.items():
        per_concept_best = defaultdict(lambda: (-1, None))  # concept -> (best_score, best_percentile)

        # Check if this is an unsupervised method (kmeans or sae)
        if 'kmeans' in con_label or 'sae' in con_label:
            # Load best clusters mapping for unsupervised methods
            best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
            if not os.path.exists(best_clusters_path):
                print(f"Warning: Best clusters file not found - {best_clusters_path}")
                continue
            best_clusters = torch.load(best_clusters_path, weights_only=False)
            
            for percentile in percentiles:
                try:
                    # Unsupervised methods use CSV format with allpairs
                    df = pd.read_csv(f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{con_label}.csv')
                except FileNotFoundError:
                    continue
                
                # Process each concept
                for concept in gt_concepts.keys():
                    if concept in best_clusters:
                        cluster_id = best_clusters[concept]['best_cluster']
                        # Find the row for this (concept, cluster) pair
                        row = df[df['concept'] == f"('{concept}', '{cluster_id}')"]
                        if not row.empty:
                            score = row.iloc[0][metric]
                            if score > per_concept_best[concept][0]:
                                per_concept_best[concept] = (score, percentile)
        else:
            # Supervised methods use PT format
            for percentile in percentiles:
                try:
                    df = torch.load(f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.pt', weights_only=False)
                except FileNotFoundError:
                    continue

                df = df[df['concept'].isin(gt_concepts.keys())]

                for _, row in df.iterrows():
                    c = row['concept']
                    score = row[metric]

                    if score > per_concept_best[c][0]:
                        per_concept_best[c] = (score, percentile)

        for concept, (score, percentile) in per_concept_best.items():
            results.append({
                'concept': concept,
                'method': method_name,
                f'best_{metric}': round(score, 4),
                'percentile': percentile
            })

    df = pd.DataFrame(results)
    
    # Display the full DataFrame if requested
    if display:
        from IPython.display import display as ipython_display
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            ipython_display(df)
    
    return df
    
""" Inversions """
def compare_best_schemes(metric_type, concept_schemes, dataset_name, model_name,
                         justobj=False, superdetector_inversion=False, xmin=None, xmax=None, weighted_avg=True,
                         include_baselines=True):

    dir = f'Quant_Results/{dataset_name}'
    best_metric_dfs = {}

    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text3').pt"
    elif model_name == 'CLIP':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(224, 224).pt'
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(560, 560).pt'
    gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)

    for concept_scheme in concept_schemes:
        # Construct con_label first
        if concept_scheme == 'avg':
            con_label = f"{model_name}_avg_patch_embeddings_percentthrumodel_100"
        elif concept_scheme == 'linsep':
            con_label = f"{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_100"
        elif concept_scheme == 'unsupervised kmeans':
            con_label = f"{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_100"
        elif concept_scheme == 'unsupervised kmeans linsep':
            con_label = f"{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_100"
        elif concept_scheme == 'sae':
            con_label = f"{model_name}_patchsae_patch_embeddings_sae_percentthrumodel_100"
        else:
            raise ValueError(f"Unrecognized concept scheme: {concept_scheme}")

        # Use the actual file naming from all_inversion_stats.py
        if superdetector_inversion:
            # Two-stage superdetector files
            pt_file_name = f"optimal_test_results_twostage_superpatch_{con_label}_f1.pt"
            csv_file_name = f"twostage_superpatch_avg_{con_label}_optimal_test.csv"
            
            # Check if CSV exists, otherwise use PT file
            if os.path.exists(os.path.join(base_dir, csv_file_name)):
                file_name = csv_file_name
            else:
                # Load PT file and convert to the format expected
                pt_path = os.path.join(base_dir, pt_file_name)
                if os.path.exists(pt_path):
                    results = torch.load(pt_path, weights_only=False)
                    # Convert to CSV format
                    rows = []
                    for concept, metrics in results.items():
                        row = {
                            'concept': concept,
                            'tp': metrics['tp'],
                            'fp': metrics['fp'],
                            'tn': metrics['tn'],
                            'fn': metrics['fn'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1': metrics['f1'],
                            'accuracy': metrics['accuracy']
                        }
                        if 'invert_percentile' in metrics:
                            row['invert_percentile'] = metrics['invert_percentile']
                        rows.append(row)
                    df_temp = pd.DataFrame(rows)
                    csv_path = os.path.join(base_dir, csv_file_name)
                    df_temp.to_csv(csv_path, index=False)
                    file_name = csv_file_name
                else:
                    print(f"Warning: Neither CSV nor PT file found for twostage {con_label}")
                    continue
        else:
            # Regular method doesn't create CSV by default, so try PT file first
            pt_file_name = f"optimal_test_results_{con_label}_f1.pt"
            csv_file_name = f"{con_label}_optimal_test.csv"
            
            # Check if CSV exists, otherwise use PT file
            if os.path.exists(os.path.join(dir, csv_file_name)):
                file_name = csv_file_name
            else:
                # Load PT file and convert to the format expected
                pt_path = os.path.join(dir, pt_file_name)
                if os.path.exists(pt_path):
                    results = torch.load(pt_path, weights_only=False)
                    # Convert to CSV format
                    rows = []
                    for concept, metrics in results.items():
                        rows.append({
                            'concept': concept,
                            'tp': metrics['tp'],
                            'fp': metrics['fp'],
                            'tn': metrics['tn'],
                            'fn': metrics['fn'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1': metrics['f1'],
                            'accuracy': metrics['accuracy']
                        })
                    df_temp = pd.DataFrame(rows)
                    csv_path = os.path.join(dir, csv_file_name)
                    df_temp.to_csv(csv_path, index=False)
                    file_name = csv_file_name
                else:
                    print(f"Warning: Neither CSV nor PT file found for {con_label}")
                    continue

        metric_path = f'{dir}/{file_name}'
        
        try:
            df = pd.read_csv(metric_path)
            df = df[df['concept'].isin(gt_samples_per_concept)]
            
            # Don't show percentiles since they vary by concept
            label = f"{concept_scheme}"
            best_metric_dfs[label] = df
            
        except FileNotFoundError:
            print(f"Warning: Optimal calibration file not found {metric_path}, skipping.")
            continue

    if include_baselines:
        for baseline_type in ['random', 'always_positive', 'always_negative']:
            baseline_path = f'{dir}/inversion_{baseline_type}_{model_name}_patch_baseline.csv'
            try:
                baseline_df = pd.read_csv(baseline_path)
                baseline_df = baseline_df[baseline_df['concept'].isin(gt_samples_per_concept)]

                if weighted_avg:
                    total_samples = sum(len(gt_samples_per_concept[c]) for c in baseline_df['concept'])
                    weighted_sum = sum(row[metric_type] * len(gt_samples_per_concept[row['concept']])
                                       for _, row in baseline_df.iterrows())
                    avg_metric = weighted_sum / total_samples if total_samples > 0 else 0
                else:
                    avg_metric = baseline_df[metric_type].mean()

                best_metric_dfs[f"Inversion Baseline\n({baseline_type})"] = baseline_df
                print(f"Added {baseline_type} baseline with avg {metric_type}: {avg_metric:.3f}")

            except FileNotFoundError:
                print(f"Warning: Baseline file not found {baseline_path}")

    title = f"Best {metric_type.capitalize()} for {model_name} Patch Schemes on {dataset_name}"
    if superdetector_inversion:
        title += "\n(Superdetector Inversion)"
    if include_baselines:
        title += "\n(Including Inversion Baselines)"

    plot_average_metrics(best_metric_dfs, metric_type, title=title, xmin=xmin, xmax=xmax)


def compare_best_schemes_per_concept(metric_type, concept_schemes, dataset_name, model_name,
                                     justobj=False,
                                     superdetector_inversion=False, concepts_to_plot=None,
                                     xmin=None, xmax=None, n_cols=3, include_baselines=True):

    base_dir = f'Quant_Results/{dataset_name}'

    # Load and filter GT
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text3').pt"
    elif model_name == 'CLIP':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(224, 224).pt'
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(560, 560).pt'
    gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)

    concept_to_best_metrics = {}

    for concept_scheme in concept_schemes:
        # Construct con_label first
        if concept_scheme == 'avg':
            con_label = f"{model_name}_avg_patch_embeddings_percentthrumodel_100"
        elif concept_scheme == 'linsep':
            con_label = f"{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_100"
        elif concept_scheme == 'unsupervised kmeans':
            con_label = f"{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_100"
        elif concept_scheme == 'unsupervised kmeans linsep':
            con_label = f"{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_100"
        elif concept_scheme == 'sae':
            con_label = f"{model_name}_patchsae_patch_embeddings_sae_percentthrumodel_100"
        else:
            raise ValueError(f"Unrecognized concept scheme: {concept_scheme}")

        # Use the EXACT file naming from all_inversion_stats.py - NO FALLBACKS
        if superdetector_inversion:
            # Two-stage superdetector: all_inversion_stats.py saves BOTH PT and CSV
            # We only use the CSV file
            file_name = f"twostage_superpatch_avg_{con_label}_optimal_test.csv"
        else:
            # Regular method: all_inversion_stats.py saves ONLY PT file
            pt_file_name = f"optimal_test_results_{con_label}_f1.pt"
            pt_path = os.path.join(base_dir, pt_file_name)
            
            if not os.path.exists(pt_path):
                print(f"Warning: Regular method results not found for {con_label}")
                print(f"  Expected PT file: {pt_file_name}")
                continue
            
            # Load PT file and convert to DataFrame for plotting
            results = torch.load(pt_path, weights_only=False)
            rows = []
            for concept, metrics in results.items():
                rows.append({
                    'concept': concept,
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'tn': metrics['tn'],
                    'fn': metrics['fn'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'accuracy': metrics['accuracy']
                })
            df = pd.DataFrame(rows)
            
            # Skip the rest of the CSV loading logic - we have df directly
            if 'kmeans' in concept_scheme or 'sae' in concept_scheme:
                # Check if concepts in df are already ground truth names or cluster IDs
                first_concept = str(df['concept'].iloc[0]) if len(df) > 0 else None
                if first_concept and first_concept.isdigit():
                    # Concepts are cluster IDs, need to map to ground truth names
                    con_label_for_alignment = con_label  # Save original con_label
                    alignment_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label_for_alignment}.pt'
                    alignment_results = torch.load(alignment_path, weights_only=False)
                    cluster_to_concept = {str(info['best_cluster']): concept for concept, info in alignment_results.items()}
                    df = df.copy()
                    df['concept'] = df['concept'].astype(str).map(cluster_to_concept)
                # else: concepts are already ground truth names, no mapping needed

            df = df[df['concept'].isin(gt_samples_per_concept_test)]

            if weighted_avg:
                total_samples = sum(len(gt_samples_per_concept_test[c]) for c in df['concept'])
                weighted_sum = sum(row[metric_type] * len(gt_samples_per_concept_test[row['concept']]) for _, row in df.iterrows())
                avg_metric = weighted_sum / total_samples if total_samples > 0 else 0
            else:
                avg_metric = df[metric_type].mean()

            scheme_name = f'{concept_scheme} superdetector cossim' if superdetector_inversion else f'{concept_scheme} concept cossim'
            summary_rows.append({
                'Scheme': scheme_name,
                f'Best Avg {metric_type.upper()}': round(avg_metric, 4),
                'Detect %': 'calibrated',
                'Invert %': 'calibrated',
                'File': pt_file_name if not superdetector_inversion else file_name
            })
            continue  # Skip the CSV loading part below

        metric_path = f'{base_dir}/{file_name}'
        optimal_path = f'Detect_Invert_Thresholds/{dataset_name}/optimal_f1_{con_label}.pt'
        
        try:
            df = pd.read_csv(metric_path)
            df = df[df['concept'].isin(gt_samples_per_concept)]
            
            # Load optimal percentiles
            try:
                optimal_data = torch.load(optimal_path, weights_only=False)
            except:
                optimal_data = {}
            
            best_metrics_per_concept = {}
            for idx, row in df.iterrows():
                concept = row['concept']
                score = row[metric_type]
                
                # Get the optimal percentiles for this concept
                if concept in optimal_data:
                    detect_pct = optimal_data[concept].get('detect_percentile', 'N/A')
                    invert_pct = optimal_data[concept].get('invert_percentile', 'N/A')
                else:
                    detect_pct = 'N/A'
                    invert_pct = 'N/A'
                    
                best_metrics_per_concept[concept] = (score, detect_pct, invert_pct)
                
            concept_to_best_metrics[concept_scheme] = best_metrics_per_concept
            
        except FileNotFoundError:
            print(f"Warning: Optimal calibration file not found {metric_path}, skipping.")
            concept_to_best_metrics[concept_scheme] = {}

    if include_baselines:
        for baseline_type in ['random', 'always_positive', 'always_negative']:
            baseline_path = f'{base_dir}/inversion_{baseline_type}_{model_name}_patch_baseline.csv'
            baseline_metrics_per_concept = {}

            try:
                baseline_df = pd.read_csv(baseline_path)
                baseline_df = baseline_df[baseline_df['concept'].isin(gt_samples_per_concept)]
                for idx, row in baseline_df.iterrows():
                    concept = row['concept']
                    score = row[metric_type]
                    baseline_metrics_per_concept[concept] = (score, 'N/A', 'N/A')

                concept_to_best_metrics[f'Baseline ({baseline_type})'] = baseline_metrics_per_concept
                print(f"Added {baseline_type} baseline")

            except FileNotFoundError:
                print(f"Warning: Baseline file not found {baseline_path}")

    # Plotting
    all_concepts = set()
    for scheme_best in concept_to_best_metrics.values():
        all_concepts.update(scheme_best.keys())

    if concepts_to_plot is None:
        concepts_to_plot = sorted(list(all_concepts))

    n_concepts = len(concepts_to_plot)
    n_rows = math.ceil(n_concepts / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, concept in enumerate(concepts_to_plot):
        ax = axes[idx]
        scores = []
        labels = []

        for concept_scheme in concept_to_best_metrics.keys():
            if concept in concept_to_best_metrics[concept_scheme]:
                score, detect, invert = concept_to_best_metrics[concept_scheme][concept]
                scores.append(score)
                if detect == 'N/A':
                    labels.append(f"{concept_scheme}")
                else:
                    labels.append(f"{concept_scheme}\n(detect={detect}, invert={invert})")
            else:
                scores.append(0.0)
                labels.append(f"{concept_scheme}\n(not found)")

        colors = sns.color_palette("husl", len(labels))
        bars = ax.barh(labels, scores, color=colors)

        for bar, value in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}", va='center', ha='left', fontsize=9, fontweight='bold')

        ax.set_title(f"{concept}", fontsize=11)
        ax.set_xlim(left=xmin if xmin is not None else 0, right=xmax if xmax is not None else 1)
        ax.set_xlabel(metric_type.capitalize())
        ax.grid(axis='x', linestyle='--', linewidth=0.5)

    for i in range(n_concepts, len(axes)):
        fig.delaxes(axes[i])

    title = f"Best {metric_type.capitalize()} per Concept ({model_name} on {dataset_name})"
    if include_baselines:
        title += "\n(Including Inversion Baselines)"
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def summarize_best_inversion_metrics(metric_type, concept_schemes, dataset_name, model_name,
                                     justobj=False, superdetector_inversion=False, weighted_avg=True, include_baselines=True, percentthrumodel=100, sample_types=None):
    base_dir = f'Quant_Results/{dataset_name}'
    summary_rows = []

    # Load and filter ground truth
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or dataset_name == 'GoEmotions':
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text2').pt"
    elif model_name == 'CLIP':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(224, 224).pt'
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(560, 560).pt'
    gt_samples_per_concept_test = torch.load(gt_path, weights_only=False)
    gt_samples_per_concept_test = filter_concept_dict(gt_samples_per_concept_test, dataset_name)

    # Default to patch if sample_types not specified
    if sample_types is None:
        sample_types = ['patch']
    
    for concept_scheme in concept_schemes:
        # For inversion, we typically only use patch sample type
        # But allow flexibility if needed
        sample_type = sample_types[0] if sample_types else 'patch'
        
        # Adjust n_clusters based on sample type
        n_clusters = 1000 if sample_type == 'patch' else 50
        
        # Construct con_label first
        if concept_scheme == 'avg':
            con_label = f"{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}"
        elif concept_scheme == 'linsep':
            con_label = f"{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}"
        elif concept_scheme == 'unsupervised kmeans':
            con_label = f"{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}"
        elif concept_scheme == 'unsupervised kmeans linsep':
            con_label = f"{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}"
        elif concept_scheme == 'sae':
            # SAE uses different naming convention
            con_label = f"{model_name}_sae_{sample_type}_dense"
        else:
            raise ValueError(f"Unrecognized concept scheme: {concept_scheme}")

        # Load best percentiles for display
        detect_percentiles_data = None
        invert_percentiles_data = None
        
        # For regular method
        if not superdetector_inversion:
            try:
                detect_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
                detect_percentiles_data = torch.load(detect_file, weights_only=False)
                
                invert_file = f'Best_Inversion_Percentiles_Cal/{dataset_name}/best_inversion_percentiles_{con_label}.pt'
                invert_percentiles_data = torch.load(invert_file, weights_only=False)
            except:
                pass
        else:
            # For two-stage superdetector
            try:
                invert_file = f'Best_Inversion_Percentiles_Cal/{dataset_name}/best_inversion_percentiles_twostage_superpatch_{con_label}.pt'
                invert_percentiles_data = torch.load(invert_file, weights_only=False)
                # Two-stage uses best detection percentiles per concept
                detect_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
                detect_percentiles_data = torch.load(detect_file, weights_only=False)
            except:
                pass

        # Use the EXACT file naming from all_inversion_stats.py - NO FALLBACKS
        if superdetector_inversion:
            # Two-stage superdetector: all_inversion_stats.py saves PT file
            pt_file_name = f"optimal_test_results_twostage_superpatch_{con_label}_f1.pt"
            pt_path = os.path.join(base_dir, pt_file_name)
            
            if not os.path.exists(pt_path):
                print(f"Warning: Two-stage superdetector results not found for {con_label}")
                print(f"  Expected PT file: {pt_file_name}")
                continue
                
            # Load PT file and convert to DataFrame for processing
            results = torch.load(pt_path, weights_only=False)
            rows = []
            for concept, metrics in results.items():
                row = {
                    'concept': concept,
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'tn': metrics['tn'],
                    'fn': metrics['fn'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1']
                }
                rows.append(row)
            df = pd.DataFrame(rows)
        else:
            # Regular method: all_inversion_stats.py saves ONLY PT file
            pt_file_name = f"optimal_test_results_{con_label}_f1.pt"
            pt_path = os.path.join(base_dir, pt_file_name)
            
            if not os.path.exists(pt_path):
                print(f"Warning: Regular method results not found for {con_label}")
                print(f"  Expected PT file: {pt_file_name}")
                continue
            
            # Load PT file and convert to DataFrame for processing
            results = torch.load(pt_path, weights_only=False)
            rows = []
            for concept, metrics in results.items():
                row = {
                    'concept': concept,
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'tn': metrics['tn'],
                    'fn': metrics['fn'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1']
                }
                rows.append(row)
            df = pd.DataFrame(rows)

        if 'kmeans' in concept_scheme or 'sae' in concept_scheme:
            # Check if concepts in df are already ground truth names or cluster IDs
            first_concept = str(df['concept'].iloc[0]) if len(df) > 0 else None
            if first_concept and first_concept.isdigit():
                # Concepts are cluster IDs, need to map to ground truth names
                alignment_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                alignment_results = torch.load(alignment_path, weights_only=False)
                cluster_to_concept = {str(info['best_cluster']): concept for concept, info in alignment_results.items()}
                df = df.copy()
                df['concept'] = df['concept'].astype(str).map(cluster_to_concept)
            # else: concepts are already ground truth names, no mapping needed

        df = df[df['concept'].isin(gt_samples_per_concept_test)]

        if weighted_avg:
            total_samples = sum(len(gt_samples_per_concept_test[c]) for c in df['concept'])
            weighted_sum = sum(row[metric_type] * len(gt_samples_per_concept_test[row['concept']]) for _, row in df.iterrows())
            avg_metric = weighted_sum / total_samples if total_samples > 0 else 0
        else:
            avg_metric = df[metric_type].mean()
            
        # Get most common percentiles from the loaded data
        detect_percentile_str = 'calibrated'
        invert_percentile_str = 'calibrated'
        
        if detect_percentiles_data and invert_percentiles_data:
            # Get all percentiles used for concepts in df
            detect_percs = []
            invert_percs = []
            
            for concept in df['concept']:
                if concept in detect_percentiles_data:
                    detect_percs.append(detect_percentiles_data[concept].get('best_percentile', None))
                if concept in invert_percentiles_data:
                    invert_percs.append(invert_percentiles_data[concept].get('best_percentile', None))
            
            # Get mode (most common) percentiles
            if detect_percs:
                from collections import Counter
                detect_counter = Counter(detect_percs)
                mode_detect = detect_counter.most_common(1)[0][0]
                detect_percentile_str = str(mode_detect)
                    
            if invert_percs:
                invert_counter = Counter(invert_percs)
                mode_invert = invert_counter.most_common(1)[0][0]
                invert_percentile_str = str(mode_invert)

        scheme_name = f'{concept_scheme} superdetector cossim' if superdetector_inversion else f'{concept_scheme} concept cossim'
        summary_rows.append({
            'Scheme': scheme_name,
            f'Best Avg {metric_type.upper()}': round(avg_metric, 4),
            'Detect %': detect_percentile_str,
            'Invert %': invert_percentile_str,
            'File': pt_file_name
        })

    if include_baselines:
        for baseline_type in ['random', 'always_positive', 'always_negative']:
            baseline_path = f'{base_dir}/inversion_{baseline_type}_{model_name}_patch_baseline.csv'
            try:
                baseline_df = pd.read_csv(baseline_path)
                baseline_df = baseline_df[baseline_df['concept'].isin(gt_samples_per_concept_test)]

                if weighted_avg:
                    total_samples = sum(len(gt_samples_per_concept_test[c]) for c in baseline_df['concept'])
                    weighted_sum = sum(row[metric_type] * len(gt_samples_per_concept_test[row['concept']]) for _, row in baseline_df.iterrows())
                    avg_metric = weighted_sum / total_samples if total_samples > 0 else 0
                else:
                    avg_metric = baseline_df[metric_type].mean()

                summary_rows.append({
                    'Scheme': f'Inversion Baseline ({baseline_type})',
                    f'Best Avg {metric_type.upper()}': round(avg_metric, 4),
                    'Detect %': 'N/A',
                    'Invert %': 'N/A',
                    'File': f'inversion_{baseline_type}_{model_name}_patch_baseline.csv'
                })
            except FileNotFoundError:
                print(f"Warning: Baseline file not found {baseline_path}")

    return pd.DataFrame(summary_rows)


def summarize_all_inversion_methods(metric_type, concept_schemes, dataset_name, 
                                  weighted_avg=True, include_baselines=True, percentthrumodel=100,
                                  concept_types=None, model_names=None, sample_types=None, show_baseline=True,
                                  save_table=True):
    """
    Comprehensive summary showing all models and both regular/superdetector methods for a dataset.
    
    Args:
        metric_type: 'f1', 'precision', 'recall', etc.
        concept_schemes: List like ['avg', 'linsep', 'unsupervised kmeans']
        dataset_name: Dataset name
        weighted_avg: Whether to weight by number of test samples
        include_baselines: Whether to include random baselines
        percentthrumodel: Percentage through model for embeddings (default: 100)
        concept_types: List of concept types to include. Options: ['avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae']
                      If None, uses concept_schemes parameter
        model_names: List of specific models to include (e.g., ['CLIP', 'Llama'])
                    If None, includes all available models for the dataset
        sample_types: List of sample types to include (e.g., ['patch', 'cls'])
                     If None, defaults to ['patch'] for inversion
        show_baseline: Whether to show baseline results in the output (default: True)
                      Note: include_baselines still controls whether baselines are computed
        save_table: Whether to save the results table to a CSV file (default: True)
    
    Returns:
        pd.DataFrame with comprehensive comparison
    """
    # Convert concept_types to concept_schemes if provided
    if concept_types is not None:
        # Map concept_types to concept_schemes used in inversion
        type_to_scheme = {
            'avg': 'avg',
            'linsep': 'linsep', 
            'kmeans': 'unsupervised kmeans',
            'linsep kmeans': 'unsupervised kmeans linsep',
            'sae': 'sae'
        }
        concept_schemes = [type_to_scheme.get(ct, ct) for ct in concept_types]
    
    # Set default sample_types if not provided
    if sample_types is None:
        sample_types = ['patch']  # Inversion typically uses patch
    
    # Determine models based on dataset type
    if model_names is not None:
        # Use user-specified models
        models = model_names
    else:
        # Default models based on dataset type
        if dataset_name in ['Stanford-Tree-Bank', 'GoEmotions'] or 'Sarcasm' in dataset_name:
            # Text datasets
            models = ['Llama', 'Qwen', 'Gemma']
        else:
            # Image datasets  
            models = ['CLIP', 'Llama']
    
    all_summary_rows = []
    
    for model_name in models:
        for superdetector in [False, True]:
            method_name = "Superdetector" if superdetector else "Regular"
            
            try:
                # Get results for this model/method combination
                results_df = summarize_best_inversion_metrics(
                    metric_type=metric_type,
                    concept_schemes=concept_schemes, 
                    dataset_name=dataset_name,
                    model_name=model_name,
                    superdetector_inversion=superdetector,
                    weighted_avg=weighted_avg,
                    include_baselines=False,  # Add baselines only once at the end
                    percentthrumodel=percentthrumodel,
                    sample_types=sample_types  # Pass through sample_types
                )
                
                # Add model and method columns
                for _, row in results_df.iterrows():
                    new_row = row.to_dict()
                    new_row['Model'] = model_name
                    new_row['Method'] = method_name
                    # Reorder scheme name to include model/method info
                    new_row['Scheme'] = f"{model_name} {method_name} {row['Scheme']}"
                    all_summary_rows.append(new_row)
                    
            except Exception as e:
                print(f"Warning: Could not process {model_name} {method_name}: {e}")
                continue
    
    # Add baselines once if requested and show_baseline is True
    if include_baselines and models and show_baseline:
        # Use first available model for baselines
        try:
            baseline_df = summarize_best_inversion_metrics(
                metric_type=metric_type,
                concept_schemes=[],  # Empty to skip concept schemes 
                dataset_name=dataset_name,
                model_name=models[0],
                superdetector_inversion=False,
                weighted_avg=weighted_avg,
                include_baselines=True,
                percentthrumodel=percentthrumodel,
                sample_types=sample_types
            )
            
            for _, row in baseline_df.iterrows():
                if 'Baseline' in row['Scheme']:
                    new_row = row.to_dict()
                    new_row['Model'] = 'N/A'
                    new_row['Method'] = 'Baseline'
                    all_summary_rows.append(new_row)
        except:
            pass
    
    if not all_summary_rows:
        print(f"No results found for dataset {dataset_name}")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_summary_rows)
    
    # Sort by Model first, then Method, then by metric value (descending)
    metric_col = f'Best Avg {metric_type.upper()}'
    if metric_col in df.columns:
        df = df.sort_values(['Model', 'Method', metric_col], ascending=[True, True, False])
    
    # Display the DataFrame directly
    print(f"\n=== {dataset_name} - All Inversion Methods Comparison ({metric_type.upper()}) ===")
    display(df)
    
    # Save to CSV file if requested
    if save_table:
        import os
        save_dir = '../Inversion_Tables'
        os.makedirs(save_dir, exist_ok=True)
        
        # Create filename with dataset name and metric type
        save_path = f'{save_dir}/{dataset_name}_all_inversion_methods_{metric_type}.csv'
        df.to_csv(save_path, index=False)
        print(f"\nTable saved to: {save_path}")
    
    return df


def summarize_detection_across_datasets(dataset_names, model_name, concept_type, sample_types, split='test', metric='f1', weighted_avg=True, percentthrumodel=100, include_errors=True, sorted=False, baselines=None):
    """
    Displays a table comparing detection scores across multiple datasets for a single model and concept type.
    Uses per-concept optimal percentiles from calibration data.
    
    Args:
        dataset_names: List of dataset names to compare
        model_name: Single model name (e.g., 'CLIP' or 'Llama')
        concept_type: Single concept type with its full specification:
                     - For supervised: 'avg' or 'linsep'
                     - For unsupervised: 'kmeans' or 'linsep kmeans'
                     - For SAE: 'sae'
        sample_types: List of sample types (e.g., ['cls', 'patch']) or single sample type string
        split: Data split to use (default: 'test')
        metric: Metric to evaluate (default: 'f1')
        weighted_avg: Whether to use weighted average (default: True)
        percentthrumodel: Percentage through model for embeddings (default: 100)
        include_errors: Whether to include ± error bars from bootstrap CI (default: True)
        sorted: Whether to sort results by metric value from highest to lowest (default: False)
        baselines: List of baseline methods to include. Options: ['random', 'prompt', 'maxtoken', 'meantoken', 'lasttoken', 'randomtoken']
                   If None, no baselines are included
    """
    import os
    import torch
    import pandas as pd
    import json
    
    # Handle single sample type as string
    if isinstance(sample_types, str):
        sample_types = [sample_types]
    
    results = []
    
    for dataset_name in dataset_names:
        for sample_type in sample_types:
            print(f"DEBUG: Processing {dataset_name} - {sample_type}")
            # Load ground-truth labels
            if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                if model_name == 'Llama':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
                elif model_name == 'Gemma':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
                elif model_name == 'Qwen':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
                else:
                    pass  # Silently skip unsupported model-dataset combinations
                    continue
            elif model_name == 'CLIP':
                gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt'
            elif model_name == 'Llama':
                gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt'
            else:
                pass  # Silently skip unsupported model-dataset combinations
                continue
                
            if not os.path.exists(gt_path):
                print(f"DEBUG: GT file not found: {gt_path}")
                continue
                
            gt_samples_per_concept = torch.load(gt_path, weights_only=False)
            gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
            
            # Construct concept label
            n_clusters = 1000 if sample_type == 'patch' else 50
            
            if concept_type == 'avg':
                con_label = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
            elif concept_type == 'linsep':
                con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
            elif concept_type == 'kmeans':
                con_label = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
            elif concept_type == 'linsep kmeans':
                con_label = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
            elif concept_type == 'sae':
                # SAE has specific percentthrumodel requirements
                if model_name == 'CLIP' and percentthrumodel != 92:
                    pass  # SAE concepts for CLIP only available at percentthrumodel=92
                    continue
                elif model_name == 'Gemma' and percentthrumodel != 81:
                    pass  # SAE concepts for Gemma only available at percentthrumodel=81
                    continue
                con_label = f'{model_name}_sae_{sample_type}_dense'
            else:
                pass  # Unknown concept type
                continue
            
            # Load best percentiles per concept from calibration
            best_percentiles_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
            
            if not os.path.exists(best_percentiles_path):
                print(f"DEBUG: Best percentiles file not found: {best_percentiles_path}")
                continue
                
            best_percentiles = torch.load(best_percentiles_path, weights_only=False)
            
            # Collect per-concept scores using their optimal percentiles
            concept_scores = []
            concept_weights = []
            concept_errors = []
            
            # Check if this is an unsupervised method
            is_unsupervised = 'kmeans' in con_label or 'sae' in con_label
            
            if is_unsupervised:
                # Load best clusters mapping for unsupervised methods
                best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                if not os.path.exists(best_clusters_path):
                    pass  # Best clusters file not found
                    continue
                best_clusters = torch.load(best_clusters_path, weights_only=False)
            
            for concept in gt_samples_per_concept:
                if concept not in best_percentiles:
                    continue
                    
                # Get the best percentile for this specific concept
                percentile = best_percentiles[concept]['best_percentile']
                
                try:
                    if is_unsupervised:
                        # Unsupervised methods use CSV format with allpairs
                        detection_metrics = pd.read_csv(f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{con_label}.csv')
                        
                        if concept in best_clusters:
                            cluster_id = best_clusters[concept]['best_cluster']
                            # Find the row for this (concept, cluster) pair
                            row = detection_metrics[detection_metrics['concept'] == f"('{concept}', '{cluster_id}')"]
                            if not row.empty:
                                score = row.iloc[0][metric]
                                concept_scores.append(score)
                                concept_weights.append(len(gt_samples_per_concept[concept]))
                                
                                # Try to load confidence interval if include_errors is True
                                if include_errors:
                                    ci_file = f'Quant_Results_with_CI/{dataset_name}/per_concept_ci_optimal_{con_label}.csv'
                                    if os.path.exists(ci_file):
                                        try:
                                            ci_df = pd.read_csv(ci_file)
                                            concept_row = ci_df[ci_df['concept'] == concept]
                                            if not concept_row.empty:
                                                error_col = f'{metric}_error'
                                                if error_col in concept_row.columns:
                                                    error = concept_row.iloc[0][error_col]
                                                    concept_errors.append(error)
                                                else:
                                                    concept_errors.append(0)
                                            else:
                                                concept_errors.append(0)
                                        except:
                                            concept_errors.append(0)
                                    else:
                                        concept_errors.append(0)
                    else:
                        # Supervised methods use PT format
                        detection_metrics = torch.load(f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.pt', weights_only=False)
                        
                        # Find the row for this concept
                        concept_row = detection_metrics[detection_metrics['concept'] == concept]
                        if not concept_row.empty:
                            score = concept_row.iloc[0][metric]
                            concept_scores.append(score)
                            concept_weights.append(len(gt_samples_per_concept[concept]))
                            
                            # Try to load confidence interval if include_errors is True
                            if include_errors:
                                ci_file = f'Quant_Results_with_CI/{dataset_name}/per_concept_ci_optimal_{con_label}.csv'
                                if os.path.exists(ci_file):
                                    try:
                                        ci_df = pd.read_csv(ci_file)
                                        concept_row = ci_df[ci_df['concept'] == concept]
                                        if not concept_row.empty:
                                            error_col = f'{metric}_error'
                                            if error_col in concept_row.columns:
                                                error = concept_row.iloc[0][error_col]
                                                concept_errors.append(error)
                                            else:
                                                concept_errors.append(0)
                                        else:
                                            concept_errors.append(0)
                                    except:
                                        concept_errors.append(0)
                                else:
                                    concept_errors.append(0)
                            
                except FileNotFoundError:
                    pass  # Detection metrics not found for this percentile
                    continue
            
            # Calculate final score and error
            print(f"DEBUG: Found {len(concept_scores)} concept scores for {dataset_name} - {sample_type}")
            if concept_scores:
                if weighted_avg:
                    total_weight = sum(concept_weights)
                    weighted_score = sum(s * w for s, w in zip(concept_scores, concept_weights)) / total_weight
                    final_score = weighted_score
                    # Calculate weighted error if available
                    if include_errors and concept_errors and len(concept_errors) == len(concept_weights):
                        weighted_error = sum(e * w for e, w in zip(concept_errors, concept_weights)) / total_weight
                    else:
                        weighted_error = 0
                else:
                    final_score = sum(concept_scores) / len(concept_scores)
                    # Calculate mean error if available
                    if include_errors and concept_errors:
                        weighted_error = sum(concept_errors) / len(concept_errors)
                    else:
                        weighted_error = 0
            else:
                final_score = 0
                weighted_error = 0
                
            # Try to load dataset-level CI if available
            if include_errors:
                dataset_ci_file = f'Quant_Results_with_CI/{dataset_name}/dataset_ci_optimal_{con_label}.json'
                if os.path.exists(dataset_ci_file):
                    try:
                        with open(dataset_ci_file, 'r') as f:
                            dataset_ci = json.load(f)
                        # Use weighted or macro based on weighted_avg setting
                        metric_key = f"{'weighted' if weighted_avg else 'macro'}_{metric}"
                        if metric_key in dataset_ci:
                            weighted_error = dataset_ci[metric_key].get('error', weighted_error)
                    except:
                        pass
            
            # Create result entry
            detection_method = 'CLS' if sample_type == 'cls' else 'SuperTokens'
            result_entry = {
                'Dataset': dataset_name,
                'Model': model_name,
                'Sample Type': sample_type,
                'Concept Method': concept_type,
                'Detection Method': detection_method,
                f'Best {metric.upper()}': round(final_score, 3)
            }
        
        # Add error column if requested
        if include_errors:
            if weighted_error > 0:
                result_entry[f'Best {metric.upper()}'] = f"{final_score:.3f} ± {weighted_error:.3f}"
            else:
                result_entry[f'Best {metric.upper()}'] = f"{final_score:.3f}"
                
            print(f"DEBUG: Appending result for {dataset_name} - {sample_type} - {concept_type}")
            results.append(result_entry)
    
    # Add baselines if requested
    if baselines:
        for dataset_name in dataset_names:
            # Load ground truth for baseline calculation
            if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                if model_name == 'Llama':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
                elif model_name == 'Gemma':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
                elif model_name == 'Qwen':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
                else:
                    continue
            elif model_name == 'CLIP':
                gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt'
            elif model_name == 'Llama':
                gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt'
            else:
                continue
                
            if os.path.exists(gt_path):
                gt_samples_per_concept = torch.load(gt_path, weights_only=False)
                gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
                
                # Add prompt baseline if requested (only for rate-based metrics, not counts)
                if 'prompt' in baselines and metric not in ['fp', 'fn', 'tp', 'tn']:
                    try:
                        prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
                        prompt_error = 0
                        
                        # Try to load prompt CI if available
                        if include_errors:
                            # First try the per-concept CSV to compute weighted error
                            prompt_ci_file = f'Quant_Results_with_CI/{dataset_name}/per_concept_ci_direct_prompt_Llama-3.2-11B-Vision.csv'
                            if os.path.exists(prompt_ci_file):
                                try:
                                    ci_df = pd.read_csv(prompt_ci_file)
                                    # Filter to concepts that are in ground truth
                                    ci_df = ci_df[ci_df['concept'].isin(gt_samples_per_concept.keys())]
                                    
                                    error_col = f'{metric}_error'
                                    if error_col in ci_df.columns:
                                        if weighted_avg:
                                            # Compute weighted average of errors
                                            total_weight = sum(len(gt_samples_per_concept[c]) for c in ci_df['concept'] if c in gt_samples_per_concept)
                                            weighted_error_sum = 0
                                            for _, row in ci_df.iterrows():
                                                concept = row['concept']
                                                if concept in gt_samples_per_concept:
                                                    weight = len(gt_samples_per_concept[concept])
                                                    weighted_error_sum += row[error_col] * weight
                                            prompt_error = weighted_error_sum / total_weight if total_weight > 0 else 0
                                        else:
                                            # Simple average of errors
                                            prompt_error = ci_df[error_col].mean()
                                except Exception as e:
                                    # Fallback to dataset-level CI if available
                                    dataset_ci_file = f'Quant_Results_with_CI/{dataset_name}/dataset_ci_direct_prompt_Llama-3.2-11B-Vision.json'
                                    if os.path.exists(dataset_ci_file):
                                        try:
                                            with open(dataset_ci_file, 'r') as f:
                                                dataset_ci = json.load(f)
                                            metric_key = f"{'weighted' if weighted_avg else 'macro'}_{metric}"
                                            if metric_key in dataset_ci:
                                                prompt_error = dataset_ci[metric_key].get('error', 0)
                                        except:
                                            pass
                        
                        result_entry = {
                            'Dataset': dataset_name,
                            'Model': model_name,
                            'Sample Type': 'N/A',
                            'Concept Method': 'prompt',
                            'Detection Method': 'N/A',
                            f'Best {metric.upper()}': round(prompt_score, 3)
                        }
                        
                        if include_errors:
                            if prompt_error > 0:
                                result_entry[f'Best {metric.upper()}'] = f"{prompt_score:.3f} ± {prompt_error:.3f}"
                            else:
                                result_entry[f'Best {metric.upper()}'] = f"{prompt_score:.3f}"
                                
                        results.append(result_entry)
                    except Exception as e:
                        pass  # Could not compute prompt baseline
                
                # Add random baseline if requested
                if 'random' in baselines and metric not in ['fp', 'fn', 'tp', 'tn']:
                    try:
                        random_score = get_weighted_random_score(dataset_name, model_name, gt_samples_per_concept, metric, split)
                        result_entry = {
                            'Dataset': dataset_name,
                            'Model': model_name,
                            'Sample Type': 'N/A',
                            'Concept Method': 'random',
                            'Detection Method': 'N/A',
                            f'Best {metric.upper()}': f"{random_score:.3f}"
                        }
                        results.append(result_entry)
                    except Exception as e:
                        pass  # Could not compute random baseline
                
                # Add other token-based baselines if requested
                token_baselines = {'maxtoken', 'meantoken', 'lasttoken', 'randomtoken', 'avgtoken'}
                requested_token_baselines = set(baselines) & token_baselines
                
                if requested_token_baselines and metric not in ['fp', 'fn', 'tp', 'tn']:
                    # Import baseline functions
                    try:
                        from utils.baseline_detection_utils import get_baseline_scores_for_dataset
                        
                        for baseline_type in requested_token_baselines:
                            # Map avgtoken to meantoken if needed
                            actual_baseline = 'meantoken' if baseline_type == 'avgtoken' else baseline_type
                            
                            try:
                                baseline_scores = get_baseline_scores_for_dataset(
                                    dataset_name, 
                                    model_name,
                                    actual_baseline,
                                    gt_samples_per_concept,
                                    metric,
                                    split
                                )
                                
                                if baseline_scores is not None and baseline_scores > 0:
                                    result_entry = {
                                        'Dataset': dataset_name,
                                        'Model': model_name,
                                        'Sample Type': 'N/A',
                                        'Concept Method': baseline_type,
                                        'Detection Method': 'N/A',
                                        f'Best {metric.upper()}': f"{baseline_scores:.3f}"
                                    }
                                    results.append(result_entry)
                            except:
                                pass  # Could not compute this baseline
                    except ImportError:
                        pass  # baseline_detection_utils not available
    
    # Create and display results DataFrame
    print(f"DEBUG: Total results before DataFrame: {len(results)}")
    if results:
        df = pd.DataFrame(results)
        
        # Sort if requested
        if sorted and f'Best {metric.upper()}' in df.columns:
            # Extract numeric values for sorting if errors are included
            if include_errors:
                sort_values = df[f'Best {metric.upper()}'].apply(lambda x: float(x.split(' ±')[0]) if ' ±' in str(x) else float(x))
                df = df.iloc[sort_values.argsort()[::-1]]
            else:
                df = df.sort_values(by=f'Best {metric.upper()}', ascending=False)
        
        # Save table to file
        save_dir = '../Figs/Paper_Tables'
        os.makedirs(save_dir, exist_ok=True)
        
        # Create filename with dataset info, model, and concept type
        datasets_str = '_'.join(dataset_names[:3])  # Limit to first 3 datasets to avoid overly long filenames
        if len(dataset_names) > 3:
            datasets_str += f'_and_{len(dataset_names)-3}_more'
        
        # Replace spaces in concept type with underscores
        concept_type_clean = concept_type.replace(' ', '_')
        
        filename = f"detection_comparison_{datasets_str}_{model_name}_{concept_type_clean}_{metric}.csv"
        save_path = os.path.join(save_dir, filename)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        print(f"Table saved to: {save_path}")
        
        return df
    else:
        return pd.DataFrame()


# Removed summarize_best_equal_percentile_inversion_metrics as it's no longer needed with calibration approach


""" Precision/Recall Curves """
from sklearn.metrics import auc


def get_style_label_color(key, pr_auc):
    style_map = {
    'avg': {'color': 'orchid', 'type': 'supervised', 'label': 'patch avg'},
    'linsep': {'color': 'indigo', 'type': 'supervised', 'label': 'patch linsep'},
    'labeled cls avg': {'color': 'goldenrod', 'type': 'supervised', 'label': 'cls avg'},
    'labeled cls linsep': {'color': 'orangered', 'type': 'supervised', 'label': 'cls linsep'},
    'unsupervised patch kmeans': {'color': 'orchid', 'type': 'unsupervised', 'label': 'patch avg'},
    'unsupervised patch linsep kmeans': {'color': 'indigo', 'type': 'unsupervised', 'label': 'patch linsep'},
    'unsupervised cls kmeans': {'color': 'goldenrod', 'type': 'unsupervised', 'label': 'cls avg'},
    'unsupervised cls linsep kmeans': {'color': 'orangered', 'type': 'unsupervised', 'label': 'cls linsep'},
    }
    style = style_map.get(key, {})
    color = style.get('color', 'gray')
    label = f"{style.get('label', key)} (AUC={pr_auc:.2f})"
    return label, color
      
def plot_pr_curves_across_methods(
    dataset_name,
    split,
    model_name,
    sample_types,
    save_path=None,
    weighted=False,
    ax=None,
    style_map=None,
):

    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    con_labels = {}

    for sample_type in sample_types:
        n_clusters = 1000 if sample_type == "patch" else 50
        con_labels[f"labeled {sample_type} avg"] = f"{model_name}_avg_{sample_type}_embeddings_percentthrumodel_100"
        con_labels[f"labeled {sample_type} linsep"] = f"{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_100"

    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
    elif model_name == 'ClIP':
        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt"
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt' 
    gt_concepts = torch.load(gt_path, weights_only=False)

    if weighted:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))

        for method_name, con_label in con_labels.items():
            weighted_prec, weighted_rec = [0.0] * len(percentiles), [0.0] * len(percentiles)
            total_weight = 0.0

            for concept, gt_idxs in gt_concepts.items():
                weight = len(gt_idxs)
                prec, rec = [], []
                for p in percentiles:
                    file_name = f"Quant_Results/{dataset_name}/detectionmetrics_per_{p}_{con_label}.pt"
                    if not os.path.exists(file_name):
                        continue
                    df = torch.load(file_name, weights_only=False)
                    row = df[df["concept"] == concept]
                    if row.empty:
                        continue
                    prec.append(row["precision"].iloc[0])
                    rec.append(row["recall"].iloc[0])

                if len(prec) == len(percentiles):
                    weighted_prec = [wp + pr * weight for wp, pr in zip(weighted_prec, prec)]
                    weighted_rec = [wr + rc * weight for wr, rc in zip(weighted_rec, rec)]
                    total_weight += weight

            if total_weight == 0:
                continue

            avg_prec = [wp / total_weight for wp in weighted_prec]
            avg_rec = [wr / total_weight for wr in weighted_rec]
            pr_auc = auc(avg_rec, avg_prec)
            best_idx = max(range(len(avg_prec)), key=lambda i: (2 * avg_prec[i] * avg_rec[i]) / (avg_prec[i] + avg_rec[i] + 1e-8))
            best_f1 = (2 * avg_prec[best_idx] * avg_rec[best_idx]) / (avg_prec[best_idx] + avg_rec[best_idx] + 1e-8)
            
            label, color = get_style_label_color(method_name, pr_auc)
            plt.plot(avg_rec, avg_prec, label=label, color=color)
            plt.plot([avg_rec[best_idx]], [avg_prec[best_idx]], marker = 'o', color=color)
            plt.plot([0, avg_rec[best_idx]], [avg_prec[best_idx]] * 2, linestyle='--', color=color, alpha=0.6)
            plt.plot([avg_rec[best_idx]] * 2, [0, avg_prec[best_idx]], linestyle='--', color=color, alpha=0.6)
            plt.text(avg_rec[best_idx], avg_prec[best_idx], f"Best F1={best_f1:.2f}", fontsize=10, color=color)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(title='Concept Type', loc="best")
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, "pr_weighted_all_methods.png"), dpi=300)
        plt.show()


def plot_pr_curves_patch_level(
    dataset_name,
    split,
    model_name,
    concept_schemes,
    percentiles,
    justobj=False,
    weighted=False,
    save_path=None,
    style_map=None
):

    base_dir = f"Quant_Results/{dataset_name}"
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
    elif model_name == 'CLIP':
        gt_path = f"GT_Samples/{dataset_name}/gt_patch_per_concept_{split}_inputsize_(224, 224).pt"
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_patch_per_concept_{split}_inputsize_(560, 560).pt' 
        
    gt_samples = torch.load(gt_path, weights_only=False)


    scheme_to_label = {
        'avg': '_avg_patch_embeddings_percentthrumodel_100.pt',
        'linsep': '_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_100.pt',
        'unsupervised kmeans': '_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_100.pt',
        'unsupervised kmeans linsep': '_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_100.pt',
    }

    if weighted:
        plt.figure(figsize=(8, 6))
        for scheme in concept_schemes:
            weighted_prec, weighted_rec = [0.0]*len(percentiles), [0.0]*len(percentiles)
            total_w = 0.0

            for concept, idxs in gt_samples.items():
                w = len(idxs)
                precs, recs = [], []
                for p in percentiles:
                    fp = f"Quant_Results/{dataset_name}/detectionmetrics_per_{p}_{model_name}{scheme_to_label[scheme]}"
                    if not os.path.exists(fp):
                        print(fp, "doesn't exist")
                        break
                    df = torch.load(fp, weights_only=False)
                    if 'kmeans' in scheme:
                        con_label = f'{model_name}_kmeans_1000{"_linsep" if "linsep" in scheme else ""}_patch_embeddings_kmeans_percentthrumodel_100'
                        align_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                        align_results = torch.load(align_path, weights_only=False)
                        cluster_to_concept = {str(info['best_cluster']): concept for concept, info in align_results.items()}
                        df = df.copy()
                        df['concept'] = df['concept'].astype(str).map(cluster_to_concept)

                    row = df[df['concept'] == concept]
                    if row.empty:
                        break
                    precs.append(row['precision'].iloc[0])
                    recs.append(row['recall'].iloc[0])
                else:
                    total_w += w
                    for i in range(len(percentiles)):
                        weighted_prec[i] += precs[i] * w
                        weighted_rec[i] += recs[i] * w

            if total_w > 0:
                avg_prec = [wp / total_w for wp in weighted_prec]
                avg_rec = [wr / total_w for wr in weighted_rec]
                pr_auc = auc(avg_rec, avg_prec)
                best_idx = max(range(len(avg_prec)), key=lambda i: (2 * avg_prec[i] * avg_rec[i]) / (avg_prec[i] + avg_rec[i] + 1e-8))
                best_f1 = (2 * avg_prec[best_idx] * avg_rec[best_idx]) / (avg_prec[best_idx] + avg_rec[best_idx] + 1e-8)

                label, color = get_style_label_color(scheme, pr_auc)
                plt.plot(avg_rec, avg_prec, label=label, color=color)
                plt.plot([avg_rec[best_idx]], [avg_prec[best_idx]], marker = 'o', color=color)
                plt.plot([0, avg_rec[best_idx]], [avg_prec[best_idx]] * 2, linestyle='--', color=color, alpha=0.6)
                plt.plot([avg_rec[best_idx]] * 2, [0, avg_prec[best_idx]], linestyle='--', color=color, alpha=0.6)
                plt.text(avg_rec[best_idx], avg_prec[best_idx], f"Best F1={best_f1:.2f}", fontsize=10, color=color)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(title='Concept Type', loc='best')
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, "pr_weighted_patch_level.png"), dpi=300)
        plt.show()


def plot_f1_scores_horizontal_bar(dataset_names, split, model_names, sample_types, weighted_avg=True, 
                                 concept_types=None, percentthrumodel=100, show_baselines=False, 
                                 sorted=True, figsize=None, save_fig=True, save_dir='../Figs/Paper_Figs',
                                 label_font_size=None, title_font_size=None):
    """
    Plots F1 scores as a vertical bar chart for different detection schemes.
    Supports multiple datasets plotted side-by-side.
    
    Args:
        dataset_names: Name of the dataset (string) or list of dataset names for side-by-side plots
        split: Data split to use ('test', 'val', etc.)
        model_names: List of model names (e.g., ['CLIP', 'Llama']) or single model name string
        sample_types: List of sample types (e.g., ['cls', 'patch'])
        weighted_avg: Whether to use weighted average (default: True)
        concept_types: List of concept types to include. Options: ['avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae']
        percentthrumodel: Percentage through model for embeddings (default: 100)
        show_baselines: Whether to include random and prompt baselines (default: False)
        sorted: Whether to sort by F1 score (default: True)
        figsize: Figure size as tuple (width, height) (default: None, auto-calculated based on number of datasets)
        save_fig: Whether to save the figure (default: True)
        save_dir: Directory to save figures (default: '../Figs/Paper_Figs')
        label_font_size: Font size for axis labels (default: None, uses paper style default)
        title_font_size: Font size for the title (default: None, uses paper style default)
    """
    # Handle single dataset name as string
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # Handle single model name as string
    if isinstance(model_names, str):
        model_names = [model_names]
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (8 * len(dataset_names), 6)
    
    # Apply paper plotting style
    paper_style = get_paper_plotting_style()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(dataset_names), figsize=figsize, sharey=True)
    if len(dataset_names) == 1:
        axes = [axes]  # Make it a list for consistency
    
    # Store all dataframes for return
    all_dfs = []
    
    # Apply the paper style to current figure
    with plt.rc_context(paper_style):
        # Process each dataset
        for idx, dataset_name in enumerate(dataset_names):
            ax = axes[idx]
            
            # Get the data using the existing function
            results = []
            metric = 'f1'
            
            # Loop through each model
            for model_name in model_names:
                # Load ground-truth labels
                if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                    if model_name == 'Llama':
                        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
                    elif model_name == 'Gemma':
                        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
                    elif model_name == 'Qwen':
                        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
                    gt_samples_per_concept = torch.load(gt_path, weights_only=False)
                elif model_name == 'CLIP':
                    gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
                elif model_name == 'Llama':
                    gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
                else:
                    continue
                gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
                
                # Construct concept label mappings
                for sample_type in sample_types:
                    n_clusters = 1000 if sample_type == 'patch' else 50
                    con_labels = {}
                    
                    # Supervised methods
                    if concept_types is None or 'avg' in concept_types:
                        con_labels['avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
                    if concept_types is None or 'linsep' in concept_types:
                        con_labels['linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
                    # Unsupervised methods
                    if concept_types is None or 'kmeans' in concept_types:
                        con_labels['kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
                    if concept_types is None or 'linsep kmeans' in concept_types:
                        con_labels['linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
                    # Add SAE for supported models and datasets
                    if concept_types is None or 'sae' in concept_types:
                        # CLIP SAE for vision datasets (only available at percentthrumodel=92)
                        if model_name == 'CLIP' and sample_type == 'patch':
                                if percentthrumodel == 92:
                                    con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                                else:
                                    print(f"Warning: SAE concepts for CLIP are only available at percentthrumodel=92, not {percentthrumodel}")
                        # Gemma SAE for text datasets (only available at percentthrumodel=81)
                        elif model_name == 'Gemma' and sample_type == 'patch':
                                if percentthrumodel == 81:
                                    con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                                else:
                                    print(f"Warning: SAE concepts for Gemma are only available at percentthrumodel=81, not {percentthrumodel}")

                    for name, con_label in con_labels.items():
                        # Load best percentiles per concept from calibration
                        best_percentiles_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
                        
                        if not os.path.exists(best_percentiles_path):
                            continue
                            
                        best_percentiles = torch.load(best_percentiles_path, weights_only=False)
                        
                        # Collect per-concept scores using their optimal percentiles
                        concept_scores = []
                        concept_weights = []
                        
                        # Check if this is an unsupervised method
                        is_unsupervised = 'kmeans' in con_label or 'sae' in con_label
                        
                        if is_unsupervised:
                            # Load best clusters mapping for unsupervised methods
                            best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                            if not os.path.exists(best_clusters_path):
                                continue
                            best_clusters = torch.load(best_clusters_path, weights_only=False)
                        
                        for concept in gt_samples_per_concept:
                            if concept not in best_percentiles:
                                continue
                                
                            # Get the best percentile for this specific concept
                            percentile = best_percentiles[concept]['best_percentile']
                            
                            try:
                                if is_unsupervised:
                                    # Unsupervised methods use CSV format with allpairs
                                    detection_metrics = pd.read_csv(f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{con_label}.csv')
                                    
                                    if concept in best_clusters:
                                        cluster_id = best_clusters[concept]['best_cluster']
                                        # Find the row for this (concept, cluster) pair
                                        row = detection_metrics[detection_metrics['concept'] == f"('{concept}', '{cluster_id}')"]
                                        if not row.empty:
                                            score = row.iloc[0][metric]
                                            concept_scores.append(score)
                                            concept_weights.append(len(gt_samples_per_concept[concept]))
                                else:
                                    # Supervised methods use PT format
                                    detection_metrics = torch.load(f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.pt', weights_only=False)
                                    
                                    # Find the row for this concept
                                    concept_row = detection_metrics[detection_metrics['concept'] == concept]
                                    if not concept_row.empty:
                                        score = concept_row.iloc[0][metric]
                                        concept_scores.append(score)
                                        concept_weights.append(len(gt_samples_per_concept[concept]))
                                        
                            except FileNotFoundError:
                                continue
                        
                        # Calculate final score
                        if concept_scores:
                            if weighted_avg:
                                total_weight = sum(concept_weights)
                                weighted_score = sum(s * w for s, w in zip(concept_scores, concept_weights)) / total_weight
                                final_score = weighted_score
                            else:
                                final_score = sum(concept_scores) / len(concept_scores)
                        else:
                            final_score = 0
                            
                        results.append({
                            'Model': model_name,
                            'Sample Type': sample_type,
                            'Method': name,
                            'F1': final_score,
                            'Is_Unsupervised': 'kmeans' in name or 'sae' in name
                        })
            
            # Add baselines if requested
            if show_baselines and model_names:
                model_name = model_names[0]
                # Load ground truth for baseline calculation
                if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                    if model_name == 'Llama':
                        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
                    elif model_name == 'Gemma':
                        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
                    elif model_name == 'Qwen':
                        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
                    gt_samples_per_concept = torch.load(gt_path, weights_only=False)
                elif model_name == 'CLIP':
                    gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
                elif model_name == 'Llama':
                    gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
                gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
                
                # Add prompt baseline
                try:
                    prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
                    results.append({
                        'Model': 'Baseline',
                        'Sample Type': 'baseline',
                        'Method': 'Prompt',
                        'F1': prompt_score,
                        'Is_Unsupervised': False
                    })
                except Exception:
                    pass
                
                # Add random baseline
                baseline_path = f'Quant_Results/{dataset_name}/random_{model_name}_cls_baseline.csv'
                if os.path.exists(baseline_path):
                    try:
                        df = pd.read_csv(baseline_path)
                        df = df[df['concept'].isin(gt_samples_per_concept)]
                        if weighted_avg:
                            total = sum(len(gt_samples_per_concept[c]) for c in df['concept'])
                            score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) for _, row in df.iterrows()) / total
                        else:
                            score = df[metric].mean()
                        results.append({
                            'Model': 'Baseline',
                            'Sample Type': 'baseline',
                            'Method': 'Random',
                            'F1': score,
                            'Is_Unsupervised': False
                        })
                    except Exception:
                        pass
            
            # Create DataFrame for this dataset
            df = pd.DataFrame(results)
            all_dfs.append(df)
            
            # Now process and plot this dataset
            if not df.empty:
                # Create labels based on available configurations
                unique_models = df['Model'].unique()
                unique_samples = df['Sample Type'].unique()
                
                # Build labels for each row
                labels = []
                for i, row in df.iterrows():
                    if row['Model'] == 'Baseline':
                        # Baselines get their method name
                        labels.append(row['Method'])
                    else:
                        # For supervised/unsupervised pairs, check if this is part of a pair
                        is_pair = False
                        if i > 0:
                            prev_row = df.iloc[i-1]
                            if (prev_row['Sample Type'] == row['Sample Type'] and 
                                ((prev_row['Method'] == 'avg' and row['Method'] == 'kmeans') or
                                 (prev_row['Method'] == 'linsep' and row['Method'] == 'linsep kmeans'))):
                                # This is the second in a pair, use same label as previous
                                label = 'SD Token' if row['Sample Type'] == 'patch' else 'CLS'
                                labels.append(label)
                                is_pair = True
                        if not is_pair and i < len(df) - 1:
                            next_row = df.iloc[i+1]
                            if (next_row['Sample Type'] == row['Sample Type'] and 
                                ((row['Method'] == 'avg' and next_row['Method'] == 'kmeans') or
                                 (row['Method'] == 'linsep' and next_row['Method'] == 'linsep kmeans'))):
                                # This is the first in a pair
                                label = 'SD Token' if row['Sample Type'] == 'patch' else 'CLS'
                                labels.append(label)
                                is_pair = True
                        
                        if not is_pair:
                            # Not part of a pair
                            parts = []
                            
                            # Only include model if there are multiple models
                            non_baseline_models = [m for m in unique_models if m != 'Baseline']
                            if len(non_baseline_models) > 1:
                                parts.append(row['Model'])
                            
                            # For patch/cls, just use the sample type label without method name
                            if row['Sample Type'] in ['patch', 'cls']:
                                sample_label = 'SD Token' if row['Sample Type'] == 'patch' else 'CLS'
                                parts.append(sample_label)
                            else:
                                # For other types (like baselines), keep the original behavior
                                # Only include sample type if there are multiple sample types
                                non_baseline_samples = [s for s in unique_samples if s != 'baseline']
                                if len(non_baseline_samples) > 1:
                                    sample_label = 'SD Token' if row['Sample Type'] == 'patch' else 'CLS'
                                    parts.append(sample_label)
                                
                                # Add method name
                                if row['Method'] == 'avg' or row['Method'] == 'kmeans':
                                    method_name = 'centroid'
                                elif row['Method'] == 'linsep' or row['Method'] == 'linsep kmeans':
                                    method_name = 'separator'
                                elif row['Method'] == 'sae':
                                    method_name = 'SAE'
                                else:
                                    method_name = row['Method']
                                
                                parts.append(method_name)
                            
                            labels.append(' '.join(parts))
                
                df['Label'] = labels
                
                # Custom sorting: patch first, then cls, then baselines
                # Within each sample type, group supervised/unsupervised pairs together
                def sort_key(row):
                    sample_order = {'patch': 0, 'cls': 1, 'baseline': 2}
                    sample_rank = sample_order.get(row['Sample Type'], 3)
                    
                    # Group centroid and separator concepts together
                    if row['Method'] == 'avg' or row['Method'] == 'kmeans':
                        method_group = 0  # centroid
                    elif row['Method'] == 'linsep' or row['Method'] == 'linsep kmeans':
                        method_group = 1  # separator
                    elif row['Method'] == 'sae':
                        method_group = 2  # sae
                    else:
                        method_group = 3  # others
                    
                    # Put supervised before unsupervised within each group
                    is_unsupervised = 1 if row['Is_Unsupervised'] else 0
                    
                    return (sample_rank, method_group, is_unsupervised, row.name)
                
                df['sort_key'] = df.apply(sort_key, axis=1)
                df = df.sort_values('sort_key').drop('sort_key', axis=1)
                df = df.reset_index(drop=True)
                
                # Reverse the order (bottom to top becomes top to bottom)
                df = df.iloc[::-1].reset_index(drop=True)
            
            # Format dataset name for title
            if dataset_name.startswith('Broden-'):
                title_name = dataset_name[7:]  # Remove "Broden-" prefix
            elif dataset_name.lower() == 'coco':
                title_name = 'COCO'
            else:
                title_name = dataset_name
            
            # Create grouped bars for supervised/unsupervised pairs
            # First, identify groups and create positions
            groups = []
            y_labels = []
            i = 0
            
            while i < len(df):
                curr_row = df.iloc[i]
                # Check if this starts a supervised/unsupervised pair
                if i + 1 < len(df):
                    next_row = df.iloc[i + 1]
                    # After reversal, unsupervised comes before supervised
                    is_pair = (curr_row['Sample Type'] == next_row['Sample Type'] and 
                              curr_row['Model'] != 'Baseline' and next_row['Model'] != 'Baseline' and
                              ((curr_row['Method'] == 'kmeans' and next_row['Method'] == 'avg') or
                               (curr_row['Method'] == 'linsep kmeans' and next_row['Method'] == 'linsep')))
                    
                    if is_pair:
                        # This is a pair
                        groups.append([i, i+1])
                        # Use capitalized label for groups
                        sample_type = df.iloc[i]['Sample Type']
                        label = 'SD Token' if sample_type == 'patch' else 'CLS' if sample_type == 'cls' else df.iloc[i]['Label']
                        y_labels.append(label)
                        i += 2
                        continue
                # Single bar
                groups.append([i])
                y_labels.append(df.iloc[i]['Label'])
                i += 1
            
            # Create x positions for groups with equal spacing
            x_pos = np.arange(len(groups))  # Equal spacing between all bars
            
            # Color scheme based on sample type
            color_map = {
                'patch': '#ff7f0e',       # Orange for SD Token
                'cls': '#9467bd',         # Purple for CLS
                'baseline': '#808080',    # Gray for baselines
                'prompt': '#8B4513'       # Brown for prompt
            }
            
            # Determine colors for each bar
            colors = []
            has_unsupervised = False
            has_supervised = False
            
            for _, row in df.iterrows():
                if row['Method'] == 'Prompt':
                    colors.append(color_map['prompt'])
                elif row['Method'] == 'Random' or row['Method'] == 'Rand':
                    colors.append(color_map['baseline'])
                elif row['Sample Type'] == 'patch':
                    colors.append(color_map['patch'])
                elif row['Sample Type'] == 'cls':
                    colors.append(color_map['cls'])
                else:
                    colors.append('#17becf')  # Cyan for others
                
                # Track if we have both supervised and unsupervised
                if row['Model'] != 'Baseline':
                    if row['Is_Unsupervised']:
                        has_unsupervised = True
                    else:
                        has_supervised = True
            
            # Create the vertical bars using proper grouping
            bar_width = 0.3  # Width of each individual bar
            gap = 0  # No gap between grouped bars
            
            for group_idx, group_indices in enumerate(groups):
                if len(group_indices) == 2:
                    # This is a supervised/unsupervised pair - create tightly grouped bars
                    # After reversal: first is unsupervised, second is supervised
                    # Unsupervised bar (left)
                    unsup_idx = group_indices[0]
                    unsup_row = df.iloc[unsup_idx]
                    ax.bar(x_pos[group_idx] - bar_width/2 - gap/2, unsup_row['F1'], 
                           width=bar_width, color=colors[unsup_idx], alpha=0.8,
                           hatch='///', edgecolor='white', linewidth=0.5)
                    
                    # Supervised bar (right)
                    sup_idx = group_indices[1]
                    sup_row = df.iloc[sup_idx]
                    ax.bar(x_pos[group_idx] + bar_width/2 + gap/2, sup_row['F1'], 
                           width=bar_width, color=colors[sup_idx], alpha=0.8)
                else:
                    # Single bar (not part of a pair)
                    df_idx = group_indices[0]
                    row = df.iloc[df_idx]
                    if row['Is_Unsupervised'] and row['Model'] != 'Baseline':
                        # Unsupervised single bar
                        ax.bar(x_pos[group_idx], row['F1'], width=bar_width*2, 
                               color=colors[df_idx], alpha=0.8,
                               hatch='///', edgecolor='white', linewidth=0.5)
                    else:
                        # Supervised or baseline single bar
                        ax.bar(x_pos[group_idx], row['F1'], width=bar_width*2, 
                               color=colors[df_idx], alpha=0.8)
            
            
            # Customize the plot
            # Set axis labels with specified font size or default from paper style
            if label_font_size is None:
                label_font_size = paper_style['axes.labelsize']
            ax.set_xlabel('Detection Schemes', fontsize=label_font_size)
            if idx == 0:  # Only show y-label on first subplot
                ax.set_ylabel('F1', fontsize=label_font_size)
            
            # Add title with dataset name
            if title_font_size is None:
                title_font_size = paper_style['font.size'] + 2  # Slightly larger than default
            ax.set_title(title_name, fontsize=title_font_size)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(y_labels, rotation=45, ha='right', fontsize=paper_style['xtick.labelsize'])  # Rotate labels for readability
            # Set ytick label size
            ax.tick_params(axis='y', labelsize=paper_style['ytick.labelsize'])
            ax.set_ylim(0, 1)  # F1 score ranges from 0 to 1
            # Reduced grid - only show major ticks
            ax.grid(axis='y', alpha=0.2, linestyle='--', which='major')
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])  # Only show 5 grid lines
            
            # Add legend if we have both supervised and unsupervised
            if has_supervised and has_unsupervised and idx == len(dataset_names) - 1:
                from matplotlib.patches import Patch
                # Use a neutral color for the legend
                legend_elements = [
                    Patch(facecolor='gray', alpha=0.8, label='Supervised'),
                    Patch(facecolor='gray', alpha=0.8, hatch='///', 
                          edgecolor='white', linewidth=0.5, label='Unsupervised')
                ]
                # Place legend on top using bbox_to_anchor
                ax.legend(handles=legend_elements, loc='upper center', 
                         bbox_to_anchor=(0.5, 1.15), ncol=2, framealpha=0.9)
        
        
        # Use tight_layout with padding to accommodate legend
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave 8% space on top for legend
        
        # Save if requested
        if save_fig:
            # Generate automatic filename (use first dataset name only)
            dataset_str = dataset_names[0]  # Use only first dataset for filename
            models_str = '_'.join(model_names)
            samples_str = '_'.join(sample_types)
            concepts_str = '_'.join(concept_types) if concept_types else 'all'
            filename = f"{dataset_str}_{models_str}_{samples_str}_{concepts_str}_f1_vertical_bar.pdf"
            save_path = os.path.join(save_dir, filename)
            
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    # Return the DataFrames for further analysis if needed
    return all_dfs if len(all_dfs) > 1 else all_dfs[0]
def plot_percentile_histogram(dataset_names, model_name, concept_type, sample_type='patch', 
                             percentthrumodel=100, split='test', figsize=(10, 6), 
                             save_fig=True, save_dir='../Figs/Histograms', colors=None, 
                             label_fontsize=14, ticklabel_fontsize=12, legend_fontsize=14, 
                             y_label=None, x_label=None):
    """
    Plots a histogram of optimal percentiles for given datasets with a model and concept type.
    Uses fixed bins at: 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    
    Args:
    dataset_names: Single dataset name (string) or list of dataset names
    model_name: Model name (e.g., 'CLIP', 'Llama', 'Gemma')
    concept_type: Type of concept ('avg', 'linsep', 'kmeans', 'linsep kmeans', or 'sae')
    sample_type: Sample type ('cls' or 'patch') (default: 'patch')
    percentthrumodel: Percentage through model for embeddings (default: 100)
    split: Data split to use (default: 'test')
    figsize: Figure size as tuple (width, height) (default: (10, 6))
    save_fig: Whether to save the figure (default: True)
    save_dir: Directory to save figures (default: '../Figs/Histograms')
    colors: List of colors for each dataset (default: None, uses matplotlib color cycle)
    label_fontsize: Font size for axis labels (default: 14)
    ticklabel_fontsize: Font size for tick labels (default: 12)
    legend_fontsize: Font size for legend (default: 14)
    y_label: Custom y-axis label (default: None, uses 'SuperActivator N')
    x_label: Custom x-axis label (default: None, uses '# Concepts that achieve highest F₁ at this Sparsity Level')
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import torch
    from collections import Counter
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure dataset_names is a list
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # Set up colors - use predefined palette
    if colors is None:
        # Use matplotlib's 'tab10' palette - bright, distinct colors
        colors = plt.cm.tab10.colors[:len(dataset_names)]
    
    # Fixed bins
    fixed_bins = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_bins = len(fixed_bins) - 1
    
    # Collect bin counts for all datasets
    all_bin_counts = {}
    
    for dataset_idx, dataset_name in enumerate(dataset_names):
            # For SAE concept type, automatically determine model and percentthrumodel
            if concept_type == 'sae':
                # Check if text dataset
                is_text_dataset = (dataset_name == 'Stanford-Tree-Bank' or 
                                 'Sarcasm' in dataset_name or 
                                 'Emotion' in dataset_name)
                
                if is_text_dataset:
                    actual_model_name = 'Gemma'
                    actual_percentthrumodel = 81
                else:
                    actual_model_name = 'CLIP'
                    actual_percentthrumodel = 81  # Use 81 as requested
            else:
                actual_model_name = model_name
                actual_percentthrumodel = percentthrumodel
            
            # Load ground truth to filter concepts
            if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                if actual_model_name == 'Llama':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
                elif actual_model_name == 'Gemma':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
                elif actual_model_name == 'Qwen':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
                else:
                    raise ValueError(f"Unknown model for text: {actual_model_name}")
            elif actual_model_name == 'CLIP':
                gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt'
            elif actual_model_name == 'Llama' and dataset_name not in ['Stanford-Tree-Bank'] and 'Sarcasm' not in dataset_name and 'Emotion' not in dataset_name:
                gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt'
            else:
                raise ValueError(f"Unknown model/dataset combination: {actual_model_name}/{dataset_name}")
                
            if not os.path.exists(gt_path):
                print(f"Warning: Ground truth file not found for {dataset_name}: {gt_path}")
                continue
                
            gt_samples_per_concept = torch.load(gt_path, weights_only=False)
            
            # Filter concepts
            gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
            valid_concepts = set(gt_samples_per_concept.keys())
            
            # Build concept label based on type
            n_clusters = 1000 if sample_type == 'patch' else 50
            
            if concept_type == 'avg':
                con_label = f'{actual_model_name}_avg_{sample_type}_embeddings_percentthrumodel_{actual_percentthrumodel}'
            elif concept_type == 'linsep':
                con_label = f'{actual_model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{actual_percentthrumodel}'
            elif concept_type == 'kmeans':
                con_label = f'{actual_model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{actual_percentthrumodel}'
            elif concept_type == 'linsep kmeans':
                con_label = f'{actual_model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{actual_percentthrumodel}'
            elif concept_type == 'sae':
                # Special handling for SAE - now using actual_model_name which is already set correctly
                if actual_model_name == 'CLIP' and sample_type == 'patch':
                    con_label = f'{actual_model_name}_sae_{sample_type}_dense'
                elif actual_model_name == 'Gemma' and sample_type == 'patch':
                    con_label = f'{actual_model_name}_sae_{sample_type}_dense'
                else:
                    raise ValueError(f"SAE not available for {actual_model_name} with sample_type={sample_type}")
            else:
                raise ValueError(f"Unknown concept type: {concept_type}")
            
            # Load best percentiles
            best_percentiles_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
            
            if not os.path.exists(best_percentiles_path):
                print(f"Warning: Best percentiles file not found for {dataset_name}: {best_percentiles_path}")
                continue
                
            best_percentiles = torch.load(best_percentiles_path, weights_only=False)
            
            # Collect percentiles for valid concepts
            percentiles_list = []
            for concept in valid_concepts:
                if concept in best_percentiles:
                    percentile = best_percentiles[concept]['best_percentile']
                    percentiles_list.append(percentile)
            
            if not percentiles_list:
                print(f"Warning: No valid percentiles found for {dataset_name}")
                continue
            
            # Count how many percentiles fall into each bin
            bin_counts = []
            for i in range(len(fixed_bins)-1):
                count = sum(1 for p in percentiles_list if fixed_bins[i] <= p < fixed_bins[i+1])
                bin_counts.append(count)
            
            all_bin_counts[dataset_name] = bin_counts
    
    # Create horizontal stacked bar chart with equally spaced positions
    # Use equally spaced y positions for clarity that these are bins
    y_positions = np.arange(n_bins)
    
    # Use a bar height that leaves gaps between bars
    bar_height = 0.6  # Height that leaves visible gaps (slightly thicker)
    
    # Initialize left positions for stacking
    left = np.zeros(n_bins)
    
    for idx, (dataset_name, bin_counts) in enumerate(all_bin_counts.items()):
        # Clean up dataset name for display
        display_name = dataset_name
        if dataset_name == 'Coco' or dataset_name == 'COCO':
            display_name = 'COCO'
        elif dataset_name.startswith('Broden-'):
            display_name = dataset_name.replace('Broden-', '')
        
        # Plot horizontal bars with visible separation
        bars = ax.barh(y_positions, bin_counts, bar_height, 
                       left=left,
                       label=display_name, 
                       color=colors[idx], 
                       edgecolor='black', linewidth=0.5, alpha=0.9)
        
        # Update left for next dataset
        left += np.array(bin_counts)
    
    # Customize plot
    if y_label is None:
        y_label = 'SuperActivator N'
    if x_label is None:
        x_label = '# Concepts that achieve highest F₁ at this Sparsity Level'
    
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    
    # Move y-axis label closer to ticks
    ax.yaxis.labelpad = 4  # Reduce padding between label and axis
    
    # Set y-axis with equally spaced positions and single value labels
    ax.set_yticks(y_positions)
    bin_labels = [f'{int(val*100)}%' for val in fixed_bins[:-1]]
    ax.set_yticklabels(bin_labels, fontsize=ticklabel_fontsize)
    ax.tick_params(axis='x', labelsize=ticklabel_fontsize)
    
    # Add grid for x-axis only to help read values
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Extend x-axis to create space at right
    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0], current_xlim[1] * 1.08)  # Add 8% more space at right
    
    # No title - removed per request
    
    # Add legend if multiple datasets
    if len(all_bin_counts) > 1:
        if concept_type == 'sae':
            # Place legend at bottom right corner inside the plot for SAE
            ax.legend(fontsize=legend_fontsize, 
                      loc='lower right',
                      ncol=3,
                      frameon=True,
                      columnspacing=1.0)
        else:
            ax.legend(fontsize=legend_fontsize, 
                      loc='upper right',
                      ncol=3,
                      frameon=True,
                      columnspacing=1.0)
    
    # No grid - removed per request
    
    plt.tight_layout(pad=3.0)  # Add padding around the plot
    plt.subplots_adjust(top=0.85, bottom=0.15)  # Add more padding at the top and bottom
    
    if save_fig:
            os.makedirs(save_dir, exist_ok=True)
            # Create filename based on datasets
            if len(all_bin_counts) == 1:
                filename = f'percentile_histogram_{list(all_bin_counts.keys())[0]}_{model_name}_{concept_type}_{sample_type}_percentthru{percentthrumodel}.png'
            else:
                datasets_str = '_'.join(all_bin_counts.keys())
                filename = f'percentile_histogram_multi_{datasets_str}_{model_name}_{concept_type}_{sample_type}_percentthru{percentthrumodel}.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Figure saved to {os.path.join(save_dir, filename)}")
    
    plt.show()
    
    return fig, ax

def plot_percentile_freqs(dataset_name, model_name, split='test', concept_types=None, sample_types=['cls', 'patch'], 
                         percentthrumodel=100, figsize=(8, 6), save_fig=True, save_dir='../Figs/Paper_Figs'):
    """
    Plots the frequency of optimal percentiles across concepts as a bar chart.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Model name (e.g., 'CLIP', 'Llama')
        split: Data split to use (default: 'test')
        concept_types: List of concept types to include. Options: ['avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae']
        sample_types: List of sample types (e.g., ['cls', 'patch'])
        percentthrumodel: Percentage through model for embeddings (default: 100)
        figsize: Figure size as tuple (width, height) (default: (8, 6))
        save_fig: Whether to save the figure (default: True)
        save_dir: Directory to save figures (default: '../Figs/Paper_Figs')
    """
    # Apply paper plotting style
    paper_style = get_paper_plotting_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Apply the paper style to current figure
    with plt.rc_context(paper_style):
        # First, load ground truth to filter concepts
        if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
            if model_name == 'Llama':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
            elif model_name == 'Gemma':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
            elif model_name == 'Qwen':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
            gt_samples_per_concept = torch.load(gt_path, weights_only=False)
        elif model_name == 'CLIP':
            gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
        elif model_name == 'Llama':
            gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        # Filter concepts
        gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
        valid_concepts = set(gt_samples_per_concept.keys())
        
        # Collect percentile frequencies for each method
        all_percentile_counts = {}
        
        for sample_type in sample_types:
            n_clusters = 1000 if sample_type == 'patch' else 50
            con_labels = {}
            
            # Build concept labels based on requested types
            if concept_types is None or 'avg' in concept_types:
                con_labels['avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'linsep' in concept_types:
                con_labels['linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'kmeans' in concept_types:
                con_labels['kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'linsep kmeans' in concept_types:
                con_labels['linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'sae' in concept_types:
                # CLIP SAE for vision datasets (only available at percentthrumodel=92)
                if model_name == 'CLIP' and sample_type == 'patch':
                        if percentthrumodel == 92:
                            con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                        else:
                            print(f"Warning: SAE concepts for CLIP are only available at percentthrumodel=92, not {percentthrumodel}")
                # Gemma SAE for text datasets (only available at percentthrumodel=81)
                elif model_name == 'Gemma' and sample_type == 'patch':
                        if percentthrumodel == 81:
                            con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                        else:
                            print(f"Warning: SAE concepts for Gemma are only available at percentthrumodel=81, not {percentthrumodel}")
            
            for name, con_label in con_labels.items():
                # Load best percentiles per concept from calibration
                best_percentiles_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
                
                if not os.path.exists(best_percentiles_path):
                    print(f"Warning: Best percentiles file not found - {best_percentiles_path}")
                    continue
                    
                best_percentiles = torch.load(best_percentiles_path, weights_only=False)
                
                # Count percentile frequencies
                percentile_counts = defaultdict(int)
                
                for concept in valid_concepts:
                    if concept in best_percentiles:
                        percentile = best_percentiles[concept]['best_percentile']
                        percentile_counts[percentile] += 1
                
                # Store counts with a descriptive key
                method_key = f"{sample_type} - {name}"
                all_percentile_counts[method_key] = dict(percentile_counts)
        
        # Get all unique percentiles
        all_percentiles = set()
        for counts in all_percentile_counts.values():
            all_percentiles.update(counts.keys())
        percentiles_sorted = sorted(all_percentiles)
        
        # Use actual percentile values for x-axis (continuous scale)
        x_values = [p * 100 for p in percentiles_sorted]  # Convert to percentage
        
        # Color mapping
        colors = {
            'patch': '#ff7f0e',  # Orange
            'cls': '#9467bd',    # Purple
        }
        
        # Plot lines for each method
        for method_key, counts in all_percentile_counts.items():
            # Get y values for this method
            y_values = [counts.get(p, 0) for p in percentiles_sorted]
            
            # Determine color based on sample type
            color = colors['patch'] if 'patch' in method_key else colors['cls']
            
            # Determine if unsupervised (for line style)
            is_unsupervised = 'kmeans' in method_key or 'sae' in method_key
            linestyle = ':' if is_unsupervised else '-'
            
            # Extract method name for label
            parts = method_key.split(' - ')
            if len(parts) == 2:
                sample_type, method = parts
                label = f"{sample_type.capitalize()} - {method}"
            else:
                label = method_key
            
            # Plot line with continuous x values
            ax.plot(x_values, y_values, color=color, linestyle=linestyle, 
                   linewidth=2, marker='o', markersize=3, 
                   label=label, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Optimal Recall Percentile', fontsize=paper_style['legend.fontsize'])
        ax.set_ylabel('# Concepts', fontsize=paper_style['legend.fontsize'])
        
        # Set x-ticks to show every 10% but only label 10, 30, 50, 70, 90
        x_ticks = [i for i in range(0, 101, 10)]
        ax.set_xticks(x_ticks)
        x_labels = []
        for i in x_ticks:
            if i in [10, 30, 50, 70, 90]:
                x_labels.append(f"{i}%")
            else:
                x_labels.append("")
        ax.tick_params(axis='x', pad=2)  # Reduce padding between labels and axis
        ax.set_xticklabels(x_labels, rotation=45, ha='right', rotation_mode='anchor')
        
        # Set x-axis limits
        ax.set_xlim(0, 100)
        
        # Force y-axis to use integer values only with at most 4 ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))
        
        # Set y-axis upper limit to max count + 0.5 or + 2
        max_count = 0
        for counts in all_percentile_counts.values():
            max_count = max(max_count, max(counts.values()) if counts else 0)
        
        if max_count <= 1:
            ax.set_ylim(0, max_count + 0.5)
        else:
            ax.set_ylim(0, max_count + 2)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add legend
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                 framealpha=0.9, fontsize=paper_style['legend.fontsize'])
        
        # Use tight_layout with padding to accommodate legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave 15% space on the right for legend
        
        # Save if requested
        if save_fig:
            # Generate automatic filename
            samples_str = '_'.join(sample_types)
            concepts_str = '_'.join(concept_types) if concept_types else 'all'
            filename = f"{dataset_name}_{model_name}_{samples_str}_{concepts_str}_percentile_freqs.pdf"
            save_path = os.path.join(save_dir, filename)
            
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        
    return all_percentile_counts

def plot_detection_scores_on_axis(ax, dataset_name, split, model_name, sample_types, 
                                 metric='f1', weighted_avg=True, concept_types=None, 
                                 baseline_types=None, percentthrumodel=100,
                                 label_font_size=12, legend_font=None, xlabel=None, ylim=None,
                                 show_legend=True, axes_font=None):
    """
    Plot detection scores on a given axis (simplified version of plot_detection_scores).
    """
    if legend_font is None:
        legend_font = label_font_size
    if axes_font is None:
        axes_font = legend_font
        
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95]
    
    # Load ground truth
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Gemma':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        else:
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt"
    else:
        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt"
    
    if os.path.exists(gt_path):
        gt_samples_per_concept = torch.load(gt_path, weights_only=False)
        gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    else:
        return
    
    # Plot baselines
    baseline_lines = []
    
    # Random baseline
    if baseline_types is None or 'random' in baseline_types:
        baseline_path = f'Quant_Results/{dataset_name}/random_{model_name}_cls_baseline.csv'
        if os.path.exists(baseline_path):
            df_baseline = pd.read_csv(baseline_path)
            df_baseline = df_baseline[df_baseline['concept'].isin(gt_samples_per_concept)]
            if weighted_avg:
                total = sum(len(gt_samples_per_concept[c]) for c in df_baseline['concept'])
                score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) for _, row in df_baseline.iterrows()) / total
            else:
                score = df_baseline[metric].mean()
            baseline_lines.append(ax.axhline(score, color='#808080', linestyle='-.', linewidth=1.75, label='Random'))
    
    # Prompt baseline
    if baseline_types is None or 'prompt' in baseline_types:
        try:
            prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
            baseline_lines.append(ax.axhline(prompt_score, color='#8B4513', linestyle='-.', linewidth=1.75, label='Prompt'))
        except:
            pass
    
    # Plot concept methods - use colors that match the bars above
    # Color map matching the bars: patch='#ff7f0e', cls='#9467bd'
    style_map = {
        'avg': {'type': 'supervised', 'label': 'Centroid'},
        'linsep': {'type': 'supervised', 'label': 'Separator'},
        'kmeans': {'type': 'unsupervised', 'label': 'Centroid'},
        'linsep kmeans': {'type': 'unsupervised', 'label': 'Separator'},
        'sae': {'type': 'sae', 'label': 'SAE'}
    }
    
    lines = []
    # Reverse order so CLS is plotted first and Token/patch lines appear on top
    for sample_type in reversed(sample_types):
        n_clusters = 1000 if sample_type == 'patch' else 50
        con_labels = {}
        
        if concept_types is None or 'avg' in concept_types:
            con_labels['avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
        if concept_types is None or 'linsep' in concept_types:
            con_labels['linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
        if concept_types is None or 'kmeans' in concept_types:
            con_labels['kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
        if concept_types is None or 'linsep kmeans' in concept_types:
            con_labels['linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
        
        for name, con_label in con_labels.items():
            scores = []
            for percentile in percentiles:
                file_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.pt'
                if not os.path.exists(file_path):
                    # Try CSV for unsupervised
                    file_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.csv'
                    if not os.path.exists(file_path):
                        scores.append(0)
                        continue
                    detection_metrics = pd.read_csv(file_path)
                else:
                    detection_metrics = torch.load(file_path, weights_only=False)
                
                detection_metrics = detection_metrics[detection_metrics['concept'].isin(gt_samples_per_concept)]
                if weighted_avg:
                    total = sum(len(gt_samples_per_concept[c]) for c in detection_metrics['concept'])
                    if total > 0:
                        score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) for _, row in detection_metrics.iterrows()) / total
                    else:
                        score = 0
                else:
                    score = detection_metrics[metric].mean() if len(detection_metrics) > 0 else 0
                scores.append(score)
            
            style = style_map[name]
            # Use colors matching the bars: patch='#e65100', cls='#7b1fa2'
            color = '#e65100' if sample_type == 'patch' else '#7b1fa2'
            kind = style['type']
            label = f"{'TOKEN' if sample_type == 'patch' else sample_type.upper()} - {style['label']}"
            linestyle = '-' if kind == 'supervised' else ':'
            
            line = ax.plot(percentiles, scores, color=color, linestyle=linestyle, 
                          marker='o', markersize=1.5, linewidth=0.875, label=label)[0]
            lines.append(line)
    
    # Formatting
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_font_size - 1)
    else:
        ax.set_xlabel("Ground Truth Concept Recall Percentage", fontsize=label_font_size - 1)
    ax.set_ylabel("F1" if metric == 'f1' else f"{metric.upper()} Score", fontsize=label_font_size, rotation=0, ha='right')
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1)
    
    # Set ticks only where we have labels
    tick_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    tick_labels = [f"{int(pos*100)}%" for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=axes_font)
    ax.tick_params(axis='x', pad=-2)  # Move labels up closer to axis
    ax.tick_params(axis='y', pad=2)  # Set y-axis padding to match top plot
    
    # Set y-axis ticks to multiples of 0.2
    ylim_min, ylim_max = ax.get_ylim()
    yticks = np.arange(0, ylim_max + 0.1, 0.2)
    yticks = yticks[(yticks >= ylim_min) & (yticks <= ylim_max)]
    # Remove 0 from labels but keep the tick
    ylabels = ['' if tick == 0 else f'{tick:.1f}' for tick in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    # Add legend only if requested
    if show_legend:
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), 
                 fontsize=legend_font, frameon=True)


def plot_combined_f1_and_detection(dataset_name, model_names, split='test', sample_types=['cls', 'patch'],
                                   concept_types=None, baseline_types=None, percentthrumodel=100,
                                   show_baselines=True, metric='f1', weighted_avg=True,
                                   figsize=(10, 5, 5), save_fig=True, save_dir='../Figs/Paper_Figs',
                                   label_font_size=12, legend_font=None, axes_font=None, title=None, xlabel=None, ylims=None,
                                   flatten_legend=False, x_positions=None, hide_top_xticks=False):
    """
    Creates a combined plot with F1 scores (top) and detection scores (bottom).
    
    Args:
        dataset_name: Name of the dataset
        model_names: List of model names or single model name
        split: Data split to use (default: 'test')
        sample_types: List of sample types (e.g., ['cls', 'patch'])
        concept_types: List of concept types to include
        baseline_types: List of baseline types to include in detection plot
        percentthrumodel: Percentage through model for embeddings
        show_baselines: Whether to show baselines in F1 plot
        metric: Metric to plot in detection scores (default: 'f1')
        weighted_avg: Whether to use weighted average in detection scores
        figsize: Figure size as (width, top_plot_height, bottom_plot_height)
        save_fig: Whether to save the figures
        save_dir: Directory to save figures
        label_font_size: Font size for x and y axis labels
        legend_font: Font size for legend. If None, uses label_font_size
        axes_font: Font size for tick labels (F1 values, concept types, percentages). If None, uses legend_font
        title: Title for the overall figure (if None, no title)
        xlabel: Custom x-axis label for detection plot
        ylims: Tuple of (ylim_top, ylim_bottom) for setting y-axis limits on both plots
        flatten_legend: Whether to arrange legend items horizontally (default: False)
        x_positions: Tuple of (rand_x, prompt_x, cls_x, token_x) for custom x positions in top plot
        hide_top_xticks: Whether to hide x-axis ticks and labels on top plot (default: False)
    """
    # Handle single model name
    if isinstance(model_names, str):
        model_names = [model_names]
    
    # Use legend_font if provided, otherwise use label_font_size
    if legend_font is None:
        legend_font = label_font_size
    if axes_font is None:
        axes_font = legend_font
    
    # Apply paper plotting style
    paper_style = get_paper_plotting_style()
    
    # Extract dimensions from figsize
    width, top_height, bottom_height = figsize
    total_height = top_height + bottom_height
    height_ratios = [top_height, bottom_height]
    
    # Create main figure with two subplots
    fig = plt.figure(figsize=(width, total_height))
    # Reduce vertical spacing if hiding top x-ticks
    hspace_val = 0.05 if hide_top_xticks else 0.45
    gs = fig.add_gridspec(2, 1, height_ratios=height_ratios, hspace=hspace_val)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    with plt.rc_context(paper_style):
        # ========== TOP PLOT: F1 Scores (Horizontal Bar) ==========
        plt.sca(ax1)
        
        # Get F1 data
        results = []
        metric = 'f1'
        
        # Loop through each model (similar to plot_f1_scores_horizontal_bar)
        for model_name in model_names:
            # Load ground-truth labels
            if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                if model_name == 'Llama':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
                elif model_name == 'Gemma':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
                elif model_name == 'Qwen':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
                gt_samples_per_concept = torch.load(gt_path, weights_only=False)
            elif model_name == 'CLIP':
                gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
            elif model_name == 'Llama':
                gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
            else:
                continue
            gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
            
            # Construct concept label mappings
            for sample_type in sample_types:
                n_clusters = 1000 if sample_type == 'patch' else 50
                con_labels = {}
                
                # Supervised methods
                if concept_types is None or 'avg' in concept_types:
                    con_labels['avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
                if concept_types is None or 'linsep' in concept_types:
                    con_labels['linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
                # Unsupervised methods
                if concept_types is None or 'kmeans' in concept_types:
                    con_labels['kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
                if concept_types is None or 'linsep kmeans' in concept_types:
                    con_labels['linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
                # Add SAE
                if concept_types is None or 'sae' in concept_types:
                    if model_name == 'CLIP' and sample_type == 'patch':
                            if percentthrumodel == 92:
                                con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                            else:
                                print(f"Warning: SAE concepts for CLIP are only available at percentthrumodel=92, not {percentthrumodel}")
                    elif model_name == 'Gemma' and sample_type == 'patch':
                            if percentthrumodel == 81:
                                con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                            else:
                                print(f"Warning: SAE concepts for Gemma are only available at percentthrumodel=81, not {percentthrumodel}")

                for name, con_label in con_labels.items():
                    # Load best percentiles per concept
                    best_percentiles_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
                    
                    if not os.path.exists(best_percentiles_path):
                        continue
                        
                    best_percentiles = torch.load(best_percentiles_path, weights_only=False)
                    
                    # Collect per-concept scores
                    concept_scores = []
                    concept_weights = []
                    
                    # Check if unsupervised
                    is_unsupervised = 'kmeans' in con_label or 'sae' in con_label
                    
                    if is_unsupervised:
                        best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                        if not os.path.exists(best_clusters_path):
                            continue
                        best_clusters = torch.load(best_clusters_path, weights_only=False)
                    
                    for concept in gt_samples_per_concept:
                        if concept not in best_percentiles:
                            continue
                            
                        percentile = best_percentiles[concept]['best_percentile']
                        
                        try:
                            if is_unsupervised:
                                detection_metrics = pd.read_csv(f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{con_label}.csv')
                                
                                if concept in best_clusters:
                                    cluster_id = best_clusters[concept]['best_cluster']
                                    row = detection_metrics[detection_metrics['concept'] == f"('{concept}', '{cluster_id}')"]
                                    if not row.empty:
                                        score = row.iloc[0][metric]
                                        concept_scores.append(score)
                                        concept_weights.append(len(gt_samples_per_concept[concept]))
                            else:
                                detection_metrics = torch.load(f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.pt', weights_only=False)
                                
                                concept_row = detection_metrics[detection_metrics['concept'] == concept]
                                if not concept_row.empty:
                                    score = concept_row.iloc[0][metric]
                                    concept_scores.append(score)
                                    concept_weights.append(len(gt_samples_per_concept[concept]))
                                    
                        except FileNotFoundError:
                            continue
                    
                    # Calculate final score
                    if concept_scores:
                        total_weight = sum(concept_weights)
                        weighted_score = sum(s * w for s, w in zip(concept_scores, concept_weights)) / total_weight
                        final_score = weighted_score
                    else:
                        final_score = 0
                        
                    results.append({
                        'Model': model_name,
                        'Sample Type': sample_type,
                        'Method': name,
                        'F1': final_score,
                        'Is_Unsupervised': 'kmeans' in name or 'sae' in name
                    })
        
        # Add baselines if requested
        if show_baselines and model_names:
            model_name = model_names[0]
            # Load ground truth for baseline calculation
            if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                if model_name == 'Llama':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
                elif model_name == 'Gemma':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
                elif model_name == 'Qwen':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
                gt_samples_per_concept = torch.load(gt_path, weights_only=False)
            elif model_name == 'CLIP':
                gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
            elif model_name == 'Llama':
                gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
            gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
            
            # Add prompt baseline
            try:
                prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
                results.append({
                    'Model': 'Baseline',
                    'Sample Type': 'baseline',
                    'Method': 'Prompt',
                    'F1': prompt_score,
                    'Is_Unsupervised': False
                })
            except Exception:
                pass
            
            # Add random baseline
            baseline_path = f'Quant_Results/{dataset_name}/random_{model_name}_cls_baseline.csv'
            if os.path.exists(baseline_path):
                try:
                    df_baseline = pd.read_csv(baseline_path)
                    df_baseline = df_baseline[df_baseline['concept'].isin(gt_samples_per_concept)]
                    total = sum(len(gt_samples_per_concept[c]) for c in df_baseline['concept'])
                    score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) for _, row in df_baseline.iterrows()) / total
                    results.append({
                        'Model': 'Baseline',
                        'Sample Type': 'baseline',
                        'Method': 'Rand',
                        'F1': score,
                        'Is_Unsupervised': False
                    })
                except Exception:
                    pass
        
        # Create DataFrame and process for plotting
        df = pd.DataFrame(results)
        
        if not df.empty:
            # Create labels
            unique_models = df['Model'].unique()
            unique_samples = df['Sample Type'].unique()
            
            labels = []
            for i, row in df.iterrows():
                if row['Model'] == 'Baseline':
                    labels.append(row['Method'])
                else:
                    is_pair = False
                    if i > 0:
                        prev_row = df.iloc[i-1]
                        if (prev_row['Sample Type'] == row['Sample Type'] and 
                            ((prev_row['Method'] == 'avg' and row['Method'] == 'kmeans') or
                             (prev_row['Method'] == 'linsep' and row['Method'] == 'linsep kmeans'))):
                            label = 'SD\nTokens' if row['Sample Type'] == 'patch' else 'CLS\n '
                            labels.append(label)
                            is_pair = True
                    if not is_pair and i < len(df) - 1:
                        next_row = df.iloc[i+1]
                        if (next_row['Sample Type'] == row['Sample Type'] and 
                            ((row['Method'] == 'avg' and next_row['Method'] == 'kmeans') or
                             (row['Method'] == 'linsep' and next_row['Method'] == 'linsep kmeans'))):
                            label = 'SD\nTokens' if row['Sample Type'] == 'patch' else 'CLS\n '
                            labels.append(label)
                            is_pair = True
                    
                    if not is_pair:
                        parts = []
                        non_baseline_models = [m for m in unique_models if m != 'Baseline']
                        if len(non_baseline_models) > 1:
                            parts.append(row['Model'])
                        
                        non_baseline_samples = [s for s in unique_samples if s != 'baseline']
                        if len(non_baseline_samples) > 1:
                            sample_label = 'SD\nTokens' if row['Sample Type'] == 'patch' else 'CLS'
                            parts.append(sample_label)
                        
                        if row['Method'] == 'avg' or row['Method'] == 'kmeans':
                            method_name = 'centroid'
                        elif row['Method'] == 'linsep' or row['Method'] == 'linsep kmeans':
                            method_name = 'separator'
                        elif row['Method'] == 'sae':
                            method_name = 'SAE'
                        else:
                            method_name = row['Method']
                        
                        parts.append(method_name)
                        labels.append(' - '.join(parts))
            
            df['Label'] = labels
            
            # Custom sorting
            def sort_key(row):
                sample_order = {'patch': 0, 'cls': 1, 'baseline': 2}
                sample_rank = sample_order.get(row['Sample Type'], 3)
                
                if row['Method'] == 'avg' or row['Method'] == 'kmeans':
                    method_group = 0
                elif row['Method'] == 'linsep' or row['Method'] == 'linsep kmeans':
                    method_group = 1
                elif row['Method'] == 'sae':
                    method_group = 2
                else:
                    method_group = 3
                
                is_unsupervised = 1 if row['Is_Unsupervised'] else 0
                
                return (sample_rank, method_group, is_unsupervised, row.name)
            
            df['sort_key'] = df.apply(sort_key, axis=1)
            df = df.sort_values('sort_key').drop('sort_key', axis=1)
            df = df.reset_index(drop=True)
            
            # Reverse the order
            df = df.iloc[::-1].reset_index(drop=True)
            
            # Create grouped bars
            groups = []
            group_labels = []
            i = 0
            
            while i < len(df):
                curr_row = df.iloc[i]
                if i + 1 < len(df):
                    next_row = df.iloc[i + 1]
                    # After reversal, unsupervised comes before supervised
                    is_pair = (curr_row['Sample Type'] == next_row['Sample Type'] and 
                              curr_row['Model'] != 'Baseline' and next_row['Model'] != 'Baseline' and
                              ((curr_row['Method'] == 'kmeans' and next_row['Method'] == 'avg') or
                               (curr_row['Method'] == 'linsep kmeans' and next_row['Method'] == 'linsep')))
                    
                    if is_pair:
                        groups.append([i, i+1])
                        sample_type = df.iloc[i]['Sample Type']
                        label = 'SD\nTokens' if sample_type == 'patch' else 'CLS\n ' if sample_type == 'cls' else df.iloc[i]['Label']
                        group_labels.append(label)
                        i += 2
                        continue
                groups.append([i])
                group_labels.append(df.iloc[i]['Label'])
                i += 1
            
            # Create x positions for groups with custom spacing
            x_pos = []
            
            if x_positions is not None:
                # Use custom x positions
                rand_x, prompt_x, cls_x, token_x = x_positions
                
                for i, group_indices in enumerate(groups):
                    # Determine what type of group this is
                    if len(group_indices) == 1:
                        row = df.iloc[group_indices[0]]
                        is_random = row['Method'] in ['Random', 'Rand']
                        is_prompt = row['Method'] == 'Prompt'
                    else:
                        # It's a paired group (supervised/unsupervised)
                        row = df.iloc[group_indices[0]]  # Check first item
                        is_random = False
                        is_prompt = False
                    
                    # Assign custom positions
                    if is_random:
                        x_pos.append(rand_x)
                    elif is_prompt:
                        x_pos.append(prompt_x)
                    elif row['Sample Type'] == 'cls':
                        x_pos.append(cls_x)
                    elif row['Sample Type'] == 'patch':
                        x_pos.append(token_x)
                    else:
                        # Default position for any other type
                        x_pos.append(0)
            else:
                # Use automatic spacing (original logic)
                current_x = 0
                
                for i, group_indices in enumerate(groups):
                    # Determine what type of group this is
                    if len(group_indices) == 1:
                        row = df.iloc[group_indices[0]]
                        is_random = row['Method'] in ['Random', 'Rand']
                        is_prompt = row['Method'] == 'Prompt'
                    else:
                        # It's a paired group (supervised/unsupervised)
                        row = df.iloc[group_indices[0]]  # Check first item
                        is_random = False
                        is_prompt = False
                    
                    # Shift specific bars left
                    if is_random:
                        x_pos.append(current_x - 0.3)  # Move Random left
                    elif is_prompt:
                        x_pos.append(current_x - 0.2)  # Move Prompt left
                    elif row['Sample Type'] == 'cls':
                        x_pos.append(current_x - 0.2)  # Move CLS groups left
                    elif row['Sample Type'] == 'patch':
                        x_pos.append(current_x - 0.2)  # Move Token groups left
                    else:
                        x_pos.append(current_x)
                    
                    # Add spacing after this group
                    if i < len(groups) - 1:  # Not the last group
                        next_group_indices = groups[i + 1]
                        if len(next_group_indices) == 1:
                            next_row = df.iloc[next_group_indices[0]]
                            next_is_prompt = next_row['Method'] == 'Prompt'
                            next_is_cls = next_row['Sample Type'] == 'cls'
                        else:
                            next_row = df.iloc[next_group_indices[0]]
                            next_is_prompt = False
                            next_is_cls = next_row['Sample Type'] == 'cls'
                        
                        # Add extra spacing between specific transitions
                        if is_random and next_is_prompt:
                            current_x += 2.5  # Extra space between Random and Prompt
                        elif is_prompt and next_is_cls:
                            current_x += 2.5  # Extra space between Prompt and CLS groups
                        elif row['Sample Type'] == 'cls' and next_row['Sample Type'] == 'patch':
                            current_x += 2.5  # Extra space between CLS and Token groups
                        else:
                            current_x += 1.2  # Normal spacing
            
            x_pos = np.array(x_pos)
            
            # Color scheme
            color_map = {
                'patch': '#e65100',  # Darker orange
                'cls': '#7b1fa2',    # Darker purple
                'baseline': '#808080',
                'prompt': '#8B4513'
            }
            
            # Determine colors
            colors = []
            for _, row in df.iterrows():
                if row['Method'] == 'Prompt':
                    colors.append(color_map['prompt'])
                elif row['Method'] == 'Random' or row['Method'] == 'Rand':
                    colors.append(color_map['baseline'])
                elif row['Sample Type'] == 'patch':
                    colors.append(color_map['patch'])
                elif row['Sample Type'] == 'cls':
                    colors.append(color_map['cls'])
                else:
                    colors.append('#17becf')
            
            # Create the vertical bars
            if x_positions is not None:
                # Use consistent bar width when custom positions are provided
                bar_width = 0.13  # Consistent width for all bars
                gap = 0  # No gap between paired bars
            else:
                # Use original variable widths
                bar_width = 0.6  # Double the width from 0.3 to 0.6
                gap = 0  # No gap between paired bars
            
            for group_idx, group_indices in enumerate(groups):
                if len(group_indices) == 2:
                    # Paired bars (supervised + unsupervised)
                    if x_positions is not None:
                        # Use consistent width with custom positions
                        # Unsupervised bar (left)
                        unsup_idx = group_indices[0]
                        unsup_row = df.iloc[unsup_idx]
                        ax1.bar(x_pos[group_idx] - bar_width, unsup_row['F1'], 
                               width=bar_width, color=colors[unsup_idx], alpha=0.8,
                               hatch='///', edgecolor='white', linewidth=0.5)
                        
                        # Supervised bar (right)
                        sup_idx = group_indices[1]
                        sup_row = df.iloc[sup_idx]
                        ax1.bar(x_pos[group_idx], sup_row['F1'], 
                               width=bar_width, color=colors[sup_idx], alpha=0.8)
                    else:
                        # Use original variable widths
                        wide_bar_width = bar_width * 2
                        # Unsupervised bar (left)
                        unsup_idx = group_indices[0]
                        unsup_row = df.iloc[unsup_idx]
                        ax1.bar(x_pos[group_idx] - wide_bar_width, unsup_row['F1'], 
                               width=wide_bar_width, color=colors[unsup_idx], alpha=0.8,
                               hatch='///', edgecolor='white', linewidth=0.5)
                        
                        # Supervised bar (right)
                        sup_idx = group_indices[1]
                        sup_row = df.iloc[sup_idx]
                        ax1.bar(x_pos[group_idx], sup_row['F1'], 
                               width=wide_bar_width, color=colors[sup_idx], alpha=0.8)
                else:
                    # Single bar
                    df_idx = group_indices[0]
                    row = df.iloc[df_idx]
                    
                    if x_positions is not None:
                        # Use consistent width for all single bars
                        single_bar_width = bar_width
                    else:
                        # Use original variable widths
                        if row['Model'] == 'Baseline':
                            single_bar_width = bar_width * 2
                        else:
                            # Non-baseline single bars get wider
                            single_bar_width = bar_width * 4
                    
                    if row['Is_Unsupervised'] and row['Model'] != 'Baseline':
                        ax1.bar(x_pos[group_idx], row['F1'], width=single_bar_width, 
                               color=colors[df_idx], alpha=0.8,
                               hatch='///', edgecolor='white', linewidth=0.5)
                    else:
                        ax1.bar(x_pos[group_idx], row['F1'], width=single_bar_width, 
                               color=colors[df_idx], alpha=0.8)
            
            # Customize F1 plot
            ax1.set_xlabel('', fontsize=label_font_size)  # No x-axis label
            ax1.set_ylabel('F1', fontsize=label_font_size, rotation=0, ha='right')
            if hide_top_xticks:
                # Hide x-axis ticks and labels
                ax1.set_xticks([])
                ax1.set_xticklabels([])
            else:
                ax1.set_xticks(x_pos)
                if x_positions is not None:
                    # When using custom positions, adjust label alignment and padding
                    ax1.set_xticklabels(group_labels, rotation=45, ha='center', fontsize=axes_font)
                    ax1.tick_params(axis='x', pad=0)  # Slightly closer to axis
                else:
                    ax1.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=axes_font)
                    ax1.tick_params(axis='x', pad=2)  # Move x-axis labels much closer
            ax1.tick_params(axis='y', which='major', labelsize=axes_font, pad=2)
            
            # Set x-axis limits explicitly to ensure bars aren't compressed
            if len(x_pos) > 0:
                if x_positions is not None:
                    # Use fixed x-axis limits when custom positions are provided
                    # Bars outside these limits will be cut off
                    ax1.set_xlim(0, 1)  # Fixed limits - adjust as needed
                else:
                    # Add padding based on bar width
                    padding = 0.5
                    ax1.set_xlim(x_pos[0] - padding, x_pos[-1] + padding)
                
            # Force aspect ratio to prevent compression
            ax1.set_aspect('auto')
            
            # Set y-axis limits first
            if ylims and ylims[0] is not None:
                ax1.set_ylim(ylims[0])
            else:
                ax1.set_ylim(0, 1)  # F1 score ranges from 0 to 1
            
            # Set y-axis ticks to multiples of 0.2
            ylim_min, ylim_max = ax1.get_ylim()
            yticks = np.arange(0, ylim_max + 0.1, 0.2)
            yticks = yticks[(yticks >= ylim_min) & (yticks <= ylim_max)]
            # Remove 0 from labels but keep the tick
            ylabels = ['' if tick == 0 else f'{tick:.1f}' for tick in yticks]
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(ylabels)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # ========== BOTTOM PLOT: Detection Scores ==========
        plt.sca(ax2)
        
        # Use the helper function to plot detection scores
        plot_detection_scores_on_axis(ax2, dataset_name, split, model_names[0], sample_types, 
                                     metric=metric, weighted_avg=weighted_avg, concept_types=concept_types,
                                     baseline_types=baseline_types, percentthrumodel=percentthrumodel,
                                     label_font_size=label_font_size, legend_font=legend_font, xlabel=xlabel,
                                     ylim=ylims[1] if ylims and len(ylims) > 1 else None,
                                     show_legend=False, axes_font=axes_font)
        
        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=label_font_size, fontweight='bold')
        
        # Save the figure
        if save_fig:
            # Construct filename
            model_str = '_'.join(model_names)
            sample_str = '_'.join(sample_types)
            filename = f'{model_str}_{dataset_name}_{sample_str}_f1_detection_combined.pdf'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined plot to: {save_path}")
        
        plt.show()
        
        # ========== CREATE SEPARATE LEGEND ==========
        with plt.rc_context(paper_style):
            fig_legend = plt.figure()
            ax_legend = fig_legend.add_subplot(111)
            ax_legend.axis('off')
            
            from matplotlib.patches import Patch
            # Darker colors: orange #e65100, purple #7b1fa2
            legend_elements = [
                Patch(facecolor='#e65100', alpha=0.8, label='SD Tokens (Ours)'),
                Patch(facecolor='#7b1fa2', alpha=0.8, label='CLS'),
                Patch(facecolor='white', alpha=0),  # Spacer
                Patch(facecolor='#8B4513', alpha=0.8, label='Prompt'),
                Patch(facecolor='#808080', alpha=0.8, label='Random'),
                Patch(facecolor='white', alpha=0),  # Spacer
                Patch(facecolor='black', alpha=0.8, label='Supervised'),
                Patch(facecolor='black', alpha=0.8, hatch='///', 
                      edgecolor='white', linewidth=0.5, label='Unsupervised')
            ]
            
            # Remove the spacers from handles but keep the space
            handles = [elem for elem in legend_elements if elem.get_facecolor() != (1.0, 1.0, 1.0, 0.0)]
            labels = ['SD Tokens (Ours)', 'CLS', '', 'Prompt', 'Random', '', 'Supervised', 'Unsupervised']
            
            # Create legend with proper spacing - always horizontal
            legend = ax_legend.legend(handles=handles, labels=[l for l in labels if l], 
                                     title='Detection Schemes',
                                     loc='center', frameon=True, fancybox=True, shadow=False,
                                     fontsize=paper_style['legend.fontsize'],
                                     title_fontsize=paper_style['legend.fontsize'],
                                     handlelength=2, handleheight=1.5,
                                     handletextpad=0.4,
                                     ncol=len(handles))  # All items in one row
            
            # Save legend
            if save_fig:
                legend_filename = f"{model_str}_{dataset_name}_{sample_str}_f1_detection_legend.pdf"
                legend_save_path = os.path.join(save_dir, legend_filename)
                
                # Get the exact bounding box of the legend
                bbox = legend.get_window_extent()
                bbox = bbox.transformed(fig_legend.dpi_scale_trans.inverted())
                
                # Print the legend size
                width_inches = bbox.width
                height_inches = bbox.height
                print(f"Legend size: {width_inches:.2f} x {height_inches:.2f} inches")
                
                plt.savefig(legend_save_path, dpi=500, bbox_inches=bbox, pad_inches=0)
                print(f"Legend saved to: {legend_save_path}")
        
        plt.show()


def plot_combined_f1_and_detection_multi(dataset_names, model_names, split='test', sample_types=['cls', 'patch'],
                                        concept_types=None, baseline_types=None, percentthrumodel=100,
                                        show_baselines=True, metric='f1', weighted_avg=True,
                                        figsize=(10, 5, 5), save_fig=True, save_dir='../Figs/Paper_Figs',
                                        label_font_size=12, legend_font=None, axes_font=None, title=None, xlabel=None, ylims=None,
                                        flatten_legend=False, x_positions=None, hide_top_xticks=False):
    """
    Creates multiple combined plots side-by-side for different datasets.
    
    Args:
        dataset_names: List of dataset names
        model_names: List of model names or single model name
        split: Data split to use (default: 'test')
        sample_types: List of sample types (e.g., ['cls', 'patch'])
        concept_types: List of concept types to include
        baseline_types: List of baseline types to include in detection plot
        percentthrumodel: Percentage through model for embeddings
        show_baselines: Whether to show baselines in F1 plot
        metric: Metric to plot in detection scores (default: 'f1')
        weighted_avg: Whether to use weighted average in detection scores
        figsize: Figure size as (total_width, top_plot_height, bottom_plot_height)
        save_fig: Whether to save the figures
        save_dir: Directory to save figures
        label_font_size: Font size for x and y axis labels
        legend_font: Font size for legend. If None, uses label_font_size
        axes_font: Font size for tick labels. If None, uses legend_font
        title: Title for the overall figure (if None, no title)
        xlabel: Custom x-axis label for detection plot
        ylims: List of tuples [(ylim_top, ylim_bottom), ...] for each dataset
        flatten_legend: Whether to arrange legend items horizontally (default: False)
        x_positions: Tuple of (rand_x, prompt_x, cls_x, token_x) for custom x positions
        hide_top_xticks: Whether to hide x-axis ticks and labels on top plot (default: False)
    """
    # Handle single dataset name for backward compatibility
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # Handle single model name
    if isinstance(model_names, str):
        model_names = [model_names]
    
    # Use legend_font if provided, otherwise use label_font_size
    if legend_font is None:
        legend_font = label_font_size
    if axes_font is None:
        axes_font = legend_font
    
    # Convert single ylims to list if needed
    if ylims is not None and len(ylims) == 2 and isinstance(ylims[0], (int, float)):
        ylims = [ylims] * len(dataset_names)
    
    # Apply paper plotting style
    paper_style = get_paper_plotting_style()
    
    # Extract dimensions from figsize
    total_width, top_height, bottom_height = figsize
    total_height = top_height + bottom_height + 0.8  # Extra space for legend on top
    height_ratios = [0.8, top_height, bottom_height]  # Legend, top plots, bottom plots
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(total_width, total_height))
    # Reduce vertical spacing if hiding top x-ticks
    hspace_val = 0.15 if hide_top_xticks else 0.25
    gs = fig.add_gridspec(3, len(dataset_names), height_ratios=height_ratios, hspace=hspace_val, wspace=0.35)
    
    with plt.rc_context(paper_style):
        # Create axes for each dataset
        top_axes = []
        bottom_axes = []
        
        for i in range(len(dataset_names)):
            top_axes.append(fig.add_subplot(gs[1, i]))
            bottom_axes.append(fig.add_subplot(gs[2, i]))
        
        # Process each dataset
        for dataset_idx, dataset_name in enumerate(dataset_names):
            ax1 = top_axes[dataset_idx]
            ax2 = bottom_axes[dataset_idx]
            
            # Get ylims for this dataset
            dataset_ylims = ylims[dataset_idx] if ylims else None
            
            # Process dataset name for title
            display_name = dataset_name
            if display_name.startswith('Broden-'):
                display_name = display_name[7:]  # Remove 'Broden-' prefix
            if display_name == 'Coco':
                display_name = 'COCO'  # Capitalize COCO properly
            
            # ========== TOP PLOT: F1 Scores (Horizontal Bar) ==========
            plt.sca(ax1)
            
            # Get F1 data (same logic as original function)
            results = []
            
            # Loop through each model
            for model_name in model_names:
                # Load ground-truth labels
                if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                    if model_name == 'Llama':
                        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
                    elif model_name == 'Gemma':
                        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
                    elif model_name == 'Qwen':
                        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
                    gt_samples_per_concept = torch.load(gt_path, weights_only=False)
                elif model_name == 'CLIP':
                    gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
                elif model_name == 'Llama':
                    gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
                else:
                    continue
                gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
                
                # Construct concept label mappings
                for sample_type in sample_types:
                    n_clusters = 1000 if sample_type == 'patch' else 50
                    con_labels = {}
                    
                    # Supervised methods
                    if concept_types is None or 'avg' in concept_types:
                        con_labels['avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
                    if concept_types is None or 'linsep' in concept_types:
                        con_labels['linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
                    # Unsupervised methods
                    if concept_types is None or 'kmeans' in concept_types:
                        con_labels['kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
                    if concept_types is None or 'linsep kmeans' in concept_types:
                        con_labels['linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
                    # Add SAE
                    if concept_types is None or 'sae' in concept_types:
                        if model_name == 'CLIP' and sample_type == 'patch':
                                if percentthrumodel == 92:
                                    con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                                else:
                                    print(f"Warning: SAE concepts for CLIP are only available at percentthrumodel=92, not {percentthrumodel}")
                        elif model_name == 'Gemma' and sample_type == 'patch':
                                if percentthrumodel == 81:
                                    con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                                else:
                                    print(f"Warning: SAE concepts for Gemma are only available at percentthrumodel=81, not {percentthrumodel}")
                    
                    for name, con_label in con_labels.items():
                        # Load best percentiles per concept
                        best_percentiles_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
                        
                        if not os.path.exists(best_percentiles_path):
                            continue
                            
                        best_percentiles = torch.load(best_percentiles_path, weights_only=False)
                        
                        # Collect per-concept scores
                        concept_scores = []
                        concept_weights = []
                        
                        # Check if unsupervised
                        is_unsupervised = 'kmeans' in con_label or 'sae' in con_label
                        
                        if is_unsupervised:
                            best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                            if not os.path.exists(best_clusters_path):
                                continue
                            best_clusters = torch.load(best_clusters_path, weights_only=False)
                        
                        for concept in gt_samples_per_concept:
                            if concept not in best_percentiles:
                                continue
                                
                            percentile = best_percentiles[concept]['best_percentile']
                            
                            try:
                                if is_unsupervised:
                                    detection_metrics = pd.read_csv(f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{con_label}.csv')
                                    
                                    if concept in best_clusters:
                                        cluster_id = best_clusters[concept]['best_cluster']
                                        row = detection_metrics[detection_metrics['concept'] == f"('{concept}', '{cluster_id}')"]
                                        if not row.empty:
                                            score = row.iloc[0][metric]
                                            concept_scores.append(score)
                                            concept_weights.append(len(gt_samples_per_concept[concept]))
                                else:
                                    detection_metrics = torch.load(f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.pt', weights_only=False)
                                    
                                    concept_row = detection_metrics[detection_metrics['concept'] == concept]
                                    if not concept_row.empty:
                                        score = concept_row.iloc[0][metric]
                                        concept_scores.append(score)
                                        concept_weights.append(len(gt_samples_per_concept[concept]))
                                        
                            except FileNotFoundError:
                                continue
                        
                        # Calculate final score
                        if concept_scores:
                            total_weight = sum(concept_weights)
                            weighted_score = sum(s * w for s, w in zip(concept_scores, concept_weights)) / total_weight
                            final_score = weighted_score
                        else:
                            final_score = 0
                            
                        results.append({
                            'Model': model_name,
                            'Sample Type': sample_type,
                            'Method': name,
                            'F1': final_score,
                            'Is_Unsupervised': 'kmeans' in name or 'sae' in name
                        })
            
            # Add baselines if requested
            if show_baselines and model_names:
                model_name = model_names[0]
                # Load ground truth for baseline calculation
                if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                    if model_name == 'Llama':
                        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
                    elif model_name == 'Gemma':
                        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
                    elif model_name == 'Qwen':
                        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
                    gt_samples_per_concept = torch.load(gt_path, weights_only=False)
                elif model_name == 'CLIP':
                    gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
                elif model_name == 'Llama':
                    gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
                gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
                
                # Add prompt baseline
                try:
                    prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
                    results.append({
                        'Model': 'Baseline',
                        'Sample Type': 'baseline',
                        'Method': 'Prompt',
                        'F1': prompt_score,
                        'Is_Unsupervised': False
                    })
                except Exception:
                    pass
                
                # Add random baseline
                baseline_path = f'Quant_Results/{dataset_name}/random_{model_name}_cls_baseline.csv'
                if os.path.exists(baseline_path):
                    try:
                        df_baseline = pd.read_csv(baseline_path)
                        df_baseline = df_baseline[df_baseline['concept'].isin(gt_samples_per_concept)]
                        total = sum(len(gt_samples_per_concept[c]) for c in df_baseline['concept'])
                        score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) for _, row in df_baseline.iterrows()) / total
                        results.append({
                            'Model': 'Baseline',
                            'Sample Type': 'baseline',
                            'Method': 'Rand',
                            'F1': score,
                            'Is_Unsupervised': False
                        })
                    except Exception:
                        pass
            
            # Create DataFrame and process for plotting
            df = pd.DataFrame(results)
            
            if not df.empty:
                # Create labels (same logic as original)
                unique_models = df['Model'].unique()
                unique_samples = df['Sample Type'].unique()
                
                labels = []
                for i, row in df.iterrows():
                    if row['Model'] == 'Baseline':
                        labels.append(row['Method'])
                    else:
                        is_pair = False
                        if i > 0:
                            prev_row = df.iloc[i-1]
                            if (prev_row['Sample Type'] == row['Sample Type'] and 
                                ((prev_row['Method'] == 'avg' and row['Method'] == 'kmeans') or
                                 (prev_row['Method'] == 'linsep' and row['Method'] == 'linsep kmeans'))):
                                label = 'SD\nTokens' if row['Sample Type'] == 'patch' else 'CLS\n '
                                labels.append(label)
                                is_pair = True
                        if not is_pair and i < len(df) - 1:
                            next_row = df.iloc[i+1]
                            if (next_row['Sample Type'] == row['Sample Type'] and 
                                ((row['Method'] == 'avg' and next_row['Method'] == 'kmeans') or
                                 (row['Method'] == 'linsep' and next_row['Method'] == 'linsep kmeans'))):
                                label = 'SD\nTokens' if row['Sample Type'] == 'patch' else 'CLS\n '
                                labels.append(label)
                                is_pair = True
                        
                        if not is_pair:
                            parts = []
                            non_baseline_models = [m for m in unique_models if m != 'Baseline']
                            if len(non_baseline_models) > 1:
                                parts.append(row['Model'])
                            
                            non_baseline_samples = [s for s in unique_samples if s != 'baseline']
                            if len(non_baseline_samples) > 1:
                                sample_label = 'SD\nTokens' if row['Sample Type'] == 'patch' else 'CLS'
                                parts.append(sample_label)
                            
                            if row['Method'] == 'avg' or row['Method'] == 'kmeans':
                                method_name = 'centroid'
                            elif row['Method'] == 'linsep' or row['Method'] == 'linsep kmeans':
                                method_name = 'separator'
                            elif row['Method'] == 'sae':
                                method_name = 'SAE'
                            else:
                                method_name = row['Method']
                            
                            parts.append(method_name)
                            labels.append(' - '.join(parts))
                
                df['Label'] = labels
                
                # Custom sorting (same as original)
                def sort_key(row):
                    sample_order = {'patch': 0, 'cls': 1, 'baseline': 2}
                    sample_rank = sample_order.get(row['Sample Type'], 3)
                    
                    if row['Method'] == 'avg' or row['Method'] == 'kmeans':
                        method_group = 0
                    elif row['Method'] == 'linsep' or row['Method'] == 'linsep kmeans':
                        method_group = 1
                    elif row['Method'] == 'sae':
                        method_group = 2
                    else:
                        method_group = 3
                    
                    is_unsupervised = 1 if row['Is_Unsupervised'] else 0
                    
                    return (sample_rank, method_group, is_unsupervised, row.name)
                
                df['sort_key'] = df.apply(sort_key, axis=1)
                df = df.sort_values('sort_key').drop('sort_key', axis=1)
                df = df.reset_index(drop=True)
                
                # Reverse the order
                df = df.iloc[::-1].reset_index(drop=True)
                
                # Create grouped bars
                groups = []
                group_labels = []
                i = 0
                
                while i < len(df):
                    curr_row = df.iloc[i]
                    if i + 1 < len(df):
                        next_row = df.iloc[i + 1]
                        # After reversal, unsupervised comes before supervised
                        is_pair = (curr_row['Sample Type'] == next_row['Sample Type'] and 
                                  curr_row['Model'] != 'Baseline' and next_row['Model'] != 'Baseline' and
                                  ((curr_row['Method'] == 'kmeans' and next_row['Method'] == 'avg') or
                                   (curr_row['Method'] == 'linsep kmeans' and next_row['Method'] == 'linsep')))
                        
                        if is_pair:
                            groups.append([i, i+1])
                            sample_type = df.iloc[i]['Sample Type']
                            label = 'SD\nTokens' if sample_type == 'patch' else 'CLS\n ' if sample_type == 'cls' else df.iloc[i]['Label']
                            group_labels.append(label)
                            i += 2
                            continue
                    groups.append([i])
                    group_labels.append(df.iloc[i]['Label'])
                    i += 1
                
                # Create x positions for groups with custom spacing
                x_pos = []
                
                if x_positions is not None:
                    # Use custom x positions
                    rand_x, prompt_x, cls_x, token_x = x_positions
                    
                    for i, group_indices in enumerate(groups):
                        # Determine what type of group this is
                        if len(group_indices) == 1:
                            row = df.iloc[group_indices[0]]
                            is_random = row['Method'] in ['Random', 'Rand']
                            is_prompt = row['Method'] == 'Prompt'
                        else:
                            # It's a paired group (supervised/unsupervised)
                            row = df.iloc[group_indices[0]]  # Check first item
                            is_random = False
                            is_prompt = False
                        
                        # Assign custom positions
                        if is_random:
                            x_pos.append(rand_x)
                        elif is_prompt:
                            x_pos.append(prompt_x)
                        elif row['Sample Type'] == 'cls':
                            x_pos.append(cls_x)
                        elif row['Sample Type'] == 'patch':
                            x_pos.append(token_x)
                        else:
                            # Default position for any other type
                            x_pos.append(0)
                else:
                    # Use automatic spacing (original logic)
                    current_x = 0
                    
                    for i, group_indices in enumerate(groups):
                        # Determine what type of group this is
                        if len(group_indices) == 1:
                            row = df.iloc[group_indices[0]]
                            is_random = row['Method'] in ['Random', 'Rand']
                            is_prompt = row['Method'] == 'Prompt'
                        else:
                            # It's a paired group (supervised/unsupervised)
                            row = df.iloc[group_indices[0]]  # Check first item
                            is_random = False
                            is_prompt = False
                        
                        # Shift specific bars left
                        if is_random:
                            x_pos.append(current_x - 0.3)  # Move Random left
                        elif is_prompt:
                            x_pos.append(current_x - 0.2)  # Move Prompt left
                        elif row['Sample Type'] == 'cls':
                            x_pos.append(current_x - 0.2)  # Move CLS groups left
                        elif row['Sample Type'] == 'patch':
                            x_pos.append(current_x - 0.2)  # Move Token groups left
                        else:
                            x_pos.append(current_x)
                        
                        # Add spacing after this group
                        if i < len(groups) - 1:  # Not the last group
                            next_group_indices = groups[i + 1]
                            if len(next_group_indices) == 1:
                                next_row = df.iloc[next_group_indices[0]]
                                next_is_prompt = next_row['Method'] == 'Prompt'
                                next_is_cls = next_row['Sample Type'] == 'cls'
                            else:
                                next_row = df.iloc[next_group_indices[0]]
                                next_is_prompt = False
                                next_is_cls = next_row['Sample Type'] == 'cls'
                            
                            # Add extra spacing between specific transitions
                            if is_random and next_is_prompt:
                                current_x += 2.5  # Extra space between Random and Prompt
                            elif is_prompt and next_is_cls:
                                current_x += 2.5  # Extra space between Prompt and CLS groups
                            elif row['Sample Type'] == 'cls' and next_row['Sample Type'] == 'patch':
                                current_x += 2.5  # Extra space between CLS and Token groups
                            else:
                                current_x += 1.2  # Normal spacing
                
                x_pos = np.array(x_pos)
                
                # Color scheme
                color_map = {
                    'patch': '#e65100',  # Darker orange
                    'cls': '#7b1fa2',    # Darker purple
                    'baseline': '#808080',
                    'prompt': '#8B4513'
                }
                
                # Determine colors
                colors = []
                for _, row in df.iterrows():
                    if row['Method'] == 'Prompt':
                        colors.append(color_map['prompt'])
                    elif row['Method'] == 'Random' or row['Method'] == 'Rand':
                        colors.append(color_map['baseline'])
                    elif row['Sample Type'] == 'patch':
                        colors.append(color_map['patch'])
                    elif row['Sample Type'] == 'cls':
                        colors.append(color_map['cls'])
                    else:
                        colors.append('#17becf')
                
                # Create the vertical bars
                if x_positions is not None:
                    # Use consistent bar width when custom positions are provided
                    bar_width = 0.11  # Consistent width for all bars
                    gap = 0  # No gap between paired bars
                else:
                    # Use original variable widths
                    bar_width = 0.6  # Double the width from 0.3 to 0.6
                    gap = 0  # No gap between paired bars
                
                for group_idx, group_indices in enumerate(groups):
                    if len(group_indices) == 2:
                        # Paired bars (supervised + unsupervised)
                        if x_positions is not None:
                            # Use consistent width with custom positions
                            # Unsupervised bar (left)
                            unsup_idx = group_indices[0]
                            unsup_row = df.iloc[unsup_idx]
                            ax1.bar(x_pos[group_idx] - bar_width, unsup_row['F1'], 
                                   width=bar_width, color=colors[unsup_idx], alpha=0.8,
                                   hatch='///', edgecolor='white', linewidth=0.5)
                            
                            # Supervised bar (right)
                            sup_idx = group_indices[1]
                            sup_row = df.iloc[sup_idx]
                            ax1.bar(x_pos[group_idx], sup_row['F1'], 
                                   width=bar_width, color=colors[sup_idx], alpha=0.8)
                        else:
                            # Use original variable widths
                            wide_bar_width = bar_width * 2
                            # Unsupervised bar (left)
                            unsup_idx = group_indices[0]
                            unsup_row = df.iloc[unsup_idx]
                            ax1.bar(x_pos[group_idx] - wide_bar_width, unsup_row['F1'], 
                                   width=wide_bar_width, color=colors[unsup_idx], alpha=0.8,
                                   hatch='///', edgecolor='white', linewidth=0.5)
                            
                            # Supervised bar (right)
                            sup_idx = group_indices[1]
                            sup_row = df.iloc[sup_idx]
                            ax1.bar(x_pos[group_idx], sup_row['F1'], 
                                   width=wide_bar_width, color=colors[sup_idx], alpha=0.8)
                    else:
                        # Single bar
                        df_idx = group_indices[0]
                        row = df.iloc[df_idx]
                        
                        if x_positions is not None:
                            # Use consistent width for all single bars
                            single_bar_width = bar_width
                        else:
                            # Use original variable widths
                            if row['Model'] == 'Baseline':
                                single_bar_width = bar_width * 2
                            else:
                                # Non-baseline single bars get wider
                                single_bar_width = bar_width * 4
                        
                        if row['Is_Unsupervised'] and row['Model'] != 'Baseline':
                            ax1.bar(x_pos[group_idx], row['F1'], width=single_bar_width, 
                                   color=colors[df_idx], alpha=0.8,
                                   hatch='///', edgecolor='white', linewidth=0.5)
                        else:
                            ax1.bar(x_pos[group_idx], row['F1'], width=single_bar_width, 
                                   color=colors[df_idx], alpha=0.8)
                
                # Customize F1 plot
                ax1.set_xlabel('', fontsize=label_font_size)  # No x-axis label
                if dataset_idx == 0:
                    ax1.set_ylabel('F1', fontsize=label_font_size, rotation=0, ha='right')
                if hide_top_xticks:
                    # Hide x-axis ticks and labels
                    ax1.set_xticks([])
                    ax1.set_xticklabels([])
                else:
                    ax1.set_xticks(x_pos)
                    if x_positions is not None:
                        # When using custom positions, adjust label alignment and padding
                        ax1.set_xticklabels(group_labels, rotation=45, ha='center', fontsize=axes_font)
                        ax1.tick_params(axis='x', pad=0)  # Slightly closer to axis
                    else:
                        ax1.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=axes_font)
                        ax1.tick_params(axis='x', pad=2)  # Move x-axis labels much closer
                ax1.tick_params(axis='y', which='major', labelsize=axes_font, pad=2)
                
                # Set x-axis limits explicitly to ensure bars aren't compressed
                if len(x_pos) > 0:
                    if x_positions is not None:
                        # Use fixed x-axis limits when custom positions are provided
                        # Bars outside these limits will be cut off
                        ax1.set_xlim(0, 1)  # Fixed limits - adjust as needed
                    else:
                        # Add padding based on bar width
                        padding = 0.5
                        ax1.set_xlim(x_pos[0] - padding, x_pos[-1] + padding)
                    
                # Force aspect ratio to prevent compression
                ax1.set_aspect('auto')
                
                # Set y-axis limits first
                if dataset_ylims and dataset_ylims[0] is not None:
                    ax1.set_ylim(dataset_ylims[0])
                else:
                    ax1.set_ylim(0, 1)  # F1 score ranges from 0 to 1
                
                # Set y-axis ticks to multiples of 0.2
                ylim_min, ylim_max = ax1.get_ylim()
                yticks = np.arange(0, ylim_max + 0.1, 0.2)
                yticks = yticks[(yticks >= ylim_min) & (yticks <= ylim_max)]
                # Remove 0 from labels but keep the tick
                ylabels = ['' if tick == 0 else f'{tick:.1f}' for tick in yticks]
                ax1.set_yticks(yticks)
                ax1.set_yticklabels(ylabels)
                ax1.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Add dataset name as title
                ax1.set_title(display_name, fontsize=label_font_size, fontweight='bold', pad=2)
            
            # ========== BOTTOM PLOT: Detection Scores ==========
            plt.sca(ax2)
            
            # Use the helper function to plot detection scores
            plot_detection_scores_on_axis(ax2, dataset_name, split, model_names[0], sample_types, 
                                         metric=metric, weighted_avg=weighted_avg, concept_types=concept_types,
                                         baseline_types=baseline_types, percentthrumodel=percentthrumodel,
                                         label_font_size=label_font_size, legend_font=legend_font, xlabel=xlabel,
                                         ylim=dataset_ylims[1] if dataset_ylims and len(dataset_ylims) > 1 else None,
                                         show_legend=False, axes_font=axes_font)
            
            # Add y-axis label only for first plot
            if dataset_idx == 0:
                ax2.set_ylabel(ax2.get_ylabel(), fontsize=label_font_size, rotation=0, ha='right')
            else:
                ax2.set_ylabel('')
        
        # ========== CREATE LEGEND ON TOP ==========
        # Create a separate axis for the legend at the top
        ax_legend = fig.add_subplot(gs[0, :])  # Span all columns
        ax_legend.axis('off')
        
        # Create legend elements (reversed order)
        legend_elements = [
            Patch(facecolor='black', alpha=0.8, hatch='///', 
                  edgecolor='white', linewidth=0.5, label='Unsupervised'),
            Patch(facecolor='black', alpha=0.8, label='Supervised'),
            Patch(facecolor='white', alpha=0),  # Spacer
            Patch(facecolor='#808080', alpha=0.8, label='Random'),
            Patch(facecolor='#8B4513', alpha=0.8, label='Prompt'),
            Patch(facecolor='white', alpha=0),  # Spacer
            Patch(facecolor='#7b1fa2', alpha=0.8, label='CLS'),
            Patch(facecolor='#e65100', alpha=0.8, label='SD Tokens (Ours)')
        ]
        
        # Remove the spacers from handles but keep the space
        handles = [elem for elem in legend_elements if elem.get_facecolor() != (1.0, 1.0, 1.0, 0.0)]
        labels = ['Unsupervised', 'Supervised', '', 'Random', 'Prompt', '', 'CLS', 'SD Tokens (Ours)']
        
        # Create legend with proper spacing - always horizontal
        legend = ax_legend.legend(handles=handles, labels=[l for l in labels if l], 
                                 title='Concept Detection Schemes',
                                 loc='center', frameon=True, fancybox=True, shadow=False,
                                 fontsize=legend_font,
                                 title_fontsize=legend_font + 1,
                                 handlelength=2, handleheight=1.2,
                                 handletextpad=0.4, columnspacing=1,
                                 ncol=len(handles))
        
        # legend.get_title().set_fontweight('bold')  # Removed bold formatting
        
        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=label_font_size + 2, fontweight='bold', y=0.98)
        
        # Save the figure
        if save_fig:
            # Construct filename
            model_str = '_'.join(model_names)
            dataset_str = '_'.join(dataset_names)
            sample_str = '_'.join(sample_types)
            filename = f'{model_str}_{dataset_str}_{sample_str}_f1_detection_combined_multi.pdf'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined multi-dataset plot to: {save_path}")
        
        plt.show()


def plot_combined_f1_and_percentiles(dataset_name, model_names, split='test', sample_types=['cls', 'patch'],
                                   concept_types=None, percentthrumodel=100, show_baselines=False,
                                   figsize=(8, 4, 4), save_fig=True, save_dir='../Figs/Paper_Figs', flatten_legend=False):
    """
    Creates a combined plot with F1 scores (top) and percentile frequencies (bottom),
    and saves a separate legend.
    
    Args:
        dataset_name: Name of the dataset
        model_names: List of model names or single model name
        split: Data split to use (default: 'test')
        sample_types: List of sample types (e.g., ['cls', 'patch'])
        concept_types: List of concept types to include
        percentthrumodel: Percentage through model for embeddings
        show_baselines: Whether to include baselines in F1 plot
        figsize: Figure size as (width, top_plot_height, bottom_plot_height)
        save_fig: Whether to save the figures
        save_dir: Directory to save figures
        flatten_legend: Whether to arrange legend items horizontally (default: False)
    """
    # Handle single model name
    if isinstance(model_names, str):
        model_names = [model_names]
    
    # Apply paper plotting style
    paper_style = get_paper_plotting_style()
    
    # Extract dimensions from figsize
    if len(figsize) == 3:
        width, top_height, bottom_height = figsize
        total_height = top_height + bottom_height
        height_ratios = [top_height, bottom_height]
    else:
        # Fallback for old-style (width, height) tuple
        width, total_height = figsize
        height_ratios = [1, 1]
    
    # Create main figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, total_height), 
                                    gridspec_kw={'height_ratios': height_ratios})
    
    with plt.rc_context(paper_style):
        # ========== TOP PLOT: F1 Scores ==========
        # This is adapted from plot_f1_scores_horizontal_bar
        
        # Get F1 data
        results = []
        metric = 'f1'
        
        # Loop through each model (similar to plot_f1_scores_horizontal_bar)
        for model_name in model_names:
            # Load ground-truth labels
            if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                if model_name == 'Llama':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
                elif model_name == 'Gemma':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
                elif model_name == 'Qwen':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
                gt_samples_per_concept = torch.load(gt_path, weights_only=False)
            elif model_name == 'CLIP':
                gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
            elif model_name == 'Llama':
                gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
            else:
                continue
            gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
            
            # Construct concept label mappings
            for sample_type in sample_types:
                n_clusters = 1000 if sample_type == 'patch' else 50
                con_labels = {}
                
                # Supervised methods
                if concept_types is None or 'avg' in concept_types:
                    con_labels['avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
                if concept_types is None or 'linsep' in concept_types:
                    con_labels['linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
                # Unsupervised methods
                if concept_types is None or 'kmeans' in concept_types:
                    con_labels['kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
                if concept_types is None or 'linsep kmeans' in concept_types:
                    con_labels['linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
                # Add SAE
                if concept_types is None or 'sae' in concept_types:
                    if model_name == 'CLIP' and sample_type == 'patch':
                            if percentthrumodel == 92:
                                con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                            else:
                                print(f"Warning: SAE concepts for CLIP are only available at percentthrumodel=92, not {percentthrumodel}")
                    elif model_name == 'Gemma' and sample_type == 'patch':
                            if percentthrumodel == 81:
                                con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                            else:
                                print(f"Warning: SAE concepts for Gemma are only available at percentthrumodel=81, not {percentthrumodel}")

                for name, con_label in con_labels.items():
                    # Load best percentiles per concept
                    best_percentiles_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
                    
                    if not os.path.exists(best_percentiles_path):
                        continue
                        
                    best_percentiles = torch.load(best_percentiles_path, weights_only=False)
                    
                    # Collect per-concept scores
                    concept_scores = []
                    concept_weights = []
                    
                    # Check if unsupervised
                    is_unsupervised = 'kmeans' in con_label or 'sae' in con_label
                    
                    if is_unsupervised:
                        best_clusters_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                        if not os.path.exists(best_clusters_path):
                            continue
                        best_clusters = torch.load(best_clusters_path, weights_only=False)
                    
                    for concept in gt_samples_per_concept:
                        if concept not in best_percentiles:
                            continue
                            
                        percentile = best_percentiles[concept]['best_percentile']
                        
                        try:
                            if is_unsupervised:
                                detection_metrics = pd.read_csv(f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{con_label}.csv')
                                
                                if concept in best_clusters:
                                    cluster_id = best_clusters[concept]['best_cluster']
                                    row = detection_metrics[detection_metrics['concept'] == f"('{concept}', '{cluster_id}')"]
                                    if not row.empty:
                                        score = row.iloc[0][metric]
                                        concept_scores.append(score)
                                        concept_weights.append(len(gt_samples_per_concept[concept]))
                            else:
                                detection_metrics = torch.load(f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.pt', weights_only=False)
                                
                                concept_row = detection_metrics[detection_metrics['concept'] == concept]
                                if not concept_row.empty:
                                    score = concept_row.iloc[0][metric]
                                    concept_scores.append(score)
                                    concept_weights.append(len(gt_samples_per_concept[concept]))
                                    
                        except FileNotFoundError:
                            continue
                    
                    # Calculate final score
                    if concept_scores:
                        total_weight = sum(concept_weights)
                        weighted_score = sum(s * w for s, w in zip(concept_scores, concept_weights)) / total_weight
                        final_score = weighted_score
                    else:
                        final_score = 0
                        
                    results.append({
                        'Model': model_name,
                        'Sample Type': sample_type,
                        'Method': name,
                        'F1': final_score,
                        'Is_Unsupervised': 'kmeans' in name or 'sae' in name
                    })
        
        # Add baselines if requested
        if show_baselines and model_names:
            model_name = model_names[0]
            # Load ground truth for baseline calculation
            if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
                if model_name == 'Llama':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
                elif model_name == 'Gemma':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
                elif model_name == 'Qwen':
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
                gt_samples_per_concept = torch.load(gt_path, weights_only=False)
            elif model_name == 'CLIP':
                gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
            elif model_name == 'Llama':
                gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
            gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
            
            # Add prompt baseline
            try:
                prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
                results.append({
                    'Model': 'Baseline',
                    'Sample Type': 'baseline',
                    'Method': 'Prompt',
                    'F1': prompt_score,
                    'Is_Unsupervised': False
                })
            except Exception:
                pass
            
            # Add random baseline
            baseline_path = f'Quant_Results/{dataset_name}/random_{model_name}_cls_baseline.csv'
            if os.path.exists(baseline_path):
                try:
                    df_baseline = pd.read_csv(baseline_path)
                    df_baseline = df_baseline[df_baseline['concept'].isin(gt_samples_per_concept)]
                    total = sum(len(gt_samples_per_concept[c]) for c in df_baseline['concept'])
                    score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) for _, row in df_baseline.iterrows()) / total
                    results.append({
                        'Model': 'Baseline',
                        'Sample Type': 'baseline',
                        'Method': 'Rand',
                        'F1': score,
                        'Is_Unsupervised': False
                    })
                except Exception:
                    pass
        
        # Create DataFrame and process for plotting
        df = pd.DataFrame(results)
        
        if not df.empty:
            # Create labels
            unique_models = df['Model'].unique()
            unique_samples = df['Sample Type'].unique()
            
            labels = []
            for i, row in df.iterrows():
                if row['Model'] == 'Baseline':
                    labels.append(row['Method'])
                else:
                    is_pair = False
                    if i > 0:
                        prev_row = df.iloc[i-1]
                        if (prev_row['Sample Type'] == row['Sample Type'] and 
                            ((prev_row['Method'] == 'avg' and row['Method'] == 'kmeans') or
                             (prev_row['Method'] == 'linsep' and row['Method'] == 'linsep kmeans'))):
                            label = 'SD\nTokens' if row['Sample Type'] == 'patch' else 'CLS\n '
                            labels.append(label)
                            is_pair = True
                    if not is_pair and i < len(df) - 1:
                        next_row = df.iloc[i+1]
                        if (next_row['Sample Type'] == row['Sample Type'] and 
                            ((row['Method'] == 'avg' and next_row['Method'] == 'kmeans') or
                             (row['Method'] == 'linsep' and next_row['Method'] == 'linsep kmeans'))):
                            label = 'SD\nTokens' if row['Sample Type'] == 'patch' else 'CLS\n '
                            labels.append(label)
                            is_pair = True
                    
                    if not is_pair:
                        parts = []
                        non_baseline_models = [m for m in unique_models if m != 'Baseline']
                        if len(non_baseline_models) > 1:
                            parts.append(row['Model'])
                        
                        non_baseline_samples = [s for s in unique_samples if s != 'baseline']
                        if len(non_baseline_samples) > 1:
                            sample_label = 'SD\nTokens' if row['Sample Type'] == 'patch' else 'CLS'
                            parts.append(sample_label)
                        
                        if row['Method'] == 'avg' or row['Method'] == 'kmeans':
                            method_name = 'centroid'
                        elif row['Method'] == 'linsep' or row['Method'] == 'linsep kmeans':
                            method_name = 'separator'
                        elif row['Method'] == 'sae':
                            method_name = 'SAE'
                        else:
                            method_name = row['Method']
                        
                        parts.append(method_name)
                        labels.append(' - '.join(parts))
            
            df['Label'] = labels
            
            # Custom sorting
            def sort_key(row):
                sample_order = {'patch': 0, 'cls': 1, 'baseline': 2}
                sample_rank = sample_order.get(row['Sample Type'], 3)
                
                if row['Method'] == 'avg' or row['Method'] == 'kmeans':
                    method_group = 0
                elif row['Method'] == 'linsep' or row['Method'] == 'linsep kmeans':
                    method_group = 1
                elif row['Method'] == 'sae':
                    method_group = 2
                else:
                    method_group = 3
                
                is_unsupervised = 1 if row['Is_Unsupervised'] else 0
                
                return (sample_rank, method_group, is_unsupervised, row.name)
            
            df['sort_key'] = df.apply(sort_key, axis=1)
            df = df.sort_values('sort_key').drop('sort_key', axis=1)
            df = df.reset_index(drop=True)
            
            # Reverse the order
            df = df.iloc[::-1].reset_index(drop=True)
            
            # Create grouped bars
            groups = []
            y_labels = []
            i = 0
            
            while i < len(df):
                curr_row = df.iloc[i]
                if i + 1 < len(df):
                    next_row = df.iloc[i + 1]
                    # After reversal, unsupervised comes before supervised
                    is_pair = (curr_row['Sample Type'] == next_row['Sample Type'] and 
                              curr_row['Model'] != 'Baseline' and next_row['Model'] != 'Baseline' and
                              ((curr_row['Method'] == 'kmeans' and next_row['Method'] == 'avg') or
                               (curr_row['Method'] == 'linsep kmeans' and next_row['Method'] == 'linsep')))
                    
                    if is_pair:
                        groups.append([i, i+1])
                        sample_type = df.iloc[i]['Sample Type']
                        label = 'SD\nTokens' if sample_type == 'patch' else 'CLS\n ' if sample_type == 'cls' else df.iloc[i]['Label']
                        y_labels.append(label)
                        i += 2
                        continue
                groups.append([i])
                y_labels.append(df.iloc[i]['Label'])
                i += 1
            
            # Create y positions for groups
            y_pos = np.arange(len(groups)) * 1.5
            
            # Color scheme
            color_map = {
                'patch': '#e65100',  # Darker orange
                'cls': '#7b1fa2',    # Darker purple
                'baseline': '#808080',
                'prompt': '#8B4513'
            }
            
            # Determine colors
            colors = []
            for _, row in df.iterrows():
                if row['Method'] == 'Prompt':
                    colors.append(color_map['prompt'])
                elif row['Method'] == 'Random' or row['Method'] == 'Rand':
                    colors.append(color_map['baseline'])
                elif row['Sample Type'] == 'patch':
                    colors.append(color_map['patch'])
                elif row['Sample Type'] == 'cls':
                    colors.append(color_map['cls'])
                else:
                    colors.append('#17becf')
            
            # Create the vertical bars
            bar_width = 0.3
            gap = 0  # No gap between paired bars
            
            # Create x positions for groups
            x_pos = np.arange(len(groups)) * 1.5
            
            for group_idx, group_indices in enumerate(groups):
                if len(group_indices) == 2:
                    # Unsupervised bar (left)
                    unsup_idx = group_indices[0]
                    unsup_row = df.iloc[unsup_idx]
                    ax1.bar(x_pos[group_idx] - bar_width/2 - gap/2, unsup_row['F1'], 
                           width=bar_width, color=colors[unsup_idx], alpha=0.8,
                           hatch='///', edgecolor='white', linewidth=0.5)
                    
                    # Supervised bar (right)
                    sup_idx = group_indices[1]
                    sup_row = df.iloc[sup_idx]
                    ax1.bar(x_pos[group_idx] + bar_width/2 + gap/2, sup_row['F1'], 
                           width=bar_width, color=colors[sup_idx], alpha=0.8)
                else:
                    # Single bar
                    df_idx = group_indices[0]
                    row = df.iloc[df_idx]
                    if row['Is_Unsupervised'] and row['Model'] != 'Baseline':
                        ax1.bar(x_pos[group_idx], row['F1'], width=bar_width*2, 
                               color=colors[df_idx], alpha=0.8,
                               hatch='///', edgecolor='white', linewidth=0.5)
                    else:
                        ax1.bar(x_pos[group_idx], row['F1'], width=bar_width*2, 
                               color=colors[df_idx], alpha=0.8)
            
            # Customize F1 plot
            ax1.set_xlabel('', fontsize=paper_style['legend.fontsize'])  # No x-axis label
            ax1.set_ylabel('F1', fontsize=paper_style['legend.fontsize'])
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(y_labels, rotation=0, ha='center')
            # Set y-axis limits
            if ylims and ylims[0] is not None:
                ax1.set_ylim(ylims[0])
            else:
                ax1.set_ylim(0, 1)  # F1 score ranges from 0 to 1
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # ========== BOTTOM PLOT: Percentile Frequencies ==========
        # Get the first model for percentile plot
        model_name = model_names[0]
        
        # Load ground truth for filtering
        if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
            if model_name == 'Llama':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
            elif model_name == 'Gemma':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
            elif model_name == 'Qwen':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
            gt_samples_per_concept = torch.load(gt_path, weights_only=False)
        elif model_name == 'CLIP':
            gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
        elif model_name == 'Llama':
            gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
        
        gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
        valid_concepts = set(gt_samples_per_concept.keys())
        
        # Collect percentile frequencies
        all_percentile_counts = {}
        
        for sample_type in sample_types:
            n_clusters = 1000 if sample_type == 'patch' else 50
            con_labels = {}
            
            if concept_types is None or 'avg' in concept_types:
                con_labels['avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'linsep' in concept_types:
                con_labels['linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'kmeans' in concept_types:
                con_labels['kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'linsep kmeans' in concept_types:
                con_labels['linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthrumodel}'
            if concept_types is None or 'sae' in concept_types:
                if model_name == 'CLIP' and sample_type == 'patch':
                        if percentthrumodel == 92:
                            con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                        else:
                            print(f"Warning: SAE concepts for CLIP are only available at percentthrumodel=92, not {percentthrumodel}")
                elif model_name == 'Gemma' and sample_type == 'patch':
                        if percentthrumodel == 81:
                            con_labels['sae'] = f'{model_name}_sae_{sample_type}_dense'
                        else:
                            print(f"Warning: SAE concepts for Gemma are only available at percentthrumodel=81, not {percentthrumodel}")
            
            for name, con_label in con_labels.items():
                best_percentiles_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
                
                if not os.path.exists(best_percentiles_path):
                    continue
                    
                best_percentiles = torch.load(best_percentiles_path, weights_only=False)
                
                percentile_counts = defaultdict(int)
                
                for concept in valid_concepts:
                    if concept in best_percentiles:
                        percentile = best_percentiles[concept]['best_percentile']
                        percentile_counts[percentile] += 1
                
                method_key = f"{sample_type} - {name}"
                all_percentile_counts[method_key] = dict(percentile_counts)
        
        # Get all unique percentiles
        all_percentiles = set()
        for counts in all_percentile_counts.values():
            all_percentiles.update(counts.keys())
        percentiles_sorted = sorted(all_percentiles)
        
        # Use actual percentile values for x-axis (continuous scale)
        x_values = [p * 100 for p in percentiles_sorted]  # Convert to percentage
        
        # Plot lines for each method
        for method_key, counts in all_percentile_counts.items():
            # Get y values for this method
            y_values = [counts.get(p, 0) for p in percentiles_sorted]
            
            # Determine color and line style
            color = color_map['patch'] if 'patch' in method_key else color_map['cls']
            is_unsupervised = 'kmeans' in method_key or 'sae' in method_key
            linestyle = ':' if is_unsupervised else '-'
            
            # Extract method name for label
            parts = method_key.split(' - ')
            if len(parts) == 2:
                sample_type, method = parts
                label = f"{sample_type.capitalize()} - {method}"
            else:
                label = method_key
            
            # Plot line with continuous x values
            ax2.plot(x_values, y_values, color=color, linestyle=linestyle, 
                    linewidth=2, marker='o', markersize=3, 
                    label=label, alpha=0.8)
        
        # Customize percentile plot
        ax2.set_xlabel('Optimal Recall Percentile', fontsize=paper_style['legend.fontsize'])
        ax2.set_ylabel('# Concepts', fontsize=paper_style['legend.fontsize'])
        
        # Set x-ticks to show every 10% but only label 10, 30, 50, 70, 90
        x_ticks = [i for i in range(0, 101, 10)]
        ax2.set_xticks(x_ticks)
        x_labels = []
        for i in x_ticks:
            if i in [10, 30, 50, 70, 90]:
                x_labels.append(f"{i}%")
            else:
                x_labels.append("")
        ax2.tick_params(axis='x', pad=2)  # Reduce padding between labels and axis
        ax2.set_xticklabels(x_labels, rotation=45, ha='right', rotation_mode='anchor')
        
        # Set x-axis limits
        ax2.set_xlim(0, 100)
        
        # Force y-axis to use integer values only with at most 4 ticks
        ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))
        
        # Set y-axis upper limit to max count + 0.5 or + 2
        max_count = 0
        for counts in all_percentile_counts.values():
            max_count = max(max_count, max(counts.values()) if counts else 0)
        
        if max_count <= 1:
            ax2.set_ylim(0, max_count + 0.5)
        else:
            ax2.set_ylim(0, max_count + 2)
        
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save main figure
        if save_fig:
            models_str = '_'.join(model_names)
            samples_str = '_'.join(sample_types)
            concepts_str = '_'.join(concept_types) if concept_types else 'all'
            filename = f"{dataset_name}_{models_str}_{samples_str}_{concepts_str}_combined.pdf"
            save_path = os.path.join(save_dir, filename)
            
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
            print(f"Main figure saved to: {save_path}")
        
        plt.show()
        
        # ========== CREATE SEPARATE LEGEND ==========
        with plt.rc_context(paper_style):
            fig_legend = plt.figure()
            ax_legend = fig_legend.add_subplot(111)
            ax_legend.axis('off')
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#ff7f0e', alpha=0.8, label='Patch-Concept'),
                Patch(facecolor='#9467bd', alpha=0.8, label='CLS-Concept'),
                Patch(facecolor='white', alpha=0),  # Spacer
                Patch(facecolor='black', alpha=0.8, label='Supervised'),
                Patch(facecolor='black', alpha=0.8, hatch='///', 
                      edgecolor='white', linewidth=0.5, label='Unsupervised')
            ]
            
            # Remove the spacer from handles but keep the space
            handles = [elem for elem in legend_elements if elem.get_facecolor() != (1.0, 1.0, 1.0, 0.0)]
            labels = ['Patch-Concept', 'CLS-Concept', '', 'Supervised', 'Unsupervised']
            
            # Create legend with proper spacing
            if flatten_legend:
                # Horizontal layout
                legend = ax_legend.legend(handles=handles, labels=[l for l in labels if l], 
                                         title='Detection Schemes',
                                         loc='center', frameon=True, fancybox=True, shadow=False,
                                         fontsize=paper_style['legend.fontsize'],
                                         title_fontsize=paper_style['legend.fontsize'],
                                         handlelength=2, handleheight=1.5,
                                         ncol=len(handles))  # All items in one row
            else:
                # Vertical layout (default)
                legend = ax_legend.legend(handles=handles, labels=[l for l in labels if l], 
                                         title='Detection Schemes',
                                         loc='center', frameon=True, fancybox=True, shadow=False,
                                         fontsize=paper_style['legend.fontsize'],
                                         title_fontsize=paper_style['legend.fontsize'],
                                         handlelength=2, handleheight=1.5)
            
            # Save legend
            if save_fig:
                legend_filename = f"{dataset_name}_{models_str}_{samples_str}_{concepts_str}_legend.pdf"
                legend_save_path = os.path.join(save_dir, legend_filename)
                
                # Get the exact bounding box of the legend
                bbox = legend.get_window_extent()
                bbox = bbox.transformed(fig_legend.dpi_scale_trans.inverted())
                
                # Print the legend size
                width_inches = bbox.width
                height_inches = bbox.height
                print(f"Legend size: {width_inches:.2f} x {height_inches:.2f} inches")
                
                plt.savefig(legend_save_path, dpi=500, bbox_inches=bbox, pad_inches=0)
                print(f"Legend saved to: {legend_save_path}")
        
        plt.show()
        
    return df, all_percentile_counts


def plot_detection_scores_multi(dataset_names, model_names, split='test', sample_types=['cls', 'patch'],
                               concept_types=None, baseline_types=None, percentthrumodel=100,
                               metric='f1', weighted_avg=True,
                               figsize=None, save_fig=True, save_dir='../Figs/Paper_Figs',
                               label_font_size=12, legend_font=None, axes_font=None, 
                               title=None, xlabel=None, ylims=None,
                               flatten_legend=False, x_positions=None):
    """
    Creates detection score plots for multiple datasets side-by-side.
    This is essentially the bottom plots from plot_combined_f1_and_detection_multi.
    
    Args:
        dataset_names: List of dataset names
        model_names: List of model names or single model name
        split: Data split to use (default: 'test')
        sample_types: List of sample types (e.g., ['cls', 'patch'])
        concept_types: List of concept types to include
        baseline_types: List of baseline types to include in detection plot
        percentthrumodel: Percentage through model for embeddings
        metric: Metric to plot in detection scores (default: 'f1')
        weighted_avg: Whether to use weighted average in detection scores
        figsize: Figure size as (width, height). If None, auto-calculated
        save_fig: Whether to save the figures
        save_dir: Directory to save figures
        label_font_size: Font size for x and y axis labels
        legend_font: Font size for legend. If None, uses label_font_size
        axes_font: Font size for tick labels. If None, uses legend_font
        title: Title for the overall figure (if None, no title)
        xlabel: Custom x-axis label for detection plot
        ylims: List of tuples [(y_min, y_max), ...] for each dataset
        flatten_legend: Whether to arrange legend items horizontally (default: False)
        x_positions: Tuple of (rand_x, prompt_x, cls_x, token_x) for custom x positions
    """
    # Handle single dataset name for backward compatibility
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # Handle single model name
    if isinstance(model_names, str):
        model_names = [model_names]
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (7 * len(dataset_names), 5)
    elif isinstance(figsize, tuple) and len(figsize) == 2:
        # If user provides (width, height), scale width by number of datasets
        figsize = (figsize[0] * len(dataset_names), figsize[1])
    
    # Use legend_font if provided, otherwise use label_font_size
    if legend_font is None:
        legend_font = label_font_size
    if axes_font is None:
        axes_font = legend_font
    
    # Convert single ylims to list if needed
    if ylims is not None and len(ylims) == 2 and isinstance(ylims[0], (int, float)):
        ylims = [ylims] * len(dataset_names)
    
    # Apply paper plotting style
    paper_style = get_paper_plotting_style()
    
    # Create figure with subplots (don't share y-axis to show ticks on all)
    fig, axes = plt.subplots(1, len(dataset_names), figsize=figsize, sharey=False)
    
    # Ensure axes is always a list
    if len(dataset_names) == 1:
        axes = [axes]
    
    # Apply the paper style to current figure
    with plt.rc_context(paper_style):
        # Process each dataset
        for idx, (dataset_name, ax) in enumerate(zip(dataset_names, axes)):
            # Format dataset name for title
            if dataset_name.startswith('Broden-'):
                title_name = dataset_name[7:]  # Remove "Broden-" prefix
            elif dataset_name.lower() == 'coco':
                title_name = 'COCO'
            else:
                title_name = dataset_name
            
            # Plot detection scores
            # Note: plot_detection_scores_on_axis expects single model_name, not list
            model_name = model_names[0] if isinstance(model_names, list) else model_names
            plot_detection_scores_on_axis(ax, dataset_name, split, model_name, sample_types,
                                        metric=metric, weighted_avg=weighted_avg,
                                        concept_types=concept_types, baseline_types=baseline_types,
                                        percentthrumodel=percentthrumodel,
                                        show_legend=False, 
                                        ylim=ylims[idx] if ylims else None,
                                        label_font_size=label_font_size,
                                        legend_font=legend_font,
                                        axes_font=axes_font,
                                        xlabel=xlabel)
            
            # Add dataset title
            ax.set_title(title_name, fontsize=label_font_size + 2, pad=10)
            
            # Show y-label on all plots with more padding
            ax.set_ylabel(metric.upper(), fontsize=label_font_size, labelpad=10)
        
        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=label_font_size + 4, y=0.98)
        
        # Create legend below all plots
        # Get the color and style information
        colors_and_styles = {
            'Prompt': {'color': '#8B4513', 'marker': 'o', 'linestyle': '-', 'is_baseline': True},
            'Random': {'color': '#808080', 'marker': 's', 'linestyle': '--', 'is_baseline': True},
            'CLS': {'color': '#9467bd', 'marker': '^', 'linestyle': '-.', 'is_baseline': False},
            'SD Tokens': {'color': '#ff7f0e', 'marker': 'D', 'linestyle': '-', 'is_baseline': False}
        }
        
        # Create custom legend elements in the order: Random, Prompt, CLS, SD Token (Ours)
        legend_elements = []
        
        # Add in specified order
        if baseline_types is None or 'random' in baseline_types:
            legend_elements.append(
                Line2D([0], [0], color=colors_and_styles['Random']['color'], lw=2,
                       marker=colors_and_styles['Random']['marker'], markersize=8,
                       linestyle=colors_and_styles['Random']['linestyle'],
                       label='Random')
            )
        
        if baseline_types is None or 'prompt' in baseline_types:
            legend_elements.append(
                Line2D([0], [0], color=colors_and_styles['Prompt']['color'], lw=2,
                       marker=colors_and_styles['Prompt']['marker'], markersize=8,
                       label='Prompt')
            )
        
        # CLS and SD Tokens
        legend_elements.extend([
            Line2D([0], [0], color=colors_and_styles['CLS']['color'], lw=2,
                   marker=colors_and_styles['CLS']['marker'], markersize=8,
                   label='CLS'),
            Line2D([0], [0], color=colors_and_styles['SD Tokens']['color'], lw=2, 
                   marker=colors_and_styles['SD Tokens']['marker'], markersize=8,
                   label='SD Token (Ours)'),
        ])
        
        # Create legend - always flat (horizontal)
        ncol = len(legend_elements)
        bbox_anchor = (0.5, -0.15)
        loc = 'upper center'
        
        fig.legend(handles=legend_elements,
                  title='Concept Detection Methods',
                  loc=loc,
                  bbox_to_anchor=bbox_anchor,
                  ncol=ncol,
                  fontsize=legend_font,
                  title_fontsize=legend_font,
                  frameon=True,
                  fancybox=True)
        
        # Adjust layout
        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=0.93)
        plt.subplots_adjust(bottom=0.15)  # Make room for legend
        
        # Save if requested
        if save_fig:
            # Generate filename using first dataset
            dataset_str = dataset_names[0]
            models_str = '_'.join(model_names)
            samples_str = '_'.join(sample_types)
            filename = f"{dataset_str}_{models_str}_{samples_str}_detection_scores_multi.pdf"
            save_path = os.path.join(save_dir, filename)
            
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()


def plot_detection_scores_overlay(dataset_names, model_name, split='test', sample_type='patch',
                                 concept_types=None, percentthrumodel=100,
                                 metric='f1', weighted_avg=True,
                                 figsize=(8, 6), save_fig=True, save_dir='../Figs/Paper_Figs',
                                 label_font_size=12, legend_font=None, axes_font=None, 
                                 title=None, xlabel=None, ylim=None,
                                 colors=None):
    """
    Creates a single plot with detection scores for multiple datasets overlaid.
    Only shows SD Token (patch) results, no baselines or CLS.
    
    Args:
        dataset_names: List of dataset names or single dataset name
        model_name: Model name (single model only)
        split: Data split to use (default: 'test')
        sample_type: Sample type - 'patch' for SD Token (default: 'patch')
        concept_types: List of concept types to include
        percentthrumodel: Percentage through model for embeddings
        metric: Metric to plot in detection scores (default: 'f1')
        weighted_avg: Whether to use weighted average in detection scores
        figsize: Figure size as (width, height)
        save_fig: Whether to save the figure
        save_dir: Directory to save figures
        label_font_size: Font size for x and y axis labels
        legend_font: Font size for legend. If None, uses label_font_size
        axes_font: Font size for tick labels. If None, uses legend_font
        title: Title for the plot (if None, no title)
        xlabel: Custom x-axis label
        ylim: Tuple of (y_min, y_max) for y-axis limits
        colors: List of colors for each dataset. If None, uses default color cycle
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    import os
    
    # Handle single dataset name
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # Use legend_font if provided, otherwise use label_font_size
    if legend_font is None:
        legend_font = label_font_size
    if axes_font is None:
        axes_font = legend_font
    
    # Apply paper plotting style
    paper_style = get_paper_plotting_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors if not provided
    if colors is None:
        # Use a nice color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_names)))
    
    # Apply the paper style to current figure
    with plt.rc_context(paper_style):
        # Store lines for legend
        lines = []
        labels = []
        
        # Process each dataset
        for idx, dataset_name in enumerate(dataset_names):
            # Format dataset name for legend
            if dataset_name.startswith('Broden-'):
                legend_name = dataset_name[7:]  # Remove "Broden-" prefix
            elif dataset_name.lower() == 'coco':
                legend_name = 'COCO'
            else:
                legend_name = dataset_name
            
            # Percentiles to plot
            percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95]
            
            # Compute detection scores for this dataset
            scores_data = compute_detection_scores(
                dataset_name, split, model_name, [sample_type], 
                metric=metric, weighted_avg=weighted_avg,
                concept_types=concept_types, percentthrumodel=percentthrumodel
            )
            
            if scores_data is not None and 'concept_data' in scores_data:
                # Get the percentiles from the returned data
                percentiles = scores_data['percentiles']
                
                # Find the appropriate key for our concept type and sample type
                # Keys are like 'labeled patch linsep', 'labeled patch avg', etc.
                for key, scores in scores_data['concept_data'].items():
                    # Check if this key matches our sample_type
                    if sample_type in key:
                        # Plot the line
                        line, = ax.plot(percentiles, scores, 
                                       color=colors[idx], 
                                       linewidth=2.5, 
                                       marker='o', 
                                       markersize=6,
                                       label=legend_name)
                        lines.append(line)
                        labels.append(legend_name)
                        break  # Only plot the first matching method
        
        # Customize the plot
        ax.set_xlabel(xlabel or 'Percentile', fontsize=label_font_size)
        ax.set_ylabel(metric.upper(), fontsize=label_font_size, labelpad=10)
        
        # Set tick label sizes
        ax.tick_params(axis='both', labelsize=axes_font)
        
        # Set limits
        ax.set_xlim(0, 1)
        if ylim:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(0, 1)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add title if provided
        if title:
            ax.set_title(title, fontsize=label_font_size + 2, pad=10)
        
        # Add legend in top right corner with two columns
        ax.legend(lines, labels, 
                 loc='upper right',
                 ncol=2,
                 fontsize=legend_font,
                 frameon=True,
                 fancybox=True,
                 shadow=False)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_fig:
            # Generate filename using first dataset
            dataset_str = dataset_names[0]
            filename = f"{dataset_str}_{model_name}_{sample_type}_overlay_detection_scores.pdf"
            save_path = os.path.join(save_dir, filename)
            
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()


def plot_percentile_histogram_all_ptm(dataset_names, model_name, concept_types, sample_type="patch",
                                     split="test", figsize=None, save_fig=True, 
                                     save_dir="../Figs/Histograms", colors=None, 
                                     label_fontsize=12, input_size=None, legend_size=12, font_size=10,
                                     percentthrumodels=None, title_font_size=14, show_legend=True):
    """
    Plots percentile histograms for specified percentthrumodel values of a model.
    Each percentthrumodel gets its own subplot arranged horizontally.
    Axes are flipped compared to plot_percentile_histogram (vertical bars instead of horizontal).
    
    Args:
        dataset_names: Single dataset name (string) or list of dataset names
        model_name: Model name (e.g., "CLIP", "Llama-Vision", "Llama-Text", "Gemma")
        concept_types: Single concept type (string) or list of concept types ("avg", "linsep", "kmeans", "linsep kmeans", or "sae")
        sample_type: Sample type ("cls" or "patch") (default: "patch")
        split: Data split to use (default: "test")
        figsize: Figure size as tuple (width, height) (default: calculated based on PTMs)
        save_fig: Whether to save the figure (default: True)
        save_dir: Directory to save figures (default: "../Figs/Histograms")
        colors: List of colors for each dataset (default: None, uses matplotlib color cycle)
        label_fontsize: Font size for axis labels (default: 12)
        input_size: Input size for determining default percentthrumodels (default: None, auto-determined)
        legend_size: Font size for the legend (default: 12)
        font_size: Font size for tick labels (percentages and numbers) (default: 10)
        percentthrumodels: List of percentthrumodel values to plot (default: None, uses model defaults)
        title_font_size: Font size for the main title "% Through {model_name} Model" (default: 14)
    """
    # Ensure dataset_names is a list
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # Ensure concept_types is a list
    if isinstance(concept_types, str):
        concept_types = [concept_types]
    
    # Determine input size if not provided
    if input_size is None:
        # Check if any dataset is a text dataset
        text_datasets = ["Stanford-Tree-Bank", "iSarcasm", "Sarcasm", "GoEmotions"]
        is_text = any(d in text_datasets or "Sarcasm" in d or "Emotion" in d for d in dataset_names)
        
        if model_name == "CLIP":
            input_size = (224, 224)
        elif model_name == "Llama-Vision":
            input_size = (560, 560)
        elif model_name == "Llama-Text":
            input_size = ("text", "text")
        elif model_name in ["Gemma", "Qwen"]:
            input_size = ("text", "text") if is_text else (560, 560)
    
    # Get percentthrumodels to use
    if percentthrumodels is None:
        # Use default percentthrumodels for this model
        percentthrumodels = get_model_default_percentthrumodels(model_name, input_size)
    
    # Check if percentthrumodels is empty
    if len(percentthrumodels) == 0:
        print("No percentthrumodels specified. Nothing to plot.")
        return None, None
    
    n_ptms = len(percentthrumodels)
    
    # Set figure size if not provided
    if figsize is None:
        figsize = (4 * n_ptms + 2, 6)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_ptms, figsize=figsize, sharey=True)
    if n_ptms == 1:
        axes = [axes]  # Make it iterable
    
    
    # Set up colors
    if colors is None:
        colors = plt.cm.tab10.colors[:len(dataset_names)]
    
    # Fixed bins (same as original function)
    fixed_bins = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_bins = len(fixed_bins) - 1
    bin_labels = [f"{int(val*100)}%" for val in fixed_bins[:-1]]
    
    # Store max count across all subplots for consistent x-axis
    max_count = 0
    
    # Process each percentthrumodel
    for ptm_idx, percentthrumodel in enumerate(percentthrumodels):
        ax = axes[ptm_idx]
        
        # Collect bin counts for all datasets at this PTM
        all_bin_counts = {}
        
        for dataset_idx, dataset_name in enumerate(dataset_names):
            # Collect percentiles from all concept types for this dataset
            all_percentiles_for_dataset = []
            
            for concept_type in concept_types:
                # For SAE concept type, automatically determine model and percentthrumodel
                if concept_type == "sae":
                    is_text_dataset = (dataset_name == "Stanford-Tree-Bank" or 
                                     "Sarcasm" in dataset_name or 
                                     "Emotion" in dataset_name)
                    if is_text_dataset:
                        actual_model_name = "Gemma"
                        actual_percentthrumodel = 81
                    else:
                        actual_model_name = "CLIP"
                        actual_percentthrumodel = 81
                else:
                    # For regular concepts, map Llama-Vision/Llama-Text back to Llama for file paths
                    if model_name in ["Llama-Vision", "Llama-Text"]:
                        actual_model_name = "Llama"
                    else:
                        actual_model_name = model_name
                    actual_percentthrumodel = percentthrumodel
            
                
                # Build concept label
                n_clusters = 1000 if sample_type == "patch" else 50
                
                if concept_type == "avg":
                    con_label = f"{actual_model_name}_avg_{sample_type}_embeddings_percentthrumodel_{actual_percentthrumodel}"
                elif concept_type == "linsep":
                    con_label = f"{actual_model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{actual_percentthrumodel}"
                elif concept_type == "kmeans":
                    con_label = f"{actual_model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{actual_percentthrumodel}"
                elif concept_type == "linsep kmeans":
                    con_label = f"{actual_model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{actual_percentthrumodel}"
                elif concept_type == "sae":
                    con_label = f"{actual_model_name}_sae_{sample_type}_dense"
                else:
                    raise ValueError(f"Unknown concept type: {concept_type}")
                
                # Load best percentiles
                best_percentiles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                           f"Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt")
                
                if not os.path.exists(best_percentiles_path):
                    print(f"Warning: Best percentiles file not found for {dataset_name} {concept_type} at PTM {percentthrumodel}")
                    continue
                
                best_percentiles = torch.load(best_percentiles_path, weights_only=False)
                
                # Collect percentiles for all concepts in the file
                for concept, data in best_percentiles.items():
                    if "best_percentile" in data:
                        percentile = data["best_percentile"]
                        all_percentiles_for_dataset.append(percentile)
            
            # After collecting from all concept types, count how many percentiles fall into each bin
            if all_percentiles_for_dataset:
                bin_counts = []
                for i in range(len(fixed_bins)-1):
                    count = sum(1 for p in all_percentiles_for_dataset if fixed_bins[i] <= p < fixed_bins[i+1])
                    bin_counts.append(count)
                
                all_bin_counts[dataset_name] = bin_counts
        
        # Calculate max stacked height for y-axis limits
        if all_bin_counts:
            # Sum counts across all datasets for each bin
            stacked_totals = np.zeros(n_bins)
            for dataset_name, bin_counts in all_bin_counts.items():
                stacked_totals += np.array(bin_counts)
            max_count = max(max_count, max(stacked_totals) if len(stacked_totals) > 0 else 0)
        
        # Create vertical stacked bar chart with equally spaced positions
        # Use equally spaced x positions for clarity that these are bins
        x_positions = np.arange(n_bins)
        
        # Use a bar width that leaves gaps between bars
        bar_width = 0.8  # Width that leaves visible gaps
        
        # Initialize bottom positions for stacking
        bottom = np.zeros(n_bins)
        
        for idx, (dataset_name, bin_counts) in enumerate(all_bin_counts.items()):
            # Clean up dataset name for display
            display_name = dataset_name
            if dataset_name == "Coco" or dataset_name == "COCO":
                display_name = "COCO"
            elif dataset_name.startswith("Broden-"):
                display_name = dataset_name.replace("Broden-", "")
            
            # Plot vertical bars with visible separation
            bars = ax.bar(x_positions, bin_counts, bar_width,
                          bottom=bottom,
                          label=display_name if ptm_idx == 0 else "",  # Only label on first subplot
                          color=colors[idx],
                          edgecolor="black", linewidth=0.5, alpha=0.9)
            
            # Update bottom for next dataset
            bottom += np.array(bin_counts)
        
        # Remove all x-axis labels
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        # Add only the tick positions
        ax.set_xticks(x_positions)
        
        # Now add the labels with rotation and font size - only every other one
        labels_to_show = []
        for i, label in enumerate(bin_labels):
            if i % 2 == 0:  # Show every other label (0, 2, 4, 6, 8, 10)
                labels_to_show.append(label)
            else:
                labels_to_show.append('')  # Empty string for labels we don't want to show
        ax.set_xticklabels(labels_to_show, rotation=45, ha='right', fontsize=font_size)
        
        ax.tick_params(axis='x', labelsize=font_size, pad=1)  # Reduce padding between ticks and labels
        ax.tick_params(axis='y', labelsize=font_size)
        
        # Customize subplot
        if ptm_idx == 0:
            ax.set_ylabel("# Concepts w/\nPeak Detection", fontsize=label_fontsize)
        
        # Add x-axis label
        ax.set_xlabel("SuperTok Percentile (N)", fontsize=label_fontsize)
        
        # Add text above subplot to show which percentthrumodel this is
        ax.text(0.5, 1.02, f"{percentthrumodel}%", transform=ax.transAxes,
                ha='center', va='bottom', fontsize=label_fontsize)
        
        # Add grid for y-axis
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Set consistent y-axis limits across all subplots with extra padding
    for ax in axes:
        ax.set_ylim(0, max_count * 1.1)  # 10% padding to ensure nothing is cut off
    
    plt.tight_layout()
    
    # Add centered label above all subplots
    fig.text(0.5, 0.98, f"% Through {model_name} Model", ha='center', va='bottom', fontsize=title_font_size)
    
    # Add single legend at bottom
    if show_legend and len(dataset_names) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels,
                  loc="lower center",
                  bbox_to_anchor=(0.5, -0.5),
                  ncol=len(dataset_names),
                  fontsize=legend_size,
                  frameon=True,
                  columnspacing=1.5)
    
    # Adjust layout to make room for the labels
    if show_legend and len(dataset_names) > 1:
        plt.subplots_adjust(bottom=0.20, top=0.85)
    else:
        plt.subplots_adjust(bottom=0.05, top=0.85)
    
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        datasets_str = "_".join(dataset_names)
        concept_types_str = "_".join(concept_types)
        filename = f"percentile_histogram_all_ptm_{datasets_str}_{model_name}_{concept_types_str}_{sample_type}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
        print(f"Figure saved to {os.path.join(save_dir, filename)}")
    
    plt.show()
    
    return fig, axes


def compute_aggregated_error_from_per_ptm_ci(dataset_name, con_label, metric, weighted_avg, results_path, baseline_name=None):
    """
    Compute aggregated error by loading individual PTM CI files and weighting by concept frequency.
    
    Args:
        dataset_name: Name of dataset
        con_label: Base concept label (without PTM suffix)
        metric: Metric to get error for (f1, precision, recall)
        weighted_avg: Whether to compute weighted average
        results_path: Path to the PT file with best_ptm_per_concept info
        baseline_name: Optional baseline name for baseline methods
    
    Returns:
        float: Aggregated standard error
    """
    try:
        # Load the results to get PTM per concept
        if not os.path.exists(results_path):
            return 0
            
        results = torch.load(results_path, weights_only=False)
        if 'best_ptm_per_concept' not in results:
            return 0
            
        # Load ground truth to get concept weights
        if 'gt_samples_per_concept' in results:
            gt_samples_per_concept = results['gt_samples_per_concept']
        else:
            # Try to load GT separately
            # Determine model and input size from con_label
            if 'CLIP' in con_label:
                model_input_size = (224, 224)
            elif 'Llama' in con_label:
                if dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                    model_input_size = ('text', 'text')
                else:
                    model_input_size = (560, 560)
            elif 'Gemma' in con_label:
                model_input_size = ('text', 'text2')
            elif 'Qwen' in con_label:
                model_input_size = ('text', 'text3')
            else:
                return 0
                
            gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt'
            if os.path.exists(gt_path):
                try:
                    gt_samples_per_concept = torch.load(gt_path, weights_only=False)
                    from utils.filter_datasets_utils import filter_concept_dict
                    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
                except:
                    return 0
            else:
                return 0
            
        # Collect errors for each concept
        concept_errors = []
        concept_weights = []
        
        for concept, ptm_info in results['best_ptm_per_concept'].items():
            ptm = ptm_info['best_ptm']
            
            # Build CI file path
            if baseline_name:
                # For baselines: per_concept_ci_optimal_{baseline}_{con_label}_percentthrumodel_{ptm}.csv
                base_label = con_label.replace(f'_{baseline_name}', '')
                ci_file = f'Quant_Results_with_CI/{dataset_name}/per_concept_ci_optimal_{baseline_name}_{base_label}_percentthrumodel_{ptm}.csv'
            else:
                # For regular: per_concept_ci_optimal_{con_label}_percentthrumodel_{ptm}.csv
                ci_file = f'Quant_Results_with_CI/{dataset_name}/per_concept_ci_optimal_{con_label}_percentthrumodel_{ptm}.csv'
            
            # Try absolute path if relative doesn't exist
            ci_path = repo_file(ci_file)
            if not os.path.exists(ci_path):
                ci_path = repo_path('Experiments', *ci_file.split('/'))
            
            if os.path.exists(ci_path):
                try:
                    ci_df = pd.read_csv(ci_path)
                    error_col = f'{metric}_error'
                    
                    if error_col in ci_df.columns:
                        # Find this concept's row
                        concept_row = ci_df[ci_df['concept'] == concept]
                        if not concept_row.empty:
                            error = concept_row.iloc[0][error_col]
                            concept_errors.append(error)
                            if concept in gt_samples_per_concept:
                                concept_weights.append(len(gt_samples_per_concept[concept]))
                            else:
                                concept_weights.append(1)
                except:
                    continue
        
        # Aggregate errors
        if concept_errors:
            if weighted_avg and concept_weights:
                # Weighted average of errors
                total_weight = sum(concept_weights)
                weighted_error = sum(e * w for e, w in zip(concept_errors, concept_weights)) / total_weight
                return weighted_error
            else:
                # Simple average
                return sum(concept_errors) / len(concept_errors)
    except Exception as e:
        pass
        
    return 0


def summarize_best_detection_scores_finding_optimal_percentthrumodel_per_concept(
        dataset_name, model_names, sample_types, metric='f1', weighted_avg=True, 
        concept_types=None, include_errors=True, sorted=False, baselines=None, save_table=True):
    """
    Reads results from per_concept_ptm_optimization.py outputs and creates a summary table.
    
    This function reads the pre-computed optimal PTM results and formats them into a 
    summary DataFrame similar to other summarize functions.
    
    Args:
        dataset_name: Name of dataset
        model_names: List of model names to analyze
        sample_types: List of sample types ('patch' or 'cls')
        metric: Metric to use (default 'f1')
        weighted_avg: Whether to compute weighted average by concept frequency
        concept_types: List of concept types to include ['avg', 'linsep']
        include_errors: Whether to include error bars (if available)
        sorted: Whether to sort results by score
        baselines: List of baseline methods to include ['maxtoken', 'meantoken', 'lasttoken', 'randomtoken', 'prompt', 'random']
        save_table: Whether to save the table to ../Figs/Paper_Tables/ (default True)
        
    Returns:
        DataFrame with detection scores where each concept uses its optimal percentthrumodel
    """
    import numpy as np
    from collections import defaultdict
    
    results = []
    
    # Directory where optimization results are stored
    optimization_dir = f'Per_Concept_PTM_Optimization/{dataset_name}'
    
    if not os.path.exists(optimization_dir):
        print(f"Warning: No optimization results found at {optimization_dir}")
        print(f"Please run per_concept_ptm_optimization.py first.")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Default concept types if not specified
    if concept_types is None:
        concept_types = ['avg', 'linsep']
    
    # Process each model and configuration
    for model_name in model_names:
        for sample_type in sample_types:
            # Process concept types
            for concept_type in concept_types:
                # Build file name for regular detection
                n_clusters = 1000 if sample_type == 'patch' else 50
                
                if concept_type == 'avg':
                    con_label = f'{model_name}_avg_{sample_type}_embeddings'
                elif concept_type == 'linsep':
                    con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False'
                elif concept_type == 'kmeans':
                    con_label = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans'
                elif concept_type == 'linsep kmeans':
                    con_label = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans'
                else:
                    continue  # Unsupported concept type
                
                # Load optimization results
                results_path = os.path.join(optimization_dir, f'optimal_ptm_results_{con_label}.pt')
                aggregate_path = os.path.join(optimization_dir, f'optimal_ptm_aggregate_{con_label}.csv')
                
                if os.path.exists(aggregate_path):
                    # Load aggregate metrics
                    aggregate_df = pd.read_csv(aggregate_path)
                    if not aggregate_df.empty:
                        aggregate_metrics = aggregate_df.iloc[0].to_dict()
                        
                        # Create result entry
                        sample_label = 'SuperTok' if sample_type == 'patch' else 'CLS'
                        method_label = f"{sample_label} {concept_type.title()}"
                        
                        # Get the requested metric
                        metric_key = f'weighted_{metric}' if weighted_avg else f'avg_{metric}'
                        if metric_key in aggregate_metrics:
                            score = aggregate_metrics[metric_key]
                        else:
                            # Fallback to basic metric name
                            score = aggregate_metrics.get(metric, np.nan)
                        
                        # Format concept type for display
                        concept_display = concept_type.replace('avg', 'Avg').replace('linsep kmeans', 'LinsepKmeans').replace('linsep', 'Linsep').replace('kmeans', 'Kmeans')
                        
                        # Set detection method based on sample type
                        if sample_type == 'patch':
                            detection_method = 'SuperTok'
                        else:
                            detection_method = 'CLS'
                        
                        result_entry = {
                            'Model': model_name,
                            'Sample Type': sample_type,
                            'Concept Type': concept_display,
                            'Detection Method': detection_method,
                            f'Best {metric.upper()}': f"{score:.3f}"
                        }
                        
                        # Add standard error if requested
                        if include_errors:
                            # First try to get SE from aggregate CSV (it's already loaded)
                            se_key = f'weighted_{metric}_se' if weighted_avg else f'macro_{metric}_se'
                            std_error = 0
                            
                            if se_key in aggregate_metrics:
                                std_error = aggregate_metrics[se_key]
                            
                            # If SE is 0, try loading from PT file
                            if std_error == 0 and os.path.exists(results_path):
                                try:
                                    detailed_results = torch.load(results_path, weights_only=False)
                                    if 'aggregate_metrics' in detailed_results:
                                        if se_key in detailed_results['aggregate_metrics']:
                                            std_error = detailed_results['aggregate_metrics'][se_key]
                                except Exception:
                                    pass
                            
                            # If still 0, compute from individual CI files
                            if std_error == 0:
                                std_error = compute_aggregated_error_from_per_ptm_ci(
                                    dataset_name, con_label, metric, weighted_avg, results_path
                                )
                            
                            if std_error > 0:
                                result_entry[f'Best {metric.upper()}'] = f"{score:.3f} ± {std_error:.3f}"
                        
                        # Add PTM distribution info if available
                        if 'ptm_distribution' in aggregate_metrics:
                            result_entry['PTM Distribution'] = aggregate_metrics['ptm_distribution']
                        
                        results.append(result_entry)
                        
                        # Print PTM distribution if available in detailed results
                        if os.path.exists(results_path):
                            try:
                                detailed_results = torch.load(results_path, weights_only=False)
                                if 'best_ptm_per_concept' in detailed_results:
                                    ptm_counts = defaultdict(int)
                                    for concept, info in detailed_results['best_ptm_per_concept'].items():
                                        ptm_counts[info['best_ptm']] += 1
                                    
                                    # Removed PTM distribution printing for cleaner output
                                    pass
                            except Exception:
                                pass
            
            # Process baselines if requested
            if baselines and sample_type == 'patch':  # Baselines only work for patch/token level
                # Remove duplicates from baselines list (handle avgtoken alias)
                processed_baselines = []
                for baseline_name in baselines:
                    if baseline_name == 'avgtoken':
                        if 'meantoken' not in processed_baselines:
                            processed_baselines.append('meantoken')
                    elif baseline_name not in processed_baselines:
                        processed_baselines.append(baseline_name)
                
                for baseline_name in processed_baselines:
                    # Build file names for baseline detection
                    for concept_type in concept_types:
                        n_clusters = 1000 if sample_type == 'patch' else 50
                        
                        if concept_type == 'avg':
                            con_label = f'{model_name}_avg_{sample_type}_embeddings_{baseline_name}'
                        elif concept_type == 'linsep':
                            con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_{baseline_name}'
                        elif concept_type == 'kmeans':
                            con_label = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_{baseline_name}'
                        elif concept_type == 'linsep kmeans':
                            con_label = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_{baseline_name}'
                        else:
                            continue
                        
                        # Load baseline optimization results
                        aggregate_path = os.path.join(optimization_dir, f'optimal_ptm_aggregate_{con_label}.csv')
                        baseline_results_path = os.path.join(optimization_dir, f'optimal_ptm_results_{con_label}.pt')
                        
                        if os.path.exists(aggregate_path):
                            # Load aggregate metrics
                            aggregate_df = pd.read_csv(aggregate_path)
                            if not aggregate_df.empty:
                                aggregate_metrics = aggregate_df.iloc[0].to_dict()
                                
                                # Map baseline names to display names
                                baseline_display_name = {
                                    'maxtoken': 'MaxTok',
                                    'meantoken': 'MeanTok',
                                    'lasttoken': 'LastTok',
                                    'randomtoken': 'RandTok'
                                }.get(baseline_name, baseline_name)
                                
                                # Get the requested metric
                                metric_key = f'weighted_{metric}' if weighted_avg else f'avg_{metric}'
                                if metric_key in aggregate_metrics:
                                    score = aggregate_metrics[metric_key]
                                else:
                                    score = aggregate_metrics.get(metric, np.nan)
                                
                                # Only add if we have a valid score
                                if not np.isnan(score):
                                    # Format concept type for display
                                    concept_display = concept_type.replace('avg', 'Avg').replace('linsep kmeans', 'LinsepKmeans').replace('linsep', 'Linsep').replace('kmeans', 'Kmeans')
                                    
                                    result_entry = {
                                        'Model': model_name,
                                        'Sample Type': sample_type,
                                        'Concept Type': concept_display,
                                        'Detection Method': baseline_display_name,
                                        f'Best {metric.upper()}': f"{score:.3f}"
                                    }
                                    
                                    # Add standard error if requested
                                    if include_errors:
                                        # First try to get SE from aggregate CSV
                                        se_key = f'weighted_{metric}_se' if weighted_avg else f'macro_{metric}_se'
                                        std_error = 0
                                        
                                        if se_key in aggregate_metrics:
                                            std_error = aggregate_metrics[se_key]
                                        
                                        # If SE is 0, try loading from PT file
                                        if std_error == 0 and os.path.exists(baseline_results_path):
                                            try:
                                                detailed_results = torch.load(baseline_results_path, weights_only=False)
                                                if 'aggregate_metrics' in detailed_results:
                                                    if se_key in detailed_results['aggregate_metrics']:
                                                        std_error = detailed_results['aggregate_metrics'][se_key]
                                            except Exception:
                                                pass
                                        
                                        # If still 0, compute from individual CI files
                                        if std_error == 0:
                                            std_error = compute_aggregated_error_from_per_ptm_ci(
                                                dataset_name, con_label, metric, weighted_avg, baseline_results_path, baseline_name
                                            )
                                        
                                        if std_error > 0:
                                            result_entry[f'Best {metric.upper()}'] = f"{score:.3f} ± {std_error:.3f}"
                                    
                                    results.append(result_entry)
                                    
                                    # Print PTM distribution for baseline
                                    if os.path.exists(baseline_results_path):
                                        try:
                                            detailed_results = torch.load(baseline_results_path, weights_only=False)
                                            if 'best_ptm_per_concept' in detailed_results:
                                                ptm_counts = defaultdict(int)
                                                for concept, info in detailed_results['best_ptm_per_concept'].items():
                                                    ptm_counts[info['best_ptm']] += 1
                                                
                                                # Removed PTM distribution printing for cleaner output
                                                pass
                                        except Exception:
                                            pass
    
    # Process non-token baselines (prompt, random)
    if baselines:
        # Load ground truth for weighted calculations
        if model_names and len(model_names) > 0:
            model_name = model_names[0]  # Use first model for GT loading
            
            # Determine correct input size based on model and dataset
            if model_name == 'CLIP':
                gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(224, 224).pt'
            elif model_name == 'Llama':
                if dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                    gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text').pt"
                else:
                    gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(560, 560).pt'
            elif model_name == 'Gemma':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text2').pt"
            elif model_name == 'Qwen':
                gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text3').pt"
            else:
                gt_path = None
                
            if gt_path and os.path.exists(gt_path):
                gt_samples_per_concept = torch.load(gt_path, weights_only=False)
                from utils.filter_datasets_utils import filter_concept_dict
                gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
                
                # Add prompt baseline if requested (only for rate-based metrics, not counts)
                if 'prompt' in baselines and metric not in ['fp', 'fn', 'tp', 'tn']:
                    for model_name in model_names:
                        try:
                            prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, 'test')
                            prompt_error = 0
                            
                            # Try to load prompt CI if available
                            if include_errors:
                                # First try the per-concept CSV to compute weighted error
                                prompt_ci_file = f'Quant_Results_with_CI/{dataset_name}/per_concept_ci_direct_prompt_Llama-3.2-11B-Vision.csv'
                                if os.path.exists(prompt_ci_file):
                                    try:
                                        ci_df = pd.read_csv(prompt_ci_file)
                                        # Filter to concepts that are in ground truth
                                        ci_df = ci_df[ci_df['concept'].isin(gt_samples_per_concept.keys())]
                                        
                                        error_col = f'{metric}_error'
                                        if error_col in ci_df.columns:
                                            if weighted_avg:
                                                # Compute weighted average of errors
                                                total_weight = sum(len(gt_samples_per_concept[c]) for c in ci_df['concept'])
                                                weighted_error_sum = 0
                                                for _, row in ci_df.iterrows():
                                                    concept = row['concept']
                                                    if concept in gt_samples_per_concept:
                                                        weight = len(gt_samples_per_concept[concept])
                                                        weighted_error_sum += row[error_col] * weight
                                                prompt_error = weighted_error_sum / total_weight if total_weight > 0 else 0
                                            else:
                                                # Simple average of errors
                                                prompt_error = ci_df[error_col].mean()
                                    except Exception as e:
                                        # Fallback to dataset-level CI if available
                                        dataset_ci_file = f'Quant_Results_with_CI/{dataset_name}/dataset_ci_direct_prompt_Llama-3.2-11B-Vision.json'
                                        if os.path.exists(dataset_ci_file):
                                            try:
                                                import json
                                                with open(dataset_ci_file, 'r') as f:
                                                    dataset_ci = json.load(f)
                                                metric_key = f"{'weighted' if weighted_avg else 'macro'}_{metric}"
                                                if metric_key in dataset_ci:
                                                    prompt_error = dataset_ci[metric_key].get('error', 0)
                                            except:
                                                pass
                            
                            result_entry = {
                                'Model': model_name,
                                'Sample Type': 'N/A',  # Prompt doesn't have sample type
                                'Concept Type': 'N/A',  # Prompt doesn't have concept type
                                'Detection Method': 'Prompt',
                                f'Best {metric.upper()}': f"{prompt_score:.3f}"
                            }
                            
                            if include_errors and prompt_error > 0:
                                result_entry[f'Best {metric.upper()}'] = f"{prompt_score:.3f} ± {prompt_error:.3f}"
                            else:
                                result_entry[f'Best {metric.upper()}'] = f"{prompt_score:.3f}"
                                
                            results.append(result_entry)
                        except Exception as e:
                            pass  # Could not compute prompt baseline
                
                # Add random baseline if requested
                if 'random' in baselines:
                    for model_name in model_names:
                        baseline_path = f'Quant_Results/{dataset_name}/random_{model_name}_cls_baseline.csv'
                        if os.path.exists(baseline_path):
                            try:
                                df = pd.read_csv(baseline_path)
                                df = df[df['concept'].isin(gt_samples_per_concept)]
                                if weighted_avg:
                                    total = sum(len(gt_samples_per_concept[c]) for c in df['concept'])
                                    if total > 0:
                                        random_score = sum(
                                            row[metric] * len(gt_samples_per_concept[row['concept']]) 
                                            for _, row in df.iterrows()
                                        ) / total
                                    else:
                                        random_score = df[metric].mean()
                                else:
                                    random_score = df[metric].mean()
                                
                                result_entry = {
                                    'Model': model_name,
                                    'Sample Type': 'N/A',  # Random doesn't have sample type
                                    'Concept Type': 'N/A',  # Random doesn't have concept type
                                    'Detection Method': 'Random',
                                    f'Best {metric.upper()}': f"{random_score:.3f}"
                                }
                                results.append(result_entry)
                            except Exception:
                                pass
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort if requested
    if sorted and results_df.shape[0] > 0:
        results_df['sort_value'] = results_df[f'Best {metric.upper()}'].apply(
            lambda x: float(str(x).split(' ±')[0]) if isinstance(x, str) else x
        )
        results_df = results_df.sort_values('sort_value', ascending=False)
        results_df = results_df.drop('sort_value', axis=1)
    
    # Save table if requested
    if save_table and not results_df.empty:
        # Create output directory
        output_dir = '../Figs/Paper_Tables'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create descriptive filename
        models_str = '_'.join(model_names)
        sample_types_str = '_'.join(sample_types)
        concept_types_str = '_'.join(concept_types) if concept_types else 'avg_linsep'
        
        # Add baseline info if present
        baseline_str = ''
        if baselines:
            unique_baselines = []
            for b in baselines:
                if b == 'avgtoken':
                    if 'meantoken' not in unique_baselines:
                        unique_baselines.append('meantoken')
                elif b not in unique_baselines:
                    unique_baselines.append(b)
            if unique_baselines:
                baseline_str = f'_with_{"+".join(unique_baselines)}_baselines'
        
        # Include weighted/unweighted info
        weight_str = 'weighted' if weighted_avg else 'unweighted'
        
        # Include errors info
        error_str = 'with_errors' if include_errors else 'no_errors'
        
        filename = f'{dataset_name}_{models_str}_{sample_types_str}_{concept_types_str}_ptm_per_concept_{metric}_{weight_str}_{error_str}{baseline_str}.csv'
        filepath = os.path.join(output_dir, filename)
        
        # Save the table
        results_df.to_csv(filepath, index=False)
        print(f"\nTable saved to: {filepath}")
    
    return results_df





def plot_optimal_ptm_distribution(
        dataset_names, model_names, concept_types=None, sample_types=None,
        detection_methods=None, figsize=None, save_fig=True, save_dir='../Figs/Paper_Figs',
        title_suffix='', text_size=None, legend_size=None, legend_pos=None):
    """
    Create a stacked bar plot showing the distribution of optimal percentthrumodel values
    across concepts for different model/concept-type combinations.
    
    Args:
        dataset_names: List of dataset names or single dataset name
        model_names: List of model names to analyze
        concept_types: List of concept types (default: ['avg', 'linsep'])
        sample_types: List of sample types (default: ['patch'])
        detection_methods: List of detection methods (default: ['regular'])
        figsize: Figure size (width, height)
        save_fig: Whether to save the figure
        save_dir: Directory to save the figure
        title_suffix: Additional text to add to title
        text_size: Dict with 'legend' and 'label' font sizes (default: {'legend': 10, 'label': 12})
        legend_size: Specific font size for legend (overrides text_size['legend'] if provided)
        legend_pos: Legend position - 'left', 'right', 'middle', or None for auto
        
    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict, Counter
    import seaborn as sns
    
    # Handle single values
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    if isinstance(model_names, str):
        model_names = [model_names]
    
    # Defaults
    if concept_types is None:
        concept_types = ['avg', 'linsep']
    if sample_types is None:
        sample_types = ['patch']
    if detection_methods is None:
        detection_methods = ['regular']
    
    # Handle text_size parameter
    if text_size is None:
        text_size = {'legend': 10, 'label': 12}
    elif isinstance(text_size, (int, float)):
        # If a single number is passed, use it for both legend and label
        text_size = {'legend': text_size, 'label': text_size}
    elif not isinstance(text_size, dict):
        # If it's not a dict or number, use defaults
        text_size = {'legend': 10, 'label': 12}
    
    # Collect PTM distributions for each configuration
    ptm_data = defaultdict(lambda: defaultdict(int))  # {(model, concept_type, detection_method): {ptm: count}}
    all_ptms = set()
    
    for dataset_name in dataset_names:
        for model_name in model_names:
            # Determine input size for this model/dataset
            if dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank']:
                if model_name == 'Llama':
                    input_size = ('text', 'text')
                elif model_name == 'Gemma':
                    input_size = ('text', 'text2')
                elif model_name == 'Qwen':
                    input_size = ('text', 'text3')
                else:
                    input_size = None
            else:  # Image datasets
                if model_name == 'CLIP':
                    input_size = (224, 224)
                elif model_name == 'Llama':
                    input_size = (560, 560)
                else:
                    input_size = None
            
            if input_size:
                # Get default PTMs for this model
                default_ptms = get_model_default_percentthrumodels(model_name, input_size)
                all_ptms.update(default_ptms)
            
            for sample_type in sample_types:
                for concept_type in concept_types:
                    for detection_method in detection_methods:
                        # Map detection method to appropriate sample type and suffix
                        if detection_method == 'regular' or detection_method == 'supertok':
                            # Both regular and supertok use patch sample type with no suffix
                            actual_sample_type = sample_type
                            method_suffix = ''
                        elif detection_method == 'cls':
                            # CLS uses cls sample type with no suffix
                            actual_sample_type = 'cls'
                            method_suffix = ''
                        else:
                            # Other detection methods (baselines) use original sample type with suffix
                            actual_sample_type = sample_type
                            method_suffix = f'_{detection_method}'
                        
                        if concept_type == 'avg':
                            con_label = f'{model_name}_avg_{actual_sample_type}_embeddings{method_suffix}'
                        else:
                            con_label = f'{model_name}_linsep_{actual_sample_type}_embeddings_BD_True_BN_False{method_suffix}'
                        
                        # Load results
                        results_path = f'Per_Concept_PTM_Optimization/{dataset_name}/optimal_ptm_results_{con_label}.pt'
                        
                        if os.path.exists(results_path):
                            try:
                                data = torch.load(results_path, weights_only=False)
                                if 'best_ptm_per_concept' in data:
                                    # Count PTMs for this configuration
                                    for concept_info in data['best_ptm_per_concept'].values():
                                        ptm = concept_info['best_ptm']
                                        # Create key that includes detection method
                                        key = (model_name, concept_type, detection_method)
                                        ptm_data[key][ptm] += 1
                                        
                                    print(f"Loaded {len(data['best_ptm_per_concept'])} concepts for {model_name} {concept_type} {detection_method}")
                            except Exception as e:
                                print(f"Warning: Could not load {results_path}: {e}")
                        else:
                            print(f"File not found: {results_path}")
    
    if not ptm_data:
        print("No PTM optimization results found!")
        return None
    
    # Sort PTMs
    sorted_ptms = sorted(all_ptms)
    
    # Prepare data for stacked bar plot
    # Order configurations to preserve the input order of models and detection methods
    configurations = []
    for model_name in model_names:
        for detection_method in detection_methods:
            for concept_type in concept_types:
                key = (model_name, concept_type, detection_method)
                if key in ptm_data:
                    configurations.append(key)
    n_configs = len(configurations)
    n_ptms = len(sorted_ptms)
    
    # Create color palette - use distinct colors for each model/concept-type/method combo
    colors = sns.color_palette("husl", n_configs)
    
    # Set up the figure
    if figsize is None:
        figsize = (max(10, len(sorted_ptms) * 0.8), 6)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Width of bars
    bar_width = 0.8
    
    # X positions for bars
    x_pos = np.arange(len(sorted_ptms))
    
    # Get unique detection methods in order
    unique_detection_methods = list(dict.fromkeys(detection_method for _, _, detection_method in configurations))
    n_detection_methods = len(unique_detection_methods)
    
    # Create color mapping for detection methods
    detection_method_colors = dict(zip(unique_detection_methods, sns.color_palette("husl", n_detection_methods)))
    
    # Reorganize configurations to group by detection method
    grouped_configs = []
    for detection_method in unique_detection_methods:
        for model, concept_type, det_method in configurations:
            if det_method == detection_method:
                grouped_configs.append((model, concept_type, det_method))
    
    # Create stacked bars
    bottom = np.zeros(len(sorted_ptms))
    legend_added = set()
    
    for model, concept_type, detection_method in grouped_configs:
        counts = []
        for ptm in sorted_ptms:
            counts.append(ptm_data[(model, concept_type, detection_method)].get(ptm, 0))
        
        # Create label only for first occurrence of each detection method
        if detection_method not in legend_added:
            if detection_method == 'regular':
                label = "Patch"
            elif detection_method == 'supertok':
                label = "SuperTok"
            elif detection_method == 'cls':
                label = "CLS"
            else:
                # Map baseline names to display names
                baseline_display = {
                    'maxtoken': 'MaxTok',
                    'meantoken': 'MeanTok', 
                    'lasttoken': 'LastTok',
                    'randomtoken': 'RandTok'
                }.get(detection_method, detection_method)
                label = baseline_display
            legend_added.add(detection_method)
        else:
            label = None  # Don't add duplicate legend entries
        
        # Use consistent color for same detection method
        color = detection_method_colors[detection_method]
        
        # Plot bar segment
        ax.bar(x_pos, counts, bar_width, bottom=bottom, 
               color=color, label=label, alpha=0.9)
        
        # Update bottom for next stack
        bottom += np.array(counts)
    
    # Customize the plot
    ax.set_xlabel('% Through Model', fontsize=text_size.get('label', 12))
    ax.set_ylabel('# of Concepts', fontsize=text_size.get('label', 12))
    
    # No title
    
    # Calculate legend font size early so we can use it for x-axis labels
    if legend_size is not None:
        legend_fontsize = legend_size
    else:
        legend_fontsize = text_size.get('legend', 10)
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{ptm}%' for ptm in sorted_ptms])
    
    # Set x-axis tick label size to match legend
    ax.tick_params(axis='x', labelsize=legend_fontsize)
    
    # Set y-axis to use integers
    max_y = int(np.max(bottom)) + 1
    ax.set_ylim(0, max_y)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    
    # Determine legend position
    if legend_pos is not None:
        # Use user-specified position
        if legend_pos.lower() == 'left':
            legend_loc = 'upper left'
        elif legend_pos.lower() == 'right':
            legend_loc = 'upper right'
        elif legend_pos.lower() == 'middle':
            legend_loc = 'upper center'
        else:
            # Default to upper left for invalid input
            legend_loc = 'upper left'
    else:
        # Auto-determine based on data distribution
        # Calculate center of mass of the data
        total_counts_by_ptm = bottom
        weighted_sum = sum(i * count for i, count in enumerate(total_counts_by_ptm))
        total_sum = sum(total_counts_by_ptm)
        if total_sum > 0:
            center_of_mass = weighted_sum / total_sum / len(sorted_ptms)
        else:
            center_of_mass = 0.5
        
        # Place legend on the opposite side of where most data is
        if center_of_mass < 0.5:
            # Data is on the left, put legend on the right
            legend_loc = 'upper right'
        else:
            # Data is on the right, put legend on the left
            legend_loc = 'upper left'
    
    # Legend inside the plot
    ax.legend(loc=legend_loc, fontsize=legend_fontsize, 
              framealpha=0.95, edgecolor='gray')
    
    # Don't add total counts on top of bars
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        
        # Create filename
        models_str = '_'.join(model_names)
        datasets_str = '_'.join(dataset_names)
        methods_str = '_'.join(detection_methods) if detection_methods != ['regular'] else 'regular'
        filename = f'ptm_distribution_{datasets_str}_{models_str}_{methods_str}.pdf'
        filepath = os.path.join(save_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")
    
    return fig


def plot_detection_scores_multi_dataset(model_name, datasets, titles, split='test', sample_types=['patch'], 
                                      metric='f1', weighted_avg=True, plot_type='both', concept_types=None,
                                      baseline_types=None, save_filename=None, ylim=None, 
                                      figsize=(18, 4), percentthrumodel=100, 
                                      label_font_size=12, legend_font=None, xlabel=None, 
                                      suptitle=None, show_legend=True, legend_on_plot=False):
    """
    Plot detection scores for multiple datasets side by side for a single model.
    
    Args:
        model_name: Model name (e.g., 'CLIP', 'Llama', 'Gemma', 'Qwen')
        datasets: List of dataset names to plot
        titles: List of titles for each subplot (must match length of datasets)
        split: Data split to use ('test', 'val', 'cal')
        sample_types: List of sample types to plot (default ['patch'] for token-level only)
        metric: Metric to plot ('f1', 'precision', 'recall', 'accuracy')
        weighted_avg: Whether to use weighted averaging
        plot_type: Type of plots to show ('supervised', 'unsupervised', 'both')
        concept_types: List of concept types to include ['avg', 'linsep', 'kmeans', 'linsep kmeans', 'sae']
        baseline_types: Baseline types to plot (if any)
        save_filename: Path to save the combined figure
        ylim: Y-axis limits (tuple or None for default (0, 1.05))
        figsize: Overall figure size (width, height) in inches (default: (18, 4))
        percentthrumodel: Percent through model value
        label_font_size: Font size for axis labels and titles (default: 12)
        legend_font: Font size for legend. If None, uses label_font_size (default: None)
        xlabel: Custom x-axis label. If None, uses default "Ground Truth Concept Recall Percentage" (default: None)
        suptitle: Super title for the entire figure
        show_legend: Whether to show legend (default: True)
        legend_on_plot: If True, includes legend on plot. If False, creates separate legend figure (default: False)
    
    Returns:
        None (displays and optionally saves the plot)
    """
    
    # Validate inputs
    if len(datasets) != len(titles):
        raise ValueError(f"Number of datasets ({len(datasets)}) must match number of titles ({len(titles)})")
    
    # Use legend_font if provided, otherwise use label_font_size
    if legend_font is None:
        legend_font = label_font_size
    
    # Calculate figure size
    n_plots = len(datasets)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Handle single dataset case
    if n_plots == 1:
        axes = [axes]
    
    # Track all lines and labels for a unified legend
    all_lines = []
    all_labels = []
    seen_labels = set()
    
    # Plot each dataset
    for i, (dataset_name, title, ax) in enumerate(zip(datasets, titles, axes)):
        # Plot on this axis
        lines, labels = _plot_detection_scores_on_axis(
            ax=ax,
            dataset_name=dataset_name,
            split=split,
            model_name=model_name,
            sample_types=sample_types,
            metric=metric,
            weighted_avg=weighted_avg,
            plot_type=plot_type,
            concept_types=concept_types,
            baseline_types=baseline_types,
            percentthrumodel=percentthrumodel,
            label_font_size=label_font_size,
            ylim=ylim,
            xlabel=None,  # We'll set it manually below
            ylabel=None,  # We'll set it manually below
            show_legend=False  # We'll add legend manually at the end
        )
        
        # Set title for this subplot
        ax.set_title(title, fontweight='bold', fontsize=label_font_size)
        
        # Format axes to match plot_detection_scores
        # Only show y-label on first subplot
        if i == 0:
            ax.set_ylabel("F1" if metric == 'f1' else f"{metric.upper()} Score", 
                         fontsize=label_font_size, rotation=0, ha='right')
        
        # Set x-label
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=label_font_size)
        else:
            ax.set_xlabel("Ground Truth Concept Recall Percentage", fontsize=label_font_size)
        
        # Set x-axis to percentages (0-100)
        ax.set_xlim(0, 100)
        
        # Set y-axis limits
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(0, 1.05)
        
        # Set ticks every 10%
        tick_positions = np.arange(0, 110, 10)  # Every 10%
        tick_labels = []
        for pos in tick_positions:
            # Label only at 10%, 30%, 50%, 70%, 90%
            if pos in [10, 30, 50, 70, 90]:
                tick_labels.append(f"{pos}%")
            else:
                tick_labels.append("")  # Empty label for other ticks
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)
        
        # Set tick label font size to match legend font
        ax.tick_params(axis='both', which='major', labelsize=legend_font)
        ax.tick_params(axis='x', pad=2)  # Reduce padding for x-axis labels
        
        # Hide 0.0 on y-axis
        yticks = ax.get_yticks()
        ylabels = []
        for tick in yticks:
            if tick == 0:
                ylabels.append("")
            else:
                ylabels.append(f"{tick:.1f}")
        ax.set_yticklabels(ylabels)
        
        # Collect unique lines and labels
        for line, label in zip(lines, labels):
            if label not in seen_labels:
                all_lines.append(line)
                all_labels.append(label)
                seen_labels.add(label)
    
    # Add overall title if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=label_font_size + 2, fontweight='bold')
    
    # Add legend to plot if requested
    if legend_on_plot and all_lines:
        # Create a compact legend on the right side of the last subplot
        axes[-1].legend(all_lines, all_labels, title="Concept Type", 
                       loc='center left', bbox_to_anchor=(1.02, 0.5), 
                       frameon=True, fontsize=legend_font, title_fontsize=legend_font, ncol=1)
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for legend on the right
    else:
        # Adjust layout without legend
        plt.tight_layout()
    
    # Save if filename provided
    if save_filename:
        plt.savefig(save_filename, dpi=500, format='pdf', bbox_inches='tight')
        print(f"Saved figure to: {save_filename}")
        
    # Create a separate figure for the legend if not on plot and requested
    if not legend_on_plot and show_legend and all_lines:
        # Calculate height based on number of entries
        n_entries = len(all_lines)
        fig_height = max(6, n_entries * 0.8)  # At least 6 inches, 0.8 inch per entry
        fig_legend = plt.figure(figsize=(4, fig_height))
        
        # Create legend on the new figure with proper spacing
        fig_legend.legend(all_lines, all_labels, title="Concept Type", loc='center', 
                         frameon=True, fontsize=legend_font, title_fontsize=legend_font,
                         labelspacing=2.0, handlelength=3.0, handletextpad=1.0)
        
        # Save the legend as a separate file
        if save_filename:
            legend_path = save_filename.replace('.pdf', '_legend.pdf')
            fig_legend.savefig(legend_path, dpi=500, format='pdf', bbox_inches='tight')
            print(f"Saved legend to: {legend_path}")
        plt.close(fig_legend)
    
    plt.show()
