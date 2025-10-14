import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.general_utils import create_binary_labels
from utils.patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence


def compute_label_correlations(gt_patches_per_concept, dataset_name, model_input_size, acts_path, 
                                   device="cuda", patch_size=14, scratch_dir=''):
    """
    Compute correlations between patch-CLS activation metrics and ground truth concept patches
    for each concept, using only test data.
    
    Args:
        dataset_name (str): Name of dataset
        model_input_size (tuple): Model input dimensions 
        acts_file (str): Activation metrics file (e.g., "patch_cls_cosine_similarities_*.csv")
        device (str): Torch device
        patch_size (int): Patch size for vision models
        scratch_dir (str): Scratch directory path
        
    Returns:
        dict: Correlation results per concept with Pearson and Spearman correlations
    """
    print(f"Loading activation metrics from: {acts_path}")
    act_metrics = pd.read_csv(acts_path)
    
    # Create binary labels for ALL patches using ALL gt_patches
    D = act_metrics.shape[0]
    all_concept_labels = create_binary_labels(D, gt_patches_per_concept)
    
    # Get patch split information and filter to test split only
    split_df = get_patch_split_df(dataset_name, model_input_size, patch_size=patch_size)
    
    # Filter to test split only
    test_mask = split_df == 'test'
    test_indices = test_mask[test_mask].index
    
    # Further filter to exclude padding patches
    relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size)
    relevant_indices_list = relevant_indices.tolist()
    
    # Get concept keys that are present in both activation metrics and ground truth
    act_concept_keys = set(act_metrics.columns)
    gt_concept_keys = set(all_concept_labels.keys())
    common_concepts = act_concept_keys & gt_concept_keys
    
    print(f"Computing correlations for {len(common_concepts)} common concepts")
    # Compute correlations for each concept
    correlation_results = {}
    
    for concept in common_concepts:
        # Get activation values for test patches
        act_values = act_metrics[concept].iloc[relevant_indices_list].values
        
        # Get ground truth binary labels for test patches
        gt_labels = all_concept_labels[concept][relevant_indices_list]
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.numpy()
        
        # Compute correlations
        pearson_corr, pearson_p = pearsonr(act_values, gt_labels)
        spearman_corr, spearman_p = spearmanr(act_values, gt_labels)
        
        correlation_results[concept] = {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr, 
            'spearman_p_value': spearman_p,
            'n_positive_samples': int(gt_labels.sum()),
            'n_total_samples': len(gt_labels),
            'positive_ratio': float(gt_labels.sum() / len(gt_labels))
        }
    return correlation_results


def save_correlation_results(correlation_results, dataset_name, scheme, model_name, scratch_dir=''):
    """
    Save correlation results to CSV file.
    
    Args:
        correlation_results (dict): Results from compute_patch_cls_correlations
        dataset_name (str): Dataset name
        con_label (str): Concept label for filename
        scratch_dir (str): Scratch directory path
    """
    # Convert to DataFrame
    results_df = pd.DataFrame.from_dict(correlation_results, orient='index')
    results_df.index.name = 'concept'
    results_df = results_df.reset_index()
    
    # Create output directory
    output_dir = f'Act_Correlations/{dataset_name}/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = f'label_correlations_{scheme}_{model_name}.csv'
    output_path = os.path.join(output_dir, output_file)
    results_df.to_csv(output_path, index=False)
    
    print(f"Correlation results saved to: {output_path}")
    return output_path


def summarize_correlations(correlation_results):
    """
    Print summary statistics of correlation results.
    
    Args:
        correlation_results (dict): Results from compute_patch_cls_correlations
    """
    # Extract correlation values
    pearson_corrs = [r['pearson_correlation'] for r in correlation_results.values() if not np.isnan(r['pearson_correlation'])]
    spearman_corrs = [r['spearman_correlation'] for r in correlation_results.values() if not np.isnan(r['spearman_correlation'])]
    
    print(f"\n=== Correlation Summary ===")
    print(f"Total concepts: {len(correlation_results)}")
    print(f"Valid Pearson correlations: {len(pearson_corrs)}")
    print(f"Valid Spearman correlations: {len(spearman_corrs)}")
    
    if pearson_corrs:
        print(f"\nPearson Correlation Stats:")
        print(f"  Mean: {np.mean(pearson_corrs):.4f}")
        print(f"  Median: {np.median(pearson_corrs):.4f}")
        print(f"  Std: {np.std(pearson_corrs):.4f}")
        print(f"  Min: {np.min(pearson_corrs):.4f}")
        print(f"  Max: {np.max(pearson_corrs):.4f}")
    
    if spearman_corrs:
        print(f"\nSpearman Correlation Stats:")
        print(f"  Mean: {np.mean(spearman_corrs):.4f}")
        print(f"  Median: {np.median(spearman_corrs):.4f}")
        print(f"  Std: {np.std(spearman_corrs):.4f}")
        print(f"  Min: {np.min(spearman_corrs):.4f}")
        print(f"  Max: {np.max(spearman_corrs):.4f}")
    
    # Show top correlations
    if correlation_results:
        sorted_by_pearson = sorted(correlation_results.items(), 
                                 key=lambda x: x[1]['pearson_correlation'] if not np.isnan(x[1]['pearson_correlation']) else -999, 
                                 reverse=True)
        
        print(f"\nTop 5 Concepts by Pearson Correlation:")
        for i, (concept, results) in enumerate(sorted_by_pearson[:5]):
            if not np.isnan(results['pearson_correlation']):
                print(f"  {i+1}. {concept}: {results['pearson_correlation']:.4f} (p={results['pearson_p_value']:.4f})")