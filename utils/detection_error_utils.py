"""
Detection Error Analysis Utilities
==================================

This module provides bootstrap-based error analysis for concept detection metrics.
It computes confidence intervals at both per-concept and dataset levels using:
- Single-level bootstrap for per-concept error bars
- Two-level bootstrap for dataset-level aggregated metrics

Key Functions:
- compute_per_concept_bootstrap: Bootstrap confidence intervals for individual concepts
- compute_dataset_level_bootstrap: Two-level bootstrap for dataset-wide metrics
- generate_error_report: Generate formatted reports with error bars
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import os
from tqdm import tqdm
import json


def bootstrap_metrics(
    tp: int, fp: int, tn: int, fn: int, 
    n_samples: int, n_bootstrap: int = 1000, 
    confidence_level: float = 0.95, seed: int = 42
) -> Dict[str, Tuple[float, float, float]]:
    """
    Bootstrap confidence intervals for detection metrics from confusion matrix counts.
    
    Args:
        tp, fp, tn, fn: Confusion matrix counts
        n_samples: Total number of samples
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility
        
    Returns:
        Dict mapping metric name -> (mean, lower_ci, upper_ci)
    """
    np.random.seed(seed)
    
    # Create sample outcomes based on counts
    # Use integers instead of tuples for efficiency
    # 0=TP, 1=FP, 2=TN, 3=FN
    outcomes = np.array(
        [0] * tp + 
        [1] * fp + 
        [2] * tn + 
        [3] * fn,
        dtype=np.int8
    )
    
    if len(outcomes) != n_samples:
        raise ValueError(f"Sum of counts ({len(outcomes)}) != n_samples ({n_samples})")
    
    # Bootstrap
    metrics_bootstrap = defaultdict(list)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = np.random.choice(outcomes, size=n_samples, replace=True)
        
        # Count outcomes efficiently
        counts = np.bincount(resampled, minlength=4)
        tp_boot = counts[0]
        fp_boot = counts[1]
        tn_boot = counts[2]
        fn_boot = counts[3]
        
        # Compute metrics
        precision = tp_boot / (tp_boot + fp_boot) if (tp_boot + fp_boot) > 0 else 0
        recall = tp_boot / (tp_boot + fn_boot) if (tp_boot + fn_boot) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp_boot + tn_boot) / n_samples
        specificity = tn_boot / (tn_boot + fp_boot) if (tn_boot + fp_boot) > 0 else 0
        fpr = fp_boot / (fp_boot + tn_boot) if (fp_boot + tn_boot) > 0 else 0
        
        metrics_bootstrap['precision'].append(precision)
        metrics_bootstrap['recall'].append(recall)
        metrics_bootstrap['f1'].append(f1)
        metrics_bootstrap['accuracy'].append(accuracy)
        metrics_bootstrap['specificity'].append(specificity)
        metrics_bootstrap['fpr'].append(fpr)
        metrics_bootstrap['tpr'].append(recall)  # TPR = recall
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    results = {}
    for metric_name, values in metrics_bootstrap.items():
        values = np.array(values)
        mean_val = np.mean(values)
        lower_ci = np.percentile(values, lower_percentile)
        upper_ci = np.percentile(values, upper_percentile)
        results[metric_name] = (mean_val, lower_ci, upper_ci)
    
    return results


def compute_per_concept_bootstrap(
    detection_results: pd.DataFrame,
    gt_samples_per_concept: Dict[str, List[int]],
    total_samples: int,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for each concept individually.
    
    Args:
        detection_results: DataFrame with columns ['concept', 'tp', 'fp', 'tn', 'fn', ...]
        gt_samples_per_concept: Ground truth samples per concept
        total_samples: Total number of samples in the dataset
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals
        seed: Random seed
        
    Returns:
        DataFrame with original metrics plus confidence intervals
    """
    results = []
    
    for _, row in tqdm(detection_results.iterrows(), total=len(detection_results), 
                       desc="Computing per-concept bootstrap"):
        concept = row['concept']
        tp, fp, tn, fn = int(row['tp']), int(row['fp']), int(row['tn']), int(row['fn'])
        
        # Get number of samples for this concept
        n_samples = tp + fp + tn + fn
        
        # Skip if no samples
        if n_samples == 0:
            continue
            
        # Bootstrap metrics
        bootstrap_results = bootstrap_metrics(
            tp, fp, tn, fn, n_samples, n_bootstrap, confidence_level, seed
        )
        
        # Create result row
        result_row = {
            'concept': concept,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'n_samples': n_samples
        }
        
        # Add metrics with confidence intervals
        for metric_name, (mean_val, lower_ci, upper_ci) in bootstrap_results.items():
            result_row[f'{metric_name}'] = mean_val
            result_row[f'{metric_name}_lower'] = lower_ci
            result_row[f'{metric_name}_upper'] = upper_ci
            result_row[f'{metric_name}_error'] = (upper_ci - lower_ci) / 2
        
        results.append(result_row)
    
    return pd.DataFrame(results)


def compute_dataset_level_bootstrap(
    detection_results: pd.DataFrame,
    gt_samples_per_concept: Dict[str, List[int]],
    activation_loader,
    percentile: float,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    aggregation: str = 'macro',
    seed: int = 42
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute two-level bootstrap for dataset-level metrics.
    Level 1: Resample concepts with replacement
    Level 2: Within each concept, resample examples with replacement
    
    Args:
        detection_results: DataFrame with per-concept results
        gt_samples_per_concept: Ground truth samples per concept
        activation_loader: Loader for accessing activations
        percentile: Detection threshold percentile
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level
        aggregation: 'macro', 'micro', or 'weighted' averaging
        seed: Random seed
        
    Returns:
        Dict mapping metric -> (mean, lower_ci, upper_ci)
    """
    np.random.seed(seed)
    
    concepts = list(detection_results['concept'].unique())
    n_concepts = len(concepts)
    
    # Calculate concept weights based on sample frequency
    concept_weights = {}
    total_gt_samples = 0
    for concept in concepts:
        n_gt_samples = len(gt_samples_per_concept.get(concept, []))
        concept_weights[concept] = n_gt_samples
        total_gt_samples += n_gt_samples
    
    # Normalize weights
    for concept in concepts:
        concept_weights[concept] = concept_weights[concept] / total_gt_samples if total_gt_samples > 0 else 1.0 / n_concepts
    
    # Store bootstrap results
    bootstrap_metrics = defaultdict(list)
    
    for boot_iter in tqdm(range(n_bootstrap), desc="Dataset-level bootstrap"):
        # Level 1: Resample concepts
        resampled_concepts = np.random.choice(concepts, size=n_concepts, replace=True)
        
        # Track overall confusion matrix for micro-averaging
        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
        concept_metrics = []
        
        for concept in resampled_concepts:
            # Get original data for this concept
            concept_row = detection_results[detection_results['concept'] == concept].iloc[0]
            
            # Total samples for this concept
            n_total = int(concept_row['tp'] + concept_row['fp'] + 
                        concept_row['tn'] + concept_row['fn'])
            
            if n_total == 0:
                continue
            
            # Level 2: Bootstrap the actual confusion matrix counts
            # Use the same approach as per-concept bootstrap
            tp_orig = int(concept_row['tp'])
            fp_orig = int(concept_row['fp']) 
            tn_orig = int(concept_row['tn'])
            fn_orig = int(concept_row['fn'])
            
            # Create outcome array like in bootstrap_metrics
            # 0=TP, 1=FP, 2=TN, 3=FN
            outcomes = np.array(
                [0] * tp_orig + 
                [1] * fp_orig + 
                [2] * tn_orig + 
                [3] * fn_orig,
                dtype=np.int8
            )
            
            # Resample with replacement
            resampled = np.random.choice(outcomes, size=n_total, replace=True)
            
            # Count outcomes
            counts = np.bincount(resampled, minlength=4)
            tp_boot = counts[0]
            fp_boot = counts[1]
            tn_boot = counts[2]
            fn_boot = counts[3]
            
            # Update totals for micro-averaging
            total_tp += tp_boot
            total_fp += fp_boot
            total_tn += tn_boot
            total_fn += fn_boot
            
            # Compute concept-level metrics for macro-averaging
            if aggregation == 'macro':
                precision = tp_boot / (tp_boot + fp_boot) if (tp_boot + fp_boot) > 0 else 0
                recall = tp_boot / (tp_boot + fn_boot) if (tp_boot + fn_boot) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                concept_metrics.append({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
        
        # Aggregate metrics
        if aggregation in ['macro', 'weighted'] and concept_metrics:
            if aggregation == 'macro':
                # Simple average across concepts
                for metric in ['precision', 'recall', 'f1']:
                    values = [m[metric] for m in concept_metrics]
                    bootstrap_metrics[f'macro_{metric}'].append(np.mean(values))
            else:  # weighted
                # Weighted average by concept frequency
                for metric in ['precision', 'recall', 'f1']:
                    weighted_sum = 0
                    total_weight = 0
                    for i, concept in enumerate(resampled_concepts):
                        if i < len(concept_metrics):
                            weighted_sum += concept_metrics[i][metric] * concept_weights.get(concept, 0)
                            total_weight += concept_weights.get(concept, 0)
                    weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0
                    bootstrap_metrics[f'weighted_{metric}'].append(weighted_avg)
        
        elif aggregation == 'micro':
            # Micro-average using totals
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            bootstrap_metrics['micro_precision'].append(precision)
            bootstrap_metrics['micro_recall'].append(recall)
            bootstrap_metrics['micro_f1'].append(f1)
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    results = {}
    for metric_name, values in bootstrap_metrics.items():
        values = np.array(values)
        mean_val = np.mean(values)
        lower_ci = np.percentile(values, lower_percentile)
        upper_ci = np.percentile(values, upper_percentile)
        results[metric_name] = (mean_val, lower_ci, upper_ci)
    
    return results


def format_metric_with_ci(mean: float, lower: float, upper: float, 
                         precision: int = 3) -> str:
    """Format metric with confidence interval."""
    return f"{mean:.{precision}f} [{lower:.{precision}f}, {upper:.{precision}f}]"


def generate_per_concept_report(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    top_n: Optional[int] = None
) -> str:
    """
    Generate a formatted report of per-concept metrics with confidence intervals.
    
    Args:
        results_df: DataFrame from compute_per_concept_bootstrap
        output_path: Optional path to save report
        top_n: Show only top N concepts by F1 score
        
    Returns:
        Formatted report string
    """
    # Sort by F1 score
    results_df = results_df.sort_values('f1', ascending=False)
    
    if top_n:
        results_df = results_df.head(top_n)
    
    # Create report
    report_lines = [
        "PER-CONCEPT DETECTION METRICS WITH CONFIDENCE INTERVALS",
        "=" * 80,
        ""
    ]
    
    # Header
    header = f"{'Concept':<30} {'F1':<25} {'Precision':<25} {'Recall':<25}"
    report_lines.append(header)
    report_lines.append("-" * 105)
    
    # Concept rows
    for _, row in results_df.iterrows():
        concept = str(row['concept'])[:30]  # Truncate long names
        
        f1_str = format_metric_with_ci(row['f1'], row['f1_lower'], row['f1_upper'])
        prec_str = format_metric_with_ci(row['precision'], row['precision_lower'], row['precision_upper'])
        rec_str = format_metric_with_ci(row['recall'], row['recall_lower'], row['recall_upper'])
        
        line = f"{concept:<30} {f1_str:<25} {prec_str:<25} {rec_str:<25}"
        report_lines.append(line)
    
    # Add confusion matrix summary
    report_lines.extend([
        "",
        "CONFUSION MATRIX SUMMARY",
        "-" * 50,
        f"{'Concept':<30} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}"
    ])
    
    for _, row in results_df.iterrows():
        concept = str(row['concept'])[:30]
        line = f"{concept:<30} {int(row['tp']):<8} {int(row['fp']):<8} {int(row['tn']):<8} {int(row['fn']):<8}"
        report_lines.append(line)
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


def generate_dataset_report(
    dataset_results: Dict[str, Tuple[float, float, float]],
    per_concept_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a formatted report of dataset-level metrics with confidence intervals.
    
    Args:
        dataset_results: Results from compute_dataset_level_bootstrap
        per_concept_df: Per-concept results for additional statistics
        output_path: Optional path to save report
        
    Returns:
        Formatted report string
    """
    report_lines = [
        "DATASET-LEVEL DETECTION METRICS WITH CONFIDENCE INTERVALS",
        "=" * 80,
        "",
        f"Total concepts evaluated: {len(per_concept_df)}",
        f"Total samples: {per_concept_df['n_samples'].sum()}",
        "",
        "AGGREGATED METRICS",
        "-" * 50
    ]
    
    # Group metrics by type for better organization
    macro_metrics = {k: v for k, v in dataset_results.items() if 'macro' in k}
    weighted_metrics = {k: v for k, v in dataset_results.items() if 'weighted' in k}
    micro_metrics = {k: v for k, v in dataset_results.items() if 'micro' in k}
    
    if macro_metrics:
        report_lines.extend(["", "Macro-averaged (unweighted):", "-" * 30])
        for metric_name, (mean_val, lower_ci, upper_ci) in sorted(macro_metrics.items()):
            formatted = format_metric_with_ci(mean_val, lower_ci, upper_ci)
            metric_display = metric_name.replace('macro_', '').replace('_', ' ').title()
            report_lines.append(f"  {metric_display:<18}: {formatted}")
    
    if weighted_metrics:
        report_lines.extend(["", "Weighted by concept frequency:", "-" * 30])
        for metric_name, (mean_val, lower_ci, upper_ci) in sorted(weighted_metrics.items()):
            formatted = format_metric_with_ci(mean_val, lower_ci, upper_ci)
            metric_display = metric_name.replace('weighted_', '').replace('_', ' ').title()
            report_lines.append(f"  {metric_display:<18}: {formatted}")
    
    if micro_metrics:
        report_lines.extend(["", "Micro-averaged (pooled):", "-" * 30])
        for metric_name, (mean_val, lower_ci, upper_ci) in sorted(micro_metrics.items()):
            formatted = format_metric_with_ci(mean_val, lower_ci, upper_ci)
            metric_display = metric_name.replace('micro_', '').replace('_', ' ').title()
            report_lines.append(f"  {metric_display:<18}: {formatted}")
    
    # Add per-concept statistics
    report_lines.extend([
        "",
        "PER-CONCEPT STATISTICS",
        "-" * 50,
        f"Mean F1 score      : {per_concept_df['f1'].mean():.3f} ± {per_concept_df['f1'].std():.3f}",
        f"Mean Precision     : {per_concept_df['precision'].mean():.3f} ± {per_concept_df['precision'].std():.3f}",
        f"Mean Recall        : {per_concept_df['recall'].mean():.3f} ± {per_concept_df['recall'].std():.3f}",
        "",
        f"Median F1 score    : {per_concept_df['f1'].median():.3f}",
        f"Min F1 score       : {per_concept_df['f1'].min():.3f} ({per_concept_df.loc[per_concept_df['f1'].idxmin(), 'concept']})",
        f"Max F1 score       : {per_concept_df['f1'].max():.3f} ({per_concept_df.loc[per_concept_df['f1'].idxmax(), 'concept']})"
    ])
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


def save_results_with_ci(
    per_concept_results: pd.DataFrame,
    dataset_results: Dict[str, Tuple[float, float, float]],
    dataset_name: str,
    con_label: str,
    percentile: float,
    output_dir: str = "Quant_Results_with_CI"
) -> None:
    """
    Save results with confidence intervals to disk.
    
    Args:
        per_concept_results: Per-concept results with CIs
        dataset_results: Dataset-level results with CIs
        dataset_name: Name of dataset
        con_label: Concept label
        percentile: Detection percentile used
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(f"{output_dir}/{dataset_name}", exist_ok=True)
    
    # Save per-concept results as CSV
    csv_path = f"{output_dir}/{dataset_name}/per_concept_ci_{percentile}_{con_label}.csv"
    per_concept_results.to_csv(csv_path, index=False)
    
    # Save dataset-level results as JSON
    json_path = f"{output_dir}/{dataset_name}/dataset_ci_{percentile}_{con_label}.json"
    
    # Convert to serializable format
    dataset_json = {}
    for metric, (mean_val, lower, upper) in dataset_results.items():
        dataset_json[metric] = {
            'mean': float(mean_val),
            'lower_ci': float(lower),
            'upper_ci': float(upper),
            'error': float((upper - lower) / 2)
        }
    
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    # Generate and save reports
    per_concept_report = generate_per_concept_report(
        per_concept_results,
        output_path=f"{output_dir}/{dataset_name}/per_concept_report_{percentile}_{con_label}.txt"
    )
    
    dataset_report = generate_dataset_report(
        dataset_results,
        per_concept_results,
        output_path=f"{output_dir}/{dataset_name}/dataset_report_{percentile}_{con_label}.txt"
    )
    
    print(f"Results saved to {output_dir}/{dataset_name}/")