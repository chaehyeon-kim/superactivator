#!/usr/bin/env python3
"""
Per-Concept PTM Optimization Script
===================================

This script finds the optimal percentthrumodel (PTM) for each individual concept
based on calibration F1 scores, then uses those PTMs to evaluate test performance.

The script:
1. Loads calibration detection results across all PTMs for each concept
2. Finds the PTM that achieves highest F1 for each concept on calibration data
3. Uses the optimal PTMs to load corresponding test results
4. Computes weighted F1 scores using concept-specific optimal PTMs
5. Saves the results for analysis
"""

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
from collections import defaultdict
from itertools import product
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.default_percentthrumodels import get_model_default_percentthrumodels
from utils.filter_datasets_utils import filter_concept_dict

# Default configuration
MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Gemma', ('text', 'text2')), ('Qwen', ('text', 'text3'))]
DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
SAMPLE_TYPES = [('patch', 1000), ('cls', 50)]
CONCEPT_TYPES = ['avg', 'linsep', 'kmeans', 'linsepkmeans']  # Include all concept types
DETECTION_METHODS = ['regular', 'maxtoken', 'meantoken', 'lasttoken', 'randomtoken']


def get_concept_label(model_name, concept_type, sample_type, n_clusters, ptm):
    """
    Construct concept label based on concept type.
    
    Args:
        model_name: Name of the model
        concept_type: Type of concept ('avg', 'linsep', 'kmeans', 'linsepkmeans')
        sample_type: 'patch' or 'cls'
        n_clusters: Number of clusters (used for kmeans)
        ptm: Percentthrumodel value
    
    Returns:
        Concept label string
    """
    if concept_type == 'avg':
        return f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{ptm}'
    elif concept_type == 'linsep':
        return f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{ptm}'
    elif concept_type == 'kmeans':
        return f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{ptm}'
    elif concept_type == 'linsepkmeans':
        return f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{ptm}'
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")


def load_detection_results_for_concept(dataset_name, con_label, percentile, split='test', load_ci=True):
    """
    Load detection results for a specific concept from a specific percentile.
    
    Args:
        dataset_name: Dataset name
        con_label: Concept label (includes PTM)
        percentile: Detection percentile
        split: 'test' or 'cal'
        load_ci: Whether to load confidence intervals from Quant_Results_with_CI
    
    Returns:
        dict with detection metrics for each concept
    """
    # For calibration split, append _cal to con_label
    if split == 'cal':
        con_label_with_split = f"{con_label}_cal"
    else:
        con_label_with_split = con_label
    
    # Try loading .pt file first
    pt_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label_with_split}.pt'
    
    if os.path.exists(pt_path):
        data = torch.load(pt_path, weights_only=False)
        # Convert DataFrame to dict if needed
        if hasattr(data, 'iterrows'):  # It's a DataFrame
            result = {}
            for _, row in data.iterrows():
                concept = row['concept']
                result[concept] = row.to_dict()
            data = result
    else:
        # Try CSV format
        csv_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label_with_split}.csv'
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Convert to dictionary format
            data = {}
            for _, row in df.iterrows():
                concept = row['concept']
                data[concept] = row.to_dict()
        else:
            return None
    
    # Load confidence intervals if requested and available
    if load_ci and split == 'test':  # CI only computed for test set
        # Try both naming patterns
        ci_paths = [
            f'Quant_Results_with_CI/{dataset_name}/per_concept_ci_{percentile}_{con_label}.csv',
            f'Quant_Results_with_CI/{dataset_name}/per_concept_ci_optimal_{con_label}.csv'
        ]
        
        ci_df = None
        for ci_path in ci_paths:
            if os.path.exists(ci_path):
                try:
                    ci_df = pd.read_csv(ci_path)
                    break
                except Exception:
                    continue
        
        if ci_df is not None:
            try:
                # Add CI information to each concept
                for _, row in ci_df.iterrows():
                    concept = row['concept']
                    if concept in data:
                        # Add error values directly from the error columns
                        for metric in ['f1', 'precision', 'recall', 'accuracy', 'tpr', 'fpr']:
                            error_col = f'{metric}_error'
                            if error_col in row:
                                # Use the error value directly as standard error
                                data[concept][f'{metric}_se'] = row[error_col]
            except Exception as e:
                print(f"Warning: Could not load CI results from {ci_path}: {e}")
    
    return data


def load_best_percentile_for_config(dataset_name, con_label, detection_method='regular'):
    """
    Load the best percentile for each concept from calibration results.
    
    Args:
        dataset_name: Dataset name
        con_label: Concept label (without _cal suffix)
        detection_method: Detection method (regular, maxtoken, etc.)
    
    Returns:
        dict mapping concept -> best_percentile
    """
    if detection_method == 'regular':
        # Regular detection uses Best_Detection_Percentiles_Cal directory
        best_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
    else:
        # Baseline methods store in Quant_Results
        best_file = f'Quant_Results/{dataset_name}/{detection_method}_best_percentiles_{con_label}.pt'
    
    try:
        if os.path.exists(best_file):
            data = torch.load(best_file, weights_only=False)
            
            # Check if it's a DataFrame (happens with some regular detection files)
            if hasattr(data, 'iterrows'):  # It's a DataFrame
                result = {}
                for _, row in data.iterrows():
                    concept = row['concept']
                    result[concept] = {
                        'best_percentile': row['best_percentile'],
                        'best_f1': row.get('best_f1', row.get('f1', 0))
                    }
                return {concept: info['best_percentile'] for concept, info in result.items()}
            elif isinstance(data, dict) and 'best_percentiles' in data:
                # Baseline format
                return data['best_percentiles']
            elif isinstance(data, dict):
                # Regular format - dict mapping concept to info including best_percentile
                return {concept: info['best_percentile'] for concept, info in data.items()}
    except Exception as e:
        print(f"Error loading {best_file}: {e}")
        return None
    
    return None


def find_best_ptm_per_concept(
    dataset_name, model_name, model_input_size, sample_type, 
    concept_type, detection_method, percentthrumodels, n_clusters=None
):
    """
    Find the best PTM for each concept based on calibration F1 scores.
    
    Returns:
        dict mapping concept -> {best_ptm, best_f1, best_percentile}
    """
    print(f"\nFinding best PTM per concept for {model_name} {concept_type} {sample_type} ({detection_method})")
    
    # Collect all calibration results across PTMs
    all_cal_results = defaultdict(lambda: defaultdict(dict))
    
    for ptm in percentthrumodels:
        # Get concept label for this PTM
        con_label = get_concept_label(model_name, concept_type, sample_type, n_clusters, ptm)
        
        # Load best percentiles for this configuration
        best_percentiles = load_best_percentile_for_config(dataset_name, con_label, detection_method)
        
        if best_percentiles is None:
            if detection_method != 'regular' or ptm == percentthrumodels[0]:
                # Only print warning once per method
                print(f"  ⚠️  No best percentiles found for PTM {ptm}")
            continue
        
        # For each concept, load its calibration results at its best percentile
        for concept, best_perc in best_percentiles.items():
            cal_results = load_detection_results_for_concept(dataset_name, con_label, best_perc, split='cal')
            
            if cal_results and concept in cal_results:
                concept_data = cal_results[concept]
                f1_score = concept_data.get('f1', 0)
                all_cal_results[concept][ptm] = {
                    'f1': f1_score,
                    'percentile': best_perc,
                    'precision': concept_data.get('precision', 0),
                    'recall': concept_data.get('recall', 0),
                    'tp': concept_data.get('tp', 0),
                    'fp': concept_data.get('fp', 0),
                    'tn': concept_data.get('tn', 0),
                    'fn': concept_data.get('fn', 0)
                }
    
    # Find best PTM for each concept
    best_ptm_per_concept = {}
    
    for concept, ptm_results in all_cal_results.items():
        if not ptm_results:
            continue
        
        # Find PTM with highest F1
        best_ptm = max(ptm_results.keys(), key=lambda ptm: ptm_results[ptm]['f1'])
        best_data = ptm_results[best_ptm]
        
        best_ptm_per_concept[concept] = {
            'best_ptm': best_ptm,
            'best_f1_cal': best_data['f1'],
            'best_percentile': best_data['percentile'],
            'best_precision_cal': best_data['precision'],
            'best_recall_cal': best_data['recall'],
            'all_ptm_results': ptm_results  # Keep all results for analysis
        }
    
    print(f"  Found best PTMs for {len(best_ptm_per_concept)} concepts")
    
    # Print summary statistics
    ptm_counts = defaultdict(int)
    for concept_info in best_ptm_per_concept.values():
        ptm_counts[concept_info['best_ptm']] += 1
    
    print("\n  PTM distribution:")
    for ptm, count in sorted(ptm_counts.items()):
        print(f"    PTM {ptm}: {count} concepts")
    
    return best_ptm_per_concept


def compute_test_results_with_optimal_ptms(
    dataset_name, model_name, model_input_size, sample_type,
    concept_type, detection_method, best_ptm_per_concept, n_clusters=None
):
    """
    Load test results using the optimal PTM for each concept and compute metrics.
    
    Returns:
        dict with per-concept test results and aggregated metrics
    """
    print(f"\nLoading test results with optimal PTMs")
    
    test_results_per_concept = {}
    
    # Load ground truth for computing weighted averages
    gt_images_per_concept_test = torch.load(
        f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt',
        weights_only=False
    )
    gt_images_per_concept_test = filter_concept_dict(gt_images_per_concept_test, dataset_name)
    
    # Process each concept with its optimal PTM
    for concept, ptm_info in best_ptm_per_concept.items():
        best_ptm = ptm_info['best_ptm']
        best_percentile = ptm_info['best_percentile']
        
        # Get concept label for this PTM
        con_label = get_concept_label(model_name, concept_type, sample_type, n_clusters, best_ptm)
        
        # Load test results
        test_results = load_detection_results_for_concept(dataset_name, con_label, best_percentile, split='test')
        
        if test_results and concept in test_results:
            concept_data = test_results[concept]
            result_entry = {
                'ptm': best_ptm,
                'percentile': best_percentile,
                'f1_test': concept_data.get('f1', 0),
                'precision_test': concept_data.get('precision', 0),
                'recall_test': concept_data.get('recall', 0),
                'tp': concept_data.get('tp', 0),
                'fp': concept_data.get('fp', 0),
                'tn': concept_data.get('tn', 0),
                'fn': concept_data.get('fn', 0),
                'f1_cal': ptm_info['best_f1_cal'],
                'n_gt_positive': len(gt_images_per_concept_test.get(concept, []))
            }
            
            # Add CI/SE information if available
            for metric in ['f1', 'precision', 'recall']:
                if f'{metric}_se' in concept_data:
                    result_entry[f'{metric}_se'] = concept_data[f'{metric}_se']
                if f'{metric}_ci_lower' in concept_data:
                    result_entry[f'{metric}_ci_lower'] = concept_data[f'{metric}_ci_lower']
                if f'{metric}_ci_upper' in concept_data:
                    result_entry[f'{metric}_ci_upper'] = concept_data[f'{metric}_ci_upper']
            
            test_results_per_concept[concept] = result_entry
    
    # Compute aggregate metrics
    if test_results_per_concept:
        # Weighted average (by number of positive samples)
        total_weight = sum(r['n_gt_positive'] for r in test_results_per_concept.values())
        
        if total_weight > 0:
            weighted_f1 = sum(r['f1_test'] * r['n_gt_positive'] for r in test_results_per_concept.values()) / total_weight
            weighted_precision = sum(r['precision_test'] * r['n_gt_positive'] for r in test_results_per_concept.values()) / total_weight
            weighted_recall = sum(r['recall_test'] * r['n_gt_positive'] for r in test_results_per_concept.values()) / total_weight
            
            # Compute weighted standard errors if available
            weighted_f1_se = 0
            weighted_precision_se = 0
            weighted_recall_se = 0
            
            for r in test_results_per_concept.values():
                weight = r['n_gt_positive'] / total_weight
                if 'f1_se' in r:
                    weighted_f1_se += (r['f1_se'] * weight) ** 2
                if 'precision_se' in r:
                    weighted_precision_se += (r['precision_se'] * weight) ** 2
                if 'recall_se' in r:
                    weighted_recall_se += (r['recall_se'] * weight) ** 2
            
            # Take square root to get pooled SE
            weighted_f1_se = np.sqrt(weighted_f1_se) if weighted_f1_se > 0 else 0
            weighted_precision_se = np.sqrt(weighted_precision_se) if weighted_precision_se > 0 else 0
            weighted_recall_se = np.sqrt(weighted_recall_se) if weighted_recall_se > 0 else 0
        else:
            weighted_f1 = weighted_precision = weighted_recall = 0
            weighted_f1_se = weighted_precision_se = weighted_recall_se = 0
        
        # Macro average (simple mean)
        macro_f1 = np.mean([r['f1_test'] for r in test_results_per_concept.values()])
        macro_precision = np.mean([r['precision_test'] for r in test_results_per_concept.values()])
        macro_recall = np.mean([r['recall_test'] for r in test_results_per_concept.values()])
        
        # Micro average (aggregate TP, FP, FN)
        total_tp = sum(r['tp'] for r in test_results_per_concept.values())
        total_fp = sum(r['fp'] for r in test_results_per_concept.values())
        total_fn = sum(r['fn'] for r in test_results_per_concept.values())
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    else:
        weighted_f1 = macro_f1 = micro_f1 = 0
        weighted_precision = macro_precision = micro_precision = 0
        weighted_recall = macro_recall = micro_recall = 0
        weighted_f1_se = weighted_precision_se = weighted_recall_se = 0
    
    aggregate_metrics = {
        'weighted_f1': weighted_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1_se': weighted_f1_se,
        'weighted_precision_se': weighted_precision_se,
        'weighted_recall_se': weighted_recall_se,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'micro_f1': micro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'n_concepts': len(test_results_per_concept)
    }
    
    print(f"  Loaded test results for {len(test_results_per_concept)} concepts")
    print(f"  Weighted F1: {weighted_f1:.3f}, Macro F1: {macro_f1:.3f}, Micro F1: {micro_f1:.3f}")
    
    return test_results_per_concept, aggregate_metrics


def save_optimization_results(
    dataset_name, model_name, sample_type, concept_type, detection_method,
    best_ptm_per_concept, test_results_per_concept, aggregate_metrics, n_clusters=None
):
    """
    Save the optimization results.
    """
    # Create output directory
    output_dir = f'Per_Concept_PTM_Optimization/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create configuration label
    if detection_method == 'regular':
        method_suffix = ''
    else:
        method_suffix = f'_{detection_method}'
    
    # Get label without PTM for saving
    if concept_type == 'avg':
        con_label = f'{model_name}_avg_{sample_type}_embeddings{method_suffix}'
    elif concept_type == 'linsep':
        con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False{method_suffix}'
    elif concept_type == 'kmeans':
        con_label = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans{method_suffix}'
    elif concept_type == 'linsepkmeans':
        con_label = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans{method_suffix}'
    
    # Save comprehensive results
    results = {
        'best_ptm_per_concept': best_ptm_per_concept,
        'test_results_per_concept': test_results_per_concept,
        'aggregate_metrics': aggregate_metrics,
        'metadata': {
            'dataset': dataset_name,
            'model': model_name,
            'sample_type': sample_type,
            'concept_type': concept_type,
            'detection_method': detection_method
        }
    }
    
    # Save as .pt file
    pt_path = os.path.join(output_dir, f'optimal_ptm_results_{con_label}.pt')
    torch.save(results, pt_path)
    
    # Also save as CSV for easy viewing
    # Create DataFrame with per-concept results
    rows = []
    for concept in best_ptm_per_concept:
        row = {
            'concept': concept,
            'best_ptm': best_ptm_per_concept[concept]['best_ptm'],
            'best_percentile': best_ptm_per_concept[concept]['best_percentile'],
            'f1_cal': best_ptm_per_concept[concept]['best_f1_cal']
        }
        
        if concept in test_results_per_concept:
            row.update({
                'f1_test': test_results_per_concept[concept]['f1_test'],
                'precision_test': test_results_per_concept[concept]['precision_test'],
                'recall_test': test_results_per_concept[concept]['recall_test'],
                'n_gt_positive': test_results_per_concept[concept]['n_gt_positive']
            })
            
            # Add SE columns if available
            if 'f1_se' in test_results_per_concept[concept]:
                row['f1_se'] = test_results_per_concept[concept]['f1_se']
            if 'precision_se' in test_results_per_concept[concept]:
                row['precision_se'] = test_results_per_concept[concept]['precision_se']
            if 'recall_se' in test_results_per_concept[concept]:
                row['recall_se'] = test_results_per_concept[concept]['recall_se']
        else:
            row.update({
                'f1_test': np.nan,
                'precision_test': np.nan,
                'recall_test': np.nan,
                'n_gt_positive': 0
            })
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('concept')
    csv_path = os.path.join(output_dir, f'optimal_ptm_results_{con_label}.csv')
    df.to_csv(csv_path, index=False)
    
    # Save aggregate metrics
    aggregate_df = pd.DataFrame([aggregate_metrics])
    aggregate_csv_path = os.path.join(output_dir, f'optimal_ptm_aggregate_{con_label}.csv')
    aggregate_df.to_csv(aggregate_csv_path, index=False)
    
    print(f"\n  Results saved to:")
    print(f"    {pt_path}")
    print(f"    {csv_path}")
    print(f"    {aggregate_csv_path}")


def process_configuration(
    dataset_name, model_name, model_input_size, sample_type,
    concept_type, detection_method, n_clusters=None
):
    """
    Process a single configuration to find optimal PTMs per concept.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {model_name} - {dataset_name} - {sample_type} - {concept_type} - {detection_method}")
    print(f"{'='*80}")
    
    # Get valid PTMs for this model
    percentthrumodels = get_model_default_percentthrumodels(model_name, model_input_size)
    
    # Find best PTM for each concept based on calibration
    best_ptm_per_concept = find_best_ptm_per_concept(
        dataset_name, model_name, model_input_size, sample_type,
        concept_type, detection_method, percentthrumodels, n_clusters
    )
    
    if not best_ptm_per_concept:
        print("  ⚠️  No concepts found with valid results")
        if detection_method != 'regular':
            print(f"      (baseline_detections.py may not have been run for {detection_method})")
        else:
            print(f"      (all_detection_stats.py may not have been run for this configuration)")
        return
    
    # Compute test results using optimal PTMs
    test_results_per_concept, aggregate_metrics = compute_test_results_with_optimal_ptms(
        dataset_name, model_name, model_input_size, sample_type,
        concept_type, detection_method, best_ptm_per_concept, n_clusters
    )
    
    # Save results
    save_optimization_results(
        dataset_name, model_name, sample_type, concept_type, detection_method,
        best_ptm_per_concept, test_results_per_concept, aggregate_metrics, n_clusters
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find optimal PTM for each concept')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process')
    parser.add_argument('--datasets', nargs='+', help='Multiple datasets to process')
    parser.add_argument('--model', type=str, help='Specific model to use')
    parser.add_argument('--models', nargs='+', help='Multiple models to use')
    parser.add_argument('--sample-type', type=str, choices=['patch', 'cls'], help='Sample type')
    parser.add_argument('--concept-types', nargs='+', choices=['avg', 'linsep', 'kmeans', 'linsepkmeans'], 
                        default=['avg', 'linsep', 'kmeans', 'linsepkmeans'], help='Concept types')
    parser.add_argument('--detection-methods', nargs='+', 
                        choices=['regular', 'maxtoken', 'meantoken', 'lasttoken', 'randomtoken'],
                        default=['regular', 'maxtoken', 'meantoken', 'lasttoken', 'randomtoken'], help='Detection methods to analyze')
    
    args = parser.parse_args()
    
    # Determine datasets to process
    if args.dataset:
        datasets_to_process = [args.dataset]
    elif args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = DATASETS
    
    # Determine models to process
    if args.models:
        models_to_process = []
        for model_name in args.models:
            found = [(m, s) for m, s in MODELS if m == model_name]
            if not found:
                print(f"Error: Model '{model_name}' not found")
                sys.exit(1)
            models_to_process.extend(found)
    elif args.model:
        models_to_process = [(m, s) for m, s in MODELS if m == args.model]
        if not models_to_process:
            print(f"Error: Model '{args.model}' not found")
            sys.exit(1)
    else:
        models_to_process = MODELS
    
    # Determine sample types
    if args.sample_type:
        sample_types_to_process = [(s, n) for s, n in SAMPLE_TYPES if s == args.sample_type]
    else:
        sample_types_to_process = SAMPLE_TYPES
    
    print(f"\nPER-CONCEPT PTM OPTIMIZATION")
    print(f"============================")
    print(f"Concept types: {args.concept_types}")
    print(f"Detection methods: {args.detection_methods}")
    print()
    
    # Process all configurations
    experiment_configs = product(
        models_to_process, datasets_to_process, sample_types_to_process,
        args.concept_types, args.detection_methods
    )
    
    for (model_name, model_input_size), dataset_name, (sample_type, n_clusters), concept_type, detection_method in experiment_configs:
        # Skip invalid combinations
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
        
        # Skip baseline methods for cls sample type
        if detection_method != 'regular' and sample_type == 'cls':
            continue
        
        try:
            # For kmeans concepts, we need to use the n_clusters value
            if concept_type in ['kmeans', 'linsepkmeans']:
                process_configuration(
                    dataset_name, model_name, model_input_size, sample_type,
                    concept_type, detection_method, n_clusters
                )
            else:
                process_configuration(
                    dataset_name, model_name, model_input_size, sample_type,
                    concept_type, detection_method, None
                )
        except Exception as e:
            import traceback
            print(f"\n  ❌ Error processing configuration: {e}")
            traceback.print_exc()
            continue
    
    print("\n✅ Per-concept PTM optimization complete!")
    print("Results saved to: Per_Concept_PTM_Optimization/")