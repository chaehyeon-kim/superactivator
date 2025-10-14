"""
Hybrid Concept Analysis Utilities

This module provides hybrid concept analysis functions that work with chunked 
embeddings for efficient processing of large datasets.

Key features:
- Chunked processing of large embedding files
- Efficient activation computation
- Threshold computation and evaluation
- Per-concept metrics extraction
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import json

from utils.compute_concepts_utils import compute_cosine_sims, compute_signed_distances
from utils.quant_concept_evals_utils import compute_concept_thresholds_over_percentiles, compute_detection_metrics_over_percentiles, compute_stats_from_counts
from utils.patch_alignment_utils import filter_patches_by_image_presence
from utils.memory_management_utils import ChunkedEmbeddingLoader, compute_activations_chunked, compute_hybrid_activations_chunked, memory_efficient_context

def load_superdetectors(dataset_name, con_label, percentile):
    """Load superdetector patches for given dataset, concept label, and percentile."""
    # Extract model name and method from con_label
    if '_avg' in con_label:
        model_name = con_label.split('_avg')[0]
        method = 'avg'
        embeddings_type = 'patch_embeddings'
    elif '_linsep' in con_label:
        model_name = con_label.split('_linsep')[0]
        method = 'linsep'
        embeddings_type = 'patch_embeddings_BD_True_BN_False'
    else:
        # Fallback for simple model names
        model_name = con_label
        method = 'avg'
        embeddings_type = 'patch_embeddings'
    
    superdetector_file = f'Superpatches/{dataset_name}/per_{percentile}_{model_name}_{method}_{embeddings_type}_percentthrumodel_100.pt'
    if not os.path.exists(superdetector_file):
        raise FileNotFoundError(f"Required superdetector file not found: {superdetector_file}")
    
    superdetectors = torch.load(superdetector_file, weights_only=False)
    return superdetectors

def load_concept_vectors(dataset_name, con_label):
    """Load concept vectors for given concept label."""
    # Extract model name and method from con_label 
    if '_avg' in con_label:
        model_name = con_label.split('_avg')[0]
        method = 'avg'
    elif '_linsep' in con_label:
        model_name = con_label.split('_linsep')[0]
        method = 'linsep'
    else:
        # Fallback for simple model names
        model_name = con_label
        method = 'avg'  # Default to avg
    
    # Load only the concepts for the specific method
    concepts = {}
    
    if method == 'avg':
        avg_concepts_file = f'avg_concepts_{model_name}_patch_embeddings_percentthrumodel_100.pt'
        avg_concepts_path = f'Concepts/{dataset_name}/{avg_concepts_file}'
        
        if os.path.exists(avg_concepts_path):
            avg_concepts = torch.load(avg_concepts_path)
            concepts.update({k + '_avg': v for k, v in avg_concepts.items()})
        else:
            raise FileNotFoundError(f"Average concept file not found: {avg_concepts_path}")
    
    elif method == 'linsep':
        linsep_concepts_file = f'linsep_concepts_BD_True_BN_False_{model_name}_patch_embeddings_percentthrumodel_100.pt'
        linsep_concepts_path = f'Concepts/{dataset_name}/{linsep_concepts_file}'
        
        if os.path.exists(linsep_concepts_path):
            linsep_concepts = torch.load(linsep_concepts_path)
            concepts.update({k + '_linsep': v for k, v in linsep_concepts.items()})
        else:
            raise FileNotFoundError(f"Linear separator concept file not found: {linsep_concepts_path}")
    
    return concepts

def get_sample_ranges(dataset_name, model_input_size, total_embeddings):
    """Get the patch ranges for each sample (image/paragraph)."""
    if model_input_size[0] == 'text':
        # For text, load token counts to determine sample boundaries
        token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
        
        if not os.path.exists(token_counts_file):
            raise FileNotFoundError(f"Token counts file not found: {token_counts_file}")
        
        token_counts = torch.load(token_counts_file, weights_only=False)
        sample_ranges = []
        start_idx = 0
        for sent_idx, word_token_counts in enumerate(token_counts):
            total_tokens = sum(word_token_counts)
            sample_ranges.append((start_idx, start_idx + total_tokens))
            start_idx += total_tokens
    else:
        # For images, calculate patches per image based on input size
        # model_input_size is (height, width), patches are 14x14
        img_height, img_width = model_input_size
        patch_size = 14
        
        patches_per_row = img_height // patch_size
        patches_per_col = img_width // patch_size
        patches_per_image = patches_per_row * patches_per_col
        
        num_images = total_embeddings // patches_per_image
        sample_ranges = []
        for img_idx in range(num_images):
            start_idx = img_idx * patches_per_image
            sample_ranges.append((start_idx, start_idx + patches_per_image))
    
    return sample_ranges

def filter_relevant_patches(indices, dataset_name, model_input_size):
    """
    Filter out padding patches/tokens using the relevant patches mask.
    
    Args:
        indices: List of patch/token indices to filter
        dataset_name: Name of dataset
        model_input_size: Model input size (for images) or 'text' identifier
    
    Returns:
        Filtered list of indices containing only real content (no padding)
    """
    if model_input_size[0] == 'text':
        # For text, load the relevant tokens mask
        mask_file = f'GT_Samples/{dataset_name}/patches_w_image_mask_inputsize_{model_input_size}.pt'
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Required relevant tokens mask file not found: {mask_file}")
        
        relevant_tokens = torch.load(mask_file, weights_only=False)
        # Filter indices based on relevant tokens mask
        filtered_indices = [idx for idx in indices if idx < len(relevant_tokens) and relevant_tokens[idx] == 1]
        return filtered_indices
    else:
        # For images, use the existing filter function
        return filter_patches_by_image_presence(indices, dataset_name, model_input_size).tolist()

def compute_hybrid_concept_vectors(concept_vectors, superdetectors, embeddings_path, sample_ranges, alpha, dataset_name, model_input_size, device='cuda'):
    """
    Computation of hybrid concept vectors using chunked embeddings.
    
    Args:
        concept_vectors: Dictionary mapping concept names to concept vectors
        superdetectors: Dictionary mapping concept names to superdetector indices
        embeddings_path: Path to embeddings file (chunked or not)
        sample_ranges: List of (start_idx, end_idx) tuples for each sample
        alpha: Alpha value for hybrid combination
        dataset_name: Name of dataset
        model_input_size: Model input size
        device: Device for computation
        
    Returns:
        Dictionary mapping concept names to lists of hybrid concept vectors (one per sample)
    """
    print(f"   🔧 Computing hybrid concept vectors (alpha={alpha:.2f})...")
    
    hybrid_concepts = {}
    loader = ChunkedEmbeddingLoader(embeddings_path, device)
    
    # For each concept, we need to compute sample-specific hybrid vectors
    for concept, concept_vector in concept_vectors.items():
        print(f"      📝 Processing concept: {concept}")
        
        # Strip the _avg or _linsep suffix to get base concept name for superdetector lookup
        base_concept = concept.replace('_avg', '').replace('_linsep', '')
        
        if base_concept not in superdetectors:
            # If no superdetectors for this concept, use pure concept vector for all samples
            hybrid_concepts[concept] = [concept_vector] * len(sample_ranges)
            continue
            
        concept_superdetectors = superdetectors[base_concept]
        sample_hybrid_vectors = []
        
        # Group superdetectors by sample for efficient processing
        superdetectors_by_sample = {}
        for sample_idx, (start_idx, end_idx) in enumerate(sample_ranges):
            sample_superdetector_indices = [
                idx for idx in concept_superdetectors 
                if start_idx <= idx < end_idx
            ]
            if sample_superdetector_indices:
                superdetectors_by_sample[sample_idx] = sample_superdetector_indices
        
        if not superdetectors_by_sample:
            # No superdetectors for any sample - use pure concept vector
            hybrid_concepts[concept] = [concept_vector] * len(sample_ranges)
            continue
        
        # For samples with superdetectors, we need to load those specific embeddings
        all_superdetector_indices = []
        for indices in superdetectors_by_sample.values():
            all_superdetector_indices.extend(indices)
        
        # Load only the superdetector embeddings we need
        if all_superdetector_indices:
            superdetector_embeddings = loader.load_specific_embeddings(all_superdetector_indices)
            
            # Create mapping from global index to embedding
            idx_to_embedding = {idx: superdetector_embeddings[i] 
                              for i, idx in enumerate(all_superdetector_indices)}
        else:
            idx_to_embedding = {}
        
        # Compute hybrid vector for each sample
        for sample_idx, (start_idx, end_idx) in enumerate(sample_ranges):
            if sample_idx in superdetectors_by_sample:
                sample_superdetector_indices = superdetectors_by_sample[sample_idx]
                
                # Filter out padding patches/tokens
                filtered_superdetector_indices = filter_relevant_patches(
                    sample_superdetector_indices, dataset_name, model_input_size
                )
                
                if len(filtered_superdetector_indices) > 0:
                    # Get embeddings for this sample's superdetectors
                    sample_superdetector_embeds = torch.stack([
                        idx_to_embedding[idx] for idx in filtered_superdetector_indices
                    ])
                    
                    # Average superdetector embeddings from this sample
                    avg_superdetector = torch.mean(sample_superdetector_embeds, dim=0)
                    
                    # Compute hybrid vector for this sample
                    hybrid_vector = alpha * concept_vector + (1 - alpha) * avg_superdetector
                else:
                    # No valid superdetectors in this sample (all were padding)
                    hybrid_vector = concept_vector
            else:
                # No superdetectors in this sample, use pure concept vector
                hybrid_vector = concept_vector
            
            sample_hybrid_vectors.append(hybrid_vector)
        
        hybrid_concepts[concept] = sample_hybrid_vectors
        
        # Clean up
        if 'superdetector_embeddings' in locals():
            del superdetector_embeddings
        del idx_to_embedding
        torch.cuda.empty_cache() if device.startswith('cuda') else None
    
    return hybrid_concepts

def compute_hybrid_thresholds_for_concepts(activations, gt_samples_per_concept_cal, percentile, dataset_name, con_label, alpha):
    """
    Compute thresholds for hybrid concepts using calibration set at given percentile.
    Works at patch level for inversion F1.
    
    Args:
        activations: Dictionary mapping concept names to activation tensors (patch-level)
        gt_samples_per_concept_cal: GT samples for calibration set (patch indices)
        percentile: Percentile to use for threshold finding
        dataset_name: Name of dataset
        con_label: Concept label for saving
        alpha: Alpha value for hybrid concepts
    
    Returns:
        Dictionary mapping concept names to thresholds
    """
    thresholds = {}
    
    for concept, concept_activations in activations.items():
        # Strip suffix to get base concept name for GT lookup
        base_concept = concept.replace('_avg', '').replace('_linsep', '')
        
        if base_concept not in gt_samples_per_concept_cal:
            continue
            
        cal_gt_patch_indices = gt_samples_per_concept_cal[base_concept]
        
        if len(cal_gt_patch_indices) == 0:
            raise ValueError(f"No calibration GT patches found for concept '{concept}'. Cannot compute threshold.")
        
        # Get activations for calibration GT patches
        cal_gt_activations = concept_activations[cal_gt_patch_indices]
        
        # Find threshold that captures the desired percentile of GT activations
        # NOTE: Use (1 - percentile) to match baseline threshold computation
        threshold = torch.quantile(cal_gt_activations, 1 - percentile).item()
        
        thresholds[concept] = threshold
    
    # Save thresholds in Hybrid_Results
    os.makedirs(f'Hybrid_Results/{dataset_name}', exist_ok=True)
    threshold_file = f'Hybrid_Results/{dataset_name}/concept_thresholds_hybrid_{con_label}_alpha_{alpha:.2f}_percentile_{percentile}.pt'
    torch.save({percentile: thresholds}, threshold_file)
    
    return thresholds

def evaluate_hybrid_performance_with_thresholds(activations, thresholds, gt_patches_per_concept, gt_samples_per_concept_test, percentile, dataset_name, con_label, alpha, model_input_size, total_embeddings):
    """
    Evaluate F1 performance using same patch filtering and test set as baseline.
    
    Args:
        activations: Dictionary mapping concept names to activation tensors (patch-level)
        thresholds: Dictionary mapping concept names to threshold values
        gt_samples_per_concept_test: GT samples for test set (patch indices)
        percentile: Percentile used (invert percentile)
        dataset_name: Name of dataset
        con_label: Concept label for saving
        alpha: Alpha value for hybrid concepts
        model_input_size: Model input size tuple
    
    Returns:
        Dictionary containing all per-concept metrics plus summary statistics
    """
    from utils.patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence
    from utils.general_utils import create_binary_labels
    
    # Get test patches with same filtering as baseline
    # Use the passed model_input_size directly
    patch_size = 14
    split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
    test_indices = torch.tensor(split_df.index[split_df == 'test'].tolist())
    
    # Apply relevance filtering to test indices (no bounds filtering - that's cheating!)
    relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size)
    
    # Get ground truth labels for all patches (same as baseline)
    # For chunked processing, we need to handle GT indices differently
    # The GT labels contain global indices, but we need to work with the total embedding count
    
    all_concept_labels = create_binary_labels(total_embeddings, gt_patches_per_concept)
    
    # Collect counts for all concepts
    tp_counts = {}
    fp_counts = {}
    fn_counts = {}
    tn_counts = {}
    thresholds_dict = {}
    additional_info = {}
    
    for concept, concept_activations in activations.items():
        # Strip suffix to get base concept name for GT lookup
        base_concept = concept.replace('_avg', '').replace('_linsep', '')
        
        if concept not in thresholds or base_concept not in gt_patches_per_concept:
            continue
            
        threshold = thresholds[concept]
        
        # Apply threshold to get predictions (same threshold for detect/invert)
        activated_patches = concept_activations > threshold
        
        # Get relevant indices as boolean mask (same filtering as baseline)
        relevant_mask = torch.zeros(len(concept_activations), dtype=torch.bool, device=concept_activations.device)
        relevant_mask[relevant_indices] = True
        
        # Get GT mask (same GT processing as baseline) - use base concept name
        gt_values = all_concept_labels[base_concept] == 1
        if isinstance(gt_values, torch.Tensor):
            gt_mask = gt_values.clone().detach().to(concept_activations.device, dtype=torch.bool)
        else:
            gt_mask = torch.tensor(gt_values, dtype=torch.bool, device=concept_activations.device)
        
        # Compute confusion matrix counts only on relevant test patches (same subset as baseline)
        relevant_activated = activated_patches & relevant_mask
        relevant_gt = gt_mask & relevant_mask
        
        tp = torch.sum(relevant_activated & relevant_gt).item()
        fp = torch.sum(relevant_activated & (~relevant_gt)).item()
        fn = torch.sum((~relevant_activated) & relevant_gt).item()
        tn = torch.sum((~relevant_activated) & (~relevant_gt)).item()
        
        # Store counts in dictionaries
        tp_counts[concept] = tp
        fp_counts[concept] = fp
        fn_counts[concept] = fn
        tn_counts[concept] = tn
        thresholds_dict[concept] = threshold
        additional_info[concept] = {
            'num_gt': torch.sum(relevant_gt).item(),
            'num_pred': torch.sum(relevant_activated).item(),
            'total_patches': torch.sum(relevant_mask).item()
        }
    
    # Use existing function to compute stats from counts
    stats_df = compute_stats_from_counts(tp_counts, fp_counts, tn_counts, fn_counts)
    
    # Convert DataFrame back to dictionary format and add additional info
    all_metrics = {}
    for _, row in stats_df.iterrows():
        concept = row['concept']
        stats = row.to_dict()
        # Add threshold and additional information
        stats['threshold'] = thresholds_dict[concept]
        stats.update(additional_info[concept])
        all_metrics[concept] = stats
    
    # Compute weighted average F1 (weighted by number of test GT samples)
    f1_scores = {concept: metrics['f1'] for concept, metrics in all_metrics.items()}
    
    # Map concept names to their base versions for GT lookup
    total_samples = 0
    weighted_sum = 0.0
    
    for concept, f1_score in f1_scores.items():
        # Strip suffix to get base concept name for GT lookup
        base_concept = concept.replace('_avg', '').replace('_linsep', '')
        
        if base_concept in gt_samples_per_concept_test:
            concept_test_samples = len(gt_samples_per_concept_test[base_concept])
            total_samples += concept_test_samples
            weighted_sum += f1_score * concept_test_samples
    
    if total_samples == 0:
        weighted_f1 = 0.0
    else:
        weighted_f1 = weighted_sum / total_samples
    
    # Add weighted F1 to metrics
    all_metrics['weighted_f1'] = weighted_f1
    all_metrics['total_test_samples'] = total_samples
    
    # Save metrics in Hybrid_Results
    os.makedirs(f'Hybrid_Results/{dataset_name}', exist_ok=True)
    metrics_file = f'Hybrid_Results/{dataset_name}/detection_metrics_hybrid_{con_label}_alpha_{alpha:.2f}_percentile_{percentile}.pt'
    torch.save({percentile: all_metrics}, metrics_file)
    
    return all_metrics

def get_concept_label(model_name, method):
    """Get the concept label for supervised methods."""
    if method == 'avg':
        return f'{model_name}_avg_patch_embeddings_percentthrumodel_100'
    elif method == 'linsep':
        return f'{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_100'

def run_hybrid_analysis_for_config(model_name, model_input_size, dataset_name, method, detect_percentile, invert_percentile, alpha_values, device='cuda', scratch_dir=''):
    """
    Run hybrid concept analysis for a single configuration using chunked processing.
    
    Args:
        detect_percentile: Percentile used to find superpatches
        invert_percentile: Percentile used for threshold computation and evaluation
    
    Returns:
        List of results for each alpha value, or None if failed
    """
    print(f"   📊 Processing {method} with detect: {detect_percentile}, invert: {invert_percentile}")
    
    try:
        # Set up embeddings path
        embeddings_file = f"{model_name}_patch_embeddings_percentthrumodel_100.pt"
        embeddings_path = f"{scratch_dir}Embeddings/{dataset_name}/{embeddings_file}"
        
        # Get embedding info
        loader = ChunkedEmbeddingLoader(embeddings_path, device)
        embedding_info = loader.get_embedding_info()
        total_embeddings = embedding_info['total_samples']
        
        print(f"   📏 Total embeddings: {total_embeddings:,}")
        
        # Get sample ranges
        sample_ranges = get_sample_ranges(dataset_name, model_input_size, total_embeddings)
        print(f"   📋 Number of samples: {len(sample_ranges)}")
        
        # Load GT data like baseline does
        gt_patches_per_concept = torch.load(
            f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt',
            weights_only=False
        )
        gt_samples_per_concept_cal = torch.load(
            f'GT_Samples/{dataset_name}/gt_patch_per_concept_cal_inputsize_{model_input_size}.pt',
            weights_only=False
        )
        gt_samples_per_concept_test = torch.load(
            f'GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt',
            weights_only=False
        )
        
        # Load concept vectors and superdetectors
        con_label = get_concept_label(model_name, method)
        concept_vectors = load_concept_vectors(dataset_name, con_label)
        superdetectors = load_superdetectors(dataset_name, con_label, detect_percentile)
        
        # Move concept vectors to device
        concept_vectors = {k: v.to(device) for k, v in concept_vectors.items()}
        
        print(f"   🎯 Loaded {len(concept_vectors)} concept vectors")
        print(f"   🔍 Loaded superdetectors for {len(superdetectors)} concepts")
        
        # Sweep alpha values
        alpha_results = []
        
        for alpha in tqdm(alpha_values, desc=f"   Alpha sweep for {method}", leave=False):
            with memory_efficient_context(device):
                # 1. Compute hybrid concept vectors using chunked processing
                hybrid_concepts = compute_hybrid_concept_vectors(
                    concept_vectors, superdetectors, embeddings_path, sample_ranges, alpha, dataset_name, model_input_size, device
                )
       
                # 2. Compute activations using per-sample hybrid vectors and chunked processing
                print(f"   🧮 Computing hybrid activations (alpha={alpha:.2f})...")
                hybrid_activations = compute_hybrid_activations_chunked(
                    embeddings_path, hybrid_concepts, sample_ranges, device, method, show_progress=False
                )
                
                # 3. Compute thresholds using calibration set
                print(f"   🎚️  Computing thresholds...")
                thresholds = compute_hybrid_thresholds_for_concepts(
                    hybrid_activations, gt_samples_per_concept_cal, invert_percentile, dataset_name, con_label, alpha
                )
                
                # 4. Evaluate performance on test set
                print(f"   📊 Evaluating performance...")
                all_metrics = evaluate_hybrid_performance_with_thresholds(
                    hybrid_activations, thresholds, gt_patches_per_concept, gt_samples_per_concept_test, invert_percentile, dataset_name, con_label, alpha, model_input_size, total_embeddings
                )
                
                # Extract per-concept F1 scores and other metrics
                per_concept_metrics = {}
                for concept, metrics in all_metrics.items():
                    if isinstance(metrics, dict) and 'f1' in metrics:
                        # Store all the important metrics for this concept
                        per_concept_metrics[concept] = {
                            'f1': float(metrics['f1']),
                            'precision': float(metrics['precision']),
                            'recall': float(metrics['recall']),
                            'accuracy': float(metrics['accuracy']),
                            'threshold': float(metrics['threshold']),
                            'num_gt': int(metrics['num_gt']),
                            'num_pred': int(metrics['num_pred']),
                            'total_patches': int(metrics['total_patches'])
                        }
                
                # Get weighted F1 and total samples from summary metrics
                weighted_f1 = all_metrics.get('weighted_f1', 0.0)
                total_test_samples = all_metrics.get('total_test_samples', 0)
                
                alpha_results.append({
                    'alpha': float(alpha),
                    'weighted_f1': float(weighted_f1),
                    'total_test_samples': int(total_test_samples),
                    'per_concept_metrics': per_concept_metrics
                })
                
                print(f"   ✅ Alpha {alpha:.2f}: F1 = {weighted_f1:.4f} ({len(per_concept_metrics)} concepts)")
                
                # Clear variables to free memory
                del hybrid_concepts, hybrid_activations, thresholds
        
        return alpha_results
        
    except Exception as e:
        print(f"   ❌ Error in hybrid analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_metric_over_alphas(dataset_name, model_name, method, metric='weighted_f1', save_path=None, show_plot=True):
    """
    Plot a given metric over alpha values for memory-efficient results.
    
    Args:
        dataset_name: Name of dataset (e.g., 'CLEVR', 'Coco')
        model_name: Name of model (e.g., 'CLIP', 'Llama') 
        method: Method type ('avg' or 'linsep')
        metric: Metric to plot ('weighted_f1', 'precision', 'recall', 'accuracy', etc.)
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    # Load results from hybrid analysis file
    results_file = f'Hybrid_Results/{dataset_name}/hybrid_analysis_{model_name}_{method}.json'
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Hybrid analysis results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        config_results = json.load(f)
    alpha_sweep = config_results['alpha_sweep']
    
    from utils.filter_datasets_utils import filter_concept_dict
    
    # Extract alpha values and metric values
    alphas = [result['alpha'] for result in alpha_sweep]
    metric_values = []
    
    for alpha_result in alpha_sweep:
        per_concept_metrics = alpha_result.get('per_concept_metrics', {})
        
        # Filter concepts to only include relevant ones for this dataset
        filtered_metrics = filter_concept_dict(per_concept_metrics, dataset_name)
        
        if metric == 'weighted_f1':
            # Compute weighted F1 from per-concept metrics
            if filtered_metrics:
                total_weight = 0
                weighted_sum = 0.0
                
                for concept, concept_metrics in filtered_metrics.items():
                    if isinstance(concept_metrics, dict) and 'f1' in concept_metrics:
                        f1_score = concept_metrics['f1']
                        weight = concept_metrics.get('num_gt', 1)  # Use num_gt as weight
                        total_weight += weight
                        weighted_sum += f1_score * weight
                
                if total_weight > 0:
                    metric_values.append(weighted_sum / total_weight)
                else:
                    metric_values.append(0.0)
            else:
                metric_values.append(0.0)
        else:
            # For other metrics, compute weighted average across filtered concepts
            if filtered_metrics:
                total_weight = 0
                weighted_sum = 0.0
                
                for concept, concept_metrics in filtered_metrics.items():
                    if isinstance(concept_metrics, dict) and metric in concept_metrics:
                        metric_value = concept_metrics[metric]
                        weight = concept_metrics.get('num_gt', 1)  # Use num_gt as weight
                        total_weight += weight
                        weighted_sum += metric_value * weight
                
                if total_weight > 0:
                    metric_values.append(weighted_sum / total_weight)
                else:
                    metric_values.append(0.0)
            else:
                metric_values.append(0.0)
    
    # Create the plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(alphas, metric_values, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Alpha\n(α=0: Pure Superdetector, α=1: Pure Concept Vector)', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{metric.replace("_", " ").title()} vs Alpha\n{model_name} on {dataset_name} ({method})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Highlight best alpha
    best_idx = np.argmax(metric_values)
    best_alpha = alphas[best_idx]
    best_value = metric_values[best_idx]
    
    ax.scatter(best_alpha, best_value, color='red', s=120, zorder=5, edgecolors='darkred', linewidth=2)
    
    # Add vertical lines at pure endpoints
    ax.axvline(x=0, color='blue', linestyle='--', alpha=0.7, linewidth=1, label='Pure Superdetector')
    ax.axvline(x=1, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Pure Concept Vector')
    
    # Add best result annotation
    ax.text(best_alpha, best_value + 0.05 * (max(metric_values) - min(metric_values)), 
            f'Best: α={best_alpha:.2f}\n{metric}={best_value:.4f}', 
            fontsize=9, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax.set_xlim(-0.05, 1.05)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()