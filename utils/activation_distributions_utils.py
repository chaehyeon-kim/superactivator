"""
Utility functions for analyzing activation distributions of patches/tokens
that don't contain specific concepts.
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import gc
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from scipy.stats import norm, kstest, gaussian_kde
import warnings
import pandas as pd
import seaborn as sns

from utils import repo_path
from utils.patch_alignment_utils import filter_patches_by_image_presence
from utils.memory_management_utils import convert_image_indices_to_patch_indices, map_global_to_split_local, ChunkedActivationLoader
from utils.filter_datasets_utils import filter_concept_dict
from utils.general_utils import retrieve_image, get_split_index_from_global_index, pad_or_resize_img, apply_paper_plotting_style, get_paper_plotting_style
from utils.default_percentthrumodels import get_model_default_percentthrumodels
from utils.embedding_utils import percent_to_layer


def get_model_total_layers(model_name: str) -> int:
    """Get the total number of layers for a given model."""
    model_layers = {
        "CLIP": 24,  # Standard CLIP ViT-L/14
        "Llama": 32,  # For text, 40 for vision but we'll use 32 as default
        "Gemma": 28,  # Gemma text model layers
        "Qwen": 32,  # Qwen text model layers
    }
    return model_layers.get(model_name, 32)  # Default to 32 if unknown


def compute_distribution_metrics(data: np.ndarray, use_gpu: bool = False) -> Dict[str, float]:
    """
    Compute various metrics that quantify the variation/spread of a distribution.
    
    Args:
        data: 1D array of values
        use_gpu: Whether to use GPU for computation (if available)
        
    Returns:
        Dictionary containing various distribution metrics:
        - std: Standard deviation
        - iqr: Interquartile range (75th - 25th percentile)
        - cv: Coefficient of variation (std/mean)
        - mad: Median absolute deviation
        - range: Max - Min
        - entropy: Shannon entropy (discretized)
        - gini: Gini coefficient (inequality measure)
    """
    if len(data) == 0:
        return {
            'std': 0.0,
            'iqr': 0.0,
            'cv': 0.0,
            'mad': 0.0,
            'range': 0.0,
            'entropy': 0.0,
            'gini': 0.0
        }
    
    metrics = {}
    
    # Use GPU if requested and available
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        data_tensor = torch.from_numpy(data).to(device)
        
        # Standard deviation
        metrics['std'] = torch.std(data_tensor).item()
        
        # Percentiles for IQR
        q75 = torch.quantile(data_tensor, 0.75).item()
        q25 = torch.quantile(data_tensor, 0.25).item()
        metrics['iqr'] = q75 - q25
        
        # Mean and CV
        mean_val = torch.mean(data_tensor).item()
        metrics['cv'] = metrics['std'] / abs(mean_val) if mean_val != 0 else float('inf')
        
        # Median absolute deviation
        median = torch.median(data_tensor).values.item()
        mad_tensor = torch.median(torch.abs(data_tensor - median)).values
        metrics['mad'] = mad_tensor.item()
        
        # Range
        metrics['range'] = (torch.max(data_tensor) - torch.min(data_tensor)).item()
        
        # Shannon entropy (using histogram)
        hist = torch.histc(data_tensor, bins=20)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        metrics['entropy'] = -(hist * torch.log(hist)).sum().item()
        
        # Gini coefficient
        sorted_data = torch.sort(data_tensor).values
        n = len(sorted_data)
        index = torch.arange(1, n + 1, device=device, dtype=torch.float32)
        gini = (2 * (index * sorted_data).sum() / (n * sorted_data.sum()) - (n + 1) / n).item()
        metrics['gini'] = gini
    else:
        # CPU version (original code)
        # Standard deviation
        metrics['std'] = np.std(data)
        
        # Interquartile range
        q75, q25 = np.percentile(data, [75, 25])
        metrics['iqr'] = q75 - q25
        
        # Coefficient of variation (normalized by mean)
        mean_val = np.mean(data)
        metrics['cv'] = metrics['std'] / abs(mean_val) if mean_val != 0 else float('inf')
        
        # Median absolute deviation
        median = np.median(data)
        metrics['mad'] = np.median(np.abs(data - median))
        
        # Range
        metrics['range'] = np.max(data) - np.min(data)
        
        # Shannon entropy (using histogram)
        hist, _ = np.histogram(data, bins=20)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        metrics['entropy'] = -np.sum(hist * np.log(hist))
        
        # Gini coefficient (measure of inequality)
        sorted_data = np.sort(data)
        n = len(sorted_data)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_data)) / (n * np.sum(sorted_data)) - (n + 1) / n
        metrics['gini'] = gini
    
    return metrics


def get_samples_without_concept(
    gt_samples_per_concept: Dict[str, List[int]], 
    concept: str,
    all_samples: set
) -> torch.Tensor:
    """
    Get indices of samples that DON'T contain a specific concept.
    
    Args:
        gt_samples_per_concept: Dictionary mapping concepts to sample indices
        concept: The concept to exclude
        all_samples: Set of all sample indices to consider
        
    Returns:
        Tensor of sample indices that don't contain the concept
    """
    # Get samples with the concept
    samples_with_concept = set(gt_samples_per_concept.get(concept, []))
    
    # Get samples without the concept
    samples_without_concept = all_samples - samples_with_concept
    
    return torch.tensor(sorted(list(samples_without_concept)))


def get_patches_from_samples(
    sample_indices: torch.Tensor,
    patches_per_sample: int,
    device: torch.device
) -> torch.Tensor:
    """
    Convert sample indices to patch indices.
    
    Args:
        sample_indices: Indices of samples
        patches_per_sample: Number of patches per sample
        device: Device to put tensors on
        
    Returns:
        Tensor of patch indices
    """
    sample_indices = sample_indices.to(device)
    patch_indices = []
    
    for sample_idx in sample_indices:
        start_idx = sample_idx * patches_per_sample
        end_idx = start_idx + patches_per_sample
        patch_indices.extend(range(start_idx, end_idx))
    
    return torch.tensor(patch_indices, dtype=torch.long, device=device)


def compute_activation_distribution_per_sample(
    activations: torch.Tensor,
    sample_indices: torch.Tensor,
    patches_per_sample: int,
    num_bins: int = 100,
    activation_range: Tuple[float, float] = (-1.0, 1.0),
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Compute activation distribution for each sample.
    
    Args:
        activations: Patch-level activations [num_patches]
        sample_indices: Indices of samples to analyze
        patches_per_sample: Number of patches per sample
        num_bins: Number of bins for the histogram
        activation_range: Range of activations (min, max)
        device: Device to use for computation
        
    Returns:
        Distribution tensor [num_samples, num_bins]
    """
    # Move to device for faster computation
    activations = activations.to(device)
    sample_indices = sample_indices.to(device)
    
    num_samples = len(sample_indices)
    distributions = torch.zeros(num_samples, num_bins, device=device)
    
    # Create bin edges
    bin_edges = torch.linspace(activation_range[0], activation_range[1], num_bins + 1, device=device)
    
    for i, sample_idx in enumerate(sample_indices):
        # Get patch indices for this sample
        start_idx = sample_idx * patches_per_sample
        end_idx = start_idx + patches_per_sample
        
        # Get activations for this sample's patches
        sample_activations = activations[start_idx:end_idx]
        
        # Compute histogram
        hist = torch.histc(sample_activations, bins=num_bins, 
                          min=activation_range[0], max=activation_range[1])
        
        # Normalize to get distribution
        if hist.sum() > 0:
            distributions[i] = hist / hist.sum()
    
    return distributions


def get_activation_distributions_for_concept_samples(
    act_loader,
    gt_samples_per_concept: Dict[str, List[int]],
    dataset_name: str,
    model_input_size: Tuple,
    device: torch.device,
    sample_type: str = 'patch',
    num_bins: int = 100,
    activation_range: Tuple[float, float] = None,
    contains_concept: bool = False
) -> Dict[str, Dict]:
    """
    Get activation distributions for samples that do or don't contain each concept.
    
    Args:
        act_loader: ChunkedActivationLoader instance
        gt_samples_per_concept: Dictionary mapping concepts to sample indices
        dataset_name: Name of the dataset
        model_input_size: Model input size
        device: Device to use
        sample_type: 'patch' or 'cls'
        num_bins: Number of bins for the histogram
        activation_range: Range of activations
        contains_concept: If True, analyze samples WITH the concept; if False, WITHOUT
        
    Returns:
        Dictionary mapping concepts to their activation distribution info
    """
    results = {}
    
    # Get activation info
    act_info = act_loader.get_activation_info()
    total_patches = act_info['total_samples']  # This is actually total patches
    num_concepts = act_info['num_concepts']
    
    # Calculate patches per sample and total image samples
    if dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank']:
        # For text datasets, we need to load token counts
        token_file = f"./Data/{dataset_name}/Embeddings/num_tokens_per_sentence_test.pt"
        if os.path.exists(token_file):
            tokens_per_sample = torch.load(token_file)
            total_samples = len(tokens_per_sample)  # Number of text samples
        else:
            raise ValueError(f"Token count file not found: {token_file}")
    else:
        # For vision datasets, fixed patches per image
        if model_input_size == (224, 224):
            patches_per_sample = 256  # 16x16
        elif model_input_size == (560, 560):
            patches_per_sample = 1600  # 40x40
        else:
            patches_per_sample = 256  # Default
        
        # Calculate total number of images from total patches
        total_samples = total_patches // patches_per_sample
    
    # Get all concepts
    concepts = list(gt_samples_per_concept.keys())
    
    print(f"Processing {len(concepts)} concepts...")
    
    # Get all test sample indices (we're only working with test data)
    all_test_samples = set()
    for concept_samples in gt_samples_per_concept.values():
        all_test_samples.update(concept_samples)
    max_test_sample_idx = max(all_test_samples) if all_test_samples else 0
    
    # If no activation range provided, compute global min/max across all concepts
    if activation_range is None:
        print("Computing global activation range across all concepts...")
        global_min = float('inf')
        global_max = float('-inf')
        
        # First pass: find global min/max
        for concept in tqdm(concepts, desc="Finding global range"):
            if concept not in gt_samples_per_concept:
                continue
                
            # Get samples based on contains_concept flag
            samples_with_concept = set(gt_samples_per_concept.get(concept, []))
            all_test_samples_list = sorted(list(all_test_samples))
            if contains_concept:
                target_samples = [s for s in all_test_samples_list if s in samples_with_concept]
            else:
                target_samples = [s for s in all_test_samples_list if s not in samples_with_concept]
            target_samples = torch.tensor(target_samples)
            
            if len(target_samples) == 0:
                continue
            
            # Get patch indices (same logic as main loop)
            if dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank']:
                # Text dataset logic - simplified for range finding
                continue  # Skip text for now, use default range
            else:
                # Vision dataset logic
                global_patch_indices = convert_image_indices_to_patch_indices(
                    target_samples.tolist(), patches_per_sample
                )
                filtered_global_patch_indices = filter_patches_by_image_presence(
                    global_patch_indices.numpy(), dataset_name, model_input_size
                )
                valid_global_indices, local_test_indices = map_global_to_split_local(
                    filtered_global_patch_indices, dataset_name, 'test', model_input_size, patch_size=14
                )
                patch_indices = local_test_indices
            
            if len(patch_indices) == 0:
                continue
            
            # Load activations and find min/max
            concept_idx = concepts.index(concept)
            test_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
            
            if test_acts is None:
                continue
                
            concept_acts = test_acts[:, concept_idx]
            filtered_acts = concept_acts[patch_indices]
            
            batch_min = filtered_acts.min().item()
            batch_max = filtered_acts.max().item()
            
            global_min = min(global_min, batch_min)
            global_max = max(global_max, batch_max)
        
        # Set activation range with small padding
        if global_min != float('inf') and global_max != float('-inf'):
            padding = (global_max - global_min) * 0.05  # 5% padding
            activation_range = (global_min - padding, global_max + padding)
            print(f"Computed global activation range: [{activation_range[0]:.4f}, {activation_range[1]:.4f}]")
        else:
            activation_range = (-1.0, 1.0)  # Fallback
            print("Using fallback activation range: [-1.0, 1.0]")
    
    # Second pass: compute distributions using global range
    for concept in tqdm(concepts, desc="Processing concepts"):
        if concept not in gt_samples_per_concept:
            continue
            
        # Get TEST samples with or without this concept based on contains_concept flag
        samples_with_concept = set(gt_samples_per_concept.get(concept, []))
        all_test_samples_list = sorted(list(all_test_samples))
        if contains_concept:
            target_samples = [s for s in all_test_samples_list if s in samples_with_concept]
            sample_type_desc = "with"
        else:
            target_samples = [s for s in all_test_samples_list if s not in samples_with_concept]
            sample_type_desc = "without"
        target_samples = torch.tensor(target_samples)
        
        if len(target_samples) == 0:
            print(f"Warning: No test samples found {sample_type_desc} concept '{concept}'")
            continue
        
        # For text datasets, we need to handle variable length
        if dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank']:
            # Get all token indices for target samples
            token_indices = []
            for sample_idx in target_samples:
                if sample_idx < len(tokens_per_sample):
                    # Calculate cumulative token count up to this sample
                    start_idx = sum(tokens_per_sample[:sample_idx])
                    end_idx = start_idx + tokens_per_sample[sample_idx]
                    token_indices.extend(range(start_idx, end_idx))
            
            patch_indices = torch.tensor(token_indices, dtype=torch.long, device=device)
        else:
            # For vision datasets, convert image indices to patch indices
            # This will be overridden below with proper filtering
            pass
        
        # For vision datasets, we need to filter out padding patches and map to test split
        if dataset_name not in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank']:
            
            # Convert image indices to global patch indices
            global_patch_indices = convert_image_indices_to_patch_indices(
                target_samples.tolist(), patches_per_sample
            )
            # Filter out padding patches using global indices
            filtered_global_patch_indices = filter_patches_by_image_presence(
                global_patch_indices.numpy(), dataset_name, model_input_size
            )
            
            # Map filtered global patch indices to test split local indices
            valid_global_indices, local_test_indices = map_global_to_split_local(
                filtered_global_patch_indices, dataset_name, 'test', model_input_size, patch_size=14
            )
            
            patch_indices = local_test_indices.to(device)
        
        # Load activations for this concept
        concept_idx = concepts.index(concept)
        
        # Get test split activations
        test_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
        
        if test_acts is None:
            print(f"Warning: Could not load test activations")
            continue
        
        # Get activations for this concept - test_acts is already [patches, concepts]
        concept_acts = test_acts[:, concept_idx].to(device)  # Move to GPU
        
        # Since we're working with test data and patch_indices are already based on test samples,
        # we can use them directly with the test activations
        filtered_acts = concept_acts[patch_indices]
        
        # Compute distribution per sample
        if dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank']:
            # For text, compute overall distribution since tokens per sample varies
            # Create histogram
            hist = torch.histc(filtered_acts, bins=num_bins, 
                              min=activation_range[0], max=activation_range[1])
            # Normalize
            if hist.sum() > 0:
                mean_distribution = hist / hist.sum()
            else:
                mean_distribution = torch.zeros(num_bins, device=device)
        else:
            # For vision, compute per-sample distributions then average
            # Since we have patches, compute overall distribution instead of per-sample
            # Create histogram for all patches
            hist = torch.histc(filtered_acts, bins=num_bins, 
                              min=activation_range[0], max=activation_range[1])
            # Normalize
            if hist.sum() > 0:
                mean_distribution = hist / hist.sum()
            else:
                mean_distribution = torch.zeros(num_bins, device=device)
        
        # Store results
        results[concept] = {
            'mean_distribution': mean_distribution.cpu(),
            'num_samples': len(target_samples),
            'total_patches': len(patch_indices),
            'activation_range': activation_range,
            'num_bins': num_bins,
            'contains_concept': contains_concept
        }
    
    return results


def get_activation_distributions_for_non_concept_samples(
    act_loader,
    gt_samples_per_concept: Dict[str, List[int]],
    dataset_name: str,
    model_input_size: Tuple,
    device: torch.device,
    sample_type: str = 'patch',
    num_bins: int = 100,
    activation_range: Tuple[float, float] = None
) -> Dict[str, Dict]:
    """Backward compatibility wrapper - analyzes samples WITHOUT concepts."""
    return get_activation_distributions_for_concept_samples(
        act_loader, gt_samples_per_concept, dataset_name, model_input_size, device,
        sample_type, num_bins, activation_range, contains_concept=False
    )


def get_activation_distributions_for_with_concept_samples(
    act_loader,
    gt_samples_per_concept: Dict[str, List[int]],
    dataset_name: str,
    model_input_size: Tuple,
    device: torch.device,
    sample_type: str = 'patch',
    num_bins: int = 100,
    activation_range: Tuple[float, float] = None
) -> Dict[str, Dict]:
    """Analyzes samples WITH concepts."""
    return get_activation_distributions_for_concept_samples(
        act_loader, gt_samples_per_concept, dataset_name, model_input_size, device,
        sample_type, num_bins, activation_range, contains_concept=True
    )


def plot_activation_distributions(
    activation_distributions: Optional[Dict[str, Dict]] = None,
    concepts_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    overall_title: Optional[str] = None,
    # Parameters for loading data internally
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    concept_type: Optional[str] = None,
    percent_thru_model: int = 100,
    contains_concept: bool = False
):
    """
    Plot activation distributions for specified concepts.
    
    Args:
        activation_distributions: Dictionary from get_activation_distributions_for_concept_samples (optional if loading internally)
        concepts_to_plot: List of concepts to plot (None = plot all)
        save_path: Path to save the figure (None = don't save)
        figsize: Figure size
        overall_title: Optional overall title for the entire figure
        dataset_name: Dataset name for internal loading (e.g. 'CLEVR')
        model_name: Model name for internal loading (e.g. 'CLIP')
        concept_type: Concept type for internal loading (e.g. 'linsep_patch_embeddings_BD_True_BN_False')
        percent_thru_model: Percent through model for internal loading (default 100)
        contains_concept: Whether to load WITH or WITHOUT concept data (default False = WITHOUT)
    """
    
    # Load data internally if not provided
    if activation_distributions is None:
        if not all([dataset_name, model_name, concept_type]):
            raise ValueError("Must provide either activation_distributions or (dataset_name, model_name, concept_type)")
        
        import torch
        
        # Determine filename suffix based on contains_concept
        suffix = "with_concept_test" if contains_concept else "test"
        filename = f'activation_distributions/{dataset_name}/activation_distributions_{model_name}_{concept_type}_percentthrumodel_{percent_thru_model}_{suffix}.pt'
        
        try:
            activation_distributions = torch.load(filename, weights_only=True)
        except FileNotFoundError:
            print(f"Error: Could not load file {filename}")
            return
    if concepts_to_plot is None:
        concepts_to_plot = list(activation_distributions.keys())
    
    # Auto-detect WITH/WITHOUT status if no title provided
    if overall_title is None:
        # Check first concept to determine WITH/WITHOUT status
        if concepts_to_plot and concepts_to_plot[0] in activation_distributions:
            first_concept_info = activation_distributions[concepts_to_plot[0]]
            contains_concept = first_concept_info.get('contains_concept', False)
            if contains_concept:
                overall_title = "Activation Distributions for Samples WITH Concepts"
            else:
                overall_title = "Activation Distributions for Samples WITHOUT Concepts"
        else:
            overall_title = "Activation Distributions"
    
    # Create figure
    num_concepts = len(concepts_to_plot)
    cols = min(3, num_concepts)
    rows = (num_concepts + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Add overall title
    fig.suptitle(overall_title, fontsize=16, y=0.98)
    
    for i, concept in enumerate(concepts_to_plot):
        if concept not in activation_distributions:
            print(f"Warning: Concept '{concept}' not found in distributions")
            continue
            
        ax = axes[i]
        dist_info = activation_distributions[concept]
        
        # Get distribution and create x-axis values
        distribution = dist_info['mean_distribution']
        activation_range = dist_info['activation_range']
        num_bins = dist_info['num_bins']
        
        # Create bin centers for x-axis
        bin_edges = np.linspace(activation_range[0], activation_range[1], num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Determine concept presence description
        contains_concept = dist_info.get('contains_concept', False)
        presence_desc = "WITH" if contains_concept else "WITHOUT"
        
        # Plot distribution
        ax.bar(bin_centers, distribution.numpy(), width=(bin_edges[1] - bin_edges[0]) * 0.8,
               alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('Activation Value', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.set_title(f'{concept} ({presence_desc})\n({dist_info["num_samples"]} samples, {dist_info["total_patches"]} patches)',
                    fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits
        ax.set_xlim(activation_range)
        
        # Add statistics
        mean_act = (bin_centers * distribution.numpy()).sum()
        ax.axvline(mean_act, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_act:.3f}')
        ax.legend(fontsize=8)
    
    # Remove empty subplots
    for i in range(num_concepts, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for overall title
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_with_without_comparison_overlayed(
    dataset_name: str,
    model_name: str,
    concept_type: str,
    percent_thru_model: int = 100,
    concepts_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot activation distributions comparing samples WITH vs WITHOUT concepts overlayed on same plots.
    Includes optimal detection and inversion threshold lines.
    
    Args:
        dataset_name: Name of dataset (e.g. 'CLEVR')
        model_name: Name of model (e.g. 'CLIP')  
        concept_type: Type of concept (e.g. 'linsep_patch_embeddings_BD_True_BN_False')
        percent_thru_model: Percent through model (default 100)
        concepts_to_plot: List of concepts to plot (None = plot all)
        save_path: Path to save figure
        figsize: Figure size
    """
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load activation distributions
    wo_concept_file = f'activation_distributions/{dataset_name}/activation_distributions_{model_name}_{concept_type}_percentthrumodel_{percent_thru_model}_test.pt'
    w_concept_file = f'activation_distributions/{dataset_name}/activation_distributions_{model_name}_{concept_type}_percentthrumodel_{percent_thru_model}_with_concept_test.pt'
    
    try:
        wo_concept_activation_distributions = torch.load(wo_concept_file, weights_only=True)
        w_concept_activation_distributions = torch.load(w_concept_file, weights_only=True)
    except FileNotFoundError as e:
        print(f"Error loading activation distribution files: {e}")
        return
    
    # Load detection thresholds
    detection_threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_{concept_type}_percentthrumodel_{percent_thru_model}.pt'
    try:
        detection_thresholds = torch.load(detection_threshold_file, weights_only=True)
    except FileNotFoundError:
        print(f"Warning: Could not load detection thresholds from {detection_threshold_file}")
        detection_thresholds = {}
    
    # Load inversion thresholds
    inversion_threshold_file = f'Best_Inversion_Percentiles_Cal/{dataset_name}/best_inversion_percentiles_{model_name}_{concept_type}_percentthrumodel_{percent_thru_model}.pt'
    inversion_thresholds = {}
    
    try:
        inversion_thresholds = torch.load(inversion_threshold_file, weights_only=True)
    except FileNotFoundError:
        pass  # Just skip inversion thresholds if not available
    
    # Determine concepts to plot
    if concepts_to_plot is None:
        concepts_to_plot = list(wo_concept_activation_distributions.keys())
    
    # Create subplots
    num_concepts = len(concepts_to_plot)
    cols = min(3, num_concepts)
    rows = (num_concepts + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Add overall title
    fig.suptitle(f'Activation Distribution Comparison: WITH vs WITHOUT Concepts\n{dataset_name} {model_name} {concept_type}', 
                 fontsize=16, y=0.98)
    
    for i, concept in enumerate(concepts_to_plot):
        if concept not in wo_concept_activation_distributions or concept not in w_concept_activation_distributions:
            print(f"Warning: Concept '{concept}' not found in both distributions")
            continue
            
        ax = axes[i]
        
        # Get distribution data for both WITH and WITHOUT
        wo_info = wo_concept_activation_distributions[concept]
        w_info = w_concept_activation_distributions[concept]
        
        wo_distribution = wo_info['mean_distribution']
        w_distribution = w_info['mean_distribution']
        
        # Use the same activation range for both (should be computed globally)
        activation_range = wo_info['activation_range']  # Both should have same range
        num_bins = wo_info['num_bins']
        
        # Create bin centers for x-axis
        bin_edges = np.linspace(activation_range[0], activation_range[1], num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Plot WITHOUT concept distribution (red, transparent)
        ax.bar(bin_centers, wo_distribution.numpy(), width=bin_width * 0.8,
               alpha=0.4, color='red', edgecolor='darkred', linewidth=0.5,
               label=f'WITHOUT {concept} ({wo_info["num_samples"]} samples)')
        
        # Plot WITH concept distribution (blue, transparent)  
        ax.bar(bin_centers, w_distribution.numpy(), width=bin_width * 0.8,
               alpha=0.4, color='blue', edgecolor='darkblue', linewidth=0.5,
               label=f'WITH {concept} ({w_info["num_samples"]} samples)')
        
        # Add detection threshold line (green)
        if concept in detection_thresholds:
            detection_thresh = detection_thresholds[concept]['best_threshold']
            ax.axvline(detection_thresh, color='green', linestyle='--', linewidth=2, 
                      label=f'Detection Threshold: {detection_thresh:.3f}')
        
        # Add inversion threshold line (orange)
        if concept in inversion_thresholds:
            inversion_thresh_data = inversion_thresholds[concept]['best_threshold']
            # Handle tuple format (threshold, index) - use just the threshold value
            if isinstance(inversion_thresh_data, (tuple, list)):
                inversion_thresh = inversion_thresh_data[0]
            else:
                inversion_thresh = inversion_thresh_data
            ax.axvline(inversion_thresh, color='orange', linestyle='--', linewidth=2,
                      label=f'Inversion Threshold: {inversion_thresh:.3f}')
        
        # Customize plot
        ax.set_xlabel('Activation Value', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.set_title(f'{concept}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(activation_range)
        
        # Add legend (smaller font to fit)
        ax.legend(fontsize=8, loc='upper right')
    
    # Remove empty subplots
    for i in range(num_concepts, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for overall title
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to: {save_path}")
    
    plt.show()


def plot_with_without_comparison(
    dataset_name: str,
    model_name: str,
    concept_type: str,
    percent_thru_model: int = 100,
    concepts_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot activation distributions comparing samples WITH vs WITHOUT concepts using nested subplots.
    Each concept gets its own subplot area, within which WITH and WITHOUT are stacked vertically.
    
    Args:
        dataset_name: Name of dataset (e.g. 'CLEVR')
        model_name: Name of model (e.g. 'CLIP')  
        concept_type: Type of concept (e.g. 'linsep_patch_embeddings_BD_True_BN_False')
        percent_thru_model: Percent through model (default 100)
        concepts_to_plot: List of concepts to plot (None = plot all)
        save_path: Path to save figure
        figsize: Figure size
    """
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    
    # Load activation distributions
    wo_concept_file = f'activation_distributions/{dataset_name}/activation_distributions_{model_name}_{concept_type}_percentthrumodel_{percent_thru_model}_test.pt'
    w_concept_file = f'activation_distributions/{dataset_name}/activation_distributions_{model_name}_{concept_type}_percentthrumodel_{percent_thru_model}_with_concept_test.pt'
    
    try:
        wo_concept_activation_distributions = torch.load(wo_concept_file, weights_only=True)
        w_concept_activation_distributions = torch.load(w_concept_file, weights_only=True)
    except FileNotFoundError as e:
        print(f"Error loading activation distribution files: {e}")
        return
    
    # Load detection thresholds
    detection_threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_{concept_type}_percentthrumodel_{percent_thru_model}.pt'
    try:
        detection_thresholds = torch.load(detection_threshold_file, weights_only=True)
    except FileNotFoundError:
        print(f"Warning: Could not load detection thresholds from {detection_threshold_file}")
        detection_thresholds = {}
    
    # Load inversion thresholds
    inversion_threshold_file = f'Best_Inversion_Percentiles_Cal/{dataset_name}/best_inversion_percentiles_{model_name}_{concept_type}_percentthrumodel_{percent_thru_model}.pt'
    inversion_thresholds = {}
    
    try:
        inversion_thresholds = torch.load(inversion_threshold_file, weights_only=True)
    except FileNotFoundError:
        pass  # Just skip inversion thresholds if not available
    
    # Determine concepts to plot
    if concepts_to_plot is None:
        concepts_to_plot = list(wo_concept_activation_distributions.keys())
    
    # Create main subplots - one for each concept
    num_concepts = len(concepts_to_plot)
    cols = min(3, num_concepts)  
    rows = (num_concepts + cols - 1) // cols
    
    # Create the main figure and grid
    fig = plt.figure(figsize=figsize)
    main_gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)
    
    # Add overall title
    fig.suptitle(f'Activation Distribution Comparison: WITH vs WITHOUT Concepts\n{dataset_name} {model_name} {concept_type}', 
                 fontsize=16, y=0.98)
    
    for i, concept in enumerate(concepts_to_plot):
        if concept not in wo_concept_activation_distributions or concept not in w_concept_activation_distributions:
            print(f"Warning: Concept '{concept}' not found in both distributions")
            continue
        
        # Calculate main subplot position
        row_idx = i // cols
        col_idx = i % cols
        
        # Create nested subplot within this concept's main area
        concept_gs = GridSpecFromSubplotSpec(2, 1, main_gs[row_idx, col_idx], 
                                           height_ratios=[1, 1], hspace=0.05)
        
        # Get distribution data for both WITH and WITHOUT
        wo_info = wo_concept_activation_distributions[concept]
        w_info = w_concept_activation_distributions[concept]
        
        wo_distribution = wo_info['mean_distribution']
        w_distribution = w_info['mean_distribution']
        
        # Use the same activation range for both (should be computed globally)
        activation_range = wo_info['activation_range']  # Both should have same range
        num_bins = wo_info['num_bins']
        
        # Create bin centers for x-axis
        bin_edges = np.linspace(activation_range[0], activation_range[1], num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Create the nested subplots
        ax_without = fig.add_subplot(concept_gs[0, 0])  # Top: WITHOUT
        ax_with = fig.add_subplot(concept_gs[1, 0])     # Bottom: WITH
        
        # Add concept title to the top subplot only
        ax_without.set_title(f'{concept}', fontsize=12, pad=15)
        
        # Plot WITHOUT concept (top subplot)
        wo_bars = ax_without.bar(bin_centers, wo_distribution.numpy(), width=bin_width * 0.8,
                                alpha=0.7, color='red', edgecolor='darkred', linewidth=0.5,
                                label=f'WITHOUT ({wo_info["num_samples"]} samples)')
        
        # Add threshold lines to WITHOUT plot (no labels, we'll add to legend separately)
        if concept in detection_thresholds:
            detection_thresh = detection_thresholds[concept]['best_threshold']
            ax_without.axvline(detection_thresh, color='green', linestyle='--', linewidth=2)
        
        if concept in inversion_thresholds:
            inversion_thresh_data = inversion_thresholds[concept]['best_threshold']
            if isinstance(inversion_thresh_data, (tuple, list)):
                inversion_thresh = inversion_thresh_data[0]
            else:
                inversion_thresh = inversion_thresh_data
            ax_without.axvline(inversion_thresh, color='orange', linestyle='--', linewidth=2)
        
        ax_without.set_ylabel('Density', fontsize=8)
        ax_without.grid(True, alpha=0.3)
        ax_without.set_xlim(activation_range)
        ax_without.set_xticklabels([])  # Remove x-axis labels from top plot
        
        # Create legend for WITHOUT plot
        legend_elements = [wo_bars]
        legend_labels = [f'WITHOUT ({wo_info["num_samples"]} samples)']
        if concept in detection_thresholds:
            legend_elements.append(plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2))
            legend_labels.append(f'Detection: {detection_thresholds[concept]["best_threshold"]:.3f}')
        if concept in inversion_thresholds:
            thresh_val = inversion_thresholds[concept]['best_threshold']
            if isinstance(thresh_val, (tuple, list)):
                thresh_val = thresh_val[0]
            legend_elements.append(plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2))
            legend_labels.append(f'Inversion: {thresh_val:.3f}')
        
        ax_without.legend(legend_elements, legend_labels, fontsize=6)
        
        # Plot WITH concept (bottom subplot) 
        w_bars = ax_with.bar(bin_centers, w_distribution.numpy(), width=bin_width * 0.8,
                            alpha=0.7, color='blue', edgecolor='darkblue', linewidth=0.5,
                            label=f'WITH ({w_info["num_samples"]} samples)')
        
        # Add threshold lines to WITH plot (no labels, we'll add to legend separately)
        if concept in detection_thresholds:
            detection_thresh = detection_thresholds[concept]['best_threshold']
            ax_with.axvline(detection_thresh, color='green', linestyle='--', linewidth=2)
        
        if concept in inversion_thresholds:
            inversion_thresh_data = inversion_thresholds[concept]['best_threshold']
            if isinstance(inversion_thresh_data, (tuple, list)):
                inversion_thresh = inversion_thresh_data[0]
            else:
                inversion_thresh = inversion_thresh_data
            ax_with.axvline(inversion_thresh, color='orange', linestyle='--', linewidth=2)
        
        ax_with.set_xlabel('Activation Value', fontsize=9)
        ax_with.set_ylabel('Density', fontsize=8) 
        ax_with.grid(True, alpha=0.3)
        ax_with.set_xlim(activation_range)
        
        # Create legend for WITH plot
        legend_elements = [w_bars]
        legend_labels = [f'WITH ({w_info["num_samples"]} samples)']
        if concept in detection_thresholds:
            legend_elements.append(plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2))
            legend_labels.append(f'Detection: {detection_thresholds[concept]["best_threshold"]:.3f}')
        if concept in inversion_thresholds:
            thresh_val = inversion_thresholds[concept]['best_threshold']
            if isinstance(thresh_val, (tuple, list)):
                thresh_val = thresh_val[0]
            legend_elements.append(plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2))
            legend_labels.append(f'Inversion: {thresh_val:.3f}')
        
        ax_with.legend(legend_elements, legend_labels, fontsize=6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for overall title
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to: {save_path}")
    
    plt.show()


def plot_single_concept_distribution(
    activation_distributions: Dict[str, Dict],
    concept: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    color: str = 'steelblue'
):
    """
    Plot activation distribution for a single concept with more detail.
    
    Args:
        activation_distributions: Dictionary from get_activation_distributions_for_non_concept_samples
        concept: Concept to plot
        save_path: Path to save the figure
        figsize: Figure size
        color: Bar color
    """
    if concept not in activation_distributions:
        raise ValueError(f"Concept '{concept}' not found in distributions")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    dist_info = activation_distributions[concept]
    distribution = dist_info['mean_distribution']
    activation_range = dist_info['activation_range']
    num_bins = dist_info['num_bins']
    
    # Determine concept presence description
    contains_concept = dist_info.get('contains_concept', False)
    presence_desc = "WITH" if contains_concept else "WITHOUT"
    
    # Create bin centers for x-axis
    bin_edges = np.linspace(activation_range[0], activation_range[1], num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot distribution
    ax.bar(bin_centers, distribution.numpy(), width=(bin_edges[1] - bin_edges[0]) * 0.8,
           alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
    
    # Calculate statistics
    dist_np = distribution.numpy()
    mean_act = (bin_centers * dist_np).sum()
    
    # Find mode
    mode_idx = dist_np.argmax()
    mode_act = bin_centers[mode_idx]
    
    # Calculate percentiles
    cumsum = np.cumsum(dist_np)
    p25_idx = np.searchsorted(cumsum, 0.25)
    p50_idx = np.searchsorted(cumsum, 0.50)
    p75_idx = np.searchsorted(cumsum, 0.75)
    
    p25 = bin_centers[p25_idx] if p25_idx < len(bin_centers) else activation_range[1]
    p50 = bin_centers[p50_idx] if p50_idx < len(bin_centers) else activation_range[1]
    p75 = bin_centers[p75_idx] if p75_idx < len(bin_centers) else activation_range[1]
    
    # Add statistical lines
    ax.axvline(mean_act, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Mean: {mean_act:.3f}')
    ax.axvline(mode_act, color='green', linestyle='--', alpha=0.7, linewidth=2, label=f'Mode: {mode_act:.3f}')
    ax.axvline(p50, color='orange', linestyle='--', alpha=0.7, linewidth=2, label=f'Median: {p50:.3f}')
    
    # Add percentile shading
    ax.axvspan(p25, p75, alpha=0.1, color='gray', label=f'IQR: [{p25:.3f}, {p75:.3f}]')
    
    # Customize plot
    ax.set_xlabel('Activation Value', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'Activation Distribution for Samples {presence_desc} "{concept}"\n'
                f'({dist_info["num_samples"]} samples, {dist_info["total_patches"]} patches)',
                fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(activation_range)
    
    # Add legend
    ax.legend(fontsize=10, loc='upper right')
    
    # Add text box with additional stats
    textstr = f'Samples: {dist_info["num_samples"]:,}\n'
    textstr += f'Patches: {dist_info["total_patches"]:,}\n'
    textstr += f'Max density: {dist_np.max():.3f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def get_activation_distributions_with_patch_level(
    act_loader,
    gt_samples_per_concept: Dict[str, List[int]],
    gt_patches_per_concept: Dict[str, List[int]],
    dataset_name: str,
    model_input_size: Tuple,
    device: torch.device,
    sample_type: str = 'patch',
    num_bins: int = 100,
    activation_range: Tuple[float, float] = None
) -> Dict[str, Dict]:
    """
    Get activation distributions for samples WITH concepts, broken down into:
    - overall: all patches from images with the concept
    - patch_with_concept: patches that actually contain the concept
    - patch_without_concept: patches from images with concept but patch doesn't contain it
    
    Args:
        act_loader: ChunkedActivationLoader instance
        gt_samples_per_concept: Dictionary mapping concepts to sample indices
        gt_patches_per_concept: Dictionary mapping concepts to patch indices that contain the concept
        dataset_name: Name of the dataset
        model_input_size: Model input size
        device: Device to use
        sample_type: 'patch' or 'cls'
        num_bins: Number of bins for the histogram
        activation_range: Range of activations
        
    Returns:
        Dictionary mapping concepts to their activation distribution info with patch-level breakdown
    """
    # First get the regular WITH concept distributions
    results = get_activation_distributions_for_concept_samples(
        act_loader, gt_samples_per_concept, dataset_name, model_input_size, device,
        sample_type, num_bins, activation_range, contains_concept=True
    )
    
    # Skip patch-level analysis for text datasets
    if dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank']:
        print("Note: Patch-level concept analysis not applicable for text datasets")
        return results
    
    # Calculate patches per sample
    if model_input_size == (224, 224):
        patches_per_sample = 256  # 16x16
    elif model_input_size == (560, 560):
        patches_per_sample = 1600  # 40x40
    else:
        patches_per_sample = 256  # Default
    
    # Get all test sample indices
    all_test_samples = set()
    for concept_samples in gt_samples_per_concept.values():
        all_test_samples.update(concept_samples)
    
    # Get all concepts
    concepts = list(gt_samples_per_concept.keys())
    
    print(f"Adding patch-level breakdown for {len(concepts)} concepts...")
    
    # Process each concept
    for concept in tqdm(concepts, desc="Processing patch-level concepts"):
        if concept not in gt_samples_per_concept or concept not in results:
            continue
        
        # Get samples and patches with this concept
        samples_with_concept = set(gt_samples_per_concept.get(concept, []))
        patches_with_concept = set(gt_patches_per_concept.get(concept, []))
        
        all_test_samples_list = sorted(list(all_test_samples))
        samples_with_concept_list = [s for s in all_test_samples_list if s in samples_with_concept]
        samples_with_tensor = torch.tensor(samples_with_concept_list)
        
        if len(samples_with_tensor) == 0:
            continue
        
        # Get all patches from images with concept
        global_patch_indices = convert_image_indices_to_patch_indices(
            samples_with_tensor.tolist(), patches_per_sample
        )
        filtered_global_patch_indices = filter_patches_by_image_presence(
            global_patch_indices.numpy(), dataset_name, model_input_size
        )
        valid_global_indices, local_test_indices = map_global_to_split_local(
            filtered_global_patch_indices, dataset_name, 'test', model_input_size, patch_size=14
        )
        
        if len(local_test_indices) == 0:
            continue
        
        # Load test activations
        concept_idx = concepts.index(concept)
        test_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
        
        if test_acts is None:
            continue
        
        concept_acts = test_acts[:, concept_idx].to(device)
        
        # Get activation range from the overall distribution
        if activation_range is None and 'activation_range' in results[concept]:
            activation_range = results[concept]['activation_range']
        
        # Separate patches that contain vs don't contain the concept
        patches_without_concept_mask = ~np.isin(valid_global_indices, list(patches_with_concept))
        patches_with_concept_mask = np.isin(valid_global_indices, list(patches_with_concept))
        
        # Compute distribution for patches WITH concept
        patches_with_indices = local_test_indices[patches_with_concept_mask]
        if len(patches_with_indices) > 0:
            patch_indices = patches_with_indices.to(device)
            filtered_acts = concept_acts[patch_indices]
            
            hist = torch.histc(filtered_acts, bins=num_bins, 
                              min=activation_range[0], max=activation_range[1])
            
            if hist.sum() > 0:
                mean_distribution = hist / hist.sum()
            else:
                mean_distribution = torch.zeros(num_bins, device=device)
            
            results[concept]['patch_with_concept'] = {
                'mean_distribution': mean_distribution.cpu(),
                'num_samples': len(samples_with_tensor),
                'total_patches': len(patch_indices),
                'activation_range': activation_range,
                'num_bins': num_bins,
                'contains_concept': True,
                'patch_contains_concept': True
            }
        
        # Compute distribution for patches WITHOUT concept (but from images with concept)
        patches_without_indices = local_test_indices[patches_without_concept_mask]
        if len(patches_without_indices) > 0:
            patch_indices = patches_without_indices.to(device)
            filtered_acts = concept_acts[patch_indices]
            
            hist = torch.histc(filtered_acts, bins=num_bins, 
                              min=activation_range[0], max=activation_range[1])
            
            if hist.sum() > 0:
                mean_distribution = hist / hist.sum()
            else:
                mean_distribution = torch.zeros(num_bins, device=device)
            
            results[concept]['patch_without_concept'] = {
                'mean_distribution': mean_distribution.cpu(),
                'num_samples': len(samples_with_tensor),
                'total_patches': len(patch_indices),
                'activation_range': activation_range,
                'num_bins': num_bins,
                'contains_concept': True,
                'patch_contains_concept': False
            }
    
    return results


def plot_concept_heatmaps_with_distributions(
    concept: str,
    dataset_name: str,
    model_name: str,
    concept_type: str,
    n_samples: int = 5,
    start_idx: int = 0,
    model_input_size: Tuple = None,
    percent_thru_model: int = 100,
    save_path: Optional[str] = None,
    figsize_per_image: Tuple[float, float] = (3, 9),
    cmap: str = 'hot',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    scratch_dir: str = ''
):
    """
    Plot concept heatmaps with corresponding activation distributions for test samples.
    Top row: Original images
    Middle row: Heatmaps showing cosine similarity/distance for each patch
    Bottom row: Distribution of activations for all patches in that image
    
    Args:
        concept: The concept to visualize
        dataset_name: Name of dataset (e.g. 'CLEVR')
        model_name: Name of model (e.g. 'CLIP')
        concept_type: Type of concept (e.g. 'linsep')
        n_samples: Number of test samples to visualize
        start_idx: Starting index in test set
        model_input_size: Model input size (will be inferred if None)
        percent_thru_model: Percentage through model
        save_path: Path to save figure
        figsize_per_image: Figure size per image (width, height)
        cmap: Colormap for heatmaps
        vmin: Minimum value for heatmap color scale
        vmax: Maximum value for heatmap color scale
        scratch_dir: Directory where activation files are stored
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from utils.general_utils import retrieve_image, get_split_index_from_global_index, pad_or_resize_img
    from utils.patch_alignment_utils import compute_patch_similarities_to_vector
    from utils.memory_management_utils import ChunkedActivationLoader
    
    # Determine model input size
    if model_input_size is None:
        if dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank']:
            print("This visualization is designed for image datasets")
            return
        else:
            if model_name == 'CLIP':
                model_input_size = (224, 224)
            elif model_name == 'Llama':
                model_input_size = (560, 560)
            else:
                raise ValueError(f"Unknown model: {model_name}")
    
    # Get concept label and activation file based on full concept type
    if concept_type == 'avg_patch_embeddings':
        con_label = f'{model_name}_avg_patch_embeddings_percentthrumodel_{percent_thru_model}'
        acts_file = f"cosine_similarities_avg_concepts_{model_name}_patch_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        con_label = f'{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}'
        acts_file = f"dists_linsep_concepts_BD_True_BN_False_{model_name}_patch_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        con_label = f"{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
        acts_file = f"cosine_similarities_kmeans_1000_concepts_{model_name}_patch_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        con_label = f"{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
        acts_file = f"dists_kmeans_1000_linsep_concepts_{model_name}_patch_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    # Load activation loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use scratch_dir from function parameter
    
    try:
        act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
    except FileNotFoundError:
        try:
            act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir='.', device=device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Activation file not found: {acts_file}")
    
    # Load concepts to get concept index
    if concept_type == 'avg_patch_embeddings':
        concepts_file = f"Concepts/{dataset_name}/avg_concepts_{model_name}_patch_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        concepts_file = f"Concepts/{dataset_name}/linsep_concepts_BD_True_BN_False_{model_name}_patch_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        concepts_file = f"Concepts/{dataset_name}/kmeans_1000_concepts_{model_name}_patch_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        concepts_file = f"Concepts/{dataset_name}/kmeans_1000_linsep_concepts_{model_name}_patch_embeddings_percentthrumodel_{percent_thru_model}.pt"
    
    if os.path.exists(concepts_file):
        concepts_data = torch.load(concepts_file, weights_only=False)
        if isinstance(concepts_data, dict):
            concept_names = list(concepts_data.keys())
        else:
            # For some formats, we need to load separately
            concept_names = None
    else:
        concept_names = None
    
    # Get concept index
    if concept_names and concept in concept_names:
        concept_idx = concept_names.index(concept)
    else:
        # Try to infer from activation loader
        print(f"Warning: Could not find concept '{concept}' in concept list, using index 0")
        concept_idx = 0
    
    # Get test image indices - always use sequential test images
    # First, we need to get the global indices of the first n test images
    import pandas as pd
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    
    # Get all test image indices (these are global indices)
    test_global_indices = metadata[metadata['split'] == 'test'].index.tolist()
    
    # Get the subset we want to show
    if start_idx + n_samples > len(test_global_indices):
        print(f"Warning: Not enough test images. Requested {n_samples} starting from {start_idx}, but only {len(test_global_indices)} test images available.")
        n_samples = min(n_samples, len(test_global_indices) - start_idx)
    
    # Get the global indices of the test images we want to show
    test_indices = test_global_indices[start_idx:start_idx + n_samples]
    
    # Calculate patches per image
    if model_input_size == (224, 224):
        patches_per_image = 256  # 16x16
        grid_size = 16
    elif model_input_size == (560, 560):
        patches_per_image = 1600  # 40x40
        grid_size = 40
    else:
        patches_per_image = 256
        grid_size = 16
    
    # Create figure with subplots (4 rows: images, individual scale heatmaps, common scale heatmaps, distributions)
    # Adjust figure height for 4 rows
    fig = plt.figure(figsize=(n_samples * figsize_per_image[0], figsize_per_image[1] * 1.33))
    
    # Use GridSpec for better control - 4 rows
    gs = gridspec.GridSpec(4, n_samples, height_ratios=[1, 1, 1, 1], hspace=0.15, wspace=0.1)
    
    # Load test activations
    test_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
    if test_acts is None:
        raise ValueError("Could not load test activations")
    
    # Get activations for this concept
    concept_acts = test_acts[:, concept_idx].cpu().numpy()
    
    # Determine global color scale for common scale heatmaps
    # We'll calculate this after collecting all patch activations
    
    # The ground truth samples are already test-set relative indices (0-based within test set)
    # So we don't need to map them
    
    # Arrays to store all patch activations for distribution alignment
    all_patch_acts = []
    
    # First pass: plot original images and collect activation data
    for i, img_idx in enumerate(test_indices):
        # Top row: Original images
        ax_img = fig.add_subplot(gs[0, i])
        try:
            img = retrieve_image(img_idx, dataset_name)
            resized_img = pad_or_resize_img(img, model_input_size)
            ax_img.imshow(resized_img)
            ax_img.set_title(f'Image {img_idx}', fontsize=10)
        except:
            ax_img.text(0.5, 0.5, f'Image {img_idx}\n(Not Found)', 
                       ha='center', va='center', transform=ax_img.transAxes)
        ax_img.axis('off')
        
        # Convert global image index to test set position
        try:
            split_name, test_set_idx = get_split_index_from_global_index(dataset_name, img_idx)
            if split_name != 'test':
                print(f"Warning: Image {img_idx} is in {split_name} split, not test split")
                all_patch_acts.append(np.zeros(patches_per_image))
                continue
        except Exception as e:
            print(f"Warning: Could not find test position for image {img_idx}: {e}")
            all_patch_acts.append(np.zeros(patches_per_image))
            continue
        
        # Get patch activations for this image
        start_patch = test_set_idx * patches_per_image
        end_patch = start_patch + patches_per_image
        
        if end_patch > len(concept_acts):
            print(f"Warning: Test index {test_set_idx} (image {img_idx}) exceeds available activations")
            all_patch_acts.append(np.zeros(patches_per_image))
            continue
        
        patch_acts = concept_acts[start_patch:end_patch]
        all_patch_acts.append(patch_acts)
    
    # Calculate common scale from all patch activations
    if vmin is None or vmax is None:
        all_acts_for_scale = np.concatenate(all_patch_acts)
        if vmin is None:
            vmin = np.percentile(all_acts_for_scale, 1)
        if vmax is None:
            vmax = np.percentile(all_acts_for_scale, 99)
    
    # Add row labels
    fig.text(0.02, 0.875, 'Original\nImage', fontsize=11, ha='right', va='center')
    fig.text(0.02, 0.625, 'Individual\nScale', fontsize=11, ha='right', va='center')
    fig.text(0.02, 0.375, 'Common\nScale', fontsize=11, ha='right', va='center')
    fig.text(0.02, 0.125, 'Activation\nDistribution', fontsize=11, ha='right', va='center')
    
    # Second pass: create heatmaps with individual colorbars
    for i, (img_idx, patch_acts) in enumerate(zip(test_indices, all_patch_acts)):
        # Middle row: Heatmap overlaid on grayscale image
        ax_heat = fig.add_subplot(gs[1, i])
        
        # Reshape to grid for heatmap
        heatmap_data = patch_acts.reshape(grid_size, grid_size)
        
        # Use individual vmin/vmax for this heatmap
        vmin_local = heatmap_data.min()
        vmax_local = heatmap_data.max()
        
        # Load image and display as grayscale background
        try:
            img = retrieve_image(img_idx, dataset_name)
            resized_img = pad_or_resize_img(img, model_input_size)
            # Display grayscale version
            ax_heat.imshow(resized_img.convert('L'), cmap='gray', alpha=0.4)
            
            # Overlay heatmap with transparency
            im = ax_heat.imshow(heatmap_data, cmap=cmap, alpha=0.6, 
                               vmin=vmin_local, vmax=vmax_local, 
                               extent=[0, model_input_size[0], model_input_size[1], 0])
        except:
            # If image loading fails, just show heatmap
            im = ax_heat.imshow(heatmap_data, cmap=cmap, vmin=vmin_local, vmax=vmax_local)
        
        ax_heat.set_title(f'Max: {vmax_local:.3f}\nMin: {vmin_local:.3f}', 
                         fontsize=9)
        ax_heat.axis('off')
        
        # Add individual colorbar for each heatmap
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax_heat)
        cax = divider.append_axes("right", size="8%", pad=0.08)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=7)
        # Set number of ticks
        from matplotlib.ticker import MaxNLocator
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()
    
    # Third row: Common scale heatmaps
    for i, (img_idx, patch_acts) in enumerate(zip(test_indices, all_patch_acts)):
        ax_heat_common = fig.add_subplot(gs[2, i])
        
        # Reshape to grid for heatmap
        heatmap_data = patch_acts.reshape(grid_size, grid_size)
        
        # Load image and display as grayscale background
        try:
            img = retrieve_image(img_idx, dataset_name)
            resized_img = pad_or_resize_img(img, model_input_size)
            # Display grayscale version
            ax_heat_common.imshow(resized_img.convert('L'), cmap='gray', alpha=0.4)
            
            # Overlay heatmap with transparency using common scale
            im_common = ax_heat_common.imshow(heatmap_data, cmap=cmap, alpha=0.6, 
                                             vmin=vmin, vmax=vmax, 
                                             extent=[0, model_input_size[0], model_input_size[1], 0])
        except:
            # If image loading fails, just show heatmap
            im_common = ax_heat_common.imshow(heatmap_data, cmap=cmap, vmin=vmin, vmax=vmax)
        
        ax_heat_common.axis('off')
    
    # Add a single colorbar for the common scale row
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.35, 0.015, 0.15])
    cbar_common = plt.colorbar(sm, cax=cbar_ax)
    cbar_common.set_label(metric_type, fontsize=9)
    cbar_common.ax.tick_params(labelsize=8)
    
    # Determine common bins for distributions
    all_acts_flat = np.concatenate(all_patch_acts)
    hist_min = np.min(all_acts_flat)
    hist_max = np.max(all_acts_flat)
    bins = np.linspace(hist_min, hist_max, 50)
    
    # Bottom row: Distributions with aligned axes
    max_height = 0
    dist_axes = []  # Keep track of distribution axes
    
    for i, patch_acts in enumerate(all_patch_acts):
        ax_dist = fig.add_subplot(gs[3, i])
        dist_axes.append(ax_dist)  # Store the axis
        
        # Create histogram
        counts, _, patches = ax_dist.hist(patch_acts, bins=bins, alpha=0.7, 
                                         color='steelblue', edgecolor='black')
        
        # Track maximum height for y-axis alignment
        max_height = max(max_height, np.max(counts))
        
        # Add mean line
        mean_val = np.mean(patch_acts)
        ax_dist.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_val:.3f}')
        
        ax_dist.set_xlabel('Activation Value', fontsize=10)
        if i == 0:
            ax_dist.set_ylabel('Count', fontsize=10)
        ax_dist.grid(True, alpha=0.3)
        ax_dist.legend(fontsize=8, loc='upper right')
    
    # Align all distribution y-axes using our stored axes
    for ax_dist in dist_axes:
        ax_dist.set_ylim(0, max_height * 1.1)
        ax_dist.set_xlim(hist_min, hist_max)
    
    # Overall title
    fig.suptitle(f'Concept Activations on Test {dataset_name} Images\n'
                f'Concept: "{concept}", Model: {model_name}',
                fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0.06, 0.02, 0.91, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def analyze_image_concept_activations(
    image_idx: int,
    concept: str,
    dataset_name: str,
    model_name: str,
    concept_type: str,
    model_input_size: Optional[Tuple[int, int]] = None,
    percent_thru_model: int = 100,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 10),
    cmap: str = 'hot',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    scratch_dir: str = '',
    show_patch_outlines: bool = True,
    n_bins: int = 30
) -> plt.Figure:
    """
    Analyze and visualize concept activations for a specific image/text sample with three distribution categories:
    1. Patches/tokens that contain the target concept
    2. Patches/tokens that contain other concepts but not the target
    3. Patches/tokens that don't contain any concept
    
    Args:
        image_idx: Global index of the image/text sample to analyze
        concept: Target concept to analyze
        dataset_name: Name of dataset (e.g. 'CLEVR', 'GoEmotions')
        model_name: Name of model (e.g. 'CLIP', 'Llama')
        concept_type: Type of concept - must be one of:
            'avg_patch_embeddings'
            'linsep_patch_embeddings_BD_True_BN_False'
            'kmeans_1000_patch_embeddings_kmeans'
            'kmeans_1000_linsep_patch_embeddings_kmeans'
        model_input_size: Model input size (will be inferred if None)
        percent_thru_model: Percentage through model
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap for heatmap
        vmin: Minimum value for heatmap color scale
        vmax: Maximum value for heatmap color scale
        scratch_dir: Directory where activation files are stored
        show_patch_outlines: Whether to show patch category outlines on heatmap (images only)
        n_bins: Number of bins for histograms
        
    Returns:
        matplotlib Figure object
    """
    # Check if this is a text dataset
    is_text_dataset = dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank', 'IMDB', 'Jailbreak']
    
    if is_text_dataset:
        # Redirect to text-specific function
        return analyze_text_concept_activations(
            sample_idx=image_idx,  # Use image_idx as sample_idx
            concept=concept,
            dataset_name=dataset_name,
            model_name=model_name,
            concept_type=concept_type,
            model_input_size=model_input_size,
            percent_thru_model=percent_thru_model,
            save_path=save_path,
            figsize=figsize,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            scratch_dir=scratch_dir,
            n_bins=n_bins
        )
    
    # Continue with image analysis
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    from utils.general_utils import retrieve_image, get_split_index_from_global_index, pad_or_resize_img
    from utils.memory_management_utils import ChunkedActivationLoader
    from utils.patch_alignment_utils import filter_patches_by_image_presence
    
    # Determine model input size
    if model_input_size is None:
        if dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank']:
            raise ValueError("This function is designed for image datasets only")
        else:
            if model_name == 'CLIP':
                model_input_size = (224, 224)
            elif model_name == 'Llama':
                model_input_size = (560, 560)
            else:
                raise ValueError(f"Unknown model: {model_name}")
    
    # Calculate patches per image and grid size
    if model_input_size == (224, 224):
        patches_per_image = 256  # 16x16
        grid_size = 16
    elif model_input_size == (560, 560):
        patches_per_image = 1600  # 40x40
        grid_size = 40
    else:
        patches_per_image = 256
        grid_size = 16
    
    patch_size_pixels = model_input_size[0] / grid_size
    
    # Get activation file name based on concept type
    sample_type = 'patch'
    if concept_type == 'avg_patch_embeddings':
        acts_file = f"cosine_similarities_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        acts_file = f"dists_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"cosine_similarities_kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    else:
        raise ValueError(f"Unknown concept type: {concept_type}. Valid options are: 'avg_patch_embeddings', 'linsep_patch_embeddings_BD_True_BN_False', 'kmeans_1000_patch_embeddings_kmeans', 'kmeans_1000_linsep_patch_embeddings_kmeans'")
    
    # Load activation loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
    except FileNotFoundError:
        try:
            act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir='.', device=device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Activation file not found: {acts_file}")
    
    # Import filter utility
    from utils.filter_datasets_utils import filter_concept_dict
    
    # Load ground truth samples per concept (image-level)
    gt_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
    print(f"Loading ground truth from: {gt_file}")
    gt_samples_per_concept = torch.load(gt_file, weights_only=False)
    # Filter to only include valid concepts for this dataset
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    # Load ground truth patches per concept (patch-level)
    gt_patches_file = f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
    print(f"Loading patch-level ground truth from: {gt_patches_file}")
    
    if not os.path.exists(gt_patches_file):
        print(f"Warning: Patch-level ground truth not found at {gt_patches_file}")
        print("Will only be able to show image-level concept presence")
        gt_patches_per_concept = {}
    else:
        gt_patches_per_concept = torch.load(gt_patches_file, weights_only=False)
        # Filter to only include valid concepts for this dataset
        gt_patches_per_concept = filter_concept_dict(gt_patches_per_concept, dataset_name)
    
    # Get all concept names
    all_concepts = list(gt_samples_per_concept.keys())
    if concept not in all_concepts:
        raise ValueError(f"Concept '{concept}' not found. Available concepts: {all_concepts[:10]}...")
    
    concept_idx = all_concepts.index(concept)
    
    # Determine which split the image is in and its position
    split_name, split_idx = get_split_index_from_global_index(dataset_name, image_idx)
    print(f"Image {image_idx} is in {split_name} split at position {split_idx}")
    
    # Load activations for this split
    if split_name == 'test':
        split_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
    elif split_name == 'val':
        split_acts = act_loader.load_split_tensor('val', dataset_name, model_input_size, patch_size=14)
    else:  # train
        split_acts = act_loader.load_split_tensor('train', dataset_name, model_input_size, patch_size=14)
    
    if split_acts is None:
        raise ValueError(f"Could not load {split_name} activations")
    
    print(f"Loaded split activations shape: {split_acts.shape}")
    
    # Get activations for all concepts for this image's patches
    start_patch = split_idx * patches_per_image
    end_patch = start_patch + patches_per_image
    
    if end_patch > split_acts.shape[0]:
        raise ValueError(f"Image index {split_idx} exceeds available activations in {split_name} split")
    
    # Calculate global patch indices for this image
    global_patch_indices = np.arange(image_idx * patches_per_image, (image_idx + 1) * patches_per_image)
    
    # Filter out padding patches using the filter function
    valid_global_patch_indices = filter_patches_by_image_presence(global_patch_indices, dataset_name, model_input_size)
    valid_global_patch_indices_set = set(valid_global_patch_indices.tolist())
    
    # Try the simpler approach first - directly load the image's activations
    # Use load_concept_range to get activations for this specific image
    global_start_patch = image_idx * patches_per_image
    global_end_patch = global_start_patch + patches_per_image
    
    try:
        # Load activations for this specific image using the global indices
        df = act_loader.load_concept_range(
            concept_names=all_concepts,
            start_idx=global_start_patch,
            end_idx=global_end_patch
        )
        # Convert to numpy array
        image_acts_all_concepts = df.values
        target_concept_acts = image_acts_all_concepts[:, concept_idx]
        
        print(f"Loaded activations using load_concept_range: shape {image_acts_all_concepts.shape}")
        
    except Exception as e:
        print(f"load_concept_range failed: {e}, falling back to split loading")
        
        # Fallback to the split tensor approach
        # Get activations for all concepts for this image's patches
        start_patch = split_idx * patches_per_image
        end_patch = start_patch + patches_per_image
        
        if end_patch > split_acts.shape[0]:
            raise ValueError(f"Image index {split_idx} exceeds available activations in {split_name} split")
        
        image_acts_all_concepts = split_acts[start_patch:end_patch, :].cpu().numpy()
        target_concept_acts = image_acts_all_concepts[:, concept_idx]
        
    # For Llama with padding, mark padded patches as NaN
    has_padding_mask = False
    if model_input_size == (560, 560):
        padding_mask_file = f'GT_Samples/{dataset_name}/patches_w_image_mask_inputsize_{model_input_size}.pt'
        if os.path.exists(padding_mask_file):
            has_padding_mask = True
            full_padding_mask = torch.load(padding_mask_file, weights_only=False)
            image_padding_mask = full_padding_mask[image_idx * patches_per_image:(image_idx + 1) * patches_per_image]
            
            # Set padded patches to NaN
            for patch_idx in range(patches_per_image):
                if not image_padding_mask[patch_idx]:
                    target_concept_acts[patch_idx] = np.nan
    
    # Categorize patches into three groups
    patches_with_target = []
    patches_with_other = []
    patches_with_none = []
    
    # Get patches that contain the target concept
    target_patches = set(gt_patches_per_concept.get(concept, []))
    
    # Get all patches that contain any concept
    all_concept_patches = set()
    for c in all_concepts:
        all_concept_patches.update(gt_patches_per_concept.get(c, []))
    
    # Categorize each patch (only valid patches, not padding)
    for i, global_patch_idx in enumerate(global_patch_indices):
        if global_patch_idx in valid_global_patch_indices_set:  # Only process valid patches
            if global_patch_idx in target_patches:
                patches_with_target.append(i)
            elif global_patch_idx in all_concept_patches:
                patches_with_other.append(i)
            else:
                patches_with_none.append(i)
    
    print(f"\nPatch breakdown for image {image_idx}:")
    print(f"- Total patches: {patches_per_image} (valid: {len(valid_global_patch_indices)}, padding: {patches_per_image - len(valid_global_patch_indices)})")
    print(f"- Patches with '{concept}': {len(patches_with_target)}")
    print(f"- Patches with other concepts: {len(patches_with_other)}")
    print(f"- Patches with no concepts: {len(patches_with_none)}")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    # Top left: Original image
    ax_img = fig.add_subplot(gs[0, 0])
    try:
        img = retrieve_image(image_idx, dataset_name)
        
        # For padded images (Llama), show only the actual image content without padding
        if has_padding_mask and model_input_size == (560, 560):
            from utils.general_utils import get_resized_dims_w_same_ar
            original_size = img.size
            resized_dims = get_resized_dims_w_same_ar(original_size, model_input_size)
            resized_width, resized_height = resized_dims
            
            # Show only the resized image without padding
            resized_img = img.resize(resized_dims)
            ax_img.imshow(resized_img)
            ax_img.set_title(f'Image {image_idx}', fontsize=12)
        else:
            # No padding (CLIP) - show full image
            resized_img = pad_or_resize_img(img, model_input_size)
            ax_img.imshow(resized_img)
            ax_img.set_title(f'Image {image_idx}', fontsize=12)
    except Exception as e:
        ax_img.text(0.5, 0.5, f'Image {image_idx}\n(Error loading: {str(e)})', 
                   ha='center', va='center', transform=ax_img.transAxes)
    ax_img.axis('off')
    
    # Top middle: Heatmap
    ax_heat = fig.add_subplot(gs[0, 1])
    
    # Reshape activations to grid
    heatmap_data = target_concept_acts.reshape(grid_size, grid_size)
    
    # Auto-scale if needed (only on valid patches)
    valid_acts = target_concept_acts[~np.isnan(target_concept_acts)]
    if vmin is None and len(valid_acts) > 0:
        vmin = np.min(valid_acts)
    if vmax is None and len(valid_acts) > 0:
        vmax = np.max(valid_acts)
    
    # Show heatmap with proper cropping for padded images
    try:
        img = retrieve_image(image_idx, dataset_name)
        
        # For padded images (Llama), we need to determine the actual image size
        if has_padding_mask and model_input_size == (560, 560):
            # Get original image size before padding
            from utils.general_utils import get_resized_dims_w_same_ar
            original_size = img.size
            resized_dims = get_resized_dims_w_same_ar(original_size, model_input_size)
            resized_width, resized_height = resized_dims
            
            # Create masked heatmap for visualization
            masked_heatmap = np.ma.masked_where(np.isnan(heatmap_data), heatmap_data)
            
            # Show the padded image with heatmap overlay
            resized_img = pad_or_resize_img(img, model_input_size)
            ax_heat.imshow(resized_img.convert('L'), cmap='gray', alpha=0.3)
            im = ax_heat.imshow(masked_heatmap, cmap=cmap, alpha=0.7, vmin=vmin, vmax=vmax,
                               extent=[0, model_input_size[0], model_input_size[1], 0])
            # Crop to show only the non-padded region
            ax_heat.set_xlim(0, resized_width)
            ax_heat.set_ylim(resized_height, 0)
        else:
            # No padding (CLIP) - show full image
            resized_img = pad_or_resize_img(img, model_input_size)
            ax_heat.imshow(resized_img.convert('L'), cmap='gray', alpha=0.3)
            im = ax_heat.imshow(heatmap_data, cmap=cmap, alpha=0.7, vmin=vmin, vmax=vmax,
                               extent=[0, model_input_size[0], model_input_size[1], 0])
    except:
        im = ax_heat.imshow(heatmap_data, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add patch outlines if requested
    if show_patch_outlines:
        # First, add gray overlay for padding patches
        for i in range(patches_per_image):
            if np.isnan(target_concept_acts[i]):  # This is a padding patch
                row = i // grid_size
                col = i % grid_size
                rect = patches.Rectangle(
                    (col * patch_size_pixels, row * patch_size_pixels),
                    patch_size_pixels, patch_size_pixels,
                    linewidth=0, facecolor='gray', alpha=0.7
                )
                ax_heat.add_patch(rect)
        
        # Then add outlines for patches with concepts
        if gt_patches_per_concept:
            # Green for target concept patches
            for patch_idx in patches_with_target:
                row = patch_idx // grid_size
                col = patch_idx % grid_size
                rect = patches.Rectangle(
                    (col * patch_size_pixels, row * patch_size_pixels),
                    patch_size_pixels, patch_size_pixels,
                    linewidth=2, edgecolor='lime', facecolor='none'
                )
                ax_heat.add_patch(rect)
            
            # Blue for other concept patches
            for patch_idx in patches_with_other:
                row = patch_idx // grid_size
                col = patch_idx % grid_size
                rect = patches.Rectangle(
                    (col * patch_size_pixels, row * patch_size_pixels),
                    patch_size_pixels, patch_size_pixels,
                    linewidth=2, edgecolor='cyan', facecolor='none'
                )
                ax_heat.add_patch(rect)
    
    ax_heat.set_title(f"'{concept}' Activation Heatmap", fontsize=12)
    ax_heat.axis('off')
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax_heat)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(metric_type, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    # Top right: Legend
    ax_legend = fig.add_subplot(gs[0, 2])
    ax_legend.axis('off')
    
    # Create legend patches
    legend_elements = []
    num_padding = patches_per_image - len(valid_global_patch_indices)
    if num_padding > 0:
        legend_elements.append(patches.Patch(facecolor='gray', alpha=0.7,
                                           label=f'Padding ({num_padding} patches)'))
    if len(patches_with_target) > 0:
        legend_elements.append(patches.Patch(facecolor='none', edgecolor='lime', linewidth=2,
                                           label=f"Contains '{concept}' ({len(patches_with_target)} patches)"))
    if len(patches_with_other) > 0:
        legend_elements.append(patches.Patch(facecolor='none', edgecolor='cyan', linewidth=2,
                                           label=f'Contains other concepts ({len(patches_with_other)} patches)'))
    if len(patches_with_none) > 0:
        legend_elements.append(patches.Patch(facecolor='white', edgecolor='black', linewidth=1,
                                           label=f'No concepts ({len(patches_with_none)} patches)'))
    
    if legend_elements:
        ax_legend.legend(handles=legend_elements, loc='center', fontsize=11, frameon=True)
    
    # Add statistics text
    stats_text = f"Image contains:\n"
    concepts_in_image = []
    for c in all_concepts:
        if image_idx in gt_samples_per_concept.get(c, []):
            concepts_in_image.append(c)
    
    if concepts_in_image:
        stats_text += f"{len(concepts_in_image)} concepts:\n"
        for i, c in enumerate(concepts_in_image[:5]):  # Show first 5
            if c == concept:
                stats_text += f"• {c} (target)\n"
            else:
                stats_text += f"• {c}\n"
        if len(concepts_in_image) > 5:
            stats_text += f"• ... and {len(concepts_in_image) - 5} more"
    else:
        stats_text += "No annotated concepts"
    
    ax_legend.text(0.5, 0.3, stats_text, transform=ax_legend.transAxes,
                  fontsize=10, ha='center', va='top')
    
    # Bottom row: Three distribution plots
    
    # Load best detection threshold if available
    best_threshold = None
    threshold_file = None
    
    # Construct the threshold file path based on concept type
    if concept_type == 'avg_patch_embeddings':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_avg_patch_embeddings_percentthrumodel_{percent_thru_model}.pt'
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}.pt'
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
    
    if threshold_file and os.path.exists(threshold_file):
        try:
            thresholds_data = torch.load(threshold_file, weights_only=False)
            if concept in thresholds_data:
                best_threshold = thresholds_data[concept]['best_threshold']
                print(f"Loaded detection threshold for '{concept}': {best_threshold:.3f}")
        except Exception as e:
            print(f"Warning: Could not load threshold from {threshold_file}: {e}")
    
    # Calculate common bin edges for all distributions (excluding NaN values)
    valid_acts_for_range = target_concept_acts[~np.isnan(target_concept_acts)]
    if len(valid_acts_for_range) > 0:
        hist_min = np.min(valid_acts_for_range)
        hist_max = np.max(valid_acts_for_range)
    else:
        hist_min, hist_max = -1, 1
    bins = np.linspace(hist_min, hist_max, n_bins)
    
    # Import for KDE
    from scipy.stats import gaussian_kde
    
    # Calculate all densities first to find max y-value for consistent scaling
    max_density = 0
    densities = {}
    
    # Pre-calculate densities for all distributions
    x_range = np.linspace(hist_min, hist_max, 200)
    
    if len(patches_with_target) > 0:
        acts_with_target = target_concept_acts[patches_with_target]
        kde_target = gaussian_kde(acts_with_target)
        density_target = kde_target(x_range)
        # Verify normalization
        dx = x_range[1] - x_range[0]
        integral = np.trapz(density_target, dx=dx)
        if abs(integral - 1.0) > 0.1:
            density_target = density_target / integral
        densities['target'] = (acts_with_target, density_target)
        max_density = max(max_density, density_target.max())
    
    if len(patches_with_other) > 0:
        acts_with_other = target_concept_acts[patches_with_other]
        kde_other = gaussian_kde(acts_with_other)
        density_other = kde_other(x_range)
        dx = x_range[1] - x_range[0]
        integral = np.trapz(density_other, dx=dx)
        if abs(integral - 1.0) > 0.1:
            density_other = density_other / integral
        densities['other'] = (acts_with_other, density_other)
        max_density = max(max_density, density_other.max())
    
    if len(patches_with_none) > 0:
        acts_with_none = target_concept_acts[patches_with_none]
        kde_none = gaussian_kde(acts_with_none)
        density_none = kde_none(x_range)
        dx = x_range[1] - x_range[0]
        integral = np.trapz(density_none, dx=dx)
        if abs(integral - 1.0) > 0.1:
            density_none = density_none / integral
        densities['none'] = (acts_with_none, density_none)
        max_density = max(max_density, density_none.max())
    
    # Add some margin to the max density
    y_max = max_density * 1.1
    
    # Distribution 1: Patches with target concept
    ax_dist1 = fig.add_subplot(gs[1, 0])
    if 'target' in densities:
        acts_with_target, density = densities['target']
        ax_dist1.plot(x_range, density, color='green', linewidth=2)
        ax_dist1.fill_between(x_range, density, alpha=0.3, color='green')
        
        # Create twin axis for counts
        ax_dist1_twin = ax_dist1.twinx()
        counts, bin_edges, _ = ax_dist1_twin.hist(acts_with_target, bins=30, density=False,
                                                  alpha=0.4, color='green', edgecolor='black', linewidth=0.5)
        
        mean_val = np.mean(acts_with_target)
        ax_dist1.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_val:.3f}')
        
        # Add superdetector threshold line
        if best_threshold is not None:
            ax_dist1.axvline(best_threshold, color='purple', linestyle='-', linewidth=2.5,
                            label=f'Superdetector: {best_threshold:.3f}')
        
        ax_dist1.set_title(f"Patches WITH '{concept}'\n({len(patches_with_target)} patches)", fontsize=11)
        
        # Compute and display variation metrics
        metrics = compute_distribution_metrics(acts_with_target)
        metric_text = f"σ={metrics['std']:.3f}, IQR={metrics['iqr']:.3f}, CV={metrics['cv']:.3f}"
        ax_dist1.text(0.02, 0.98, metric_text, transform=ax_dist1.transAxes,
                     fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='white', alpha=0.7))
        
        ax_dist1_twin.set_ylabel('Patch Count', fontsize=9)
    else:
        ax_dist1.text(0.5, 0.5, f"No patches contain\n'{concept}'", 
                     ha='center', va='center', transform=ax_dist1.transAxes, fontsize=12)
        ax_dist1.set_title(f"Patches WITH '{concept}'", fontsize=11)
        # Create empty twin axis
        ax_dist1_twin = ax_dist1.twinx()
        ax_dist1_twin.set_ylim(0, 1)
    
    ax_dist1.set_xlabel('Activation Value', fontsize=10)
    ax_dist1.set_ylabel('Probability Density', fontsize=10)
    ax_dist1.grid(True, alpha=0.3, axis='x')  # Only show x-axis grid
    ax_dist1.set_xlim(hist_min, hist_max)
    ax_dist1.set_ylim(0, y_max)
    
    # Distribution 2: Patches with other concepts
    ax_dist2 = fig.add_subplot(gs[1, 1])
    if 'other' in densities:
        acts_with_other, density = densities['other']
        ax_dist2.plot(x_range, density, color='blue', linewidth=2)
        ax_dist2.fill_between(x_range, density, alpha=0.3, color='blue')
        
        mean_val = np.mean(acts_with_other)
        ax_dist2.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_val:.3f}')
        
        # Add superdetector threshold line
        if best_threshold is not None:
            ax_dist2.axvline(best_threshold, color='purple', linestyle='-', linewidth=2.5,
                            label=f'Superdetector: {best_threshold:.3f}')
        
        ax_dist2.set_title(f'Patches with OTHER concepts\n({len(patches_with_other)} patches)', fontsize=11)
        
        # Compute and display variation metrics
        metrics = compute_distribution_metrics(acts_with_other)
        metric_text = f"σ={metrics['std']:.3f}, IQR={metrics['iqr']:.3f}, CV={metrics['cv']:.3f}"
        ax_dist2.text(0.02, 0.98, metric_text, transform=ax_dist2.transAxes,
                     fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='white', alpha=0.7))
    else:
        ax_dist2.text(0.5, 0.5, 'No patches contain\nother concepts', 
                     ha='center', va='center', transform=ax_dist2.transAxes, fontsize=12)
        ax_dist2.set_title('Patches with OTHER concepts', fontsize=11)
    
    ax_dist2.set_xlabel('Activation Value', fontsize=10)
    ax_dist2.set_ylabel('Density', fontsize=10)
    ax_dist2.grid(True, alpha=0.3, axis='x')  # Only show x-axis grid
    ax_dist2.set_xlim(hist_min, hist_max)
    ax_dist2.set_ylim(0, y_max)
    
    # Distribution 3: Patches with no concepts
    ax_dist3 = fig.add_subplot(gs[1, 2])
    if 'none' in densities:
        acts_with_none, density = densities['none']
        ax_dist3.plot(x_range, density, color='gray', linewidth=2)
        ax_dist3.fill_between(x_range, density, alpha=0.3, color='gray')
        
        mean_val = np.mean(acts_with_none)
        ax_dist3.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_val:.3f}')
        
        # Add superdetector threshold line
        if best_threshold is not None:
            ax_dist3.axvline(best_threshold, color='purple', linestyle='-', linewidth=2.5,
                            label=f'Superdetector: {best_threshold:.3f}')
        
        ax_dist3.set_title(f'Patches with NO concepts\n({len(patches_with_none)} patches)', fontsize=11)
        
        # Compute and display variation metrics
        metrics = compute_distribution_metrics(acts_with_none)
        metric_text = f"σ={metrics['std']:.3f}, IQR={metrics['iqr']:.3f}, CV={metrics['cv']:.3f}"
        ax_dist3.text(0.02, 0.98, metric_text, transform=ax_dist3.transAxes,
                     fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='white', alpha=0.7))
    else:
        ax_dist3.text(0.5, 0.5, 'All patches contain\nsome concept', 
                     ha='center', va='center', transform=ax_dist3.transAxes, fontsize=12)
        ax_dist3.set_title('Patches with NO concepts', fontsize=11)
    
    ax_dist3.set_xlabel('Activation Value', fontsize=10)
    ax_dist3.set_ylabel('Density', fontsize=10)
    ax_dist3.grid(True, alpha=0.3, axis='x')  # Only show x-axis grid
    ax_dist3.set_xlim(hist_min, hist_max)
    ax_dist3.set_ylim(0, y_max)
    
    # Add superdetector threshold to all plots if we're showing empty plots too
    if best_threshold is not None:
        for ax in [ax_dist1, ax_dist2, ax_dist3]:
            # Check if the plot doesn't already have the threshold line
            if not any('Superdetector' in line.get_label() for line in ax.get_lines()):
                ax.axvline(best_threshold, color='purple', linestyle='-', linewidth=2.5,
                          label=f'Superdetector: {best_threshold:.3f}', alpha=0.5)
    
    # Overall title
    fig.suptitle(f"Concept '{concept}' Activation Analysis for Image {image_idx}\n"
                f"Dataset: {dataset_name}, Model: {model_name}, Concept Type: {concept_type}",
                fontsize=14, y=0.98)
    
    # Add a single legend at the top right for mean and superdetector threshold
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Mean'),
        Line2D([0], [0], color='purple', linestyle='-', linewidth=2.5, label='Superdetector Threshold')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), 
               fontsize=10, frameon=True, fancybox=True, shadow=True,
               borderpad=0.3, columnspacing=0.5, handletextpad=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    # Print summary statistics
    print(f"\nActivation Statistics for '{concept}':")
    if len(patches_with_target) > 0:
        acts_target = target_concept_acts[patches_with_target]
        mean_target = np.mean(acts_target)
        std_target = np.std(acts_target)
        cv_target = std_target / abs(mean_target) if mean_target != 0 else float('inf')
        print(f"- Patches WITH target: mean={mean_target:.3f}, std={std_target:.3f}, CV={cv_target:.3f}")
    if len(patches_with_other) > 0:
        acts_other = target_concept_acts[patches_with_other]
        mean_other = np.mean(acts_other)
        std_other = np.std(acts_other)
        cv_other = std_other / abs(mean_other) if mean_other != 0 else float('inf')
        print(f"- Patches WITH other: mean={mean_other:.3f}, std={std_other:.3f}, CV={cv_other:.3f}")
    if len(patches_with_none) > 0:
        acts_none = target_concept_acts[patches_with_none]
        mean_none = np.mean(acts_none)
        std_none = np.std(acts_none)
        cv_none = std_none / abs(mean_none) if mean_none != 0 else float('inf')
        print(f"- Patches WITH none:  mean={mean_none:.3f}, std={std_none:.3f}, CV={cv_none:.3f}")
    
    plt.show()
    
    return fig


def analyze_text_concept_activations(
    sample_idx: int,
    concept: str,
    dataset_name: str,
    model_name: str,
    concept_type: str,
    model_input_size: Optional[Tuple[str, str]] = None,
    percent_thru_model: int = 100,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 10),
    cmap: str = 'coolwarm',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    scratch_dir: str = '',
    n_bins: int = 30
) -> plt.Figure:
    """
    Analyze and visualize concept activations for a specific text sample.
    
    Args:
        sample_idx: Global index of the text sample to analyze
        concept: Target concept to analyze
        dataset_name: Name of dataset (e.g. 'GoEmotions')
        model_name: Name of model (e.g. 'Llama')
        concept_type: Type of concept
        model_input_size: Model input size (e.g. ('text', 'text'))
        percent_thru_model: Percentage through model
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap for token coloring
        vmin/vmax: Value range for colormap
        scratch_dir: Directory where activation files are stored
        n_bins: Number of bins for histograms
        
    Returns:
        matplotlib Figure object
    """
    from matplotlib.gridspec import GridSpec
    from utils.memory_management_utils import ChunkedActivationLoader
    from utils.text_visualization_utils import (
        get_glob_tok_indices_from_sent_idx, 
        get_color_for_sim,
        retrieve_sentence,
        clean_text_spacing
    )
    from scipy.stats import gaussian_kde
    import matplotlib.cm as cm
    
    # Import filter utility
    from utils.filter_datasets_utils import filter_concept_dict
    
    # Determine model input size
    if model_input_size is None:
        if model_name == 'Llama':
            model_input_size = ('text', 'text')
        elif model_name == 'Gemma':
            model_input_size = ('text', 'text2')
        elif model_name == 'Qwen':
            model_input_size = ('text', 'text3')
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    # Get activation file name
    sample_type = 'patch'  # Even for text, the files use 'patch'
    if concept_type == 'avg_patch_embeddings':
        acts_file = f"cosine_similarities_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        acts_file = f"dists_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    # Load activation loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
    
    # Load ground truth samples per concept
    gt_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
    gt_samples_per_concept = torch.load(gt_file, weights_only=False)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    # Load ground truth tokens per concept
    gt_tokens_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
    if os.path.exists(gt_tokens_file):
        gt_tokens_per_concept = torch.load(gt_tokens_file, weights_only=False)
        gt_tokens_per_concept = filter_concept_dict(gt_tokens_per_concept, dataset_name)
    else:
        gt_tokens_per_concept = {}
    
    # Load tokens
    tokens_file = f"GT_Samples/{dataset_name}/tokens_inputsize_{model_input_size}.pt"
    tokens_list = torch.load(tokens_file, weights_only=False)
    
    # Get tokens for this sample and clean them
    raw_tokens = tokens_list[sample_idx]
    clean_tokens = [token.replace("Ġ", "") for token in raw_tokens]  # Remove GPT tokenizer spacing marker
    
    # Get original text
    original_text = retrieve_sentence(sample_idx, dataset_name)
    if not original_text:
        # Fallback to joined tokens
        original_text = " ".join([t for t in clean_tokens if t and t != "[EMPTY]"])
    
    # Calculate global token indices
    start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sample_idx, tokens_list)
    
    # Load activations for this sample
    all_concepts = list(gt_samples_per_concept.keys())
    if concept not in all_concepts:
        raise ValueError(f"Concept '{concept}' not found. Available concepts: {all_concepts}")
    
    concept_idx = act_loader.get_concept_index(concept)
    
    # Load activations for all concepts at once
    sentence_acts = act_loader.load_tensor_range(start_idx, end_idx).cpu().numpy()
    token_acts = sentence_acts[:, concept_idx]
    
    # Auto-scale if needed
    if vmin is None:
        vmin = np.min(token_acts)
    if vmax is None:
        vmax = np.max(token_acts)
    
    # Get ground truth tokens for this concept
    tokens_with_target = set()
    if concept in gt_tokens_per_concept:
        # Check which tokens in our range are in the ground truth
        for i, global_idx in enumerate(range(start_idx, end_idx)):
            if global_idx in gt_tokens_per_concept[concept]:
                tokens_with_target.add(i)
    
    # Get tokens with any concept
    tokens_with_any_concept = set()
    for c in all_concepts:
        if c in gt_tokens_per_concept:
            for i, global_idx in enumerate(range(start_idx, end_idx)):
                if global_idx in gt_tokens_per_concept[c]:
                    tokens_with_any_concept.add(i)
    
    # Categorize tokens
    tokens_with_other = tokens_with_any_concept - tokens_with_target
    tokens_with_none = set(range(len(raw_tokens))) - tokens_with_any_concept
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    # Top: Text visualization with colored tokens as HTML-like display
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis('off')
    
    # Create colormap
    colormap = cm.get_cmap(cmap)
    
    # Title
    ax_text.text(0.5, 0.98, f"Text Sample {sample_idx}: Concept '{concept}' Heatmap", 
                fontsize=14, fontweight='bold',
                ha='center', va='top', transform=ax_text.transAxes)
    
    # Build HTML-style token visualization
    # Use matplotlib's text rendering similar to plot_all_concept_activations_on_sentence
    x_pos = 0.02
    y_pos = 0.85
    line_height = 0.06
    max_width = 0.96
    
    # Process tokens in groups to handle wrapping
    for i, (token, act) in enumerate(zip(clean_tokens, token_acts)):
        if not token or token == "[EMPTY]":
            continue
            
        # Get color for this token based on activation
        color_rgb = get_color_for_sim(act, vmin, vmax, colormap)
        # Convert RGB string to matplotlib color
        import re
        rgb_match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color_rgb)
        if rgb_match:
            r, g, b = [int(x)/255 for x in rgb_match.groups()]
            bg_color = (r, g, b)
        else:
            bg_color = 'white'
        
        # Add star marker if token contains target concept
        display_token = token
        if i in tokens_with_target:
            display_token = token + "★"
        
        # Create text with colored background
        token_text = ax_text.text(x_pos, y_pos, f" {display_token} ", 
                                 transform=ax_text.transAxes,
                                 fontsize=11,
                                 bbox=dict(boxstyle="round,pad=0.3", 
                                         facecolor=bg_color, 
                                         alpha=0.8,
                                         edgecolor='gray',
                                         linewidth=0.5))
        
        # Get text extent to calculate position
        renderer = fig.canvas.get_renderer()
        bbox = token_text.get_window_extent(renderer=renderer)
        inv = ax_text.transAxes.inverted()
        bbox_data = inv.transform(bbox)
        token_width = bbox_data[1][0] - bbox_data[0][0]
        
        # Update position
        x_pos += token_width + 0.005
        
        # Wrap to next line if needed
        if x_pos > max_width:
            x_pos = 0.02
            y_pos -= line_height
    
    # Add colorbar to the right side
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.25])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(metric_type, fontsize=10)
    
    # Find max activation token for display
    if len(token_acts) > 0:
        max_idx = np.argmax(token_acts)
        max_token = clean_tokens[max_idx] if max_idx < len(clean_tokens) else "?"
        max_act = token_acts[max_idx]
        
        # Add max token info
        ax_text.text(0.02, 0.02, f"Max activation: {max_act:.3f} at token '{max_token}'",
                    transform=ax_text.transAxes, fontsize=10, style='italic')
    
    # Load best detection threshold if available
    best_threshold = None
    threshold_file = None
    
    # Construct the threshold file path based on concept type
    if concept_type == 'avg_patch_embeddings':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_avg_patch_embeddings_percentthrumodel_{percent_thru_model}.pt'
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}.pt'
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
    
    if threshold_file and os.path.exists(threshold_file):
        try:
            thresholds_data = torch.load(threshold_file, weights_only=False)
            if concept in thresholds_data:
                best_threshold = thresholds_data[concept]['best_threshold']
                print(f"Loaded detection threshold for '{concept}': {best_threshold:.3f}")
        except Exception as e:
            print(f"Warning: Could not load threshold from {threshold_file}: {e}")
    
    # Bottom: Three histograms
    # Distribution 1: Tokens with target concept
    ax_dist1 = fig.add_subplot(gs[1, 0])
    if len(tokens_with_target) > 0:
        acts_with_target = token_acts[list(tokens_with_target)]
        
        # Create smooth density curve using KDE
        kde = gaussian_kde(acts_with_target)
        x_range = np.linspace(vmin, vmax, 200)
        density = kde(x_range)
        ax_dist1.plot(x_range, density, color='green', linewidth=2)
        ax_dist1.fill_between(x_range, density, alpha=0.3, color='green')
        
        mean_val = np.mean(acts_with_target)
        ax_dist1.axvline(mean_val, color='red', linestyle='--', linewidth=2)
        
        # Add superdetector threshold line
        if best_threshold is not None:
            ax_dist1.axvline(best_threshold, color='purple', linestyle='-', linewidth=2.5)
        
        ax_dist1.set_title(f"Tokens WITH '{concept}'\n({len(tokens_with_target)} tokens)", fontsize=11)
        
        # Display metrics
        metrics = compute_distribution_metrics(acts_with_target)
        metric_text = f"μ={mean_val:.3f}, σ={metrics['std']:.3f}, CV={metrics['cv']:.3f}"
        ax_dist1.text(0.02, 0.98, metric_text, transform=ax_dist1.transAxes,
                     fontsize=8, verticalalignment='top', 
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    else:
        ax_dist1.text(0.5, 0.5, f"No tokens contain\n'{concept}'", 
                     ha='center', va='center', transform=ax_dist1.transAxes, fontsize=12)
        ax_dist1.set_title(f"Tokens WITH '{concept}'", fontsize=11)
    
    ax_dist1.set_xlabel('Activation Value', fontsize=10)
    ax_dist1.set_ylabel('Density', fontsize=10)
    ax_dist1.grid(True, alpha=0.3)
    ax_dist1.set_xlim(vmin, vmax)
    
    # Distribution 2: Tokens with other concepts
    ax_dist2 = fig.add_subplot(gs[1, 1])
    if len(tokens_with_other) > 0:
        acts_with_other = token_acts[list(tokens_with_other)]
        
        kde = gaussian_kde(acts_with_other)
        density = kde(x_range)
        ax_dist2.plot(x_range, density, color='blue', linewidth=2)
        ax_dist2.fill_between(x_range, density, alpha=0.3, color='blue')
        
        mean_val = np.mean(acts_with_other)
        ax_dist2.axvline(mean_val, color='red', linestyle='--', linewidth=2)
        
        # Add superdetector threshold line
        if best_threshold is not None:
            ax_dist2.axvline(best_threshold, color='purple', linestyle='-', linewidth=2.5)
        
        ax_dist2.set_title(f'Tokens with OTHER concepts\n({len(tokens_with_other)} tokens)', fontsize=11)
        
        metrics = compute_distribution_metrics(acts_with_other)
        metric_text = f"μ={mean_val:.3f}, σ={metrics['std']:.3f}, CV={metrics['cv']:.3f}"
        ax_dist2.text(0.02, 0.98, metric_text, transform=ax_dist2.transAxes,
                     fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    else:
        ax_dist2.text(0.5, 0.5, 'No tokens contain\nother concepts', 
                     ha='center', va='center', transform=ax_dist2.transAxes, fontsize=12)
        ax_dist2.set_title('Tokens with OTHER concepts', fontsize=11)
    
    ax_dist2.set_xlabel('Activation Value', fontsize=10)
    ax_dist2.set_ylabel('Density', fontsize=10)
    ax_dist2.grid(True, alpha=0.3)
    ax_dist2.set_xlim(vmin, vmax)
    
    # Distribution 3: Tokens with no concepts
    ax_dist3 = fig.add_subplot(gs[1, 2])
    if len(tokens_with_none) > 0:
        acts_with_none = token_acts[list(tokens_with_none)]
        
        kde = gaussian_kde(acts_with_none)
        density = kde(x_range)
        ax_dist3.plot(x_range, density, color='gray', linewidth=2)
        ax_dist3.fill_between(x_range, density, alpha=0.3, color='gray')
        
        mean_val = np.mean(acts_with_none)
        ax_dist3.axvline(mean_val, color='red', linestyle='--', linewidth=2)
        
        # Add superdetector threshold line
        if best_threshold is not None:
            ax_dist3.axvline(best_threshold, color='purple', linestyle='-', linewidth=2.5)
        
        ax_dist3.set_title(f'Tokens with NO concepts\n({len(tokens_with_none)} tokens)', fontsize=11)
        
        metrics = compute_distribution_metrics(acts_with_none)
        metric_text = f"μ={mean_val:.3f}, σ={metrics['std']:.3f}, CV={metrics['cv']:.3f}"
        ax_dist3.text(0.02, 0.98, metric_text, transform=ax_dist3.transAxes,
                     fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    else:
        ax_dist3.text(0.5, 0.5, 'All tokens contain\nsome concept', 
                     ha='center', va='center', transform=ax_dist3.transAxes, fontsize=12)
        ax_dist3.set_title('Tokens with NO concepts', fontsize=11)
    
    ax_dist3.set_xlabel('Activation Value', fontsize=10)
    ax_dist3.set_ylabel('Density', fontsize=10)
    ax_dist3.grid(True, alpha=0.3)
    ax_dist3.set_xlim(vmin, vmax)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Mean'),
        Line2D([0], [0], color='purple', linestyle='-', linewidth=2.5, label='Superdetector Threshold'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=12, label='GT Token')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
               fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Overall title
    fig.suptitle(f"Concept '{concept}' Token-Level Analysis\n"
                f"Dataset: {dataset_name}, Model: {model_name}",
                fontsize=14, y=0.99)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.97])
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    # Print summary
    print(f"\nToken-level analysis for '{concept}':")
    print(f"- Total tokens: {len(raw_tokens)}")
    print(f"- Tokens WITH target concept: {len(tokens_with_target)}")
    print(f"- Tokens WITH other concepts: {len(tokens_with_other)}")
    print(f"- Tokens WITH no concepts: {len(tokens_with_none)}")
    print(f"- Activation range: [{vmin:.3f}, {vmax:.3f}]")
    
    plt.show()
    
    return fig


def analyze_concept_activations_global(
    concepts: Union[str, List[str]],
    dataset_name: str,
    model_name: str,
    concept_type: str,
    sim_concepts: Optional[Union[List[str], List[List[str]]]] = None,
    model_input_size: Optional[Tuple[int, int]] = None,
    percent_thru_model: int = 100,
    save_path: Optional[str] = None,
    figsize_per_row: Tuple[float, float] = (20, 5),
    cmap: str = 'hot',
    scratch_dir: str = '',
    n_bins: int = 50
) -> plt.Figure:
    """
    Analyze concept activations across ALL test images with four distribution categories:
    1. Patches that contain the target concept
    2. Patches that contain OTHER concepts from images WITHOUT the target concept
    3. Patches that contain semantically similar concepts from images WITHOUT the target concept
    4. Non-Concept patches (no concepts) from images WITHOUT the target concept
    
    Args:
        concepts: Target concept(s) to analyze - can be a single concept string or list of concepts
        dataset_name: Name of dataset (e.g. 'CLEVR')
        model_name: Name of model (e.g. 'CLIP')
        concept_type: Type of concept - must be one of:
            'avg_patch_embeddings'
            'linsep_patch_embeddings_BD_True_BN_False'
            'kmeans_1000_patch_embeddings_kmeans'
            'kmeans_1000_linsep_patch_embeddings_kmeans'
        sim_concepts: List of semantically similar concepts (for single concept) or 
                     list of lists (one list per concept in concepts)
        model_input_size: Model input size (will be inferred if None)
        percent_thru_model: Percentage through model
        save_path: Path to save figure
        figsize_per_row: Figure size per row (width, height)
        cmap: Colormap for heatmap
        scratch_dir: Directory where activation files are stored
        n_bins: Number of bins for histograms
        
    Returns:
        matplotlib Figure object
    """
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    from utils.general_utils import retrieve_image, get_split_index_from_global_index, pad_or_resize_img
    from utils.memory_management_utils import ChunkedActivationLoader
    from utils.patch_alignment_utils import filter_patches_by_image_presence
    from scipy.stats import gaussian_kde
    
    # Determine dataset type and model input size
    is_text_dataset = dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank', 'IMDB', 'Jailbreak']
    
    if model_input_size is None:
        if is_text_dataset:
            # Set model input size based on text model type
            if model_name == 'Llama':
                model_input_size = ('text', 'text')
            elif model_name == 'Gemma':
                model_input_size = ('text', 'text2')
            elif model_name == 'Qwen':
                model_input_size = ('text', 'text3')
            else:
                raise ValueError(f"Unknown text model: {model_name}")
        else:
            # Image models
            if model_name == 'CLIP':
                model_input_size = (224, 224)
            elif model_name == 'Llama':
                model_input_size = (560, 560)
            else:
                raise ValueError(f"Unknown image model: {model_name}")
    
    # Calculate patches/tokens per sample
    if is_text_dataset:
        # For text, we'll load token counts dynamically
        units_per_sample = None  # Will be loaded from file
        unit_type = 'token'
    else:
        # For images, calculate patches per image
        if model_input_size == (224, 224):
            units_per_sample = 256  # 16x16
        elif model_input_size == (560, 560):
            units_per_sample = 1600  # 40x40
        else:
            units_per_sample = 256
        unit_type = 'patch'
    
    # Get activation file name based on concept type
    sample_type = 'patch'  # Both text and image datasets use 'patch' in filenames
    if concept_type == 'avg_patch_embeddings':
        acts_file = f"cosine_similarities_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        acts_file = f"dists_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"cosine_similarities_kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    # Load activation loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
    except FileNotFoundError:
        try:
            act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir='.', device=device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Activation file not found: {acts_file}")
    
    # Import filter utility
    from utils.filter_datasets_utils import filter_concept_dict
    
    # Load ground truth samples per concept (sample-level) - TEST ONLY
    gt_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
    print(f"Loading ground truth from: {gt_file}")
    gt_samples_per_concept = torch.load(gt_file, weights_only=False)
    # Filter to only include valid concepts for this dataset
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    # Load ground truth patches/tokens per concept (unit-level)
    # Note: For text datasets, the file might be named gt_patch_per_concept instead of gt_patches_per_concept
    gt_patches_file = f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
    if not os.path.exists(gt_patches_file):
        # Try alternative naming convention for text datasets
        gt_patches_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
    print(f"Loading {unit_type}-level ground truth from: {gt_patches_file}")
    
    if not os.path.exists(gt_patches_file):
        print(f"Warning: Patch-level ground truth not found at {gt_patches_file}")
        print("Will only be able to show image-level concept presence")
        gt_patches_per_concept = {}
    else:
        gt_patches_per_concept = torch.load(gt_patches_file, weights_only=False)
        # Filter to only include valid concepts for this dataset
        gt_patches_per_concept = filter_concept_dict(gt_patches_per_concept, dataset_name)
    
    # Handle single concept or list of concepts
    if isinstance(concepts, str):
        concepts = [concepts]
    
    # Get all concept names
    all_concepts = list(gt_samples_per_concept.keys())
    
    # Validate all requested concepts exist
    concept_indices = {}
    for concept in concepts:
        if concept not in all_concepts:
            raise ValueError(f"Concept '{concept}' not found. Available concepts: {all_concepts[:10]}...")
        concept_indices[concept] = all_concepts.index(concept)
    
    # Load test activations - we need all of them
    print("Loading all test activations...")
    test_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
    if test_acts is None:
        raise ValueError("Could not load test activations")
    
    # Get number of test samples and units
    num_test_units = test_acts.shape[0]
    
    # Get all test sample global indices to map units correctly
    import pandas as pd
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    test_global_indices = metadata[metadata['split'] == 'test'].index.tolist()
    print(f"Found {len(test_global_indices)} test samples")
    
    if is_text_dataset:
        # Load token counts for text datasets - these are stored with model_input_size in the filename
        token_count_file = f"GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt"
        
        if os.path.exists(token_count_file):
            all_token_counts = torch.load(token_count_file, weights_only=False)
            
            # Extract test split token counts based on test_global_indices
            tokens_per_sample = []
            for test_idx in test_global_indices:
                if test_idx < len(all_token_counts):
                    token_count_list = all_token_counts[test_idx]
                    # Sum the token counts for all words in this sample
                    if isinstance(token_count_list, list):
                        total_tokens = sum(token_count_list)
                        tokens_per_sample.append(total_tokens)
                    else:
                        # In case it's already a single number
                        tokens_per_sample.append(token_count_list)
                else:
                    raise ValueError(f"Test index {test_idx} out of bounds for token counts (length {len(all_token_counts)})")
            
            num_test_samples = len(tokens_per_sample)
            total_test_tokens = sum(tokens_per_sample)
            print(f"Analyzing {num_test_samples} test texts ({total_test_tokens} tokens total)")
            
            # Verify token counts match activation size
            if total_test_tokens != num_test_units:
                print(f"WARNING: Token count mismatch! Expected {total_test_tokens} tokens but have {num_test_units} activations")
        else:
            raise ValueError(f"Token count file not found: {token_count_file}")
    else:
        num_test_samples = num_test_units // units_per_sample
        print(f"Analyzing {num_test_samples} test images ({num_test_units} patches)")
    
    # We'll process each concept separately in the loop below
    
    # Get patches that contain any concept (will be used for all concepts)
    patches_with_any_concept = set()
    for c in all_concepts:
        patches_with_any_concept.update(gt_patches_per_concept.get(c, []))
    
    # Handle sim_concepts format - convert single list to list of lists if needed
    if sim_concepts is not None:
        if isinstance(concepts, str):
            # Single concept - sim_concepts should be a simple list
            if sim_concepts and isinstance(sim_concepts[0], list):
                raise ValueError("For a single concept, sim_concepts should be a simple list, not a list of lists")
            sim_concepts_processed = [sim_concepts]
        else:
            # Multiple concepts
            if sim_concepts and not isinstance(sim_concepts[0], list):
                raise ValueError("For multiple concepts, sim_concepts should be a list of lists (one per concept)")
            if len(sim_concepts) != len(concepts):
                raise ValueError(f"sim_concepts must have the same length as concepts. Got {len(sim_concepts)} lists for {len(concepts)} concepts")
            sim_concepts_processed = sim_concepts
    else:
        sim_concepts_processed = [None] * len(concepts) if isinstance(concepts, list) else [None]
    
    # Create a mapping from global sample index to test position
    global_to_test_pos = {global_idx: i for i, global_idx in enumerate(test_global_indices)}
    
    # Vectorized mask creation
    print(f"Building {unit_type} masks with vectorized operations...")
    
    if is_text_dataset:
        # For text datasets, we need to map tokens to their samples
        total_units = num_test_units
        all_test_unit_indices = torch.arange(total_units, device=device)
        
        # Create mapping from token to sample
        test_sample_indices = torch.zeros(total_units, dtype=torch.long, device=device)
        current_idx = 0
        for sample_idx, token_count in enumerate(tokens_per_sample):
            test_sample_indices[current_idx:current_idx + token_count] = sample_idx
            current_idx += token_count
        
        # All tokens are valid for text
        is_valid_patch = torch.ones(total_units, dtype=torch.bool, device=device)
    else:
        # For image datasets
        total_units = num_test_samples * units_per_sample
        all_test_unit_indices = torch.arange(total_units, device=device)
        
        # Vectorized computation of which test image each patch belongs to (0-indexed within test set)
        test_sample_indices = torch.div(all_test_unit_indices, units_per_sample, rounding_mode='floor')
        
        # Load padding mask if it exists
        padding_mask_file = f'GT_Samples/{dataset_name}/patches_w_image_mask_inputsize_{model_input_size}.pt'
        if os.path.exists(padding_mask_file):
            print("Loading pre-computed padding mask...")
            full_padding_mask = torch.load(padding_mask_file, weights_only=False)
            # Extract only test portion of padding mask
            test_padding_mask = []
            for test_global_idx in test_global_indices:
                start_patch = test_global_idx * units_per_sample
                end_patch = start_patch + units_per_sample
                test_padding_mask.append(full_padding_mask[start_patch:end_patch])
            is_valid_patch = torch.cat(test_padding_mask).to(device).bool()
        else:
            print("No padding mask found, assuming all patches are valid...")
            is_valid_patch = torch.ones(total_units, dtype=torch.bool, device=device)
    
    # The processing logic has been moved to the per-concept loop below
    
    # Print metrics explanation before creating the figure
    print("\n" + "="*60)
    print("METRICS EXPLANATION:")
    print("="*60)
    print(f"n     = number of {unit_type}s")
    print("μ     = mean activation value")
    print("σ     = standard deviation (spread of the distribution)")
    print("IQR   = interquartile range (75th - 25th percentile)")
    print("\nVISUAL INDICATORS:")
    print("Red dashed line   = mean value")
    print("Purple solid line = superdetector threshold")
    print("="*60 + "\n")
    
    # Load thresholds for all concepts
    threshold_file = None
    if concept_type == 'avg_patch_embeddings':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_avg_patch_embeddings_percentthrumodel_{percent_thru_model}.pt'
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}.pt'
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
    
    thresholds_data = {}
    if threshold_file and os.path.exists(threshold_file):
        try:
            thresholds_data = torch.load(threshold_file, weights_only=False)
            print(f"Loaded detection thresholds from {threshold_file}")
        except Exception as e:
            print(f"Warning: Could not load threshold from {threshold_file}: {e}")
    
    # Figure size - one row per concept, 4 distributions per row
    n_concepts = len(concepts)
    fig = plt.figure(figsize=(figsize_per_row[0], figsize_per_row[1] * n_concepts))
    gs = GridSpec(n_concepts, 4, wspace=0.3, hspace=0.5)
    
    # Process each concept
    for concept_idx, concept in enumerate(concepts):
        print(f"\nProcessing concept: {concept}")
        
        # Get concept index for activations
        concept_act_idx = concept_indices[concept]
        concept_acts_gpu = test_acts[:, concept_act_idx]  # Keep on GPU
        
        # Get samples that contain the target concept (these are already test-specific indices)
        samples_with_target = set(gt_samples_per_concept.get(concept, []))
        
        # Get units (patches/tokens) that contain the target concept (these are global indices)
        units_with_target = set(gt_patches_per_concept.get(concept, []))
        
        # Get units that contain semantically similar concepts for this concept
        units_with_sim_concepts = set()
        current_sim_concepts = sim_concepts_processed[concept_idx]
        if current_sim_concepts:
            for sim_c in current_sim_concepts:
                if sim_c in gt_patches_per_concept:
                    units_with_sim_concepts.update(gt_patches_per_concept.get(sim_c, []))
        
        # Convert samples with target concept to tensor
        samples_with_target_list = list(samples_with_target)
        if samples_with_target_list:
            samples_with_target_tensor = torch.tensor(samples_with_target_list, device=device)
            is_from_target_sample = torch.isin(test_sample_indices, samples_with_target_tensor)
        else:
            is_from_target_sample = torch.zeros(total_units, dtype=torch.bool, device=device)
        
        # Create masks for unit-level ground truth
        is_target_unit = torch.zeros(total_units, dtype=torch.bool, device=device)
        is_other_concept_unit = torch.zeros(total_units, dtype=torch.bool, device=device)
        is_sim_concept_unit = torch.zeros(total_units, dtype=torch.bool, device=device)
        
        if is_text_dataset:
            # For text datasets, gt_patches_per_concept contains GLOBAL token indices
            # We need to map these to test-specific token indices
            
            # First, calculate the global token range for each test sample
            global_token_start_per_sample = {}
            current_global_idx = 0
            
            # Calculate global token positions for ALL samples (not just test)
            for sample_idx in range(len(all_token_counts)):
                global_token_start_per_sample[sample_idx] = current_global_idx
                current_global_idx += sum(all_token_counts[sample_idx])
            
            # Now map ground truth to test token positions
            current_test_token_idx = 0
            for test_pos, global_sample_idx in enumerate(test_global_indices):
                sample_token_count = tokens_per_sample[test_pos]
                global_start = global_token_start_per_sample[global_sample_idx]
                global_end = global_start + sample_token_count
                
                # Check each token in this sample
                for i in range(sample_token_count):
                    global_token_idx = global_start + i
                    test_token_idx = current_test_token_idx + i
                    
                    if global_token_idx in units_with_target:
                        is_target_unit[test_token_idx] = True
                    elif global_token_idx in units_with_sim_concepts:
                        is_sim_concept_unit[test_token_idx] = True
                    elif global_token_idx in patches_with_any_concept:
                        is_other_concept_unit[test_token_idx] = True
                        
                current_test_token_idx += sample_token_count
        else:
            # For image datasets, process patches
            for test_pos, global_img_idx in enumerate(test_global_indices):
                # Calculate global patch range for this image
                global_patch_start = global_img_idx * units_per_sample
                global_patch_end = global_patch_start + units_per_sample
                
                # Calculate test patch range
                test_patch_start = test_pos * units_per_sample
                test_patch_end = test_patch_start + units_per_sample
                
                # Check each patch in this image
                for i in range(units_per_sample):
                    global_patch_idx = global_patch_start + i
                    test_patch_idx = test_patch_start + i
                    
                    if global_patch_idx in units_with_target:
                        is_target_unit[test_patch_idx] = True
                    elif global_patch_idx in units_with_sim_concepts:
                        is_sim_concept_unit[test_patch_idx] = True
                    elif global_patch_idx in patches_with_any_concept:
                        is_other_concept_unit[test_patch_idx] = True
        
        # Extract activations for the 4 categories
        # 1. Units containing target concept
        mask_1 = is_valid_patch & is_target_unit
        acts_1_target = concept_acts_gpu[mask_1].cpu().numpy()
        
        # 2. OTHER concept units from samples WITHOUT target
        mask_2 = is_valid_patch & ~is_from_target_sample & is_other_concept_unit
        acts_2_other_no_target = concept_acts_gpu[mask_2].cpu().numpy()
        
        # 3. Semantically similar concept units from samples WITHOUT target
        mask_3 = is_valid_patch & ~is_from_target_sample & is_sim_concept_unit
        acts_3_sim_no_target = concept_acts_gpu[mask_3].cpu().numpy()
        
        # 4. Non-Concept units from samples WITHOUT target
        mask_4 = is_valid_patch & ~is_from_target_sample & ~is_target_unit & ~is_other_concept_unit & ~is_sim_concept_unit
        acts_4_bg_no_target = concept_acts_gpu[mask_4].cpu().numpy()
        
        print(f"{unit_type.capitalize()} counts for '{concept}':")
        print(f"  1. {unit_type.capitalize()}s WITH concept: {len(acts_1_target)}")
        print(f"  2. OTHER concepts from {unit_type}s WITHOUT concept: {len(acts_2_other_no_target)}")
        if current_sim_concepts:
            print(f"  3. Semantically similar concepts from {unit_type}s WITHOUT concept: {len(acts_3_sim_no_target)}")
        else:
            print(f"  3. Semantically similar concepts: Not specified")
        print(f"  4. Non-Concept from {unit_type}s WITHOUT concept: {len(acts_4_bg_no_target)}")
        
        # Get threshold for this concept
        best_threshold = None
        if concept in thresholds_data:
            best_threshold = thresholds_data[concept]['best_threshold']
            print(f"  Detection threshold: {best_threshold:.3f}")
        
        # Determine activation range for this concept
        all_acts = np.concatenate([
            acts_1_target if len(acts_1_target) > 0 else [],
            acts_2_other_no_target if len(acts_2_other_no_target) > 0 else [],
            acts_3_sim_no_target if len(acts_3_sim_no_target) > 0 else [],
            acts_4_bg_no_target if len(acts_4_bg_no_target) > 0 else []
        ])
        
        if len(all_acts) == 0:
            continue
            
        hist_min = np.min(all_acts)
        hist_max = np.max(all_acts)
        x_range = np.linspace(hist_min, hist_max, 200)
        
        # Calculate all densities first to find max y-value for consistent scaling
        max_density = 0
        densities = {}
        
        # Pre-calculate densities for all distributions
        # gaussian_kde already normalizes to integrate to 1
        if len(acts_1_target) > 0:
            kde_1 = gaussian_kde(acts_1_target)
            density_1 = kde_1(x_range)
            # Verify normalization (integral should be ~1)
            dx = x_range[1] - x_range[0]
            integral_1 = np.trapz(density_1, dx=dx)
            if abs(integral_1 - 1.0) > 0.1:  # If not normalized, normalize it
                density_1 = density_1 / integral_1
            densities['target'] = density_1
            max_density = max(max_density, density_1.max())
        
        if len(acts_2_other_no_target) > 0:
            kde_2 = gaussian_kde(acts_2_other_no_target)
            density_2 = kde_2(x_range)
            dx = x_range[1] - x_range[0]
            integral_2 = np.trapz(density_2, dx=dx)
            if abs(integral_2 - 1.0) > 0.1:
                density_2 = density_2 / integral_2
            densities['other'] = density_2
            max_density = max(max_density, density_2.max())
        
        if len(acts_3_sim_no_target) > 0:
            kde_3 = gaussian_kde(acts_3_sim_no_target)
            density_3 = kde_3(x_range)
            dx = x_range[1] - x_range[0]
            integral_3 = np.trapz(density_3, dx=dx)
            if abs(integral_3 - 1.0) > 0.1:
                density_3 = density_3 / integral_3
            densities['sim'] = density_3
            max_density = max(max_density, density_3.max())
        
        if len(acts_4_bg_no_target) > 0:
            kde_4 = gaussian_kde(acts_4_bg_no_target)
            density_4 = kde_4(x_range)
            dx = x_range[1] - x_range[0]
            integral_4 = np.trapz(density_4, dx=dx)
            if abs(integral_4 - 1.0) > 0.1:
                density_4 = density_4 / integral_4
            densities['bg'] = density_4
            max_density = max(max_density, density_4.max())
        
        # Add some margin to the max density
        y_max = max_density * 1.1
        
        # Format sim_concepts label
        if current_sim_concepts:
            sim_label = f"Semantically similar\n({', '.join(current_sim_concepts[:2])}{'...' if len(current_sim_concepts) > 2 else ''})"
        else:
            sim_label = "Semantically similar\n(none specified)"
        
        # Four distributions for this concept
        sample_type_label = "texts" if is_text_dataset else "images"
        distributions = [
            (acts_1_target, f"{unit_type.capitalize()}s WITH\n'{concept}'", 'green', 0, 'target'),
            (acts_2_other_no_target, f"OTHER concepts from\n{sample_type_label} WITHOUT '{concept}'", 'blue', 1, 'other'),
            (acts_3_sim_no_target, sim_label, 'orange', 2, 'sim'),
            (acts_4_bg_no_target, f"Non-Concept from\n{sample_type_label} WITHOUT '{concept}'", 'gray', 3, 'bg')
        ]
        
        for acts, title, color, col, key in distributions:
            ax = fig.add_subplot(gs[concept_idx, col])
            
            if key in densities:
                # Use pre-calculated density
                density = densities[key]
                ax.plot(x_range, density, color=color, linewidth=2)
                ax.fill_between(x_range, density, alpha=0.3, color=color)
                
                # Add mean line
                mean_val = np.mean(acts)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2)
                
                # Add superdetector threshold line
                if best_threshold is not None:
                    ax.axvline(best_threshold, color='purple', linestyle='-', linewidth=2.5)
                
                # Create twin axis for counts
                ax2 = ax.twinx()
                
                # Create histogram on the twin axis (for counts)
                counts, bin_edges, _ = ax2.hist(acts, bins=50, density=False, 
                                               alpha=0.4, color=color, edgecolor='black', linewidth=0.5)
                
                # Set y-axis labels
                if col == 0:
                    ax.set_ylabel('Probability Density', fontsize=9, fontweight='bold')
                else:
                    ax.set_ylabel('Probability Density', fontsize=9)
                
                if col == 3:  # Rightmost column
                    ax2.set_ylabel(f'{unit_type.capitalize()} Count', fontsize=9, fontweight='bold')
                else:
                    ax2.set_ylabel(f'{unit_type.capitalize()} Count', fontsize=9)
                
                # Compute and display variation metrics
                metrics = compute_distribution_metrics(acts)
                metric_text = f"n={len(acts)}\nμ={mean_val:.3f}\nσ={metrics['std']:.3f}\nIQR={metrics['iqr']:.3f}\nCV={metrics['cv']:.3f}"
                ax.text(0.02, 0.98, metric_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                # Create empty twin axis to maintain consistency
                ax2 = ax.twinx()
                ax2.set_ylim(0, 1)
            
            # Add row label for first column
            if col == 0 and concept_idx == 0:
                ax.text(-0.3, 0.5, f"Concept: '{concept}'", transform=ax.transAxes,
                        fontsize=11, fontweight='bold', rotation=90, va='center')
            
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Activation Value', fontsize=9)
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_xlim(hist_min, hist_max)
            ax.set_ylim(0, y_max)
            
            # Format tick labels for readability
            ax.tick_params(axis='y', labelsize=8)
            ax2.tick_params(axis='y', labelsize=8)
    
    # No overall title as requested
    
    # Add a legend/key for threshold lines at top right
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Mean'),
        Line2D([0], [0], color='purple', linestyle='-', linewidth=2.5, label='Superdetector Threshold')
    ]
    
    # Adjust legend position based on number of concepts
    legend_y = 0.95 - (0.05 * (n_concepts - 1))
    legend_ax = fig.add_axes([0.88, legend_y, 0.11, 0.04])
    legend_ax.axis('off')
    legend_ax.legend(handles=legend_elements, loc='center', ncol=1, fontsize=9, 
                     title='Lines', title_fontsize=10, frameon=True,
                     fancybox=True, shadow=True)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig


def analyze_concept_std_distributions_global(
    concepts: Union[str, List[str]],
    dataset_name: str,
    model_name: str,
    concept_type: str,
    sim_concepts: Optional[Union[List[str], List[List[str]]]] = None,
    model_input_size: Optional[Tuple[int, int]] = None,
    percent_thru_model: int = 100,
    save_path: Optional[str] = None,
    figsize_per_row: Tuple[float, float] = (20, 5),
    scratch_dir: str = '',
    n_bins: int = 50,
    std_range: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """
    Analyze standard deviation distributions across samples for concept activations.
    Computes std per image/text sample for four distribution categories:
    1. Samples where patches/tokens contain the target concept
    2. Samples with OTHER concepts (excluding target concept samples)
    3. Samples with semantically similar concepts (excluding target concept samples)
    4. Non-Concept samples with no concepts (excluding target concept samples)
    
    Args:
        concepts: Target concept(s) to analyze - can be a single concept string or list of concepts
        dataset_name: Name of dataset (e.g. 'CLEVR')
        model_name: Name of model (e.g. 'CLIP')
        concept_type: Type of concept - must be one of:
            'avg_patch_embeddings'
            'linsep_patch_embeddings_BD_True_BN_False'
            'kmeans_1000_patch_embeddings_kmeans'
            'kmeans_1000_linsep_patch_embeddings_kmeans'
        sim_concepts: List of semantically similar concepts (for single concept) or 
                     list of lists (one list per concept in concepts)
        model_input_size: Model input size (will be inferred if None)
        percent_thru_model: Percentage through model
        save_path: Path to save figure
        figsize_per_row: Figure size per row (width, height)
        scratch_dir: Directory where activation files are stored
        n_bins: Number of bins for std histograms
        std_range: Range for std values (min, max) - if None, will be auto-determined
        
    Returns:
        matplotlib Figure object showing std distributions
    """
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    from utils.memory_management_utils import ChunkedActivationLoader
    from utils.patch_alignment_utils import filter_patches_by_image_presence
    from scipy.stats import gaussian_kde
    from collections import defaultdict
    
    # Ensure concepts is a list
    if isinstance(concepts, str):
        concepts = [concepts]
        if sim_concepts and not isinstance(sim_concepts[0], list):
            sim_concepts = [sim_concepts]
    
    n_concepts = len(concepts)
    
    # Determine dataset type and model input size
    is_text_dataset = dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank', 'IMDB', 'Jailbreak']
    
    if model_input_size is None:
        if is_text_dataset:
            if model_name == 'Llama':
                model_input_size = ('text', 'text')
            elif model_name == 'Gemma':
                model_input_size = ('text', 'text2')
            elif model_name == 'Qwen':
                model_input_size = ('text', 'text3')
            else:
                raise ValueError(f"Unknown text model: {model_name}")
        else:
            if model_name == 'CLIP':
                model_input_size = (224, 224)
            elif model_name == 'Llama':
                model_input_size = (560, 560)
            else:
                raise ValueError(f"Unknown image model: {model_name}")
    
    # Calculate patches/tokens per sample
    if is_text_dataset:
        unit_type = 'token'
        # For text datasets, we need to load token counts
        # Try multiple possible locations
        possible_paths = [
            f"./Data/{dataset_name}/Embeddings/num_patches_per_sentence_test.pt",
            f"SCRATCH_DIR/Data/{dataset_name}/Embeddings/num_patches_per_sentence_test.pt",
            f"Data/{dataset_name}/Embeddings/num_patches_per_sentence_test.pt",
            f"./Data/{dataset_name}/Embeddings/num_tokens_per_sentence_test.pt",
            f"SCRATCH_DIR/Data/{dataset_name}/Embeddings/num_tokens_per_sentence_test.pt",
            f"Data/{dataset_name}/Embeddings/num_tokens_per_sentence_test.pt"
        ]
        
        token_file_found = False
        for token_file in possible_paths:
            if os.path.exists(token_file):
                tokens_per_sample = torch.load(token_file)
                token_file_found = True
                print(f"Loaded token counts from: {token_file}")
                break
        
        if not token_file_found:
            # If no file found, we need to estimate or skip
            print(f"Warning: Could not find token count file for {dataset_name}")
            print("Attempting to estimate from activation data...")
            # We'll handle this after loading the activation loader
            tokens_per_sample = None
    else:
        if model_input_size == (224, 224):
            patches_per_sample = 256  # 16x16
        elif model_input_size == (560, 560):
            patches_per_sample = 1600  # 40x40
        else:
            patches_per_sample = 256
        unit_type = 'patch'
    
    # Get activation file name based on concept type
    sample_type = 'patch'
    if concept_type == 'avg_patch_embeddings':
        acts_file = f"cosine_similarities_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        acts_file = f"dists_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"cosine_similarities_kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    # Load ground truth samples
    gt_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
    
    print(f"Loading ground truth from: {gt_file}")
    gt_samples_per_concept = torch.load(gt_file)
    
    # Load ground truth patches/tokens if available
    gt_patches_per_concept = None
    if not is_text_dataset:
        gt_patches_file = f"GT_Samples/{dataset_name}/gt_patches_per_concept_test_inputsize_{model_input_size}.pt"
        if os.path.exists(gt_patches_file):
            print(f"Loading ground truth patches from: {gt_patches_file}")
            gt_patches_per_concept = torch.load(gt_patches_file)
    
    # Load concepts to get indices
    # Handle different concept type formats
    if concept_type == 'avg_patch_embeddings':
        concepts_filename = f"avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        concepts_filename = f"linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        concepts_filename = f"kmeans_1000_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        concepts_filename = f"kmeans_1000_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    else:
        # Fallback to old logic
        concepts_filename = f"{concept_type.replace('_patch_embeddings', '')}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    
    concepts_file = f"Concepts/{dataset_name}/{concepts_filename}"
    
    if os.path.exists(concepts_file):
        concepts_data = torch.load(concepts_file, weights_only=False)
        if isinstance(concepts_data, dict):
            all_concept_names = list(concepts_data.keys())
        else:
            raise ValueError(f"Unexpected format in concepts file: {concepts_file}")
    else:
        raise FileNotFoundError(f"Concepts file not found: {concepts_file}")
    
    # Load activation loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
    except FileNotFoundError:
        try:
            act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir='.', device=device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Activation file not found: {acts_file}")
    
    # Get total samples
    act_info = act_loader.get_activation_info()
    total_patches = act_info['total_samples']
    
    if is_text_dataset:
        if tokens_per_sample is not None:
            total_samples = len(tokens_per_sample)
        else:
            # Estimate from ground truth data
            # Get the maximum sample index to determine total samples
            max_sample_idx = 0
            for sample_list in gt_samples_per_concept.values():
                if sample_list:
                    max_sample_idx = max(max_sample_idx, max(sample_list))
            
            total_samples = max_sample_idx + 1  # Assuming 0-based indexing
            
            # Create a dummy tokens_per_sample assuming uniform distribution
            # This is not accurate but allows the function to proceed
            avg_tokens_per_sample = total_patches // total_samples if total_samples > 0 else 100
            tokens_per_sample = torch.full((total_samples,), avg_tokens_per_sample, dtype=torch.long)
            print(f"Warning: Using estimated uniform token distribution")
            print(f"Estimated {total_samples} samples with average {avg_tokens_per_sample} tokens per sample")
            print(f"Results may not be accurate for variable-length text samples!")
    else:
        total_samples = total_patches // patches_per_sample
    
    # Create figure
    fig = plt.figure(figsize=(figsize_per_row[0], figsize_per_row[1] * n_concepts))
    gs = GridSpec(n_concepts, 4, figure=fig, hspace=0.3, wspace=0.25)
    
    # Process each concept
    for concept_idx, concept in enumerate(concepts):
        print(f"\nProcessing concept '{concept}'...")
        
        if concept not in all_concept_names:
            print(f"Warning: Concept '{concept}' not found in concepts file")
            continue
        
        concept_index = all_concept_names.index(concept)
        
        # Get current sim_concepts for this concept
        if sim_concepts and concept_idx < len(sim_concepts):
            current_sim_concepts = sim_concepts[concept_idx]
        else:
            current_sim_concepts = []
        
        # Get samples with and without the target concept
        samples_with_target = set(gt_samples_per_concept.get(concept, []))
        samples_without_target = set(range(total_samples)) - samples_with_target
        
        print(f"  Samples with '{concept}': {len(samples_with_target)}")
        print(f"  Samples without '{concept}': {len(samples_without_target)}")
        
        # Initialize std collections for each category
        stds_1_target = []  # stds from samples WITH target concept
        stds_2_other_no_target = []  # stds from samples with OTHER concepts but NOT target
        stds_3_sim_no_target = []  # stds from samples with SIMILAR concepts but NOT target
        stds_4_bg_no_target = []  # stds from background samples (no concepts) but NOT target
        
        # Process samples in batches
        batch_size = 100 if is_text_dataset else 50
        
        # Process samples WITH target concept
        print(f"  Computing stds for samples with target concept...")
        
        # Batch process samples for efficiency
        target_samples_list = sorted(list(samples_with_target))
        batch_size = 50  # Process 50 samples at a time
        
        for batch_start in tqdm(range(0, len(target_samples_list), batch_size), desc="Target batches"):
            batch_end = min(batch_start + batch_size, len(target_samples_list))
            batch_samples = target_samples_list[batch_start:batch_end]
            
            # Calculate indices for all samples in batch
            all_unit_indices = []
            sample_boundaries = [0]  # Track where each sample starts
            
            if is_text_dataset:
                # For text samples, we need to handle variable length sequences
                if tokens_per_sample is not None:
                    # Vectorized computation for text samples
                    cumsum_tokens = torch.cumsum(torch.cat([torch.zeros(1), tokens_per_sample]), dim=0)
                    for sample_idx in batch_samples:
                        start_token = int(cumsum_tokens[sample_idx])
                        end_token = int(cumsum_tokens[sample_idx + 1])
                        unit_indices = list(range(start_token, end_token))
                        all_unit_indices.extend(unit_indices)
                        sample_boundaries.append(len(all_unit_indices))
                else:
                    # If we don't have exact token counts, skip this batch
                    print(f"Warning: Skipping batch due to missing token counts")
                    continue
            else:
                # Vectorized computation for image patches
                for sample_idx in batch_samples:
                    start_patch = sample_idx * patches_per_sample
                    end_patch = start_patch + patches_per_sample
                    unit_indices = list(range(start_patch, end_patch))
                    all_unit_indices.extend(unit_indices)
                    sample_boundaries.append(len(all_unit_indices))
            
            if len(all_unit_indices) > 0:
                # Load all activations for batch at once
                min_idx = min(all_unit_indices)
                max_idx = max(all_unit_indices) + 1
                
                # Load the range of activations - stays on GPU if available
                chunk_acts = act_loader.load_chunk_range(min_idx, max_idx)
                
                # Extract just the concept column we need
                if len(chunk_acts.shape) == 2:
                    all_activations = chunk_acts[:, concept_index]
                else:
                    all_activations = chunk_acts
                
                # Get only the rows we need (vectorized)
                relative_indices = torch.tensor([idx - min_idx for idx in all_unit_indices], device=all_activations.device)
                batch_activations = all_activations[relative_indices]
                
                # Calculate CVs for each sample in batch (vectorized)
                for i in range(len(batch_samples)):
                    start_idx = sample_boundaries[i]
                    end_idx = sample_boundaries[i + 1]
                    
                    if end_idx > start_idx:
                        sample_acts = batch_activations[start_idx:end_idx]
                        
                        # Calculate std on GPU
                        std_act = torch.std(sample_acts, unbiased=False).item()
                        stds_1_target.append(std_act)
        
        # Process samples WITHOUT target concept
        print(f"  Computing stds for samples without target concept...")
        
        # Categorize samples without target
        other_concept_samples = []
        similar_concept_samples = []
        no_concept_samples = []
        
        for sample_idx in samples_without_target:
            # Check what concepts this sample has
            sample_concepts = []
            for c in all_concept_names:
                if sample_idx in gt_samples_per_concept.get(c, []):
                    sample_concepts.append(c)
            
            if not sample_concepts:
                no_concept_samples.append(sample_idx)
            else:
                # Check if it has similar concepts
                has_similar = any(c in current_sim_concepts for c in sample_concepts)
                if has_similar:
                    similar_concept_samples.append(sample_idx)
                else:
                    other_concept_samples.append(sample_idx)
        
        # Process each category
        for samples_list, stds_list, category_name in [
            (other_concept_samples, stds_2_other_no_target, "other concepts"),
            (similar_concept_samples, stds_3_sim_no_target, "similar concepts"),
            (no_concept_samples, stds_4_bg_no_target, "background")
        ]:
            print(f"  Processing {category_name} samples ({len(samples_list)} samples)...")
            
            # Limit samples for efficiency but process in batches
            samples_to_process = samples_list[:500]
            batch_size = 50
            
            for batch_start in tqdm(range(0, len(samples_to_process), batch_size), desc=f"{category_name} batches"):
                batch_end = min(batch_start + batch_size, len(samples_to_process))
                batch_samples = samples_to_process[batch_start:batch_end]
                
                # Calculate indices for all samples in batch
                all_unit_indices = []
                sample_boundaries = [0]
                
                if is_text_dataset:
                    # For text samples, we need to handle variable length sequences
                    if tokens_per_sample is not None:
                        # Vectorized computation for text samples
                        cumsum_tokens = torch.cumsum(torch.cat([torch.zeros(1), tokens_per_sample]), dim=0)
                        for sample_idx in batch_samples:
                            start_token = int(cumsum_tokens[sample_idx])
                            end_token = int(cumsum_tokens[sample_idx + 1])
                            unit_indices = list(range(start_token, end_token))
                            all_unit_indices.extend(unit_indices)
                            sample_boundaries.append(len(all_unit_indices))
                    else:
                        # If we don't have exact token counts, skip this batch
                        print(f"Warning: Skipping batch due to missing token counts")
                        continue
                else:
                    # Vectorized computation for image patches
                    for sample_idx in batch_samples:
                        start_patch = sample_idx * patches_per_sample
                        end_patch = start_patch + patches_per_sample
                        unit_indices = list(range(start_patch, end_patch))
                        all_unit_indices.extend(unit_indices)
                        sample_boundaries.append(len(all_unit_indices))
                
                if len(all_unit_indices) > 0:
                    # Load all activations for batch at once
                    min_idx = min(all_unit_indices)
                    max_idx = max(all_unit_indices) + 1
                    
                    # Load the range of activations - stays on GPU if available
                    chunk_acts = act_loader.load_chunk_range(min_idx, max_idx)
                    
                    # Extract just the concept column we need
                    if len(chunk_acts.shape) == 2:
                        all_activations = chunk_acts[:, concept_index]
                    else:
                        all_activations = chunk_acts
                    
                    # Get only the rows we need (vectorized)
                    relative_indices = torch.tensor([idx - min_idx for idx in all_unit_indices], device=all_activations.device)
                    batch_activations = all_activations[relative_indices]
                    
                    # Calculate CVs for each sample in batch (vectorized)
                    for i in range(len(batch_samples)):
                        start_idx = sample_boundaries[i]
                        end_idx = sample_boundaries[i + 1]
                        
                        if end_idx > start_idx:
                            sample_acts = batch_activations[start_idx:end_idx]
                            
                            # Calculate std on GPU
                            std_act = torch.std(sample_acts, unbiased=False).item()
                            stds_list.append(std_act)
        
        # Determine std range for plotting
        all_stds = stds_1_target + stds_2_other_no_target + stds_3_sim_no_target + stds_4_bg_no_target
        
        if std_range is None and len(all_stds) > 0:
            std_min = 0  # std is always non-negative
            std_max = np.percentile(all_stds, 95)  # Use 95th percentile to avoid extreme outliers
        elif std_range is not None:
            std_min, std_max = std_range
        else:
            std_min, std_max = 0, 1
        
        x_range = np.linspace(std_min, std_max, 200)
        
        # Format sim_concepts label
        if current_sim_concepts:
            sim_label = f"Semantically similar\n({', '.join(current_sim_concepts[:2])}{'...' if len(current_sim_concepts) > 2 else ''})"
        else:
            sim_label = "Semantically similar\n(none specified)"
        
        # Four std distributions for this concept
        sample_type_label = "texts" if is_text_dataset else "images"
        distributions = [
            (stds_1_target, f"{sample_type_label.capitalize()} WITH\n'{concept}'", 'green', 0),
            (stds_2_other_no_target, f"OTHER concepts from\n{sample_type_label} WITHOUT '{concept}'", 'blue', 1),
            (stds_3_sim_no_target, sim_label, 'orange', 2),
            (stds_4_bg_no_target, f"Non-Concept from\n{sample_type_label} WITHOUT '{concept}'", 'gray', 3)
        ]
        
        for stds, title, color, col in distributions:
            ax = fig.add_subplot(gs[concept_idx, col])
            
            if len(stds) > 0:
                # Create smooth density curve using KDE
                kde = gaussian_kde(stds)
                density = kde(x_range)
                ax.plot(x_range, density, color=color, linewidth=2)
                ax.fill_between(x_range, density, alpha=0.3, color=color)
                
                # Add statistics
                mean_std = np.mean(stds)
                median_std = np.median(stds)
                ax.axvline(mean_std, color='red', linestyle='--', linewidth=2)
                ax.axvline(median_std, color='purple', linestyle=':', linewidth=2)
                
                # Display statistics
                stats_text = f"n={len(stds)}\nμ={mean_std:.3f}\nmed={median_std:.3f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
            
            # Add row label for first column
            if col == 0:
                ax.set_ylabel(f"Concept: '{concept}'", fontsize=11, fontweight='bold')
            
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Standard Deviation', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(std_min, std_max)
    
    # Add legend at top right
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Mean Std'),
        Line2D([0], [0], color='purple', linestyle=':', linewidth=2, label='Median Std')
    ]
    
    # Place legend in top right using figure coordinates
    plt.figlegend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(0.99, 0.99), bbox_transform=fig.transFigure,
                  fontsize=9, title='Statistics', title_fontsize=10, 
                  frameon=True, fancybox=True, shadow=True)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_concept_heatmaps_with_distributions_advanced(
    concept: str,
    dataset_name: str,
    model_name: str,
    concept_type: str,
    image_indices: Optional[List[int]] = None,
    n_samples: int = 5,
    start_idx: int = 0,
    model_input_size: Tuple = None,
    percent_thru_model: int = 100,
    save_path: Optional[str] = None,
    figsize_per_image: Tuple[float, float] = (3, 6),
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_image_labels: bool = True,
    show_gt_patches: bool = False,
    gt_patches_per_concept: Optional[Dict] = None,
    scratch_dir: str = ''
):
    """
    Advanced version of plot_concept_heatmaps_with_distributions with more options.
    
    Additional Args:
        image_indices: Specific test image indices to visualize (overrides n_samples/start_idx)
        show_image_labels: Whether to show image indices in titles
        show_gt_patches: Whether to outline ground truth patches that contain the concept
        gt_patches_per_concept: Ground truth patch labels (required if show_gt_patches=True)
        scratch_dir: Directory where activation files are stored
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches
    from utils.general_utils import retrieve_image
    from utils.memory_management_utils import ChunkedActivationLoader
    
    # Determine model input size
    if model_input_size is None:
        if dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank']:
            print("This visualization is designed for image datasets")
            return
        else:
            if model_name == 'CLIP':
                model_input_size = (224, 224)
            elif model_name == 'Llama':
                model_input_size = (560, 560)
            else:
                raise ValueError(f"Unknown model: {model_name}")
    
    # Get concept label and activation file based on full concept type
    if concept_type == 'avg_patch_embeddings':
        con_label = f'{model_name}_avg_patch_embeddings_percentthrumodel_{percent_thru_model}'
        acts_file = f"cosine_similarities_avg_concepts_{model_name}_patch_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        con_label = f'{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}'
        acts_file = f"dists_linsep_concepts_BD_True_BN_False_{model_name}_patch_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        con_label = f"{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
        acts_file = f"cosine_similarities_kmeans_1000_concepts_{model_name}_patch_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        con_label = f"{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
        acts_file = f"dists_kmeans_1000_linsep_concepts_{model_name}_patch_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    # Load activation loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use scratch_dir from function parameter
    
    try:
        act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
    except FileNotFoundError:
        try:
            act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir='.', device=device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Activation file not found: {acts_file}")
    
    # Load concepts to get concept index
    sample_type = 'patch'  # For visualization, we're always using patch-level
    concepts_file = f"Concepts/{dataset_name}/avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    if not os.path.exists(concepts_file):
        concepts_file = f"Concepts/{dataset_name}/{concept_type}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    
    concept_idx = 0  # Default
    if os.path.exists(concepts_file):
        concepts_data = torch.load(concepts_file, weights_only=False)
        if isinstance(concepts_data, dict):
            concept_names = list(concepts_data.keys())
            if concept in concept_names:
                concept_idx = concept_names.index(concept)
    
    # Determine which images to show
    if image_indices is None:
        test_indices = list(range(start_idx, start_idx + n_samples))
    else:
        test_indices = image_indices
        n_samples = len(image_indices)
    
    # Calculate patches per image
    if model_input_size == (224, 224):
        patches_per_image = 256  # 16x16
        grid_size = 16
    elif model_input_size == (560, 560):
        patches_per_image = 1600  # 40x40
        grid_size = 40
    else:
        patches_per_image = 256
        grid_size = 16
    
    # Load ground truth patches if requested
    gt_patches_for_concept = None
    if show_gt_patches and gt_patches_per_concept is None:
        gt_patches_file = f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
        if os.path.exists(gt_patches_file):
            gt_patches_per_concept = torch.load(gt_patches_file, weights_only=False)
            gt_patches_for_concept = set(gt_patches_per_concept.get(concept, []))
    elif show_gt_patches and gt_patches_per_concept:
        gt_patches_for_concept = set(gt_patches_per_concept.get(concept, []))
    
    # Create figure
    fig = plt.figure(figsize=(n_samples * figsize_per_image[0], figsize_per_image[1]))
    gs = gridspec.GridSpec(2, n_samples, height_ratios=[1, 1], hspace=0.3, wspace=0.2)
    
    # Load test activations
    test_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
    if test_acts is None:
        raise ValueError("Could not load test activations")
    
    concept_acts = test_acts[:, concept_idx].cpu().numpy()
    
    # Auto-scale if needed
    if vmin is None or vmax is None:
        sample_acts = []
        for idx in test_indices:
            start_patch = idx * patches_per_image
            end_patch = start_patch + patches_per_image
            if end_patch <= len(concept_acts):
                sample_acts.extend(concept_acts[start_patch:end_patch])
        
        if sample_acts:
            if vmin is None:
                vmin = np.percentile(sample_acts, 5)
            if vmax is None:
                vmax = np.percentile(sample_acts, 95)
    
    all_patch_acts = []
    
    # Process each test image
    for i, img_idx in enumerate(test_indices):
        start_patch = img_idx * patches_per_image
        end_patch = start_patch + patches_per_image
        
        if end_patch > len(concept_acts):
            print(f"Warning: Image {img_idx} exceeds available activations")
            continue
        
        patch_acts = concept_acts[start_patch:end_patch]
        all_patch_acts.append(patch_acts)
        
        # Reshape to grid
        heatmap_data = patch_acts.reshape(grid_size, grid_size)
        
        # Top row: Heatmap
        ax_heat = fig.add_subplot(gs[0, i])
        
        # Load and display image
        try:
            img = retrieve_image(img_idx, dataset_name, model_input_size)
            ax_heat.imshow(img, aspect='auto')
            im = ax_heat.imshow(heatmap_data, cmap=cmap, alpha=0.6, 
                               vmin=vmin, vmax=vmax, aspect='auto')
        except:
            im = ax_heat.imshow(heatmap_data, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Add ground truth patch outlines if requested
        if show_gt_patches and gt_patches_for_concept:
            patch_size_pixels = model_input_size[0] / grid_size
            for patch_idx in range(patches_per_image):
                global_patch_idx = start_patch + patch_idx
                if global_patch_idx in gt_patches_for_concept:
                    row = patch_idx // grid_size
                    col = patch_idx % grid_size
                    rect = patches.Rectangle(
                        (col * patch_size_pixels, row * patch_size_pixels),
                        patch_size_pixels, patch_size_pixels,
                        linewidth=2, edgecolor='lime', facecolor='none'
                    )
                    ax_heat.add_patch(rect)
        
        # Title
        if show_image_labels:
            ax_heat.set_title(f'Test Image {img_idx}', fontsize=10)
        else:
            ax_heat.set_title(f'Image {i+1}', fontsize=10)
        ax_heat.axis('off')
        
        # Colorbar for first image
        if i == 0:
            cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.set_label(metric_type, fontsize=8)
            cbar.ax.tick_params(labelsize=8)
    
    # Common bins for distributions
    all_acts_flat = np.concatenate(all_patch_acts)
    hist_min = np.min(all_acts_flat)
    hist_max = np.max(all_acts_flat)
    bins = np.linspace(hist_min, hist_max, 50)
    
    # Bottom row: Distributions
    max_height = 0
    for i, patch_acts in enumerate(all_patch_acts):
        ax_dist = fig.add_subplot(gs[1, i])
        
        counts, _, _ = ax_dist.hist(patch_acts, bins=bins, alpha=0.7, 
                                   color='steelblue', edgecolor='black')
        max_height = max(max_height, np.max(counts))
        
        # Statistics
        mean_val = np.mean(patch_acts)
        median_val = np.median(patch_acts)
        ax_dist.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_val:.3f}')
        ax_dist.axvline(median_val, color='green', linestyle='--', linewidth=2,
                       label=f'Median: {median_val:.3f}')
        
        ax_dist.set_xlabel('Activation Value', fontsize=10)
        if i == 0:
            ax_dist.set_ylabel('Count', fontsize=10)
        ax_dist.grid(True, alpha=0.3)
        ax_dist.legend(fontsize=8, loc='upper right')
    
    # Align y-axes
    for i in range(len(all_patch_acts)):
        ax_dist = fig.axes[n_samples + i]
        ax_dist.set_ylim(0, max_height * 1.1)
        ax_dist.set_xlim(hist_min, hist_max)
    
    # Title
    title = f'Concept "{concept}" - {metric_type} Heatmaps and Patch Activation Distributions\n'
    title += f'Dataset: {dataset_name}, Model: {model_name}, Concept Type: {concept_type}'
    if show_gt_patches:
        title += ' (Green outlines show GT patches with concept)'
    fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def analyze_concept_activation_overlay(
    concept: str,
    dataset_name: str,
    model_name: str,
    concept_type: str,
    similar_concepts: Optional[List[str]] = None,
    model_input_size: Optional[Tuple[int, int]] = None,
    percent_thru_model: int = 100,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    scratch_dir: str = '',
    n_bins: int = 50
) -> plt.Figure:
    """
    Analyze and overlay concept activation distributions.
    Shows:
    - Patches/tokens containing the target concept
    - If similar_concepts provided: patches/tokens containing similar concepts
    - If not: background patches/tokens from samples that don't contain the target concept at all
    
    Args:
        concept: Target concept to analyze
        dataset_name: Name of dataset (e.g. 'CLEVR', 'GoEmotions')
        model_name: Name of model (e.g. 'CLIP', 'Llama')
        concept_type: Type of concept (e.g. 'linsep_patch_embeddings_BD_True_BN_False')
        similar_concepts: Optional list of semantically similar concepts
        model_input_size: Model input size (will be inferred if None)
        percent_thru_model: Percentage through model
        save_path: Path to save figure
        figsize: Figure size
        scratch_dir: Directory where activation files are stored
        n_bins: Number of bins for histograms
        
    Returns:
        matplotlib Figure object
    """
    import matplotlib.patches as patches
    from utils.general_utils import retrieve_image, get_split_index_from_global_index, pad_or_resize_img
    from utils.memory_management_utils import ChunkedActivationLoader
    from utils.patch_alignment_utils import filter_patches_by_image_presence
    from scipy.stats import gaussian_kde
    
    # Determine dataset type and model input size
    is_text_dataset = dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank', 'IMDB', 'Jailbreak']
    
    if model_input_size is None:
        if is_text_dataset:
            # Set model input size based on text model type
            if model_name == 'Llama':
                model_input_size = ('text', 'text')
            elif model_name == 'Gemma':
                model_input_size = ('text', 'text2')
            elif model_name == 'Qwen':
                model_input_size = ('text', 'text3')
            else:
                raise ValueError(f"Unknown text model: {model_name}")
        else:
            # Image models
            if model_name == 'CLIP':
                model_input_size = (224, 224)
            elif model_name == 'Llama':
                model_input_size = (560, 560)
            else:
                raise ValueError(f"Unknown image model: {model_name}")
    
    # Calculate patches/tokens per sample
    if is_text_dataset:
        units_per_sample = None  # Will be loaded from file
        unit_type = 'token'
    else:
        # For images, calculate patches per image
        if model_input_size == (224, 224):
            units_per_sample = 256  # 16x16
        elif model_input_size == (560, 560):
            units_per_sample = 1600  # 40x40
        else:
            units_per_sample = 256
        unit_type = 'patch'
    
    # Get activation file name
    sample_type = 'patch'
    if concept_type == 'avg_patch_embeddings':
        acts_file = f"cosine_similarities_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        acts_file = f"dists_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"cosine_similarities_kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    # Load activation loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
    except FileNotFoundError:
        try:
            act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir='.', device=device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Activation file not found: {acts_file}")
    
    # Load ground truth
    gt_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
    print(f"Loading ground truth from: {gt_file}")
    gt_samples_per_concept = torch.load(gt_file, weights_only=False)
    
    # Load patch/token-level ground truth
    gt_patches_file = f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
    if not os.path.exists(gt_patches_file):
        gt_patches_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
    print(f"Loading {unit_type}-level ground truth from: {gt_patches_file}")
    
    if not os.path.exists(gt_patches_file):
        print(f"Warning: {unit_type}-level ground truth not found")
        gt_patches_per_concept = {}
    else:
        gt_patches_per_concept = torch.load(gt_patches_file, weights_only=False)
    
    # Get all concept names and find target concept index
    all_concepts = list(gt_samples_per_concept.keys())
    if concept not in all_concepts:
        raise ValueError(f"Concept '{concept}' not found. Available concepts: {all_concepts[:10]}...")
    concept_idx = all_concepts.index(concept)
    
    # Load test activations
    print("Loading all test activations...")
    test_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
    if test_acts is None:
        raise ValueError("Could not load test activations")
    
    # Get metadata
    import pandas as pd
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    test_global_indices = metadata[metadata['split'] == 'test'].index.tolist()
    print(f"Found {len(test_global_indices)} test samples")
    
    # Get number of test units
    num_test_units = test_acts.shape[0]
    
    if is_text_dataset:
        # Load token counts for text datasets
        token_count_file = f"GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt"
        
        if os.path.exists(token_count_file):
            all_token_counts = torch.load(token_count_file, weights_only=False)
            
            # Extract test split token counts
            tokens_per_sample = []
            for test_idx in test_global_indices:
                if test_idx < len(all_token_counts):
                    token_count_list = all_token_counts[test_idx]
                    total_tokens = sum(token_count_list)
                    tokens_per_sample.append(total_tokens)
            
            num_test_samples = len(tokens_per_sample)
            total_test_tokens = sum(tokens_per_sample)
            print(f"Analyzing {num_test_samples} test texts ({total_test_tokens} tokens total)")
    else:
        num_test_samples = num_test_units // units_per_sample
        print(f"Analyzing {num_test_samples} test images ({num_test_units} patches)")
    
    # Build masks
    print(f"Building {unit_type} masks...")
    
    if is_text_dataset:
        total_units = num_test_units
        test_sample_indices = torch.zeros(total_units, dtype=torch.long, device=device)
        current_idx = 0
        for sample_idx, token_count in enumerate(tokens_per_sample):
            test_sample_indices[current_idx:current_idx + token_count] = sample_idx
            current_idx += token_count
        
        is_valid_unit = torch.ones(total_units, dtype=torch.bool, device=device)
    else:
        total_units = num_test_samples * units_per_sample
        all_test_unit_indices = torch.arange(total_units, device=device)
        test_sample_indices = torch.div(all_test_unit_indices, units_per_sample, rounding_mode='floor')
        
        # Load padding mask if exists
        padding_mask_file = f'GT_Samples/{dataset_name}/patches_w_image_mask_inputsize_{model_input_size}.pt'
        if os.path.exists(padding_mask_file):
            print("Loading pre-computed padding mask...")
            full_padding_mask = torch.load(padding_mask_file, weights_only=False)
            test_padding_mask = []
            for test_global_idx in test_global_indices:
                start_patch = test_global_idx * units_per_sample
                end_patch = start_patch + units_per_sample
                test_padding_mask.append(full_padding_mask[start_patch:end_patch])
            is_valid_unit = torch.cat(test_padding_mask).to(device).bool()
        else:
            is_valid_unit = torch.ones(total_units, dtype=torch.bool, device=device)
    
    # Get samples containing target concept
    samples_with_target = set(gt_samples_per_concept.get(concept, []))
    samples_with_target_list = list(samples_with_target)
    if samples_with_target_list:
        samples_with_target_tensor = torch.tensor(samples_with_target_list, device=device)
        is_from_target_sample = torch.isin(test_sample_indices, samples_with_target_tensor)
    else:
        is_from_target_sample = torch.zeros(total_units, dtype=torch.bool, device=device)
    
    # Get units containing target concept
    units_with_target = set(gt_patches_per_concept.get(concept, []))
    
    # Create masks
    is_target_unit = torch.zeros(total_units, dtype=torch.bool, device=device)
    
    if is_text_dataset:
        # Map global token indices to test-specific indices
        global_token_start_per_sample = {}
        current_global_idx = 0
        
        for sample_idx in range(len(all_token_counts)):
            global_token_start_per_sample[sample_idx] = current_global_idx
            current_global_idx += sum(all_token_counts[sample_idx])
        
        current_test_token_idx = 0
        for test_pos, global_sample_idx in enumerate(test_global_indices):
            sample_token_count = tokens_per_sample[test_pos]
            global_start = global_token_start_per_sample[global_sample_idx]
            
            for i in range(sample_token_count):
                global_token_idx = global_start + i
                test_token_idx = current_test_token_idx + i
                
                if global_token_idx in units_with_target:
                    is_target_unit[test_token_idx] = True
                    
            current_test_token_idx += sample_token_count
    else:
        # For images
        for test_pos, global_img_idx in enumerate(test_global_indices):
            global_patch_start = global_img_idx * units_per_sample
            test_patch_start = test_pos * units_per_sample
            
            for i in range(units_per_sample):
                global_patch_idx = global_patch_start + i
                test_patch_idx = test_patch_start + i
                
                if global_patch_idx in units_with_target:
                    is_target_unit[test_patch_idx] = True
    
    # Get concept activations
    concept_acts_gpu = test_acts[:, concept_idx]
    
    # Extract activations for target concept
    mask_target = is_valid_unit & is_target_unit
    acts_target = concept_acts_gpu[mask_target].cpu().numpy()
    
    # Extract activations for comparison
    if similar_concepts:
        # Get units containing similar concepts
        units_with_similar = set()
        is_similar_unit = torch.zeros(total_units, dtype=torch.bool, device=device)
        
        for sim_concept in similar_concepts:
            if sim_concept in gt_patches_per_concept:
                units_with_similar.update(gt_patches_per_concept[sim_concept])
        
        if is_text_dataset:
            # Map similar concept units
            current_test_token_idx = 0
            for test_pos, global_sample_idx in enumerate(test_global_indices):
                sample_token_count = tokens_per_sample[test_pos]
                global_start = global_token_start_per_sample[global_sample_idx]
                
                for i in range(sample_token_count):
                    global_token_idx = global_start + i
                    test_token_idx = current_test_token_idx + i
                    
                    if global_token_idx in units_with_similar:
                        is_similar_unit[test_token_idx] = True
                        
                current_test_token_idx += sample_token_count
        else:
            # For images
            for test_pos, global_img_idx in enumerate(test_global_indices):
                global_patch_start = global_img_idx * units_per_sample
                test_patch_start = test_pos * units_per_sample
                
                for i in range(units_per_sample):
                    global_patch_idx = global_patch_start + i
                    test_patch_idx = test_patch_start + i
                    
                    if global_patch_idx in units_with_similar:
                        is_similar_unit[test_patch_idx] = True
        
        # Get similar concept activations from ALL samples
        mask_similar = is_valid_unit & is_similar_unit
        acts_comparison = concept_acts_gpu[mask_similar].cpu().numpy()
        comparison_label = f"Similar concepts: {', '.join(similar_concepts[:3])}{'...' if len(similar_concepts) > 3 else ''}"
    else:
        # Get background from samples WITHOUT target concept
        mask_background = is_valid_unit & ~is_from_target_sample & ~is_target_unit
        acts_comparison = concept_acts_gpu[mask_background].cpu().numpy()
        comparison_label = f"Non-Concept (no '{concept}')"
    
    print(f"\n{unit_type.capitalize()} counts:")
    print(f"  {unit_type.capitalize()}s WITH '{concept}': {len(acts_target)}")
    print(f"  {comparison_label}: {len(acts_comparison)}")
    
    # Load detection threshold
    threshold_file = None
    if concept_type == 'avg_patch_embeddings':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_avg_patch_embeddings_percentthrumodel_{percent_thru_model}.pt'
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}.pt'
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
    
    best_threshold = None
    if threshold_file and os.path.exists(threshold_file):
        try:
            thresholds_data = torch.load(threshold_file, weights_only=False)
            if concept in thresholds_data:
                best_threshold = thresholds_data[concept]['best_threshold']
                print(f"  Detection threshold: {best_threshold:.3f}")
        except:
            pass
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine range
    all_acts = np.concatenate([acts_target, acts_comparison])
    hist_min = np.min(all_acts) - 0.01
    hist_max = np.max(all_acts) + 0.01
    x_range = np.linspace(hist_min, hist_max, 200)
    
    # Plot distributions using KDE
    if len(acts_target) > 1:
        kde_target = gaussian_kde(acts_target)
        density_target = kde_target(x_range)
        ax.plot(x_range, density_target, color='green', linewidth=3, 
                label=f"{unit_type.capitalize()}s WITH '{concept}' (n={len(acts_target)})")
        ax.fill_between(x_range, density_target, alpha=0.3, color='green')
    
    if len(acts_comparison) > 1:
        kde_comparison = gaussian_kde(acts_comparison)
        density_comparison = kde_comparison(x_range)
        color = 'orange' if similar_concepts else 'gray'
        ax.plot(x_range, density_comparison, color=color, linewidth=3,
                label=f"{comparison_label} (n={len(acts_comparison)})")
        ax.fill_between(x_range, density_comparison, alpha=0.3, color=color)
    
    # Add threshold line
    if best_threshold is not None:
        ax.axvline(best_threshold, color='purple', linestyle='-', linewidth=2.5,
                  label=f'Detection threshold: {best_threshold:.3f}')
    
    # Add mean lines
    if len(acts_target) > 0:
        mean_target = np.mean(acts_target)
        ax.axvline(mean_target, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(mean_target, ax.get_ylim()[1]*0.9, f'μ={mean_target:.3f}', 
                rotation=90, va='bottom', ha='right', color='darkgreen')
    
    if len(acts_comparison) > 0:
        mean_comparison = np.mean(acts_comparison)
        color = 'darkorange' if similar_concepts else 'darkgray'
        ax.axvline(mean_comparison, color=color, linestyle='--', linewidth=2, alpha=0.7)
        ax.text(mean_comparison, ax.get_ylim()[1]*0.9, f'μ={mean_comparison:.3f}', 
                rotation=90, va='bottom', ha='right', color=color)
    
    # Customize plot
    ax.set_xlabel('Activation Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f"Activation Distributions for '{concept}'\n{dataset_name} - {model_name}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Add statistics box
    stats_text = f"Statistics:\n"
    if len(acts_target) > 0:
        stats_text += f"Target: μ={np.mean(acts_target):.3f}, σ={np.std(acts_target):.3f}\n"
    if len(acts_comparison) > 0:
        comp_name = "Similar" if similar_concepts else "Non-Concept"
        stats_text += f"{comp_name}: μ={np.mean(acts_comparison):.3f}, σ={np.std(acts_comparison):.3f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    return fig


def test_gaussian_vs_mixture(activations: np.ndarray, 
                            n_components: int = 2,
                            plot: bool = False,
                            concept_name: str = None,
                            use_gpu: bool = True,
                            threshold: Optional[float] = None,
                            signal_dist: str = 'gaussian',
                            background_acts: Optional[np.ndarray] = None) -> Dict[str, Union[float, Dict, str]]:
    """
    Test whether positive activations are better modeled by a single Gaussian or mixture of Gaussians.
    
    This function helps identify if there are distinct subpopulations in the activation patterns,
    potentially corresponding to superdetector tokens (high activations) vs regular activations.
    
    Args:
        activations: 1D array of activation values (should be positive activations only)
        n_components: Number of components for the mixture model (default: 2)
        plot: Whether to create visualization plots
        concept_name: Name of the concept for plot titles
        use_gpu: Whether to use GPU for faster computation if available (default: True)
        threshold: Optional detection threshold to display on plot
        signal_dist: Distribution type for components - 'gaussian' or 'student-t' (default: 'gaussian')
        background_acts: Optional background activations to compute overlap metrics
        
    Returns:
        Dictionary containing:
        - 'best_model': 'single' or 'mixture'
        - 'single_gaussian': Parameters and metrics for single Gaussian
        - 'mixture_gaussian': Parameters and metrics for mixture model
        - 'bic_difference': BIC(single) - BIC(mixture), positive favors mixture
        - 'aic_difference': AIC(single) - AIC(mixture), positive favors mixture
        - 'likelihood_ratio': Log likelihood ratio test statistic
        - 'p_value': P-value for likelihood ratio test
        - 'ks_test_single': Kolmogorov-Smirnov test for single Gaussian
        - 'ks_test_mixture': KS test for mixture (using empirical CDF)
        - 'separation_score': If mixture, how well separated the components are
    """
    # Ensure we have a 1D array
    activations = np.asarray(activations).flatten()
    # NOTE: We now keep ALL activations, including negative values
    
    if len(activations) < 10:
        return {
            'best_model': 'insufficient_data',
            'error': f'Only {len(activations)} positive activations, need at least 10'
        }
    
    # Reshape for sklearn
    X = activations.reshape(-1, 1)
    
    # Fit single Gaussian
    single_mean = np.mean(activations)
    single_std = np.std(activations, ddof=1)
    single_var = single_std ** 2
    
    # Log likelihood for single Gaussian
    single_log_likelihood = np.sum(norm.logpdf(activations, single_mean, single_std))
    
    # BIC and AIC for single Gaussian (2 parameters: mean and variance)
    n_params_single = 2
    single_bic = -2 * single_log_likelihood + n_params_single * np.log(len(activations))
    single_aic = -2 * single_log_likelihood + 2 * n_params_single
    
    # KS test for single Gaussian
    ks_stat_single, ks_pval_single = kstest(activations, 'norm', args=(single_mean, single_std))
    
    # Check if GPU is available and requested
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    # Fit mixture based on distribution type
    if signal_dist == 'gaussian':
        # Fit Gaussian mixture
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
            gmm.fit(X)
        
        # Get mixture parameters
        mixture_means = gmm.means_.flatten()
        mixture_stds = np.sqrt(gmm.covariances_.flatten())
        mixture_weights = gmm.weights_
        mixture_dfs = None  # Not applicable for Gaussian
        
        # Log likelihood for mixture
        mixture_log_likelihood = gmm.score(X) * len(activations)
    else:  # signal_dist == 'student-t'
        # Fit Student-t mixture using custom EM algorithm
        from scipy.special import gammaln, digamma, polygamma
        
        # Initialize with k-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Initialize parameters
        mixture_means = np.zeros(n_components)
        mixture_stds = np.zeros(n_components)
        mixture_dfs = np.ones(n_components) * 5.0  # Start with df=5
        mixture_weights = np.zeros(n_components)
        
        for k in range(n_components):
            mask = labels == k
            mixture_weights[k] = np.mean(mask)
            if np.sum(mask) > 1:
                mixture_means[k] = np.mean(activations[mask])
                mixture_stds[k] = np.std(activations[mask], ddof=1)
            else:
                mixture_means[k] = np.mean(activations)
                mixture_stds[k] = np.std(activations, ddof=1)
        
        # EM algorithm for Student-t mixture
        max_iter = 100
        tol = 1e-6
        prev_ll = -np.inf
        
        for iteration in range(max_iter):
            # E-step: compute responsibilities
            log_probs = np.zeros((len(activations), n_components))
            for k in range(n_components):
                # Student-t log pdf
                df = mixture_dfs[k]
                loc = mixture_means[k]
                scale = mixture_stds[k]
                
                z = (activations - loc) / scale
                log_probs[:, k] = (gammaln((df + 1) / 2) - gammaln(df / 2) - 
                                  0.5 * np.log(df * np.pi) - np.log(scale) - 
                                  (df + 1) / 2 * np.log(1 + z**2 / df) + 
                                  np.log(mixture_weights[k]))
            
            # Normalize
            log_probs_max = np.max(log_probs, axis=1, keepdims=True)
            log_probs = log_probs - log_probs_max
            probs = np.exp(log_probs)
            responsibilities = probs / np.sum(probs, axis=1, keepdims=True)
            
            # Compute log likelihood
            ll = np.sum(log_probs_max.flatten() + np.log(np.sum(probs, axis=1)))
            
            # Check convergence
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
            
            # M-step: update parameters
            for k in range(n_components):
                resp_k = responsibilities[:, k]
                n_k = np.sum(resp_k)
                
                if n_k > 1e-10:
                    # Update weights
                    mixture_weights[k] = n_k / len(activations)
                    
                    # Update mean
                    mixture_means[k] = np.sum(resp_k * activations) / n_k
                    
                    # Update scale (iterative for Student-t)
                    z = (activations - mixture_means[k]) / mixture_stds[k]
                    u = (mixture_dfs[k] + 1) / (mixture_dfs[k] + z**2)
                    mixture_stds[k] = np.sqrt(np.sum(resp_k * u * (activations - mixture_means[k])**2) / n_k)
                    
                    # Update df using expected log-likelihood approach
                    # E[log(1 + z^2/df)] and E[u] where u = (df+1)/(df+z^2)
                    e_log_u = np.sum(resp_k * (np.log(u) + digamma((mixture_dfs[k] + 1)/2) - digamma(mixture_dfs[k]/2))) / n_k
                    e_u = np.sum(resp_k * u) / n_k
                    
                    # Newton-Raphson update for df (simplified)
                    # This solves: digamma((df+1)/2) - digamma(df/2) - log(df) - 1/df - e_log_u + e_u = 0
                    for _ in range(5):  # A few Newton iterations
                        df = mixture_dfs[k]
                        f = digamma((df + 1)/2) - digamma(df/2) - np.log(df) - 1/df - e_log_u + e_u
                        f_prime = 0.5 * (polygamma(1, (df + 1)/2) - polygamma(1, df/2)) - 1/df + 1/(df**2)
                        df_new = df - f / f_prime
                        df_new = np.clip(df_new, 1.5, 50.0)  # Keep df in reasonable range
                        if abs(df_new - df) < 0.1:
                            break
                        mixture_dfs[k] = df_new
        
        # Compute proper log-likelihood of the mixture model
        # For each data point: log(sum_k(weight_k * pdf_k(x)))
        mixture_pdfs = np.zeros((len(activations), n_components))
        for k in range(n_components):
            mixture_pdfs[:, k] = mixture_weights[k] * stats.t.pdf(activations, 
                                                                  df=mixture_dfs[k], 
                                                                  loc=mixture_means[k], 
                                                                  scale=mixture_stds[k])
        mixture_log_likelihood = np.sum(np.log(np.sum(mixture_pdfs, axis=1) + 1e-10))
    
    # Sort components by mean
    sorted_idx = np.argsort(mixture_means)
    mixture_means = mixture_means[sorted_idx]
    mixture_stds = mixture_stds[sorted_idx]
    mixture_weights = mixture_weights[sorted_idx]
    if mixture_dfs is not None:
        mixture_dfs = mixture_dfs[sorted_idx]
    
    # BIC and AIC for mixture
    # For Gaussian: n_components * 3 - 1 parameters (means, variances, and weights minus 1)
    # For Student-t: n_components * 4 - 1 parameters (means, scales, dfs, and weights minus 1)
    if signal_dist == 'student-t':
        n_params_mixture = n_components * 4 - 1
    else:
        n_params_mixture = n_components * 3 - 1
    mixture_bic = -2 * mixture_log_likelihood + n_params_mixture * np.log(len(activations))
    mixture_aic = -2 * mixture_log_likelihood + 2 * n_params_mixture
    
    # Likelihood ratio test
    lr_statistic = 2 * (mixture_log_likelihood - single_log_likelihood)
    df = n_params_mixture - n_params_single
    p_value = 1 - stats.chi2.cdf(lr_statistic, df)
    
    # KS test for mixture (using empirical CDF)
    # Generate samples from the fitted mixture
    n_samples = 10000
    if signal_dist == 'gaussian':
        mixture_samples = gmm.sample(n_samples)[0].flatten()
    else:  # Student-t mixture
        # Generate samples from Student-t mixture
        mixture_samples = []
        for _ in range(n_samples):
            # Choose component
            comp = np.random.choice(n_components, p=mixture_weights)
            # Sample from Student-t
            sample = stats.t.rvs(df=mixture_dfs[comp], loc=mixture_means[comp], 
                               scale=mixture_stds[comp], size=1)
            mixture_samples.append(sample[0])
        mixture_samples = np.array(mixture_samples)
    
    ks_stat_mixture, ks_pval_mixture = stats.ks_2samp(activations, mixture_samples)
    
    # Calculate separation score for mixture (if applicable)
    separation_score = 0
    if n_components == 2 and len(mixture_means) == 2:
        # Separation = difference in means / average std
        mean_diff = abs(mixture_means[1] - mixture_means[0])
        avg_std = np.mean(mixture_stds)
        separation_score = mean_diff / avg_std if avg_std > 0 else 0
    
    # Calculate overlap with background distribution if provided
    background_overlaps = None
    if background_acts is not None and len(background_acts) > 10:
        background_overlaps = []
        
        # Create histogram-based approximation of background distribution
        n_bins_bg = min(200, max(50, len(background_acts)//100))
        hist_bg, bin_edges_bg = np.histogram(background_acts, bins=n_bins_bg, density=True)
        bin_centers_bg = (bin_edges_bg[:-1] + bin_edges_bg[1:]) / 2
        
        # Compute overlap for each mixture component
        for k in range(n_components):
            # Define integration range
            x_min = min(background_acts.min(), mixture_means[k] - 4*mixture_stds[k])
            x_max = max(background_acts.max(), mixture_means[k] + 4*mixture_stds[k])
            x_range = np.linspace(x_min, x_max, 1000)
            
            # Non-Concept PDF (interpolated)
            bg_pdf = np.interp(x_range, bin_centers_bg, hist_bg, left=0, right=0)
            
            # Component PDF
            if signal_dist == 'gaussian':
                comp_pdf = stats.norm.pdf(x_range, mixture_means[k], mixture_stds[k])
            else:  # Student-t
                comp_pdf = stats.t.pdf(x_range, df=mixture_dfs[k], loc=mixture_means[k], 
                                     scale=mixture_stds[k])
            
            # Compute overlap (intersection of PDFs)
            overlap = np.trapz(np.minimum(bg_pdf, comp_pdf), x_range)
            background_overlaps.append(overlap)
    
    # Determine best model
    bic_diff = single_bic - mixture_bic
    aic_diff = single_aic - mixture_aic
    
    # Model selection based on BIC (lower is better)
    best_model = 'mixture' if bic_diff > 0 else 'single'
    
    # Additional check: if mixture components are too similar, prefer single
    if best_model == 'mixture' and separation_score < 0.5:
        best_model = 'single'
    
    results = {
        'best_model': best_model,
        'single_gaussian': {
            'mean': single_mean,
            'std': single_std,
            'log_likelihood': single_log_likelihood,
            'bic': single_bic,
            'aic': single_aic,
            'ks_statistic': ks_stat_single,
            'ks_pvalue': ks_pval_single
        },
        'mixture_gaussian': {
            'means': mixture_means.tolist(),
            'stds': mixture_stds.tolist(),
            'weights': mixture_weights.tolist(),
            'dfs': mixture_dfs.tolist() if mixture_dfs is not None else None,
            'log_likelihood': mixture_log_likelihood,
            'bic': mixture_bic,
            'aic': mixture_aic,
            'ks_statistic': ks_stat_mixture,
            'ks_pvalue': ks_pval_mixture,
            'background_overlaps': background_overlaps
        },
        'bic_difference': bic_diff,
        'aic_difference': aic_diff,
        'likelihood_ratio': lr_statistic,
        'p_value': p_value,
        'separation_score': separation_score
    }
    
    # Create visualization if requested
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Histogram with fitted distributions
        ax = axes[0]
        # Many more bins for fine-grained resolution
        n_bins = min(200, max(100, len(activations) // 3))
        counts, bins, _ = ax.hist(activations, bins=n_bins, density=True, 
                                 alpha=0.7, color='gray', edgecolor='black', linewidth=0.5)
        
        x_range = np.linspace(activations.min(), activations.max(), 1000)
        
        # Plot in specific order for proper layering
        # 1. Single Gaussian (bottom layer)
        single_pdf = norm.pdf(x_range, single_mean, single_std)
        ax.plot(x_range, single_pdf, 'b-', linewidth=2, alpha=0.7,
                label=f'Single Gaussian (BIC={single_bic:.1f})', zorder=1)
        
        # 2. Full Mixture (middle layer)
        mixture_pdf = np.zeros_like(x_range)
        component_colors = ['orange', 'green']
        for i in range(n_components):
            if signal_dist == 'gaussian':
                component_pdf = mixture_weights[i] * norm.pdf(x_range, mixture_means[i], mixture_stds[i])
            else:  # Student-t
                component_pdf = mixture_weights[i] * stats.t.pdf(x_range, df=mixture_dfs[i], 
                                                                loc=mixture_means[i], scale=mixture_stds[i])
            mixture_pdf += component_pdf
        
        dist_label = 'Student-t Mixture' if signal_dist == 'student-t' else 'Mixture'
        ax.plot(x_range, mixture_pdf, 'r-', linewidth=3, 
                label=f'{dist_label} (BIC={mixture_bic:.1f})', zorder=2)
        
        # 3. Individual Components (top layer, dotted)
        for i in range(n_components):
            if signal_dist == 'gaussian':
                component_pdf = mixture_weights[i] * norm.pdf(x_range, mixture_means[i], mixture_stds[i])
                comp_label = f'Component {i+1} (w={mixture_weights[i]:.2f}, μ={mixture_means[i]:.3f})'
            else:  # Student-t
                component_pdf = mixture_weights[i] * stats.t.pdf(x_range, df=mixture_dfs[i], 
                                                                loc=mixture_means[i], scale=mixture_stds[i])
                comp_label = f'Component {i+1} (w={mixture_weights[i]:.2f}, μ={mixture_means[i]:.3f}, ν={mixture_dfs[i]:.1f})'
            
            # Add background overlap if available
            if background_overlaps is not None:
                comp_label += f', bg overlap={background_overlaps[i]:.2%}'
                
            ax.plot(x_range, component_pdf, '--', alpha=0.9, linewidth=2.5,
                   color=component_colors[i % len(component_colors)],
                   label=comp_label,
                   zorder=3+i)
        
        # Add threshold line if provided (highest layer)
        if threshold is not None:
            ax.axvline(x=threshold, color='darkgreen', linestyle='-', linewidth=2.5, 
                      label=f'Detection Threshold ({threshold:.3f})', alpha=0.9, zorder=10)
        
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution Fitting{" - " + concept_name if concept_name else ""}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Model comparison
        ax = axes[1]
        
        metrics = ['BIC', 'AIC', '-Log Likelihood']
        single_values = [single_bic, single_aic, -single_log_likelihood]
        mixture_values = [mixture_bic, mixture_aic, -mixture_log_likelihood]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, single_values, width, label='Single Gaussian', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, mixture_values, width, label='Mixture', color='red', alpha=0.7)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value (lower is better)')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        # Add summary text
        summary = f"Best Model: {best_model.upper()}\n"
        summary += f"BIC diff: {bic_diff:.2f}\n"
        summary += f"LR test p-value: {p_value:.3f}\n"
        if n_components == 2:
            summary += f"Separation score: {separation_score:.2f}"
        
        ax.text(0.02, 0.98, summary, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        plt.show()
    
    return results





def analyze_gaussian_mixture_for_concepts(
    concepts: Optional[Union[str, List[str]]] = None,
    dataset_name: str = None,
    model_name: str = None,
    concept_type: str = None,
    model_input_size: Optional[Tuple[int, int]] = None,
    percent_thru_model: int = 100,
    n_components: int = 2,
    scratch_dir: str = '',
    save_results: bool = False,
    save_path: Optional[str] = None,
    show_threshold: bool = True
) -> Dict[str, Any]:
    """
    Analyze whether GT positive patch/token activations are better modeled as single or mixture of Gaussians.
    This analyzes activations from specific patches/tokens that contain the target concept,
    not just from images/texts that contain the concept.
    
    Args:
        concepts: Single concept name, list of concept names, or None for all concepts
                 (e.g. 'blue_sphere', ['red_cube', 'green_cylinder'], or None)
        dataset_name: Name of dataset (e.g. 'CLEVR', 'COCO', 'Broden')
        model_name: Name of model (e.g. 'clipRN50', 'CLIP')
        concept_type: Type of concept - must be one of:
            'avg_patch_embeddings'
            'linsep_patch_embeddings_BD_True_BN_False'
            'kmeans_1000_patch_embeddings_kmeans'
            'kmeans_1000_linsep_patch_embeddings_kmeans'
        model_input_size: Model input size (will be inferred if None)
        percent_thru_model: Percentage through model (default 100)
        n_components: Number of Gaussian components to test (default 2)
        scratch_dir: Directory where activation files are stored
        save_results: Whether to save results to file (default False)
        save_path: Custom path to save results (if None, uses default location)
        show_threshold: Whether to show detection threshold on plots (default True)
        
    Returns:
        Dictionary containing:
        - 'concept_results': Individual results for each concept
        - 'summary': Summary statistics across all concepts
        - 'dataset_info': Information about the dataset and model
    """
    # Determine if text dataset
    is_text_dataset = dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank']
    
    # Infer model input size if not provided
    if model_input_size is None:
        if is_text_dataset:
            model_input_size = 'text'  # For text datasets
        elif 'CLIP' in model_name or 'clip' in model_name:
            model_input_size = (224, 224)
        elif 'Llama' in model_name or 'llama' in model_name:
            model_input_size = (560, 560)
        else:
            model_input_size = (224, 224)
    
    
    # Get activation file name based on concept type
    sample_type = 'patch'
    if concept_type == 'avg_patch_embeddings':
        acts_file = f"cosine_similarities_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        acts_file = f"dists_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"cosine_similarities_kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    # Load ground truth patches/tokens per concept (unit-level)
    # Note: For text datasets, the file might be named gt_patch_per_concept instead of gt_patches_per_concept
    gt_patches_file = f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
    if not os.path.exists(gt_patches_file):
        # Try alternative naming convention for text datasets
        gt_patches_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
    
    if not os.path.exists(gt_patches_file):
        print(f"Error: Patch-level ground truth not found at {gt_patches_file}")
        return None
    
    try:
        # Try with weights_only first, fall back if it fails
        try:
            gt_patches_per_concept = torch.load(gt_patches_file, weights_only=True)
        except:
            gt_patches_per_concept = torch.load(gt_patches_file)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {gt_patches_file}")
        return None
    
    # Load activation loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load activation loader
    try:
        act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
    except FileNotFoundError as e:
        print(f"Error: Could not find activation file: {acts_file}")
        print(f"Error details: {e}")
        return None
    
    # Get activation info
    act_info = act_loader.get_activation_info()
    num_concepts = act_info['num_concepts']
    
    # Load concepts to get proper indices (matching reference function)
    # Handle different concept type formats
    if concept_type == 'avg_patch_embeddings':
        concepts_filename = f"avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        concepts_filename = f"linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        concepts_filename = f"kmeans_1000_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        concepts_filename = f"kmeans_1000_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    else:
        concepts_filename = f"{concept_type.replace('_patch_embeddings', '')}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    
    concepts_file = f"Concepts/{dataset_name}/{concepts_filename}"
    
    if os.path.exists(concepts_file):
        concepts_data = torch.load(concepts_file, weights_only=False)
        if isinstance(concepts_data, dict):
            all_concept_names = list(concepts_data.keys())
        else:
            raise ValueError(f"Unexpected format in concepts file: {concepts_file}")
    else:
        raise FileNotFoundError(f"Concepts file not found: {concepts_file}")
    
    # Load thresholds if requested
    thresholds = None
    if show_threshold:
        # Determine threshold file name based on concept type
        if 'cosine' in acts_file:
            threshold_prefix = 'cosine'
        else:
            threshold_prefix = 'distance'
            
        # Construct the threshold file path based on concept type (matching reference function)
        threshold_file = None
        if concept_type == 'avg_patch_embeddings':
            threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_avg_patch_embeddings_percentthrumodel_{percent_thru_model}.pt'
        elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
            threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}.pt'
        elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
            threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
        elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
            threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
        
        # Load thresholds (matching reference function exactly)
        if threshold_file and os.path.exists(threshold_file):
            try:
                thresholds_data = torch.load(threshold_file, weights_only=False)
                thresholds = thresholds_data
            except Exception as e:
                print(f"Warning: Could not load threshold from {threshold_file}: {e}")
        else:
            print(f"Warning: Threshold file not found: {threshold_file}")
    
    # Determine which concepts to analyze
    if concepts is None:
        # Analyze all concepts in the ground truth
        gt_concept_names = list(gt_patches_per_concept.keys())
        # Filter out any non-concept entries if they exist
        concepts_to_analyze = [c for c in gt_concept_names 
                             if not c.startswith('_') and c != 'metadata']
    elif isinstance(concepts, str):
        concepts_to_analyze = [concepts]
    else:
        concepts_to_analyze = list(concepts)
    
    
    # Filter to only include valid concepts for this dataset
    gt_patches_per_concept = filter_concept_dict(gt_patches_per_concept, dataset_name)
    
    # Results storage
    results = {}
    mixture_count = 0
    single_count = 0
    insufficient_count = 0
    no_data_count = 0
    
    # Analyze each requested concept
    for i, concept_name in enumerate(concepts_to_analyze):
        # Get concept index from the concept names list (matching reference function)
        if concept_name not in all_concept_names:
            print(f"Warning: Concept '{concept_name}' not found in concepts file")
            continue
            
        concept_idx = all_concept_names.index(concept_name)
        
        # Get positive patch/token indices
        positive_patch_indices = gt_patches_per_concept.get(concept_name, [])
        
        # If no patches found, skip this concept
        if len(positive_patch_indices) == 0:
            print(f"No positive patches found for concept '{concept_name}'")
            results[concept_name] = {
                'best_model': 'no_data',
                'n_positive_activations': 0
            }
            insufficient_count += 1
            continue
            
        # Collect positive activations from patches
        positive_activations = []
        
        # Load test activations (matching reference function)
        test_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
        if test_acts is None:
            print(f"Could not load test activations for concept '{concept_name}'")
            continue
            
        # Keep data on GPU if available for faster processing
        if str(device) == 'cuda' and not test_acts.is_cuda:
            test_acts = test_acts.cuda()
            
        # Get activations for this concept
        concept_acts = test_acts[:, concept_idx]
        
        # Get test image indices to map global to test-specific indices
        metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
        test_global_indices = metadata[metadata['split'] == 'test'].index.tolist()
        
        # Calculate patches/tokens per sample
        if is_text_dataset:
            # For text datasets, we need token counts per sample
            # This would require loading token count files - simplified for now
            print(f"Text dataset token mapping not fully implemented yet")
            # Just use indices as-is for now
            for patch_idx in positive_patch_indices:
                if patch_idx < len(concept_acts):
                    act_value = concept_acts[patch_idx].item()
                    positive_activations.append(act_value)
        else:
            # For images, calculate patches per image
            if model_input_size == (224, 224):
                patches_per_image = 256  # 16x16
            elif model_input_size == (560, 560):
                patches_per_image = 1600  # 40x40
            else:
                patches_per_image = 256
                
            # Map global patch indices to test-specific indices
            for global_patch_idx in positive_patch_indices:
                # Find which global image this patch belongs to
                global_img_idx = global_patch_idx // patches_per_image
                patch_within_img = global_patch_idx % patches_per_image
                
                # Check if this image is in the test set
                if global_img_idx in test_global_indices:
                    # Get the position of this image in the test set
                    test_img_position = test_global_indices.index(global_img_idx)
                    # Calculate the test-specific patch index
                    test_patch_idx = test_img_position * patches_per_image + patch_within_img
                    
                    if test_patch_idx < len(concept_acts):
                        act_value = concept_acts[test_patch_idx].item()
                        # For distances (linsep), we want all values; for cosine sim, only positive
                        if 'dists' in acts_file or act_value > 0:
                            positive_activations.append(act_value)
        
        # Run analysis if we have enough data
        if len(positive_activations) >= 10:
            # Get threshold for this concept if available (matching reference function)
            concept_threshold = None
            if thresholds is not None and isinstance(thresholds, dict):
                if concept_name in thresholds:
                    concept_threshold = thresholds[concept_name]['best_threshold']
            
            analysis = test_gaussian_vs_mixture(
                np.array(positive_activations),
                n_components=n_components,
                plot=True,  # Always create plots
                concept_name=concept_name,
                use_gpu=True,  # Use GPU for faster processing
                threshold=concept_threshold
            )
            
            results[concept_name] = analysis
            results[concept_name]['n_positive_activations'] = len(positive_activations)
            
            # Update counts
            if analysis['best_model'] == 'mixture':
                mixture_count += 1
            elif analysis['best_model'] == 'single':
                single_count += 1
        else:
            results[concept_name] = {
                'best_model': 'insufficient_data',
                'n_positive_activations': len(positive_activations)
            }
            insufficient_count += 1
    
    # Calculate summary statistics
    summary = {
        'total_concepts_analyzed': len(results),
        'prefer_mixture': mixture_count,
        'prefer_single': single_count,
        'insufficient_data': insufficient_count,
        'no_data': no_data_count,
        'mixture_percentage': (mixture_count / (mixture_count + single_count) * 100) 
                            if (mixture_count + single_count) > 0 else 0
    }
    
    # Find concepts with highest separation scores
    separation_scores = []
    for concept_name, result in results.items():
        if result['best_model'] == 'mixture':
            separation_scores.append((concept_name, result['separation_score']))
    
    separation_scores.sort(key=lambda x: x[1], reverse=True)
    summary['top_separated_concepts'] = separation_scores[:10]
    
    
    # Save results
    if save_results:
        if save_path is None:
            output_dir = repo_path(
                "Experiments",
                "Quant_Results",
                dataset_name,
                "gaussian_mixture_analysis",
            )
            os.makedirs(output_dir, exist_ok=True)
            save_path = output_dir / (
                f"gaussian_mixture_analysis_{model_name}_{concept_type}_ptm{percent_thru_model}.pt"
            )
        
        torch.save({
            'concept_results': results,
            'summary': summary,
            'dataset_info': {
                'dataset': dataset_name,
                'model': model_name,
                'concept_type': concept_type,
                'model_input_size': model_input_size,
                'percent_thru_model': percent_thru_model,
                'n_components': n_components
            }
        }, save_path)
        
        print(f"\nResults saved to: {save_path}")
    
    return {
        'concept_results': results,
        'summary': summary,
        'dataset_info': {
            'dataset': dataset_name,
            'model': model_name,
            'concept_type': concept_type,
            'model_input_size': model_input_size,
            'percent_thru_model': percent_thru_model,
            'n_components': n_components
        }
    }


def analyze_concept_unified_mixture_decomposition(
    concepts: Optional[Union[str, List[str]]] = None,
    dataset_name: str = None,
    model_name: str = None,
    concept_type: str = None,
    model_input_size: Optional[Tuple[int, int]] = None,
    percent_thru_model: int = 100,
    scratch_dir: str = '',
    n_bins: int = 100,
    show_threshold: bool = True,
    background_method: str = 'gaussian',
    mixture_method: str = 'em',
    save_plots: bool = False,
    save_path: Optional[str] = None,
    show_plots: bool = True,
    signal_dist: str = 'gaussian'
) -> Dict[str, Any]:
    """
    print(f"\n=== Starting unified mixture decomposition analysis ===")
    print(f"Dataset: {dataset_name}, Model: {model_name}")
    print(f"Concept type: {concept_type}")
    print(f"Percent through model: {percent_thru_model}")
    
    Unified analysis combining Gaussian mixture modeling and decomposition analysis.
    
    Creates a 3-panel visualization showing:
    1. Non-Concept vs GT positive activations overlay
    2. GT activations fitted with 2-component Gaussian mixture
    3. GT decomposed into background + excess signal components
    
    Args:
        concepts: Single concept name, list of concept names, or None for all concepts
        dataset_name: Name of dataset (e.g. 'CLEVR', 'COCO', 'Broden')
        model_name: Name of model (e.g. 'CLIP', 'Llama')
        concept_type: Type of concept - must be one of:
            'avg_patch_embeddings'
            'linsep_patch_embeddings_BD_True_BN_False'
            'kmeans_1000_patch_embeddings_kmeans'
            'kmeans_1000_linsep_patch_embeddings_kmeans'
        model_input_size: Model input size (will be inferred if None)
        percent_thru_model: Percentage through model (default 100)
        scratch_dir: Directory where activation files are stored
        n_bins: Number of bins for histograms
        show_threshold: Whether to show detection threshold on plots
        background_method: 'gaussian' or 'kde' for background distribution fitting
        mixture_method: 'mle' or 'em' for mixture fitting
        save_plots: Whether to save individual plots
        save_path: Directory to save plots (if None, uses default location)
        show_plots: Whether to display plots (set False for batch processing)
        signal_dist: 'gaussian' or 'student-t' for signal component distribution
        
    Returns:
        Dictionary containing:
        - 'concept_results': Results for each concept including:
            - 'two_gaussian_mixture': Mixture coefficients, separability, and fit quality
            - 'background_signal_decomposition': Mixture coefficients, separability, and fit quality
            - 'data_stats': Number of GT positive and background samples
        - 'summary': Summary statistics across all concepts
    """
    print(f"\n=== Starting unified mixture decomposition analysis ===")
    print(f"Dataset: {dataset_name}, Model: {model_name}")
    print(f"Concept type: {concept_type}")
    print(f"Percent through model: {percent_thru_model}")
    
    print(f"\n=== Starting unified mixture decomposition analysis ===")
    print(f"Dataset: {dataset_name}, Model: {model_name}")
    print(f"Concept type: {concept_type}")
    print(f"Percent through model: {percent_thru_model}")
    
    # Determine if text dataset
    is_text_dataset = dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank', 'IMDB', 'Jailbreak']
    unit_type = 'token' if is_text_dataset else 'patch'
    
    # Infer model input size if not provided
    if model_input_size is None:
        if is_text_dataset:
            if model_name == 'Llama':
                model_input_size = ('text', 'text')
            elif model_name == 'Gemma':
                model_input_size = ('text', 'text2')
            elif model_name == 'Qwen':
                model_input_size = ('text', 'text3')
            else:
                raise ValueError(f"Unknown text model: {model_name}")
        else:
            if 'CLIP' in model_name or 'clip' in model_name:
                model_input_size = (224, 224)
            elif 'Llama' in model_name or 'llama' in model_name:
                model_input_size = (560, 560)
            else:
                model_input_size = (224, 224)
    
    # Get activation file name based on concept type
    sample_type = 'patch'
    if concept_type == 'avg_patch_embeddings':
        acts_file = f"cosine_similarities_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        acts_file = f"dists_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"cosine_similarities_kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    # Load ground truth patches/tokens per concept
    print(f"\nLoading ground truth data...")
    gt_patches_file = f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
    if not os.path.exists(gt_patches_file):
        gt_patches_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
    
    if not os.path.exists(gt_patches_file):
        print(f"Error: Patch-level ground truth not found at {gt_patches_file}")
        return None
    
    try:
        gt_patches_per_concept = torch.load(gt_patches_file, weights_only=False)
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return None
    
    # Load ground truth samples per concept
    gt_samples_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
    
    try:
        gt_samples_per_concept = torch.load(gt_samples_file, weights_only=False)
    except Exception as e:
        print(f"Error loading ground truth samples file: {e}")
        return None
    
    # Filter to only include valid concepts
    gt_patches_per_concept = filter_concept_dict(gt_patches_per_concept, dataset_name)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    # Load activation loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Loading activation file: {acts_file}")
    
    try:
        act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
    except FileNotFoundError as e:
        print(f"Error: Could not find activation file: {acts_file}")
        return None
    
    # Load test activations
    print(f"Loading test activations...")
    test_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
    if test_acts is None:
        raise ValueError("Could not load test activations")
    print(f"Test activations shape: {test_acts.shape}")
    
    # Load concepts to get proper indices
    if concept_type == 'avg_patch_embeddings':
        concepts_filename = f"avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        concepts_filename = f"linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        concepts_filename = f"kmeans_1000_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        concepts_filename = f"kmeans_1000_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    
    concepts_file = f"Concepts/{dataset_name}/{concepts_filename}"
    
    if os.path.exists(concepts_file):
        concepts_data = torch.load(concepts_file, weights_only=False)
        if isinstance(concepts_data, dict):
            all_concept_names = list(concepts_data.keys())
        else:
            raise ValueError(f"Unexpected format in concepts file: {concepts_file}")
    else:
        raise FileNotFoundError(f"Concepts file not found: {concepts_file}")
    
    # Load thresholds if requested
    thresholds = None
    if show_threshold:
        threshold_file = None
        if concept_type == 'avg_patch_embeddings':
            threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_avg_patch_embeddings_percentthrumodel_{percent_thru_model}.pt'
        elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
            threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}.pt'
        elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
            threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
        elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
            threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
        
        if threshold_file and os.path.exists(threshold_file):
            try:
                thresholds = torch.load(threshold_file, weights_only=False)
            except Exception as e:
                print(f"Warning: Could not load thresholds: {e}")
    
    # Get test metadata
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    test_global_indices = metadata[metadata['split'] == 'test'].index.tolist()
    
    # Calculate patches/tokens per sample
    if not is_text_dataset:
        if model_input_size == (224, 224):
            patches_per_image = 256  # 16x16
        elif model_input_size == (560, 560):
            patches_per_image = 1600  # 40x40
        else:
            patches_per_image = 256
    
    # Determine which concepts to analyze
    if concepts is None:
        gt_concept_names = list(gt_patches_per_concept.keys())
        concepts_to_analyze = [c for c in gt_concept_names 
                             if not c.startswith('_') and c != 'metadata']
    elif isinstance(concepts, str):
        concepts_to_analyze = [concepts]
    else:
        concepts_to_analyze = list(concepts)
    
    print(f"\nAnalyzing {len(concepts_to_analyze)} concepts: {concepts_to_analyze[:5]}{'...' if len(concepts_to_analyze) > 5 else ''}")
    
    # Results storage
    results = {}
    
    # Analyze each concept
    for concept_idx, concept_name in enumerate(concepts_to_analyze):
        print(f"\n--- Processing concept {concept_idx+1}/{len(concepts_to_analyze)}: '{concept_name}' ---")
        
        # Get concept index
        if concept_name not in all_concept_names:
            print(f"Warning: Concept '{concept_name}' not found in concepts file")
            continue
        
        concept_idx = all_concept_names.index(concept_name)
        
        # Get concept activations
        concept_acts = test_acts[:, concept_idx]
        
        # Keep on GPU if available for faster processing
        if not concept_acts.is_cuda and device.type == 'cuda':
            concept_acts = concept_acts.cuda()
        
        # Check if we need to filter padding patches
        has_padding = not is_text_dataset and model_input_size == (560, 560)
        
        # 1. Get GT positive patch activations
        print(f"  Collecting GT positive activations...")
        positive_patch_indices = gt_patches_per_concept.get(concept_name, [])
        
        # Filter out padding patches if needed
        if has_padding and len(positive_patch_indices) > 0:
            positive_patch_indices = filter_patches_by_image_presence(
                positive_patch_indices, dataset_name, model_input_size
            )
            positive_patch_indices = positive_patch_indices.tolist()
        
        if len(positive_patch_indices) == 0:
            print(f"  No positive patches found for concept '{concept_name}'")
            continue
        
        print(f"  Found {len(positive_patch_indices)} positive patches")
        
        # Collect GT positive activations efficiently
        if is_text_dataset:
            # TODO: Implement proper global to test mapping for text
            gt_positive_acts = []
            for patch_idx in positive_patch_indices:
                if patch_idx < len(concept_acts):
                    act_value = concept_acts[patch_idx].item()
                    gt_positive_acts.append(act_value)
            gt_positive_acts = np.array(gt_positive_acts)
        else:
            # Fully vectorized approach for images
            positive_patch_indices = np.array(positive_patch_indices)
            
            # Vectorized computation of global image indices
            global_img_indices = positive_patch_indices // patches_per_image
            patch_within_imgs = positive_patch_indices % patches_per_image
            
            # Filter to only include patches from test images
            # First create a set of test global indices for fast lookup
            test_global_set = set(test_global_indices)
            
            # Filter positive patches to only those from test images
            test_mask = np.array([img_idx in test_global_set for img_idx in global_img_indices])
            test_global_img_indices = global_img_indices[test_mask]
            test_patch_within_imgs = patch_within_imgs[test_mask]
            
            if len(test_global_img_indices) == 0:
                gt_positive_acts = np.array([])
            else:
                # Create mapping from global to test position
                global_to_test_pos = {global_idx: test_pos for test_pos, global_idx in enumerate(test_global_indices)}
                
                # Get test positions for the filtered indices
                test_positions = np.array([global_to_test_pos[idx] for idx in test_global_img_indices])
                
                # Compute test patch indices
                test_patch_indices = test_positions * patches_per_image + test_patch_within_imgs
                
                # Filter indices within bounds
                valid_indices = test_patch_indices[test_patch_indices < len(concept_acts)]
            
            # Extract activations in one go
            if len(valid_indices) > 0:
                if device.type == 'cuda' and concept_acts.is_cuda:
                    indices_tensor = torch.from_numpy(valid_indices).long().to(device)
                    gt_positive_acts = concept_acts[indices_tensor].cpu().numpy()
                else:
                    gt_positive_acts = concept_acts[valid_indices].cpu().numpy() if hasattr(concept_acts, 'cpu') else concept_acts[valid_indices].numpy()
            else:
                gt_positive_acts = np.array([])
        
        # 2. Get background patch activations (from samples WITHOUT the concept)
        print(f"  Collecting background activations...")
        samples_with_concept = set(gt_samples_per_concept.get(concept_name, []))
        
        if is_text_dataset:
            # TODO: Implement for text datasets
            print("Text dataset background extraction not fully implemented")
            continue
        else:
            # Fully vectorized background collection
            samples_with_concept_arr = np.array(list(samples_with_concept))
            
            # Create mask for test positions without concept
            all_test_positions = np.arange(len(test_global_indices))
            has_concept = np.zeros(len(all_test_positions), dtype=bool)
            if len(samples_with_concept_arr) > 0:
                valid_concept_samples = samples_with_concept_arr[samples_with_concept_arr < len(all_test_positions)]
                has_concept[valid_concept_samples] = True
            
            positions_without_concept = all_test_positions[~has_concept]
            
            if len(positions_without_concept) > 0:
                # Vectorized computation of all patch indices
                n_patches = min(patches_per_image, (len(concept_acts) - 1) // len(test_global_indices) + 1)
                patch_offsets = np.arange(n_patches)
                
                # Create all background indices at once
                base_indices = positions_without_concept * patches_per_image
                all_bg_indices = base_indices[:, np.newaxis] + patch_offsets[np.newaxis, :]
                background_indices = all_bg_indices.flatten()
                
                # Filter valid indices
                background_indices = background_indices[background_indices < len(concept_acts)]
                
                # Apply padding mask filtering if needed
                if has_padding:
                    try:
                        # Try to call filter_patches_by_image_presence with case handling
                        # First convert test indices to global indices
                        test_to_global_patch_mapping = {}
                        for test_idx in background_indices:
                            test_img_idx = test_idx // patches_per_image
                            patch_within_img = test_idx % patches_per_image
                            
                            if test_img_idx < len(test_global_indices):
                                global_img_idx = test_global_indices[test_img_idx]
                                global_patch_idx = global_img_idx * patches_per_image + patch_within_img
                                test_to_global_patch_mapping[test_idx] = global_patch_idx
                        
                        # Get all global indices
                        global_background_indices = list(test_to_global_patch_mapping.values())
                        
                        if len(global_background_indices) > 0:
                            # Try with original dataset name first
                            try:
                                filtered_global_indices = filter_patches_by_image_presence(
                                    global_background_indices, dataset_name, model_input_size
                                )
                            except FileNotFoundError:
                                # Try with capitalized version (e.g., COCO -> Coco)
                                alt_dataset_name = dataset_name.capitalize()
                                print(f"  Trying alternative dataset name: {alt_dataset_name}")
                                filtered_global_indices = filter_patches_by_image_presence(
                                    global_background_indices, alt_dataset_name, model_input_size
                                )
                            
                            # Convert result to set for fast lookup
                            if hasattr(filtered_global_indices, 'tolist'):
                                filtered_global_set = set(filtered_global_indices.tolist())
                            else:
                                filtered_global_set = set(filtered_global_indices)
                            
                            # Keep only test indices whose global counterparts passed the filter
                            filtered_background_indices = []
                            for test_idx, global_idx in test_to_global_patch_mapping.items():
                                if global_idx in filtered_global_set:
                                    filtered_background_indices.append(test_idx)
                            
                            orig_count = len(background_indices)
                            if len(filtered_background_indices) > 0:
                                background_indices = np.array(filtered_background_indices)
                                print(f"  Filtered out {orig_count - len(background_indices)} padding patches from background")
                            else:
                                print(f"  Warning: Padding filter removed all {orig_count} background patches! Keeping unfiltered.")
                                # Keep original indices if filtering removes everything
                    except Exception as e:
                        print(f"  Warning: Error applying padding filter: {e}. Continuing without filtering.")
                
                # Extract all at once
                if device.type == 'cuda' and concept_acts.is_cuda:
                    indices_tensor = torch.from_numpy(background_indices).long().to(device)
                    background_acts = concept_acts[indices_tensor].cpu().numpy()
                else:
                    background_acts = concept_acts[background_indices].cpu().numpy() if hasattr(concept_acts, 'cpu') else concept_acts[background_indices].numpy()
            else:
                background_acts = np.array([])
        
        # Skip if not enough data
        print(f"  GT positive: {len(gt_positive_acts)} samples, Non-Concept: {len(background_acts)} samples")
        if len(gt_positive_acts) < 10 or len(background_acts) < 50:
            print(f"  ⚠️  Insufficient data, skipping concept")
            results[concept_name] = {
                'success': False,
                'message': f'Insufficient data: {len(gt_positive_acts)} GT positive, {len(background_acts)} background'
            }
            continue
        
        # Analysis 1: Standard 2-component mixture on GT data
        print(f"  Running {signal_dist} mixture modeling...")
        gmm_result = test_gaussian_vs_mixture(
            gt_positive_acts,
            n_components=2,
            plot=False,  # We'll create custom plots
            concept_name=concept_name,
            use_gpu=True,
            signal_dist=signal_dist,
            background_acts=background_acts
        )
        
        # Analysis 2: Non-Concept + signal decomposition using empirical distribution
        print(f"  Running background + signal decomposition with empirical distribution...")
        # Always use histogram-based approximation for fast empirical distribution
        print(f"    Using fast histogram approximation for {len(background_acts)} background samples...")
        # Create fast histogram-based PDF with adaptive bins
        n_bins = min(200, max(50, len(background_acts)//100))
        hist, bin_edges = np.histogram(background_acts, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create interpolated PDF function
        from scipy.interpolate import interp1d
        # Extend the range slightly to handle edge cases
        extended_bins = np.concatenate([[bin_edges[0] - 1], bin_centers, [bin_edges[-1] + 1]])
        extended_hist = np.concatenate([[0], hist, [0]])
        fast_pdf = interp1d(extended_bins, extended_hist, kind='linear', bounds_error=False, fill_value=0)
        
        background_dist = {
            'type': 'histogram',
            'pdf': lambda x: np.maximum(fast_pdf(x), 1e-10),
            'log_pdf': lambda x: np.log(np.maximum(fast_pdf(x), 1e-10))
        }
        decomp_mixture_fit = fit_constrained_mixture(gt_positive_acts, background_dist, method=mixture_method, use_gpu=True, allow_background_shift=True, signal_dist=signal_dist)
        
        if not decomp_mixture_fit['success']:
            results[concept_name] = {
                'success': False,
                'message': 'Failed to fit decomposition mixture model'
            }
            continue
        
        # No bootstrap test
        
        # Determine x range for all plots (needed for calculations too)
        all_data = np.concatenate([gt_positive_acts, background_acts])
        x_min, x_max = np.min(all_data), np.max(all_data)
        x_range = np.linspace(x_min, x_max, 1000)
        
        # Create unified 3-panel visualization
        print(f"  Creating visualization...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel 1: Non-Concept vs GT overlay
        ax = axes[0]
        ax.hist(background_acts, bins=n_bins, density=True, alpha=0.6, 
                color='gray', edgecolor='black', label=f'Non-Concept (n={len(background_acts)})')
        ax.hist(gt_positive_acts, bins=n_bins, density=True, alpha=0.7,
                color='green', edgecolor='darkgreen', label=f'GT Positive (n={len(gt_positive_acts)})')
        
        # Add threshold if available
        if thresholds and concept_name in thresholds:
            threshold_val = thresholds[concept_name]['best_threshold']
            ax.axvline(threshold_val, color='purple', linestyle='-', linewidth=2.5, label='Detection threshold')
        
        ax.set_title('Non-Concept vs GT Positive Activations', fontsize=12)
        ax.set_xlabel(metric_type)
        ax.set_ylabel('Density')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: GT with 2-component Gaussian mixture
        ax = axes[1]
        ax.hist(gt_positive_acts, bins=n_bins, density=True, alpha=0.5,
                color='green', edgecolor='darkgreen')
        
        # Add threshold on panel 2
        if thresholds and concept_name in thresholds:
            threshold_val = thresholds[concept_name]['best_threshold']
            ax.axvline(threshold_val, color='purple', linestyle='-', linewidth=2.5, alpha=0.7)
        
        # Always show mixture model, regardless of best_model
        gmm_means = gmm_result['mixture_gaussian']['means']
        gmm_stds = gmm_result['mixture_gaussian']['stds'] 
        gmm_weights = gmm_result['mixture_gaussian']['weights']
        gmm_dfs = gmm_result['mixture_gaussian']['dfs']
        bg_overlaps = gmm_result['mixture_gaussian']['background_overlaps']
        
        # Sort components by mean to have consistent coloring
        if gmm_dfs is None:  # Gaussian
            component_data = list(zip(gmm_means, gmm_stds, gmm_weights))
            component_data.sort(key=lambda x: x[0])  # Sort by mean
            gmm_means, gmm_stds, gmm_weights = zip(*component_data)
        else:  # Student-t
            component_data = list(zip(gmm_means, gmm_stds, gmm_weights, gmm_dfs))
            component_data.sort(key=lambda x: x[0])  # Sort by mean
            gmm_means, gmm_stds, gmm_weights, gmm_dfs = zip(*component_data)
        
        # Plot individual components with consistent colors (orange, red)
        colors = ['orange', 'red']
        for i in range(len(gmm_means)):
            if gmm_dfs is None:  # Gaussian
                component_pdf = gmm_weights[i] * stats.norm.pdf(x_range, gmm_means[i], gmm_stds[i])
            else:  # Student-t
                component_pdf = gmm_weights[i] * stats.t.pdf(x_range, df=gmm_dfs[i], 
                                                            loc=gmm_means[i], scale=gmm_stds[i])
            color = colors[i] if i < len(colors) else f'C{i}'
            ax.plot(x_range, component_pdf, linestyle='--', linewidth=2.5, alpha=0.8,
                   color=color)
        
        # Plot full mixture
        if gmm_dfs is None:  # Gaussian
            mixture_pdf = sum(gmm_weights[i] * stats.norm.pdf(x_range, gmm_means[i], gmm_stds[i]) 
                            for i in range(len(gmm_means)))
        else:  # Student-t
            mixture_pdf = sum(gmm_weights[i] * stats.t.pdf(x_range, df=gmm_dfs[i], 
                                                          loc=gmm_means[i], scale=gmm_stds[i]) 
                            for i in range(len(gmm_means)))
        ax.plot(x_range, mixture_pdf, 'b-', linewidth=3, alpha=0.9)
        
        # Add average likelihood and mixture coefficients
        mixture_ll = gmm_result['mixture_gaussian']['log_likelihood']
        n_samples = len(gt_positive_acts)
        avg_ll = mixture_ll / n_samples
        weights_str = ', '.join([f'{w:.2f}' for w in gmm_weights])
        separation_text = f"Avg log-likelihood: {avg_ll:.4f}\nWeights: [{weights_str}]"
        
        # Add df values for Student-t
        if gmm_dfs is not None:
            dfs_str = ', '.join([f'{df:.1f}' for df in gmm_dfs])
            separation_text += f"\ndfs: [{dfs_str}]"
        
        # Add overlap metrics if available
        if bg_overlaps is not None:
            overlaps_str = ', '.join([f'{o:.2%}' for o in bg_overlaps])
            separation_text += f"\nBg overlaps: [{overlaps_str}]"
        
        ax.text(0.02, 0.98, separation_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        dist_type = 'Student-t' if signal_dist == 'student-t' else 'Gaussian'
        ax.set_title(f'GT with 2-Component {dist_type} Mixture', fontsize=12)
        ax.set_xlabel(metric_type)
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: GT decomposed into background + excess signal
        ax = axes[2]
        ax.hist(gt_positive_acts, bins=n_bins, density=True, alpha=0.5,
                color='green', edgecolor='darkgreen')
        
        # Add threshold on panel 3
        if thresholds and concept_name in thresholds:
            threshold_val = thresholds[concept_name]['best_threshold']
            ax.axvline(threshold_val, color='purple', linestyle='-', linewidth=2.5, alpha=0.7)
        
        # Get components from decomposition
        pi = decomp_mixture_fit['pi']
        background_pdf = decomp_mixture_fit['background']['pdf']
        signal_pdf = decomp_mixture_fit['signal']['pdf']
        mixture_pdf = decomp_mixture_fit['mixture_pdf']
        
        # Plot decomposition components with consistent colors
        bg_component = (1 - pi) * background_pdf(x_range)
        signal_component = pi * signal_pdf(x_range)
        
        ax.plot(x_range, bg_component, 'orange', linewidth=2.5, linestyle='--', alpha=0.8)
        ax.plot(x_range, signal_component, 'red', linewidth=2.5, linestyle='--', alpha=0.8)
        ax.plot(x_range, mixture_pdf(x_range), 'b-', linewidth=3, alpha=0.9)
        
        # Add average likelihood, mixture coefficients, and shift
        decomp_ll = decomp_mixture_fit['log_likelihood']
        n_samples = len(gt_positive_acts)
        avg_ll = decomp_ll / n_samples
        bg_shift = decomp_mixture_fit['background'].get('shift', 0.0)
        decomp_text = f"Avg log-likelihood: {avg_ll:.4f}\nWeights: [{1-pi:.2f}, {pi:.2f}]"
        if abs(bg_shift) > 0.001:
            decomp_text += f"\nNon-Concept shift: {bg_shift:.3f}"
        
        # Compute overlaps between components and actual background distribution
        # Get actual background PDF (from raw background_acts)
        actual_bg_hist, actual_bg_bins = np.histogram(background_acts, bins=n_bins, density=True)
        actual_bg_centers = (actual_bg_bins[:-1] + actual_bg_bins[1:]) / 2
        from scipy.interpolate import interp1d
        actual_bg_pdf_interp = interp1d(actual_bg_centers, actual_bg_hist, kind='linear', 
                                       bounds_error=False, fill_value=0)
        actual_bg_pdf = lambda x: np.maximum(actual_bg_pdf_interp(x), 1e-10)
        
        # Compute overlap between fitted background component and actual background
        bg_component_overlap = np.trapz(np.minimum(background_pdf(x_range), actual_bg_pdf(x_range)), x_range)
        
        # Compute overlap between signal component and actual background
        signal_bg_overlap = np.trapz(np.minimum(signal_pdf(x_range), actual_bg_pdf(x_range)), x_range)
        
        # Add overlaps to text
        decomp_text += f"\nBg overlaps: [{bg_component_overlap:.2%}, {signal_bg_overlap:.2%}]"
        
        ax.text(0.02, 0.98, decomp_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        signal_type = decomp_mixture_fit['signal'].get('type', 'gaussian')
        if signal_type == 'student-t':
            df_val = decomp_mixture_fit['signal'].get('df', 2.0)
            ax.set_title(f'GT = Empirical Non-Concept + Student-t Signal (df={df_val:.1f})', fontsize=12)
        else:
            ax.set_title('GT = Empirical Non-Concept + Gaussian Signal', fontsize=12)
        ax.set_xlabel(metric_type)
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(f"Unified Mixture Analysis: '{concept_name}' ({dataset_name} - {model_name})", fontsize=14)
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            if save_path is None:
                save_path = f"Unified_Mixture_Plots/{dataset_name}/{model_name}_{concept_type}_percentthrumodel_{percent_thru_model}"
            os.makedirs(save_path, exist_ok=True)
            plot_file = os.path.join(save_path, f"{concept_name}_unified_mixture.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"  Saved plot to: {plot_file}")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        # Calculate separability metrics
        def calculate_bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
            """Calculate Bhattacharyya distance between two Gaussians."""
            var1 = sigma1 ** 2
            var2 = sigma2 ** 2
            avg_var = (var1 + var2) / 2
            term1 = 0.25 * ((mu1 - mu2) ** 2) / avg_var
            term2 = 0.25 * np.log((var1 * var2) / (avg_var ** 2))
            return term1 + term2
        
        # Extract metrics for GMM - always use mixture metrics
        gmm_means = gmm_result['mixture_gaussian']['means']
        gmm_stds = gmm_result['mixture_gaussian']['stds']
        gmm_weights = gmm_result['mixture_gaussian']['weights']
        
        gmm_metrics = {
            'mixture_coefficients': gmm_weights,
            'background_overlaps': gmm_result['mixture_gaussian']['background_overlaps'],
            'distribution_type': signal_dist,
            'separability': {
                'separation_score': gmm_result['separation_score'],
                'mean_distance': abs(gmm_means[1] - gmm_means[0]),
                'normalized_distance': abs(gmm_means[1] - gmm_means[0]) / np.mean(gmm_stds),
                'bhattacharyya_distance': calculate_bhattacharyya_distance(
                    gmm_means[0], gmm_stds[0], gmm_means[1], gmm_stds[1]
                )
            },
            'fit_quality': {
                'bic': gmm_result['mixture_gaussian']['bic'],
                'aic': gmm_result['mixture_gaussian']['aic'],
                'log_likelihood': gmm_result['mixture_gaussian']['log_likelihood']
            },
            'best_model': gmm_result['best_model']  # Keep track of what was actually best
        }
        
        # Extract metrics for decomposition (using empirical stats for KDE)
        bg_mean = np.mean(background_acts)
        bg_std = np.std(background_acts)
        bg_shift = decomp_mixture_fit['background'].get('shift', 0.0)
        
        # Adjust background mean for shift
        effective_bg_mean = bg_mean + bg_shift
        
        decomp_metrics = {
            'mixture_coefficients': [1 - pi, pi],  # [background, signal]
            'background_shift': bg_shift,
            'background_overlaps': [bg_component_overlap, signal_bg_overlap],  # [bg component overlap, signal overlap]
            'signal_distribution_type': signal_dist,
            'separability': {
                'mean_distance': abs(decomp_mixture_fit['signal']['mean'] - effective_bg_mean),
                'normalized_distance': abs(decomp_mixture_fit['signal']['mean'] - effective_bg_mean) / 
                                     np.mean([decomp_mixture_fit['signal']['std'], bg_std]),
                'bhattacharyya_distance': calculate_bhattacharyya_distance(
                    effective_bg_mean, bg_std,
                    decomp_mixture_fit['signal']['mean'], decomp_mixture_fit['signal']['std']
                )
            },
            'fit_quality': {
                'log_likelihood': decomp_mixture_fit['log_likelihood']
            }
        }
        
        # Store simplified results
        results[concept_name] = {
            'success': True,
            'two_gaussian_mixture': gmm_metrics,
            'background_signal_decomposition': decomp_metrics,
            'data_stats': {
                'n_gt_positive': len(gt_positive_acts),
                'n_background': len(background_acts)
            }
        }
    
    # Summary statistics
    print(f"\n=== Analysis complete ===")
    successful_results = [r for r in results.values() if r.get('success', False)]
    
    summary = {
        'total_concepts_analyzed': len(results),
        'successful_analyses': len(successful_results)
    }
    
    print(f"\nSummary:")
    print(f"  Total concepts analyzed: {summary['total_concepts_analyzed']}")
    print(f"  Successful analyses: {summary['successful_analyses']}")
    
    return {
        'concept_results': results,
        'summary': summary,
        'dataset_info': {
            'dataset': dataset_name,
            'model': model_name,
            'concept_type': concept_type,
            'percent_thru_model': percent_thru_model
        }
    }


def fit_background_distribution(background_data: np.ndarray, 
                                method: str = 'gaussian',
                                kde_bandwidth: Optional[float] = None,
                                use_fast_kde: bool = True) -> Dict:
    """
    Fit a distribution to background-only data.
    
    Args:
        background_data: Array of background activation values
        method: 'gaussian' or 'kde'
        kde_bandwidth: Bandwidth for KDE (if None, uses Scott's rule)
        use_fast_kde: Use fast KDE implementation for large datasets
    
    Returns:
        Dictionary with fitted distribution parameters and functions
    """
    if method == 'gaussian':
        mean = np.mean(background_data)
        std = np.std(background_data)
        
        return {
            'type': 'gaussian',
            'mean': mean,
            'std': std,
            'pdf': lambda x: stats.norm.pdf(x, mean, std),
            'log_pdf': lambda x: stats.norm.logpdf(x, mean, std)
        }
    
    elif method == 'kde':
        # For large datasets, use a fast approximate KDE
        if use_fast_kde and len(background_data) > 5000:
            # Subsample for efficiency
            subsample_size = min(5000, len(background_data))
            indices = np.random.choice(len(background_data), subsample_size, replace=False)
            kde_data = background_data[indices]
        else:
            kde_data = background_data
            
        kde = stats.gaussian_kde(kde_data, bw_method=kde_bandwidth)
        
        # Create optimized PDF function
        def fast_pdf(x):
            if isinstance(x, np.ndarray):
                return kde.pdf(x).flatten() if x.ndim > 1 else kde.pdf(x)
            else:
                return kde.pdf(x)
                
        def fast_log_pdf(x):
            pdf_vals = fast_pdf(x)
            return np.log(pdf_vals + 1e-10)
        
        return {
            'type': 'kde',
            'kde': kde,
            'pdf': fast_pdf,
            'log_pdf': fast_log_pdf
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def fit_constrained_mixture(observed_data: np.ndarray,
                           background_dist: Dict,
                           signal_bounds: Optional[Tuple[float, float]] = None,
                           initial_pi: float = 0.2,
                           method: str = 'em',
                           use_gpu: bool = True,
                           allow_background_shift: bool = False,
                           signal_dist: str = 'gaussian') -> Dict:
    """
    Fit a mixture model: f_obs(x) ≈ (1-π)f_background(x - δ) + π f_signal(x)
    where f_background can be shifted by δ and we estimate π, δ, and f_signal.
    
    Args:
        observed_data: Array of observed activation values
        background_dist: Dictionary from fit_background_distribution
        signal_bounds: Optional bounds for signal mean (min, max)
        initial_pi: Initial guess for mixing proportion
        method: 'mle' for maximum likelihood or 'em' for EM algorithm
        use_gpu: Whether to use GPU acceleration when available (for EM method)
        allow_background_shift: Whether to allow horizontal shift of background distribution
        signal_dist: 'gaussian' or 'student-t' for signal component distribution
    
    Returns:
        Dictionary with mixture parameters and component distributions
    """
    n_obs = len(observed_data)
    
    # If allowing background shift, find optimal shift first
    best_shift = 0.0
    if allow_background_shift:
        print("    Finding optimal background shift...")
        # Try different shifts and find the one with best log-likelihood
        shift_range = np.linspace(-0.2, 0.2, 11)  # Reduced from 21 to 11 for speed
        best_ll = -np.inf
        
        for shift in shift_range:
            # Create shifted background PDF
            shifted_pdf = lambda x: background_dist['pdf'](x - shift)
            
            # Quick EM with fixed iterations to evaluate this shift
            # Initialize parameters
            high_acts = observed_data[observed_data > np.percentile(observed_data, 75)]
            test_signal_mean = np.mean(high_acts) if len(high_acts) > 0 else np.mean(observed_data)
            test_signal_std = np.std(observed_data) * 0.5
            test_pi = 0.2
            
            # Run 5 EM iterations (reduced from 10 for speed)
            bg_probs = shifted_pdf(observed_data)
            
            # For student-t, we use df=2 as default during optimization
            test_df = 2.0 if signal_dist == 'student-t' else None
            
            for _ in range(5):
                # E-step
                if signal_dist == 'student-t':
                    signal_probs = stats.t.pdf(observed_data, df=test_df, loc=test_signal_mean, scale=test_signal_std)
                else:
                    signal_probs = stats.norm.pdf(observed_data, test_signal_mean, test_signal_std)
                    
                numerator = test_pi * signal_probs
                denominator = (1 - test_pi) * bg_probs + test_pi * signal_probs + 1e-10
                responsibilities = numerator / denominator
                
                # M-step
                n_signal = responsibilities.sum()
                test_pi = n_signal / n_obs
                
                if n_signal > 1:
                    test_signal_mean = (responsibilities * observed_data).sum() / n_signal
                    test_signal_var = (responsibilities * (observed_data - test_signal_mean)**2).sum() / n_signal
                    test_signal_std = np.sqrt(max(test_signal_var, 1e-6))
            
            # Compute log-likelihood
            mixture_probs = (1 - test_pi) * bg_probs + test_pi * signal_probs
            ll = np.log(np.maximum(mixture_probs, 1e-10)).sum()
            
            if ll > best_ll:
                best_ll = ll
                best_shift = shift
        
        print(f"    Optimal shift: {best_shift:.3f}")
        
        # Create permanently shifted background distribution
        original_pdf = background_dist['pdf']
        original_log_pdf = background_dist['log_pdf'] if 'log_pdf' in background_dist else lambda x: np.log(original_pdf(x) + 1e-10)
        
        background_dist = background_dist.copy()
        background_dist['pdf'] = lambda x: original_pdf(x - best_shift)
        background_dist['log_pdf'] = lambda x: original_log_pdf(x - best_shift)
        background_dist['shift'] = best_shift
    
    if method == 'mle':
        # Use MLE to fit signal component and mixing proportion
        
        def neg_log_likelihood(params):
            """Negative log-likelihood of mixture model."""
            if signal_dist == 'student-t':
                pi = params[0]
                signal_mean = params[1]
                signal_std = params[2]
                signal_df = params[3]
                
                # Ensure valid parameters
                if pi <= 0 or pi >= 1 or signal_std <= 0 or signal_df <= 0:
                    return np.inf
                    
                # Compute mixture likelihood for each point
                bg_probs = background_dist['pdf'](observed_data)
                signal_probs = stats.t.pdf(observed_data, df=signal_df, loc=signal_mean, scale=signal_std)
            else:
                pi = params[0]
                signal_mean = params[1]
                signal_std = params[2]
                
                # Ensure valid parameters
                if pi <= 0 or pi >= 1 or signal_std <= 0:
                    return np.inf
                
                # Compute mixture likelihood for each point
                bg_probs = background_dist['pdf'](observed_data)
                signal_probs = stats.norm.pdf(observed_data, signal_mean, signal_std)
            
            mixture_probs = (1 - pi) * bg_probs + pi * signal_probs
            
            # Avoid log(0)
            mixture_probs = np.maximum(mixture_probs, 1e-10)
            
            return -np.sum(np.log(mixture_probs))
        
        # Set bounds
        if signal_dist == 'student-t':
            bounds = [
                (0.01, 0.99),  # pi
                (signal_bounds if signal_bounds else (observed_data.min(), observed_data.max())),  # signal mean
                (0.01, 10 * np.std(observed_data)),  # signal std
                (1.0, 30.0)  # degrees of freedom
            ]
            
            # Initial guess
            initial_guess = [
                initial_pi,
                np.mean(observed_data[observed_data > np.percentile(observed_data, 80)]),  # High activations
                np.std(observed_data) * 0.5,
                2.0  # Initial df
            ]
        else:
            bounds = [
                (0.01, 0.99),  # pi
                (signal_bounds if signal_bounds else (observed_data.min(), observed_data.max())),  # signal mean
                (0.01, 10 * np.std(observed_data))  # signal std
            ]
            
            # Initial guess
            initial_guess = [
                initial_pi,
                np.mean(observed_data[observed_data > np.percentile(observed_data, 80)]),  # High activations
                np.std(observed_data) * 0.5
            ]
        
        # Optimize
        result = differential_evolution(neg_log_likelihood, bounds, seed=42, maxiter=1000)
        
        if result.success:
            if signal_dist == 'student-t':
                pi_opt, signal_mean_opt, signal_std_opt, signal_df_opt = result.x
                
                return {
                    'success': True,
                    'pi': pi_opt,
                    'signal': {
                        'type': 'student-t',
                        'mean': signal_mean_opt,
                        'std': signal_std_opt,
                        'df': signal_df_opt,
                        'pdf': lambda x: stats.t.pdf(x, df=signal_df_opt, loc=signal_mean_opt, scale=signal_std_opt)
                    },
                    'background': background_dist,
                    'log_likelihood': -result.fun,
                    'mixture_pdf': lambda x: (1 - pi_opt) * background_dist['pdf'](x) + 
                                           pi_opt * stats.t.pdf(x, df=signal_df_opt, loc=signal_mean_opt, scale=signal_std_opt)
                }
            else:
                pi_opt, signal_mean_opt, signal_std_opt = result.x
                
                return {
                    'success': True,
                    'pi': pi_opt,
                    'signal': {
                        'type': 'gaussian',
                        'mean': signal_mean_opt,
                        'std': signal_std_opt,
                        'pdf': lambda x: stats.norm.pdf(x, signal_mean_opt, signal_std_opt)
                    },
                    'background': background_dist,
                    'log_likelihood': -result.fun,
                    'mixture_pdf': lambda x: (1 - pi_opt) * background_dist['pdf'](x) + 
                                           pi_opt * stats.norm.pdf(x, signal_mean_opt, signal_std_opt)
                }
        else:
            return {'success': False, 'message': 'Optimization failed'}
    
    elif method == 'em':
        # EM algorithm for mixture fitting with fixed background
        
        # Check if GPU acceleration is available and requested
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        if device.type == 'cuda' and len(observed_data) > 1000:  # Use GPU for larger datasets
            # GPU-accelerated EM algorithm
            obs_tensor = torch.from_numpy(observed_data).float().to(device)
            
            # Initialize signal parameters
            high_threshold = torch.quantile(obs_tensor, 0.75)
            high_acts = obs_tensor[obs_tensor > high_threshold]
            signal_mean = high_acts.mean() if len(high_acts) > 0 else obs_tensor.mean()
            signal_std = obs_tensor.std() * 0.5
            pi = torch.tensor(initial_pi, device=device)
            
            # For student-t, use fixed df for simplicity in EM
            if signal_dist == 'student-t':
                signal_df = torch.tensor(2.0, device=device)  # Can be optimized later
            
            max_iter = 30  # Reduced from 100 for speed
            tol = 1e-5  # Slightly relaxed tolerance
            prev_ll = -float('inf')
            
            # Pre-compute background probabilities once (this is the slow part)
            bg_probs_np = background_dist['pdf'](observed_data)
            bg_probs = torch.from_numpy(bg_probs_np).float().to(device)
            
            for iteration in range(max_iter):
                # E-step: compute responsibilities
                
                # Signal probabilities using torch distributions
                if signal_dist == 'student-t':
                    # Use torch's StudentT distribution
                    signal_distribution = torch.distributions.StudentT(df=signal_df, loc=signal_mean, scale=signal_std)
                    signal_probs = torch.exp(signal_distribution.log_prob(obs_tensor))
                else:
                    signal_distribution = torch.distributions.Normal(signal_mean, signal_std)
                    signal_probs = torch.exp(signal_distribution.log_prob(obs_tensor))
                
                # Posterior probability that each point belongs to signal
                numerator = pi * signal_probs
                denominator = (1 - pi) * bg_probs + pi * signal_probs + 1e-10
                responsibilities = numerator / denominator
                
                # M-step: update parameters
                n_signal = responsibilities.sum()
                pi = n_signal / n_obs
                
                if n_signal > 1:  # Avoid division by zero
                    signal_mean = (responsibilities * obs_tensor).sum() / n_signal
                    signal_var = (responsibilities * (obs_tensor - signal_mean)**2).sum() / n_signal
                    signal_std = torch.sqrt(torch.maximum(signal_var, torch.tensor(1e-6, device=device)))
                    
                    # Update df for Student-t distribution
                    if signal_dist == 'student-t':
                        # Compute expected values needed for df update
                        z = (obs_tensor - signal_mean) / signal_std
                        u = (signal_df + 1) / (signal_df + z**2)
                        
                        # Expected log(u) weighted by responsibilities
                        e_log_u = (responsibilities * torch.log(u)).sum() / n_signal
                        # Expected u weighted by responsibilities  
                        e_u = (responsibilities * u).sum() / n_signal
                        
                        # Newton-Raphson for df (simplified, a few iterations)
                        for _ in range(3):
                            df = signal_df
                            # Note: torch.digamma is the digamma function
                            f = torch.digamma((df + 1)/2) - torch.digamma(df/2) - torch.log(df) - 1/df - e_log_u + e_u
                            # Polygamma function for derivative
                            f_prime = 0.5 * (torch.special.polygamma(1, (df + 1)/2) - torch.special.polygamma(1, df/2)) - 1/df + 1/(df**2)
                            df_new = df - f / f_prime
                            df_new = torch.clamp(df_new, min=1.5, max=50.0)
                            if torch.abs(df_new - df) < 0.1:
                                break
                            signal_df = df_new
                
                # Compute log-likelihood
                mixture_probs = (1 - pi) * bg_probs + pi * signal_probs
                ll = torch.log(torch.maximum(mixture_probs, torch.tensor(1e-10, device=device))).sum()
                
                # Check convergence
                if abs(ll.item() - prev_ll) < tol:
                    break
                prev_ll = ll.item()
            
            # Convert results back to CPU/numpy
            if signal_dist == 'student-t':
                return {
                    'success': True,
                    'pi': pi.cpu().item(),
                    'signal': {
                        'type': 'student-t',
                        'mean': signal_mean.cpu().item(),
                        'std': signal_std.cpu().item(),
                        'df': signal_df.cpu().item(),
                        'pdf': lambda x: stats.t.pdf(x, df=signal_df.cpu().item(), loc=signal_mean.cpu().item(), scale=signal_std.cpu().item())
                    },
                    'background': background_dist,
                    'log_likelihood': ll.cpu().item(),
                    'responsibilities': responsibilities.cpu().numpy(),
                    'mixture_pdf': lambda x: (1 - pi.cpu().item()) * background_dist['pdf'](x) + 
                                           pi.cpu().item() * stats.t.pdf(x, df=signal_df.cpu().item(), loc=signal_mean.cpu().item(), scale=signal_std.cpu().item()),
                    'n_iterations': iteration + 1
                }
            else:
                return {
                    'success': True,
                    'pi': pi.cpu().item(),
                    'signal': {
                        'type': 'gaussian',
                        'mean': signal_mean.cpu().item(),
                        'std': signal_std.cpu().item(),
                        'pdf': lambda x: stats.norm.pdf(x, signal_mean.cpu().item(), signal_std.cpu().item())
                    },
                    'background': background_dist,
                    'log_likelihood': ll.cpu().item(),
                    'responsibilities': responsibilities.cpu().numpy(),
                    'mixture_pdf': lambda x: (1 - pi.cpu().item()) * background_dist['pdf'](x) + 
                                           pi.cpu().item() * stats.norm.pdf(x, signal_mean.cpu().item(), signal_std.cpu().item()),
                    'n_iterations': iteration + 1
                }
        else:
            # CPU version (original code)
            # Initialize signal parameters
            high_acts = observed_data[observed_data > np.percentile(observed_data, 75)]
            signal_mean = np.mean(high_acts) if len(high_acts) > 0 else np.mean(observed_data)
            signal_std = np.std(observed_data) * 0.5
            pi = initial_pi
            
            # For student-t, use fixed df for simplicity in EM
            if signal_dist == 'student-t':
                signal_df = 2.0  # Can be optimized later
            
            max_iter = 30  # Reduced from 100 for speed
            tol = 1e-5  # Slightly relaxed tolerance
            prev_ll = -np.inf
            
            # Pre-compute background probabilities once (this is the slow part)
            bg_probs = background_dist['pdf'](observed_data)
            
            for iteration in range(max_iter):
                # E-step: compute responsibilities
                if signal_dist == 'student-t':
                    signal_probs = stats.t.pdf(observed_data, df=signal_df, loc=signal_mean, scale=signal_std)
                else:
                    signal_probs = stats.norm.pdf(observed_data, signal_mean, signal_std)
                
                # Posterior probability that each point belongs to signal
                numerator = pi * signal_probs
                denominator = (1 - pi) * bg_probs + pi * signal_probs + 1e-10
                responsibilities = numerator / denominator
                
                # M-step: update parameters
                n_signal = np.sum(responsibilities)
                pi = n_signal / n_obs
                
                if n_signal > 1:  # Avoid division by zero
                    signal_mean = np.sum(responsibilities * observed_data) / n_signal
                    signal_var = np.sum(responsibilities * (observed_data - signal_mean)**2) / n_signal
                    signal_std = np.sqrt(max(signal_var, 1e-6))
                    
                    # Update df for Student-t distribution
                    if signal_dist == 'student-t':
                        from scipy.special import digamma, polygamma
                        
                        # Compute expected values needed for df update
                        z = (observed_data - signal_mean) / signal_std
                        u = (signal_df + 1) / (signal_df + z**2)
                        
                        # Expected log(u) weighted by responsibilities
                        e_log_u = np.sum(responsibilities * np.log(u)) / n_signal
                        # Expected u weighted by responsibilities  
                        e_u = np.sum(responsibilities * u) / n_signal
                        
                        # Newton-Raphson for df (simplified, a few iterations)
                        for _ in range(3):
                            df = signal_df
                            f = digamma((df + 1)/2) - digamma(df/2) - np.log(df) - 1/df - e_log_u + e_u
                            f_prime = 0.5 * (polygamma(1, (df + 1)/2) - polygamma(1, df/2)) - 1/df + 1/(df**2)
                            df_new = df - f / f_prime
                            df_new = np.clip(df_new, 1.5, 50.0)
                            if abs(df_new - df) < 0.1:
                                break
                            signal_df = df_new
                
                # Compute log-likelihood
                mixture_probs = (1 - pi) * bg_probs + pi * signal_probs
                ll = np.sum(np.log(np.maximum(mixture_probs, 1e-10)))
                
                # Check convergence
                if abs(ll - prev_ll) < tol:
                    break
                prev_ll = ll
            
            if signal_dist == 'student-t':
                return {
                    'success': True,
                    'pi': pi,
                    'signal': {
                        'type': 'student-t',
                        'mean': signal_mean,
                        'std': signal_std,
                        'df': signal_df,
                        'pdf': lambda x: stats.t.pdf(x, df=signal_df, loc=signal_mean, scale=signal_std)
                    },
                    'background': background_dist,
                    'log_likelihood': ll,
                    'responsibilities': responsibilities,
                    'mixture_pdf': lambda x: (1 - pi) * background_dist['pdf'](x) + 
                                           pi * stats.t.pdf(x, df=signal_df, loc=signal_mean, scale=signal_std),
                    'n_iterations': iteration + 1
                }
            else:
                return {
                    'success': True,
                    'pi': pi,
                    'signal': {
                        'type': 'gaussian',
                        'mean': signal_mean,
                        'std': signal_std,
                        'pdf': lambda x: stats.norm.pdf(x, signal_mean, signal_std)
                    },
                    'background': background_dist,
                    'log_likelihood': ll,
                    'responsibilities': responsibilities,
                    'mixture_pdf': lambda x: (1 - pi) * background_dist['pdf'](x) + 
                                           pi * stats.norm.pdf(x, signal_mean, signal_std),
                    'n_iterations': iteration + 1
                }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def bootstrap_mixture_test(background_data: np.ndarray,
                          observed_data: np.ndarray,
                          background_dist: Dict,
                          n_bootstrap: int = 100,
                          random_state: int = 42) -> Dict:
    """
    Bootstrap test for whether observed data needs a signal component beyond background.
    
    Args:
        background_data: Original background samples
        observed_data: Observed data to test
        background_dist: Fitted background distribution
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed
    
    Returns:
        Dictionary with p-value and bootstrap statistics
    """
    np.random.seed(random_state)
    n_obs = len(observed_data)
    
    # Fit mixture to observed data
    obs_mixture = fit_constrained_mixture(observed_data, background_dist, method='mle', use_gpu=True)
    if not obs_mixture['success']:
        return {'success': False, 'message': 'Failed to fit observed mixture'}
    
    obs_improvement = obs_mixture['log_likelihood'] - np.sum(background_dist['log_pdf'](observed_data))
    
    # Bootstrap: sample from background and test if mixture fits better
    bootstrap_improvements = []
    
    for i in range(n_bootstrap):
        # Sample from background distribution
        if background_dist['type'] == 'gaussian':
            bootstrap_sample = np.random.normal(background_dist['mean'], 
                                              background_dist['std'], 
                                              n_obs)
        else:  # KDE
            # Resample from background data and add small noise
            indices = np.random.choice(len(background_data), n_obs, replace=True)
            bootstrap_sample = background_data[indices]
            bootstrap_sample += np.random.normal(0, 0.01 * np.std(background_data), n_obs)
        
        # Fit mixture to bootstrap sample
        boot_mixture = fit_constrained_mixture(bootstrap_sample, background_dist, method='mle', use_gpu=True)
        
        if boot_mixture['success']:
            boot_improvement = boot_mixture['log_likelihood'] - np.sum(background_dist['log_pdf'](bootstrap_sample))
            bootstrap_improvements.append(boot_improvement)
    
    bootstrap_improvements = np.array(bootstrap_improvements)
    p_value = np.mean(bootstrap_improvements >= obs_improvement) if len(bootstrap_improvements) > 0 else 1.0
    
    return {
        'success': True,
        'observed_improvement': obs_improvement,
        'bootstrap_improvements': bootstrap_improvements,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def analyze_concept_activation_decomposition(
    concepts: Optional[Union[str, List[str]]] = None,
    dataset_name: str = None,
    model_name: str = None,
    concept_type: str = None,
    model_input_size: Optional[Tuple[int, int]] = None,
    percent_thru_model: int = 100,
    scratch_dir: str = '',
    n_bins: int = 100,
    show_threshold: bool = True,
    background_method: str = 'gaussian',
    mixture_method: str = 'em',
    n_bootstrap: int = 100,
    save_plots: bool = False,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Decompose GT positive patch activations using mixture model:
    f_observed(x) ≈ (1-π)f_background(x) + π f_signal(x)
    
    This rigorously shows that the observed GT distribution is composed of:
    1. A background component (fixed from control samples without the concept)
    2. A signal component (what's uniquely present for the concept)
    
    The method estimates the mixing proportion π and the signal distribution parameters.
    
    Args:
        concepts: Single concept name, list of concept names, or None for all concepts
        dataset_name: Name of dataset (e.g. 'CLEVR', 'COCO', 'Broden')
        model_name: Name of model (e.g. 'CLIP', 'Llama')
        concept_type: Type of concept - must be one of:
            'avg_patch_embeddings'
            'linsep_patch_embeddings_BD_True_BN_False'
            'kmeans_1000_patch_embeddings_kmeans'
            'kmeans_1000_linsep_patch_embeddings_kmeans'
        model_input_size: Model input size (will be inferred if None)
        percent_thru_model: Percentage through model (default 100)
        scratch_dir: Directory where activation files are stored
        n_bins: Number of bins for histograms
        show_threshold: Whether to show detection threshold on plots
        background_method: 'gaussian' or 'kde' for background distribution fitting
        mixture_method: 'mle' or 'em' for mixture fitting
        save_plots: Whether to save individual plots
        save_path: Directory to save plots (if None, uses default location)
        
    Returns:
        Dictionary containing:
        - 'concept_results': Results for each concept including:
            - 'mixture_fit': Fitted mixture model parameters
            - 'pi': Mixing proportion (fraction of signal)
            - 'signal_params': Parameters of signal distribution
            - 'bootstrap_test': Results of bootstrap significance test
            - 'responsibilities': Posterior probability each point is signal
        - 'summary': Summary statistics across all concepts
    """
    # Determine if text dataset
    is_text_dataset = dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank', 'IMDB', 'Jailbreak']
    unit_type = 'token' if is_text_dataset else 'patch'
    
    # Infer model input size if not provided
    if model_input_size is None:
        if is_text_dataset:
            if model_name == 'Llama':
                model_input_size = ('text', 'text')
            elif model_name == 'Gemma':
                model_input_size = ('text', 'text2')
            elif model_name == 'Qwen':
                model_input_size = ('text', 'text3')
            else:
                raise ValueError(f"Unknown text model: {model_name}")
        else:
            if 'CLIP' in model_name or 'clip' in model_name:
                model_input_size = (224, 224)
            elif 'Llama' in model_name or 'llama' in model_name:
                model_input_size = (560, 560)
            else:
                model_input_size = (224, 224)
    
    # Get activation file name based on concept type
    sample_type = 'patch'
    if concept_type == 'avg_patch_embeddings':
        acts_file = f"cosine_similarities_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        acts_file = f"dists_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"cosine_similarities_kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Cosine Similarity'
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        n_clusters = 1000
        acts_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        metric_type = 'Distance to Boundary'
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    # Load ground truth patches/tokens per concept
    gt_patches_file = f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
    if not os.path.exists(gt_patches_file):
        gt_patches_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
    
    if not os.path.exists(gt_patches_file):
        print(f"Error: Patch-level ground truth not found at {gt_patches_file}")
        return None
    
    try:
        gt_patches_per_concept = torch.load(gt_patches_file, weights_only=False)
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return None
    
    # Load ground truth samples per concept
    gt_samples_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
    
    try:
        gt_samples_per_concept = torch.load(gt_samples_file, weights_only=False)
    except Exception as e:
        print(f"Error loading ground truth samples file: {e}")
        return None
    
    # Filter to only include valid concepts
    gt_patches_per_concept = filter_concept_dict(gt_patches_per_concept, dataset_name)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    # Load activation loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
    except FileNotFoundError as e:
        print(f"Error: Could not find activation file: {acts_file}")
        return None
    
    # Load test activations
    test_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
    if test_acts is None:
        raise ValueError("Could not load test activations")
    
    # Load concepts to get proper indices
    if concept_type == 'avg_patch_embeddings':
        concepts_filename = f"avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
        concepts_filename = f"linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
        concepts_filename = f"kmeans_1000_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
        concepts_filename = f"kmeans_1000_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    
    concepts_file = f"Concepts/{dataset_name}/{concepts_filename}"
    
    if os.path.exists(concepts_file):
        concepts_data = torch.load(concepts_file, weights_only=False)
        if isinstance(concepts_data, dict):
            all_concept_names = list(concepts_data.keys())
        else:
            raise ValueError(f"Unexpected format in concepts file: {concepts_file}")
    else:
        raise FileNotFoundError(f"Concepts file not found: {concepts_file}")
    
    # Load thresholds if requested
    thresholds = None
    if show_threshold:
        threshold_file = None
        if concept_type == 'avg_patch_embeddings':
            threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_avg_patch_embeddings_percentthrumodel_{percent_thru_model}.pt'
        elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
            threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}.pt'
        elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
            threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
        elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
            threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
        
        if threshold_file and os.path.exists(threshold_file):
            try:
                thresholds = torch.load(threshold_file, weights_only=False)
            except Exception as e:
                print(f"Warning: Could not load thresholds: {e}")
    
    # Get test metadata
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    test_global_indices = metadata[metadata['split'] == 'test'].index.tolist()
    
    # Calculate patches/tokens per sample
    if not is_text_dataset:
        if model_input_size == (224, 224):
            patches_per_image = 256  # 16x16
        elif model_input_size == (560, 560):
            patches_per_image = 1600  # 40x40
        else:
            patches_per_image = 256
    
    # Determine which concepts to analyze
    if concepts is None:
        gt_concept_names = list(gt_patches_per_concept.keys())
        concepts_to_analyze = [c for c in gt_concept_names 
                             if not c.startswith('_') and c != 'metadata']
    elif isinstance(concepts, str):
        concepts_to_analyze = [concepts]
    else:
        concepts_to_analyze = list(concepts)
    
    # Results storage
    results = {}
    
    # Analyze each concept
    for concept_name in concepts_to_analyze:
        
        # Get concept index
        if concept_name not in all_concept_names:
            print(f"Warning: Concept '{concept_name}' not found in concepts file")
            continue
        
        concept_idx = all_concept_names.index(concept_name)
        
        # Get concept activations
        concept_acts = test_acts[:, concept_idx]
        
        # Keep on GPU if available for faster processing
        if not concept_acts.is_cuda and device.type == 'cuda':
            concept_acts = concept_acts.cuda()
        
        # 1. Get GT positive patch activations
        positive_patch_indices = gt_patches_per_concept.get(concept_name, [])
        
        if len(positive_patch_indices) == 0:
            print(f"No positive patches found for concept '{concept_name}'")
            continue
        
        # Collect GT positive activations efficiently
        if is_text_dataset:
            # TODO: Implement proper global to test mapping for text
            gt_positive_acts = []
            for patch_idx in positive_patch_indices:
                if patch_idx < len(concept_acts):
                    act_value = concept_acts[patch_idx].item()
                    gt_positive_acts.append(act_value)
            gt_positive_acts = np.array(gt_positive_acts)
        else:
            # Vectorized approach for images using GPU when possible
            valid_test_indices = []
            
            # Pre-compute test positions for all global indices
            test_pos_map = {idx: pos for pos, idx in enumerate(test_global_indices)}
            
            for global_patch_idx in positive_patch_indices:
                global_img_idx = global_patch_idx // patches_per_image
                patch_within_img = global_patch_idx % patches_per_image
                
                if global_img_idx in test_pos_map:
                    test_img_position = test_pos_map[global_img_idx]
                    test_patch_idx = test_img_position * patches_per_image + patch_within_img
                    
                    if test_patch_idx < len(concept_acts):
                        valid_test_indices.append(test_patch_idx)
            
            # Batch extract activations using GPU tensor operations
            if valid_test_indices:
                if device.type == 'cuda':
                    indices_tensor = torch.tensor(valid_test_indices, device=device)
                    gt_positive_acts = concept_acts[indices_tensor].cpu().numpy()
                else:
                    gt_positive_acts = concept_acts[valid_test_indices].numpy()
            else:
                gt_positive_acts = np.array([])
        
        # 2. Get background patch activations (from samples WITHOUT the concept)
        samples_with_concept = set(gt_samples_per_concept.get(concept_name, []))
        
        if is_text_dataset:
            # TODO: Implement for text datasets
            print("Text dataset background extraction not fully implemented")
            continue
        else:
            # Vectorized approach for background collection
            background_indices = []
            
            # Pre-collect all background patch indices
            for test_pos in range(len(test_global_indices)):
                if test_pos not in samples_with_concept:
                    # This image doesn't have the concept - add all its patches
                    start_patch = test_pos * patches_per_image
                    end_patch = min(start_patch + patches_per_image, len(concept_acts))
                    background_indices.extend(range(start_patch, end_patch))
            
            # Batch extract background activations
            if background_indices:
                if device.type == 'cuda':
                    indices_tensor = torch.tensor(background_indices, device=device)
                    background_acts = concept_acts[indices_tensor].cpu().numpy()
                else:
                    background_acts = concept_acts[background_indices].numpy()
            else:
                background_acts = np.array([])
        
        
        # Skip if not enough data
        if len(gt_positive_acts) < 10 or len(background_acts) < 50:
            results[concept_name] = {
                'success': False,
                'message': f'Insufficient data: {len(gt_positive_acts)} GT positive, {len(background_acts)} background'
            }
            continue
        
        # 3. Fit background distribution
        background_dist = fit_background_distribution(background_acts, method=background_method)
        
        # 4. Fit constrained mixture model to observed (GT positive) data
        mixture_fit = fit_constrained_mixture(gt_positive_acts, background_dist, method=mixture_method, use_gpu=True)
        
        if not mixture_fit['success']:
            results[concept_name] = {
                'success': False,
                'message': 'Failed to fit mixture model'
            }
            continue
        
        # 5. Bootstrap test for significance
        if n_bootstrap > 0:
            bootstrap_test = bootstrap_mixture_test(background_acts, gt_positive_acts, 
                                                  background_dist, n_bootstrap=n_bootstrap)
        else:
            bootstrap_test = {'success': False, 'message': 'Bootstrap test skipped'}
        
        # 6. Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Determine x range for all plots
        all_data = np.concatenate([gt_positive_acts, background_acts])
        x_min, x_max = np.min(all_data), np.max(all_data)
        x_range = np.linspace(x_min, x_max, 1000)
        
        # Get components from mixture fit
        pi = mixture_fit['pi']
        background_pdf = mixture_fit['background']['pdf']
        signal_pdf = mixture_fit['signal']['pdf']
        mixture_pdf = mixture_fit['mixture_pdf']
        
        # Plot 1: Non-Concept distribution (control samples)
        ax = axes[0]
        ax.hist(background_acts, bins=n_bins, density=True, alpha=0.7, 
                color='gray', edgecolor='black', label='Non-Concept data')
        ax.plot(x_range, background_pdf(x_range), 'k-', linewidth=2, label='Fitted background')
        ax.set_title(f'Non-Concept Distribution (n={len(background_acts)})', fontsize=12)
        ax.set_xlabel(metric_type)
        ax.set_ylabel('Density')
        ax.legend()
        
        # Plot 2: Observed (GT positive) vs Non-Concept
        ax = axes[1]
        ax.hist(gt_positive_acts, bins=n_bins, density=True, alpha=0.7,
                color='green', edgecolor='darkgreen', label='Observed (GT positive)')
        ax.plot(x_range, background_pdf(x_range), 'k--', linewidth=2, alpha=0.7, label='Non-Concept')
        
        # Shade excess region
        obs_hist, bin_edges = np.histogram(gt_positive_acts, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bg_at_bins = background_pdf(bin_centers)
        excess_mask = obs_hist > bg_at_bins
        
        if np.any(excess_mask):
            ax.fill_between(bin_centers[excess_mask], bg_at_bins[excess_mask], 
                           obs_hist[excess_mask], alpha=0.3, color='red',
                           label='Excess over background')
        
        # Add threshold if available
        if thresholds and concept_name in thresholds:
            threshold_val = thresholds[concept_name]['best_threshold']
            ax.axvline(threshold_val, color='purple', linestyle='-', linewidth=2.5, label='Detection threshold')
        
        ax.set_title(f'Observed vs Non-Concept (n={len(gt_positive_acts)})', fontsize=12)
        ax.set_xlabel(metric_type)
        ax.set_ylabel('Density')
        ax.legend()
        
        # Plot 3: Mixture decomposition
        ax = axes[2]
        ax.hist(gt_positive_acts, bins=n_bins, density=True, alpha=0.5,
                color='green', edgecolor='darkgreen', label='Observed data')
        
        # Plot mixture components
        bg_component = (1 - pi) * background_pdf(x_range)
        signal_component = pi * signal_pdf(x_range)
        
        ax.plot(x_range, bg_component, 'gray', linewidth=2, linestyle='--',
                label=f'Non-Concept component ({(1-pi)*100:.1f}%)')
        ax.plot(x_range, signal_component, 'red', linewidth=2, linestyle='--',
                label=f'Signal component ({pi*100:.1f}%)')
        ax.plot(x_range, mixture_pdf(x_range), 'b-', linewidth=3, alpha=0.8,
                label='Fitted mixture')
        
        # Fill areas to show decomposition
        ax.fill_between(x_range, 0, bg_component, alpha=0.3, color='gray')
        ax.fill_between(x_range, bg_component, bg_component + signal_component, 
                       alpha=0.3, color='red')
        
        ax.set_title('Mixture Decomposition: (1-π)×Non-Concept + π×Signal', fontsize=12)
        ax.set_xlabel(metric_type)
        ax.set_ylabel('Density')
        ax.legend()
        
        # Plot 4: Signal component and responsibilities
        ax = axes[3]
        
        if 'responsibilities' in mixture_fit:
            # Show posterior responsibilities
            resp = mixture_fit['responsibilities']
            scatter = ax.scatter(gt_positive_acts, np.random.uniform(-0.1, 0.1, len(gt_positive_acts)),
                               c=resp, cmap='RdYlGn', alpha=0.6, s=20)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('P(signal | x)', fontsize=10)
            ax.set_ylim(-0.5, 1.5)
            ax.set_ylabel('(jittered for visibility)')
            
            # Overlay signal distribution
            ax2 = ax.twinx()
            ax2.plot(x_range, signal_pdf(x_range), 'r-', linewidth=3, alpha=0.7, label='Signal distribution')
            ax2.fill_between(x_range, signal_pdf(x_range), alpha=0.2, color='red')
            ax2.set_ylabel('Signal density', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        else:
            # Just show signal distribution
            ax.plot(x_range, signal_pdf(x_range), 'r-', linewidth=3, label='Signal distribution')
            ax.fill_between(x_range, signal_pdf(x_range), alpha=0.3, color='red')
            ax.set_ylabel('Density')
        
        # Add statistics
        signal_mean = mixture_fit['signal']['mean']
        signal_std = mixture_fit['signal']['std']
        stats_text = f"Signal parameters:\nμ = {signal_mean:.3f}\nσ = {signal_std:.3f}\nπ = {pi:.3f}"
        
        if bootstrap_test['success']:
            stats_text += f"\n\nBootstrap test:\np-value = {bootstrap_test['p_value']:.4f}"
            if bootstrap_test['significant']:
                stats_text += "\n✓ Significant"
            else:
                stats_text += "\n✗ Not significant"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title('Signal Component & Posterior Responsibilities', fontsize=12)
        ax.set_xlabel(metric_type)
        ax.legend(loc='lower right')
        
        # Overall title
        fig.suptitle(f"Concept '{concept_name}' - GT Activation Decomposition (Non-Concept + Concept-specific)", fontsize=14)
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            if save_path is None:
                save_path = f"Activation_Difference_Plots/{dataset_name}/{model_name}_{concept_type}_percentthrumodel_{percent_thru_model}"
            os.makedirs(save_path, exist_ok=True)
            plot_file = os.path.join(save_path, f"{concept_name}_difference_analysis.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"  Saved plot to: {plot_file}")
        
        plt.show()
        
        # Store results
        results[concept_name] = {
            'success': True,
            'mixture_fit': {
                'pi': pi,
                'signal_mean': signal_mean,
                'signal_std': signal_std,
                'background_mean': background_dist['mean'] if background_dist['type'] == 'gaussian' else np.mean(background_acts),
                'background_std': background_dist['std'] if background_dist['type'] == 'gaussian' else np.std(background_acts),
                'log_likelihood': mixture_fit['log_likelihood']
            },
            'bootstrap_test': bootstrap_test,
            'n_gt_positive': len(gt_positive_acts),
            'n_background': len(background_acts),
            'responsibilities': mixture_fit.get('responsibilities', None)
        }
    
    # Summary statistics
    successful_results = [r for r in results.values() if r.get('success', False)]
    significant_results = [r for r in successful_results if r.get('bootstrap_test', {}).get('significant', False)]
    
    summary = {
        'total_concepts_analyzed': len(results),
        'successful_decompositions': len(successful_results),
        'significant_decompositions': len(significant_results),
        'avg_signal_proportion': np.mean([r['mixture_fit']['pi'] for r in successful_results]) if successful_results else 0,
        'avg_signal_mean': np.mean([r['mixture_fit']['signal_mean'] for r in successful_results]) if successful_results else 0,
        'concepts_with_significant_signal': [name for name, r in results.items() 
                                           if r.get('success', False) and r.get('bootstrap_test', {}).get('significant', False)]
    }
    
    return {
        'concept_results': results,
        'summary': summary,
        'dataset_info': {
            'dataset': dataset_name,
            'model': model_name,
            'concept_type': concept_type,
            'percent_thru_model': percent_thru_model
        }
    }


def compute_background_thresholds_and_detection(
    dataset_name: str,
    concept_type: str,
    model_name: str,
    sample_type: str,
    percentthrumodel: int,
    validation_split: str = 'val',
    test_split: str = 'test',
    background_percentile: float = 98.0,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    device: str = 'cuda',
    scratch_dir: str = '',
    verbose: bool = True
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute background activation thresholds and evaluate detection performance.
    
    This function:
    1. Computes the threshold as the Nth percentile of background activations 
       (patches/tokens from samples that don't contain the concept)
    2. Evaluates detection on ground truth positive test samples
    3. Returns per-concept detection rates with confidence intervals
    
    Args:
        dataset_name: Name of dataset (e.g., 'CLEVR', 'COCO')
        concept_type: Type of concepts to use ('avg', 'linsep', or 'kmeans_n' where n is number of clusters)
        model_name: Model name (e.g., 'Llama', 'CLIP')
        sample_type: 'patch' or 'cls'
        percentthrumodel: Percentage through model (e.g., 81, 92, 100)
        validation_split: Split to use for computing thresholds (default: 'val')
        test_split: Split to use for evaluation (default: 'test')
        background_percentile: Percentile of background activations to use as threshold (default: 98.0)
        n_bootstrap: Number of bootstrap samples for confidence intervals (default: 1000)
        confidence_level: Confidence level for intervals (default: 0.95)
        device: Device to use for computation
        scratch_dir: Base directory for data files
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing:
        - 'per_concept_detection': Dict mapping concept_id to detection percentage
        - 'per_concept_ci_lower': Dict mapping concept_id to lower confidence interval
        - 'per_concept_ci_upper': Dict mapping concept_id to upper confidence interval
        - 'weighted_mean_detection': Weighted mean detection rate across all concepts
        - 'weighted_mean_ci_lower': Lower CI for weighted mean
        - 'weighted_mean_ci_upper': Upper CI for weighted mean
        - 'concept_weights': Dict mapping concept_id to weight (number of test samples)
        - 'thresholds': Dict mapping concept_id to computed threshold
        
    Error bars explanation:
        - For individual concepts: Wilson score interval (better for proportions than normal approximation)
        - For weighted mean: Bootstrap with beta resampling within each concept's CI to preserve uncertainty
    """
    
    # Load ground truth samples
    gt_samples_path = os.path.join('GT_Samples', dataset_name, 
                                   f'{sample_type}_gt_samples_per_concept_sorted_by_split.pt')
    gt_samples_per_concept = torch.load(gt_samples_path, weights_only=False)
    
    # Load concept vectors
    concepts_dir = os.path.join('Concepts', dataset_name)
    
    if concept_type == 'avg':
        concepts_file = f'avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}.pt'
    elif concept_type == 'linsep':
        concepts_file = f'linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}.pt'
    elif concept_type.startswith('kmeans'):
        n_clusters = int(concept_type.split('_')[1])
        concepts_file = f'kmeans_{n_clusters}_{sample_type}_embeddings_{model_name}_kmeans_percentthrumodel_{percentthrumodel}.pt'
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    concepts = torch.load(os.path.join(concepts_dir, concepts_file), weights_only=False)
    
    # Determine model input size
    model_input_size = (224, 224) if model_name == 'CLIP' else (560, 560)
    
    # Initialize activation loader
    if concept_type in ['avg', 'linsep']:
        activation_dir = 'Cosine_Similarities' if concept_type == 'avg' else 'Distances'
        embeddings_file = f'{model_name}_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}.pt'
        
        if concept_type == 'avg':
            activation_file = f'avg_cosine_similarities_{embeddings_file}'
        else:
            activation_file = f'linsep_signed_distances_{embeddings_file}'
    else:  # kmeans
        activation_dir = 'Distances'
        n_clusters = int(concept_type.split('_')[1])
        embeddings_file = f'{model_name}_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}.pt'
        activation_file = f'kmeans_{n_clusters}_distances_{embeddings_file}'
    
    # Create activation loader
    activation_loader = ChunkedActivationLoader(
        dataset_name=dataset_name,
        embeddings_file=activation_file,
        scratch_dir=os.path.join(scratch_dir, activation_dir),
        device=device
    )
    
    # Results storage
    per_concept_thresholds = {}
    per_concept_detection_rates = {}
    per_concept_ci_lower = {}
    per_concept_ci_upper = {}
    concept_weights = {}
    
    # Process each concept
    concept_ids = sorted(gt_samples_per_concept.keys())
    
    for concept_id in tqdm(concept_ids, desc="Processing concepts", disable=not verbose):
        if verbose:
            print(f"\nProcessing concept {concept_id}")
            
        # Get ground truth samples for this concept
        gt_val_indices = gt_samples_per_concept[concept_id].get(validation_split, [])
        gt_test_indices = gt_samples_per_concept[concept_id].get(test_split, [])
        
        if len(gt_test_indices) == 0:
            if verbose:
                print(f"  Skipping concept {concept_id}: no test samples")
            continue
            
        # Compute threshold from validation background
        threshold = _compute_background_threshold_for_concept(
            concept_id, concepts, gt_val_indices, activation_loader, 
            dataset_name, model_input_size, sample_type, concept_type,
            background_percentile, device, verbose
        )
        per_concept_thresholds[concept_id] = threshold
        
        # Evaluate detection on test set
        detection_rate, ci_lower, ci_upper = _evaluate_detection_with_confidence(
            concept_id, concepts, gt_test_indices, activation_loader,
            dataset_name, model_input_size, sample_type, concept_type,
            threshold, confidence_level, device, verbose
        )
        
        per_concept_detection_rates[concept_id] = detection_rate
        per_concept_ci_lower[concept_id] = ci_lower
        per_concept_ci_upper[concept_id] = ci_upper
        concept_weights[concept_id] = len(gt_test_indices)
        
        if verbose:
            print(f"  Detection rate: {detection_rate:.2%} [{ci_lower:.2%}, {ci_upper:.2%}]")
            print(f"  Test samples: {len(gt_test_indices)}")
    
    # Compute weighted mean and its confidence interval
    weighted_mean, weighted_ci_lower, weighted_ci_upper = _compute_weighted_mean_with_bootstrap(
        per_concept_detection_rates, concept_weights, per_concept_ci_lower, per_concept_ci_upper,
        n_bootstrap, confidence_level
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Overall weighted detection rate: {weighted_mean:.2%} [{weighted_ci_lower:.2%}, {weighted_ci_upper:.2%}]")
        print(f"Total concepts evaluated: {len(per_concept_detection_rates)}")
        print(f"Total test samples: {sum(concept_weights.values())}")
    
    return {
        'per_concept_detection': per_concept_detection_rates,
        'per_concept_ci_lower': per_concept_ci_lower,
        'per_concept_ci_upper': per_concept_ci_upper,
        'weighted_mean_detection': weighted_mean,
        'weighted_mean_ci_lower': weighted_ci_lower,
        'weighted_mean_ci_upper': weighted_ci_upper,
        'concept_weights': concept_weights,
        'thresholds': per_concept_thresholds
    }


def _compute_background_threshold_for_concept(concept_id: int, concepts: torch.Tensor,
                                gt_positive_indices: List[int],
                                activation_loader: ChunkedActivationLoader,
                                dataset_name: str, model_input_size: Tuple[int, int],
                                sample_type: str, concept_type: str, percentile: float,
                                device: str, verbose: bool = True) -> float:
    """Compute threshold as percentile of background activations."""
    
    background_activations = []
    
    # If patch-level, we need to get all patches and filter out those from positive images
    if sample_type == 'patch':
        # Get all patch indices
        all_patch_indices = list(range(activation_loader.total_samples))
        
        # Filter to only patches that correspond to actual image content (not padding)
        valid_patch_indices = filter_patches_by_image_presence(
            all_patch_indices, dataset_name, model_input_size
        ).tolist()
        
        # Convert positive image indices to patch indices
        positive_patch_indices = set()
        patches_per_image = (model_input_size[0] // 14) * (model_input_size[1] // 14)
        
        for img_idx in gt_positive_indices:
            start_patch = img_idx * patches_per_image
            end_patch = start_patch + patches_per_image
            positive_patch_indices.update(range(start_patch, end_patch))
        
        # Non-Concept patches = valid patches - positive patches
        background_indices = [idx for idx in valid_patch_indices if idx not in positive_patch_indices]
        
    else:  # cls-level
        # Non-Concept samples = all samples - positive samples
        positive_set = set(gt_positive_indices)
        all_indices = list(range(activation_loader.total_samples))
        background_indices = [idx for idx in all_indices if idx not in positive_set]
    
    if verbose:
        print(f"  Computing threshold from {len(background_indices)} background samples")
    
    # Get activations for background samples
    concept_vector = concepts[concept_id].to(device)
    
    # Process in chunks to avoid memory issues
    chunk_size = 10000
    for i in range(0, len(background_indices), chunk_size):
        chunk_indices = background_indices[i:i+chunk_size]
        
        # Get activations directly from loader
        chunk_activations = activation_loader.get_embeddings(chunk_indices)
        
        # For kmeans, we need to compute negative distance
        if concept_type.startswith('kmeans'):
            chunk_activations = -chunk_activations[:, concept_id]
        else:
            chunk_activations = chunk_activations[:, concept_id]
        
        background_activations.extend(chunk_activations.cpu().numpy())
    
    # Compute percentile threshold
    threshold = np.percentile(background_activations, percentile)
    
    if verbose:
        print(f"  Non-Concept activation range: [{np.min(background_activations):.4f}, {np.max(background_activations):.4f}]")
        print(f"  {percentile}th percentile threshold: {threshold:.4f}")
    
    return float(threshold)


def _evaluate_detection_with_confidence(concept_id: int, concepts: torch.Tensor,
                                       gt_test_indices: List[int],
                                       activation_loader: ChunkedActivationLoader,
                                       dataset_name: str, model_input_size: Tuple[int, int],
                                       sample_type: str, concept_type: str, threshold: float,
                                       confidence_level: float, device: str,
                                       verbose: bool = True) -> Tuple[float, float, float]:
    """Evaluate detection rate on test samples and compute confidence intervals."""
    
    detected_samples = []
    
    if sample_type == 'patch':
        # For each test image, check if ANY patch exceeds threshold
        patches_per_image = (model_input_size[0] // 14) * (model_input_size[1] // 14)
        
        for img_idx in gt_test_indices:
            # Get all patches for this image
            start_patch = img_idx * patches_per_image
            end_patch = start_patch + patches_per_image
            patch_indices = list(range(start_patch, end_patch))
            
            # Filter to valid patches
            valid_patches = filter_patches_by_image_presence(
                patch_indices, dataset_name, model_input_size
            ).tolist()
            
            # Get activations for all patches
            patch_activations = activation_loader.get_embeddings(valid_patches)
            
            # Extract activations for this concept
            if concept_type.startswith('kmeans'):
                patch_activations = -patch_activations[:, concept_id]
            else:
                patch_activations = patch_activations[:, concept_id]
            
            # Check if any patch exceeds threshold
            detected = (patch_activations > threshold).any().item()
            detected_samples.append(detected)
            
    else:  # cls-level
        # Get activations for test samples
        test_activations = activation_loader.get_embeddings(gt_test_indices)
        
        # Extract activations for this concept
        if concept_type.startswith('kmeans'):
            test_activations = -test_activations[:, concept_id]
        else:
            test_activations = test_activations[:, concept_id]
        
        detected_samples = (test_activations > threshold).cpu().numpy()
    
    # Compute detection rate
    detection_rate = np.mean(detected_samples)
    n_detected = np.sum(detected_samples)
    n_total = len(detected_samples)
    
    # Compute confidence intervals using Wilson score interval
    ci_lower, ci_upper = _wilson_score_confidence_interval(n_detected, n_total, confidence_level)
    
    if verbose:
        print(f"  Detected {n_detected}/{n_total} samples ({detection_rate:.2%})")
    
    return detection_rate, ci_lower, ci_upper


def _wilson_score_confidence_interval(successes: int, trials: int, confidence: float) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.
    
    This is more accurate than normal approximation, especially for extreme proportions
    and small sample sizes.
    """
    if trials == 0:
        return 0.0, 0.0
    
    p = successes / trials
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    denominator = 1 + z**2 / trials
    centre_adjusted_probability = p + z**2 / (2 * trials)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials)
    
    lower = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    
    # Clamp to [0, 1]
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    
    return lower, upper


def _compute_weighted_mean_with_bootstrap(detection_rates: Dict[int, float],
                                         weights: Dict[int, int],
                                         ci_lowers: Dict[int, float],
                                         ci_uppers: Dict[int, float],
                                         n_bootstrap: int,
                                         confidence_level: float) -> Tuple[float, float, float]:
    """Compute weighted mean and its confidence interval using bootstrap."""
    
    # Convert to arrays
    concept_ids = sorted(detection_rates.keys())
    rates = np.array([detection_rates[c] for c in concept_ids])
    w = np.array([weights[c] for c in concept_ids])
    lowers = np.array([ci_lowers[c] for c in concept_ids])
    uppers = np.array([ci_uppers[c] for c in concept_ids])
    
    # Normalize weights
    w = w / w.sum()
    
    # Compute weighted mean
    weighted_mean = np.sum(rates * w)
    
    # Bootstrap for confidence interval
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Sample concepts with replacement
        indices = np.random.choice(len(concept_ids), size=len(concept_ids), replace=True)
        
        # Sample rates within their confidence intervals
        sampled_rates = []
        for i in indices:
            # Use beta distribution to sample within CI
            # This preserves the uncertainty in each concept's rate
            lower, upper = lowers[i], uppers[i]
            if lower == upper:
                sampled_rate = lower
            else:
                # Convert CI to beta parameters (method of moments)
                mean = rates[i]
                variance = ((upper - lower) / 4) ** 2  # Approximate variance
                
                # Ensure valid parameters
                if variance > 0 and mean > 0 and mean < 1:
                    alpha = mean * ((mean * (1 - mean) / variance) - 1)
                    beta = (1 - mean) * ((mean * (1 - mean) / variance) - 1)
                    if alpha > 0 and beta > 0:
                        sampled_rate = np.random.beta(alpha, beta)
                    else:
                        sampled_rate = np.random.uniform(lower, upper)
                else:
                    sampled_rate = mean
                    
            sampled_rates.append(sampled_rate)
        
        # Compute weighted mean for this bootstrap sample
        bootstrap_weights = w[indices]
        bootstrap_weights = bootstrap_weights / bootstrap_weights.sum()
        bootstrap_mean = np.sum(sampled_rates * bootstrap_weights)
        bootstrap_means.append(bootstrap_mean)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return mean_estimate, (ci_lower, ci_upper)


def plot_concept_activation_grid(
    concepts: List[str],
    dataset_name: str,
    model_name: str,
    concept_type: str,
    model_input_size: Tuple[int, int],
    percentthrumodels: Optional[List[int]] = None,
    scratch_dir: str = '',
    n_bins: int = 50,
    figsize_per_plot: Tuple[float, float] = (4, 3),
    show_thresholds: bool = True,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    max_concepts: int = 10,
    use_gpu: bool = True,
    batch_process: bool = True
) -> Dict[str, any]:
    """
    Display background vs GT positive activation overlays in a grid format.
    Rows represent concepts, columns represent percentthrumodel values.
    
    Args:
        concepts: List of concept names to analyze
        dataset_name: Name of dataset (e.g. 'CLEVR', 'COCO', 'Broden')
        model_name: Name of model (e.g. 'CLIP', 'Llama')
        concept_type: Type of concept - must be one of:
            'avg_patch_embeddings'
            'linsep_patch_embeddings_BD_True_BN_False'
            'kmeans_1000_patch_embeddings_kmeans'
            'kmeans_1000_linsep_patch_embeddings_kmeans'
        model_input_size: Model input size (e.g. (224, 224) for CLIP, (560, 560) for Llama, ('text', 'text') for text)
        percentthrumodels: List of percentthrumodel values (defaults to model-specific values)
        scratch_dir: Directory where activation files are stored
        n_bins: Number of bins for histograms
        figsize_per_plot: Size of each subplot
        show_thresholds: Whether to show detection thresholds
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        max_concepts: Maximum number of concepts to display
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n=== Creating concept activation grid ===")
    print(f"Dataset: {dataset_name}, Model: {model_name}")
    print(f"Concept type: {concept_type}")
    
    # Determine if text dataset
    is_text_dataset = dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank', 'IMDB', 'Jailbreak']
    unit_type = 'token' if is_text_dataset else 'patch'
    
    # Get default percentthrumodels if not provided
    if percentthrumodels is None:
        percentthrumodels = get_model_default_percentthrumodels(model_name, model_input_size)
    
    print(f"Using percentthrumodel values: {percentthrumodels}")
    
    # Don't limit concepts - process all of them
    print(f"Processing all {len(concepts)} concepts")
    
    # Determine metric type based on concept type
    sample_type = 'patch'
    if concept_type in ['avg_patch_embeddings', 'kmeans_1000_patch_embeddings_kmeans']:
        metric_type = 'Cosine Similarity'
        acts_prefix = 'cosine_similarities'
    else:
        metric_type = 'Distance to Boundary'
        acts_prefix = 'dists'
    
    # Load ground truth data (same for all percentthrumodels)
    print(f"\nLoading ground truth data...")
    gt_patches_file = f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
    if not os.path.exists(gt_patches_file):
        gt_patches_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
    
    if not os.path.exists(gt_patches_file):
        print(f"Error: Patch-level ground truth not found at {gt_patches_file}")
        return None
    
    try:
        gt_patches_per_concept = torch.load(gt_patches_file, weights_only=False)
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return None
    
    # Load ground truth samples
    gt_samples_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
    
    try:
        gt_samples_per_concept = torch.load(gt_samples_file, weights_only=False)
    except Exception as e:
        print(f"Error loading ground truth samples file: {e}")
        return None
    
    # Filter to valid concepts
    gt_patches_per_concept = filter_concept_dict(gt_patches_per_concept, dataset_name)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    # Load metadata
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    test_global_indices = metadata[metadata['split'] == 'test'].index.tolist()
    
    # Calculate patches per sample
    if not is_text_dataset:
        if model_input_size == (224, 224):
            patches_per_image = 256  # 16x16
        elif model_input_size == (560, 560):
            patches_per_image = 1600  # 40x40
        else:
            patches_per_image = 256
    
    has_padding = not is_text_dataset and model_input_size == (560, 560)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create figure
    n_rows = len(concepts)
    n_cols = len(percentthrumodels)
    fig_width = n_cols * figsize_per_plot[0]
    fig_height = n_rows * figsize_per_plot[1]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Results storage
    results = {}
    
    # Track global x-axis limits for consistent scaling
    global_x_min = float('inf')
    global_x_max = float('-inf')
    
    # Process each percentthrumodel
    for col_idx, percent_thru_model in enumerate(percentthrumodels):
        print(f"\n--- Processing percentthrumodel {percent_thru_model} ---")
        
        # Get activation file name
        if concept_type == 'avg_patch_embeddings':
            acts_file = f"{acts_prefix}_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
            acts_file = f"{acts_prefix}_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
            n_clusters = 1000
            acts_file = f"{acts_prefix}_kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
            n_clusters = 1000
            acts_file = f"{acts_prefix}_kmeans_{n_clusters}_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        
        # Load activations
        try:
            act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
        except FileNotFoundError as e:
            print(f"Error: Could not find activation file: {acts_file}")
            # Create empty plots for this column
            for row_idx in range(n_rows):
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            continue
        
        # Load test activations
        test_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
        if test_acts is None:
            print(f"Could not load test activations for percentthrumodel {percent_thru_model}")
            continue
        
        print(f"Test activations shape: {test_acts.shape}")
        
        # Load concepts to get indices
        if concept_type == 'avg_patch_embeddings':
            concepts_filename = f"avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
            concepts_filename = f"linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
            concepts_filename = f"kmeans_1000_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
            concepts_filename = f"kmeans_1000_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        
        concepts_file = f"Concepts/{dataset_name}/{concepts_filename}"
        
        if os.path.exists(concepts_file):
            concepts_data = torch.load(concepts_file, weights_only=False)
            if isinstance(concepts_data, dict):
                all_concept_names = list(concepts_data.keys())
            else:
                print(f"Unexpected format in concepts file: {concepts_file}")
                continue
        else:
            print(f"Concepts file not found: {concepts_file}")
            continue
        
        # Load thresholds if requested
        thresholds = None
        if show_thresholds:
            threshold_file = None
            if concept_type == 'avg_patch_embeddings':
                threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_avg_patch_embeddings_percentthrumodel_{percent_thru_model}.pt'
            elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
                threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}.pt'
            elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
                threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
            elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
                threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
            
            if threshold_file and os.path.exists(threshold_file):
                try:
                    thresholds = torch.load(threshold_file, weights_only=False)
                except Exception as e:
                    print(f"Warning: Could not load thresholds: {e}")
        
        # Batch process concepts if requested and on GPU
        if batch_process and device.type == 'cuda' and use_gpu:
            # Get all concept indices at once
            concept_indices = []
            valid_concepts = []
            for concept_name in concepts:
                if concept_name in all_concept_names:
                    concept_indices.append(all_concept_names.index(concept_name))
                    valid_concepts.append(concept_name)
            
            if concept_indices:
                # Get all concept activations at once on GPU
                concept_indices_tensor = torch.tensor(concept_indices, device=device)
                all_concept_acts = test_acts[:, concept_indices_tensor]  # Shape: (n_patches, n_concepts)
                
                # Pre-compute background indices once (same for all concepts)
                if not is_text_dataset:
                    all_test_positions = torch.arange(len(test_global_indices), device=device)
                    n_patches = min(patches_per_image, (test_acts.shape[0] - 1) // len(test_global_indices) + 1)
                    patch_offsets = torch.arange(n_patches, device=device)
        
        # Process each concept
        for row_idx, concept_name in enumerate(concepts):
            ax = axes[row_idx, col_idx]
            
            # Get concept index
            if concept_name not in all_concept_names:
                ax.text(0.5, 0.5, 'Concept not found', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            concept_idx = all_concept_names.index(concept_name)
            
            # Get concept activations
            if batch_process and device.type == 'cuda' and use_gpu and concept_name in valid_concepts:
                # Use pre-fetched GPU data
                valid_idx = valid_concepts.index(concept_name)
                concept_acts = all_concept_acts[:, valid_idx]
            else:
                # Original single concept approach
                concept_acts = test_acts[:, concept_idx]
                
                # Keep on GPU if available
                if not concept_acts.is_cuda and device.type == 'cuda' and use_gpu:
                    concept_acts = concept_acts.cuda()
            
            # Get GT positive patch indices
            positive_patch_indices = gt_patches_per_concept.get(concept_name, [])
            
            # Filter out padding patches if needed
            if has_padding and len(positive_patch_indices) > 0:
                positive_patch_indices = filter_patches_by_image_presence(
                    positive_patch_indices, dataset_name, model_input_size
                )
                positive_patch_indices = positive_patch_indices.tolist()
            
            if len(positive_patch_indices) == 0:
                ax.text(0.5, 0.5, 'No positive patches', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Collect GT positive activations (simplified version)
            if is_text_dataset:
                gt_positive_acts = []
                for patch_idx in positive_patch_indices:
                    if patch_idx < len(concept_acts):
                        act_value = concept_acts[patch_idx].item()
                        gt_positive_acts.append(act_value)
                gt_positive_acts = np.array(gt_positive_acts)
            else:
                # Vectorized approach for images
                positive_patch_indices = np.array(positive_patch_indices)
                global_img_indices = positive_patch_indices // patches_per_image
                patch_within_imgs = positive_patch_indices % patches_per_image
                
                test_global_set = set(test_global_indices)
                test_mask = np.array([img_idx in test_global_set for img_idx in global_img_indices])
                test_global_img_indices = global_img_indices[test_mask]
                test_patch_within_imgs = patch_within_imgs[test_mask]
                
                if len(test_global_img_indices) == 0:
                    gt_positive_acts = np.array([])
                else:
                    global_to_test_pos = {global_idx: test_pos for test_pos, global_idx in enumerate(test_global_indices)}
                    test_positions = np.array([global_to_test_pos[idx] for idx in test_global_img_indices])
                    test_patch_indices = test_positions * patches_per_image + test_patch_within_imgs
                    valid_indices = test_patch_indices[test_patch_indices < len(concept_acts)]
                    
                    if len(valid_indices) > 0:
                        if device.type == 'cuda' and concept_acts.is_cuda and use_gpu:
                            # Keep on GPU until final conversion
                            indices_tensor = torch.from_numpy(valid_indices).long().to(device)
                            gt_positive_acts_gpu = concept_acts[indices_tensor]
                            # Only move to CPU at the end
                            gt_positive_acts = gt_positive_acts_gpu.cpu().numpy()
                        else:
                            gt_positive_acts = concept_acts[valid_indices].cpu().numpy() if hasattr(concept_acts, 'cpu') else concept_acts[valid_indices].numpy()
                    else:
                        gt_positive_acts = np.array([])
            
            # Get background activations
            samples_with_concept = set(gt_samples_per_concept.get(concept_name, []))
            
            if is_text_dataset:
                background_acts = np.array([])  # Skip for text datasets in this simplified version
            else:
                # Vectorized background collection
                samples_with_concept_arr = np.array(list(samples_with_concept))
                all_test_positions = np.arange(len(test_global_indices))
                has_concept = np.zeros(len(all_test_positions), dtype=bool)
                if len(samples_with_concept_arr) > 0:
                    valid_concept_samples = samples_with_concept_arr[samples_with_concept_arr < len(all_test_positions)]
                    has_concept[valid_concept_samples] = True
                
                positions_without_concept = all_test_positions[~has_concept]
                
                if len(positions_without_concept) > 0:
                    n_patches = min(patches_per_image, (len(concept_acts) - 1) // len(test_global_indices) + 1)
                    patch_offsets = np.arange(n_patches)
                    base_indices = positions_without_concept * patches_per_image
                    all_bg_indices = base_indices[:, np.newaxis] + patch_offsets[np.newaxis, :]
                    background_indices = all_bg_indices.flatten()
                    background_indices = background_indices[background_indices < len(concept_acts)]
                    
                    if device.type == 'cuda' and concept_acts.is_cuda and use_gpu:
                        # Keep on GPU until final conversion
                        indices_tensor = torch.from_numpy(background_indices).long().to(device)
                        background_acts_gpu = concept_acts[indices_tensor]
                        # Sample if too many for histogram computation
                        if len(background_acts_gpu) > 100000:
                            sample_indices = torch.randperm(len(background_acts_gpu), device=device)[:100000]
                            background_acts_gpu = background_acts_gpu[sample_indices]
                        background_acts = background_acts_gpu.cpu().numpy()
                    else:
                        background_acts = concept_acts[background_indices].cpu().numpy() if hasattr(concept_acts, 'cpu') else concept_acts[background_indices].numpy()
                        if len(background_acts) > 100000:
                            background_acts = np.random.choice(background_acts, 100000, replace=False)
                else:
                    background_acts = np.array([])
            
            # Plot if we have data
            if len(gt_positive_acts) > 0 and len(background_acts) > 0:
                # Update global min/max
                all_acts = np.concatenate([background_acts, gt_positive_acts])
                global_x_min = min(global_x_min, np.min(all_acts))
                global_x_max = max(global_x_max, np.max(all_acts))
                
                ax.hist(background_acts, bins=n_bins, density=True, alpha=0.6, 
                        color='gray', edgecolor='black', label=f'Bg ({len(background_acts)})')
                ax.hist(gt_positive_acts, bins=n_bins, density=True, alpha=0.7,
                        color='green', edgecolor='darkgreen', label=f'GT+ ({len(gt_positive_acts)})')
                
                # Add threshold if available
                if thresholds and concept_name in thresholds:
                    threshold_val = thresholds[concept_name]['best_threshold']
                    ax.axvline(threshold_val, color='purple', linestyle='-', linewidth=2, label='Thresh')
                
                # Compute overlap metric
                # Method 1: Histogram intersection (area of overlap)
                hist_bg, bin_edges = np.histogram(background_acts, bins=n_bins, density=True)
                hist_gt, _ = np.histogram(gt_positive_acts, bins=bin_edges, density=True)
                bin_width = bin_edges[1] - bin_edges[0]
                overlap_area = np.sum(np.minimum(hist_bg, hist_gt)) * bin_width
                
                # Method 2: Separability score (1 - overlap_area)
                separability = max(0, 1 - overlap_area)
                
                # Method 3: Distribution statistics
                bg_mean, bg_std = np.mean(background_acts), np.std(background_acts)
                gt_mean, gt_std = np.mean(gt_positive_acts), np.std(gt_positive_acts)
                
                # Cohen's d (effect size)
                pooled_std = np.sqrt((bg_std**2 + gt_std**2) / 2)
                cohens_d = abs(gt_mean - bg_mean) / pooled_std if pooled_std > 0 else 0
                
                # Add text with metrics
                metrics_text = f'Sep: {separability:.2f}\nd: {cohens_d:.2f}'
                ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right', fontsize=12,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # Formatting
                if row_idx == 0:
                    # Get layer number based on model
                    if model_name == 'CLIP':
                        total_layers = 24
                    elif model_name == 'Llama' and is_text_dataset:
                        total_layers = 32
                    elif model_name == 'Llama' and not is_text_dataset:
                        total_layers = 40
                    elif model_name == 'Gemma':
                        total_layers = 28
                    elif model_name == 'Qwen':
                        total_layers = 32
                    else:
                        total_layers = 24  # Default
                    
                    layer_num = percent_to_layer(percent_thru_model, total_layers, model_name)
                    ax.set_title(f'{percent_thru_model}% (L{layer_num})', fontsize=14)
                if col_idx == 0:
                    ax.set_ylabel(f'{concept_name[:15]}...', fontsize=14)
                
                # Only show x-label on bottom row
                if row_idx == n_rows - 1:
                    ax.set_xlabel(metric_type, fontsize=12)
                
                ax.tick_params(labelsize=10)
                ax.grid(True, alpha=0.3)
                
                # Legend only on first plot
                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Store results
            key = f"{concept_name}_{percent_thru_model}"
            if len(gt_positive_acts) > 0 and len(background_acts) > 0:
                results[key] = {
                    'n_gt_positive': len(gt_positive_acts),
                    'n_background': len(background_acts),
                    'gt_mean': gt_mean if 'gt_mean' in locals() else np.mean(gt_positive_acts),
                    'bg_mean': bg_mean if 'bg_mean' in locals() else np.mean(background_acts),
                    'gt_std': gt_std if 'gt_std' in locals() else np.std(gt_positive_acts),
                    'bg_std': bg_std if 'bg_std' in locals() else np.std(background_acts),
                    'overlap_area': overlap_area if 'overlap_area' in locals() else None,
                    'separability': separability if 'separability' in locals() else None,
                    'cohens_d': cohens_d if 'cohens_d' in locals() else None
                }
            else:
                results[key] = {
                    'n_gt_positive': len(gt_positive_acts),
                    'n_background': len(background_acts),
                    'gt_mean': np.mean(gt_positive_acts) if len(gt_positive_acts) > 0 else None,
                    'bg_mean': np.mean(background_acts) if len(background_acts) > 0 else None,
                    'gt_std': np.std(gt_positive_acts) if len(gt_positive_acts) > 0 else None,
                    'bg_std': np.std(background_acts) if len(background_acts) > 0 else None,
                    'overlap_area': None,
                    'separability': None,
                    'cohens_d': None
                }
    
    # Overall title and layout
    fig.suptitle(f'{dataset_name} - {model_name}: Non-Concept vs GT Positive Activations', fontsize=18)
    
    # Apply consistent x-axis limits to all subplots
    if global_x_min != float('inf') and global_x_max != float('-inf'):
        # Add some padding (5% of range on each side)
        x_range = global_x_max - global_x_min
        x_padding = 0.05 * x_range
        x_min_padded = global_x_min - x_padding
        x_max_padded = global_x_max + x_padding
        
        # Apply to all subplots
        for row in axes:
            for ax in row:
                if ax.has_data():
                    ax.set_xlim(x_min_padded, x_max_padded)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return results


def compute_concept_activation_distributions(
    concepts: List[str],
    dataset_name: str,
    model_name: str,
    concept_type: str,
    model_input_size: Tuple[int, int],
    percentthrumodels: Optional[List[int]] = None,
    scratch_dir: str = '',
    n_bins: int = 50,
    compute_thresholds: bool = True,
    max_concepts: int = 10,
    use_gpu: bool = True,
    batch_process: bool = True,
    max_background_samples: int = 100000
) -> Dict[str, Any]:
    """
    Compute background vs GT positive activation distributions for concepts across layers.
    
    Args:
        concepts: List of concept names to analyze
        dataset_name: Name of dataset (e.g. 'CLEVR', 'COCO', 'Broden')
        model_name: Name of model (e.g. 'CLIP', 'Llama')
        concept_type: Type of concept - must be one of:
            'avg_patch_embeddings'
            'linsep_patch_embeddings_BD_True_BN_False'
            'kmeans_1000_patch_embeddings_kmeans'
            'kmeans_1000_linsep_patch_embeddings_kmeans'
        model_input_size: Model input size (e.g. (224, 224) for CLIP, (560, 560) for Llama, ('text', 'text') for text)
        percentthrumodels: List of percentthrumodel values (defaults to model-specific values)
        scratch_dir: Directory where activation files are stored
        n_bins: Number of bins for histograms
        compute_thresholds: Whether to load detection thresholds
        max_concepts: Maximum number of concepts to process
        use_gpu: Whether to use GPU for computation
        batch_process: Whether to batch process concepts on GPU
        max_background_samples: Maximum number of background samples to use
        
    Returns:
        Dictionary containing:
        - 'distributions': Dict[str, Dict] with keys "{concept}_{percentthru}" containing:
            - 'gt_positive_acts': Array of GT positive activations
            - 'background_acts': Array of background activations
            - 'n_gt_positive': Number of GT positive samples
            - 'n_background': Number of background samples
            - 'gt_mean', 'gt_std': GT distribution statistics
            - 'bg_mean', 'bg_std': Non-Concept distribution statistics
            - 'overlap_area': Histogram intersection area
            - 'separability': 1 - overlap_area
            - 'cohens_d': Cohen's d effect size
            - 'auc_roc': Area Under the ROC Curve
            - 'threshold': Detection threshold (if compute_thresholds=True)
        - 'metadata': Dict containing:
            - 'dataset_name', 'model_name', 'concept_type'
            - 'is_text_dataset': Boolean
            - 'unit_type': 'token' or 'patch'
            - 'metric_type': 'Cosine Similarity' or 'Distance to Boundary'
            - 'percentthrumodels': List of percentthrumodel values used
            - 'concepts': List of concepts processed
            - 'global_x_min', 'global_x_max': Global x-axis limits
    """
    print(f"\n=== Computing concept activation distributions ===")
    print(f"Dataset: {dataset_name}, Model: {model_name}")
    print(f"Concept type: {concept_type}")
    
    # Determine if text dataset
    is_text_dataset = dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank', 'IMDB', 'Jailbreak']
    unit_type = 'token' if is_text_dataset else 'patch'
    
    # Get default percentthrumodels if not provided
    if percentthrumodels is None:
        percentthrumodels = get_model_default_percentthrumodels(model_name, model_input_size)
    
    print(f"Using percentthrumodel values: {percentthrumodels}")
    
    # Don't limit concepts - process all of them
    print(f"Processing all {len(concepts)} concepts")
    
    # Determine metric type based on concept type
    sample_type = 'patch'
    if concept_type in ['avg_patch_embeddings', 'kmeans_1000_patch_embeddings_kmeans']:
        metric_type = 'Cosine Similarity'
        acts_prefix = 'cosine_similarities'
    else:
        metric_type = 'Distance to Boundary'
        acts_prefix = 'dists'
    
    # Load ground truth data
    print(f"\nLoading ground truth data...")
    gt_patches_file = f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
    if not os.path.exists(gt_patches_file):
        gt_patches_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
    
    if not os.path.exists(gt_patches_file):
        raise FileNotFoundError(f"Patch-level ground truth not found at {gt_patches_file}")
    
    gt_patches_per_concept = torch.load(gt_patches_file, weights_only=False)
    gt_samples_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
    gt_samples_per_concept = torch.load(gt_samples_file, weights_only=False)
    
    # Filter to valid concepts
    gt_patches_per_concept = filter_concept_dict(gt_patches_per_concept, dataset_name)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    # Load metadata
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    test_global_indices = metadata[metadata['split'] == 'test'].index.tolist()
    
    # Calculate patches per sample
    if not is_text_dataset:
        if model_input_size == (224, 224):
            patches_per_image = 256  # 16x16
        elif model_input_size == (560, 560):
            patches_per_image = 1600  # 40x40
        else:
            patches_per_image = 256
    
    has_padding = not is_text_dataset and model_input_size == (560, 560)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Results storage
    distributions = {}
    global_x_min = float('inf')
    global_x_max = float('-inf')
    
    # Process each percentthrumodel
    for percent_thru_model in percentthrumodels:
        print(f"\n--- Processing percentthrumodel {percent_thru_model} ---")
        
        # Get activation file name
        if concept_type == 'avg_patch_embeddings':
            acts_file = f"{acts_prefix}_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
            acts_file = f"{acts_prefix}_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
            n_clusters = 1000
            acts_file = f"{acts_prefix}_kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
            n_clusters = 1000
            acts_file = f"{acts_prefix}_kmeans_{n_clusters}_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        
        # Load activations
        try:
            act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
        except FileNotFoundError:
            print(f"Error: Could not find activation file: {acts_file}")
            # Store empty data for this percentthrumodel
            for concept_name in concepts:
                key = f"{concept_name}_{percent_thru_model}"
                distributions[key] = {
                    'gt_positive_acts': np.array([]),
                    'background_acts': np.array([]),
                    'n_gt_positive': 0,
                    'n_background': 0,
                    'gt_mean': None,
                    'gt_std': None,
                    'bg_mean': None,
                    'bg_std': None,
                    'overlap_area': None,
                    'separability': None,
                    'cohens_d': None,
                    'threshold': None,
                    'error': 'Activation file not found'
                }
            continue
        
        # Load test activations
        test_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
        if test_acts is None:
            print(f"Could not load test activations for percentthrumodel {percent_thru_model}")
            continue
        
        print(f"Test activations shape: {test_acts.shape}")
        
        # Load concepts to get indices
        if concept_type == 'avg_patch_embeddings':
            concepts_filename = f"avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
            concepts_filename = f"linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
            concepts_filename = f"kmeans_1000_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
            concepts_filename = f"kmeans_1000_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        
        concepts_file = f"Concepts/{dataset_name}/{concepts_filename}"
        
        if not os.path.exists(concepts_file):
            print(f"Concepts file not found: {concepts_file}")
            continue
            
        concepts_data = torch.load(concepts_file, weights_only=False)
        if isinstance(concepts_data, dict):
            all_concept_names = list(concepts_data.keys())
        else:
            print(f"Unexpected format in concepts file: {concepts_file}")
            continue
        
        # Load thresholds if requested
        thresholds = {}
        if compute_thresholds:
            threshold_file = None
            if concept_type == 'avg_patch_embeddings':
                threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_avg_patch_embeddings_percentthrumodel_{percent_thru_model}.pt'
            elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
                threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}.pt'
            elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
                threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
            elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
                threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
            
            if threshold_file and os.path.exists(threshold_file):
                try:
                    thresholds = torch.load(threshold_file, weights_only=False)
                except Exception as e:
                    print(f"Warning: Could not load thresholds: {e}")
        
        # Process each concept
        for concept_name in concepts:
            key = f"{concept_name}_{percent_thru_model}"
            
            # Get concept index
            if concept_name not in all_concept_names:
                distributions[key] = {
                    'gt_positive_acts': np.array([]),
                    'background_acts': np.array([]),
                    'n_gt_positive': 0,
                    'n_background': 0,
                    'gt_mean': None,
                    'gt_std': None,
                    'bg_mean': None,
                    'bg_std': None,
                    'overlap_area': None,
                    'separability': None,
                    'cohens_d': None,
                    'threshold': None,
                    'error': 'Concept not found'
                }
                continue
            
            concept_idx = all_concept_names.index(concept_name)
            concept_acts = test_acts[:, concept_idx]
            
            # Keep on GPU if available
            if not concept_acts.is_cuda and device.type == 'cuda' and use_gpu:
                concept_acts = concept_acts.cuda()
            
            # Get GT positive patch indices
            positive_patch_indices = gt_patches_per_concept.get(concept_name, [])
            
            # Filter out padding patches if needed
            if has_padding and len(positive_patch_indices) > 0:
                positive_patch_indices = filter_patches_by_image_presence(
                    positive_patch_indices, dataset_name, model_input_size
                )
                positive_patch_indices = positive_patch_indices.tolist()
            
            if len(positive_patch_indices) == 0:
                distributions[key] = {
                    'gt_positive_acts': np.array([]),
                    'background_acts': np.array([]),
                    'n_gt_positive': 0,
                    'n_background': 0,
                    'gt_mean': None,
                    'gt_std': None,
                    'bg_mean': None,
                    'bg_std': None,
                    'overlap_area': None,
                    'separability': None,
                    'cohens_d': None,
                    'threshold': None,
                    'error': 'No positive patches'
                }
                continue
            
            # Collect GT positive activations
            if is_text_dataset:
                gt_positive_acts = []
                for patch_idx in positive_patch_indices:
                    if patch_idx < len(concept_acts):
                        act_value = concept_acts[patch_idx].item()
                        gt_positive_acts.append(act_value)
                gt_positive_acts = np.array(gt_positive_acts)
            else:
                # Vectorized approach for images
                positive_patch_indices = np.array(positive_patch_indices)
                global_img_indices = positive_patch_indices // patches_per_image
                patch_within_imgs = positive_patch_indices % patches_per_image
                
                test_global_set = set(test_global_indices)
                test_mask = np.array([img_idx in test_global_set for img_idx in global_img_indices])
                test_global_img_indices = global_img_indices[test_mask]
                test_patch_within_imgs = patch_within_imgs[test_mask]
                
                if len(test_global_img_indices) == 0:
                    gt_positive_acts = np.array([])
                else:
                    global_to_test_pos = {global_idx: test_pos for test_pos, global_idx in enumerate(test_global_indices)}
                    test_positions = np.array([global_to_test_pos[idx] for idx in test_global_img_indices])
                    test_patch_indices = test_positions * patches_per_image + test_patch_within_imgs
                    valid_indices = test_patch_indices[test_patch_indices < len(concept_acts)]
                    
                    if len(valid_indices) > 0:
                        if device.type == 'cuda' and concept_acts.is_cuda and use_gpu:
                            indices_tensor = torch.from_numpy(valid_indices).long().to(device)
                            gt_positive_acts_gpu = concept_acts[indices_tensor]
                            gt_positive_acts = gt_positive_acts_gpu.cpu().numpy()
                        else:
                            gt_positive_acts = concept_acts[valid_indices].cpu().numpy() if hasattr(concept_acts, 'cpu') else concept_acts[valid_indices].numpy()
                    else:
                        gt_positive_acts = np.array([])
            
            # Get background activations
            samples_with_concept = set(gt_samples_per_concept.get(concept_name, []))
            
            if is_text_dataset:
                background_acts = np.array([])  # Skip for text datasets in this simplified version
            else:
                # Vectorized background collection
                samples_with_concept_arr = np.array(list(samples_with_concept))
                all_test_positions = np.arange(len(test_global_indices))
                has_concept = np.zeros(len(all_test_positions), dtype=bool)
                if len(samples_with_concept_arr) > 0:
                    valid_concept_samples = samples_with_concept_arr[samples_with_concept_arr < len(all_test_positions)]
                    has_concept[valid_concept_samples] = True
                
                positions_without_concept = all_test_positions[~has_concept]
                
                if len(positions_without_concept) > 0:
                    n_patches = min(patches_per_image, (len(concept_acts) - 1) // len(test_global_indices) + 1)
                    patch_offsets = np.arange(n_patches)
                    base_indices = positions_without_concept * patches_per_image
                    all_bg_indices = base_indices[:, np.newaxis] + patch_offsets[np.newaxis, :]
                    background_indices = all_bg_indices.flatten()
                    background_indices = background_indices[background_indices < len(concept_acts)]
                    
                    if device.type == 'cuda' and concept_acts.is_cuda and use_gpu:
                        indices_tensor = torch.from_numpy(background_indices).long().to(device)
                        background_acts_gpu = concept_acts[indices_tensor]
                        if len(background_acts_gpu) > max_background_samples:
                            sample_indices = torch.randperm(len(background_acts_gpu), device=device)[:max_background_samples]
                            background_acts_gpu = background_acts_gpu[sample_indices]
                        background_acts = background_acts_gpu.cpu().numpy()
                    else:
                        background_acts = concept_acts[background_indices].cpu().numpy() if hasattr(concept_acts, 'cpu') else concept_acts[background_indices].numpy()
                        if len(background_acts) > max_background_samples:
                            background_acts = np.random.choice(background_acts, max_background_samples, replace=False)
                else:
                    background_acts = np.array([])
            
            # Compute statistics if we have data
            if len(gt_positive_acts) > 0 and len(background_acts) > 0:
                # Update global min/max
                all_acts = np.concatenate([background_acts, gt_positive_acts])
                global_x_min = min(global_x_min, np.min(all_acts))
                global_x_max = max(global_x_max, np.max(all_acts))
                
                # Compute histograms for overlap calculation
                hist_bg, bin_edges = np.histogram(background_acts, bins=n_bins, density=True)
                hist_gt, _ = np.histogram(gt_positive_acts, bins=bin_edges, density=True)
                bin_width = bin_edges[1] - bin_edges[0]
                overlap_area = np.sum(np.minimum(hist_bg, hist_gt)) * bin_width
                
                # Separability score
                separability = max(0, 1 - overlap_area)
                
                # Distribution statistics
                bg_mean, bg_std = np.mean(background_acts), np.std(background_acts)
                gt_mean, gt_std = np.mean(gt_positive_acts), np.std(gt_positive_acts)
                
                # Cohen's d
                pooled_std = np.sqrt((bg_std**2 + gt_std**2) / 2)
                cohens_d = abs(gt_mean - bg_mean) / pooled_std if pooled_std > 0 else 0
                
                # Compute AUC-ROC
                # Create labels: 1 for GT positive, 0 for background
                y_true = np.concatenate([np.ones(len(gt_positive_acts)), np.zeros(len(background_acts))])
                y_scores = np.concatenate([gt_positive_acts, background_acts])
                
                try:
                    auc_roc = roc_auc_score(y_true, y_scores)
                except Exception as e:
                    print(f"Warning: Could not compute AUC for {concept_name}: {e}")
                    auc_roc = None
                
                # Get threshold if available
                threshold = None
                if thresholds and concept_name in thresholds:
                    threshold = thresholds[concept_name]['best_threshold']
                
                distributions[key] = {
                    'gt_positive_acts': gt_positive_acts,
                    'background_acts': background_acts,
                    'n_gt_positive': len(gt_positive_acts),
                    'n_background': len(background_acts),
                    'gt_mean': gt_mean,
                    'gt_std': gt_std,
                    'bg_mean': bg_mean,
                    'bg_std': bg_std,
                    'overlap_area': overlap_area,
                    'separability': separability,
                    'cohens_d': cohens_d,
                    'auc_roc': auc_roc,
                    'threshold': threshold,
                    'bin_edges': bin_edges  # Store for consistent binning in plotting
                }
            else:
                distributions[key] = {
                    'gt_positive_acts': gt_positive_acts,
                    'background_acts': background_acts,
                    'n_gt_positive': len(gt_positive_acts),
                    'n_background': len(background_acts),
                    'gt_mean': np.mean(gt_positive_acts) if len(gt_positive_acts) > 0 else None,
                    'gt_std': np.std(gt_positive_acts) if len(gt_positive_acts) > 0 else None,
                    'bg_mean': np.mean(background_acts) if len(background_acts) > 0 else None,
                    'bg_std': np.std(background_acts) if len(background_acts) > 0 else None,
                    'overlap_area': None,
                    'separability': None,
                    'cohens_d': None,
                    'auc_roc': None,
                    'threshold': None,
                    'error': 'Insufficient data'
                }
    
    # Prepare metadata
    metadata_dict = {
        'dataset_name': dataset_name,
        'model_name': model_name,
        'concept_type': concept_type,
        'is_text_dataset': is_text_dataset,
        'unit_type': unit_type,
        'metric_type': metric_type,
        'percentthrumodels': percentthrumodels,
        'concepts': concepts,
        'global_x_min': global_x_min if global_x_min != float('inf') else None,
        'global_x_max': global_x_max if global_x_max != float('-inf') else None,
        'n_bins': n_bins
    }
    
    return {
        'distributions': distributions,
        'metadata': metadata_dict
    }


def plot_concept_activation_distributions(
    computation_results: Dict[str, Any],
    figsize_per_plot: Tuple[float, float] = (4, 3),
    show_thresholds: bool = True,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    use_consistent_xlim: bool = True,
    alpha_bg: float = 0.6,
    alpha_gt: float = 0.7,
    show_legend: bool = True,
    show_metrics: bool = True
) -> None:
    """
    Plot the computed concept activation distributions in a grid format.
    
    Args:
        computation_results: Results from compute_concept_activation_distributions
        figsize_per_plot: Size of each subplot
        show_thresholds: Whether to show detection thresholds
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        use_consistent_xlim: Whether to use the same x-axis limits for all plots
        alpha_bg: Alpha value for background histogram
        alpha_gt: Alpha value for GT positive histogram
        show_legend: Whether to show legend on first plot
        show_metrics: Whether to show separability and Cohen's d metrics
    """
    # Extract data
    distributions = computation_results['distributions']
    metadata = computation_results['metadata']
    
    dataset_name = metadata['dataset_name']
    model_name = metadata['model_name']
    is_text_dataset = metadata['is_text_dataset']
    metric_type = metadata['metric_type']
    percentthrumodels = metadata['percentthrumodels']
    concepts = metadata['concepts']
    global_x_min = metadata['global_x_min']
    global_x_max = metadata['global_x_max']
    n_bins = metadata['n_bins']
    
    # Create figure
    n_rows = len(concepts)
    n_cols = len(percentthrumodels)
    fig_width = n_cols * figsize_per_plot[0]
    fig_height = n_rows * figsize_per_plot[1]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Get total layers for title
    if model_name == 'CLIP':
        total_layers = 24
    elif model_name == 'Llama' and is_text_dataset:
        total_layers = 32
    elif model_name == 'Llama' and not is_text_dataset:
        total_layers = 40
    elif model_name == 'Gemma':
        total_layers = 28
    elif model_name == 'Qwen':
        total_layers = 32
    else:
        total_layers = 24  # Default
    
    # Plot each cell
    for row_idx, concept_name in enumerate(concepts):
        for col_idx, percent_thru_model in enumerate(percentthrumodels):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            
            key = f"{concept_name}_{percent_thru_model}"
            if key not in distributions:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            dist_data = distributions[key]
            
            # Check for errors
            if 'error' in dist_data:
                ax.text(0.5, 0.5, dist_data['error'], ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10, wrap=True)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            gt_positive_acts = dist_data['gt_positive_acts']
            background_acts = dist_data['background_acts']
            
            # Plot histograms if we have data
            if len(gt_positive_acts) > 0 and len(background_acts) > 0:
                # Use stored bin edges if available for consistency
                if 'bin_edges' in dist_data:
                    bin_edges = dist_data['bin_edges']
                    ax.hist(background_acts, bins=bin_edges, density=True, alpha=alpha_bg,
                           color='gray', edgecolor='black', label=f'Bg ({len(background_acts)})')
                    ax.hist(gt_positive_acts, bins=bin_edges, density=True, alpha=alpha_gt,
                           color='green', edgecolor='darkgreen', label=f'GT+ ({len(gt_positive_acts)})')
                else:
                    ax.hist(background_acts, bins=n_bins, density=True, alpha=alpha_bg,
                           color='gray', edgecolor='black', label=f'Bg ({len(background_acts)})')
                    ax.hist(gt_positive_acts, bins=n_bins, density=True, alpha=alpha_gt,
                           color='green', edgecolor='darkgreen', label=f'GT+ ({len(gt_positive_acts)})')
                
                # Add threshold if available and requested
                if show_thresholds and dist_data.get('threshold') is not None:
                    ax.axvline(dist_data['threshold'], color='purple', linestyle='-', 
                             linewidth=2, label='Thresh')
                
                # Add metrics text if requested
                if show_metrics and dist_data.get('separability') is not None:
                    metrics_text = f'Sep: {dist_data["separability"]:.2f}\nd: {dist_data["cohens_d"]:.2f}'
                    ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='right', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # Formatting
                if row_idx == 0:
                    # Add title with layer number
                    layer_num = percent_to_layer(percent_thru_model, total_layers, model_name)
                    ax.set_title(f'{percent_thru_model}% (L{layer_num})', fontsize=12)
                
                if col_idx == 0:
                    # Truncate long concept names
                    ylabel = concept_name if len(concept_name) <= 15 else f'{concept_name[:15]}...'
                    ax.set_ylabel(ylabel, fontsize=12)
                
                # Only show x-label on bottom row
                if row_idx == n_rows - 1:
                    ax.set_xlabel(metric_type, fontsize=10)
                
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.3)
                
                # Legend only on first plot
                if show_legend and row_idx == 0 and col_idx == 0:
                    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            else:
                # No valid data
                error_msg = 'No GT+' if len(gt_positive_acts) == 0 else 'No Bg'
                ax.text(0.5, 0.5, error_msg, ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Apply consistent x-axis limits if requested
    if use_consistent_xlim and global_x_min is not None and global_x_max is not None:
        # Add some padding (5% of range on each side)
        x_range = global_x_max - global_x_min
        x_padding = 0.05 * x_range
        x_min_padded = global_x_min - x_padding
        x_max_padded = global_x_max + x_padding
        
        # Apply to all subplots
        for row in axes:
            for ax in row:
                if ax.has_data():
                    ax.set_xlim(x_min_padded, x_max_padded)
    
    # Overall title
    fig.suptitle(f'{dataset_name} - {model_name}: Non-Concept vs GT Positive Activations', fontsize=16)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    # Show if requested  
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def compute_separation_over_percentthru(
    dataset_name: str,
    model_name: str,
    concept_type: str,
    model_input_size: Tuple[int, int],
    metrics_to_compute: List[str] = ['separability', 'auc_roc'],
    scratch_dir: str = '',
    percentthrumodels: Optional[List[int]] = None,
    concepts: Optional[List[str]] = None,
    n_bins: int = 50,
    max_concepts: int = 10,
    compute_thresholds: bool = True,
    use_gpu: bool = True,
    batch_process: bool = True,
    max_background_samples: int = 100000,
    compute_superdetector_separation: bool = False,
    background_percentile: float = 0.995,
    compute_background_detection: bool = False,
    validation_split: str = 'cal'
) -> Dict[str, Any]:
    """
    Compute concept activation distributions and separation metrics across layers.
    This is based on plot_concept_activation_grid but returns computed results instead of plotting.
    
    Args:
        dataset_name: Name of dataset (e.g. 'CLEVR', 'COCO', 'Broden')
        model_name: Name of model (e.g. 'CLIP', 'Llama')
        concept_type: Type of concept - must be one of:
            'avg_patch_embeddings'
            'linsep_patch_embeddings_BD_True_BN_False'
            'kmeans_1000_patch_embeddings_kmeans'
            'kmeans_1000_linsep_patch_embeddings_kmeans'
        model_input_size: Model input size (e.g. (224, 224) for CLIP, (560, 560) for Llama)
        metrics_to_compute: List of metrics to compute ('separability', 'auc_roc', 'cohens_d', 'overlap_area')
        scratch_dir: Directory where activation files are stored
        percentthrumodels: List of percentthrumodel values (defaults to model-specific values)
        concepts: List of concept names to analyze (if None, uses all available)
        n_bins: Number of bins for histograms
        max_concepts: Maximum number of concepts to process
        compute_thresholds: Whether to load detection thresholds
        use_gpu: Whether to use GPU for computation
        batch_process: Whether to batch process concepts on GPU
        max_background_samples: Maximum number of background samples to use
        compute_superdetector_separation: Whether to compute superdetector separation (not implemented yet)
        background_percentile: Non-Concept percentile threshold for detection (default: 0.995 = 99.5th percentile)
        compute_background_detection: Whether to compute detection rates using background percentile threshold
        validation_split: Split to use for computing background thresholds (default: 'cal')
        
    Returns:
        Dictionary containing:
        - 'results': Dict with keys "{concept}_{percentthru}" containing all metrics
        - 'metadata': Dict with dataset info, concepts, percentthrumodels, etc.
        - 'averaged_results': Dict with metrics averaged across concepts for each percentthru
        - 'background_detection_results': (if compute_background_detection=True) Dict mapping concept names to lists of detection rates
        - 'averaged_background_detection': (if compute_background_detection=True) List of averaged detection rates per layer
        - 'gt_mass_above_threshold_results': (if compute_background_detection=True) Dict mapping concept names to lists of GT mass above threshold
        - 'averaged_gt_mass_above_threshold': (if compute_background_detection=True) List of averaged GT mass above threshold per layer
    """
    print(f"\n=== Computing concept separation metrics ===")
    print(f"Dataset: {dataset_name}, Model: {model_name}")
    print(f"Concept type: {concept_type}")
    print(f"Metrics to compute: {metrics_to_compute}")
    
    # Determine if text dataset
    is_text_dataset = dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank', 'IMDB', 'Jailbreak']
    unit_type = 'token' if is_text_dataset else 'patch'
    
    # Get default percentthrumodels if not provided
    if percentthrumodels is None:
        percentthrumodels = get_model_default_percentthrumodels(model_name, model_input_size)
    
    print(f"Using percentthrumodel values: {percentthrumodels}")
    
    # Determine metric type based on concept type
    sample_type = 'patch'
    if concept_type in ['avg_patch_embeddings', 'kmeans_1000_patch_embeddings_kmeans']:
        metric_type = 'Cosine Similarity'
        acts_prefix = 'cosine_similarities'
    else:
        metric_type = 'Distance to Boundary'
        acts_prefix = 'dists'
    
    # Load ground truth data (same for all percentthrumodels)
    print(f"\nLoading ground truth data...")
    gt_patches_file = f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
    if not os.path.exists(gt_patches_file):
        gt_patches_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
    
    if not os.path.exists(gt_patches_file):
        raise FileNotFoundError(f"Patch-level ground truth not found at {gt_patches_file}")
    
    gt_patches_per_concept = torch.load(gt_patches_file, weights_only=False)
    gt_samples_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
    gt_samples_per_concept = torch.load(gt_samples_file, weights_only=False)
    
    # Filter to valid concepts
    gt_patches_per_concept = filter_concept_dict(gt_patches_per_concept, dataset_name)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    # Get concepts if not provided
    if concepts is None:
        concepts = list(gt_patches_per_concept.keys())
        print(f"Found {len(concepts)} concepts")
    
    # Don't limit concepts - process all of them
    print(f"Processing all {len(concepts)} concepts")
    
    # Load metadata
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    test_global_indices = metadata[metadata['split'] == 'test'].index.tolist()
    
    # Calculate patches per sample
    if not is_text_dataset:
        if model_input_size == (224, 224):
            patches_per_image = 256  # 16x16
        elif model_input_size == (560, 560):
            patches_per_image = 1600  # 40x40
        else:
            patches_per_image = 256
    
    has_padding = not is_text_dataset and model_input_size == (560, 560)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Results storage
    results = {}
    averaged_results = {metric: [] for metric in metrics_to_compute}
    if compute_background_detection:
        background_detection_results = {}
        averaged_background_detection = []
        gt_mass_above_threshold_results = {}
        averaged_gt_mass_above_threshold = []
    
    # Track global x-axis limits for consistent scaling
    global_x_min = float('inf')
    global_x_max = float('-inf')
    
    # Process each percentthrumodel
    for percent_thru_model in percentthrumodels:
        print(f"\n--- Processing percentthrumodel {percent_thru_model} ---")
        
        # Get activation file name
        if concept_type == 'avg_patch_embeddings':
            acts_file = f"{acts_prefix}_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
            acts_file = f"{acts_prefix}_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
            n_clusters = 1000
            acts_file = f"{acts_prefix}_kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
            n_clusters = 1000
            acts_file = f"{acts_prefix}_kmeans_{n_clusters}_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        
        # Load activations
        try:
            act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
        except FileNotFoundError as e:
            print(f"Error: Could not find activation file: {acts_file}")
            continue
        
        # Load test activations
        test_acts = act_loader.load_split_tensor('test', dataset_name, model_input_size, patch_size=14)
        if test_acts is None:
            print(f"Could not load test activations for percentthrumodel {percent_thru_model}")
            continue
        
        # Load validation activations if computing background detection
        val_acts = None
        if compute_background_detection:
            val_acts = act_loader.load_split_tensor(validation_split, dataset_name, model_input_size, patch_size=14)
            if val_acts is None:
                print(f"Could not load validation activations for percentthrumodel {percent_thru_model}")
                compute_background_detection_for_layer = False
            else:
                compute_background_detection_for_layer = True
        else:
            compute_background_detection_for_layer = False
        
        # Load concepts to get indices
        if concept_type == 'avg_patch_embeddings':
            concepts_filename = f"avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
            concepts_filename = f"linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
            concepts_filename = f"kmeans_1000_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
            concepts_filename = f"kmeans_1000_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
        
        concepts_file = f"Concepts/{dataset_name}/{concepts_filename}"
        
        if not os.path.exists(concepts_file):
            print(f"Concepts file not found: {concepts_file}")
            continue
            
        concepts_data = torch.load(concepts_file, weights_only=False)
        if isinstance(concepts_data, dict):
            all_concept_names = list(concepts_data.keys())
        else:
            print(f"Unexpected format in concepts file: {concepts_file}")
            continue
        
        # Load thresholds if requested
        thresholds = {}
        if compute_thresholds:
            threshold_file = None
            if concept_type == 'avg_patch_embeddings':
                threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_avg_patch_embeddings_percentthrumodel_{percent_thru_model}.pt'
            elif concept_type == 'linsep_patch_embeddings_BD_True_BN_False':
                threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}.pt'
            elif concept_type == 'kmeans_1000_patch_embeddings_kmeans':
                threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
            elif concept_type == 'kmeans_1000_linsep_patch_embeddings_kmeans':
                threshold_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_{percent_thru_model}.pt'
            
            if threshold_file and os.path.exists(threshold_file):
                try:
                    thresholds = torch.load(threshold_file, weights_only=False)
                except Exception as e:
                    print(f"Warning: Could not load thresholds: {e}")
        
        # Initialize metrics for this layer
        layer_metrics = {metric: [] for metric in metrics_to_compute}
        if compute_background_detection_for_layer:
            layer_detection_rates = []
            layer_gt_mass_above_threshold = []
            
            # Load validation metadata for background threshold computation
            val_metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
            val_global_indices = val_metadata[val_metadata['split'] == validation_split].index.tolist()
        
        # Process each concept
        for concept_name in concepts:
            key = f"{concept_name}_{percent_thru_model}"
            
            # Get concept index
            if concept_name not in all_concept_names:
                results[key] = {
                    'gt_positive_acts': np.array([]),
                    'background_acts': np.array([]),
                    'n_gt_positive': 0,
                    'n_background': 0,
                    'error': 'Concept not found'
                }
                continue
            
            concept_idx = all_concept_names.index(concept_name)
            concept_acts = test_acts[:, concept_idx]
            
            # Keep on GPU if available
            if not concept_acts.is_cuda and device.type == 'cuda' and use_gpu:
                concept_acts = concept_acts.cuda()
            
            # Get GT positive patch indices
            positive_patch_indices = gt_patches_per_concept.get(concept_name, [])
            
            # Filter out padding patches if needed
            if has_padding and len(positive_patch_indices) > 0:
                positive_patch_indices = filter_patches_by_image_presence(
                    positive_patch_indices, dataset_name, model_input_size
                )
                positive_patch_indices = positive_patch_indices.tolist()
            
            if len(positive_patch_indices) == 0:
                results[key] = {
                    'gt_positive_acts': np.array([]),
                    'background_acts': np.array([]),
                    'n_gt_positive': 0,
                    'n_background': 0,
                    'error': 'No positive patches'
                }
                continue
            
            # Collect GT positive activations
            if is_text_dataset:
                gt_positive_acts = []
                for patch_idx in positive_patch_indices:
                    if patch_idx < len(concept_acts):
                        act_value = concept_acts[patch_idx].item()
                        gt_positive_acts.append(act_value)
                gt_positive_acts = np.array(gt_positive_acts)
            else:
                # Vectorized approach for images
                positive_patch_indices = np.array(positive_patch_indices)
                global_img_indices = positive_patch_indices // patches_per_image
                patch_within_imgs = positive_patch_indices % patches_per_image
                
                test_global_set = set(test_global_indices)
                test_mask = np.array([img_idx in test_global_set for img_idx in global_img_indices])
                test_global_img_indices = global_img_indices[test_mask]
                test_patch_within_imgs = patch_within_imgs[test_mask]
                
                if len(test_global_img_indices) == 0:
                    gt_positive_acts = np.array([])
                else:
                    global_to_test_pos = {global_idx: test_pos for test_pos, global_idx in enumerate(test_global_indices)}
                    test_positions = np.array([global_to_test_pos[idx] for idx in test_global_img_indices])
                    test_patch_indices = test_positions * patches_per_image + test_patch_within_imgs
                    valid_indices = test_patch_indices[test_patch_indices < len(concept_acts)]
                    
                    if len(valid_indices) > 0:
                        if device.type == 'cuda' and concept_acts.is_cuda and use_gpu:
                            indices_tensor = torch.from_numpy(valid_indices).long().to(device)
                            gt_positive_acts_gpu = concept_acts[indices_tensor]
                            gt_positive_acts = gt_positive_acts_gpu.cpu().numpy()
                        else:
                            gt_positive_acts = concept_acts[valid_indices].cpu().numpy() if hasattr(concept_acts, 'cpu') else concept_acts[valid_indices].numpy()
                    else:
                        gt_positive_acts = np.array([])
            
            # Get background activations
            samples_with_concept = set(gt_samples_per_concept.get(concept_name, []))
            
            if is_text_dataset:
                background_acts = np.array([])  # Skip for text datasets in this simplified version
            else:
                # Vectorized background collection
                samples_with_concept_arr = np.array(list(samples_with_concept))
                all_test_positions = np.arange(len(test_global_indices))
                has_concept = np.zeros(len(all_test_positions), dtype=bool)
                if len(samples_with_concept_arr) > 0:
                    valid_concept_samples = samples_with_concept_arr[samples_with_concept_arr < len(all_test_positions)]
                    has_concept[valid_concept_samples] = True
                
                positions_without_concept = all_test_positions[~has_concept]
                
                if len(positions_without_concept) > 0:
                    n_patches = min(patches_per_image, (len(concept_acts) - 1) // len(test_global_indices) + 1)
                    patch_offsets = np.arange(n_patches)
                    base_indices = positions_without_concept * patches_per_image
                    all_bg_indices = base_indices[:, np.newaxis] + patch_offsets[np.newaxis, :]
                    background_indices = all_bg_indices.flatten()
                    background_indices = background_indices[background_indices < len(concept_acts)]
                    
                    if device.type == 'cuda' and concept_acts.is_cuda and use_gpu:
                        indices_tensor = torch.from_numpy(background_indices).long().to(device)
                        background_acts_gpu = concept_acts[indices_tensor]
                        if len(background_acts_gpu) > max_background_samples:
                            sample_indices = torch.randperm(len(background_acts_gpu), device=device)[:max_background_samples]
                            background_acts_gpu = background_acts_gpu[sample_indices]
                        background_acts = background_acts_gpu.cpu().numpy()
                    else:
                        background_acts = concept_acts[background_indices].cpu().numpy() if hasattr(concept_acts, 'cpu') else concept_acts[background_indices].numpy()
                        if len(background_acts) > max_background_samples:
                            background_acts = np.random.choice(background_acts, max_background_samples, replace=False)
                else:
                    background_acts = np.array([])
            
            # Compute metrics if we have data
            if len(gt_positive_acts) > 0 and len(background_acts) > 0:
                # Update global min/max
                all_acts = np.concatenate([background_acts, gt_positive_acts])
                global_x_min = min(global_x_min, np.min(all_acts))
                global_x_max = max(global_x_max, np.max(all_acts))
                
                result_dict = {
                    'gt_positive_acts': gt_positive_acts,
                    'background_acts': background_acts,
                    'n_gt_positive': len(gt_positive_acts),
                    'n_background': len(background_acts),
                    'gt_mean': np.mean(gt_positive_acts),
                    'gt_std': np.std(gt_positive_acts),
                    'bg_mean': np.mean(background_acts),
                    'bg_std': np.std(background_acts)
                }
                
                # Compute requested metrics
                if 'overlap_area' in metrics_to_compute or 'separability' in metrics_to_compute:
                    # Compute histograms for overlap calculation
                    hist_bg, bin_edges = np.histogram(background_acts, bins=n_bins, density=True)
                    hist_gt, _ = np.histogram(gt_positive_acts, bins=bin_edges, density=True)
                    bin_width = bin_edges[1] - bin_edges[0]
                    overlap_area = np.sum(np.minimum(hist_bg, hist_gt)) * bin_width
                    result_dict['overlap_area'] = overlap_area
                    result_dict['bin_edges'] = bin_edges
                    
                    if 'separability' in metrics_to_compute:
                        separability = max(0, 1 - overlap_area)
                        result_dict['separability'] = separability
                        layer_metrics['separability'].append(separability)
                
                if 'cohens_d' in metrics_to_compute:
                    # Cohen's d
                    pooled_std = np.sqrt((result_dict['bg_std']**2 + result_dict['gt_std']**2) / 2)
                    cohens_d = abs(result_dict['gt_mean'] - result_dict['bg_mean']) / pooled_std if pooled_std > 0 else 0
                    result_dict['cohens_d'] = cohens_d
                    layer_metrics['cohens_d'].append(cohens_d)
                
                if 'auc_roc' in metrics_to_compute or 'auc' in metrics_to_compute:
                    # Compute AUC-ROC
                    y_true = np.concatenate([np.ones(len(gt_positive_acts)), np.zeros(len(background_acts))])
                    y_scores = np.concatenate([gt_positive_acts, background_acts])
                    
                    try:
                        auc_roc = roc_auc_score(y_true, y_scores)
                        result_dict['auc_roc'] = auc_roc
                        if 'auc_roc' in metrics_to_compute:
                            layer_metrics['auc_roc'].append(auc_roc)
                        if 'auc' in metrics_to_compute:
                            layer_metrics['auc'].append(auc_roc)
                    except Exception as e:
                        print(f"Warning: Could not compute AUC for {concept_name}: {e}")
                        result_dict['auc_roc'] = None
                
                # Get threshold if available
                if thresholds and concept_name in thresholds:
                    result_dict['threshold'] = thresholds[concept_name]['best_threshold']
                else:
                    result_dict['threshold'] = None
                
                # Compute background detection if requested
                if compute_background_detection_for_layer:
                    # Get validation background activations for this concept
                    val_concept_acts = val_acts[:, concept_idx]
                    
                    # Get validation samples without this concept
                    val_samples_with_concept = set(gt_samples_per_concept.get(concept_name, []))
                    val_positions_without_concept = [i for i, idx in enumerate(val_global_indices) 
                                                   if idx not in val_samples_with_concept]
                    
                    if len(val_positions_without_concept) > 0 and not is_text_dataset:
                        # Collect validation background activations
                        val_bg_indices = []
                        for pos in val_positions_without_concept:
                            start_idx = pos * patches_per_image
                            end_idx = min((pos + 1) * patches_per_image, len(val_concept_acts))
                            val_bg_indices.extend(range(start_idx, end_idx))
                        
                        val_bg_acts = val_concept_acts[val_bg_indices].cpu().numpy() if hasattr(val_concept_acts, 'cpu') else val_concept_acts[val_bg_indices].numpy()
                        
                        # Compute background threshold (background_percentile is already 0-1, so multiply by 100)
                        bg_threshold = np.percentile(val_bg_acts, background_percentile * 100)
                        result_dict['background_threshold'] = bg_threshold
                        
                        # Compute GT mass above background threshold
                        gt_mass_above_threshold = np.mean(gt_positive_acts > bg_threshold)
                        result_dict['gt_mass_above_threshold'] = gt_mass_above_threshold
                        
                        # Now compute detection rate on test set
                        # Get test samples with this concept
                        test_samples_with_concept = [i for i, idx in enumerate(test_global_indices) 
                                                   if idx in gt_samples_per_concept.get(concept_name, [])]
                        
                        if len(test_samples_with_concept) > 0:
                            detected_count = 0
                            for test_pos in test_samples_with_concept:
                                # Check if any patch in this image exceeds threshold
                                start_idx = test_pos * patches_per_image
                                end_idx = min((test_pos + 1) * patches_per_image, len(concept_acts))
                                image_patches = concept_acts[start_idx:end_idx]
                                
                                if torch.any(image_patches > bg_threshold):
                                    detected_count += 1
                            
                            detection_rate = detected_count / len(test_samples_with_concept)
                            result_dict['background_detection_rate'] = detection_rate
                            layer_detection_rates.append(detection_rate)
                            layer_gt_mass_above_threshold.append(gt_mass_above_threshold)
                
                results[key] = result_dict
            else:
                results[key] = {
                    'gt_positive_acts': gt_positive_acts,
                    'background_acts': background_acts,
                    'n_gt_positive': len(gt_positive_acts),
                    'n_background': len(background_acts),
                    'error': 'Insufficient data'
                }
        
        # Average metrics across concepts for this layer
        for metric in metrics_to_compute:
            if metric in layer_metrics and layer_metrics[metric]:
                averaged_results[metric].append(np.mean(layer_metrics[metric]))
            else:
                averaged_results[metric].append(np.nan)
        
        # Average background detection rates if computed
        if compute_background_detection_for_layer and layer_detection_rates:
            averaged_background_detection.append(np.mean(layer_detection_rates))
            averaged_gt_mass_above_threshold.append(np.mean(layer_gt_mass_above_threshold))
            # Store per-concept detection rates and GT mass for this layer
            for concept_name in concepts:
                key = f"{concept_name}_{percent_thru_model}"
                if key in results and 'background_detection_rate' in results[key]:
                    if concept_name not in background_detection_results:
                        background_detection_results[concept_name] = []
                    background_detection_results[concept_name].append(results[key]['background_detection_rate'])
                if key in results and 'gt_mass_above_threshold' in results[key]:
                    if concept_name not in gt_mass_above_threshold_results:
                        gt_mass_above_threshold_results[concept_name] = []
                    gt_mass_above_threshold_results[concept_name].append(results[key]['gt_mass_above_threshold'])
    
    # Prepare metadata
    metadata_dict = {
        'dataset_name': dataset_name,
        'model_name': model_name,
        'concept_type': concept_type,
        'is_text_dataset': is_text_dataset,
        'unit_type': unit_type,
        'metric_type': metric_type,
        'percentthrumodels': percentthrumodels,
        'concepts': concepts,
        'global_x_min': global_x_min if global_x_min != float('inf') else None,
        'global_x_max': global_x_max if global_x_max != float('-inf') else None,
        'n_bins': n_bins,
        'background_percentile': background_percentile if compute_background_detection else None
    }
    
    result_dict = {
        'results': results,
        'metadata': metadata_dict,
        'averaged_results': averaged_results
    }
    
    if compute_background_detection:
        result_dict['background_detection_results'] = background_detection_results
        result_dict['averaged_background_detection'] = averaged_background_detection
        result_dict['gt_mass_above_threshold_results'] = gt_mass_above_threshold_results
        result_dict['averaged_gt_mass_above_threshold'] = averaged_gt_mass_above_threshold
        if 'background_detection_rate' not in averaged_results:
            averaged_results['background_detection_rate'] = averaged_background_detection
    
    return result_dict


def plot_gt_mass_above_threshold(
    computation_results: Dict[str, Any],
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True,
    use_layer_numbers: bool = False,
    label_size: int = 12,
    tick_label_size: Optional[int] = None,
    show_title: bool = True,
    custom_ylabel: Optional[str] = None,
    custom_title: Optional[str] = None,
    line_styles: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
    legend_loc: str = 'best',
    ylim: Optional[Tuple[float, float]] = None,
    show_error_bars: bool = True,
    plot_individual_concepts: bool = False,
    highlight_percentile: bool = True,
    overlap_framing: bool = False,
    legend_text: Optional[str] = None
) -> None:
    """
    Plot the average fraction of GT positive activations that exceed the background 
    percentile threshold across layers.
    
    Args:
        computation_results: Results from compute_separation_over_percentthru with compute_background_detection=True
        figsize: Figure size
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        use_layer_numbers: Whether to use layer numbers on x-axis (vs percentthrumodel)
        label_size: Font size for labels
        tick_label_size: Font size for tick labels (defaults to label_size-2 if not provided)
        show_title: Whether to show title
        custom_title: Custom title (overrides default)
        custom_ylabel: Custom y-axis label (overrides default)
        line_styles: Dict mapping concept names to line styles
        colors: Dict mapping concept names to colors
        show_legend: Not used - legends are not shown
        legend_loc: Not used - legends are not shown
        ylim: Y-axis limits (default: (0, 1) for fractions)
        show_error_bars: Whether to show error bars for averaged GT mass
        plot_individual_concepts: Whether to plot individual concept lines
        highlight_percentile: Whether to show the background percentile used in title
        overlap_framing: If True, plots (1 - fraction) to show overlap instead of separation
        legend_text: Optional text to display in bottom right corner (if None or empty, no text is shown)
    """
    # Apply paper plotting style
    apply_paper_plotting_style()
    
    # Extract data
    if 'gt_mass_above_threshold_results' not in computation_results:
        raise ValueError("No GT mass above threshold results found. Make sure compute_background_detection=True was used.")
    
    results = computation_results['results']
    metadata = computation_results['metadata']
    gt_mass_results = computation_results['gt_mass_above_threshold_results']
    averaged_gt_mass = computation_results.get('averaged_gt_mass_above_threshold', [])
    
    dataset_name = metadata['dataset_name']
    model_name = metadata['model_name']
    percentthrumodels = metadata['percentthrumodels']
    concepts = metadata['concepts']
    is_text_dataset = metadata['is_text_dataset']
    
    # Get background percentile from metadata or default
    background_percentile = metadata.get('background_percentile', 0.995) * 100  # Convert to percentage
    
    # Get total layers for conversion
    if model_name == 'CLIP':
        total_layers = 24
    elif model_name == 'Llama' and is_text_dataset:
        total_layers = 32
    elif model_name == 'Llama' and not is_text_dataset:
        total_layers = 40
    elif model_name == 'Gemma':
        total_layers = 28
    elif model_name == 'Qwen':
        total_layers = 32
    else:
        total_layers = 24
    
    # Default colors for concepts
    default_colors = plt.cm.tab10(np.linspace(0, 1, len(concepts)))
    if colors is None:
        colors = {concept: default_colors[i] for i, concept in enumerate(concepts)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # X-axis values
    if use_layer_numbers:
        x_values = [percent_to_layer(p, total_layers, model_name) for p in percentthrumodels]
        x_label = 'Layer'
    else:
        x_values = percentthrumodels
        x_label = '% Through Model'
    
    # Plot individual concepts if requested
    if plot_individual_concepts:
        for concept in concepts:
            if concept in gt_mass_results:
                gt_mass_values = gt_mass_results[concept]
                if len(gt_mass_values) == len(x_values):
                    # Apply overlap framing if requested
                    plot_values = [1 - v for v in gt_mass_values] if overlap_framing else gt_mass_values
                    ax.plot(x_values, plot_values,
                           label=concept,
                           linestyle=line_styles.get(concept, '-') if line_styles else '-',
                           color=colors.get(concept, None),
                           alpha=0.7, linewidth=1.5, 
                           marker='o', markersize=3)
    
    # Plot average GT mass
    if averaged_gt_mass and len(averaged_gt_mass) == len(x_values):
        # Apply overlap framing if requested
        plot_avg_values = [1 - v for v in averaged_gt_mass] if overlap_framing else averaged_gt_mass
        
        if show_error_bars and len(concepts) > 1 and plot_individual_concepts:
            # Calculate standard deviation across concepts for error bars
            stds = []
            for i in range(len(percentthrumodels)):
                mass_at_layer = []
                for concept in concepts:
                    if concept in gt_mass_results and i < len(gt_mass_results[concept]):
                        mass_at_layer.append(gt_mass_results[concept][i])
                
                if mass_at_layer:
                    stds.append(np.std(mass_at_layer))
                else:
                    stds.append(0)
            
            errors = [std / np.sqrt(len(concepts)) for std in stds]
            label_text = 'Average' if plot_individual_concepts else ('GT Mass Below Threshold' if overlap_framing else 'GT Mass Above Threshold')
            ax.errorbar(x_values, plot_avg_values, yerr=errors,
                       label=label_text,
                       linestyle='-',
                       color='black' if plot_individual_concepts else 'purple',
                       capsize=3, alpha=0.8, linewidth=2.5,
                       marker='o', markersize=3)
        else:
            label_text = 'Average' if plot_individual_concepts else ('GT Mass Below Threshold' if overlap_framing else 'GT Mass Above Threshold')
            ax.plot(x_values, plot_avg_values,
                   label=label_text,
                   linestyle='-',
                   color='black' if plot_individual_concepts else 'purple',
                   linewidth=2.5,
                   marker='o', markersize=3)
    
    # Formatting - Move x-axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(x_label, fontsize=label_size, labelpad=10)  # Add padding to raise label
    
    if custom_ylabel == "":
        # Don't set any y-label
        pass
    elif custom_ylabel:
        ax.set_ylabel(custom_ylabel, fontsize=label_size)
    else:
        if overlap_framing:
            ax.set_ylabel('Fraction of GT Distribution Below Threshold', fontsize=label_size)
        else:
            ax.set_ylabel('Fraction of GT Distribution Above Threshold', fontsize=label_size)
    
    if custom_title == "":
        # Don't set any title
        pass
    elif show_title:
        if custom_title:
            ax.set_title(custom_title, fontsize=label_size + 2, pad=20)  # Add padding to avoid overlap with x-axis
        else:
            title = f'{dataset_name} - {model_name}'
            if highlight_percentile:
                title += f'\n(GT Mass Above {background_percentile:.1f}th Percentile Non-Concept Threshold)'
            ax.set_title(title, fontsize=label_size + 2, pad=20)  # Add padding to avoid overlap with x-axis
    
    # Set y-axis limits
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, 1.05)  # Default for fractions
    
    # Use tick_label_size if provided, otherwise default to label_size-2
    tick_size = tick_label_size if tick_label_size is not None else label_size-2
    ax.tick_params(axis='both', labelsize=tick_size)
    
    # Set y-axis ticks at 0.25, 0.5, 0.75, 1.0
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    
    # Set x-axis ticks at every 25% when using percentthrumodel
    if not use_layer_numbers:
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    ax.grid(True, alpha=0.3)
    
    # Add legend if legend_text is provided
    if legend_text:
        # Create a custom legend entry with the same style as the plotted line
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='purple', linewidth=2.5, 
                                 marker='o', markersize=3, label=legend_text)]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=label_size-2,
                 frameon=True, fancybox=True, framealpha=0.8)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    # Show if requested  
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_true_concept_tokens_above_threshold(
    computation_results: Dict[str, Any],
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True,
    use_layer_numbers: bool = False,
    label_size: int = 12,
    custom_ylabel: Optional[str] = None,
    custom_title: Optional[str] = None,
    line_styles: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
    legend_loc: str = 'best',
    ylim: Optional[Tuple[float, float]] = None,
    show_error_bars: bool = True,
    plot_individual_concepts: bool = False,
    highlight_percentile: bool = True,
    overlap_framing: bool = False,
    legend_text: Optional[str] = None,
    background_percentile: float = 0.99
) -> None:
    """
    Plot the percentage of true concept TOKENS that exceed the background 
    percentile threshold across layers.
    
    Args:
        computation_results: Results from compute_separation_over_percentthru
        figsize: Figure size
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        use_layer_numbers: Whether to use layer numbers on x-axis (vs percentthrumodel)
        label_size: Font size for labels
        custom_title: Custom title (if empty string "", no title is shown)
        custom_ylabel: Custom y-axis label (overrides default)
        line_styles: Dict mapping concept names to line styles
        colors: Dict mapping concept names to colors
        show_legend: Not used - legends are not shown
        legend_loc: Not used - legends are not shown
        ylim: Y-axis limits (default: (0, 1) for fractions)
        show_error_bars: Whether to show error bars for averaged values
        plot_individual_concepts: Whether to plot individual concept lines
        highlight_percentile: Whether to show the background percentile used in title
        overlap_framing: If True, plots (1 - fraction) to show overlap instead of separation
        legend_text: Optional text to display in bottom right corner (if None or empty, no text is shown)
        background_percentile: Background percentile to use (default 0.99 for 99th percentile)
    """
    # Apply paper plotting style
    apply_paper_plotting_style()
    
    # Extract data
    results = computation_results['results']
    metadata = computation_results['metadata']
    
    dataset_name = metadata['dataset_name']
    model_name = metadata['model_name']
    percentthrumodels = metadata['percentthrumodels']
    concepts = metadata['concepts']
    is_text_dataset = metadata['is_text_dataset']
    
    # Get total layers for conversion
    if model_name == 'CLIP':
        total_layers = 24
    elif model_name == 'Llama' and is_text_dataset:
        total_layers = 32
    elif model_name == 'Llama' and not is_text_dataset:
        total_layers = 40
    elif model_name == 'Gemma':
        total_layers = 28
    elif model_name == 'Qwen':
        total_layers = 32
    else:
        total_layers = 24
    
    # Default colors for concepts
    default_colors = plt.cm.tab10(np.linspace(0, 1, len(concepts)))
    if colors is None:
        colors = {concept: default_colors[i] for i, concept in enumerate(concepts)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # X-axis values
    if use_layer_numbers:
        x_values = [percent_to_layer(p, total_layers, model_name) for p in percentthrumodels]
        x_label = 'Layer'
    else:
        x_values = percentthrumodels
        x_label = '% Through Model'
    
    # Calculate fraction of GT tokens above threshold for each concept
    concept_token_fractions = {}
    
    for concept in concepts:
        token_fractions = []
        
        for percent_thru_model in percentthrumodels:
            key = f"{concept}_{percent_thru_model}"
            if key in results and 'error' not in results[key]:
                result_data = results[key]
                gt_positive_acts = result_data.get('gt_positive_acts', np.array([]))
                background_acts = result_data.get('background_acts', np.array([]))
                
                if len(gt_positive_acts) > 0 and len(background_acts) > 0:
                    # Calculate threshold from background distribution
                    threshold = np.percentile(background_acts, background_percentile * 100)
                    
                    # Calculate fraction of GT tokens above threshold
                    fraction_above = np.mean(gt_positive_acts > threshold)
                    token_fractions.append(fraction_above)
                else:
                    token_fractions.append(0.0)  # No data
            else:
                token_fractions.append(0.0)  # Error or missing data
        
        if len(token_fractions) == len(percentthrumodels):
            concept_token_fractions[concept] = token_fractions
    
    # Calculate average across concepts
    if concept_token_fractions:
        avg_token_fractions = []
        for i in range(len(percentthrumodels)):
            values_at_layer = [concept_token_fractions[c][i] for c in concept_token_fractions]
            avg_token_fractions.append(np.mean(values_at_layer))
    else:
        avg_token_fractions = []
    
    # Plot individual concepts if requested
    if plot_individual_concepts and concept_token_fractions:
        for concept in concepts:
            if concept in concept_token_fractions:
                plot_values = concept_token_fractions[concept]
                if overlap_framing:
                    plot_values = [1 - v for v in plot_values]
                
                ax.plot(x_values, plot_values,
                       label=concept,
                       linestyle=line_styles.get(concept, '-') if line_styles else '-',
                       color=colors.get(concept, None),
                       alpha=0.7, linewidth=1.5, 
                       marker='o', markersize=3)
    
    # Plot average
    if avg_token_fractions and len(avg_token_fractions) == len(x_values):
        plot_avg_values = [1 - v for v in avg_token_fractions] if overlap_framing else avg_token_fractions
        
        if show_error_bars and len(concepts) > 1 and plot_individual_concepts:
            # Calculate standard deviation across concepts for error bars
            stds = []
            for i in range(len(percentthrumodels)):
                values_at_layer = [concept_token_fractions[c][i] for c in concept_token_fractions if c in concept_token_fractions]
                if values_at_layer:
                    stds.append(np.std(values_at_layer))
                else:
                    stds.append(0)
            
            errors = [std / np.sqrt(len(concepts)) for std in stds]
            label_text = 'Average' if plot_individual_concepts else ('True Concept Tokens Below Threshold' if overlap_framing else 'True Concept Tokens Above Threshold')
            ax.errorbar(x_values, plot_avg_values, yerr=errors,
                       label=label_text,
                       linestyle='-',
                       color='black' if plot_individual_concepts else 'purple',
                       capsize=3, alpha=0.8, linewidth=2.5,
                       marker='o', markersize=3)
        else:
            label_text = 'Average' if plot_individual_concepts else ('True Concept Tokens Below Threshold' if overlap_framing else 'True Concept Tokens Above Threshold')
            ax.plot(x_values, plot_avg_values,
                   label=label_text,
                   linestyle='-',
                   color='black' if plot_individual_concepts else 'purple',
                   linewidth=2.5,
                   marker='o', markersize=3)
    
    # Formatting - Move x-axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(x_label, fontsize=label_size)
    
    if custom_ylabel == "":
        # Don't set any y-label
        pass
    elif custom_ylabel:
        ax.set_ylabel(custom_ylabel, fontsize=label_size)
    else:
        if overlap_framing:
            ax.set_ylabel('Fraction of True Concept Tokens Below Threshold', fontsize=label_size)
        else:
            ax.set_ylabel('Fraction of True Concept Tokens Above Threshold', fontsize=label_size)
    
    # Handle title
    if custom_title == "":
        # Don't set any title
        pass
    elif custom_title:
        ax.set_title(custom_title, fontsize=label_size + 2, pad=20)  # Add padding to avoid overlap with x-axis
    else:
        # No default title when custom_title is None
        pass
    
    # Set y-axis limits
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, 1.05)  # Default for fractions
    
    ax.tick_params(axis='both', labelsize=label_size-2)
    
    # Set y-axis ticks at 0.25, 0.5, 0.75, 1.0
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    
    # Format x-axis ticks with % when using percentthrumodel
    if not use_layer_numbers:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    ax.grid(True, alpha=0.3)
    
    # Add legend if legend_text is provided
    if legend_text:
        # Create a custom legend entry with the same style as the plotted line
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='purple', linewidth=2.5, 
                                 marker='o', markersize=3, label=legend_text)]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=label_size-2,
                 frameon=True, fancybox=True, framealpha=0.8)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    # Show if requested  
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_background_detection_rates(
    computation_results: Dict[str, Any],
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True,
    use_layer_numbers: bool = False,
    label_size: int = 12,
    show_title: bool = True,
    custom_ylabel: Optional[str] = None,
    custom_title: Optional[str] = None,
    line_styles: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
    legend_loc: str = 'best',
    ylim: Optional[Tuple[float, float]] = None,
    show_error_bars: bool = True,
    plot_individual_concepts: bool = False,
    highlight_percentile: bool = True,
    legend_text: Optional[str] = None
) -> None:
    """
    Plot the percentage of GT images that have at least one patch greater than 
    the background percentile threshold across layers.
    
    Args:
        computation_results: Results from compute_separation_over_percentthru with compute_background_detection=True
        figsize: Figure size
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        use_layer_numbers: Whether to use layer numbers on x-axis (vs percentthrumodel)
        label_size: Font size for labels
        show_title: Whether to show title
        custom_title: Custom title (overrides default)
        custom_ylabel: Custom y-axis label (overrides default)
        line_styles: Dict mapping concept names to line styles
        colors: Dict mapping concept names to colors
        show_legend: Not used - legends are not shown
        legend_loc: Not used - legends are not shown
        ylim: Y-axis limits (default: (0, 1) for percentages)
        show_error_bars: Whether to show error bars for averaged detection
        plot_individual_concepts: Whether to plot individual concept lines
        highlight_percentile: Whether to show the background percentile used in title
        legend_text: Optional text to display in bottom right corner (if None or empty, no text is shown)
    """
    # Apply paper plotting style
    apply_paper_plotting_style()
    
    # Extract data
    if 'background_detection_results' not in computation_results:
        raise ValueError("No background detection results found. Make sure compute_background_detection=True was used.")
    
    results = computation_results['results']
    metadata = computation_results['metadata']
    background_detection_results = computation_results['background_detection_results']
    averaged_detection = computation_results.get('averaged_background_detection', [])
    
    dataset_name = metadata['dataset_name']
    model_name = metadata['model_name']
    percentthrumodels = metadata['percentthrumodels']
    concepts = metadata['concepts']
    is_text_dataset = metadata['is_text_dataset']
    
    # Get background percentile from first result that has it
    background_percentile = None
    for key, result in results.items():
        if 'background_threshold' in result:
            # Find the percentile value from the parameters used
            for concept in concepts:
                for percent in percentthrumodels:
                    if key == f"{concept}_{percent}":
                        # Extract from the computation parameters
                        background_percentile = computation_results.get('metadata', {}).get('background_percentile', 99.5)
                        break
                if background_percentile is not None:
                    break
        if background_percentile is not None:
            break
    
    # Get total layers for conversion
    if model_name == 'CLIP':
        total_layers = 24
    elif model_name == 'Llama' and is_text_dataset:
        total_layers = 32
    elif model_name == 'Llama' and not is_text_dataset:
        total_layers = 40
    elif model_name == 'Gemma':
        total_layers = 28
    elif model_name == 'Qwen':
        total_layers = 32
    else:
        total_layers = 24
    
    # Default colors for concepts
    default_colors = plt.cm.tab10(np.linspace(0, 1, len(concepts)))
    if colors is None:
        colors = {concept: default_colors[i] for i, concept in enumerate(concepts)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # X-axis values
    if use_layer_numbers:
        x_values = [percent_to_layer(p, total_layers, model_name) for p in percentthrumodels]
        x_label = 'Layer'
    else:
        x_values = percentthrumodels
        x_label = '% Through Model'
    
    # Plot individual concepts if requested
    if plot_individual_concepts:
        for concept in concepts:
            if concept in background_detection_results:
                detection_rates = background_detection_results[concept]
                if len(detection_rates) == len(x_values):
                    ax.plot(x_values, detection_rates,
                           label=concept,
                           linestyle=line_styles.get(concept, '-') if line_styles else '-',
                           color=colors.get(concept, None),
                           marker='o', markersize=6,
                           alpha=0.7, linewidth=1.5)
    
    # Plot average detection rate
    if averaged_detection and len(averaged_detection) == len(x_values):
        if show_error_bars and len(concepts) > 1 and plot_individual_concepts:
            # Calculate standard deviation across concepts for error bars
            stds = []
            for i in range(len(percentthrumodels)):
                rates_at_layer = []
                for concept in concepts:
                    if concept in background_detection_results and i < len(background_detection_results[concept]):
                        rates_at_layer.append(background_detection_results[concept][i])
                
                if rates_at_layer:
                    stds.append(np.std(rates_at_layer))
                else:
                    stds.append(0)
            
            errors = [std / np.sqrt(len(concepts)) for std in stds]
            ax.errorbar(x_values, averaged_detection, yerr=errors,
                       label='Average' if plot_individual_concepts else f'Detection Rate',
                       linestyle='-',
                       color='black' if plot_individual_concepts else 'blue',
                       marker='o', markersize=6,
                       capsize=3, alpha=0.8, linewidth=2.5)
        else:
            ax.plot(x_values, averaged_detection,
                   label='Average' if plot_individual_concepts else f'Detection Rate',
                   linestyle='-',
                   color='black' if plot_individual_concepts else 'blue',
                   marker='o', markersize=6,
                   linewidth=2.5)
    
    # Formatting
    ax.set_xlabel(x_label, fontsize=label_size)
    
    if custom_ylabel == "":
        # Don't set any y-label
        pass
    elif custom_ylabel:
        ax.set_ylabel(custom_ylabel, fontsize=label_size)
    else:
        ax.set_ylabel('% of GT Images Detected', fontsize=label_size)
    
    if custom_title == "":
        # Don't set any title
        pass
    elif show_title:
        if custom_title:
            ax.set_title(custom_title, fontsize=label_size+2)
        else:
            title = f'{dataset_name} - {model_name}: Non-Concept Detection'
            if highlight_percentile and background_percentile is not None:
                title += f'\n(>{background_percentile:.1f}th percentile of background)'
            ax.set_title(title, fontsize=label_size+2)
    
    # Set y-axis to percentage scale
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, 1.05)  # 0 to 105% to show full range
    
    # Set y-axis ticks at 25%, 50%, 75%, 100%
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    ax.grid(True, alpha=0.3)
    
    # Legend removed - no longer showing legend
    
    ax.tick_params(labelsize=label_size-2)
    
    # Format x-axis ticks with % when using percentthrumodel
    if not use_layer_numbers:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    # Add legend if legend_text is provided
    if legend_text:
        # Create a custom legend entry with the same style as the plotted line
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='blue', linewidth=2.5, 
                                 marker='o', markersize=3, label=legend_text)]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=label_size-2,
                 frameon=True, fancybox=True, framealpha=0.8)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_activation_distributions_grid(
    computation_results: Dict[str, Any],
    figsize: Tuple[float, float] = (12, 9),
    show_thresholds: bool = True,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    max_concepts: int = 10,
    concepts: Optional[List[str]] = None,
    n_bins: int = 50,
    alpha_bg: float = 0.6,
    alpha_gt: float = 0.7,
    show_legend: bool = True,
    show_metrics: bool = True,
    label_size: int = 10,
    x_padding_fraction: float = 0.0,
    xlabel_size: Optional[int] = None,
    concept_xlims: Optional[Tuple[float, float]] = None,
    legend_size: Optional[int] = None,
    percent_size: Optional[int] = None,
    concept_size: Optional[int] = None,
    plot_type: str = 'density',
    x_ticks: Optional[List[float]] = None,
    percentthrumodels: Optional[List[int]] = None
) -> None:
    """
    Plot activation distributions as histograms in a grid format (like original plot_concept_activation_grid).
    
    Args:
        computation_results: Results from compute_separation_over_percentthru
        figsize: Size of the entire figure (width, height)
        show_thresholds: Whether to show detection thresholds
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        max_concepts: Maximum number of concepts to display (ignored if concepts is provided)
        concepts: List of specific concepts to plot (overrides max_concepts)
        n_bins: Number of bins for histograms
        alpha_bg: Alpha value for background histogram
        alpha_gt: Alpha value for GT positive histogram
        show_legend: Whether to show legend on first plot
        show_metrics: Whether to show separability and Cohen's d metrics
        label_size: Font size for labels and metrics
        x_padding_fraction: Fraction of data range to pad on each side (0.0 = no padding, 0.005 = 0.5%)
        xlabel_size: Font size for x-axis labels (defaults to label_size if not specified)
        concept_xlims: Single (min, max) tuple for x-axis limits to apply to all plots
        legend_size: Font size for legend text (defaults to label_size if not specified)
        percent_size: Font size for percentage labels (defaults to label_size if not specified)
        concept_size: Font size for concept names (defaults to label_size if not specified)
        plot_type: Type of plot - 'density' for KDE smooth curves or 'histogram' for traditional histograms
        x_ticks: List of x-axis tick values (if None, uses automatic spacing of 3)
        percentthrumodels: List of percentthrumodel values to plot (if None or empty, uses all available)
    """
    # Apply paper plotting style from general_utils
    apply_paper_plotting_style()
    
    # Extract data
    results = computation_results['results']
    metadata = computation_results['metadata']
    
    dataset_name = metadata['dataset_name']
    model_name = metadata['model_name']
    
    # Use provided percentthrumodels or default to all available
    if percentthrumodels is None or len(percentthrumodels) == 0:
        percentthrumodels = metadata['percentthrumodels']
    else:
        # Validate that requested percentthrumodels exist in results
        available_percentthrus = metadata['percentthrumodels']
        percentthrumodels = [p for p in percentthrumodels if p in available_percentthrus]
        if not percentthrumodels:
            raise ValueError(f"None of the requested percentthrumodels found in results. Available: {available_percentthrus}")
    
    # Use provided concepts or default to metadata concepts
    if concepts is None:
        concepts = metadata['concepts'][:max_concepts]
    else:
        # Validate that requested concepts exist in results
        available_concepts = metadata['concepts']
        concepts = [c for c in concepts if c in available_concepts]
        if not concepts:
            raise ValueError(f"None of the requested concepts found in results. Available: {available_concepts[:10]}...")
    
    is_text_dataset = metadata['is_text_dataset']
    metric_type = metadata['metric_type']
    
    # Get total layers for title
    if model_name == 'CLIP':
        total_layers = 24
    elif model_name == 'Llama' and is_text_dataset:
        total_layers = 32
    elif model_name == 'Llama' and not is_text_dataset:
        total_layers = 40
    elif model_name == 'Gemma':
        total_layers = 28
    elif model_name == 'Qwen':
        total_layers = 32
    else:
        total_layers = 24
    
    # Create figure
    n_rows = len(concepts)
    n_cols = len(percentthrumodels)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                            gridspec_kw={'hspace': 0.15, 'wspace': 0.1})
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Track x-axis limits - now global across all concepts for alignment
    global_x_min = float('inf')
    global_x_max = float('-inf')
    
    # First pass: collect min/max across ALL concepts and layers
    for concept_name in concepts:
        for percent_thru_model in percentthrumodels:
            key = f"{concept_name}_{percent_thru_model}"
            if key in results and 'error' not in results[key]:
                result_data = results[key]
                gt_positive_acts = result_data.get('gt_positive_acts', np.array([]))
                background_acts = result_data.get('background_acts', np.array([]))
                
                if len(gt_positive_acts) > 0 and len(background_acts) > 0:
                    # Get the actual data range
                    all_acts = np.concatenate([gt_positive_acts, background_acts])
                    global_x_min = min(global_x_min, np.min(all_acts))
                    global_x_max = max(global_x_max, np.max(all_acts))
    
    # Apply padding to global limits
    if global_x_min != float('inf') and global_x_max != float('-inf'):
        if x_padding_fraction > 0:
            x_range = global_x_max - global_x_min
            x_padding = x_padding_fraction * x_range
            global_x_limits = (global_x_min - x_padding, global_x_max + x_padding)
        else:
            # Use exact data range without padding
            global_x_limits = (global_x_min, global_x_max)
    else:
        global_x_limits = None
    
    # Plot each cell
    for row_idx, concept_name in enumerate(concepts):
        for col_idx, percent_thru_model in enumerate(percentthrumodels):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            
            key = f"{concept_name}_{percent_thru_model}"
            if key not in results:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=label_size)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            result_data = results[key]
            
            # Check for errors
            if 'error' in result_data:
                ax.text(0.5, 0.5, result_data['error'], ha='center', va='center', 
                       transform=ax.transAxes, fontsize=label_size, wrap=True)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            gt_positive_acts = result_data.get('gt_positive_acts', np.array([]))
            background_acts = result_data.get('background_acts', np.array([]))
            
            # Plot distributions if we have data
            if len(gt_positive_acts) > 0 and len(background_acts) > 0:
                # Use global x-axis limits for alignment
                if global_x_limits is not None:
                    x_min, x_max = global_x_limits
                else:
                    # Fallback to computing range from current data
                    all_acts = np.concatenate([background_acts, gt_positive_acts])
                    x_min, x_max = np.min(all_acts), np.max(all_acts)
                
                if plot_type == 'density':
                    # Create smooth x values for plotting
                    x_smooth = np.linspace(x_min, x_max, 500)
                    
                    # Create KDE for background
                    kde_bg = gaussian_kde(background_acts)
                    y_bg = kde_bg(x_smooth)
                    ax.fill_between(x_smooth, y_bg, alpha=alpha_bg, color='#505050', 
                                   label=f'Out-of-Concept Tokens ({len(background_acts)})')
                    
                    # Create KDE for GT positive
                    kde_gt = gaussian_kde(gt_positive_acts)
                    y_gt = kde_gt(x_smooth)
                    ax.fill_between(x_smooth, y_gt, alpha=alpha_gt, color='green',
                                   label=f'In-Concept Tokens ({len(gt_positive_acts)})')
                else:  # histogram
                    # Use consistent bins for both histograms
                    bins = np.linspace(x_min, x_max, n_bins)
                    
                    ax.hist(background_acts, bins=bins, density=True, alpha=alpha_bg,
                           color='#505050', label=f'Out-of-Concept Tokens ({len(background_acts)})')
                    ax.hist(gt_positive_acts, bins=bins, density=True, alpha=alpha_gt,
                           color='green', label=f'In-Concept Tokens ({len(gt_positive_acts)})')
                
                # Add threshold if available and requested
                if show_thresholds and result_data.get('threshold') is not None:
                    ax.axvline(result_data['threshold'], color='purple', linestyle='-', 
                             linewidth=2, label='Thresh')
                
                # AUC metric text removed per request
                
                # Formatting
                if row_idx == 0:
                    # Add title with just percent_thru_model
                    ax.set_title(f'{percent_thru_model}%', fontsize=percent_size if percent_size is not None else label_size)
                
                if col_idx == 0:
                    # Add concept name as horizontal text to the left of the plot - capitalized and italicized
                    # If concept has "::", use only the part after it
                    display_name = concept_name.split("::")[-1] if "::" in concept_name else concept_name
                    concept_label = display_name.capitalize() if len(display_name) <= 20 else f'{display_name[:20].capitalize()}...'
                    ax.text(-0.15, 0.5, concept_label, transform=ax.transAxes,
                           verticalalignment='center', horizontalalignment='right', 
                           fontsize=concept_size if concept_size is not None else label_size, rotation=0, style='italic')
                
                # Show x ticks on all rows
                ax.tick_params(axis='x', bottom=True, labelbottom=True, length=4, width=1.0, color='black', labelsize=8, direction='inout')
                
                # Set x ticks at intervals of 2, including negative values
                x_min, x_max = ax.get_xlim()
                # Start from the nearest multiple of 2 below x_min
                start_tick = np.floor(x_min / 2) * 2
                # Generate ticks from start to x_max
                x_ticks = np.arange(start_tick, x_max + 1, 2)
                # Only include ticks within the actual range
                x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]
                ax.set_xticks(x_ticks)
                
                # Only show x-label on bottom row
                if row_idx == n_rows - 1:
                    # Use "Activation (s)" for x-axis label
                    x_label = "Activation (s)"
                    # Use xlabel_size if provided, otherwise use label_size
                    ax.set_xlabel(x_label, fontsize=xlabel_size if xlabel_size is not None else label_size)
                
                ax.tick_params(axis='y', left=False, labelleft=False)  # Remove y ticks and labels
                ax.grid(False)  # No grid lines
                
                # Set y-axis to start at 0
                ax.set_ylim(bottom=0)
                
                # Apply x-limits - use custom xlims if provided, otherwise use global limits
                if concept_xlims is not None:
                    # User provided custom limits for all plots
                    ax.set_xlim(concept_xlims)
                    ax.margins(x=0)
                elif global_x_limits is not None:
                    # Use global limits for alignment
                    ax.set_xlim(global_x_limits)
                    ax.margins(x=0)
                
                # Don't put legend on individual plots - we'll put it on top
            else:
                # No valid data
                error_msg = 'No GT+' if len(gt_positive_acts) == 0 else 'No Bg'
                ax.text(0.5, 0.5, error_msg, ha='center', va='center', transform=ax.transAxes, fontsize=label_size)
                ax.set_xticks([])
                ax.set_yticks([])
    
    # No overall title as requested
    
    # Add legend at the bottom
    if show_legend:
        # Create a dummy plot to get legend handles
        dummy_ax = fig.add_subplot(111, frameon=False)
        dummy_ax.hist([], alpha=alpha_bg, color='#505050', edgecolor='black', label='Out-of-Concept Tokens')
        dummy_ax.hist([], alpha=alpha_gt, color='#FF9500', edgecolor='#CC7700', label='In-Concept Tokens')
        if show_thresholds:
            dummy_ax.axvline(0, color='purple', linestyle='-', linewidth=2, label='Threshold')
        
        # Place legend at the bottom center with proper transform
        handles, labels = dummy_ax.get_legend_handles_labels()
        # Use figure transform to make position independent of subplot size
        fig.legend(handles, labels, loc='upper center', ncol=len(handles), 
                  fontsize=legend_size if legend_size is not None else label_size, 
                  bbox_to_anchor=(0.54, 0), bbox_transform=fig.transFigure)
        
        # Hide the dummy axes
        dummy_ax.set_visible(False)
    
    # Add overall label at the top, centered relative to plot area (not including concept labels)
    # Since concept labels take 0.08 of the figure width, center relative to [0.08, 1.0]
    fig.text(0.54, 1.005, '% Through Model', ha='center', va='bottom', fontsize=label_size + 2)
    
    # Adjust layout to accommodate horizontal concept names on the left, title at top, and legend at bottom
    # Leave more space at bottom for legend that won't change with figure size
    plt.tight_layout(rect=[0.08, 0.14 if show_legend else 0, 1, 0.94], h_pad=0.5, w_pad=0.5)
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    # Show if requested  
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_separation_metrics(
    computation_results: Dict[str, Any],
    metrics_to_plot: List[str] = ['auc'],
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True,
    use_layer_numbers: bool = False,
    label_size: int = 12,
    show_title: bool = True,
    custom_ylabel: Optional[str] = None,
    custom_title: Optional[str] = None,
    line_styles: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
    legend_loc: str = 'best',
    ylim: Optional[Tuple[float, float]] = None,
    show_error_bars: bool = True,
    plot_individual_concepts: bool = False
) -> None:
    """
    Plot separation metrics as line plots across layers.
    
    Args:
        computation_results: Results from compute_separation_over_percentthru
        metrics_to_plot: List of metrics to plot ('auc', 'separability', 'cohens_d', 'overlap_area')
        figsize: Figure size
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        use_layer_numbers: Whether to use layer numbers on x-axis (vs percentthrumodel)
        label_size: Font size for labels
        show_title: Whether to show title
        custom_title: Custom title (overrides default)
        custom_ylabel: Custom y-axis label (overrides default)
        line_styles: Dict mapping metric names to line styles
        colors: Dict mapping metric names to colors
        show_legend: Whether to show legend
        legend_loc: Legend location
        ylim: Y-axis limits
        show_error_bars: Whether to show error bars for averaged metrics
        plot_individual_concepts: Whether to plot individual concept lines (vs just average)
    """
    # Apply paper plotting style
    apply_paper_plotting_style()
    
    # Extract data
    results = computation_results['results']
    metadata = computation_results['metadata']
    
    dataset_name = metadata['dataset_name']
    model_name = metadata['model_name']
    percentthrumodels = metadata['percentthrumodels']
    concepts = metadata['concepts']
    is_text_dataset = metadata['is_text_dataset']
    
    # Get total layers for conversion
    if model_name == 'CLIP':
        total_layers = 24
    elif model_name == 'Llama' and is_text_dataset:
        total_layers = 32
    elif model_name == 'Llama' and not is_text_dataset:
        total_layers = 40
    elif model_name == 'Gemma':
        total_layers = 28
    elif model_name == 'Qwen':
        total_layers = 32
    else:
        total_layers = 24
    
    # Default styles and colors
    default_line_styles = {
        'separability': '-',
        'auc': '-',
        'auc_roc': '-',
        'cohens_d': '--',
        'overlap_area': ':'
    }
    
    default_colors = {
        'separability': 'blue',
        'auc': 'green',
        'auc_roc': 'green',
        'cohens_d': 'red',
        'overlap_area': 'purple'
    }
    
    if line_styles is None:
        line_styles = default_line_styles
    if colors is None:
        colors = default_colors
    
    # Metric display names
    metric_names = {
        'separability': 'Separability',
        'auc': 'AUC-ROC',
        'auc_roc': 'AUC-ROC',
        'cohens_d': "Cohen's d",
        'overlap_area': 'Overlap Area'
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # X-axis values
    if use_layer_numbers:
        x_values = [percent_to_layer(p, total_layers, model_name) for p in percentthrumodels]
        x_label = 'Layer'
    else:
        x_values = percentthrumodels
        x_label = '% Through Model'
    
    # Plot each metric
    for metric in metrics_to_plot:
        # Handle 'auc' -> 'auc_roc' mapping
        metric_key = 'auc_roc' if metric == 'auc' else metric
        
        if metric_key not in metric_names:
            print(f"Warning: Unknown metric '{metric}', skipping")
            continue
        
        # Collect metric values across concepts and layers
        if plot_individual_concepts:
            # Plot each concept separately
            for concept in concepts:
                concept_values = []
                for percent in percentthrumodels:
                    key = f"{concept}_{percent}"
                    if key in results and metric_key in results[key]:
                        value = results[key][metric_key]
                        concept_values.append(value if value is not None else np.nan)
                    else:
                        concept_values.append(np.nan)
                
                ax.plot(x_values, concept_values,
                       label=f'{metric_names[metric_key]} - {concept}',
                       linestyle=line_styles.get(metric, '-'),
                       alpha=0.6, linewidth=1.5)
        
        # Always plot average
        metric_values = []
        metric_stds = []
        
        for percent in percentthrumodels:
            values_at_layer = []
            
            for concept in concepts:
                key = f"{concept}_{percent}"
                if key in results and metric_key in results[key]:
                    value = results[key][metric_key]
                    if value is not None:
                        values_at_layer.append(value)
            
            # Average across concepts for this layer
            if values_at_layer:
                metric_values.append(np.mean(values_at_layer))
                metric_stds.append(np.std(values_at_layer))
            else:
                metric_values.append(np.nan)
                metric_stds.append(0)
        
        # Plot average with optional error bars
        if show_error_bars and len(concepts) > 1:
            # Calculate standard error
            errors = [std / np.sqrt(len(concepts)) for std in metric_stds]
            ax.errorbar(x_values, metric_values, yerr=errors,
                       label=metric_names[metric_key] + (' (avg)' if plot_individual_concepts else ''),
                       linestyle=line_styles.get(metric, '-'),
                       color=colors.get(metric, None),
                       capsize=3, alpha=0.8, linewidth=2)
        else:
            ax.plot(x_values, metric_values,
                   label=metric_names[metric_key] + (' (avg)' if plot_individual_concepts else ''),
                   linestyle=line_styles.get(metric, '-'),
                   color=colors.get(metric, None),
                   linewidth=2)
    
    # Formatting
    ax.set_xlabel(x_label, fontsize=label_size)
    
    if custom_ylabel == "":
        # Don't set any y-label
        pass
    elif custom_ylabel:
        ax.set_ylabel(custom_ylabel, fontsize=label_size)
    else:
        if len(metrics_to_plot) == 1:
            ax.set_ylabel(metric_names.get(metrics_to_plot[0] if metrics_to_plot[0] != 'auc' else 'auc_roc', 
                                          'Metric Value'), fontsize=label_size)
        else:
            ax.set_ylabel('Metric Value', fontsize=label_size)
    
    if custom_title == "":
        # Don't set any title
        pass
    elif show_title:
        if custom_title:
            ax.set_title(custom_title, fontsize=label_size+2)
        else:
            title = f'{dataset_name} - {model_name}: Separation Metrics Across Layers'
            if len(concepts) == 1:
                title += f'\n{concepts[0]}'
            elif not plot_individual_concepts:
                title += f'\n(Averaged over {len(concepts)} concepts)'
            ax.set_title(title, fontsize=label_size+2)
    
    ax.grid(True, alpha=0.3)
    
    if ylim:
        ax.set_ylim(ylim)
    
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=label_size-2, framealpha=0.9)
    
    ax.tick_params(labelsize=label_size-2)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)





def plot_single_sample_activation_distribution(
    sample_index: int,
    concept_name: str,
    dataset_name: str,
    model_name: str,
    concept_type: str = "avg_patch_embeddings",
    percentthrumodel: int = 100,
    scratch_dir: str = "",
    model_input_size: Union[int, Tuple[int, int]] = (224, 224),
    figsize: Optional[Tuple[float, float]] = None,
    fontsize: Optional[int] = None,
    legendsize: Optional[int] = None,
    n_bins: int = 30,
    show_threshold: bool = True,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    alpha_positive: float = 0.9,
    alpha_negative: float = 0.9,
    color_positive: str = "#FF9500",
    color_negative: str = "#4d4d4d",
    show_stats: bool = True,
    use_gpu: bool = True,
    is_image: bool = True,
    plot_type: str = "histogram",
    show_global: bool = False,
    global_alpha: float = 0.3,
    x_lim: Optional[Tuple[float, float]] = None,
    count_max: Optional[float] = None
) -> Dict[str, Any]:
    """
    Plot activation distributions for GT positive vs GT negative patches/tokens 
    within a single sample for a given concept.
    
    Args:
        sample_index: Index of the sample (image or text document) to analyze
        concept_name: Name of the concept to analyze
        dataset_name: Name of the dataset
        model_name: Name of the model (e.g., "CLIP", "Llama")
        concept_type: Type of concept ("avg_patch_embeddings" or "linsep_patch_embeddings_BD_True_BN_False")
        percentthrumodel: Percentage through model
        scratch_dir: Directory containing activations
        model_input_size: Input size for the model - can be int (e.g., 224) or tuple (e.g., (224, 224))
        figsize: Figure size as (width, height). If None, uses (10, 6)
        fontsize: Font size for labels. If None, uses default from paper plotting style (10)
        legendsize: Font size for legend. If None, uses fontsize - 1
        n_bins: Number of bins for histogram
        show_threshold: Whether to show the detection threshold
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        alpha_positive: Alpha value for positive patches
        alpha_negative: Alpha value for negative patches
        color_positive: Color for positive patches
        color_negative: Color for negative patches
        show_stats: Whether to show statistics on the plot
        use_gpu: Whether to use GPU for computation
        is_image: Whether this is an image dataset (True) or text dataset (False)
        plot_type: Type of plot - "histogram" for traditional histograms or "kde" for kernel density estimation (smooth curves)
        show_global: Whether to show global test set distributions in the background
        global_alpha: Alpha transparency for global distributions (only used if show_global=True)
        x_lim: Optional tuple (min, max) to set x-axis limits. If None, limits are set automatically.
        count_max: Optional maximum value for the y-axis of the count histogram. If None, scales automatically.
    
    Returns:
        Dictionary containing:
        - "positive_activations": Array of activations for GT positive patches/tokens
        - "negative_activations": Array of activations for GT negative patches/tokens
        - "statistics": Dict with mean, std, and other stats for both distributions
        - "threshold": Detection threshold value (if available)
    """
    # Apply paper plotting style
    apply_paper_plotting_style()
    
    # Set figsize default if not provided
    if figsize is None:
        figsize = (10, 6)
    
    # Get fontsize from paper style if not provided
    if fontsize is None:
        fontsize = get_paper_plotting_style()['font.size']
    
    # Set legendsize default if not provided
    if legendsize is None:
        legendsize = fontsize - 1
    
    # Determine if text dataset
    is_text_dataset = dataset_name in ["Sarcasm", "iSarcasm", "GoEmotions", "Stanford-Tree-Bank", "IMDB", "Jailbreak"]
    unit_type = "token" if is_text_dataset else "patch"
    
    # Determine metric type based on concept type
    sample_type = "patch"
    if concept_type in ["avg_patch_embeddings", "kmeans_1000_patch_embeddings_kmeans"]:
        metric_type = "Cosine Similarity"
        acts_prefix = "cosine_similarities"
    else:
        metric_type = "Distance to Boundary"
        acts_prefix = "dists"
    
    # Load ground truth data
    print(f"Loading ground truth data for {dataset_name}...")
    
    # Format the input size string for filename
    if is_text_dataset:
        # Text datasets use ('text', 'text') format
        if isinstance(model_input_size, tuple):
            input_size_str = str(model_input_size)
        else:
            input_size_str = "('text', 'text')"
    else:
        # Image datasets use (width, height) format
        if isinstance(model_input_size, tuple):
            input_size_str = str(model_input_size)  # e.g., "(224, 224)"
        else:
            input_size_str = f'({model_input_size}, {model_input_size})'
    
    gt_patches_file = f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{input_size_str}.pt"
    if not os.path.exists(gt_patches_file):
        gt_patches_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{input_size_str}.pt"
    
    if not os.path.exists(gt_patches_file):
        raise FileNotFoundError(f"Patch-level ground truth not found at {gt_patches_file}")
    
    gt_patches_per_concept = torch.load(gt_patches_file, weights_only=False)
    
    # Filter to valid concepts
    gt_patches_per_concept = filter_concept_dict(gt_patches_per_concept, dataset_name)
    
    # Check if concept exists
    if concept_name not in gt_patches_per_concept:
        raise ValueError(f"Concept '{concept_name}' not found in ground truth data. Available concepts: {list(gt_patches_per_concept.keys())[:10]}...")
    
    # Load activations
    print(f"Loading activations...")
    if concept_type == "avg_patch_embeddings":
        acts_filename = f"{acts_prefix}_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}_chunk_0.pt"
    elif concept_type == "linsep_patch_embeddings_BD_True_BN_False":
        acts_filename = f"{acts_prefix}_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}_chunk_0.pt"
    
    acts_dir = f"{scratch_dir}/{'Cosine_Similarities' if acts_prefix == 'cosine_similarities' else 'Distances'}/{dataset_name}"
    acts_path = f"{acts_dir}/{acts_filename}"
    
    if not os.path.exists(acts_path):
        raise FileNotFoundError(f"Activations not found at {acts_path}")
    
    # Load activations
    acts_data_dict = torch.load(acts_path, weights_only=False)
    if isinstance(acts_data_dict, dict) and 'activations' in acts_data_dict:
        acts_data = acts_data_dict['activations']
    else:
        acts_data = acts_data_dict
    
    # Load concepts to get concept index
    if concept_type == "avg_patch_embeddings":
        concepts_filename = f"avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}.pt"
    elif concept_type == "linsep_patch_embeddings_BD_True_BN_False":
        concepts_filename = f"linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}.pt"
    
    concepts_file = f"Concepts/{dataset_name}/{concepts_filename}"
    
    if not os.path.exists(concepts_file):
        raise FileNotFoundError(f"Concepts file not found: {concepts_file}")
        
    concepts_data = torch.load(concepts_file, weights_only=False)
    if isinstance(concepts_data, dict):
        all_concept_names = list(concepts_data.keys())
    else:
        raise ValueError(f"Unexpected format in concepts file: {concepts_file}")
    
    if concept_name not in all_concept_names:
        raise ValueError(f"Concept '{concept_name}' not found in concepts. Available: {all_concept_names[:10]}...")
    
    concept_idx = all_concept_names.index(concept_name)
    
    # Get activations for the specific sample
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    if is_text_dataset:
        # For text, we need to load token counts to map sample index to token indices
        # Format the input size string for token counts file
        if isinstance(model_input_size, tuple):
            token_input_size_str = str(model_input_size)
        else:
            # For text, default is ('text', 'text')
            token_input_size_str = "('text', 'text')"
            
        token_counts_file = f"GT_Samples/{dataset_name}/token_counts_inputsize_{token_input_size_str}.pt"
        if not os.path.exists(token_counts_file):
            raise FileNotFoundError(f"Token counts file not found at {token_counts_file}")
            
        token_counts = torch.load(token_counts_file, weights_only=False)
        
        # Calculate start and end token indices for this sample
        start_token_idx = 0
        for i in range(sample_index):
            if i >= len(token_counts):
                raise ValueError(f"Sample index {sample_index} out of range. Dataset has {len(token_counts)} samples.")
            start_token_idx += sum(token_counts[i])
        
        if sample_index >= len(token_counts):
            raise ValueError(f"Sample index {sample_index} out of range. Dataset has {len(token_counts)} samples.")
            
        num_tokens_in_sample = sum(token_counts[sample_index])
        end_token_idx = start_token_idx + num_tokens_in_sample
        
        # Get activations for all tokens in this sample
        sample_acts = acts_data[start_token_idx:end_token_idx, concept_idx]
        
        if use_gpu and torch.cuda.is_available():
            sample_acts = sample_acts.to(device)
        
        # Get GT positive tokens for this concept
        gt_positive_tokens = gt_patches_per_concept[concept_name]  # In text, "patches" are actually tokens
        
        # Filter to tokens from this specific sample
        sample_positive_tokens = []
        for token_idx in gt_positive_tokens:
            if start_token_idx <= token_idx < end_token_idx:
                # Convert to local token index within the sample
                local_token_idx = token_idx - start_token_idx
                sample_positive_tokens.append(local_token_idx)
        
        # Create mask for positive vs negative tokens
        positive_mask = torch.zeros(num_tokens_in_sample, dtype=torch.bool, device=device)
        if sample_positive_tokens:
            positive_mask[sample_positive_tokens] = True
        
        negative_mask = ~positive_mask
        
        # Get activations for positive and negative tokens
        positive_acts = sample_acts[positive_mask].cpu().numpy()
        negative_acts = sample_acts[negative_mask].cpu().numpy()
        
        # Store for later use
        image_positive_patches = sample_positive_tokens  # For consistency with image variable name
        patches_per_image = num_tokens_in_sample  # For consistency with image variable name
    else:
        # For images, calculate patch indices for this image
        patches_per_image = 196  # 14x14 for standard vision transformers
        start_patch_idx = sample_index * patches_per_image
        end_patch_idx = start_patch_idx + patches_per_image
        
        # Get activations for all patches in this image
        sample_acts = acts_data[start_patch_idx:end_patch_idx, concept_idx]
        
        if use_gpu and torch.cuda.is_available():
            sample_acts = sample_acts.to(device)
        
        # Get GT positive patches for this concept
        gt_positive_patches = gt_patches_per_concept[concept_name]
        
        # Filter to patches from this specific image
        image_positive_patches = []
        for patch_idx in gt_positive_patches:
            if start_patch_idx <= patch_idx < end_patch_idx:
                # Convert to local patch index within the image
                local_patch_idx = patch_idx - start_patch_idx
                image_positive_patches.append(local_patch_idx)
        
        # Create mask for positive vs negative patches
        positive_mask = torch.zeros(patches_per_image, dtype=torch.bool, device=device)
        if image_positive_patches:
            positive_mask[image_positive_patches] = True
        
        negative_mask = ~positive_mask
        
        # Get activations for positive and negative patches
        positive_acts = sample_acts[positive_mask].cpu().numpy()
        negative_acts = sample_acts[negative_mask].cpu().numpy()
    
    # Load threshold if requested
    threshold_value = None
    if show_threshold:
        threshold_file = None
        if concept_type == "avg_patch_embeddings":
            threshold_file = f"Thresholds/{dataset_name}/all_percentiles_{model_name}_avg_patch_embeddings_percentthrumodel_{percentthrumodel}.pt"
        elif concept_type == "linsep_patch_embeddings_BD_True_BN_False":
            threshold_file = f"Thresholds/{dataset_name}/all_percentiles_{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}.pt"
        
        if threshold_file and os.path.exists(threshold_file):
            try:
                threshold_data = torch.load(threshold_file, weights_only=False)
                if concept_name in threshold_data:
                    percentiles = threshold_data[concept_name]["percentiles"]
                    thresholds = threshold_data[concept_name]["thresholds"]
                    # Use 95th percentile as default threshold
                    idx_95 = (percentiles == 95).nonzero(as_tuple=True)[0]
                    if len(idx_95) > 0:
                        threshold_value = thresholds[idx_95[0]].item()
            except Exception as e:
                print(f"Warning: Could not load threshold: {e}")
    
    # Compute global distributions if requested
    global_positive_acts = None
    global_negative_acts = None
    
    if show_global:
        print("Computing global test set distributions...")
        
        # Load test metadata to get all test sample indices
        try:
            metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
            test_global_indices = metadata[metadata['split'] == 'test'].index.tolist()
        except:
            print("Warning: Could not load metadata for global distributions")
            show_global = False
        
        # Load ground truth samples file
        if show_global:
            try:
                if is_text_dataset:
                    if isinstance(model_input_size, tuple):
                        input_size_str = str(model_input_size)
                    else:
                        input_size_str = "('text', 'text')"
                else:
                    if isinstance(model_input_size, tuple):
                        input_size_str = str(model_input_size)
                    else:
                        input_size_str = f'({model_input_size}, {model_input_size})'
                
                gt_samples_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{input_size_str}.pt"
                gt_samples_per_concept = torch.load(gt_samples_file, weights_only=False)
                gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
            except:
                print("Warning: Could not load ground truth samples for global distributions")
                show_global = False
        
        if show_global:
            # Get all GT positive patches/tokens in test set
            gt_positive_patches_test = [p for p in gt_patches_per_concept[concept_name] 
                                       if p < len(acts_data)]  # Ensure within bounds
            
            # Get all test activations for this concept
            test_acts_concept = acts_data[:, concept_idx]
            
            if is_text_dataset:
                # For text, we need to map tokens to test samples
                # This is complex, so for now just use all positive tokens
                global_positive_acts = test_acts_concept[gt_positive_patches_test].cpu().numpy()
                # For negative, sample from all non-positive tokens
                all_indices = set(range(len(test_acts_concept)))
                negative_indices = list(all_indices - set(gt_positive_patches_test))
                if len(negative_indices) > 100000:  # Subsample if too many
                    negative_indices = np.random.choice(negative_indices, 100000, replace=False)
                global_negative_acts = test_acts_concept[negative_indices].cpu().numpy()
            else:
                # For images, filter to only test images
                patches_per_image = 196 if model_input_size == (224, 224) else 1600
                
                # Get positive patches from test images only
                test_positive_patches = []
                for patch_idx in gt_positive_patches_test:
                    img_idx = patch_idx // patches_per_image
                    if img_idx in test_global_indices:
                        test_positive_patches.append(patch_idx)
                
                if test_positive_patches:
                    global_positive_acts = test_acts_concept[test_positive_patches].cpu().numpy()
                
                # Get negative patches from test images without the concept
                samples_with_concept = set(gt_samples_per_concept.get(concept_name, []))
                test_negative_patches = []
                
                for test_idx in test_global_indices:
                    if test_idx not in samples_with_concept:
                        # Add all patches from this image
                        start_idx = test_idx * patches_per_image
                        end_idx = start_idx + patches_per_image
                        test_negative_patches.extend(range(start_idx, min(end_idx, len(test_acts_concept))))
                
                # Subsample if too many
                if len(test_negative_patches) > 100000:
                    test_negative_patches = np.random.choice(test_negative_patches, 100000, replace=False)
                
                if test_negative_patches:
                    global_negative_acts = test_acts_concept[test_negative_patches].cpu().numpy()
            
            print(f"Global distributions - Positive: {len(global_positive_acts) if global_positive_acts is not None else 0}, "
                  f"Negative: {len(global_negative_acts) if global_negative_acts is not None else 0}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Store global distributions for scaling later
    global_neg_hist_data = None
    global_pos_hist_data = None
    
    # Plot global distributions first (so they appear behind)
    if show_global and global_positive_acts is not None and global_negative_acts is not None:
        if plot_type == "kde":
            from scipy.stats import gaussian_kde
            # We'll scale these after plotting the local distributions
            pass
        else:  # histogram
            # We'll use KDE for smooth curves instead of histograms
            # Store the raw data for later KDE computation
            global_neg_hist_data = (global_negative_acts, None)
            global_pos_hist_data = (global_positive_acts, None)
    
    # Plot distributions
    if plot_type == "kde":
        # Use kernel density estimation for smooth curves
        from scipy.stats import gaussian_kde
        
        # Plot global distributions first if requested
        if show_global and global_positive_acts is not None and global_negative_acts is not None:
            # Compute local KDE peaks first for scaling
            local_neg_peak = 0
            local_pos_peak = 0
            
            if len(negative_acts) > 1:
                kde_local_neg = gaussian_kde(negative_acts)
                x_range_neg = np.linspace(negative_acts.min(), negative_acts.max(), 200)
                local_neg_peak = np.max(kde_local_neg(x_range_neg))
            
            if len(positive_acts) > 1:
                kde_local_pos = gaussian_kde(positive_acts)
                x_range_pos = np.linspace(positive_acts.min(), positive_acts.max(), 200)
                local_pos_peak = np.max(kde_local_pos(x_range_pos))
            
            # Plot scaled global KDEs
            if len(global_negative_acts) > 1 and local_neg_peak > 0:
                kde_global_neg = gaussian_kde(global_negative_acts)
                x_range = np.linspace(global_negative_acts.min(), global_negative_acts.max(), 200)
                global_neg_values = kde_global_neg(x_range)
                global_neg_peak = np.max(global_neg_values)
                
                if global_neg_peak > 0:
                    scale_factor = local_neg_peak / global_neg_peak
                    scaled_values = global_neg_values * scale_factor
                    # Just fill, no outline
                    ax.fill_between(x_range, scaled_values, alpha=global_alpha, color='gray', label="Global: Out-of-Concept")
            
            if len(global_positive_acts) > 1 and local_pos_peak > 0:
                kde_global_pos = gaussian_kde(global_positive_acts)
                x_range = np.linspace(global_positive_acts.min(), global_positive_acts.max(), 200)
                global_pos_values = kde_global_pos(x_range)
                global_pos_peak = np.max(global_pos_values)
                
                if global_pos_peak > 0:
                    scale_factor = local_pos_peak / global_pos_peak
                    scaled_values = global_pos_values * scale_factor
                    # Just fill, no outline
                    ax.fill_between(x_range, scaled_values, alpha=global_alpha, color='darkgreen', label="Global: In-Concept")
        
        # Create twin axis for sample counts
        ax2 = ax.twinx()
        
        # Set count axis limit if provided
        if count_max is not None:
            ax2.set_ylim(0, count_max)
        
        # Now plot sample distributions as histograms on the count axis
        # First determine common bin edges for both histograms
        all_sample_acts = []
        if len(negative_acts) > 0:
            all_sample_acts.extend(negative_acts)
        if len(positive_acts) > 0:
            all_sample_acts.extend(positive_acts)
        
        if len(all_sample_acts) > 0:
            # Create common bins for both histograms
            bin_edges = np.linspace(np.min(all_sample_acts), np.max(all_sample_acts), n_bins + 1)
            
            if len(negative_acts) > 0:
                # Plot histogram on twin axis for counts
                ax2.hist(negative_acts, bins=bin_edges, density=False, alpha=alpha_negative,
                        color=color_negative, edgecolor="none", label="Sample: Out-of-Concept")
            
            if len(positive_acts) > 0:
                # Plot histogram on twin axis for counts
                ax2.hist(positive_acts, bins=bin_edges, density=False, alpha=alpha_positive,
                        color=color_positive, edgecolor="none", label="Sample: In-Concept")
                
        # Ensure x-axis covers both distributions
        if len(negative_acts) > 0 and len(positive_acts) > 0:
            all_acts = np.concatenate([negative_acts, positive_acts])
            margin = (all_acts.max() - all_acts.min()) * 0.1
            ax.set_xlim(all_acts.min() - margin, all_acts.max() + margin)
            
    else:  # histogram
        # Plot global distributions first (so they appear behind)
        if show_global and global_neg_hist_data is not None and global_pos_hist_data is not None:
            from scipy.stats import gaussian_kde
            
            # Extract the raw activation data
            global_negative_acts = global_neg_hist_data[0]
            global_positive_acts = global_pos_hist_data[0]
            
            # Get the peak heights of local distributions
            local_neg_hist, _ = np.histogram(negative_acts, bins=n_bins, density=True) if len(negative_acts) > 0 else (np.array([0]), None)
            local_pos_hist, _ = np.histogram(positive_acts, bins=n_bins, density=True) if len(positive_acts) > 0 else (np.array([0]), None)
            
            local_neg_peak = np.max(local_neg_hist) if len(local_neg_hist) > 0 else 1
            local_pos_peak = np.max(local_pos_hist) if len(local_pos_hist) > 0 else 1
            
            # Plot global distributions as smooth KDE curves
            if len(global_negative_acts) > 1 and len(negative_acts) > 0:
                try:
                    # Create KDE for global negative distribution
                    kde_global_neg = gaussian_kde(global_negative_acts)
                    # Create a smooth x-range that covers the data range
                    x_min = min(global_negative_acts.min(), negative_acts.min() if len(negative_acts) > 0 else global_negative_acts.min())
                    x_max = max(global_negative_acts.max(), negative_acts.max() if len(negative_acts) > 0 else global_negative_acts.max())
                    x_range = np.linspace(x_min, x_max, 300)
                    global_neg_values = kde_global_neg(x_range)
                    global_neg_peak_kde = np.max(global_neg_values)
                    
                    if global_neg_peak_kde > 0:
                        # Scale to match local histogram peak
                        scale_factor = local_neg_peak / global_neg_peak_kde
                        scaled_values = global_neg_values * scale_factor
                        # Just fill, no outline
                        ax.fill_between(x_range, scaled_values, alpha=global_alpha, color='gray', label="Global: Out-of-Concept")
                except:
                    # Fallback to histogram if KDE fails
                    pass
            
            if len(global_positive_acts) > 1 and len(positive_acts) > 0:
                try:
                    # Create KDE for global positive distribution
                    kde_global_pos = gaussian_kde(global_positive_acts)
                    # Create a smooth x-range that covers the data range
                    x_min = min(global_positive_acts.min(), positive_acts.min() if len(positive_acts) > 0 else global_positive_acts.min())
                    x_max = max(global_positive_acts.max(), positive_acts.max() if len(positive_acts) > 0 else global_positive_acts.max())
                    x_range = np.linspace(x_min, x_max, 300)
                    global_pos_values = kde_global_pos(x_range)
                    global_pos_peak_kde = np.max(global_pos_values)
                    
                    if global_pos_peak_kde > 0:
                        # Scale to match local histogram peak
                        scale_factor = local_pos_peak / global_pos_peak_kde
                        scaled_values = global_pos_values * scale_factor
                        # Just fill, no outline
                        ax.fill_between(x_range, scaled_values, alpha=global_alpha, color='#CC7700', label="Global: In-Concept")
                except:
                    # Fallback to histogram if KDE fails
                    pass
        
        # Now plot sample distributions as counts on the second axis
        # First create the twin axis here so we can plot on it
        ax2 = ax.twinx()
        
        # Set count axis limit if provided
        if count_max is not None:
            ax2.set_ylim(0, count_max)
        
        # Determine common bin edges for both histograms
        all_sample_acts = []
        if len(negative_acts) > 0:
            all_sample_acts.extend(negative_acts)
        if len(positive_acts) > 0:
            all_sample_acts.extend(positive_acts)
        
        if len(all_sample_acts) > 0:
            # Create common bins for both histograms
            bin_edges = np.linspace(np.min(all_sample_acts), np.max(all_sample_acts), n_bins + 1)
            
            if len(negative_acts) > 0:
                # Plot counts on twin axis
                n_neg, bins_neg, _ = ax2.hist(negative_acts, bins=bin_edges, density=False, alpha=alpha_negative,
                        color=color_negative, edgecolor="#2d2d2d", 
                        label="Sample: Out-of-Concept")
            
            if len(positive_acts) > 0:
                # Plot counts on twin axis
                n_pos, bins_pos, _ = ax2.hist(positive_acts, bins=bin_edges, density=False, alpha=alpha_positive,
                        color=color_positive, edgecolor="none",
                        label="Sample: In-Concept")
    
    
    # Add threshold line
    if threshold_value is not None:
        ax.axvline(threshold_value, color="purple", linestyle="--", linewidth=2, 
                   label=f"Threshold (95th %ile)")
    
    # Add optimal threshold from calibration set
    best_threshold_file = None
    if concept_type == "avg_patch_embeddings":
        best_threshold_file = f"Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}.pt"
    elif concept_type == "linsep_patch_embeddings_BD_True_BN_False":
        best_threshold_file = f"Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}.pt"
    
    if best_threshold_file and os.path.exists(best_threshold_file):
        try:
            best_thresholds_data = torch.load(best_threshold_file, weights_only=False)
            if concept_name in best_thresholds_data:
                best_threshold = best_thresholds_data[concept_name].get('best_threshold')
                if best_threshold is not None:
                    ax.axvline(best_threshold, color="#00ff00", linestyle=(0, (5, 5)), linewidth=2.33)
                    
                    # Add text label to the right of the line
                    # Get current y-axis limits to position text vertically
                    y_min, y_max = ax.get_ylim()
                    y_pos = y_min + (y_max - y_min) * 0.88  # Position at 88% height
                    
                    # Add text with smaller background highlight
                    text = ax.text(best_threshold + 0.10, y_pos, "SuperActivator", 
                           color="black", fontsize=legendsize + 1, 
                           verticalalignment='center', horizontalalignment='left',
                           weight='bold', 
                           bbox=dict(boxstyle="round,pad=0.15", facecolor="#00ff00", alpha=0.25, edgecolor="none"))
                    
                    print(f"Added optimal threshold from calibration: {best_threshold:.4f}")
        except Exception as e:
            print(f"Warning: Could not load best threshold: {e}")
    
    # Compute statistics
    statistics = {}
    if len(positive_acts) > 0:
        statistics['positive'] = {
            'mean': np.mean(positive_acts),
            'std': np.std(positive_acts),
            'min': np.min(positive_acts),
            'max': np.max(positive_acts),
            'count': len(positive_acts)
        }
    else:
        statistics['positive'] = {'count': 0}
    
    if len(negative_acts) > 0:
        statistics['negative'] = {
            'mean': np.mean(negative_acts),
            'std': np.std(negative_acts),
            'min': np.min(negative_acts),
            'max': np.max(negative_acts),
            'count': len(negative_acts)
        }
    else:
        statistics['negative'] = {'count': 0}
    
    # Compute separation metrics if both distributions exist
    if len(positive_acts) > 0 and len(negative_acts) > 0:
        # Cohen's d
        pooled_std = np.sqrt((statistics['positive']['std']**2 + statistics['negative']['std']**2) / 2)
        cohens_d = (statistics['positive']['mean'] - statistics['negative']['mean']) / pooled_std if pooled_std > 0 else 0
        
        # Compute overlap
        hist_pos, bin_edges = np.histogram(positive_acts, bins=n_bins, density=True)
        hist_neg, _ = np.histogram(negative_acts, bins=bin_edges, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        overlap_area = np.sum(np.minimum(hist_pos, hist_neg)) * bin_width
        
        statistics['separation'] = {
            'cohens_d': cohens_d,
            'overlap_area': overlap_area,
            'separability': max(0, 1 - overlap_area)
        }
    
    # Add statistics to plot if requested
    if show_stats and len(positive_acts) > 0 and len(negative_acts) > 0:
        stats_text = f"Positive: μ={statistics['positive']['mean']:.3f}, σ={statistics['positive']['std']:.3f}\n"
        stats_text += f"Negative: μ={statistics['negative']['mean']:.3f}, σ={statistics['negative']['std']:.3f}\n"
        if "separation" in statistics:
            stats_text += f"Cohen's d: {statistics['separation']['cohens_d']:.2f}\n"
            stats_text += f"Separability: {statistics['separation']['separability']:.2f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment="top", fontsize=fontsize,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    # If we haven't created ax2 yet (for non-histogram plot types), create it now
    if 'ax2' not in locals():
        ax2 = ax.twinx()
    
    # Labels and formatting
    ax.set_xlabel("Concept Activation (s)", fontsize=fontsize)
    ax.set_ylabel("Frequency", fontsize=fontsize)
    
    # Set x-axis ticks every 2.0 including negative values
    x_min, x_max = ax.get_xlim()
    # Start from the nearest multiple of 2 below x_min
    x_start = np.floor(x_min / 2.0) * 2.0
    x_ticks = np.arange(x_start, x_max + 2.0, 2.0)
    ax.set_xticks(x_ticks)
    
    # Get legend handles and labels from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    # Combine handles and labels
    handles = handles1 + handles2
    labels = labels1 + labels2
    
    # Create desired order: Sample on top row, Global on bottom row
    # Swapped positions of In-Concept Sample and Out-of-Concept Global
    desired_order = ['Sample: Out-of-Concept', 'Global: Out-of-Concept', 'Sample: In-Concept', 'Global: In-Concept']
    
    # Reorder handles and labels
    ordered_handles = []
    ordered_labels = []
    
    for desired_label in desired_order:
        if desired_label in labels:
            idx = labels.index(desired_label)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])
    
    # Create legend above the plot in two rows
    # Sample distributions on top row, Global distributions on bottom row
    legend = ax.legend(ordered_handles, ordered_labels, 
                      loc='upper center', bbox_to_anchor=(0.5, 1.30),
                      ncol=2, fontsize=legendsize, markerscale=0.67,
                      frameon=True, columnspacing=2.5, handletextpad=0.8,
                      edgecolor='darkgray', fancybox=True, handlelength=2.0)
    
    # Set x-axis ticks with smaller fontsize (2 points smaller)
    ax.tick_params(axis='x', labelsize=fontsize - 2)
    
    # Hide y-axis ticks on the left
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Hide the right y-axis completely
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    
    # Ensure both y-axes start at 0
    ax.set_ylim(bottom=0)
    
    # For ax2, respect count_max if provided, otherwise just set bottom to 0
    if count_max is not None:
        ax2.set_ylim(0, count_max)
    else:
        ax2.set_ylim(bottom=0)
    
    # Set x-axis limits if provided
    if x_lim is not None:
        ax.set_xlim(x_lim)
    
    plt.tight_layout()
    # Adjust layout to make room for legend above plot and move bottom elements down
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    # Save if requested
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:  # Only create directory if there is one
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # Return results
    return {
        'positive_activations': positive_acts,
        'negative_activations': negative_acts,
        'statistics': statistics,
        'threshold': threshold_value,
        'sample_index': sample_index,
        'concept_name': concept_name,
        'n_positive_patches': len(image_positive_patches),
        'n_negative_patches': patches_per_image - len(image_positive_patches)
    }


def plot_combined_distributions_and_gt_mass(
    computation_results: Dict[str, Any],
    concepts: Optional[List[str]] = None,
    percentthrumodels: Optional[List[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    # Font sizes
    header_size: int = 14,
    label_size: int = 12,
    concept_size: int = 12,
    legend_size: int = 10,
    # Grid plot specific
    max_concepts: int = 10,
    n_bins: int = 50,
    alpha_bg: float = 0.6,
    alpha_gt: float = 0.7,
    show_legend_grid: bool = True,
    show_thresholds: bool = True,
    plot_type: str = 'density',
    x_padding_fraction: float = 0.0,
    concept_xlims: Optional[Tuple[float, float]] = None,
    # GT mass plot specific
    use_layer_numbers: bool = False,
    custom_ylabel_mass: Optional[str] = None,
    custom_title_mass: Optional[str] = None,
    ylim_mass: Optional[Tuple[float, float]] = None,
    show_error_bars: bool = True,
    plot_individual_concepts_mass: bool = False,
    highlight_percentile: bool = True,
    overlap_framing: bool = False,
    legend_text_mass: Optional[str] = None,
    all_concepts: bool = False
) -> None:
    """
    Create a combined figure with activation distributions grid on the left and GT mass plot on the right.
    
    Args:
        computation_results: Results from compute_separation_over_percentthru
        concepts: List of specific concepts to plot (overrides max_concepts)
        percentthrumodels: List of percentthrumodel values to plot
        figsize: Figure size for the entire combined plot
        save_path: Path to save the combined figure
        show_plot: Whether to display the plot
        
        # Font sizes
        header_size: Font size for the two "% Through Model" headers
        label_size: Font size for all axis labels, tick labels, and percentages
        concept_size: Font size for concept names on the left
        legend_size: Font size for legends
        
        # Grid plot specific arguments
        max_concepts: Maximum number of concepts to display (ignored if concepts is provided)
        n_bins: Number of bins for histograms
        alpha_bg: Alpha value for background histogram
        alpha_gt: Alpha value for GT positive histogram
        show_legend_grid: Whether to show legend on grid plot
        show_thresholds: Whether to show detection thresholds
        plot_type: 'density' for KDE smooth curves or 'histogram' for traditional histograms
        x_padding_fraction: Fraction of data range to pad on each side
        concept_xlims: Optional tuple (min, max) to set x-axis limits for all histogram plots
        
        # GT mass plot specific arguments
        use_layer_numbers: Whether to use layer numbers on x-axis (vs percentthrumodel)
        custom_ylabel_mass: Custom y-axis label for GT mass plot
        custom_title_mass: Custom title for GT mass plot (if "", no title)
        ylim_mass: Y-axis limits for GT mass plot
        show_error_bars: Whether to show error bars for averaged GT mass
        plot_individual_concepts_mass: Whether to plot individual concept lines
        highlight_percentile: Whether to show the background percentile used in title
        overlap_framing: If True, plots (1 - fraction) to show overlap instead of separation
        legend_text_mass: Optional text to display in legend for GT mass plot
        all_concepts: If True, plot all concepts as transparent lines with only 'Avg' in legend
    """
    # Apply paper plotting style
    apply_paper_plotting_style()
    
    # Extract metadata to determine grid dimensions
    metadata = computation_results['metadata']
    
    # Use provided concepts or default to metadata concepts for histograms
    if concepts is None:
        concepts_hist = metadata['concepts'][:max_concepts]
    else:
        # Validate that requested concepts exist
        available_concepts = metadata['concepts']
        concepts_hist = [c for c in concepts if c in available_concepts]
        if not concepts_hist:
            raise ValueError(f"None of the requested concepts found in results")
    
    # Always use ALL concepts from metadata for GT mass plot
    concepts_mass = metadata['concepts']
    
    # Use provided percentthrumodels for histograms or default to all available
    if percentthrumodels is None or len(percentthrumodels) == 0:
        percentthrumodels_hist = metadata['percentthrumodels']
    else:
        percentthrumodels_hist = percentthrumodels
    
    # Always use all default percentthrumodels for GT mass plot
    percentthrumodels_mass = metadata['percentthrumodels']
    
    n_rows = len(concepts_hist)
    n_cols = len(percentthrumodels_hist)
    
    # Use provided figsize or set a default
    if figsize is None:
        figsize = (12, 8)  # Default size
    
    # Create figure with GridSpec for precise control
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=figsize)
    
    # Create main grid with 1 row and 2 columns
    # Set width ratios to 70% for left plot, 30% for right plot
    gs_main = gridspec.GridSpec(1, 2, figure=fig, 
                                width_ratios=[7, 3],  # 70%, 30%
                                wspace=0.22)  # Slightly reduced space between the two plots
    
    # Create sub-gridspec for the activation distributions grid
    gs_grid = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, 
                                               subplot_spec=gs_main[0],
                                               hspace=0.15, wspace=0.1)
    
    # Create axes for grid plot
    axes_grid = []
    for i in range(n_rows):
        row_axes = []
        for j in range(n_cols):
            ax = fig.add_subplot(gs_grid[i, j])
            row_axes.append(ax)
        axes_grid.append(row_axes)
    axes_grid = np.array(axes_grid)
    
    # Create axis for GT mass plot
    ax_mass = fig.add_subplot(gs_main[1])
    
    # Note: All font sizes are now explicitly set by the user
    
    # First, plot the activation distributions grid
    # We'll implement the grid plotting logic here rather than calling the function
    # to have full control over the axes
    
    # Create color mapping for concepts (red, orange, pink colors)
    # Define specific colors for up to the first few concepts
    warm_colors = ['red', 'darkorange', 'hotpink', 'crimson', 'coral', 'deeppink']
    
    # Use the defined colors for concepts, cycling if needed
    concept_colors = []
    for i in range(len(concepts_hist)):
        concept_colors.append(warm_colors[i % len(warm_colors)])
    
    concept_color_map = {concept: concept_colors[i] for i, concept in enumerate(concepts_hist)}
    
    # Track x-axis limits for alignment
    global_x_min = float('inf')
    global_x_max = float('-inf')
    
    # First pass: collect min/max across ALL concepts and layers
    for concept_name in concepts_hist:
        for percent_thru_model in percentthrumodels_hist:
            key = f"{concept_name}_{percent_thru_model}"
            if key in computation_results['results'] and 'error' not in computation_results['results'][key]:
                result_data = computation_results['results'][key]
                gt_positive_acts = result_data.get('gt_positive_acts', np.array([]))
                background_acts = result_data.get('background_acts', np.array([]))
                
                if len(gt_positive_acts) > 0 and len(background_acts) > 0:
                    all_acts = np.concatenate([gt_positive_acts, background_acts])
                    global_x_min = min(global_x_min, np.min(all_acts))
                    global_x_max = max(global_x_max, np.max(all_acts))
    
    # Apply padding to global limits
    if global_x_min != float('inf') and global_x_max != float('-inf'):
        if x_padding_fraction > 0:
            x_range = global_x_max - global_x_min
            x_padding = x_padding_fraction * x_range
            global_x_limits = (global_x_min - x_padding, global_x_max + x_padding)
        else:
            global_x_limits = (global_x_min, global_x_max)
    else:
        global_x_limits = None
    
    # Plot each cell in the grid
    for row_idx, concept_name in enumerate(concepts_hist):
        for col_idx, percent_thru_model in enumerate(percentthrumodels_hist):
            ax = axes_grid[row_idx, col_idx]
            
            key = f"{concept_name}_{percent_thru_model}"
            if key not in computation_results['results']:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=label_size)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            result_data = computation_results['results'][key]
            
            if 'error' in result_data:
                ax.text(0.5, 0.5, result_data['error'], ha='center', va='center', 
                       transform=ax.transAxes, fontsize=label_size, wrap=True)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            gt_positive_acts = result_data.get('gt_positive_acts', np.array([]))
            background_acts = result_data.get('background_acts', np.array([]))
            
            if len(gt_positive_acts) > 0 and len(background_acts) > 0:
                # Use concept_xlims if provided, otherwise global x-axis limits for alignment
                if concept_xlims is not None:
                    x_min, x_max = concept_xlims
                elif global_x_limits is not None:
                    x_min, x_max = global_x_limits
                else:
                    all_acts = np.concatenate([background_acts, gt_positive_acts])
                    x_min, x_max = np.min(all_acts), np.max(all_acts)
                
                if plot_type == 'density':
                    from scipy.stats import gaussian_kde
                    x_smooth = np.linspace(x_min, x_max, 500)
                    
                    kde_bg = gaussian_kde(background_acts)
                    y_bg = kde_bg(x_smooth)
                    ax.fill_between(x_smooth, y_bg, alpha=alpha_bg, color='#505050', 
                                   label=f'Out-of-Concept ({len(background_acts)})')
                    
                    kde_gt = gaussian_kde(gt_positive_acts)
                    y_gt = kde_gt(x_smooth)
                    ax.fill_between(x_smooth, y_gt, alpha=alpha_gt, color='#FF9500',
                                   label=f'In-Concept ({len(gt_positive_acts)})')
                else:  # histogram
                    bins = np.linspace(x_min, x_max, n_bins)
                    ax.hist(background_acts, bins=bins, density=True, alpha=alpha_bg,
                           color='#505050', label=f'Out-of-Concept ({len(background_acts)})')
                    ax.hist(gt_positive_acts, bins=bins, density=True, alpha=alpha_gt,
                           color='#FF9500', label=f'In-Concept ({len(gt_positive_acts)})')
                
                # Add threshold if available
                if show_thresholds and result_data.get('threshold') is not None:
                    ax.axvline(result_data['threshold'], color='purple', linestyle='-', 
                             linewidth=2, label='Thresh')
                
                # Formatting
                if row_idx == 0:
                    ax.set_title(f'{percent_thru_model}%', fontsize=label_size, pad=3)
                
                if col_idx == 0:
                    display_name = concept_name.split("::")[-1] if "::" in concept_name else concept_name
                    concept_label = display_name.capitalize() if len(display_name) <= 20 else f'{display_name[:20].capitalize()}...'
                    
                    # Add concept label text without underlining
                    text_obj = ax.text(-0.15, 0.5, concept_label, transform=ax.transAxes,
                                      verticalalignment='center', horizontalalignment='right', 
                                      fontsize=concept_size, rotation=0, style='italic')
                
                # Apply x-limits first - use concept_xlims if provided, otherwise global_x_limits
                if concept_xlims is not None:
                    ax.set_xlim(concept_xlims)
                    ax.margins(x=0)
                    x_min, x_max = concept_xlims
                elif global_x_limits is not None:
                    ax.set_xlim(global_x_limits)
                    ax.margins(x=0)
                    x_min, x_max = global_x_limits
                
                # Now set x ticks at intervals of 2 based on the actual limits
                # Round to nearest even number for cleaner ticks
                x_start = int(np.ceil(x_min / 2) * 2)
                x_end = int(np.floor(x_max / 2) * 2)
                x_ticks_grid = np.arange(x_start, x_end + 1, 2)
                # Ensure we have at least some ticks
                if len(x_ticks_grid) == 0:
                    x_ticks_grid = np.array([int(np.round((x_min + x_max) / 2))])
                ax.set_xticks(x_ticks_grid)
                
                # X-axis formatting - straddle ticks, labels only on bottom row
                if row_idx == n_rows - 1:
                    ax.tick_params(axis='x', bottom=True, labelbottom=True, length=4, 
                                  labelsize=label_size, labelcolor='black', direction='inout')
                else:
                    ax.tick_params(axis='x', bottom=True, labelbottom=False, length=4, direction='inout')
                
                # Only show x-label on bottom row
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Act (s)", fontsize=label_size, labelpad=2)
                
                # For leftmost plots, add a rho symbol at the top
                if col_idx == 0:
                    # Add rho symbol at the top of the y-axis
                    ax.text(-0.11, 1.02, r"$\rho$", transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='center',
                           fontsize=label_size, rotation=0)
                
                ax.tick_params(axis='y', left=False, labelleft=False)
                ax.grid(False)
                
                # Set y-axis to start at 0
                ax.set_ylim(bottom=0)
            else:
                error_msg = 'No GT+' if len(gt_positive_acts) == 0 else 'No Bg'
                ax.text(0.5, 0.5, error_msg, ha='center', va='center', 
                       transform=ax.transAxes, fontsize=label_size)
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Add legend for grid plot at the bottom
    if show_legend_grid:
        # Create dummy plot to get legend handles
        dummy_ax = fig.add_subplot(111, frameon=False)
        dummy_ax.hist([], alpha=alpha_bg, color='#505050', edgecolor='black', label='Out-of-Concept Tokens')
        dummy_ax.hist([], alpha=alpha_gt, color='#FF9500', edgecolor='#CC7700', label='In-Concept Tokens')
        if show_thresholds:
            dummy_ax.axvline(0, color='purple', linestyle='-', linewidth=2, label='Threshold')
        
        handles, labels = dummy_ax.get_legend_handles_labels()
        # Position legend below the grid, centered over just the plots
        # Move it down more by using a fixed negative value
        legend_y = -0.08  # Fixed position further below the figure
        fig.legend(handles, labels, loc='upper center', ncol=len(handles), 
                  fontsize=legend_size, 
                  bbox_to_anchor=(0.33, legend_y), bbox_transform=fig.transFigure)
        dummy_ax.set_visible(False)
    
    # Add title for grid section (centered over just the plots, not including concept labels)
    # The grid takes 70% of width, but concept labels extend to the left
    # Estimate: concept labels take about 10% of the 70%, so plots are from ~7% to 70%
    # Center of plots: (7% + 70%) / 2 = 38.5%, adjust to ~35%, then left by 0.05 to 0.30, then right by 0.03 to 0.33
    fig.text(0.33, 1.015, '% Through Model', ha='center', va='bottom', 
             fontsize=header_size, transform=fig.transFigure)
    
    # Add title for overlap section (centered over the right plot)
    # Right plot is from 70% to 100%, center is at 85%, then left by 0.05 to 0.80, then right by 0.03 to 0.83
    fig.text(0.835, 1.015, '% Through Model', ha='center', va='bottom', 
             fontsize=header_size, transform=fig.transFigure)
    
    # Now plot the overlap metrics instead of GT mass
    # We'll compute overlap for selected concepts and average across ALL concepts
    
    # Compute overlap metrics for all concepts
    overlap_results = {}  # Will store overlap for each concept
    
    # Process all concepts in the metadata (not just selected ones)
    for concept_name in concepts_mass:
        overlap_per_layer = []
        
        for percent_thru_model in percentthrumodels_mass:
            key = f"{concept_name}_{percent_thru_model}"
            
            if key in computation_results['results'] and 'error' not in computation_results['results'][key]:
                result_data = computation_results['results'][key]
                gt_positive_acts = result_data.get('gt_positive_acts', np.array([]))
                background_acts = result_data.get('background_acts', np.array([]))
                
                if len(gt_positive_acts) > 0 and len(background_acts) > 0:
                    # Compute overlap as probability mass of union between distributions
                    # First, determine the range for histogram bins
                    all_acts = np.concatenate([gt_positive_acts, background_acts])
                    min_val, max_val = np.min(all_acts), np.max(all_acts)
                    
                    # Create histogram bins
                    n_overlap_bins = 100  # Number of bins for overlap calculation
                    bins = np.linspace(min_val, max_val, n_overlap_bins + 1)
                    
                    # Compute normalized histograms (probability density)
                    hist_gt, _ = np.histogram(gt_positive_acts, bins=bins, density=True)
                    hist_bg, _ = np.histogram(background_acts, bins=bins, density=True)
                    
                    # Convert to probability mass by multiplying by bin width
                    bin_width = bins[1] - bins[0]
                    prob_gt = hist_gt * bin_width
                    prob_bg = hist_bg * bin_width
                    
                    # Compute overlap as the minimum probability at each bin (intersection)
                    # The overlap is the sum of minimum probabilities across all bins
                    overlap_prob_mass = np.sum(np.minimum(prob_gt, prob_bg))
                    
                    # Debug: Check probability mass sums
                    total_prob_gt = np.sum(prob_gt)
                    total_prob_bg = np.sum(prob_bg)
                    
                    # Comment out debug output now that we've verified it's working
                    # if len(overlap_results) < 2 and len(overlap_per_layer) < 2:
                    #     print(f"\nDEBUG - Overlap calculation for {concept_name} at {percent_thru_model}%:")
                    #     print(f"  GT samples: {len(gt_positive_acts)}, BG samples: {len(background_acts)}")
                    #     print(f"  Value range: [{min_val:.3f}, {max_val:.3f}]")
                    #     print(f"  GT mean: {np.mean(gt_positive_acts):.3f}, std: {np.std(gt_positive_acts):.3f}")
                    #     print(f"  BG mean: {np.mean(background_acts):.3f}, std: {np.std(background_acts):.3f}")
                    #     print(f"  Total prob mass GT: {total_prob_gt:.3f} (should be ~1.0)")
                    #     print(f"  Total prob mass BG: {total_prob_bg:.3f} (should be ~1.0)")
                    #     print(f"  Overlap prob mass: {overlap_prob_mass:.3f}")
                    #     print(f"  Non-zero overlap bins: {np.sum(np.minimum(prob_gt, prob_bg) > 0)} / {n_overlap_bins}")
                    
                    # Ensure overlap is between 0 and 1
                    overlap_prob_mass = np.clip(overlap_prob_mass, 0, 1)
                    
                    overlap_per_layer.append(overlap_prob_mass)
                else:
                    overlap_per_layer.append(np.nan)
            else:
                overlap_per_layer.append(np.nan)
        
        if overlap_per_layer and not all(np.isnan(overlap_per_layer)):
            overlap_results[concept_name] = overlap_per_layer
    
    # Compute average overlap across ALL concepts
    averaged_overlap = []
    for i in range(len(percentthrumodels_mass)):
        values_at_layer = []
        for concept_name in overlap_results:
            if i < len(overlap_results[concept_name]):
                val = overlap_results[concept_name][i]
                if not np.isnan(val):
                    values_at_layer.append(val)
        
        if values_at_layer:
            averaged_overlap.append(np.mean(values_at_layer))
        else:
            averaged_overlap.append(np.nan)
    
    print(f"Computed overlap for {len(overlap_results)} concepts")
    print(f"Average overlap values: {[f'{v:.3f}' if not np.isnan(v) else 'nan' for v in averaged_overlap]}")
    
    # Print overlap values for the selected concepts
    print("\nOverlap values for selected concepts:")
    print(f"Percentthrumodels: {percentthrumodels_mass}")
    for concept in concepts_hist:
        if concept in overlap_results:
            values = overlap_results[concept]
            formatted_values = [f'{v:.3f}' if not np.isnan(v) else 'nan' for v in values]
            print(f"{concept}: {formatted_values}")
    
    # Plot on the mass axis
    # Move x-axis to top
    ax_mass.xaxis.tick_top()
    ax_mass.xaxis.set_label_position('top')
    
    # X-axis values for GT mass plot (always use all default percentthrumodels)
    if use_layer_numbers:
        # Get total layers for conversion
        model_name = metadata['model_name']
        is_text_dataset = metadata['is_text_dataset']
        if model_name == 'CLIP':
            total_layers = 24
        elif model_name == 'Llama' and is_text_dataset:
            total_layers = 32
        elif model_name == 'Llama' and not is_text_dataset:
            total_layers = 40
        else:
            total_layers = 24
        
        x_values = [percent_to_layer(p, total_layers, model_name) for p in percentthrumodels_mass]
        x_label = 'Layer'
    else:
        x_values = percentthrumodels_mass
        x_label = '% Through Model'
    
    # Plot individual concepts' overlap
    if all_concepts and overlap_results:
        # Plot ALL concepts as transparent light pink lines without markers
        first_concept = True
        for concept in overlap_results:
            plot_values = overlap_results[concept]
            
            # Handle NaN values by interpolating or skipping
            valid_indices = [j for j, v in enumerate(plot_values) if not np.isnan(v)]
            if valid_indices:
                valid_x = [x_values[j] for j in valid_indices]
                valid_y = [plot_values[j] for j in valid_indices]
                
                # Add label only for the first concept
                label = 'Per-Concept' if first_concept else None
                first_concept = False
                
                ax_mass.plot(valid_x, valid_y,
                            color='lightpink',
                            alpha=0.3, linewidth=1, 
                            marker=None,
                            label=label)
    elif overlap_results and concepts_hist:
        # Plot only selected concepts from concepts_hist with colors and markers
        # Use the same warm colors (red, orange, pink) as the concept labels
        warm_colors = ['red', 'darkorange', 'hotpink', 'crimson', 'coral', 'deeppink']
        concept_colors = [warm_colors[i % len(warm_colors)] for i in range(len(concepts_hist))]
        for i, concept in enumerate(concepts_hist):
            if concept in overlap_results:
                plot_values = overlap_results[concept]
                
                # Handle NaN values by interpolating or skipping
                valid_indices = [j for j, v in enumerate(plot_values) if not np.isnan(v)]
                if valid_indices:
                    valid_x = [x_values[j] for j in valid_indices]
                    valid_y = [plot_values[j] for j in valid_indices]
                    
                    # Remove "material::" prefix from concept name for legend and capitalize
                    display_name = concept.replace("material::", "") if concept.startswith("material::") else concept
                    display_name = display_name.capitalize()
                    
                    ax_mass.plot(valid_x, valid_y,
                                label=display_name,
                                color=concept_colors[i],
                                alpha=0.8, linewidth=1.5, 
                                marker='o', markersize=2)
    
    # Plot average overlap across ALL concepts
    if averaged_overlap and len(averaged_overlap) == len(x_values):
        # Handle NaN values
        valid_indices = [j for j, v in enumerate(averaged_overlap) if not np.isnan(v)]
        if valid_indices:
            valid_x = [x_values[j] for j in valid_indices]
            valid_y = [averaged_overlap[j] for j in valid_indices]
            
            if show_error_bars and len(overlap_results) > 1 and not all_concepts:
                # Calculate standard deviation across all concepts
                stds = []
                for i in range(len(percentthrumodels_mass)):
                    values_at_layer = []
                    for concept in overlap_results:
                        if i < len(overlap_results[concept]):
                            val = overlap_results[concept][i]
                            if not np.isnan(val):
                                values_at_layer.append(val)
                    if values_at_layer:
                        stds.append(np.std(values_at_layer))
                    else:
                        stds.append(np.nan)
                
                # Filter out NaN error values
                valid_errors = [stds[j] / np.sqrt(len(overlap_results)) for j in valid_indices]
                
                avg_label = 'Average' if all_concepts else f'Avg ({len(overlap_results)} concepts)'
                ax_mass.errorbar(valid_x, valid_y, yerr=valid_errors,
                                label=avg_label,
                                linestyle='-',
                                color='darkviolet',
                                capsize=3, alpha=0.9, linewidth=2.5,
                                marker='o', markersize=5)
            else:
                avg_label = 'Average' if all_concepts else f'Avg ({len(overlap_results)} concepts)'
                ax_mass.plot(valid_x, valid_y,
                            label=avg_label,
                            linestyle='-',
                            color='darkviolet',
                            linewidth=2.5,
                            marker='o', markersize=5)
    
    # Formatting - no x-axis label since we have the header
    # ax_mass.set_xlabel(x_label, fontsize=label_size, labelpad=10)  # Commented out to avoid duplicate
    
    if custom_ylabel_mass == "":
        pass
    elif custom_ylabel_mass:
        ax_mass.set_ylabel(custom_ylabel_mass, fontsize=label_size, labelpad=2)
    else:
        ax_mass.set_ylabel('Distribution Overlap\n(Probability Mass)', fontsize=label_size, labelpad=2)
    
    # Title
    if custom_title_mass == "":
        pass
    elif custom_title_mass:
        ax_mass.set_title(custom_title_mass, fontsize=label_size + 2, pad=20)
    else:
        # No default title
        pass
    
    # Y-axis
    if ylim_mass:
        ax_mass.set_ylim(ylim_mass)
    else:
        ax_mass.set_ylim(0, 1.05)
    
    # Set tick parameters - outside ticks only for right plot
    # Use small padding to bring labels down closer to plot
    ax_mass.tick_params(axis='x', which='major', labelsize=label_size, length=3,
                       pad=2, labelcolor='black', direction='out')
    ax_mass.tick_params(axis='y', which='major', labelsize=label_size, length=3,
                       pad=2, labelcolor='black', direction='out')
    # Set y-axis ticks at 0.2 intervals with one decimal place
    ax_mass.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_mass.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    
    # X-axis ticks
    if not use_layer_numbers:
        ax_mass.set_xticks([0, 25, 50, 75, 100])
        ax_mass.set_xlim(0, 100)  # Set exact limits
        ax_mass.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    ax_mass.grid(True, alpha=0.3)
    
    # Always add legend to show what lines are plotted
    handles, labels = ax_mass.get_legend_handles_labels()
    if handles:  # Only add legend if there are items to show
        # Separate individual concepts from average
        concept_handles = []
        concept_labels = []
        avg_handle = None
        avg_label = None
        per_concept_handle = None
        per_concept_label = None
        
        for h, l in zip(handles, labels):
            if l == 'Average' or ('Avg (' in l and 'concepts)' in l):
                avg_handle = h
                avg_label = l
            elif l == 'Per-Concept':
                # Create a thicker line for the legend while keeping plot lines thin
                from matplotlib.lines import Line2D
                per_concept_handle = Line2D([0], [0], color='lightpink', 
                                          linewidth=3, alpha=0.8)
                per_concept_label = l
            else:
                concept_handles.append(h)
                concept_labels.append(l)
        
        # Build final handles and labels lists
        final_handles = []
        final_labels = []
        
        # Add per-concept if it exists
        if per_concept_handle is not None:
            final_handles.append(per_concept_handle)
            final_labels.append(per_concept_label)
        
        # Add individual concepts
        final_handles.extend(concept_handles)
        final_labels.extend(concept_labels)
        
        # Add average last
        if avg_handle is not None:
            final_handles.append(avg_handle)
            final_labels.append(avg_label)
        
        # Create single legend with all items
        if final_handles:
            ax_mass.legend(final_handles, final_labels, 
                          loc='upper right',
                          ncol=1,
                          fontsize=legend_size,
                          frameon=True, 
                          fancybox=True, 
                          framealpha=0.8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure right plot aligns with grid - get positions of top and bottom grid plots
    if axes_grid.size > 0:
        top_ax = axes_grid[0, 0]
        bottom_ax = axes_grid[-1, 0]
        
        # Get the y-positions of the grid
        top_pos = top_ax.get_position()
        bottom_pos = bottom_ax.get_position()
        
        # Get and adjust the mass plot position
        mass_pos = ax_mass.get_position()
        
        # Get the title position of the top left plot to align with
        # The title is above the plot, so we need to leave space for it
        title_height = 0.03  # Approximate height needed for title
        
        # Set the mass plot to align with the actual plot area of the grid (not including titles)
        # Lower the top by 0.025
        ax_mass.set_position([mass_pos.x0, bottom_pos.y0, 
                             mass_pos.width, top_pos.y1 - bottom_pos.y0 - 0.025])
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def compute_overlap_across_layers(
    datasets: List[str],
    models: List[str],
    concept_types: List[str] = ['avg', 'linsep'],
    sample_type: str = 'patch',
    save_path: Optional[str] = None,
    percentthrus: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 8),
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
    scratch_dir: str = ''
) -> Dict:
    """
    Compute and plot probability mass overlap across layers for multiple datasets and models.
    
    Args:
        datasets: List of dataset names (e.g., ['CLEVR', 'Coco', 'Sarcasm'])
        models: List of model names (e.g., ['ViT-B-16', 'Llama'])
        concept_types: List of concept types to analyze (default: ['avg', 'linsep'])
        sample_type: Type of samples ('patch' or 'cls') - use 'patch' for patch/token-level analysis, 'cls' for CLS token only
                     Note: For text datasets, 'patch' refers to token-level analysis
        save_path: Path to save the figure (optional)
        percentthrus: List of percentthru values to analyze (default: all available)
        figsize: Figure size tuple
        colors: Dict mapping dataset names to colors (optional)
        linestyles: Dict mapping concept types to line styles (optional)
        scratch_dir: Directory containing activation files (default: '')
        
    Returns:
        Dict containing overlap results for each configuration
    """
    import seaborn as sns
    from pathlib import Path
    from utils.filter_datasets_utils import DATASET_TO_CONCEPTS
    import warnings
    
    # Import the function to get model-specific percentthrumodels
    from utils.default_percentthrumodels import get_model_default_percentthrumodels
    
    # Default percentthrus if not specified - will be set per model later
    if percentthrus is None:
        use_model_defaults = True
    else:
        use_model_defaults = False
    
    # Default colors for datasets if not specified
    if colors is None:
        color_palette = sns.color_palette('husl', len(datasets))
        colors = {ds: color_palette[i] for i, ds in enumerate(datasets)}
    
    # Default line styles for concept types if not specified
    if linestyles is None:
        linestyles = {
            'avg': '-',
            'linsep': '--',
            'kmeans': ':'
        }
    
    results = {}
    
    # Set up the plot
    fig, axes = plt.subplots(1, len(models), figsize=figsize, squeeze=False)
    axes = axes[0]
    
    # Process each model
    for model_idx, model in enumerate(models):
        ax = axes[model_idx]
        
        # Process each dataset
        for dataset in datasets:
            # Get valid concepts for this dataset
            valid_concepts = DATASET_TO_CONCEPTS.get(dataset, [])
            
            # Get dataset-specific percentthrus if using defaults
            if use_model_defaults:
                # Determine model input size for this specific dataset
                is_text_dataset = dataset in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'IMDB', 'iSarcasmEval', 'Stanford-Tree-Bank', 'jailbreak']
                
                if model in ['ViT-B-16', 'CLIP']:
                    model_input_size_for_ptm = (224, 224)
                elif model == 'Llama':
                    # Use appropriate percentthrus based on THIS dataset, not all datasets
                    if is_text_dataset:
                        model_input_size_for_ptm = ('text', 'text')
                    else:
                        model_input_size_for_ptm = (560, 560)
                elif model == 'Gemma':
                    model_input_size_for_ptm = ('text', 'text2')
                elif model == 'Qwen':
                    model_input_size_for_ptm = ('text', 'text3')
                else:
                    model_input_size_for_ptm = (224, 224)
                
                dataset_percentthrus = get_model_default_percentthrumodels(model, model_input_size_for_ptm)
            else:
                dataset_percentthrus = percentthrus
            
            # Process each concept type
            for concept_type in concept_types:
                overlaps_by_layer = []
                
                # Process each layer (percentthru)
                for percentthru in tqdm(dataset_percentthrus, desc=f"{model} {dataset} {concept_type}"):
                    # Determine activation file name based on concept type
                    if concept_type == 'avg':
                        acts_filename = f"cosine_similarities_avg_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
                        acts_dir = os.path.join(scratch_dir, 'Cosine_Similarities', dataset)
                    elif concept_type == 'linsep':
                        acts_filename = f"dists_linsep_concepts_BD_True_BN_False_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
                        acts_dir = os.path.join(scratch_dir, 'Distances', dataset)
                    else:
                        print(f"Warning: Unknown concept type: {concept_type}")
                        continue
                    
                    # Try to load using ChunkedActivationLoader
                    try:
                        from utils.memory_management_utils import ChunkedActivationLoader
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        act_loader = ChunkedActivationLoader(dataset, acts_filename, scratch_dir=scratch_dir, device=device)
                        
                        # Determine model input size for this model/dataset combination
                        if model in ['ViT-B-16', 'CLIP']:
                            load_model_input_size = (224, 224)
                        elif model == 'Llama':
                            if dataset in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'IMDB', 'iSarcasmEval']:
                                load_model_input_size = ('text', 'text')
                            else:
                                load_model_input_size = (560, 560)
                        elif model == 'Gemma':
                            load_model_input_size = ('text', 'text2')
                        elif model == 'Qwen':
                            load_model_input_size = ('text', 'text3')
                        else:
                            load_model_input_size = (224, 224)
                        
                        # Load test activations
                        test_acts = act_loader.load_split_tensor('test', dataset, model_input_size=load_model_input_size, patch_size=14)
                        if test_acts is None:
                            print(f"Warning: Could not load test activations for {dataset} {model} {concept_type} ptm={percentthru}")
                            continue
                            
                        # Load concepts to get indices
                        if concept_type == 'avg':
                            concepts_filename = f"avg_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
                        else:
                            concepts_filename = f"linsep_concepts_BD_True_BN_False_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
                        
                        concepts_file = f"Concepts/{dataset}/{concepts_filename}"
                        if not os.path.exists(concepts_file):
                            print(f"Warning: Concepts file not found: {concepts_file}")
                            continue
                            
                        # Load concepts data, suppressing warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=FutureWarning)
                            try:
                                concepts_data = torch.load(concepts_file, map_location='cpu', weights_only=True)
                            except:
                                # Fallback to weights_only=False if the file contains non-tensor objects
                                concepts_data = torch.load(concepts_file, map_location='cpu', weights_only=False)
                        all_concept_names = list(concepts_data.keys())
                        
                    except Exception as e:
                        print(f"Warning: Error loading activations for {dataset} {model} {concept_type} ptm={percentthru}: {e}")
                        continue
                    
                    # Load ground truth data
                    # Determine if this is a text dataset
                    is_text_dataset = dataset in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'IMDB', 'iSarcasmEval']
                    
                    if model in ['ViT-B-16', 'CLIP']:
                        model_input_size = (224, 224)
                        patches_per_image = 256  # 16x16
                    elif model == 'Llama':
                        if is_text_dataset:
                            model_input_size = ('text', 'text')
                            # For text, we'll handle token counting differently
                        else:
                            model_input_size = (560, 560)
                            patches_per_image = 1600  # 40x40
                    elif model == 'Gemma':
                        model_input_size = ('text', 'text2')
                        patches_per_image = None  # Variable for text
                    elif model == 'Qwen':
                        model_input_size = ('text', 'text3')
                        patches_per_image = None  # Variable for text
                    else:
                        model_input_size = (224, 224)
                        patches_per_image = 256
                    
                    # Load GT patches and samples
                    gt_patches_file = f"GT_Samples/{dataset}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
                    if not os.path.exists(gt_patches_file):
                        gt_patches_file = f"GT_Samples/{dataset}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
                    
                    gt_samples_file = f"GT_Samples/{dataset}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
                    
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=FutureWarning)
                            try:
                                gt_patches_per_concept = torch.load(gt_patches_file, map_location='cpu', weights_only=True)
                                gt_samples_per_concept = torch.load(gt_samples_file, map_location='cpu', weights_only=True)
                            except:
                                # Fallback if files contain non-tensor objects
                                gt_patches_per_concept = torch.load(gt_patches_file, map_location='cpu', weights_only=False)
                                gt_samples_per_concept = torch.load(gt_samples_file, map_location='cpu', weights_only=False)
                    except Exception as e:
                        print(f"Warning: Could not load GT data: {e}")
                        continue
                    
                    # Get test metadata
                    import pandas as pd
                    metadata = pd.read_csv(f'../Data/{dataset}/metadata.csv')
                    test_global_indices = metadata[metadata['split'] == 'test'].index.tolist()
                    test_pos_map = {idx: pos for pos, idx in enumerate(test_global_indices)}
                    
                    # Compute overlap for each concept and average
                    concept_overlaps = []
                    
                    # Get indices for all valid concepts at once
                    valid_concept_indices = []
                    valid_concept_names = []
                    for concept in valid_concepts:
                        if concept in all_concept_names:
                            valid_concept_indices.append(all_concept_names.index(concept))
                            valid_concept_names.append(concept)
                    
                    if not valid_concept_indices:
                        continue
                    
                    # Process all concepts in parallel on GPU
                    if device.type == 'cuda' and test_acts.is_cuda:
                        # Get all concept activations at once (keep on GPU)
                        all_concept_acts = test_acts[:, valid_concept_indices]  # shape: [n_patches, n_concepts]
                    else:
                        all_concept_acts = test_acts[:, valid_concept_indices]
                    
                    # Process each concept
                    for concept_idx_in_batch, concept in enumerate(valid_concept_names):
                        # Get concept activations for this concept
                        concept_acts = all_concept_acts[:, concept_idx_in_batch]
                        
                        # Get GT positive patch indices
                        positive_patch_indices = gt_patches_per_concept.get(concept, [])
                        if len(positive_patch_indices) == 0:
                            continue
                        
                        # Collect GT positive activations
                        if is_text_dataset:
                            # For text datasets, patch indices are token indices
                            # Vectorized filtering
                            positive_patch_indices_np = np.array(positive_patch_indices)
                            valid_mask = positive_patch_indices_np < len(concept_acts)
                            valid_test_indices = positive_patch_indices_np[valid_mask].tolist()
                        else:
                            # For image datasets - vectorized computation
                            positive_patch_indices_np = np.array(positive_patch_indices)
                            global_img_indices = positive_patch_indices_np // patches_per_image
                            patch_within_imgs = positive_patch_indices_np % patches_per_image
                            
                            # Vectorized mapping
                            test_positions = np.array([test_pos_map.get(idx, -1) for idx in global_img_indices])
                            valid_mask = test_positions >= 0
                            
                            if np.any(valid_mask):
                                valid_test_positions = test_positions[valid_mask]
                                valid_patch_within = patch_within_imgs[valid_mask]
                                test_patch_indices = valid_test_positions * patches_per_image + valid_patch_within
                                
                                # Final filter for bounds
                                final_valid_mask = test_patch_indices < len(concept_acts)
                                valid_test_indices = test_patch_indices[final_valid_mask].tolist()
                                
                                # Filter out padding patches using filter_patches_by_image_presence
                                from utils.patch_alignment_utils import filter_patches_by_image_presence
                                if valid_test_indices:
                                    valid_test_indices_filtered = filter_patches_by_image_presence(
                                        valid_test_indices, dataset, model_input_size
                                    )
                                    valid_test_indices = valid_test_indices_filtered.tolist()
                            else:
                                valid_test_indices = []
                        
                        if len(valid_test_indices) == 0:
                            continue
                        
                        # Extract GT positive activations (keep on GPU)
                        if device.type == 'cuda' and concept_acts.is_cuda:
                            indices_tensor = torch.tensor(valid_test_indices, device=device)
                            gt_positive_acts = concept_acts[indices_tensor]
                        else:
                            gt_positive_acts = concept_acts[valid_test_indices]
                            if hasattr(gt_positive_acts, 'cpu'):
                                gt_positive_acts = gt_positive_acts.cpu()
                        
                        # Get background activations
                        samples_with_concept = set(gt_samples_per_concept.get(concept, []))
                        all_test_positions = torch.arange(len(test_global_indices), device=device)
                        has_concept = torch.zeros(len(all_test_positions), dtype=torch.bool, device=device)
                        
                        if len(samples_with_concept) > 0:
                            valid_concept_samples = [s for s in samples_with_concept if s in test_pos_map.values()]
                            if valid_concept_samples:
                                has_concept[valid_concept_samples] = True
                        
                        positions_without_concept = all_test_positions[~has_concept]
                        if positions_without_concept.is_cuda:
                            positions_without_concept = positions_without_concept.cpu()
                        
                        if len(positions_without_concept) > 0:
                            # Collect background patch/token indices
                            background_indices = []
                            
                            if is_text_dataset:
                                # For text, sample tokens from documents without the concept
                                # This is simplified - ideally we'd map through actual token counts
                                max_tokens_to_sample = min(10000, len(concept_acts))
                                # Random sample from available indices - vectorized approach
                                all_indices = torch.arange(len(concept_acts), device=device)
                                # Create mask for non-GT indices
                                mask = torch.ones(len(concept_acts), dtype=torch.bool, device=device)
                                mask[valid_test_indices] = False
                                available_indices = all_indices[mask]
                                
                                if len(available_indices) > max_tokens_to_sample:
                                    perm = torch.randperm(len(available_indices), device=device)[:max_tokens_to_sample]
                                    background_indices = available_indices[perm].cpu().numpy()
                                else:
                                    background_indices = available_indices.cpu().numpy()
                            else:
                                # For images, sample patches from images without the concept
                                # More efficient vectorized approach
                                n_bg_images = min(100, len(positions_without_concept))
                                n_patches_per_img = min(patches_per_image, 50)
                                
                                # Create grid of indices
                                bg_positions = positions_without_concept[:n_bg_images]
                                if isinstance(bg_positions, torch.Tensor):
                                    bg_positions = bg_positions.cpu().numpy()
                                patch_offsets = np.arange(n_patches_per_img)
                                
                                # Vectorized computation of all indices
                                bg_positions_expanded = np.repeat(bg_positions, n_patches_per_img)
                                patch_offsets_tiled = np.tile(patch_offsets, n_bg_images)
                                background_indices = bg_positions_expanded * patches_per_image + patch_offsets_tiled
                                
                                # Filter valid indices
                                background_indices = background_indices[background_indices < len(concept_acts)]
                                
                                # Filter out padding patches for background as well
                                if len(background_indices) > 0 and not is_text_dataset:
                                    from utils.patch_alignment_utils import filter_patches_by_image_presence
                                    background_indices_filtered = filter_patches_by_image_presence(
                                        background_indices.tolist(), dataset, model_input_size
                                    )
                                    background_indices = background_indices_filtered.numpy()
                            
                            if len(background_indices) > 0:
                                # Extract background activations (keep on GPU)
                                if device.type == 'cuda' and concept_acts.is_cuda:
                                    indices_tensor = torch.tensor(background_indices, device=device)
                                    background_acts = concept_acts[indices_tensor]
                                    # Sample if too many
                                    if len(background_acts) > 10000:
                                        perm = torch.randperm(len(background_acts), device=device)[:10000]
                                        background_acts = background_acts[perm]
                                else:
                                    background_acts = concept_acts[background_indices]
                                    if hasattr(background_acts, 'cpu'):
                                        background_acts = background_acts.cpu()
                                    if len(background_acts) > 10000:
                                        indices = torch.randperm(len(background_acts))[:10000]
                                        background_acts = background_acts[indices]
                            else:
                                background_acts = torch.tensor([])
                        else:
                            background_acts = torch.tensor([])
                        
                        # Compute probability mass overlap
                        if len(gt_positive_acts) > 0 and len(background_acts) > 0:
                            # Ensure tensors
                            if not isinstance(gt_positive_acts, torch.Tensor):
                                gt_positive_acts = torch.tensor(gt_positive_acts)
                            if not isinstance(background_acts, torch.Tensor):
                                background_acts = torch.tensor(background_acts)
                            
                            # Move to GPU if available
                            if device.type == 'cuda':
                                if not gt_positive_acts.is_cuda:
                                    gt_positive_acts = gt_positive_acts.to(device)
                                if not background_acts.is_cuda:
                                    background_acts = background_acts.to(device)
                            
                            # Determine bin range
                            min_val = torch.min(torch.min(gt_positive_acts), torch.min(background_acts))
                            max_val = torch.max(torch.max(gt_positive_acts), torch.max(background_acts))
                            
                            # Create bins
                            bins = torch.linspace(min_val.item(), max_val.item(), 101, device=gt_positive_acts.device)
                            bin_width = (max_val - min_val) / 100
                            
                            # Use torch.histc for histogram
                            hist_gt = torch.histc(gt_positive_acts, bins=100, min=min_val.item(), max=max_val.item())
                            hist_bg = torch.histc(background_acts, bins=100, min=min_val.item(), max=max_val.item())
                            
                            # Normalize to probability density
                            hist_gt = hist_gt / (gt_positive_acts.shape[0] * bin_width)
                            hist_bg = hist_bg / (background_acts.shape[0] * bin_width)
                            
                            # Convert to probability mass
                            prob_gt = hist_gt * bin_width
                            prob_bg = hist_bg * bin_width
                            
                            # Compute overlap
                            overlap_prob_mass = torch.sum(torch.minimum(prob_gt, prob_bg)).item()
                            
                            # Ensure result is in valid range
                            overlap = min(max(overlap_prob_mass, 0.0), 1.0)
                            concept_overlaps.append(overlap)
                    
                    # Average across concepts
                    if concept_overlaps:
                        avg_overlap = sum(concept_overlaps) / len(concept_overlaps)
                        overlaps_by_layer.append(avg_overlap)
                    else:
                        overlaps_by_layer.append(float('nan'))
                
                # Store results
                key = f"{dataset}_{model}_{concept_type}"
                results[key] = {
                    'percentthrus': dataset_percentthrus[:len(overlaps_by_layer)],
                    'overlaps': overlaps_by_layer,
                    'dataset': dataset,
                    'model': model,
                    'concept_type': concept_type
                }
                
                # Plot if we have valid data
                if overlaps_by_layer and not all(np.isnan(overlaps_by_layer)):
                    ax.plot(
                        dataset_percentthrus[:len(overlaps_by_layer)],
                        overlaps_by_layer,
                        color=colors[dataset],
                        linestyle=linestyles[concept_type],
                        label=None,  # Don't add individual labels
                        linewidth=2,
                        marker='o',
                        markersize=4
                    )
        
        # Customize subplot
        ax.set_xlabel('Percent Through Model (%)')
        ax.set_ylabel('Probability Mass Overlap')
        ax.set_title(f'{model}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)
        
        # Add custom legend to first subplot
        if model_idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = []
            
            # Add legend entries for concept types
            if 'avg' in concept_types:
                legend_elements.append(Line2D([0], [0], color='gray', linestyle='-', 
                                            linewidth=2, label='avg'))
            if 'linsep' in concept_types:
                legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', 
                                            linewidth=2, label='linsep'))
            if 'kmeans' in concept_types:
                legend_elements.append(Line2D([0], [0], color='gray', linestyle=':', 
                                            linewidth=2, label='kmeans'))
            
            ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
                     fancybox=True, framealpha=0.8)
    
    # Overall title
    fig.suptitle('Probability Mass Overlap Across Layers', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return results


def compute_overlap_across_layers_data_old(
    datasets: List[str],
    models: List[str],
    concept_types: List[str] = ['avg', 'linsep'],
    sample_type: str = 'patch',
    percentthrus: Optional[List[int]] = None,
    scratch_dir: str = '',
    background_percentile: float = 0.99,
    validation_split: str = 'cal'
) -> Dict:
    """
    Compute GT mass above threshold and detection rates across layers for multiple datasets and models.
    
    Args:
        datasets: List of dataset names (e.g., ['CLEVR', 'Coco', 'Sarcasm'])
        models: List of model names (e.g., ['ViT-B-16', 'Llama'])
        concept_types: List of concept types to analyze (default: ['avg', 'linsep'])
        sample_type: Type of samples ('patch' or 'cls') - use 'patch' for patch/token-level analysis
        percentthrus: List of percentthru values to analyze (default: model-specific)
        scratch_dir: Directory containing activation files (default: '')
        background_percentile: Percentile of background distribution to use as threshold (default: 0.99 = 99%)
        validation_split: Split to use for computing background thresholds (default: 'cal')
        
    Returns:
        Dict containing results for each configuration with structure:
        {
            'dataset_model_concepttype': {
                'percentthrus': list of percentthru values,
                'gt_mass_above_threshold': list of averaged GT mass above threshold values,
                'gt_mass_per_concept': dict mapping concept names to GT mass lists,
                'detection_rates': list of averaged detection rate values,
                'detection_rates_per_concept': dict mapping concept names to detection rate lists,
                'dataset': dataset name,
                'model': model name,
                'concept_type': concept type,
                'background_percentile': percentile used
            },
            ...
        }
    """
    import warnings
    from pathlib import Path
    from utils.filter_datasets_utils import DATASET_TO_CONCEPTS
    from utils.default_percentthrumodels import get_model_default_percentthrumodels
    
    # Default percentthrus if not specified - will be set per model later
    if percentthrus is None:
        use_model_defaults = True
    else:
        use_model_defaults = False
    
    results = {}
    
    # Process each model
    for model in models:
        # Process each dataset
        for dataset in datasets:
            # Check if this is a valid model-dataset combination
            is_text_dataset = dataset in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'IMDB', 'iSarcasmEval', 'Stanford-Tree-Bank', 'jailbreak']
            is_image_dataset = dataset in ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'surgery_fig1', 'surgery']
            
            # Skip invalid combinations
            if model in ['ViT-B-16', 'CLIP'] and is_text_dataset:
                print(f"Skipping invalid combination: {model} (image model) with {dataset} (text dataset)")
                continue
            elif model in ['Gemma', 'Qwen'] and is_image_dataset:
                print(f"Skipping invalid combination: {model} (text model) with {dataset} (image dataset)")
                continue
            # Note: Llama can handle both text and images (Llama-Vision and Llama-Text)
            
            # Get valid concepts for this dataset
            valid_concepts = DATASET_TO_CONCEPTS.get(dataset, [])
            
            # Get dataset-specific percentthrus if using defaults
            if use_model_defaults:
                # Determine model input size for this specific dataset
                if model in ['ViT-B-16', 'CLIP']:
                    model_input_size_for_ptm = (224, 224)
                elif model == 'Llama':
                    # Use appropriate percentthrus based on THIS dataset, not all datasets
                    if is_text_dataset:
                        model_input_size_for_ptm = ('text', 'text')
                    else:
                        model_input_size_for_ptm = (560, 560)
                elif model == 'Gemma':
                    model_input_size_for_ptm = ('text', 'text2')
                elif model == 'Qwen':
                    model_input_size_for_ptm = ('text', 'text3')
                else:
                    model_input_size_for_ptm = (224, 224)
                
                dataset_percentthrus = get_model_default_percentthrumodels(model, model_input_size_for_ptm)
            else:
                dataset_percentthrus = percentthrus
            
            # Process each concept type
            for concept_type in concept_types:
                overlaps_by_layer = []
                
                # Process each layer (percentthru)
                for percentthru in tqdm(dataset_percentthrus, desc=f"{model} {dataset} {concept_type}"):
                    # Determine activation file name based on concept type
                    if concept_type == 'avg':
                        acts_filename = f"cosine_similarities_avg_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
                        acts_dir = os.path.join(scratch_dir, 'Cosine_Similarities', dataset)
                    elif concept_type == 'linsep':
                        acts_filename = f"dists_linsep_concepts_BD_True_BN_False_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
                        acts_dir = os.path.join(scratch_dir, 'Distances', dataset)
                    else:
                        print(f"Warning: Unknown concept type: {concept_type}")
                        continue
                    
                    # Try to load using ChunkedActivationLoader
                    try:
                        from utils.memory_management_utils import ChunkedActivationLoader
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        act_loader = ChunkedActivationLoader(dataset, acts_filename, scratch_dir=scratch_dir, device=device)
                        
                        # Determine model input size for this model/dataset combination
                        if model in ['ViT-B-16', 'CLIP']:
                            load_model_input_size = (224, 224)
                        elif model == 'Llama':
                            if dataset in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'IMDB', 'iSarcasmEval']:
                                load_model_input_size = ('text', 'text')
                            else:
                                load_model_input_size = (560, 560)
                        elif model == 'Gemma':
                            load_model_input_size = ('text', 'text2')
                        elif model == 'Qwen':
                            load_model_input_size = ('text', 'text3')
                        else:
                            load_model_input_size = (224, 224)
                        
                        # Load test activations
                        test_acts = act_loader.load_split_tensor('test', dataset, model_input_size=load_model_input_size, patch_size=14)
                        if test_acts is None:
                            print(f"Warning: Could not load test activations for {dataset} {model} {concept_type} ptm={percentthru}")
                            continue
                            
                        # Load concepts to get indices
                        if concept_type == 'avg':
                            concepts_filename = f"avg_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
                        else:
                            concepts_filename = f"linsep_concepts_BD_True_BN_False_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
                        
                        concepts_file = f"Concepts/{dataset}/{concepts_filename}"
                        if not os.path.exists(concepts_file):
                            print(f"Warning: Concepts file not found: {concepts_file}")
                            continue
                            
                        # Load concepts data, suppressing warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=FutureWarning)
                            try:
                                concepts_data = torch.load(concepts_file, map_location='cpu', weights_only=True)
                            except:
                                # Fallback to weights_only=False if the file contains non-tensor objects
                                concepts_data = torch.load(concepts_file, map_location='cpu', weights_only=False)
                        all_concept_names = list(concepts_data.keys())
                        
                    except Exception as e:
                        print(f"Warning: Error loading activations for {dataset} {model} {concept_type} ptm={percentthru}: {e}")
                        continue
                    
                    # Load ground truth data
                    # Determine if this is a text dataset
                    is_text_dataset = dataset in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'IMDB', 'iSarcasmEval']
                    
                    if model in ['ViT-B-16', 'CLIP']:
                        model_input_size = (224, 224)
                        patches_per_image = 256  # 16x16
                    elif model == 'Llama':
                        if is_text_dataset:
                            model_input_size = ('text', 'text')
                            # For text, we'll handle token counting differently
                        else:
                            model_input_size = (560, 560)
                            patches_per_image = 1600  # 40x40
                    elif model == 'Gemma':
                        model_input_size = ('text', 'text2')
                        patches_per_image = None  # Variable for text
                    elif model == 'Qwen':
                        model_input_size = ('text', 'text3')
                        patches_per_image = None  # Variable for text
                    else:
                        model_input_size = (224, 224)
                        patches_per_image = 256
                    
                    # Load GT patches and samples
                    gt_patches_file = f"GT_Samples/{dataset}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
                    if not os.path.exists(gt_patches_file):
                        gt_patches_file = f"GT_Samples/{dataset}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
                    
                    gt_samples_file = f"GT_Samples/{dataset}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
                    
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=FutureWarning)
                            try:
                                gt_patches_per_concept = torch.load(gt_patches_file, map_location='cpu', weights_only=True)
                                gt_samples_per_concept = torch.load(gt_samples_file, map_location='cpu', weights_only=True)
                            except:
                                # Fallback if files contain non-tensor objects
                                gt_patches_per_concept = torch.load(gt_patches_file, map_location='cpu', weights_only=False)
                                gt_samples_per_concept = torch.load(gt_samples_file, map_location='cpu', weights_only=False)
                    except Exception as e:
                        print(f"Warning: Could not load GT data: {e}")
                        continue
                    
                    # Get test metadata
                    import pandas as pd
                    metadata = pd.read_csv(f'../Data/{dataset}/metadata.csv')
                    test_global_indices = metadata[metadata['split'] == 'test'].index.tolist()
                    test_pos_map = {idx: pos for pos, idx in enumerate(test_global_indices)}
                    
                    # Compute overlap for each concept and average
                    concept_overlaps = []
                    
                    # Get indices for all valid concepts at once
                    valid_concept_indices = []
                    valid_concept_names = []
                    for concept in valid_concepts:
                        if concept in all_concept_names:
                            valid_concept_indices.append(all_concept_names.index(concept))
                            valid_concept_names.append(concept)
                    
                    if not valid_concept_indices:
                        continue
                    
                    # Process all concepts in parallel on GPU
                    if device.type == 'cuda' and test_acts.is_cuda:
                        # Get all concept activations at once (keep on GPU)
                        all_concept_acts = test_acts[:, valid_concept_indices]  # shape: [n_patches, n_concepts]
                    else:
                        all_concept_acts = test_acts[:, valid_concept_indices]
                    
                    # Process each concept
                    for concept_idx_in_batch, concept in enumerate(valid_concept_names):
                        # Get concept activations for this concept
                        concept_acts = all_concept_acts[:, concept_idx_in_batch]
                        
                        # Get GT positive patch indices
                        positive_patch_indices = gt_patches_per_concept.get(concept, [])
                        if len(positive_patch_indices) == 0:
                            continue
                        
                        # Collect GT positive activations
                        if is_text_dataset:
                            # For text datasets, patch indices are token indices
                            # Vectorized filtering
                            positive_patch_indices_np = np.array(positive_patch_indices)
                            valid_mask = positive_patch_indices_np < len(concept_acts)
                            valid_test_indices = positive_patch_indices_np[valid_mask].tolist()
                        else:
                            # For image datasets - vectorized computation
                            positive_patch_indices_np = np.array(positive_patch_indices)
                            global_img_indices = positive_patch_indices_np // patches_per_image
                            patch_within_imgs = positive_patch_indices_np % patches_per_image
                            
                            # Vectorized mapping
                            test_positions = np.array([test_pos_map.get(idx, -1) for idx in global_img_indices])
                            valid_mask = test_positions >= 0
                            
                            if np.any(valid_mask):
                                valid_test_positions = test_positions[valid_mask]
                                valid_patch_within = patch_within_imgs[valid_mask]
                                test_patch_indices = valid_test_positions * patches_per_image + valid_patch_within
                                
                                # Final filter for bounds
                                final_valid_mask = test_patch_indices < len(concept_acts)
                                valid_test_indices = test_patch_indices[final_valid_mask].tolist()
                                
                                # Filter out padding patches using filter_patches_by_image_presence
                                from utils.patch_alignment_utils import filter_patches_by_image_presence
                                if valid_test_indices:
                                    valid_test_indices_filtered = filter_patches_by_image_presence(
                                        valid_test_indices, dataset, model_input_size
                                    )
                                    valid_test_indices = valid_test_indices_filtered.tolist()
                            else:
                                valid_test_indices = []
                        
                        if len(valid_test_indices) == 0:
                            continue
                        
                        # Extract GT positive activations (keep on GPU)
                        if device.type == 'cuda' and concept_acts.is_cuda:
                            indices_tensor = torch.tensor(valid_test_indices, device=device)
                            gt_positive_acts = concept_acts[indices_tensor]
                        else:
                            gt_positive_acts = concept_acts[valid_test_indices]
                            if hasattr(gt_positive_acts, 'cpu'):
                                gt_positive_acts = gt_positive_acts.cpu()
                        
                        # Get background activations
                        samples_with_concept = set(gt_samples_per_concept.get(concept, []))
                        all_test_positions = torch.arange(len(test_global_indices), device=device)
                        has_concept = torch.zeros(len(all_test_positions), dtype=torch.bool, device=device)
                        
                        if len(samples_with_concept) > 0:
                            valid_concept_samples = [s for s in samples_with_concept if s in test_pos_map.values()]
                            if valid_concept_samples:
                                has_concept[valid_concept_samples] = True
                        
                        positions_without_concept = all_test_positions[~has_concept]
                        if positions_without_concept.is_cuda:
                            positions_without_concept = positions_without_concept.cpu()
                        
                        if len(positions_without_concept) > 0:
                            # Collect background patch/token indices
                            background_indices = []
                            
                            if is_text_dataset:
                                # For text, sample tokens from documents without the concept
                                # This is simplified - ideally we'd map through actual token counts
                                max_tokens_to_sample = min(10000, len(concept_acts))
                                # Random sample from available indices - vectorized approach
                                all_indices = torch.arange(len(concept_acts), device=device)
                                # Create mask for non-GT indices
                                mask = torch.ones(len(concept_acts), dtype=torch.bool, device=device)
                                mask[valid_test_indices] = False
                                available_indices = all_indices[mask]
                                
                                if len(available_indices) > max_tokens_to_sample:
                                    perm = torch.randperm(len(available_indices), device=device)[:max_tokens_to_sample]
                                    background_indices = available_indices[perm].cpu().numpy()
                                else:
                                    background_indices = available_indices.cpu().numpy()
                            else:
                                # For images, sample patches from images without the concept
                                # More efficient vectorized approach
                                n_bg_images = min(100, len(positions_without_concept))
                                n_patches_per_img = min(patches_per_image, 50)
                                
                                # Create grid of indices
                                bg_positions = positions_without_concept[:n_bg_images]
                                if isinstance(bg_positions, torch.Tensor):
                                    bg_positions = bg_positions.cpu().numpy()
                                patch_offsets = np.arange(n_patches_per_img)
                                
                                # Vectorized computation of all indices
                                bg_positions_expanded = np.repeat(bg_positions, n_patches_per_img)
                                patch_offsets_tiled = np.tile(patch_offsets, n_bg_images)
                                background_indices = bg_positions_expanded * patches_per_image + patch_offsets_tiled
                                
                                # Filter valid indices
                                background_indices = background_indices[background_indices < len(concept_acts)]
                                
                                # Filter out padding patches for background as well
                                if len(background_indices) > 0 and not is_text_dataset:
                                    from utils.patch_alignment_utils import filter_patches_by_image_presence
                                    background_indices_filtered = filter_patches_by_image_presence(
                                        background_indices.tolist(), dataset, model_input_size
                                    )
                                    background_indices = background_indices_filtered.numpy()
                            
                            if len(background_indices) > 0:
                                # Extract background activations (keep on GPU)
                                if device.type == 'cuda' and concept_acts.is_cuda:
                                    indices_tensor = torch.tensor(background_indices, device=device)
                                    background_acts = concept_acts[indices_tensor]
                                    # Sample if too many
                                    if len(background_acts) > 10000:
                                        perm = torch.randperm(len(background_acts), device=device)[:10000]
                                        background_acts = background_acts[perm]
                                else:
                                    background_acts = concept_acts[background_indices]
                                    if hasattr(background_acts, 'cpu'):
                                        background_acts = background_acts.cpu()
                                    if len(background_acts) > 10000:
                                        indices = torch.randperm(len(background_acts))[:10000]
                                        background_acts = background_acts[indices]
                            else:
                                background_acts = torch.tensor([])
                        else:
                            background_acts = torch.tensor([])
                        
                        # Compute probability mass overlap
                        if len(gt_positive_acts) > 0 and len(background_acts) > 0:
                            # Ensure tensors
                            if not isinstance(gt_positive_acts, torch.Tensor):
                                gt_positive_acts = torch.tensor(gt_positive_acts)
                            if not isinstance(background_acts, torch.Tensor):
                                background_acts = torch.tensor(background_acts)
                            
                            # Move to GPU if available
                            if device.type == 'cuda':
                                if not gt_positive_acts.is_cuda:
                                    gt_positive_acts = gt_positive_acts.to(device)
                                if not background_acts.is_cuda:
                                    background_acts = background_acts.to(device)
                            
                            # Determine bin range
                            min_val = torch.min(torch.min(gt_positive_acts), torch.min(background_acts))
                            max_val = torch.max(torch.max(gt_positive_acts), torch.max(background_acts))
                            
                            # Create bins
                            bins = torch.linspace(min_val.item(), max_val.item(), 101, device=gt_positive_acts.device)
                            bin_width = (max_val - min_val) / 100
                            
                            # Use torch.histc for histogram
                            hist_gt = torch.histc(gt_positive_acts, bins=100, min=min_val.item(), max=max_val.item())
                            hist_bg = torch.histc(background_acts, bins=100, min=min_val.item(), max=max_val.item())
                            
                            # Normalize to probability density
                            hist_gt = hist_gt / (gt_positive_acts.shape[0] * bin_width)
                            hist_bg = hist_bg / (background_acts.shape[0] * bin_width)
                            
                            # Convert to probability mass
                            prob_gt = hist_gt * bin_width
                            prob_bg = hist_bg * bin_width
                            
                            # Compute overlap
                            overlap_prob_mass = torch.sum(torch.minimum(prob_gt, prob_bg)).item()
                            
                            # Ensure result is in valid range
                            overlap = min(max(overlap_prob_mass, 0.0), 1.0)
                            concept_overlaps.append(overlap)
                    
                    # Average across concepts
                    if concept_overlaps:
                        avg_overlap = sum(concept_overlaps) / len(concept_overlaps)
                        overlaps_by_layer.append(avg_overlap)
                    else:
                        overlaps_by_layer.append(float('nan'))
                
                # Store results
                key = f"{dataset}_{model}_{concept_type}"
                results[key] = {
                    'percentthrus': dataset_percentthrus[:len(overlaps_by_layer)],
                    'overlaps': overlaps_by_layer,
                    'dataset': dataset,
                    'model': model,
                    'concept_type': concept_type
                }
    
    return results


def plot_overlap_across_layers(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
    title: Optional[str] = None,
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot probability mass overlap results across layers.
    
    Args:
        results: Results dictionary from compute_overlap_across_layers_data
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple
        colors: Dict mapping dataset names to colors (optional)
        linestyles: Dict mapping concept types to line styles (optional)
        show_legend: Whether to show the legend
        title: Custom title for the plot (default: 'Probability Mass Overlap Across Layers')
        datasets: List of dataset names to include in the plot (default: all datasets in results)
        models: List of model names to include in the plot (default: all models in results)
        
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    
    # Filter results based on datasets and models parameters
    if datasets is not None or models is not None:
        filtered_results = {}
        for key, data in results.items():
            include_dataset = datasets is None or data['dataset'] in datasets
            include_model = models is None or data['model'] in models
            if include_dataset and include_model:
                filtered_results[key] = data
        results = filtered_results
    
    # Extract unique models and datasets from filtered results
    models = sorted(list(set(r['model'] for r in results.values())))
    datasets = sorted(list(set(r['dataset'] for r in results.values())))
    concept_types = sorted(list(set(r['concept_type'] for r in results.values())))
    
    # Default colors for datasets if not specified
    if colors is None:
        color_palette = sns.color_palette('husl', len(datasets))
        colors = {ds: color_palette[i] for i, ds in enumerate(datasets)}
    
    # Default line styles for concept types if not specified
    if linestyles is None:
        linestyles = {
            'avg': '-',
            'linsep': '--',
            'kmeans': ':'
        }
    
    # Set up the plot
    fig, axes = plt.subplots(1, len(models), figsize=figsize, squeeze=False)
    axes = axes[0]
    
    # Plot each model
    for model_idx, model in enumerate(models):
        ax = axes[model_idx]
        
        # Plot each result
        for key, data in results.items():
            if data['model'] != model:
                continue
            
            # Extract data
            dataset = data['dataset']
            concept_type = data['concept_type']
            percentthrus = data['percentthrus']
            overlaps = data['overlaps']
            
            # Plot if we have valid data
            if overlaps and not all(np.isnan(overlaps)):
                ax.plot(
                    percentthrus,
                    overlaps,
                    color=colors.get(dataset, 'gray'),
                    linestyle=linestyles.get(concept_type, '-'),
                    label=None,  # Don't add individual labels
                    linewidth=2,
                    marker='o',
                    markersize=4
                )
        
        # Customize subplot
        ax.set_xlabel('Percent Through Model (%)')
        ax.set_ylabel('Probability Mass Overlap')
        ax.set_title(f'{model}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)
        
        # Add custom legend to first subplot
        if model_idx == 0 and show_legend:
            legend_elements = []
            
            # Add legend entries for datasets
            for ds in datasets:
                legend_elements.append(Line2D([0], [0], color=colors.get(ds, 'gray'), 
                                            linewidth=2, label=ds))
            
            # Add separator if we have both datasets and concept types
            if len(datasets) > 0 and len(concept_types) > 1:
                legend_elements.append(Line2D([0], [0], color='none', label=''))
            
            # Add legend entries for concept types
            if 'avg' in concept_types:
                legend_elements.append(Line2D([0], [0], color='gray', linestyle='-', 
                                            linewidth=2, label='avg'))
            if 'linsep' in concept_types:
                legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', 
                                            linewidth=2, label='linsep'))
            if 'kmeans' in concept_types:
                legend_elements.append(Line2D([0], [0], color='gray', linestyle=':', 
                                            linewidth=2, label='kmeans'))
            
            ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
                     fancybox=True, framealpha=0.8)
    
    # Overall title
    if title is None:
        title = 'Probability Mass Overlap Across Layers'
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def compute_background_detection_across_layers_data(
    datasets: List[str],
    models: List[str],
    concept_types: List[str] = ['avg', 'linsep'],
    sample_type: str = 'patch',
    percentthrus: Optional[List[int]] = None,
    scratch_dir: str = '',
    background_percentile: float = 0.995,
    validation_split: str = 'cal',
    max_background_samples: int = 100000
) -> Dict:
    """
    Compute background detection rates across layers for multiple datasets and models.
    
    Background detection rate is the percentage of GT images that have at least one patch/token
    exceeding the background percentile threshold (computed from non-concept samples).
    
    Args:
        datasets: List of dataset names (e.g., ['CLEVR', 'Coco', 'Sarcasm'])
        models: List of model names (e.g., ['ViT-B-16', 'Llama'])
        concept_types: List of concept types to analyze (default: ['avg', 'linsep'])
        sample_type: Type of samples ('patch' or 'cls') - use 'patch' for patch/token-level analysis
        percentthrus: List of percentthru values to analyze (default: model-specific)
        scratch_dir: Directory containing activation files (default: '')
        background_percentile: Percentile of background distribution to use as threshold (default: 0.995 = 99.5%)
        validation_split: Split to use for computing background thresholds (default: 'cal')
        max_background_samples: Maximum number of background samples to use for threshold computation
        
    Returns:
        Dict containing detection results for each configuration with structure:
        {
            'dataset_model_concepttype': {
                'percentthrus': list of percentthru values,
                'detection_rates': list of averaged detection rates,
                'detection_rates_per_concept': dict mapping concept names to detection rate lists,
                'gt_mass_above_threshold': list of averaged GT mass above threshold,
                'gt_mass_per_concept': dict mapping concept names to GT mass lists,
                'dataset': dataset name,
                'model': model name,
                'concept_type': concept type,
                'background_percentile': percentile used
            },
            ...
        }
    """
    import warnings
    from pathlib import Path
    
    # Default percentthrus if not specified - will be set per model later
    if percentthrus is None:
        use_model_defaults = True
    else:
        use_model_defaults = False
    
    results = {}
    
    # Process each model
    for model in models:
        # Process each dataset
        for dataset in datasets:
            # Check if this is a valid model-dataset combination
            is_text_dataset = dataset in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'IMDB', 'iSarcasmEval', 'Stanford-Tree-Bank', 'jailbreak']
            is_image_dataset = dataset in ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'surgery_fig1', 'surgery']
            
            # Skip invalid combinations
            if model in ['ViT-B-16', 'CLIP'] and is_text_dataset:
                print(f"Skipping invalid combination: {model} (image model) with {dataset} (text dataset)")
                continue
            elif model in ['Gemma', 'Qwen'] and is_image_dataset:
                print(f"Skipping invalid combination: {model} (text model) with {dataset} (image dataset)")
                continue
            
            # Get valid concepts for this dataset
            valid_concepts = DATASET_TO_CONCEPTS.get(dataset, [])
            if not valid_concepts:
                print(f"Warning: No valid concepts found for dataset {dataset}")
                continue
            
            # Get dataset-specific percentthrus if using defaults
            if use_model_defaults:
                # Determine model input size for this specific dataset
                if model in ['ViT-B-16', 'CLIP']:
                    model_input_size = (224, 224)
                elif model == 'Llama':
                    if is_text_dataset:
                        model_input_size = ('text', 'text')
                    else:
                        model_input_size = (560, 560)
                elif model == 'Gemma':
                    model_input_size = ('text', 'text2')
                elif model == 'Qwen':
                    model_input_size = ('text', 'text3')
                else:
                    model_input_size = (224, 224)
                
                dataset_percentthrus = get_model_default_percentthrumodels(model, model_input_size)
            else:
                dataset_percentthrus = percentthrus
            
            # Process each concept type
            for concept_type in concept_types:
                detection_rates_by_layer = []
                detection_rates_per_concept_by_layer = {concept: [] for concept in valid_concepts}
                gt_mass_by_layer = []
                gt_mass_per_concept_by_layer = {concept: [] for concept in valid_concepts}
                
                # Process each layer (percentthru)
                for percentthru in tqdm(dataset_percentthrus, desc=f"{model} {dataset} {concept_type}"):
                    # Determine activation file name based on concept type
                    if concept_type == 'avg':
                        acts_filename = f"cosine_similarities_avg_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
                        acts_dir = os.path.join(scratch_dir, 'Cosine_Similarities', dataset)
                    elif concept_type == 'linsep':
                        acts_filename = f"dists_linsep_concepts_BD_True_BN_False_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
                        acts_dir = os.path.join(scratch_dir, 'Distances', dataset)
                    else:
                        print(f"Warning: Unknown concept type: {concept_type}")
                        continue
                    
                    # Try to load using ChunkedActivationLoader
                    try:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        act_loader = ChunkedActivationLoader(dataset, acts_filename, scratch_dir=scratch_dir, device=device)
                        
                        # Load test activations
                        test_acts = act_loader.load_split_tensor('test', dataset, model_input_size, patch_size=14)
                        if test_acts is None:
                            print(f"Warning: Could not load test activations for {dataset} {model} {concept_type} ptm={percentthru}")
                            continue
                        
                        # Load validation activations for threshold computation
                        val_acts = act_loader.load_split_tensor(validation_split, dataset, model_input_size, patch_size=14)
                        if val_acts is None:
                            print(f"Warning: Could not load validation activations for {dataset} {model} {concept_type} ptm={percentthru}")
                            continue
                            
                        # Load concepts to get indices
                        if concept_type == 'avg':
                            concepts_filename = f"avg_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
                        else:
                            concepts_filename = f"linsep_concepts_BD_True_BN_False_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
                        
                        concepts_file = f"Concepts/{dataset}/{concepts_filename}"
                        if not os.path.exists(concepts_file):
                            print(f"Warning: Concepts file not found: {concepts_file}")
                            continue
                            
                        # Load concepts data
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=FutureWarning)
                            try:
                                concepts_data = torch.load(concepts_file, map_location='cpu', weights_only=True)
                            except:
                                concepts_data = torch.load(concepts_file, map_location='cpu', weights_only=False)
                        all_concept_names = list(concepts_data.keys())
                        
                    except Exception as e:
                        print(f"Warning: Error loading data for {dataset} {model} {concept_type} ptm={percentthru}: {e}")
                        continue
                    
                    # Load ground truth data
                    if model in ['ViT-B-16', 'CLIP']:
                        patches_per_image = 256  # 16x16
                    elif model == 'Llama':
                        if is_text_dataset:
                            patches_per_image = None  # Variable for text
                        else:
                            patches_per_image = 1600  # 40x40
                    else:
                        patches_per_image = 256
                    
                    # Load GT patches and samples
                    gt_patches_file = f"GT_Samples/{dataset}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
                    if not os.path.exists(gt_patches_file):
                        gt_patches_file = f"GT_Samples/{dataset}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
                    
                    gt_samples_file = f"GT_Samples/{dataset}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
                    
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=FutureWarning)
                            try:
                                gt_patches_per_concept = torch.load(gt_patches_file, map_location='cpu', weights_only=True)
                                gt_samples_per_concept = torch.load(gt_samples_file, map_location='cpu', weights_only=True)
                            except:
                                gt_patches_per_concept = torch.load(gt_patches_file, map_location='cpu', weights_only=False)
                                gt_samples_per_concept = torch.load(gt_samples_file, map_location='cpu', weights_only=False)
                    except Exception as e:
                        print(f"Warning: Could not load GT data: {e}")
                        continue
                    
                    # Filter to valid concepts
                    gt_patches_per_concept = filter_concept_dict(gt_patches_per_concept, dataset)
                    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset)
                    
                    # Get metadata
                    metadata = pd.read_csv(f'../Data/{dataset}/metadata.csv')
                    test_global_indices = metadata[metadata['split'] == 'test'].index.tolist()
                    val_global_indices = metadata[metadata['split'] == validation_split].index.tolist()
                    test_pos_map = {idx: pos for pos, idx in enumerate(test_global_indices)}
                    
                    # Compute detection rates for each concept
                    concept_detection_rates = []
                    concept_gt_masses = []
                    
                    for concept in valid_concepts:
                        if concept not in all_concept_names:
                            continue
                        
                        concept_idx = all_concept_names.index(concept)
                        
                        # Get validation background activations for this concept
                        val_concept_acts = val_acts[:, concept_idx]
                        
                        # Get validation samples without this concept
                        val_samples_with_concept = set(gt_samples_per_concept.get(concept, []))
                        val_positions_without_concept = [i for i, idx in enumerate(val_global_indices) 
                                                       if idx not in val_samples_with_concept]
                        
                        if len(val_positions_without_concept) == 0:
                            continue
                        
                        # Collect validation background activations
                        if is_text_dataset:
                            # For text, sample from all tokens not in concept documents
                            # This is simplified - ideally we'd track token boundaries
                            val_bg_acts = val_concept_acts.cpu().numpy()
                            if len(val_bg_acts) > max_background_samples:
                                val_bg_acts = np.random.choice(val_bg_acts, max_background_samples, replace=False)
                        else:
                            # For images, collect patches from non-concept images
                            val_bg_indices = []
                            for pos in val_positions_without_concept[:min(len(val_positions_without_concept), 1000)]:
                                start_idx = pos * patches_per_image
                                end_idx = min((pos + 1) * patches_per_image, len(val_concept_acts))
                                val_bg_indices.extend(range(start_idx, end_idx))
                            
                            if len(val_bg_indices) > max_background_samples:
                                val_bg_indices = np.random.choice(val_bg_indices, max_background_samples, replace=False)
                            
                            val_bg_acts = val_concept_acts[val_bg_indices].cpu().numpy()
                        
                        # Compute background threshold
                        bg_threshold = np.percentile(val_bg_acts, background_percentile * 100)
                        
                        # Get test concept activations
                        test_concept_acts = test_acts[:, concept_idx]
                        
                        # Get GT positive patches
                        positive_patch_indices = gt_patches_per_concept.get(concept, [])
                        if len(positive_patch_indices) == 0:
                            continue
                        
                        # Collect GT positive activations
                        if is_text_dataset:
                            # For text datasets
                            gt_positive_acts = []
                            for patch_idx in positive_patch_indices:
                                if patch_idx < len(test_concept_acts):
                                    gt_positive_acts.append(test_concept_acts[patch_idx].item())
                            gt_positive_acts = np.array(gt_positive_acts)
                        else:
                            # For image datasets - map to test indices
                            positive_patch_indices_np = np.array(positive_patch_indices)
                            global_img_indices = positive_patch_indices_np // patches_per_image
                            patch_within_imgs = positive_patch_indices_np % patches_per_image
                            
                            # Map to test positions
                            test_positions = []
                            valid_patches = []
                            for global_idx, patch_idx in zip(global_img_indices, patch_within_imgs):
                                if global_idx in test_pos_map:
                                    test_positions.append(test_pos_map[global_idx])
                                    valid_patches.append(patch_idx)
                            
                            if len(test_positions) > 0:
                                test_positions = np.array(test_positions)
                                valid_patches = np.array(valid_patches)
                                test_patch_indices = test_positions * patches_per_image + valid_patches
                                
                                # Filter valid indices
                                valid_mask = test_patch_indices < len(test_concept_acts)
                                test_patch_indices = test_patch_indices[valid_mask]
                                
                                if len(test_patch_indices) > 0:
                                    gt_positive_acts = test_concept_acts[test_patch_indices].cpu().numpy()
                                else:
                                    gt_positive_acts = np.array([])
                            else:
                                gt_positive_acts = np.array([])
                        
                        if len(gt_positive_acts) == 0:
                            continue
                        
                        # Compute GT mass above threshold
                        gt_mass_above_threshold = np.mean(gt_positive_acts > bg_threshold)
                        concept_gt_masses.append(gt_mass_above_threshold)
                        gt_mass_per_concept_by_layer[concept].append(gt_mass_above_threshold)
                        
                        # Compute detection rate on test set
                        # Get test samples with this concept
                        test_samples_with_concept = []
                        for sample_idx in gt_samples_per_concept.get(concept, []):
                            if sample_idx in test_pos_map:
                                test_samples_with_concept.append(test_pos_map[sample_idx])
                        
                        if len(test_samples_with_concept) == 0:
                            continue
                        
                        # Check which samples have at least one patch/token above threshold
                        detected_samples = 0
                        
                        if is_text_dataset:
                            # For text, we need to map tokens to documents
                            # This is simplified - checking if any GT positive token is above threshold
                            if np.any(gt_positive_acts > bg_threshold):
                                # If any token is above threshold, we detect the concept
                                # This is a simplification - ideally we'd track which document each token belongs to
                                detection_rate = gt_mass_above_threshold  # Use GT mass as proxy
                            else:
                                detection_rate = 0.0
                        else:
                            # For images, check each image individually
                            for test_pos in test_samples_with_concept:
                                start_idx = test_pos * patches_per_image
                                end_idx = min((test_pos + 1) * patches_per_image, len(test_concept_acts))
                                image_acts = test_concept_acts[start_idx:end_idx].cpu().numpy()
                                
                                if np.any(image_acts > bg_threshold):
                                    detected_samples += 1
                            
                            detection_rate = detected_samples / len(test_samples_with_concept)
                        
                        concept_detection_rates.append(detection_rate)
                        detection_rates_per_concept_by_layer[concept].append(detection_rate)
                    
                    # Average across concepts for this layer
                    if concept_detection_rates:
                        avg_detection_rate = np.mean(concept_detection_rates)
                        detection_rates_by_layer.append(avg_detection_rate)
                    else:
                        detection_rates_by_layer.append(np.nan)
                    
                    if concept_gt_masses:
                        avg_gt_mass = np.mean(concept_gt_masses)
                        gt_mass_by_layer.append(avg_gt_mass)
                    else:
                        gt_mass_by_layer.append(np.nan)
                
                # Store results
                key = f"{dataset}_{model}_{concept_type}"
                results[key] = {
                    'percentthrus': dataset_percentthrus[:len(detection_rates_by_layer)],
                    'detection_rates': detection_rates_by_layer,
                    'detection_rates_per_concept': detection_rates_per_concept_by_layer,
                    'gt_mass_above_threshold': gt_mass_by_layer,
                    'gt_mass_per_concept': gt_mass_per_concept_by_layer,
                    'dataset': dataset,
                    'model': model,
                    'concept_type': concept_type,
                    'background_percentile': background_percentile
                }
    
    return results


def plot_background_detection_across_layers(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
    title: Optional[str] = None,
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    plot_gt_mass: bool = False,
    show_error_bars: bool = True
) -> plt.Figure:
    """
    Plot background detection rates across layers.
    
    Args:
        results: Results dictionary from compute_background_detection_across_layers_data
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple
        colors: Dict mapping dataset names to colors (optional)
        linestyles: Dict mapping concept types to line styles (optional)
        show_legend: Whether to show the legend
        title: Custom title for the plot (default: 'Background Detection Rates Across Layers')
        datasets: List of dataset names to include in the plot (default: all datasets in results)
        models: List of model names to include in the plot (default: all models in results)
        plot_gt_mass: If True, plot GT mass above threshold instead of detection rates
        show_error_bars: Whether to show standard error bars
        
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    
    # Filter results based on datasets and models parameters
    if datasets is not None or models is not None:
        filtered_results = {}
        for key, data in results.items():
            include_dataset = datasets is None or data['dataset'] in datasets
            include_model = models is None or data['model'] in models
            if include_dataset and include_model:
                filtered_results[key] = data
        results = filtered_results
    
    # Extract unique models and datasets from filtered results
    models = sorted(list(set(r['model'] for r in results.values())))
    datasets = sorted(list(set(r['dataset'] for r in results.values())))
    concept_types = sorted(list(set(r['concept_type'] for r in results.values())))
    
    # Default colors for datasets if not specified
    if colors is None:
        color_palette = sns.color_palette('husl', len(datasets))
        colors = {ds: color_palette[i] for i, ds in enumerate(datasets)}
    
    # Default line styles for concept types if not specified
    if linestyles is None:
        linestyles = {
            'avg': '-',
            'linsep': '--',
            'kmeans': ':'
        }
    
    # Set up the plot
    fig, axes = plt.subplots(1, len(models), figsize=figsize, squeeze=False)
    axes = axes[0]
    
    # Plot each model
    for model_idx, model in enumerate(models):
        ax = axes[model_idx]
        
        # Plot each result
        for key, data in results.items():
            if data['model'] != model:
                continue
            
            # Extract data
            dataset = data['dataset']
            concept_type = data['concept_type']
            percentthrus = data['percentthrus']
            
            if plot_gt_mass:
                values = data['gt_mass_above_threshold']
                per_concept_values = data['gt_mass_per_concept']
            else:
                values = data['detection_rates']
                per_concept_values = data['detection_rates_per_concept']
            
            # Plot if we have valid data
            if values and not all(np.isnan(values)):
                # Calculate error bars if requested
                yerr = None
                if show_error_bars and per_concept_values:
                    # Calculate standard error across concepts for each layer
                    errors = []
                    for i in range(len(values)):
                        layer_values = []
                        for concept_values in per_concept_values.values():
                            if i < len(concept_values) and not np.isnan(concept_values[i]):
                                layer_values.append(concept_values[i])
                        
                        if len(layer_values) > 1:
                            std = np.std(layer_values)
                            se = std / np.sqrt(len(layer_values))
                            errors.append(se)
                        else:
                            errors.append(0)
                    yerr = errors[:len(values)]
                
                # Plot with or without error bars
                if yerr is not None:
                    ax.errorbar(
                        percentthrus,
                        values,
                        yerr=yerr,
                        color=colors.get(dataset, 'gray'),
                        linestyle=linestyles.get(concept_type, '-'),
                        label=None,
                        linewidth=2,
                        marker='o',
                        markersize=4,
                        capsize=3,
                        alpha=0.8
                    )
                else:
                    ax.plot(
                        percentthrus,
                        values,
                        color=colors.get(dataset, 'gray'),
                        linestyle=linestyles.get(concept_type, '-'),
                        label=None,
                        linewidth=2,
                        marker='o',
                        markersize=4
                    )
        
        # Customize subplot
        ax.set_xlabel('Percent Through Model (%)')
        if plot_gt_mass:
            ax.set_ylabel('GT Mass Above Threshold')
        else:
            ax.set_ylabel('Detection Rate')
        ax.set_title(f'{model}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Add custom legend to first subplot
        if model_idx == 0 and show_legend:
            legend_elements = []
            
            # Add legend entries for datasets
            for ds in datasets:
                legend_elements.append(Line2D([0], [0], color=colors.get(ds, 'gray'), 
                                            linewidth=2, label=ds))
            
            # Add separator
            if len(datasets) > 0 and len(concept_types) > 1:
                legend_elements.append(Line2D([0], [0], color='none', label=''))
            
            # Add legend entries for concept types
            for ct in concept_types:
                legend_elements.append(Line2D([0], [0], color='gray', linestyle=linestyles.get(ct, '-'), 
                                            linewidth=2, label=ct))
            
            ax.legend(handles=legend_elements, loc='best', frameon=True, 
                     fancybox=True, framealpha=0.8)
    
    # Overall title
    if title is None:
        if plot_gt_mass:
            title = 'GT Mass Above Background Threshold Across Layers'
        else:
            title = 'Background Detection Rates Across Layers'
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

# Import the new compute function
from utils.compute_thresholded_metrics import compute_overlap_across_layers_data


# Plot GT mass above threshold and detection rates side by side
def plot_gt_mass_and_detection_side_by_side(
    results: Dict,
    dataset: Optional[Union[str, List[str]]] = None,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    models: Optional[List[str]] = None,
    concept_types: Optional[List[str]] = None,
    colors_concept_types: Optional[Dict[str, str]] = None,
    linestyles_models: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
    suptitle: Optional[str] = None,
    show_error_bars: bool = True,
    overlap_framing: bool = True,
    text_size: int = 12,
    legend_size: int = 10,
    left_label: Optional[str] = None,
    right_label: Optional[str] = None,
    save_file: bool = False
) -> plt.Figure:
    """
    Plot GT mass above threshold and detection rates side by side for one or more datasets.
    
    Args:
        results: Results dictionary from compute_overlap_across_layers_data
        dataset: Dataset name(s) to plot. Can be:
                 - None: plot all datasets in results
                 - str: plot single dataset (original behavior)
                 - List[str]: plot multiple datasets (one per row)
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple. If None, automatically set based on number of datasets
        models: List of model names to include in the plot (default: all models in results) 
        concept_types: List of concept types to include (default: all in results)
        colors_concept_types: Dict mapping concept types to colors (optional)
        linestyles_models: Dict mapping model names to line styles (optional)
        show_legend: Whether to show the legend
        suptitle: Overall title for the figure
        show_error_bars: Whether to show standard error bars
        overlap_framing: If True, plots (1 - value) to show overlap instead of separation
        text_size: Font size for axis labels and ticks (default: 12)
        legend_size: Font size for legend text (default: 10)
        left_label: Custom title for left subplot (default: auto-generated)
        right_label: Custom title for right subplot (default: auto-generated)
        save_file: If True, automatically saves to ../Figs/Paper_Figs/multi_dataset_avg_distr_overlap.pdf (default: False)
        
    Returns:
        matplotlib Figure object
    """
    # Determine datasets to plot
    if dataset is None:
        # Plot all datasets in results
        datasets_to_plot = sorted(list(set(data['dataset'] for data in results.values())))
    elif isinstance(dataset, str):
        # Single dataset (backward compatibility)
        datasets_to_plot = [dataset]
    else:
        # List of datasets
        datasets_to_plot = dataset
    
    # Set default figure size based on number of datasets
    if figsize is None:
        figsize = (16, 6 * len(datasets_to_plot))
    
    # If single dataset, use original implementation
    if len(datasets_to_plot) == 1:
        # Filter results for the specified dataset
        filtered_results = {}
        for key, data in results.items():
            if data['dataset'] == datasets_to_plot[0]:
                include_model = models is None or data['model'] in models
                include_concept_type = concept_types is None or data['concept_type'] in concept_types
                if include_model and include_concept_type:
                    filtered_results[key] = data
        
        if not filtered_results:
            raise ValueError(f"No results found for dataset '{datasets_to_plot[0]}' after filtering")
        
        results = filtered_results
    
    # Extract unique values
    all_models = sorted(list(set(r['model'] for r in results.values())))
    all_concept_types = sorted(list(set(r['concept_type'] for r in results.values())))
    
    # Default colors for models if not specified
    if colors_concept_types is None:  # Keep parameter name for backwards compatibility
        color_palette = sns.color_palette('husl', len(all_models))
        colors_models = {model: color_palette[i] for i, model in enumerate(all_models)}
    else:
        colors_models = colors_concept_types  # In case user passed custom colors
    
    # Default line styles for concept types if not specified
    if linestyles_models is None:  # Keep parameter name for backwards compatibility
        line_styles = ['-', '--', '-.', ':']
        linestyles_concept_types = {ct: line_styles[i % len(line_styles)] for i, ct in enumerate(all_concept_types)}
    else:
        linestyles_concept_types = linestyles_models  # In case user passed custom line styles
    
    # Create figure with subplots based on number of datasets
    if len(datasets_to_plot) == 1:
        # Single dataset - original layout
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes = np.array([[axes[0], axes[1]]])  # Make it 2D for consistency
    else:
        # Multiple datasets - one row per dataset
        fig, axes = plt.subplots(len(datasets_to_plot), 2, figsize=figsize, 
                                 gridspec_kw={'hspace': 0.3})
    
    # Plot each dataset
    for row_idx, dataset_name in enumerate(datasets_to_plot):
        # Get axes for this row
        if len(datasets_to_plot) == 1:
            ax1, ax2 = axes[0]
        else:
            ax1, ax2 = axes[row_idx]
        
        # Filter results for this dataset
        dataset_results = {}
        for key, data in results.items():
            if data['dataset'] == dataset_name:
                include_model = models is None or data['model'] in models
                include_concept_type = concept_types is None or data['concept_type'] in concept_types
                if include_model and include_concept_type:
                    dataset_results[key] = data
        
        if not dataset_results:
            continue
        
        # Plot GT mass above threshold (left)
        for key, data in dataset_results.items():
            model = data['model']
            concept_type = data['concept_type']
            percentthrus = data['percentthrus']
            gt_mass_values = data['gt_mass_above_threshold']
            
            if gt_mass_values and not all(np.isnan(v) for v in gt_mass_values):
                # Don't invert - we want to show the actual GT mass above threshold
                plot_values = gt_mass_values
                
                # Create label
                concept_display = 'LinSep' if concept_type == 'linsep' else concept_type.capitalize()
                label = f"{model} - {concept_display}"
            
            # Plot with error bars if requested
            if show_error_bars and 'gt_mass_per_concept' in data:
                # Calculate standard error across concepts for each layer
                errors = []
                for i in range(len(gt_mass_values)):
                    layer_values = []
                    for concept_values in data['gt_mass_per_concept'].values():
                        if i < len(concept_values) and not np.isnan(concept_values[i]):
                            layer_values.append(concept_values[i])
                    
                    if len(layer_values) > 1:
                        std = np.std(layer_values)
                        se = std / np.sqrt(len(layer_values))
                        errors.append(se)
                    else:
                        errors.append(0)
                
                ax1.errorbar(
                    percentthrus,
                    plot_values,
                    yerr=errors[:len(plot_values)],
                    color=colors_models.get(model, 'gray'),
                    linestyle=linestyles_concept_types.get(concept_type, '-'),
                    label=label,
                    linewidth=2,
                    capsize=3,
                    alpha=0.8,
                    zorder=10
                )
            else:
                ax1.plot(
                    percentthrus,
                    plot_values,
                    color=colors_models.get(model, 'gray'),
                    linestyle=linestyles_concept_types.get(concept_type, '-'),
                    label=label,
                    linewidth=2,
                    zorder=10
                )
    
    # Plot detection rates (right)
    for key, data in results.items():
        model = data['model']
        concept_type = data['concept_type']
        percentthrus = data['percentthrus']
        detection_values = data['detection_rates']
        
        if detection_values and not all(np.isnan(v) for v in detection_values):
            # For detection rates, we want to show the actual rate, not inverted
            plot_values = detection_values
            
            # Create label
            concept_display = 'LinSep' if concept_type == 'linsep' else concept_type.capitalize()
            label = f"{model} - {concept_display}"
            
            # Plot with error bars if requested
            if show_error_bars and 'detection_rates_per_concept' in data:
                # Calculate standard error across concepts for each layer
                errors = []
                for i in range(len(detection_values)):
                    layer_values = []
                    for concept_values in data['detection_rates_per_concept'].values():
                        if i < len(concept_values) and not np.isnan(concept_values[i]):
                            layer_values.append(concept_values[i])
                    
                    if len(layer_values) > 1:
                        std = np.std(layer_values)
                        se = std / np.sqrt(len(layer_values))
                        errors.append(se)
                    else:
                        errors.append(0)
                
                ax2.errorbar(
                    percentthrus,
                    plot_values,
                    yerr=errors[:len(plot_values)],
                    color=colors_models.get(model, 'gray'),
                    linestyle=linestyles_concept_types.get(concept_type, '-'),
                    label=label,
                    linewidth=2,
                    capsize=3,
                    alpha=0.8,
                    zorder=10
                )
            else:
                ax2.plot(
                    percentthrus,
                    plot_values,
                    color=colors_models.get(model, 'gray'),
                    linestyle=linestyles_concept_types.get(concept_type, '-'),
                    label=label,
                    linewidth=2,
                    zorder=10
                )
    
    # Determine if this is a text or image dataset
    is_text_dataset = dataset in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'IMDB', 'iSarcasmEval', 'Stanford-Tree-Bank', 'jailbreak']
    
    # Customize left subplot
    ax1.set_xlabel('Percent Through Model', fontsize=text_size)
    if left_label is None:
        left_label = '% True-Concept Token Distribution > 99% Non-Concept'
    ax1.set_title(left_label, fontsize=text_size)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.05)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}%'.format(x)))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax1.tick_params(axis='both', labelsize=text_size)
    
    # Customize right subplot
    ax2.set_xlabel('Percent Through Model', fontsize=text_size)
    if right_label is None:
        right_label = f"% True-Concept {'Text Samples' if is_text_dataset else 'Images'} w/ a Token > 99% Non-Concept"
    ax2.set_title(right_label, fontsize=text_size)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1.05)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}%'.format(x)))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax2.tick_params(axis='both', labelsize=text_size)
    
    # Add legend
    if show_legend:
        # Get unique handles and labels from first subplot
        handles, labels = ax1.get_legend_handles_labels()
        # Remove duplicates while preserving order
        seen = set()
        unique_handles = []
        unique_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        
        # Sort handles and labels to group by model (each column = one model)
        sorted_items = []
        for model in all_models:
            for concept_type in sorted(all_concept_types):
                concept_display = 'LinSep' if concept_type == 'linsep' else concept_type.capitalize()
                target_label = f"{model} - {concept_display}"
                for h, l in zip(unique_handles, unique_labels):
                    if l == target_label:
                        sorted_items.append((h, l))
                        break
        
        # Extract sorted handles and labels
        sorted_handles = [item[0] for item in sorted_items]
        sorted_labels = [item[1] for item in sorted_items]
        
        # Calculate number of columns (number of models)
        ncol = len(all_models)
        
        # Create legend below the plots with 2 rows
        fig.legend(sorted_handles, sorted_labels, loc='lower center', 
                  bbox_to_anchor=(0.5, -0.15), ncol=ncol, 
                  frameon=True, fancybox=True, framealpha=0.8, fontsize=legend_size)
    
    # Overall title
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)
    
    # Adjust subplot spacing to bring plots closer together
    plt.subplots_adjust(wspace=0.15)  # Reduce horizontal space between subplots
    plt.tight_layout()
    
    # Save if path provided or save_file is True
    if save_file:
        # Auto-generate save path with dataset name
        save_dir = "../Figs/Paper_Figs/"
        os.makedirs(save_dir, exist_ok=True)
        # Include dataset name in the filename
        save_path = os.path.join(save_dir, f"{dataset}_gt_mass_detection_overlap.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    elif save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_detection_rates_per_concept(
    results: Dict,
    dataset: str,
    model: str,
    concept_type: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    text_size: float = 9.5,
    legend_size: int = 9,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
) -> plt.Figure:
    """
    Plot detection rates for a single model and concept type, showing both average and per-concept lines.
    
    Args:
        results: Results dictionary from compute_overlap_across_layers_data
        dataset: Dataset name to plot
        model: Model name to plot
        concept_type: Concept type to plot
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple (default: (8, 6))
        text_size: Font size for axis labels and ticks (default: 9.5)
        legend_size: Font size for legend text (default: 9)
        title: Custom title for the plot (default: auto-generated)
        ylabel: Custom y-axis label (default: "Detection Rate")
        xlabel: Custom x-axis label (default: "Percent Through Model")
        
    Returns:
        matplotlib Figure object
    """
    # Filter results for the specified dataset, model, and concept type
    filtered_data = None
    for key, data in results.items():
        if (data['dataset'] == dataset and 
            data['model'] == model and 
            data['concept_type'] == concept_type):
            filtered_data = data
            break
    
    if filtered_data is None:
        raise ValueError(f"No results found for dataset='{dataset}', model='{model}', concept_type='{concept_type}'")
    
    # Extract data
    percentthrus = filtered_data['percentthrus']
    avg_detection = filtered_data['detection_rates']
    per_concept_detection = filtered_data.get('detection_rates_per_concept', {})
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot per-concept lines (light blue, thin, transparent)
    if per_concept_detection:
        for concept_name, concept_values in per_concept_detection.items():
            if concept_values and not all(np.isnan(v) for v in concept_values):
                ax.plot(
                    percentthrus[:len(concept_values)],
                    concept_values,
                    color='lightblue',
                    alpha=0.3,
                    linewidth=1,
                    zorder=5
                )
        
        # Add one representative line to legend
        ax.plot([], [], color='lightblue', alpha=0.5, linewidth=3, label='Per-Concept')
    
    # Plot average line (dark blue with dots)
    if avg_detection and not all(np.isnan(v) for v in avg_detection):
        ax.plot(
            percentthrus[:len(avg_detection)],
            avg_detection,
            color='darkblue',
            marker='o',
            markersize=6,
            linewidth=2,
            label='Average',
            zorder=10
        )
    
    # Set labels and title
    if xlabel is None:
        xlabel = "% Through Model"
    if ylabel is None:
        ylabel = ""  # No y-label by default
    if title is None:
        # Determine if this is a text or image dataset
        text_datasets = ['Sarcasm', 'iSarcasm', 'GoEmotions', 'IMDB', 'iSarcasmEval', 'Stanford-Tree-Bank', 'jailbreak']
        is_text_dataset = dataset in text_datasets
        
        if is_text_dataset:
            title = '% True-Concept Text Samples w/ a Token\n> 99% Non-Concept Token Distr'
        else:
            title = '% True-Concept Images w/ a Token\n> 99% Non-Concept Token Distr'
    
    ax.set_xlabel(xlabel, fontsize=text_size)
    if ylabel:  # Only set if not empty
        ax.set_ylabel(ylabel, fontsize=text_size)
    ax.set_title(title, fontsize=text_size)
    
    # Set axis properties
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])  # Y ticks every 25%
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}%'.format(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.tick_params(axis='both', labelsize=text_size)
    
    # Add legend
    ax.legend(fontsize=legend_size, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {os.path.abspath(save_path)}")
    
    return fig
