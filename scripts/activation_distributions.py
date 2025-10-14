#!/usr/bin/env python3
"""
Script to analyze activation distributions for patches/tokens based on concept presence.
Can analyze:
1. Patches from samples WITHOUT the concept (--mode without)
2. Patches from samples WITH the concept (--mode with)
3. Patches from samples WITH the concept, with patch-level breakdown (--mode with_patch_level)

Usage: 
  python scripts/activation_distributions.py                                    # Run all combinations
  python scripts/activation_distributions.py --dataset CLEVR                   # Run only CLEVR
  python scripts/activation_distributions.py --model CLIP --dataset CLEVR      # Run only CLEVR CLIP
  python scripts/activation_distributions.py --concept_type avg               # Run only avg concepts
  python scripts/activation_distributions.py --mode with_patch_level          # Run with patch-level breakdown
"""

import os
import sys
import torch
import argparse
from tqdm import tqdm
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.memory_management_utils import ChunkedActivationLoader
from utils.activation_distributions_utils import (
    get_activation_distributions_for_non_concept_samples,
    get_activation_distributions_for_with_concept_samples,
    get_activation_distributions_with_patch_level,
    plot_activation_distributions
)

# Global configurations (same as all_detection_stats.py)
MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Qwen', ('text', 'text3'))]
DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
SAMPLE_TYPES = [('patch', 1000), ('cls', 50)]
CONCEPT_TYPES = ['avg', 'linsep', 'kmeans', 'linsep_kmeans']
SCRATCH_DIR = ''

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCENT_THRU_MODEL = 100


def get_con_label_and_acts_file(model_name, concept_type, sample_type='patch', percent_thru_model=100, n_clusters=500):
    """
    Get concept label and activation file name following the naming conventions.
    """
    if concept_type == 'avg':
        con_label = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}'
        acts_file = f"cosine_similarities_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'linsep':
        con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}'
        acts_file = f"dists_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'kmeans':
        con_label = f"{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
        acts_file = f"cosine_similarities_kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif concept_type == 'linsep_kmeans':
        con_label = f"{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
        acts_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")
    
    return con_label, acts_file


def main(dataset_name, model_name, concept_type, mode='without', percent_thru_model=100, n_clusters=500):
    """Main function to compute activation distributions based on concept presence.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        concept_type: Type of concepts (avg, linsep, kmeans, linsep_kmeans)
        mode: Analysis mode - 'without', 'with', or 'with_patch_level'
        percent_thru_model: Percentage through model
        n_clusters: Number of clusters for kmeans
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine model input size based on model and dataset
    if dataset_name in ['Sarcasm', 'iSarcasm', 'GoEmotions', 'Stanford-Tree-Bank']:
        # Text datasets
        if model_name == 'Llama':
            model_input_size = ('text', 'text')
        elif model_name == 'Qwen':
            model_input_size = ('text', 'text3')
        else:
            raise ValueError(f"Model {model_name} not compatible with text dataset {dataset_name}")
    else:
        # Vision datasets
        if model_name == 'CLIP':
            model_input_size = (224, 224)
        elif model_name == 'Llama':
            model_input_size = (560, 560)
        else:
            raise ValueError(f"Model {model_name} not compatible with vision dataset {dataset_name}")
    
    # Get concept label and activation file name
    sample_type = 'patch'  # Always use patch-level analysis
    con_label, acts_file = get_con_label_and_acts_file(
        model_name, concept_type, sample_type, percent_thru_model, n_clusters
    )
    
    print(f"Concept label: {con_label}")
    
    # Load ground truth samples per concept
    # Ground truth files are in Experiments/GT_Samples/ directory
    gt_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
    if not os.path.exists(gt_file):
        # Try with string format for text datasets
        gt_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_text_text.pt"
    
    print(f"Loading ground truth from: {gt_file}")
    gt_samples_per_concept = torch.load(gt_file)
    
    # Load activation loader
    print(f"Loading activations from: {acts_file}")
    scratch_dir = SCRATCH_DIR
    
    # ChunkedActivationLoader automatically determines the correct directory based on filename
    # Files with 'dists_' or 'linsep' -> Distances/, others -> Cosine_Similarities/
    
    # First try scratch directory
    try:
        act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=scratch_dir, device=device)
    except FileNotFoundError:
        # Try local Experiments directory
        try:
            act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir='.', device=device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Activation file not found in either scratch or local directory: {acts_file}\n"
                                   f"Please run the compute_activations.py script first to generate activation files.")
    
    # Get activation distributions based on mode
    if mode == 'without':
        print("Computing activation distributions for samples WITHOUT concepts...")
        activation_distributions = get_activation_distributions_for_non_concept_samples(
            act_loader=act_loader,
            gt_samples_per_concept=gt_samples_per_concept,
            dataset_name=dataset_name,
            model_input_size=model_input_size,
            device=device,
            sample_type=sample_type
        )
    elif mode == 'with':
        print("Computing activation distributions for samples WITH concepts...")
        activation_distributions = get_activation_distributions_for_with_concept_samples(
            act_loader=act_loader,
            gt_samples_per_concept=gt_samples_per_concept,
            dataset_name=dataset_name,
            model_input_size=model_input_size,
            device=device,
            sample_type=sample_type
        )
    elif mode == 'with_patch_level':
        print("Computing activation distributions for samples WITH concepts (with patch-level breakdown)...")
        
        # Load ground truth patches per concept for patch-level analysis
        gt_patches_file = f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
        if not os.path.exists(gt_patches_file):
            print(f"Warning: Patch-level ground truth not found at {gt_patches_file}")
            print("Falling back to regular WITH concept analysis")
            activation_distributions = get_activation_distributions_for_with_concept_samples(
                act_loader=act_loader,
                gt_samples_per_concept=gt_samples_per_concept,
                dataset_name=dataset_name,
                model_input_size=model_input_size,
                device=device,
                sample_type=sample_type
            )
        else:
            gt_patches_per_concept = torch.load(gt_patches_file, weights_only=False)
            activation_distributions = get_activation_distributions_with_patch_level(
                act_loader=act_loader,
                gt_samples_per_concept=gt_samples_per_concept,
                gt_patches_per_concept=gt_patches_per_concept,
                dataset_name=dataset_name,
                model_input_size=model_input_size,
                device=device,
                sample_type=sample_type
            )
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'without', 'with', or 'with_patch_level'")
    
    # Save results
    output_dir = f"activation_distributions/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create appropriate filename based on mode
    if mode == 'without':
        output_file = os.path.join(output_dir, f"activation_distributions_{con_label}_test.pt")
    elif mode == 'with':
        output_file = os.path.join(output_dir, f"activation_distributions_{con_label}_with_concept_test.pt")
    elif mode == 'with_patch_level':
        output_file = os.path.join(output_dir, f"activation_distributions_{con_label}_with_patch_level_test.pt")
    
    torch.save(activation_distributions, output_file)
    print(f"Saved activation distributions to: {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    num_concepts_shown = 0
    
    if mode == 'with_patch_level':
        # Special handling for patch-level breakdown
        for concept, dist_info in activation_distributions.items():
            if num_concepts_shown < 3:  # Show first 3 concepts
                print(f"\n{concept}:")
                
                # Overall WITH concept stats
                if 'mean_distribution' in dist_info:
                    print(f"  Overall (all patches from images WITH concept):")
                    print(f"    Number of images: {dist_info['num_samples']}")
                    print(f"    Total patches: {dist_info['total_patches']}")
                
                # Patches that contain the concept
                if 'patch_with_concept' in dist_info:
                    patch_info = dist_info['patch_with_concept']
                    print(f"  Patches that CONTAIN the concept:")
                    print(f"    Total patches: {patch_info['total_patches']}")
                
                # Patches that don't contain the concept
                if 'patch_without_concept' in dist_info:
                    patch_info = dist_info['patch_without_concept']
                    print(f"  Patches that DON'T contain the concept:")
                    print(f"    Total patches: {patch_info['total_patches']}")
                
                num_concepts_shown += 1
    else:
        # Regular mode (with or without)
        for concept, dist_info in activation_distributions.items():
            if 'mean_distribution' in dist_info and num_concepts_shown < 5:  # Show first 5 concepts
                mean_dist = dist_info['mean_distribution']
                print(f"\n{concept}:")
                if mode == 'without':
                    print(f"  Number of samples WITHOUT concept: {dist_info['num_samples']}")
                else:
                    print(f"  Number of samples WITH concept: {dist_info['num_samples']}")
                print(f"  Total patches analyzed: {dist_info['total_patches']}")
                
                # Calculate mean activation value
                activation_range = dist_info['activation_range']
                num_bins = dist_info['num_bins']
                bin_edges = torch.linspace(activation_range[0], activation_range[1], num_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                mean_activation = (bin_centers * mean_dist).sum().item()
                
                print(f"  Mean activation value: {mean_activation:.4f}")
                print(f"  Distribution range: [{mean_dist.min().item():.4f}, {mean_dist.max().item():.4f}]")
                num_concepts_shown += 1
    
    if len(activation_distributions) > num_concepts_shown:
        print(f"\n... and {len(activation_distributions) - num_concepts_shown} more concepts")
    
    print(f"\nTotal concepts analyzed: {len(activation_distributions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute activation distributions based on concept presence")
    parser.add_argument("--dataset", type=str, help="Filter to specific dataset (e.g., CLEVR, Coco)")
    parser.add_argument("--model", type=str, help="Filter to specific model (e.g., CLIP, Llama)")
    parser.add_argument("--concept_type", type=str, help="Filter to specific concept type (avg, linsep, kmeans, linsep_kmeans)")
    parser.add_argument("--sample_type", type=str, help="Filter to specific sample type (patch, cls)")
    parser.add_argument("--mode", type=str, default="all", choices=['without', 'with', 'with_patch_level', 'all'],
                        help="Analysis mode: 'without' (samples without concept), 'with' (samples with concept), 'with_patch_level' (with patch breakdown), 'all' (run all modes)")
    parser.add_argument("--percent_thru_model", type=int, default=100, 
                        help="Percentage through model (default: 100)")
    
    args = parser.parse_args()
    
    # Filter configurations based on arguments
    models = MODELS if not args.model else [(m, s) for m, s in MODELS if m == args.model]
    datasets = DATASETS if not args.dataset else [args.dataset]
    concept_types = CONCEPT_TYPES if not args.concept_type else [args.concept_type]
    sample_types = SAMPLE_TYPES if not args.sample_type else [(s, n) for s, n in SAMPLE_TYPES if s == args.sample_type]
    
    # Create experiment configurations
    experiment_configs = product(models, datasets, sample_types, concept_types)
    
    # Determine which modes to run
    if args.mode == 'all':
        modes_to_run = ['without', 'with', 'with_patch_level']
    else:
        modes_to_run = [args.mode]
    
    successful = []
    failed = []
    
    for (model_name, model_input_size), dataset_name, (sample_type, n_clusters), concept_type in experiment_configs:
        # Skip invalid dataset-input size combinations (same logic as all_detection_stats.py)
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
        
        # Run for each mode
        for mode in modes_to_run:
            # Skip patch-level mode for text datasets or cls sample type
            if mode == 'with_patch_level' and (dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions'] or sample_type == 'cls'):
                continue
                
            print(f"\n{'='*60}")
            print(f"Processing: {dataset_name} {model_name} {concept_type} {sample_type} (mode: {mode})")
            print('='*60)
            
            try:
                main(dataset_name, model_name, concept_type, mode, args.percent_thru_model, n_clusters)
                successful.append((dataset_name, model_name, concept_type, sample_type, mode))
                print(f"✓ SUCCESS: {dataset_name} {model_name} {concept_type} {sample_type} (mode: {mode})")
            except Exception as e:
                failed.append((dataset_name, model_name, concept_type, sample_type, mode, str(e)))
                print(f"✗ FAILED: {dataset_name} {model_name} {concept_type} {sample_type} (mode: {mode})")
                print(f"  Error: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    mode_desc = {
        'without': "WITHOUT CONCEPTS",
        'with': "WITH CONCEPTS", 
        'with_patch_level': "WITH CONCEPTS (PATCH-LEVEL BREAKDOWN)",
        'all': "ALL MODES"
    }
    print(f"ACTIVATION DISTRIBUTIONS ({mode_desc.get(args.mode, args.mode)}) - SUMMARY")
    print('='*60)
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✓ Successfully processed:")
        for item in successful:
            if len(item) == 5:  # New format with mode
                dataset, model, concept_type, sample_type, mode = item
                print(f"  - {dataset} {model} {concept_type} {sample_type} (mode: {mode})")
            else:  # Old format without mode
                dataset, model, concept_type, sample_type = item
                print(f"  - {dataset} {model} {concept_type} {sample_type}")
    
    if failed:
        print(f"\n✗ Failed to process:")
        for item in failed:
            if len(item) == 6:  # New format with mode
                dataset, model, concept_type, sample_type, mode, error = item
                print(f"  - {dataset} {model} {concept_type} {sample_type} (mode: {mode}): {error[:100]}...")
            else:  # Old format without mode
                dataset, model, concept_type, sample_type, error = item
                print(f"  - {dataset} {model} {concept_type} {sample_type}: {error[:100]}...")