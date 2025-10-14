#!/usr/bin/env python3
"""
Script to compute inversion statistics for concept detection across models and datasets.

Usage examples:
    # Process a single dataset
    python scripts/all_inversion_stats.py --dataset CLEVR
    
    # Process multiple specific datasets
    python scripts/all_inversion_stats.py --datasets CLEVR Coco
    
    # Process specific dataset with specific model
    python scripts/all_inversion_stats.py --dataset CLEVR --model CLIP
    
    # Process specific dataset with specific sample type
    python scripts/all_inversion_stats.py --dataset Coco --sample-type patch
    
    # List available datasets
    python scripts/all_inversion_stats.py --list-datasets
    
    # Run default configuration (processes datasets specified in DATASETS variable)
    python scripts/all_inversion_stats.py
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os
from collections import defaultdict
from itertools import product
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gc
import time

from utils.compute_concepts_utils import gpu_kmeans, compute_linear_separators, compute_avg_concept_vectors
from utils.activation_utils import compute_cosine_sims, compute_signed_distances
from utils.unsupervised_utils import compute_detection_metrics_over_percentiles_allpairs, find_best_clusters_per_concept_from_detectionmetrics, filter_and_save_best_clusters, get_matched_concepts_and_data, \
compute_concept_thresholds_over_percentiles_all_pairs
from utils.superdetector_inversion_utils import all_superdetector_inversions_across_percentiles, detect_then_invert_superdetector_twostage_metrics, find_optimal_twostage_superdetector_thresholds, detect_then_invert_twostage_superdetector_with_optimal_thresholds
from utils.quant_concept_evals_utils import find_optimal_detect_invert_thresholds, compute_concept_thresholds_over_percentiles, compute_detection_metrics_over_percentiles, detect_then_invert_metrics_over_percentiles, detect_then_invert_with_optimal_thresholds
from utils.gt_concept_segmentation_utils import map_concepts_to_patch_indices, map_concepts_to_image_indices
from utils.memory_management_utils import ChunkedEmbeddingLoader, MatchedConceptActivationLoader
from utils.filter_datasets_utils import filter_concept_dict
from utils.default_percentthrumodels import ALL_PERCENTTHRUMODELS, get_model_default_percentthrumodels
# Removed unused import: create_matched_concept_loader


MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Gemma', ('text', 'text2')), ('Qwen', ('text', 'text3'))]
DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
SAMPLE_TYPES = [('patch', 1000)]


DEVICE = "cuda"
PERCENT_THRU_MODEL = 100  # Default value, can be overridden by command line
SCRATCH_DIR = ''
PERCENTILES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BATCH_SIZE = 5000  # Further increased for maximum GPU utilization

    
def get_files_for_avg(model_name, sample_type, percent_thru_model):
    con_label = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}'
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    concepts_file = f'avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt'
    cossim_file = f"cosine_similarities_{concepts_file[:-3]}.pt"
    return con_label, embeddings_file, concepts_file, cossim_file


def get_files_for_linsep(model_name, sample_type, percent_thru_model):
    con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}'
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    concepts_file = f'linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt'
    cossim_file = f"dists_{concepts_file[:-3]}.pt"
    return con_label, embeddings_file, concepts_file, cossim_file

def get_files_for_reg_kmeans(model_name, n_clusters, sample_type, percent_thru_model):
    """
    Constructs filenames and labels for regular k-means concept pipeline.

    Args:
        model_name (str): Name of the model (e.g., 'vit_b_32')
        n_clusters (int): Number of clusters used in k-means
        sample_type (str): Type of embedding source (e.g., 'patch', 'cls')
        percent_thru_model (int): Percentage through model (e.g., 100)

    Returns:
        tuple: (con_label, embeddings_file, concepts_file, cossim_file)
    """
    con_label = f"{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    concepts_file = f"kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    cossim_file = f"cosine_similarities_{concepts_file[:-3]}.pt"
    return con_label, embeddings_file, concepts_file, cossim_file


def get_files_for_linsep_kmeans(model_name, n_clusters, sample_type, percent_thru_model):
    """
    Constructs filenames and labels for linear separator k-means concept pipeline.

    Args:
        model_name (str): Name of the model (e.g., 'vit_b_32')
        n_clusters (int): Number of clusters used in k-means
        sample_type (str): Type of embedding source (e.g., 'patch', 'cls')
        percent_thru_model (int): Percentage through model (e.g., 100)

    Returns:
        tuple: (con_label, embeddings_file, dists_file, dists_path)
    """
    con_label = f"{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    concepts_file = f"kmeans_{n_clusters}_linsep_concepts_{embeddings_file}"
    dists_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{embeddings_file[:-3]}.pt"
    return con_label, embeddings_file, concepts_file, dists_file


def get_files_for_sae(model_name, sample_type, percent_thru_model):
    """
    Constructs filenames and labels for SAE (Sparse Autoencoder) concept pipeline.
    Note: SAE is only available for CLIP model with patch embeddings.

    Args:
        model_name (str): Name of the model (must be 'CLIP')
        sample_type (str): Type of embedding source (must be 'patch')
        percent_thru_model (int): Percentage through model (e.g., 100)

    Returns:
        tuple: (con_label, embeddings_file, concepts_file, acts_file) or None if not applicable
    """
    if model_name != 'CLIP' or sample_type != 'patch':
        return None
    
    # SAE uses patchsae for CLIP patch embeddings
    sae_name = 'patchsae'
    con_label = f"{model_name}_sae_{sae_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}"
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    # SAE doesn't have a traditional concepts file, but we'll use a placeholder
    concepts_file = f"sae_{sae_name}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    acts_file = f"sae_acts_{sae_name}_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    return con_label, embeddings_file, concepts_file, acts_file


def get_all_files(model_name, sample_type, n_clusters, percent_thru_model):
    all_files = []
    all_files.append(get_files_for_avg(model_name, sample_type, percent_thru_model))
    all_files.append(get_files_for_linsep(model_name, sample_type, percent_thru_model))
    all_files.append(get_files_for_reg_kmeans(model_name, n_clusters, sample_type, percent_thru_model))
    all_files.append(get_files_for_linsep_kmeans(model_name, n_clusters, sample_type, percent_thru_model))
    return all_files

def get_cluster_labels(dataset_name, kmeans_concept_file):
    print("loading gt clusters from kmeans")
    train_cluster_to_samples = torch.load(f'Concepts/{dataset_name}/train_samples_{kmeans_concept_file}')
    test_cluster_to_samples = torch.load(f'Concepts/{dataset_name}/test_samples_{kmeans_concept_file}')
    cluster_to_samples = defaultdict(list)
    for cluster, samples in train_cluster_to_samples.items():
        cluster_to_samples[cluster].extend(samples)
    for cluster, samples in test_cluster_to_samples.items():
        cluster_to_samples[cluster].extend(samples)
    for cluster in cluster_to_samples:
        cluster_to_samples[cluster] = sorted(cluster_to_samples[cluster])
    cluster_to_samples = dict(cluster_to_samples)
    return cluster_to_samples


def get_act_metrics(dataset_name, acts_file):
    from utils.memory_management_utils import ChunkedActivationLoader
    
    # Ensure we use .pt files, not .csv
    if acts_file.endswith('.csv'):
        acts_file = acts_file.replace('.csv', '.pt')
    
    if 'sae_acts' in acts_file:
        print("Setting up SAE activation loader...")
        loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=SCRATCH_DIR, device=DEVICE)
    elif 'linsep' in acts_file:
        print("Setting up distance loader...")
        loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=SCRATCH_DIR, device=DEVICE)
    else:
        print("Setting up cosine similarity loader...")
        loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=SCRATCH_DIR, device=DEVICE)
    
    if loader.is_chunked:
        print(f"   Found {len(loader.chunk_files)} chunks")
    else:
        print(f"   Single file loader")
    
    return loader

        
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inversion statistics on specified dataset(s)')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process (e.g., CLEVR, Coco, Broden-Pascal)')
    parser.add_argument('--datasets', nargs='+', help='Multiple datasets to process')
    parser.add_argument('--model', type=str, help='Specific model to use (e.g., CLIP, Llama, Qwen)')
    parser.add_argument('--models', nargs='+', help='Multiple models to use')
    parser.add_argument('--sample-type', type=str, choices=['patch', 'cls'], help='Sample type to process')
    parser.add_argument('--concept-types', nargs='+', choices=['avg', 'linsep', 'kmeans', 'linsepkmeans'], 
                        help='Concept types to process (default: all)')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets and exit')
    parser.add_argument('--percentthrumodels', nargs='+', type=int, default=ALL_PERCENTTHRUMODELS, 
                        help=f'List of percentages through model layers to use (default: every 2 layers for all models)')
    
    args = parser.parse_args()
    
    # List available datasets if requested
    if args.list_datasets:
        print("Available datasets:")
        all_datasets = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
        for dataset in all_datasets:
            print(f"  - {dataset}")
        sys.exit(0)
    
    # Determine which datasets to process
    if args.dataset:
        datasets_to_process = [args.dataset]
    elif args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = DATASETS
    
    # Determine which models to process
    all_available_models = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), 
                          ('Gemma', ('text', 'text2')), ('Qwen', ('text', 'text3'))]
    
    if args.models:
        models_to_process = []
        for model_name in args.models:
            found = [(m, s) for m, s in all_available_models if m == model_name]
            if not found:
                print(f"Error: Model '{model_name}' not found")
                sys.exit(1)
            models_to_process.extend(found)
    elif args.model:
        models_to_process = [(m, s) for m, s in MODELS if m == args.model]
        if not models_to_process:
            print(f"Error: Model '{args.model}' not found in configured models")
            sys.exit(1)
    else:
        models_to_process = MODELS
    
    # Get list of percentthrumodels to process
    # If --percentthrumodels is specified, use those values
    # Otherwise, use model-specific defaults based on selected models
    if args.percentthrumodels != ALL_PERCENTTHRUMODELS:  # User specified custom values
        percentthrumodels = args.percentthrumodels
    else:
        # Collect all unique percentthrumodels for the selected models
        percentthrumodels = set()
        for model_name, model_input_size in models_to_process:
            model_defaults = get_model_default_percentthrumodels(model_name, model_input_size)
            percentthrumodels.update(model_defaults)
        percentthrumodels = sorted(list(percentthrumodels))
        print(f"Using model-specific default percentthrumodels: {percentthrumodels}")
    
    # Determine which sample types to process
    if args.sample_type:
        sample_types_to_process = [(s, n) for s, n in SAMPLE_TYPES if s == args.sample_type]
        if not sample_types_to_process:
            print(f"Error: Sample type '{args.sample_type}' not found in configured sample types")
            sys.exit(1)
    else:
        sample_types_to_process = SAMPLE_TYPES
    
    # Validate datasets
    available_datasets = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
    for dataset in datasets_to_process:
        if dataset not in available_datasets:
            print(f"Error: Dataset '{dataset}' not recognized. Available datasets: {', '.join(available_datasets)}")
            sys.exit(1)
    
    # Determine which concept types to process
    if args.concept_types:
        # Map user-friendly names to function names
        concept_type_mapping = {
            'avg': 'avg',
            'linsep': 'linsep',
            'kmeans': 'kmeans_1000',
            'linsepkmeans': 'linsep_kmeans_1000'
        }
        concept_types_to_process = [concept_type_mapping[ct] for ct in args.concept_types]
    else:
        # Default to all concept types
        concept_types_to_process = ['avg', 'linsep', 'kmeans_1000', 'linsep_kmeans_1000']
    
    print(f"Processing datasets: {', '.join(datasets_to_process)}")
    print(f"Using models: {', '.join([m for m, _ in models_to_process])}")
    print(f"Using sample types: {', '.join([s for s, _ in sample_types_to_process])}")
    print(f"Using concept types: {', '.join(concept_types_to_process)}")
    
    # Loop through all percentthrumodels
    for PERCENT_THRU_MODEL in percentthrumodels:
        print(f"\n{'='*60}")
        print(f"Processing with PERCENT_THRU_MODEL = {PERCENT_THRU_MODEL}")
        print(f"{'='*60}\n")
        
        experiment_configs = product(models_to_process, datasets_to_process, sample_types_to_process)
        for (model_name, model_input_size), dataset_name, (sample_type, n_clusters) in experiment_configs:
            # Skip this model if the current percentthrumodel is not in its default list
            model_default_percentiles = get_model_default_percentthrumodels(model_name, model_input_size)
            if PERCENT_THRU_MODEL not in model_default_percentiles:
                continue
            # Skip invalid dataset-input size combinations
            if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue
            if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue
            
            print(f"Processing model {model_name} dataset {dataset_name} sample type {sample_type}")
            #get gt
            gt_patches_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt')
            gt_patches_per_concept_test = torch.load(f'GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt')
            gt_patches_per_concept_cal = torch.load(f'GT_Samples/{dataset_name}/gt_patch_per_concept_cal_inputsize_{model_input_size}.pt')
            
            # Filter to only relevant concepts for this dataset
            gt_patches_per_concept = filter_concept_dict(gt_patches_per_concept, dataset_name)
            gt_patches_per_concept_test = filter_concept_dict(gt_patches_per_concept_test, dataset_name)
            gt_patches_per_concept_cal = filter_concept_dict(gt_patches_per_concept_cal, dataset_name)
            print(f"  Filtered to {len(gt_patches_per_concept)} concepts for {dataset_name}")

            #load embeds using chunked loader
            print("Setting up chunked embedding loader...")
            embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt"
            embedding_loader = ChunkedEmbeddingLoader(dataset_name, embeddings_file, SCRATCH_DIR, DEVICE)
            
            print(f"📊 Embedding info: {embedding_loader.get_embedding_info()}")
    
            all_files = get_all_files(model_name, sample_type, n_clusters, PERCENT_THRU_MODEL)
            
            # Filter files based on selected concept types
            filtered_files = []
            for con_label, embeddings_file, concepts_file, acts_file in all_files:
                # Check if this concept type should be processed
                if 'avg' in con_label and 'avg' in concept_types_to_process:
                    filtered_files.append((con_label, embeddings_file, concepts_file, acts_file))
                elif 'linsep' in con_label and 'kmeans' not in con_label and 'linsep' in concept_types_to_process:
                    filtered_files.append((con_label, embeddings_file, concepts_file, acts_file))
                elif 'kmeans' in con_label and 'linsep' not in con_label and 'kmeans_1000' in concept_types_to_process:
                    filtered_files.append((con_label, embeddings_file, concepts_file, acts_file))
                elif 'kmeans' in con_label and 'linsep' in con_label and 'linsep_kmeans_1000' in concept_types_to_process:
                    filtered_files.append((con_label, embeddings_file, concepts_file, acts_file))
            
            for con_label, _, concepts_file, acts_file in filtered_files:  
                print(con_label)
                
                # Check if activation file exists (important for SAE which may not be generated for all configs)
                if 'sae' in con_label:
                    acts_path = os.path.join(SCRATCH_DIR, 'SAE_Acts', dataset_name, acts_file)
                elif 'dists_' in acts_file:
                    acts_path = os.path.join(SCRATCH_DIR, 'Distances', dataset_name, acts_file)
                else:
                    acts_path = os.path.join(SCRATCH_DIR, 'Cosine_Similarities', dataset_name, acts_file)
                
                # Check for either direct file or chunked files
                chunks_info_path = acts_path.replace('.pt', '_chunks_info.json')
                chunk_0_path = acts_path.replace('.pt', '_chunk_0.pt')
                    
                #get act metrics loader
                print(f"\nLoading activation metrics: {acts_file}")
                act_metrics = get_act_metrics(dataset_name, acts_file)
                
                # Process based on concept type
                if 'kmeans' in con_label or 'sae' in con_label:  # Unsupervised concepts
                    if 'sae' in con_label:
                        num_sae_units = len(act_metrics.columns)
                        # Create a dictionary mapping SAE unit indices to identity vectors
                        # Create identity vectors one at a time to avoid creating full matrix
                        concepts = {}
                        for i in range(num_sae_units):
                            vec = torch.zeros(num_sae_units, device=DEVICE)
                            vec[i] = 1.0
                            concepts[i] = vec
                    else:
                        concepts = torch.load(f'Concepts/{dataset_name}/{concepts_file}')
                    
                    # Get matched loader and data for unsupervised concepts
                    matched_loader, matched_gt_patches_per_concept_cal, matched_gt_patches_per_concept_test, \
                    matched_gt_patches_per_concept, matched_concepts = get_matched_concepts_and_data(dataset_name,
                                                                                    con_label,
                                                                                    None,  # This parameter is ignored anyway
                                                                                    gt_patches_per_concept_cal,
                                                                                    gt_patches_per_concept_test,
                                                                                    gt_patches_per_concept,
                                                                                    concepts=concepts,
                                                                                    scratch_dir=SCRATCH_DIR
                                                                                    )
                    
                    # Filter matched concepts to only those relevant for this dataset
                    matched_concepts = filter_concept_dict(matched_concepts, dataset_name)
                    matched_gt_patches_per_concept_cal = filter_concept_dict(matched_gt_patches_per_concept_cal, dataset_name)
                    matched_gt_patches_per_concept_test = filter_concept_dict(matched_gt_patches_per_concept_test, dataset_name)
                    matched_gt_patches_per_concept = filter_concept_dict(matched_gt_patches_per_concept, dataset_name)
                    
                    # Update the matched loader to only include filtered concepts
                    if hasattr(matched_loader, 'concept_to_cluster'):
                        # Filter the concept mappings in the loader
                        filtered_concept_to_cluster = {c: cluster_id for c, cluster_id in matched_loader.concept_to_cluster.items() 
                                                       if c in matched_concepts}
                        matched_loader.concept_to_cluster = filtered_concept_to_cluster
                        matched_loader.cluster_to_concept = {cluster_id: c for c, cluster_id in filtered_concept_to_cluster.items()}
                        matched_loader.available_clusters = list(matched_loader.cluster_to_concept.keys())
                        matched_loader.matched_concepts = list(filtered_concept_to_cluster.keys())
                else:  # Supervised concepts
                    concepts = torch.load(f'Concepts/{dataset_name}/{concepts_file}')
                    
                    # Filter concepts to match the filtered ground truth
                    concepts = filter_concept_dict(concepts, dataset_name)
                    
                    # For supervised concepts, use the activation loader directly
                    matched_loader = act_metrics
                    matched_concepts = concepts
                    matched_gt_patches_per_concept_cal = gt_patches_per_concept_cal
                    matched_gt_patches_per_concept_test = gt_patches_per_concept_test
                    matched_gt_patches_per_concept = gt_patches_per_concept
                
                # Run the evaluation pipeline
                print("="*80)
                print(f"Running evaluation pipeline for {con_label}")
                print(f"Number of concepts to process: {len(matched_concepts)}")
                print(f"Dataset: {dataset_name}, Model: {model_name}")
                print("="*80)
                
                # Common parameters for all functions
                eval_params = {
                    'device': DEVICE,
                    'dataset_name': dataset_name,
                    'model_input_size': model_input_size,
                    'con_label': con_label,
                    'patch_size': 14,
                    'agglomerate_type': 'avg'
                }
                
                # Step 1: Regular inversion optimization (using pre-computed detection percentiles)
                print("\n[Step 1/7] Finding optimal inversion percentiles for regular method...")
                print(f"Processing {len(matched_concepts)} concepts with {len(PERCENTILES)} percentiles")
                
                detect_then_invert_metrics_over_percentiles(
                    PERCENTILES, PERCENTILES, matched_loader, matched_concepts, 
                    matched_gt_patches_per_concept, matched_gt_patches_per_concept_cal,
                    eval_params['device'], eval_params['dataset_name'], 
                    eval_params['model_input_size'], eval_params['con_label'],
                    all_object_patches=None, 
                    patch_size=eval_params['patch_size'],
                    embedding_loader=embedding_loader
                )
                
                # Step 2: Find optimal inversion thresholds for regular method
                print("\n[Step 2/7] Finding optimal inversion thresholds for regular method...")
                find_optimal_detect_invert_thresholds(PERCENTILES, 
                                                    eval_params['dataset_name'], eval_params['con_label'])
                
                # Step 3: Test set evaluation with optimal regular thresholds
                print("\n[Step 3/7] Evaluating regular method on test set...")
                detect_then_invert_with_optimal_thresholds(
                    matched_loader, matched_concepts, matched_gt_patches_per_concept, 
                    matched_gt_patches_per_concept_test, eval_params['device'], 
                    eval_params['dataset_name'], eval_params['model_input_size'], 
                    eval_params['con_label'], embedding_loader=embedding_loader
                )
                
                # Step 4: Compute superdetector inversions (prerequisite for two-stage method)
                print("\n[Step 4/7] Computing superdetector inversions...")
                
                # Always use ground truth concept names for consistency
                # Filter to only concepts that exist in the ground truth for this dataset
                concept_names = list(matched_concepts.keys())
                
                # For k-means, we need to ensure concepts are available in the matched loader
                if hasattr(matched_loader, 'concept_to_cluster'):
                    # Only include concepts that have been matched to clusters
                    available_concepts = set(matched_loader.concept_to_cluster.keys())
                    filtered_concept_names = [c for c in concept_names 
                                            if c in matched_gt_patches_per_concept_cal 
                                            and c in available_concepts]
                else:
                    # For supervised methods, just check ground truth
                    filtered_concept_names = [c for c in concept_names 
                                            if c in matched_gt_patches_per_concept_cal]
                
                print(f"Using concept names: {len(filtered_concept_names)} concepts (filtered from {len(concept_names)})")
                
                all_superdetector_inversions_across_percentiles(
                    PERCENTILES, eval_params['agglomerate_type'], embedding_loader, matched_loader,
                    filtered_concept_names, matched_gt_patches_per_concept_cal, 
                    eval_params['dataset_name'], eval_params['model_input_size'], 
                    eval_params['con_label'], eval_params['device'], 
                    patch_size=eval_params['patch_size'], local=True, split='cal',
                    scratch_dir=SCRATCH_DIR
                )
                
                # Step 5: Two-stage superdetector calibration metrics
                print("\n[Step 5/7] Evaluating two-stage superdetector metrics on calibration set...")
                
                # Always use matched_concepts which contains ground truth names
                concepts_for_eval = matched_concepts
                
                detect_then_invert_superdetector_twostage_metrics(
                    PERCENTILES, matched_loader, concepts_for_eval, 
                    matched_gt_patches_per_concept, matched_gt_patches_per_concept_cal,
                    embedding_loader, eval_params['device'], eval_params['dataset_name'], 
                    eval_params['model_input_size'], eval_params['con_label'],
                    all_object_patches=None, patch_size=eval_params['patch_size'], 
                    agglomerate_type=eval_params['agglomerate_type'], split='cal',
                    scratch_dir=SCRATCH_DIR
                )
                
                # Step 6: Find optimal two-stage thresholds
                print("\n[Step 6/7] Finding optimal two-stage superdetector thresholds...")
                find_optimal_twostage_superdetector_thresholds(
                    PERCENTILES, eval_params['dataset_name'], 
                    eval_params['con_label'], eval_params['model_input_size'], 
                    agglomerate_type=eval_params['agglomerate_type']
                )
                
                # Step 7: Test set evaluation with optimal two-stage thresholds
                print("\n[Step 7/7] Evaluating two-stage superdetector on test set...")
                detect_then_invert_twostage_superdetector_with_optimal_thresholds(
                    matched_loader, concepts_for_eval, matched_gt_patches_per_concept, 
                    matched_gt_patches_per_concept_test, embedding_loader, eval_params['device'], 
                    eval_params['dataset_name'], eval_params['model_input_size'], 
                    eval_params['con_label'], agglomerate_type=eval_params['agglomerate_type'],
                    split='test', scratch_dir=SCRATCH_DIR
                )
                
                # Clean up loader to free memory
                if 'matched_loader' in locals() and hasattr(matched_loader, 'close'):
                    matched_loader.close()
                del matched_loader
            if 'act_metrics' in locals():
                if hasattr(act_metrics, 'close'):
                    act_metrics.close()
                del act_metrics
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # Clean up embedding loader
        if hasattr(embedding_loader, 'close'):
            embedding_loader.close()
        del embedding_loader
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
