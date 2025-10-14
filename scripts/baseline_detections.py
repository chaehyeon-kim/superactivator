import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import sys
import os
import argparse
import gc
from collections import defaultdict
from itertools import product
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.compute_concepts_utils import gpu_kmeans, compute_linear_separators
from utils.activation_utils import compute_cosine_sims, compute_signed_distances
from utils.memory_management_utils import ChunkedActivationLoader
from utils.filter_datasets_utils import filter_concept_dict
from utils.default_percentthrumodels import ALL_PERCENTTHRUMODELS, get_model_default_percentthrumodels
from utils.baseline_detection_utils import (compute_aggregated_activation_thresholds_over_percentiles,
                                           compute_aggregated_detection_metrics_over_percentiles,
                                           find_best_aggregated_detection_percentiles_cal,
                                           compute_max_activation_thresholds_over_percentiles,
                                           compute_baseline_detection_metrics_over_percentiles,
                                           find_best_baseline_detection_percentiles_cal,
                                           compute_aggregated_activation_thresholds_over_percentiles_all_pairs,
                                           compute_aggregated_detection_metrics_over_percentiles_allpairs,
                                           find_best_clusters_per_concept_from_aggregated_detectionmetrics,
                                           filter_and_save_best_clusters_aggregated,
                                           save_best_percentiles_for_kmeans)

# Try to import precomputation utilities if available
try:
    from utils.baseline_detection_utils_chunked import precompute_all_aggregations
    PRECOMPUTE_AVAILABLE = True
except ImportError:
    PRECOMPUTE_AVAILABLE = False

MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Gemma', ('text', 'text2')), ('Qwen', ('text', 'text3'))]

DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
SAMPLE_TYPES = [('patch', 1000)]  # Only patch/token analysis for baseline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCENT_THRU_MODEL = 100  # Default value, can be overridden by command line
SCRATCH_DIR = ''
PERCENTILES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BATCH_SIZE = 300


def get_files_for_avg(model_name, n_clusters, sample_type, percent_thru_model):
    con_label = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}'
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    concepts_file = f'avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt'
    cossim_file = f"cosine_similarities_{concepts_file[:-3]}.pt"
    return con_label, embeddings_file, concepts_file, cossim_file


def get_files_for_linsep(model_name, n_clusters, sample_type, percent_thru_model):
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


def get_files_for_sae(model_name, n_clusters, sample_type, percent_thru_model):
    """
    Constructs filenames and labels for SAE (Sparse Autoencoder) concept pipeline.
    Note: Only CLIP and Gemma models have SAE support at specific percentthrumodels.

    Args:
        model_name (str): Name of the model (e.g., 'CLIP', 'Gemma')
        n_clusters (int): Not used for SAE but kept for consistency
        sample_type (str): Type of embedding source (e.g., 'patch', 'cls')
        percent_thru_model (int): Must be 92 for CLIP or 81 for Gemma

    Returns:
        tuple: (con_label, embeddings_file, concepts_file, acts_file) or None if model doesn't support SAE
    """
    # Only CLIP and Gemma have SAE support at specific percentthrumodels
    if model_name == 'CLIP' and percent_thru_model != 92:
        return None
    elif model_name == 'Gemma' and percent_thru_model != 81:
        return None
    elif model_name not in ['CLIP', 'Gemma']:
        return None
        
    con_label = f"{model_name}_sae_{sample_type}_dense"
    # SAE doesn't use embeddings file, but we need a placeholder
    embeddings_file = None
    # SAE concepts are the SAE dictionary itself
    concepts_file = f"sae_concepts_{model_name}_{sample_type}_dense.pt"
    # SAE uses pre-computed dense activations with special naming
    # CLIP uses 'clipscope', Gemma uses 'gemmascope'
    scope_name = f"{model_name.lower()}scope"
    acts_file = f"{scope_name}_{sample_type}_dense"
    return con_label, embeddings_file, concepts_file, acts_file


def get_all_files(model_name, n_clusters, sample_type, percent_thru_model, concept_types=None):
    if concept_types is None:
        concept_types = ['avg', 'linsep', 'kmeans', 'linsepkmeans']
    
    all_files = []
    if 'avg' in concept_types:
        all_files.append(get_files_for_avg(model_name, n_clusters, sample_type, percent_thru_model))
    if 'linsep' in concept_types:
        all_files.append(get_files_for_linsep(model_name, n_clusters, sample_type, percent_thru_model))
    if 'kmeans' in concept_types:
        all_files.append(get_files_for_reg_kmeans(model_name, n_clusters, sample_type, percent_thru_model))
    if 'linsepkmeans' in concept_types:
        all_files.append(get_files_for_linsep_kmeans(model_name, n_clusters, sample_type, percent_thru_model))
    if 'sae' in concept_types:
        sae_files = get_files_for_sae(model_name, n_clusters, sample_type, percent_thru_model)
        if sae_files is not None:
            all_files.append(sae_files)
    
    return all_files
    

def get_act_metrics(dataset_name, acts_file):
    # Use ChunkedActivationLoader to handle both chunked and non-chunked files
    loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=SCRATCH_DIR)
    
    # Get activation info
    info = loader.get_activation_info()
    if info['is_chunked']:
        print(f"   Loading chunked activation file ({info['num_chunks']} chunks, {info['total_samples']:,} samples)...")
    else:
        print(f"   Loading single activation file...")
    
    # Return loader instead of loading full dataframe
    return loader


def compute_thresholds_with_fallback(
    is_kmeans, con_label, act_loader, gt_samples_per_concept_cal, 
    PERCENTILES, DEVICE, dataset_name, aggregation_method, 
    model_input_size, patch_size=14, random_seed=42, n_vectors=1, n_concepts_to_print=0):
    """Helper function to compute thresholds"""
    
    if is_kmeans:
        compute_aggregated_activation_thresholds_over_percentiles_all_pairs(
            act_loader,
            gt_samples_per_concept_cal,
            PERCENTILES,
            DEVICE,
            dataset_name,
            con_label,
            aggregation_method=aggregation_method,
            model_input_size=model_input_size,
            patch_size=patch_size,
            random_seed=random_seed
        )
    else:
        compute_aggregated_activation_thresholds_over_percentiles(
            gt_samples_per_concept_cal,
            act_loader,
            PERCENTILES,
            DEVICE,
            dataset_name,
            con_label,
            aggregation_method=aggregation_method,
            model_input_size=model_input_size,
            patch_size=patch_size,
            n_vectors=n_vectors,
            n_concepts_to_print=n_concepts_to_print,
            random_seed=random_seed
        )
      


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute baseline detection using maximum activation method')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process')
    parser.add_argument('--datasets', nargs='+', help='Multiple datasets to process')
    parser.add_argument('--model', type=str, help='Specific model to use')
    parser.add_argument('--models', nargs='+', help='Multiple models to process')
    parser.add_argument('--sample-type', type=str, choices=['patch'], help='Sample type to process (only patch/token supported)')
    parser.add_argument('--concept-types', nargs='+', choices=['avg', 'linsep', 'kmeans', 'linsepkmeans', 'sae'], 
                        default=['avg', 'linsep', 'kmeans', 'linsepkmeans'],
                        help='Concept types to compute thresholds for (default: all)')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets and exit')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Batch size for processing (default: {BATCH_SIZE})')
    parser.add_argument('--percentiles', nargs='+', type=float, help='Percentiles to compute thresholds for')
    parser.add_argument('--percentthrumodels', nargs='+', type=int, default=ALL_PERCENTTHRUMODELS, 
                        help=f'List of percentages through model layers to use (default: every 2 layers for all models)')
    parser.add_argument('--ptm', nargs='+', type=int, dest='percentthrumodels',
                        help='Shorthand for --percentthrumodels')
    parser.add_argument('--force-percentthrumodel', action='store_true',
                        help='Force processing of specified percentthrumodels even if not in model defaults')
    parser.add_argument('--compute-thresholds-only', action='store_true', 
                        help='Only compute thresholds, skip detection evaluation')
    parser.add_argument('--skip-thresholds', action='store_true',
                        help='Skip threshold computation (assumes thresholds already exist)')
    parser.add_argument('--aggregation-methods', nargs='+', choices=['max', 'mean', 'last', 'random'],
                        default=['max', 'mean', 'last', 'random'],
                        help='Aggregation methods to use (default: all)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for random aggregation method')
    parser.add_argument('--no-precompute-aggregations', action='store_true',
                        help='Disable precomputation of aggregations (process methods sequentially)')
    
    args = parser.parse_args()
    
    # List available datasets if requested
    if args.list_datasets:
        print("Available datasets:")
        all_datasets = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
        for dataset in all_datasets:
            print(f"  - {dataset}")
        sys.exit(0)
    
    # List available models if requested
    if args.list_models:
        print("Available models:")
        all_models = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), 
                      ('Gemma', ('text', 'text2')), ('Qwen', ('text', 'text3'))]
        for model_name, input_size in all_models:
            print(f"  - {model_name}: {input_size}")
        sys.exit(0)
    
    # Get list of percentthrumodels to process
    percentthrumodels = args.percentthrumodels
    
    # Determine which datasets to process
    if args.dataset:
        datasets_to_process = [args.dataset]
    elif args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = DATASETS
    
    # Determine which models to process
    if args.model:
        models_to_process = [(m, s) for m, s in MODELS if m == args.model]
        if not models_to_process:
            # Try to find in all available models
            all_models = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), 
                          ('Gemma', ('text', 'text2')), ('Qwen', ('text', 'text3'))]
            models_to_process = [(m, s) for m, s in all_models if m == args.model]
            if not models_to_process:
                print(f"Error: Model '{args.model}' not found")
                sys.exit(1)
    elif args.models:
        # Handle multiple models
        all_models = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), 
                      ('Gemma', ('text', 'text2')), ('Qwen', ('text', 'text3'))]
        models_to_process = []
        for model_name in args.models:
            model_matches = [(m, s) for m, s in all_models if m == model_name]
            if not model_matches:
                print(f"Error: Model '{model_name}' not found")
                sys.exit(1)
            models_to_process.extend(model_matches)
        # Remove duplicates while preserving order
        seen = set()
        models_to_process = [x for x in models_to_process if not (x in seen or seen.add(x))]
    else:
        models_to_process = MODELS
    
    # Determine which sample types to process
    if args.sample_type:
        sample_types_to_process = [(s, n) for s, n in SAMPLE_TYPES if s == args.sample_type]
        if not sample_types_to_process:
            # Use default cluster sizes
            if args.sample_type == 'patch':
                sample_types_to_process = [('patch', 1000)]
            else:
                sample_types_to_process = [('cls', 50)]
    else:
        sample_types_to_process = SAMPLE_TYPES
    
    # Update batch size if specified
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    
    # Update percentiles if specified
    if args.percentiles:
        PERCENTILES = args.percentiles
    
    # Loop through all percentthrumodels
    for PERCENT_THRU_MODEL in percentthrumodels:
        # If only SAE is requested, skip invalid percentthrumodels
        if args.concept_types == ['sae']:
            if PERCENT_THRU_MODEL not in [81, 92]:
                continue
        
        # Check if any valid configurations exist for this percentthrumodel
        has_valid_configs = False
        experiment_configs = list(product(models_to_process, datasets_to_process, sample_types_to_process))
        
        for (model_name, model_input_size), dataset_name, (sample_type, n_clusters) in experiment_configs:
            # If only SAE is requested, check if this model supports SAE at this percentthrumodel
            if args.concept_types == ['sae']:
                if not ((model_name == 'CLIP' and PERCENT_THRU_MODEL == 92) or 
                        (model_name == 'Gemma' and PERCENT_THRU_MODEL == 81)):
                    continue
            else:
                # Skip this model if the current percentthrumodel is not in its default list
                # unless --force-percentthrumodel is specified
                if not args.force_percentthrumodel:
                    model_default_percentiles = get_model_default_percentthrumodels(model_name, model_input_size)
                    if PERCENT_THRU_MODEL not in model_default_percentiles:
                        continue
                    
            # Skip invalid dataset-input size combinations
            if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue
            if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue
                
            has_valid_configs = True
            break
            
        if not has_valid_configs:
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing AGGREGATED DETECTION with PERCENT_THRU_MODEL = {PERCENT_THRU_MODEL}")
        print(f"{'='*60}\n")
        
        # Process experiment configurations
        for (model_name, model_input_size), dataset_name, (sample_type, n_clusters) in experiment_configs:
            # If only SAE is requested, skip models that don't support SAE at this percentthrumodel
            if args.concept_types == ['sae']:
                if not ((model_name == 'CLIP' and PERCENT_THRU_MODEL == 92) or 
                        (model_name == 'Gemma' and PERCENT_THRU_MODEL == 81)):
                    continue
            else:
                # Skip this model if the current percentthrumodel is not in its default list
                # unless --force-percentthrumodel is specified
                if not args.force_percentthrumodel:
                    model_default_percentiles = get_model_default_percentthrumodels(model_name, model_input_size)
                    if PERCENT_THRU_MODEL not in model_default_percentiles:
                        continue
                    
            # Skip invalid dataset-input size combinations
            if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue
            if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue
            
            print(f"\nProcessing model {model_name} dataset {dataset_name} sample type {sample_type}")
            
            # Clear GPU cache before processing
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            # Get ground truth values
            if sample_type == 'patch':   
                gt_samples_per_concept_cal = torch.load(f"GT_Samples/{dataset_name}/gt_patch_per_concept_cal_inputsize_{model_input_size}.pt")   
                gt_samples_per_concept_test = torch.load(f'GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt')
                # Also need image-level GT for detection metrics
                gt_images_per_concept_test = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt')
                gt_images_per_concept_cal = torch.load(f"GT_Samples/{dataset_name}/gt_samples_per_concept_cal_inputsize_{model_input_size}.pt")
            else:
                gt_samples_per_concept_cal = torch.load(f"GT_Samples/{dataset_name}/gt_samples_per_concept_cal_inputsize_{model_input_size}.pt")
                gt_samples_per_concept_test = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt')
                gt_images_per_concept_test = gt_samples_per_concept_test
                gt_images_per_concept_cal = gt_samples_per_concept_cal
            
            # Filter to only relevant concepts for this dataset
            gt_samples_per_concept_cal = filter_concept_dict(gt_samples_per_concept_cal, dataset_name)
            gt_samples_per_concept_test = filter_concept_dict(gt_samples_per_concept_test, dataset_name)
            gt_images_per_concept_test = filter_concept_dict(gt_images_per_concept_test, dataset_name)
            gt_images_per_concept_cal = filter_concept_dict(gt_images_per_concept_cal, dataset_name)
            print(f"  Filtered to {len(gt_samples_per_concept_cal)} concepts for {dataset_name}")
      
            all_files = get_all_files(model_name, n_clusters, sample_type, PERCENT_THRU_MODEL, args.concept_types)
            
            if not all_files:
                continue
                
            for con_label, embeddings_file, concepts_file, acts_file in all_files:
                print(f"\n{con_label}")
                
                # Load concepts
                if 'sae' in con_label:
                    print("   SAE detected - concepts are the SAE units")
                    concepts = None  # Will be handled differently
                elif 'kmeans' in con_label and 'linsep' not in con_label:
                    # For regular kmeans, concepts are cluster centroids
                    concepts = torch.load(f'Concepts/{dataset_name}/{concepts_file}')
                    print(f"   Loaded {len(concepts)} kmeans cluster concepts")
                else:
                    concepts = torch.load(f'Concepts/{dataset_name}/{concepts_file}')
                
                # Get activation loader
                try:
                    act_loader = get_act_metrics(dataset_name, acts_file)
                    info = act_loader.get_activation_info()
                    print(f"   Activation info: {info['total_samples']:,} samples, {info['num_concepts']} concepts")
                except FileNotFoundError as e:
                    print(f"   ⚠️  Activation file not found: {acts_file}, skipping...")
                    print(f"   Error: {e}")
                    continue
                
                # Precompute aggregations by default (unless disabled)
                precomputed_aggregations = {}
                if not args.no_precompute_aggregations and len(args.aggregation_methods) > 1:
                    if PRECOMPUTE_AVAILABLE:
                        print(f"\n   ⚡ Precomputing aggregations for {len(args.aggregation_methods)} methods...")
                        try:
                            precomputed_aggregations = precompute_all_aggregations(
                                act_loader, 
                                args.aggregation_methods,
                                model_input_size,
                                dataset_name,
                                patch_size=14,
                                device=DEVICE,
                                random_seed=args.random_seed
                            )
                            print(f"   ✓ Precomputed aggregations ready")
                        except Exception as e:
                            print(f"   ⚠️  Precomputation failed: {e}, processing methods sequentially")
                    else:
                        print(f"   ℹ️  Precomputation utilities not available, processing methods sequentially")
                        print(f"      To enable, ensure utils.baseline_detection_utils_chunked is available")
                
                # Process all aggregation methods
                for aggregation_method in args.aggregation_methods:
                    method_name = {'max': 'maxtoken', 'mean': 'meantoken', 'last': 'lasttoken', 'random': 'randomtoken'}[aggregation_method]
                    print(f"\n   Processing {aggregation_method.upper()} aggregation method...")
                    
                    # Step 1: Compute thresholds on calibration set (unless skipped)
                    if not args.skip_thresholds:
                        print(f"   Computing {method_name} thresholds using {aggregation_method} activations...")
                        if 'kmeans' in con_label:
                            print(f"   Computing {method_name} all-pairs thresholds for kmeans concepts...")
                        elif 'sae' in con_label:
                            print(f"   Computing {method_name} all-pairs thresholds for SAE units...")
                        
                        # SAE and kmeans both use all-pairs approach
                        is_allpairs = 'kmeans' in con_label or 'sae' in con_label
                        
                        compute_thresholds_with_fallback(
                            is_kmeans=is_allpairs,
                            con_label=con_label,
                            act_loader=act_loader,
                            gt_samples_per_concept_cal=gt_samples_per_concept_cal,
                            PERCENTILES=PERCENTILES,
                            DEVICE=DEVICE,
                            dataset_name=dataset_name,
                            aggregation_method=aggregation_method,
                            model_input_size=model_input_size,
                            patch_size=14,
                            random_seed=args.random_seed
                        )
                
                    if args.compute_thresholds_only:
                        print("   Threshold computation complete. Skipping detection evaluation.")
                        continue
                    
                    # Step 2: Evaluate detection
                    if 'kmeans' in con_label or 'sae' in con_label:
                        # For kmeans and SAE, use all-pairs detection
                        concept_type = 'SAE units' if 'sae' in con_label else 'kmeans'
                        print(f"\n   Computing {method_name} all-pairs detection metrics on TEST set for {concept_type}...")
                        
                        # For SAE, n_clusters represents the number of SAE units
                        if 'sae' in con_label:
                            # Get number of SAE units from activation info
                            n_units = act_loader.get_activation_info()['num_concepts']
                        else:
                            n_units = n_clusters
                            
                        compute_aggregated_detection_metrics_over_percentiles_allpairs(
                            PERCENTILES,
                            gt_images_per_concept_test,
                            dataset_name,
                            model_input_size,
                            DEVICE,
                            con_label,
                            act_loader,
                            aggregation_method=aggregation_method,
                            sample_type=sample_type,
                            patch_size=14,
                            n_clusters=n_units,
                            random_seed=args.random_seed
                        )
                        
                        # Step 3: Evaluate detection on CALIBRATION set
                        print(f"\n   Computing {method_name} all-pairs detection metrics on CALIBRATION set for {concept_type}...")
                        compute_aggregated_detection_metrics_over_percentiles_allpairs(
                            PERCENTILES,
                            gt_images_per_concept_cal,
                            dataset_name,
                            model_input_size,
                            DEVICE,
                            con_label + "_cal",
                            act_loader,
                            aggregation_method=aggregation_method,
                            sample_type=sample_type,
                            patch_size=14,
                            n_clusters=n_units,
                            random_seed=args.random_seed
                        )
                        
                        # Find best clusters/units per concept for both splits
                        unit_type = 'SAE units' if 'sae' in con_label else 'clusters'
                        for split_name, label_suffix in [('TEST', ''), ('CALIBRATION', '_cal')]:
                            print(f"\n   Finding best {method_name} {unit_type} per concept for {split_name} set...")
                            best_clusters = find_best_clusters_per_concept_from_aggregated_detectionmetrics(
                                dataset_name,
                                model_name,
                                sample_type,
                                metric_type='f1',
                                percentiles=PERCENTILES,
                                con_label=con_label + label_suffix,
                                aggregation_method=aggregation_method
                            )
                            filter_and_save_best_clusters_aggregated(dataset_name, con_label + label_suffix, aggregation_method)
                            
                        # Also save best percentiles for kmeans/SAE concepts (needed for per_concept_ptm_optimization.py)
                        print(f"\n   Saving best {method_name} percentiles for {concept_type} concepts...")
                        save_best_percentiles_for_kmeans(dataset_name, con_label, aggregation_method)
                    else:
                        # For supervised concepts, use regular detection
                        print(f"\n   Computing {method_name} detection metrics on TEST set...")
                        test_metrics = compute_aggregated_detection_metrics_over_percentiles(
                            PERCENTILES,
                            gt_images_per_concept_test,
                            act_loader,
                            dataset_name,
                            model_input_size,
                            DEVICE,
                            con_label,
                            aggregation_method=aggregation_method,
                            patch_size=14,
                            random_seed=args.random_seed
                        )
                        
                        # Step 3: Evaluate detection on CALIBRATION set
                        print(f"\n   Computing {method_name} detection metrics on CALIBRATION set...")
                        cal_metrics = compute_aggregated_detection_metrics_over_percentiles(
                            PERCENTILES,
                            gt_images_per_concept_cal,
                            act_loader,
                            dataset_name,
                            model_input_size,
                            DEVICE,
                            con_label + "_cal",
                            aggregation_method=aggregation_method,
                            patch_size=14,
                            random_seed=args.random_seed
                        )
                        
                        # Step 4: Find best percentiles based on calibration F1
                        print(f"\n   Finding best {method_name} detection percentiles...")
                        find_best_aggregated_detection_percentiles_cal(
                            dataset_name,
                            con_label,
                            PERCENTILES,
                            sample_type,
                            aggregation_method=aggregation_method
                        )
                
                    
                    # Clear GPU memory after each aggregation method
                    if DEVICE == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Clear memory after processing all aggregation methods for this concept file
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                
            print(f"\nCompleted all processing for {dataset_name} - {model_name} - {sample_type}")