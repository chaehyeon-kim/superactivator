import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import sys
import os
import argparse
from collections import defaultdict
from itertools import product
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.compute_concepts_utils import gpu_kmeans, compute_linear_separators
from utils.activation_utils import compute_cosine_sims, compute_signed_distances
from utils.unsupervised_utils import compute_detection_metrics_over_percentiles_allpairs, \
find_best_clusters_per_concept_from_detectionmetrics, filter_and_save_best_clusters, get_matched_concepts_and_data, \
compute_concept_thresholds_over_percentiles_all_pairs
from utils.superdetector_inversion_utils import find_all_superdetector_patches, all_superdetector_inversions_across_percentiles
from utils.quant_concept_evals_utils import detect_then_invert_metrics_over_percentiles, compute_concept_thresholds_over_percentiles
from utils.gt_concept_segmentation_utils import map_concepts_to_patch_indices, map_concepts_to_image_indices
from utils.default_percentthrumodels import ALL_PERCENTTHRUMODELS, get_model_default_percentthrumodels


MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Gemma', ('text', 'text2')), ('Qwen', ('text', 'text3'))]

DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
SAMPLE_TYPES = [('patch', 1000), ('cls', 50)]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCENT_THRU_MODEL = 100  # Default value, can be overridden by command line
SCRATCH_DIR = ''
BATCH_SIZE = 2000  # Increased for better GPU utilization (adjust based on GPU memory)


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
    return all_files
    
     


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute activations for concept detection')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process')
    parser.add_argument('--datasets', nargs='+', help='Multiple datasets to process')
    parser.add_argument('--model', type=str, help='Specific model to use')
    parser.add_argument('--models', nargs='+', help='Multiple models to use')
    parser.add_argument('--sample-type', type=str, choices=['patch', 'cls'], help='Sample type to process')
    parser.add_argument('--concept-types', nargs='+', choices=['avg', 'linsep', 'kmeans', 'linsepkmeans'], 
                        default=['avg', 'linsep', 'kmeans', 'linsepkmeans'],
                        help='Concept types to compute activations for (default: all)')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets and exit')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Batch size for processing (default: {BATCH_SIZE})')
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
    
    # List available models if requested
    if args.list_models:
        print("Available models:")
        all_models = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), 
                      ('Gemma', ('text', 'text2')), ('Qwen', ('text', 'text3'))]
        for model_name, input_size in all_models:
            print(f"  - {model_name}: {input_size}")
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
            # Try to find in all available models
            models_to_process = [(m, s) for m, s in all_available_models if m == args.model]
            if not models_to_process:
                print(f"Error: Model '{args.model}' not found")
                sys.exit(1)
    else:
        models_to_process = MODELS
    
    # Get list of percentthrumodels to process
    percentthrumodels = args.percentthrumodels
    
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
    
    # Get list of percentthrumodels to process
    if args.percentthrumodels != ALL_PERCENTTHRUMODELS:  # User specified custom values
        percentthrumodels = args.percentthrumodels
    else:
        # Compute model-specific defaults based on selected models
        percentthrumodels = set()
        for model_name, model_input_size in models_to_process:
            model_defaults = get_model_default_percentthrumodels(model_name, model_input_size)
            percentthrumodels.update(model_defaults)
        percentthrumodels = sorted(list(percentthrumodels))
    
    print(f"Using percentthrumodels ({len(percentthrumodels)} values): {percentthrumodels}")
    
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
            all_files = get_all_files(model_name, n_clusters, sample_type, PERCENT_THRU_MODEL, args.concept_types)
            if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue
            if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue

            
            for con_label, embeddings_file, concepts_file, acts_file in all_files:
                print(f"Processing model {model_name} dataset {dataset_name} sample type {sample_type}")
                print(con_label)
                
                # Pass embedding path instead of loading into memory
                embeddings_path = f"{SCRATCH_DIR}Embeddings/{dataset_name}/{embeddings_file}"
                print(f"Using embeddings from: {embeddings_path}")
                
                concepts = torch.load(f'Concepts/{dataset_name}/{concepts_file}')

                if 'linsep' in acts_file:
                    compute_signed_distances(embeds=embeddings_path,  # Pass path instead of tensor
                                                concepts=concepts, 
                                                dataset_name=dataset_name, 
                                                device=DEVICE,
                                                output_file=acts_file, 
                                                scratch_dir=SCRATCH_DIR, 
                                                batch_size=BATCH_SIZE)
                    torch.cuda.empty_cache()            
                    torch.cuda.ipc_collect()           

                else:
                    compute_cosine_sims(embeddings=embeddings_path,  # Pass path instead of tensor
                                                concepts=concepts, 
                                                output_file=acts_file,
                                                dataset_name=dataset_name, 
                                                device=DEVICE,
                                                batch_size=BATCH_SIZE, 
                                                scratch_dir=SCRATCH_DIR)
                    torch.cuda.empty_cache()            
                    torch.cuda.ipc_collect()
                