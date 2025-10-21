import torch
import sys
import argparse
from collections import defaultdict
from itertools import product

from utils.unsupervised_utils import compute_concept_thresholds_over_percentiles_all_pairs
from utils.quant_concept_evals_utils import compute_concept_thresholds_over_percentiles
from utils.filter_datasets_utils import filter_concept_dict
from utils.default_percentthrumodels import ALL_PERCENTTHRUMODELS, get_model_default_percentthrumodels


MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Gemma', ('text', 'text2')), ('Qwen', ('text', 'text3'))]

DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
SAMPLE_TYPES = [('patch', 1000), ('cls', 50)]

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
    
    # Add SAE if applicable (CLIP patch only) - not controlled by concept_types
    sae_files = get_files_for_sae(model_name, sample_type, percent_thru_model)
    if sae_files is not None:
        all_files.append(sae_files)
    
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
      


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute validation thresholds for concept detection')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process')
    parser.add_argument('--datasets', nargs='+', help='Multiple datasets to process')
    parser.add_argument('--model', type=str, help='Specific model to use')
    parser.add_argument('--models', nargs='+', help='Multiple models to use')
    parser.add_argument('--sample-type', type=str, choices=['patch', 'cls'], help='Sample type to process')
    parser.add_argument('--concept-types', nargs='+', choices=['avg', 'linsep', 'kmeans', 'linsepkmeans'], 
                        default=['avg', 'linsep', 'kmeans', 'linsepkmeans'],
                        help='Concept types to compute thresholds for (default: all)')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets and exit')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Batch size for processing (default: {BATCH_SIZE})')
    parser.add_argument('--percentiles', nargs='+', type=float, help='Percentiles to compute thresholds for')
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
            # Skip invalid dataset-input size combinations
            if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue
            if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue
            
            
            print(f"Processing model {model_name} dataset {dataset_name} sample type {sample_type}")
            #get gt values from calibration dataset
            if sample_type == 'patch':   
                gt_samples_per_concept_cal = torch.load(f"GT_Samples/{dataset_name}/gt_patch_per_concept_cal_inputsize_{model_input_size}.pt")   
            else:
                gt_samples_per_concept_cal = torch.load(f"GT_Samples/{dataset_name}/gt_samples_per_concept_cal_inputsize_{model_input_size}.pt")
            
            # Filter to only relevant concepts for this dataset
            gt_samples_per_concept_cal = filter_concept_dict(gt_samples_per_concept_cal, dataset_name)
            print(f"  Filtered to {len(gt_samples_per_concept_cal)} concepts for {dataset_name}")
      
            all_files = get_all_files(model_name, n_clusters, sample_type, PERCENT_THRU_MODEL, args.concept_types)
            for con_label, embeddings_file, concepts_file, acts_file in all_files:
                print(con_label)
                
                # Let ChunkedActivationLoader handle finding the file (chunked or not)
                # It will check for both regular and chunked versions
                
                #load concepts
                if 'sae' in con_label:
                    # For SAE, we don't have a separate concepts file
                    # The concepts are implicitly the SAE units themselves
                    print("   SAE detected - concepts are the SAE units")
                    concepts = None  # Will be handled differently in threshold computation
                else:
                    concepts = torch.load(f'Concepts/{dataset_name}/{concepts_file}')
                
                #compute raw act metrics with cal embeds and concept
                try:
                    act_loader = get_act_metrics(dataset_name, acts_file)
                    info = act_loader.get_activation_info()
                    # Print concise info without the full concept list
                    print(f"Activation info: {info['total_samples']:,} samples, {info['num_concepts']} concepts, {info['num_chunks']} chunks")
                except FileNotFoundError as e:
                    print(f"   ⚠️  Activation file not found: {acts_file}, skipping...")
                    print(f"   Error: {e}")
                    continue
          
                #compute threshold for those concepts
                if 'kmeans' in con_label or 'sae' in con_label:
                    print("computing thresholds over percentiles (chunked)")
                    # Use chunked version that only loads calibration data
                    compute_concept_thresholds_over_percentiles_all_pairs(act_loader, 
                                                                          gt_samples_per_concept_cal, 
                                                                          PERCENTILES, DEVICE,
                                                                          dataset_name, con_label)  
                else:
                    print("computing thresholds over percentiles")
                    # Use the original function which we'll make compatible with loaders
                    compute_concept_thresholds_over_percentiles(gt_samples_per_concept_cal, 
                                                                act_loader, PERCENTILES, DEVICE, 
                                                                dataset_name, con_label, n_vectors=1,
                                                                n_concepts_to_print=0)