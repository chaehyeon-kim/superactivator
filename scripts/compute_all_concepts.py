from lib2to3.fixes.fix_input import context
import torch
import sys
import argparse
from collections import defaultdict
from itertools import product

from utils.compute_concepts_utils import gpu_kmeans, compute_linear_separators, compute_avg_concept_vectors, create_binary_labels
from utils.general_utils import get_split_df
from utils.memory_management_utils import ChunkedEmbeddingLoader
from sklearn.metrics import f1_score, precision_recall_curve
import numpy as np
import time
from utils.filter_datasets_utils import filter_concept_dict
from utils.default_percentthrumodels import ALL_PERCENTTHRUMODELS, get_model_default_percentthrumodels
from utils.patch_alignment_utils import get_patch_split_df


MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Gemma', ('text', 'text2')), ('Qwen', ('text', 'text3'))]
DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
SAMPLE_TYPES = [('patch', 1000), ('cls', 50)]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCENT_THRU_MODEL = 100  # Default value, can be overridden by command line
SCRATCH_DIR = ''
BATCH_SIZE = 1000  # Increased from 500 for better GPU utilization
PRELOAD_ALL_CHUNKS = True  # Load all chunks into memory for faster training

def get_gt(sample_type, dataset_name, model_input_size):
    if sample_type == 'patch':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt')
        gt_samples_per_concept_train = torch.load(f'GT_Samples/{dataset_name}/gt_patch_per_concept_train_inputsize_{model_input_size}.pt')
    else:
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt')
        gt_samples_per_concept_train = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_train_inputsize_{model_input_size}.pt')
    return gt_samples_per_concept, gt_samples_per_concept_train

    
def get_files_for_avg(model_name, sample_type, percent_thru_model):
    con_label = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}'
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    concepts_file = f'avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt'
    cossim_file = f"cosine_similarities_{concepts_file[:-3]}.csv"
    return con_label, embeddings_file, concepts_file, cossim_file


def get_files_for_linsep(model_name, sample_type, percent_thru_model):
    con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}'
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    concepts_file = f'linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt'
    cossim_file = f"dists_{concepts_file[:-3]}.csv"
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
    cossim_file = f"cosine_similarities_{concepts_file[:-3]}.csv"
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
    dists_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{embeddings_file[:-3]}.csv"
    return con_label, embeddings_file, concepts_file, dists_file


def get_all_files(model_name, sample_type, n_clusters, percent_thru_model, concept_types=None):
    if concept_types is None:
        concept_types = ['avg', 'linsep', 'kmeans', 'linsepkmeans']
    
    all_files = []
    if 'avg' in concept_types:
        all_files.append(get_files_for_avg(model_name, sample_type, percent_thru_model))
    if 'linsep' in concept_types:
        all_files.append(get_files_for_linsep(model_name, sample_type, percent_thru_model))
    if 'kmeans' in concept_types:
        all_files.append(get_files_for_reg_kmeans(model_name, n_clusters, sample_type, percent_thru_model))
    if 'linsepkmeans' in concept_types:
        all_files.append(get_files_for_linsep_kmeans(model_name, n_clusters, sample_type, percent_thru_model))
    return all_files

def compute_and_report_f1_scores(concepts, gt_samples_per_concept, loader, dataset_name, sample_type, model_input_size=224):
    """
    Compute and report F1 scores for learned concepts.
    Used primarily for one-pass method which doesn't compute F1 during training.
    """
    print("\n   Computing F1 scores for one-pass concepts (using sample)...")
    
    # Get train/test split
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name, model_input_size=model_input_size)
    else:
        split_df = get_split_df(dataset_name)
    
    test_indices = list(split_df[split_df == 'test'].index)
    
    # Sample a subset for faster evaluation (e.g., 10k samples max)
    max_test_samples = min(10000, len(test_indices))
    if len(test_indices) > max_test_samples:
        print(f"   Sampling {max_test_samples} out of {len(test_indices)} test samples for speed...")
        test_indices = np.random.choice(test_indices, max_test_samples, replace=False).tolist()
    
    # Create binary labels
    total_samples = loader.total_samples
    all_labels = create_binary_labels(total_samples, gt_samples_per_concept)
    
    # Load test embeddings
    print(f"   Loading {len(test_indices)} test embeddings...")
    t_load = time.time()
    test_embeddings = loader.load_specific_embeddings(test_indices)
    print(f"   Loading completed in {time.time() - t_load:.2f} seconds")
    
    f1_scores = []
    
    # Compute F1 for each concept
    for concept_name, concept_vector in concepts.items():
        if concept_name not in all_labels:
            continue
            
        # Get labels for this concept
        labels = all_labels[concept_name][test_indices]
        
        # Compute activations
        concept_vector = concept_vector.to(test_embeddings.device)
        activations = test_embeddings @ concept_vector
        
        # Find optimal threshold using precision-recall curve
        activations_cpu = activations.cpu().numpy()
        labels_cpu = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        
        # Skip if no positive samples
        if labels_cpu.sum() == 0:
            continue
            
        precisions, recalls, thresholds = precision_recall_curve(labels_cpu, activations_cpu)
        f1_scores_at_thresholds = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        # Get best F1 score
        best_f1_idx = np.argmax(f1_scores_at_thresholds)
        best_f1 = f1_scores_at_thresholds[best_f1_idx]
        f1_scores.append(best_f1)
    
    if f1_scores:
        min_f1 = min(f1_scores)
        max_f1 = max(f1_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)
        
        print(f"\n   ===== F1 SCORES SUMMARY =====")
        print(f"   Number of concepts evaluated: {len(f1_scores)}")
        print(f"   Min F1: {min_f1:.4f}")
        print(f"   Max F1: {max_f1:.4f}")
        print(f"   Avg F1: {avg_f1:.4f}")
        print(f"   =============================\n")
    else:
        print("   Warning: No F1 scores could be computed (no test samples)")


def get_cluster_labels(dataset_name, kmeans_concept_file):
    print("loading gt clusters from kmeans")
    train_cluster_to_samples = torch.load(f'Concepts/{dataset_name}/train_samples_{kmeans_concept_file}')
    test_cluster_to_samples = torch.load(f'Concepts/{dataset_name}/test_samples_{kmeans_concept_file}')
    cal_cluster_to_samples = torch.load(f'Concepts/{dataset_name}/cal_samples_{kmeans_concept_file}')
    
    cluster_to_samples = defaultdict(list)
    for cluster, samples in train_cluster_to_samples.items():
        cluster_to_samples[cluster].extend(samples)
    for cluster, samples in test_cluster_to_samples.items():
        cluster_to_samples[cluster].extend(samples)
    for cluster, samples in cal_cluster_to_samples.items():
        cluster_to_samples[cluster].extend(samples)
        
    for cluster in cluster_to_samples:
        cluster_to_samples[cluster] = sorted(cluster_to_samples[cluster])
    cluster_to_samples = dict(cluster_to_samples)
    return cluster_to_samples

def get_unsupervised_concepts(embeddings_file, n_clusters, dataset_name, concepts_file, model_input_size, sample_type, model_name, loader, num_workers=None, use_batched=False, use_onepass=False, compute_f1=False):
    # Construct full path for backward compatibility with functions that expect it
    embeddings_path = f"{SCRATCH_DIR}Embeddings/{dataset_name}/{embeddings_file}"
    if 'linsep' in concepts_file:
        kmeans_concepts_file = "_".join(part for part in concepts_file.split("_") if "linsep" not in part)
        cluster_to_samples = get_cluster_labels(dataset_name, kmeans_concepts_file)
        
        # Always use chunked version for memory efficiency
        if use_onepass:
            print("   Using one-pass accumulation method for unsupervised concepts...")
        else:
            print("   Using chunked linear separators for unsupervised concepts...")
        # For k-means with 1000 clusters, can use either batched or parallel training
        concepts, logs = compute_linear_separators(embeddings_path, cluster_to_samples, dataset_name, sample_type, model_input_size, 
                                  device=DEVICE, output_file=concepts_file, lr=0.001, epochs=1000, batch_size=256, patience=20, 
                                  tolerance=0.001, weight_decay=1e-4, lr_step_size=5, lr_gamma=0.8, balance_data=True,  # Always balance data
                                  balance_negatives=False, num_workers=num_workers, use_batched=use_batched, use_onepass=use_onepass)
        
        # If one-pass method, compute F1 scores for reporting (optional - can be slow)
        if use_onepass and compute_f1:
            compute_and_report_f1_scores(concepts, cluster_to_samples, loader, dataset_name, sample_type, model_input_size)
    else:
        # Always use chunked K-means for memory efficiency
        print("   Using chunked K-means clustering...")
        concepts, _, _, _ = gpu_kmeans(n_clusters=n_clusters, embeddings_path=embeddings_path, dataset_name=dataset_name,
                                      device=DEVICE, model_input_size=model_input_size,
                                      concepts_filename=concepts_file, sample_type=sample_type,
                                      map_samples=True)
        
    return concepts


def get_supervised_concepts(embeddings_file, dataset_name, concepts_file, model_input_size, sample_type, loader, num_workers=None):
    # Construct full path for backward compatibility with functions that expect it
    embeddings_path = f"{SCRATCH_DIR}Embeddings/{dataset_name}/{embeddings_file}"
    gt_samples_per_concept, gt_samples_per_concept_train = get_gt(sample_type, dataset_name, model_input_size)
    
    # Filter concepts to only those relevant for this dataset
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    gt_samples_per_concept_train = filter_concept_dict(gt_samples_per_concept_train, dataset_name)
    
    print(f"computing concepts for {len(gt_samples_per_concept)} concepts (filtered from original)")
    
    if 'linsep' in concepts_file:
        # Always use chunked version for memory efficiency
        print("   Using chunked linear separators for supervised concepts...")
        concepts, logs = compute_linear_separators(embeddings_path, gt_samples_per_concept, dataset_name, 
                                                 sample_type=sample_type, device=DEVICE,
                                                 model_input_size=model_input_size,
                                                 output_file=concepts_file, batch_size=64,  # Reduced batch size
                                                 lr=0.001, epochs=1000, patience=20, tolerance=0.001,
                                                 weight_decay=0.0001, lr_step_size=5, lr_gamma=0.8,
                                                 balance_data=True,
                                                 balance_negatives=False,
                                                 num_workers=num_workers)

        
    else:
        # Always use chunked version for memory efficiency
        print("   Using chunked average concept computation...")
        concepts = compute_avg_concept_vectors(gt_samples_per_concept_train, loader, 
                                             dataset_name=dataset_name, output_file=concepts_file)
    return concepts

        
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Compute concepts for various models and datasets')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process')
    parser.add_argument('--datasets', nargs='+', help='Multiple datasets to process')
    parser.add_argument('--model', type=str, help='Specific model to use')
    parser.add_argument('--models', nargs='+', help='Multiple models to use')
    parser.add_argument('--sample-type', type=str, choices=['patch', 'cls'], help='Sample type to process')
    parser.add_argument('--concept-types', nargs='+', choices=['avg', 'linsep', 'kmeans', 'linsepkmeans'], 
                        default=['avg', 'linsep', 'kmeans', 'linsepkmeans'],
                        help='Concept types to compute (default: all)')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets and exit')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Batch size for processing (default: {BATCH_SIZE})')
    parser.add_argument('--percentthrumodels', nargs='+', type=int,
                        help='List of percentages through model layers to use (default: model-specific values)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers for training concepts (default: auto-detect based on GPUs)')
    parser.add_argument('--use-batched', action='store_true',
                        help='Use batched training with single D×K model for k-means concepts (more memory efficient)')
    parser.add_argument('--use-onepass', action='store_true',
                        help='Use one-pass accumulation method for linsep kmeans (faster for large datasets)')
    parser.add_argument('--compute-f1', action='store_true',
                        help='Compute and report F1 scores after one-pass method (slower due to loading test embeddings)')
    
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
    if args.percentthrumodels:
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
            if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue
            if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue
            
            print(f"Processing model {model_name} dataset {dataset_name} sample type {sample_type}")
            #get gt values
            if sample_type == 'patch':
                gt_samples_per_concept = torch.load(f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt")   
            else:
                gt_samples_per_concept = torch.load(f"GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt")
            
            # Filter to only relevant concepts for this dataset
            gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
            print(f"  Filtered to {len(gt_samples_per_concept)} concepts for {dataset_name}")


            all_files = get_all_files(model_name, sample_type, n_clusters, PERCENT_THRU_MODEL, args.concept_types)
            for con_label, embeddings_file, concepts_file, acts_file in all_files:
                print(con_label)
                    
                #load embeddings (handles both chunked and non-chunked)
                loader = ChunkedEmbeddingLoader(dataset_name, embeddings_file, SCRATCH_DIR, device=DEVICE)
                
                # Check if embeddings are chunked
                info = loader.get_embedding_info()
                if info['is_chunked']:
                    print(f"   Loading chunked embeddings ({info['num_chunks']} chunks, {info['total_samples']:,} samples)...")
                
                # Pass loader to concept computation functions
                # They will decide whether to use chunked processing based on data size
                
                if 'kmeans' in con_label: #unsupervised concepts
                    get_unsupervised_concepts(embeddings_file, n_clusters, dataset_name, concepts_file, model_input_size, sample_type, model_name, loader, num_workers=args.num_workers, use_batched=args.use_batched, use_onepass=args.use_onepass, compute_f1=args.compute_f1)
                else: #supervised concepts
                    #compute concepts
                    get_supervised_concepts(embeddings_file, dataset_name, concepts_file, model_input_size, sample_type, loader, num_workers=args.num_workers)
