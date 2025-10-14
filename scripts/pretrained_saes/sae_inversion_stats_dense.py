"""
Compute inversion statistics for dense filtered SAE units.
This version uses the same functions as the regular pipeline.
"""

import torch
import os
import sys
import argparse
from itertools import product
from tqdm import tqdm

# Ensure we're in the right directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(script_dir, '../..'))
os.chdir(workspace_dir)

sys.path.append(workspace_dir)

from utils.quant_concept_evals_utils import (detect_then_invert_metrics_over_percentiles,
                                            find_optimal_detect_invert_thresholds,
                                            detect_then_invert_with_optimal_thresholds)
from utils.superdetector_inversion_utils import (all_superdetector_inversions_across_percentiles,
                                                detect_then_invert_superdetector_twostage_metrics,
                                                find_optimal_twostage_superdetector_thresholds,
                                                detect_then_invert_twostage_superdetector_with_optimal_thresholds)
from utils.unsupervised_utils import get_matched_concepts_and_data
from utils.memory_management_utils import ChunkedActivationLoader, ChunkedEmbeddingLoader
from utils.filter_datasets_utils import filter_concept_dict

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRATCH_DIR = ''
PERCENTILES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# Default datasets for SAE
ALL_DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
IMAGE_DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces']
TEXT_DATASETS = ['Sarcasm', 'iSarcasm', 'GoEmotions']


def get_sae_act_loader(dataset_name, model_type, sample_type):
    """Get activation loader for SAE dense activations."""
    # Map model type to the correct activation filename
    if model_type == 'CLIP':
        acts_file = f"clipscope_{sample_type}_dense.pt"
    else:  # Gemma
        acts_file = f"gemmascope_{sample_type}_dense.pt"
    
    # Use ChunkedActivationLoader which will look in the correct directory
    loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=SCRATCH_DIR)
    
    # Get activation info
    info = loader.get_activation_info()
    if info['is_chunked']:
        print(f"   Loading chunked activation file ({info['num_chunks']} chunks, {info['total_samples']:,} samples)...")
    else:
        print(f"   Loading single activation file...")
    
    return loader


def main():
    parser = argparse.ArgumentParser(description='Compute inversion statistics for SAE dense activations')
    parser.add_argument('--datasets', nargs='+', 
                       default=ALL_DATASETS,
                       help='Datasets to process')
    parser.add_argument('--models', nargs='+', default=['CLIP', 'Gemma'],
                       choices=['CLIP', 'Gemma'], help='Model types to process')
    parser.add_argument('--sample-types', nargs='+', default=['patch'],
                       help='Sample types to process (only patch supports inversion)')
    parser.add_argument('--percentiles', nargs='+', type=float, 
                       help='Percentiles to use for inversion')
    parser.add_argument('--percentthrumodel', type=int, default=92,
                       help='Percentage through model for SAE (default: 92 for CLIP)')
    
    args = parser.parse_args()
    
    # Update percentiles if specified
    global PERCENTILES
    if args.percentiles:
        PERCENTILES = args.percentiles
    
    # Validate inputs
    # Filter to only patch sample type
    args.sample_types = [s for s in args.sample_types if s == 'patch']
    if not args.sample_types:
        print("Warning: Only patch sample type supports inversion. Setting to patch.")
        args.sample_types = ['patch']
    
    # Process all requested datasets
    datasets_to_process = args.datasets
    
    print(f"🚀 Starting SAE inversion statistics computation using regular pipeline functions")
    print(f"   Datasets: {datasets_to_process}")
    print(f"   Models: {args.models}")
    print(f"   Sample types: {args.sample_types}")
    print(f"   Percentiles: {PERCENTILES}")
    
    for dataset_name in datasets_to_process:
        # Determine which models to use for this dataset
        if dataset_name in IMAGE_DATASETS:
            valid_models = [m for m in args.models if m == 'CLIP']
        else:  # Text datasets
            valid_models = [m for m in args.models if m == 'Gemma']
        
        if not valid_models:
            print(f"⚠️  Skipping {dataset_name} - no valid models specified for this dataset type")
            continue
            
        for model_type in valid_models:
            # Set model input size based on dataset type
            if dataset_name in IMAGE_DATASETS:
                model_input_size = (224, 224)  # CLIP input size
            else:  # Text datasets
                model_input_size = ('text', 'text2')  # Gemma input size
            
            # Set percent through model based on model type
            if model_type == 'CLIP':
                percent_thru_model = 92  # Default for CLIP SAE
            else:  # Gemma
                percent_thru_model = 81  # Default for Gemma SAE
            
            for sample_type in args.sample_types:
                print(f"\n{'='*60}")
                print(f"Processing {dataset_name} - {model_type} - {sample_type}")
                
                try:
                    # Load ground truth patches
                    gt_patches_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt')
                    gt_patches_per_concept_test = torch.load(f'GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt')
                    gt_patches_per_concept_cal = torch.load(f'GT_Samples/{dataset_name}/gt_patch_per_concept_cal_inputsize_{model_input_size}.pt')
                    
                    # Filter to only relevant concepts for this dataset
                    gt_patches_per_concept = filter_concept_dict(gt_patches_per_concept, dataset_name)
                    gt_patches_per_concept_test = filter_concept_dict(gt_patches_per_concept_test, dataset_name)
                    gt_patches_per_concept_cal = filter_concept_dict(gt_patches_per_concept_cal, dataset_name)
                    print(f"   Filtered to {len(gt_patches_per_concept)} concepts for {dataset_name}")
                    
                    # Setup embedding loader (using SAE embeddings)
                    print("   Setting up chunked embedding loader...")
                    embeddings_file = f"{model_type}_SAE_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
                    embedding_loader = ChunkedEmbeddingLoader(dataset_name, embeddings_file, SCRATCH_DIR, DEVICE)
                    print(f"   📊 Embedding info: {embedding_loader.get_embedding_info()}")
                    
                    # Get activation loader
                    acts_file = f"clipscope_{sample_type}_dense.pt"
                    act_loader = get_sae_act_loader(dataset_name, model_type, sample_type)
                    info = act_loader.get_activation_info()
                    print(f"   Activation info: {info['total_samples']:,} samples, {info['num_concepts']} SAE units")
                    
                    # Create con_label for SAE
                    con_label = f"{model_type}_sae_{sample_type}_dense"
                    
                    # For SAE, create identity vectors as concepts
                    print("   Creating identity vectors for SAE units...")
                    num_sae_units = info['num_concepts']
                    concepts = {}
                    for i in range(num_sae_units):
                        vec = torch.zeros(num_sae_units, device=DEVICE)
                        vec[i] = 1.0
                        concepts[str(i)] = vec
                    
                    # Get matched loader and data for unsupervised concepts
                    print("   Getting matched concepts and data...")
                    
                    matched_loader, matched_gt_patches_per_concept_cal, matched_gt_patches_per_concept_test, \
                    matched_gt_patches_per_concept, matched_concepts = get_matched_concepts_and_data(
                        dataset_name,
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
                        filtered_concept_to_cluster = {c: cluster_id for c, cluster_id in matched_loader.concept_to_cluster.items() 
                                                     if c in matched_concepts}
                        matched_loader.concept_to_cluster = filtered_concept_to_cluster
                        matched_loader.cluster_to_concept = {cluster_id: c for c, cluster_id in filtered_concept_to_cluster.items()}
                        matched_loader.available_clusters = list(matched_loader.cluster_to_concept.keys())
                        matched_loader.matched_concepts = list(filtered_concept_to_cluster.keys())
                    
                    # Run the evaluation pipeline
                    print("="*80)
                    print(f"Running evaluation pipeline for {con_label}")
                    print(f"Number of concepts to process: {len(matched_concepts)}")
                    print(f"Dataset: {dataset_name}, Model: {model_type}")
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
                    
                    # Filter concept names
                    concept_names = list(matched_concepts.keys())
                    if hasattr(matched_loader, 'concept_to_cluster'):
                        available_concepts = set(matched_loader.concept_to_cluster.keys())
                        filtered_concept_names = [c for c in concept_names 
                                                if c in matched_gt_patches_per_concept_cal 
                                                and c in available_concepts]
                    else:
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
                    
                    print(f"\n✅ Completed inversion statistics for {dataset_name} - {model_type} - {sample_type}")
                    
                except FileNotFoundError as e:
                    print(f"❌ Error: Required file not found - {e}")
                    continue
                except Exception as e:
                    print(f"❌ Error processing {dataset_name} - {model_type} - {sample_type}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"\n✨ SAE inversion statistics computation complete!")
    print(f"   Results saved to: {os.path.join(workspace_dir, 'Best_Inversion_Percentiles_Cal')}")


if __name__ == "__main__":
    main()