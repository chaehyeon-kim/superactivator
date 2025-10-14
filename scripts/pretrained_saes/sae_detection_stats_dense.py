"""
Compute detection statistics for dense filtered SAE units.
This version uses the same functions as the regular pipeline.
"""

import torch
import pandas as pd
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

from utils.quant_concept_evals_utils import compute_detection_metrics_over_percentiles, find_best_detection_percentiles_cal
from utils.unsupervised_utils import (compute_detection_metrics_over_percentiles_allpairs, 
                                     find_best_clusters_per_concept_from_detectionmetrics,
                                     filter_and_save_best_clusters, get_matched_concepts_and_data)
from utils.superdetector_inversion_utils import find_all_superdetector_patches
from utils.memory_management_utils import ChunkedActivationLoader
from utils.filter_datasets_utils import filter_concept_dict

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRATCH_DIR = ''
PERCENTILES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# Default datasets for SAE
IMAGE_DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces']
TEXT_DATASETS = ['Sarcasm', 'iSarcasm', 'GoEmotions']


def get_sae_act_loader(dataset_name, model_type, sample_type):
    """Get activation loader for SAE dense activations."""
    # Map model type to the correct activation filename
    if model_type == 'CLIP':
        acts_file = f"clipscope_{sample_type}_dense.pt"
    else:  # Gemma
        acts_file = f"gemmascope_{sample_type}_dense.pt"
    
    # Use the standard ChunkedActivationLoader - it will now detect SAE_Activations_Dense folder
    loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=SCRATCH_DIR)
    
    # Get activation info
    info = loader.get_activation_info()
    if info['is_chunked']:
        print(f"   Loading chunked activation file ({info['num_chunks']} chunks, {info['total_samples']:,} samples)...")
    else:
        print(f"   Loading single activation file...")
    
    return loader


def main():
    parser = argparse.ArgumentParser(description='Compute detection statistics for SAE dense activations')
    parser.add_argument('--datasets', nargs='+', 
                       help='Datasets to process')
    parser.add_argument('--models', nargs='+', default=['CLIP', 'Gemma'],
                       choices=['CLIP', 'Gemma'], help='Model types to process')
    parser.add_argument('--sample-types', nargs='+', 
                       help='Sample types to process')
    parser.add_argument('--percentiles', nargs='+', type=float, 
                       help='Percentiles to use for detection')
    
    args = parser.parse_args()
    
    # Update percentiles if specified
    global PERCENTILES
    if args.percentiles:
        PERCENTILES = args.percentiles
    
    # Determine datasets
    if args.datasets:
        datasets_to_process = args.datasets
    else:
        # Default: all datasets
        datasets_to_process = IMAGE_DATASETS + TEXT_DATASETS
    
    print(f"🚀 Starting SAE detection statistics computation using regular pipeline functions")
    print(f"   Datasets: {datasets_to_process}")
    print(f"   Models: {args.models}")
    print(f"   Percentiles: {PERCENTILES}")
    
    for dataset_name in datasets_to_process:
        for model_type in args.models:
            # Skip invalid combinations
            if model_type == 'CLIP' and dataset_name in TEXT_DATASETS:
                continue
            if model_type == 'Gemma' and dataset_name in IMAGE_DATASETS:
                continue
            
            # Determine sample types
            if model_type == 'CLIP':
                if args.sample_types:
                    sample_types = [s for s in args.sample_types if s in ['patch', 'cls']]
                else:
                    sample_types = ['patch', 'cls']
                model_input_size = (224, 224)
                # For SAE clustering, we use more clusters since we have many units
                n_clusters = 5000  # Adjust as needed
            else:  # Gemma
                if args.sample_types:
                    sample_types = [s for s in args.sample_types if s in ['patch', 'token', 'cls']]
                else:
                    sample_types = ['patch', 'cls']
                model_input_size = ('text', 'text2')
                n_clusters = 3000  # Adjust as needed
            
            for sample_type in sample_types:
                print(f"\n{'='*60}")
                print(f"Processing {dataset_name} - {model_type} - {sample_type}")
                
                try:
                    # Load ground truth samples for test and calibration
                    if sample_type in ['patch', 'token']:
                        if sample_type == 'patch':
                            gt_samples_test_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
                            gt_samples_cal_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_cal_inputsize_{model_input_size}.pt"
                            # Also need image-level GT for detection metrics
                            gt_images_test_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
                            gt_images_cal_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_cal_inputsize_{model_input_size}.pt"
                        else:  # token
                            # For text, patch-level means token-level
                            gt_samples_test_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
                            gt_samples_cal_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_cal_inputsize_{model_input_size}.pt"
                            # Sentence-level GT
                            gt_images_test_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
                            gt_images_cal_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_cal_inputsize_{model_input_size}.pt"
                    else:  # cls
                        gt_samples_test_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
                        gt_samples_cal_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_cal_inputsize_{model_input_size}.pt"
                        gt_images_test_file = gt_samples_test_file
                        gt_images_cal_file = gt_samples_cal_file
                    
                    print(f"   Loading ground truth files...")
                    gt_samples_per_concept_test = torch.load(gt_samples_test_file)
                    gt_samples_per_concept_cal = torch.load(gt_samples_cal_file)
                    gt_images_per_concept_test = torch.load(gt_images_test_file)
                    gt_images_per_concept_cal = torch.load(gt_images_cal_file)
                    
                    # Filter to only relevant concepts for this dataset
                    gt_samples_per_concept_test = filter_concept_dict(gt_samples_per_concept_test, dataset_name)
                    gt_samples_per_concept_cal = filter_concept_dict(gt_samples_per_concept_cal, dataset_name)
                    gt_images_per_concept_test = filter_concept_dict(gt_images_per_concept_test, dataset_name)
                    gt_images_per_concept_cal = filter_concept_dict(gt_images_per_concept_cal, dataset_name)
                    print(f"   Filtered to {len(gt_images_per_concept_test)} concepts for {dataset_name}")
                    
                    # Get activation loader
                    try:
                        act_loader = get_sae_act_loader(dataset_name, model_type, sample_type)
                        info = act_loader.get_activation_info()
                        print(f"   Activation info: {info['total_samples']:,} samples, {info['num_concepts']} SAE units")
                    except Exception as e:
                        print(f"Error in get_sae_act_loader: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                    
                    # Create con_label for SAE
                    con_label = f"{model_type}_sae_{sample_type}_dense"
                    
                    # SAE units are treated as unsupervised concepts
                    print("   SAE units are treated as unsupervised concepts...")
                    
                    # Step 1: Compute detection metrics on TEST set
                    print("   Computing detection metrics over all pairs on TEST set")
                    compute_detection_metrics_over_percentiles_allpairs(
                        PERCENTILES,
                        gt_images_per_concept_test,  # Use image-level GT for detection metrics
                        dataset_name,
                        model_input_size,
                        DEVICE,
                        con_label,
                        act_loader,
                        scratch_dir=SCRATCH_DIR, 
                        sample_type=sample_type,
                        patch_size=14,
                        n_clusters=n_clusters
                    )
                    
                    # Step 1b: Compute detection metrics on CALIBRATION set
                    print("   Computing detection metrics over all pairs on CALIBRATION set")
                    compute_detection_metrics_over_percentiles_allpairs(
                        PERCENTILES,
                        gt_images_per_concept_cal,  # Use image-level GT for detection metrics
                        dataset_name,
                        model_input_size,
                        DEVICE,
                        con_label + "_cal",
                        act_loader,
                        scratch_dir=SCRATCH_DIR, 
                        sample_type=sample_type,
                        patch_size=14,
                        n_clusters=n_clusters
                    )

                    # Step 2: Find best clusters per concept for TEST set
                    print("   Matching concepts/clusters by detection rates for TEST set")
                    best_clusters_by_detect_test = find_best_clusters_per_concept_from_detectionmetrics(
                        dataset_name,
                        model_type,
                        sample_type,
                        metric_type='f1',
                        percentiles=PERCENTILES, 
                        con_label=con_label
                    )
                    filter_and_save_best_clusters(dataset_name, con_label)
                    
                    # Step 2b: Find best clusters per concept for CALIBRATION set
                    print("   Matching concepts/clusters by detection rates for CALIBRATION set")
                    best_clusters_by_detect_cal = find_best_clusters_per_concept_from_detectionmetrics(
                        dataset_name,
                        model_type,
                        sample_type,
                        metric_type='f1',
                        percentiles=PERCENTILES, 
                        con_label=con_label + "_cal"
                    )
                    filter_and_save_best_clusters(dataset_name, con_label + "_cal")
                    
                    # Step 3: Find best detection percentiles on calibration set
                    print("   Finding best detection percentiles on calibration set")
                    find_best_detection_percentiles_cal(dataset_name, con_label, PERCENTILES, sample_type)

                    # Step 4: Write superdetectors to file (only for patch-level analysis)
                    if sample_type == 'patch':
                        print("   Writing superdetectors")
                        # For SAE/unsupervised, we need to get matched concepts first
                        # Get the correct acts file name based on model type
                        if model_type == 'CLIP':
                            acts_file_name = f"clipscope_{sample_type}_dense.pt"
                        else:  # Gemma
                            acts_file_name = f"gemmascope_{sample_type}_dense.pt"
                        
                        matched_acts_loader, matched_gt_cal, matched_gt_test, _, _ = get_matched_concepts_and_data(
                            dataset_name,
                            con_label,
                            act_loader,
                            gt_samples_per_concept_cal=gt_samples_per_concept_cal,
                            gt_samples_per_concept_test=gt_samples_per_concept_test,
                            gt_samples_per_concept=None,
                            concepts=None,
                            acts_file=acts_file_name  # Provide the correct acts file name
                        )
                        # For SAE/unsupervised, use the matched concept names and matched ground truth
                        concept_names = list(matched_acts_loader.concept_names)
                        # Use the matched ground truth which has the same concept names as the loader
                        for percentile in tqdm(PERCENTILES):
                            find_all_superdetector_patches(percentile, matched_acts_loader, concept_names,
                                                         matched_gt_test, dataset_name, 
                                                         model_input_size, con_label, DEVICE)
                    
                    print(f"✅ Completed detection statistics for {dataset_name} - {model_type} - {sample_type}")
                    
                except FileNotFoundError as e:
                    print(f"❌ Error: Required file not found - {e}")
                    continue
                except Exception as e:
                    print(f"❌ Error processing {dataset_name} - {model_type} - {sample_type}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"\n✨ SAE detection statistics computation complete!")
    print(f"   Results saved to: {os.path.join(workspace_dir, 'Best_Detection_Percentiles_Cal')}")


if __name__ == "__main__":
    main()