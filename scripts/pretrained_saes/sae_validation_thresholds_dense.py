"""
Compute validation thresholds for dense filtered SAE activations.
This version uses the same functions as the regular pipeline.
"""

import torch
import pandas as pd
import os
import sys
import argparse
from itertools import product

# Ensure we're in the right directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(script_dir, '../..'))
os.chdir(workspace_dir)

sys.path.append(workspace_dir)

from utils.quant_concept_evals_utils import compute_concept_thresholds_over_percentiles
from utils.unsupervised_utils import compute_concept_thresholds_over_percentiles_all_pairs
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
    parser = argparse.ArgumentParser(description='Compute validation thresholds for SAE dense activations')
    parser.add_argument('--datasets', nargs='+', 
                       help='Datasets to process')
    parser.add_argument('--models', nargs='+', default=['CLIP', 'Gemma'],
                       choices=['CLIP', 'Gemma'], help='Model types to process')
    parser.add_argument('--sample-types', nargs='+', 
                       help='Sample types to process')
    parser.add_argument('--percentiles', nargs='+', type=float, 
                       help='Percentiles to compute thresholds for')
    
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
    
    print(f"🚀 Starting SAE validation threshold computation using regular pipeline functions")
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
            
            # Determine sample types - use 'patch' for both image and text
            if args.sample_types:
                sample_types = [s for s in args.sample_types if s in ['patch', 'cls']]
            else:
                sample_types = ['patch', 'cls']
            
            # Model input size
            if model_type == 'CLIP':
                model_input_size = (224, 224)
            else:  # Gemma
                model_input_size = ('text', 'text2')
            
            for sample_type in sample_types:
                print(f"\n{'='*60}")
                print(f"Processing {dataset_name} - {model_type} - {sample_type}")
                
                try:
                    # First check if dense activations exist
                    dense_path = os.path.join(SCRATCH_DIR, "SAE_Activations_Dense", dataset_name)
                    if model_type == 'CLIP':
                        dense_file = f"clipscope_{sample_type}_dense"
                    else:
                        dense_file = f"gemmascope_{sample_type}_dense"
                    
                    # Check for single file or chunked files
                    single_file = os.path.join(dense_path, f"{dense_file}.pt")
                    chunk_file = os.path.join(dense_path, f"{dense_file}_chunk_0.pt")
                    
                    if not os.path.exists(single_file) and not os.path.exists(chunk_file):
                        print(f"   ⚠️  Skipping - no dense activations found for {sample_type}")
                        continue
                    
                    # Load ground truth samples for calibration
                    if sample_type in ['patch', 'token']:
                        if sample_type == 'patch':
                            gt_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_cal_inputsize_{model_input_size}.pt"
                        else:  # token
                            gt_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_cal_inputsize_{model_input_size}.pt"
                    else:  # cls
                        gt_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_cal_inputsize_{model_input_size}.pt"
                    
                    print(f"   Loading ground truth from: {gt_file}")
                    gt_samples_per_concept_cal = torch.load(gt_file)
                    
                    # Filter to only relevant concepts for this dataset
                    gt_samples_per_concept_cal = filter_concept_dict(gt_samples_per_concept_cal, dataset_name)
                    print(f"   Filtered to {len(gt_samples_per_concept_cal)} concepts for {dataset_name}")
                    
                    # Get activation loader
                    act_loader = get_sae_act_loader(dataset_name, model_type, sample_type)
                    
                    # Verify the loader is looking in the right place
                    print(f"   Loader base path: {act_loader.base_path}")
                    print(f"   Loader folder: {act_loader.folder}")
                    
                    info = act_loader.get_activation_info()
                    print(f"   Activation info: {info['total_samples']:,} samples, {info['num_concepts']} SAE units")
                    
                    # Create con_label for SAE
                    con_label = f"{model_type}_sae_{sample_type}_dense"
                    
                    # Compute thresholds using the same function as regular pipeline
                    print("   Computing thresholds over percentiles (SAE units as concepts)...")
                    compute_concept_thresholds_over_percentiles_all_pairs(
                        act_loader, 
                        gt_samples_per_concept_cal, 
                        PERCENTILES, 
                        DEVICE,
                        dataset_name, 
                        con_label
                    )
                    
                    print(f"✅ Completed threshold computation for {dataset_name} - {model_type} - {sample_type}")
                    
                except FileNotFoundError as e:
                    print(f"❌ Error: Required file not found - {e}")
                    continue
                except Exception as e:
                    print(f"❌ Error processing {dataset_name} - {model_type} - {sample_type}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"\n✨ SAE validation threshold computation complete!")
    print(f"   Results saved to: {os.path.join(workspace_dir, 'Thresholds')}")


if __name__ == "__main__":
    main()