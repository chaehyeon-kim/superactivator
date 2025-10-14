import torch
import sys
import os
import argparse
from itertools import product
sys.path.append(os.path.abspath("../"))

import utils.gt_concept_segmentation_utils as gt_concept_segmentation_utils

from utils.gt_concept_segmentation_utils import map_concepts_to_image_indices, map_concepts_to_patch_indices, sort_mapping_by_split


PERCENT_THRU_MODEL = 100  # Default value, can be overridden by command line
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_TYPES = ['patch']
DATASETS = ['CLEVR']
MODEL_INPUT_SIZES = [(224, 224)]  # Only CLIP input size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute ground truth samples for image datasets')
    parser.add_argument('--percentthrumodel', type=int, default=100, help='Percentage through model layers to use (default: 100)')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process')
    parser.add_argument('--datasets', nargs='+', help='Multiple datasets to process')
    parser.add_argument('--sample-type', type=str, choices=['cls', 'patch'], help='Specific sample type to process')
    parser.add_argument('--sample-types', nargs='+', choices=['cls', 'patch'], help='Multiple sample types to process')
    
    args = parser.parse_args()
    
    # Update PERCENT_THRU_MODEL from command line if provided
    PERCENT_THRU_MODEL = args.percentthrumodel
    
    # Determine which datasets to process
    if args.dataset:
        datasets_to_process = [args.dataset]
    elif args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = DATASETS
        
    # Determine which sample types to process
    if args.sample_type:
        sample_types_to_process = [args.sample_type]
    elif args.sample_types:
        sample_types_to_process = args.sample_types
    else:
        sample_types_to_process = SAMPLE_TYPES
    
    experiment_configs = product(datasets_to_process, MODEL_INPUT_SIZES, sample_types_to_process)
    for dataset_name, model_input_size, sample_type in experiment_configs:
        print(f"Computing gt for dataset {dataset_name} input size {model_input_size} sample type {sample_type}")
        if sample_type == 'cls':
            gt_samples_per_concept = map_concepts_to_image_indices(dataset_name=dataset_name,
                                                                  model_input_size=model_input_size)
        else:
            gt_samples_per_concept = map_concepts_to_patch_indices(dataset_name=dataset_name,
                                                                  model_input_size=model_input_size)
        
        sort_mapping_by_split(gt_samples_per_concept, dataset_name,
                              sample_type=sample_type, model_input_size=model_input_size)
            