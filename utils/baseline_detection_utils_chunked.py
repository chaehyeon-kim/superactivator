"""
Chunked version of baseline detection that only loads the data it needs.
"""

import torch
import os
from tqdm import tqdm
from collections import defaultdict


def compute_aggregated_activation_thresholds_over_percentiles_all_pairs_chunked(
        loader, 
        gt_samples_per_concept_cal,
        percentiles,
        device,
        dataset_name,
        con_label,
        aggregation_method='max',
        model_input_size=None,
        patch_size=14,
        random_seed=42,
        batch_size=50000):  # Process this many patches at a time
    """
    Compute thresholds by loading only the patches we need in chunks.
    """
    import random
    from utils.patch_alignment_utils import compute_patches_per_image, filter_patches_by_image_presence
    from utils.general_utils import get_split_df
    
    # Set random seed if using random method
    if aggregation_method == 'random':
        random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Get method name for file naming
    method_name = {
        'max': 'maxtoken',
        'mean': 'meantoken',
        'last': 'lasttoken',
        'random': 'randomtoken'
    }[aggregation_method]
    
    cache_dir = f'Thresholds/{dataset_name}'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{method_name}_all_percentiles_allpairs_{con_label}.pt')
    
    print(f"\n   [CHUNKED] Computing {method_name} thresholds by loading only needed patches...")
    
    # Get loader info
    info = loader.get_activation_info()
    n_clusters = len(info['concept_names'])
    cluster_labels = info['concept_names']
    
    # Calculate patches per image
    patches_per_image = compute_patches_per_image(patch_size, model_input_size)
    
    # Get split info
    split_df = get_split_df(dataset_name)
    
    # Initialize storage for all thresholds
    all_thresholds = {p: {} for p in percentiles}
    
    # Process each concept
    for concept_idx, (concept_name, cal_indices) in enumerate(tqdm(gt_samples_per_concept_cal.items(), 
                                                                   desc=f"Computing {method_name} thresholds")):
        if len(cal_indices) == 0:
            continue
        
        # Filter patches and group by image
        cal_samples_tensor = torch.tensor(cal_indices, device=device)
        filtered_cal_samples = filter_patches_by_image_presence(cal_samples_tensor, dataset_name, model_input_size)
        
        if len(filtered_cal_samples) == 0:
            continue
        
        # Group patches by image
        image_to_patches = defaultdict(list)
        for patch_idx in filtered_cal_samples.cpu().numpy():
            img_idx = patch_idx // patches_per_image
            if split_df.get(img_idx) == 'cal':  # Only calibration images
                image_to_patches[img_idx].append(patch_idx)
        
        if not image_to_patches:
            continue
        
        # Process images in batches to compute aggregated values
        aggregated_values_per_cluster = defaultdict(list)
        
        # Sort patches by index for efficient chunk loading
        all_patches_needed = []
        patch_to_image = {}
        for img_idx, patches in image_to_patches.items():
            for patch_idx in patches:
                all_patches_needed.append(patch_idx)
                patch_to_image[patch_idx] = img_idx
        
        all_patches_needed.sort()
        
        # Process patches in batches
        for batch_start in range(0, len(all_patches_needed), batch_size):
            batch_end = min(batch_start + batch_size, len(all_patches_needed))
            batch_patches = all_patches_needed[batch_start:batch_end]
            
            # Load only these specific patches
            min_patch = min(batch_patches)
            max_patch = max(batch_patches)
            
            # Load the range containing our patches
            batch_acts = loader.load_tensor_range(min_patch, max_patch + 1)
            if batch_acts is None:
                continue
            
            # Extract only the patches we need
            patch_indices = torch.tensor([p - min_patch for p in batch_patches], device=device)
            needed_acts = batch_acts[patch_indices]  # Shape: [n_patches, n_clusters]
            
            # Group by image and compute aggregation
            for i, patch_idx in enumerate(batch_patches):
                img_idx = patch_to_image[patch_idx]
                
                # Store activations for this image
                if img_idx not in aggregated_values_per_cluster:
                    aggregated_values_per_cluster[img_idx] = []
                
                aggregated_values_per_cluster[img_idx].append(needed_acts[i])
            
            # Clean up
            del batch_acts, needed_acts
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Now compute aggregations for each image
        final_aggregated_values = defaultdict(list)
        
        for img_idx, act_list in aggregated_values_per_cluster.items():
            if not act_list:
                continue
                
            # Stack all activations for this image
            img_acts = torch.stack(act_list)  # Shape: [n_patches_in_image, n_clusters]
            
            # Compute aggregation
            if aggregation_method == 'max':
                agg_values = img_acts.max(dim=0)[0]
            elif aggregation_method == 'mean':
                agg_values = img_acts.mean(dim=0)
            elif aggregation_method == 'last':
                agg_values = img_acts[-1]
            elif aggregation_method == 'random':
                rand_idx = random.randint(0, len(img_acts) - 1)
                agg_values = img_acts[rand_idx]
            
            # Store aggregated values for each cluster
            for cluster_idx in range(n_clusters):
                final_aggregated_values[cluster_idx].append(agg_values[cluster_idx].item())
        
        # Compute percentiles for each cluster
        for cluster_idx in range(n_clusters):
            if cluster_idx in final_aggregated_values and len(final_aggregated_values[cluster_idx]) > 0:
                values = torch.tensor(final_aggregated_values[cluster_idx])
                sorted_values, _ = torch.sort(values)
                n_values = len(sorted_values)
                
                for p in percentiles:
                    idx = int(p * n_values)
                    if idx >= n_values:
                        idx = n_values - 1
                    threshold = sorted_values[idx].item()
                    all_thresholds[p][(concept_name, cluster_labels[cluster_idx])] = (threshold, float('nan'))
    
    # Save results
    print(f"   Saving thresholds to {cache_file}")
    torch.save(all_thresholds, cache_file)
    
    return all_thresholds