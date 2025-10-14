import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import os
import gc
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from utils.patch_alignment_utils import compute_patches_per_image, get_image_idx_from_global_patch_idx, filter_patches_by_image_presence
from utils.general_utils import get_split_df
import random
from torch.nn.utils.rnn import pad_sequence

def compute_aggregated_activation_thresholds_over_percentiles(gt_samples_per_concept_cal: Dict[str, List[int]], 
                                                            act_loader,
                                                            percentiles: List[float], 
                                                            device: str, 
                                                            dataset_name: str,
                                                            con_label: str,
                                                            aggregation_method: str = 'max',
                                                            model_input_size: Tuple = None,
                                                            patch_size: int = 14,
                                                            n_vectors: int = 1,
                                                            n_concepts_to_print: int = 0,
                                                            random_seed: int = 42,
                                                            is_kmeans: bool = False) -> None:
    """
    Compute concept detection thresholds based on aggregated activation per image/paragraph at different percentiles.
    
    This function computes thresholds using different aggregation methods across all tokens/patches
    for each image/paragraph (not individual patches/tokens).
    
    Args:
        gt_samples_per_concept_cal: Dictionary mapping concept names to lists of calibration patch/token indices
        act_loader: ChunkedActivationLoader containing activations
        percentiles: List of percentiles to compute thresholds for (e.g., [0.02, 0.05, 0.1, ...])
        device: Device to use for computation ('cuda' or 'cpu')
        dataset_name: Name of the dataset
        con_label: Label for the concept configuration
        aggregation_method: Method to aggregate activations ('max', 'mean', 'last', 'random')
        model_input_size: Model input size (needed for patch calculations)
        patch_size: Size of patches (default: 14)
        n_vectors: Number of concept vectors (default: 1)
        n_concepts_to_print: Number of concepts to print debug info for (default: 0)
        random_seed: Random seed for 'random' aggregation method
    """
    
    # Get activation info
    info = act_loader.get_activation_info()
    is_image_dataset = model_input_size and isinstance(model_input_size, tuple) and model_input_size[0] != 'text'
    
    # Calculate patches per image if needed
    if is_image_dataset:
        patches_per_image = compute_patches_per_image(patch_size, model_input_size)
    
    # Initialize storage for all percentile thresholds
    all_percentile_thresholds = {}
    
    # Get concept names from activation loader
    info = act_loader.get_activation_info()
    concept_names = info['concept_names']
    concept_to_idx = {name: idx for idx, name in enumerate(concept_names)}
    
    # Set random seed for reproducibility
    if aggregation_method == 'random':
        random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    print(f"Computing {aggregation_method} activations per image/paragraph for thresholds on GT positive samples...")
    
    # OPTIMIZATION: Import get_split_df at module level to avoid repeated imports
    split_df = get_split_df(dataset_name)
    
    # Get total samples from loader
    loader_info = act_loader.get_activation_info()
    total_samples = loader_info['total_samples']
    
    # Determine range of calibration data to load
    if is_image_dataset:
        # Get all calibration images
        cal_image_indices = [i for i in range(total_samples // patches_per_image) if split_df.get(i) == 'cal']
        if not cal_image_indices:
            print("   No calibration images found")
            return
            
        # Convert to patch indices
        min_patch_idx = min(cal_image_indices) * patches_per_image
        max_patch_idx = (max(cal_image_indices) + 1) * patches_per_image - 1
        
        print(f"   Loading activations for {len(cal_image_indices)} calibration images ({max_patch_idx - min_patch_idx + 1:,} patches)...")
        
        # Load all calibration patches
        cal_acts_tensor = act_loader.load_chunk_range(min_patch_idx, max_patch_idx + 1)
        if cal_acts_tensor is None:
            print("   Failed to load calibration data")
            return
    else:
        # Text datasets - get all calibration sentences
        tokens_file = f'GT_Samples/{dataset_name}/tokens_inputsize_{model_input_size}.pt'
        if not os.path.exists(tokens_file):
            raise FileNotFoundError(f"Tokens file not found: {tokens_file}")
        tokens_list = torch.load(tokens_file, weights_only=False)
        
        # Get calibration sentence indices
        cal_sent_indices = [i for i in range(len(tokens_list)) if split_df.get(i) == 'cal']
        if not cal_sent_indices:
            print("   No calibration sentences found")
            return
            
        # Calculate token range
        cumulative_tokens = 0
        min_token_idx = None
        max_token_idx = None
        
        for sent_idx in range(len(tokens_list)):
            sent_length = len(tokens_list[sent_idx])
            if sent_idx == min(cal_sent_indices) and min_token_idx is None:
                min_token_idx = cumulative_tokens
            if sent_idx == max(cal_sent_indices):
                max_token_idx = cumulative_tokens + sent_length - 1
            cumulative_tokens += sent_length
            
        print(f"   Loading activations for {len(cal_sent_indices)} calibration sentences ({max_token_idx - min_token_idx + 1:,} tokens)...")
        
        # Load all calibration tokens
        cal_acts_tensor = act_loader.load_chunk_range(min_token_idx, max_token_idx + 1)
    
    if cal_acts_tensor is None:
        print("   Failed to load calibration data")
        return
        
    cal_acts_tensor = cal_acts_tensor.to(device)
    
    # Create a mapping from original indices to tensor row positions
    if is_image_dataset:
        idx_to_position = {idx: pos for pos, idx in enumerate(range(min_patch_idx, max_patch_idx + 1))}
    else:
        idx_to_position = {idx: pos for pos, idx in enumerate(range(min_token_idx, max_token_idx + 1))}
    
    # OPTIMIZATION: Process all concepts in parallel and aggregate to image/paragraph level
    aggregated_values_list = []
    valid_concepts = []
    
    if is_image_dataset:
        # Load padding mask
        patch_mask = torch.load(f'GT_Samples/{dataset_name}/patches_w_image_mask_inputsize_{model_input_size}.pt', 
                               map_location=device, weights_only=False)
        
        # OPTIMIZATION: Collect all unique calibration images across all concepts first
        all_cal_images = set()
        concept_to_images = defaultdict(set)
        
        for concept_name in concept_names:
            if concept_name not in gt_samples_per_concept_cal:
                continue
                
            gt_patches = gt_samples_per_concept_cal[concept_name]
            if not gt_patches:
                continue
                
            # Group GT patches by image
            for patch_idx in gt_patches:
                img_idx = patch_idx // patches_per_image
                if split_df.get(img_idx) == 'cal':  # Only calibration images
                    all_cal_images.add(img_idx)
                    concept_to_images[concept_name].add(img_idx)
        
        if not all_cal_images:
            print("   No calibration images found")
            return
        
        print(f"   Processing {len(all_cal_images)} unique calibration images for {len(concept_to_images)} concepts...")
        
        # OPTIMIZATION: Precompute aggregated activations for ALL images and ALL concepts at once
        image_to_aggregated = {}  # img_idx -> tensor of shape [num_concepts]
        
        # Process images in batches
        batch_size = 100
        all_cal_images_list = sorted(list(all_cal_images))
        
        for batch_start in tqdm(range(0, len(all_cal_images_list), batch_size), desc="Processing image batches"):
            batch_end = min(batch_start + batch_size, len(all_cal_images_list))
            batch_images = all_cal_images_list[batch_start:batch_end]
            
            # Collect all patches for this batch of images
            batch_positions = []
            image_boundaries = []
            
            for img_idx in batch_images:
                start_patch = img_idx * patches_per_image
                end_patch = (img_idx + 1) * patches_per_image
                
                # Get patch mask for this image
                img_mask = patch_mask[start_patch:end_patch]
                if not img_mask.any():
                    image_boundaries.append(None)
                    continue
                    
                # Map patch indices to positions in loaded tensor
                image_positions = []
                for local_idx in range(patches_per_image):
                    global_patch_idx = start_patch + local_idx
                    if global_patch_idx in idx_to_position and img_mask[local_idx]:
                        image_positions.append(idx_to_position[global_patch_idx])
                
                if image_positions:
                    start_pos = len(batch_positions)
                    batch_positions.extend(image_positions)
                    end_pos = len(batch_positions)
                    image_boundaries.append((start_pos, end_pos))
                else:
                    image_boundaries.append(None)
            
            if batch_positions:
                # Get activations for all patches in batch, all concepts at once
                positions_tensor = torch.tensor(batch_positions, device=device)
                batch_acts = cal_acts_tensor[positions_tensor]  # Shape: [n_patches, n_concepts]
                
                # Aggregate for each image
                for i, img_idx in enumerate(batch_images):
                    if image_boundaries[i] is None:
                        continue
                        
                    start_pos, end_pos = image_boundaries[i]
                    img_acts = batch_acts[start_pos:end_pos]  # Shape: [n_patches_in_img, n_concepts]
                    
                    # Compute aggregation for ALL concepts at once
                    if aggregation_method == 'max':
                        agg_acts = img_acts.max(dim=0)[0]  # Shape: [n_concepts]
                    elif aggregation_method == 'mean':
                        agg_acts = img_acts.mean(dim=0)
                    elif aggregation_method == 'last':
                        agg_acts = img_acts[-1]
                    elif aggregation_method == 'random':
                        random_idx = torch.randint(0, len(img_acts), (1,), device=device).item()
                        agg_acts = img_acts[random_idx]
                    
                    image_to_aggregated[img_idx] = agg_acts
            
            # Clear memory
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Now collect aggregated values per concept from precomputed data
        for concept_idx, concept_name in enumerate(concept_names):
            if concept_name not in concept_to_images:
                continue
            
            # Get aggregated values for this concept's GT images
            gt_images = sorted(list(concept_to_images[concept_name]))
            aggregated_values = []
            
            for img_idx in gt_images:
                if img_idx in image_to_aggregated:
                    # Extract the aggregated value for this concept
                    agg_value = image_to_aggregated[img_idx][concept_idx]
                    aggregated_values.append(agg_value)
            
            # Convert to tensor and add to list
            if aggregated_values:
                aggregated_values_tensor = torch.stack(aggregated_values)
                aggregated_values_list.append(aggregated_values_tensor)
                valid_concepts.append(concept_name)
                
    else:
        # Text dataset - load tokens list to get sentence boundaries
        tokens_file = f'GT_Samples/{dataset_name}/tokens_inputsize_{model_input_size}.pt'
        if not os.path.exists(tokens_file):
            raise FileNotFoundError(f"Tokens file not found: {tokens_file}")
            
        tokens_list = torch.load(tokens_file, weights_only=False)
        print(f"   Loaded tokens for {len(tokens_list)} sentences/paragraphs")
        
        # Build mapping from global token index to sentence index
        token_to_sent_idx = {}
        current_pos = 0
        for sent_idx, sent_tokens in enumerate(tokens_list):
            for local_idx in range(len(sent_tokens)):
                token_to_sent_idx[current_pos + local_idx] = (sent_idx, local_idx)
            current_pos += len(sent_tokens)
        
        # Pre-compute global starts for efficiency
        global_starts = [0]
        for i in range(len(tokens_list)):
            global_starts.append(global_starts[-1] + len(tokens_list[i]))
        
        # OPTIMIZATION: Collect all unique calibration sentences across all concepts first
        all_cal_sentences = set()
        concept_to_sentences = defaultdict(set)
        
        for concept_name in concept_names:
            if concept_name not in gt_samples_per_concept_cal:
                continue
                
            gt_tokens = gt_samples_per_concept_cal[concept_name]
            if not gt_tokens:
                continue
            
            # Get sentences containing GT positive tokens
            for token_idx in gt_tokens:
                if token_idx in token_to_sent_idx:
                    sent_idx, _ = token_to_sent_idx[token_idx]
                    if split_df.get(sent_idx) == 'cal':  # Only calibration sentences
                        all_cal_sentences.add(sent_idx)
                        concept_to_sentences[concept_name].add(sent_idx)
        
        if not all_cal_sentences:
            print("   No calibration sentences found")
            return
        
        print(f"   Processing {len(all_cal_sentences)} unique calibration sentences for {len(concept_to_sentences)} concepts...")
        
        # OPTIMIZATION: Precompute aggregated activations for ALL sentences and ALL concepts at once
        sentence_to_aggregated = {}  # sent_idx -> tensor of shape [num_concepts]
        
        # Process sentences in batches
        batch_size = 200
        all_cal_sentences_list = sorted(list(all_cal_sentences))
        
        for batch_start in tqdm(range(0, len(all_cal_sentences_list), batch_size), desc="Processing sentence batches"):
            batch_end = min(batch_start + batch_size, len(all_cal_sentences_list))
            batch_sentences = all_cal_sentences_list[batch_start:batch_end]
            
            # Collect all tokens for this batch of sentences
            batch_positions = []
            sentence_boundaries = []
            
            for sent_idx in batch_sentences:
                sent_tokens = tokens_list[sent_idx]
                if len(sent_tokens) == 0:
                    sentence_boundaries.append(None)
                    continue
                
                # Get global token indices for this sentence
                global_start = global_starts[sent_idx]
                
                # Get positions in loaded tensor for this sentence
                sentence_positions = []
                for local_idx in range(len(sent_tokens)):
                    global_token_idx = global_start + local_idx
                    if global_token_idx in idx_to_position:
                        sentence_positions.append(idx_to_position[global_token_idx])
                
                if sentence_positions:
                    start_pos = len(batch_positions)
                    batch_positions.extend(sentence_positions)
                    end_pos = len(batch_positions)
                    sentence_boundaries.append((start_pos, end_pos))
                else:
                    sentence_boundaries.append(None)
            
            if batch_positions:
                # Get activations for all tokens in batch, all concepts at once
                positions_tensor = torch.tensor(batch_positions, device=device)
                batch_acts = cal_acts_tensor[positions_tensor]  # Shape: [n_tokens, n_concepts]
                
                # Aggregate for each sentence
                for i, sent_idx in enumerate(batch_sentences):
                    if sentence_boundaries[i] is None:
                        continue
                        
                    start_pos, end_pos = sentence_boundaries[i]
                    sent_acts = batch_acts[start_pos:end_pos]  # Shape: [n_tokens_in_sent, n_concepts]
                    
                    # Compute aggregation for ALL concepts at once
                    if aggregation_method == 'max':
                        agg_acts = sent_acts.max(dim=0)[0]  # Shape: [n_concepts]
                    elif aggregation_method == 'mean':
                        agg_acts = sent_acts.mean(dim=0)
                    elif aggregation_method == 'last':
                        agg_acts = sent_acts[-1]
                    elif aggregation_method == 'random':
                        random_idx = torch.randint(0, len(sent_acts), (1,), device=device).item()
                        agg_acts = sent_acts[random_idx]
                    
                    sentence_to_aggregated[sent_idx] = agg_acts
            
            # Clear memory
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Now collect aggregated values per concept from precomputed data
        for concept_idx, concept_name in enumerate(concept_names):
            if concept_name not in concept_to_sentences:
                continue
            
            # Get aggregated values for this concept's GT sentences
            gt_sentences = sorted(list(concept_to_sentences[concept_name]))
            aggregated_values = []
            
            for sent_idx in gt_sentences:
                if sent_idx in sentence_to_aggregated:
                    # Extract the aggregated value for this concept
                    agg_value = sentence_to_aggregated[sent_idx][concept_idx]
                    aggregated_values.append(agg_value)
            
            # Convert to tensor and add to list
            if aggregated_values:
                aggregated_values_tensor = torch.stack(aggregated_values)
                aggregated_values_list.append(aggregated_values_tensor)
                valid_concepts.append(concept_name)
    
    # OPTIMIZATION 3: Compute thresholds for all percentiles at once
    print(f"   Computing thresholds for {len(percentiles)} percentiles across {len(valid_concepts)} concepts...")
    
    # Pad sequences for batch processing
    if aggregated_values_list:
        padded_values = pad_sequence(aggregated_values_list, batch_first=True, padding_value=float('nan'))
        
        # Convert percentiles to tensor
        # Note: We compute 1 - percentile because a "0.9 percentile" threshold means
        # we want to keep 90% of GT positives, so we need the 10th percentile of values
        percentiles_tensor = torch.tensor([1 - p for p in percentiles], device=device)
        
        # Compute all thresholds at once - shape: (num_percentiles, num_concepts)
        batch_thresholds = torch.nanquantile(padded_values, percentiles_tensor, dim=1)
        
        # Organize results
        all_thresholds = {}
        for i, p in enumerate(percentiles):
            all_thresholds[p] = {}
            for j, concept in enumerate(valid_concepts):
                threshold = batch_thresholds[i, j].item()
                all_thresholds[p][concept] = (threshold, float('nan'))
    else:
        all_thresholds = {p: {} for p in percentiles}
    
    # Save results
    method_name = {
        'max': 'maxtoken',
        'mean': 'meantoken',
        'last': 'lasttoken',
        'random': 'randomtoken'
    }[aggregation_method]
    
    # Save all percentiles in one file
    all_save_path = f'Thresholds/{dataset_name}/{method_name}_all_percentiles_{con_label}.pt'
    os.makedirs(os.path.dirname(all_save_path), exist_ok=True)
    torch.save(all_thresholds, all_save_path)
    
    # Also save individual percentile files for compatibility
    for p in percentiles:
        save_path = f'Thresholds/{dataset_name}/per_{p*100}_{con_label}_{method_name}.pt'
        torch.save(all_thresholds[p], save_path)
    
    print(f"   Saved {method_name} thresholds to {all_save_path}")
    
    # Clean up memory
    del cal_acts_tensor
    if 'padded_values' in locals():
        del padded_values
    if 'batch_thresholds' in locals():
        del batch_thresholds
    torch.cuda.empty_cache()
    
    # Print sample thresholds if requested
    if n_concepts_to_print > 0:
        print(f"\nSample thresholds for first {n_concepts_to_print} concepts:")
        for i, concept in enumerate(valid_concepts[:n_concepts_to_print]):
            print(f"  {concept}:")
            for p in percentiles[:3]:  # Show first 3 percentiles
                print(f"    {p*100}%: {all_thresholds[p].get(concept, (0, 0))[0]:.4f}")


# Compatibility wrapper for old function name
def compute_max_activation_thresholds_over_percentiles(gt_samples_per_concept_cal: Dict[str, List[int]], 
                                                      act_loader,
                                                      percentiles: List[float], 
                                                      device: str, 
                                                      dataset_name: str,
                                                      con_label: str,
                                                      model_input_size: Tuple = None,
                                                      patch_size: int = 14,
                                                      n_vectors: int = 1,
                                                      n_concepts_to_print: int = 0) -> None:
    """Wrapper for backward compatibility - calls aggregated version with 'max'."""
    return compute_aggregated_activation_thresholds_over_percentiles(
        gt_samples_per_concept_cal, act_loader, percentiles, device, dataset_name,
        con_label, aggregation_method='max', model_input_size=model_input_size,
        patch_size=patch_size, n_vectors=n_vectors, n_concepts_to_print=n_concepts_to_print
    )


def compute_aggregated_detection_metrics_over_percentiles(percentiles: List[float],
                                                         gt_images_per_concept: Dict[str, List[int]],
                                                         act_loader,
                                                         dataset_name: str,
                                                         model_input_size: Tuple,
                                                         device: str,
                                                         con_label: str,
                                                         aggregation_method: str = 'max',
                                                         patch_size: int = 14,
                                                         random_seed: int = 42) -> pd.DataFrame:
    """
    Compute detection metrics using aggregated activation approach.
    
    An image/paragraph is detected if its aggregated activation across all tokens/patches
    exceeds the threshold (computed from calibration data).
    
    Args:
        percentiles: List of percentiles to evaluate
        gt_images_per_concept: Ground truth image/paragraph indices per concept
        act_loader: ChunkedActivationLoader containing activations
        dataset_name: Name of the dataset
        model_input_size: Model input size
        device: Device for computation
        con_label: Concept configuration label
        aggregation_method: Method to aggregate activations ('max', 'mean', 'last', 'random')
        patch_size: Size of patches (default: 14)
        random_seed: Random seed for 'random' aggregation method
        
    Returns:
        DataFrame containing detection metrics
    """
    
    # Get method name for file loading
    method_name = {
        'max': 'maxtoken',
        'mean': 'meantoken',
        'last': 'lasttoken',
        'random': 'randomtoken'
    }[aggregation_method]
    
    # Set random seed if using random method
    if aggregation_method == 'random':
        random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Load thresholds computed on calibration set
    # Remove "_cal" suffix if present since thresholds are always computed on calibration set
    threshold_con_label = con_label.replace("_cal", "") if con_label.endswith("_cal") else con_label
    threshold_file = f"Thresholds/{dataset_name}/{method_name}_all_percentiles_{threshold_con_label}.pt"
    if not os.path.exists(threshold_file):
        raise FileNotFoundError(f"Threshold file not found: {threshold_file}. Run {method_name} threshold computation first.")
    
    all_thresholds = torch.load(threshold_file)
    
    # Get activation info
    info = act_loader.get_activation_info()
    total_samples = info['total_samples']
    num_concepts = info['num_concepts']
    
    # Determine if we're evaluating calibration or test set
    is_calibration = "_cal" in con_label
    
    # Check if this is an image or text dataset
    is_image_dataset = model_input_size and isinstance(model_input_size, tuple) and model_input_size[0] != 'text'
    
    # Calculate patches per image if needed
    if is_image_dataset:
        patches_per_image = compute_patches_per_image(patch_size, model_input_size)
        num_images = total_samples // patches_per_image
    else:
        # For text, we need to determine number of paragraphs
        # This should come from dataset metadata
        num_paragraphs = total_samples  # This is a simplification - should be fixed
    
    # Results storage
    results = []
    
    # Get concept names from loader
    info = act_loader.get_activation_info()
    concept_names = info['concept_names']
    concept_to_idx = {name: idx for idx, name in enumerate(concept_names)}
    
    print(f"Computing baseline detection metrics for {len(concept_names)} concepts...")
    
    # Pre-load padding mask once for image datasets
    if is_image_dataset:
        patch_mask = torch.load(f'GT_Samples/{dataset_name}/patches_w_image_mask_inputsize_{model_input_size}.pt', 
                               map_location=device, weights_only=False)
    
    # MAJOR OPTIMIZATION: Compute aggregated activations ONCE before the percentile loop
    print(f"Computing {aggregation_method} aggregated activations once for all percentiles...")
    
    if is_image_dataset:
        # Get split info
        split_df = get_split_df(dataset_name)
        
        # Create image mask based on whether we're evaluating calibration or test set
        if is_calibration:
            image_mask = torch.tensor([split_df.get(i) == 'cal' for i in range(num_images)], device=device)
        else:
            image_mask = torch.tensor([split_df.get(i) == 'test' for i in range(num_images)], device=device)
        
        num_images_to_process = image_mask.sum().item()
        print(f"   Processing {num_images_to_process} {'calibration' if is_calibration else 'test'} images...")
        
        # Load all activations once
        print(f"   Loading all activations...")
        all_acts = act_loader.load_tensor_range(0, total_samples)
        if all_acts is None:
            raise RuntimeError("Failed to load activations")
        all_acts = all_acts.to(device)
        
        # Reshape to [num_images, patches_per_image, num_concepts]
        reshaped_acts = all_acts.reshape(num_images, patches_per_image, num_concepts)
        
        # Apply patch mask efficiently
        patch_mask_reshaped = patch_mask.reshape(num_images, patches_per_image).to(device)
        
        # Compute aggregation for all images at once
        print(f"   Computing {aggregation_method} activations across all images...")
        if aggregation_method == 'max':
            # Set masked patches to -inf so they don't affect max
            masked_acts = reshaped_acts.clone()
            masked_acts[~patch_mask_reshaped.unsqueeze(-1).expand_as(reshaped_acts)] = float('-inf')
            agg_acts = masked_acts.max(dim=1)[0]  # Shape: [num_images, num_concepts]
        elif aggregation_method == 'mean':
            # Compute mean only over valid patches
            masked_acts = reshaped_acts * patch_mask_reshaped.unsqueeze(-1).float()
            valid_counts = patch_mask_reshaped.sum(dim=1, keepdim=True).clamp(min=1)
            agg_acts = masked_acts.sum(dim=1) / valid_counts
        elif aggregation_method == 'last':
            # Vectorized: Get last valid patch for each image
            agg_acts = torch.zeros(num_images, num_concepts, device=device)
            # Find last valid index for each image
            reversed_mask = patch_mask_reshaped.flip(dims=[1])
            last_valid_idx = patches_per_image - 1 - reversed_mask.argmax(dim=1)
            # Use advanced indexing to get values
            valid_images = patch_mask_reshaped.any(dim=1)
            agg_acts[valid_images] = reshaped_acts[valid_images, last_valid_idx[valid_images]]
        elif aggregation_method == 'random':
            # OPTIMIZATION: Fully vectorized random selection
            agg_acts = torch.zeros(num_images, num_concepts, device=device)
            valid_counts = patch_mask_reshaped.sum(dim=1)
            valid_images_mask = valid_counts > 0
            
            if valid_images_mask.any():
                # Generate random indices for valid images
                valid_images_idx = valid_images_mask.nonzero(as_tuple=True)[0]
                random_factors = torch.rand(len(valid_images_idx), device=device)
                random_indices = (random_factors * valid_counts[valid_images_idx]).floor().long()
                
                # Create cumulative sum for mapping
                cumsum = patch_mask_reshaped.cumsum(dim=1)
                
                # Vectorized patch selection using gather
                selected_patches = torch.zeros(len(valid_images_idx), dtype=torch.long, device=device)
                for i, (img_idx, rand_idx) in enumerate(zip(valid_images_idx, random_indices)):
                    # Find the patch where cumsum equals rand_idx + 1
                    target_cumsum = rand_idx + 1
                    patch_idx = (cumsum[img_idx] == target_cumsum).nonzero(as_tuple=True)[0]
                    if len(patch_idx) > 0:
                        selected_patches[i] = patch_idx[0]
                
                # Gather values using advanced indexing
                agg_acts[valid_images_idx] = reshaped_acts[valid_images_idx, selected_patches]
        
        # Clean up the large tensors we don't need anymore
        del all_acts, reshaped_acts
        if 'masked_acts' in locals():
            del masked_acts
        torch.cuda.empty_cache()
        
        # Only keep relevant images' aggregated activations
        relevant_images_idx = image_mask.nonzero(as_tuple=True)[0]
        relevant_agg_acts = agg_acts[relevant_images_idx] if len(relevant_images_idx) > 0 else torch.empty(0, num_concepts, device=device)
    
    else:
        # Text datasets - similar optimization
        tokens_file = f'GT_Samples/{dataset_name}/tokens_inputsize_{model_input_size}.pt'
        if not os.path.exists(tokens_file):
            raise FileNotFoundError(f"Tokens file not found: {tokens_file}")
        tokens_list = torch.load(tokens_file, weights_only=False)
        
        # Get split info
        split_df = get_split_df(dataset_name)
        
        # Create sentence mask for cal/test filtering
        num_sentences = len(tokens_list)
        if is_calibration:
            sentence_mask = torch.tensor([split_df.get(i) == 'cal' for i in range(num_sentences)], device=device)
        else:
            sentence_mask = torch.tensor([split_df.get(i) == 'test' for i in range(num_sentences)], device=device)
        
        num_sentences_to_process = sentence_mask.sum().item()
        print(f"   Processing {num_sentences_to_process} {'calibration' if is_calibration else 'test'} sentences...")
        
        # Build sentence boundaries and lengths
        sentence_boundaries = []
        sentence_lengths = []
        pos = 0
        for sent_tokens in tokens_list:
            sent_len = len(sent_tokens)
            sentence_boundaries.append((pos, pos + sent_len))
            sentence_lengths.append(sent_len)
            pos += sent_len
        
        # Load all activations at once
        print(f"   Loading all text activations...")
        all_acts = act_loader.load_tensor_range(0, total_samples)
        if all_acts is None:
            raise RuntimeError("Failed to load activations")
        all_acts = all_acts.to(device)
        
        # Pre-allocate aggregated activations for all sentences
        agg_acts = torch.zeros(num_sentences, num_concepts, device=device)
        
        # Compute aggregation for all sentences
        print(f"   Computing {aggregation_method} activations for all sentences...")
        
        # Pre-generate random indices for all sentences if needed
        if aggregation_method == 'random':
            random_indices = []
            for start, end in sentence_boundaries:
                if end > start:
                    random_indices.append(start + torch.randint(end - start, (1,)).item())
                else:
                    random_indices.append(start)
        
        # Process sentences in batches for better cache efficiency
        batch_size = 1000
        for batch_start in range(0, num_sentences, batch_size):
            batch_end = min(batch_start + batch_size, num_sentences)
            
            for sent_idx in range(batch_start, batch_end):
                start, end = sentence_boundaries[sent_idx]
                if end > start:  # Non-empty sentence
                    if aggregation_method == 'max':
                        agg_acts[sent_idx] = all_acts[start:end].max(dim=0)[0]
                    elif aggregation_method == 'mean':
                        agg_acts[sent_idx] = all_acts[start:end].mean(dim=0)
                    elif aggregation_method == 'last':
                        agg_acts[sent_idx] = all_acts[end-1]
                    elif aggregation_method == 'random':
                        agg_acts[sent_idx] = all_acts[random_indices[sent_idx]]
        
        # Clean up
        del all_acts
        torch.cuda.empty_cache()
        
        # Only keep relevant sentences' aggregated activations
        relevant_sentences_idx = sentence_mask.nonzero(as_tuple=True)[0]
        relevant_agg_acts = agg_acts[relevant_sentences_idx] if len(relevant_sentences_idx) > 0 else torch.empty(0, num_concepts, device=device)
    
    # Now iterate through percentiles using the pre-computed aggregated activations
    for percentile in tqdm(percentiles, desc="Evaluating percentiles"):
        if percentile not in all_thresholds:
            print(f"Warning: No thresholds found for percentile {percentile}")
            continue
            
        percentile_thresholds = all_thresholds[percentile]
        
        # OPTIMIZATION: Process concepts in batches
        # First, prepare threshold tensor for all concepts
        threshold_tensor = torch.full((num_concepts,), float('inf'), device=device)
        valid_concepts = []
        
        for concept_idx, concept_name in enumerate(concept_names):
            if concept_name in percentile_thresholds and concept_name in gt_images_per_concept:
                threshold_data = percentile_thresholds[concept_name]
                if isinstance(threshold_data, tuple):
                    threshold_tensor[concept_idx] = threshold_data[0]
                else:
                    threshold_tensor[concept_idx] = threshold_data
                valid_concepts.append((concept_idx, concept_name))
        
        if not valid_concepts:
            continue
            
        # Process all valid concepts together using pre-computed aggregated activations
        all_detections = {}  # concept_name -> set of detected items
        
        if is_image_dataset:
            # Apply thresholds to pre-computed aggregated activations
            if len(relevant_images_idx) > 0 and relevant_agg_acts.shape[0] > 0:
                # Compare against thresholds for this percentile
                detections = relevant_agg_acts >= threshold_tensor.unsqueeze(0)  # Shape: [num_relevant_images, num_concepts]
                
                # Extract detected images for each concept
                for concept_idx, concept_name in valid_concepts:
                    detected_mask = detections[:, concept_idx]
                    if detected_mask.any():
                        # Map back to original image indices
                        detected_idx = relevant_images_idx[detected_mask]
                        all_detections[concept_name] = set(detected_idx.tolist())
                    else:
                        all_detections[concept_name] = set()
            else:
                # No relevant images, all concepts have empty detection sets
                for _, concept_name in valid_concepts:
                    all_detections[concept_name] = set()
        else:
            # Text datasets
            if len(relevant_sentences_idx) > 0 and relevant_agg_acts.shape[0] > 0:
                # Compare against thresholds for this percentile
                detections = relevant_agg_acts >= threshold_tensor.unsqueeze(0)  # Shape: [num_relevant_sentences, num_concepts]
                
                # Extract detected sentences for each concept
                for concept_idx, concept_name in valid_concepts:
                    detected_mask = detections[:, concept_idx]
                    if detected_mask.any():
                        # Map back to original sentence indices
                        detected_idx = relevant_sentences_idx[detected_mask]
                        all_detections[concept_name] = set(detected_idx.tolist())
                    else:
                        all_detections[concept_name] = set()
            else:
                # No relevant sentences, all concepts have empty detection sets
                for _, concept_name in valid_concepts:
                    all_detections[concept_name] = set()
        
        # Compute metrics for all concepts
        for concept_idx, concept_name in valid_concepts:
            # Get ground truth and detected samples
            gt_positive_samples = set(gt_images_per_concept.get(concept_name, []))
            detected_samples = all_detections.get(concept_name, set())
            
            # Compute metrics
            tp = len(detected_samples & gt_positive_samples)
            fp = len(detected_samples - gt_positive_samples)
            fn = len(gt_positive_samples - detected_samples)
            
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store results
            results.append({
                'concept': concept_name,
                'percentile': percentile,
                'threshold': threshold_tensor[concept_idx].item(),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'n_detected': len(detected_samples),
                'n_gt_positive': len(gt_positive_samples)
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    save_dir = f"Quant_Results/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Use appropriate filename based on whether this is calibration or test evaluation
    if is_calibration:
        # Remove the "_cal" suffix from con_label for the base name
        base_con_label = con_label.replace("_cal", "")
        save_path = f"{save_dir}/detectfirst_{method_name}_cal_{base_con_label}.csv"
    else:
        save_path = f"{save_dir}/detectfirst_{method_name}_test_{con_label}.csv"
    
    results_df.to_csv(save_path, index=False)
    print(f"Saved {method_name} detection results to {save_path}")
    
    # Print summary statistics
    if len(results_df) > 0:
        best_f1_per_concept = results_df.groupby('concept')['f1'].max()
        print(f"\n{method_name.capitalize()} detection summary:")
        print(f"  Mean best F1 across concepts: {best_f1_per_concept.mean():.4f}")
        print(f"  Concepts with F1 > 0.5: {(best_f1_per_concept > 0.5).sum()} / {len(best_f1_per_concept)}")
    
    # Clean up the pre-computed aggregated activations
    del relevant_agg_acts
    if 'agg_acts' in locals():
        del agg_acts
    torch.cuda.empty_cache()
    
    return results_df


# Compatibility wrapper for old function name
def compute_baseline_detection_metrics_over_percentiles(percentiles: List[float],
                                                       gt_images_per_concept: Dict[str, List[int]],
                                                       act_loader,
                                                       dataset_name: str,
                                                       model_input_size: Tuple,
                                                       device: str,
                                                       con_label: str,
                                                       patch_size: int = 14) -> pd.DataFrame:
    """Wrapper for backward compatibility - calls aggregated version with 'max'."""
    return compute_aggregated_detection_metrics_over_percentiles(
        percentiles, gt_images_per_concept, act_loader, dataset_name,
        model_input_size, device, con_label, aggregation_method='max',
        patch_size=patch_size
    )


def find_best_aggregated_detection_percentiles_cal(dataset_name: str,
                                                  con_label: str,
                                                  percentiles: List[float],
                                                  sample_type: str,
                                                  aggregation_method: str = 'max') -> None:
    """
    Find the best detection percentile for each concept based on calibration F1 scores.
    
    This function analyzes the calibration detection results to find which percentile
    threshold gives the best F1 score for each concept.
    
    Args:
        dataset_name: Name of the dataset
        con_label: Concept configuration label
        percentiles: List of percentiles that were evaluated
        sample_type: 'patch' or 'cls'
        aggregation_method: Method used for aggregation
    """
    
    # Get method name for file loading
    method_name = {
        'max': 'maxtoken',
        'mean': 'meantoken',
        'last': 'lasttoken',
        'random': 'randomtoken'
    }[aggregation_method]
    
    # Load calibration results
    cal_results_file = f"Quant_Results/{dataset_name}/detectfirst_{method_name}_cal_{con_label}.csv"
    if not os.path.exists(cal_results_file):
        print(f"Warning: Calibration results not found at {cal_results_file}")
        return
    
    cal_df = pd.read_csv(cal_results_file)
    
    # Find best percentile for each concept based on F1
    best_percentiles = {}
    best_f1_scores = {}
    
    for concept in cal_df['concept'].unique():
        concept_df = cal_df[cal_df['concept'] == concept]
        best_idx = concept_df['f1'].idxmax()
        best_row = concept_df.loc[best_idx]
        
        best_percentiles[concept] = best_row['percentile']
        best_f1_scores[concept] = best_row['f1']
    
    # Save best percentiles
    best_percentiles_data = {
        'best_percentiles': best_percentiles,
        'best_f1_scores': best_f1_scores,
        'percentiles_evaluated': percentiles,
        'sample_type': sample_type
    }
    
    save_path = f"Quant_Results/{dataset_name}/{method_name}_best_percentiles_{con_label}.pt"
    torch.save(best_percentiles_data, save_path)
    print(f"Saved best {method_name} percentiles to {save_path}")
    
    # Print summary
    print(f"\nBest {method_name} detection percentiles summary:")
    print(f"  Average best F1: {torch.tensor(list(best_f1_scores.values())).mean().item():.4f}")
    print(f"  Most common best percentile: {pd.Series(list(best_percentiles.values())).mode().values[0]}")


def find_best_baseline_detection_percentiles_cal(dataset_name: str, 
                                                con_label: str, 
                                                percentiles: List[float],
                                                sample_type: str) -> None:
    """Wrapper for backward compatibility - calls aggregated version with 'max'."""
    return find_best_aggregated_detection_percentiles_cal(
        dataset_name, con_label, percentiles, sample_type, aggregation_method='max'
    )


def compute_aggregated_activation_thresholds_over_percentiles_all_pairs(loader, 
                                                                       gt_samples_per_concept_cal: Dict[str, List[int]],
                                                                       percentiles: List[float],
                                                                       device: str,
                                                                       dataset_name: str,
                                                                       con_label: str,
                                                                       aggregation_method: str = 'max',
                                                                       model_input_size: Tuple = None,
                                                                       patch_size: int = 14,
                                                                       random_seed: int = 42) -> Dict:
    """
    Computes activation thresholds for every (concept, cluster) pair using aggregated activations.
    
    This function computes thresholds for kmeans concepts using all-pairs matching,
    where each concept can potentially match with any cluster.
    
    Args:
        loader: ChunkedActivationLoader containing activations
        gt_samples_per_concept_cal: Dictionary mapping concept names to calibration sample indices
        percentiles: List of percentiles to compute thresholds for
        device: Device to use for computation
        dataset_name: Name of the dataset
        con_label: Label for the concept configuration
        aggregation_method: Method to aggregate activations ('max', 'mean', 'last', 'random')
        model_input_size: Model input size (for patch calculations)
        patch_size: Size of patches
        random_seed: Random seed for 'random' aggregation
        
    Returns:
        Dictionary mapping percentile -> {(concept, cluster): (threshold, nan)}
    """
    import os
    import gc
    from tqdm import tqdm
    from collections import defaultdict
    
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
    
    all_thresholds = {}
    new_percentiles = set(percentiles)
    print(f"   Computing {method_name} thresholds for all percentiles: {sorted(new_percentiles)}")
    
    # Initialize threshold storage
    for p in new_percentiles:
        all_thresholds[p] = {}
    
    # Get loader info
    info = loader.get_activation_info()
    is_image_dataset = model_input_size and isinstance(model_input_size, tuple) and model_input_size[0] != 'text'
    
    # Calculate patches per image if needed
    if is_image_dataset:
        patches_per_image = compute_patches_per_image(patch_size, model_input_size)
        num_images = info['total_samples'] // patches_per_image
    else:
        # For text datasets, need token counts
        token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
        if os.path.exists(token_counts_file):
            token_counts = torch.load(token_counts_file, weights_only=False)
            tokens_per_paragraph = [sum(sent_tokens) if isinstance(sent_tokens, list) else sent_tokens 
                                  for sent_tokens in token_counts]
            num_paragraphs = len(tokens_per_paragraph)
        else:
            # Fallback - treat each token as its own paragraph
            num_paragraphs = info['total_samples']
            tokens_per_paragraph = [1] * num_paragraphs
    
    # Get cluster labels
    cluster_labels = info['concept_names']
    n_clusters = len(cluster_labels)
    
    # Collect ALL calibration patches across ALL concepts first (like regular pipeline)
    if is_image_dataset:
        split_df = get_split_df(dataset_name)
        
        all_cal_patches = set()
        concept_to_patches = defaultdict(set)
        
        for concept_name, cal_indices in gt_samples_per_concept_cal.items():
            if len(cal_indices) == 0:
                continue
            
            # Filter out padding patches
            cal_samples_tensor = torch.tensor(cal_indices, device=device)
            filtered_cal_samples = filter_patches_by_image_presence(cal_samples_tensor, dataset_name, model_input_size)
            
            for patch_idx in filtered_cal_samples.cpu().numpy():
                img_idx = patch_idx // patches_per_image
                if split_df.get(img_idx) == 'cal':  # Only calibration images
                    all_cal_patches.add(patch_idx)
                    concept_to_patches[concept_name].add(patch_idx)
        
        all_cal_patches = sorted(list(all_cal_patches))
        
        if not all_cal_patches:
            print("   No calibration patches found")
            return all_thresholds
        
        print(f"   Processing {len(all_cal_patches):,} calibration patches...")
        
        # Process calibration patches in chunks
        chunk_size = 100000  # Same as regular pipeline
        
        # Store activations by concept and image
        concept_image_values = defaultdict(lambda: defaultdict(list))  # concept -> img_idx -> list of patch activations
        
        for chunk_start in tqdm(range(0, len(all_cal_patches), chunk_size), 
                               desc=f"Loading {len(all_cal_patches):,} cal patches"):
            chunk_end = min(chunk_start + chunk_size, len(all_cal_patches))
            chunk_patches = all_cal_patches[chunk_start:chunk_end]
            
            # Load this chunk efficiently
            min_idx = min(chunk_patches)
            max_idx = max(chunk_patches)
            
            # Load the range for this chunk
            chunk_range = loader.load_chunk_range(min_idx, max_idx + 1)
            if chunk_range is None:
                continue
            
            # Extract only the patches we need
            local_positions = [idx - min_idx for idx in chunk_patches]
            chunk_acts = chunk_range[local_positions].to(device)  # Shape: [n_patches, n_clusters]
            del chunk_range
            
            # Map each patch to its concept(s) and image
            for i, patch_idx in enumerate(chunk_patches):
                img_idx = patch_idx // patches_per_image
                
                # Find which concepts this patch belongs to
                for concept_name, concept_patches in concept_to_patches.items():
                    if patch_idx in concept_patches:
                        concept_image_values[concept_name][img_idx].append(chunk_acts[i])
            
            # Clean up
            del chunk_acts
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Now compute aggregated values and thresholds for each concept
        print(f"   Computing {aggregation_method} aggregation and thresholds...")
        
        for concept_name in tqdm(gt_samples_per_concept_cal.keys(), desc="Processing concepts"):
            if concept_name not in concept_image_values:
                continue
            
            # Aggregate patches by image
            aggregated_values_all_clusters = []  # Will be shape [n_images, n_clusters]
            
            for img_idx, patch_acts_list in concept_image_values[concept_name].items():
                if not patch_acts_list:
                    continue
                
                # Stack patches for this image
                img_acts = torch.stack(patch_acts_list)  # Shape: [n_patches_in_image, n_clusters]
                
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
                
                aggregated_values_all_clusters.append(agg_values)
            
            if aggregated_values_all_clusters:
                # Stack all aggregated values for this concept
                all_agg_values = torch.stack(aggregated_values_all_clusters)  # Shape: [n_images, n_clusters]
                
                # Compute percentiles for each cluster
                for cluster_idx, cluster_label in enumerate(cluster_labels):
                    cluster_values = all_agg_values[:, cluster_idx]
                    
                    if len(cluster_values) > 0:
                        # Compute percentiles
                        percentiles_tensor = torch.tensor([1 - p for p in new_percentiles], device=device)
                        thresholds = torch.quantile(cluster_values, percentiles_tensor, interpolation='linear')
                        
                        # Store thresholds
                        for p_idx, p in enumerate(new_percentiles):
                            all_thresholds[p][(concept_name, cluster_label)] = (thresholds[p_idx].item(), float('nan'))
    else:
        # Text dataset implementation
        # Get cluster labels from loader
        loader_info = loader.get_activation_info()
        cluster_labels = [str(i) for i in range(loader_info['num_concepts'])]
        
        # Load token information to map tokens to sentences/paragraphs
        tokens_file = f'GT_Samples/{dataset_name}/tokens_inputsize_{model_input_size}.pt'
        if not os.path.exists(tokens_file):
            print(f"   Tokens file not found: {tokens_file}")
            return all_thresholds
            
        tokens_info = torch.load(tokens_file, weights_only=False)
        # Handle both dict and list formats
        if isinstance(tokens_info, dict):
            all_tokens = tokens_info['all_tokens']
        else:
            all_tokens = tokens_info
        
        # Get split df
        split_df = get_split_df(dataset_name)
        
        all_cal_tokens = set()
        concept_to_tokens = defaultdict(set)
        
        for concept_name, cal_indices in gt_samples_per_concept_cal.items():
            if len(cal_indices) == 0:
                continue
            
            for token_idx in cal_indices:
                # Map token to sentence/paragraph
                sent_idx = None
                current_pos = 0
                for i, tokens_list in enumerate(all_tokens):
                    if current_pos <= token_idx < current_pos + len(tokens_list):
                        sent_idx = i
                        break
                    current_pos += len(tokens_list)
                
                if sent_idx is not None and split_df.get(sent_idx) == 'cal':
                    all_cal_tokens.add(token_idx)
                    concept_to_tokens[concept_name].add(token_idx)
        
        all_cal_tokens = sorted(list(all_cal_tokens))
        
        if not all_cal_tokens:
            print("   No calibration tokens found")
            return all_thresholds
        
        print(f"   Processing {len(all_cal_tokens):,} calibration tokens...")
        
        # Process calibration tokens in chunks
        chunk_size = 100000
        
        # Store activations by concept and sentence
        concept_sent_values = defaultdict(lambda: defaultdict(list))  # concept -> sent_idx -> list of token activations
        
        # Build token to sentence mapping
        token_to_sent = {}
        current_pos = 0
        for sent_idx, tokens_list in enumerate(all_tokens):
            for local_idx in range(len(tokens_list)):
                token_to_sent[current_pos + local_idx] = sent_idx
            current_pos += len(tokens_list)
        
        for chunk_start in tqdm(range(0, len(all_cal_tokens), chunk_size), 
                               desc=f"Loading {len(all_cal_tokens):,} cal tokens"):
            chunk_end = min(chunk_start + chunk_size, len(all_cal_tokens))
            chunk_tokens = all_cal_tokens[chunk_start:chunk_end]
            
            # Load this chunk efficiently
            min_idx = min(chunk_tokens)
            max_idx = max(chunk_tokens)
            
            # Load the range for this chunk
            chunk_range = loader.load_chunk_range(min_idx, max_idx + 1)
            if chunk_range is None:
                continue
            
            # Extract only the tokens we need
            local_positions = [idx - min_idx for idx in chunk_tokens]
            chunk_acts = chunk_range[local_positions].to(device)  # Shape: [n_tokens, n_clusters]
            del chunk_range
            
            # Map each token to its concept(s) and sentence
            for i, token_idx in enumerate(chunk_tokens):
                sent_idx = token_to_sent.get(token_idx)
                if sent_idx is None:
                    continue
                
                # Find which concepts this token belongs to
                for concept_name, concept_tokens in concept_to_tokens.items():
                    if token_idx in concept_tokens:
                        concept_sent_values[concept_name][sent_idx].append(chunk_acts[i])
            
            # Clean up
            del chunk_acts
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Now compute aggregated values and thresholds for each concept
        print(f"   Computing {aggregation_method} aggregation and thresholds...")
        
        for concept_name in tqdm(gt_samples_per_concept_cal.keys(), desc="Processing concepts"):
            if concept_name not in concept_sent_values:
                continue
            
            # Aggregate tokens by sentence
            aggregated_values_all_clusters = []  # Will be shape [n_sentences, n_clusters]
            
            for sent_idx, token_acts_list in concept_sent_values[concept_name].items():
                if not token_acts_list:
                    continue
                
                # Stack tokens for this sentence
                sent_acts = torch.stack(token_acts_list)  # Shape: [n_tokens_in_sent, n_clusters]
                
                # Compute aggregation
                if aggregation_method == 'max':
                    agg_values = sent_acts.max(dim=0)[0]
                elif aggregation_method == 'mean':
                    agg_values = sent_acts.mean(dim=0)
                elif aggregation_method == 'last':
                    agg_values = sent_acts[-1]
                elif aggregation_method == 'random':
                    rand_idx = random.randint(0, len(sent_acts) - 1)
                    agg_values = sent_acts[rand_idx]
                
                aggregated_values_all_clusters.append(agg_values)
            
            if aggregated_values_all_clusters:
                # Stack all aggregated values for this concept
                all_agg_values = torch.stack(aggregated_values_all_clusters)  # Shape: [n_sentences, n_clusters]
                
                # Compute percentiles for each cluster
                for cluster_idx, cluster_label in enumerate(cluster_labels):
                    cluster_values = all_agg_values[:, cluster_idx]
                    
                    if len(cluster_values) > 0:
                        # Compute percentiles
                        percentiles_tensor = torch.tensor([1 - p for p in new_percentiles], device=device)
                        thresholds = torch.quantile(cluster_values, percentiles_tensor, interpolation='linear')
                        
                        # Store thresholds
                        for p_idx, p in enumerate(new_percentiles):
                            all_thresholds[p][(concept_name, cluster_label)] = (thresholds[p_idx].item(), float('nan'))
    
    # Save thresholds
    torch.save(all_thresholds, cache_file)
    print(f"   Saved {method_name} all-pairs thresholds to {cache_file}")
    
    return all_thresholds


def compute_aggregated_detection_metrics_over_percentiles_allpairs(percentiles: List[float],
                                                                  gt_images_per_concept_split: Dict[str, List[int]],
                                                                  dataset_name: str,
                                                                  model_input_size: Tuple,
                                                                  device: str,
                                                                  con_label: str,
                                                                  loader,
                                                                  aggregation_method: str = 'max',
                                                                  sample_type: str = 'patch',
                                                                  patch_size: int = 14,
                                                                  n_clusters: int = 1000,
                                                                  random_seed: int = 42,
                                                                  scratch_dir: str = '') -> None:
    """
    Computes detection metrics over multiple percentiles for all (concept, cluster) pairs using aggregated activations.
    
    This function is the aggregated version of compute_detection_metrics_over_percentiles_allpairs,
    supporting different aggregation methods (max, mean, last, random) for kmeans concepts.
    
    Args:
        percentiles: List of percentiles to evaluate
        gt_images_per_concept_split: Ground truth image/paragraph indices per concept
        dataset_name: Name of the dataset
        model_input_size: Model input size
        device: Device for computation
        con_label: Concept configuration label
        loader: ChunkedActivationLoader
        aggregation_method: Method to aggregate activations ('max', 'mean', 'last', 'random')
        sample_type: 'patch' or 'cls'
        patch_size: Size of patches
        n_clusters: Number of clusters
        random_seed: Random seed for 'random' aggregation
        scratch_dir: Scratch directory (for compatibility)
    """
    import os
    import gc
    from tqdm import tqdm
    from collections import defaultdict
    from utils.general_utils import get_split_df
    from utils.quant_concept_evals_utils import compute_stats_from_counts
    
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
    
    # Ensure device
    device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
    
    # Get actual number of clusters from loader
    info = loader.get_activation_info()
    n_clusters = info['num_concepts']
    
    # Load thresholds for all percentiles
    threshold_label = con_label.replace("_cal", "") if con_label.endswith("_cal") else con_label
    threshold_file = f'Thresholds/{dataset_name}/{method_name}_all_percentiles_allpairs_{threshold_label}.pt'
    
    if not os.path.exists(threshold_file):
        raise FileNotFoundError(f"Threshold file not found: {threshold_file}. Run {method_name} all-pairs threshold computation first.")
    
    thresholds = torch.load(threshold_file, weights_only=False)
    
    # Get split info
    split_df = get_split_df(dataset_name)
    if con_label.endswith("_cal"):
        eval_split = 'cal'
    else:
        eval_split = 'test'
    
    # Create masks for the evaluation split
    split_array = split_df.values if hasattr(split_df, 'values') else split_df.to_numpy()
    eval_mask = torch.tensor(split_array == eval_split, device=device)
    eval_indices_set = set(torch.where(eval_mask)[0].cpu().numpy())
    
    # Check if this is an image or text dataset
    is_image_dataset = model_input_size and isinstance(model_input_size, tuple) and model_input_size[0] != 'text'
    
    print(f"   Computing {method_name} detection metrics for {eval_split} set...")
    
    # Precompute aggregated activations for all images/paragraphs
    if sample_type == 'patch':
        if is_image_dataset:
            # Image dataset with patches
            patches_per_image = compute_patches_per_image(patch_size, model_input_size)
            total_patches = info['total_samples']
            num_images = total_patches // patches_per_image
            
            # Load padding mask
            patch_mask = torch.load(f'GT_Samples/{dataset_name}/patches_w_image_mask_inputsize_{model_input_size}.pt',
                                   map_location=device, weights_only=False)
            
            # Allocate storage for aggregated activations
            aggregated_activations = torch.zeros(num_images, n_clusters, dtype=torch.float32)
            
            # Process images in chunks
            images_per_chunk = 500
            for img_start in tqdm(range(0, num_images, images_per_chunk), desc="Computing aggregated activations"):
                img_end = min(img_start + images_per_chunk, num_images)
                
                # Load patch activations for this chunk of images
                patch_start = img_start * patches_per_image
                patch_end = img_end * patches_per_image
                
                chunk_acts = loader.load_tensor_range(patch_start, patch_end)
                num_images_in_chunk = img_end - img_start
                
                if device == 'cuda' and not chunk_acts.is_cuda:
                    chunk_acts = chunk_acts.cuda()
                
                # Reshape to [num_images_in_chunk, patches_per_image, n_clusters]
                reshaped = chunk_acts.reshape(num_images_in_chunk, patches_per_image, -1)
                
                # Process each image
                for local_img_idx in range(num_images_in_chunk):
                    global_img_idx = img_start + local_img_idx
                    
                    # Get mask for this image's patches
                    img_patch_start = global_img_idx * patches_per_image
                    img_patch_end = img_patch_start + patches_per_image
                    img_mask = patch_mask[img_patch_start:img_patch_end]
                    
                    # Get valid patches (non-padding)
                    valid_indices = torch.where(img_mask == 1)[0]
                    
                    if len(valid_indices) > 0:
                        valid_acts = reshaped[local_img_idx, valid_indices]  # [n_valid_patches, n_clusters]
                        
                        # Compute aggregated activation based on method
                        if aggregation_method == 'max':
                            agg_acts = valid_acts.max(dim=0)[0]
                        elif aggregation_method == 'mean':
                            agg_acts = valid_acts.mean(dim=0)
                        elif aggregation_method == 'last':
                            agg_acts = valid_acts[-1]
                        elif aggregation_method == 'random':
                            random_idx = random.randint(0, len(valid_acts) - 1)
                            agg_acts = valid_acts[random_idx]
                        
                        aggregated_activations[global_img_idx] = agg_acts.cpu()
                
                del chunk_acts, reshaped
                gc.collect()
        else:
            # Text dataset with variable-length paragraphs
            token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
            if not os.path.exists(token_counts_file):
                raise FileNotFoundError(f"Token counts file not found: {token_counts_file}")
            
            token_counts = torch.load(token_counts_file, weights_only=False)
            tokens_per_paragraph = [sum(sent_tokens) if isinstance(sent_tokens, list) else sent_tokens
                                  for sent_tokens in token_counts]
            num_paragraphs = len(tokens_per_paragraph)
            
            # Allocate storage
            aggregated_activations = torch.zeros(num_paragraphs, n_clusters, dtype=torch.float32)
            
            # Calculate cumulative token positions
            cumulative_tokens = [0]
            for tokens in tokens_per_paragraph:
                cumulative_tokens.append(cumulative_tokens[-1] + tokens)
            
            # Process paragraphs in chunks
            paragraphs_per_chunk = 100
            for para_start in tqdm(range(0, num_paragraphs, paragraphs_per_chunk), desc="Computing aggregated activations"):
                para_end = min(para_start + paragraphs_per_chunk, num_paragraphs)
                
                # Load tokens for this chunk of paragraphs
                token_start = cumulative_tokens[para_start]
                token_end = cumulative_tokens[para_end]
                
                chunk_acts = loader.load_tensor_range(token_start, token_end)
                if device == 'cuda' and not chunk_acts.is_cuda:
                    chunk_acts = chunk_acts.cuda()
                
                # Process each paragraph
                chunk_offset = 0
                for para_idx in range(para_start, para_end):
                    para_tokens = tokens_per_paragraph[para_idx]
                    para_acts = chunk_acts[chunk_offset:chunk_offset + para_tokens]
                    
                    if para_tokens > 0:
                        # Compute aggregated activation
                        if aggregation_method == 'max':
                            agg_acts = para_acts.max(dim=0)[0]
                        elif aggregation_method == 'mean':
                            agg_acts = para_acts.mean(dim=0)
                        elif aggregation_method == 'last':
                            agg_acts = para_acts[-1]
                        elif aggregation_method == 'random':
                            random_idx = random.randint(0, para_tokens - 1)
                            agg_acts = para_acts[random_idx]
                        
                        aggregated_activations[para_idx] = agg_acts.cpu()
                    
                    chunk_offset += para_tokens
                
                del chunk_acts
                gc.collect()
    else:
        # CLS-level activations are already at image/paragraph level
        aggregated_activations = loader.load_full_tensor()
        if aggregated_activations.numel() * 4 < torch.cuda.get_device_properties(0).total_memory * 0.8:
            aggregated_activations = aggregated_activations.cuda()
    
    # Get cluster labels
    cluster_labels = info['concept_names']
    
    # Process each percentile
    for per in tqdm(percentiles, desc="Computing detection metrics"):
        save_path = f'Quant_Results/{dataset_name}/detectionmetrics_{method_name}_allpairs_per_{per}_{con_label}.csv'
        
        if per not in thresholds:
            print(f"   Warning: No thresholds found for percentile {per}")
            continue
        
        curr_thresholds = thresholds[per]
        
        # Storage for metrics
        fp_count, tp_count, tn_count, fn_count = {}, {}, {}, {}
        
        # Process each concept
        for concept in gt_images_per_concept_split.keys():
            gt_images = set(gt_images_per_concept_split[concept]) & eval_indices_set
            
            # Process each cluster
            for cluster_idx in range(n_clusters):
                cluster_str = cluster_labels[cluster_idx] if cluster_idx < len(cluster_labels) else str(cluster_idx)
                key = (concept, cluster_str)
                
                if key in curr_thresholds:
                    threshold = curr_thresholds[key][0]
                    
                    # Find activated images/paragraphs
                    activated_mask = aggregated_activations[:, cluster_idx] >= threshold
                    
                    # Get indices of activated samples in the evaluation split
                    if aggregated_activations.is_cuda:
                        activated_in_split = activated_mask & eval_mask
                        activated_indices = set(torch.where(activated_in_split)[0].cpu().numpy())
                    else:
                        activated_indices = set()
                        for idx in torch.where(activated_mask)[0].tolist():
                            if idx in eval_indices_set:
                                activated_indices.add(idx)
                    
                    # Compute metrics
                    tp = len(gt_images & activated_indices)
                    fp = len(activated_indices - gt_images)
                    fn = len(gt_images - activated_indices)
                    tn = len(eval_indices_set) - (tp + fp + fn)
                    
                    tp_count[key] = tp
                    fp_count[key] = fp
                    fn_count[key] = fn
                    tn_count[key] = tn
        
        # Compute and save metrics
        metrics = compute_stats_from_counts(tp_count, fp_count, tn_count, fn_count)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        metrics.to_csv(save_path, index=False)
        print(f"   Saved {method_name} all-pairs detection metrics to {save_path}")
    
    # Clean up
    del aggregated_activations
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


def find_best_clusters_per_concept_from_aggregated_detectionmetrics(dataset_name: str,
                                                                   model_name: str,
                                                                   sample_type: str,
                                                                   metric_type: str,
                                                                   percentiles: List[float],
                                                                   con_label: str,
                                                                   aggregation_method: str = 'max') -> Dict:
    """
    For each semantic concept, finds the best cluster + percentile that maximizes the given metric.
    
    This is the aggregated version of find_best_clusters_per_concept_from_detectionmetrics,
    supporting different aggregation methods.
    
    Args:
        dataset_name: Name of dataset
        model_name: Name of model (CLIP, Llama, etc.)
        sample_type: 'patch' or 'cls'
        metric_type: Metric to maximize ('f1', 'tpr', 'fpr')
        percentiles: List of percentiles to search over
        con_label: Concept configuration label
        aggregation_method: Method used for aggregation
        
    Returns:
        best_cluster_per_concept: dict mapping concept -> dict with best_cluster, best_score, best_percentile
    """
    import ast
    import os
    from tqdm import tqdm
    
    # Get method name
    method_name = {
        'max': 'maxtoken',
        'mean': 'meantoken',
        'last': 'lasttoken',
        'random': 'randomtoken'
    }[aggregation_method]
    
    # Where detection metrics are saved
    base_dir = f'Quant_Results/{dataset_name}'
    
    best_cluster_per_concept = {}
    
    for per in tqdm(percentiles, desc=f"Searching {method_name} percentiles"):
        # Load detection metrics at this percentile
        detectionmetrics_path = f"{base_dir}/detectionmetrics_{method_name}_allpairs_per_{per}_{con_label}.csv"
        
        try:
            metrics_df = pd.read_csv(detectionmetrics_path)
        except Exception as e:
            print(f"   Warning: Missing or error loading {detectionmetrics_path}: {e}")
            continue
        
        # Process each row
        for idx, row in metrics_df.iterrows():
            concept, cluster = ast.literal_eval(row['concept'])
            cluster = str(cluster).strip("'\"")
            score = row[metric_type]
            
            # Only update if score is better
            if concept not in best_cluster_per_concept or score > best_cluster_per_concept[concept]['best_score']:
                best_cluster_per_concept[concept] = {
                    'best_cluster': cluster,
                    'best_score': score,
                    'best_percentile': per
                }
    
    # Save results
    save_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{method_name}_{con_label}.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_cluster_per_concept, save_path)
    print(f"   Saved best {method_name} matches to {save_path}")
    
    return best_cluster_per_concept


def filter_and_save_best_clusters_aggregated(dataset_name: str, 
                                            con_label: str,
                                            aggregation_method: str = 'max') -> None:
    """
    Filters detection metrics CSVs to only include best cluster per concept for aggregated methods.
    
    This is the aggregated version of filter_and_save_best_clusters,
    supporting different aggregation methods.
    
    Args:
        dataset_name: Name of dataset
        con_label: Concept group label
        aggregation_method: Method used for aggregation
    """
    import ast
    import os
    import glob
    
    # Get method name
    method_name = {
        'max': 'maxtoken',
        'mean': 'meantoken',
        'last': 'lasttoken',
        'random': 'randomtoken'
    }[aggregation_method]
    
    metrics_dir = f"Quant_Results/{dataset_name}"
    
    # Try to load best clusters info
    try:
        best_clusters_by_detect = torch.load(
            f'Unsupervised_Matches/{dataset_name}/bestdetects_{method_name}_{con_label}.pt',
            weights_only=False
        )
    except FileNotFoundError:
        print(f"   Warning: No best clusters file found for {method_name} {con_label}")
        return
    
    # Pattern for all matching detection metric CSV files
    pattern = os.path.join(metrics_dir, f"detectionmetrics_{method_name}_allpairs_per_*_{con_label}.csv")
    
    metric_files = glob.glob(pattern)
    if not metric_files:
        print(f"   Warning: No detection metric files found matching pattern: {pattern}")
        return
    
    for metric_file in metric_files:
        try:
            # Check if file is empty
            if os.path.getsize(metric_file) == 0:
                print(f"   Warning: Empty file {metric_file}, skipping")
                continue
            
            # Load the detection metrics CSV
            df = pd.read_csv(metric_file)
            
            if df.empty:
                print(f"   Warning: No data in {metric_file}, skipping")
                continue
            
            # Filter to only include best clusters
            filtered_rows = []
            
            for idx, row in df.iterrows():
                # Parse the concept tuple
                concept, cluster = ast.literal_eval(row['concept'])
                
                # Check if this is the best cluster for this concept
                if concept in best_clusters_by_detect:
                    best_info = best_clusters_by_detect[concept]
                    if str(cluster) == str(best_info['best_cluster']):
                        # Update the concept column to just be the concept name
                        row['concept'] = concept
                        filtered_rows.append(row)
            
            # Create filtered dataframe and save
            if filtered_rows:
                filtered_df = pd.DataFrame(filtered_rows)
                # Remove '_allpairs' and add method name
                base_filename = os.path.basename(metric_file)
                base_filename = base_filename.replace(f"_{method_name}_allpairs", "")
                base_filename = base_filename.replace("detectionmetrics_", f"detectfirst_{method_name}_")
                
                save_path = os.path.join(metrics_dir, base_filename)
                filtered_df.to_csv(save_path, index=False)
                print(f"   Saved filtered {method_name} metrics: {base_filename} ({len(filtered_df)} concepts)")
            else:
                print(f"   Warning: No matching data found in {metric_file}")
        
        except pd.errors.EmptyDataError:
            print(f"   Error: Empty CSV file {metric_file}")
            continue
        except Exception as e:
            print(f"   Error processing {metric_file}: {e}")
            continue


def save_best_percentiles_for_kmeans(dataset_name: str, con_label: str, aggregation_method: str = 'max') -> None:
    """
    Extract and save best percentiles for kmeans concepts from filtered detection results.
    This is needed for per_concept_ptm_optimization.py compatibility.
    
    Args:
        dataset_name: Name of dataset
        con_label: Concept configuration label
        aggregation_method: Method used for aggregation
    """
    import ast
    
    method_name = {
        'max': 'maxtoken',
        'mean': 'meantoken',
        'last': 'lasttoken',
        'random': 'randomtoken'
    }[aggregation_method]
    
    # Get percentiles from the first available file
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    # Read the filtered calibration results
    cal_results_file = f"Quant_Results/{dataset_name}/detectfirst_{method_name}_per_0.02_{con_label}_cal.csv"
    
    try:
        if not os.path.exists(cal_results_file):
            # Try other percentiles if 0.02 doesn't exist
            for perc in [0.05, 0.1, 0.2]:
                cal_results_file = f"Quant_Results/{dataset_name}/detectfirst_{method_name}_per_{perc}_{con_label}_cal.csv"
                if os.path.exists(cal_results_file):
                    break
            else:
                print(f"   Warning: No filtered calibration results found for {method_name} {con_label}")
                return
        
        # Find best percentile for each concept based on F1
        best_percentiles = {}
        best_f1_scores = {}
        
        # For kmeans, we need to read all percentile files and find best F1 for each concept
        for perc in percentiles:
            try:
                perc_file = f"Quant_Results/{dataset_name}/detectfirst_{method_name}_per_{perc}_{con_label}_cal.csv"
                if os.path.exists(perc_file):
                    perc_df = pd.read_csv(perc_file)
                    for _, row in perc_df.iterrows():
                        concept = row['concept']
                        f1 = row['f1']
                        
                        if concept not in best_f1_scores or f1 > best_f1_scores[concept]:
                            best_percentiles[concept] = perc
                            best_f1_scores[concept] = f1
            except Exception as e:
                continue
        
        if best_percentiles:
            # Save best percentiles in the same format as supervised concepts
            best_percentiles_data = {
                'best_percentiles': best_percentiles,
                'best_f1_scores': best_f1_scores,
                'percentiles_evaluated': percentiles,
                'sample_type': 'patch'
            }
            
            save_path = f"Quant_Results/{dataset_name}/{method_name}_best_percentiles_{con_label}.pt"
            torch.save(best_percentiles_data, save_path)
            print(f"   Saved best {method_name} percentiles to {save_path}")
            
    except Exception as e:
        print(f"   Error saving best percentiles for {method_name} {con_label}: {e}")