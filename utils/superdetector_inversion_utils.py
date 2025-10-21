"""
Superdetector inversion utilities that work only with tensors and loaders.
No DataFrame support, no duplicate functions.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc
import os
import math
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Union, Optional, Tuple

from utils.memory_management_utils import (
    ChunkedEmbeddingLoader, ChunkedActivationLoader, MatchedConceptActivationLoader,
    convert_image_indices_to_patch_indices, map_global_to_split_local,
    convert_text_sentence_to_token_indices
)
from utils.quant_concept_evals_utils import get_patch_split_df, filter_patches_by_image_presence
# from utils.patch_mapping_helpers import get_patch_indices_for_concepts_and_images
from utils.patch_alignment_utils import (
    get_patch_range_for_image, get_patch_range_for_text, 
    compute_patches_per_image, filter_patches_by_image_presence as filter_patches_utils
)
from utils.quant_concept_evals_utils import detect_then_invert_metrics_over_percentiles as loader_compatible_fn
from utils.quant_concept_evals_utils import create_binary_labels, compute_stats_from_counts


def find_all_superdetector_patches(percentile: float, 
                                 act_loader: Union[ChunkedActivationLoader, MatchedConceptActivationLoader],
                                 concept_names: List[str],
                                 gt_samples_per_concept_test: Dict,
                                 dataset_name: str,
                                 model_input_size: Tuple,
                                 con_label: str,
                                 device: str):
    """
    Find superdetector patches for all concepts using a loader.
    OPTIMIZED: Batch processes concepts to minimize data loading.
    
    Args:
        percentile: Percentile threshold for superdetector selection
        act_loader: Activation loader (ChunkedActivationLoader or MatchedConceptActivationLoader)
        concept_names: List of concept names corresponding to tensor columns
        gt_samples_per_concept_test: Ground truth test samples per concept
        dataset_name: Dataset name
        model_input_size: Model input size
        con_label: Concept label
        device: Device for computation
        
    Returns:
        Dict mapping concept -> list of superdetector patch indices
    """
    all_superdetectors = {}
    concept_to_idx = {name: i for i, name in enumerate(concept_names)}
    
    # Load ground truth patches once
    gt_patch_file = f'GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt'
    if os.path.exists(gt_patch_file):
        gt_patches = torch.load(gt_patch_file, weights_only=False)
    else:
        gt_patches = None
    
    # OPTIMIZATION: Batch process concepts with similar patch indices
    concepts_to_process = []
    concept_patch_map = {}
    
    # First pass: collect all patch indices needed
    all_patch_indices_set = set()
    for concept in gt_samples_per_concept_test.keys():
        if concept not in concept_to_idx:
            all_superdetectors[concept] = []
            continue
            
        test_image_indices = gt_samples_per_concept_test[concept]
        if not test_image_indices:
            all_superdetectors[concept] = []
            continue
        
        # Get ground truth patch indices for this concept
        assert gt_patches and concept in gt_patches, f"Missing ground truth patches for concept: {concept}"
        concept_patch_indices = list(gt_patches.get(concept, []))
        
        if not concept_patch_indices:
            all_superdetectors[concept] = []
            continue
        
        concepts_to_process.append(concept)
        concept_patch_map[concept] = concept_patch_indices
        all_patch_indices_set.update(concept_patch_indices)
    
    if not concepts_to_process:
        return all_superdetectors
    
    # OPTIMIZATION: Load all needed patches at once
    all_patch_indices = sorted(list(all_patch_indices_set))
    min_idx = all_patch_indices[0]
    max_idx = all_patch_indices[-1] + 1
    
    print(f"Loading activations for {len(concepts_to_process)} concepts, {len(all_patch_indices)} unique patches...")
    
    # Batch load activations for all concepts
    if isinstance(act_loader, MatchedConceptActivationLoader):
        # Load tensor range for all matched concepts
        range_tensor = act_loader.load_tensor_range(min_idx, max_idx)
        
        # Get concept to column index mapping
        concept_to_col_idx = {concept: i for i, concept in enumerate(act_loader.concept_names)}
        
        # Process each concept
        for concept in concepts_to_process:
            if concept not in concept_to_col_idx:
                all_superdetectors[concept] = []
                continue
                
            concept_col_idx = concept_to_col_idx[concept]
            concept_patch_indices = concept_patch_map[concept]
            relative_indices = [idx - min_idx for idx in concept_patch_indices]
            
            # Extract activations for this concept's patches
            concept_activations = range_tensor[relative_indices, concept_col_idx].to(device)
            
            # Find top percentile patches
            threshold = torch.quantile(concept_activations, 1 - percentile)
            superdetector_mask = concept_activations >= threshold
            superdetector_indices = [concept_patch_indices[i] for i, is_super in enumerate(superdetector_mask) if is_super]
            
            all_superdetectors[concept] = superdetector_indices
    else:
        # For ChunkedActivationLoader, batch load the tensor range
        range_tensor = act_loader.load_tensor_range(min_idx, max_idx)
        
        for concept in concepts_to_process:
            concept_idx = concept_to_idx[concept]
            concept_patch_indices = concept_patch_map[concept]
            relative_indices = [idx - min_idx for idx in concept_patch_indices]
            
            # Extract activations for this concept
            concept_activations = range_tensor[relative_indices, concept_idx].to(device)
            
            # Find top percentile patches
            threshold = torch.quantile(concept_activations, 1 - percentile)
            superdetector_mask = concept_activations >= threshold
            superdetector_indices = [concept_patch_indices[i] for i, is_super in enumerate(superdetector_mask) if is_super]
            
            all_superdetectors[concept] = superdetector_indices
    
    # Handle concepts that weren't processed
    for concept in gt_samples_per_concept_test.keys():
        if concept not in all_superdetectors:
            all_superdetectors[concept] = []
    
    return all_superdetectors


def get_superdetector_vector(superdetector_indices: List[int],
                           embedding_loader: ChunkedEmbeddingLoader,
                           concept_idx: int,
                           act_loader: Union[ChunkedActivationLoader, MatchedConceptActivationLoader],
                           concept_name: str,
                           agglomerate_type: str = 'avg',
                           device: str = 'cuda') -> torch.Tensor:
    """
    Compute superdetector vector from embeddings.
    
    Args:
        superdetector_indices: Indices of superdetector patches
        embedding_loader: Chunked embedding loader
        concept_idx: Column index of the concept in activation tensor
        act_loader: Activation loader for getting activation weights
        concept_name: Name of the concept (for MatchedConceptActivationLoader)
        agglomerate_type: 'avg' or 'max' aggregation
        device: Device for computation
        
    Returns:
        Superdetector vector
    """
    if not superdetector_indices:
        # Return zero vector if no superdetectors
        emb_dim = embedding_loader.embedding_dim
        return torch.zeros(emb_dim, device=device)
    
    # Load embeddings for superdetector patches
    super_embeds = embedding_loader.load_specific_embeddings(superdetector_indices)
    
    # Load activations for superdetector patches using unified method
    super_acts = act_loader.load_concept_activations_for_indices(concept_name, superdetector_indices, device)
    
    if agglomerate_type == 'avg':
        # Weighted average by activation strength
        weights = torch.softmax(super_acts, dim=0)
        superdetector_vector = torch.sum(super_embeds * weights.unsqueeze(-1), dim=0)
    elif agglomerate_type == 'max':
        # Take embedding with highest activation
        max_idx = torch.argmax(super_acts)
        superdetector_vector = super_embeds[max_idx]
    else:
        raise ValueError(f"Unknown agglomerate type: {agglomerate_type}")
    
    return superdetector_vector


def batch_superdetector_inversions(
    percentiles: List[float],  # Takes multiple percentiles but computes once
    agglomerate_type: str,
    embedding_loader: ChunkedEmbeddingLoader,
    act_loader: Union[ChunkedActivationLoader, MatchedConceptActivationLoader],
    concept_names: List[str],
    gt_samples_per_concept_test: Dict,
    dataset_name: str,
    model_input_size: Tuple,
    con_label: str,
    device: str,
    patch_size: int = 14,
    local: bool = False,
    split: str = 'cal',
    batch_size: int = 10,  # Smaller batch size for memory efficiency
    scratch_dir: str = ''
):
    """
    Compute superdetector inversions ONCE and saves the same result for all percentiles. 
    The percentiles are used later during evaluation to determine similarity thresholds, 
    NOT to change the inversion computation.
    
    Key insight: The inversion is just cosine similarity between patches and the
    superdetector vector. This doesn't change with percentile.
    
    Args:
        percentiles: List of percentiles (used for evaluation, not computation)
        agglomerate_type: Aggregation method ('avg' or 'max')
        embedding_loader: Chunked embedding loader
        act_loader: Activation loader
        concept_names: List of concept names
        gt_samples_per_concept_test: Ground truth test samples
        dataset_name: Dataset name
        model_input_size: Model input size
        con_label: Concept label
        device: Device for computation
        patch_size: Patch size
        local: If True, compute per-image superdetectors
        split: Data split to process
        batch_size: Batch size for processing
        scratch_dir: Scratch directory for saving chunked files
    """
    
    if not local:
        # For global superdetectors, fall back to original implementation
        for percentile in percentiles:
            batch_superdetector_inversions(
                [percentile], agglomerate_type, embedding_loader, act_loader,
                concept_names, gt_samples_per_concept_test, dataset_name,
                model_input_size, con_label, device, patch_size, local, split
            )
        return
    
    print(f"\nOptimized computation for {len(percentiles)} percentiles on {split} split")
    
    # Get detection thresholds
    best_detection_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
    if not os.path.exists(best_detection_file):
        raise FileNotFoundError(f"Best detection percentiles required: {best_detection_file}")
    best_detection_percentiles = torch.load(best_detection_file, weights_only=False)
    
    detection_thresholds = {}
    if 'kmeans' not in con_label and 'sae' not in con_label:
        all_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt', weights_only=False)
        for concept, info in best_detection_percentiles.items():
            if concept in concept_names:
                best_perc = info['best_percentile']
                detection_thresholds[concept] = all_thresholds[best_perc][concept][0] if isinstance(all_thresholds[best_perc][concept], tuple) else all_thresholds[best_perc][concept]
    else:
        # Handle kmeans and SAE thresholds (unsupervised methods)
        raw_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt', weights_only=False)
        alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)
        
        for concept, info in best_detection_percentiles.items():
            if concept in concept_names:
                best_perc = info['best_percentile']
                cluster_id = alignment_results[concept]['best_cluster']
                key = (concept, cluster_id)
                if best_perc in raw_thresholds and key in raw_thresholds[best_perc]:
                    detection_thresholds[concept] = raw_thresholds[best_perc][key][0] if isinstance(raw_thresholds[best_perc][key], tuple) else raw_thresholds[best_perc][key]
    
    # Setup
    loader_info = act_loader.get_activation_info() if hasattr(act_loader, 'get_activation_info') else act_loader.get_info()
    total_patches = loader_info['total_samples']
    n_concepts = len(concept_names)
    concept_to_idx = {name: i for i, name in enumerate(concept_names)}
    
    # Create temporary file for incremental saving to avoid memory issues
    save_dir = os.path.join(scratch_dir, 'Superpatches', dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    temp_path = os.path.join(save_dir, f'temp_superpatch_{agglomerate_type}_inv_{con_label}.npy')
    
    # Use memory-mapped array to avoid loading everything into memory
    import numpy as np
    inversions_mmap = np.memmap(temp_path, dtype='float32', mode='w+', shape=(total_patches, n_concepts))
    
    # Get samples to process - ONLY cal and test splits (skip train to save compute)
    metadata_df = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    # Get cal and test samples
    cal_samples = metadata_df[metadata_df['split'] == 'cal'].index.tolist()
    test_samples = metadata_df[metadata_df['split'] == 'test'].index.tolist()
    samples_to_process = cal_samples + test_samples
    n_samples = len(samples_to_process)
    
    # Handle text vs image datasets
    if model_input_size[0] == 'text':
        # For text datasets, we don't have patches per image
        # Instead, we have variable number of tokens per sentence
        patches_per_image = None  # We'll handle this differently for text
        is_text_dataset = True
    else:
        patches_per_image = compute_patches_per_image(patch_size, model_input_size)
        is_text_dataset = False
    
    if is_text_dataset:
        print(f"Processing {len(cal_samples)} cal + {len(test_samples)} test = {n_samples} sentences (skipping train)")
    else:
        print(f"Processing {len(cal_samples)} cal + {len(test_samples)} test = {n_samples} images (skipping train)")
    
    # IMPORTANT: Initialize ALL patches to -1 first, then we'll only compute for cal/test
    # This ensures train patches stay at -1 and maintain correct global indices
    inversions_mmap[:, :] = -1.0
    
    # Track which patches we've processed to ensure we don't miss any
    processed_patches = set()
    
    if is_text_dataset:
        print(f"Processing {n_samples} sentences in batches of {batch_size}...")
    else:
        print(f"Processing {n_samples} images in batches of {batch_size}...")
    
    # OPTIMIZATION: Pre-compute all superdetector vectors first
    print("Computing superdetector vectors for all concepts...")
    superdetector_vectors = {}
    
    # Find all superdetector patches for each concept
    for concept in tqdm(concept_names, desc="Finding superdetectors"):
        if concept not in detection_thresholds:
            continue
            
        concept_idx = concept_to_idx[concept]
        threshold = detection_thresholds[concept]
        
        # Use unified method to find indices above threshold
        super_indices = act_loader.find_indices_above_threshold(concept, threshold)
        
        
        if len(super_indices) > 0:
            # Load embeddings for superdetector patches
            super_embeds = embedding_loader.load_specific_embeddings(super_indices).to(device)
            
            # Get activations for weighting using unified method
            super_acts = act_loader.load_concept_activations_for_indices(concept, super_indices, device)
            
            if agglomerate_type == 'avg':
                # Weighted average by activation strength
                weights = torch.softmax(super_acts, dim=0)
                superdetector_vec = torch.sum(super_embeds * weights.unsqueeze(-1), dim=0)
            elif agglomerate_type == 'max':
                # Take embedding with highest activation
                max_idx = torch.argmax(super_acts)
                superdetector_vec = super_embeds[max_idx]
            else:
                raise ValueError(f"Unknown agglomerate type: {agglomerate_type}")
            
            superdetector_vectors[concept_idx] = F.normalize(superdetector_vec.unsqueeze(0), dim=1)
            
            # Clean up memory after each concept
            del super_embeds, super_acts
            if 'weights' in locals():
                del weights
            torch.cuda.empty_cache()
        else:
            print(f"Warning: No superdetector patches found for {concept}")
    
    # Process in batches for memory efficiency
    for batch_start in tqdm(range(0, n_samples, batch_size), desc="Computing inversions"):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_samples = samples_to_process[batch_start:batch_end]
        
        # Pre-compute all patch ranges and indices for the batch
        batch_patch_data = []
        all_global_patches = []
        
        if is_text_dataset:
            # For text datasets, we need to load token counts to get variable-length sequences
            import glob
            token_files = glob.glob(f'GT_Samples/{dataset_name}/token_counts_inputsize_*.pt')
            if not token_files:
                token_files = glob.glob(f'GT_Samples/{dataset_name}/token_counts.pt')
            if not token_files:
                raise FileNotFoundError(f"No token counts file found for {dataset_name}")
            
            # Determine correct token file based on model - MUST match exactly
            if model_input_size and model_input_size[0] == 'text':
                token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
                if not os.path.exists(token_counts_file):
                    raise FileNotFoundError(f"Required token counts file not found: {token_counts_file}. "
                                          f"This MUST match the model input size {model_input_size}")
            else:
                raise ValueError(f"Cannot determine correct token counts file for non-text input size: {model_input_size}")
            
            token_counts_per_sentence = torch.load(token_counts_file, weights_only=False)
            
            # Compute cumulative token positions
            token_counts_flat = torch.tensor([sum(x) if isinstance(x, list) else x for x in token_counts_per_sentence])
            sentence_starts = torch.cat([torch.tensor([0]), token_counts_flat.cumsum(0)[:-1]])
            
            for global_sent_idx in batch_samples:
                global_start = sentence_starts[global_sent_idx].item()
                if global_sent_idx + 1 < len(sentence_starts):
                    global_end = sentence_starts[global_sent_idx + 1].item()
                else:
                    global_end = sentence_starts[global_sent_idx].item() + token_counts_flat[global_sent_idx].item()
                global_patches = list(range(global_start, global_end))
                
                batch_patch_data.append({
                    'global_patches': global_patches,
                    'n_patches': len(global_patches)
                })
        else:
            # For image datasets, use fixed patches per image
            for global_img_idx in batch_samples:
                global_start = global_img_idx * patches_per_image
                global_end = (global_img_idx + 1) * patches_per_image
                global_patches = list(range(global_start, global_end))
                
                # Since we're processing multiple splits (cal + test), just use global indices
                # No need to map to split-local indices
                batch_patch_data.append({
                    'global_patches': global_patches,
                    'n_patches': len(global_patches)
                })
        
        # Collect all global patches for this batch
        for img_data in batch_patch_data:
            all_global_patches.extend(img_data['global_patches'])
        
        if not all_global_patches:
            continue
        
        # Load activations for entire batch at once using GLOBAL indices
        min_global = min(all_global_patches)
        max_global = max(all_global_patches) + 1
        
        
        if isinstance(act_loader, MatchedConceptActivationLoader):
            # Load tensor range for all concepts
            # First load the underlying tensor data
            underlying_loader = act_loader.activation_loader
            # Load to CPU first to avoid GPU OOM
            batch_acts_tensor = underlying_loader.load_tensor_range(min_global, max_global)
            
            # Create dictionary mapping concept names to their activations
            batch_acts_dict = {}
            for concept_name in concept_names:
                # Get cluster ID for this concept
                cluster_id = act_loader.concept_to_cluster.get(concept_name)
                if cluster_id and cluster_id in underlying_loader.columns:
                    # Get column index for this cluster
                    col_idx = underlying_loader.columns.index(cluster_id)
                    # Only move the specific column to GPU when needed
                    batch_acts_dict[concept_name] = batch_acts_tensor[:, col_idx].to(device)
        else:
            # Load tensor range
            batch_acts_tensor = act_loader.load_tensor_range(min_global, max_global)
            batch_acts_tensor = batch_acts_tensor.to(device)
        
        # Process each image individually to minimize memory usage
        for img_data in batch_patch_data:
            global_patches = img_data['global_patches']
            n_patches = img_data['n_patches']
            
            if n_patches == 0:
                continue
            
            # Load embeddings only for this image's patches
            # For SAE, process on CPU if many patches to avoid GPU OOM
            if 'sae' in con_label and n_patches > 100:
                img_embeds = embedding_loader.load_specific_embeddings(global_patches).to('cpu')
                img_embeds_norm = F.normalize(img_embeds, dim=1)
                compute_device = 'cpu'
            else:
                img_embeds = embedding_loader.load_specific_embeddings(global_patches).to(device)
                img_embeds_norm = F.normalize(img_embeds, dim=1)
                compute_device = device
            
            # Compute similarities with all superdetector vectors at once
            if len(superdetector_vectors) > 0:
                # Stack all superdetector vectors
                super_vecs = torch.cat([vec.to(compute_device) for vec in superdetector_vectors.values()], dim=0)
                concept_indices = list(superdetector_vectors.keys())
                
                # Compute all similarities at once [n_patches x n_concepts]
                similarities = torch.matmul(img_embeds_norm, super_vecs.t())
                
                # Store results directly to memory-mapped array
                # Convert to numpy on GPU then transfer
                similarities_np = similarities.cpu().numpy()
                for i, c_idx in enumerate(concept_indices):
                    inversions_mmap[global_patches, c_idx] = similarities_np[:, i]
                
                # Track that we processed these patches
                processed_patches.update(global_patches)
                
                # Clean up this image's embeddings immediately
                del img_embeds, img_embeds_norm, similarities
                torch.cuda.empty_cache()
        
        # Clean up batch memory
        if 'batch_acts_dict' in locals():
            del batch_acts_dict
        if 'batch_acts_tensor' in locals():
            del batch_acts_tensor
        # Also clean up the superdetector vectors if they exist
        if 'super_vecs' in locals():
            del super_vecs
        torch.cuda.empty_cache()
    
    # Verify we processed all cal/test patches
    expected_patches = set()
    if is_text_dataset:
        # For text, use variable token lengths
        # MUST use correct token counts file for the model
        if model_input_size and model_input_size[0] == 'text':
            token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
            if not os.path.exists(token_counts_file):
                raise FileNotFoundError(f"Required token counts file not found: {token_counts_file}")
        else:
            raise ValueError(f"Cannot determine correct token counts file for non-text input size: {model_input_size}")
        
        token_counts_per_sentence = torch.load(token_counts_file, weights_only=False)
        token_counts_flat = torch.tensor([sum(x) if isinstance(x, list) else x for x in token_counts_per_sentence])
        sentence_starts = torch.cat([torch.tensor([0]), token_counts_flat.cumsum(0)[:-1]])
        
        for sample_idx in samples_to_process:
                start_patch = sentence_starts[sample_idx].item()
                if sample_idx + 1 < len(sentence_starts):
                    end_patch = sentence_starts[sample_idx + 1].item()
                else:
                    end_patch = sentence_starts[sample_idx].item() + token_counts_flat[sample_idx].item()
                expected_patches.update(range(start_patch, end_patch))
    else:
        # For images, use fixed patches per image
        for sample_idx in samples_to_process:
            start_patch = sample_idx * patches_per_image
            end_patch = (sample_idx + 1) * patches_per_image
            expected_patches.update(range(start_patch, end_patch))
    
    missing_patches = expected_patches - processed_patches
    if missing_patches:
        print(f"WARNING: {len(missing_patches)} patches were not processed!")
        print(f"First few missing: {sorted(list(missing_patches))[:10]}")
    else:
        print(f"✓ All {len(expected_patches)} cal/test patches processed successfully")
    
    # Verify train patches are still -1
    train_samples = metadata_df[metadata_df['split'] == 'train'].index.tolist()
    if train_samples:
        # Check a few train patches to verify they're still -1
        sample_train_idx = train_samples[0]
        if is_text_dataset:
            # For text, get the first token of the first train sentence
            if 'token_counts_per_sentence' in locals():
                token_counts_flat = torch.tensor([sum(x) if isinstance(x, list) else x for x in token_counts_per_sentence])
                sentence_starts = torch.cat([torch.tensor([0]), token_counts_flat.cumsum(0)[:-1]])
                sample_patch_idx = sentence_starts[sample_train_idx].item()
            else:
                sample_patch_idx = 0  # Fallback
        else:
            sample_patch_idx = sample_train_idx * patches_per_image
        
        if inversions_mmap[sample_patch_idx, 0] != -1.0:
            print("WARNING: Train patches were not properly skipped!")
        else:
            print(f"✓ Train patches correctly marked as -1 (checked sample patch {sample_patch_idx})")
    
    # Flush memory-mapped array to disk
    del inversions_mmap
    
    # Save in chunked format compatible with ChunkedActivationLoader
    print("\nSaving inversions in chunked format...")
    
    # Calculate chunk sizes (similar to activation_utils.py)
    chunk_size_gb = 10.0  # Target size for each chunk in GB
    bytes_per_value = 4  # float32
    bytes_per_gb = 1024 * 1024 * 1024
    values_per_row = n_concepts
    bytes_per_row = values_per_row * bytes_per_value
    rows_per_chunk = int((chunk_size_gb * bytes_per_gb) / bytes_per_row)
    rows_per_chunk = max(1, rows_per_chunk)  # Ensure at least one row per chunk
    
    # Calculate number of chunks
    num_chunks = int(np.ceil(total_patches / rows_per_chunk))
    
    print(f"  Total patches: {total_patches:,}")
    print(f"  Concepts: {n_concepts}")
    print(f"  Rows per chunk: {rows_per_chunk:,}")
    print(f"  Number of chunks: {num_chunks}")
    
    # Base filename for chunks
    base_filename = f'superpatch_{agglomerate_type}_inv_{con_label}'
    
    # Save each chunk
    chunk_info = {
        'num_chunks': num_chunks,
        'total_samples': total_patches,
        'concept_names': concept_names,
        'num_concepts': n_concepts,
        'chunks': [],
        'metadata': {
            'train_patches_value': -1.0,
            'train_patches_skipped': True,
            'splits_processed': ['cal', 'test'],
            'note': 'Train patches are set to -1.0 to save computation'
        }
    }
    
    # Read from memory-mapped array and save in chunks
    mmap_read = np.memmap(temp_path, dtype='float32', mode='r', shape=(total_patches, n_concepts))
    
    for chunk_idx in tqdm(range(num_chunks), desc="Saving chunks"):
        start_idx = chunk_idx * rows_per_chunk
        end_idx = min((chunk_idx + 1) * rows_per_chunk, total_patches)
        chunk_samples = end_idx - start_idx
        
        # Read chunk from memory-mapped array
        chunk_data = mmap_read[start_idx:end_idx].copy()
        chunk_tensor = torch.from_numpy(chunk_data)
        
        # Save chunk with metadata
        chunk_filename = f'{base_filename}_chunk_{chunk_idx}.pt'
        chunk_path = os.path.join(save_dir, chunk_filename)
        
        torch.save({
            'activations': chunk_tensor,
            'concept_names': concept_names,
            'start_idx': start_idx,
            'end_idx': end_idx
        }, chunk_path)
        
        # Add to chunk info
        chunk_info['chunks'].append({
            'file': chunk_filename,
            'chunk_idx': chunk_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'samples': chunk_samples
        })
        
        # Clear memory
        del chunk_data, chunk_tensor
        gc.collect()
    
    # Save chunk info JSON
    info_filename = f'{base_filename}_chunks_info.json'
    info_path = os.path.join(save_dir, info_filename)
    
    with open(info_path, 'w') as f:
        json.dump(chunk_info, f, indent=2)
    
    print(f"\n✅ Saved {num_chunks} chunks to {save_dir}")
    print(f"   Chunk info: {info_filename}")
    
    # Clean up temp file
    del mmap_read
    os.remove(temp_path)



def all_superdetector_inversions_across_percentiles(percentiles: List[float],
                                                  agglomerate_type: str,
                                                  embedding_loader: ChunkedEmbeddingLoader,
                                                  act_loader: Union[ChunkedActivationLoader, MatchedConceptActivationLoader],
                                                  concept_names: List[str],
                                                  gt_samples_per_concept_test: Dict,
                                                  dataset_name: str,
                                                  model_input_size: Tuple,
                                                  con_label: str,
                                                  device: str,
                                                  patch_size: int = 14,
                                                  local: bool = False,
                                                  split: str = 'cal',
                                                  scratch_dir: str = ''):
    """
    Compute superdetector inversions across multiple percentiles.
    Now uses optimized batch processing.
    """
    # Process all percentiles at once
    # Adjust batch size based on concept type - kmeans with many clusters needs smaller batches
    if 'sae' in con_label:
        batch_size = 5  # Very small batch for SAE due to large number of patches
    elif 'kmeans_1000' in con_label:
        batch_size = 100  # Increased from 10 for faster processing
    elif 'kmeans' in con_label:
        batch_size = 20  # Moderate batch for other kmeans
    else:
        batch_size = 50  # Larger batch for avg/linsep with fewer concepts
    
    # Debug info about embedding dimensions
    embed_info = embedding_loader.get_embedding_info()
    print(f"\nEmbedding info: {embed_info['embedding_dim']} dimensions, {embed_info['total_samples']} samples")
    
    batch_superdetector_inversions(
        percentiles=percentiles,
        agglomerate_type=agglomerate_type,
        embedding_loader=embedding_loader,
        act_loader=act_loader,
        concept_names=concept_names,
        gt_samples_per_concept_test=gt_samples_per_concept_test,
        dataset_name=dataset_name,
        model_input_size=model_input_size,
        con_label=con_label,
        device=device,
        patch_size=patch_size,
        local=local,
        split=split,
        batch_size=batch_size,  # Dynamic based on concept type
        scratch_dir=scratch_dir
    )



def detect_then_invert_superdetector_twostage_metrics(
    invert_percentiles: List[float],
    act_loader: Union[ChunkedActivationLoader, MatchedConceptActivationLoader],
    concepts: Dict,
    gt_patches_per_concept: Dict,
    gt_patches_per_concept_test: Dict,
    embedding_loader: ChunkedEmbeddingLoader,
    device: str,
    dataset_name: str,
    model_input_size: Tuple,
    con_label: str,
    all_object_patches: Optional[set] = None,
    patch_size: int = 14,
    agglomerate_type: str = 'avg',
    split: str = 'cal',
    scratch_dir: str = ''
):
    """
    Two-stage superdetector method:
    Stage 1: Only images with superdetector patches can be "detected"
    Stage 2: Among detected images, evaluate all patches using superdetector similarities
    
    Key: Non-detected images contribute all their patches as "negative" to final F1
    """
    from utils.patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence
    from utils.patch_alignment_utils import compute_patches_per_image, get_patch_range_for_image
    from utils.quant_concept_evals_utils import create_binary_labels, compute_stats_from_counts
    from utils.gt_concept_segmentation_utils import remap_text_ground_truth_indices
    from tqdm import tqdm
    import pandas as pd
    
    print(f"\n=== Two-Stage Superdetector Evaluation on {split} set ===")
    
    # GLOBAL INDICES APPROACH: Use the same approach as fixed regular method
    from utils.general_utils import get_split_df
    
    # Get split indices  
    if model_input_size[0] == 'text':
        # For text: get sentence-level split, then map to all token indices
        split_df = get_split_df(dataset_name)
        token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
        if not os.path.exists(token_counts_file):
            raise FileNotFoundError(f"Required token counts file not found: {token_counts_file}")
        token_counts_per_sentence = torch.load(token_counts_file, weights_only=False)
        
        # Use ALL token indices - no split filtering here (same as fixed regular method)
        loader_info = act_loader.get_activation_info() if hasattr(act_loader, 'get_activation_info') else act_loader.get_info()
        total_tokens = loader_info['total_samples']
        relevant_indices = torch.arange(total_tokens)
    else:
        # For images: use patch-based split filtering
        split_df = get_patch_split_df(dataset_name, model_input_size=model_input_size, patch_size=patch_size)
        split_indices = torch.tensor(split_df.index[split_df == split].tolist())
        relevant_indices = filter_patches_by_image_presence(split_indices, dataset_name, model_input_size)
    
    # Get ground truth labels for this split
    loader_info = act_loader.get_activation_info() if hasattr(act_loader, 'get_activation_info') else act_loader.get_info()
    total_samples = loader_info['total_samples']
    
    # GLOBAL INDICES: Keep ground truth in global indices (same as fixed regular method)
    if model_input_size[0] == 'text':
        # Filter ground truth to only include tokens from sentences in this split
        split_sentence_indices = [i for i in range(len(token_counts_per_sentence)) if split_df.get(i) == split]
        
        # Map to token ranges for this split 
        split_token_ranges = []
        current_token = 0
        for sent_idx in range(len(token_counts_per_sentence)):
            num_tokens = sum(token_counts_per_sentence[sent_idx]) if isinstance(token_counts_per_sentence[sent_idx], list) else token_counts_per_sentence[sent_idx]
            if sent_idx in split_sentence_indices:
                split_token_ranges.extend(range(current_token, current_token + num_tokens))
            current_token += num_tokens
        
        # Filter ground truth to tokens in this split, but keep global indices
        filtered_gt = {}
        for concept, indices in gt_patches_per_concept_test.items():
            # Keep only tokens that are in the target split, but preserve global indices
            filtered_gt[concept] = [idx for idx in indices if idx in split_token_ranges]
        
        all_concept_labels = create_binary_labels(total_samples, filtered_gt)
    else:
        # Filter ground truth to split indices (for images, keep existing logic)
        split_gt = {}
        for concept, indices in gt_patches_per_concept_test.items():
            split_indices_set = set(relevant_indices.tolist())
            split_gt[concept] = [idx for idx in indices if idx in split_indices_set]
        all_concept_labels = create_binary_labels(total_samples, split_gt)
    
    # Load best detection percentiles and thresholds
    best_detection_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
    if not os.path.exists(best_detection_file):
        raise FileNotFoundError(f"Best detection percentiles required: {best_detection_file}")
    best_detection_percentiles = torch.load(best_detection_file, weights_only=False)
    
    # Load detection thresholds
    detection_thresholds = {}
    if 'kmeans' not in con_label and 'sae' not in con_label:
        all_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt', weights_only=False)
        for concept, info in best_detection_percentiles.items():
            if concept in concepts:
                best_perc = info['best_percentile']
                detection_thresholds[concept] = all_thresholds[best_perc][concept][0] if isinstance(all_thresholds[best_perc][concept], tuple) else all_thresholds[best_perc][concept]
    else:
        # Handle kmeans and SAE thresholds (unsupervised methods)
        raw_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt', weights_only=False)
        alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)
        
        for concept, info in best_detection_percentiles.items():
            if concept in concepts:
                best_perc = info['best_percentile']
                cluster_id = alignment_results[concept]['best_cluster']
                key = (concept, cluster_id)
                if best_perc in raw_thresholds and key in raw_thresholds[best_perc]:
                    detection_thresholds[concept] = raw_thresholds[best_perc][key][0] if isinstance(raw_thresholds[best_perc][key], tuple) else raw_thresholds[best_perc][key]
    
    # Load superdetector inversion similarities using ChunkedActivationLoader
    inversion_file = f'superpatch_{agglomerate_type}_inv_{con_label}_chunks_info.json'
    inversion_path = os.path.join(scratch_dir, 'Superpatches', dataset_name, inversion_file)
    
    # Check if chunked version exists, otherwise fall back to old format
    if os.path.exists(inversion_path):
        print("Loading chunked superdetector inversions...")
        # Use ChunkedActivationLoader to load the inversions
        inversion_loader = ChunkedActivationLoader(
            dataset_name=dataset_name,
            acts_file=f'superpatch_{agglomerate_type}_inv_{con_label}.pt',
            scratch_dir=scratch_dir,  # Base scratch dir
            device='cpu'  # Load to CPU for memory efficiency
        )
        inversions_shape = (inversion_loader.total_samples, len(inversion_loader.columns))
    else:
        # Fallback to old single-file format (check both scratch and local)
        old_inversion_file_scratch = os.path.join(scratch_dir, 'Superpatches', dataset_name, f'superpatch_{agglomerate_type}_inv_{con_label}.pt')
        old_inversion_file_local = f'Superpatches/{dataset_name}/superpatch_{agglomerate_type}_inv_{con_label}.pt'
        
        if os.path.exists(old_inversion_file_scratch):
            old_inversion_file = old_inversion_file_scratch
        elif os.path.exists(old_inversion_file_local):
            old_inversion_file = old_inversion_file_local
        else:
            raise FileNotFoundError(f"Superdetector inversions not found in either chunked or single-file format")
        
        print("Loading single-file superdetector inversions (legacy format)...")
        # Create temporary chunked files for compatibility
        temp_inversions = torch.load(old_inversion_file, weights_only=False, map_location='cpu')
        inversions_shape = temp_inversions.shape
        
        # Create memory-mapped array for legacy format
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp:
            mmap_path = tmp.name
        
        np.save(mmap_path, temp_inversions.numpy())
        del temp_inversions
        gc.collect()
        
        inversions_mmap = np.memmap(mmap_path, dtype='float32', mode='r', shape=inversions_shape)
    
    # Get metadata for grouping patches by image/sentence
    metadata_df = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    split_metadata = metadata_df[metadata_df['split'] == split]
    images_in_split = split_metadata.index.tolist()
    
    # Handle text vs image datasets
    if model_input_size[0] == 'text':
        patches_per_image = None  # Variable length for text
        is_text_dataset = True
        # Load token counts for mapping tokens to sentences
        # MUST use correct token counts file for the model
        token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
        if not os.path.exists(token_counts_file):
            raise FileNotFoundError(f"Required token counts file not found: {token_counts_file}. "
                                  f"This MUST match the model input size {model_input_size}")
        
        token_counts_per_sentence = torch.load(token_counts_file, weights_only=False)
        token_counts_flat = torch.tensor([sum(x) if isinstance(x, list) else x for x in token_counts_per_sentence])
        sentence_starts = torch.cat([torch.tensor([0]), token_counts_flat.cumsum(0)[:-1]])
    else:
        patches_per_image = compute_patches_per_image(patch_size, model_input_size)
        is_text_dataset = False
    
    # Filter concepts to only those that exist in ground truth
    available_gt_concepts = set(gt_patches_per_concept_test.keys())
    concept_names = [c for c in concepts.keys() if c in available_gt_concepts]
    
    if len(concept_names) == 0:
        print(f"WARNING: No matching concepts found between concepts dict and ground truth!")
        print(f"  Concepts dict has: {list(concepts.keys())[:5]}...")
        print(f"  Ground truth has: {list(available_gt_concepts)[:5]}...")
        return
    
    concept_to_idx = {name: i for i, name in enumerate(concept_names)}
    
    # OPTIMIZATION: Convert to numpy for faster operations
    relevant_indices_np = relevant_indices.cpu().numpy()
    relevant_indices_set = set(relevant_indices_np.tolist())
    
    # OPTIMIZATION: Vectorized computation of image/sentence-to-patch/token mappings
    image_to_relevant_patches = {}
    
    # Create a boolean mask for relevant indices
    max_patch_idx = max(relevant_indices_set) + 1
    is_relevant = np.zeros(max_patch_idx, dtype=bool)
    is_relevant[relevant_indices_np] = True
    
    for global_img_idx in images_in_split:
        if is_text_dataset:
            # For text, map sentence to tokens
            start_patch = sentence_starts[global_img_idx].item()
            if global_img_idx + 1 < len(sentence_starts):
                end_patch = sentence_starts[global_img_idx + 1].item()
            else:
                end_patch = sentence_starts[global_img_idx].item() + token_counts_flat[global_img_idx].item()
        else:
            # For images, use get_patch_range_for_image
            start_patch, end_patch = get_patch_range_for_image(global_img_idx, patch_size, model_input_size)
        
        # Use numpy slicing for faster checking
        if end_patch <= max_patch_idx:
            patch_mask = is_relevant[start_patch:end_patch]
            if np.any(patch_mask):
                relevant_patches = np.arange(start_patch, end_patch)[patch_mask].tolist()
                image_to_relevant_patches[global_img_idx] = relevant_patches
    
    # Stage 1: Detect images ONCE for each concept using best detection thresholds
    print("\n[Stage 1] Detecting images for each concept...")
    detected_images_per_concept = {}
    
    for concept in tqdm(concept_names, desc="Detecting concepts"):
        if concept not in detection_thresholds:
            continue
            
        concept_idx = concept_to_idx[concept]
        detection_threshold = detection_thresholds[concept]
        
        # OPTIMIZATION: Load only this concept's activations
        if isinstance(act_loader, MatchedConceptActivationLoader):
            # For MatchedConceptActivationLoader, returns tensor directly
            concept_acts_tensor = act_loader[[concept]]
            if concept_acts_tensor.numel() == 0:
                print(f"Warning: Concept {concept} not found in activations")
                continue
            concept_acts_full = concept_acts_tensor.squeeze().to(device)
        else:
            # For ChunkedActivationLoader, use load_specific_concepts
            concept_acts_full = act_loader.load_specific_concepts([concept]).squeeze().to(device)
        
        # Extract only the relevant indices
        concept_acts_tensor = concept_acts_full[relevant_indices]
        
        # Find detected images for this concept (VECTORIZED)
        # Create a mapping from relevant patch index to image index
        patch_to_image = torch.zeros(len(relevant_indices), dtype=torch.long, device=device)
        
        # Build reverse mapping for fast lookup
        relevant_idx_to_pos = {idx: pos for pos, idx in enumerate(relevant_indices.tolist())}
        
        for img_idx, patches in image_to_relevant_patches.items():
            positions = [relevant_idx_to_pos[p] for p in patches if p in relevant_idx_to_pos]
            if positions:
                patch_to_image[positions] = img_idx
        
        # Find all patches that exceed threshold
        exceeds_threshold = concept_acts_tensor >= detection_threshold
        
        # Get unique images that have at least one patch exceeding threshold
        detected_image_indices = patch_to_image[exceeds_threshold]
        detected_images = set(detected_image_indices[detected_image_indices > 0].cpu().tolist())
        
        detected_images_per_concept[concept] = detected_images
        print(f"   {concept}: {len(detected_images)}/{len(images_in_split)} images detected")
    
    # Stage 2: Test different inversion thresholds on the detected images
    print("\n[Stage 2] Testing inversion thresholds...")
    
    # Process each inversion percentile
    for invert_percentile in tqdm(invert_percentiles, desc="Inversion percentiles"):
        results = {}
        
        for concept in concept_names:
            if concept not in detection_thresholds or concept not in detected_images_per_concept:
                continue
                
            concept_idx = concept_to_idx[concept]
            detected_images = detected_images_per_concept[concept]
            
            # Evaluate inversion on ALL patches, but only for detected images
            # Non-detected images contribute all patches as "negative"
            
            all_predictions = torch.zeros(len(relevant_indices), dtype=torch.bool, device=device)
            # Get labels for the relevant indices
            concept_labels_tensor = all_concept_labels[concept]
            all_labels = concept_labels_tensor[relevant_indices].to(device).bool()
            
            # For detected images: evaluate patches using inversion threshold
            # Load inversions using appropriate method
            if 'inversion_loader' in locals():
                # Using ChunkedActivationLoader
                # The columns in the inversion loader are concept names, not indices
                relevant_inversions = inversion_loader.load_concept_activations_for_indices(
                    concept_name=concept,  # Use concept name, not index
                    indices=relevant_indices.cpu().numpy().tolist(),
                    device=device
                )
            else:
                # Using legacy memory-mapped format
                concept_inversions_np = inversions_mmap[:, concept_idx]
                relevant_inversions_np = concept_inversions_np[relevant_indices.cpu().numpy()]
                relevant_inversions = torch.from_numpy(relevant_inversions_np.copy()).to(device)
            
            # Get all patches from detected images
            all_patches_from_detected_images = []
            for global_img_idx in detected_images:
                if global_img_idx in image_to_relevant_patches:
                    all_patches_from_detected_images.extend(image_to_relevant_patches[global_img_idx])
            
            if len(all_patches_from_detected_images) > 0:
                # Convert to local indices for the relevant_indices tensor (OPTIMIZED)
                # Create mapping once for O(1) lookups
                global_to_local = {global_idx: i for i, global_idx in enumerate(relevant_indices.tolist())}
                local_all_detected = [global_to_local[idx] for idx in all_patches_from_detected_images if idx in global_to_local]
                
                if len(local_all_detected) > 0:
                    # Calculate inversion threshold using ALL patches from detected images
                    detected_similarities = relevant_inversions[local_all_detected]
                    
                    # Filter out any -1 values (train patches) just in case
                    valid_similarities = detected_similarities[detected_similarities > -0.5]
                    if len(valid_similarities) > 0:
                        inversion_threshold = torch.quantile(valid_similarities, invert_percentile).item()
                    else:
                        # Fallback if somehow all values are -1
                        inversion_threshold = 0.0
                    
                    # Apply inversion threshold to all patches from detected images
                    for local_idx in local_all_detected:
                        if relevant_inversions[local_idx] >= inversion_threshold:
                            all_predictions[local_idx] = True
            
            # Note: Patches in non-detected images remain False (negative predictions)
            
            # Compute metrics
            tp = torch.sum(all_predictions & all_labels).item()
            fp = torch.sum(all_predictions & ~all_labels).item()
            tn = torch.sum(~all_predictions & ~all_labels).item()
            fn = torch.sum(~all_predictions & all_labels).item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
            
            results[concept] = {
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy
            }
        
        # Save results with two-stage prefix
        os.makedirs(f'Quant_Results/{dataset_name}', exist_ok=True)
        save_path = f'Quant_Results/{dataset_name}/twostage_superdetector_invert_{invert_percentile}_{agglomerate_type}_{con_label}_{split}.pt'
        torch.save(results, save_path)
        print(f"Saved two-stage results: {save_path}")
    
    # Clean up
    if 'inversion_loader' in locals():
        # Clean up ChunkedActivationLoader
        if hasattr(inversion_loader, 'close'):
            inversion_loader.close()
        del inversion_loader
    else:
        # Clean up memory-mapped array
        del inversions_mmap
        os.remove(mmap_path)


def find_optimal_twostage_superdetector_thresholds(
    invert_percentiles: List[float],
    dataset_name: str,
    con_label: str,
    model_input_size: Tuple,
    agglomerate_type: str = 'avg',
    optimization_metric: str = 'f1'
):
    """
    Find optimal inversion thresholds for two-stage superdetector method.
    """
    print(f"\nFinding optimal two-stage superdetector thresholds using {optimization_metric}...")
    
    # Load all two-stage calibration results
    all_results = {}
    for invert_perc in invert_percentiles:
        result_file = f'Quant_Results/{dataset_name}/twostage_superdetector_invert_{invert_perc}_{agglomerate_type}_{con_label}_cal.pt'
        if os.path.exists(result_file):
            results = torch.load(result_file, weights_only=False)
            all_results[invert_perc] = results
        else:
            print(f"Warning: Missing results file {result_file}")
    
    if not all_results:
        print("No two-stage calibration results found!")
        return {}
    
    # Find best inversion percentile for each concept
    optimal_thresholds = {}
    best_inversion_percentiles = {}
    
    # Get all concepts from first result file
    first_result = next(iter(all_results.values()))
    concepts = list(first_result.keys())
    
    for concept in concepts:
        best_score = -1
        best_perc = None
        
        for invert_perc, results in all_results.items():
            if concept in results:
                score = results[concept][optimization_metric]
                if score > best_score:
                    best_score = score
                    best_perc = invert_perc
        
        if best_perc is not None:
            optimal_thresholds[concept] = {
                'invert_percentile': best_perc,
                f'best_{optimization_metric}': best_score
            }
            best_inversion_percentiles[concept] = {
                'best_percentile': best_perc,
                f'best_{optimization_metric}': best_score
            }
            print(f"   {concept}: best inversion percentile = {best_perc}, {optimization_metric} = {best_score:.4f}")
    
    # Save best inversion percentiles
    os.makedirs(f'Best_Inversion_Percentiles_Cal/{dataset_name}', exist_ok=True)
    inversion_file = f'Best_Inversion_Percentiles_Cal/{dataset_name}/best_inversion_percentiles_twostage_superpatch_{con_label}.pt'
    torch.save(best_inversion_percentiles, inversion_file)
    
    print(f"Saved best two-stage inversion percentiles to {inversion_file}")
    return optimal_thresholds


def detect_then_invert_twostage_superdetector_with_optimal_thresholds(
    act_loader: Union[ChunkedActivationLoader, MatchedConceptActivationLoader],
    concepts: Dict,
    gt_samples_per_concept: Dict,
    gt_samples_per_concept_test: Dict,
    embedding_loader: ChunkedEmbeddingLoader,
    device: str,
    dataset_name: str,
    model_input_size: Tuple,
    con_label: str,
    agglomerate_type: str = 'avg',
    all_object_patches: Optional[set] = None,
    patch_size: int = 14,
    split: str = 'test',
    scratch_dir: str = ''
):
    """
    Evaluate two-stage superdetector method on test set using optimal thresholds.
    """
    print(f"\n=== Two-Stage Superdetector Test Evaluation with Optimal Thresholds ===")
    
    # Load best inversion percentiles  
    inversion_file = f'Best_Inversion_Percentiles_Cal/{dataset_name}/best_inversion_percentiles_twostage_superpatch_{con_label}.pt'
    if not os.path.exists(inversion_file):
        raise FileNotFoundError(f"Best two-stage inversion percentiles not found: {inversion_file}")
    best_inversion_percentiles = torch.load(inversion_file, weights_only=False)
    
    # Get unique inversion percentiles to evaluate
    unique_invert_percentiles = list(set(info['best_percentile'] for info in best_inversion_percentiles.values()))
    
    print(f"Evaluating {len(unique_invert_percentiles)} unique inversion percentiles on test set...")
    
    # Run two-stage evaluation for each unique percentile
    for invert_perc in unique_invert_percentiles:
        detect_then_invert_superdetector_twostage_metrics(
            [invert_perc], act_loader, concepts, gt_samples_per_concept, gt_samples_per_concept_test,
            embedding_loader, device, dataset_name, model_input_size, con_label,
            all_object_patches, patch_size, agglomerate_type, split
        )
    
    # Collect results for each concept using their optimal percentile
    final_results = {}
    for concept, info in best_inversion_percentiles.items():
        if concept not in concepts:
            continue
            
        optimal_invert_perc = info['best_percentile']
        result_file = f'Quant_Results/{dataset_name}/twostage_superdetector_invert_{optimal_invert_perc}_{agglomerate_type}_{con_label}_{split}.pt'
        
        if os.path.exists(result_file):
            results = torch.load(result_file, weights_only=False)
            if concept in results:
                final_results[concept] = results[concept].copy()
                final_results[concept]['invert_percentile'] = optimal_invert_perc
    
    # Save final test results
    save_path = f'Quant_Results/{dataset_name}/optimal_test_results_twostage_superpatch_{con_label}_f1.pt'
    torch.save(final_results, save_path)
    
    # Also save as CSV for visualization compatibility
    if final_results:
        import pandas as pd
        rows = []
        for concept, metrics in final_results.items():
            row = {
                'concept': concept,
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'tn': metrics['tn'],
                'fn': metrics['fn'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy']
            }
            if 'invert_percentile' in metrics:
                row['invert_percentile'] = metrics['invert_percentile']
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = f'Quant_Results/{dataset_name}/twostage_superpatch_avg_{con_label}_optimal_test.csv'
        df.to_csv(csv_path, index=False)
        
        avg_precision = sum(r['precision'] for r in final_results.values()) / len(final_results)
        avg_recall = sum(r['recall'] for r in final_results.values()) / len(final_results)
        avg_f1 = sum(r['f1'] for r in final_results.values()) / len(final_results)
        
        print(f"\nTwo-Stage Superdetector Test Results:")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Results saved to: {save_path}")
        print(f"CSV saved to: {csv_path}")
    else:
        print("No results computed")


def detect_then_invert_superdetector_calibration_metrics(
    invert_percentiles: List[float],
    act_loader: Union[ChunkedActivationLoader, MatchedConceptActivationLoader],
    concepts: Dict,
    gt_patches_per_concept: Dict,
    gt_patches_per_concept_test: Dict,
    device: str,
    dataset_name: str,
    model_input_size: Tuple,
    con_label: str,
    all_object_patches: Optional[set] = None,
    patch_size: int = 14,
    agglomerate_type: str = 'avg'
):
    """
    Superdetector-specific calibration evaluation that saves with superpatch prefix.
    """
    from utils.quant_concept_evals_utils_loader import detect_then_invert_metrics
    from utils.patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence
    from utils.general_utils import get_split_df
    from utils.quant_concept_evals_utils import create_binary_labels
    from utils.gt_concept_segmentation_utils import remap_text_ground_truth_indices
    from tqdm import tqdm
    
    # GLOBAL INDICES APPROACH: Use the same approach as fixed regular method
    if model_input_size[0] == 'text':
        # For text: get sentence-level split, then use all token indices
        split_df = get_split_df(dataset_name)
        token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
        if not os.path.exists(token_counts_file):
            raise FileNotFoundError(f"Required token counts file not found: {token_counts_file}")
        token_counts_per_sentence = torch.load(token_counts_file, weights_only=False)
        
        # Use ALL token indices - no split filtering here (same as fixed regular method)
        loader_info = act_loader.get_activation_info() if hasattr(act_loader, 'get_activation_info') else act_loader.get_info()
        total_tokens = loader_info['total_samples']
        relevant_indices = torch.arange(total_tokens)
    else:
        # For images: use patch-based split filtering  
        split_df = get_patch_split_df(dataset_name, model_input_size=model_input_size, patch_size=patch_size)
        cal_indices = torch.tensor(split_df.index[split_df == 'cal'].tolist())
        relevant_indices = filter_patches_by_image_presence(cal_indices, dataset_name, model_input_size)
    
    # Get ground truth labels from calibration set
    loader_info = act_loader.get_activation_info() if hasattr(act_loader, 'get_activation_info') else act_loader.get_info()
    total_samples = loader_info['total_samples']
    
    # GLOBAL INDICES: Keep ground truth in global indices (same as fixed regular method)
    if model_input_size[0] == 'text':
        # Filter ground truth to only include tokens from calibration sentences, but keep global indices
        cal_sentence_indices = [i for i in range(len(token_counts_per_sentence)) if split_df.get(i) == 'cal']
        
        # Map to token ranges for calibration split 
        cal_token_ranges = []
        current_token = 0
        for sent_idx in range(len(token_counts_per_sentence)):
            num_tokens = sum(token_counts_per_sentence[sent_idx]) if isinstance(token_counts_per_sentence[sent_idx], list) else token_counts_per_sentence[sent_idx]
            if sent_idx in cal_sentence_indices:
                cal_token_ranges.extend(range(current_token, current_token + num_tokens))
            current_token += num_tokens
        
        # Filter ground truth to tokens in calibration split, but preserve global indices
        filtered_gt = {}
        for concept, indices in gt_patches_per_concept_test.items():
            # Keep only tokens that are in calibration split, but preserve global indices
            filtered_gt[concept] = [idx for idx in indices if idx in cal_token_ranges]
        
        all_concept_labels = create_binary_labels(total_samples, filtered_gt)
    else:
        # For image datasets, filter ground truth to only include calibration indices
        cal_gt = {}
        for concept, indices in gt_patches_per_concept_test.items():
            # Filter to only include indices that are in the calibration set
            cal_indices_set = set(relevant_indices.tolist())
            cal_gt[concept] = [idx for idx in indices if idx in cal_indices_set]
        all_concept_labels = create_binary_labels(total_samples, cal_gt)
    
    # Use best detection percentiles
    best_detection_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
    if not os.path.exists(best_detection_file):
        raise FileNotFoundError(f"Best detection percentiles required: {best_detection_file}")
    best_detection_percentiles = torch.load(best_detection_file, weights_only=False)
    
    # Group by unique detection percentiles
    unique_detect_percs = set()
    concept_to_detect_perc = {}
    for concept, info in best_detection_percentiles.items():
        detect_perc = info['best_percentile']
        unique_detect_percs.add(detect_perc)
        concept_to_detect_perc[concept] = detect_perc
    
    # OPTIMIZATION: Process by inversion percentile instead of detection percentile
    pbar = tqdm(invert_percentiles, desc="Evaluating superdetector thresholds")
    for invert_percentile in pbar:
        # Group concepts by detection percentile for batch processing
        concepts_by_detect = {}
        
        for concept, info in best_detection_percentiles.items():
            if concept not in concepts:
                continue
            detect_perc = info['best_percentile']
            
            # Only process if invert >= detect
            if invert_percentile >= detect_perc:
                if detect_perc not in concepts_by_detect:
                    concepts_by_detect[detect_perc] = []
                concepts_by_detect[detect_perc].append(concept)
        
        # Process each detection group
        for detect_perc, concept_list in concepts_by_detect.items():
            # Batch process all concepts with same detection percentile
            filtered_concepts = {c: concepts[c] for c in concept_list if c in concepts}
            
            if filtered_concepts:
                # Call detect_then_invert_metrics once for this batch
                metrics_results = detect_then_invert_metrics(
                    detect_perc, [invert_percentile],
                    act_loader, filtered_concepts,
                    gt_patches_per_concept, gt_patches_per_concept_test,
                    relevant_indices, all_concept_labels,
                    device, dataset_name, model_input_size, con_label,
                    all_object_patches=all_object_patches,
                    patch_size=patch_size
                )
                
                # Save results with superpatch prefix
                if invert_percentile in metrics_results:
                    os.makedirs(f'Quant_Results/{dataset_name}', exist_ok=True)
                    save_path = f'Quant_Results/{dataset_name}/detectfirst_{detect_perc}_invert_{invert_percentile}_superpatch_{agglomerate_type}_inv_{con_label}.pt'
                    torch.save(metrics_results[invert_percentile], save_path)


# This function has been removed and replaced with detect_then_invert_superdetector_twostage_metrics


def find_optimal_superdetector_thresholds(invert_percentiles: List[float],
                                        dataset_name: str,
                                        con_label: str,
                                        model_input_size: Tuple,
                                        optimization_metric: str = 'f1',
                                        agglomerate_type: str = 'avg'):
    """
    Find optimal inversion thresholds for superdetector method using best detection percentiles.
    """
    print(f"Finding optimal superdetector inversion thresholds based on {optimization_metric}...")
    
    # Load the best detection percentiles from Step 2
    best_detection_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
    if not os.path.exists(best_detection_file):
        raise FileNotFoundError(f"Best detection percentiles not found: {best_detection_file}")
    
    best_detection_percentiles = torch.load(best_detection_file, weights_only=False)
    print(f"Using best detection percentiles from {best_detection_file}")
    
    optimal_thresholds = {}
    
    # Find optimal inversion threshold for each concept using its best detection threshold
    for concept, detection_info in tqdm(best_detection_percentiles.items(), desc="Optimizing inversion thresholds"):
        best_detect_p = detection_info['best_percentile']
        best_score = -float('inf')
        best_invert_p = None
        
        # Only try inversion percentiles >= detection percentile
        valid_invert_percentiles = [p for p in invert_percentiles if p >= best_detect_p]
        
        for invert_p in valid_invert_percentiles:
            pt_filename = f"Quant_Results/{dataset_name}/detectfirst_{best_detect_p}_invert_{invert_p}_superpatch_{agglomerate_type}_inv_{con_label}.pt"
            
            try:
                results = torch.load(pt_filename, weights_only=False)
                if concept in results and optimization_metric in results[concept]:
                    score = results[concept][optimization_metric]
                    if not math.isnan(score) and score > best_score:
                        best_score = score
                        best_invert_p = invert_p
            except Exception as e:
                print(f"Error loading {pt_filename}: {e}")
                continue
        
        if best_invert_p is not None:
            optimal_thresholds[concept] = {
                'detect_percentile': best_detect_p,
                'invert_percentile': best_invert_p,
                f'best_{optimization_metric}': best_score
            }
    
    # Save inversion percentiles to Best_Inversion_Percentiles_Cal
    os.makedirs(f'Best_Inversion_Percentiles_Cal/{dataset_name}', exist_ok=True)
    best_inversion_percentiles = {}
    for concept, info in optimal_thresholds.items():
        best_inversion_percentiles[concept] = {
            'best_percentile': info['invert_percentile'],
            f'best_{optimization_metric}': info[f'best_{optimization_metric}']
        }
    inversion_file = f'Best_Inversion_Percentiles_Cal/{dataset_name}/best_inversion_percentiles_superpatch_{con_label}.pt'
    torch.save(best_inversion_percentiles, inversion_file)
    
    print(f"Saved best inversion percentiles to {inversion_file}")
    return optimal_thresholds


def detect_then_invert_locally_with_optimal_thresholds(act_loader: Union[ChunkedActivationLoader, MatchedConceptActivationLoader],
                                                      concepts: Dict,
                                                      gt_samples_per_concept: Dict,
                                                      gt_samples_per_concept_test: Dict,
                                                      device: str,
                                                      dataset_name: str,
                                                      model_input_size: Tuple,
                                                      con_label: str,
                                                      embedding_loader: ChunkedEmbeddingLoader,
                                                      agglomerate_type: str = 'avg',
                                                      all_object_patches: Optional[set] = None,
                                                      patch_size: int = 14,
                                                      split: str = 'test'):
    """
    Evaluate superdetector method on test set using optimal thresholds from separate files.
    Uses Best_Detection_Percentiles_Cal for detection and Best_Inversion_Percentiles_Cal for inversion.
    
    Args:
        act_loader: Activation loader (ChunkedActivationLoader or MatchedConceptActivationLoader)
        split: Data split to evaluate on ('test' by default)
    """
    
    # Load best detection percentiles
    detection_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
    if not os.path.exists(detection_file):
        raise FileNotFoundError(f"Best detection percentiles not found: {detection_file}")
    best_detection_percentiles = torch.load(detection_file, weights_only=False)
    
    # Load best inversion percentiles  
    inversion_file = f'Best_Inversion_Percentiles_Cal/{dataset_name}/best_inversion_percentiles_superpatch_{con_label}.pt'
    if not os.path.exists(inversion_file):
        raise FileNotFoundError(f"Best inversion percentiles not found: {inversion_file}")
    best_inversion_percentiles = torch.load(inversion_file, weights_only=False)
    
    print(f"Evaluating superdetector method on test set with optimal thresholds...")
    
    # Get detection thresholds for each concept
    detection_thresholds = {}
    if 'kmeans' not in con_label and 'sae' not in con_label:
        all_detect_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt', weights_only=False)
        for concept, info in best_detection_percentiles.items():
            best_perc = info['best_percentile']
            detection_thresholds[concept] = all_detect_thresholds[best_perc][concept][0] if isinstance(all_detect_thresholds[best_perc][concept], tuple) else all_detect_thresholds[best_perc][concept]
    else:
        # Handle kmeans and SAE thresholds (unsupervised methods)
        raw_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt', weights_only=False)
        alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)
        
        for concept, info in best_detection_percentiles.items():
            best_perc = info['best_percentile']
            cluster_id = alignment_results[concept]['best_cluster']
            key = (concept, cluster_id)
            if best_perc in raw_thresholds and key in raw_thresholds[best_perc]:
                detection_thresholds[concept] = raw_thresholds[best_perc][key][0] if isinstance(raw_thresholds[best_perc][key], tuple) else raw_thresholds[best_perc][key]
    
    # OPTIMIZATION: Batch load inversion thresholds
    print("Loading inversion thresholds...")
    inversion_thresholds = {}
    
    # GLOBAL INDICES APPROACH: Use the same approach as fixed regular method
    from utils.general_utils import get_split_df
    
    if model_input_size[0] == 'text':
        # For text: use all token indices (same as fixed regular method)
        split_df = get_split_df(dataset_name)
        token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
        if not os.path.exists(token_counts_file):
            raise FileNotFoundError(f"Required token counts file not found: {token_counts_file}")
        
        loader_info = act_loader.get_activation_info() if hasattr(act_loader, 'get_activation_info') else act_loader.get_info()
        total_tokens = loader_info['total_samples']
        test_relevant_indices = torch.arange(total_tokens)
    else:
        # For images: use patch-based split filtering
        split_df = get_patch_split_df(dataset_name, model_input_size=model_input_size, patch_size=patch_size)
        test_indices = torch.tensor(split_df.index[split_df == 'test'].tolist())
        test_relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size)
    
    # Group concepts by inversion percentile to minimize file loads
    concepts_by_percentile = {}
    for concept, inv_info in best_inversion_percentiles.items():
        best_inv_perc = inv_info['best_percentile']
        if best_inv_perc not in concepts_by_percentile:
            concepts_by_percentile[best_inv_perc] = []
        concepts_by_percentile[best_inv_perc].append(concept)
    
    # Load inversions using ChunkedActivationLoader if available
    inversion_file = f'superpatch_{agglomerate_type}_inv_{con_label}_chunks_info.json'
    inversion_path = os.path.join(scratch_dir, 'Superpatches', dataset_name, inversion_file)
    
    if os.path.exists(inversion_path):
        # Use ChunkedActivationLoader
        print("Loading chunked inversions for threshold calculation...")
        inversion_loader = ChunkedActivationLoader(
            dataset_name=dataset_name,
            acts_file=f'superpatch_{agglomerate_type}_inv_{con_label}.pt',
            scratch_dir=scratch_dir,
            device='cpu'
        )
        
        # Extract thresholds for all concepts
        for inv_perc, concept_list in concepts_by_percentile.items():
            for concept in concept_list:
                if concept in concepts:
                    concept_idx = list(concepts.keys()).index(concept)
                    # Load inversions for this concept at relevant indices
                    relevant_inversions = inversion_loader.load_specific_indices_and_concepts(
                        indices=test_relevant_indices.cpu().numpy().tolist(),
                        concept_indices=[concept_idx]
                    ).squeeze()
                    # Filter out any -1 values (train patches) before computing threshold
                    valid_inversions = relevant_inversions[relevant_inversions > -0.5]
                    if len(valid_inversions) > 0:
                        threshold = torch.quantile(valid_inversions, inv_perc).item()
                    else:
                        threshold = 0.0
                    inversion_thresholds[concept] = threshold
        
        # Clean up loader
        if hasattr(inversion_loader, 'close'):
            inversion_loader.close()
        del inversion_loader
    else:
        # Fallback to legacy single-file format
        for inv_perc, concept_list in concepts_by_percentile.items():
            # Check both scratch and local directories
            inversion_results_file_scratch = os.path.join(scratch_dir, 'Superpatches', dataset_name, f'superpatch_{agglomerate_type}_inv_{con_label}.pt')
            inversion_results_file_local = f'Superpatches/{dataset_name}/superpatch_{agglomerate_type}_inv_{con_label}.pt'
            
            if os.path.exists(inversion_results_file_scratch):
                inversion_results_file = inversion_results_file_scratch
            elif os.path.exists(inversion_results_file_local):
                inversion_results_file = inversion_results_file_local
            else:
                continue
                
            inversions = torch.load(inversion_results_file, weights_only=False)
            
            for concept in concept_list:
                if concept in concepts:
                    concept_idx = list(concepts.keys()).index(concept)
                    concept_inversions = inversions[:, concept_idx]
                    relevant_inversions = concept_inversions[test_relevant_indices]
                    # Filter out any -1 values (train patches) before computing threshold
                    valid_inversions = relevant_inversions[relevant_inversions > -0.5]
                    if len(valid_inversions) > 0:
                        threshold = torch.quantile(valid_inversions, inv_perc).item()
                    else:
                        threshold = 0.0
                    inversion_thresholds[concept] = threshold
    
    # Now perform evaluation on test set using local superdetectors
    # Filter concepts to only those that exist in ground truth
    available_gt_concepts = set(gt_samples_per_concept_test.keys())
    concept_names = [c for c in concepts.keys() if c in available_gt_concepts]
    
    if len(concept_names) == 0:
        print(f"WARNING: No matching concepts found between concepts dict and ground truth!")
        print(f"  Concepts dict has: {list(concepts.keys())[:5]}...")
        print(f"  Ground truth has: {list(available_gt_concepts)[:5]}...")
        return
    
    # Convert relevant_indices to set for O(1) lookup
    relevant_indices_set = set(test_relevant_indices.tolist())
    
    # Create binary labels for ground truth and move to GPU
    all_concept_labels = create_binary_labels(len(split_df), gt_samples_per_concept_test)
    # Keep labels on CPU for indexing
    all_concept_labels_gpu = all_concept_labels
    
    # Initialize results storage
    fp_counts = {}
    fn_counts = {}
    tp_counts = {}
    tn_counts = {}
    
    # Process each image/sample in test set
    if isinstance(model_input_size, tuple) and model_input_size[0] == 'text':
        # For text, get number of test samples from metadata
        metadata_df = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
        test_metadata = metadata_df[metadata_df['split'] == 'test']
        test_sample_indices = test_metadata.index.tolist()
        n_test_samples = len(test_sample_indices)
        print(f"Evaluating {n_test_samples} test text samples with local superdetectors...")
    else:
        patches_per_image = compute_patches_per_image(patch_size, model_input_size)
        n_test_images = len(test_relevant_indices) // patches_per_image
        print(f"Evaluating {n_test_images} test images with local superdetectors...")
    
    # OPTIMIZATION: Process all concepts together for each image
    # This reduces redundant embedding and activation loading
    
    # Prepare concepts with valid thresholds
    valid_concepts = [(c, i) for i, c in enumerate(concept_names) 
                      if c in detection_thresholds and c in inversion_thresholds]
    
    if not valid_concepts:
        print("No valid concepts with thresholds found")
        return
    
    # Initialize counts for all concepts
    for concept, _ in valid_concepts:
        fp_counts[concept] = 0
        fn_counts[concept] = 0
        tp_counts[concept] = 0
        tn_counts[concept] = 0
    
    # Process each test sample once, evaluating all concepts
    samples_to_process = test_sample_indices if (isinstance(model_input_size, tuple) and model_input_size[0] == 'text') else range(n_test_images)
    
    # OPTIMIZATION: Process in batches for better GPU utilization
    batch_size = 10
    patches_per_image = compute_patches_per_image(patch_size, model_input_size)
    
    for batch_start in tqdm(range(0, len(samples_to_process), batch_size), desc="Processing test batches"):
        batch_end = min(batch_start + batch_size, len(samples_to_process))
        batch_samples = samples_to_process[batch_start:batch_end] if isinstance(samples_to_process, list) else list(range(batch_start, batch_end))
        
        # Collect all patches for this batch
        batch_patch_ranges = []
        for sample_idx in batch_samples:
            # Get patch/token range for this sample
            if isinstance(model_input_size, tuple) and model_input_size[0] == 'text':
                start_patch_idx, end_patch_idx = get_patch_range_for_text(sample_idx, dataset_name, model_input_size)
            else:
                start_patch_idx, end_patch_idx = get_patch_range_for_image(sample_idx, patch_size, model_input_size)
            
            batch_patch_ranges.append((start_patch_idx, end_patch_idx))
        
        # Find min/max patches for batch loading
        all_batch_patches = []
        for start, end in batch_patch_ranges:
            all_batch_patches.extend(range(start, end))
        
        if not all_batch_patches:
            continue
            
        min_batch_patch = min(all_batch_patches)
        max_batch_patch = max(all_batch_patches) + 1
        
        # OPTIMIZATION: Batch load embeddings for all images at once
        batch_embeds = embedding_loader.load_specific_embeddings(all_batch_patches).to(device)
        
        # OPTIMIZATION: Batch load activations for all concepts and all images at once
        if isinstance(act_loader, MatchedConceptActivationLoader):
            concepts_to_load = [c for c, _ in valid_concepts]
            batch_acts_df = act_loader.load_concept_range(concepts_to_load, min_batch_patch, max_batch_patch)
            # Convert to tensor for faster processing
            batch_acts = torch.stack([
                torch.tensor(batch_acts_df[c].values, device=device) 
                for c, _ in valid_concepts
            ], dim=1)  # [n_patches, n_concepts]
        else:
            batch_acts_tensor = act_loader.load_tensor_range(min_batch_patch, max_batch_patch)
            concept_indices = [i for _, i in valid_concepts]
            batch_acts = batch_acts_tensor[:, concept_indices].to(device)
        
        # Process each image in the batch
        for local_idx, (sample_idx, (start_patch_idx, end_patch_idx)) in enumerate(zip(batch_samples, batch_patch_ranges)):
            image_patches = list(range(start_patch_idx, end_patch_idx))
            
            # Get embeddings and activations for this image from batch data
            local_start = start_patch_idx - min_batch_patch
            local_end = end_patch_idx - min_batch_patch
            image_embeds = batch_embeds[local_start:local_end]
            image_acts_all = batch_acts[local_start:local_end]  # [patches_per_image, n_concepts]
            
            # Create relevance mask
            if isinstance(model_input_size, tuple) and model_input_size[0] == 'text':
                relevant_mask = torch.ones(len(image_patches), dtype=torch.bool, device=device)
            else:
                relevant_mask = torch.tensor([p in relevant_indices_set for p in image_patches], device=device)
                
                if not relevant_mask.any():
                    continue
            
            # Normalize embeddings once
            image_embeds_norm = torch.nn.functional.normalize(image_embeds, dim=1)
            
            # Process all concepts for this image
            for concept_idx, (concept, _) in enumerate(valid_concepts):
                detect_thresh = detection_thresholds[concept]
                invert_thresh = inversion_thresholds[concept]
                
                # Get activations for this concept
                image_acts = image_acts_all[:, concept_idx]
                
                # Find superdetector patches
                super_mask = (image_acts >= detect_thresh) & relevant_mask
                
                if super_mask.any():
                    super_indices_local = torch.where(super_mask)[0]
                    super_embeds = image_embeds[super_indices_local]
                    super_acts = image_acts[super_indices_local]
                    
                    # Compute superdetector vector
                    if agglomerate_type == 'avg':
                        weights = torch.softmax(super_acts, dim=0)
                        local_super_vec = torch.sum(super_embeds * weights.unsqueeze(-1), dim=0)
                    else:  # max
                        max_idx = torch.argmax(super_acts)
                        local_super_vec = super_embeds[max_idx]
                    
                    # Compute similarities
                    local_super_vec_norm = torch.nn.functional.normalize(local_super_vec.unsqueeze(0), dim=1)
                    similarities = torch.matmul(image_embeds_norm, local_super_vec_norm.t()).squeeze()
                    predictions = similarities >= invert_thresh
                else:
                    predictions = torch.zeros(len(image_patches), dtype=torch.bool, device=device)
                
                # Get ground truth
                gt_values = all_concept_labels_gpu[concept][image_patches].to(device)
                
                # Count only relevant patches
                relevant_preds = predictions[relevant_mask]
                relevant_gt = gt_values[relevant_mask]
                
                # Update counts
                tp_counts[concept] += ((relevant_preds == 1) & (relevant_gt == 1)).sum().item()
                fp_counts[concept] += ((relevant_preds == 1) & (relevant_gt == 0)).sum().item()
                fn_counts[concept] += ((relevant_preds == 0) & (relevant_gt == 1)).sum().item()
                tn_counts[concept] += ((relevant_preds == 0) & (relevant_gt == 0)).sum().item()
        
        # Clean up GPU memory after each batch
        del batch_embeds, batch_acts
        if 'batch_acts_df' in locals():
            del batch_acts_df
        torch.cuda.empty_cache()
    
    # Compute metrics
    results = {}
    for concept in concept_names:
        if concept in fp_counts:
            # compute_stats_from_counts expects the counts as individual integers
            tp = tp_counts[concept]
            fp = fp_counts[concept]
            tn = tn_counts[concept]
            fn = fn_counts[concept]
            
            # Compute metrics directly
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            results[concept] = {
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy
            }
    
    # Save results
    os.makedirs(f'Quant_Results/{dataset_name}', exist_ok=True)
    save_path = f'Quant_Results/{dataset_name}/optimal_test_results_superpatch_{con_label}_f1.pt'
    torch.save(results, save_path)
    print(f"Results saved to: {save_path}")
    
    # Print summary
    if results:
        f1_scores = [v['f1'] for v in results.values() if 'f1' in v]
        if f1_scores:
            f1_tensor = torch.tensor(f1_scores)
            print(f"\nTest Set Results Summary:")
            print(f"  Average F1: {f1_tensor.mean():.3f} ± {f1_tensor.std():.3f}")
            print(f"  Min F1: {f1_tensor.min():.3f}, Max F1: {f1_tensor.max():.3f}")


