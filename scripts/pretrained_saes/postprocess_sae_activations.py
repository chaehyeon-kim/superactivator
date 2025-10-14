"""
Postprocess SAE activations: prune based on calibration statistics and convert to dense format.
This combines the functionality of prune_sae_units_sparse.py and convert_sparse_to_dense.py
"""

import torch
import numpy as np
import json
import os
import gc
import pandas as pd
from tqdm import tqdm
import argparse
from collections import defaultdict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.general_utils import get_split_df

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRATCH_DIR = ''

# Default pruning parameters
CLIP_PRUNING_CONFIG = {
    'target_units': 5000,
    'dead_threshold': 200,          # min activation count
    'ubiquitous_threshold': 0.02,   # max activation fraction
    'strength_percentile': 75,      # keep units above this percentile of p99 values
}

GEMMA_PRUNING_CONFIG = {
    'target_units': 3000,
    'dead_threshold': 500,
    'ubiquitous_threshold': 0.03,
    'strength_percentile': 70,
}


def get_calibration_indices(dataset_name, sample_type, model_type):
    """
    Get indices for calibration split samples.
    For patch/token types, converts image/text indices to patch/token indices.
    """
    # Load split information
    # When running from Experiments/, Data is one level up
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    split_series = metadata['split']
    
    # Get calibration sample indices
    cal_mask = split_series == 'cal'
    cal_indices = cal_mask[cal_mask].index.tolist()
    
    if not cal_indices:
        print(f"WARNING: No calibration samples found for {dataset_name}")
        return None, None
    
    cal_start = min(cal_indices)
    cal_end = max(cal_indices) + 1
    
    print(f"   Calibration samples: {cal_start} to {cal_end-1} (total: {len(cal_indices)})")
    
    # For patch/token samples, convert to patch/token indices
    if sample_type == 'patch':
        # Each image has 256 patches (16x16)
        patches_per_image = 256
        cal_start_patches = cal_start * patches_per_image
        cal_end_patches = cal_end * patches_per_image
        print(f"   Calibration patches: {cal_start_patches} to {cal_end_patches-1}")
        return cal_start_patches, cal_end_patches
        
    elif sample_type == 'patch' and model_type == 'Gemma':
        # For text, we need token counts to map paragraph indices to token indices
        # When running from Experiments/, GT_Samples is at the same level
        token_counts_file = f"GT_Samples/{dataset_name}/token_counts_inputsize_('text', 'text2').pt"
        if os.path.exists(token_counts_file):
            token_counts = torch.load(token_counts_file, weights_only=True)
            token_lengths = [sum(x) if isinstance(x, list) else x for x in token_counts]
            
            # Find token indices for calibration paragraphs
            cal_start_tokens = sum(token_lengths[:cal_start])
            cal_end_tokens = sum(token_lengths[:cal_end])
            print(f"   Calibration patches (tokens): {cal_start_tokens} to {cal_end_tokens-1}")
            return cal_start_tokens, cal_end_tokens
        else:
            print(f"   WARNING: Token counts file not found: {token_counts_file}")
            return None, None
    
    # For CLS samples, use image/text indices directly
    return cal_start, cal_end


def compute_unit_statistics_sparse(dataset_name, model_type, sample_type='patch'):
    """
    Compute statistics for each SAE unit from sparse activations.
    Only uses calibration split for computing statistics.
    """
    print(f"\n📊 Computing unit statistics for {dataset_name} - {model_type} - {sample_type}")
    print(f"   Using device: {DEVICE}")
    print(f"   Using CALIBRATION split only for statistics")
    
    # Get calibration indices
    cal_start, cal_end = get_calibration_indices(dataset_name, sample_type, model_type)
    if cal_start is None:
        return None, 0, 0
    
    # Determine file paths
    if model_type == 'CLIP':
        acts_dir = f"{SCRATCH_DIR}SAE_Activations_Sparse/{dataset_name}"
        prefix = f"clipscope_{sample_type}_sparse_layer22"
        n_expected_units = 65536
    else:  # Gemma
        acts_dir = f"{SCRATCH_DIR}SAE_Activations_Sparse/{dataset_name}"
        prefix = f"gemmascope_{sample_type}_sparse_layer34"
        n_expected_units = 16384
    
    # Check for chunk info
    chunk_info_path = os.path.join(acts_dir, f"{prefix}_chunks_info.json")
    
    if os.path.exists(chunk_info_path):
        with open(chunk_info_path, 'r') as f:
            chunk_info = json.load(f)
        n_chunks = chunk_info['num_chunks']
        total_samples = chunk_info['total_samples']
        n_features = chunk_info['n_features']
        topk = chunk_info['topk']
    else:
        # Single file
        single_file = os.path.join(acts_dir, f"{prefix}.pt")
        if not os.path.exists(single_file):
            raise FileNotFoundError(f"No sparse activation files found at {acts_dir}/{prefix}*")
        
        data = torch.load(single_file, map_location='cpu', weights_only=True)
        n_chunks = 1
        total_samples = data['indices'].shape[0]
        n_features = n_expected_units
        topk = data['topk']
        del data
    
    print(f"   Total samples: {total_samples:,}")
    print(f"   Number of features: {n_features:,}")
    print(f"   Top-K: {topk}")
    
    # Initialize statistics arrays on GPU for speed
    if DEVICE == 'cuda':
        unit_active_count = torch.zeros(n_features, dtype=torch.int64, device=DEVICE)
        unit_value_sum = torch.zeros(n_features, dtype=torch.float32, device=DEVICE)
        unit_value_max = torch.zeros(n_features, dtype=torch.float32, device=DEVICE)
    else:
        unit_active_count = np.zeros(n_features, dtype=np.int64)
        unit_value_sum = np.zeros(n_features, dtype=np.float32)
        unit_value_max = np.zeros(n_features, dtype=np.float32)
    
    # For percentile estimation
    n_top_values = 1000
    unit_top_values = defaultdict(list)
    
    # Track global sample index
    global_sample_idx = 0
    
    # Process chunks
    for chunk_idx in tqdm(range(n_chunks), desc="Processing sparse chunks"):
        if n_chunks > 1:
            chunk_file = os.path.join(acts_dir, f"{prefix}_chunk_{chunk_idx}.pt")
        else:
            chunk_file = os.path.join(acts_dir, f"{prefix}.pt")
        
        # Load sparse chunk
        chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=True)
        indices = chunk_data['indices']  # (n_samples, k)
        values = chunk_data['values']    # (n_samples, k)
        
        # Check which samples in this chunk are in calibration split
        chunk_samples = indices.shape[0]
        chunk_start_idx = global_sample_idx
        chunk_end_idx = global_sample_idx + chunk_samples
        
        # Calculate overlap with calibration range
        overlap_start = max(0, cal_start - chunk_start_idx)
        overlap_end = min(chunk_samples, cal_end - chunk_start_idx)
        
        # Skip chunk if no calibration samples
        if overlap_start >= overlap_end:
            global_sample_idx += chunk_samples
            continue
        
        # Extract only calibration samples from this chunk
        indices = indices[overlap_start:overlap_end]
        values = values[overlap_start:overlap_end]
        
        if DEVICE == 'cuda':
            # GPU processing - much faster!
            indices_gpu = indices.to(DEVICE)
            values_gpu = values.to(DEVICE)
            
            # Flatten for scatter operations
            flat_indices = indices_gpu.reshape(-1)
            flat_values = values_gpu.reshape(-1)
            
            # Remove zeros (padding)
            mask = flat_values > 0
            flat_indices = flat_indices[mask]
            flat_values = flat_values[mask]
            
            # Update counts
            ones = torch.ones_like(flat_indices, dtype=torch.int64)
            unit_active_count.scatter_add_(0, flat_indices, ones)
            
            # Update value sums
            unit_value_sum.scatter_add_(0, flat_indices, flat_values)
            
            # Update max values
            unit_value_max = torch.scatter_reduce(
                unit_value_max, 0, flat_indices, flat_values, 
                reduce='amax', include_self=True
            )
            
            # Collect for percentiles
            flat_indices_cpu = flat_indices.cpu().numpy()
            flat_values_cpu = flat_values.cpu().numpy()
            
        else:
            # CPU processing
            for i in range(indices.shape[0]):
                sample_indices = indices[i]
                sample_values = values[i]
                
                # Only process non-zero values
                mask = sample_values > 0
                if mask.any():
                    active_indices = sample_indices[mask]
                    active_values = sample_values[mask]
                    
                    # Update counts
                    np.add.at(unit_active_count, active_indices, 1)
                    
                    # Update sums
                    np.add.at(unit_value_sum, active_indices, active_values)
                    
                    # Update max
                    np.maximum.at(unit_value_max, active_indices, active_values)
            
            # Flatten for collecting top values
            flat_indices_cpu = indices[values > 0]
            flat_values_cpu = values[values > 0]
        
        # Collect top values per unit for percentile estimation
        for unit_id, value in zip(flat_indices_cpu, flat_values_cpu):
            unit_top_values[unit_id].append(float(value))
            if len(unit_top_values[unit_id]) > n_top_values * 2:
                unit_top_values[unit_id] = sorted(unit_top_values[unit_id], reverse=True)[:n_top_values]
        
        # Update global sample index
        global_sample_idx += chunk_samples
        
        del chunk_data
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # Move statistics back to CPU if needed
    if DEVICE == 'cuda':
        unit_active_count = unit_active_count.cpu().numpy()
        unit_value_sum = unit_value_sum.cpu().numpy()
        unit_value_max = unit_value_max.cpu().numpy()
    
    # Compute final statistics
    print("📈 Computing derived statistics...")
    final_stats = {}
    
    # Only process units that were active
    active_units = np.where(unit_active_count > 0)[0]
    print(f"   Active units: {len(active_units):,} / {n_features:,}")
    
    # Calculate total calibration samples
    cal_samples = cal_end - cal_start
    
    for unit_id in tqdm(active_units, desc="Finalizing stats"):
        active_count = unit_active_count[unit_id]
        
        # Basic statistics
        active_fraction = active_count / cal_samples
        
        # Compute percentiles from collected values
        if unit_id in unit_top_values and unit_top_values[unit_id]:
            values = sorted(unit_top_values[unit_id], reverse=True)
            n_vals = len(values)
            p95 = values[min(int(0.05 * n_vals), n_vals-1)]
            p99 = values[min(int(0.01 * n_vals), n_vals-1)]
        else:
            p95 = p99 = 0.0
        
        final_stats[unit_id] = {
            'active_count': int(active_count),
            'active_fraction': float(active_fraction),
            'max_value': float(unit_value_max[unit_id]),
            'avg_value': float(unit_value_sum[unit_id] / active_count),
            'p95': float(p95),
            'p99': float(p99),
        }
    
    print(f"✅ Computed statistics for {len(final_stats)} active units")
    
    return final_stats, cal_samples, n_features


def get_adaptive_pruning_config(dataset_name, model_type, sample_type, cal_samples):
    """Calculate adaptive pruning thresholds based on ground truth concept frequencies."""
    from utils.filter_datasets_utils import filter_concept_dict
    
    # Load GT calibration samples
    if model_type == 'CLIP':
        model_input_size = (224, 224)
    else:  # Gemma
        model_input_size = ('text', 'text2')
    
    if sample_type in ['patch', 'token']:
        gt_file = f"GT_Samples/{dataset_name}/gt_patch_per_concept_cal_inputsize_{model_input_size}.pt"
    else:  # cls
        gt_file = f"GT_Samples/{dataset_name}/gt_samples_per_concept_cal_inputsize_{model_input_size}.pt"
    
    try:
        gt_samples = torch.load(gt_file)
        gt_samples = filter_concept_dict(gt_samples, dataset_name)
        
        # Calculate concept frequencies
        concept_counts = {concept: len(indices) for concept, indices in gt_samples.items()}
        
        if not concept_counts:
            raise ValueError("No concepts found after filtering")
        
        min_count = min(concept_counts.values())
        max_count = max(concept_counts.values())
        n_concepts = len(concept_counts)
        
        # Calculate adaptive thresholds
        # Much more relaxed for SAE features which are inherently sparse
        if model_type == 'Gemma' and sample_type == 'patch':
            # For patch-level text, use very relaxed threshold
            # SAE features might only activate on 1-5% of positive samples
            dead_threshold = max(10, int(min_count * 0.01))  # 1% of rarest concept, min 10
        elif model_type == 'Gemma':
            dead_threshold = int(min_count * 0.1)  # 10% for CLS
        else:
            dead_threshold = int(min_count * 0.75)  # Original for vision
        ubiquitous_threshold = (max_count / cal_samples) * 1.25
        
        # Strength percentile based on concept count (option b)
        if n_concepts == 1:
            strength_percentile = 50
        elif n_concepts < 10:
            strength_percentile = 60
        elif n_concepts < 30:
            strength_percentile = 70
        else:
            strength_percentile = 80
        
        print(f"\n📊 Adaptive thresholds (based on {n_concepts} concepts):")
        print(f"   Rarest concept: {min_count:,} samples")
        print(f"   Most common: {max_count:,} samples")
        print(f"   → dead_threshold: {dead_threshold}")
        print(f"   → ubiquitous_threshold: {ubiquitous_threshold:.4f}")
        print(f"   → strength_percentile: {strength_percentile}")
        
        return {
            'dead_threshold': dead_threshold,
            'ubiquitous_threshold': ubiquitous_threshold,
            'strength_percentile': strength_percentile,
            'target_units': 10000  # Not used in adaptive mode
        }
        
    except Exception as e:
        print(f"\n⚠️  Could not compute adaptive thresholds: {e}")
        print(f"   Falling back to default config")
        return CLIP_PRUNING_CONFIG if model_type == 'CLIP' else GEMMA_PRUNING_CONFIG


def apply_pruning_criteria(unit_stats, cal_samples, model_type, dataset_name=None, sample_type='patch'):
    """Apply pruning criteria to filter units."""
    # Use adaptive config if dataset_name provided, otherwise fall back to fixed
    if dataset_name:
        config = get_adaptive_pruning_config(dataset_name, model_type, sample_type, cal_samples)
    else:
        config = CLIP_PRUNING_CONFIG if model_type == 'CLIP' else GEMMA_PRUNING_CONFIG
    
    print(f"\n🔪 Applying pruning criteria...")
    print(f"   Initial active units: {len(unit_stats):,}")
    
    # Start with all active units
    remaining_units = set(unit_stats.keys())
    
    # Step 1: Activity filters
    print("\n1️⃣ Activity filters...")
    dead_units = {u for u in remaining_units if unit_stats[u]['active_count'] < config['dead_threshold']}
    ubiquitous_units = {u for u in remaining_units if unit_stats[u]['active_fraction'] > config['ubiquitous_threshold']}
    
    removed = dead_units | ubiquitous_units
    remaining_units -= removed
    
    print(f"   Removed: {len(removed):,} ({len(dead_units)} too rare, {len(ubiquitous_units)} too common)")
    print(f"   Remaining: {len(remaining_units):,}")
    
    # Step 2: Strength filter
    print("\n2️⃣ Strength filter...")
    p99_values = [unit_stats[u]['p99'] for u in remaining_units]
    strength_threshold = np.percentile(p99_values, config['strength_percentile'])
    
    weak_units = {u for u in remaining_units if unit_stats[u]['p99'] < strength_threshold}
    remaining_units -= weak_units
    
    print(f"   Threshold: {strength_threshold:.4f} (p{config['strength_percentile']})")
    print(f"   Removed: {len(weak_units):,} weak units")
    print(f"   Remaining: {len(remaining_units):,}")
    
    kept_units = list(remaining_units)
    
    print(f"\n✅ Final unit count: {len(kept_units):,}")
    
    return kept_units


def save_filtered_sparse_activations(dataset_name, model_type, kept_units, sample_type='patch'):
    """
    Filter sparse activations to only include kept units.
    Applies the same kept_units (from calibration) to ALL splits.
    """
    print(f"\n💾 Filtering sparse activations for {dataset_name} - {model_type} - {sample_type}")
    print(f"   Applying calibration-based pruning to ALL splits")
    
    # Convert kept_units to set for fast lookup
    kept_units_set = set(kept_units)
    kept_units_list = sorted(kept_units)
    unit_id_to_new_idx = {unit_id: idx for idx, unit_id in enumerate(kept_units_list)}
    
    # Paths
    if model_type == 'CLIP':
        input_dir = f"{SCRATCH_DIR}SAE_Activations_Sparse/{dataset_name}"
        prefix = f"clipscope_{sample_type}_sparse_layer22"
        output_prefix = f"clipscope_{sample_type}_filtered"
    else:
        input_dir = f"{SCRATCH_DIR}SAE_Activations_Sparse/{dataset_name}"
        prefix = f"gemmascope_{sample_type}_sparse_layer34"
        output_prefix = f"gemmascope_{sample_type}_filtered"
    
    output_dir = f"{SCRATCH_DIR}SAE_Activations_Filtered/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load chunk info
    chunk_info_path = os.path.join(input_dir, f"{prefix}_chunks_info.json")
    if os.path.exists(chunk_info_path):
        with open(chunk_info_path, 'r') as f:
            chunk_info = json.load(f)
        n_chunks = chunk_info['num_chunks']
    else:
        n_chunks = 1
        chunk_info = {'chunks': [{'file': f"{prefix}.pt"}]}
    
    # New chunk info
    new_chunk_info = {
        'chunks': [],
        'total_samples': 0,
        'n_features': len(kept_units_list),
        'n_features_original': chunk_info.get('n_features', 65536 if model_type == 'CLIP' else 16384),
        'kept_units': kept_units_list,
        'format': 'sparse_filtered',
        'topk': chunk_info.get('topk', 32),
    }
    
    # Process each chunk
    for chunk_idx in tqdm(range(n_chunks), desc="Filtering chunks"):
        if n_chunks > 1:
            input_file = os.path.join(input_dir, f"{prefix}_chunk_{chunk_idx}.pt")
            output_file = os.path.join(output_dir, f"{output_prefix}_chunk_{chunk_idx}.pt")
        else:
            input_file = os.path.join(input_dir, f"{prefix}.pt")
            output_file = os.path.join(output_dir, f"{output_prefix}.pt")
        
        # Load sparse chunk
        chunk_data = torch.load(input_file, map_location='cpu', weights_only=True)
        indices = chunk_data['indices']
        values = chunk_data['values']
        
        # Filter to kept units using vectorized approach for speed
        n_samples, k = indices.shape
        n_features_original = chunk_info.get('n_features', 65536 if model_type == 'CLIP' else 16384)
        
        # Create lookup tensor for fast membership checking
        lookup = torch.zeros(n_features_original, dtype=torch.bool)
        lookup[kept_units] = True
        
        # Create remapping tensor
        remap = torch.full((n_features_original,), -1, dtype=torch.int32)
        for new_idx, old_idx in enumerate(kept_units):
            remap[old_idx] = new_idx
        
        # Check which indices to keep (vectorized)
        keep_masks = lookup[indices]  # Shape: [n_samples, k]
        
        # Prepare output tensors
        filtered_indices = torch.zeros_like(indices)
        filtered_values = torch.zeros_like(values)
        
        # Process samples in batches to balance speed and memory
        batch_size = 10000
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            
            for i in range(batch_start, batch_end):
                kept_mask = keep_masks[i]
                if kept_mask.any():
                    kept_idx = indices[i][kept_mask]
                    kept_val = values[i][kept_mask]
                    
                    # Remap indices
                    new_idx = remap[kept_idx]
                    
                    # Sort by value descending
                    sort_idx = torch.argsort(kept_val, descending=True)
                    new_idx = new_idx[sort_idx]
                    kept_val = kept_val[sort_idx]
                    
                    n_kept = len(new_idx)
                    filtered_indices[i, :n_kept] = new_idx
                    filtered_values[i, :n_kept] = kept_val
        
        # Save filtered data
        filtered_data = {
            'indices': filtered_indices,
            'values': filtered_values,
            'format': 'sparse_filtered',
            'topk': chunk_data.get('topk', indices.shape[1])
        }
        
        torch.save(filtered_data, output_file)
        
        # Update chunk info
        n_samples = filtered_indices.shape[0]
        if n_chunks > 1:
            new_chunk_info['chunks'].append({
                'file': os.path.basename(output_file),
                'start_idx': new_chunk_info['total_samples'],
                'end_idx': new_chunk_info['total_samples'] + n_samples,
                'n_samples': n_samples
            })
        new_chunk_info['total_samples'] += n_samples
        
        del chunk_data, filtered_data
        gc.collect()
    
    # Save chunk info
    if n_chunks > 1:
        new_chunk_info['num_chunks'] = n_chunks
        info_path = os.path.join(output_dir, f"{output_prefix}_chunks_info.json")
        with open(info_path, 'w') as f:
            json.dump(new_chunk_info, f, indent=2)
    
    # Save unit mapping
    mapping_path = os.path.join(output_dir, f"{output_prefix}_unit_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump({
            'kept_units': [int(x) for x in kept_units_list],
            'unit_id_to_new_idx': {int(k): int(v) for k, v in unit_id_to_new_idx.items()},
            'n_original': int(new_chunk_info['n_features_original']),
            'n_filtered': int(len(kept_units_list)),
        }, f, indent=2)
    
    print(f"✅ Filtered activations saved")
    print(f"   Reduced features: {new_chunk_info['n_features_original']:,} → {len(kept_units_list):,}")
    
    return output_dir


def convert_sparse_to_dense_chunks(dataset_name, model_type, sample_type='patch', 
                                  chunk_size_gb=2.0, use_filtered=True):
    """
    Convert sparse activations to dense format for downstream compatibility.
    """
    print(f"\n🔄 Converting sparse to dense for {dataset_name} - {model_type} - {sample_type}")
    
    # Determine paths
    if use_filtered:
        input_dir = f"{SCRATCH_DIR}SAE_Activations_Filtered/{dataset_name}"
        if model_type == 'CLIP':
            prefix = f"clipscope_{sample_type}_filtered"
            n_features_key = 'n_filtered'
        else:
            prefix = f"gemmascope_{sample_type}_filtered"
            n_features_key = 'n_filtered'
        
        # Load unit mapping
        mapping_path = os.path.join(input_dir, f"{prefix}_unit_mapping.json")
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        n_features = mapping[n_features_key]
    else:
        # For CLS samples without filtering
        input_dir = f"{SCRATCH_DIR}SAE_Activations_Sparse/{dataset_name}"
        if model_type == 'CLIP':
            prefix = f"clipscope_{sample_type}_sparse_layer22"
            n_features = 65536
        else:
            prefix = f"gemmascope_{sample_type}_sparse_layer34"
            n_features = 16384
    
    # Output directory
    output_dir = f"{SCRATCH_DIR}SAE_Activations_Dense/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load chunk info
    chunk_info_path = os.path.join(input_dir, f"{prefix}_chunks_info.json")
    if os.path.exists(chunk_info_path):
        with open(chunk_info_path, 'r') as f:
            chunk_info = json.load(f)
        n_chunks = chunk_info.get('num_chunks', 1)
        total_samples = chunk_info['total_samples']
    else:
        # Single file
        sparse_file = os.path.join(input_dir, f"{prefix}.pt")
        data = torch.load(sparse_file, map_location='cpu', weights_only=True)
        n_chunks = 1
        total_samples = data['indices'].shape[0]
        del data
    
    print(f"   Total samples: {total_samples:,}")
    print(f"   Feature dimension: {n_features:,}")
    
    # Calculate samples per dense chunk
    bytes_per_sample = n_features * 4  # float32
    samples_per_dense_chunk = int(chunk_size_gb * 1024 * 1024 * 1024 / bytes_per_sample)
    
    # Process and save as dense chunks
    current_dense_samples = []
    dense_chunk_idx = 0
    global_sample_idx = 0
    
    dense_chunk_info = {
        'chunks': [],
        'total_samples': 0,
        'feature_dim': n_features,
        'num_chunks': 0
    }
    
    output_prefix = f"{model_type.lower()}scope_{sample_type}_dense"
    
    for sparse_chunk_idx in tqdm(range(n_chunks), desc="Converting to dense"):
        # Load sparse chunk
        if n_chunks > 1:
            sparse_file = os.path.join(input_dir, f"{prefix}_chunk_{sparse_chunk_idx}.pt")
        else:
            sparse_file = os.path.join(input_dir, f"{prefix}.pt")
        
        sparse_data = torch.load(sparse_file, map_location='cpu', weights_only=True)
        indices = sparse_data['indices']
        values = sparse_data['values']
        
        # Convert batch of samples to dense (more memory efficient)
        batch_size = min(1000, indices.shape[0])  # Process in smaller batches for CLS
        
        for batch_start in range(0, indices.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, indices.shape[0])
            batch_indices = indices[batch_start:batch_end]
            batch_values = values[batch_start:batch_end]
            
            # Create dense batch
            dense_batch = torch.zeros(batch_end - batch_start, n_features, dtype=torch.float32)
            
            for i in range(batch_end - batch_start):
                sample_indices = batch_indices[i]
                sample_values = batch_values[i]
                
                # Only process non-zero values
                mask = sample_values > 0
                if mask.any():
                    dense_batch[i, sample_indices[mask]] = sample_values[mask]
            
            # Add to current samples
            for i in range(dense_batch.shape[0]):
                current_dense_samples.append(dense_batch[i])
            
            del dense_batch
            
            # Save chunk if ready
            if len(current_dense_samples) >= samples_per_dense_chunk:
                # Stack and save
                dense_chunk = torch.stack(current_dense_samples)
                chunk_file = os.path.join(output_dir, f"{output_prefix}_chunk_{dense_chunk_idx}.pt")
                torch.save(dense_chunk, chunk_file)
                
                # Update chunk info
                dense_chunk_info['chunks'].append({
                    'file': os.path.basename(chunk_file),
                    'start_idx': global_sample_idx,
                    'end_idx': global_sample_idx + len(current_dense_samples),
                    'n_samples': len(current_dense_samples)
                })
                
                global_sample_idx += len(current_dense_samples)
                dense_chunk_idx += 1
                current_dense_samples = []
                
                # Clear memory
                del dense_chunk
                gc.collect()
        
        del sparse_data
        gc.collect()
    
    # Save final chunk
    if current_dense_samples:
        dense_chunk = torch.stack(current_dense_samples)
        
        if dense_chunk_idx == 0:
            # Single chunk - save without chunk suffix
            chunk_file = os.path.join(output_dir, f"{output_prefix}.pt")
        else:
            chunk_file = os.path.join(output_dir, f"{output_prefix}_chunk_{dense_chunk_idx}.pt")
        
        torch.save(dense_chunk, chunk_file)
        
        dense_chunk_info['chunks'].append({
            'file': os.path.basename(chunk_file),
            'start_idx': global_sample_idx,
            'end_idx': global_sample_idx + len(current_dense_samples),
            'n_samples': len(current_dense_samples)
        })
        
        dense_chunk_idx += 1
    
    # Finalize chunk info
    dense_chunk_info['total_samples'] = sum(chunk['n_samples'] for chunk in dense_chunk_info['chunks'])
    dense_chunk_info['num_chunks'] = len(dense_chunk_info['chunks'])
    
    # Save chunk info if multiple chunks
    if dense_chunk_info['num_chunks'] > 1:
        info_path = os.path.join(output_dir, f"{output_prefix}_chunks_info.json")
        with open(info_path, 'w') as f:
            json.dump(dense_chunk_info, f, indent=2)
    
    print(f"✅ Dense activations saved: {dense_chunk_info['total_samples']} samples in {dense_chunk_info['num_chunks']} chunks")
    print(f"   Location: {output_dir}")
    
    return output_dir


def generate_summary_statistics(dataset_name, model_type, sample_type, dense_output_dir, n_features):
    """
    Generate summary statistics about the dense activations.
    """
    print(f"\n📈 Generating summary statistics for {dataset_name} - {model_type} - {sample_type}")
    
    # Load dense activations
    output_prefix = f"{model_type.lower()}scope_{sample_type}_dense"
    dense_file = os.path.join(dense_output_dir, f"{output_prefix}.pt")
    
    if not os.path.exists(dense_file):
        # Check for chunked version
        chunk_file = os.path.join(dense_output_dir, f"{output_prefix}_chunk_0.pt")
        if os.path.exists(chunk_file):
            dense_file = chunk_file
        else:
            print(f"   Warning: Dense file not found at {dense_file}")
            return
    
    # Load the data
    dense_activations = torch.load(dense_file, map_location='cpu')
    print(f"   Dense activation shape: {dense_activations.shape}")
    print(f"   Feature dimension: {n_features:,}")
    
    # Count non-zero activations per sample
    non_zero_mask = dense_activations > 0
    non_zero_per_sample = non_zero_mask.sum(dim=1)
    
    # Overall statistics
    total_samples = dense_activations.shape[0]
    samples_with_activations = (non_zero_per_sample > 0).sum().item()
    
    print(f"\n📊 Activation Statistics:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Samples with non-zero activations: {samples_with_activations:,} ({samples_with_activations/total_samples*100:.1f}%)")
    print(f"   Average activations per sample: {non_zero_per_sample.float().mean():.1f}")
    print(f"   Max activations per sample: {non_zero_per_sample.max().item()}")
    
    # Image/sentence level statistics
    if sample_type == 'patch' and model_type == 'CLIP':
        # For image patches, convert to image-level statistics
        patches_per_image = 256
        # Use the actual number of samples in this chunk
        n_samples_in_chunk = non_zero_per_sample.shape[0]
        n_images = n_samples_in_chunk // patches_per_image
        
        # Only process complete images
        n_complete_samples = n_images * patches_per_image
        if n_complete_samples < n_samples_in_chunk:
            print(f"   Note: Truncating {n_samples_in_chunk - n_complete_samples} patches to align with image boundaries")
            non_zero_per_sample = non_zero_per_sample[:n_complete_samples]
        
        # Reshape to image level
        patch_activations_per_image = non_zero_per_sample.view(n_images, patches_per_image)
        patches_with_activations_per_image = (patch_activations_per_image > 0).sum(dim=1)
        images_with_activations = (patches_with_activations_per_image > 0).sum().item()
        
        print(f"\n🖼️  Image-level Statistics:")
        print(f"   Total images: {n_images:,}")
        print(f"   Images with activated patches: {images_with_activations:,} ({images_with_activations/n_images*100:.1f}%)")
        print(f"   Average activated patches per image: {patches_with_activations_per_image.float().mean():.1f}")
        print(f"   Max activated patches per image: {patches_with_activations_per_image.max().item()}")
        
    elif sample_type == 'patch' and model_type == 'Gemma':
        # For tokens, convert to paragraph-level statistics using token counts
        # When running from Experiments/, GT_Samples is at the same level
        token_counts_file = f"GT_Samples/{dataset_name}/token_counts_inputsize_('text', 'text2').pt"
        if os.path.exists(token_counts_file):
            token_counts = torch.load(token_counts_file, weights_only=True)
            token_lengths = [sum(x) if isinstance(x, list) else x for x in token_counts]
            n_paragraphs = len(token_lengths)
            
            # Map tokens to paragraphs
            paragraph_activations = []
            patches_per_paragraph = []
            token_idx = 0
            
            for para_idx, n_tokens in enumerate(token_lengths):
                if token_idx + n_tokens > len(non_zero_per_sample):
                    # Skip incomplete paragraphs at the end
                    break
                paragraph_tokens = non_zero_per_sample[token_idx:token_idx + n_tokens]
                tokens_with_activations = (paragraph_tokens > 0).sum().item()
                paragraph_activations.append(tokens_with_activations)
                patches_per_paragraph.append(n_tokens)
                token_idx += n_tokens
            
            paragraph_activations = torch.tensor(paragraph_activations)
            patches_per_paragraph = torch.tensor(patches_per_paragraph)
            paragraphs_with_activations = (paragraph_activations > 0).sum().item()
            n_paragraphs_processed = len(paragraph_activations)
            
            print(f"\n📝 Paragraph-level Statistics:")
            print(f"   Total paragraphs: {n_paragraphs_processed:,}")
            print(f"   Paragraphs with activated patches: {paragraphs_with_activations:,} ({paragraphs_with_activations/n_paragraphs_processed*100:.1f}%)")
            print(f"   Average activated patches per paragraph: {paragraph_activations.float().mean():.1f}")
            print(f"   Max activated patches per paragraph: {paragraph_activations.max().item()}")
            print(f"   Average patches (tokens) per paragraph: {patches_per_paragraph.float().mean():.1f}")
        else:
            print(f"   Warning: Token counts file not found: {token_counts_file}")
    
    elif sample_type == 'cls':
        # For CLS, each sample is already at image/sentence level
        print(f"\n🎯 CLS Statistics:")
        if model_type == 'CLIP':
            print(f"   Images with CLS activations: {samples_with_activations:,} ({samples_with_activations/total_samples*100:.1f}%)")
        else:
            print(f"   Sentences with CLS activations: {samples_with_activations:,} ({samples_with_activations/total_samples*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Postprocess SAE activations: prune and convert to dense')
    parser.add_argument('--datasets', nargs='+', 
                       default=None,
                       help='Datasets to process (default: all datasets)')
    parser.add_argument('--models', nargs='+', default=None,
                       choices=['CLIP', 'Gemma'], help='Model types to process (default: all models)')
    parser.add_argument('--sample-types', nargs='+', default=None,
                       help='Sample types to process (default: patch, cls)')
    parser.add_argument('--clip-target', type=int, default=5000,
                       help='Target number of CLIP units to keep')
    parser.add_argument('--gemma-target', type=int, default=3000,
                       help='Target number of Gemma units to keep')
    
    args = parser.parse_args()
    
    # Define all available datasets
    IMAGE_DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces']
    TEXT_DATASETS = ['Sarcasm', 'iSarcasm', 'GoEmotions']
    ALL_DATASETS = IMAGE_DATASETS + TEXT_DATASETS
    
    # Set defaults if not specified
    if args.datasets is None:
        datasets_to_process = ALL_DATASETS
    else:
        datasets_to_process = args.datasets
        
    if args.models is None:
        models_to_process = ['CLIP', 'Gemma']
    else:
        models_to_process = args.models
        
    if args.sample_types is None:
        sample_types_to_process = ['patch', 'cls']
    else:
        sample_types_to_process = args.sample_types
    
    # Update targets
    CLIP_PRUNING_CONFIG['target_units'] = args.clip_target
    GEMMA_PRUNING_CONFIG['target_units'] = args.gemma_target
    
    print("🚀 Starting SAE activation postprocessing")
    print(f"   Datasets: {datasets_to_process}")
    print(f"   Models: {models_to_process}")
    print(f"   Sample types: {sample_types_to_process}")
    print(f"   CLIP target: {args.clip_target:,} units")
    print(f"   Gemma target: {args.gemma_target:,} units")
    print("   Note: CLS samples skip pruning automatically")
    
    for dataset in datasets_to_process:
        for model in models_to_process:
            # Skip invalid combinations
            if model == 'CLIP' and dataset in ['Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue
            if model == 'Gemma' and dataset not in ['Sarcasm', 'iSarcasm', 'GoEmotions']:
                continue
            
            # Determine sample types
            # Use patch for both image patches and text tokens
            sample_types = [s for s in sample_types_to_process if s in ['patch', 'cls']]
            
            for sample_type in sample_types:
                print(f"\n{'='*60}")
                print(f"Processing {dataset} - {model} - {sample_type}")
                
                try:
                    if sample_type != 'cls':
                        # Patch/token samples: do full pruning based on calibration data
                        
                        # Step 1: Compute statistics from calibration data only
                        unit_stats, cal_samples, n_features = compute_unit_statistics_sparse(
                            dataset, model, sample_type
                        )
                        
                        if unit_stats is None:
                            print("Skipping due to missing calibration data")
                            continue
                        
                        # Step 2: Apply pruning criteria
                        kept_units = apply_pruning_criteria(
                            unit_stats, cal_samples, model, dataset_name=dataset, sample_type=sample_type
                        )
                        
                        # Step 3: Filter sparse activations (applies to all splits)
                        output_dir = save_filtered_sparse_activations(
                            dataset, model, kept_units, sample_type
                        )
                        
                        # Step 4: Convert to dense format
                        dense_output_dir = convert_sparse_to_dense_chunks(
                            dataset, model, sample_type, use_filtered=True
                        )
                        
                        # Step 5: Generate summary statistics
                        generate_summary_statistics(dataset, model, sample_type, dense_output_dir, len(kept_units))
                        
                    else:
                        # CLS samples: only keep features active in calibration set
                        print("\n🎯 Processing CLS with calibration-active features only")
                        
                        # Step 1: Get active features from calibration data
                        unit_stats, cal_samples, n_features = compute_unit_statistics_sparse(
                            dataset, model, sample_type
                        )
                        
                        if unit_stats is None:
                            print("Skipping due to missing calibration data")
                            continue
                        
                        # For CLS: keep all features that were active at least once in cal set
                        active_units = sorted([unit_id for unit_id, stats in unit_stats.items() 
                                             if stats['active_count'] > 0])
                        
                        print(f"\n✅ Keeping {len(active_units)} features active in calibration set")
                        print(f"   (out of {n_features} total features)")
                        
                        # Step 2: Filter sparse activations to only active features
                        save_filtered_sparse_activations(dataset, model, active_units, sample_type)
                        
                        # Step 3: Convert filtered sparse to dense
                        dense_output_dir = convert_sparse_to_dense_chunks(
                            dataset, model, sample_type, use_filtered=True
                        )
                        
                        # Step 4: Generate summary statistics
                        generate_summary_statistics(dataset, model, sample_type, dense_output_dir, len(active_units))
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
    
    print(f"\n✨ Postprocessing complete!")


if __name__ == "__main__":
    main()