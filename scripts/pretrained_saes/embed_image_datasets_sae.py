"""
Script to compute embeddings specifically for SAE pipeline.
Extracts layer-22 residual tokens from CLIP without normalization.
Saves in the same format as regular embeddings but with SAE-specific naming.
"""

import sys
import os
import torch
import gc
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import pandas as pd

# Use OpenCLIP for the recommended checkpoint
import open_clip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.general_utils import load_images
from utils.embedding_utils import layer_to_percent

PERCENT_THRU_MODEL = 92  # Layer 22 out of 24 layers = ~92% (22/24)
# Verify the layer mapping
print("\n=== SAE Image Embedding Configuration ===")
print(f"Target: Layer 22 (0-indexed 21) out of 24 CLIP layers")
actual_percent = layer_to_percent(21, 24, zero_indexed=True, model_name="CLIP-SAE")
print(f"Using PERCENT_THRU_MODEL = {PERCENT_THRU_MODEL}% (should be ~{actual_percent:.1f}%)\n")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DATASET_NAMES = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces']
SCRATCH_DIR = ''

# SAE-specific configuration
MODEL_NAME = "ViT-L-14"
CHECKPOINT = "laion2b_s32b_b82k"
INPUT_SIZE = 224
BATCH_SIZE = 100
SAMPLES_PER_CHUNK = 50000
CHUNK_IF_LARGER_GB = 10

# Use CLIP_SAE as the model identifier for file naming
MODEL_IDENTIFIER = "CLIP_SAE"


def get_layer22_embeddings(model, images, preprocess, device):
    """
    Extract layer-22 residual tokens from CLIP model using hook.
    
    Returns:
        torch.Tensor: Shape [batch_size, 257, 1024] (CLS + 256 patches)
    """
    # Preprocess images
    image_tensors = torch.stack([preprocess(img) for img in images]).to(device)
    
    # Use hook to capture layer 22 output
    layer_output = []
    
    def hook_fn(module, input, output):
        layer_output.append(output)
    
    # Register hook on layer 22 (0-indexed, so layer 22 = index 22)
    handle = model.visual.transformer.resblocks[22].register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model.encode_image(image_tensors)
    
    # Remove hook
    handle.remove()
    
    # Get the captured output
    hidden_states = layer_output[0]
    
    return hidden_states




def save_raw_chunk(chunk_tensors, chunk_idx, chunks_dir, prefix='chunk'):
    """Save a raw chunk of embeddings."""
    chunk_tensor = torch.cat(chunk_tensors, dim=0)
    chunk_file = os.path.join(chunks_dir, f'{prefix}_{chunk_idx}.pt')
    torch.save(chunk_tensor, chunk_file)
    return chunk_file, chunk_tensor.shape


def compute_sae_embeddings(dataset_name, model, preprocess, device):
    """
    Compute raw SAE embeddings for a dataset (no normalization).
    Saves in the same chunked format as regular embeddings.
    """
    print(f"\n=== Processing {dataset_name} ===")
    
    # Load images - first return value is all images in correct order
    all_images, _, _ = load_images(
        dataset_name=dataset_name, 
        model_input_size=(INPUT_SIZE, INPUT_SIZE)
    )
    
    print(f"  Total images: {len(all_images)}")
    
    # Setup output directories - use same Embeddings folder as regular embeddings
    base_dir = f'{SCRATCH_DIR}Embeddings/{dataset_name}'
    os.makedirs(base_dir, exist_ok=True)
    
    # File naming: CLIP_SAE_[cls/patch]_embeddings_percentthrumodel_92.pt
    cls_file = f'{base_dir}/{MODEL_IDENTIFIER}_cls_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
    patch_file = f'{base_dir}/{MODEL_IDENTIFIER}_patch_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
    
    # Initialize chunk directories
    cls_chunks_dir = os.path.dirname(cls_file)
    patch_chunks_dir = os.path.dirname(patch_file)
    
    cls_chunk_info = {
        'chunks': [],
        'total_samples': 0,
        'embedding_dim': 1024,
        'num_chunks': 0
    }
    patch_chunk_info = {
        'chunks': [],
        'total_samples': 0,
        'embedding_dim': 1024,
        'num_chunks': 0
    }
    
    # Tracking for chunks
    cls_current_chunk_tensors = []
    patch_current_chunk_tensors = []
    cls_current_chunk_size = 0
    patch_current_chunk_size = 0
    cls_chunk_idx = 0
    patch_chunk_idx = 0
    total_cls_samples = 0
    total_patch_samples = 0
    
    print(f"\nComputing raw embeddings from layer 22...")
    
    # Process all images
    n_batches = (len(all_images) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {len(all_images)} total images in {n_batches} batches...")
    
    for batch_idx in tqdm(range(n_batches), desc="Computing embeddings"):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min((batch_idx + 1) * BATCH_SIZE, len(all_images))
        batch_images = all_images[batch_start:batch_end]
        actual_batch_size = len(batch_images)
        
        # Get real layer-22 embeddings (raw, no normalization)
        embeddings = get_layer22_embeddings(model, batch_images, preprocess, device)
        
        # Split CLS and patch tokens
        cls_embeddings = embeddings[:, 0, :]  # [batch_size, 1024]
        patch_embeddings = embeddings[:, 1:, :]  # [batch_size, 256, 1024]
        
        # Flatten patches
        patch_embeddings_flat = patch_embeddings.reshape(-1, patch_embeddings.shape[-1])  # [batch_size * 256, 1024]
        
        # Add to current chunks
        cls_current_chunk_tensors.append(cls_embeddings.cpu())
        patch_current_chunk_tensors.append(patch_embeddings_flat.cpu())
        
        cls_current_chunk_size += actual_batch_size
        patch_current_chunk_size += patch_embeddings_flat.shape[0]
        
        total_cls_samples += actual_batch_size
        total_patch_samples += patch_embeddings_flat.shape[0]
        
        # Save CLS chunk if ready
        is_last_batch = (batch_idx == n_batches - 1)
        if cls_current_chunk_size >= SAMPLES_PER_CHUNK or is_last_batch:
            chunk_file, chunk_shape = save_raw_chunk(
                cls_current_chunk_tensors, cls_chunk_idx, cls_chunks_dir,
                f'{MODEL_IDENTIFIER}_cls_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}_chunk'
            )
            
            cls_chunk_info['chunks'].append({
                'file': os.path.basename(chunk_file),
                'start_idx': total_cls_samples - cls_current_chunk_size,
                'end_idx': total_cls_samples,
                'shape': list(chunk_shape)
            })
            
            cls_current_chunk_tensors = []
            cls_current_chunk_size = 0
            cls_chunk_idx += 1
        
        # Save patch chunk if ready
        if patch_current_chunk_size >= SAMPLES_PER_CHUNK or is_last_batch:
            chunk_file, chunk_shape = save_raw_chunk(
                patch_current_chunk_tensors, patch_chunk_idx, patch_chunks_dir,
                f'{MODEL_IDENTIFIER}_patch_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}_chunk'
            )
            
            patch_chunk_info['chunks'].append({
                'file': os.path.basename(chunk_file),
                'start_idx': total_patch_samples - patch_current_chunk_size,
                'end_idx': total_patch_samples,
                'shape': list(chunk_shape)
            })
            
            patch_current_chunk_tensors = []
            patch_current_chunk_size = 0
            patch_chunk_idx += 1
        
        # Clear GPU cache
        del embeddings, cls_embeddings, patch_embeddings, patch_embeddings_flat
        torch.cuda.empty_cache()
        gc.collect()
    
    # Update final chunk info
    cls_chunk_info['total_samples'] = total_cls_samples
    cls_chunk_info['num_chunks'] = cls_chunk_idx
    
    patch_chunk_info['total_samples'] = total_patch_samples
    patch_chunk_info['num_chunks'] = patch_chunk_idx
    
    # Save chunk info files
    cls_info_file = cls_file.replace('.pt', '_chunks_info.json')
    patch_info_file = patch_file.replace('.pt', '_chunks_info.json')
    
    with open(cls_info_file, 'w') as f:
        json.dump(cls_chunk_info, f, indent=2)
    
    with open(patch_info_file, 'w') as f:
        json.dump(patch_chunk_info, f, indent=2)
    
    print(f"✅ CLS embeddings saved: {total_cls_samples} samples in {cls_chunk_idx} chunks")
    print(f"   Location: {cls_file.replace('.pt', '_chunk_*.pt') if cls_chunk_idx > 1 else cls_file}")
    print(f"✅ Patch embeddings saved: {total_patch_samples} samples in {patch_chunk_idx} chunks")
    print(f"   Location: {patch_file.replace('.pt', '_chunk_*.pt') if patch_chunk_idx > 1 else patch_file}")
    
    return cls_chunk_info, patch_chunk_info


def main():
    """Main function to process all datasets."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute SAE image embeddings for specified datasets')
    parser.add_argument('--datasets', nargs='+', 
                        choices=DEFAULT_DATASET_NAMES,
                        default=DEFAULT_DATASET_NAMES,
                        help='Datasets to process (default: all datasets)')
    args = parser.parse_args()
    
    dataset_names = args.datasets
    
    print("🚀 Starting SAE embedding extraction")
    print(f"   Model: OpenCLIP {MODEL_NAME} ({CHECKPOINT})")
    print(f"   Layer: 22 (residual tokens)")
    print(f"   Device: {DEVICE}")
    print(f"   Datasets: {dataset_names}")
    
    # Load OpenCLIP model
    print("\n📦 Loading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, 
        pretrained=CHECKPOINT,
        device=DEVICE
    )
    model.eval()
    print("✓ Model loaded successfully")
    
    # Process each dataset
    for dataset_name in dataset_names:
        try:
            compute_sae_embeddings(dataset_name, model, preprocess, DEVICE)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n✨ SAE embedding extraction complete!")


if __name__ == "__main__":
    main()