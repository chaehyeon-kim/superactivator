"""
Compute SAE activations for both image and text datasets.
This version saves activations in sparse format to save space and computation.
Processes embeddings through pretrained SAEs (CLIP-Scope for images, Gemma Scope for text).
"""

import torch
import torch.nn.functional as F
import sys
import os
import gc
import json
import argparse
from tqdm import tqdm
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.memory_management_utils import ChunkedEmbeddingLoader

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRATCH_DIR = ''

# Default datasets
IMAGE_DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces']
TEXT_DATASETS = ['Sarcasm', 'iSarcasm', 'GoEmotions']

# SAE model configurations
CLIP_SAE_CONFIG = {
    'percent_thru_model': 92,
    'embedding_dim': 1024,
    'n_features': 65536,
    'topk': 32,
    'model_identifier': 'CLIP_SAE'
}

GEMMA_SAE_CONFIG = {
    'percent_thru_model': 81,
    'embedding_dim': 3584,  # Gemma-9b embedding dimension
    'n_features': 16384,  # Using 16k width
    'model_identifier': 'Gemma_SAE'
}


def process_clip_sae_sparse(dataset_name, sample_type='patch', batch_size=131072):
    """
    Process CLIP embeddings through CLIP-Scope SAE and save as sparse format.
    
    Args:
        dataset_name: Name of the dataset
        sample_type: 'patch' or 'cls'
        batch_size: Batch size for processing
    """
    print(f"\n=== Processing {dataset_name} - {sample_type} with CLIP-Scope SAE (SPARSE) ===")
    
    # Load CLIP-Scope SAE
    try:
        from clipscope import TopKSAE
    except ImportError:
        print("Error: clipscope not installed. Run: pip install clipscope pillow")
        return
    
    # Load SAE checkpoint for layer 22 residual
    sae = TopKSAE.from_pretrained(checkpoint="22_resid/1200013184.pt", device=DEVICE)
    sae.eval()
    print(f"✓ Loaded CLIP-Scope SAE (Top-K={CLIP_SAE_CONFIG['topk']}, features={CLIP_SAE_CONFIG['n_features']})")
    
    # Setup paths
    embeddings_dir = f"{SCRATCH_DIR}Embeddings/{dataset_name}"
    embeddings_file = f"{CLIP_SAE_CONFIG['model_identifier']}_{sample_type}_embeddings_percentthrumodel_{CLIP_SAE_CONFIG['percent_thru_model']}.pt"
    embeddings_path = os.path.join(embeddings_dir, embeddings_file)
    
    # Output paths - using 'sparse' suffix
    output_dir = f"{SCRATCH_DIR}SAE_Activations_Sparse/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"clipscope_{sample_type}_sparse_layer22.pt"
    output_path = os.path.join(output_dir, output_file)
    
    # Load embeddings using ChunkedEmbeddingLoader
    print(f"Loading embeddings from: {embeddings_path}")
    loader = ChunkedEmbeddingLoader(
        dataset_name=dataset_name,
        embeddings_file=embeddings_file,
        scratch_dir=SCRATCH_DIR,
        device='cpu'  # Load to CPU first, move to GPU in batches
    )
    total_samples = loader.total_samples
    
    # Note: The embeddings now contain train (placeholders), cal, and test data
    # in that order. We'll process all of them through the SAE
    
    print(f"Total {sample_type} samples: {total_samples}")
    print(f"Processing in batches of {batch_size}...")
    
    # Process in chunks and save as sparse format
    chunk_size_gb = 2  # Smaller chunks since sparse format is more efficient
    # Estimate: 32 features * 4 bytes each = 128 bytes per sample
    samples_per_chunk = int(chunk_size_gb * 1024 * 1024 * 1024 / 128)
    
    current_indices = []
    current_values = []
    current_chunk_size = 0
    chunk_idx = 0
    chunk_info = {
        'chunks': [],
        'total_samples': 0,
        'n_features': CLIP_SAE_CONFIG['n_features'],
        'topk': CLIP_SAE_CONFIG['topk'],
        'format': 'sparse_topk'
    }
    
    with torch.no_grad():
        # Process embeddings chunk by chunk using the loader's iterator
        global_idx = 0
        for embed_chunk, start_idx, end_idx in loader.iter_chunks():
            print(f"Processing embedding chunk: rows {start_idx}-{end_idx}")
            
            # Process this chunk in batches
            for batch_start in tqdm(range(0, embed_chunk.shape[0], batch_size), 
                                  desc=f"Computing SAE activations (chunk {start_idx}-{end_idx})"):
                batch_end = min(batch_start + batch_size, embed_chunk.shape[0])
                batch_embeddings = embed_chunk[batch_start:batch_end].to(DEVICE)
                
                # Ensure correct shape
                assert batch_embeddings.shape[1] == CLIP_SAE_CONFIG['embedding_dim'], \
                    f"Expected embedding dim {CLIP_SAE_CONFIG['embedding_dim']}, got {batch_embeddings.shape[1]}"
                
                # Forward through SAE
                out = sae.forward_verbose(batch_embeddings)
                latents = out["latent"]  # (batch_size, 65536) - sparse with TopK=32 nonzeros
                
                # Extract Top-K values and indices for sparse storage
                # Since SAE already enforces sparsity, we just need to find the non-zeros
                # But to be safe and consistent, we'll use topk
                topk_values, topk_indices = torch.topk(latents, k=CLIP_SAE_CONFIG['topk'], dim=1)
                
                # Move to CPU and store
                current_indices.append(topk_indices.cpu())
                current_values.append(topk_values.cpu())
                current_chunk_size += topk_indices.shape[0]
                
                # Save chunk if it's large enough
                if current_chunk_size >= samples_per_chunk and global_idx + batch_embeddings.shape[0] < total_samples:
                    # Stack all indices and values
                    chunk_indices = torch.cat(current_indices, dim=0)
                    chunk_values = torch.cat(current_values, dim=0)
                    
                    # Save as sparse format
                    chunk_file = output_path.replace('.pt', f'_chunk_{chunk_idx}.pt')
                    torch.save({
                        'indices': chunk_indices,
                        'values': chunk_values,
                        'format': 'sparse_topk',
                        'topk': CLIP_SAE_CONFIG['topk']
                    }, chunk_file)
                    
                    chunk_info['chunks'].append({
                        'file': os.path.basename(chunk_file),
                        'start_idx': global_idx - current_chunk_size + batch_embeddings.shape[0],
                        'end_idx': global_idx,
                        'n_samples': chunk_indices.shape[0]
                    })
                    
                    print(f"  Saved sparse chunk {chunk_idx}: {chunk_indices.shape[0]} samples")
                    chunk_idx += 1
                    current_indices = []
                    current_values = []
                    current_chunk_size = 0
                    
                    # Clear memory
                    del chunk_indices, chunk_values
                    gc.collect()
                    torch.cuda.empty_cache()
                
                global_idx += batch_embeddings.shape[0]
                
                # Clear batch memory
                del batch_embeddings, latents, topk_values, topk_indices
                torch.cuda.empty_cache()
    
    # Save final chunk
    if current_indices:
        chunk_indices = torch.cat(current_indices, dim=0)
        chunk_values = torch.cat(current_values, dim=0)
        
        if chunk_idx == 0:
            # Only one chunk - save directly
            torch.save({
                'indices': chunk_indices,
                'values': chunk_values,
                'format': 'sparse_topk',
                'topk': CLIP_SAE_CONFIG['topk']
            }, output_path)
            print(f"✅ Saved sparse SAE activations: {chunk_indices.shape} to {output_path}")
            
            # Compute and print unique features
            unique_features = set()
            mask = chunk_values > 0
            active_features = chunk_indices[mask].numpy()
            unique_features.update(active_features.tolist())
            
            print(f"\n🔍 Unique Feature Summary:")
            print(f"   Total possible features: {CLIP_SAE_CONFIG['n_features']:,}")
            print(f"   Unique active features: {len(unique_features):,}")
            print(f"   Coverage: {len(unique_features)/CLIP_SAE_CONFIG['n_features']*100:.1f}%")
            print(f"   Sparsity: {(1 - len(unique_features)/CLIP_SAE_CONFIG['n_features'])*100:.1f}%")
        else:
            # Multiple chunks - save last chunk
            chunk_file = output_path.replace('.pt', f'_chunk_{chunk_idx}.pt')
            torch.save({
                'indices': chunk_indices,
                'values': chunk_values,
                'format': 'sparse_topk',
                'topk': CLIP_SAE_CONFIG['topk']
            }, chunk_file)
            
            chunk_info['chunks'].append({
                'file': os.path.basename(chunk_file),
                'start_idx': total_samples - chunk_indices.shape[0],
                'end_idx': total_samples,
                'n_samples': chunk_indices.shape[0]
            })
            
            # Save chunk info
            chunk_info['total_samples'] = total_samples
            chunk_info['num_chunks'] = chunk_idx + 1
            info_file = output_path.replace('.pt', '_chunks_info.json')
            with open(info_file, 'w') as f:
                json.dump(chunk_info, f, indent=2)
            
            print(f"✅ Saved {chunk_idx + 1} sparse chunks for {total_samples} total samples")
            print(f"   Chunk info: {info_file}")
            
            # Compute and print unique features statistics
            print("\n📊 Computing unique feature statistics...")
            unique_features = set()
            
            # Re-read chunks to count unique features
            for i in range(chunk_idx + 1):
                chunk_file = output_path.replace('.pt', f'_chunk_{i}.pt')
                chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=True)
                indices = chunk_data['indices']
                values = chunk_data['values']
                
                # Find all non-zero features
                mask = values > 0
                active_features = indices[mask].numpy()
                unique_features.update(active_features.tolist())
                
                del chunk_data
            
            print(f"\n🔍 Unique Feature Summary:")
            print(f"   Total possible features: {CLIP_SAE_CONFIG['n_features']:,}")
            print(f"   Unique active features: {len(unique_features):,}")
            print(f"   Coverage: {len(unique_features)/CLIP_SAE_CONFIG['n_features']*100:.1f}%")
            print(f"   Sparsity: {(1 - len(unique_features)/CLIP_SAE_CONFIG['n_features'])*100:.1f}%")


def process_gemma_sae_sparse(dataset_name, sample_type='patch', batch_size=262144):
    """
    Process Gemma embeddings through Gemma Scope SAE and save as sparse format.
    
    Args:
        dataset_name: Name of the dataset
        sample_type: 'patch' (for token embeddings) or 'cls'
        batch_size: Batch size for processing
    """
    print(f"\n=== Processing {dataset_name} - {sample_type} with Gemma Scope SAE (SPARSE) ===")
    
    # Load Gemma Scope SAE
    try:
        from sae_lens import SAE
    except ImportError:
        print("Error: sae-lens not installed. Run: pip install sae-lens")
        return
    
    # Load SAE for layer 34 
    sae = SAE.from_pretrained(
        release="gemma-scope-9b-pt-res-canonical",
        sae_id="layer_34/width_16k/canonical",
        device=DEVICE,
    )
    sae.eval()
    
    # Get actual feature dimension from SAE
    n_features = sae.W_dec.shape[0]
    print(f"✓ Loaded Gemma Scope SAE (features={n_features})")
    
    # Update config with actual feature size
    GEMMA_SAE_CONFIG['n_features'] = n_features
    
    # Setup paths
    embeddings_dir = f"{SCRATCH_DIR}Embeddings/{dataset_name}"
    # Use sample_type directly (patch for tokens, cls for cls)
    embeddings_file = f"{GEMMA_SAE_CONFIG['model_identifier']}_{sample_type}_embeddings_percentthrumodel_{GEMMA_SAE_CONFIG['percent_thru_model']}.pt"
    embeddings_path = os.path.join(embeddings_dir, embeddings_file)
    
    # Output paths
    output_dir = f"{SCRATCH_DIR}SAE_Activations_Sparse/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"gemmascope_{sample_type}_sparse_layer34.pt"
    output_path = os.path.join(output_dir, output_file)
    
    # Load embeddings
    print(f"Loading embeddings from: {embeddings_path}")
    loader = ChunkedEmbeddingLoader(
        dataset_name=dataset_name,
        embeddings_file=embeddings_file,
        scratch_dir=SCRATCH_DIR,
        device='cpu'
    )
    total_samples = loader.total_samples
    
    # Note: The embeddings now contain train (placeholders), cal, and test data
    # in that order. We'll process all of them through the SAE
    
    print(f"Total {sample_type} samples: {total_samples}")
    print(f"Processing in batches of {batch_size}...")
    
    # For Gemma, we'll select top-K from the SAE output
    # Using k=64 for text (can be adjusted)
    topk = 64
    
    # Process and save
    chunk_size_gb = 2
    samples_per_chunk = int(chunk_size_gb * 1024 * 1024 * 1024 / (topk * 8))  # 8 bytes per entry
    
    current_indices = []
    current_values = []
    current_chunk_size = 0
    chunk_idx = 0
    chunk_info = {
        'chunks': [],
        'total_samples': 0,
        'n_features': n_features,
        'topk': topk,
        'format': 'sparse_topk'
    }
    
    with torch.no_grad():
        global_idx = 0
        for embed_chunk, start_idx, end_idx in loader.iter_chunks():
            print(f"Processing embedding chunk: rows {start_idx}-{end_idx}")
            
            for batch_start in tqdm(range(0, embed_chunk.shape[0], batch_size), 
                                  desc=f"Computing SAE activations (chunk {start_idx}-{end_idx})"):
                batch_end = min(batch_start + batch_size, embed_chunk.shape[0])
                batch_embeddings = embed_chunk[batch_start:batch_end].to(DEVICE)
                
                # Forward through SAE using SAELens API
                latents = sae.encode(batch_embeddings)  # (batch_size, n_features)
                
                # Extract Top-K for sparse storage
                topk_values, topk_indices = torch.topk(latents, k=topk, dim=1)
                
                # Move to CPU and store
                current_indices.append(topk_indices.cpu())
                current_values.append(topk_values.cpu())
                current_chunk_size += topk_indices.shape[0]
                
                # Save chunk if needed
                if current_chunk_size >= samples_per_chunk and global_idx + batch_embeddings.shape[0] < total_samples:
                    chunk_indices = torch.cat(current_indices, dim=0)
                    chunk_values = torch.cat(current_values, dim=0)
                    
                    chunk_file = output_path.replace('.pt', f'_chunk_{chunk_idx}.pt')
                    torch.save({
                        'indices': chunk_indices,
                        'values': chunk_values,
                        'format': 'sparse_topk',
                        'topk': topk
                    }, chunk_file)
                    
                    chunk_info['chunks'].append({
                        'file': os.path.basename(chunk_file),
                        'start_idx': global_idx - current_chunk_size + batch_embeddings.shape[0],
                        'end_idx': global_idx,
                        'n_samples': chunk_indices.shape[0]
                    })
                    
                    print(f"  Saved sparse chunk {chunk_idx}: {chunk_indices.shape[0]} samples")
                    chunk_idx += 1
                    current_indices = []
                    current_values = []
                    current_chunk_size = 0
                    
                    del chunk_indices, chunk_values
                    gc.collect()
                    torch.cuda.empty_cache()
                
                global_idx += batch_embeddings.shape[0]
                del batch_embeddings, latents, topk_values, topk_indices
                torch.cuda.empty_cache()
    
    # Save final chunk
    if current_indices:
        chunk_indices = torch.cat(current_indices, dim=0)
        chunk_values = torch.cat(current_values, dim=0)
        
        if chunk_idx == 0:
            torch.save({
                'indices': chunk_indices,
                'values': chunk_values,
                'format': 'sparse_topk',
                'topk': topk
            }, output_path)
            print(f"✅ Saved sparse SAE activations: {chunk_indices.shape} to {output_path}")
            
            # Compute and print unique features
            unique_features = set()
            mask = chunk_values > 0
            active_features = chunk_indices[mask].numpy()
            unique_features.update(active_features.tolist())
            
            print(f"\n🔍 Unique Feature Summary:")
            print(f"   Total possible features: {n_features:,}")
            print(f"   Unique active features: {len(unique_features):,}")
            print(f"   Coverage: {len(unique_features)/n_features*100:.1f}%")
            print(f"   Sparsity: {(1 - len(unique_features)/n_features)*100:.1f}%")
        else:
            chunk_file = output_path.replace('.pt', f'_chunk_{chunk_idx}.pt')
            torch.save({
                'indices': chunk_indices,
                'values': chunk_values,
                'format': 'sparse_topk',
                'topk': topk
            }, chunk_file)
            
            chunk_info['chunks'].append({
                'file': os.path.basename(chunk_file),
                'start_idx': total_samples - chunk_indices.shape[0],
                'end_idx': total_samples,
                'n_samples': chunk_indices.shape[0]
            })
            
            chunk_info['total_samples'] = total_samples
            chunk_info['num_chunks'] = chunk_idx + 1
            info_file = output_path.replace('.pt', '_chunks_info.json')
            with open(info_file, 'w') as f:
                json.dump(chunk_info, f, indent=2)
            
            print(f"✅ Saved {chunk_idx + 1} sparse chunks for {total_samples} total samples")
            
            # Compute and print unique features statistics
            print("\n📊 Computing unique feature statistics...")
            unique_features = set()
            
            # Re-read chunks to count unique features
            for i in range(chunk_idx + 1):
                chunk_file = output_path.replace('.pt', f'_chunk_{i}.pt')
                chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=True)
                indices = chunk_data['indices']
                values = chunk_data['values']
                
                # Find all non-zero features
                mask = values > 0
                active_features = indices[mask].numpy()
                unique_features.update(active_features.tolist())
                
                del chunk_data
            
            print(f"\n🔍 Unique Feature Summary:")
            print(f"   Total possible features: {n_features:,}")
            print(f"   Unique active features: {len(unique_features):,}")
            print(f"   Coverage: {len(unique_features)/n_features*100:.1f}%")
            print(f"   Sparsity: {(1 - len(unique_features)/n_features)*100:.1f}%")


def main():
    """Main function to process SAE activations in sparse format."""
    parser = argparse.ArgumentParser(description='Compute sparse SAE activations for image and text datasets')
    parser.add_argument('--datasets', nargs='+', 
                        help='Datasets to process (default: all datasets)')
    parser.add_argument('--sample-type', choices=['patch', 'cls', 'token', 'all'], default='all',
                        help='Sample type to process')
    parser.add_argument('--batch-size', type=int, 
                        help='Batch size for processing')
    args = parser.parse_args()
    
    # Determine which datasets to process
    if args.datasets:
        # Validate dataset names
        all_datasets = IMAGE_DATASETS + TEXT_DATASETS
        invalid_datasets = [d for d in args.datasets if d not in all_datasets]
        if invalid_datasets:
            print(f"Error: Invalid datasets: {invalid_datasets}")
            print(f"Valid datasets: {all_datasets}")
            sys.exit(1)
        
        image_datasets = [d for d in args.datasets if d in IMAGE_DATASETS]
        text_datasets = [d for d in args.datasets if d in TEXT_DATASETS]
    else:
        # Default: process all datasets
        image_datasets = IMAGE_DATASETS
        text_datasets = TEXT_DATASETS
    
    print("🚀 Starting SPARSE SAE activation computation")
    print(f"   Image datasets: {image_datasets}")
    print(f"   Text datasets: {text_datasets}")
    
    # Process image datasets with CLIP-Scope
    if image_datasets:
        batch_size = args.batch_size or 131072  # Default for CLIP-Scope
        
        for dataset_name in image_datasets:
            if args.sample_type in ['patch', 'all']:
                try:
                    process_clip_sae_sparse(dataset_name, sample_type='patch', batch_size=batch_size)
                except Exception as e:
                    print(f"❌ Error processing {dataset_name} patches: {e}")
                    import traceback
                    traceback.print_exc()
            
            if args.sample_type in ['cls', 'all']:
                try:
                    process_clip_sae_sparse(dataset_name, sample_type='cls', batch_size=batch_size)
                except Exception as e:
                    print(f"❌ Error processing {dataset_name} cls: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Clear GPU memory between datasets
            gc.collect()
            torch.cuda.empty_cache()
    
    # Process text datasets with Gemma Scope
    if text_datasets:
        batch_size = args.batch_size or 262144  # Default for Gemma Scope
        
        for dataset_name in text_datasets:
            if args.sample_type in ['patch', 'token', 'all']:
                try:
                    process_gemma_sae_sparse(dataset_name, sample_type='patch', batch_size=batch_size)
                except Exception as e:
                    print(f"❌ Error processing {dataset_name} tokens: {e}")
                    import traceback
                    traceback.print_exc()
            
            if args.sample_type in ['cls', 'all']:
                try:
                    process_gemma_sae_sparse(dataset_name, sample_type='cls', batch_size=batch_size)
                except Exception as e:
                    print(f"❌ Error processing {dataset_name} cls: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Clear GPU memory between datasets
            gc.collect()
            torch.cuda.empty_cache()
    
    print("\n✨ SPARSE SAE activation computation complete!")
    print(f"   Outputs saved to: {SCRATCH_DIR}SAE_Activations_Sparse/")


if __name__ == "__main__":
    main()