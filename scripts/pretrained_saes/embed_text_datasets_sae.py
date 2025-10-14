"""
Script to compute text embeddings specifically for SAE pipeline using Gemma-2-9B.
Extracts layer-34 residual tokens from Gemma-2 without normalization.
Saves in the same format as regular embeddings but with SAE-specific naming.
"""

import sys
import os
import torch
import gc
import json
from tqdm import tqdm
import numpy as np
import argparse

# Use transformers for Gemma-2
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.general_utils import load_text
from utils.embedding_utils import layer_to_percent

PERCENT_THRU_MODEL = 81  # Layer 34 out of 42 layers = ~81% (34/42)
# Verify the layer mapping
print("\n=== SAE Text Embedding Configuration ===")
print(f"Target: Layer 34 (0-indexed 33) out of 42 Gemma-2 layers")
actual_percent = layer_to_percent(33, 42, zero_indexed=True, model_name="Gemma-SAE")
print(f"Using PERCENT_THRU_MODEL = {PERCENT_THRU_MODEL}% (should be ~{actual_percent:.1f}%)\n")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DATASET_NAMES = ['Sarcasm', 'iSarcasm', 'GoEmotions']
SCRATCH_DIR = ''

# SAE-specific configuration for Gemma-2-9B
MODEL_NAME = "google/gemma-2-9b"
LAYER_INDEX = 34  # Target layer (0-indexed: block 34)
HIDDEN_SIZE = 3584
BATCH_SIZE = 32  # Smaller batch size for large model
SAMPLES_PER_CHUNK = 50000
CHUNK_IF_LARGER_GB = 10

# Use Gemma_SAE as the model identifier for file naming
MODEL_IDENTIFIER = "Gemma_SAE"


def get_layer34_embeddings(model, tokenizer, texts, device):
    """
    Extract layer-34 residual tokens from Gemma-2 model.
    
    Args:
        model: Gemma model
        tokenizer: Gemma tokenizer
        texts: List of text strings
        device: Device for computation
    
    Returns:
        tuple: (cls_embeddings, token_embeddings, token_to_text_map)
            - cls_embeddings: Mean-pooled embeddings across all valid tokens
            - token_embeddings: All individual token embeddings
            - token_to_text_map: Mapping from token index to (text_index, position)
    """
    # Tokenize texts (no special tokens to match SAE training and avoid BOS outliers)
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,  # Standard max length for most datasets
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=False
    ).to(device)
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
    
    # Extract layer-34 residual tokens (index 35 in hidden_states)
    # hidden_states[0] = embeddings, hidden_states[1-42] = after blocks 0-41
    # So layer 34 (after block 34) is at index 35
    layer34_hidden = outputs.hidden_states[35]  # [batch_size, seq_len, 3584]
    
    # Extract all token embeddings, masking out padding
    batch_size, seq_len, hidden_dim = layer34_hidden.shape
    attention_mask = tokenized['attention_mask']
    
    # Compute mean-pooled CLS embeddings and collect all tokens
    cls_embeddings_list = []
    all_tokens = []
    token_to_text_map = []
    
    for batch_idx in range(batch_size):
        # Get valid (non-padded) positions
        valid_positions = attention_mask[batch_idx].bool()
        valid_tokens = layer34_hidden[batch_idx][valid_positions]  # [valid_seq_len, 3584]
        
        # Compute mean pooling for CLS embedding (mean across all valid tokens)
        cls_embedding = valid_tokens.mean(dim=0)  # [3584]
        cls_embeddings_list.append(cls_embedding)
        
        # Add tokens to collection
        all_tokens.append(valid_tokens)
        
        # Create mapping from token index to (text_index, token_position)
        for pos_idx in range(valid_tokens.shape[0]):
            token_to_text_map.append((batch_idx, pos_idx))
    
    # Stack CLS embeddings
    cls_embeddings = torch.stack(cls_embeddings_list, dim=0)  # [batch_size, 3584]
    
    # Concatenate all valid tokens
    token_embeddings = torch.cat(all_tokens, dim=0)  # [total_valid_tokens, 3584]
    
    return cls_embeddings, token_embeddings, token_to_text_map


def save_raw_chunk(chunk_tensors, chunk_idx, chunks_dir, prefix='chunk'):
    """Save a raw chunk of embeddings."""
    chunk_tensor = torch.cat(chunk_tensors, dim=0)
    chunk_file = os.path.join(chunks_dir, f'{prefix}_{chunk_idx}.pt')
    torch.save(chunk_tensor, chunk_file)
    return chunk_file, chunk_tensor.shape


def compute_sae_text_embeddings(dataset_name, model, tokenizer, device):
    """
    Compute raw SAE embeddings for a text dataset (no normalization).
    Saves in the same chunked format as regular embeddings.
    """
    print(f"\n=== Processing {dataset_name} ===")
    
    # Load text data - first return value is all texts in correct order
    all_texts, _, _, _ = load_text(dataset_name)
    
    print(f"  Total texts: {len(all_texts)}")
    
    # Setup output directories - use same Embeddings folder as regular embeddings
    base_dir = f'{SCRATCH_DIR}Embeddings/{dataset_name}'
    os.makedirs(base_dir, exist_ok=True)
    
    # File naming: Gemma_SAE_[cls/patch]_embeddings_percentthrumodel_81.pt
    cls_file = f'{base_dir}/{MODEL_IDENTIFIER}_cls_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
    patch_file = f'{base_dir}/{MODEL_IDENTIFIER}_patch_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
    
    # Initialize chunk directories
    cls_chunks_dir = os.path.dirname(cls_file)
    patch_chunks_dir = os.path.dirname(patch_file)
    
    cls_chunk_info = {
        'chunks': [],
        'total_samples': 0,
        'embedding_dim': HIDDEN_SIZE,
        'num_chunks': 0
    }
    patch_chunk_info = {
        'chunks': [],
        'total_samples': 0,
        'embedding_dim': HIDDEN_SIZE,
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
    
    print(f"\nComputing raw embeddings from layer {LAYER_INDEX}...")
    
    # Process all texts
    n_batches = (len(all_texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {len(all_texts)} total texts in {n_batches} batches...")
    
    for batch_idx in tqdm(range(n_batches), desc="Computing embeddings"):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min((batch_idx + 1) * BATCH_SIZE, len(all_texts))
        batch_texts = all_texts[batch_start:batch_end]
        actual_batch_size = len(batch_texts)
        
        # Get real layer-34 embeddings (raw, no normalization)
        cls_embeddings, token_embeddings, token_map = get_layer34_embeddings(
            model, tokenizer, batch_texts, device
        )
        
        # Add to current chunks
        cls_current_chunk_tensors.append(cls_embeddings.cpu())
        patch_current_chunk_tensors.append(token_embeddings.cpu())
        
        cls_current_chunk_size += actual_batch_size
        patch_current_chunk_size += token_embeddings.shape[0]
        
        total_cls_samples += actual_batch_size
        total_patch_samples += token_embeddings.shape[0]
        
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
        del cls_embeddings, token_embeddings
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
    print(f"✅ Token embeddings saved: {total_patch_samples} samples in {patch_chunk_idx} chunks")
    print(f"   Location: {patch_file.replace('.pt', '_chunk_*.pt') if patch_chunk_idx > 1 else patch_file}")
    
    return cls_chunk_info, patch_chunk_info


def main():
    """Main function to process all text datasets."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute SAE text embeddings for specified datasets')
    parser.add_argument('--datasets', nargs='+', 
                        choices=DEFAULT_DATASET_NAMES,
                        default=DEFAULT_DATASET_NAMES,
                        help='Datasets to process (default: all datasets)')
    args = parser.parse_args()
    
    dataset_names = args.datasets
    
    print("🚀 Starting SAE text embedding extraction")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Layer: {LAYER_INDEX} (residual tokens)")
    print(f"   Device: {DEVICE}")
    print(f"   Datasets: {dataset_names}")
    
    # Load Gemma-2 model and tokenizer (same approach as embed_text_datasets.py)
    print("\n📦 Loading Gemma-2 model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model = model.to(DEVICE)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process each dataset
    for dataset_name in dataset_names:
        try:
            compute_sae_text_embeddings(dataset_name, model, tokenizer, DEVICE)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n✨ SAE text embedding extraction complete!")


if __name__ == "__main__":
    main()