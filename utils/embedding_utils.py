"""Utils for Embedding Functions"""

import os
import torch
import gc
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.general_utils import get_split_df
from utils.patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence


### Unified Layer-Percent Conversion Functions ###

# Global set to track which configurations have been printed
_printed_percent_configs = set()

def percent_to_layer(percent_thru_model, total_layers, model_name=""):
    """
    Convert percent through model to layer index (0-based).
    
    Args:
        percent_thru_model: Percentage through model (0-100)
        total_layers: Total number of layers in the model
        model_name: Optional model name for logging
        
    Returns:
        layer_index: 0-based layer index
    """
    # Calculate raw layer position
    raw_layer = (percent_thru_model / 100) * total_layers
    
    # Convert to 0-based index
    layer_index = int(raw_layer) - 1
    
    # Ensure within valid range
    layer_index = max(0, min(layer_index, total_layers - 1))
    
    if model_name:
        # Create a unique key for this configuration
        config_key = (model_name, percent_thru_model, total_layers)
        if config_key not in _printed_percent_configs:
            _printed_percent_configs.add(config_key)
            print(f"{model_name}: percent_thru_model={percent_thru_model}% -> layer {layer_index} (0-indexed) or layer {layer_index + 1} (1-indexed)")
    
    return layer_index


def layer_to_percent(layer_index, total_layers, zero_indexed=True, model_name=""):
    """
    Convert layer index to percent through model.
    
    Args:
        layer_index: Layer index
        total_layers: Total number of layers in the model
        zero_indexed: Whether the layer_index is 0-based (default: True)
        model_name: Optional model name for logging
        
    Returns:
        percent_thru_model: Percentage through model (0-100)
    """
    # Convert to 1-based if needed
    if zero_indexed:
        layer_1indexed = layer_index + 1
    else:
        layer_1indexed = layer_index
    
    # Calculate percentage
    percent_thru_model = (layer_1indexed / total_layers) * 100
    
    if model_name:
        layer_display = layer_index if zero_indexed else layer_index - 1
        # Create a unique key for this configuration
        config_key = (model_name, layer_index, total_layers, zero_indexed)
        if config_key not in _printed_percent_configs:
            _printed_percent_configs.add(config_key)
            print(f"{model_name}: layer {layer_display} (0-indexed) or layer {layer_1indexed} (1-indexed) -> percent_thru_model={percent_thru_model:.1f}%")
    
    return percent_thru_model

class ChunkedEmbeddingDataset(Dataset):
    """
    PyTorch Dataset that loads embeddings on-demand from chunked files.
    Maintains global indexing while loading from chunk-local indices.
    """
    def __init__(self, embeddings_path, indices, labels, device='cpu'):
        """
        Args:
            embeddings_path: Path to embeddings file (chunked or not)
            indices: List of global indices to include in dataset
            labels: Tensor of labels corresponding to indices
            device: Device to load embeddings to
        """
        from utils.memory_management_utils import ChunkedEmbeddingLoader
        
        self.loader = ChunkedEmbeddingLoader(embeddings_path, device=device)
        self.indices = indices
        self.labels = labels
        self.device = device
        
        # Cache for recently loaded chunks to avoid repeated loading
        self.chunk_cache = {}
        # Increase cache size based on available memory and number of chunks
        if hasattr(self.loader, 'chunk_info') and self.loader.is_chunked:
            num_chunks = self.loader.chunk_info['num_chunks']
            # Try to cache more chunks if we have many chunks
            # But limit based on memory constraints
            self.max_cache_size = min(num_chunks, 20)  # Cache up to 20 chunks
        else:
            self.max_cache_size = 3
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        label = self.labels[idx]
        
        # Get chunk number and local index
        chunk_num, local_idx = self.loader.global_to_chunk_index(global_idx)
        
        # Track chunk access pattern
        if not hasattr(self, 'chunk_access_count'):
            self.chunk_access_count = {}
        if chunk_num not in self.chunk_access_count:
            self.chunk_access_count[chunk_num] = 0
        self.chunk_access_count[chunk_num] += 1
        
        # Print chunk access pattern periodically
        if idx > 0 and idx % 1000 == 0:
            print(f"    Chunk access pattern after {idx} samples: {dict(sorted(self.chunk_access_count.items()))}")
        
        # Check if chunk is in cache
        if chunk_num not in self.chunk_cache:
            # Load chunk
            if self.loader.is_chunked:
                try:
                    chunk_path = os.path.join(self.loader.chunks_dir, 
                                            self.loader.chunk_info['chunks'][chunk_num]['file'])
                    # if idx < 5:
                    #     print(f"Loading chunk from: {chunk_path}")
                    
                    chunk_data = torch.load(chunk_path, map_location=self.device)
                    
                    # Handle different storage formats
                    if isinstance(chunk_data, dict):
                        if 'normalized_embeddings' in chunk_data:
                            chunk_embeddings = chunk_data['normalized_embeddings']
                        elif 'embeddings' in chunk_data:
                            chunk_embeddings = chunk_data['embeddings']
                        else:
                            raise ValueError(f"Chunk dict does not contain 'normalized_embeddings' or 'embeddings' keys. Keys: {chunk_data.keys()}")
                    else:
                        # Handle case where chunk is stored directly as tensor
                        chunk_embeddings = chunk_data
                    
                    # if idx < 5:
                    #     print(f"Loaded chunk {chunk_num} with shape: {chunk_embeddings.shape}")
                        
                    # Add to cache
                    self.chunk_cache[chunk_num] = chunk_embeddings
                except Exception as e:
                    print(f"Error loading chunk {chunk_num}: {e}")
                    print(f"Chunk info: {self.loader.chunk_info['chunks'][chunk_num]}")
                    raise
                
                # Remove least recently used chunk if cache is full
                if len(self.chunk_cache) > self.max_cache_size:
                    # Don't remove the chunk we just loaded
                    removable_chunks = [k for k in self.chunk_cache.keys() if k != chunk_num]
                    if removable_chunks:
                        # Remove the chunk with the smallest number (approximation of LRU)
                        chunk_to_remove = min(removable_chunks)
                        del self.chunk_cache[chunk_to_remove]
                        gc.collect()
            else:
                # For non-chunked files, load all at once
                if 0 not in self.chunk_cache:
                    data = torch.load(self.loader.embeddings_path, map_location=self.device)
                    if 'normalized_embeddings' in data:
                        self.chunk_cache[0] = data['normalized_embeddings']
                    else:
                        self.chunk_cache[0] = data['embeddings']
        
        # Get embedding from cache
        try:
            embedding = self.chunk_cache[chunk_num][local_idx]
        except KeyError as e:
            print(f"KeyError accessing chunk {chunk_num}, local_idx {local_idx}")
            print(f"Chunks in cache: {list(self.chunk_cache.keys())}")
            print(f"Cache size: {len(self.chunk_cache)}, max size: {self.max_cache_size}")
            raise
        
        return embedding, label


### For Computing Embeddings ###

def get_intermediate_representations(model, processor, images, device, percent_thru_model):
    """
    Extracts embeddings from chosen layer of given model for each patch and class token in each image.

    Args:
        model: The CLIP model to generate embeddings.
        processor: The processor used for transforming the images and text.
        images: A list of PIL.Image objects.
        device: The device to move the tensors to.
        percent_thru_model (int) : For patch concepts, percentage through model intermediate rep is extracted from.

    Returns:
        torch.Tensor: The generated embeddings per patch per image.
    """
    # Extracts embeddings from a specific layer of the CLIP image encoder. (will need to change if you use different model)
    total_layers = len(model.vision_model.encoder.layers)
    layer_index = percent_to_layer(percent_thru_model, total_layers, model_name="CLIP") 
    layer_output = []
    
    def hook(module, input, output):
        layer_output.append(output)
    
    # Register the hook to the specified layer
    layer = model.vision_model.encoder.layers[layer_index]
    handle = layer.register_forward_hook(hook)
    
    # Preprocess the image and convert it to tensor format
    processed_images = processor(images=images, return_tensors="pt", padding=True).to(device)
    
    # The output of the hook contains the patch embeddings
    with torch.no_grad():
        model.get_image_features(pixel_values=processed_images['pixel_values'])
    all_embeddings = layer_output[0][0]
    handle.remove()
    return all_embeddings


def get_intermediate_representations_llama(model, processor, images, device, percent_thru_model):
    """
    Extracts intermediate vision embeddings from LLaMA-Vision-Instruct.

    Args:
        model: LLaMA-Vision-Instruct model.
        processor: HuggingFace processor.
        images: List of PIL.Image objects.
        device: Torch device.
        percent_thru_model: Percent through vision encoder (0–100) to extract from.

    Returns:
        torch.Tensor: Shape [n_images, seq_len, hidden_dim] for selected layer.
    """
    inputs = processor(images, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Llama vision model has 32 transformer layers + 8 global transformer layers = 40 total
    total_layers = 40
    target_layer_idx = percent_to_layer(percent_thru_model, total_layers, model_name="Llama")
    
    # Storage for the target layer output
    target_output = None
    
    def capture_layer_output(module, input, output):
        nonlocal target_output
        if isinstance(output, tuple) and len(output) > 0:
            target_output = output[0]
        elif isinstance(output, torch.Tensor):
            target_output = output
        elif hasattr(output, 'last_hidden_state'):
            target_output = output.last_hidden_state
    
    # Register hook on the target layer
    if target_layer_idx < 32:
        # Target is in main transformer
        handle = model.vision_model.transformer.layers[target_layer_idx].register_forward_hook(capture_layer_output)
    else:
        # Target is in global transformer (layers 32-39)
        global_layer_idx = target_layer_idx - 32
        handle = model.vision_model.global_transformer.layers[global_layer_idx].register_forward_hook(capture_layer_output)
    
    try:
        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=inputs["pixel_values"],
                aspect_ratio_ids=inputs["aspect_ratio_ids"],
                aspect_ratio_mask=inputs["aspect_ratio_mask"],
                output_hidden_states=False,  # We don't need all hidden states
                output_attentions=False,
                return_dict=True
            )
    finally:
        # Always remove the hook
        handle.remove()
    
    if target_output is None:
        raise ValueError(f"Failed to capture output from layer {target_layer_idx}")
    
    # Reshape if needed - the output might be [batch_size, seq_len, hidden_dim]
    # or might have an extra dimension
    if target_output.dim() == 4:
        # If shape is [batch_size, num_tiles, seq_len, hidden_dim], flatten tiles
        batch_size, num_tiles, seq_len, hidden_dim = target_output.shape
        target_output = target_output.view(batch_size, num_tiles * seq_len, hidden_dim)
    
    return target_output



def get_clip_both_embeddings(model, processor, images, device, percent_thru_model):
    """
    Extracts both CLS and patch embeddings from CLIP in a single pass.
    
    Returns:
        tuple: (cls_embeddings, patch_embeddings)
    """
    all_embeddings = get_intermediate_representations(model, processor, images, device, percent_thru_model)
    
    # Extract CLS embeddings
    cls_embeddings = all_embeddings[:, 0, :]
    
    # Extract patch embeddings
    patch_embeddings = all_embeddings[:, 1:, :]
    patch_embeddings = patch_embeddings.reshape(-1, patch_embeddings.size(-1))
    
    return cls_embeddings, patch_embeddings

def get_llama_both_embeddings(model, processor, images, device, percent_thru_model):
    """
    Extracts CLS and first-tile *true patch* embeddings (1600 tokens only) from LLaMA-Vision-Instruct.

    Returns:
        tuple:
            - cls_embeddings: [n_images, hidden_dim]
            - patch_embeddings: [n_images * 1600, hidden_dim]
    """
    embeddings = get_intermediate_representations_llama(
        model, processor, images, device, percent_thru_model
    )  # [n_images, 6432, 1280]

    # Extract CLS token
    cls_embeddings = embeddings[:, 0, :]  # [n_images, 1280]

    # Extract only true patch tokens from tile 1 (excluding delimiters)
    # CLS = position 0, tile 1 = positions 1 through 1600 (inclusive)
    first_tile_patches = embeddings[:, 1:1601, :]  # [n_images, 1600, 1280]

    # Flatten across images
    patch_embeddings = first_tile_patches.reshape(-1, embeddings.shape[-1])  # [n_images * 1600, 1280]

    return cls_embeddings, patch_embeddings



def get_text_both_embeddings(model, processor, text_samples, device, percent_thru_model=100):
    """
    Extracts both token-level (patch) and mean-pooled (cls) embeddings from text in a single pass.
    
    Returns:
        tuple: (cls_embeddings, patch_embeddings)
    """
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    
    # Load layer mappings from JSON file
    import json
    import os
    layer_mappings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      'Model_Layer_Mappings', 'percentthrumodel_to_layer_mappings.json')
    
    with open(layer_mappings_path, 'r') as f:
        layer_mappings = json.load(f)
    
    # Determine model name and get total layers from mappings
    model_name = "Unknown"
    total_layers = None
    
    if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
        model_path = model.config._name_or_path
        if 'llama' in model_path.lower():
            model_name = "Llama-Text"
            total_layers = layer_mappings['Llama-Text']['total_layers']  # 32
        elif 'gemma' in model_path.lower():
            if 'gemma-2' in model_path.lower():
                model_name = "Gemma-Text"  # Using Gemma-2 entry from mappings
                total_layers = layer_mappings['Gemma-2']['total_layers']  # 42
            else:
                model_name = "Gemma-Text"
                total_layers = layer_mappings['Gemma-Text']['total_layers']  # 28
        elif 'qwen' in model_path.lower():
            model_name = "Qwen-Text"
            total_layers = layer_mappings['Qwen-Text']['total_layers']  # 32
    
    all_hidden_states = []
    mean_embeddings = []
    
    with torch.no_grad():
        for text_sample in text_samples:
            inputs = tokenizer(
                text_sample,
                add_special_tokens=False,
                padding=False,
                return_tensors="pt"
            ).to(device)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Verify total_layers was set from mappings
            if total_layers is None:
                raise ValueError(f"Could not determine total layers for model {model_name}. Model path: {model.config._name_or_path if hasattr(model, 'config') else 'Unknown'}")
            
            # Calculate which layer to extract from based on percent_thru_model
            if percent_thru_model == 100:
                # Use the last layer
                target_hidden = outputs.hidden_states[-1]
            else:
                # Calculate layer index using percent_to_layer
                layer_index = percent_to_layer(percent_thru_model, total_layers, model_name=model_name)
                # hidden_states[0] is embeddings, hidden_states[1] is after layer 0, etc.
                # So we need layer_index + 1 to get the output after layer layer_index
                hidden_state_index = layer_index + 1
                
                # Verify we have enough hidden states
                if hidden_state_index >= len(outputs.hidden_states):
                    raise ValueError(f"Model {model_name} only has {len(outputs.hidden_states)-1} layers but trying to access layer {layer_index}")
                
                target_hidden = outputs.hidden_states[hidden_state_index]
            
            # Store token-level embeddings
            all_hidden_states.append(target_hidden.squeeze(0))  # (seq_len, hidden_dim)
            
            # Store mean pooled embeddings
            mean_emb = target_hidden.mean(dim=1)  # (1, hidden_dim)
            mean_embeddings.append(mean_emb)
    
    # Concatenate results
    patch_embeddings = torch.cat(all_hidden_states, dim=0)  # (total_tokens, hidden_dim)
    cls_embeddings = torch.cat(mean_embeddings, dim=0)  # (n_samples, hidden_dim)
    
    return cls_embeddings, patch_embeddings

def compute_raw_batch_embeddings(images, embedding_fxn, model, processor, device, 
                                 percent_thru_model, dataset_name, batch_size=100):
    """
    Compute raw embeddings for images in batches and split into train/test sets.

    Args:
        images (list): List of PIL.Image objects.
        embedding_fxn (function): Function to compute embeddings.
        model: Model for generating embeddings.
        processor: Processor for preprocessing images.
        device: Device to run the model on.
        percent_thru_model (int): Model layer from which to extract representations.
        dataset_name (str): Dataset name (used for loading metadata).
        batch_size (int): Number of images per batch.

    Returns:
        tuple: (train_embeddings, test_embeddings)
    """
    print("Computing embeddings in batches...")
    embeddings = []
    n_batches = (len(images) + batch_size - 1) // batch_size  # Compute number of batches
    for i in tqdm(range(n_batches), desc="Computing embeddings"):
        batch_images = images[i * batch_size:(i + 1) * batch_size]
        batch_embeddings = embedding_fxn(model, processor, batch_images, device, percent_thru_model)
        print(f"batch embeddings: {batch_embeddings.shape}")
        embeddings.append(batch_embeddings.cpu())
        del batch_embeddings
        torch.cuda.empty_cache()
    embeddings = torch.cat(embeddings, dim=0)  # Concatenate all batch embeddings
    # print(f"Extracted embeddings of shape: {embeddings.shape}")
    return embeddings



def compute_train_avg_and_norm(embeddings, dataset_name, model_input_size, sample_type):
    if sample_type == 'patch':
        #Load split_df
        split_df = get_patch_split_df(dataset_name, model_input_size)

        # Create boolean masks for train/test
        train_mask = split_df[split_df == 'train']

        #Filter out the embeddings that are 'padding'
        relevant_indices = filter_patches_by_image_presence(train_mask.index, dataset_name, model_input_size).tolist()
        final_mask = train_mask.loc[train_mask.index.intersection(relevant_indices)]
    else:
        split_df = get_split_df(dataset_name)
        final_mask = split_df[split_df == 'train']
    
    # Apply masks to embeddings
    train_embeddings = embeddings[final_mask.index.to_list()].float()
    
    mean_train_embedding = train_embeddings.mean(dim=0)
    train_norm = train_embeddings.norm(dim=1, keepdim=True).mean()
    return mean_train_embedding, train_norm
    

def center_and_normalize_embeddings(embeddings, dataset_name, model_input_size, sample_type):
    """
    Center and normalize embeddings using statistics from the training set.

    Args:
        train_embeddings (torch.Tensor): Tensor of training set embeddings.
        test_embeddings (torch.Tensor): Tensor of test set embeddings.

    Returns:
        tuple: (normalized_train_embeddings, normalized_test_embeddings)
    """
    mean_train_embedding, train_norm = compute_train_avg_and_norm(embeddings, dataset_name, model_input_size, sample_type)
    centered_embeddings = embeddings - mean_train_embedding
    norm_embeddings = centered_embeddings / train_norm

    return norm_embeddings, mean_train_embedding, train_norm



def get_training_indices(dataset_name, model_input_size, sample_type):
    """Get training indices for either patch or cls embeddings."""
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name, model_input_size)
        train_mask = split_df[split_df == 'train']
        relevant_indices = filter_patches_by_image_presence(train_mask.index, dataset_name, model_input_size).tolist()
        final_train_mask = train_mask.loc[train_mask.index.intersection(relevant_indices)]
    else:  # cls
        split_df = get_split_df(dataset_name)
        final_train_mask = split_df[split_df == 'train']
    
    return torch.tensor(final_train_mask.index.to_list())


def initialize_stats(embedding_dim):
    """Initialize statistics accumulators for training data."""
    sum_embeddings = torch.zeros(embedding_dim, dtype=torch.float64)
    sum_squared_norms = 0.0
    n_train_samples = 0
    return sum_embeddings, sum_squared_norms, n_train_samples


def update_training_stats(embeddings, indices, train_indices, sum_embeddings, sum_squared_norms, n_train_samples):
    """Update training statistics with a batch of embeddings."""
    is_train = torch.isin(indices, train_indices)
    
    if is_train.any():
        train_embeddings = embeddings[is_train].double()
        sum_embeddings += train_embeddings.sum(dim=0)
        sum_squared_norms += (train_embeddings.norm(dim=1) ** 2).sum().item()
        n_train_samples += is_train.sum().item()
    
    return sum_embeddings, sum_squared_norms, n_train_samples


def save_chunk_if_ready(current_chunk_tensors, current_chunk_size, total_samples,
                       chunk_idx, chunks_dir, chunk_info, samples_per_chunk, is_last_batch):
    """Save accumulated tensors as a chunk if size threshold met or last batch."""
    if current_chunk_size >= samples_per_chunk or is_last_batch:
        if not current_chunk_tensors:  # Safety check
            return current_chunk_tensors, current_chunk_size, chunk_idx
            
        chunk_tensor = torch.cat(current_chunk_tensors, dim=0)
        chunk_file = os.path.join(chunks_dir, f'chunk_{chunk_idx}.pt')
        torch.save(chunk_tensor, chunk_file)
        
        chunk_info['chunks'].append({
            'file': f'chunk_{chunk_idx}.pt',
            'start_idx': total_samples - current_chunk_size,
            'end_idx': total_samples,
            'shape': list(chunk_tensor.shape)
        })
        
        del chunk_tensor
        gc.collect()
        
        return [], 0, chunk_idx + 1
    
    return current_chunk_tensors, current_chunk_size, chunk_idx


def compute_final_statistics(sum_embeddings, sum_squared_norms, n_train_samples):
    """Compute mean and norm from accumulated statistics."""
    mean_train_embedding = (sum_embeddings / n_train_samples).float()
    train_norm = np.sqrt(sum_squared_norms / n_train_samples)
    return mean_train_embedding, train_norm


def save_statistics_file(stats_file, mean_train_embedding, train_norm, n_train_samples, embedding_dim):
    """Save training statistics to file."""
    torch.save({
        'mean_train_embedding': mean_train_embedding,
        'train_norm': train_norm,
        'n_train_samples': n_train_samples,
        'embedding_dim': embedding_dim
    }, stats_file)


def setup_file_paths(model_name, dataset_name, percent_thru_model, scratch_dir):
    """Set up file paths for cls and patch embeddings."""
    base_dir = f'{scratch_dir}Embeddings/{dataset_name}'
    cls_file = f'{base_dir}/{model_name}_cls_embeddings_percentthrumodel_{percent_thru_model}.pt'
    patch_file = f'{base_dir}/{model_name}_patch_embeddings_percentthrumodel_{percent_thru_model}.pt'
    os.makedirs(base_dir, exist_ok=True)
    return cls_file, patch_file


def initialize_chunk_processing(cls_file, patch_file):
    """Initialize directories and data structures for chunk processing."""
    cls_chunks_dir = cls_file.replace('.pt', '_raw_chunks')
    patch_chunks_dir = patch_file.replace('.pt', '_raw_chunks')
    os.makedirs(cls_chunks_dir, exist_ok=True)
    os.makedirs(patch_chunks_dir, exist_ok=True)
    
    cls_chunk_info = {'chunks': []}
    patch_chunk_info = {'chunks': []}
    
    return cls_chunks_dir, patch_chunks_dir, cls_chunk_info, patch_chunk_info


def process_batch_embeddings(batch_data, batch_idx, batch_size, embedding_fxn, model, processor, 
                           device, percent_thru_model, embedding_dim):
    """Process a batch of data and return cls and patch embeddings."""
    cls_embeddings, patch_embeddings = embedding_fxn(model, processor, batch_data, device, percent_thru_model)
    
    # Initialize embedding dimension on first batch
    if embedding_dim is None:
        embedding_dim = cls_embeddings.shape[1]
    
    return cls_embeddings, patch_embeddings, embedding_dim


def save_final_results(cls_file, patch_file, cls_final_chunk_info, patch_final_chunk_info):
    """Save chunk info JSON files for both embedding types."""
    cls_info_file = cls_file.replace('.pt', '_chunks_info.json')
    patch_info_file = patch_file.replace('.pt', '_chunks_info.json')
    
    with open(cls_info_file, 'w') as f:
        json.dump(cls_final_chunk_info, f, indent=2)
    
    with open(patch_info_file, 'w') as f:
        json.dump(patch_final_chunk_info, f, indent=2)
    
    print(f"CLS embeddings saved with {len(cls_final_chunk_info['chunks'])} chunks")
    print(f"Patch embeddings saved with {len(patch_final_chunk_info['chunks'])} chunks")


def process_and_save_chunks(chunk_info, chunks_dir, mean_train_embedding, train_norm, 
                           output_file, embedding_type):
    """Process raw chunks by normalizing and saving them."""
    final_chunk_info = {
        'num_chunks': len(chunk_info['chunks']),
        'total_samples': chunk_info.get('total_samples', 0),
        'embedding_dim': mean_train_embedding.shape[0],
        'stats_file': os.path.basename(output_file.replace('.pt', '_stats.pt')),
        'chunks': []
    }
    
    for idx, chunk_meta in enumerate(tqdm(chunk_info['chunks'], desc=f"Normalizing {embedding_type} chunks")):
        raw_chunk_file = os.path.join(chunks_dir, chunk_meta['file'])
        raw_embeddings = torch.load(raw_chunk_file)
        
        # Normalize
        centered_embeddings = raw_embeddings - mean_train_embedding
        norm_embeddings = centered_embeddings / train_norm
        
        # Save normalized chunk
        chunk_file = output_file.replace('.pt', f'_chunk_{idx}.pt')
        torch.save({'normalized_embeddings': norm_embeddings}, chunk_file)
        
        final_chunk_info['chunks'].append({
            'file': os.path.basename(chunk_file),
            'start_idx': chunk_meta['start_idx'],
            'end_idx': chunk_meta['end_idx'],
            'shape': list(norm_embeddings.shape)
        })
        
        # Clean up
        del raw_embeddings, centered_embeddings, norm_embeddings
        os.remove(raw_chunk_file)
        gc.collect()
    
    # Remove the directory only if it's empty
    try:
        os.rmdir(chunks_dir)
    except OSError:
        # Directory might not be empty due to other files or concurrent access
        shutil.rmtree(chunks_dir)
    
    return final_chunk_info


def compute_batch_embeddings_both(images_or_text, embedding_fxn, model, processor, device, 
                                  percent_thru_model, dataset_name, model_input_size,
                                  batch_size=100, scratch_dir="",
                                  chunk_if_larger_gb=10, samples_per_chunk=50000, model_name=None):
    """
    Compute both CLS and patch embeddings in a single pass.
    Returns a dictionary with both types of embeddings.
    """
    # Set up file paths - extract model name from function name or use provided
    if model_name is None:
        if 'clip' in embedding_fxn.__name__.lower():
            model_name = 'CLIP'
        elif 'llama' in embedding_fxn.__name__.lower():
            model_name = 'Llama'
        else:
            model_name = 'Model'  # Generic fallback
    
    cls_file, patch_file = setup_file_paths(model_name, dataset_name, percent_thru_model, scratch_dir)
    
    # Get training indices for both types
    patch_train_indices = get_training_indices(dataset_name, model_input_size, 'patch')
    cls_train_indices = get_training_indices(dataset_name, model_input_size, 'cls')
    
    # Initialize stats for both types
    cls_sum_embeddings = None
    cls_sum_squared_norms = 0.0
    cls_n_train_samples = 0
    
    patch_sum_embeddings = None
    patch_sum_squared_norms = 0.0
    patch_n_train_samples = 0
    
    total_cls_samples = 0
    total_patch_samples = 0
    embedding_dim = None
    
    # Process in large chunks
    n_batches = (len(images_or_text) + batch_size - 1) // batch_size
    
    # Initialize chunk processing
    cls_chunks_dir, patch_chunks_dir, cls_chunk_info, patch_chunk_info = initialize_chunk_processing(cls_file, patch_file)
    
    current_cls_chunk_tensors = []
    current_patch_chunk_tensors = []
    current_cls_chunk_size = 0
    current_patch_chunk_size = 0
    cls_chunk_idx = 0
    patch_chunk_idx = 0
    
    print("Pass 1: Computing embeddings and statistics...")
    for i in tqdm(range(n_batches), desc="Computing embeddings"):
        batch_data = images_or_text[i * batch_size:(i + 1) * batch_size]
        
        # Process batch using helper function
        cls_embeddings, patch_embeddings, embedding_dim = process_batch_embeddings(
            batch_data, i, batch_size, embedding_fxn, model, processor, 
            device, percent_thru_model, embedding_dim
        )
        
        if cls_sum_embeddings is None:
            cls_sum_embeddings, cls_sum_squared_norms, cls_n_train_samples = initialize_stats(embedding_dim)
            patch_sum_embeddings, patch_sum_squared_norms, patch_n_train_samples = initialize_stats(embedding_dim)
        
        # Process CLS embeddings
        batch_start_idx = i * batch_size
        cls_batch_cpu = cls_embeddings.cpu()
        current_cls_chunk_tensors.append(cls_batch_cpu)
        
        # Track CLS indices and update stats
        cls_batch_indices = torch.arange(batch_start_idx, batch_start_idx + len(cls_batch_cpu))
        cls_sum_embeddings, cls_sum_squared_norms, cls_n_train_samples = update_training_stats(
            cls_batch_cpu, cls_batch_indices, cls_train_indices, 
            cls_sum_embeddings, cls_sum_squared_norms, cls_n_train_samples
        )
        
        current_cls_chunk_size += len(cls_batch_cpu)
        total_cls_samples += len(cls_batch_cpu)
        
        # Process patch embeddings
        patch_batch_cpu = patch_embeddings.cpu()
        current_patch_chunk_tensors.append(patch_batch_cpu)
        
        # Track patch indices and update stats
        n_patches_per_image = len(patch_embeddings) // len(batch_data)
        patch_batch_start_idx = batch_start_idx * n_patches_per_image
        patch_batch_indices = torch.arange(patch_batch_start_idx, patch_batch_start_idx + len(patch_batch_cpu))
        patch_sum_embeddings, patch_sum_squared_norms, patch_n_train_samples = update_training_stats(
            patch_batch_cpu, patch_batch_indices, patch_train_indices, 
            patch_sum_embeddings, patch_sum_squared_norms, patch_n_train_samples
        )
        
        current_patch_chunk_size += len(patch_batch_cpu)
        total_patch_samples += len(patch_batch_cpu)
        
        # Save CLS chunk if large enough
        current_cls_chunk_tensors, current_cls_chunk_size, cls_chunk_idx = save_chunk_if_ready(
            current_cls_chunk_tensors, current_cls_chunk_size, total_cls_samples,
            cls_chunk_idx, cls_chunks_dir, cls_chunk_info, samples_per_chunk, i == n_batches - 1
        )
        
        # Save patch chunk if large enough
        current_patch_chunk_tensors, current_patch_chunk_size, patch_chunk_idx = save_chunk_if_ready(
            current_patch_chunk_tensors, current_patch_chunk_size, total_patch_samples,
            patch_chunk_idx, patch_chunks_dir, patch_chunk_info, samples_per_chunk, i == n_batches - 1
        )
        
        del cls_embeddings, patch_embeddings
        torch.cuda.empty_cache()
        gc.collect()
    
    # Compute final statistics
    cls_mean_train_embedding, cls_train_norm = compute_final_statistics(
        cls_sum_embeddings, cls_sum_squared_norms, cls_n_train_samples
    )
    
    patch_mean_train_embedding, patch_train_norm = compute_final_statistics(
        patch_sum_embeddings, patch_sum_squared_norms, patch_n_train_samples
    )
    
    print(f"CLS training statistics computed from {cls_n_train_samples} samples")
    print(f"Patch training statistics computed from {patch_n_train_samples} samples")
    
    # Save statistics
    cls_stats_file = cls_file.replace('.pt', '_stats.pt')
    patch_stats_file = patch_file.replace('.pt', '_stats.pt')
    
    save_statistics_file(cls_stats_file, cls_mean_train_embedding, cls_train_norm, 
                        cls_n_train_samples, embedding_dim)
    
    save_statistics_file(patch_stats_file, patch_mean_train_embedding, patch_train_norm,
                        patch_n_train_samples, embedding_dim)
    
    # Pass 2: Normalize and save chunks
    print("\nPass 2: Normalizing and saving final chunks...")
    
    # Add total_samples to chunk_info
    cls_chunk_info['total_samples'] = total_cls_samples
    patch_chunk_info['total_samples'] = total_patch_samples
    
    # Process chunks using helper function
    cls_final_chunk_info = process_and_save_chunks(
        cls_chunk_info, cls_chunks_dir, cls_mean_train_embedding, cls_train_norm, cls_file, 'CLS'
    )
    
    patch_final_chunk_info = process_and_save_chunks(
        patch_chunk_info, patch_chunks_dir, patch_mean_train_embedding, patch_train_norm, patch_file, 'patch'
    )
    
    # Save final results
    save_final_results(cls_file, patch_file, cls_final_chunk_info, patch_final_chunk_info)
    
    return {
        'cls': {
            'is_chunked': True,
            'chunk_info': cls_final_chunk_info,
            'mean_train_embedding': cls_mean_train_embedding,
            'train_norm': cls_train_norm,
            'n_chunks': len(cls_final_chunk_info['chunks'])
        },
        'patch': {
            'is_chunked': True,
            'chunk_info': patch_final_chunk_info,
            'mean_train_embedding': patch_mean_train_embedding,
            'train_norm': patch_train_norm,
            'n_chunks': len(patch_final_chunk_info['chunks'])
        }
    }

def compute_batch_embeddings(images_or_text, model, processor, device, 
                            percent_thru_model, dataset_name, model_input_size,
                            batch_size=100, scratch_dir="", 
                            chunk_if_larger_gb=10, samples_per_chunk=50000, model_name=None):
    """
    Memory-efficient chunked processing that computes both cls and patch embeddings in one pass.
    Always processes both types to avoid redundant computation.
    """
    # Determine which embedding function to use based on model_input_size
    if model_input_size == (224, 224):
        both_fn = get_clip_both_embeddings
        if model_name is None:
            model_name = 'CLIP'
    elif model_input_size == (560, 560):
        both_fn = get_llama_both_embeddings
        if model_name is None:
            model_name = 'Llama'
    elif model_input_size == ('text', 'text'):
        both_fn = get_text_both_embeddings
        if model_name is None:
            model_name = 'Llama'
    elif model_input_size == ('text', 'text2'):
        both_fn = get_text_both_embeddings
        if model_name is None:
            model_name = 'Gemma'
    elif model_input_size == ('text', 'text3'):
        both_fn = get_text_both_embeddings
        if model_name is None:
            model_name = 'Qwen'
    else:
        raise ValueError(f"Unknown model_input_size: {model_input_size}")
    
    return compute_batch_embeddings_both(images_or_text, both_fn, model, processor, device,
                                       percent_thru_model, dataset_name, model_input_size,
                                       batch_size, scratch_dir, chunk_if_larger_gb, samples_per_chunk, model_name)

