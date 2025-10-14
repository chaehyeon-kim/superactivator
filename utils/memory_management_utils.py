"""
Memory Management Utilities for Processing Chunked Embeddings

This module provides utilities for efficiently processing large embedding files that 
have been split into chunks to reduce memory usage.

Key Features:
- Auto-detection of chunked vs non-chunked embeddings
- Incremental loading and processing of chunks
- Memory-efficient aggregation of results
- Index mapping between global and chunk-local indices
- Unified interface for both chunked and non-chunked embeddings
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
import gc
from typing import Dict, List, Tuple, Optional, Union, Iterator, Callable, Any
from contextlib import contextmanager
from tqdm import tqdm

import numpy as np
from utils.convert_to_memmap import convert_embeddings_to_memmap, MemmapEmbeddingLoader


class ChunkedEmbeddingLoader:
    """
    Loader for chunked embedding files that provides memory-efficient access.
    
    Supports both chunked files (split into multiple .pt files) and regular files.
    Provides iterator interface for processing chunks incrementally.
    """
    
    def __init__(self, dataset_name: str, embeddings_file: str, scratch_dir: str = '', device: str = 'cpu', use_memmap: bool = True):
        """
        Initialize the chunked embedding loader.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'CLEVR')
            embeddings_file: Filename of embeddings (e.g., 'Llama_patch_embeddings_percentthrumodel_100.pt')
            scratch_dir: Base scratch directory (e.g., '/path/to/scratch/')
            device: Device to load embeddings on
            use_memmap: Whether to use memory-mapped files for faster loading
        """
        self.dataset_name = dataset_name
        self.embeddings_file = embeddings_file
        self.scratch_dir = scratch_dir
        self.device = device
        self.use_memmap = use_memmap
        
        # GPU cache for frequently accessed chunks (3-5x speedup)
        self.gpu_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cached_chunks = 4  # Cache 4x9GB chunks on 47GB GPU (~36GB cache, ~11GB free)
        self.embeddings_path = os.path.join(scratch_dir, 'Embeddings', dataset_name, embeddings_file)
        self.is_chunked = False
        self.memmap_loader = None
        self.chunk_info = None
        self.total_samples = 0
        self.embedding_dim = 0
        self.chunks_dir = os.path.dirname(self.embeddings_path)
        
        # Check if embeddings are chunked
        self._detect_chunked_embeddings()
    
    def get_cache_stats(self):
        """Get GPU cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.2%}",
            'cached_chunks': len(self.gpu_cache),
            'max_chunks': self.max_cached_chunks
        }
    
    def clear_cache(self):
        """Clear the GPU cache to free memory."""
        self.gpu_cache.clear()
        gc.collect()
        if self.device != 'cpu':
            torch.cuda.empty_cache()
        
    def _detect_chunked_embeddings(self):
        """Detect if embeddings are split into chunks or use memmap."""
        # First, check if we should use memmap
        if self.use_memmap and not self.embeddings_path.endswith('_chunked.pt'):
            base_name = os.path.basename(self.embeddings_path).replace('.pt', '')
            memmap_info_path = os.path.join(self.chunks_dir, f"{base_name}_memmap_info.pt")
            memmap_data_path = os.path.join(self.chunks_dir, f"{base_name}_memmap.dat")
            
            if os.path.exists(memmap_info_path) and os.path.exists(memmap_data_path):
                # Load existing memmap
                print(f"Using existing memory-mapped embeddings")
                self.memmap_loader = MemmapEmbeddingLoader(memmap_data_path, memmap_info_path)
                self.total_samples = self.memmap_loader.shape[0]
                self.embedding_dim = self.memmap_loader.shape[1]
                return
            elif os.path.exists(self.embeddings_path):
                # Create memmap if source exists
                print(f"Converting to memory-mapped format for faster loading...")
                try:
                    memmap_path, info_path = convert_embeddings_to_memmap(self.embeddings_path, self.chunks_dir)
                    self.memmap_loader = MemmapEmbeddingLoader(memmap_path, info_path)
                    self.total_samples = self.memmap_loader.shape[0]
                    self.embedding_dim = self.memmap_loader.shape[1]
                    return
                except Exception as e:
                    print(f"Failed to create memmap: {e}. Falling back to regular loading.")
                    self.memmap_loader = None
        
        # Look for chunk info file
        base_name = os.path.splitext(self.embeddings_path)[0]
        chunk_info_file = f"{base_name}_chunks_info.json"
        
        if os.path.exists(chunk_info_file):
            # Load chunk information
            with open(chunk_info_file, 'r') as f:
                self.chunk_info = json.load(f)
            
            self.is_chunked = True
            self.total_samples = self.chunk_info['total_samples']
            self.embedding_dim = self.chunk_info['embedding_dim']
            
            # print(f"   Detected chunked embeddings: {self.chunk_info['num_chunks']} chunks")
            # print(f"   Total samples: {self.total_samples:,}, Embedding dim: {self.embedding_dim}")
              
        else:
            raise FileNotFoundError(f"Embedding file not found: {chunk_info_file}")
    
    def global_to_chunk_index(self, global_idx: int) -> Tuple[int, int]:
        """
        Convert global embedding index to (chunk_number, local_index_in_chunk).
        
        Args:
            global_idx: Index in the full embedding tensor
            
        Returns:
            Tuple of (chunk_number, local_index_in_chunk)
            
        Raises:
            ValueError: If global_idx is out of bounds
            RuntimeError: If embeddings are not chunked
        """
        if not self.is_chunked:
            # For non-chunked embeddings, everything is in chunk 0
            return (0, global_idx)
        
        if global_idx < 0 or global_idx >= self.total_samples:
            raise ValueError(f"Global index {global_idx} out of bounds [0, {self.total_samples})")
        
        # Find which chunk contains this global index
        for chunk_num, chunk_data in enumerate(self.chunk_info['chunks']):
            start_idx = chunk_data['start_idx']
            end_idx = chunk_data['end_idx']
            
            if start_idx <= global_idx < end_idx:
                local_idx = global_idx - start_idx
                return (chunk_num, local_idx)
        
        raise RuntimeError(f"Could not find chunk for global index {global_idx}")
    
    def chunk_to_global_index(self, chunk_num: int, local_idx: int) -> int:
        """
        Convert (chunk_number, local_index_in_chunk) to global embedding index.
        
        Args:
            chunk_num: Chunk number
            local_idx: Index within the chunk
            
        Returns:
            Global index in the full embedding tensor
            
        Raises:
            ValueError: If chunk_num or local_idx is out of bounds
            RuntimeError: If embeddings are not chunked
        """
        if not self.is_chunked:
            # For non-chunked embeddings, local_idx is the global index
            return local_idx
        
        if chunk_num < 0 or chunk_num >= self.chunk_info['num_chunks']:
            raise ValueError(f"Chunk number {chunk_num} out of bounds [0, {self.chunk_info['num_chunks']})")
        
        chunk_data = self.chunk_info['chunks'][chunk_num]
        start_idx = chunk_data['start_idx']
        end_idx = chunk_data['end_idx']
        chunk_size = end_idx - start_idx
        
        if local_idx < 0 or local_idx >= chunk_size:
            raise ValueError(f"Local index {local_idx} out of bounds [0, {chunk_size}) for chunk {chunk_num}")
        
        return start_idx + local_idx
    
    def global_indices_to_chunk_map(self, global_indices: List[int]) -> Dict[int, List[Tuple[int, int]]]:
        """
        Convert a list of global indices to a mapping of chunk_num -> [(global_idx, local_idx), ...].
        
        This is useful for efficiently processing multiple indices by grouping them by chunk.
        
        Args:
            global_indices: List of global indices
            
        Returns:
            Dictionary mapping chunk numbers to lists of (global_idx, local_idx) tuples
        """
        chunk_map = {}
        
        for global_idx in global_indices:
            chunk_num, local_idx = self.global_to_chunk_index(global_idx)
            
            if chunk_num not in chunk_map:
                chunk_map[chunk_num] = []
            
            chunk_map[chunk_num].append((global_idx, local_idx))
        
        return chunk_map
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embeddings."""
        if self.is_chunked:
            return {
                'is_chunked': True,
                'num_chunks': self.chunk_info['num_chunks'],
                'total_samples': self.total_samples,
                'embedding_dim': self.embedding_dim,
                'chunk_info': self.chunk_info
            }
        else:
            # Load info from single file
            data = torch.load(self.embeddings_path, map_location=self.device, weights_only=True)
            if 'normalized_embeddings' in data:
                embeddings = data['normalized_embeddings']
            else:
                embeddings = data['embeddings']
            
            info = {
                'is_chunked': False,
                'total_samples': embeddings.shape[0],
                'embedding_dim': embeddings.shape[1],
                'data_keys': list(data.keys())
            }
            
            # Clean up
            del data, embeddings
            gc.collect()
            
            return info
    
    def load_specific_embeddings(self, global_indices: List[int]) -> torch.Tensor:
        """
        Load specific embeddings by their global indices.
        Memory-efficient: only loads the necessary chunks or uses memmap.
        
        Args:
            global_indices: List of global indices to load
            
        Returns:
            Tensor containing the requested embeddings in the same order as global_indices
        """
        # Use memmap loader if available (fastest!)
        if self.memmap_loader is not None:
            embeddings = self.memmap_loader.load_specific_embeddings(global_indices)
            return embeddings.to(self.device)
        
        if not self.is_chunked:
            # For non-chunked embeddings, load all and index
            data = torch.load(self.embeddings_path, map_location=self.device)
            if 'normalized_embeddings' in data:
                embeddings = data['normalized_embeddings']
            else:
                embeddings = data['embeddings']
            
            selected_embeddings = embeddings[global_indices]
            
            # Clean up
            del data, embeddings
            gc.collect()
            
            return selected_embeddings
        
        # For chunked embeddings, load only necessary chunks
        chunk_map = self.global_indices_to_chunk_map(global_indices)
        
        # Create result tensor
        result_embeddings = torch.empty(len(global_indices), self.embedding_dim, 
                                      device=self.device, dtype=torch.float32)
        
        # Create mapping from global_idx to result position
        global_to_result_pos = {global_idx: i for i, global_idx in enumerate(global_indices)}
        
        # Load embeddings from each required chunk (with GPU caching)
        for chunk_num, chunk_indices in chunk_map.items():
            # Check GPU cache first (3-5x speedup)
            if chunk_num in self.gpu_cache:
                chunk_embeddings = self.gpu_cache[chunk_num]
                self.cache_hits += 1
            else:
                # Cache miss - load from disk
                chunk_path = os.path.join(self.chunks_dir, self.chunk_info['chunks'][chunk_num]['file'])
                chunk_data = torch.load(chunk_path, map_location=self.device, weights_only=False)
                
                # Handle both tensor and dict formats
                if isinstance(chunk_data, torch.Tensor):
                    chunk_embeddings = chunk_data
                elif isinstance(chunk_data, dict):
                    if 'normalized_embeddings' in chunk_data:
                        chunk_embeddings = chunk_data['normalized_embeddings']
                    else:
                        chunk_embeddings = chunk_data['embeddings']
                else:
                    raise ValueError(f"Unexpected chunk data type: {type(chunk_data)}")
                
                # Add to GPU cache if there's room
                if len(self.gpu_cache) < self.max_cached_chunks:
                    self.gpu_cache[chunk_num] = chunk_embeddings
                else:
                    # Evict oldest chunk (simple FIFO policy)
                    oldest_chunk = next(iter(self.gpu_cache))
                    del self.gpu_cache[oldest_chunk]
                    self.gpu_cache[chunk_num] = chunk_embeddings
                
                self.cache_misses += 1
                if 'chunk_data' in locals():
                    del chunk_data
            
            # Extract required embeddings from this chunk
            for global_idx, local_idx in chunk_indices:
                result_pos = global_to_result_pos[global_idx]
                result_embeddings[result_pos] = chunk_embeddings[local_idx]
            
            # Clean up chunk - only delete if they exist (not from cache)
            if 'chunk_data' in locals():
                del chunk_data
            del chunk_embeddings  # This always exists
            gc.collect()
        
        return result_embeddings
    
    def load_full_embeddings(self) -> torch.Tensor:
        """
        Load all embeddings into memory at once.
        WARNING: Use only for small files or when you have enough memory.
        """
        if not self.is_chunked:
            # Load single file
            data = torch.load(self.embeddings_path, map_location=self.device)
            if 'normalized_embeddings' in data:
                embeddings = data['normalized_embeddings']
            else:
                embeddings = data['embeddings']
            return embeddings
        else:
            # Load and concatenate all chunks
            # print(f"   WARNING: Loading all chunks into memory ({self.chunk_info['num_chunks']} chunks)")
            
            all_embeddings = []
            for i in range(self.chunk_info['num_chunks']):
                chunk_path = os.path.join(self.chunks_dir, self.chunk_info['chunks'][i]['file'])
                chunk_data = torch.load(chunk_path, map_location=self.device, weights_only=True)
                
                if 'normalized_embeddings' in chunk_data:
                    chunk_embeddings = chunk_data['normalized_embeddings']
                else:
                    chunk_embeddings = chunk_data['embeddings']
                
                all_embeddings.append(chunk_embeddings)
                
                # Clean up
                del chunk_data
                gc.collect()
            
            return torch.cat(all_embeddings, dim=0)
    
    def iter_chunks(self, chunk_size: Optional[int] = None) -> Iterator[Tuple[torch.Tensor, int, int]]:
        """
        Iterate over embedding chunks.
        
        Args:
            chunk_size: Size of chunks to yield (only used for non-chunked files)
            
        Yields:
            Tuple of (embeddings_chunk, start_idx, end_idx)
        """
        # Use memmap if available
        if self.memmap_loader is not None:
            if chunk_size is None:
                chunk_size = 50000  # Default chunk size for memmap
            
            total_samples = self.memmap_loader.shape[0]
            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)
                indices = list(range(start_idx, end_idx))
                chunk_embeddings = self.memmap_loader.load_specific_embeddings(indices).to(self.device)
                yield chunk_embeddings, start_idx, end_idx
                del chunk_embeddings
                gc.collect()
            return
            
        if self.is_chunked:
            # Iterate over existing chunks
            for i, chunk_info in enumerate(self.chunk_info['chunks']):
                chunk_path = os.path.join(self.chunks_dir, chunk_info['file'])
                # Use mmap_mode=None to prevent memory mapping issues
                with open(chunk_path, 'rb') as f:
                    chunk_data = torch.load(f, map_location=self.device, weights_only=True)
                
                # Handle both dictionary and tensor formats
                if isinstance(chunk_data, dict):
                    if 'normalized_embeddings' in chunk_data:
                        chunk_embeddings = chunk_data['normalized_embeddings'].clone()
                    else:
                        chunk_embeddings = chunk_data['embeddings'].clone()
                else:
                    # Direct tensor format (e.g., from SAE embeddings)
                    chunk_embeddings = chunk_data.clone()
                
                start_idx = chunk_info['start_idx']
                end_idx = chunk_info['end_idx']
                
                yield chunk_embeddings, start_idx, end_idx
                
                # Clean up - only delete the original data
                # The yielded tensor will be cleaned up by the caller
                del chunk_data
                gc.collect()
        else:
            # Split single file into chunks on-the-fly
            data = torch.load(self.embeddings_path, map_location=self.device)
            if 'normalized_embeddings' in data:
                embeddings = data['normalized_embeddings']
            else:
                embeddings = data['embeddings']
            
            if chunk_size is None:
                # Return entire file as single chunk
                chunk_embeddings = embeddings.to(self.device)
                yield chunk_embeddings, 0, embeddings.shape[0]
            else:
                # Split into chunks of specified size
                total_samples = embeddings.shape[0]
                for start_idx in range(0, total_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_samples)
                    chunk_embeddings = embeddings[start_idx:end_idx].to(self.device)
                    yield chunk_embeddings, start_idx, end_idx
                    
                    # Clean up
                    del chunk_embeddings
                    gc.collect()
            
            # Clean up full embeddings
            del data, embeddings
            gc.collect()

def process_embeddings_chunked(
    dataset_name: str,
    embeddings_file: str,
    processing_func: Callable[[torch.Tensor, int, int], Any],
    aggregation_func: Callable[[List[Any]], Any],
    scratch_dir: str = '',
    device: str = 'cuda',
    chunk_size: Optional[int] = None,
    show_progress: bool = True
) -> Any:
    """
    Process embeddings in chunks and aggregate results.
    
    Args:
        dataset_name: Name of the dataset
        embeddings_file: Filename of embeddings
        processing_func: Function to process each chunk (chunk, start_idx, end_idx) -> result
        aggregation_func: Function to aggregate all results from chunks
        scratch_dir: Base scratch directory
        device: Device to use for processing
        chunk_size: Chunk size for non-chunked files
        show_progress: Whether to show progress bar
        
    Returns:
        Aggregated result from all chunks
    """
    loader = ChunkedEmbeddingLoader(dataset_name, embeddings_file, scratch_dir, device)
    results = []
    
    chunk_iter = loader.iter_chunks(chunk_size)
    if show_progress:
        if loader.is_chunked:
            chunk_iter = tqdm(chunk_iter, total=loader.chunk_info['num_chunks'], 
                            desc="Processing chunks")
        else:
            chunk_iter = tqdm(chunk_iter, desc="Processing chunks")
    
    for chunk_embeddings, start_idx, end_idx in chunk_iter:
        result = processing_func(chunk_embeddings, start_idx, end_idx)
        results.append(result)
        
        # Force memory cleanup
        del chunk_embeddings
        gc.collect()
        
        # Force GPU memory cleanup if using CUDA
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    return aggregation_func(results)

def compute_activations_chunked(
    dataset_name: str,
    embeddings_file: str,
    concept_vectors: Dict[str, torch.Tensor],
    sample_ranges: List[Tuple[int, int]],
    scratch_dir: str = '',
    device: str = 'cuda',
    method: str = 'avg',
    show_progress: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Compute activations between embeddings and concept vectors using chunked processing.
    
    Args:
        dataset_name: Name of the dataset
        embeddings_file: Filename of embeddings
        concept_vectors: Dictionary mapping concept names to vectors
        sample_ranges: List of (start_idx, end_idx) for each sample
        scratch_dir: Base scratch directory
        device: Device for computation
        method: 'avg' for cosine similarity, 'linsep' for signed distances
        show_progress: Whether to show progress
        
    Returns:
        Dictionary mapping concept names to activation tensors
    """
    
    def process_chunk(chunk_embeddings, chunk_start_idx, chunk_end_idx):
        """Process a single chunk of embeddings."""
        chunk_activations = {}
        
        for concept_name, concept_vector in concept_vectors.items():
            concept_vector = concept_vector.to(device)
            
            if method == 'linsep':
                # Signed distances for linear separator
                activations = torch.matmul(chunk_embeddings, concept_vector)
            else:
                # Cosine similarities for average concepts
                activations = F.cosine_similarity(
                    chunk_embeddings,
                    concept_vector.unsqueeze(0).expand_as(chunk_embeddings),
                    dim=1
                )
            
            chunk_activations[concept_name] = activations  # Keep on GPU for faster processing
        
        return chunk_activations, chunk_start_idx, chunk_end_idx
    
    def aggregate_results(results):
        """Aggregate activation results from all chunks."""
        all_activations = {}
        
        # Initialize activation lists for each concept
        for concept_name in concept_vectors.keys():
            all_activations[concept_name] = []
        
        # Sort results by start index to maintain order
        results.sort(key=lambda x: x[1])
        
        # Concatenate activations for each concept
        for chunk_activations, _, _ in results:
            for concept_name, activations in chunk_activations.items():
                all_activations[concept_name].append(activations)
        
        # Concatenate tensors for each concept
        final_activations = {}
        for concept_name, activation_list in all_activations.items():
            final_activations[concept_name] = torch.cat(activation_list, dim=0).to(device)
        
        return final_activations
    
    return process_embeddings_chunked(
        dataset_name=dataset_name,
        embeddings_file=embeddings_file,
        processing_func=process_chunk,
        aggregation_func=aggregate_results,
        scratch_dir=scratch_dir,
        device=device,
        show_progress=show_progress
    )

def compute_hybrid_activations_chunked(
    dataset_name: str,
    embeddings_file: str,
    hybrid_concepts: Dict[str, List[torch.Tensor]],
    sample_ranges: List[Tuple[int, int]],
    scratch_dir: str = '',
    device: str = 'cuda',
    method: str = 'avg',
    show_progress: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Compute activations between embeddings and per-sample hybrid concept vectors using chunked processing.
    
    Args:
        dataset_name: Name of the dataset
        embeddings_file: Filename of embeddings
        hybrid_concepts: Dictionary mapping concept names to lists of hybrid vectors (one per sample)
        sample_ranges: List of (start_idx, end_idx) for each sample
        scratch_dir: Base scratch directory
        device: Device for computation
        method: 'avg' for cosine similarity, 'linsep' for signed distances
        show_progress: Whether to show progress
        
    Returns:
        Dictionary mapping concept names to activation tensors
    """
    
    def process_chunk(chunk_embeddings, chunk_start_idx, chunk_end_idx):
        """Process a single chunk of embeddings, respecting sample boundaries."""
        # CRITICAL: We need to maintain the exact same order as the original
        # by processing samples in order and only including the parts that fall within this chunk
        
        chunk_activations = {}
        
        for concept_name, concept_hybrid_vectors in hybrid_concepts.items():
            concept_activation_parts = []
            
            # Process samples in order to maintain global indexing
            for sample_idx, (sample_start, sample_end) in enumerate(sample_ranges):
                # Check if sample overlaps with this chunk
                overlap_start = max(sample_start, chunk_start_idx)
                overlap_end = min(sample_end, chunk_end_idx)
                
                if overlap_start < overlap_end:
                    # Get embeddings for this sample within the chunk
                    # These indices are relative to the chunk
                    sample_chunk_start = overlap_start - chunk_start_idx
                    sample_chunk_end = overlap_end - chunk_start_idx
                    sample_embeddings = chunk_embeddings[sample_chunk_start:sample_chunk_end]
                    
                    # Get hybrid vector for this sample
                    sample_hybrid_vector = concept_hybrid_vectors[sample_idx].to(device)
                    
                    if method == 'linsep':
                        # Signed distances for linear separator
                        sample_activations = torch.matmul(sample_embeddings, sample_hybrid_vector)
                    else:
                        # Cosine similarities for average concepts
                        sample_activations = F.cosine_similarity(
                            sample_embeddings,
                            sample_hybrid_vector.unsqueeze(0).expand_as(sample_embeddings),
                            dim=1
                        )
                    
                    concept_activation_parts.append(sample_activations.cpu())
            
            # Concatenate all activation parts for this concept in this chunk
            # This maintains the global ordering because we processed samples in order
            if concept_activation_parts:
                chunk_activations[concept_name] = torch.cat(concept_activation_parts, dim=0)
            else:
                # No activations for this concept in this chunk
                chunk_activations[concept_name] = torch.empty(0, device='cpu')
        
        return chunk_activations, chunk_start_idx, chunk_end_idx
    
    def aggregate_results(results):
        """Aggregate activation results from all chunks."""
        all_activations = {}
        
        # Initialize activation lists for each concept
        for concept_name in hybrid_concepts.keys():
            all_activations[concept_name] = []
        
        # Sort results by start index to maintain order
        results.sort(key=lambda x: x[1])
        
        # Concatenate activations for each concept
        for chunk_activations, _, _ in results:
            for concept_name, activations in chunk_activations.items():
                if len(activations) > 0:  # Only add non-empty tensors
                    all_activations[concept_name].append(activations)
        
        # Concatenate tensors for each concept
        final_activations = {}
        for concept_name, activation_list in all_activations.items():
            if activation_list:
                final_activations[concept_name] = torch.cat(activation_list, dim=0).to(device)
            else:
                final_activations[concept_name] = torch.empty(0, device=device)
        
        return final_activations
    
    return process_embeddings_chunked(
        dataset_name=dataset_name,
        embeddings_file=embeddings_file,
        processing_func=process_chunk,
        aggregation_func=aggregate_results,
        scratch_dir=scratch_dir,
        device=device,
        show_progress=show_progress
    )

def load_embeddings_by_indices(dataset_name: str, embeddings_file: str, global_indices: List[int], 
                              scratch_dir: str = '', device: str = 'cuda') -> torch.Tensor:
    """
    Convenience function to load specific embeddings by their global indices.
    
    Args:
        dataset_name: Name of the dataset
        embeddings_file: Filename of embeddings
        global_indices: List of global indices to load
        scratch_dir: Base scratch directory
        device: Device to load embeddings on
        
    Returns:
        Tensor containing the requested embeddings
    """
    loader = ChunkedEmbeddingLoader(dataset_name, embeddings_file, scratch_dir, device)
    return loader.load_specific_embeddings(global_indices)

def convert_global_to_chunk_indices(dataset_name: str, embeddings_file: str, global_indices: List[int],
                                  scratch_dir: str = '') -> Dict[int, List[Tuple[int, int]]]:
    """
    Convenience function to convert global indices to chunk mapping.
    
    Args:
        dataset_name: Name of the dataset
        embeddings_file: Filename of embeddings
        global_indices: List of global indices
        scratch_dir: Base scratch directory
        
    Returns:
        Dictionary mapping chunk numbers to lists of (global_idx, local_idx) tuples
    """
    loader = ChunkedEmbeddingLoader(dataset_name, embeddings_file, scratch_dir, 'cpu')
    return loader.global_indices_to_chunk_map(global_indices)

@contextmanager
def memory_efficient_context(device: str = 'cuda'):
    """
    Context manager for memory-efficient processing.
    Automatically cleans up memory at the end.
    """
    try:
        yield
    finally:
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

def estimate_memory_usage(embedding_shape: Tuple[int, int], dtype_size: int = 4) -> float:
    """
    Estimate memory usage in GB for embeddings.
    
    Args:
        embedding_shape: (num_samples, embedding_dim)
        dtype_size: Size of data type in bytes (4 for float32)
        
    Returns:
        Estimated memory usage in GB
    """
    total_elements = embedding_shape[0] * embedding_shape[1]
    total_bytes = total_elements * dtype_size
    return total_bytes / (1024 ** 3)

def get_optimal_chunk_size(embedding_shape: Tuple[int, int], max_memory_gb: float = 8.0, dtype_size: int = 4) -> int:
    """
    Calculate optimal chunk size based on available memory.
    
    Args:
        embedding_shape: (num_samples, embedding_dim)
        max_memory_gb: Maximum memory to use in GB
        dtype_size: Size of data type in bytes
        
    Returns:
        Optimal chunk size (number of samples)
    """
    total_samples, embedding_dim = embedding_shape
    
    # Calculate how many samples fit in the memory limit
    max_bytes = max_memory_gb * (1024 ** 3)
    bytes_per_sample = embedding_dim * dtype_size
    
    chunk_size = int(max_bytes / bytes_per_sample)
    
    # Ensure chunk size is at least 1 and not more than total samples
    chunk_size = max(1, min(chunk_size, total_samples))
    
    return chunk_size

def check_chunked_embeddings_status(embeddings_path: str) -> Dict[str, Any]:
    """
    Check the status of chunked embeddings and provide diagnostics.
    
    Args:
        embeddings_path: Path to embedding file
        
    Returns:
        Dictionary with status information
    """
    base_name = os.path.splitext(embeddings_path)[0]
    chunk_info_file = f"{base_name}_chunks_info.json"
    
    status = {
        'original_file_exists': os.path.exists(embeddings_path),
        'chunk_info_exists': os.path.exists(chunk_info_file),
        'is_chunked': False,
        'all_chunks_exist': False,
        'missing_chunks': [],
        'chunk_count': 0,
        'total_size_gb': 0.0
    }
    
    if status['chunk_info_exists']:
        with open(chunk_info_file, 'r') as f:
            chunk_info = json.load(f)
        
        status['is_chunked'] = True
        status['chunk_count'] = chunk_info['num_chunks']
        
        # Check if all chunks exist
        chunks_dir = os.path.dirname(embeddings_path)
        missing_chunks = []
        total_size = 0
        
        for i, chunk_data in enumerate(chunk_info['chunks']):
            chunk_path = os.path.join(chunks_dir, chunk_data['file'])
            if os.path.exists(chunk_path):
                total_size += os.path.getsize(chunk_path)
            else:
                missing_chunks.append(i)
        
        status['missing_chunks'] = missing_chunks
        status['all_chunks_exist'] = len(missing_chunks) == 0
        status['total_size_gb'] = total_size / (1024 ** 3)
    
    elif status['original_file_exists']:
        # Get size of original file
        file_size = os.path.getsize(embeddings_path)
        status['total_size_gb'] = file_size / (1024 ** 3)
    
    return status

def print_chunked_embedding_summary(embeddings_path: str):
    """Print a summary of chunked embedding status."""
    status = check_chunked_embeddings_status(embeddings_path)
    
    print(f"Embedding Status: {os.path.basename(embeddings_path)}")
    print(f"   Original file exists: {'Yes' if status['original_file_exists'] else 'No'}")
    print(f"   Is chunked: {'Yes' if status['is_chunked'] else 'No'}")
    
    if status['is_chunked']:
        print(f"   Chunk count: {status['chunk_count']}")
        print(f"   All chunks exist: {'Yes' if status['all_chunks_exist'] else 'No'}")
        if status['missing_chunks']:
            print(f"   Missing chunks: {status['missing_chunks']}")
    
    print(f"   Total size: {status['total_size_gb']:.2f} GB")


class ChunkedActivationLoader:
    """
    Loader for chunked activation files (cosine similarities or distances).
    Only supports PyTorch tensor (.pt) formats.
    Provides memory-efficient access to activation metrics without loading full files.
    """
    
    def __init__(self, dataset_name: str, acts_file: str, scratch_dir: str = '', device: str = 'cuda'):
        """
        Initialize the chunked activation loader.
        
        Args:
            dataset_name: Name of dataset (e.g., 'CLEVR')
            acts_file: Name of activation file (e.g., 'cosine_similarities_avg_concepts_Llama_patch_embeddings_percentthrumodel_100.pt')
            scratch_dir: Base scratch directory (e.g., '/path/to/scratch/')
            device: Device to load tensors on ('cuda' or 'cpu')
        """
        self.dataset_name = dataset_name
        self.acts_file = acts_file
        self.scratch_dir = scratch_dir
        self.device = device
        
        # GPU cache for activation chunks (3-5x speedup)
        self.gpu_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cached_chunks = 4  # Cache 4x9GB chunks on 47GB GPU (~36GB cache, ~11GB free)
        
        self.is_chunked = False
        self.chunk_files = []
        self.total_samples = 0
        self.chunk_size = None  # Will be determined from actual chunks
        self.columns = []
        self.file_format = 'pt'  # Only PT supported
        
        # Detect folder type from filename
        if 'superpatch' in acts_file:
            self.folder = "Superpatches"
        elif 'sae_acts' in acts_file:
            self.folder = "SAE_Acts"
        elif ('clipscope' in acts_file or 'gemmascope' in acts_file) and 'dense' in acts_file:
            self.folder = "SAE_Activations_Dense"
        elif 'dists_' in acts_file or 'linsep' in acts_file:
            self.folder = "Distances"
        else:
            self.folder = "Cosine_Similarities"
        
        self.base_path = os.path.join(scratch_dir, self.folder, dataset_name)
        
        # Force .pt extension
        if not acts_file.endswith('.pt'):
            self.acts_file = acts_file + '.pt'
        else:
            self.acts_file = acts_file
        self.file_format = 'pt'
        
        self.full_file_path = os.path.join(self.base_path, self.acts_file)
        
        # Detect chunked files
        self._detect_chunked_files()
    
    def get_cache_stats(self):
        """Get GPU cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.2%}",
            'cached_chunks': len(self.gpu_cache),
            'max_chunks': self.max_cached_chunks
        }
        
    def _detect_chunked_files(self):
        """Detect if activation files are split into chunks."""
        # Look for chunk files
        base_name = self.acts_file.replace('.pt', '')
        chunk_0_file = os.path.join(self.base_path, f"{base_name}_chunk_0.pt")
        chunk_info_file = os.path.join(self.base_path, f"{base_name}_chunks_info.json")
        
        if os.path.exists(chunk_0_file):
            self.is_chunked = True
            
            if chunk_info_file and os.path.exists(chunk_info_file):
                # For PT format, use chunk info file if available
                with open(chunk_info_file, 'r') as f:
                    chunk_info = json.load(f)
                self.total_samples = chunk_info['total_samples']
                
                # Handle different chunk info formats
                if 'concept_names' in chunk_info:
                    self.columns = chunk_info['concept_names']
                elif 'feature_dim' in chunk_info:
                    # SAE format - generate column names based on feature dimension
                    num_features = chunk_info['feature_dim']
                    if 'sae' in self.acts_file or 'dense' in self.acts_file:
                        self.columns = [str(i) for i in range(num_features)]
                    else:
                        self.columns = [f'concept_{i}' for i in range(num_features)]
                else:
                    # Fall back to loading first chunk to determine columns
                    self.columns = None  # Will be set later when loading chunks
                
                # Get chunk size from chunk info or actual chunk files
                if 'chunk_size' in chunk_info:
                    self.chunk_size = chunk_info['chunk_size']
                else:
                    # We'll determine chunk size after finding chunk files
                    self.chunk_size = None
                
                # Find all chunk files
                for i in range(chunk_info['num_chunks']):
                    chunk_file = os.path.join(self.base_path, f"{base_name}_chunk_{i}.pt")
                    self.chunk_files.append(chunk_file)
                
                # If chunk_size wasn't in info, determine from first chunk
                if self.chunk_size is None and self.chunk_files:
                    first_chunk = torch.load(self.chunk_files[0], map_location=self.device, weights_only=True)
                    if isinstance(first_chunk, torch.Tensor):
                        self.chunk_size = first_chunk.shape[0]
                    elif isinstance(first_chunk, dict):
                        self.chunk_size = first_chunk['activations'].shape[0]
                    else:
                        raise ValueError(f"Cannot determine chunk size from chunk format")
                    del first_chunk
                    gc.collect()
            else:
                # Find all chunk files
                chunk_idx = 0
                while True:
                    chunk_file = os.path.join(self.base_path, f"{base_name}_chunk_{chunk_idx}.pt")
                    
                    if not os.path.exists(chunk_file):
                        break
                    self.chunk_files.append(chunk_file)
                    chunk_idx += 1
                
                # For PT files without chunk info, load first chunk to get metadata
                if self.chunk_files:
                    first_chunk = torch.load(self.chunk_files[0], map_location=self.device, weights_only=True)
                    
                    # Handle different formats
                    if isinstance(first_chunk, torch.Tensor):
                        # SAE format - raw tensor
                        num_units = first_chunk.shape[1]
                        if 'sae_acts' in self.acts_file:
                            self.columns = [str(i) for i in range(num_units)]
                        else:
                            self.columns = [str(i) for i in range(num_units)]
                        # Count total samples from all chunks and infer chunk size
                        total_samples = 0
                        self.chunk_size = first_chunk.shape[0]
                        for chunk_file in self.chunk_files:
                            chunk_data = torch.load(chunk_file, map_location=self.device, weights_only=True)
                            total_samples += chunk_data.shape[0]
                        self.total_samples = total_samples
                    elif isinstance(first_chunk, dict):
                        # Regular format with metadata
                        self.columns = first_chunk['concept_names']
                        # Count total samples from all chunks and infer chunk size
                        total_samples = 0
                        self.chunk_size = first_chunk['activations'].shape[0]
                        for chunk_file in self.chunk_files:
                            chunk_data = torch.load(chunk_file, map_location=self.device, weights_only=True)
                            total_samples += chunk_data['activations'].shape[0]
                        self.total_samples = total_samples
                    else:
                        raise ValueError(f"Unexpected chunk format in {self.chunk_files[0]}")
            # print(f"   Detected chunked activation file: {len(self.chunk_files)} chunks")
            # print(f"   Total samples: {self.total_samples:,}, Concepts: {len(self.columns)}")
            
        elif os.path.exists(self.full_file_path):
            # Non-chunked file exists - support it for small files like SAE dense
            self.is_chunked = False
            # Load the file to get metadata
            data = torch.load(self.full_file_path, map_location=self.device, weights_only=True)
            
            if isinstance(data, torch.Tensor):
                # SAE format - raw tensor
                self.total_samples = data.shape[0]
                num_units = data.shape[1]
                if 'sae_acts' in self.acts_file or 'sae' in self.acts_file:
                    self.columns = [str(i) for i in range(num_units)]
                else:
                    self.columns = [str(i) for i in range(num_units)]
            elif isinstance(data, dict):
                # Regular format with metadata
                self.total_samples = data['activations'].shape[0]
                self.columns = data['concept_names']
            else:
                raise ValueError(f"Unexpected file format in {self.full_file_path}")
            
            # For non-chunked files, treat as single chunk
            self.chunk_files = [self.full_file_path]
            self.chunk_size = self.total_samples
        else:
            # Check if this is a chunking issue
            chunk_0_file = os.path.join(self.base_path, f"{base_name}_chunk_0.pt")
            if not os.path.exists(chunk_0_file):
                raise FileNotFoundError(
                    f"No chunked activation files found.\n"
                    f"Expected to find: {chunk_0_file}\n"
                    f"Please ensure the activation files are generated in chunked format."
                )
    
    def load_chunk_range(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Load activation data for a specific range of indices.
        
        Args:
            start_idx: Starting global index
            end_idx: Ending global index (exclusive)
            
        Returns:
            Tensor with activation data for the specified range
        """
        if not self.is_chunked:
            # Load from single file
            # For SAE activations, need weights_only=False since they're saved as regular tensors
            try:
                data = torch.load(self.full_file_path, map_location=self.device, weights_only=True)
            except:
                data = torch.load(self.full_file_path, map_location=self.device, weights_only=False)
            
            # Handle different formats
            if isinstance(data, torch.Tensor):
                # SAE activations are raw tensors
                activations = data[start_idx:end_idx]
            elif isinstance(data, dict):
                # Regular activations with metadata
                activations = data['activations'][start_idx:end_idx]
            else:
                raise ValueError(f"Unexpected data format in {self.full_file_path}")
                
            # Move to CPU if needed for compatibility
            if activations.is_cuda and self.device == 'cpu':
                activations = activations.cpu()
            
            # Clean up
            del data
            gc.collect()
            
            return activations
        
        # Load from chunked files
        chunks_to_load = []
        current_offset = 0
        
        # For PT files, we need to load chunk metadata to know sizes
        for chunk_file in self.chunk_files:
            chunk_data = torch.load(chunk_file, map_location=self.device, weights_only=True)
            
            # Handle different formats
            if isinstance(chunk_data, torch.Tensor):
                # SAE format - raw tensor
                chunk_size = chunk_data.shape[0]
                chunk_activations_full = chunk_data
            elif isinstance(chunk_data, dict):
                # Regular format with metadata
                chunk_size = chunk_data['activations'].shape[0]
                chunk_activations_full = chunk_data['activations']
            else:
                raise ValueError(f"Unexpected chunk format in {chunk_file}")
            
            chunk_start = current_offset
            chunk_end = current_offset + chunk_size
            
            # Check if this chunk overlaps with our desired range
            if chunk_start < end_idx and chunk_end > start_idx:
                # Calculate which rows to load from this chunk
                local_start = max(0, start_idx - chunk_start)
                local_end = min(chunk_size, end_idx - chunk_start)
                
                if local_end > local_start:
                    chunk_activations = chunk_activations_full[local_start:local_end]
                    # Move to CPU if needed
                    if chunk_activations.is_cuda and self.device == 'cpu':
                        chunk_activations = chunk_activations.cpu()
                    chunks_to_load.append(chunk_activations)
            
            current_offset = chunk_end
            
            # Clean up
            del chunk_data, chunk_activations_full
            gc.collect()
            
            # Early termination if we've loaded everything we need
            if current_offset >= end_idx:
                break
        
        if chunks_to_load:
            # Concatenate all loaded chunks
            result_tensor = torch.cat(chunks_to_load, dim=0)
            # Clear chunks from memory
            del chunks_to_load
            gc.collect()
            return result_tensor
        else:
            # Return empty tensor with correct shape
            return torch.zeros((0, len(self.columns)), device=self.device)
    
    def load_full_dataframe(self) -> pd.DataFrame:
        """
        Load the complete activation dataframe.
        WARNING: This may use a lot of memory for large files.
        """
        if not self.is_chunked:
            data = torch.load(self.full_file_path, map_location=self.device, weights_only=True)
            
            # Handle different formats
            if isinstance(data, torch.Tensor):
                # SAE activations are raw tensors
                df = pd.DataFrame(data.numpy(), columns=self.columns)
            elif isinstance(data, dict):
                # Regular activations with metadata
                df = pd.DataFrame(data['activations'].numpy(), columns=self.columns)
            else:
                raise ValueError(f"Unexpected data format in {self.full_file_path}")
            
            # Clean up
            del data
            gc.collect()
            
            return df
        
        # Load and concatenate all chunks
        chunks = []
        for chunk_file in self.chunk_files:
            chunk_data = torch.load(chunk_file, map_location=self.device, weights_only=True)
            
            # Handle different formats
            if isinstance(chunk_data, torch.Tensor):
                # SAE format - raw tensor
                chunk_df = pd.DataFrame(chunk_data.numpy(), columns=self.columns)
            elif isinstance(chunk_data, dict):
                # Regular format with metadata
                chunk_df = pd.DataFrame(chunk_data['activations'].numpy(), columns=self.columns)
            else:
                raise ValueError(f"Unexpected chunk format in {chunk_file}")
            
            chunks.append(chunk_df)
            del chunk_data
            gc.collect()
        
        full_df = pd.concat(chunks, ignore_index=True)
        # Clear chunks from memory
        del chunks
        gc.collect()
        return full_df
    
    def load_specific_concepts(self, concept_names: List[str], return_tensor: bool = True) -> Union[pd.DataFrame, torch.Tensor]:
        """
        Load activation data for specific concepts/clusters only.
        Memory-efficient: only loads the necessary columns.
        
        Args:
            concept_names: List of concept/cluster names to load
            return_tensor: If True, return torch.Tensor (default). If False, return DataFrame (legacy)
            
        Returns:
            Tensor or DataFrame with activation data for the specified concepts only
        """
        # Filter concept names to only those that exist
        available_concepts = [name for name in concept_names if name in self.columns]
        if not available_concepts:
            print(f"Warning: None of the requested concepts {concept_names} found in data")
            if return_tensor:
                return torch.empty(0, 0, device=self.device)
            else:
                return pd.DataFrame()
        
        if not self.is_chunked:
            # Load from single file
            data = torch.load(self.full_file_path, map_location=self.device, weights_only=True)
            
            # Handle different formats
            if isinstance(data, torch.Tensor):
                # SAE format - raw tensor
                activations = data
            elif isinstance(data, dict):
                # Regular format with metadata
                activations = data['activations']
            else:
                raise ValueError(f"Unexpected data format in {self.full_file_path}")
                
            # Get indices of the concepts we want
            concept_indices = [self.columns.index(name) for name in available_concepts]
            selected_activations = activations[:, concept_indices]
            
            # Clean up
            del data, activations
            gc.collect()
            
            # Return tensor directly - NO DATAFRAME
            if return_tensor:
                return selected_activations
            else:
                # Only create DataFrame if explicitly requested (legacy compatibility)
                return pd.DataFrame(selected_activations.cpu().numpy(), columns=available_concepts)
        
        # Load from chunked files
        chunks_data = []
        
        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            # Check GPU cache first (3-5x speedup)
            if chunk_idx in self.gpu_cache:
                chunk_data = self.gpu_cache[chunk_idx]
                self.cache_hits += 1
            else:
                # Cache miss - load from disk
                chunk_data = torch.load(chunk_file, map_location=self.device, weights_only=True)
                
                # Add to GPU cache if there's room
                if len(self.gpu_cache) < self.max_cached_chunks:
                    self.gpu_cache[chunk_idx] = chunk_data
                else:
                    # Evict oldest chunk (simple FIFO policy)
                    oldest_chunk = next(iter(self.gpu_cache))
                    del self.gpu_cache[oldest_chunk]
                    self.gpu_cache[chunk_idx] = chunk_data
                
                self.cache_misses += 1
            
            # Handle different formats
            if isinstance(chunk_data, torch.Tensor):
                # SAE format - raw tensor
                chunk_activations = chunk_data
            elif isinstance(chunk_data, dict):
                # Regular format with metadata
                chunk_activations = chunk_data['activations']
            else:
                raise ValueError(f"Unexpected chunk format in {chunk_file}")
            
            # Get indices of the concepts we want
            concept_indices = [self.columns.index(name) for name in available_concepts]
            selected_activations = chunk_activations[:, concept_indices]
            
            if return_tensor:
                # Store tensors directly - NO DATAFRAME
                chunks_data.append(selected_activations)
            else:
                # Only create DataFrame if explicitly requested (legacy)
                chunk_df = pd.DataFrame(selected_activations.cpu().numpy(), columns=available_concepts)
                chunks_data.append(chunk_df)
            
            del chunk_data, chunk_activations
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        if chunks_data:
            if return_tensor:
                # Concatenate tensors directly on GPU
                result_tensor = torch.cat(chunks_data, dim=0)
                del chunks_data
                gc.collect()
                return result_tensor
            else:
                # Concatenate DataFrames (legacy)
                result_df = pd.concat(chunks_data, ignore_index=True)
                del chunks_data
                gc.collect()
                return result_df
        else:
            if return_tensor:
                return torch.empty(0, len(available_concepts), device=self.device)
            else:
                return pd.DataFrame(columns=available_concepts)
    
    def load_concept_range(self, concept_names: List[str], start_idx: int, end_idx: int) -> pd.DataFrame:
        """
        Load activation data for specific concepts and sample range.
        Most memory-efficient option.
        
        Args:
            concept_names: List of concept/cluster names to load
            start_idx: Starting sample index
            end_idx: Ending sample index (exclusive)
            
        Returns:
            DataFrame with activation data for the specified concepts and range
        """
        # Filter concept names to only those that exist
        available_concepts = [name for name in concept_names if name in self.columns]
        if not available_concepts:
            print(f"Warning: None of the requested concepts {concept_names} found in data")
            return pd.DataFrame()
        
        if not self.is_chunked:
            # Load from single file
            data = torch.load(self.full_file_path, map_location=self.device, weights_only=True)
            
            # Handle different formats
            if isinstance(data, torch.Tensor):
                # SAE format - raw tensor
                activations = data[start_idx:end_idx]
            elif isinstance(data, dict):
                # Regular format with metadata
                activations = data['activations'][start_idx:end_idx]
            else:
                raise ValueError(f"Unexpected data format in {self.full_file_path}")
                
            # Get indices of the concepts we want
            concept_indices = [self.columns.index(name) for name in available_concepts]
            selected_activations = activations[:, concept_indices]
            df = pd.DataFrame(selected_activations.numpy(), columns=available_concepts)
            
            # Clean up
            del data, activations
            gc.collect()
            
            return df
        
        # Load from chunked files
        chunks_to_load = []
        current_offset = 0
        
        # For PT files, we need to load chunk metadata to know sizes
        for chunk_file in self.chunk_files:
            chunk_data = torch.load(chunk_file, map_location=self.device, weights_only=True)
            
            # Handle different formats
            if isinstance(chunk_data, torch.Tensor):
                # SAE format - raw tensor
                chunk_size = chunk_data.shape[0]
                chunk_activations_full = chunk_data
            elif isinstance(chunk_data, dict):
                # Regular format with metadata
                chunk_size = chunk_data['activations'].shape[0]
                chunk_activations_full = chunk_data['activations']
            else:
                raise ValueError(f"Unexpected chunk format in {chunk_file}")
                
            chunk_start = current_offset
            chunk_end = current_offset + chunk_size
            
            # Check if this chunk overlaps with our desired range
            if chunk_start < end_idx and chunk_end > start_idx:
                # Calculate which rows to load from this chunk
                local_start = max(0, start_idx - chunk_start)
                local_end = min(chunk_size, end_idx - chunk_start)
                
                if local_end > local_start:
                    chunk_activations = chunk_activations_full[local_start:local_end]
                    # Get indices of the concepts we want
                    concept_indices = [self.columns.index(name) for name in available_concepts]
                    selected_activations = chunk_activations[:, concept_indices]
                    chunk_df = pd.DataFrame(selected_activations.cpu().numpy(), columns=available_concepts)
                    chunks_to_load.append(chunk_df)
            
            current_offset = chunk_end
            del chunk_data, chunk_activations_full
            gc.collect()
            
            # Early termination if we've loaded everything we need
            if current_offset >= end_idx:
                break
        
        if chunks_to_load:
            # Concatenate all loaded chunks
            result_df = pd.concat(chunks_to_load, ignore_index=True)
            # Clear chunks from memory
            del chunks_to_load
            gc.collect()
            return result_df
        else:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=available_concepts)
    
    def get_activation_info(self) -> Dict[str, Any]:
        """Get information about the activation data."""
        return {
            'total_samples': self.total_samples,
            'num_concepts': len(self.columns),
            'concept_names': self.columns,
            'is_chunked': self.is_chunked,
            'num_chunks': len(self.chunk_files) if self.is_chunked else 1,
            'file_type': 'distances' if 'dists' in self.acts_file else 'cosine_similarities'
        }
    
    def load_full_tensor(self) -> torch.Tensor:
        """
        Load the complete activation data as a tensor.
        More memory-efficient than load_full_dataframe as it avoids pandas conversion.
        
        Returns:
            torch.Tensor of shape (num_samples, num_concepts)
        """
        if not self.is_chunked:
            data = torch.load(self.full_file_path, map_location=self.device, weights_only=True)
            
            # Handle different formats
            if isinstance(data, torch.Tensor):
                # SAE activations are raw tensors
                return data
            elif isinstance(data, dict):
                # Regular activations with metadata
                return data['activations']
            else:
                raise ValueError(f"Unexpected data format in {self.full_file_path}")
        
        # Load and concatenate all chunks
        chunks = []
        for chunk_file in self.chunk_files:
            chunk_data = torch.load(chunk_file, map_location=self.device, weights_only=True)
            
            # Handle different formats
            if isinstance(chunk_data, torch.Tensor):
                chunks.append(chunk_data)
            elif isinstance(chunk_data, dict):
                chunks.append(chunk_data['activations'])
            else:
                raise ValueError(f"Unexpected chunk format in {chunk_file}")
            
            # Clean up chunk data
            del chunk_data
            gc.collect()
        
        # Concatenate all chunks
        full_tensor = torch.cat(chunks, dim=0)
        
        # Clear chunks from memory
        del chunks
        gc.collect()
        return full_tensor
    
    def load_split_tensor(self, split: str, dataset_name: str, model_input_size: tuple, patch_size: int = 14) -> torch.Tensor:
        """
        Load activation data for a specific split only (train/test/cal).
        Significantly reduces memory usage compared to loading full tensor.
        
        Args:
            split: Split to load ('train', 'test', or 'cal')
            dataset_name: Dataset name for loading split information
            model_input_size: Model input size
            patch_size: Patch size (default 14)
            
        Returns:
            torch.Tensor containing only the specified split's data
        """
        # Import here to avoid circular imports
        from utils.patch_alignment_utils import get_patch_split_df
        
        # Get split information
        split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
        split_indices = torch.tensor(split_df.index[split_df == split].tolist())
        
        if len(split_indices) == 0:
            print(f"Warning: No samples found for split '{split}'")
            return torch.empty(0, len(self.columns))
        
        # print(f"   Loading {split} split: {len(split_indices):,} samples out of {self.total_samples:,} total")
        
        # For non-chunked files, just index directly
        if not self.is_chunked:
            data = torch.load(self.full_file_path, map_location=self.device, weights_only=True)
            
            # Handle different formats
            if isinstance(data, torch.Tensor):
                result = data[split_indices]
            elif isinstance(data, dict):
                result = data['activations'][split_indices]
            else:
                raise ValueError(f"Unexpected data format in {self.full_file_path}")
            
            # Clean up
            del data
            gc.collect()
            return result
        
        # For chunked files, we need to load only the relevant chunks
        # First, sort indices for efficient chunk loading
        sorted_indices = torch.sort(split_indices)[0]
        
        # Determine chunk size from first chunk
        first_chunk_data = torch.load(self.chunk_files[0], map_location=self.device, weights_only=True)
        if isinstance(first_chunk_data, torch.Tensor):
            chunk_size = first_chunk_data.shape[0]
        elif isinstance(first_chunk_data, dict):
            chunk_size = first_chunk_data['activations'].shape[0]
        else:
            raise ValueError("Unexpected chunk format")
        del first_chunk_data
        gc.collect()
        
        # Collect data from relevant chunks
        result_tensors = []
        
        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, self.total_samples)
            
            # Find indices that fall within this chunk
            mask = (sorted_indices >= chunk_start) & (sorted_indices < chunk_end)
            chunk_indices = sorted_indices[mask]
            
            if len(chunk_indices) > 0:
                # Load this chunk
                chunk_data = torch.load(chunk_file, map_location=self.device, weights_only=True)
                
                # Extract activations
                if isinstance(chunk_data, torch.Tensor):
                    chunk_activations = chunk_data
                elif isinstance(chunk_data, dict):
                    chunk_activations = chunk_data['activations']
                else:
                    raise ValueError(f"Unexpected chunk format in {chunk_file}")
                
                # Convert global indices to local chunk indices
                local_indices = chunk_indices - chunk_start
                
                # Extract relevant rows
                result_tensors.append(chunk_activations[local_indices])
                
                # Clean up
                del chunk_data, chunk_activations
                gc.collect()
        
        # Concatenate all results
        if result_tensors:
            result = torch.cat(result_tensors, dim=0)
        else:
            result = torch.empty(0, len(self.columns))
        
        # Clear intermediate results
        del result_tensors
        gc.collect()
        
        return result
    
    def load_tensor_range(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Load activation data for a specific range as a tensor.
        
        Args:
            start_idx: Starting sample index
            end_idx: Ending sample index (exclusive)
            
        Returns:
            torch.Tensor of shape (end_idx - start_idx, num_concepts)
        """
        if not self.is_chunked:
            # Load from single file
            data = torch.load(self.full_file_path, map_location=self.device, weights_only=True)
            
            # Handle different formats
            if isinstance(data, torch.Tensor):
                # SAE format - raw tensor
                result = data[start_idx:end_idx]
            elif isinstance(data, dict):
                # Regular format with metadata
                result = data['activations'][start_idx:end_idx]
            else:
                raise ValueError(f"Unexpected data format in {self.full_file_path}")
            
            # Clean up
            del data
            gc.collect()
            return result
        
        # For chunked files, determine which chunks we need
        first_chunk = start_idx // self.chunk_size
        last_chunk = (end_idx - 1) // self.chunk_size
        
        result_parts = []
        for chunk_idx in range(first_chunk, last_chunk + 1):
            if chunk_idx >= len(self.chunk_files):
                break
                
            try:
                chunk_data = torch.load(self.chunk_files[chunk_idx], map_location=self.device, weights_only=True)
            except Exception as e:
                if "Weights only load failed" in str(e):
                    # Fall back to weights_only=False if needed
                    chunk_data = torch.load(self.chunk_files[chunk_idx], map_location=self.device, weights_only=False)
                else:
                    raise
            
            # Handle different formats
            if isinstance(chunk_data, torch.Tensor):
                chunk_activations = chunk_data
            elif isinstance(chunk_data, dict):
                chunk_activations = chunk_data['activations']
            else:
                raise ValueError(f"Unexpected chunk format")
            
            # Determine slice within this chunk
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = chunk_start + len(chunk_activations)
            
            # Calculate relative indices within this chunk
            rel_start = max(0, start_idx - chunk_start)
            rel_end = min(len(chunk_activations), end_idx - chunk_start)
            
            if rel_start < rel_end:
                result_parts.append(chunk_activations[rel_start:rel_end])
            
            # Clean up
            del chunk_data
            gc.collect()
        
        if result_parts:
            return torch.cat(result_parts, dim=0)
        else:
            # Return empty tensor with correct shape
            return torch.empty(0, len(self.columns))
    
    def __len__(self):
        """Return the total number of samples in the activation data."""
        return self.total_samples
    
    def get_concept_index(self, concept_name: str) -> int:
        """
        Get the column index for a given concept name.
        
        Args:
            concept_name: Name of the concept (e.g., '0', '1', etc. for clusters)
            
        Returns:
            Column index for the concept
            
        Raises:
            ValueError if concept not found
        """
        if concept_name in self.columns:
            return self.columns.index(concept_name)
        else:
            raise ValueError(f"Concept '{concept_name}' not found in columns")
    
    def get_concept_indices(self, concept_names: List[str]) -> List[int]:
        """
        Get column indices for multiple concept names.
        
        Args:
            concept_names: List of concept names
            
        Returns:
            List of column indices
        """
        return [self.get_concept_index(name) for name in concept_names]
    
    def load_tensor_for_concepts(self, concept_names: List[str], start_idx: int = None, end_idx: int = None) -> torch.Tensor:
        """
        Load activation data as a tensor for specific concepts only.
        
        Args:
            concept_names: List of concept names to load
            start_idx: Optional starting index
            end_idx: Optional ending index
            
        Returns:
            torch.Tensor of shape (num_samples, num_concepts)
        """
        # Get indices for requested concepts
        concept_indices = self.get_concept_indices(concept_names)
        
        # Load the appropriate range
        if start_idx is not None and end_idx is not None:
            full_tensor = self.load_tensor_range(start_idx, end_idx)
        else:
            full_tensor = self.load_full_tensor()
        
        # Return only requested concept columns
        return full_tensor[:, concept_indices]
    
    @property
    def values(self):
        """
        Property to provide DataFrame-like interface for tensor access.
        Returns the full tensor (memory intensive - use with caution).
        """
        return self.load_full_tensor()
    
    def load_concept_activations_for_indices(self, concept_name: str, indices: List[int], device: str = None) -> torch.Tensor:
        """
        Load activations for a specific concept at specific indices.
        Unified interface that handles both dense and sparse index loading.
        
        Args:
            concept_name: Name of the concept/column to load
            indices: List of indices to load
            device: Device to load tensors on (defaults to self.device)
            
        Returns:
            torch.Tensor with activations for the specified indices
        """
        if device is None:
            device = self.device
            
        if not indices:
            return torch.empty(0, device=device)
            
        # Get concept index
        if concept_name not in self.columns:
            raise ValueError(f"Concept {concept_name} not found in columns")
        concept_idx = self.columns.index(concept_name)
        
        # Sort indices for efficient loading
        sorted_indices = sorted(indices)
        min_idx = sorted_indices[0]
        max_idx = sorted_indices[-1] + 1
        
        # Load the range containing all indices
        range_tensor = self.load_tensor_range(min_idx, max_idx)
        
        # Extract activations for the concept
        concept_acts = range_tensor[:, concept_idx]
        
        # Map back to original indices
        relative_indices = [idx - min_idx for idx in indices]
        result = concept_acts[relative_indices].to(device)
        
        # Clean up
        del range_tensor, concept_acts
        gc.collect()
        
        return result
    
    def find_indices_above_threshold(self, concept_name: str, threshold: float, 
                                   chunk_size: int = 50000) -> List[int]:
        """
        Find all indices where activation for a concept exceeds threshold.
        Memory-efficient chunked processing.
        
        Args:
            concept_name: Name of the concept to check
            threshold: Threshold value
            chunk_size: Size of chunks to process at once
            
        Returns:
            List of indices where activation > threshold
        """
        if concept_name not in self.columns:
            raise ValueError(f"Concept {concept_name} not found in columns")
        concept_idx = self.columns.index(concept_name)
        
        indices_above_threshold = []
        
        # Process in chunks
        for chunk_start in range(0, self.total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.total_samples)
            
            # Load chunk
            chunk_tensor = self.load_tensor_range(chunk_start, chunk_end)
            chunk_acts = chunk_tensor[:, concept_idx].to(self.device)
            
            # Find indices above threshold
            mask = chunk_acts > threshold
            chunk_indices = torch.where(mask)[0] + chunk_start
            indices_above_threshold.extend(chunk_indices.tolist())
            
            # Clean up
            del chunk_tensor, chunk_acts
            torch.cuda.empty_cache()
        
        return indices_above_threshold


class MatchedConceptActivationLoader:
    """
    Wrapper around ChunkedActivationLoader that provides concept-matched functionality.
    Maps cluster IDs to concept names and provides the same interface as a matched DataFrame.
    """
    
    def __init__(self, activation_loader: ChunkedActivationLoader, cluster_to_concept_mapping: Dict[str, str]):
        """
        Initialize with a ChunkedActivationLoader and concept mapping.
        
        Args:
            activation_loader: ChunkedActivationLoader instance
            cluster_to_concept_mapping: Dict mapping cluster_id -> concept_name
        """
        self.activation_loader = activation_loader
        self.cluster_to_concept = cluster_to_concept_mapping
        self.concept_to_cluster = {v: k for k, v in cluster_to_concept_mapping.items()}
        
        # Get available matched clusters
        self.available_clusters = [cluster_id for cluster_id in cluster_to_concept_mapping.keys() 
                                  if cluster_id in activation_loader.columns]
        self.matched_concepts = [cluster_to_concept_mapping[cluster_id] for cluster_id in self.available_clusters]
        
    @property
    def columns(self):
        """Return available concept names (like DataFrame.columns)"""
        # For consistency with supervised methods, return concept names instead of cluster IDs
        return self.matched_concepts
    
    @property 
    def concept_names(self):
        """Return the concept names corresponding to available clusters"""
        return self.matched_concepts
        
    def __getitem__(self, concept_names):
        """
        Load activations for specific concept names (like DataFrame column selection).
        
        Args:
            concept_names: Single concept name (str) or list of concept names
            
        Returns:
            DataFrame with activations for the specified concepts
        """
        if isinstance(concept_names, str):
            concept_names = [concept_names]
        
        # Convert concept names to cluster IDs
        cluster_ids = [self.concept_to_cluster[name] for name in concept_names 
                      if name in self.concept_to_cluster]
        
        if not cluster_ids:
            print(f"Warning: None of the requested concepts {concept_names} found in mapping")
            return torch.empty(0, 0)
        
        return self.activation_loader.load_specific_concepts(cluster_ids)
    
    def load_by_concept_names(self, concept_names):
        """
        Load activations by concept names instead of cluster IDs.
        
        Args:
            concept_names: List of concept names
            
        Returns:
            DataFrame with activations, columns are cluster IDs
        """
        cluster_ids = [self.concept_to_cluster[name] for name in concept_names 
                      if name in self.concept_to_cluster]
        
        if not cluster_ids:
            print(f"Warning: None of the requested concepts {concept_names} found in mapping")
            return torch.empty(0, 0)
            
        return self.activation_loader.load_specific_concepts(cluster_ids)
    
    def load_range(self, concept_names, start_idx: int, end_idx: int):
        """
        Load activations for specific concept names and sample range.
        
        Args:
            concept_names: Single concept name (str) or list of concept names
            start_idx: Starting sample index  
            end_idx: Ending sample index (exclusive)
            
        Returns:
            Tensor with activations for the specified concepts and range
        """
        if isinstance(concept_names, str):
            concept_names = [concept_names]
        
        # Convert concept names to cluster IDs
        cluster_ids = [self.concept_to_cluster[name] for name in concept_names 
                      if name in self.concept_to_cluster]
        
        if not cluster_ids:
            print(f"Warning: None of the requested concepts {concept_names} found in mapping")
            return torch.empty(0, 0)
            
        return self.activation_loader.load_concept_range(cluster_ids, start_idx, end_idx)
    
    def get_concept_for_cluster(self, cluster_id: str) -> str:
        """Get the concept name for a cluster ID"""
        return self.cluster_to_concept.get(cluster_id, cluster_id)
    
    def get_cluster_for_concept(self, concept_name: str) -> str:
        """Get the cluster ID for a concept name"""
        return self.concept_to_cluster.get(concept_name, concept_name)
    
    def get_activation_info(self) -> Dict[str, Any]:
        """
        Get information about the activation data.
        Delegates to the underlying activation loader but updates concept names.
        
        Returns:
            Dictionary with total_samples, num_concepts, concept_names, is_chunked
        """
        info = self.activation_loader.get_activation_info()
        # Update to return only matched concepts
        info['num_concepts'] = len(self.matched_concepts)
        info['concept_names'] = self.matched_concepts
        return info
    
    def load_chunk_range(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Load activation data for a specific range of indices.
        Returns only matched concepts.
        
        Args:
            start_idx: Starting global index
            end_idx: Ending global index (exclusive)
            
        Returns:
            Tensor with only matched concept columns
        """
        # Load full chunk from underlying loader
        full_chunk = self.activation_loader.load_chunk_range(start_idx, end_idx)
        
        # Filter to only matched concepts
        if self.available_clusters:
            # Get indices of available clusters in the full columns list
            all_columns = self.activation_loader.columns
            cluster_indices = [all_columns.index(cluster) for cluster in self.available_clusters if cluster in all_columns]
            return full_chunk[:, cluster_indices]
        else:
            return torch.zeros((0, 0), device=full_chunk.device)
    
    def to_csv(self, filepath: str):
        """
        Save all matched activations to CSV (like DataFrame.to_csv).
        NOTE: This loads all matched concepts at once for backward compatibility.
        For memory-efficient operations, use the loader methods directly.
        """
        if self.available_clusters:
            df = self.activation_loader.load_specific_concepts(self.available_clusters)
            df.to_csv(filepath)
        else:
            print("No matched clusters available to save")
            
    def load_concept_range(self, concept_names, start_idx: int, end_idx: int):
        """
        Load activations for specific concept names and sample range.
        This is an alias for load_range that takes concept names.
        
        Args:
            concept_names: List of concept names
            start_idx: Starting sample index  
            end_idx: Ending sample index (exclusive)
            
        Returns:
            DataFrame with activations for the specified concepts and range
        """
        # Convert concept names to cluster IDs
        cluster_ids = []
        for name in concept_names:
            if name in self.concept_to_cluster:
                cluster_ids.append(self.concept_to_cluster[name])
            elif name in self.available_clusters:
                # If it's already a cluster ID, use it directly
                cluster_ids.append(name)
                
        if not cluster_ids:
            print(f"Warning: None of the requested concepts {concept_names} found in data")
            return pd.DataFrame()
            
        return self.activation_loader.load_concept_range(cluster_ids, start_idx, end_idx)
    
    def load_tensor_range(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Load activation tensor for a specific range of indices.
        Returns only matched concepts as a tensor.
        
        Args:
            start_idx: Starting global index
            end_idx: Ending global index (exclusive)
            
        Returns:
            Tensor with all matched concept columns
        """
        # Use the existing load_chunk_range method which returns a tensor
        return self.load_chunk_range(start_idx, end_idx)
    
    def get_info(self):
        """Get information about the matched activation loader"""
        base_info = self.activation_loader.get_activation_info()
        return {
            **base_info,
            'matched_clusters': len(self.available_clusters),
            'matched_concepts': len(self.matched_concepts),
            'cluster_to_concept_mapping': self.cluster_to_concept
        }
    
    def load_full_tensor(self) -> torch.Tensor:
        """
        Load the complete activation data as a tensor for matched concepts only.
        
        Returns:
            torch.Tensor of shape (num_samples, num_matched_concepts)
        """
        # Load full tensor from underlying loader
        full_tensor = self.activation_loader.load_full_tensor()
        
        # Get indices of matched clusters
        all_columns = self.activation_loader.columns
        matched_indices = [all_columns.index(cluster_id) for cluster_id in self.available_clusters]
        
        # Return only matched concept columns
        return full_tensor[:, matched_indices]
    
    def load_split_tensor(self, split: str, dataset_name: str, model_input_size: tuple, patch_size: int = 14) -> torch.Tensor:
        """
        Load activation data for a specific split only (train/test/cal) for matched concepts.
        Significantly reduces memory usage compared to loading full tensor.
        
        Args:
            split: Split to load ('train', 'test', or 'cal')
            dataset_name: Dataset name for loading split information
            model_input_size: Model input size
            patch_size: Patch size (default 14)
            
        Returns:
            torch.Tensor containing only the specified split's data for matched concepts
        """
        # Load split tensor from underlying loader
        split_tensor = self.activation_loader.load_split_tensor(split, dataset_name, model_input_size, patch_size)
        
        # Get indices of matched clusters
        all_columns = self.activation_loader.columns
        matched_indices = [all_columns.index(cluster_id) for cluster_id in self.available_clusters]
        
        # Return only matched concept columns
        return split_tensor[:, matched_indices]
    
    def load_concept_activations_for_indices(self, concept_name: str, indices: List[int], device: str = None) -> torch.Tensor:
        """
        Load activations for a specific concept at specific indices.
        Handles concept name to cluster ID mapping.
        
        Args:
            concept_name: Name of the concept to load
            indices: List of indices to load
            device: Device to load tensors on (defaults to underlying loader's device)
            
        Returns:
            torch.Tensor with activations for the specified indices
        """
        # Convert concept name to cluster ID if needed
        if concept_name in self.concept_to_cluster:
            cluster_id = self.concept_to_cluster[concept_name]
        elif concept_name in self.available_clusters:
            cluster_id = concept_name
        else:
            raise ValueError(f"Concept {concept_name} not found in matched concepts")
            
        # Use underlying loader's method
        return self.activation_loader.load_concept_activations_for_indices(cluster_id, indices, device)
    
    def find_indices_above_threshold(self, concept_name: str, threshold: float, chunk_size: int = 50000) -> List[int]:
        """
        Find all indices where activation for a concept exceeds threshold.
        
        Args:
            concept_name: Name of the concept to check
            threshold: Threshold value
            chunk_size: Size of chunks to process at once
            
        Returns:
            List of indices where activation > threshold
        """
        # Convert concept name to cluster ID if needed
        if concept_name in self.concept_to_cluster:
            cluster_id = self.concept_to_cluster[concept_name]
        elif concept_name in self.available_clusters:
            cluster_id = concept_name
        else:
            raise ValueError(f"Concept {concept_name} not found in matched concepts")
            
        # Use underlying loader's method
        return self.activation_loader.find_indices_above_threshold(cluster_id, threshold, chunk_size)


# Index Conversion Utilities
def get_split_indices(dataset_name: str, split: str, model_input_size: tuple, 
                     patch_size: int = 14) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    Get indices for a specific split and create global-to-local mapping.
    
    Args:
        dataset_name: Dataset name
        split: Split name ('train', 'test', 'cal')
        model_input_size: Model input size
        patch_size: Patch size for images
        
    Returns:
        Tuple of:
        - split_indices: Tensor of global indices for this split
        - global_to_local: Dict mapping global indices to local indices
    """
    from utils.general_utils import get_split_df
    from utils.patch_alignment_utils import get_patch_split_df
    
    if isinstance(model_input_size, tuple) and model_input_size[0] == 'text':
        # For text datasets
        split_df = get_split_df(dataset_name)
        split_indices = [idx for idx, s in split_df.items() if s == split]
        split_indices = torch.tensor(sorted(split_indices))
    else:
        # For image datasets
        split_df = get_patch_split_df(dataset_name, model_input_size, patch_size)
        split_indices = torch.tensor(split_df.index[split_df == split].tolist())
    
    # Create global to local mapping
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(split_indices.tolist())}
    
    return split_indices, global_to_local


def convert_image_indices_to_patch_indices(image_indices: Union[List[int], torch.Tensor], 
                                         patches_per_image: int,
                                         split: Optional[str] = None,
                                         dataset_name: Optional[str] = None,
                                         model_input_size: Optional[tuple] = None) -> torch.Tensor:
    """
    Convert image indices to patch indices, optionally within a specific split.
    
    Args:
        image_indices: Image indices to convert
        patches_per_image: Number of patches per image
        split: If provided, convert to local indices within this split
        dataset_name: Required if split is provided
        model_input_size: Required if split is provided
        
    Returns:
        Tensor of patch indices (global or local depending on split parameter)
    """
    if isinstance(image_indices, list):
        image_indices = torch.tensor(image_indices)
    
    # Get all patch indices for these images
    patch_indices = []
    for img_idx in image_indices:
        start_patch = img_idx * patches_per_image
        end_patch = start_patch + patches_per_image
        patch_indices.extend(range(start_patch, end_patch))
    
    patch_indices = torch.tensor(patch_indices)
    
    # If split is provided, convert to local indices
    if split is not None:
        if dataset_name is None or model_input_size is None:
            raise ValueError("dataset_name and model_input_size required when split is provided")
        
        _, global_to_local = get_split_indices(dataset_name, split, model_input_size)
        
        # Convert global patch indices to local
        local_indices = []
        for patch_idx in patch_indices.tolist():
            if patch_idx in global_to_local:
                local_indices.append(global_to_local[patch_idx])
        
        patch_indices = torch.tensor(local_indices)
    
    return patch_indices


def map_global_to_split_local(global_indices: Union[List[int], torch.Tensor],
                            dataset_name: str,
                            split: str,
                            model_input_size: tuple,
                            patch_size: int = 14) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map global indices to split-specific local indices.
    
    Args:
        global_indices: Global indices to map
        dataset_name: Dataset name
        split: Target split ('train', 'test', 'cal')
        model_input_size: Model input size
        patch_size: Patch size for images
        
    Returns:
        Tuple of:
        - valid_global_indices: Global indices that belong to the split
        - local_indices: Corresponding local indices within the split
    """
    if isinstance(global_indices, list):
        global_indices = torch.tensor(global_indices)
    
    split_indices, global_to_local = get_split_indices(dataset_name, split, model_input_size, patch_size)
    
    # Find which global indices belong to this split
    valid_mask = torch.isin(global_indices, split_indices)
    valid_global_indices = global_indices[valid_mask]
    
    # Map to local indices
    local_indices = torch.tensor([global_to_local[idx.item()] for idx in valid_global_indices])
    
    return valid_global_indices, local_indices


def get_text_token_mapping(dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get token-to-sentence mapping for text datasets.
    
    Args:
        dataset_name: Dataset name
        
    Returns:
        Tuple of:
        - sentence_starts: Start token index for each sentence
        - sentence_ends: End token index for each sentence
    """
    import glob
    
    # Find token counts file
    token_files = glob.glob(f'GT_Samples/{dataset_name}/token_counts_inputsize_*.pt')
    if not token_files:
        raise FileNotFoundError(f"No token counts file found for {dataset_name}")
    
    token_counts_per_sentence = torch.load(token_files[0], weights_only=False)
    token_counts_flat = torch.tensor([sum(x) if isinstance(x, list) else x for x in token_counts_per_sentence])
    
    sentence_starts = torch.cat([torch.tensor([0]), token_counts_flat.cumsum(0)[:-1]])
    sentence_ends = token_counts_flat.cumsum(0)
    
    return sentence_starts, sentence_ends


def convert_text_sentence_to_token_indices(sentence_indices: Union[List[int], torch.Tensor],
                                         dataset_name: str,
                                         split: Optional[str] = None) -> torch.Tensor:
    """
    Convert sentence indices to token indices for text datasets.
    
    Args:
        sentence_indices: Sentence indices to convert
        dataset_name: Dataset name
        split: If provided, return local indices within split
        
    Returns:
        Tensor of token indices
    """
    if isinstance(sentence_indices, list):
        sentence_indices = torch.tensor(sentence_indices)
    
    sentence_starts, sentence_ends = get_text_token_mapping(dataset_name)
    
    token_indices = []
    for sent_idx in sentence_indices:
        start = sentence_starts[sent_idx].item()
        end = sentence_ends[sent_idx].item()
        token_indices.extend(range(start, end))
    
    token_indices = torch.tensor(token_indices)
    
    # If split is provided, convert to local indices
    if split is not None:
        _, global_to_local = get_split_indices(dataset_name, split, ('text', 'text'))
        
        local_indices = []
        for token_idx in token_indices.tolist():
            if token_idx in global_to_local:
                local_indices.append(global_to_local[token_idx])
        
        token_indices = torch.tensor(local_indices)
    
    return token_indices