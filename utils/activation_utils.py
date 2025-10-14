###For Computing Similarity Metrics###
import torch
import torch.nn.functional as F
import csv
import json
import os
import gc
import numpy as np
from tqdm import tqdm
import ctypes
from utils.general_utils import filter_coco_concepts
# Removed write_batch_cosine_sims as it's no longer needed for tensor storage
        

def compute_cosine_sims(embeddings, concepts, output_file, dataset_name, device, scratch_dir='', batch_size=32, chunk_if_larger_gb=10):
    """
    Compute cosine similarities between embeddings and concepts with support for chunked embeddings.
    
    Args:
        embeddings: Either a torch tensor or a path to embedding file (chunked or non-chunked)
        concepts: Dictionary of concept vectors
        output_file: Output filename (now .pt instead of .csv)
        dataset_name: Name of dataset
        device: Device for computation
        scratch_dir: Scratch directory prefix
        batch_size: Batch size for processing
        chunk_if_larger_gb: Threshold for output chunking
    """
    from utils.memory_management_utils import ChunkedEmbeddingLoader
    
    if dataset_name == 'Coco' and 'kmeans' not in output_file:
        concept_keys = filter_coco_concepts(list(concepts.keys()))
    else:
        concept_keys = list(concepts.keys())

    all_concept_embeddings = {k: v.to(device) for k, v in concepts.items() if k in concept_keys}
    all_concept_embeddings_tensor = torch.stack([all_concept_embeddings[k] for k in concept_keys])
    

    base_path = os.path.join(scratch_dir, 'Cosine_Similarities', dataset_name)
    os.makedirs(base_path, exist_ok=True)
    
    # Change output file extension from .csv to .pt
    if output_file.endswith('.csv'):
        output_file = output_file[:-4] + '.pt'
    
    # Check if embeddings is a path or tensor
    if isinstance(embeddings, str):
        # Parse dataset name and file from path
        # Expected format: {scratch_dir}Embeddings/{dataset_name}/{embeddings_file}
        path_parts = embeddings.split('/')
        embeddings_file = path_parts[-1]
        dataset_name_from_path = path_parts[-2]
        scratch_dir_from_path = '/'.join(path_parts[:-3]) + '/' if len(path_parts) > 3 else ''
        
        # Use ChunkedEmbeddingLoader
        # print(f"Loading embeddings from: {embeddings}")
        loader = ChunkedEmbeddingLoader(dataset_name_from_path, embeddings_file, scratch_dir_from_path, device='cpu')  # Load to CPU first
        total_samples = loader.total_samples
        
        # Estimate output size for tensor storage
        bytes_per_value = 4  # float32
        bytes_per_gb = 1024 * 1024 * 1024
        total_values = total_samples * len(concept_keys)
        estimated_size_gb = (total_values * bytes_per_value) / bytes_per_gb
        
        if chunk_if_larger_gb is not None and estimated_size_gb > chunk_if_larger_gb:
            num_output_chunks = int(np.ceil(estimated_size_gb / chunk_if_larger_gb))
            print(f"Estimated tensor size {estimated_size_gb:.2f}GB exceeds {chunk_if_larger_gb}GB threshold")
        else:
            num_output_chunks = 1
            print(f"Estimated tensor size {estimated_size_gb:.2f}GB - saving as 1 chunk")
        
        rows_per_output_chunk = total_samples // num_output_chunks
        print(f"Will save as {num_output_chunks} output chunk(s)")
        
        # Process embeddings chunk by chunk
        current_output_chunk = 0
        current_row_in_output_chunk = 0
        
        # Preallocate tensor for current output chunk instead of accumulating lists
        # For the last chunk, it might be smaller than rows_per_output_chunk
        current_chunk_size = min(rows_per_output_chunk, total_samples - current_output_chunk * rows_per_output_chunk)
        # Use pinned memory for faster GPU->CPU transfers
        output_tensor = torch.zeros((current_chunk_size, len(concept_keys)), dtype=torch.float32, pin_memory=True)
        
        # Create progress bar for overall progress
        total_batches = (total_samples + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="Computing cosine similarities")
        
        # Create CUDA stream for overlapped execution
        stream = torch.cuda.Stream() if device != 'cpu' else None
        
        with torch.no_grad():
            for embed_chunk, start_idx, end_idx in loader.iter_chunks():
                embed_chunk = embed_chunk.to(device)
                
                # Process this embedding chunk in batches
                for i in range(0, embed_chunk.shape[0], batch_size):
                    batch_end = min(i + batch_size, embed_chunk.shape[0])
                    batch_embeddings = embed_chunk[i:batch_end]
                    
                    # Compute cosine similarities
                    cosine_similarities = F.cosine_similarity(
                        batch_embeddings.unsqueeze(1),
                        all_concept_embeddings_tensor.unsqueeze(0),
                        dim=2
                    )
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Determine how many rows we can fit in current chunk
                    batch_rows = cosine_similarities.shape[0]
                    rows_remaining_in_chunk = rows_per_output_chunk - current_row_in_output_chunk
                    
                    if batch_rows <= rows_remaining_in_chunk:
                        # Entire batch fits in current chunk
                        output_tensor[current_row_in_output_chunk:current_row_in_output_chunk + batch_rows].copy_(cosine_similarities, non_blocking=True)
                        current_row_in_output_chunk += batch_rows
                        
                        # Check if we've filled the current chunk completely
                        if current_row_in_output_chunk == rows_per_output_chunk and current_output_chunk < num_output_chunks - 1:
                            # Synchronize to ensure all data is transferred before saving
                            if device != 'cpu':
                                torch.cuda.synchronize()
                            
                            # Save current chunk
                            chunk_output_file = output_file.replace('.pt', f'_chunk_{current_output_chunk}.pt')
                            chunk_output_path = os.path.join(base_path, chunk_output_file)
                            
                            torch.save({
                                'activations': output_tensor,
                                'concept_names': concept_keys,
                                'start_idx': current_output_chunk * rows_per_output_chunk,
                                'end_idx': (current_output_chunk + 1) * rows_per_output_chunk
                            }, chunk_output_path)
                            
                            print(f"Output chunk {current_output_chunk} completed")
                            del output_tensor
                            gc.collect()
                            
                            # Start next chunk
                            current_output_chunk += 1
                            current_row_in_output_chunk = 0
                            
                            # Preallocate tensor for next chunk
                            current_chunk_size = min(rows_per_output_chunk, total_samples - current_output_chunk * rows_per_output_chunk)
                            output_tensor = torch.zeros((current_chunk_size, len(concept_keys)), dtype=torch.float32, pin_memory=True)
                    else:
                        # Batch spans multiple chunks
                        rows_to_add = rows_remaining_in_chunk
                        output_tensor[current_row_in_output_chunk:current_row_in_output_chunk + rows_to_add].copy_(cosine_similarities[:rows_to_add], non_blocking=True)
                        
                        # Synchronize to ensure all data is transferred before saving
                        if device != 'cpu':
                            torch.cuda.synchronize()
                        
                        # Save current chunk
                        chunk_output_file = output_file.replace('.pt', f'_chunk_{current_output_chunk}.pt')
                        chunk_output_path = os.path.join(base_path, chunk_output_file)
                        
                        torch.save({
                            'activations': output_tensor[:rows_per_output_chunk],
                            'concept_names': concept_keys,
                            'start_idx': current_output_chunk * rows_per_output_chunk,
                            'end_idx': (current_output_chunk + 1) * rows_per_output_chunk
                        }, chunk_output_path)
                        
                        print(f"Output chunk {current_output_chunk} completed")
                        del output_tensor
                        gc.collect()
                        
                        # Start next chunk
                        current_output_chunk += 1
                        current_row_in_output_chunk = 0
                        
                        # Preallocate tensor for next chunk
                        remaining_rows = total_samples - (current_output_chunk * rows_per_output_chunk)
                        output_tensor = torch.zeros((min(rows_per_output_chunk, remaining_rows), len(concept_keys)), dtype=torch.float32)
                        
                        # Add remaining rows from current batch
                        remaining_batch_rows = batch_rows - rows_to_add
                        output_tensor[:remaining_batch_rows].copy_(cosine_similarities[rows_to_add:], non_blocking=True)
                        current_row_in_output_chunk = remaining_batch_rows
                    
                    del batch_embeddings, cosine_similarities
                
                # Clean up embedding chunk
                del embed_chunk
                torch.cuda.empty_cache()
                gc.collect()
                
                # Clear the loader's cache to prevent memory accumulation
                loader.clear_cache()
                
                # Force Python to release memory to OS
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except:
                    pass  # malloc_trim might not be available on all systems
        
        # Close progress bar
        pbar.close()
        
        # Save remaining data in the last chunk
        if current_row_in_output_chunk > 0:
            # Synchronize to ensure all data is transferred before saving
            if device != 'cpu':
                torch.cuda.synchronize()
                
            chunk_output_file = output_file.replace('.pt', f'_chunk_{current_output_chunk}.pt')
            chunk_output_path = os.path.join(base_path, chunk_output_file)
            
            torch.save({
                'activations': output_tensor[:current_row_in_output_chunk],
                'concept_names': concept_keys,
                'start_idx': current_output_chunk * rows_per_output_chunk,
                'end_idx': current_output_chunk * rows_per_output_chunk + current_row_in_output_chunk
            }, chunk_output_path)
            
            del output_tensor
        
        # Use actual number of chunks created (current_output_chunk is 0-indexed)
        actual_num_chunks = current_output_chunk + 1
        print(f"All {actual_num_chunks} output chunk(s) saved successfully")
        print(f"Cosine similarities saved to: {base_path}")
        
        # Save chunk info for easier loading later
        chunk_info = {
            'num_chunks': actual_num_chunks,
            'total_samples': total_samples,
            'concept_names': concept_keys,
            'num_concepts': len(concept_keys),
            'chunks': []
        }
        
        for i in range(actual_num_chunks):
            chunk_file = output_file.replace('.pt', f'_chunk_{i}.pt')
            print(f"  - {chunk_file}")
            chunk_info['chunks'].append({
                'file': chunk_file,
                'chunk_idx': i
            })
        
        # Save chunk info
        chunk_info_path = os.path.join(base_path, output_file.replace('.pt', '_chunks_info.json'))
        with open(chunk_info_path, 'w') as f:
            json.dump(chunk_info, f, indent=2)
        
        return
    
    # Original code for tensor input
    if chunk_if_larger_gb is not None:
        bytes_per_value = 4  # float32
        bytes_per_gb = 1024 * 1024 * 1024
        total_values = embeddings.shape[0] * len(concept_keys)
        estimated_size_gb = (total_values * bytes_per_value) / bytes_per_gb
        
        # Calculate number of chunks needed
        if estimated_size_gb > chunk_if_larger_gb:
            num_chunks = int(np.ceil(estimated_size_gb / chunk_if_larger_gb))
            print(f"Estimated tensor size {estimated_size_gb:.2f}GB exceeds {chunk_if_larger_gb}GB threshold")
        else:
            num_chunks = 1
            print(f"Estimated tensor size {estimated_size_gb:.2f}GB - saving as 1 chunk")
            
        rows_per_chunk = embeddings.shape[0] // num_chunks
        print(f"Saving as {num_chunks} chunk(s)")
        
        # Process and save in chunks for tensor input
        chunk_info = {
            'num_chunks': num_chunks,
            'total_samples': embeddings.shape[0],
            'concept_names': concept_keys,
            'num_concepts': len(concept_keys),
            'chunks': []
        }
        
        for chunk_idx in range(num_chunks):
            start_row = chunk_idx * rows_per_chunk
            end_row = (chunk_idx + 1) * rows_per_chunk if chunk_idx < num_chunks - 1 else embeddings.shape[0]
            
            chunk_output_file = output_file.replace('.pt', f'_chunk_{chunk_idx}.pt')
            chunk_output_path = os.path.join(base_path, chunk_output_file)
            
            print(f"Processing chunk {chunk_idx}: rows {start_row}-{end_row}")
            
            # Collect all similarities for this chunk
            chunk_data = []
            
            with torch.no_grad():
                for i in tqdm(range(start_row, end_row, batch_size), 
                            desc=f"Computing cosine similarities (chunk {chunk_idx})"):
                    batch_end = min(i + batch_size, end_row)
                    batch_embeddings = embeddings[i:batch_end].to(device)
                    
                    cosine_similarities = F.cosine_similarity(
                        batch_embeddings.unsqueeze(1),
                        all_concept_embeddings_tensor.unsqueeze(0),
                        dim=2
                    )
                    
                    chunk_data.extend(cosine_similarities.cpu().tolist())
                    del batch_embeddings, cosine_similarities
            
            # Save chunk
            chunk_tensor = torch.tensor(chunk_data, dtype=torch.float32)
            torch.save({
                'activations': chunk_tensor,
                'concept_names': concept_keys,
                'start_idx': start_row,
                'end_idx': end_row
            }, chunk_output_path)
            
            print(f"Chunk {chunk_idx} saved to {chunk_output_path}")
            chunk_info['chunks'].append({
                'file': chunk_output_file,
                'chunk_idx': chunk_idx
            })
            
            del chunk_data, chunk_tensor
        
        # Save chunk info
        chunk_info_path = os.path.join(base_path, output_file.replace('.pt', '_chunks_info.json'))
        with open(chunk_info_path, 'w') as f:
            json.dump(chunk_info, f, indent=2)
        
        print(f"All {num_chunks} chunk(s) saved successfully")
        print(f"Cosine similarities saved to: {base_path}")
        for i in range(num_chunks):
            chunk_file = output_file.replace('.pt', f'_chunk_{i}.pt')
            print(f"  - {chunk_file}")
        return
    
    # Save as single file (only if chunk_if_larger_gb is None)
    output_path = os.path.join(base_path, output_file)
    
    # Collect all similarities
    all_sims = []
    
    with torch.no_grad():
        for i in tqdm(range(0, embeddings.shape[0], batch_size), desc="Computing cosine similarities"):
            batch_end = min(i + batch_size, embeddings.shape[0])
            batch_embeddings = embeddings[i:batch_end].to(device)
            
            cosine_similarities = F.cosine_similarity(
                batch_embeddings.unsqueeze(1),
                all_concept_embeddings_tensor.unsqueeze(0),
                dim=2
            )
            
            all_sims.extend(cosine_similarities.cpu().tolist())
            del batch_embeddings, cosine_similarities
    
    # Save as tensor with metadata
    sims_tensor = torch.tensor(all_sims, dtype=torch.float32)
    torch.save({
        'activations': sims_tensor,
        'concept_names': concept_keys,
        'total_samples': embeddings.shape[0],
        'num_concepts': len(concept_keys)
    }, output_path)
    
    print(f"\nCosine similarity results saved at {output_path}")


# Removed CSV writing functions as they're no longer needed for tensor storage
        
def compute_signed_distances(embeds, concepts, dataset_name, device, output_file, scratch_dir, batch_size=100, chunk_if_larger_gb=10):
    """
    Computes signed distances between embeddings and cluster directions with chunking support.

    Args:
        embeds: Either a torch tensor or a path to embedding file (chunked or non-chunked)
        concepts (dict): concept_id -> weight tensor (1D)
        dataset_name (str): Used for output folder
        device (str): 'cuda' or 'cpu'
        output_file (str): Filename to write to (now .pt instead of .csv)
        scratch_dir (str): Scratch directory prefix
        batch_size (int): Batch size for processing
        chunk_if_larger_gb (float): If file size exceeds this, save as chunks. Set to None to disable chunking.
    """
    from utils.memory_management_utils import ChunkedEmbeddingLoader
    
    concept_ids = [str(k) for k in concepts.keys()]
    output_dir = os.path.join(scratch_dir, "Distances", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Change output file extension from .csv to .pt
    if output_file.endswith('.csv'):
        output_file = output_file[:-4] + '.pt'

    # Prepare weight matrix
    weight_matrix = torch.stack([concepts[cid] for cid in concept_ids]).to(device)
    
    # Check if embeds is a path or tensor
    if isinstance(embeds, str):
        # Parse dataset name and file from path
        # Expected format: {scratch_dir}Embeddings/{dataset_name}/{embeddings_file}
        path_parts = embeds.split('/')
        embeddings_file = path_parts[-1]
        dataset_name_from_path = path_parts[-2]
        scratch_dir_from_path = '/'.join(path_parts[:-3]) + '/' if len(path_parts) > 3 else ''
        
        # Use ChunkedEmbeddingLoader
        # print(f"Loading embeddings from: {embeds}")
        loader = ChunkedEmbeddingLoader(dataset_name_from_path, embeddings_file, scratch_dir_from_path, device='cpu')  # Load to CPU first
        total_samples = loader.total_samples
        
        # Estimate output size for tensor storage
        bytes_per_value = 4  # float32
        bytes_per_gb = 1024 * 1024 * 1024
        total_values = total_samples * len(concept_ids)
        estimated_size_gb = (total_values * bytes_per_value) / bytes_per_gb
        
        if chunk_if_larger_gb is not None and estimated_size_gb > chunk_if_larger_gb:
            num_output_chunks = int(np.ceil(estimated_size_gb / chunk_if_larger_gb))
            print(f"Estimated tensor size {estimated_size_gb:.2f}GB exceeds {chunk_if_larger_gb}GB threshold")
        else:
            num_output_chunks = 1
            print(f"Estimated tensor size {estimated_size_gb:.2f}GB - saving as 1 chunk")
        
        rows_per_output_chunk = total_samples // num_output_chunks
        print(f"Will save as {num_output_chunks} output chunk(s)")
        
        # Process embeddings chunk by chunk
        current_output_chunk = 0
        current_row_in_output_chunk = 0
        
        # Preallocate tensor for current output chunk instead of accumulating lists
        # For the last chunk, it might be smaller than rows_per_output_chunk
        current_chunk_size = min(rows_per_output_chunk, total_samples - current_output_chunk * rows_per_output_chunk)
        output_tensor = torch.zeros((current_chunk_size, len(concept_ids)), dtype=torch.float32, pin_memory=True)
        
        # Create progress bar for overall progress
        total_batches = (total_samples + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="Computing signed distances")
        
        with torch.no_grad():
            for embed_chunk, start_idx, end_idx in loader.iter_chunks():
                embed_chunk = embed_chunk.to(device)
                
                # Ensure weight matrix has same dtype as embeddings
                weight_matrix_typed = weight_matrix.to(embed_chunk.dtype)
                
                # Process this embedding chunk in batches
                for i in range(0, embed_chunk.shape[0], batch_size):
                    batch_end = min(i + batch_size, embed_chunk.shape[0])
                    batch_embeddings = embed_chunk[i:batch_end]
                    
                    # Compute signed distances
                    sims = batch_embeddings @ weight_matrix_typed.T
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Determine how many rows we can fit in current chunk
                    batch_rows = sims.shape[0]
                    rows_remaining_in_chunk = rows_per_output_chunk - current_row_in_output_chunk
                    
                    if batch_rows <= rows_remaining_in_chunk:
                        # Entire batch fits in current chunk
                        output_tensor[current_row_in_output_chunk:current_row_in_output_chunk + batch_rows].copy_(sims, non_blocking=True)
                        current_row_in_output_chunk += batch_rows
                        
                        # Check if we've filled the current chunk completely
                        if current_row_in_output_chunk == rows_per_output_chunk and current_output_chunk < num_output_chunks - 1:
                            # Save current chunk
                            chunk_output_file = output_file.replace('.pt', f'_chunk_{current_output_chunk}.pt')
                            chunk_output_path = os.path.join(output_dir, chunk_output_file)
                            
                            torch.save({
                                'activations': output_tensor,
                                'concept_names': concept_ids,
                                'start_idx': current_output_chunk * rows_per_output_chunk,
                                'end_idx': (current_output_chunk + 1) * rows_per_output_chunk
                            }, chunk_output_path)
                            
                            print(f"Output chunk {current_output_chunk} completed")
                            del output_tensor
                            gc.collect()
                            
                            # Start next chunk
                            current_output_chunk += 1
                            current_row_in_output_chunk = 0
                            
                            # Preallocate tensor for next chunk
                            current_chunk_size = min(rows_per_output_chunk, total_samples - current_output_chunk * rows_per_output_chunk)
                            output_tensor = torch.zeros((current_chunk_size, len(concept_ids)), dtype=torch.float32, pin_memory=True)
                    else:
                        # Batch spans multiple chunks
                        rows_to_add = rows_remaining_in_chunk
                        output_tensor[current_row_in_output_chunk:current_row_in_output_chunk + rows_to_add].copy_(sims[:rows_to_add], non_blocking=True)
                        
                        # Save current chunk
                        chunk_output_file = output_file.replace('.pt', f'_chunk_{current_output_chunk}.pt')
                        chunk_output_path = os.path.join(output_dir, chunk_output_file)
                        
                        torch.save({
                            'activations': output_tensor[:rows_per_output_chunk],
                            'concept_names': concept_ids,
                            'start_idx': current_output_chunk * rows_per_output_chunk,
                            'end_idx': (current_output_chunk + 1) * rows_per_output_chunk
                        }, chunk_output_path)
                        
                        print(f"Output chunk {current_output_chunk} completed")
                        del output_tensor
                        gc.collect()
                        
                        # Start next chunk
                        current_output_chunk += 1
                        current_row_in_output_chunk = 0
                        
                        # Preallocate tensor for next chunk
                        current_chunk_size = min(rows_per_output_chunk, total_samples - current_output_chunk * rows_per_output_chunk)
                        output_tensor = torch.zeros((current_chunk_size, len(concept_ids)), dtype=torch.float32, pin_memory=True)
                        
                        # Add remaining rows from current batch
                        remaining_batch_rows = batch_rows - rows_to_add
                        output_tensor[:remaining_batch_rows].copy_(sims[rows_to_add:], non_blocking=True)
                        current_row_in_output_chunk = remaining_batch_rows
                    
                    del batch_embeddings, sims
                
                # Clean up embedding chunk
                del embed_chunk
                torch.cuda.empty_cache()
                gc.collect()
                
                # Clear the loader's cache to prevent memory accumulation
                loader.clear_cache()
                
                # Force Python to release memory to OS
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except:
                    pass  # malloc_trim might not be available on all systems
        
        # Close progress bar
        pbar.close()
        
        # Save remaining data in the last chunk
        if current_row_in_output_chunk > 0:
            chunk_output_file = output_file.replace('.pt', f'_chunk_{current_output_chunk}.pt')
            chunk_output_path = os.path.join(output_dir, chunk_output_file)
            
            torch.save({
                'activations': output_tensor[:current_row_in_output_chunk],
                'concept_names': concept_ids,
                'start_idx': current_output_chunk * rows_per_output_chunk,
                'end_idx': current_output_chunk * rows_per_output_chunk + current_row_in_output_chunk
            }, chunk_output_path)
            
            del output_tensor
        
        # Use actual number of chunks created (current_output_chunk is 0-indexed)
        actual_num_chunks = current_output_chunk + 1
        print(f"All {actual_num_chunks} output chunk(s) saved successfully")
        print(f"Signed distances saved to: {output_dir}/")
        
        # Save chunk info for easier loading later
        chunk_info = {
            'num_chunks': actual_num_chunks,
            'total_samples': total_samples,
            'concept_names': concept_ids,
            'num_concepts': len(concept_ids),
            'chunks': []
        }
        
        for i in range(actual_num_chunks):
            chunk_file = output_file.replace('.pt', f'_chunk_{i}.pt')
            print(f"  - {chunk_file}")
            chunk_info['chunks'].append({
                'file': chunk_file,
                'chunk_idx': i
            })
        
        # Save chunk info
        chunk_info_path = os.path.join(output_dir, output_file.replace('.pt', '_chunks_info.json'))
        with open(chunk_info_path, 'w') as f:
            json.dump(chunk_info, f, indent=2)
        
        return
    
    # Original code for tensor input
    if chunk_if_larger_gb is not None:
        bytes_per_value = 4  # float32
        bytes_per_gb = 1024 * 1024 * 1024
        total_values = embeds.shape[0] * len(concept_ids)
        estimated_size_gb = (total_values * bytes_per_value) / bytes_per_gb
        
        # Calculate number of chunks needed
        if estimated_size_gb > chunk_if_larger_gb:
            num_chunks = int(np.ceil(estimated_size_gb / chunk_if_larger_gb))
            print(f"Estimated tensor size {estimated_size_gb:.2f}GB exceeds {chunk_if_larger_gb}GB threshold")
        else:
            num_chunks = 1
            print(f"Estimated tensor size {estimated_size_gb:.2f}GB - saving as 1 chunk")
            
        rows_per_chunk = embeds.shape[0] // num_chunks
        print(f"Saving as {num_chunks} chunk(s)")
        
        # Prepare weight matrix outside loop
        weight_matrix = torch.stack([concepts[cid] for cid in concept_ids]).to(device).to(embeds.dtype)
        
        # Process and save in chunks for tensor input
        chunk_info = {
            'num_chunks': num_chunks,
            'total_samples': embeds.shape[0],
            'concept_names': concept_ids,
            'num_concepts': len(concept_ids),
            'chunks': []
        }
        
        for chunk_idx in range(num_chunks):
            start_row = chunk_idx * rows_per_chunk
            end_row = (chunk_idx + 1) * rows_per_chunk if chunk_idx < num_chunks - 1 else embeds.shape[0]
            
            chunk_output_file = output_file.replace('.pt', f'_chunk_{chunk_idx}.pt')
            chunk_output_path = os.path.join(output_dir, chunk_output_file)
            
            print(f"Processing chunk {chunk_idx}: rows {start_row}-{end_row}")
            
            # Collect all distances for this chunk
            chunk_data = []
            
            with torch.no_grad():
                for i in tqdm(range(start_row, end_row, batch_size), 
                            desc=f"Computing signed distances (chunk {chunk_idx})"):
                    batch_end = min(i + batch_size, end_row)
                    batch_embeddings = embeds[i:batch_end].to(device)
                    
                    # Compute signed distances
                    sims = batch_embeddings @ weight_matrix.T
                    
                    chunk_data.extend(sims.cpu().tolist())
                    del batch_embeddings, sims
            
            # Save chunk
            chunk_tensor = torch.tensor(chunk_data, dtype=torch.float32)
            torch.save({
                'activations': chunk_tensor,
                'concept_names': concept_ids,
                'start_idx': start_row,
                'end_idx': end_row
            }, chunk_output_path)
            
            print(f"Chunk {chunk_idx} saved to {chunk_output_path}")
            chunk_info['chunks'].append({
                'file': chunk_output_file,
                'chunk_idx': chunk_idx
            })
            
            del chunk_data, chunk_tensor
        
        # Save chunk info
        chunk_info_path = os.path.join(output_dir, output_file.replace('.pt', '_chunks_info.json'))
        with open(chunk_info_path, 'w') as f:
            json.dump(chunk_info, f, indent=2)
        
        print(f"All {num_chunks} chunk(s) saved successfully")
        print(f"Signed distances saved to: {output_dir}/")
        for i in range(num_chunks):
            chunk_file = output_file.replace('.pt', f'_chunk_{i}.pt')
            print(f"  - {chunk_file}")
        return

    # Save as single file (only if chunk_if_larger_gb is None)
    output_path = os.path.join(output_dir, output_file)
    
    # Prepare weight matrix outside loop
    weight_matrix = torch.stack([concepts[cid] for cid in concept_ids]).to(device).to(embeds.dtype)
    
    # Collect all distances
    all_dists = []
    
    with torch.no_grad():
        for i in tqdm(range(0, embeds.shape[0], batch_size), desc="Computing signed distances"):
            batch_end = min(i + batch_size, embeds.shape[0])
            batch_embeddings = embeds[i:batch_end].to(device)
            
            # Compute signed distances
            sims = batch_embeddings @ weight_matrix.T
            
            all_dists.extend(sims.cpu().tolist())
            del batch_embeddings, sims
    
    # Save as tensor with metadata
    dists_tensor = torch.tensor(all_dists, dtype=torch.float32)
    torch.save({
        'activations': dists_tensor,
        'concept_names': concept_ids,
        'total_samples': embeds.shape[0],
        'num_concepts': len(concept_ids)
    }, output_path)
    
    print(f"\nSigned distances saved to: {output_path}")
