"""
Utility functions shared by SAE evaluation scripts.
"""

import os

def create_sae_symlinks(activations_path, dataset_name, scratch_dir):
    """
    Create symlinks for SAE activation files in Cosine_Similarities folder
    so they can be loaded by ChunkedActivationLoader.
    
    Returns:
        acts_file: The filename to use with ChunkedActivationLoader
    """
    acts_file = os.path.basename(activations_path)
    temp_link_dir = f"{scratch_dir}Cosine_Similarities/{dataset_name}"
    os.makedirs(temp_link_dir, exist_ok=True)
    temp_link_path = f"{temp_link_dir}/{acts_file}"
    
    # Create symlink if it doesn't exist (or remove broken symlink)
    if os.path.islink(temp_link_path):
        # Remove existing symlink (it might be broken or pointing to wrong place)
        os.unlink(temp_link_path)
    elif os.path.exists(temp_link_path):
        # If it's a real file, not a symlink, don't overwrite
        print(f"   Using existing file: {temp_link_path}")
        return acts_file
    
    os.symlink(activations_path, temp_link_path)
    
    # Also symlink chunk files if they exist
    # Handle both .csv and .pt extensions
    if acts_file.endswith('.csv'):
        base_name = acts_file.replace('.csv', '')
        ext = '.csv'
    else:
        base_name = acts_file.replace('.pt', '')
        ext = '.pt'
        
    chunk_idx = 0
    while True:
        chunk_file = f"{scratch_dir}SAE_Acts/{dataset_name}/{base_name}_chunk_{chunk_idx}{ext}"
        if not os.path.exists(chunk_file):
            break
        chunk_link = f"{temp_link_dir}/{base_name}_chunk_{chunk_idx}{ext}"
        
        # Handle chunk symlinks same way
        if os.path.islink(chunk_link):
            os.unlink(chunk_link)
        elif os.path.exists(chunk_link):
            # Real file exists, skip
            chunk_idx += 1
            continue
            
        os.symlink(chunk_file, chunk_link)
        chunk_idx += 1
        
    # Also symlink the chunks_info.json file
    chunks_info_file = f"{scratch_dir}SAE_Acts/{dataset_name}/{base_name}_chunks_info.json"
    chunks_info_link = f"{temp_link_dir}/{base_name}_chunks_info.json"
    if os.path.exists(chunks_info_file):
        if os.path.islink(chunks_info_link):
            os.unlink(chunks_info_link)
        elif not os.path.exists(chunks_info_link):
            os.symlink(chunks_info_file, chunks_info_link)
    
    return acts_file