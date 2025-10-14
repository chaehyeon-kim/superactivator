"""
Utils for analyzing concept emergence and evolution across model layers.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import os
from tqdm import tqdm

from utils.general_utils import compute_cossim_w_vector


def extract_clip_embeddings_all_layers(model, processor, images, device):
    """
    Extract patch embeddings from all layers of CLIP vision encoder.
    
    Args:
        model: CLIP model
        processor: CLIP processor
        images: List of PIL images
        device: torch device
        
    Returns:
        Dict[int, torch.Tensor]: Dictionary mapping layer percentage to patch embeddings
    """
    num_layers = len(model.vision_model.encoder.layers)
    layer_outputs = {}
    
    # Calculate layer percentages
    layer_percentages = []
    for i in range(num_layers):
        percentage = int((i / (num_layers - 1)) * 100)
        layer_percentages.append(percentage)
    
    # Register hooks for all layers
    handles = []
    for i, layer in enumerate(model.vision_model.encoder.layers):
        percentage = layer_percentages[i]
        
        def make_hook(pct):
            def hook(module, input, output):
                # Store the output (hidden_states)
                layer_outputs[pct] = output[0].clone()  # output[0] is hidden_states
            return hook
        
        handle = layer.register_forward_hook(make_hook(percentage))
        handles.append(handle)
    
    # Process images through model
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        # Forward pass - this will trigger all hooks
        _ = model.vision_model(**inputs)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Process collected outputs to get patch embeddings
    embeddings_by_layer = {}
    for percentage, layer_output in layer_outputs.items():
        # Remove CLS token (first position) to get patch embeddings
        patch_embeddings = layer_output[:, 1:, :]  # [batch_size, num_patches, hidden_dim]
        
        # Flatten to [total_patches, hidden_dim]
        patch_embeddings = patch_embeddings.reshape(-1, patch_embeddings.size(-1))
        
        
        embeddings_by_layer[percentage] = patch_embeddings
    
    return embeddings_by_layer


def extract_llama_embeddings_all_layers(model, processor, images, device):
    """
    Extract patch embeddings from all layers of Llama vision encoder WITHOUT using the projector.
    
    NOTE: This returns embeddings from each layer of the vision model (1280-dim)
    instead of projecting them to the language model space (4096-dim).
    
    Args:
        model: Llama vision model (MllamaForConditionalGeneration)
        processor: Llama processor
        images: List of PIL images
        device: torch device
        
    Returns:
        Dict[int, torch.Tensor]: Dictionary mapping layer percentage to patch embeddings (1280-dim)
    """
    # Llama vision model has transformer.layers instead of encoder.layers
    num_layers = len(model.vision_model.transformer.layers)
    layer_outputs = {}
    
    # Calculate layer percentages
    layer_percentages = []
    for i in range(num_layers):
        percentage = int((i / (num_layers - 1)) * 100)
        layer_percentages.append(percentage)
    
    # Also capture the final output
    final_output = None
    
    def capture_final_output(module, input, output):
        nonlocal final_output
        final_output = output[0].clone()
    
    # Register hooks for all layers
    handles = []
    for i, layer in enumerate(model.vision_model.transformer.layers):
        percentage = layer_percentages[i]
        
        def make_hook(pct):
            def hook(module, input, output):
                # Store the output (hidden_states)
                layer_outputs[pct] = output[0].clone()  # output[0] is hidden_states
            return hook
        
        handle = layer.register_forward_hook(make_hook(percentage))
        handles.append(handle)
    
    # Hook to capture final vision model output
    final_handle = model.vision_model.register_forward_hook(capture_final_output)
    handles.append(final_handle)
    
    # Process images through model
    inputs = processor(images,
                      add_special_tokens=False,
                      return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Forward pass - this will trigger all hooks
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"],
            aspect_ratio_ids=inputs["aspect_ratio_ids"],
            aspect_ratio_mask=inputs["aspect_ratio_mask"],
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True
        )
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Process collected outputs to get patch embeddings WITHOUT projection
    embeddings_by_layer = {}
    
    # Process each layer
    for percentage, layer_output in layer_outputs.items():
        # Extract patch embeddings WITHOUT projecting to language model space
        # All layers will have the same dimensionality (1280)
        
        # For the final layer, we also use the raw output
        if percentage == 100:
            # Use the actual final vision output
            cross_attention_states = final_output if final_output is not None else layer_output
        else:
            # For intermediate layers, use the layer output directly
            cross_attention_states = layer_output
        
        # Extract patch embeddings (excluding CLS tokens)
        all_embs = []
        
        # Handle the tiling structure - 4 tiles per image
        for i in range(0, len(images) * 4, 4):  # Only look at first tile for consistency
            if i < cross_attention_states.shape[0]:
                curr_emb = cross_attention_states[i, :, :]  
                embs_no_cls = curr_emb[1:, :]  # Exclude the first token (cls token)
                all_embs.append(embs_no_cls)
        
        if all_embs:
            patch_embeddings = torch.stack(all_embs)
            patch_embeddings = patch_embeddings.reshape(-1, patch_embeddings.shape[2])
        else:
            # Fallback if structure is different
            patch_embeddings = cross_attention_states[:, 1:, :].reshape(-1, cross_attention_states.shape[-1])
        
        embeddings_by_layer[percentage] = patch_embeddings
    
    return embeddings_by_layer


def extract_embeddings_all_layers(model, processor, images, device, model_name):
    """
    Extract embeddings from all layers for different model types.
    
    Args:
        model: Model (CLIP or Llama)
        processor: Model processor  
        images: List of PIL images
        device: torch device
        model_name: Model name ('clip' or 'llama')
        
    Returns:
        Dict[int, torch.Tensor]: Dictionary mapping layer percentage to patch embeddings
    """
    if model_name.lower() == 'clip':
        return extract_clip_embeddings_all_layers(model, processor, images, device)
    elif model_name.lower() == 'llama':
        return extract_llama_embeddings_all_layers(model, processor, images, device)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")


def load_final_layer_stats(dataset_name, model_name, device=None):
    """
    Load the final layer normalization statistics.
    
    Args:
        dataset_name: Name of dataset (e.g., 'CLEVR')
        model_name: Name of model (e.g., 'clip')
        device: Device to load tensors to
        
    Returns:
        mean_train_embedding, train_norm
    """
    # First try to load from the stats JSON (much faster)
    stats_file = 'Embeddings/embedding_stats.json'
    if os.path.exists(stats_file):
        try:
            from utils.embedding_stats_utils import load_normalization_stats
            
            # Try different capitalizations for model name
            model_name_variants = [model_name.upper(), model_name.capitalize(), model_name.lower()]
            
            for model_variant in model_name_variants:
                mean_embedding, train_norm = load_normalization_stats(
                    dataset_name=dataset_name,
                    model_name=model_variant,
                    embedding_type='patch',
                    percent_thru_model=100,
                    device=device,
                    stats_file=stats_file
                )
                
                if mean_embedding is not None and train_norm is not None:
                    print(f"Loaded stats from JSON: mean_embedding shape: {mean_embedding.shape}, train_norm: {train_norm}")
                    return mean_embedding, train_norm
            
            print("Stats not found in JSON for any model name variant, falling back to loading full embeddings file...")
        except Exception as e:
            print(f"Error loading from stats JSON: {e}, falling back to full file...")
    
    # Fallback to loading the full embeddings file
    embeddings_dir = f'Embeddings/{dataset_name}/'
    
    # Try different capitalizations for file naming
    model_name_variants = [model_name.upper(), model_name.capitalize(), model_name.lower()]
    
    embeddings_file = None
    for model_variant in model_name_variants:
        test_file = os.path.join(embeddings_dir, f"{model_variant}_patch_embeddings_percentthrumodel_100.pt")
        if os.path.exists(test_file):
            embeddings_file = test_file
            break
    
    if embeddings_file is None:
        raise FileNotFoundError(f"Could not find final layer embeddings file for model '{model_name}' in {embeddings_dir}")
    
    # Load the embeddings dictionary to CPU first to avoid GPU memory issues
    print(f"Loading embeddings from: {embeddings_file}")
    embeddings_dict = torch.load(embeddings_file, map_location='cpu')
    
    print(f"Keys in embeddings_dict: {list(embeddings_dict.keys())}")
    
    # Only extract what we need and move to device
    mean_embedding = embeddings_dict['mean_train_embedding']
    train_norm = embeddings_dict['train_norm']
    
    # Clear the large embeddings from memory
    del embeddings_dict
    
    print(f"mean_embedding shape: {mean_embedding.shape}")
    print(f"train_norm shape: {train_norm.shape}")
    
    # Move to device if specified
    if device is not None:
        mean_embedding = mean_embedding.to(device)
        train_norm = train_norm.to(device)
    
    return mean_embedding, train_norm


def load_concept_vectors(dataset_name, concept_type='avg', device=None, model_name='clip'):
    """
    Load concept vectors from the Concepts folder.
    
    Args:
        dataset_name: Name of dataset
        concept_type: Type of concept ('avg' or 'linsep_bd_true_bn_false')
        device: Device to load tensors to
        model_name: Name of model (e.g., 'clip')
        
    Returns:
        Dict mapping concept names to concept vectors
    """
    concepts_dir = f'Concepts/{dataset_name}/'
    
    # Try different capitalizations for model name
    model_name_variants = [model_name.upper(), model_name.capitalize(), model_name.lower()]
    
    concept_file = None
    for model_variant in model_name_variants:
        if concept_type == 'avg':
            test_file = os.path.join(concepts_dir, f'avg_concepts_{model_variant}_patch_embeddings_percentthrumodel_100.pt')
        elif concept_type == 'linsep_bd_true_bn_false':
            test_file = os.path.join(concepts_dir, f'linsep_concepts_BD_True_BN_False_{model_variant}_patch_embeddings_percentthrumodel_100.pt')
        else:
            raise ValueError(f"Unknown concept_type: {concept_type}")
        
        if os.path.exists(test_file):
            concept_file = test_file
            break
    
    if concept_file is None:
        raise FileNotFoundError(f"Could not find concept file for model '{model_name}' and type '{concept_type}' in {concepts_dir}")
    
    print(f"Loading concepts from: {concept_file}")
    
    # Load the concepts dictionary
    concepts_dict = torch.load(concept_file, map_location=device)
    
    # Move concept vectors to device if specified
    if device is not None:
        for concept_name, concept_vector in concepts_dict.items():
            concepts_dict[concept_name] = concept_vector.to(device)
    
    return concepts_dict


def normalize_embeddings_with_final_layer_stats(embeddings, final_mean, final_norm):
    """
    Normalize embeddings using final layer statistics.
    
    Args:
        embeddings: Embeddings tensor
        final_mean: Mean from final layer training set
        final_norm: Average norm from final layer training set
        
    Returns:
        Normalized embeddings
    """
    centered_embeddings = embeddings - final_mean
    normalized_embeddings = centered_embeddings / final_norm
    
    return normalized_embeddings


def compute_layer_concept_similarities(embeddings_by_layer, concept_vectors, final_mean, final_norm, projector=None):
    """
    Compute cosine similarities between layer embeddings and concept vectors.
    Projects intermediate embeddings to final space if projector is provided.
    
    Args:
        embeddings_by_layer: Dict mapping layer percentage to embeddings
        concept_vectors: Dict mapping concept names to vectors
        final_mean: Final layer mean for normalization (4096-dim)
        final_norm: Final layer norm for normalization
        projector: Optional projection layer to transform 1280->4096 dims
        
    Returns:
        Dict mapping layer percentage to concept similarities
    """
    similarities_by_layer = {}
    
    # Check concept vector dimension
    first_concept = list(concept_vectors.values())[0]
    concept_dim = first_concept.shape[0]
    print(f"Concept vectors dimension: {concept_dim}")
    print(f"Final layer stats dimension: {final_mean.shape[0]}")
    if projector is not None:
        print(f"Projector available: {projector.in_features} -> {projector.out_features}")
    
    for layer_pct, embeddings in embeddings_by_layer.items():
        print(f"Processing layer {layer_pct}% - embeddings shape: {embeddings.shape}")
        print(f"  Embeddings stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
        
        embedding_dim = embeddings.shape[1]
        
        if embedding_dim == concept_dim:
            # Same dimension - use existing normalization
            print(f"  Using existing normalization (dims match: {embedding_dim})")
            processed_embeddings = embeddings
        elif projector is not None and (embedding_dim == projector.in_features or 
                                        (projector.in_features == 7680 and embedding_dim == 1280)):
            # Project to final space using model's projection layer
            print(f"  Projecting {embedding_dim} -> {projector.out_features} using model's projector")
            
            # For Llama's multi-modal projector, we need to handle the 7680 input
            # The 7680 comes from concatenating multiple features
            if projector.in_features == 7680 and embedding_dim == 1280:
                # Replicate the 1280-dim embedding 6 times to get 7680
                # This simulates the concatenation that happens in the full model
                replicated_embeddings = embeddings.repeat(1, 6)  # [N, 1280] -> [N, 7680]
                print(f"    Replicated embeddings to shape: {replicated_embeddings.shape}")
            else:
                replicated_embeddings = embeddings
            
            with torch.no_grad():
                processed_embeddings = projector(replicated_embeddings)
            print(f"    Projected embeddings shape: {processed_embeddings.shape}")
        else:
            # Skip layers with incompatible dimensions
            print(f"  Skipping layer (embedding_dim={embedding_dim}, no compatible projector)")
            layer_similarities = {}
            for concept_name in concept_vectors.keys():
                layer_similarities[concept_name] = torch.zeros(embeddings.shape[0], device=embeddings.device)
            similarities_by_layer[layer_pct] = layer_similarities
            print()
            continue
        
        # Normalize using final layer stats
        normalized_embeddings = normalize_embeddings_with_final_layer_stats(
            processed_embeddings, final_mean, final_norm
        )
        
        print(f"  Normalized stats: mean={normalized_embeddings.mean():.4f}, std={normalized_embeddings.std():.4f}")
        
        # Compute similarities with each concept
        layer_similarities = {}
        for concept_name, concept_vector in concept_vectors.items():
            similarities = compute_cossim_w_vector(concept_vector, normalized_embeddings)
            layer_similarities[concept_name] = similarities
            print(f"  {concept_name}: sim_mean={similarities.mean():.4f}, sim_max={similarities.max():.4f}")
        
        similarities_by_layer[layer_pct] = layer_similarities
        print()  # Empty line for readability
    
    return similarities_by_layer


def visualize_patch_concept_heatmaps(similarities_by_layer, concept_names, image_idx, 
                                   original_image, patch_size=14, 
                                   dataset_name='CLEVR', save_path=None, model_input_size=(224, 224), con_label=None, per_layer_scale=False, selected_concepts=None, show_plot=True):
    """
    Create patch-level heatmaps overlaid on original image for each concept across layers.
    Optimized version with GPU acceleration and reduced matplotlib overhead.
    
    Args:
        similarities_by_layer: Dict mapping layer percentage to concept similarities
        concept_names: List of concept names to visualize
        image_idx: Index of the image to visualize
        original_image: PIL Image of the original image
        patch_size: Size of each patch (default 14 for CLIP)
        dataset_name: Name of dataset for saving
        save_path: Optional path to save figure
        model_input_size: Model input size for resizing
        con_label: Optional concept label for file naming (e.g., 'clip_avg')
        per_layer_scale: If True, each layer uses its own color scale. If False, uses global scale.
        selected_concepts: Optional list of concept names to filter to
        show_plot: If False, disable interactive display (faster when just saving)
    """
    from utils.general_utils import pad_or_resize_img
    from utils.patch_alignment_utils import calculate_patch_indices, get_patch_range_for_image, compute_patches_per_image
    
    # Filter to selected concepts if provided
    if selected_concepts is not None:
        concept_names = [name for name in concept_names if name in selected_concepts]
    
    # Get layer percentages and sort them
    layer_percentages = sorted(similarities_by_layer.keys())
    
    # Calculate proper patch dimensions using existing functions
    patches_per_image = compute_patches_per_image(patch_size=patch_size, model_input_size=model_input_size)
    patches_per_row, patches_per_col, _ = calculate_patch_indices(image_idx, 0, patch_size, model_input_size)
    
    print(f"Using patches_per_image={patches_per_image}, grid={patches_per_row}x{patches_per_col}")
    
    # Resize image to model input size (same as during embedding extraction)
    resized_image = pad_or_resize_img(original_image, model_input_size)
    grayscale_image = resized_image.convert('L')
    
    # Determine color scale range
    if per_layer_scale:
        # Per-layer scales will be calculated individually
        vmin_global, vmax_global = None, None
    else:
        # Global scale across all layers and concepts for this image
        all_values = []
        for layer_pct in layer_percentages:
            for concept_name in concept_names:
                similarities = similarities_by_layer[layer_pct][concept_name]
                start_idx = image_idx * patches_per_image
                end_idx = (image_idx + 1) * patches_per_image
                image_similarities = similarities[start_idx:end_idx]
                all_values.extend(image_similarities.cpu().numpy())
        
        vmin_global, vmax_global = min(all_values), max(all_values)
    
    # GPU-accelerated preprocessing if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Pre-compute all similarity grids on GPU for faster processing
    all_similarity_grids = {}
    if device.type == 'cuda':
        with torch.no_grad():
            for concept_name in concept_names:
                concept_grids = {}
                for layer_pct in layer_percentages:
                    similarities = similarities_by_layer[layer_pct][concept_name]
                    
                    # Extract similarities for the specific image
                    start_idx = image_idx * patches_per_image
                    end_idx = (image_idx + 1) * patches_per_image
                    image_similarities = similarities[start_idx:end_idx]
                    
                    # Move to GPU if not already there
                    if not image_similarities.is_cuda:
                        image_similarities = image_similarities.to(device)
                    
                    # Reshape on GPU
                    similarity_grid = image_similarities.reshape(patches_per_col, patches_per_row)
                    concept_grids[layer_pct] = similarity_grid
                    
                all_similarity_grids[concept_name] = concept_grids
    
    # Create figure with subplots: 1 row for original images + rows for concepts, columns = layers
    fig, axes = plt.subplots(len(concept_names) + 1, len(layer_percentages), 
                            figsize=(3 * len(layer_percentages), 3 * (len(concept_names) + 1)))
    
    # Handle single concept case - axes will be (2, n_layers) for 1 concept + 1 original row
    if len(concept_names) == 1:
        axes = axes.reshape(2, -1)
    
    # First row: show original image across all columns
    for layer_idx, layer_pct in enumerate(layer_percentages):
        ax = axes[0, layer_idx]
        ax.imshow(resized_image)
        ax.set_title(f'Layer {layer_pct}%', fontsize=10)
        ax.axis('off')
    
    # Add row label for original image
    axes[0, 0].text(-0.1, 0.5, 'Original\nImage', fontsize=12,
                    verticalalignment='center', horizontalalignment='right',
                    transform=axes[0, 0].transAxes)
    
    # Subsequent rows: concept heatmaps
    for concept_idx, concept_name in enumerate(concept_names):
        # Add concept label to the left of the first column
        axes[concept_idx + 1, 0].text(-0.1, 0.5, f'{concept_name}', fontsize=12,
                                    verticalalignment='center', horizontalalignment='right',
                                    transform=axes[concept_idx + 1, 0].transAxes)
        
        for layer_idx, layer_pct in enumerate(layer_percentages):
            ax = axes[concept_idx + 1, layer_idx]  # +1 to account for original image row
            
            # Get pre-computed similarity grid or compute on-the-fly
            if device.type == 'cuda' and concept_name in all_similarity_grids:
                similarity_grid = all_similarity_grids[concept_name][layer_pct].cpu().numpy()
            else:
                # Original CPU path
                similarities = similarities_by_layer[layer_pct][concept_name]
                start_idx = image_idx * patches_per_image
                end_idx = (image_idx + 1) * patches_per_image
                image_similarities = similarities[start_idx:end_idx]
                similarity_grid = image_similarities.reshape(patches_per_col, patches_per_row).cpu().numpy()
            
            # Show grayscale image as base
            ax.imshow(grayscale_image, cmap='gray', alpha=0.6)
            
            # Determine color scale for this layer
            if per_layer_scale:
                # Calculate per-layer scale across all concepts for this layer
                layer_values = []
                for cn in concept_names:
                    layer_similarities = similarities_by_layer[layer_pct][cn]
                    layer_image_similarities = layer_similarities[start_idx:end_idx]
                    layer_values.extend(layer_image_similarities.cpu().numpy())
                vmin_layer, vmax_layer = min(layer_values), max(layer_values)
            else:
                vmin_layer, vmax_layer = vmin_global, vmax_global
            
            # Overlay heatmap with transparency
            image_width, image_height = resized_image.size
            heatmap_overlay = ax.imshow(similarity_grid, cmap='hot', alpha=0.6, 
                                      extent=(0, image_width, image_height, 0),
                                      vmin=vmin_layer, vmax=vmax_layer)
            
            # Set title and remove axes
            ax.set_title(f'Max: {similarity_grid.max():.3f}', fontsize=10)
            ax.axis('off')
    
    # Add a single color bar for the entire plot
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(heatmap_overlay, cax=cbar_ax)
    if per_layer_scale:
        cbar.set_label('Cosine Similarity (per-layer scale)')
    else:
        cbar.set_label('Cosine Similarity')
    
    # Set main title
    fig.suptitle(f'Concept Emergence Across Layers - Image {image_idx}', fontsize=16)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Save figure
    if save_path is None:
        save_dir = f'../Figs/{dataset_name}/layer_analysis'
        os.makedirs(save_dir, exist_ok=True)
        
        # Build filename with scaling type
        scale_suffix = "_per_layer_scale" if per_layer_scale else ""
        if con_label:
            save_path = os.path.join(save_dir, f'patch_heatmaps_image_{image_idx}_{con_label}{scale_suffix}.png')
        else:
            save_path = os.path.join(save_dir, f'patch_heatmaps_image_{image_idx}{scale_suffix}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)  # Close figure to free memory


def save_layer_embeddings(embeddings_by_layer, dataset_name, model_name, prefix='test_images'):
    """
    Save layer embeddings to disk with proper naming convention.
    
    Args:
        embeddings_by_layer: Dict mapping layer percentage to embeddings
        dataset_name: Name of dataset
        model_name: Name of model
        prefix: Prefix for filename
    """
    embeddings_dir = f'Embeddings/{dataset_name}/'
    os.makedirs(embeddings_dir, exist_ok=True)
    
    for layer_pct, embeddings in embeddings_by_layer.items():
        filename = f'{prefix}_{model_name}_percentthrumodel_{layer_pct}_patch_embeddings.pt'
        filepath = os.path.join(embeddings_dir, filename)
        torch.save(embeddings, filepath)
    
    print(f"Layer embeddings saved to {embeddings_dir}")