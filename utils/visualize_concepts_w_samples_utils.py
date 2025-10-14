import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import math
import ast
from tqdm import tqdm

import torch.nn.functional as F
import torch

import sys
import os
sys.path.append(os.path.abspath(".."))

from utils.general_utils import retrieve_image, get_split_df, pad_or_resize_img, load_images, create_image_loader_function
from utils.patch_alignment_utils import compute_patches_per_image, calculate_patch_location, compute_patch_similarities_to_vector, get_image_idx_from_global_patch_idx, get_patch_split_df, calculate_patch_indices, filter_patches_by_image_presence

######### only when there's gt labels #########
def get_user_category(concept_columns):
    """
    Helper function to get the user's choice for concept category.

    Args:
        concept_columns (list): A list of concept column names.

    Returns:
        tuple: The selected category and the concept columns.
    """
    print("Available concept categories:")
    categories = sorted(set(col.split('::')[0] for col in concept_columns))  # Get unique concept categories
    for idx, category in enumerate(categories):
        print(f"{idx + 1}. {category}")
    
    while True:
        try:
            choice = int(input(f"Enter the number of the concept category you want to choose (1-{len(categories)}): "))
            if 1 <= choice <= len(categories):
                return categories[choice - 1], concept_columns
            else:
                print("Invalid choice, please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_user_concept(category, concept_columns):
    """
    Helper function to get the user's choice for specific concept within a category.

    Args:
        category (str): The selected concept category.
        concept_columns (list): A list of concept column names.

    Returns:
        str: The selected concept.
    """
    # Filter concepts based on selected category
    concepts = [col for col in concept_columns if col.startswith(category)]
    print(f"\nAvailable concepts in category '{category}':")
    for idx, concept in enumerate(concepts):
        print(f"{idx + 1}. {concept.split('::')[1]}")  # Display the specific concept (e.g., 'red', 'cube', etc.)
    
    while True:
        try:
            choice = int(input(f"Enter the number of the specific concept you want to choose (1-{len(concepts)}): "))
            if 1 <= choice <= len(concepts):
                return concepts[choice - 1]
            else:
                print("Invalid choice, please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def plot_aligned_images(acts_loader, con_label, concept_key=None, k=5, dataset_name='CLEVR', metric_type='Cosine Similarity', save_image=False, test_only=True):
    """
    Plot images that align well with a selected concept.

    Args:
        acts_loader: ChunkedActivationLoader instance or DataFrame with activations.
        con_label (str): label to put in path of saved image.
        concept_key (str): The concept to visualize. If None, will prompt user.
        k (int): Number of top images to display. Defaults to 5.
        dataset_name (str): The name of the dataset. Defaults to 'CLEVR'.
        metric_type (str): Type of metric being visualized.
        save_image (bool): Whether to save png of plots.
        test_only (bool): Whether to only consider test samples.

    Returns:
        None
    """
    # Load activations - handle both loader and DataFrame inputs
    if hasattr(acts_loader, 'load_full_dataframe'):
        comp_df = acts_loader.load_full_dataframe()
    else:
        comp_df = acts_loader  # Assume it's already a DataFrame
    
    # Filter for test samples if requested
    if test_only:
        metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
        test_indices = metadata[metadata['split'] == 'test'].index
        comp_df = comp_df.loc[comp_df.index.intersection(test_indices)]
    
    # Get the user's choice of concept if not provided
    concept_columns = list(comp_df.columns)
    if not concept_key:
        category, concept_columns = get_user_category(concept_columns)
        concept_key = get_user_concept(category, concept_columns)
    
    # Check if concept exists
    if concept_key not in comp_df.columns:
        print(f"Concept '{concept_key}' not found. Available concepts:")
        print(sorted(comp_df.columns)[:10], "...")
        return
    
    # Sort by cosine similarity and get the top k highest values for the specified concept
    top_k_indices = comp_df.nlargest(k, concept_key).index.tolist()
    
    # Calculate the number of rows and columns for the plot
    n_cols = k  # All images in one row
    n_rows = 1  # Single row

    # Plot the top k images based on cosine similarity
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * k, 5))
    if k == 1:
        axes = [axes]  # Ensure axes is always a list
    elif n_rows == 1:
        axes = axes.flatten()  # Flatten for consistency

    plt.suptitle(f"Top {k} Images with Highest {metric_type} to: Concept {concept_key}", fontsize=16)
    
    # Load only the images we need
    loaded_images = {}
    for idx in top_k_indices:
        loaded_images[idx] = retrieve_image(idx, dataset_name, test_only=False)
    
    for rank, idx in enumerate(top_k_indices):
        if rank >= len(axes):  # In case there are fewer images than axes
            break
        
        # Get the image from our loaded images
        img = loaded_images[idx]

        value = comp_df.loc[idx, concept_key]
        axes[rank].imshow(img)
        axes[rank].set_title(f"Rank {rank+1}: Image {idx}\n{metric_type} = {value:.4f}")
        axes[rank].axis('off')
    
    # Hide unused axes
    for rank in range(len(top_k_indices), len(axes)):
        axes[rank].axis('off')

    plt.tight_layout()
    
    if save_image:
        save_path = f'../Figs/{dataset_name}/most_aligned_w_concepts/concept_{concept_key}_{k}__{con_label}.jpg'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        
    plt.show()
    
    
###### Patch Similarities #####
def neighboring_patch_comparisons(image_index, patch_index_in_image, loader, 
                                  dataset_name, model_input_size, patch_size=14, 
                                  save_path=None):
    """
    Plots a heatmap of cosine similarity between a chosen patch's embedding and all other patches in an image.
    The heatmap is overlayed on the original image, and the original image is shown separately to the left.

    Args:
        image_index (int): The index of the image in the dataset.
        patch_index_in_image (int): The index of the patch within the image.
        loader: ChunkedEmbeddingLoader instance for loading embeddings.
        dataset_name (str): The name of the dataset.
        model_input_size (tuple): The input size to which the image should be resized (width, height).
        patch_size (int): The size of each patch.
        save_path (str) : Where to save path.

    Returns:
        None: Displays the heatmap overlayed on the image.
    """
    # Retrieve the image
    image = retrieve_image(image_index, dataset_name, test_only=False)
    resized_image = pad_or_resize_img(image, model_input_size)

    # Calculate patch indices
    patches_per_row, patches_per_col, global_patch_idx = calculate_patch_indices(
        image_index, patch_index_in_image, patch_size, model_input_size
    )

    # Calculate indices for all patches in the image
    patches_per_image = patches_per_row * patches_per_col
    image_start_idx = image_index * patches_per_image
    image_patch_indices = list(range(image_start_idx, image_start_idx + patches_per_image))
    
    # Load all necessary embeddings in one go
    all_indices_needed = image_patch_indices + [global_patch_idx]
    all_indices_needed = list(set(all_indices_needed))  # Remove duplicates
    embeddings = loader.load_specific_embeddings(all_indices_needed)
    
    # Create a mapping from global indices to loaded position
    idx_to_position = {idx: pos for pos, idx in enumerate(all_indices_needed)}
    
    # Get the selected patch embedding
    selected_patch_position = idx_to_position[global_patch_idx]
    selected_patch_embedding = embeddings[selected_patch_position]
    
    # Get embeddings for all patches in the image
    image_patch_embeddings = torch.stack([
        embeddings[idx_to_position[idx]] for idx in image_patch_indices
    ])

    # Compute cosine similarities
    cos_sims = F.cosine_similarity(
        selected_patch_embedding.unsqueeze(0),
        image_patch_embeddings
    ).cpu()

    # Reshape similarities to match the patch grid
    cos_sim_grid = cos_sims.reshape(patches_per_col, patches_per_row)

    # Plot the heatmap
    plot_patches_sim_to_vector(
        cos_sim_grid, resized_image, patch_size, image_index, patch_index_in_image, save_path=save_path,
        plot_title = f'Patch Similarity (Image {image_index}, Patch {patch_index_in_image})',
        bar_title='Cosine Similarity with Chosen Patch'
    )
    

def make_image_with_highlighted_patch(image, left, top, right, bottom, model_input_size=(224, 224), plot_image_title=None, grayscale=False):
    """
    Helper function to return the original image with a red rectangle highlighting the patch.

    Args:
        image (PIL.Image): The original image to return.
        left, top, right, bottom (int): The coordinates of the patch.
        model_input_size (tuple): The size to which the image is resized during embedding.
        plot_image_title (str, optional): Title for the plot.

    Returns:
        PIL.Image: The image with the highlighted patch.
    """
    # Resize the image to match the embedding process
    resized_image = pad_or_resize_img(image, model_input_size)
    
    # Convert to numpy for cropping if needed
    image_np = np.array(resized_image.convert('RGB'))
    
    # Only crop for LLAMA (560x560), not for CLIP (224x224)
    if model_input_size == (224, 224):
        display_image = image_np
    else:
        # Crop to remove padding for LLAMA
        display_image = top_left_crop_to_original_aspect(image_np, image.size, resized_image.size)
    
    # Convert back to PIL
    image_with_patch = Image.fromarray(display_image)
    if grayscale:
        image_with_patch = image_with_patch.convert('L').convert('RGB')
    
    # Draw the rectangle on the image
    draw = ImageDraw.Draw(image_with_patch)
    
    # Check if patch is within bounds for LLAMA
    if model_input_size != (224, 224):
        crop_height = display_image.shape[0]
        if bottom <= crop_height:
            draw.rectangle([left, top, right, bottom], outline="blue", width=5)
    else:
        draw.rectangle([left, top, right, bottom], outline="blue", width=5)

    if plot_image_title is not None:
        plt.imshow(image_with_patch)
        plt.title(plot_image_title)
        plt.axis('off')
        plt.show()

    return image_with_patch
    
    
def plot_patches_w_corr_images(patch_indices, concept_cos_sims, images, overall_title, model_input_size,
                               save_path=None, patch_size=14, metric_type='CosSim'):
    """
    Helper function to plot the original images with highlighted patches and the patches themselves.

    Args:
        patch_indices (list): List of patch indices to plot.
        concept_cos_sims (pd.Series): Cosine similarity values for the concept.
        images (list of PIL.Image): List of original images.
        overall_title (str): Title of the figure.
        save_path (str): Where to save plots.
        patch_size (int): Size of each patch.
        model_input_size (tuple): The size to which the image is resized during embedding.

    Returns:
        None: Returns the images with highlighted patches and corresponding patches.
    """
    #just display figure if it already was computed
    # if os.path.exists(save_path):
    #     plt.figure(figsize=(15, 10))
    #     plt.imshow(Image.open(save_path))
    #     plt.axis('off')
    #     plt.show()
    #     return
    
    top_n = len(patch_indices)
    
    fig, axes = plt.subplots(2, top_n, figsize=(top_n * 3, 6))

    for i, patch_idx in enumerate(patch_indices):
        # Determine the image index
        #image_idx = patch_idx // ((model_input_size[0] // patch_size) * (model_input_size[1] // patch_size))get_
        image_idx = get_image_idx_from_global_patch_idx(patch_idx, model_input_size, patch_size)
        image = images[image_idx]

        # Calculate the patch location
        left, top, right, bottom = calculate_patch_location(image, patch_idx, patch_size, model_input_size)

        # Highlight the patch
        image_with_patch = make_image_with_highlighted_patch(image, left, top, right, bottom, model_input_size)

        # Plot the image with highlighted patch
        axes[0, i].imshow(image_with_patch)
        axes[0, i].set_title(f'Image {image_idx}')
        axes[0, i].axis('off')

        # Resize and prepare image for display
        resized_image = pad_or_resize_img(image, model_input_size)
        image_np = np.array(resized_image.convert("RGB"))
        
        # Only crop for LLAMA (560x560), not for CLIP (224x224)
        if model_input_size == (224, 224):
            display_image = image_np
        else:
            # Crop to remove padding for LLAMA
            display_image = top_left_crop_to_original_aspect(image_np, image.size, resized_image.size)
        
        # Convert back to PIL for drawing
        display_image_pil = Image.fromarray(display_image)
        draw = ImageDraw.Draw(display_image_pil)
        
        # Draw rectangle around patch
        if model_input_size != (224, 224):
            # For cropped images, check if patch is within bounds
            crop_height = display_image.shape[0]
            if bottom <= crop_height:
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
        else:
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
        
        # Show the image with highlighted patch
        axes[1, i].imshow(display_image_pil)
        try:
            axes[1, i].set_title(f'Patch {patch_idx} ({metric_type}: {concept_cos_sims[patch_idx]:.2f})')
        except:
            axes[1, i].set_title(f'Patch {patch_idx} ({metric_type}: {concept_cos_sims.iloc[patch_idx]:.2f})')
        
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.suptitle(overall_title, fontsize=16, y=1.05)
                   
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
      
    plt.show()

    

def plot_top_patches_for_concept(concept_label, cos_sims, images, dataset_name, save_path='', 
                                 top_n=5, patch_size=14, model_input_size=(224, 224), metric_type='CosSim',
                                 test_only=True):
    """
    Given a concept label, plot the given patches that align most with concept 
    with their original images and highlight the patch locations.

    Args:
        concept_label (str): The concept label for which to plot patches.
        cos_sims (pd.DataFrame): DataFrame with cosine similarities, where rows are patch indices
                                 and columns are concept labels.
        images (list of PIL.Image): List of PIL Image objects, the original images.
        save_path (str): Where to save the resulting figure.
        top_n (int): The number of patches to plot for each concept.
        patch_size (int): Size of each patch (default is 14).
        model_input_size (tuple): The size to which the image is resized during embedding.

    Returns:
        None: Displays the patches and their respective images with highlighted locations.
    """   
    split_df = get_patch_split_df(dataset_name, patch_size=14, model_input_size=model_input_size)
    
    if test_only:
        test_image_indices = split_df[split_df == 'test'].index
        cos_sims = cos_sims.loc[test_image_indices]
    
    #Get the cosine similarity values for the specified concept
    concept_cos_sims = cos_sims[concept_label]
    
    #Sort the patches by cosine similarity in descending order
    top_patch_indices = concept_cos_sims.nlargest(top_n).index 

    #Call the helper function to plot the images and patches
    overall_title = f'{top_n} Test Patches Most Activated by Concept {concept_label}'
    plot_patches_w_corr_images(top_patch_indices, concept_cos_sims, images, overall_title, save_path=save_path, patch_size=patch_size, model_input_size=model_input_size, metric_type=metric_type)
    

def plot_most_similar_patches_w_heatmaps_and_corr_images(concept_label, acts_loader, con_label, dataset_name, model_input_size, vmin=None, vmax=None, save_path="", patch_size=14, top_n=5, metric_type='Cosine Similarity', test_only=True):
    """
    Plots the most similar patches with a chosen concept, as well as the heatmaps for that concept and the corresponding image.
    
    Args:
        concept_label (str): The concept to visualize.
        acts_loader: ChunkedActivationLoader instance or DataFrame with activations.
        con_label (str): Label for saving.
        dataset_name (str): Name of the dataset.
        model_input_size (tuple): Model input size.
        vmin (float): Minimum value for heatmap color scale.
        vmax (float): Maximum value for heatmap color scale.
        save_path (str): Where to save the figure.
        patch_size (int): Size of patches.
        top_n (int): Number of top patches to show.
        metric_type (str): Type of metric.
        test_only (bool): Whether to only use test samples.
    """
    
    # Load activations - handle both loader and DataFrame inputs
    if hasattr(acts_loader, 'load_full_dataframe'):
        cos_sims = acts_loader.load_full_dataframe()
    else:
        cos_sims = acts_loader  # Assume it's already a DataFrame
    
    # Filter for test samples if needed
    if test_only:
        split_df = get_patch_split_df(dataset_name, model_input_size, patch_size)
        test_indices = split_df[split_df == 'test'].index
        cos_sims_filtered = cos_sims.loc[cos_sims.index.intersection(test_indices)]
    else:
        cos_sims_filtered = cos_sims
    
    # Check if concept exists
    if concept_label not in cos_sims.columns:
        print(f"Concept '{concept_label}' not found. Available concepts:")
        print(sorted(cos_sims.columns)[:10], "...")
        return
    
    # Get top patches
    most_similar_patches = cos_sims_filtered[concept_label].sort_values(ascending=False).head(top_n).index
    
    fig, axes = plt.subplots(2, top_n, figsize=(top_n * 3, 6))
    if top_n == 1:
        axes = axes.reshape(-1, 1)
    
    heatmaps = {}
    images_w_patches = []
    
    # Calculate patches per image
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Determine which images we need to load
    image_indices_needed = set()
    for patch_idx in most_similar_patches:
        image_idx = get_image_idx_from_global_patch_idx(patch_idx, model_input_size, patch_size)
        image_indices_needed.add(image_idx)
    
    # Load only the needed images
    loaded_images = {}
    for image_idx in image_indices_needed:
        loaded_images[image_idx] = retrieve_image(image_idx, dataset_name, test_only=False)
    
    for patch_idx in most_similar_patches:
        # Determine the image index
        image_idx = get_image_idx_from_global_patch_idx(patch_idx, model_input_size, patch_size)
        image = loaded_images[image_idx]
        
        # Get patch activations for this image to create heatmap
        start_idx = image_idx * patches_per_image
        end_idx = start_idx + patches_per_image
        image_patch_acts = cos_sims[concept_label].iloc[start_idx:end_idx]
        
        # Reshape to 2D heatmap
        heatmap = torch.tensor(image_patch_acts.values).reshape(patches_per_col, patches_per_row)
        
        # Calculate the patch location
        left, top, right, bottom = calculate_patch_location(image, patch_idx, patch_size, model_input_size)
        
        # Highlight the patch
        image_with_patch = make_image_with_highlighted_patch(image, left, top, right, bottom, model_input_size, grayscale=True)
        
        heatmaps[image_idx] = heatmap
        images_w_patches.append(image_with_patch)
    
    # Determine the global color scale range across all heatmaps
    if vmin is None or vmax is None:
        all_values = [value.item() for heatmap in heatmaps.values() for row in heatmap for value in row]
        vmin, vmax = min(all_values), max(all_values)
    
    for i, patch_idx in enumerate(most_similar_patches):
        image_idx = patch_idx // patches_per_image
        image = loaded_images[image_idx]
        resized_image = pad_or_resize_img(image, model_input_size)
        
        # Plot the original image
        axes[0, i].imshow(resized_image)
        axes[0, i].set_title(f'Image {image_idx}')
        axes[0, i].axis('off')
        
        # Plot the image with highlighted patch and heatmap
        heatmap = heatmaps[image_idx]
        axes[1, i].imshow(images_w_patches[i], alpha=0.6)
        heatmap_overlay = axes[1, i].imshow(heatmap, cmap='hot', alpha=0.4, 
                                           extent=(0, model_input_size[0], model_input_size[1], 0),
                                           vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Max = {round(heatmap.max().item(), 4)}\nMin = {round(heatmap.min().item(), 4)}')
        axes[1, i].axis('off')
    
    # Add a color bar for the heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(heatmap_overlay, cax=cbar_ax)
    cbar.set_label(metric_type)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    plt.suptitle(f"Most Activated {'Test' if test_only else ''} Patches by Concept {concept_label}", fontsize=16, y=1.05)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    
    plt.show()
    

def plot_patchsims_for_concept(concept_label, heatmaps, image_indices, images, model_input_size, 
                               dataset_name, top_n=7, save_file=None, 
                               metric_type='Cosine Similarity', vmin=None, vmax=None):
    """
    Plots patch similarities for a single concept across multiple images with a consistent color scale,
    using precomputed heatmaps provided in the heatmaps parameter. The original image is displayed above its corresponding heatmap.

    Args:
        concept_label (str): The name of the concept to be visualized.
        image_indices (list of int): A list of image indices to visualize for the given concept.
        heatmaps (dict): A dictionary where keys are image indices and values are precomputed heatmaps.
        images (list of PIL.Image): A list of images.
        model_input_size (tuple): The dimensions (width, height) to which the image is resized for model input.
        dataset_name (str): Dataset name (default is 'CLEVR').
        save_file (str): Where to save the heatmap png. If None, the plot is not saved.
    """
    
    # Determine the global color scale range across all heatmaps
    if not vmin or not vmax:
        all_values = [value for heatmap in heatmaps.values() for row in heatmap for value in row]
        vmin, vmax = min(all_values), max(all_values)

    # Create a figure with a size based on the number of images
    fig, axes = plt.subplots(2, top_n, figsize=(top_n * 3, 6))  # 2 rows: 1 for images, 1 for heatmaps

    for i, image_index in enumerate(image_indices):
        # Retrieve the image and corresponding heatmap
        image = retrieve_image(image_index, dataset_name)
        heatmap = heatmaps[image_index]

        # Resize the image
        resized_image = pad_or_resize_img(image, model_input_size)

        # Get the axes for the image and heatmap
        ax_image = axes[0, i]  # Top row for the image
        ax_heatmap = axes[1, i]  # Bottom row for the heatmap

        # Plot the image in the top row (ax_image)
        ax_image.imshow(resized_image)
        ax_image.set_title(f'Image {image_index}')
        ax_image.axis('off')

        # Plot the heatmap in the bottom row (ax_heatmap)
        heatmap_overlay = ax_heatmap.imshow(heatmap, cmap='hot', alpha=0.6, vmin=vmin, vmax=vmax, extent=[0, model_input_size[0], model_input_size[1], 0])

        ax_heatmap.set_title(f'Heatmap Max = {round(heatmap.max().item(), 4)}\nHeatmap Min = {round(heatmap.min().item(), 4)}')
        #ax_heatmap.set_title(f'Heatmap Max = {round(heatmap.max().item(), 2)}')
        ax_heatmap.imshow(resized_image.convert('L'), cmap='gray', alpha=0.4)
        ax_heatmap.axis('off')

    # Add a color bar for the heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position of the color bar
    cbar = plt.colorbar(heatmap_overlay, cax=cbar_ax)
    cbar.set_label(metric_type)

    # Adjust layout to prevent overlap
    plt.suptitle(f"Concept {concept_label} Activations", fontsize=16, y=1.05)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the color bar
    

    # Optionally save the figure
    if save_file:
        save_path = f'../Figs/{dataset_name}/patch_alignment_to_concept/{concept_label}_allimages.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=500)

    plt.show()
    

def plot_patchsims_heatmaps_all_concepts(concept_labels, heatmaps, image_indices,
                                           model_input_size, dataset_name, top_n=7, 
                                           save_file=None, metric_type='Cosine Similarity', vmin=None, vmax=None):
    """
    Plots patch similarities for multiple concepts across multiple images with a consistent color scale.
    The original images are displayed only once at the top, with heatmaps for each concept in subsequent rows.

    Args:
        concept_labels (list of str): List of concept names to be visualized.
        image_indices (list of int): A list of image indices to visualize for the given concepts.
        heatmaps (dict): A dictionary where keys are image indices and values are precomputed heatmaps.
        model_input_size (tuple): The dimensions (width, height) to which the image is resized for model input.
        dataset_name (str): Dataset name (default is 'CLEVR').
        save_file (str): Where to save the heatmap png. If None, the plot is not saved.
    """

    # Determine the global color scale range across all heatmaps
    if vmin is None or vmax is None:
        vmin, vmax = np.inf, -np.inf
        for k, v in heatmaps.items():
            for k2, v2 in v.items():
                vmin = min(vmin, v2.min())
                vmax = max(vmax, v2.max())
#     if vmin is None or vmax is None:
#         all_vals = []
#         for concept_dict in heatmaps.values():
#             for heatmap in concept_dict.values():
#                 all_vals.append(heatmap.flatten())
#         all_vals = np.concatenate(all_vals)

#         # Set vmin/vmax using percentiles to ignore extreme outliers
#         vmin = np.percentile(all_vals, 1)
#         vmax = np.percentile(all_vals, 99)

    # Create a figure with a size based on the number of concepts and images
    num_concepts = len(concept_labels)
    fig, axes = plt.subplots(num_concepts + 1, top_n, figsize=(top_n * 3, (num_concepts + 1) * 3))  # 1 row for images, other rows for heatmaps

    # First, plot the original images on the top row
    for i, image_index in enumerate(image_indices):
        image = retrieve_image(image_index, dataset_name)
        resized_image = pad_or_resize_img(image, model_input_size)

        ax_image = axes[0, i]  # Top row for images
        ax_image.imshow(resized_image)
        ax_image.set_title(f'Image {image_index}')
        ax_image.axis('off')

    # Set the row labels for each concept once (to the left of the row)
    axes[0, 0].text(-0.1, 0.5, 'Original\nImage', fontsize=20,
                              verticalalignment='center', horizontalalignment='right',
                              transform=axes[0, 0].transAxes)
    for j, concept_label in enumerate(concept_labels):
        axes[j + 1, 0].text(-0.1, 0.5, f'{concept_label}', fontsize=20,
                              verticalalignment='center', horizontalalignment='right',
                              transform=axes[j + 1, 0].transAxes)

    # Then, loop through images and concepts to plot the heatmaps
    for i, image_index in enumerate(image_indices):
        image = retrieve_image(image_index, dataset_name)
        resized_image = pad_or_resize_img(image, model_input_size)

        for j, concept_label in enumerate(concept_labels):
            heatmap = heatmaps[concept_label][image_index]  # Access heatmap for this concept and image

            ax_heatmap = axes[j + 1, i]  # Heatmap rows are below the image row
            ax_heatmap.imshow(heatmap, cmap='hot', alpha=0.6, vmin=vmin, vmax=vmax,
                              extent=[0, model_input_size[0], model_input_size[1], 0])
            ax_heatmap.set_title(f'Heatmap Max = {round(heatmap.max().item(), 4)}\n'
                                 f'Heatmap Min = {round(heatmap.min().item(), 4)}')
            ax_heatmap.imshow(resized_image.convert('L'), cmap='gray', alpha=0.4)  # Overlay the image in gray
            ax_heatmap.axis('off')

    # Add a color bar for the heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap='hot')
    sm.set_array([])  # You don't need to set any specific array data.
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(metric_type)

    # Adjust layout to prevent overlap
    plt.suptitle(f"Concept Activations on Random Test {dataset_name} Images", fontsize=16, y=1)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the color bar
    
    # Optionally save the figure
    if save_file:
        save_path = f'../Figs/{dataset_name}/patch_alignment_to_concept/{concept_label}_allimages.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=500)

    plt.show()
    

def plot_patchsims_all_concepts(img_idx, heatmaps, model_input_size, dataset_name, 
                                metric_type = 'Cosine Similarity', vmin=None, vmax=None, 
                                sort_by_act=False, save_file=None):
    """
    Plots all patch similarities for all concepts with a consistent color scale.

    Args:
        img_idx (int): Index of the image in the metadata.
        heatmaps (dict): Dictionary where keys are concepts and values are heatmaps.
        model_input_size (tuple): Resized image size.
        dataset_name (str): Dataset name (default is 'CLEVR').
        save_file (str): Where to save heatmap png.
    """
    print(f"Plotting image {img_idx}:")
    # Retrieve the image and convert to grayscale
    image = retrieve_image(img_idx, dataset_name)
    resized_image = pad_or_resize_img(image, model_input_size)
    grayscale_image = resized_image.convert('L')

    # Determine the global color scale range across all heatmaps
    if not vmin or not vmax:
        all_values = [value for heatmap in heatmaps.values() for row in heatmap for value in row]
        vmin, vmax = min(all_values), max(all_values)

    # Plot each heatmap
    if sort_by_act:
        concepts = sorted(heatmaps.keys(), key=lambda x: heatmaps[x].max().item(), reverse=True)
    else:
        concepts = list(heatmaps.keys())
    n_concepts = len(concepts)
    plt.figure(figsize=(16, 4 * ((n_concepts + 2) // 3)))  # Adjust figure size dynamically

    for i, concept in enumerate(concepts):
        heatmap = heatmaps[concept]
        ax = plt.subplot((n_concepts + 2) // 3, 3, i + 1)

        # Plot the grayscale image with heatmap overlay
        image_width, image_height = grayscale_image.size
        heatmap_overlay = ax.imshow(heatmap, cmap='hot', alpha=0.4, extent=(0, image_width, image_height, 0),
                                   vmin=vmin, vmax=vmax)
        
        # ax.set_title(f'Heatmap Avg = {round(heatmap.mean().item(), 4)}\nHeatmap Max = {round(heatmap.max().item(), 4)}\nHeatmap Min = {round(heatmap.min().item(), 4)}')
        
        # Plot the grayscale image with heatmap overlay
        ax.imshow(grayscale_image, cmap='gray', alpha=0.6)
        plt.title(f'{concept}\nHeatmap Max = {round(heatmap.max().item(), 4)}\nHeatmap Min = {round(heatmap.min().item(), 4)}')
        plt.axis('off')

    # Add a single color bar for the entire plot
    cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position of the color bar
    cbar = plt.colorbar(heatmap_overlay, cax=cbar_ax)
    cbar.set_label(metric_type)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the color bar
    
    if save_file:
        save_path = f'../Figs/{dataset_name}/patch_alignment_to_concept/allconcepts_{save_file}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        
    plt.show()


def animate_patch_similarities(image_index, loader, dataset_name, model_input_size, 
                              patch_size=14, save_path=None, fps=2, show_animation=True,
                              skip_patches=1, max_frames=None, figsize=(12, 6)):
    """
    Creates an animation cycling through all patches in an image, showing similarity heatmaps.
    
    Args:
        image_index (int): The index of the image in the dataset.
        loader: ChunkedEmbeddingLoader instance for loading embeddings.
        dataset_name (str): The name of the dataset.
        model_input_size (tuple): The input size to which the image should be resized (width, height).
        patch_size (int): The size of each patch.
        save_path (str): Where to save the animation (as .gif or .mp4).
        fps (int): Frames per second for the animation.
        show_animation (bool): Whether to display the animation in the notebook.
        skip_patches (int): Skip every N patches to reduce animation size (1 = no skip).
        max_frames (int): Maximum number of frames to include (None = all patches).
        figsize (tuple): Figure size (width, height) in inches.
        
    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    import matplotlib
    
    # Set matplotlib parameters to reduce animation size
    if show_animation:
        matplotlib.rcParams['animation.embed_limit'] = 40  # Increase limit to 40MB
    
    # Retrieve the image
    image = retrieve_image(image_index, dataset_name, test_only=False)
    resized_image = pad_or_resize_img(image, model_input_size)
    
    # Calculate patch dimensions
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Calculate indices for all patches in the image
    image_start_idx = image_index * patches_per_image
    image_patch_indices = list(range(image_start_idx, image_start_idx + patches_per_image))
    
    # Load all embeddings for this image
    embeddings = loader.load_specific_embeddings(image_patch_indices)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: original image with highlighted patch
    ax1.set_title(f'Image {image_index}')
    ax1.axis('off')
    
    # Right plot: heatmap
    ax2.set_title('Patch Similarity Heatmap')
    ax2.axis('off')
    
    # Initialize plots
    im1 = ax1.imshow(resized_image)
    im2 = ax2.imshow(np.zeros((patches_per_col, patches_per_row)), cmap='hot', alpha=0.5, 
                     extent=(0, model_input_size[0], model_input_size[1], 0), vmin=0, vmax=1)
    im2_bg = ax2.imshow(resized_image, alpha=0.5)
    
    # Create colorbar
    cbar = plt.colorbar(im2, ax=ax2)
    cbar.set_label('Cosine Similarity with Selected Patch')
    
    # Rectangle for highlighting current patch
    rect = plt.Rectangle((0, 0), patch_size, patch_size, 
                        edgecolor='red', facecolor='none', linewidth=3)
    ax1.add_patch(rect)
    
    def animate(frame):
        """Update function for animation."""
        patch_index_in_image = frame
        
        # Calculate patch position
        row, col = divmod(patch_index_in_image, patches_per_row)
        left = col * patch_size
        top = row * patch_size
        
        # Update highlighted patch rectangle
        rect.set_xy((left, top))
        
        # Get the selected patch embedding
        selected_patch_embedding = embeddings[patch_index_in_image]
        
        # Compute cosine similarities with all patches
        cos_sims = F.cosine_similarity(
            selected_patch_embedding.unsqueeze(0),
            embeddings
        ).cpu()
        
        # Reshape to grid
        cos_sim_grid = cos_sims.reshape(patches_per_col, patches_per_row)
        
        # Update heatmap
        im2.set_array(cos_sim_grid)
        im2.set_clim(cos_sim_grid.min(), cos_sim_grid.max())
        
        # Update title with current patch index
        ax2.set_title(f'Similarity to Patch {patch_index_in_image} (Row {row}, Col {col})')
        
        return [rect, im2]
    
    # Determine which frames to include
    if max_frames is not None:
        frames = list(range(0, min(patches_per_image, max_frames), skip_patches))
    else:
        frames = list(range(0, patches_per_image, skip_patches))
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=frames, 
                        interval=1000/fps, blit=True, repeat=True)
    
    # Save animation if requested
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved to {save_path}")
    
    # Show animation
    if show_animation:
        plt.close()  # Close the static plot
        return HTML(anim.to_jshtml())
    else:
        plt.show()
        return anim


def show_patch_similarities_grid(image_index, loader, dataset_name, model_input_size, 
                                patch_size=14, n_patches_to_show=9, patch_indices=None):
    """
    Shows a grid of patch similarity heatmaps for selected patches in an image.
    
    Args:
        image_index (int): The index of the image in the dataset.
        loader: ChunkedEmbeddingLoader instance for loading embeddings.
        dataset_name (str): The name of the dataset.
        model_input_size (tuple): The input size to which the image should be resized (width, height).
        patch_size (int): The size of each patch.
        n_patches_to_show (int): Number of patches to show (if patch_indices not provided).
        patch_indices (list): Specific patch indices to show (optional).
        
    Returns:
        None: Displays the grid of images.
    """
    # Retrieve the image
    image = retrieve_image(image_index, dataset_name, test_only=False)
    resized_image = pad_or_resize_img(image, model_input_size)
    
    # Calculate patch dimensions
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Calculate indices for all patches in the image
    image_start_idx = image_index * patches_per_image
    image_patch_indices = list(range(image_start_idx, image_start_idx + patches_per_image))
    
    # Load all embeddings for this image
    embeddings = loader.load_specific_embeddings(image_patch_indices)
    
    # Determine which patches to show
    if patch_indices is None:
        # Sample evenly across the image
        step = max(1, patches_per_image // n_patches_to_show)
        patch_indices = list(range(0, patches_per_image, step))[:n_patches_to_show]
    
    # Calculate grid layout
    n_cols = min(3, len(patch_indices))
    n_rows = (len(patch_indices) + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each patch similarity
    for idx, patch_idx in enumerate(patch_indices):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx]
        
        # Calculate patch position
        patch_row, patch_col = divmod(patch_idx, patches_per_row)
        
        # Get the selected patch embedding
        selected_patch_embedding = embeddings[patch_idx]
        
        # Compute cosine similarities with all patches
        cos_sims = F.cosine_similarity(
            selected_patch_embedding.unsqueeze(0),
            embeddings
        ).cpu()
        
        # Reshape to grid
        cos_sim_grid = cos_sims.reshape(patches_per_col, patches_per_row)
        
        # Plot the image with heatmap overlay
        ax.imshow(resized_image)
        heatmap = ax.imshow(cos_sim_grid, cmap='hot', alpha=0.5, 
                           extent=(0, model_input_size[0], model_input_size[1], 0),
                           vmin=0, vmax=1)
        
        # Highlight the selected patch
        left = patch_col * patch_size
        top = patch_row * patch_size
        rect = plt.Rectangle((left, top), patch_size, patch_size,
                           edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        
        ax.set_title(f'Patch {patch_idx} (R{patch_row}, C{patch_col})')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(patch_indices), n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx, col_idx].axis('off')
    
    plt.suptitle(f'Patch Similarities for Image {image_index}', fontsize=16)
    plt.tight_layout()
    plt.show()


def show_patch_similarities_simple(image_index, loader, dataset_name, model_input_size, 
                                 patch_size=14, start_patch=None, end_patch=None, delay=0.5):
    """
    Simple version: Shows patch similarity heatmaps one after another.
    
    Args:
        image_index (int): The index of the image in the dataset.
        loader: ChunkedEmbeddingLoader instance for loading embeddings.
        dataset_name (str): The name of the dataset.
        model_input_size (tuple): The input size to which the image should be resized (width, height).
        patch_size (int): The size of each patch.
        start_patch (int): Starting patch index (local to image). If None, starts from 0.
        end_patch (int): Ending patch index (local to image, exclusive). If None, goes to last patch.
        delay (float): Delay in seconds between frames.
    """
    from IPython.display import clear_output, display
    import time
    
    # Retrieve the image
    image = retrieve_image(image_index, dataset_name, test_only=False)
    resized_image = pad_or_resize_img(image, model_input_size)
    
    # Calculate patch dimensions
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Calculate indices for all patches in the image
    image_start_idx = image_index * patches_per_image
    image_patch_indices = list(range(image_start_idx, image_start_idx + patches_per_image))
    
    # Load all embeddings for this image
    embeddings = loader.load_specific_embeddings(image_patch_indices)
    
    # Determine patch range
    if start_patch is None:
        start_patch = 0
    if end_patch is None:
        end_patch = patches_per_image
    
    # Validate range
    start_patch = max(0, min(start_patch, patches_per_image - 1))
    end_patch = max(start_patch + 1, min(end_patch, patches_per_image))
    
    # Create list of patches to show
    patches_to_show = list(range(start_patch, end_patch))
    
    print(f"Showing patches {start_patch} to {end_patch-1} ({len(patches_to_show)} total patches)...")
    
    # Pre-compute min/max values across all patches for consistent colorbar
    all_similarities = []
    for patch_idx in patches_to_show:
        selected_patch_embedding = embeddings[patch_idx]
        cos_sims = F.cosine_similarity(
            selected_patch_embedding.unsqueeze(0),
            embeddings
        ).cpu()
        all_similarities.append(cos_sims)
    
    # Find global min/max for consistent color scaling
    vmin = min(sims.min().item() for sims in all_similarities)
    vmax = max(sims.max().item() for sims in all_similarities)
    
    # Show each frame
    for i, patch_idx in enumerate(patches_to_show):
        # Clear previous output
        clear_output(wait=True)
        
        # Calculate patch position
        patch_row, patch_col = divmod(patch_idx, patches_per_row)
        
        # Use pre-computed similarities
        cos_sims = all_similarities[i]
        
        # Reshape to grid
        cos_sim_grid = cos_sims.reshape(patches_per_col, patches_per_row)
        
        # Create figure with colorbar space
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # Display the image
        ax.imshow(resized_image)
        
        # Overlay the similarity heatmap with consistent scale
        im = ax.imshow(cos_sim_grid, cmap='hot', alpha=0.5, 
                       extent=(0, model_input_size[0], model_input_size[1], 0),
                       vmin=vmin, vmax=vmax)
        
        # Highlight the selected patch with red rectangle
        left = patch_col * patch_size
        top = patch_row * patch_size
        rect = plt.Rectangle((left, top), patch_size, patch_size,
                           edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        
        # Add colorbar with consistent scale
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)
        
        ax.set_title(f'Patch {patch_idx} (Row {patch_row}, Col {patch_col}) - Frame {i+1}/{len(patches_to_show)}')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Wait before next frame
        if i < len(patches_to_show) - 1:
            time.sleep(delay)
    
    print("Animation complete!")


def show_patch_similarities_sequence(image_index, loader, dataset_name, model_input_size, 
                                   patch_size=14, n_patches_to_show=20, delay=0.5, interactive=False):
    """
    Shows patch similarity heatmaps one after another with a delay, creating an animation effect.
    
    Args:
        image_index (int): The index of the image in the dataset.
        loader: ChunkedEmbeddingLoader instance for loading embeddings.
        dataset_name (str): The name of the dataset.
        model_input_size (tuple): The input size to which the image should be resized (width, height).
        patch_size (int): The size of each patch.
        n_patches_to_show (int): Number of patches to show in sequence.
        delay (float): Delay in seconds between frames.
        interactive (bool): If True, use interactive mode for smoother updates.
    """
    from IPython.display import clear_output, display
    import time
    
    if interactive:
        plt.ion()  # Turn on interactive mode
    
    # Retrieve the image
    image = retrieve_image(image_index, dataset_name, test_only=False)
    resized_image = pad_or_resize_img(image, model_input_size)
    
    # Calculate patch dimensions
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Calculate indices for all patches in the image
    image_start_idx = image_index * patches_per_image
    image_patch_indices = list(range(image_start_idx, image_start_idx + patches_per_image))
    
    # Load all embeddings for this image
    embeddings = loader.load_specific_embeddings(image_patch_indices)
    
    # Sample patches evenly
    step = max(1, patches_per_image // n_patches_to_show)
    patches_to_show = list(range(0, patches_per_image, step))[:n_patches_to_show]
    
    # Create figure once if using interactive mode
    if interactive:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.show(block=False)
    
    # Create figure that we'll update
    for i, patch_idx in enumerate(patches_to_show):
        if not interactive:
            # Clear previous output and create new figure
            clear_output(wait=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        else:
            # Clear axes for updating
            ax1.clear()
            ax2.clear()
        
        # Calculate patch position
        patch_row, patch_col = divmod(patch_idx, patches_per_row)
        
        # Left: Original image with highlighted patch
        ax1.imshow(resized_image)
        left = patch_col * patch_size
        top = patch_row * patch_size
        rect = plt.Rectangle((left, top), patch_size, patch_size,
                           edgecolor='red', facecolor='none', linewidth=3)
        ax1.add_patch(rect)
        ax1.set_title(f'Image {image_index} - Patch {patch_idx}')
        ax1.axis('off')
        
        # Get the selected patch embedding and compute similarities
        selected_patch_embedding = embeddings[patch_idx]
        cos_sims = F.cosine_similarity(
            selected_patch_embedding.unsqueeze(0),
            embeddings
        ).cpu()
        
        # Reshape to grid
        cos_sim_grid = cos_sims.reshape(patches_per_col, patches_per_row)
        
        # Right: Heatmap overlay
        ax2.imshow(resized_image, alpha=0.5)
        heatmap = ax2.imshow(cos_sim_grid, cmap='hot', alpha=0.5, 
                           extent=(0, model_input_size[0], model_input_size[1], 0),
                           vmin=0, vmax=1)
        ax2.set_title(f'Similarity to Patch {patch_idx} (Row {patch_row}, Col {patch_col})')
        ax2.axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(heatmap, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity')
        
        plt.suptitle(f'Frame {i+1}/{len(patches_to_show)}', fontsize=14)
        plt.tight_layout()
        
        if interactive:
            # Update the display
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            # Show the figure
            display(fig)
            plt.close(fig)
        
        # Wait before showing next frame (except for last frame)
        if i < len(patches_to_show) - 1:
            time.sleep(delay)
    
    if interactive:
        plt.ioff()  # Turn off interactive mode
        
    print(f"\nAnimation complete! Showed {len(patches_to_show)} patches.")


def plot_patches_sim_to_vector(cos_sim_grid, resized_image, patch_size, image_index, patch_index_in_image, save_path=None, plot_title=None, bar_title=None, show_plot=True):
    """
    Plot a heatmap of cosine similarities overlayed on the resized image.
    """
    image_width, image_height = resized_image.size
    patches_per_row = image_width // patch_size
    row, col = divmod(patch_index_in_image, patches_per_row)
    left = col * patch_size
    top = row * patch_size
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(resized_image)

    # Overlay the similarity heatmap
    heatmap = ax.imshow(cos_sim_grid, cmap='hot', alpha=0.5, extent=(0, image_width, image_height, 0))

    # Highlight the selected patch
    if patch_index_in_image >= 0:
        rect = plt.Rectangle(
            (left, top), patch_size, patch_size,
            edgecolor='red', facecolor='none', linewidth=2
        )
        ax.add_patch(rect)

    # Add colorbar and labels
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label(bar_title)
    ax.set_title(plot_title)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')

    if show_plot:
        plt.show()
    plt.clf()
    plt.close(fig)
        
    return torch.tensor(heatmap.get_array())


def binarize_patchsims_for_concept(concept_label, threshold, heatmaps, image_indices, images,
                                   dataset_name, metric_type, model_input_size, top_n=7, 
                                   save_file=None):
    """
    Plots binarized patch similarities for a single concept across multiple images,
    using precomputed heatmaps that are binarized based on the provided threshold.
    The original image is displayed above its corresponding binarized heatmap.

    Args:
        concept_label (str): The name of the concept to be visualized.
        image_indices (list of int): A list of image indices to visualize for the given concept.
        heatmaps (dict): A dictionary where keys are image indices and values are precomputed heatmaps (torch.Tensor).
        images (list of PIL.Image): A list of images.
        model_input_size (tuple): The dimensions (width, height) to which the image is resized for model input.
        dataset_name (str): Dataset name (default is 'CLEVR').
        top_n (int): Number of images to display.
        threshold (float): Threshold value for binarization.
        save_file (str): Where to save the resulting plot. If None, the plot is not saved.
    """
    fig, axes = plt.subplots(2, top_n, figsize=(top_n * 3, 6))
    
    for i, image_index in enumerate(image_indices):
        # Retrieve image and heatmap
        image = retrieve_image(image_index, dataset_name)
        heatmap = heatmaps[image_index]
        
        # Resize image to the model's input size
        resized_image = pad_or_resize_img(image, model_input_size)
        image_width, image_height = resized_image.size
        
        # Convert heatmap to numpy
        heatmap_np = heatmap.detach().cpu().numpy()
        
        # Binarize the heatmap
        mask = heatmap_np >= threshold
        
        # Upsample mask to the image's dimensions
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
        mask_resized = mask_img.resize((image_width, image_height), resample=Image.NEAREST)
        mask_resized = np.array(mask_resized) > 127
        
        # Convert image to normalized RGB numpy array
        image_np = np.array(resized_image.convert("RGB")) / 255.0
        H, W, _ = image_np.shape
        
        # Create an RGBA composite image
        rgba_image = np.zeros((H, W, 4))
        rgba_image[mask_resized] = np.concatenate([
            image_np[mask_resized],
            np.full((mask_resized.sum(), 1), 1)
        ], axis=1)
        rgba_image[~mask_resized] = np.array([0, 0, 0, 1])
        
        # Plot original image on top
        axes[0, i].imshow(image_np)
        axes[0, i].set_title(f'Image {image_index}')
        axes[0, i].axis('off')
        
        # Plot binarized heatmap below
        axes[1, i].imshow(rgba_image)
        axes[1, i].set_title(f'Binarized Heatmap')
        axes[1, i].axis('off')

    plt.suptitle(f"Concept {concept_label} Binarized {metric_type} (Th={threshold:.5f})", fontsize=16, y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 1])

    if save_file:
        save_path = f'../Figs/{dataset_name}/patch_alignment_to_concept/{concept_label}_binarized.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        print(f"Binarized figure saved to {save_path}")

    plt.show()
    
    
def plot_binarized_patchsims_all_concepts(concept_labels, percentile, heatmaps, image_indices, images,
                                          thresholds, metric_type, model_input_size, 
                                          dataset_name, top_n=7, 
                                          save_file=None, nonconcept_thresholds=False):
    """
    Plots binarized patch similarities for multiple concepts across multiple images.

    Args:
        concept_labels (list of str): List of concept names.
        heatmaps (dict): Dictionary where keys are concept names, values are heatmaps per image.
        image_indices (list of int): List of image indices to visualize.
        images (list of PIL.Image): List of images.
        threshold (float): Threshold for binarization.
        model_input_size (tuple): Image resize dimensions for the model.
        dataset_name (str): Name of the dataset.
        save_file (str): Path to save the plot.
    """

    num_concepts = len(concept_labels)
    fig, axes = plt.subplots(num_concepts + 1, top_n, figsize=(top_n * 3, (num_concepts + 1) * 3))

    if top_n == 1:
        axes = np.expand_dims(axes, axis=1)

    # Plot original images in the top row
    for i, image_index in enumerate(image_indices):
        image = retrieve_image(image_index, dataset_name)
        resized_image = pad_or_resize_img(image, model_input_size)
        axes[0, i].imshow(resized_image)
        axes[0, i].set_title(f'Image {image_index}')
        axes[0, i].axis('off')

    # Add concept labels to the left of each row
    axes[0, 0].text(-0.1, 0.5, 'Original\nImage', fontsize=20,
                    verticalalignment='center', horizontalalignment='right', 
                    transform=axes[0, 0].transAxes)

    for j, concept_label in enumerate(concept_labels):
        axes[j + 1, 0].text(-0.1, 0.5, f'{concept_label}\n(thr = {thresholds[concept_label][0]:.2f})', fontsize=20,
                            verticalalignment='center', horizontalalignment='right', 
                            transform=axes[j + 1, 0].transAxes)

    # Loop through each concept and image to plot binarized heatmaps
    for i, image_index in enumerate(image_indices):
        for j, concept_label in enumerate(concept_labels):
            heatmap = heatmaps[concept_label][image_index].detach().cpu().numpy()
            if nonconcept_thresholds:
                mask = heatmap < thresholds[concept_label][0]
            else:
                mask = heatmap >= thresholds[concept_label][0]

            # Resize mask to match the image dimensions
            mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
            mask_resized = np.array(mask_resized) > 127
            
            # Convert image to normalized RGB numpy array
            image = retrieve_image(image_index, dataset_name)
            resized_image = pad_or_resize_img(image, model_input_size)
            image_np = np.array(resized_image.convert("RGB")) / 255.0
            
            # Overlay mask with transparency
            rgba_image = np.zeros((*image_np.shape[:2], 4))
            rgba_image[mask_resized] = np.concatenate([image_np[mask_resized], np.full((mask_resized.sum(), 1), 1)], axis=1)
            rgba_image[~mask_resized] = [0, 0, 0, 1]

            # Plot the binarized heatmap
            axes[j + 1, i].imshow(rgba_image)
            axes[j + 1, i].axis('off')
            
    if nonconcept_thresholds:
        plt.suptitle(f"Binarized negative concept {metric_type} at {percentile*100}% percentile for {dataset_name}", fontsize=16, y=1)
    else:
        plt.suptitle(f"Binarized {metric_type} at {percentile*100}% percentile for {dataset_name}", fontsize=16, y=1)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_file:
        plt.savefig(save_file, dpi=300)

    plt.show()
    
    
def plot_binarized_patchsims_single_image_multiple_thresholds(
    concept_labels, heatmaps, image_index, images,
    thresholds_dict, metric_type, model_input_size, 
    dataset_name, save_file=None, nonconcept_thresholds=False):
    """
    Plots binarized patch similarities for multiple concepts at multiple thresholds for a single image.

    Args:
        concept_labels (list of str): List of concept names.
        heatmaps (dict): Dictionary where keys are concept names, values are heatmaps per image.
        image_index (int): Index of the image to visualize.
        images (list of PIL.Image): List of images.
        thresholds_dict (dict): Maps from concept to list of thresholds to visualize.
        model_input_size (tuple): Image resize dimensions for the model.
        metric_type (str): Type of similarity metric used.
        dataset_name (str): Name of the dataset.
        save_file (str, optional): Path to save the plot.
        nonconcept_thresholds (bool): Whether the thresholds indicate non-concept behavior.
    """
    percentiles = list(thresholds_dict.keys())
    num_thresholds = len(thresholds_dict.keys())

    num_concepts = len(concept_labels)
    fig, axes = plt.subplots(num_concepts + 1, num_thresholds, figsize=(num_thresholds * 3, (num_concepts + 1) * 3))
    if num_thresholds == 1:
        axes = np.expand_dims(axes, axis=1)

    # Plot original image in the top row
    image = images[image_index]
    resized_image = pad_or_resize_img(image, model_input_size)
    for col in range(num_thresholds):
        axes[0, col].imshow(resized_image)
        axes[0, col].set_title(f"Image {image_index}")
        axes[0, col].axis('off')

    # Label for original image row
    axes[0, 0].text(-0.1, 0.5, 'Original\nImage', fontsize=20,
                    verticalalignment='center', horizontalalignment='right', 
                    transform=axes[0, 0].transAxes)

    for row, concept_label in enumerate(concept_labels):
        axes[row + 1, 0].text(-0.1, 0.5, f'{concept_label}', fontsize=20,
                              verticalalignment='center', horizontalalignment='right', 
                              transform=axes[row + 1, 0].transAxes)

        col = 0
        for percentile in percentiles:
            threshold = thresholds_dict[percentile][concept_label][0]
            heatmap = heatmaps[concept_label].detach().cpu().numpy()
            if nonconcept_thresholds:
                mask = heatmap < threshold
            else:
                mask = heatmap >= threshold

            # Resize mask to match the image dimensions
            mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
            mask_resized = np.array(mask_resized) > 127

            # Convert image to normalized RGB numpy array
            image_np = np.array(resized_image.convert("RGB")) / 255.0

            # Overlay mask with transparency
            rgba_image = np.zeros((*image_np.shape[:2], 4))
            rgba_image[mask_resized] = np.concatenate([image_np[mask_resized], np.full((mask_resized.sum(), 1), 1)], axis=1)
            rgba_image[~mask_resized] = [0, 0, 0, 1]

            axes[row + 1, col].imshow(rgba_image)
            axes[row + 1, col].axis('off')
            axes[row + 1, col].set_title(f"{percentile*100:.0f}%")
            col += 1

    if nonconcept_thresholds:
        plt.suptitle(f"Binarized negative concept {metric_type} for {dataset_name}", fontsize=16, y=1)
    else:
        plt.suptitle(f"Binarized {metric_type} for {dataset_name}", fontsize=16, y=1)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if save_file:
        plt.savefig(save_file, dpi=300)
    plt.show()
    
    
def plot_binarized_patchsims_pos_neg_single_image(
    image_index, images,
    positive_heatmap, negative_heatmap,
    threshold_pos, threshold_neg,
    model_input_size, dataset_name,
    metric_type="similarity", save_file=None
):
    """
    Plots a single image with binarized patch similarities for positive and negative thresholds.
    Green overlay indicates positive concept activation, red overlay indicates negative concept activation.

    Args:
        image_index (int): Index of the image to visualize.
        images (list of PIL.Image): List of images.
        positive_heatmap (torch.Tensor): Heatmap tensor for positive concept.
        negative_heatmap (torch.Tensor): Heatmap tensor for negative concept.
        threshold_pos (float): Threshold for positive activation.
        threshold_neg (float): Threshold for negative activation.
        model_input_size (tuple): Resize dimensions for model input.
        dataset_name (str): Dataset name for labeling.
        metric_type (str): Similarity or distance type.
        save_file (str): Optional path to save output image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Load and resize image
    image = images[image_index]
    resized_image = pad_or_resize_img(image, model_input_size)
    image_np = np.array(resized_image.convert("RGB")) / 255.0

    # Binarize masks
    pos_mask = positive_heatmap >= threshold_pos
    neg_mask = negative_heatmap < threshold_neg

    # Resize masks to match image shape
    pos_mask = Image.fromarray((pos_mask.cpu().numpy() * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
    neg_mask = Image.fromarray((neg_mask.cpu().numpy() * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
    pos_mask = np.array(pos_mask) > 127
    neg_mask = np.array(neg_mask) > 127

    # Compute overlap and exclusive regions
    both_mask = pos_mask & neg_mask
    pos_only = pos_mask & ~both_mask
    neg_only = neg_mask & ~both_mask

    # Create overlay
    rgba_overlay = np.zeros((*image_np.shape[:2], 4))
    rgba_overlay[pos_only] = [0, 1, 0, 0.6]      # Green
    rgba_overlay[neg_only] = [1, 0, 0, 0.6]      # Red
    rgba_overlay[both_mask] = [0.5, 0, 0.5, 0.6] # Purple

    # Composite overlay with image
    overlay_img = image_np.copy()
    out = overlay_img.copy()
    for c in range(3):
        out[..., c] = rgba_overlay[..., 3] * rgba_overlay[..., c] + (1 - rgba_overlay[..., 3]) * overlay_img[..., c]

    # Plot
    ax.imshow(out)
    ax.axis('off')
    ax.set_title(f"Image {image_index}\nGreen: ≥ {threshold_pos:.3f}, Red: < {threshold_neg:.3f}, Purple: Overlap")

    plt.suptitle(f"Binarized Patch Activations - {dataset_name} [{metric_type}]", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_file:
        plt.savefig(save_file, dpi=300)
    plt.show()
    
    
    
def plot_binarized_patchsims_pos_neg_grid(
    image_index, images,
    heatmaps_dict, positive_thresholds, negative_thresholds,
    model_input_size, dataset_name,
    metric_type="similarity", save_file=None
):
    """
    Plots a grid of binarized patch activations where columns are positive concepts and rows are negative concepts.

    Args:
        image_index (int): Index of the image to visualize.
        images (list of PIL.Image): List of PIL images.
        heatmaps_dict (dict): Nested dict with heatmaps_dict[concept][image_index] = heatmap (Tensor).
        thresholds_pos (dict): Dict mapping positive concept to threshold.
        thresholds_neg (dict): Dict mapping negative concept to threshold.
        pos_concepts (list of str): Positive concept names (columns).
        neg_concepts (list of str): Negative concept names (rows).
        model_input_size (tuple): Resize dimensions.
        dataset_name (str): Dataset name for title.
        metric_type (str): Similarity or distance.
        save_file (str): Optional path to save.
    """
    concepts = heatmaps_dict.keys()
    n_rows = len(concepts)
    n_cols = len(concepts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    image = images[image_index]
    resized_image = pad_or_resize_img(image, model_input_size)
    image_np = np.array(resized_image.convert("RGB")) / 255.0

    for row_idx, neg_concept in enumerate(concepts):
        for col_idx, pos_concept in enumerate(concepts):
            ax = axes[row_idx, col_idx]

            pos_map = heatmaps_dict[pos_concept]
            neg_map = heatmaps_dict[neg_concept]
            pos_thresh = positive_thresholds[pos_concept][0]
            neg_thresh = negative_thresholds[neg_concept][0]

            pos_mask = pos_map >= pos_thresh
            neg_mask = neg_map < neg_thresh

            pos_mask = Image.fromarray((pos_mask.cpu().numpy() * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
            neg_mask = Image.fromarray((neg_mask.cpu().numpy() * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
            pos_mask = np.array(pos_mask) > 127
            neg_mask = np.array(neg_mask) > 127

            both_mask = pos_mask & neg_mask
            pos_only = pos_mask & ~both_mask
            neg_only = neg_mask & ~both_mask

            rgba_overlay = np.zeros((*image_np.shape[:2], 4))
            rgba_overlay[pos_only] = [0, 1, 0, 0.6]      # Green
            rgba_overlay[neg_only] = [1, 0, 0, 0.6]      # Red
            rgba_overlay[both_mask] = [0.5, 0, 0.5, 0.6] # Purple

            out = image_np.copy()
            for c in range(3):
                out[..., c] = rgba_overlay[..., 3] * rgba_overlay[..., c] + (1 - rgba_overlay[..., 3]) * image_np[..., c]

            ax.imshow(out)
            ax.axis("off")

            if row_idx == 0:
                ax.set_title(f"{pos_concept}\nthr={pos_thresh:.2f}", fontsize=10)

        # Add label inside leftmost axis (first column of each row)
        axes[row_idx, 0].text(
            -0.05, 0.5,
            f"{neg_concept}\nthr={negative_thresholds[neg_concept][0]:.2f}",
            transform=axes[row_idx, 0].transAxes,
            va='center', ha='right',
            fontsize=10
        )

    plt.tight_layout(rect=[0.08, 0.02, 1, 0.92])  # leave space for row labels, less top padding
    plt.suptitle(
        f"Image {image_index} - Positive (Cols) vs Negative (Rows) [{metric_type}]",
        fontsize=14,
        x=0.6, y=0.93  # x centers, y controls vertical placement
    )

    if save_file:
        plt.savefig(save_file, dpi=300)
    plt.show()
    


def plot_binarized_patchsims_two_concepts(
    percentile,
    images,
    heatmaps1,
    heatmaps2,
    threshold1,
    threshold2,
    model_input_size,
    dataset_name,
    concept1_label="Concept 1",
    concept2_label="Concept 2",
    metric_type="similarity",
    save_file=None
):
    """
    Plots a horizontal strip of images with binarized patch activations from two concept heatmaps.

    Args:
        image_indices (list): List of image indices to visualize.
        images (list of PIL.Image): List of all dataset images.
        heatmaps1 (dict): Mapping from image index to heatmap for Concept 1.
        heatmaps2 (dict): Mapping from image index to heatmap for Concept 2.
        threshold1 (float): Activation threshold for Concept 1.
        threshold2 (float): Activation threshold for Concept 2.
        model_input_size (tuple): Resize dimensions for model input.
        dataset_name (str): Dataset name for title.
        concept1_label (str): Label for Concept 1.
        concept2_label (str): Label for Concept 2.
        metric_type (str): Similarity or distance type.
        save_file (str): Optional path to save output image.
    """
    image_indices = list(heatmaps1.keys())
    n = len(image_indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, img_idx in zip(axes, image_indices):
        image = images[img_idx]
        resized_image = pad_or_resize_img(image, model_input_size)
        image_np = np.array(resized_image.convert("RGB")) / 255.0

        # Get and binarize masks
        h1 = heatmaps1[img_idx]
        h2 = heatmaps2[img_idx]
        m1 = (h1 >= threshold1).cpu().numpy()
        m2 = (h2 >= threshold2).cpu().numpy()

        # Resize to image resolution
        m1 = Image.fromarray((m1 * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
        m2 = Image.fromarray((m2 * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
        m1 = np.array(m1) > 127
        m2 = np.array(m2) > 127

        # Color overlays
        both_mask = m1 & m2
        only1 = m1 & ~both_mask
        only2 = m2 & ~both_mask

        rgba = np.zeros((*image_np.shape[:2], 4))
        rgba[only1] = [0, 1, 0, 0.6]       # Green
        rgba[only2] = [0, 0, 1, 0.6]       # Blue
        rgba[both_mask] = [0.5, 0, 0.5, 0.6]  # Purple

        composite = image_np.copy()
        for c in range(3):
            composite[..., c] = rgba[..., 3] * rgba[..., c] + (1 - rgba[..., 3]) * composite[..., c]

        ax.imshow(composite)
        ax.set_title(f"Image {img_idx}", fontsize=12)
        ax.axis("off")

    # Legend below
    green_patch = mpatches.Patch(color='green', label=concept1_label)
    blue_patch = mpatches.Patch(color='blue', label=concept2_label)
    purple_patch = mpatches.Patch(color='purple', label='Overlap')
    fig.legend(handles=[green_patch, blue_patch, purple_patch], loc='lower center', ncol=3, fontsize=12)

    plt.suptitle(f"Binarized Patch Activations at {round(percentile*100)}% percentile", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    if save_file:
        plt.savefig(save_file, dpi=300)
    plt.show()
    
    
    
def plot_concept_evolution_over_iterations(
    image_index, heatmaps_over_time, all_images,
    dataset_name, model_input_size, concept_labels,
    fine_tuning_params, top_n=1, metric_type='Distance to Decision Boundary',
    vmin=None, vmax=None, save_file=None, epochs_to_plot=None):
    """
    Plot the evolution of heatmaps for a single image and multiple concepts over iterations.
    Rows = concepts, Columns = iterations (initial + fine-tuned steps)
    """
    total_epochs = sum(epoch_count for _, epoch_count in fine_tuning_params)
    num_concepts = len(concept_labels)

    # Build list of per-epoch patch percentages (e.g., ['init', 'init', 'init', 0.2, 0.2])
    epoch_patch_labels = []
    for patch_percent, num_epochs in fine_tuning_params:
        epoch_patch_labels.extend([patch_percent] * num_epochs)
        
    # If no specific epoch list provided, use all
    if epochs_to_plot is None:
        epochs_to_plot = list(range(1, total_epochs + 1))
    epochs_to_plot = [ep for ep in epochs_to_plot if 1 <= ep <= total_epochs]
    num_epochs_to_plot = len(epochs_to_plot)
    
    # Compute vmin/vmax if needed
    if vmin is None or vmax is None:
        vals = np.concatenate([
            heatmap.flatten() for heatmap_list in heatmaps_over_time.values() for heatmap in heatmap_list
        ])
        vmin = np.min(vals)
        vmax = np.max(vals)

    # Create figure
    fig, axes = plt.subplots(num_concepts + 1, num_epochs_to_plot, figsize=(num_epochs_to_plot * 2.0, (num_concepts + 1) * 2))

    # Get and prepare image
    image = pad_or_resize_img(all_images[image_index], model_input_size)
    image_gray = image.convert("L")

    # Top row: original image + titles for selected epochs
    for j, epoch_num in enumerate(epochs_to_plot):
        ax = axes[0, j]
        ax.imshow(image)
        ax.axis('off')

        patch_label = epoch_patch_labels[epoch_num - 1]
        label = f"{patch_label}%" if patch_label != 'init' else "All Patches"
        ax.set_title(f"{label}\n(Epoch {epoch_num})", fontsize=9)

    axes[0, 0].text(-0.05, 0.5, f"Img {image_index}", fontsize=12,
                    va='center', ha='right', transform=axes[0, 0].transAxes)

    # Plot heatmaps for each concept
    for i, concept_label in enumerate(concept_labels):
        for j, epoch_num in enumerate(epochs_to_plot):
            ax = axes[i + 1, j]
            heatmap = heatmaps_over_time[concept_label][j]  # 0-based indexing

            ax.imshow(heatmap, cmap='hot', alpha=0.6, vmin=vmin, vmax=vmax,
                      extent=[0, model_input_size[0], model_input_size[1], 0])
            ax.imshow(image_gray, cmap='gray', alpha=0.4)
            ax.set_title(f"Max={round(heatmap.max().item(), 4)}\nMin={round(heatmap.min().item(), 4)}", fontsize=8)
            ax.axis('off')

        # Concept label
        axes[i + 1, 0].text(-0.05, 0.5, concept_label, fontsize=12,
                            va='center', ha='right', transform=axes[i + 1, 0].transAxes)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap='hot')
    sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax).set_label(metric_type)

    plt.suptitle("Concept Activation Evolution Over Epochs", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_file:
        plt.savefig(save_file, dpi=500, bbox_inches='tight')

    plt.show()
    
    
def top_left_crop_to_original_aspect(arr, original_size, padded_size):
    """
    Crops the top-left region of a padded image/array to match the original image's aspect ratio.

    Args:
        arr (np.ndarray): Input image or heatmap array (H, W, ...) to crop.
        original_size (tuple): (W_orig, H_orig) of original image.
        padded_size (tuple): (W_pad, H_pad) of the padded image.

    Returns:
        np.ndarray: Cropped array from top-left corner to original aspect ratio.
    """
    W_orig, H_orig = original_size
    W_pad, H_pad = padded_size

    target_height = int(W_pad * H_orig / W_orig)
    cropped = arr[:target_height, :W_pad]

    return cropped


def plot_binarized_patchsims_with_raw_heatmaps(
    concept_labels, heatmaps, image_index, images,
    thresholds_dict, metric_type, model_input_size, 
    dataset_name, save_file=None, 
    nonconcept_thresholds=False, vmin=None, vmax=None,
    show_colorbar_ticks=True, seg_concept=None):

    percentiles = list(thresholds_dict.keys())
    num_thresholds = len(percentiles)
    num_concepts = len(concept_labels)

    # Clean matplotlib styling
    plt.rcParams.update({'font.size': 10})
    
    # Create grid layout: rows = concepts, cols = original + heatmap + percentiles
    num_concepts = len(concept_labels)
    num_cols = 2 + len(percentiles)  # Original, heatmap, then percentiles
    
    # Create figure with grid
    fig_width = 2.5 * num_cols  # Width per column
    fig_height = 2.5 * num_concepts  # Height per row
    fig, axes = plt.subplots(num_concepts, num_cols, figsize=(fig_width, fig_height),
                            gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    
    # Handle single row/column cases
    if num_concepts == 1:
        axes = axes.reshape(1, -1)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Set white background
    fig.patch.set_facecolor('white')

    # Load segmentations if needed (using efficient loader)
    concept_mask = None
    if seg_concept is not None and 'Broden' in dataset_name:
        from utils.segmentation_loader import get_concept_segmentation
        concept_mask = get_concept_segmentation(dataset_name, image_index, seg_concept)
    
    # Load image internally if not provided
    if images is None:
        image = retrieve_image(image_index, dataset_name)
    else:
        image = images[image_index]
    resized_image = pad_or_resize_img(image, model_input_size)
    image_np = np.array(resized_image.convert("RGB")) / 255.0
    
    # Create separate images: one with segmentation for original, one clean for heatmaps
    image_np_clean = image_np.copy()  # Clean version for heatmaps and overlays
    image_np_with_seg = image_np.copy()  # Version with segmentation for original image only
    
    # Apply segmentation overlay only to the segmentation version
    if concept_mask is not None:
        try:
            from utils.general_utils import pad_or_resize_img_tensor
            from scipy import ndimage
            
            # Resize mask to match image size
            if isinstance(concept_mask, torch.Tensor):
                resized_mask = pad_or_resize_img_tensor(concept_mask, model_input_size, is_mask=True)
                overlay_mask = resized_mask.cpu().numpy()
            else:
                overlay_mask = concept_mask
            
            # Apply thick yellow outline only to the segmentation version
            if overlay_mask is not None:
                # Create thicker outline by dilating multiple times
                dilated_mask = overlay_mask.copy()
                for _ in range(12):  # Apply dilation 12 times for much thicker outline (2x previous)
                    dilated_mask = ndimage.binary_dilation(dilated_mask)
                edges = dilated_mask & ~overlay_mask
                
                # Also add edge detection to include boundaries at image edges
                # Create a boundary mask for edges of the image
                h, w = overlay_mask.shape
                edge_mask = np.zeros_like(overlay_mask, dtype=bool)
                edge_width = 12  # Same thickness as dilation
                edge_mask[:edge_width, :] = True
                edge_mask[-edge_width:, :] = True
                edge_mask[:, :edge_width] = True
                edge_mask[:, -edge_width:] = True
                
                # Include edges that touch the image boundary
                edges_at_boundary = edge_mask & dilated_mask
                edges = edges | edges_at_boundary
                
                # Apply yellow outline only to the segmentation version
                image_np_with_seg[edges == 1] = [1.0, 1.0, 0.0]  # Yellow color (normalized)
        except Exception as e:
            print(f"Warning: Could not apply segmentation overlay: {e}")
    elif seg_concept is not None and 'Broden' in dataset_name:
        print(f"Warning: Segmentation concept '{seg_concept}' not found in image {image_index}")
    
    # Only crop for LLAMA (560x560), not for CLIP (224x224)
    if model_input_size == (224, 224):
        image_cropped_clean = image_np_clean
        image_cropped_with_seg = image_np_with_seg
    else:
        image_cropped_clean = top_left_crop_to_original_aspect(image_np_clean, image.size, resized_image.size)
        image_cropped_with_seg = top_left_crop_to_original_aspect(image_np_with_seg, image.size, resized_image.size)

    if vmin is None or vmax is None:
        all_values = np.concatenate([
            heatmaps[concept].detach().cpu().numpy().flatten()
            for concept in concept_labels
        ])
        vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

    # First adjust the layout to get final positions
    plt.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.02, wspace=0.05, hspace=0.05)
    
    # Now get the actual subplot positions after adjustment
    test_ax = axes[0, 0]
    bbox = test_ax.get_position()
    col_width = bbox.width
    col_spacing = axes[0, 1].get_position().x0 - (bbox.x0 + bbox.width)  # Space between columns
    
    # Add column headers aligned with subplot centers
    for col_idx in range(num_cols):
        # Calculate center position of each column accounting for spacing
        ax_pos = axes[0, col_idx].get_position()
        col_center_x = ax_pos.x0 + ax_pos.width / 2
        
        if col_idx == 0:
            # For original image column, create label with highlighted GT concept
            if seg_concept:
                # Format the GT concept name
                display_seg = seg_concept
                if dataset_name == 'Broden-OpenSurfaces' and 'material' in seg_concept:
                    if '::' in seg_concept:
                        display_seg = seg_concept.split('::')[-1].capitalize()
                
                # Create text with partial highlighting using dynamic positioning
                # Get approximate text widths (in figure coordinates)
                # Using rough character width estimate
                char_width = 0.006  # Approximate width per character at fontsize 11
                
                # Calculate positions based on text content
                part1 = 'Image (GT '
                part2 = display_seg  # Only highlight the concept name
                part3 = ')'
                
                # Calculate total width to center the whole label
                total_width = len(part1) * char_width + len(part2) * char_width + len(part3) * char_width
                start_x = col_center_x - total_width / 2 + 0.01  # Shift slightly right
                
                # First part: "Image (GT "
                text1 = fig.text(start_x + len(part1) * char_width / 2, 0.96, part1, 
                               ha='center', va='center', fontweight='bold', fontsize=11)
                
                # Second part: highlighted concept name only
                text2_x = start_x + len(part1) * char_width + len(part2) * char_width / 2
                text2 = fig.text(text2_x, 0.96, part2, 
                               ha='center', va='center', fontsize=11,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', 
                                       edgecolor='none', alpha=0.8))
                
                # Third part: closing parenthesis
                text3_x = start_x + len(part1) * char_width + len(part2) * char_width + len(part3) * char_width / 2
                text3 = fig.text(text3_x, 0.96, part3, 
                               ha='center', va='center', fontweight='bold', fontsize=11)
            else:
                fig.text(col_center_x, 0.96, 'Image', 
                        ha='center', va='center', fontweight='bold', fontsize=12)
        elif col_idx == 1:
            fig.text(col_center_x, 0.96, 'Heatmap', 
                    ha='center', va='center', fontweight='bold', fontsize=12)
        else:
            fig.text(col_center_x, 0.96, f'{percentiles[col_idx-2]*100:.0f}%', 
                    ha='center', va='center', fontweight='bold', fontsize=12)
    
    # Process each concept (row)
    for concept_idx, concept_label in enumerate(concept_labels):
        # Get concept display name
        display_label = concept_label
        if dataset_name == 'Broden-OpenSurfaces' and 'material' in concept_label:
            if '::' in concept_label:
                display_label = concept_label.split('::')[-1].capitalize()
        
        # Get the actual vertical position of this row's subplot
        row_ax = axes[concept_idx, 0]
        row_bbox = row_ax.get_position()
        row_center_y = row_bbox.y0 + row_bbox.height / 2
        
        # Add row label (concept name) to the right of the grid in italics
        # Position it between the last column and the colorbar
        fig.text(0.89, row_center_y, display_label, 
                ha='left', va='center', fontstyle='italic', fontsize=11)
        
        # Column 0: Original image with segmentation (only show once in top-left)
        ax = axes[concept_idx, 0]
        ax.axis('off')
        
        if concept_idx == 0:  # Only show original image in the first row
            ax.imshow(image_cropped_with_seg)
            ax.set_title('', fontsize=10)
        
        # Column 1: Raw heatmap
        ax = axes[concept_idx, 1]
        ax.axis('off')
        
        heatmap = heatmaps[concept_label].detach().cpu().numpy()
        heatmap_resized = Image.fromarray(heatmap).resize(resized_image.size, resample=Image.NEAREST)
        heatmap_resized = np.array(heatmap_resized)
        
        # Only crop for LLAMA (560x560), not for CLIP (224x224)
        if model_input_size == (224, 224):
            heatmap_cropped = heatmap_resized
        else:
            heatmap_cropped = top_left_crop_to_original_aspect(heatmap_resized, image.size, resized_image.size)
        
        im = ax.imshow(heatmap_cropped, cmap='hot', vmin=vmin, vmax=vmax)
        
        # Columns 2+: Binarized versions at different thresholds
        for thresh_idx, percentile in enumerate(percentiles):
            ax = axes[concept_idx, 2 + thresh_idx]
            ax.axis('off')
            
            # Get threshold and create mask
            threshold = thresholds_dict[percentile][concept_label][0]
            mask = heatmap >= threshold if not nonconcept_thresholds else heatmap < threshold
            mask[np.isnan(heatmap)] = False
            
            # Resize mask to match image size
            mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
            mask_resized = np.array(mask_resized) > 127
            
            # Create RGBA image showing only masked regions with black background
            rgba_image = np.zeros((*image_np_clean.shape[:2], 4))
            rgba_image[mask_resized] = np.concatenate([image_np_clean[mask_resized], np.ones((mask_resized.sum(), 1))], axis=1)
            rgba_image[~mask_resized] = [0, 0, 0, 1]  # Black background for non-masked areas
            
            # Only crop for LLAMA (560x560), not for CLIP (224x224)
            if model_input_size == (224, 224):
                rgba_image_cropped = rgba_image
            else:
                rgba_image_cropped = top_left_crop_to_original_aspect(rgba_image, image.size, resized_image.size)
            
            ax.imshow(rgba_image_cropped)
    
    # Add colorbar on the right side of the figure
    if show_colorbar_ticks:
        # Create a new axis for the colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_label('Activation Strength', fontsize=10)
        cb.ax.tick_params(labelsize=9)
    
    # Layout was already adjusted at the beginning, no need to do it again
    
    if save_file:
        plt.savefig(save_file, dpi=500, format='pdf', bbox_inches='tight')
    
    plt.show()


def plot_image_with_concept_heatmaps(
    concept_labels, heatmaps, image_index, images,
    model_input_size, dataset_name, save_file=None,
    vmin=None, vmax=None, show_colorbar_ticks=True, 
    seg_concept=None, heatmap_alpha=1.0, font_size=14, figure_width=None, cmap='hot',
    highlight_gt_concept=True, save_colorbar=False):
    """
    Plot original image with segmentation and concept heatmaps side by side.
    
    Args:
        concept_labels: List of concept names
        heatmaps: Dict mapping concept names to heatmap tensors
        image_index: Index of the image to visualize
        images: List of images or None (will load if None)
        model_input_size: Tuple of (width, height) for model input
        dataset_name: Name of the dataset
        save_file: Path to save the figure (optional)
        vmin/vmax: Min/max values for heatmap colorscale
        show_colorbar_ticks: Whether to show colorbar
        seg_concept: Ground truth concept for segmentation overlay
        heatmap_alpha: Alpha (transparency) for heatmap overlay (0-1)
        font_size: Base font size for text elements
        figure_width: Total figure width in inches (height auto-calculated)
        cmap: Colormap name for heatmaps (default: 'hot')
        highlight_gt_concept: Whether to highlight the GT concept with yellow background
        save_colorbar: Whether to save colorbar as separate file with _colorbar suffix
    """
    from utils.general_utils import get_paper_plotting_style
    
    num_concepts = len(concept_labels)
    num_cols = 1 + num_concepts  # Original image + one heatmap per concept
    
    # Apply paper plotting style
    plt.rcParams.update(get_paper_plotting_style())
    # Override font size if specified
    if font_size != 14:  # If not default
        plt.rcParams.update({'font.size': font_size})
    
    # Create figure with single row
    if figure_width is None:
        fig_width = 3.0 * num_cols
    else:
        fig_width = figure_width
    
    # Calculate height based on width and aspect ratio
    # Each subplot should be roughly square, plus extra space for labels
    subplot_width = fig_width / num_cols
    subplot_height = subplot_width  # Square subplots
    fig_height = subplot_height * 1.3  # Add 30% for labels and padding
    
    fig, axes = plt.subplots(1, num_cols, figsize=(fig_width, fig_height),
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    
    # Handle single column case
    if num_cols == 1:
        axes = [axes]
    
    # Set white background
    fig.patch.set_facecolor('white')
    
    # Load segmentations if needed
    concept_mask = None
    if seg_concept is not None and 'Broden' in dataset_name:
        from utils.segmentation_loader import get_concept_segmentation
        concept_mask = get_concept_segmentation(dataset_name, image_index, seg_concept)
    
    # Load image
    if images is None:
        image = retrieve_image(image_index, dataset_name)
    else:
        image = images[image_index]
    resized_image = pad_or_resize_img(image, model_input_size)
    image_np = np.array(resized_image.convert("RGB")) / 255.0
    
    # Create clean version for heatmap overlays
    image_np_clean = image_np.copy()
    
    # Apply segmentation overlay only to the version for the first column
    if concept_mask is not None:
        try:
            from utils.general_utils import pad_or_resize_img_tensor
            from scipy import ndimage
            
            # Resize mask to match image size
            if isinstance(concept_mask, torch.Tensor):
                resized_mask = pad_or_resize_img_tensor(concept_mask, model_input_size, is_mask=True)
                overlay_mask = resized_mask.cpu().numpy()
            else:
                overlay_mask = concept_mask
            
            # Apply thick yellow outline
            if overlay_mask is not None:
                dilated_mask = overlay_mask.copy()
                for _ in range(12):
                    dilated_mask = ndimage.binary_dilation(dilated_mask)
                edges = dilated_mask & ~overlay_mask
                
                # Handle edges at image boundary
                h, w = overlay_mask.shape
                edge_mask = np.zeros_like(overlay_mask, dtype=bool)
                edge_width = 12
                edge_mask[:edge_width, :] = True
                edge_mask[-edge_width:, :] = True
                edge_mask[:, :edge_width] = True
                edge_mask[:, -edge_width:] = True
                
                edges_at_boundary = edge_mask & dilated_mask
                edges = edges | edges_at_boundary
                
                image_np[edges == 1] = [1.0, 1.0, 0.0]  # Yellow
        except Exception as e:
            print(f"Warning: Could not apply segmentation overlay: {e}")
    
    # Crop if needed (only for LLAMA)
    if model_input_size == (224, 224):
        image_cropped = image_np
        image_cropped_clean = image_np_clean
    else:
        image_cropped = top_left_crop_to_original_aspect(image_np, image.size, resized_image.size)
        image_cropped_clean = top_left_crop_to_original_aspect(image_np_clean, image.size, resized_image.size)
    
    # Compute global vmin/vmax if not provided
    if vmin is None or vmax is None:
        all_values = np.concatenate([
            heatmaps[concept].detach().cpu().numpy().flatten()
            for concept in concept_labels
        ])
        vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
    
    # Column 0: Original image with segmentation
    ax = axes[0]
    ax.imshow(image_cropped)
    ax.axis('off')
    
    # Add title for original image above it (like the other columns)
    if seg_concept:
        display_seg = seg_concept
        if dataset_name == 'Broden-OpenSurfaces' and 'material' in seg_concept:
            if '::' in seg_concept:
                display_seg = seg_concept.split('::')[-1].capitalize()
        
        if highlight_gt_concept:
            # Top label: "Image" - will be positioned later with fig.text
            # ax.set_title('Image', fontweight='bold', fontsize=font_size)
            
            # Bottom label: "(GT concept)"
            y_pos_bottom = -0.05
            
            # Add prefix
            prefix = '(GT '
            t2 = ax.text(0.42, y_pos_bottom, prefix, transform=ax.transAxes,
                        ha='right', va='top', fontsize=font_size)  # Same size as other labels
            
            # Add concept with yellow background
            t3 = ax.text(0.64, y_pos_bottom, f'{display_seg}', transform=ax.transAxes,
                        ha='center', va='top', 
                        fontstyle='italic', fontsize=font_size,  # Same size as other labels
                        bbox=dict(boxstyle='round,pad=0.05', facecolor='yellow', 
                                 edgecolor='none', alpha=0.8))
            
            # Add suffix
            t4 = ax.text(0.85, y_pos_bottom, ')', transform=ax.transAxes,
                        ha='left', va='top', fontsize=font_size)  # Same size as other labels
        else:
            # Top label: "Image" - will be positioned later with fig.text
            # ax.set_title('Image', fontweight='bold', fontsize=font_size)
            
            # Bottom label: "(GT concept)" without highlighting
            ax.text(0.5, -0.05, f'(GT $\it{{{display_seg}}}$)', transform=ax.transAxes,
                    ha='center', va='top', fontsize=font_size)
    else:
        # Simple centered title - will be positioned later with fig.text
        # ax.set_title('Image', fontweight='bold', fontsize=font_size+1)
        pass
    
    # Columns 1+: Heatmaps for each concept
    for idx, concept_label in enumerate(concept_labels):
        ax = axes[idx + 1]
        ax.axis('off')
        
        # Get and process heatmap
        heatmap = heatmaps[concept_label].detach().cpu().numpy()
        heatmap_resized = Image.fromarray(heatmap).resize(resized_image.size, resample=Image.NEAREST)
        heatmap_resized = np.array(heatmap_resized)
        
        # Crop if needed
        if model_input_size == (224, 224):
            heatmap_cropped = heatmap_resized
        else:
            heatmap_cropped = top_left_crop_to_original_aspect(heatmap_resized, image.size, resized_image.size)
        
        # Display image with heatmap overlay (use clean version without segmentation)
        ax.imshow(image_cropped_clean)  # Show clean image underneath
        im = ax.imshow(heatmap_cropped, cmap=cmap, vmin=vmin, vmax=vmax, alpha=heatmap_alpha)
        
        # Add concept label at bottom
        display_label = concept_label
        if dataset_name == 'Broden-OpenSurfaces' and 'material' in concept_label:
            if '::' in concept_label:
                display_label = concept_label.split('::')[-1].capitalize()
        
        # Position label below the subplot
        ax.text(0.5, -0.05, display_label, transform=ax.transAxes,
                ha='center', va='top', fontsize=font_size, fontstyle='italic')
    
    # Adjust layout - minimal space on left, more space at bottom for labels
    plt.subplots_adjust(left=0.02, right=0.92 if show_colorbar_ticks else 0.98, 
                        top=0.80, bottom=0.15, wspace=0.05, hspace=0.05)
    
    # Add both headers at the same height using fig.text
    # First add "Image" header
    image_ax_pos = axes[0].get_position()
    image_header_x = (image_ax_pos.x0 + image_ax_pos.x1) / 2
    header_y = image_ax_pos.y1 + 0.05  # Position above the axes
    
    fig.text(image_header_x, header_y, 'Image', 
            ha='center', va='bottom', fontweight='bold', fontsize=font_size)
    
    # Add "Concept Activations" header above the heatmap columns at same height
    if num_concepts > 0:
        # Get position of first heatmap column
        first_heatmap_ax = axes[1]
        last_heatmap_ax = axes[-1]
        
        # Calculate center position for the header
        first_pos = first_heatmap_ax.get_position()
        last_pos = last_heatmap_ax.get_position()
        header_x = (first_pos.x0 + last_pos.x1) / 2
        
        # Use the same header_y as "Image"
        fig.text(header_x, header_y, 'Concept Activations', 
                ha='center', va='bottom', fontweight='bold', fontsize=font_size)
    
    # Add colorbar
    if show_colorbar_ticks:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.65])
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_label('Activation Strength', fontsize=font_size)
        cb.ax.tick_params(labelsize=font_size-2)
    
    if save_file:
        plt.savefig(save_file, dpi=500, format='pdf', bbox_inches='tight')
        
        # Save colorbar separately if requested
        if save_colorbar:
            # Generate colorbar filename
            import os
            base_name, ext = os.path.splitext(save_file)
            colorbar_file = f"{base_name}_colorbar{ext}"
            
            # Create a new figure just for the colorbar
            fig_cbar = plt.figure(figsize=(1.5, 4))
            ax_cbar = fig_cbar.add_axes([0.2, 0.1, 0.2, 0.8])
            
            # Create a dummy mappable with the same colormap and normalization
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            
            # Add colorbar
            cbar = fig_cbar.colorbar(sm, cax=ax_cbar)
            
            # Only add ticks and labels if show_colorbar_ticks is True
            if show_colorbar_ticks:
                cbar.set_label('Activation Strength', fontsize=font_size)
                cbar.ax.tick_params(labelsize=font_size-2)
            else:
                # Remove ticks and labels for clean colorbar
                cbar.set_ticks([])
                cbar.ax.set_yticklabels([])
            
            # Save the colorbar figure
            fig_cbar.savefig(colorbar_file, dpi=500, format='pdf', bbox_inches='tight')
            plt.close(fig_cbar)
            print(f"Colorbar saved to: {colorbar_file}")
    
    plt.show()


    
    
# def plot_superpatches_on_heatmaps(
#     concept_labels, heatmaps, image_index, images,
#     thresholds, metric_type, model_input_size,
#     dataset_name, save_file=None,
#     vmin=None, vmax=None):


#     num_concepts = len(concept_labels)
#     concepts_per_row = 3
#     num_rows = int(np.ceil(num_concepts / concepts_per_row))

#     fig, axes = plt.subplots(num_rows, concepts_per_row + 1, figsize=((concepts_per_row + 1) * 4, num_rows * 4))
#     axes = np.array(axes)

#     image = images[image_index]
#     resized_image = pad_or_resize_img(image, model_input_size)
#     image_np = np.array(resized_image.convert("RGB")) / 255.0
#     image_cropped = top_left_crop_to_original_aspect(image_np, image.size, resized_image.size)

#     if vmin is None or vmax is None:
#         all_values = np.concatenate([
#             heatmaps[concept].detach().cpu().numpy().flatten()
#             for concept in concept_labels
#         ])
#         vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

#     colorbar_im = None

#     # Turn off all axes initially
#     for i in range(num_rows):
#         for j in range(concepts_per_row + 1):
#             axes[i, j].axis('off')

#     # Manually position original image between heatmap rows
#     orig_ax = fig.add_axes([0.01, 0.65 - (1 / (num_rows * 2)), 0.25, 0.3])
#     orig_ax.imshow(image_cropped)
#     orig_ax.axis('off')
#     orig_ax.set_title("Original")

#     for idx, concept_label in enumerate(concept_labels):
#         row = idx // concepts_per_row
#         col = (idx % concepts_per_row) + 1

#         heatmap_orig = heatmaps[concept_label].detach().cpu().numpy()
#         heatmap_resized = Image.fromarray(heatmap_orig).resize(resized_image.size, resample=Image.NEAREST)
#         heatmap_resized = np.array(heatmap_resized)
#         # Only crop for LLAMA (560x560), not for CLIP (224x224)
#         if model_input_size == (224, 224):
#             heatmap_cropped = heatmap_resized
#         else:
#             heatmap_cropped = top_left_crop_to_original_aspect(heatmap_resized, image.size, resized_image.size)

#         axes[row, col].imshow(image_cropped, alpha=1.0)
#         im = axes[row, col].imshow(heatmap_cropped, cmap='magma', interpolation='nearest', vmin=vmin, vmax=vmax, alpha=0.8)
#         axes[row, col].axis('off')
#         axes[row, col].set_title(concept_label)

#         if colorbar_im is None:
#             colorbar_im = im

#         grid_size = heatmap_orig.shape[0]
#         patch_h = resized_image.size[1] / grid_size
#         patch_w = resized_image.size[0] / grid_size
#         threshold = thresholds[concept_label][0]

#         for i_patch in range(grid_size):
#             for j_patch in range(grid_size):
#                 if heatmap_orig[i_patch, j_patch] >= threshold:
#                     x = j_patch * patch_w
#                     y = i_patch * patch_h
#                     rect = mpatches.Rectangle((x, y), patch_w, patch_h,
#                                               linewidth=2, edgecolor='deepskyblue', facecolor='none')
#                     axes[row, col].add_patch(rect)

#     # Horizontal colorbar centered under the plots
#     fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.2, wspace=0.1, hspace=0.2)
#     cbar_width = 0.6
#     cbar_center = 0.5 - cbar_width / 2
#     cbar_ax = fig.add_axes([cbar_center, 0.1, cbar_width, 0.02])
#     fig.colorbar(colorbar_im, cax=cbar_ax, orientation='horizontal', label='Cosine Similarity to Concept')

#     if save_file:
#         plt.savefig(save_file, dpi=300)
#     plt.show()
def filter_and_plot_concept_images(
    metadata_path,
    required_concepts,
    dataset_name='Coco',
    chosen_split='train',
    start_idx=0,
    n_images=10,
    plot=True,
    exclude_concepts=None,
    show_seg=False
):
    """
    Filters metadata for rows with specified binary concepts and plots first n image thumbnails.

    Args:
        metadata_path (str): Path to metadata CSV.
        required_concepts (list): Concept column names required to be 1 (e.g., ['has_color_red']).
        dataset_name (str): Name of the dataset (e.g., 'Coco', 'Broden-OpenSurfaces', 'Broden-Pascal').
        chosen_split (str): Split to filter on ('train', 'test', etc.).
        n_images (int): Number of matching images to display.
        plot (bool): Whether to plot the images.
        exclude_concepts (list): Concept column names required to be 0 (excluded). Default is None.
        show_seg (bool): Whether to show segmentation overlays with yellow outline (for Broden datasets only).
    """
    import numpy as np
    from utils.general_utils import pad_or_resize_img_tensor
    
    metadata = pd.read_csv(metadata_path)

    # Validate concept columns
    for concept in required_concepts:
        if concept not in metadata.columns:
            raise ValueError(f"Missing concept column: {concept}")
    
    # Load segmentations if needed
    all_segs = None
    if show_seg and 'Broden' in dataset_name:
        all_segs = torch.load(f'../Data/{dataset_name}/segmentations.pt')
    
    if exclude_concepts:
        for concept in exclude_concepts:
            if concept not in metadata.columns:
                raise ValueError(f"Missing concept column: {concept}")

    # Apply filtering
    mask = metadata['split'] == chosen_split
    for concept in required_concepts:
        mask &= metadata[concept] == 1
    
    # Apply exclusion filtering
    if exclude_concepts:
        for concept in exclude_concepts:
            mask &= metadata[concept] == 0

    filtered_df = metadata[mask][start_idx:start_idx+n_images]

    if plot:
        n_imgs = len(filtered_df)
        n_cols = min(n_imgs, 5)  # Max 5 images per row
        n_rows = (n_imgs + 4) // 5  # Calculate number of rows needed
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        
        # Ensure axes is always 2D array for consistent indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Flatten axes for easy iteration
        axes_flat = axes.flatten()

        for idx, (metadata_idx, row) in enumerate(filtered_df.iterrows()):
            ax = axes_flat[idx]
            try:
                img = Image.open(f'../Data/{dataset_name}/{row["image_path"]}')
                img_array = np.array(img)
                
                # Generate title - handle material concepts for Broden-OpenSurfaces
                title_parts = [f'Index: {metadata_idx}']
                if dataset_name == 'Broden-OpenSurfaces':
                    for concept in required_concepts:
                        if row[concept] == 1 and 'material' in concept:
                            # Extract part after :: and capitalize
                            if '::' in concept:
                                material_name = concept.split('::')[-1].capitalize()
                                title_parts.append(material_name)
                            else:
                                title_parts.append(concept)
                        elif row[concept] == 1:
                            title_parts.append(concept)
                else:
                    # For other datasets, just add the concept names
                    for concept in required_concepts:
                        if row[concept] == 1:
                            title_parts.append(concept)
                
                # Add segmentation overlay if requested
                if show_seg and 'Broden' in dataset_name and all_segs is not None:
                    concept_masks = all_segs[metadata_idx]
                    
                    # Find the first required concept that exists in the masks
                    overlay_mask = None
                    for concept in required_concepts:
                        if concept in concept_masks:
                            concept_mask = concept_masks[concept]
                            # Resize mask to match image size
                            if isinstance(concept_mask, torch.Tensor):
                                resized_mask = pad_or_resize_img_tensor(concept_mask, img_array.shape[:2], is_mask=True)
                                overlay_mask = resized_mask.cpu().numpy()
                            else:
                                overlay_mask = concept_mask
                            break
                    
                    # Apply yellow outline if mask exists
                    if overlay_mask is not None:
                        from scipy import ndimage
                        # Find edges of the mask
                        edges = ndimage.binary_dilation(overlay_mask) & ~overlay_mask
                        # Apply yellow outline
                        img_array = img_array.copy()
                        img_array[edges == 1] = [255, 255, 0]  # Yellow color
                
                ax.imshow(img_array)
                ax.axis('off')
                ax.set_title(' | '.join(title_parts))
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Image load failed: {str(e)}", ha='center')
                ax.axis('off')
        
        # Hide any unused subplots
        for idx in range(len(filtered_df), len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.tight_layout()
        plt.show()



def plot_aligned_images_chunked(concept_key, dataset_name, acts_loader, 
                               con_label='', k=5, metric_type='Cosine Similarity', 
                               save_image=False, test_only=True):
    """
    Chunked version: Plot images that align well with a selected concept using pre-initialized loader.
    
    Args:
        concept_key (str): The concept to visualize (e.g., 'color::red').
        dataset_name (str): Name of the dataset.
        acts_loader: Pre-initialized ChunkedActivationLoader instance.
        con_label (str): Label to put in path of saved image.
        k (int): Number of top images to display.
        metric_type (str): Type of metric being visualized.
        save_image (bool): Whether to save the plot.
        test_only (bool): Whether to only consider test samples.
    """
    
    # Load the full dataframe (this handles chunking internally)
    comp_df = acts_loader.load_full_dataframe()
    
    # Filter for test samples if requested
    if test_only:
        metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
        test_indices = metadata[metadata['split'] == 'test'].index
        comp_df = comp_df.loc[comp_df.index.intersection(test_indices)]
    
    # Check if concept exists
    if concept_key not in comp_df.columns:
        print(f"Concept '{concept_key}' not found. Available concepts:")
        print(sorted(comp_df.columns)[:10], "...")
        return
    
    # Get top k samples
    top_k_indices = comp_df.nlargest(k, concept_key).index.tolist()
    
    # Calculate grid layout
    n_cols = min(k, 5)
    n_rows = (k + 4) // 5
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    plt.suptitle(f"Top {k} Images with Highest {metric_type} to: {concept_key}", fontsize=16)
    
    for rank, idx in enumerate(top_k_indices):
        if rank >= len(axes):
            break
        
        # Retrieve image
        img = retrieve_image(idx, dataset_name, test_only=False)
        value = comp_df.loc[idx, concept_key]
        
        axes[rank].imshow(img)
        axes[rank].set_title(f"Rank {rank+1}: Image {idx}\n{metric_type} = {value:.4f}")
        axes[rank].axis('off')
    
    # Hide unused axes
    for rank in range(len(top_k_indices), len(axes)):
        axes[rank].axis('off')
    
    plt.tight_layout()
    
    if save_image:
        save_path = f'../Figs/{dataset_name}/most_aligned_w_concepts/concept_{concept_key}_{k}__{con_label}.jpg'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
    
    plt.show()


def plot_most_similar_patches_w_heatmaps_chunked(concept_label, dataset_name, acts_loader, 
                                                embeds_loader, con_label='', 
                                                model_input_size=(224, 224), top_n=5, 
                                                metric_type='Cosine Similarity', test_only=True,
                                                save_path=None):
    """
    Chunked version: Plots the most similar patches with heatmaps using pre-initialized loaders.
    
    Args:
        concept_label (str): The concept to visualize.
        dataset_name (str): Name of the dataset.
        acts_loader: Pre-initialized ChunkedActivationLoader instance.
        embeds_loader: Pre-initialized ChunkedEmbeddingLoader instance.
        con_label (str): Label for saving.
        model_input_size (tuple): Model input size.
        top_n (int): Number of top patches to show.
        metric_type (str): Type of metric.
        test_only (bool): Whether to only use test samples.
        save_path (str): Where to save the figure.
    """
    # Load all images
    all_images, _, _ = load_images(dataset_name)
    
    # Load activations
    cos_sims = acts_loader.load_full_dataframe()
    
    # Filter for test samples if needed
    if test_only:
        split_df = get_patch_split_df(dataset_name, patch_size=14, model_input_size=model_input_size)
        test_indices = split_df[split_df == 'test'].index
        cos_sims = cos_sims.loc[cos_sims.index.intersection(test_indices)]
    
    # Get top patches
    if concept_label not in cos_sims.columns:
        print(f"Concept '{concept_label}' not found.")
        return
    
    concept_cos_sims = cos_sims[concept_label]
    most_similar_patches = concept_cos_sims.nlargest(top_n).index.tolist()
    
    # Create figure
    fig, axes = plt.subplots(2, top_n, figsize=(top_n * 3, 6))
    if top_n == 1:
        axes = axes.reshape(2, 1)
    
    # Process each top patch
    for i, patch_idx in enumerate(most_similar_patches):
        # Get image index and patch info
        patches_per_image = (model_input_size[0] // 14) * (model_input_size[1] // 14)
        image_idx = patch_idx // patches_per_image
        patch_in_image = patch_idx % patches_per_image
        
        image = all_images[image_idx]
        resized_image = pad_or_resize_img(image, model_input_size)
        
        # Plot original image
        axes[0, i].imshow(resized_image)
        axes[0, i].set_title(f'Image {image_idx}')
        axes[0, i].axis('off')
        
        # Calculate patch location and highlight
        patches_per_row = model_input_size[0] // 14
        patch_row, patch_col = divmod(patch_in_image, patches_per_row)
        left = patch_col * 14
        top = patch_row * 14
        
        # Create highlighted version
        image_with_patch = resized_image.copy()
        draw = ImageDraw.Draw(image_with_patch)
        draw.rectangle([left, top, left + 14, top + 14], outline="red", width=3)
        
        # Load embeddings for this image's patches
        image_start_idx = image_idx * patches_per_image
        image_patch_indices = list(range(image_start_idx, image_start_idx + patches_per_image))
        image_embeddings = embeds_loader.load_specific_embeddings(image_patch_indices)
        
        # Get concept vector (you might need to load this separately)
        # For now, using the embedding of the top patch as a proxy
        concept_vector = image_embeddings[patch_in_image]
        
        # Compute similarities for heatmap
        similarities = F.cosine_similarity(
            concept_vector.unsqueeze(0),
            image_embeddings
        ).cpu().reshape(model_input_size[1] // 14, model_input_size[0] // 14)
        
        # Plot heatmap
        axes[1, i].imshow(image_with_patch.convert('L'), cmap='gray', alpha=0.4)
        heatmap = axes[1, i].imshow(similarities, cmap='hot', alpha=0.6,
                                   extent=[0, model_input_size[0], model_input_size[1], 0])
        axes[1, i].set_title(f'Similarity: {concept_cos_sims.iloc[patch_idx]:.3f}')
        axes[1, i].axis('off')
    
    plt.suptitle(f"Top {top_n} Patches for Concept: {concept_label}", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    
    plt.show()


def plot_patchsims_all_concepts_from_loader(img_idx, concept_labels, acts_loader, dataset_name,
                                           model_input_size=(224, 224), patch_size=14,
                                           metric_type='Cosine Similarity', vmin=None, vmax=None, 
                                           sort_by_act=False, save_file=None):
    """
    Wrapper: Uses activation loader to create heatmaps and plot all concepts for one image.
    
    Args:
        img_idx (int): Index of the image.
        concept_labels (list): List of concept names to visualize.
        acts_loader: ChunkedActivationLoader instance.
        dataset_name (str): Dataset name.
        model_input_size (tuple): Model input size.
        patch_size (int): Size of patches.
        Other args same as plot_patchsims_all_concepts.
    """
    # Load activations
    cos_sims = acts_loader.load_full_dataframe()
    
    # Calculate patches per image
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Get patch indices for this image
    start_idx = img_idx * patches_per_image
    end_idx = start_idx + patches_per_image
    
    # Extract activations for this image and reshape into heatmaps
    heatmaps = {}
    for concept in concept_labels:
        if concept in cos_sims.columns:
            # Get activations for all patches in this image
            image_acts = cos_sims[concept].iloc[start_idx:end_idx]
            # Reshape into 2D heatmap
            heatmap = torch.tensor(image_acts.values).reshape(patches_per_col, patches_per_row)
            heatmaps[concept] = heatmap
        else:
            print(f"Warning: Concept '{concept}' not found in activations")
    
    # Call the existing function
    plot_patchsims_all_concepts(img_idx, heatmaps, model_input_size, dataset_name,
                               metric_type, vmin, vmax, sort_by_act, save_file)


def plot_patchsims_heatmaps_all_concepts_from_loader(concept_labels, image_indices, acts_loader, 
                                                    dataset_name, model_input_size=(224, 224), 
                                                    patch_size=14, top_n=7, save_file=None, 
                                                    metric_type='Cosine Similarity', vmin=None, vmax=None):
    """
    Wrapper: Uses activation loader to create heatmaps for multiple concepts across multiple images.
    
    Args:
        concept_labels (list): List of concept names.
        image_indices (list): List of image indices to visualize.
        acts_loader: ChunkedActivationLoader instance.
        dataset_name (str): Dataset name.
        model_input_size (tuple): Model input size.
        patch_size (int): Size of patches.
        Other args same as plot_patchsims_heatmaps_all_concepts.
    """
    # Load all images
    all_images, _, _ = load_images(dataset_name)
    
    # Load activations
    cos_sims = acts_loader.load_full_dataframe()
    
    # Calculate patches per image
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Create heatmaps dictionary structure
    heatmaps = {}
    for concept in concept_labels:
        if concept not in cos_sims.columns:
            print(f"Warning: Concept '{concept}' not found")
            continue
        heatmaps[concept] = {}
        
        for img_idx in image_indices:
            # Get patch indices for this image
            start_idx = img_idx * patches_per_image
            end_idx = start_idx + patches_per_image
            
            # Extract and reshape activations
            image_acts = cos_sims[concept].iloc[start_idx:end_idx]
            heatmap = torch.tensor(image_acts.values).reshape(patches_per_col, patches_per_row)
            heatmaps[concept][img_idx] = heatmap
    
    # Use only the first top_n images
    image_indices = image_indices[:top_n]
    
    # Call the existing function
    plot_patchsims_heatmaps_all_concepts(concept_labels, heatmaps, image_indices, all_images,
                                       model_input_size, dataset_name, top_n, 
                                       save_file, metric_type, vmin, vmax)


def plot_superpatches_on_heatmaps(
    concept_labels, heatmaps, image_index, images,
    thresholds, metric_type, model_input_size,
    dataset_name, save_file=None,
    vmin=None, vmax=None, show_colorbar_ticks=True,
    separate_cbar=False, heatmap_alpha=0.6, figure_width=None):
    """
    Plot superpatches on heatmaps with proper font styling.
    
    Args:
        figure_width: Figure width in inches. Height is calculated automatically based on rows.
    """
    # Apply paper plotting style
    from utils.general_utils import get_paper_plotting_style
    plt.rcParams.update(get_paper_plotting_style())
    
    import matplotlib.colors as colors
    
    num_concepts = len(concept_labels)
    concepts_per_row = 3
    num_rows = int(np.ceil(num_concepts / concepts_per_row))

    # Calculate figure dimensions
    if figure_width is None:
        fig_width = 5.5
    else:
        fig_width = figure_width
    
    # Height scales with number of rows
    fig_height = 2.2 * num_rows
    
    # Create figure without subplots for custom layout
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # We'll manually create axes as needed
    axes = np.zeros((num_rows, concepts_per_row + 1), dtype=object)

    # Load image internally if not provided
    if images is None:
        image = retrieve_image(image_index, dataset_name)
    else:
        image = images[image_index]
    resized_image = pad_or_resize_img(image, model_input_size)
    image_np = np.array(resized_image.convert("RGB")) / 255.0
    # Only crop for LLAMA (560x560), not for CLIP (224x224)
    if model_input_size == (224, 224):
        image_cropped = image_np
    else:
        image_cropped = top_left_crop_to_original_aspect(image_np, image.size, resized_image.size)

    if vmin is None or vmax is None:
        all_values = np.concatenate([
            heatmaps[concept].detach().cpu().numpy().flatten()
            for concept in concept_labels
        ])
        vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

    colorbar_im = None

    # For the original image - position it at the top
    # Will update size after calculating ax_width and ax_height
    orig_ax = fig.add_axes([0.01, 0.5, 0.22, 0.22])  # Temporary position
    orig_ax.imshow(image_cropped)
    orig_ax.axis('off')
    orig_ax.set_title("Original", fontsize=plt.rcParams['font.size'], fontweight='bold')

    # Calculate positions for heatmap axes
    ax_width = 0.22  # Width of each axis
    # Scale ax_height based on number of rows to fit better
    ax_height = 0.8 / (num_rows + 0.5)  # Dynamic height based on rows
    x_spacing = 0.23  # Horizontal spacing
    x_offset = 0.28  # Start position for heatmaps
    
    # Position original image with same size as heatmaps, aligned with first row
    # Use consistent spacing: x_offset - x_spacing to maintain even column spacing
    orig_x = x_offset - x_spacing
    # Calculate y position after we know y_start
    # Will update position after calculating heatmap positions
    
    # Calculate vertical spacing between rows
    # Position rows to be centered vertically in the figure
    if num_rows == 1:
        y_spacing = 0
        y_start = 0.5 - ax_height/2  # Center single row
        # Position the original image aligned with the single row
        orig_ax.set_position([orig_x, y_start, ax_width, ax_height])
    else:
        # For multiple rows, make them nearly touching
        vertical_gap = -0.07  # Moderate overlap to bring rows closer
        y_spacing = ax_height + vertical_gap
        
        # Calculate total height and center vertically
        total_height = num_rows * ax_height + (num_rows - 1) * vertical_gap
        y_start = 0.5 + total_height/2 - ax_height
    
    # Now position the original image aligned with the first row
    orig_ax.set_position([orig_x, y_start, ax_width, ax_height])
    
    for idx, concept_label in enumerate(concept_labels):
        row = idx // concepts_per_row
        col = (idx % concepts_per_row)

        # Create axis at calculated position
        x_pos = x_offset + col * x_spacing
        y_pos = y_start - row * y_spacing
        ax = fig.add_axes([x_pos, y_pos, ax_width, ax_height])
        axes[row, col + 1] = ax

        heatmap_orig = heatmaps[concept_label].detach().cpu().numpy()
        print("min:", heatmap_orig.min())
        print("max:", heatmap_orig.max())
        heatmap_resized = Image.fromarray(heatmap_orig).resize(resized_image.size, resample=Image.NEAREST)
        heatmap_resized = np.array(heatmap_resized)
        # Only crop for LLAMA (560x560), not for CLIP (224x224)
        if model_input_size == (224, 224):
            heatmap_cropped = heatmap_resized
        else:
            heatmap_cropped = top_left_crop_to_original_aspect(heatmap_resized, image.size, resized_image.size)

        ax.imshow(image_cropped, alpha=1.0)
        
        # Apply mild smoothing to the values only (not the shapes)
        from scipy.ndimage import gaussian_filter
        # Use a small sigma to smooth color transitions without blurring patch boundaries
        heatmap_smoothed = gaussian_filter(heatmap_cropped, sigma=0.6)
        
        # Use TwoSlopeNorm to center colormap at 0
        # Use the provided vmin/vmax or fall back to this heatmap's range
        heatmap_vmin = vmin if vmin is not None else heatmap_smoothed.min()
        heatmap_vmax = vmax if vmax is not None else heatmap_smoothed.max()
        norm = colors.TwoSlopeNorm(vmin=heatmap_vmin, vcenter=0, vmax=heatmap_vmax)
        im = ax.imshow(heatmap_smoothed, cmap='coolwarm', interpolation='nearest', norm=norm, alpha=heatmap_alpha)
        ax.axis('off')
        ax.set_title(concept_label.capitalize(), fontstyle='italic', fontsize=plt.rcParams['font.size'] - 1)

        if colorbar_im is None:
            colorbar_im = im

        grid_size = heatmap_orig.shape[0]
        patch_h = resized_image.size[1] / grid_size
        patch_w = resized_image.size[0] / grid_size
        threshold = thresholds[concept_label]

        for i_patch in range(grid_size):
            for j_patch in range(grid_size):
                if heatmap_orig[i_patch, j_patch] >= threshold:
                    x = j_patch * patch_w
                    y = i_patch * patch_h
                    rect = mpatches.Rectangle((x, y), patch_w, patch_h,
                                              linewidth=1, edgecolor='#00ff00', facecolor='none')
                    ax.add_patch(rect)

    if separate_cbar:
        # Save main plot without colorbar
        if save_file:
            plt.savefig(save_file, dpi=500, format='pdf', bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Create separate horizontal colorbar figure (rotated from main plot)
        cbar_fig, cbar_ax = plt.subplots(figsize=(6, 0.5))
        cbar_fig.subplots_adjust(left=0.05, right=0.95, top=0.6, bottom=0.3)
        
        # Create a mappable object for the colorbar
        # Use global min/max from all heatmaps for consistent colorbar
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
        sm.set_array([])
        
        # Add horizontal colorbar with same font size
        cbar = cbar_fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(metric_type, fontsize=plt.rcParams['font.size'])
        
        if not show_colorbar_ticks:
            cbar_ax.set_xticks([])
        else:
            # Keep tick label font size consistent
            cbar_ax.tick_params(labelsize=plt.rcParams['font.size'] - 2)
        
        # Save colorbar separately
        if save_file:
            cbar_filename = save_file.rsplit('.', 1)[0] + '_colorbar.' + save_file.rsplit('.', 1)[1]
            plt.savefig(cbar_filename, dpi=500, format='pdf', bbox_inches='tight')
        plt.show()
    else:
        # Add horizontal colorbar aligned with second row of heatmaps
        cbar_width = ax_width  # Same width as the image
        cbar_height = 0.015
        cbar_x = orig_x  # Same x position as original image
        # Position colorbar lower, below the center of second row
        if num_rows > 1:
            cbar_y = y_start - y_spacing + ax_height/2 - cbar_height/2 - 0.03  # Slightly higher (was -0.04)
        else:
            cbar_y = y_start - ax_height - 0.04  # Below single row if only one row (was -0.05)
        cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
        cbar = fig.colorbar(colorbar_im, cax=cbar_ax, orientation='horizontal')
        # Add newline before 'alignment' if it's in the metric_type
        label_text = metric_type.replace(' alignment', '\nalignment') if 'alignment' in metric_type else metric_type
        cbar.set_label(label_text, fontsize=plt.rcParams['font.size'] - 1)  # Same as legend
        cbar.ax.tick_params(labelsize=plt.rcParams['font.size'] - 2)
        
        if not show_colorbar_ticks:
            # Hide tick labels but keep the label
            cbar_ax.set_xticks([])
        
        # Add legend for superdetector tokens under original image, aligned with second row
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        # Create a square marker using Line2D with square marker style
        legend_elements = [Line2D([0], [0], marker='s', color='w', 
                                markerfacecolor='none', markeredgecolor='#00ff00',
                                markeredgewidth=1.5, markersize=8,
                                label=' SuperActivators')]
        # Position legend higher, above the center of second row
        if num_rows > 1:
            # Position legend significantly above colorbar
            legend_y = y_start - y_spacing + ax_height/2 + 0.06  # Much higher
        else:
            legend_y = y_start - ax_height - 0.02  # Below single row
        legend_x = orig_x + ax_width/2  # Center under original image
        legend = fig.legend(handles=legend_elements, loc='center', 
                          bbox_to_anchor=(legend_x, legend_y + 0.003), frameon=False,
                          fontsize=plt.rcParams['font.size'] - 1,
                          handlelength=0.8,  # Slightly increase marker length
                          handletextpad=0.2,  # Reduced padding between marker and text
                          alignment='center')  # Center-align multi-line text

        if save_file:
            plt.savefig(save_file, dpi=500, format='pdf', bbox_inches='tight', facecolor='white')
        plt.show()


def plot_superpatches_on_heatmaps_rotated(
    concept_labels, heatmaps, image_index, images,
    thresholds, metric_type, model_input_size,
    dataset_name, save_file=None,
    vmin=None, vmax=None, show_colorbar_ticks=True,
    separate_cbar=False, heatmap_alpha=0.6, figure_width=None):
    """
    Rotated version of plot_superpatches_on_heatmaps.
    Original image at top center, with 2 columns of 3 heatmaps each below.
    
    Args:
        figure_width: Figure width in inches. Height is calculated automatically.
    """
    # Apply paper plotting style
    from utils.general_utils import get_paper_plotting_style
    plt.rcParams.update(get_paper_plotting_style())
    
    import matplotlib.colors as colors
    
    num_concepts = len(concept_labels)
    if num_concepts > 6:
        raise ValueError("This function supports maximum 6 concepts (2 columns x 3 rows)")
    
    # Calculate figure dimensions
    if figure_width is None:
        fig_width = 5.5
    else:
        fig_width = figure_width
    
    # Height includes space for original image at top plus 3 rows of heatmaps
    fig_height = 6.0  # Adjust as needed
    
    # Create figure without subplots for custom layout
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Load image internally if not provided
    if images is None:
        image = retrieve_image(image_index, dataset_name)
    else:
        image = images[image_index]
    resized_image = pad_or_resize_img(image, model_input_size)
    image_np = np.array(resized_image.convert("RGB")) / 255.0
    # Only crop for LLAMA (560x560), not for CLIP (224x224)
    if model_input_size == (224, 224):
        image_cropped = image_np
    else:
        image_cropped = top_left_crop_to_original_aspect(image_np, image.size, resized_image.size)

    if vmin is None or vmax is None:
        all_values = np.concatenate([
            heatmaps[concept].detach().cpu().numpy().flatten()
            for concept in concept_labels
        ])
        vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

    colorbar_im = None

    # Position original image at top center
    orig_width = 0.3
    orig_height = 0.25
    orig_x = 0.5 - orig_width/2  # Center horizontally
    orig_y = 0.7  # Position near top
    orig_ax = fig.add_axes([orig_x, orig_y, orig_width, orig_height])
    orig_ax.imshow(image_cropped)
    orig_ax.axis('off')
    orig_ax.set_title("Original", fontsize=plt.rcParams['font.size'])

    # Calculate positions for heatmap axes in 2 columns x 3 rows
    ax_width = 0.35  # Width of each axis
    ax_height = 0.18  # Height of each axis
    x_spacing = 0.5  # Horizontal spacing between columns
    y_spacing = 0.22  # Vertical spacing between rows
    
    # Starting positions for the two columns
    left_col_x = 0.1
    right_col_x = 0.55
    
    # Starting y position (below the original image)
    y_start = 0.45
    
    for idx, concept_label in enumerate(concept_labels):
        # Determine column and row
        col = idx % 2  # 0 for left, 1 for right
        row = idx // 2  # 0, 1, or 2
        
        # Calculate position
        x_pos = left_col_x if col == 0 else right_col_x
        y_pos = y_start - row * y_spacing
        
        ax = fig.add_axes([x_pos, y_pos, ax_width, ax_height])

        heatmap_orig = heatmaps[concept_label].detach().cpu().numpy()
        print("min:", heatmap_orig.min())
        print("max:", heatmap_orig.max())
        heatmap_resized = Image.fromarray(heatmap_orig).resize(resized_image.size, resample=Image.NEAREST)
        heatmap_resized = np.array(heatmap_resized)
        # Only crop for LLAMA (560x560), not for CLIP (224x224)
        if model_input_size == (224, 224):
            heatmap_cropped = heatmap_resized
        else:
            heatmap_cropped = top_left_crop_to_original_aspect(heatmap_resized, image.size, resized_image.size)

        ax.imshow(image_cropped, alpha=1.0)
        
        # Apply mild smoothing to the values only (not the shapes)
        from scipy.ndimage import gaussian_filter
        # Use a small sigma to smooth color transitions without blurring patch boundaries
        heatmap_smoothed = gaussian_filter(heatmap_cropped, sigma=0.6)
        
        # Use TwoSlopeNorm to center colormap at 0
        # Use the provided vmin/vmax or fall back to this heatmap's range
        heatmap_vmin = vmin if vmin is not None else heatmap_smoothed.min()
        heatmap_vmax = vmax if vmax is not None else heatmap_smoothed.max()
        norm = colors.TwoSlopeNorm(vmin=heatmap_vmin, vcenter=0, vmax=heatmap_vmax)
        im = ax.imshow(heatmap_smoothed, cmap='coolwarm', interpolation='nearest', norm=norm, alpha=heatmap_alpha)
        ax.axis('off')
        ax.set_title(concept_label.capitalize(), fontstyle='italic', fontsize=plt.rcParams['font.size'] - 1)

        if colorbar_im is None:
            colorbar_im = im

        grid_size = heatmap_orig.shape[0]
        patch_h = resized_image.size[1] / grid_size
        patch_w = resized_image.size[0] / grid_size
        threshold = thresholds[concept_label]

        for i_patch in range(grid_size):
            for j_patch in range(grid_size):
                if heatmap_orig[i_patch, j_patch] >= threshold:
                    x = j_patch * patch_w
                    y = i_patch * patch_h
                    rect = mpatches.Rectangle((x, y), patch_w, patch_h,
                                              linewidth=1, edgecolor='#00ff00', facecolor='none')
                    ax.add_patch(rect)

    if separate_cbar:
        # Save main plot without colorbar
        if save_file:
            plt.savefig(save_file, dpi=500, format='pdf', bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Create separate horizontal colorbar figure
        cbar_fig, cbar_ax = plt.subplots(figsize=(6, 0.5))
        cbar_fig.subplots_adjust(left=0.05, right=0.95, top=0.6, bottom=0.3)
        
        # Create a mappable object for the colorbar
        # Use global min/max from all heatmaps for consistent colorbar
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
        sm.set_array([])
        
        # Add horizontal colorbar with same font size
        cbar = cbar_fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(metric_type, fontsize=plt.rcParams['font.size'])
        
        if not show_colorbar_ticks:
            cbar_ax.set_xticks([])
        else:
            # Keep tick label font size consistent
            cbar_ax.tick_params(labelsize=plt.rcParams['font.size'] - 2)
        
        # Save colorbar separately
        if save_file:
            cbar_filename = save_file.rsplit('.', 1)[0] + '_colorbar.' + save_file.rsplit('.', 1)[1]
            plt.savefig(cbar_filename, dpi=500, format='pdf', bbox_inches='tight')
        plt.show()
    else:
        # Original horizontal colorbar centered at bottom
        cbar_width = 0.6
        cbar_center = 0.5 - cbar_width / 2
        cbar_ax = fig.add_axes([cbar_center, 0.02, cbar_width, 0.03])
        cbar = fig.colorbar(colorbar_im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(metric_type, fontsize=plt.rcParams['font.size'])
        cbar.ax.tick_params(labelsize=plt.rcParams['font.size'] - 2)
        
        if not show_colorbar_ticks:
            # Hide tick labels but keep the label
            cbar_ax.set_xticks([])

        if save_file:
            plt.savefig(save_file, dpi=500, format='pdf', bbox_inches='tight', facecolor='white')
        plt.show()


def plot_best_detecting_clusters_calibrated(dataset_name: str,
                                           model_name: str,
                                           sample_type: str,
                                           n_clusters: int,
                                           acts_loader,
                                           concepts_to_show=None,
                                           top_n_clusters: int = 3,
                                           top_n_samples: int = 5,
                                           metric: str = 'f1',
                                           model_input_size=None,
                                           test_only: bool = True,
                                           percent_thru_model: int = 100,
                                           save_dir=None):
    """
    Plot the best detecting clusters for concepts, using the percentile that performed best on calibration set.
    
    Args:
        dataset_name: Name of dataset
        model_name: Name of model (e.g., 'CLIP', 'Llama')
        sample_type: 'patch' or 'cls'
        n_clusters: Number of clusters in k-means
        acts_loader: Activation loader
        concepts_to_show: List of specific concepts to show (None for all)
        top_n_clusters: Number of best clusters to show per concept
        top_n_samples: Number of samples to show per cluster
        metric: Metric to use for ranking clusters ('f1', 'precision', 'recall')
        model_input_size: Model input size (required for patch)
        test_only: Whether to use only test samples
        percent_thru_model: Percentage through model
        save_dir: Directory to save figures
    """
    from calibration_selection_utils import find_best_percentile_from_calibration_allpairs
    import ast
    
    # Construct concept label
    con_label = f"{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
    
    # Find best percentile using calibration results
    print(f"Finding best percentile using calibration set performance...")
    try:
        best_percentile, cal_matches = find_best_percentile_from_calibration_allpairs(
            dataset_name, 
            con_label,
            concepts_to_include=concepts_to_show,
            metric=metric
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure you've run all_detection_stats.py to generate calibration results.")
        return
    
    print(f"Selected percentile {best_percentile} based on calibration {metric}")
    
    # Load test results at the best percentile
    test_path = f"Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{best_percentile}_{con_label}.csv"
    
    if not os.path.exists(test_path):
        print(f"Test results not found at best percentile: {test_path}")
        return
    
    # Load and parse test results
    test_df = pd.read_csv(test_path)
    
    # Parse concept column and find best clusters for each concept on test set
    test_matches = {}
    
    for concept in cal_matches.keys():
        concept_rows = []
        for _, row in test_df.iterrows():
            concept_tuple = ast.literal_eval(row['concept'])
            if concept_tuple[0] == concept:
                concept_rows.append({
                    'cluster_id': concept_tuple[1],
                    metric: row[metric]
                })
        
        if concept_rows:
            concept_df = pd.DataFrame(concept_rows)
            top_clusters = concept_df.nlargest(top_n_clusters, metric)
            test_matches[concept] = [
                (row['cluster_id'], row[metric]) 
                for _, row in top_clusters.iterrows()
            ]
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot results
    print(f"\nShowing best detecting clusters at percentile {best_percentile}:")
    print("="*60)
    
    for concept_name, cluster_list in test_matches.items():
        print(f"\nConcept: {concept_name}")
        print(f"Calibration best: cluster {cal_matches[concept_name][0]} ({metric}={cal_matches[concept_name][1]:.3f})")
        print(f"Test set top {top_n_clusters} clusters:")
        
        for cluster_id, metric_value in cluster_list:
            print(f"  - Cluster {cluster_id}: {metric}={metric_value:.3f}")
        
        # Plot based on sample type
        if sample_type == 'cls':
            # For CLS, create subplot for all clusters
            n_cols = min(top_n_clusters, len(cluster_list))
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
            if n_cols == 1:
                axes = [axes]
            
            for idx, (cluster_id, metric_value) in enumerate(cluster_list[:n_cols]):
                plt.sca(axes[idx])
                
                # Use existing visualization function
                plot_aligned_images(
                    acts_loader=acts_loader,
                    con_label=f"cluster_{cluster_id}",
                    concept_key=str(cluster_id),
                    k=top_n_samples,
                    dataset_name=dataset_name,
                    metric_type=f'Cluster {cluster_id} Activation',
                    save_image=False,
                    test_only=test_only
                )
                
                axes[idx].set_title(f'Cluster {cluster_id} → {concept_name}\n({metric}={metric_value:.3f}, percentile={best_percentile})')
            
            plt.suptitle(f'Best Detecting Clusters for "{concept_name}" (Cal-selected p={best_percentile})', fontsize=14, y=1.02)
            plt.tight_layout()
            
            if save_dir:
                save_path = os.path.join(save_dir, f"{concept_name.replace('::', '_')}_best_clusters_p{best_percentile}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        elif sample_type == 'patch':
            # For patch embeddings, show heatmaps
            if model_input_size is None:
                raise ValueError("model_input_size required for patch visualization")
            
            for idx, (cluster_id, metric_value) in enumerate(cluster_list):
                plot_most_similar_patches_w_heatmaps_and_corr_images(
                    concept_label=str(cluster_id),
                    acts_loader=acts_loader,
                    con_label=f"{concept_name}_cluster_{cluster_id}_p{best_percentile}",
                    dataset_name=dataset_name,
                    model_input_size=model_input_size,
                    top_n=top_n_samples,
                    metric_type=f'Cluster {cluster_id} → {concept_name} ({metric}={metric_value:.3f}, p={best_percentile})',
                    test_only=test_only,
                    save_path=f"{save_dir}/{concept_name.replace('::', '_')}_cluster_{cluster_id}_p{best_percentile}.png" if save_dir else None
                )


def compare_calibration_vs_fixed_percentiles(dataset_name: str,
                                            model_name: str,
                                            sample_type: str,
                                            n_clusters: int,
                                            fixed_percentiles=None,
                                            metric: str = 'f1',
                                            percent_thru_model: int = 100):
    """
    Compare performance of calibration-selected percentile vs fixed percentiles.
    
    Args:
        dataset_name: Name of dataset
        model_name: Model name
        sample_type: 'patch' or 'cls'
        n_clusters: Number of clusters
        fixed_percentiles: List of fixed percentiles to compare against
        metric: Metric to compare
        percent_thru_model: Percentage through model
    """
    from calibration_selection_utils import find_best_percentile_from_calibration_allpairs
    import ast
    
    if fixed_percentiles is None:
        fixed_percentiles = [0.1, 0.5, 0.9]
    
    con_label = f"{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
    
    # Get calibration-selected percentile
    try:
        best_percentile, cal_matches = find_best_percentile_from_calibration_allpairs(
            dataset_name, con_label, metric=metric
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Compare performance
    results = []
    
    # Add calibration-selected result
    test_path = f"Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{best_percentile}_{con_label}.csv"
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        # Calculate average best metric for each concept
        concept_best_metrics = []
        for concept in cal_matches.keys():
            concept_rows = []
            for _, row in test_df.iterrows():
                concept_tuple = ast.literal_eval(row['concept'])
                if concept_tuple[0] == concept:
                    concept_rows.append(row[metric])
            if concept_rows:
                concept_best_metrics.append(max(concept_rows))
        
        results.append({
            'method': f'Calibration-selected (p={best_percentile})',
            'percentile': best_percentile,
            f'test_avg_best_{metric}': np.mean(concept_best_metrics)
        })
    
    # Add fixed percentile results
    for p in fixed_percentiles:
        test_path = f"Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{p}_{con_label}.csv"
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            concept_best_metrics = []
            for concept in cal_matches.keys():
                concept_rows = []
                for _, row in test_df.iterrows():
                    concept_tuple = ast.literal_eval(row['concept'])
                    if concept_tuple[0] == concept:
                        concept_rows.append(row[metric])
                if concept_rows:
                    concept_best_metrics.append(max(concept_rows))
            
            results.append({
                'method': f'Fixed (p={p})',
                'percentile': p,
                f'test_avg_best_{metric}': np.mean(concept_best_metrics)
            })
    
    # Create comparison plot
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if 'Calibration' in row['method'] else 'blue' for _, row in results_df.iterrows()]
    bars = plt.bar(results_df['method'], results_df[f'test_avg_best_{metric}'], color=colors)
    
    # Add value labels on bars
    for bar, value in zip(bars, results_df[f'test_avg_best_{metric}']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.ylabel(f'Average Best {metric.upper()} on Test Set')
    plt.title(f'Calibration-Selected vs Fixed Percentiles\n{dataset_name} - {model_name} {sample_type} (n_clusters={n_clusters})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    print("\nComparison Results:")
    print(results_df.to_string(index=False))
    
    return results_df


def plot_patches_by_activation_levels(acts_loader, concept_name, percentiles, 
                                     dataset_name, model_input_size,
                                     n_patches=5, patch_size=14, test_only=True,
                                     save_dir=None, thresholds_dict=None):
    """
    Plot patches at different activation levels for a concept:
    - Top n most activated patches
    - Bottom n least activated patches  
    - n patches closest to each percentile threshold
    
    Args:
        acts_loader: ChunkedActivationLoader containing activations
        concept_name: Name of the concept to visualize
        percentiles: List of percentiles (e.g., [0.05, 0.5, 0.95])
        dataset_name: Name of the dataset
        model_input_size: Model input size tuple
        n_patches: Number of patches to show for each category
        patch_size: Size of patches (default 14)
        test_only: Whether to only use test samples
        save_dir: Directory to save figures (optional)
        thresholds_dict: Pre-computed thresholds dict where thresholds_dict[percentile][concept_name] gives threshold
    """
    
    # Load activations for the concept
    print(f"Loading activations for concept: {concept_name}")
    
    # Get concept index
    if concept_name not in acts_loader.columns:
        raise ValueError(f"Concept '{concept_name}' not found. Available concepts: {acts_loader.columns[:10]}...")
    
    concept_idx = acts_loader.columns.index(concept_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Filter for test samples and non-padding patches
    if test_only:
        split_df = get_patch_split_df(dataset_name, model_input_size, patch_size)
        test_indices = split_df[split_df == 'test'].index.tolist()
        
        # Filter out padding patches
        print("Filtering out padding patches...")
        relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size).tolist()
        
        # Load activations in larger chunks for GPU efficiency
        activations = []
        patch_indices = []
        
        # Larger chunk size for efficiency
        chunk_size = 100000
        
        print(f"Loading {len(relevant_indices)} non-padding test patches in chunks of {chunk_size}...")
        for i in tqdm(range(0, len(relevant_indices), chunk_size), desc="Loading chunks"):
            chunk_indices = relevant_indices[i:i+chunk_size]
            chunk_acts = acts_loader.load_concept_activations_for_indices(
                concept_name, chunk_indices, device=device
            )
            # Keep on GPU for now
            activations.append(chunk_acts)
            patch_indices.extend(chunk_indices)
        
        # Concatenate on GPU
        activations_tensor = torch.cat(activations)
        patch_indices = torch.tensor(patch_indices, device=device)
    else:
        # Load all activations for this concept (filtering out padding)
        print("Loading all activations (this may take a while)...")
        
        # Get all indices and filter out padding
        all_indices = list(range(acts_loader.total_samples))
        print("Filtering out padding patches from all data...")
        relevant_indices = filter_patches_by_image_presence(all_indices, dataset_name, model_input_size).tolist()
        
        # Load activations in chunks
        chunk_size = 100000
        activations = []
        patch_indices = []
        
        print(f"Loading {len(relevant_indices)} non-padding patches in chunks of {chunk_size}...")
        for i in tqdm(range(0, len(relevant_indices), chunk_size), desc="Loading chunks"):
            chunk_indices = relevant_indices[i:i+chunk_size]
            chunk_acts = acts_loader.load_concept_activations_for_indices(
                concept_name, chunk_indices, device=device
            )
            activations.append(chunk_acts)
            patch_indices.extend(chunk_indices)
        
        activations_tensor = torch.cat(activations)
        patch_indices = torch.tensor(patch_indices, device=device)
    
    # Get thresholds - either from provided dict or calculate
    if thresholds_dict is not None:
        print("Using provided thresholds...")
        thresholds_list = []
        for p in percentiles:
            if p in thresholds_dict and concept_name in thresholds_dict[p]:
                threshold_val = thresholds_dict[p][concept_name]
                # Handle if it's a list/tensor with single value
                if hasattr(threshold_val, '__len__'):
                    threshold_val = threshold_val[0]
                thresholds_list.append(threshold_val)
            else:
                raise ValueError(f"Threshold for percentile {p} and concept '{concept_name}' not found in thresholds_dict")
        thresholds_tensor = torch.tensor(thresholds_list, device=device)
    else:
        # Calculate percentile thresholds on GPU
        print("Calculating percentiles...")
        thresholds_tensor = torch.quantile(
            activations_tensor, 
            torch.tensor([p for p in percentiles], device=device)
        )
    
    # Find patches for each category on GPU
    print("Finding patches for each category...")
    patches_to_plot = {}
    
    # 1. Top n most activated patches - use torch.topk for efficiency
    top_values, top_indices = torch.topk(activations_tensor, n_patches)
    patches_to_plot['Most Activated'] = [
        (patch_indices[idx].item(), val.item()) 
        for idx, val in zip(top_indices, top_values)
    ]
    
    # 2. Bottom n least activated patches
    bottom_values, bottom_indices = torch.topk(activations_tensor, n_patches, largest=False)
    patches_to_plot['Least Activated'] = [
        (patch_indices[idx].item(), val.item()) 
        for idx, val in zip(bottom_indices, bottom_values)
    ]
    
    # 3. Patches closest to each percentile - vectorized computation
    for i, (percentile, threshold) in enumerate(zip(percentiles, thresholds_tensor)):
        # Compute distances on GPU
        distances = torch.abs(activations_tensor - threshold)
        _, closest_indices = torch.topk(distances, n_patches, largest=False)
        
        patches_to_plot[f'Percentile {int(percentile*100)}%'] = [
            (patch_indices[idx].item(), activations_tensor[idx].item()) 
            for idx in closest_indices
        ]
    
    # Calculate figure dimensions
    n_categories = len(patches_to_plot)
    fig, axes = plt.subplots(n_categories, n_patches, 
                            figsize=(n_patches * 3, n_categories * 3))
    
    if n_categories == 1:
        axes = axes.reshape(1, -1)
    if n_patches == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each category
    for row_idx, (category_name, patch_list) in enumerate(patches_to_plot.items()):
        for col_idx, (patch_idx, activation_val) in enumerate(patch_list):
            ax = axes[row_idx, col_idx]
            
            # Get image index and load image
            image_idx = get_image_idx_from_global_patch_idx(patch_idx, model_input_size, patch_size)
            image = retrieve_image(image_idx, dataset_name, test_only=False)
            
            # Calculate patch location
            left, top, right, bottom = calculate_patch_location(image, patch_idx, patch_size, model_input_size)
            
            # Resize image and create visualization
            resized_image = pad_or_resize_img(image, model_input_size)
            
            # Calculate patches per row/col for title
            patches_per_row = model_input_size[0] // patch_size
            patches_per_col = model_input_size[1] // patch_size
            
            # Convert to numpy array for display
            image_np = np.array(resized_image.convert("RGB"))
            
            # Only crop for LLAMA (560x560), not for CLIP (224x224)
            if model_input_size == (224, 224):
                display_image = image_np
            else:
                # Crop to remove padding for LLAMA
                display_image = top_left_crop_to_original_aspect(image_np, image.size, resized_image.size)
            
            # Convert back to PIL for drawing
            display_image_pil = Image.fromarray(display_image)
            draw = ImageDraw.Draw(display_image_pil)
            
            # Adjust patch coordinates if image was cropped
            if model_input_size != (224, 224):
                # For cropped images, we need to ensure the rectangle is within bounds
                crop_height = display_image.shape[0]
                if bottom <= crop_height:  # Only draw if patch is within cropped area
                    draw.rectangle([left, top, right, bottom], outline="red", width=3)
            else:
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
            
            # Show the image with red rectangle
            ax.imshow(display_image_pil)
            ax.set_title(f"Alignment: {activation_val:.3f}", fontsize=10)
            
            ax.axis('off')
            
            # Add category label on first column
            if col_idx == 0:
                ax.text(-0.1, 0.5, category_name, transform=ax.transAxes,
                       fontsize=12, fontweight='bold', va='center', ha='right',
                       rotation=90)
    
    # Add overall title
    plt.suptitle(f"Patches for Concept: {concept_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{concept_name}_activation_levels.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics (compute on GPU)
    print(f"\nSummary for concept '{concept_name}':")
    print(f"Total patches analyzed: {len(activations_tensor)}")
    print(f"Activation range: [{activations_tensor.min().item():.3f}, {activations_tensor.max().item():.3f}]")
    print(f"Percentile thresholds: {dict(zip([f'{int(p*100)}%' for p in percentiles], [t.item() for t in thresholds_tensor]))}")










