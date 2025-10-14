"""General utils"""
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

def retrieve_image(img_idx, dataset_name, test_only=False):
    """
    Retrieves an image from the specified dataset based on the given index.

    Args:
        img_idx (int): Index of the image in the dataset's metadata.
        dataset_name (str): Name of the dataset (default is 'CLEVR').

    Returns:
        PIL.Image: The image corresponding to the specified index.
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    image_path = os.path.join(f'../Data/{dataset_name}', metadata.iloc[img_idx]['image_path'])
    # if test_only:
    #     metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)
    image = Image.open(image_path).convert("RGB")
    return image


def create_image_loader_function(dataset_name):
    """
    Creates a function that loads images on demand by index.
    
    Args:
        dataset_name (str): Name of the dataset.
        
    Returns:
        callable: A function that takes an image index and returns a PIL Image.
    """
    def load_image(image_index):
        return retrieve_image(image_index, dataset_name)
    return load_image


def load_images(dataset_name, model_input_size=None):
    """
    Load images from a dataset.

    Args:
        dataset_name (str): The name of the dataset. Defaults to 'CLEVR'.

    Returns:
        list: A list of PIL.Image objects.
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
        
    image_paths = metadata['image_path'].tolist()
    if 'split' in metadata.columns:
        splits = metadata['split'].tolist()

    all_images, train_images, test_images = [], [], []
    for idx, info in tqdm(metadata.iterrows(), total=len(metadata), desc="Loading Images"):
        image_filename = info['image_path']
        # Always load from local Data directory
        image = Image.open(f'../Data/{dataset_name}/{image_filename}').convert("RGB")
        if model_input_size: #reshape image if it's bigger than the model input size (could deal with this through tiles)
            new_width, new_height = get_resized_dims_w_same_ar(image.size, model_input_size)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        all_images.append(image)
        
        if 'split' in metadata.columns:
            split = info['split']
            if split == "train":
                train_images.append(image)
            else:
                test_images.append(image)
    print(f"Loaded {len(all_images)} images from local Data directory.")
    
    # Check for duplicates in the metadata
    if len(metadata) != len(metadata['image_path'].unique()):
        print(f"WARNING: Duplicate image paths found in metadata!")
        print(f"  - Total rows: {len(metadata)}")
        print(f"  - Unique images: {len(metadata['image_path'].unique())}")

    return all_images, train_images, test_images



def load_text(dataset_name):
    """
    Load one text sample per unique sentence (sample_filename), assigning to train/test based on split.

    Args:
        dataset_name (str): Dataset name (e.g., "IMDB").

    Returns:
        tuple: (all_text, train_text, test_text)
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')

    # Use drop_duplicates to avoid reading the same file multiple times
#     unique_samples = metadata.drop_duplicates(subset='sample_filename')

    all_text, train_text, test_text, cal_text = [], [], [], []

    # for _, row in tqdm(unique_samples.iterrows(), total=len(unique_samples), desc="Loading unique sentences"):
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Loading unique sentences"):
        #filename = row['sample_filename']
        filename = row['text_path']
        with open(f'../Data/{dataset_name}/{filename}', 'r') as f:
            text = f.read()
        all_text.append(text)
        if row['split'] == 'train':
            train_text.append(text)
        elif row['split'] == 'test':
            test_text.append(text)
        elif row['split'] == 'cal':
            cal_text.append(text)

    print(f"Loaded {len(all_text)} unique text samples.")
    return all_text, train_text, test_text, cal_text





# def pad_tensors_to_size(tensors, input_model_size):
#     """
#     Pads the given tensors to match the target size (input_model_size) by adding -1 values to the right and bottom.
#     The original tensor will be placed in the top-left corner, and the rest will be filled with -1.

#     Args:
#         masks (torch.Tensor): The input masks with shape (num_images, height, width).
#         input_model_size (tuple): The target size (height, width) to pad to.

#     Returns:
#         torch.Tensor: The padded masks with shape (num_images, input_model_size[0], input_model_size[1]).
#     """
#     num_images, orig_height, orig_width = tensors.shape
#     target_height, target_width = input_model_size

#     # If the tensor is smaller than the target size, we add padding
#     pad_height = max(target_height - orig_height, 0)
#     pad_width = max(target_width - orig_width, 0)

#     # Add padding with -1 values to the right and bottom of the tensors
#     padded_tensors = F.pad(tensors, (0, pad_width, 0, pad_height), value=-1)
    
#     return padded_tensors

def get_resized_dims_w_same_ar(original_dims, model_input_size):
    # Calculate the aspect ratio
    orig_width, orig_height = original_dims
    model_input_width, model_input_height = model_input_size
    aspect_ratio = orig_width / orig_height

    # Determine the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        # Width is the larger dimension
        new_width = model_input_width
        new_height = int(new_width / aspect_ratio)
    else:
        # Height is the larger dimension
        new_height = model_input_height
        new_width = int(new_height * aspect_ratio)
    return (new_width, new_height)


# def pad_or_resize_img(image, model_input_size, fill_color=(0, 0, 0)):
#     """
#     Resizes an image in the same way the preprocessors do.

#     Args:
#         image (PIL.Image): The input image.
#         model_input_size (tuple): The target size as (width, height).

#     Returns:
#         PIL.Image: The resized and padded image.
#     """
#     if model_input_size == (224, 224): #CLIP -> just resize image
#         processed_image = image.resize(model_input_size)
#     elif model_input_size == (560, 560): #Llama -> resize while maintaining aspect ratio
#         new_width, new_height = get_resized_dims_w_same_ar(image.size, model_input_size)

#         # Resize the image while maintaining aspect ratio
#         image = image.resize((new_width, new_height), Image.LANCZOS)

#         # Calculate padding if needed
#         pad_width = model_input_size[0] - new_width
#         pad_height = model_input_size[1] - new_height

#         # Calculate padding on each side (right and bottom)
#         left = 0
#         top = 0
#         right = pad_width
#         bottom = pad_height

#         # Ensure the image is in RGB mode if it's not
#         if image.mode != 'RGB':
#             image = image.convert('RGB')

#         # Add padding only to the right and bottom
#         processed_image = ImageOps.expand(image, border=(left, top, right, bottom), fill=fill_color)
        
#     return processed_image

# def pad_or_resize_img_tensor(image_tensor, model_input_size):
#     """
#     Pads or resizes an image tensor to match the target model input size.
#     If the image is smaller than model_input_size, it adds padding; if larger, it resizes.

#     Args:
#         image_tensor (torch.Tensor): The input image tensor (C, H, W).
#         model_input_size (tuple): The target size as (height, width).

#     Returns:
#         torch.Tensor: The adjusted image tensor with shape (C, model_input_size[0], model_input_size[1]).
#     """
#     # Convert numpy array to tensor if needed
#     if isinstance(image_tensor, np.ndarray):
#         image_tensor = torch.tensor(image_tensor)

#     # Convert bool tensor to float (since bool tensors can't be processed as images)
#     if image_tensor.dtype == torch.bool:
#         image_tensor = image_tensor.float()
        
#     # Ensure the tensor is in (H, W) format
#     if image_tensor.ndim == 3:  # (C, H, W) -> Convert to (H, W) by taking a single channel
#         image_tensor = image_tensor[0]
        
#     # Convert the 2D tensor to a PIL Image (grayscale)
#     transform_to_pil = transforms.ToPILImage()
#     image = transform_to_pil(image_tensor.unsqueeze(0))  # Add a channel dim (1, H, W)

#     # Call the padding/resizing function
#     padded_image = pad_or_resize_img(image, model_input_size)
    
#     # Convert to grayscale
#     grayscale_padded_image = padded_image.convert("L")
    
#     # Convert to tensor
#     transform = transforms.ToTensor()
#     padded_grayscale_tensor = transform(grayscale_padded_image).squeeze(0)  # Remove channel dimension (C, H, W) -> (H, W)
    
#     binary_padded_grayscale_tensor = (padded_grayscale_tensor > 0.5).int()

#     return binary_padded_grayscale_tensor

def pad_or_resize_img(image, model_input_size, fill_color=(0, 0, 0), is_mask=False):
    """
    Resizes an image or binary mask in the same way the preprocessors do.

    Args:
        image (PIL.Image): The input image.
        model_input_size (tuple): The target size as (width, height).
        fill_color (tuple or int): Fill color for padding.
        is_mask (bool): Whether the input is a binary mask (uses nearest neighbor resampling).

    Returns:
        PIL.Image: The resized and padded image.
    """
    resample_method = Image.NEAREST if is_mask else Image.LANCZOS

    if model_input_size == (224, 224):  # CLIP
        processed_image = image.resize(model_input_size, resample=resample_method)

    elif model_input_size == (560, 560):  # LLaMA
        new_width, new_height = get_resized_dims_w_same_ar(image.size, model_input_size)
        image = image.resize((new_width, new_height), resample=resample_method)

        pad_width = model_input_size[0] - new_width
        pad_height = model_input_size[1] - new_height
        left, top, right, bottom = 0, 0, pad_width, pad_height

        if not is_mask and image.mode != 'RGB':
            image = image.convert('RGB')

        # Override fill_color for grayscale masks
        fill_color = 0 if is_mask else fill_color

        processed_image = ImageOps.expand(image, border=(left, top, right, bottom), fill=fill_color)

    return processed_image


def pad_or_resize_img_tensor(image_tensor, model_input_size, is_mask=False):
    """
    Pads or resizes a binary image tensor to match the target model input size.

    Args:
        image_tensor (torch.Tensor): The input mask (C, H, W) or (H, W).
        model_input_size (tuple): The target size as (height, width).

    Returns:
        torch.Tensor: A resized binary tensor (H, W) with values in {0, 1}.
    """
    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.tensor(image_tensor)

    if image_tensor.dtype == torch.bool:
        image_tensor = image_tensor.float()

    if image_tensor.ndim == 3:
        image_tensor = image_tensor[0]

    transform_to_pil = transforms.ToPILImage()
    image = transform_to_pil(image_tensor.unsqueeze(0))

    padded_image = pad_or_resize_img(image, model_input_size, is_mask=is_mask)

    # Convert to grayscale if needed
    grayscale_padded_image = padded_image.convert("L")

    # Convert to NumPy array (pixel values 0–255)
    padded_np = np.array(grayscale_padded_image)

    # Binarize *before* converting to tensor
    binary_np = (padded_np > 0).astype(np.uint8)  # or np.int32 if you prefer

    # Convert to tensor
    binary_tensor = torch.from_numpy(binary_np).int()
    return binary_tensor



def retrieve_topn_samples(dataset_name, top_n, start_idx=0, split='test'):
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    if split == 'both':
        my_indices = metadata.index[start_idx:top_n+start_idx]
    else:
        filtered = metadata[metadata['split'] == split]
        my_indices = filtered.iloc[start_idx:start_idx + top_n].index
    return my_indices

def retrieve_topn_images_byconcepts(dataset_name, top_n, concepts, split='test'):
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    split_indices = metadata[metadata['split'] == split]
    
    split_concept_indices = []
    for idx in split_indices.index:
        has_both_concepts = True
        for concept in concepts:
            if metadata[concept].iloc[idx] != 1:
                has_both_concepts = False
                break
        if has_both_concepts:
            split_concept_indices.append(idx)
        if len(split_concept_indices) >= top_n:
            break
        
    return split_concept_indices


def retrieve_present_concepts(sample_idx, dataset_name):
    # Define the path to the metadata file based on the dataset
    data_dir = f'../Data/{dataset_name}/'
    metadata_df = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')

    # Select the metadata for the image at img_idx
    img_metadata = metadata_df.iloc[sample_idx]

    # Extract the column names for categories and supercategories
    category_columns = [col for col in metadata_df.columns if col not in ['image_path']]

    # Initialize an empty list to store the present concepts
    present_concepts = []

    # Iterate through all the categories and supercategories
    for concept in category_columns:
        if img_metadata[concept] == 1:
            present_concepts.append(concept) 

    return present_concepts


def get_split_df(dataset_name):
    """
    Expands an image-level metadata DataFrame to a per-patch split DataFrame.

    Args:
        image_metadata_df (pd.DataFrame): DataFrame containing image-level metadata, including a "split" column.
        num_patches (int): Number of patches per image (e.g., 14x14 = 196 patches).

    Returns:
        pd.DataFrame: A new DataFrame where each patch has its own row and inherits the split from the image.
    """
    per_sample_metadata_df = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    split_df = per_sample_metadata_df['split']
    
    return split_df


def get_global_index_from_split_index(dataset_name, split_name, nth_in_split):
    """
    Converts the nth sample in a specific split to its global index in the dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
        split_name (str): Split name ('train', 'test', 'cal', etc.).
        nth_in_split (int): The nth sample within the specified split (0-indexed).
        
    Returns:
        int: Global index of the sample in the full dataset.
        
    Example:
        # Get the global index of the 5th test sample
        global_idx = get_global_index_from_split_index('CLEVR', 'test', 4)
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    
    # Get all indices for the specified split
    split_indices = metadata[metadata['split'] == split_name].index
    
    if nth_in_split >= len(split_indices):
        raise ValueError(f"Index {nth_in_split} out of bounds for split '{split_name}' "
                        f"which has {len(split_indices)} samples")
    
    # Return the global index
    return split_indices[nth_in_split]


def get_split_index_from_global_index(dataset_name, global_index):
    """
    Converts a global index to its position within its split.
    
    Args:
        dataset_name (str): Name of the dataset.
        global_index (int): Global index in the full dataset.
        
    Returns:
        tuple: (split_name, nth_in_split) where split_name is the split the sample belongs to
               and nth_in_split is its 0-indexed position within that split.
               
    Example:
        # Get split info for global index 1000
        split_name, nth = get_split_index_from_global_index('CLEVR', 1000)
        # Returns e.g., ('test', 42) meaning it's the 43rd test sample
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    
    if global_index >= len(metadata):
        raise ValueError(f"Global index {global_index} out of bounds for dataset with {len(metadata)} samples")
    
    # Get the split for this index
    split_name = metadata.iloc[global_index]['split']
    
    # Get all indices for this split
    split_indices = metadata[metadata['split'] == split_name].index
    
    # Find the position within the split
    nth_in_split = list(split_indices).index(global_index)
    
    return split_name, nth_in_split


def create_binary_labels(D, gt_samples_per_concept):
    """
    Create binary labels for each embedding based on concept indices.
    
    Args:
        D number of embeddings
        concept_gt_patches (set): Set of indices where the concept is present.
    
    Returns:
        torch.Tensor: A binary label tensor of shape (N,), where 1 indicates the presence of the concept.
    """
    all_concept_labels = {}
    for concept, samples in gt_samples_per_concept.items():
        concept_labels = torch.zeros(D, dtype=torch.float32)
        if len(samples) > 0:
            # Convert to tensor for faster indexing
            indices = torch.tensor(list(samples), dtype=torch.long)
            concept_labels[indices] = 1
        all_concept_labels[concept] = concept_labels
    return all_concept_labels

def compute_cossim_w_vector(vector, embeddings):
    """
    Computes the cosine similarity between a given vector and all embeddings in the embeddings tensor.

    Args:
        vector (torch.Tensor): A tensor of shape (D,) representing the random vector.
        embeddings (torch.Tensor): A tensor of shape (N, D) containing the embeddings, 
            where N is the number of embeddings, and D is the dimension of each embedding.

    Returns:
        torch.Tensor: A tensor of cosine similarities between the random vector and all embeddings.
    """
    vector = vector.to(embeddings.device)
    cosine_similarities = F.cosine_similarity(embeddings, vector.unsqueeze(0), dim=1)
    return cosine_similarities


###Visualizations###
def plot_image_with_attributes(image_index, dataset_name='CLEVR', save_image=False, test_only=True):
    """
    Plots an image with its associated attributes from a dataset.

    Args:
        image_index (int): The index of the image in the dataset.
        dataset_name (str): Name of the dataset to load the image from.
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    # if test_only:
    #     metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 11)

    info = metadata.iloc[image_index]
    attributes = [attr for attr in info.index if ((attr not in ['image_path', 'class', 'split']) and (info.loc[attr] == 1))]

    image_path = info.loc['image_path']
    img = Image.open(f'../Data/{dataset_name}/{image_path}')

    # Create a new image with extra space at the bottom to accommodate the text
    text_height = 4  # Adjust the height of the text area
    new_img = Image.new('RGB', (img.width, img.height + text_height * 15), color=(255, 255, 255))  # Added extra space for multiple lines
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)

    # Prepare the text string with attributes separated by commas
    attribute_text = ', '.join(attributes)

    # Wrap the text if it's too wide
    max_width = img.width  # Keep some padding from the edge
    words = attribute_text.split(', ')
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line}, {word}" if current_line else word
        test_bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = test_bbox[2] - test_bbox[0]
        if test_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line + ",")  # Add comma at the end of each line
            current_line = word
    lines.append(current_line)  # No comma at the end of the last line

    # Draw the text on the new image, below the original image
    y_offset = img.height + 5  # Start a little below the image
    for line in lines:
        draw.text((0, y_offset), line, font=font, fill="black")
        y_offset += text_height  # Move down for the next line of text

    # Show the image with the attributes underneath it
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()

    # Save the new image with attributes underneath
    if save_image:
        output_image_path = f'../Figs/{dataset_name}/examples/example_{image_index}.jpg'
        new_img.save(output_image_path, dpi=(500, 500))
        

def plot_random_image_samples(dataset_name='CLEVR', num_samples=10, save_image=True):
    """
    Plots random sample images with their attributes from a given dataset.

    Args:
        dataset_name (str): Name of the dataset to load images from.
        num_samples (int): Number of sample images to plot.
        save_image (Boolean): Whether to save png file of image.
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    total_samples = len(metadata)

    random_indices = np.random.choice(total_samples, num_samples, replace=False)

    for idx in random_indices:
        plot_image_with_attributes(idx, dataset_name, save_image=save_image)
        

def get_coco_concepts():
    high_freq_concepts = ['accessory', 'animal', 'appliance', 'bench', 'book', 'bottle', 'bowl', 'bus', 'car', 
                          'chair', 'couch', 'cup', 'dining table', 'electronic', 'food', 'furniture', 'indoor', 
                          'kitchen', 'motorcycle', 'outdoor', 'person', 'pizza', 'potted plant', 'sports', 
                          'train', 'truck', 'tv', 'umbrella', 'vehicle']
    return high_freq_concepts

def filter_coco_concepts(original_concepts_list):
    high_freq_concepts = get_coco_concepts()
    filtered_concepts = [c for c in original_concepts_list if c in high_freq_concepts]
    return filtered_concepts


def get_paper_plotting_style():
    """
    Returns the default plotting style dictionary for paper-ready figures.
    Uses serif font and appropriate sizes for publication.
    
    Returns:
        dict: Dictionary of matplotlib rcParams settings
    """
    return {
        "font.family": "DejaVu Serif",
        "font.size": 10,        # match NeurIPS body text
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,    # captions are 9pt
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black"
    }


def apply_paper_plotting_style():
    """
    Applies the default paper plotting style to matplotlib.
    This modifies the global matplotlib settings.
    """
    plt.rcParams.update(get_paper_plotting_style())

        

