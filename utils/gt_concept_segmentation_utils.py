import torch
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import pandas as pd
import random
from collections import defaultdict
import re

import matplotlib.pyplot as plt
from PIL import Image
import pprint

from pycocotools.coco import COCO

import importlib
import utils.general_utils
importlib.reload(utils.general_utils)
import utils.patch_alignment_utils
importlib.reload(utils.patch_alignment_utils)
from utils.general_utils import load_images, retrieve_present_concepts, pad_or_resize_img_tensor, filter_coco_concepts, get_split_df
from utils.patch_alignment_utils import get_image_idx_from_global_patch_idx, get_patch_split_df
from typing import Dict, List, Set

### all-dataset purpose ###

def remap_text_ground_truth_indices(gt_indices: Set[int], dataset_name: str, split: str = 'test', model_input_size=None) -> Set[int]:
    """
    Remap ground truth indices from global token indices to split-specific indices.
    
    For text datasets, ground truth indices are stored as global token indices across
    the entire dataset. However, when we load a specific split (cal/train/test), the
    tokens are renumbered starting from 0. This function maps the global indices to
    split-specific indices.
    
    Args:
        gt_indices: Set of global token indices
        dataset_name: Name of the dataset
        split: Which split to map to ('cal', 'train', or 'test')
        model_input_size: Tuple specifying model input size, e.g., ('text', 'text') for Llama
        
    Returns:
        Set of remapped indices for the specific split
    """
    import glob
    from utils.general_utils import get_split_df
    
    # Load token counts - MUST match the model being used
    if model_input_size and model_input_size[0] == 'text':
        token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
        if not os.path.exists(token_counts_file):
            raise FileNotFoundError(f"Required token counts file not found: {token_counts_file}. "
                                  f"This MUST match the model input size {model_input_size}")
    else:
        # Fallback for legacy code or non-text models
        token_files = glob.glob(f'GT_Samples/{dataset_name}/token_counts_inputsize_*.pt')
        if not token_files:
            raise FileNotFoundError(f"No token counts file found for {dataset_name}")
        token_counts_file = token_files[0]
    token_counts = torch.load(token_counts_file, weights_only=False)
    
    # Convert to tokens per sentence
    tokens_per_sentence = [sum(sent_tokens) if isinstance(sent_tokens, list) else sent_tokens 
                          for sent_tokens in token_counts]
    
    # Get split info - REVERT to original working method
    split_df = get_split_df(dataset_name)
    
    # Calculate cumulative token counts
    cumulative_tokens = [0]
    for i in range(len(tokens_per_sentence)):
        cumulative_tokens.append(cumulative_tokens[-1] + tokens_per_sentence[i])
    
    # Get sentences for the target split
    target_sentences = [i for i in range(len(tokens_per_sentence)) if split_df.get(i) == split]
    
    # Build mapping from global token index to split-specific token index
    global_to_split_idx = {}
    split_token_idx = 0
    
    for sent_idx in target_sentences:
        global_start = cumulative_tokens[sent_idx]
        global_end = cumulative_tokens[sent_idx + 1]
        for global_idx in range(global_start, global_end):
            global_to_split_idx[global_idx] = split_token_idx
            split_token_idx += 1
    
    # Map the indices
    remapped_indices = set()
    for idx in gt_indices:
        if idx in global_to_split_idx:
            remapped_indices.add(global_to_split_idx[idx])
    
    return remapped_indices
def plot_seg_maps(dataset_name, input_image_size=(224, 224), img_idx=-1):
    data_dir = f'Data/{dataset_name}/'
    metadata = pd.read_csv(f'{data_dir}/metadata.csv')
    n_images = len(metadata)
    
    # Select a random image index
    if img_idx < 0:
        img_idx = random.randint(0, n_images - 1)
    img_path = metadata.iloc[img_idx]['image_path']
    image = Image.open(f'{data_dir}/{img_path}')
                       
    # Plot the original image
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(f"Image {img_idx}")
    ax.axis('off')

    # Retrieve the segmentation maps for the selected image
    seg_maps = retrieve_all_concept_segmentations(img_idx, dataset_name)
    
    # Prepare the segmentation maps for plotting
    # We will plot segmentation maps in groups of 3 per row
    num_maps = len(seg_maps)
    num_rows = (num_maps + 2) // 3  # Calculate the number of rows (3 maps per row)

    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    
    # Flatten axs array to make it easier to index
    axs = axs.flatten()
    
    # Resize the image to the input size
    resized_image = np.array(image.resize(input_image_size))
    
    # Loop through each segmentation map and plot it
    for i, (concept_key, seg_map) in enumerate(seg_maps.items()):
        # Create a binary mask where 1 is for the object area, and 0 is for background
        binary_mask = np.array(seg_map).astype(np.uint8)  # Convert to 0 (background) and 1 (object)

        # Create a masked image: set pixels to black where the mask is 0 (background)
        masked_image = np.zeros_like(resized_image)  # Create a black image as the background
        masked_image[binary_mask == 1] = resized_image[binary_mask == 1]  # Keep object pixels

        # Plot the image with the mask applied
        axs[i].imshow(masked_image)
        axs[i].set_title(f"{concept_key}")
        axs[i].axis('off')
    
    # Hide unused axes
    for i in range(num_maps, len(axs)):
        axs[i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    
def retrieve_all_concept_segmentations(img_idx, dataset_name):
    if 'CLEVR' in dataset_name:
        return retrieve_all_concept_segmentations_clevr(img_idx, dataset_name)
    elif dataset_name == 'Coco':
        return retrieve_all_concept_segmentations_coco(img_idx) 
    # elif dataset_name == 'Coco-Cal':
    #     return retrieve_all_concept_segmentations_coco(img_idx, cal=True)
    elif dataset_name == 'Broden-Pascal' or dataset_name == 'Broden-OpenSurfaces':
        return retrieve_all_concept_segmentations_broden(img_idx, dataset_name)
    elif dataset_name == 'Surgery':
        return retrieve_all_concept_segmentations_surgery(img_idx) 
    


def sort_mapping_by_split(all_samples_per_concept, dataset_name, sample_type, model_input_size):
    if sample_type =='patch':
        split_df = get_patch_split_df(dataset_name, model_input_size)
    else:
        split_df = get_split_df(dataset_name)
    
    # Initialize a regular dictionary to map split -> concept -> list of samples
    all_splits_dic = {}

    # Loop through each concept and its samples
    for concept, samples in all_samples_per_concept.items():
        for sample in samples:
            split = split_df.iloc[sample]
            
            # Initialize the dictionaries if they do not exist yet
            if split not in all_splits_dic:
                all_splits_dic[split] = {}
            if concept not in all_splits_dic[split]:
                all_splits_dic[split][concept] = []
            
            # Append the sample to the appropriate place
            all_splits_dic[split][concept].append(sample)
    
    gt_samples_per_concept_train = all_splits_dic['train']
    gt_samples_per_concept_test = all_splits_dic['test']
    gt_samples_per_concept_cal = all_splits_dic['cal']
    
    if sample_type == 'cls':
        sample_type = 'samples'
    torch.save(gt_samples_per_concept_train, f'GT_Samples/{dataset_name}/gt_{sample_type}_per_concept_train_inputsize_{model_input_size}.pt')
    torch.save(gt_samples_per_concept_test, f'GT_Samples/{dataset_name}/gt_{sample_type}_per_concept_test_inputsize_{model_input_size}.pt')
    torch.save(gt_samples_per_concept_cal, f'GT_Samples/{dataset_name}/gt_{sample_type}_per_concept_cal_inputsize_{model_input_size}.pt')
    print(f'train, test, and cal mappings saved :)')
    return gt_samples_per_concept_train, gt_samples_per_concept_test, gt_samples_per_concept_cal
    


def find_closest_to_gt(unsupervised_concepts, metrics, gt_patch_concepts, device):
    gt_metrics = pd.DataFrame()
    # Find the most aligned concept
    alignment_results = {}
    for gt_key, gt_embedding in gt_patch_concepts.items():
        max_similarity = float('-inf')
        best_match = None

        for unsupervised_key, unsupervised_embedding in unsupervised_concepts.items():
            # Compute cosine similarity using PyTorch's cosine_similarity
            similarity = F.cosine_similarity(gt_embedding.unsqueeze(0).to(device), unsupervised_embedding.unsqueeze(0).to(device)).item()

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = unsupervised_key

        alignment_results[gt_key] = (best_match, gt_embedding, max_similarity)
        gt_metrics[gt_key] = metrics[str(best_match)]
    alignment_results = dict(sorted(alignment_results.items(), key=lambda x: x[0]))
    return alignment_results, gt_metrics


### just for patches ###
def sort_patch_embeddings_by_concept(gt_patches_per_concept, embeddings, patch_size=14):
    """
    Maps patch indices to patch embeddings for each concept.

    Args:
        gt_patches_per_concept (dict): A dictionary where keys are concepts and values are lists of (image_idx, patch_idx).
        embeddings (torch.Tensor): A tensor containing the precomputed patch embeddings of shape (N, P, 1024),
            where N is the number of images, P is the number of patches (16x16), and 1024 is the embedding dimension.
        patch_size (int): Size of the patches (default is 14).

    Returns:
        dict: A dictionary where keys are concepts and values are lists of patch embeddings.
    """
    concept_embeddings = defaultdict(list)
    
    for concept, patches in gt_patches_per_concept.items():
        for idx in range(embeddings.shape[0]):
            if idx in patches:
                concept_embeddings[concept].append(embeddings[idx, :])
    return concept_embeddings


def map_concepts_to_image_indices(dataset_name, model_input_size):
    """
    Maps concepts to image indices based on a metadata CSV file.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        defaultdict(list): A dictionary where keys are concepts and values are lists of image indices 
                           that contain the respective concept.
    """
    metadata = pd.read_csv(f'Data/{dataset_name}/metadata.csv')
    concepts = [col for col in metadata.columns if col != "image_path" and col != 'split']
    if dataset_name =='Coco' or dataset_name == 'Coco-Cal':
        concepts = filter_coco_concepts(concepts)
    
    concept_to_images = defaultdict(list)
    for idx, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        for concept in concepts:
            if row[concept] == 1:
                concept_to_images[concept].append(idx)
    sorted_concept_to_images = dict(sorted(concept_to_images.items()))
    
    torch.save(sorted_concept_to_images, f'GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt')
    print(f"Concept to image dic saved to'GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt :)")
    return sorted_concept_to_images


def map_concepts_to_patch_indices(dataset_name, model_input_size, patch_size=14):
    """
    Maps concepts to patch indices based on object masks and metadata.
    """
    metadata = pd.read_csv(f'Data/{dataset_name}/metadata.csv')
    curr_concepts = [col for col in metadata.columns if col not in ['split', 'class', 'image_path']]
    if dataset_name in ['Coco', 'Coco-Cal']:
        curr_concepts = filter_coco_concepts(metadata.columns)
    
    h, w = model_input_size
    patches_per_row = w // patch_size
    concept_patch_indices = defaultdict(list)

    all_segs = None
    if 'Broden' in dataset_name:
        all_segs =  torch.load(f'Data/{dataset_name}/segmentations.pt')
        
    for idx, info in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        active_concepts = [concept for concept in curr_concepts if info[concept] == 1]
        if len(active_concepts) == 0:
            continue

        if dataset_name == 'Coco-Cal':
            img_idx = int(os.path.splitext(os.path.basename(info['image_path']))[0])
            concept_masks = retrieve_all_concept_segmentations(img_idx, dataset_name)
        elif 'Broden' in dataset_name:
            concept_masks = all_segs[idx]
        else:
            concept_masks = retrieve_all_concept_segmentations(idx, dataset_name)

        for concept in active_concepts:
            if concept not in concept_masks:
                print(f"[WARN] Concept '{concept}' not found in masks for image {idx}")
                continue

            concept_mask = concept_masks[concept]
            resized_concept_mask = pad_or_resize_img_tensor(concept_mask, model_input_size, is_mask=True)

            if resized_concept_mask.shape != torch.Size([h, w]):
                print(f"[ERROR] Resized mask shape {resized_concept_mask.shape} doesn't match model_input_size {model_input_size}")
                continue

            found = False
            for patch_idx in range(patches_per_row ** 2):
                row_idx = patch_idx // patches_per_row
                col_idx = patch_idx % patches_per_row
                i_start, j_start = row_idx * patch_size, col_idx * patch_size

                mask_window = resized_concept_mask[i_start:i_start + patch_size, j_start:j_start + patch_size]
                if torch.any(mask_window == 1):  # If patch contains part of the object
                    global_patch_index = (idx * (patches_per_row ** 2)) + patch_idx
                    concept_patch_indices[concept].append(global_patch_index)
                    found = True

            if not found:
                print(f"[INFO] Concept '{concept}' not found in any patch for image {idx}")
                # Optional: Plot mask to debug
                plt.figure()
                plt.title(f"No patches found for concept: {concept}, image: {idx}")
                plt.imshow(resized_concept_mask.cpu(), cmap='gray')
                plt.savefig(f'debug_no_patch_{dataset_name}_{idx}_{concept}.png')
                plt.close()

    sorted_concept_patch_indices = dict(sorted(concept_patch_indices.items()))
    os.makedirs(f'GT_Samples/{dataset_name}', exist_ok=True)
    torch.save(sorted_concept_patch_indices, f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt')
    print(f"[DONE] Saved patch indices to GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt")
    
    return sorted_concept_patch_indices


# def compute_attention_masks(all_text_samples, processor, dataset_name, model_input_size):
#     tokens_list = []
#     relevant_tokens_list = []
#     token_counts_per_sentence = []

#     for text_samples in tqdm(all_text_samples):
#         inputs = processor(
#                     None,
#                     text_samples,
#                     add_special_tokens=False,
#                     padding=True,
#                     token=os.environ.get("HF_TOKEN"),
#                     return_tensors="pt"
#                 )

#         tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
#         print("tokens:", tokens)
#         tokens_list.append(tokens)

#         # Split text into words (this assumes each sample is one sentence!)
#         word_list = text_samples.split()
#         print("word list:", word_list)
#         token_counts = []
#         for word in word_list:
#             print(processor.tokenizer.tokenize(word))
#             n_tokens = len(processor.tokenizer.tokenize(word))
#             token_counts.append(n_tokens)
#         print("token counts:", token_counts)
#         token_counts_per_sentence.append(token_counts)

#         for token in tokens:
#             if token == '<|begin_of_text|>':
#                 relevant_tokens_list.append(0)
#             else:
#                 relevant_tokens_list.append(1)

#     relevant_tokens = torch.tensor(relevant_tokens_list)

#     # Save all outputs
#     torch.save(tokens_list, f'GT_Samples/{dataset_name}/tokens.pt')
#     torch.save(token_counts_per_sentence, f'GT_Samples/{dataset_name}/token_counts.pt')
#     torch.save(relevant_tokens, f'GT_Samples/{dataset_name}/patches_w_image_mask_inputsize_{model_input_size}.pt')

#     return tokens_list, token_counts_per_sentence, relevant_tokens


# def compute_attention_masks(all_text_samples, processor, dataset_name, model_input_size):
#     tokens_list = []
#     relevant_tokens_list = []
#     token_counts_per_sentence = []

#     for text_samples in tqdm(all_text_samples):
#         inputs = processor(
#                     None,
#                     text_samples,
#                     add_special_tokens=False,
#                     padding=False,
#                     token=os.environ.get("HF_TOKEN"),
#                     return_tensors="pt"
#                 )

#         tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
#         tokens_list.append(tokens)

#         # Split text into words (this assumes each sample is one sentence!)
#         word_list = text_samples.split()
#         token_counts = []
#         for word in word_list:
#             n_tokens = len(processor.tokenizer.tokenize(word))
#             token_counts.append(n_tokens)
#         token_counts_per_sentence.append(token_counts)

#         for token in tokens:
#             if token == '<|begin_of_text|>':
#                 relevant_tokens_list.append(0)
#             else:
#                 relevant_tokens_list.append(1)

#     relevant_tokens = torch.tensor(relevant_tokens_list)

#     # Save all outputs
#     torch.save(tokens_list, f'GT_Samples/{dataset_name}/tokens.pt')
#     torch.save(token_counts_per_sentence, f'GT_Samples/{dataset_name}/token_counts.pt')
#     torch.save(relevant_tokens, f'GT_Samples/{dataset_name}/patches_w_image_mask_inputsize_{model_input_size}.pt')

    return tokens_list, token_counts_per_sentence, relevant_tokens


def compute_attention_masks(all_text_samples, processor, dataset_name, model_input_size):
    tokens_list = []
    relevant_tokens_list = []
    token_counts_per_sentence = []
    word_to_token_map_list = []

    # Determine if processor has a tokenizer attribute (Llama) or is itself a tokenizer (Mistral)
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    for sentence_id, text in tqdm(enumerate(all_text_samples), total=len(all_text_samples), desc="Tokenizing"):
        # Tokenize using *exactly the same settings* as get_llama_text_patch_embeddings
        inputs = tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        token_ids = inputs["input_ids"].squeeze().tolist()
        offsets = inputs["offset_mapping"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        tokens_list.append(tokens)

        for tok in tokens:
            relevant_tokens_list.append(0 if tok == "<|begin_of_text|>" else 1)

        # Word spans from raw text
        words = list(re.finditer(r"\S+", text))
        word_spans = [(m.start(), m.end()) for m in words]

        word_to_token_ids = defaultdict(list)

        # First pass: match tokens to words based on character offsets
        for word_idx, (w_start, w_end) in enumerate(word_spans):
            matched_token_ids = []
            for token_idx, (t_start, t_end) in enumerate(offsets):
                if t_end > w_start and t_start < w_end:
                    matched_token_ids.append(token_idx)
            word_to_token_ids[word_idx] = matched_token_ids
        
        # Second pass: assign unmatched tokens to previous word
        matched_tokens = set()
        for word_tokens in word_to_token_ids.values():
            matched_tokens.update(word_tokens)
        
        for token_idx, token in enumerate(tokens):
            if token_idx not in matched_tokens:
                # Find the nearest previous matched token and assign to its word
                prev_word = None
                for prev_idx in range(token_idx - 1, -1, -1):  # Go backwards
                    if prev_idx in matched_tokens:
                        # Find which word this previous token belongs to
                        for word_idx, word_tokens in word_to_token_ids.items():
                            if prev_idx in word_tokens:
                                prev_word = word_idx
                                break
                        break
                
                if prev_word is not None:
                    word_to_token_ids[prev_word].append(token_idx)
                    matched_tokens.add(token_idx)  # Update matched tokens
                elif len(word_to_token_ids) > 0:
                    # If no previous word found, assign to first word
                    first_word = min(word_to_token_ids.keys())
                    word_to_token_ids[first_word].append(token_idx)
                    matched_tokens.add(token_idx)
        
        # Count tokens per word
        token_counts = [len(word_to_token_ids[i]) for i in range(len(word_spans))]

        token_counts_per_sentence.append(token_counts)
        word_to_token_map_list.append(dict(word_to_token_ids))

    relevant_tokens = torch.tensor(relevant_tokens_list)
    out_dir = f'GT_Samples/{dataset_name}'
    os.makedirs(out_dir, exist_ok=True)

    torch.save(tokens_list, f'{out_dir}/tokens_inputsize_{model_input_size}.pt')
    torch.save(token_counts_per_sentence, f'{out_dir}/token_counts_inputsize_{model_input_size}.pt')
    torch.save(word_to_token_map_list, f'{out_dir}/word_to_token_map_inputsize_{model_input_size}.pt')
    torch.save(relevant_tokens, f'{out_dir}/patches_w_image_mask_inputsize_{model_input_size}.pt')

    print(f"\n✅ Saved tokenized outputs to {out_dir}")
    return tokens_list, token_counts_per_sentence, word_to_token_map_list, relevant_tokens



def map_sentences_to_concept_gt_jailbreak(dataset_name, model_input_size):
    concept_to_sentences = defaultdict(list)
    concept_to_sentences_train = defaultdict(list)
    concept_to_sentences_test = defaultdict(list)
    metadata = pd.read_csv(f'Data/{dataset_name}/metadata.csv')
    for idx, row in metadata.iterrows():
        class_as_list = row['class'].split()
        if class_as_list[0] == 'benign': #might need to change this if you change how you get concepts
            continue
        concept = " ".join(class_as_list[1:])
        concept_to_sentences[concept].append(idx)
        if row['split'] == 'train':
            concept_to_sentences_train[concept].append(idx)
        elif row['split'] == 'test':
            concept_to_sentences_test[concept].append(idx)
        
    
    sorted_concept_to_sentences = dict(sorted(concept_to_sentences.items()))
    sorted_concept_to_sentences_train = dict(sorted(concept_to_sentences_train.items()))
    sorted_concept_to_sentences_test = dict(sorted(concept_to_sentences_test.items()))
    torch.save(sorted_concept_to_sentences, f'GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt')
    torch.save(sorted_concept_to_sentences_train, f'GT_Samples/{dataset_name}/gt_samples_per_concept_train_inputsize_{model_input_size}.pt')
    torch.save(sorted_concept_to_sentences_test, f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt')
    print(f"Concept to sentence dic saved to 'GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt'")
    return sorted_concept_to_sentences, sorted_concept_to_sentences_train, sorted_concept_to_sentences_test


# def map_sentence_to_concept_gt(dataset_name, model_input_size, one_indexed):
#     """
#     Constructs token-level and sentence-level concept GT from text.
#     - Token-level mappings come from word_level_sentiment.csv + word_to_token_map.pt
#     - Sentence-level mappings come from sentence_level_sentiment.csv
#     - Uses get_split_df() to assign splits
#     """

#     # Load data
#     word_df = pd.read_csv(f'../Data/{dataset_name}/word_level_sentiment.csv')
#     sentence_df = pd.read_csv(f'../Data/{dataset_name}/sentence_level_sentiment.csv')
#     word_to_token_map_list = torch.load(f'GT_Samples/{dataset_name}/word_to_token_map.pt')
#     tokens_list = torch.load(f'GT_Samples/{dataset_name}/tokens.pt')  # needed for global index offset
#     split_df = get_split_df(dataset_name)

#     # Infer concept columns
#     word_concepts = [col for col in word_df.columns if col not in ['sentence_id', 'word', 'sentiment_label']]
#     sentence_concepts = [col for col in sentence_df.columns if col not in ['sentence_id', 'sentence', 'sentiment_label']]

#     # Token-level concept tracking
#     token_indices = defaultdict(list)
#     token_indices_train = defaultdict(list)
#     token_indices_test = defaultdict(list)
#     token_indices_cal = defaultdict(list)

#     # Sentence-level concept tracking
#     sentence_indices = defaultdict(set)
#     sentence_indices_train = defaultdict(set)
#     sentence_indices_test = defaultdict(set)
#     sentence_indices_cal = defaultdict(set)

#     word_groups = word_df.groupby("sentence_id")
#     global_token_ptr = 0

#     for sent_idx, word_to_token_map in tqdm(enumerate(word_to_token_map_list), total=len(word_to_token_map_list), desc="Mapping token-level concepts"):
#         split = split_df.loc[sent_idx]
#         sentence_id = sent_idx + 1

#         if sentence_id not in word_groups.groups:
#             raise ValueError(f"Missing sentence_id {sentence_id} in word_df")

#         sentence_words = word_groups.get_group(sentence_id)

#         # print(f"\n--- Sentence {sent_idx} (Split: {split}) ---")
#         # pprint.pprint(word_to_token_map)

#         for word_idx, row in enumerate(sentence_words.itertuples()):
#             word_text = getattr(row, 'word')
#             # print(f"\n  Word {word_idx} ('{word_text}')")

#             for concept in word_concepts:
#                 if getattr(row, concept) == 1:
#                     local_token_ids = word_to_token_map.get(word_idx, [])
#                     token_ids = [global_token_ptr + t for t in local_token_ids]

#                     token_indices[concept].extend(token_ids)
#                     if split == 'train':
#                         token_indices_train[concept].extend(token_ids)
#                     elif split == 'test':
#                         token_indices_test[concept].extend(token_ids)
#                     elif split == 'cal':
#                         token_indices_cal[concept].extend(token_ids)

#         # ✅ Update global index using length of this sentence's token list
#         n_tokens_in_sentence = len(tokens_list[sent_idx])
#         global_token_ptr += n_tokens_in_sentence

#     # === Sentence-level mappings from sentence-level labels ===
#     for row_idx, row in sentence_df.iterrows():
#         split = split_df.loc[row_idx]

#         for concept in sentence_concepts:
#             if row[concept] == 1:
#                 sentence_indices[concept].add(row_idx)
#                 if split == 'train':
#                     sentence_indices_train[concept].add(row_idx)
#                 elif split == 'test':
#                     sentence_indices_test[concept].add(row_idx)
#                 elif split == 'cal':
#                     sentence_indices_cal[concept].add(row_idx)

#     # === Save all outputs ===
#     out_dir = f'GT_Samples/{dataset_name}'
#     os.makedirs(out_dir, exist_ok=True)

#     def save_and_sort(d, name, prefix):
#         sorted_d = dict(sorted((k, sorted(v)) for k, v in d.items()))
#         torch.save(sorted_d, f"{out_dir}/{prefix}{name}_inputsize_{model_input_size}.pt")
#         return sorted_d

#     d_all = save_and_sort(token_indices, "", prefix='gt_patches_per_concept')
#     d_train = save_and_sort(token_indices_train, "_train", prefix='gt_patch_per_concept')
#     d_test = save_and_sort(token_indices_test, "_test", prefix='gt_patch_per_concept')
#     d_cal = save_and_sort(token_indices_cal, "_cal", prefix='gt_patch_per_concept')

#     s_all = save_and_sort(sentence_indices, "", prefix='gt_samples_per_concept')
#     s_train = save_and_sort(sentence_indices_train, "_train", prefix='gt_samples_per_concept')
#     s_test = save_and_sort(sentence_indices_test, "_test", prefix='gt_samples_per_concept')
#     s_cal = save_and_sort(sentence_indices_cal, "_cal", prefix='gt_samples_per_concept')

#     print(f"\n✅ Saved token-level and sentence-level concept indices to {out_dir}")
#     return {
#         "token_indices": (d_all, d_train, d_test, d_cal),
#         "sentence_level_indices": (s_all, s_train, s_test, s_cal),
#     }


def map_sentence_to_concept_gt(dataset_name, model_input_size, one_indexed):
    """
    Constructs token-level and sentence-level concept GT from text.
    - Token-level mappings come from word_level_sentiment.csv + word_to_token_map.pt
    - Sentence-level mappings come from sentence_level_sentiment.csv
    - Uses get_split_df() to assign splits
    """

    # Load data
    if 'Sarcasm' in dataset_name:
        word_df = pd.read_csv(f'Data/{dataset_name}/word_level_sarcasm.csv')
        sentence_df = pd.read_csv(f'Data/{dataset_name}/paragraph_level_sarcasm.csv')
    elif 'Emotion' in dataset_name:
        word_df = pd.read_csv(f'Data/{dataset_name}/word_level_emotion.csv')
        sentence_df = pd.read_csv(f'Data/{dataset_name}/paragraph_level_emotion.csv')
    else:
        word_df = pd.read_csv(f'Data/{dataset_name}/word_level_sentiment.csv')
        sentence_df = pd.read_csv(f'Data/{dataset_name}/sentence_level_sentiment.csv')
    word_to_token_map_list = torch.load(f'GT_Samples/{dataset_name}/word_to_token_map_inputsize_{model_input_size}.pt', weights_only=False)
    tokens_list = torch.load(f'GT_Samples/{dataset_name}/tokens_inputsize_{model_input_size}.pt', weights_only=False)  # needed for global index offset
    split_df = get_split_df(dataset_name)

    # Infer concept columns
    word_concepts = [col for col in word_df.columns if col not in ['paragraph_id', 'sentence_id', 'word', 'sentiment_label']]
    sentence_concepts = [col for col in sentence_df.columns if col not in ['paragraph_id', 'sentence_id', 'sentence', 'sentiment_label']]

    # Token-level concept tracking
    token_indices = defaultdict(list)
    token_indices_train = defaultdict(list)
    token_indices_test = defaultdict(list)
    token_indices_cal = defaultdict(list)

    # Sentence-level concept tracking
    sentence_indices = defaultdict(set)
    sentence_indices_train = defaultdict(set)
    sentence_indices_test = defaultdict(set)
    sentence_indices_cal = defaultdict(set)

    if 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        word_groups = word_df.groupby("paragraph_id")
    else:
        word_groups = word_df.groupby("sentence_id")
    global_token_ptr = 0

    for sent_idx, word_to_token_map in tqdm(enumerate(word_to_token_map_list), total=len(word_to_token_map_list), desc="Mapping token-level concepts"):
        split = split_df.loc[sent_idx]
        sentence_id = sent_idx + 1

        if sentence_id not in word_groups.groups:
            print(f"Warning: Missing sentence_id {sentence_id} in word_df, skipping...")
            continue

        sentence_words = word_groups.get_group(sentence_id)

        # print(f"\n--- Sentence {sent_idx} (Split: {split}) ---")
        # pprint.pprint(word_to_token_map)

        for word_idx, row in enumerate(sentence_words.itertuples()):
            word_text = getattr(row, 'word')
            # print(f"\n  Word {word_idx} ('{word_text}')")

            for concept in word_concepts:
                if getattr(row, concept) == 1:
                    local_token_ids = word_to_token_map.get(word_idx, [])
                    token_ids = [global_token_ptr + t for t in local_token_ids]

                    token_indices[concept].extend(token_ids)
                    if split == 'train':
                        token_indices_train[concept].extend(token_ids)
                    elif split == 'test':
                        token_indices_test[concept].extend(token_ids)
                    elif split == 'cal':
                        token_indices_cal[concept].extend(token_ids)

        # ✅ Update global index using length of this sentence's token list
        n_tokens_in_sentence = len(tokens_list[sent_idx])
        global_token_ptr += n_tokens_in_sentence

    # === Sentence-level mappings from sentence-level labels ===
    for row_idx, row in sentence_df.iterrows():
        split = split_df.loc[row_idx]

        for concept in sentence_concepts:
            if row[concept] == 1:
                sentence_indices[concept].add(row_idx)
                if split == 'train':
                    sentence_indices_train[concept].add(row_idx)
                elif split == 'test':
                    sentence_indices_test[concept].add(row_idx)
                elif split == 'cal':
                    sentence_indices_cal[concept].add(row_idx)

    # === Save all outputs ===
    out_dir = f'GT_Samples/{dataset_name}'
    os.makedirs(out_dir, exist_ok=True)

    def save_and_sort(d, name, prefix):
        sorted_d = dict(sorted((k, sorted(v)) for k, v in d.items()))
        torch.save(sorted_d, f"{out_dir}/{prefix}{name}_inputsize_{model_input_size}.pt")
        return sorted_d

    d_all = save_and_sort(token_indices, "", prefix='gt_patches_per_concept')
    d_train = save_and_sort(token_indices_train, "_train", prefix='gt_patch_per_concept')
    d_test = save_and_sort(token_indices_test, "_test", prefix='gt_patch_per_concept')
    d_cal = save_and_sort(token_indices_cal, "_cal", prefix='gt_patch_per_concept')

    s_all = save_and_sort(sentence_indices, "", prefix='gt_samples_per_concept')
    s_train = save_and_sort(sentence_indices_train, "_train", prefix='gt_samples_per_concept')
    s_test = save_and_sort(sentence_indices_test, "_test", prefix='gt_samples_per_concept')
    s_cal = save_and_sort(sentence_indices_cal, "_cal", prefix='gt_samples_per_concept')

    print(f"\n✅ Saved token-level and sentence-level concept indices to {out_dir}")
    return {
        "token_indices": (d_all, d_train, d_test, d_cal),
        "sentence_level_indices": (s_all, s_train, s_test, s_cal),
    }


def print_paragraph_or_sentence_gt_examples(dataset_name, model_input_size, num_examples=5):
    """
    Print GT examples for a given dataset at the sentence or paragraph level.

    Args:
        dataset_name (str): Dataset folder name (e.g., 'Sarcasm').
        model_input_size (str): Input size token for file naming (e.g., '224').
        level (str): Either 'sentence' or 'paragraph'.
        num_examples (int): Number of examples to show per concept.
    """
    if dataset_name == 'Stanford-Tree-Bank':
        level = 'sentence'
    else:
        level = 'paragraph'
    
    gt_dir = f"GT_Samples/{dataset_name}"
    data_dir = f"Data/{dataset_name}"

    # Load GT dictionaries
    concept_gt = torch.load(f"{gt_dir}/gt_samples_per_concept_inputsize_{model_input_size}.pt", weights_only=False)
    token_gt = torch.load(f"{gt_dir}/gt_patches_per_concept_inputsize_{model_input_size}.pt", weights_only=False)
    tokenized_sentences = torch.load(f"{gt_dir}/tokens_inputsize_{model_input_size}.pt", weights_only=False)

    # Load sentences or paragraphs
    if 'Sarcasm' in dataset_name:
        paragraph_df = pd.read_csv(f"{data_dir}/paragraph_level_sarcasm.csv")
        texts = paragraph_df["text"].tolist()
    elif 'Emotion' in dataset_name:
        paragraph_df = pd.read_csv(f"{data_dir}/paragraph_level_emotion.csv")
        texts = paragraph_df["text"].tolist()
    else:
        text_df = pd.read_csv(f"{data_dir}/sentence_level_sentiment.csv")
        texts = text_df["sentence"].tolist()

    print(f"\n=== {level.capitalize()}-Level GT Examples ===")
    for concept, indices in list(concept_gt.items())[:num_examples]:
        print(f"\nConcept: {concept}")
        for i in sorted(indices)[:num_examples]:
            if i < len(texts):
                print(f"[{i}] {texts[i]}")

    print("\n=== Token-Level GT Examples ===")
    for concept, token_indices in list(token_gt.items())[:num_examples]:
        print(f"\nConcept: {concept}")
        flat_tokens = [tok for sent in tokenized_sentences for tok in sent]
        for tok_id in sorted(token_indices)[:num_examples]:
            if tok_id < len(flat_tokens):
                print(f"[{tok_id}] Token: {flat_tokens[tok_id]}")
        

def map_concepts_to_token_indices(dataset_name, tokens_list, relevant_tokens, model_input_size):
    """
    Maps concepts to image indices based on a metadata CSV file.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        defaultdict(list): A dictionary where keys are concepts and values are lists of image indices 
                           that contain the respective concept.
    """
    metadata = pd.read_csv(f'Data/{dataset_name}/metadata.csv')
    
    
    concepts = [col for col in metadata.columns if col != "sample_filename"]
    concept_to_tokens = defaultdict(list)
    
    overall_idx = 0
    for sentence_idx, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        for token in tokens_list[sentence_idx]:
            for concept in concepts:
                if row[concept] == 1 and relevant_tokens[overall_idx] == 1: #if concept is present and not special token
                    concept_to_tokens[concept].append(overall_idx)
            overall_idx +=1
            
    sorted_concept_to_tokens = dict(sorted(concept_to_tokens.items()))
    
    torch.save(sorted_concept_to_tokens, f'GT_Samples/{dataset_name}/gt_patch_per_concept_inputsize_{model_input_size}.pt')
    print(f"Concept to token dic saved to 'GT_Samples/{dataset_name}/gt_patch_per_concept_inputsize_{model_input_size}.pt'")
    return sorted_concept_to_tokens
    


### CLEVR ###
def extract_CLEVR_concept_pixels_batch(images, gray_tolerance=20):
    """
    Extract foreground masks for a batch of images by identifying non-grayscale pixels.

    Args:
        images (torch.Tensor): Batch of images with shape (N, H, W, C), where N is the batch size, 
                               H and W are image height and width, and C is the number of channels (3 for RGB).
        gray_tolerance (int): Tolerance value for determining if a pixel is grayscale. 
                              A pixel is considered grayscale if the absolute difference 
                              between any two color channels (R, G, B) is less than or equal to this value.

    Returns:
        torch.Tensor: Foreground masks with shape (N, H, W), where each mask is a binary image (1 for foreground, 0 for background).
    """
    r, g, b = images[:, :, :, 0], images[:, :, :, 1], images[:, :, :, 2]
    is_gray = (torch.abs(r - g) <= gray_tolerance) & \
              (torch.abs(g - b) <= gray_tolerance) & \
              (torch.abs(r - b) <= gray_tolerance)
    is_foreground = ~is_gray
    return is_foreground.to(dtype=torch.float32)

def compute_all_concept_masks_clevr(dataset_name='CLEVR', batch_size=32):
    """
    Compute foreground concept masks for a dataset of images in batches. Optionally, plot a subset of images with masks.

    Args:
        images (list[PIL.Image.Image]): List of images to process, where each image is a PIL Image.
        n_to_plot (int): Number of example images to plot with their corresponding masks (default: 5).
        batch_size (int): Number of images to process in each batch (default: 32).
        save_path (str): Where to save masks.

    Returns:
        torch.Tensor: A tensor of foreground masks for all images with shape (N, 224, 224), where N is the number of images.
    """
    images, _, _ = load_images(dataset_name=dataset_name)
    
    # Initialize an empty tensor for object masks
    num_images = len(images)
    object_masks = []
    
    # Convert PIL images to tensors
    print("Converting images to tensors...")
    image_tensors = [torch.tensor(np.array(img), dtype=torch.float32) for img in images]
    image_tensors = torch.stack(image_tensors)  # Shape: (N, H, W, C)
    
    # Process in batches
    num_batches = (len(images) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="batch"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_images)
        batch_images = image_tensors[batch_start:batch_end]  # Shape: (B, H, W, C)

        # Extract masks
        masks = extract_CLEVR_concept_pixels_batch(batch_images, gray_tolerance=20).cpu()  # Shape: (B, H, W)

        # # pad masks to match the model input size
        # padded_masks = pad_tensors_to_size(masks, input_model_size)
        for i in range(masks.shape[0]):
            object_masks.append(masks[i, :, :])
    
    #save concepts
    torch.save(object_masks, f'Data/{dataset_name}/object_segmentations.pt')
    print(f'Masks saved to Data/{dataset_name}/object_segmentations.pt')

    
def retrieve_all_concept_segmentations_clevr(img_idx, dataset_name,n_attributes=2):
    object_masks_all_images =  torch.load(f'Data/{dataset_name}/object_segmentations.pt')
    object_mask = object_masks_all_images[img_idx]
    present_concepts = retrieve_present_concepts(img_idx, dataset_name)
    object_masks = {}
    for concept in present_concepts:
        object_masks[concept] = object_mask
    return object_masks


#### Broden #####
def retrieve_all_concept_segmentations_broden(img_idx, dataset_name):
    object_masks_all_images =  torch.load(f'Data/{dataset_name}/segmentations.pt')
    return object_masks_all_images[img_idx]


    
#### Coco #####
def retrieve_all_concept_segmentations_coco(img_idx, show_debug=False):
    # Load stacked metadata file
    metadata = pd.read_csv(f'Data/Coco/metadata.csv')  # assumes cal + val stacked
    row = metadata.loc[img_idx]
    cal = row['split'] == 'cal'
    image_filename = os.path.basename(row['image_path'])  # e.g., "000000123456.jpg"

    # Choose appropriate annotation file and image folder
    ann_file = (
        'Data/Coco-Cal/annotations/instances_train2017.json'
        if cal else 'Data/Coco/instances_val2017.json'
    )
    image_folder = (
        'Data/Coco-Cal/train2017'
        if cal else 'Data/Coco/val2017'
    )
    image_path = os.path.join(image_folder, image_filename)

    # Initialize COCO API
    coco = COCO(ann_file)

    # Find matching COCO image ID by filename
    img_id = None
    for img_info in coco.loadImgs(coco.getImgIds()):
        if img_info['file_name'] == image_filename:
            img_id = img_info['id']
            break

    if img_id is None:
        raise ValueError(f"Image filename {image_filename} not found in COCO annotations.")

    # Load image metadata
    img_metadata = coco.loadImgs(img_id)[0]

    # Get annotation IDs and load annotations
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    # Load category metadata
    cats = coco.loadCats(coco.getCatIds())
    category_name_to_supercategory = {cat['name']: cat['supercategory'] for cat in cats}
    category_id_to_name = {cat['id']: cat['name'] for cat in cats}

    # Segmentation masks per concept
    concept_seg_maps = {}

    for ann in anns:
        category_id = ann['category_id']
        category_name = category_id_to_name[category_id]
        supercategory_name = category_name_to_supercategory[category_name]

        seg_map = coco.annToMask(ann)

        # Merge by category
        if category_name in concept_seg_maps:
            concept_seg_maps[category_name] = np.logical_or(concept_seg_maps[category_name], seg_map)
        else:
            concept_seg_maps[category_name] = seg_map

        # Merge by supercategory
        if supercategory_name in concept_seg_maps:
            concept_seg_maps[supercategory_name] = np.logical_or(concept_seg_maps[supercategory_name], seg_map)
        else:
            concept_seg_maps[supercategory_name] = seg_map

    # ========== DEBUGGING SECTION ==========
    if show_debug:
        print(f"[DEBUG] Image index: {img_idx}")
        print(f"[DEBUG] Image file: {image_filename}")
        print(f"[DEBUG] Image ID in COCO: {img_id}")
        print(f"[DEBUG] Found concepts: {list(concept_seg_maps.keys())}")

        # Load original image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Plot original image and overlay first concept mask
        if concept_seg_maps:
            first_concept = list(concept_seg_maps.keys())[0]
            mask = concept_seg_maps[first_concept]

            plt.figure(figsize=(10, 5))
            plt.imshow(image_np)
            plt.imshow(mask, alpha=0.5, cmap='jet')
            plt.title(f"Overlay of concept: {first_concept}")
            patch = mpatches.Patch(color='red', label='Segmentation overlay')
            plt.legend(handles=[patch])
            plt.axis('off')
            plt.show()
        else:
            print("[DEBUG] No masks found for this image.")

    return concept_seg_maps


# def retrieve_all_concept_segmentations_coco(img_idx):
#     metadata = pd.read_csv(f'../Data/Coco/metadata.csv')
#     if metadata.loc[img_idx, 'split'] == 'cal':
#         cal = True
#     else:
#         cal = False
    
#     # Path to COCO annotations file
#     if cal:
#         ann_file = '../Data/Coco-Cal/annotations/instances_train2017.json'
#     else:
#         ann_file = '../Data/Coco/instances_val2017.json'

#     # Initialize COCO API
#     coco = COCO(ann_file)

#     if not cal:
#         # Get the image ID using the index
#         img_ids = coco.getImgIds()
#         img_id = img_ids[img_idx]
#     else:
#         img_id = img_idx

#     # Load the image metadata
#     img_metadata = coco.loadImgs(img_id)[0]

#     # Get the annotations (masks) for this image
#     ann_ids = coco.getAnnIds(imgIds=img_metadata['id'], iscrowd=None)
#     anns = coco.loadAnns(ann_ids)

#     # Get all categories and supercategories
#     cats = coco.loadCats(coco.getCatIds())
#     category_name_to_supercategory = {cat['name']: cat['supercategory'] for cat in cats}

#     # Initialize a dictionary to store segmentation maps by concept
#     concept_seg_maps = {}

#     # Loop through annotations and retrieve masks for each category/supercategory present in the image
#     for ann in anns:
#         category_id = ann['category_id']
#         category_name = coco.loadCats(category_id)[0]['name']
#         supercategory_name = category_name_to_supercategory[category_name]

#         # Get the mask for this annotation and add it to both the category and supercategory lists
#         seg_map = coco.annToMask(ann)
        
#         # Resize the segmentation map to the input image size
#         # seg_map_pil = Image.fromarray(seg_map)  # Convert to PIL image for resizing
#         # seg_map_resized = seg_map_pil.resize(input_image_size, Image.NEAREST)  # Resize to input size

#         # Convert back to NumPy array after resizing
#         # resized_mask = np.array(seg_map_resized)

#         # Combine masks for the same category
#         if category_name in concept_seg_maps:
#             concept_seg_maps[category_name] = np.logical_or(concept_seg_maps[category_name], seg_map)
#         else:
#             concept_seg_maps[category_name] = seg_map

#         # Combine masks for the same supercategory
#         if supercategory_name in concept_seg_maps:
#             concept_seg_maps[supercategory_name] = np.logical_or(concept_seg_maps[supercategory_name], seg_map)
#         else:
#             concept_seg_maps[supercategory_name] = seg_map


#     # If no masks were found for a concept, it will not be included in the dictionary
#     return concept_seg_maps


###### Surgery #######
def retrieve_all_concept_segmentations_surgery(img_idx, input_image_size=(224, 224)):
    organs = {0: 'background', 1: 'liver', 2: 'gallbladder', 3: 'hepatocystic_triangle'}
    all_segmentations = torch.load('Data/Surgery/segmentations.pt')
    img_segmentations = all_segmentations[img_idx]

    concept_segmentations = {}
    for organ_num, organ in organs.items():
        # Create a binary mask where the organ is present (1s where organ_num is found, else 0s)
        organ_seg = (img_segmentations == organ_num).to(torch.uint8)

        # Resize to input_image_size if needed
        if organ_seg.shape != input_image_size:
            organ_seg = F.interpolate(organ_seg.unsqueeze(0).unsqueeze(0).float(), size=input_image_size, mode="nearest").squeeze(0).squeeze(0).to(torch.uint8)
            
        concept_segmentations[organ] = organ_seg

    return concept_segmentations
    
     
    