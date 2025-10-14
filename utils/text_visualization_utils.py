import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from itertools import combinations
import seaborn as sns
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import io
import base64
import numpy as np
from utils.general_utils import get_split_df
from itertools import chain
from collections import defaultdict

#### Computations ####
# def flatten_token_list(tokens_list):
#     """
#     Flattens a nested list of tokens into a single list.

#     Args:
#         tokens_list (List[List[str]]): A list of sentences, where each sentence is a list of tokens.

#     Returns:
#         List[str]: A flat list containing all tokens in order.
#     """
#     flat_tokens_list = []
#     for token_list in tokens_list:
#         for token in token_list:
#             flat_tokens_list.append(token)
#     return flat_tokens_list
def flatten_token_list(tokens_list):
    """
    Efficiently flattens a nested list of tokens into a single list.

    Args:
        tokens_list (List[List[str]]): A list of sentences, where each sentence is a list of tokens.

    Returns:
        List[str]: A flat list containing all tokens in order.
    """
    return list(chain.from_iterable(tokens_list))



def get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list):
    # Step 1: Determine the starting index of the sentence in the flat list
    start_idx = sum(len(tokens) for tokens in tokens_list[:sentence_idx])
    end_idx = start_idx + len(tokens_list[sentence_idx])
    return start_idx, end_idx

def get_sent_idx_from_global_token_idx(global_token_idx, tokens_list):
    """
    Get the sentence index from a global token index
    
    Args:
        global_token_idx: The index of a token in the flattened list of all tokens
        tokens_list: List of lists where each inner list contains tokens for a sentence
        
    Returns:
        tuple: (sentence_idx, token_idx_within_sentence)
    """
    current_idx = 0
    for sent_idx, tokens in enumerate(tokens_list):
        if current_idx <= global_token_idx < current_idx + len(tokens):
            token_idx_within_sentence = global_token_idx - current_idx
            return sent_idx, token_idx_within_sentence
        current_idx += len(tokens)
    
    raise ValueError(f"Global token index {global_token_idx} out of bounds")

# # Extract the tokens from the sentence
# sent_idx = 15
# start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sent_idx, tokens_list)
# tokens = tokens_list[sent_idx]

# token_indices = [i for i, token in enumerate(flattened_tokens_list[start_idx:end_idx])]
# print(f"Token indices: {token_indices}")
# # gt_tokens = sorted([(idx, flattened_tokens_list[idx+start_idx]) for idx in gt_samples_per_concept['bird'] if idx >= start_idx and idx < end_idx], key=lambda x: x[0])
# # print(gt_tokens)


def remove_leading_token(token_list):
    """
    For GPT-based tokenizers, leading 'Ġ' indicates a space.
    For RoBERTa tokenizers, this might be a different pattern.
    """
    cleaned_tokens = []
    for token in token_list:
        if token.startswith('Ġ'):
            cleaned_tokens.append(token[1:])  # Strip the leading Ġ
        else:
            cleaned_tokens.append(token)
    return cleaned_tokens

#### Conversions ####
def get_color_for_sim(sim, vmin, vmax, cmap):
    """Convert similarity score to RGB color using the colormap."""
    norm = (sim - vmin) / (vmax - vmin)
    norm = max(0.0, min(1.0, norm))  # Clamp to [0.0, 1.0] - ensure float!
    rgba = cmap(float(norm))  # Explicitly convert to float for colormap
    return f"rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})"

def make_colorbar_image(vmin, vmax, cmap_name="coolwarm"):
    """Create a colorbar image and return as base64 string."""
    fig, ax = plt.subplots(figsize=(8, 0.5))
    
    # Create colorbar
    cmap = plt.cm.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label('Similarity Score')
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    return f'<img src="data:image/png;base64,{img_str}" style="width:400px; margin-top:10px;">'


def retrieve_sentence(sentence_idx, dataset_name):
    """
    Retrieve the original text for a sentence from the dataset metadata.
    
    Args:
        sentence_idx: Index of the sentence
        dataset_name: Name of the dataset
        
    Returns:
        str: The original sentence text with cleaned spacing
    """
    import os
    import pandas as pd
    
    # Read metadata
    metadata_path = f'../Data/{dataset_name}/metadata.csv'
    if not os.path.exists(metadata_path):
        print(f"Warning: metadata file not found at {metadata_path}")
        return None
        
    metadata_df = pd.read_csv(metadata_path)
    
    text = None
    
    # First check if there's a direct text column
    if 'sample_text' in metadata_df.columns:
        # Get unique texts by file_idx or by order
        if 'file_idx' in metadata_df.columns:
            # Get the text for this file_idx
            file_rows = metadata_df[metadata_df['file_idx'] == sentence_idx]
            if not file_rows.empty:
                text = file_rows.iloc[0]['sample_text']
        else:
            # Get unique texts in order
            unique_texts = metadata_df['sample_text'].unique()
            if sentence_idx < len(unique_texts):
                text = unique_texts[sentence_idx]
    
    # If no sample_text column, try reading from files
    if text is None:
        if 'sample_filename' in metadata_df.columns:
            file_col = 'sample_filename'
        elif 'text_path' in metadata_df.columns:
            file_col = 'text_path'
        else:
            print(f"Warning: Could not find text column or file path column in metadata")
            return None
            
        # Get unique files
        if 'file_idx' in metadata_df.columns:
            # Use file_idx to find the right file
            file_rows = metadata_df[metadata_df['file_idx'] == sentence_idx]
            if not file_rows.empty:
                sample_file = file_rows.iloc[0][file_col]
                file_path = os.path.join(f'../Data/{dataset_name}', sample_file)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
        else:
            # Fall back to old method
            sentence_files = metadata_df[file_col].unique()
            current_idx = 0
            for sample_file in sorted(sentence_files):
                if current_idx == sentence_idx:
                    file_path = os.path.join(f'../Data/{dataset_name}', sample_file)
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                    break
                current_idx += 1
    
    if text is None:
        print(f"Warning: sentence index {sentence_idx} not found in dataset")
        return None
    
    # Clean the text spacing before returning
    return clean_text_spacing(text)


def clean_text_spacing(text):
    """
    Clean text to ensure proper spacing and punctuation placement.
    
    Args:
        text: The text to clean
        
    Returns:
        str: Cleaned text with proper spacing
    """
    import re
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix spacing around punctuation
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?;:\'"\)])', r'\1', text)
    
    # Add space after punctuation if missing (except at end)
    text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)
    
    # Fix spacing around quotes and parentheses
    text = re.sub(r'(["\(])\s+', r'\1', text)
    text = re.sub(r'\s+(["\)])', r'\1', text)
    
    # Fix apostrophes
    text = re.sub(r'\s+\'', "'", text)
    text = re.sub(r'\'\s+', "'", text)
    
    # Clean up any remaining double spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def align_tokens_to_text(tokens, original_text):
    """
    Align tokens to character positions in the original text.
    
    Args:
        tokens: List of tokens (with possible Ġ prefixes)
        original_text: The original text string
        
    Returns:
        List of tuples (start_pos, end_pos, token_idx) for each token found in text
    """
    import re
    
    # Clean the original text
    original_text = clean_text_spacing(original_text)
    
    # Clean tokens and handle empty tokens
    cleaned_tokens = []
    token_indices = []
    for i, token in enumerate(tokens):
        cleaned = token.replace("Ġ", "")
        if cleaned and cleaned != "[EMPTY]":
            cleaned_tokens.append(cleaned)
            token_indices.append(i)
    
    # Create a mapping from character positions to tokens
    char_to_token = []
    
    # Try to match tokens to the original text
    text_lower = original_text.lower()
    current_pos = 0
    
    for i, (token, orig_idx) in enumerate(zip(cleaned_tokens, token_indices)):
        token_lower = token.lower()
        
        # Don't skip punctuation tokens - they need to be aligned too
            
        # Find the token in the remaining text
        pos = text_lower.find(token_lower, current_pos)
        
        if pos != -1:
            # Found the token
            start = pos
            end = pos + len(token)
            char_to_token.append((start, end, orig_idx))
            current_pos = end
        else:
            # Try partial matching for tokens that might be split differently
            # For example, "don't" might be tokenized as "don" + "'t"
            partial_match = False
            for j in range(1, len(token) + 1):
                partial = token_lower[:j]
                pos = text_lower.find(partial, current_pos)
                if pos == current_pos or (pos == current_pos + 1 and text_lower[current_pos] == ' '):
                    # Partial match found
                    if pos == current_pos + 1:
                        current_pos += 1
                    start = pos
                    end = pos + len(partial)
                    char_to_token.append((start, end, orig_idx))
                    current_pos = end
                    partial_match = True
                    break
            
            if not partial_match:
                # Skip this token if we can't find it
                continue
    
    return char_to_token


def get_sentence_category(sentence_idx, dataset_name):
    """
    Determine the category/class of a sentence based on dataset conventions.
    
    Args:
        sentence_idx: Index of the sentence
        dataset_name: Name of the dataset
        
    Returns:
        str: Category/class label for the sentence
    """
    if dataset_name.lower() == 'imdb':
        # Even indices are positive, odd are negative
        return 'positive' if sentence_idx % 2 == 0 else 'negative'
    elif dataset_name.lower() in ['sarcasm', 'isarcasm']:
        # Assuming binary classification - you may need to load labels
        # This is a placeholder - replace with actual label loading
        return 'sarcastic' if sentence_idx % 2 == 0 else 'non-sarcastic'
    elif dataset_name.lower() == 'stanford_tree_bank':
        # Stanford Sentiment Treebank has multiple classes
        # This is a placeholder - replace with actual label loading
        classes = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
        return classes[sentence_idx % 5]
    else:
        return 'unknown'


#### Additional Helper Functions ####
def get_word_from_indices(global_indices, flat_tokens_list):
    """
    Retrieve tokens from a flattened token list given their global indices.

    Args:
        global_indices (List[int]): Global indices of tokens in the flattened list.
        flat_tokens_list (List[str]): Flattened list of all tokens.

    Returns:
        List[str]: Tokens corresponding to the given indices.
    """
    return [flat_tokens_list[idx] for idx in global_indices]


def get_top_token_indices_for_concept_simple(sims_per_concept, concept, k=5):
    """
    Returns indices of tokens with the highest activation for a given concept.

    Args:
        sims_per_concept (dict): Dictionary where keys are concepts and values are lists of similarity scores.
        concept (str): The concept for which to find top tokens.
        k (int): Number of top tokens to return. Defaults to 5.

    Returns:
        Tuple: A tuple containing:
            - List[int]: Indices of top-k most similar tokens.
            - List[float]: Corresponding similarity scores.
    """
    sims = sims_per_concept[concept]
    top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    top_sims = [sims[i] for i in top_indices]
    return top_indices, top_sims


def get_top_token_indices_for_concept(act_loader, tokens_list, concept, dataset_name, top_k=5, split='test', chunk_size=50000):
    """
    Returns indices of tokens with the highest activation for the given concept.

    Args:
        act_loader: ChunkedActivationLoader instance
        tokens_list: List of tokenized sentences
        concept (str): Concept column to search by
        dataset_name (str): Dataset name
        top_k (int): Number of top tokens to return
        split (str): 'train', 'test', or 'both'
        chunk_size (int): Size of chunks to process at once

    Returns:
        List of indices (or a single index if top_k=1).
    """
    import heapq
    
    # Get concept column index
    concept_idx = act_loader.get_concept_index(concept)
    
    # Determine valid token indices based on split
    if split != 'both':
        # Step 1: Load split info
        split_df = get_split_df(dataset_name)

        # Step 2: Determine sentence indices that match the split
        valid_sentence_indices = split_df[split_df == split].index.tolist()

        # Step 3: Convert sentence-level mask to token-level indices
        valid_token_indices = set()
        idx = 0
        for i, tokens in enumerate(tokens_list):
            if i in valid_sentence_indices:
                valid_token_indices.update(range(idx, idx + len(tokens)))
            idx += len(tokens)
    else:
        valid_token_indices = None  # All indices are valid
    
    # Use a min heap to track top k values
    top_k_heap = []
    
    # Process data in chunks
    for chunk_start in range(0, len(act_loader), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(act_loader))
        
        # Load chunk
        chunk_tensor = act_loader.load_tensor_range(chunk_start, chunk_end)
        concept_acts = chunk_tensor[:, concept_idx]
        
        # Process each activation in the chunk
        for i, activation in enumerate(concept_acts):
            global_idx = chunk_start + i
            
            # Skip if not in valid split
            if valid_token_indices is not None and global_idx not in valid_token_indices:
                continue
            
            # Use negative activation for min heap (to get max values)
            if len(top_k_heap) < top_k:
                heapq.heappush(top_k_heap, (activation.item(), global_idx))
            elif activation > top_k_heap[0][0]:
                heapq.heapreplace(top_k_heap, (activation.item(), global_idx))
        
        # Clean up
        del chunk_tensor, concept_acts
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Extract indices from heap and sort by activation (descending)
    top_items = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
    top_indices = [item[1] for item in top_items]
    
    return top_indices if top_k > 1 else top_indices[0] if top_indices else None


def user_select_concept(concepts):
    """
    Interactive function to let user select a concept from a list.
    
    Args:
        concepts: List of available concepts
        
    Returns:
        str: Selected concept name
    """
    for i, concept in enumerate(concepts):
        print(f"{i}: {concept}")
    idx = int(input("Select a concept by index: "))
    return concepts[idx]


def get_sentences_by_metric(
    dataset_name,
    concept,
    acts_loader, 
    tokens_list,
    split='test',
    k=5,
    aggregate='mean',
    highest=True
):
    """
    Find top-k sentences with highest/lowest aggregate similarity to a concept.
    Filters by dataset split before computing.
    
    Args:
        dataset_name (str): Name of the dataset.
        concept (str): The concept to find sentences for.
        acts_loader: ChunkedActivationLoader instance.
        tokens_list (list): List of token lists for all sentences.
        split (str): Which split to use ('train', 'val', 'test'). Defaults to 'test'.
        k (int): Number of sentences to return. Defaults to 5.
        aggregate (str): How to aggregate token similarities ('mean' or 'max'). Defaults to 'mean'.
        highest (bool): If True, return highest scoring sentences; if False, return lowest.
    
    Returns:
        list: List of tuples (sentence_idx, score) for top-k sentences.
    """
    # Get concept index
    concept_idx = acts_loader.get_concept_index(concept)
    
    # Load split information
    from utils.general_utils import get_split_df
    split_df = get_split_df(dataset_name)
    
    # Get sentence indices for the requested split
    split_mask = split_df == split
    split_sentence_indices = split_df[split_mask].index.tolist()
    
    sentence_scores = []
    
    # Process each sentence in the split
    for sent_idx in split_sentence_indices:
        if sent_idx >= len(tokens_list):
            continue
            
        # Get global token indices for this sentence
        start_idx = sum(len(tokens) for tokens in tokens_list[:sent_idx])
        end_idx = start_idx + len(tokens_list[sent_idx])
        
        # Load activations for this sentence
        acts = acts_loader.load_tensor_range(start_idx, end_idx)
        concept_acts = acts[:, concept_idx].cpu().numpy()
        
        # Aggregate (mean or max)
        if aggregate == 'mean':
            score = concept_acts.mean()
        elif aggregate == 'max':
            score = concept_acts.max()
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")
        
        sentence_scores.append((sent_idx, score))
    
    # Sort by score
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=highest)
    
    return sorted_sentences[:k]


def compute_avg_token_similarities(embeddings, flat_tokens_list):
    token_to_indices = defaultdict(list)

    # Step 1: map tokens to their embedding indices
    for idx, token in enumerate(flat_tokens_list):
        token_to_indices[token].append(idx)

    token_to_stats = {}

    for token, indices in token_to_indices.items():
        count = len(indices)

        if count < 2:
            token_to_stats[token] = {"avg_sim": float('nan'), "count": count}
            continue

        # Get embeddings for the token
        token_embeds = embeddings[indices]  # shape (count, hidden_dim)

        # Compute pairwise cosine similarities
        sims = []
        for i, j in combinations(range(count), 2):
            sim = cosine_similarity(token_embeds[i].unsqueeze(0), token_embeds[j].unsqueeze(0))
            sims.append(sim.item())

        avg_sim = sum(sims) / len(sims)
        token_to_stats[token] = {"avg_sim": avg_sim, "count": count}

    return token_to_stats


def get_colormap_color(score, colormap, norm):
    return matplotlib.colors.to_hex(colormap(norm(score)))


def plot_colorbar(vmin, vmax, cmap, orientation='horizontal'):
    """
    Creates a colorbar visualization and returns it as a base64-encoded image.
    
    Args:
        vmin: Minimum value for the colorbar
        vmax: Maximum value for the colorbar
        cmap: Colormap to use (string name or colormap object)
        orientation: 'horizontal' or 'vertical'
    
    Returns:
        str: HTML img tag with base64-encoded colorbar
    """
    fig, ax = plt.subplots(figsize=(6, 1) if orientation == 'horizontal' else (1, 6))
    
    # Create a colormap normalization
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Create the colorbar
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                     cax=ax, orientation=orientation)
    
    # Save to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    
    # Encode the image to base64 string
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return f'<img src="data:image/png;base64,{img_base64}" style="width: 300px;">'


def highlight_tokens_with_legend(tokens, scores, score_to_color_fn, title=""):
    """
    Creates HTML visualization of tokens with color-coded background based on scores.
    
    Args:
        tokens: List of token strings
        scores: List of scores corresponding to each token
        score_to_color_fn: Function that maps a score to a color
        title: Optional title for the visualization
    
    Returns:
        HTML string for display
    """
    html_parts = []
    
    if title:
        html_parts.append(f"<h3>{title}</h3>")
    
    html_parts.append('<div style="line-height: 2.5;">')
    
    for token, score in zip(tokens, scores):
        color = score_to_color_fn(score)
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; '
            f'margin: 2px; border-radius: 3px; display: inline-block;">'
            f'{token}</span>'
        )
    
    html_parts.append('</div>')
    
    return ''.join(html_parts)


#### Visualizations ####
import matplotlib.pyplot as plt

def visualize_token_activations(
    concept, 
    sims_per_concept, 
    tokens, 
    cmap_name="coolwarm", 
    vmin=None, 
    vmax=None,
    width=12, 
    height=1.5,
    title_prefix="",
    gt_token_mask=None
):
    """
    Visualize token activations as a horizontal bar with colored segments.
    
    Args:
        concept: The concept name
        sims_per_concept: Dictionary mapping concepts to lists of similarity scores
        tokens: List of tokens
        cmap_name: Name of the colormap to use
        vmin: Minimum value for colormap (if None, computed from data)
        vmax: Maximum value for colormap (if None, computed from data)
        width: Figure width
        height: Figure height  
        title_prefix: Optional prefix for the title
        gt_token_mask: Optional boolean mask indicating ground truth tokens
    """
    sims = sims_per_concept[concept]
    
    # Compute vmin/vmax if not provided
    if vmin is None or vmax is None:
        all_sims = [s for concept_sims in sims_per_concept.values() for s in concept_sims]
        if vmin is None:
            vmin = min(all_sims)
        if vmax is None:
            vmax = max(all_sims)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    
    # Get colormap
    cmap = plt.cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Create bars for each token
    x_positions = np.arange(len(tokens))
    bar_width = 0.8
    
    for i, (token, sim) in enumerate(zip(tokens, sims)):
        color = cmap(norm(sim))
        bar = ax.bar(i, 1, bar_width, color=color, edgecolor='black', linewidth=0.5)
        
        # Add star for GT tokens
        if gt_token_mask and gt_token_mask[i]:
            ax.text(i, 0.5, '★', ha='center', va='center', 
                   color='white', fontsize=16, fontweight='bold')
    
    # Customize plot
    ax.set_xlim(-0.5, len(tokens) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticks([])
    ax.set_xlabel('Tokens')
    
    title = f"{title_prefix}{concept} Activations"
    ax.set_title(title)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Activation Score')
    
    plt.tight_layout()
    plt.show()


def plot_concept_activation_comparison(
    sentence_idx,
    concepts,
    act_loader,
    tokens_list,
    dataset_name,
    gt_samples_per_concept=None,
    cmap_name="coolwarm",
    width=15,
    height_per_concept=2,
    vmin=None,
    vmax=None
):
    """
    Plot activation comparisons for multiple concepts on the same sentence.
    
    Args:
        sentence_idx: Index of the sentence to visualize
        concepts: List of concept names to compare
        act_loader: MatchedConceptActivationLoader instance
        tokens_list: List of token lists
        dataset_name: Name of the dataset
        gt_samples_per_concept: Optional dict mapping concepts to GT token indices
        cmap_name: Colormap name
        width: Figure width
        height_per_concept: Height per concept subplot
        vmin/vmax: Optional fixed colormap limits
    """
    from utils.general_utils import get_paper_plotting_style
    
    # Apply paper plotting style
    plt.rcParams.update(get_paper_plotting_style())
    
    # Get tokens for this sentence
    tokens = [tok.replace("Ġ", "") for tok in tokens_list[sentence_idx]]
    start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list)
    
    # Load activations
    sentence_acts = act_loader.load_tensor_range(start_idx, end_idx).cpu().numpy()
    
    # Get similarities for each concept
    sims_per_concept = {}
    for concept in concepts:
        concept_idx = act_loader.get_concept_index(concept)
        sims_per_concept[concept] = sentence_acts[:, concept_idx].tolist()
    
    # Compute vmin/vmax if not provided
    if vmin is None or vmax is None:
        all_sims = [s for sims in sims_per_concept.values() for s in sims]
        if vmin is None:
            vmin = min(all_sims)
        if vmax is None:
            vmax = max(all_sims)
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(concepts), 1, figsize=(width, height_per_concept * len(concepts)))
    if len(concepts) == 1:
        axes = [axes]
    
    # Get colormap
    cmap = plt.cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot each concept
    for idx, (ax, concept) in enumerate(zip(axes, concepts)):
        sims = sims_per_concept[concept]
        
        # Create bars for each token
        x_positions = np.arange(len(tokens))
        bar_width = 0.8
        
        # Get GT mask for this concept
        gt_mask = [False] * len(tokens)
        if gt_samples_per_concept and concept in gt_samples_per_concept:
            for gt_idx in gt_samples_per_concept[concept]:
                if start_idx <= gt_idx < end_idx:
                    gt_mask[gt_idx - start_idx] = True
        
        for i, (token, sim) in enumerate(zip(tokens, sims)):
            color = cmap(norm(sim))
            bar = ax.bar(i, 1, bar_width, color=color, edgecolor='black', linewidth=0.5)
            
            # Add star for GT tokens
            if gt_mask[i]:
                ax.text(i, 0.5, '★', ha='center', va='center',
                       color='white', fontsize=14, fontweight='bold')
        
        # Customize subplot
        ax.set_xlim(-0.5, len(tokens) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks(x_positions)
        
        # Only show x-labels on bottom subplot
        if idx == len(concepts) - 1:
            ax.set_xticklabels(tokens, rotation=45, ha='right')
        else:
            ax.set_xticklabels([])
        
        ax.set_yticks([])
        ax.set_ylabel(concept, rotation=0, ha='right', va='center', fontweight='bold')
        
        # Add max activation value to title
        max_sim = max(sims)
        max_token_idx = sims.index(max_sim)
        ax.set_title(f"Max: {max_sim:.3f} at '{tokens[max_token_idx]}'", loc='right', fontsize=10)
    
    # Add single colorbar for all subplots
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Activation Score')
    
    # Overall title
    sentence_class = get_sentence_category(sentence_idx, dataset_name)
    fig.suptitle(f'Sentence {sentence_idx} ({sentence_class}): Multi-Concept Activation Comparison', 
                 fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.show()


def plot_activation_statistics(
    concept,
    act_loader,
    tokens_list,
    dataset_name,
    split='test',
    n_bins=50
):
    """
    Plot statistics about concept activations across the dataset.
    
    Args:
        concept: Concept name
        act_loader: MatchedConceptActivationLoader instance
        tokens_list: List of token lists
        dataset_name: Name of the dataset
        split: Which split to analyze ('train' or 'test')
        n_bins: Number of bins for histogram
    """
    # Get split indices
    split_df = get_split_df(dataset_name)
    split_indices = split_df[split_df == split].index.tolist()
    
    # Collect all activations for this concept
    concept_idx = act_loader.get_concept_index(concept)
    all_activations = []
    
    for sent_idx in split_indices:
        if sent_idx < len(tokens_list):
            start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sent_idx, tokens_list)
            acts = act_loader.load_tensor_range(start_idx, end_idx)[:, concept_idx].cpu().numpy()
            all_activations.extend(acts.tolist())
    
    all_activations = np.array(all_activations)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax1.hist(all_activations, bins=n_bins, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(all_activations), color='red', linestyle='--', label=f'Mean: {np.mean(all_activations):.3f}')
    ax1.axvline(np.median(all_activations), color='green', linestyle='--', label=f'Median: {np.median(all_activations):.3f}')
    ax1.set_xlabel('Activation Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{concept} Activation Distribution ({split} split)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot with percentiles
    ax2.boxplot(all_activations, vert=True, widths=0.5)
    
    # Add percentile lines
    percentiles = [2, 10, 25, 50, 75, 90, 98]
    percentile_values = np.percentile(all_activations, percentiles)
    
    for p, val in zip(percentiles, percentile_values):
        ax2.axhline(val, color='red', alpha=0.5, linestyle=':', linewidth=1)
        ax2.text(1.3, val, f'{p}%: {val:.3f}', va='center', fontsize=9)
    
    ax2.set_ylabel('Activation Value')
    ax2.set_title(f'{concept} Activation Percentiles')
    ax2.set_xticklabels([concept])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Activation Statistics for "{concept}" ({dataset_name})', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics for '{concept}':")
    print(f"Total tokens: {len(all_activations)}")
    print(f"Mean: {np.mean(all_activations):.4f}")
    print(f"Std: {np.std(all_activations):.4f}")
    print(f"Min: {np.min(all_activations):.4f}")
    print(f"Max: {np.max(all_activations):.4f}")
    print(f"\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}%: {np.percentile(all_activations, p):.4f}")


def plot_all_concept_activations_on_sentence(
    sentence_idx,
    act_loader,
    tokens_list,
    dataset_name,
    cmap_name="coolwarm",
    gt_samples_per_concept=None,
    vmin=None,
    vmax=None
):
    from IPython.display import HTML, display
    from utils.general_utils import get_paper_plotting_style
    
    # Apply paper plotting style
    import matplotlib.pyplot as plt
    plt.rcParams.update(get_paper_plotting_style())

    def clean_token(token):
        return token.replace("Ġ", "")  # Strip GPT/RoBERTa word boundary marker

    concepts = act_loader.columns
    raw_tokens = tokens_list[sentence_idx]
    tokens = [clean_token(tok) for tok in raw_tokens]
    start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list)

    # Load activations for this sentence
    sentence_acts = act_loader.load_tensor_range(start_idx, end_idx).cpu().numpy()
    
    # Get per-concept similarity scores for this sentence
    sims_per_concept = {
        concept: sentence_acts[:, act_loader.get_concept_index(concept)].tolist()
        for concept in concepts
    }

    # Compute vmin/vmax from all sims if not provided
    if vmin is None or vmax is None:
        all_sims = [sim for sims in sims_per_concept.values() for sim in sims]
        vmin = min(all_sims) if vmin is None else vmin
        vmax = max(all_sims) if vmax is None else vmax

    sentence_class = get_sentence_category(sentence_idx, dataset_name)
    print(f"\nSentence {sentence_idx} ({sentence_class}):\n")

    # Optional ground truth display
    if gt_samples_per_concept is not None:
        print("Ground Truth Concept Labels:\n")
        for concept in concepts:
            concept_token_mask = [False] * len(tokens)
            for idx in gt_samples_per_concept.get(concept, []):
                if start_idx <= idx < end_idx:
                    concept_token_mask[idx - start_idx] = True

            highlighted_tokens = [
                f"<span style='background-color: yellow'>{token}</span>" if is_labeled else token
                for token, is_labeled in zip(tokens, concept_token_mask)
            ]
            display(HTML(f"<b>{concept}:</b> {' '.join(highlighted_tokens)}"))
        print("\n")

    # Heatmap display
    print("Concept Activation Heatmaps:\n")

    # For each concept, create an HTML row displaying the tokens with background colors based on similarity
    cmap = plt.cm.get_cmap(cmap_name)
    
    for concept in concepts:
        sims = sims_per_concept[concept]
        
        # Create HTML for tokens with colored backgrounds
        html_tokens = []
        for token, sim in zip(tokens, sims):
            color = get_color_for_sim(sim, vmin, vmax, cmap)
            html_tokens.append(
                f"<span style='background-color: {color}; padding: 2px; margin: 1px; "
                f"border-radius: 3px; display: inline-block'>{token}</span>"
            )
        
        # Find max activation token
        max_sim = max(sims)
        max_idx = sims.index(max_sim)
        max_token = tokens[max_idx]
        
        html = f"<div style='margin-bottom: 10px'>"
        html += f"<b>{concept}</b> (max: {max_sim:.3f} at '{max_token}'): "
        html += " ".join(html_tokens)
        html += "</div>"
        
        display(HTML(html))
    
    # Display colorbar
    colorbar_html = make_colorbar_image(vmin, vmax, cmap_name)
    display(HTML(colorbar_html))


def visualize_concept_examples(
    concept,
    act_loader,
    tokens_list, 
    dataset_name,
    n_examples=3,
    cmap_name="coolwarm",
    show_distribution=True
):
    """
    Visualize top positive and negative examples for a concept with optional
    activation distribution plot.
    
    Args:
        concept: The concept to visualize
        act_loader: MatchedConceptActivationLoader instance
        tokens_list: List of token lists for all sentences
        dataset_name: Name of the dataset
        n_examples: Number of examples to show for each category
        cmap_name: Colormap name for visualizations
        show_distribution: Whether to show the activation distribution plot
    """
    from IPython.display import HTML, display
    import numpy as np
    
    # Get concept index
    concept_idx = act_loader.get_concept_index(concept)
    
    # Get split information
    split_df = get_split_df(dataset_name)
    test_indices = split_df[split_df == 'test'].index.tolist()
    
    # Collect paragraph-level statistics
    paragraph_stats = []
    
    for sent_idx in test_indices:
        if sent_idx < len(tokens_list):
            start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sent_idx, tokens_list)
            
            # Load activations for this paragraph
            acts = act_loader.load_tensor_range(start_idx, end_idx)
            concept_acts = acts[:, concept_idx].cpu().numpy()
            
            # Store statistics
            paragraph_stats.append({
                'sent_idx': sent_idx,
                'max_activation': np.max(concept_acts),
                'mean_activation': np.mean(concept_acts),
                'tokens': tokens_list[sent_idx]
            })
    
    # Sort by max activation
    paragraph_stats.sort(key=lambda x: x['max_activation'], reverse=True)
    
    # Get top positive and negative examples
    top_positive = paragraph_stats[:n_examples]
    top_negative = paragraph_stats[-n_examples:]
    
    # Also get some near-zero examples
    paragraph_stats_by_abs = sorted(paragraph_stats, key=lambda x: abs(x['max_activation']))
    near_zero = paragraph_stats_by_abs[:n_examples]
    
    # Optionally show distribution
    if show_distribution:
        all_max_acts = [p['max_activation'] for p in paragraph_stats]
        
        plt.figure(figsize=(10, 4))
        plt.hist(all_max_acts, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('Max Activation per Paragraph')
        plt.ylabel('Count')
        plt.title(f'Distribution of Max Activations for "{concept}" (test set)')
        plt.grid(True, alpha=0.3)
        
        # Mark example positions
        for ex in top_positive:
            plt.axvline(ex['max_activation'], color='green', alpha=0.5, linewidth=1)
        for ex in top_negative:
            plt.axvline(ex['max_activation'], color='red', alpha=0.5, linewidth=1)
        for ex in near_zero:
            plt.axvline(ex['max_activation'], color='gray', alpha=0.5, linewidth=1)
        
        plt.show()
    
    # Display examples
    print(f"\nConcept: {concept}\n")
    
    # Get colormap
    cmap = plt.cm.get_cmap(cmap_name)
    
    # Compute global vmin/vmax from all examples
    all_example_indices = (
        [p['sent_idx'] for p in top_positive] +
        [p['sent_idx'] for p in top_negative] +
        [p['sent_idx'] for p in near_zero]
    )
    
    all_acts = []
    for sent_idx in all_example_indices:
        start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sent_idx, tokens_list)
        acts = act_loader.load_tensor_range(start_idx, end_idx)[:, concept_idx].cpu().numpy()
        all_acts.extend(acts.tolist())
    
    vmin, vmax = min(all_acts), max(all_acts)
    
    # Helper function to display examples
    def display_examples(examples, category_name, category_color):
        display(HTML(f"<h3 style='color: {category_color}'>{category_name}</h3>"))
        
        for i, ex in enumerate(examples):
            sent_idx = ex['sent_idx']
            tokens = [tok.replace("Ġ", "") for tok in ex['tokens']]
            
            # Get activations
            start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sent_idx, tokens_list)
            acts = act_loader.load_tensor_range(start_idx, end_idx)
            concept_acts = acts[:, concept_idx].cpu().numpy()
            
            # Create colored tokens
            html_tokens = []
            for token, act in zip(tokens, concept_acts):
                color = get_color_for_sim(act, vmin, vmax, cmap)
                html_tokens.append(
                    f"<span style='background-color: {color}; padding: 2px; margin: 1px; "
                    f"border-radius: 3px; display: inline-block'>{token}</span>"
                )
            
            # Find max token
            max_idx = np.argmax(concept_acts)
            max_token = tokens[max_idx]
            
            sentence_class = get_sentence_category(sent_idx, dataset_name)
            
            html = f"<div style='margin-bottom: 15px; padding: 10px; border-left: 3px solid {category_color}'>"
            html += f"<b>Example {i+1}</b> (Sentence {sent_idx}, {sentence_class}) - "
            html += f"Max: {ex['max_activation']:.3f} at '{max_token}'<br>"
            html += " ".join(html_tokens)
            html += "</div>"
            
            display(HTML(html))
    
    # Display each category
    display_examples(top_positive, "Top Positive Activations", "green")
    display_examples(near_zero, "Near Zero Activations", "gray")
    display_examples(top_negative, "Top Negative Activations", "red")
    
    # Display colorbar
    colorbar_html = make_colorbar_image(vmin, vmax, cmap_name)
    display(HTML(colorbar_html))


def visualize_concept_examples_with_gt(
    concept,
    act_loader,
    tokens_list,
    dataset_name,
    gt_samples_per_concept=None,
    n_examples=3,
    cmap_name="coolwarm"
):
    """
    Visualize concept examples separated by ground truth presence.
    Shows examples where the concept is actually present (GT=True) vs not (GT=False).
    
    Args:
        concept: The concept to visualize
        act_loader: MatchedConceptActivationLoader instance
        tokens_list: List of token lists
        dataset_name: Name of the dataset
        gt_samples_per_concept: Dict mapping concepts to GT token indices
        n_examples: Number of examples per category
        cmap_name: Colormap name
    """
    from IPython.display import HTML, display
    import os
    
    # Get concept index
    concept_idx = act_loader.get_concept_index(concept)
    
    # Load ground truth if not provided
    if gt_samples_per_concept is None:
        gt_path = f'GT_samples/{dataset_name}/gt_samples_per_concept_test.pt'
        if os.path.exists(gt_path):
            gt_samples_per_concept = torch.load(gt_path)
            print(f"Loaded GT samples from {gt_path}")
        else:
            print(f"Warning: No GT samples found at {gt_path}")
            return
    
    # Get GT indices for this concept
    gt_indices = set(gt_samples_per_concept.get(concept, []))
    if not gt_indices:
        print(f"Warning: No ground truth samples found for concept '{concept}'")
        gt_indices = set()
    
    # Get test split indices
    split_df = get_split_df(dataset_name)
    test_sentence_indices = split_df[split_df == 'test'].index.tolist()
    
    # Compute max activation per paragraph and track which paragraphs have GT tokens
    paragraph_data = []
    idx = 0
    for sent_idx, tokens in enumerate(tokens_list):
        if sent_idx in test_sentence_indices:
            start_idx = idx
            end_idx = idx + len(tokens)
            
            # Get activations for all tokens in this paragraph
            paragraph_acts = act_loader.load_tensor_range(start_idx, end_idx)
            concept_acts = paragraph_acts[:, concept_idx].cpu().numpy()
            max_activation = concept_acts.max()
            
            # Check if any token in this paragraph is GT
            paragraph_token_indices = list(range(start_idx, end_idx))
            has_gt = any(token_idx in gt_indices for token_idx in paragraph_token_indices)
            
            paragraph_data.append({
                'sent_idx': sent_idx,
                'max_activation': max_activation,
                'has_gt': has_gt
            })
        idx += len(tokens)
    
    # Split by ground truth
    gt_true_paragraphs = [p for p in paragraph_data if p['has_gt']]
    gt_false_paragraphs = [p for p in paragraph_data if not p['has_gt']]
    
    # Sort by max activation
    gt_true_sorted = sorted(gt_true_paragraphs, key=lambda x: x['max_activation'], reverse=True)
    gt_false_sorted = sorted(gt_false_paragraphs, key=lambda x: x['max_activation'], reverse=True)
    
    # Get examples for each category
    categories = {}
    
    # GT True categories
    if len(gt_true_sorted) >= n_examples:
        categories["GT True - Most Positive"] = gt_true_sorted[:n_examples]
        categories["GT True - Most Negative"] = gt_true_sorted[-n_examples:]
        # For near zero, sort by absolute value of max activation
        gt_true_by_abs = sorted(gt_true_paragraphs, key=lambda x: abs(x['max_activation']))
        categories["GT True - Near Zero"] = gt_true_by_abs[:n_examples]
    else:
        categories["GT True - Most Positive"] = gt_true_sorted
        categories["GT True - Most Negative"] = []
        categories["GT True - Near Zero"] = []
    
    # GT False categories
    if len(gt_false_sorted) >= n_examples:
        categories["GT False - Most Positive"] = gt_false_sorted[:n_examples]
        categories["GT False - Most Negative"] = gt_false_sorted[-n_examples:]
        # For near zero, sort by absolute value of max activation
        gt_false_by_abs = sorted(gt_false_paragraphs, key=lambda x: abs(x['max_activation']))
        categories["GT False - Near Zero"] = gt_false_by_abs[:n_examples]
    else:
        categories["GT False - Most Positive"] = gt_false_sorted
        categories["GT False - Most Negative"] = []
        categories["GT False - Near Zero"] = []
    
    # Collect all scores for consistent colormap scale
    all_scores = []
    for category_paragraphs in categories.values():
        for paragraph in category_paragraphs:
            sent_idx = paragraph['sent_idx']
            start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sent_idx, tokens_list)
            paragraph_acts = act_loader.load_tensor_range(start_idx, end_idx)
            all_scores.extend(paragraph_acts[:, concept_idx].cpu().numpy().tolist())
    
    if all_scores:
        vmin, vmax = min(all_scores), max(all_scores)
    else:
        vmin, vmax = -1, 1  # Default range if no scores
    
    # Create visualization
    html_blocks = []
    
    for category_name, paragraphs in categories.items():
        if paragraphs:  # Only show categories with examples
            html_blocks.append(f"<h3>{category_name}</h3>")
            
            for i, paragraph in enumerate(paragraphs):
                sent_idx = paragraph['sent_idx']
                max_activation = paragraph['max_activation']
                start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sent_idx, tokens_list)
                
                tokens = tokens_list[sent_idx]
                paragraph_acts = act_loader.load_tensor_range(start_idx, end_idx)
                sims = paragraph_acts[:, concept_idx].cpu().numpy().tolist()
                
                # Find which token has the max activation
                max_token_idx = np.argmax(sims)
                max_token = remove_leading_token([tokens[max_token_idx]])[0]
                
                # Check which tokens are GT
                token_is_gt = []
                for j in range(len(tokens)):
                    global_idx = start_idx + j
                    token_is_gt.append(global_idx in gt_indices)
                
                # Create colored HTML tokens
                cmap = plt.cm.get_cmap(cmap_name)
                html_tokens = []
                for j, (token, sim, is_gt) in enumerate(zip(tokens, sims, token_is_gt)):
                    clean_token = remove_leading_token([token])[0]
                    color = get_color_for_sim(sim, vmin, vmax, cmap)
                    
                    # Add star for GT tokens
                    star = "★ " if is_gt else ""
                    
                    border_color = "gold" if is_gt else "transparent"
                    html_tokens.append(
                        f"<span style='background-color: {color}; padding: 2px 4px; margin: 1px; "
                        f"border-radius: 3px; display: inline-block; "
                        f"border: 2px solid {border_color}'>"
                        f"{star}{clean_token}</span>"
                    )
                
                sentence_class = get_sentence_category(sent_idx, dataset_name)
                
                html = f"<div style='margin-bottom: 15px; padding: 10px; background: #f5f5f5; border-radius: 5px'>"
                html += f"<b>Sentence {sent_idx}</b> ({sentence_class}) - "
                html += f"Max activation: {max_activation:.3f} at '{max_token}'<br>"
                html += f"<div style='margin-top: 5px'>{' '.join(html_tokens)}</div>"
                html += "</div>"
                
                html_blocks.append(html)
    
    # Add colorbar
    colorbar_html = make_colorbar_image(vmin, vmax, cmap_name)
    
    # Print summary statistics
    print(f"\nConcept: {concept}")
    print(f"GT True paragraphs in test set: {len(gt_true_paragraphs)}")
    print(f"GT False paragraphs in test set: {len(gt_false_paragraphs)}")
    print(f"Activation range: [{vmin:.3f}, {vmax:.3f}]")
    
    full_html = f"""
    <div>
        {''.join(html_blocks)}
        <div style="margin-top: 20px;">{colorbar_html}</div>
    </div>
    """
    display(HTML(full_html))


# def plot_binarized_token_activations_with_raw_heatmap(
#     sentence_idx,
#     concept,
#     act_loader,
#     tokens_list,
#     dataset_name,
#     thresholds_dict,
#     gt_samples_per_concept=None,
#     cmap_name="coolwarm",
#     vmin=None,
#     vmax=None,
#     save_file=None,
#     use_matplotlib=True,
#     auto_scale=True
# ):
#     """
#     Creates a hybrid visualization showing:
#     1. Original sentence with ground truth concept tokens highlighted
#     2. Raw heatmap with activation values for each token
#     3. Binarized versions at different threshold percentiles
    
#     Args:
#         sentence_idx: Index of the sentence to visualize
#         concept: The concept to visualize
#         act_loader: Activation loader with concept activations
#         tokens_list: List of token lists for all sentences
#         dataset_name: Name of the dataset
#         thresholds_dict: Dictionary mapping percentiles to threshold values
#                         e.g., {0.9: threshold_90, 0.5: threshold_50, 0.02: threshold_02}
#         gt_samples_per_concept: Optional dict mapping concepts to GT token indices
#         cmap_name: Colormap name for the heatmap
#         vmin/vmax: Min/max values for colormap scaling
#         save_file: Optional path to save the figure
#         use_matplotlib: If True, use matplotlib (for saving figures). If False, use HTML display.
#         auto_scale: If True and vmin/vmax are provided, will extend them if data exceeds range
#     """
#     from IPython.display import HTML, display
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as mpatches
#     from matplotlib.colors import Normalize
    
#     def clean_token(token):
#         # Strip GPT/RoBERTa word boundary marker but keep the token
#         cleaned = token.replace("Ġ", "")
#         # Don't skip empty tokens - they might have activations
#         return cleaned if cleaned else "[EMPTY]"
    
#     # Get tokens and activations for this sentence
#     raw_tokens = tokens_list[sentence_idx]
#     tokens = [clean_token(tok) for tok in raw_tokens]
#     start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list)
    
#     print(f"\n[DEBUG plot_binarized] Sentence {sentence_idx}")
#     print(f"[DEBUG] Raw tokens (first 10): {raw_tokens[:10]}")
#     print(f"[DEBUG] Cleaned tokens (first 10): {tokens[:10]}")
    
#     # Get concept index
#     concept_idx = act_loader.get_concept_index(concept)
    
#     # Load activations for this sentence
#     sentence_acts = act_loader.load_tensor_range(start_idx, end_idx).cpu().numpy()
#     concept_acts = sentence_acts[:, concept_idx]
    
    
#     # Determine vmin/vmax if not provided
#     if vmin is None:
#         vmin = concept_acts.min()
#     if vmax is None:
#         vmax = concept_acts.max()
    
    
#     # Get ground truth indices for this sentence if available
#     gt_token_mask = [False] * len(tokens)
#     if gt_samples_per_concept is not None and concept in gt_samples_per_concept:
#         for idx in gt_samples_per_concept[concept]:
#             if start_idx <= idx < end_idx:
#                 gt_token_mask[idx - start_idx] = True
    
#     # Sort thresholds by percentile (high to low)
#     sorted_percentiles = sorted(thresholds_dict.keys(), reverse=True)
    
#     # Get sentence category
#     sentence_class = get_sentence_category(sentence_idx, dataset_name)
    
#     # If using HTML display (default) and not saving to file (or saving as HTML)
#     if not use_matplotlib and (not save_file or save_file.endswith('.html')):
#         # Get colormap
#         cmap = plt.cm.get_cmap(cmap_name)
        
#         html_blocks = []
        
#         # Header
#         html_blocks.append(f"<h3>Sentence {sentence_idx} ({sentence_class}) - Concept: {concept}</h3>")
        
        
        
#         # Try to get original sentence text
#         original_text = retrieve_sentence(sentence_idx, dataset_name)
#         if not original_text:
#             # Fallback to token-based display
#             original_text = " ".join(tokens)
        
        
#         # Function to map tokens to positions in original text
#         def create_highlighted_text(text, tokens, highlights, cmap=None, vmin=None, vmax=None, style_type='gt'):
#             """
#             Map tokens back to original text and apply highlighting.
#             style_type: 'gt' for ground truth, 'heatmap' for colored, 'binary' for threshold
#             """
#             # Get token alignments to original text
#             alignments = align_tokens_to_text(tokens, text)
            
#             if not alignments:
#                 # Fallback to token-based display if alignment fails
#                 result_parts = []
#                 max_idx = min(len(tokens), len(highlights))
                
#                 for i in range(max_idx):
#                     token = tokens[i]
#                     cleaned = token.replace("Ġ", "")
#                     if not cleaned or cleaned == "[EMPTY]":
#                         continue
                        
#                     if style_type == 'gt':
#                         if highlights[i]:
#                             result_parts.append(f'<span style="background-color: yellow; padding: 2px; border-radius: 3px;">{cleaned}</span>')
#                         else:
#                             result_parts.append(cleaned)
#                     elif style_type == 'heatmap':
#                         color = get_color_for_sim(highlights[i], vmin, vmax, cmap)
#                         result_parts.append(f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;">{cleaned}</span>')
#                     elif style_type == 'binary':
#                         threshold, act_value = highlights[i]
#                         if act_value >= threshold:
#                             result_parts.append(f'<span style="background-color: white; color: black; padding: 2px; border-radius: 3px;">{cleaned}</span>')
#                         else:
#                             result_parts.append(f'<span style="background-color: black; color: black; padding: 2px; border-radius: 3px;">{cleaned}</span>')
                
#                 return ' '.join(result_parts)
            
#             # Build the highlighted text character by character
#             result = []
#             last_pos = 0
            
#             for start, end, token_idx in token_positions:
#                 # Add any unhighlighted text before this token
#                 if start > last_pos:
#                     result.append(text[last_pos:start])
                
#                 # Get the actual text for this token
#                 token_text = text[start:end]
                
#                 # Apply highlighting based on style
#                 if style_type == 'gt':
#                     if token_idx < len(highlights) and highlights[token_idx]:
#                         result.append(f'<span style="background-color: yellow; padding: 2px; border-radius: 3px;">{token_text}</span>')
#                     else:
#                         result.append(token_text)
                
#                 elif style_type == 'heatmap':
#                     if token_idx < len(highlights):
#                         color = get_color_for_sim(highlights[token_idx], vmin, vmax, cmap)
#                         result.append(f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;">{token_text}</span>')
#                     else:
#                         result.append(token_text)
                
#                 elif style_type == 'binary':
#                     if token_idx < len(highlights):
#                         threshold, act_value = highlights[token_idx]
#                         if act_value >= threshold:
#                             result.append(f'<span style="background-color: white; color: black; padding: 2px; border-radius: 3px;">{token_text}</span>')
#                         else:
#                             result.append(f'<span style="background-color: black; color: black; padding: 2px; border-radius: 3px;">{token_text}</span>')
#                     else:
#                         result.append(token_text)
                
#                 last_pos = end
            
#             # Add any remaining text
#             if last_pos < len(text):
#                 result.append(text[last_pos:])
            
#             return ''.join(result)
        
#         # 1. Original sentence with GT highlights
#         gt_highlighted = create_highlighted_text(original_text, tokens, gt_token_mask, style_type='gt')
#         html_blocks.append(f"<div style='margin-bottom: 15px;'><b>Original Sentence with <i>{concept}</i> highlighted:</b> {gt_highlighted}</div>")
        
#         # 2. Raw heatmap
#         heatmap_highlighted = create_highlighted_text(original_text, tokens, concept_acts, 
#                                                      cmap=cmap, vmin=vmin, vmax=vmax, style_type='heatmap')
#         html_blocks.append(f"<div style='margin-bottom: 15px;'><b>Raw Concept Alignment Heatmap:</b> {heatmap_highlighted}</div>")
        
#         # 3. Binarized versions at different thresholds
#         for percentile in sorted_percentiles:
#             threshold = thresholds_dict[percentile]
            
#             # Create pairs of (threshold, activation) for binary highlighting
#             threshold_pairs = [(threshold, act_val) for act_val in concept_acts]
#             binary_highlighted = create_highlighted_text(original_text, tokens, threshold_pairs, style_type='binary')
#             html_blocks.append(f"<div style='margin-bottom: 15px;'><b>{percentile*100:.0f}%:</b> {binary_highlighted}</div>")
        
#         # Display everything
#         full_html = f"""
#         <div style='padding: 20px; background: #f9f9f9; border-radius: 10px;'>
#             {''.join(html_blocks)}
#         </div>
#         """
#         display(HTML(full_html))
        
#         # Save HTML if requested
#         if save_file:
#             if save_file.endswith('.html'):
#                 save_html_to_file(full_html, save_file)
#             elif save_file.endswith(('.png', '.pdf')):
#                 save_html_as_image(full_html, save_file)
#             else:
#                 print(f"Warning: save_file '{save_file}' not .html, .png, or .pdf. Use use_matplotlib=True for other formats.")
        
#         return
    
#     # Create figure with better spacing
#     num_rows = 2 + len(sorted_percentiles)  # Original + heatmap + threshold rows
#     fig_height = 1.0 + 1.0 + len(sorted_percentiles) * 0.8  # Height for single-line headers (no title)
#     fig, axes = plt.subplots(num_rows, 1, figsize=(12.5, fig_height), 
#                             gridspec_kw={'height_ratios': [0.8, 0.8] + [0.8]*len(sorted_percentiles),
#                                        'hspace': 0.02})  # Minimal spacing between rows
    
#     # Set white background
#     fig.patch.set_facecolor('white')
    
#     # Get original text from metadata
#     original_text = retrieve_sentence(sentence_idx, dataset_name)
#     if not original_text:
#         # Fallback to token-based display
#         original_text = " ".join(tokens)
    
#     # 1. Original sentence with GT highlights
#     ax = axes[0]
#     ax.axis('off')
    
#     # Display the full original text with inline highlighting
#     # Get token alignments
#     alignments = align_tokens_to_text(raw_tokens, original_text)
    
#     # Build text with matplotlib Text artists
#     # Right-aligned header position (same as other headers)
#     header_x = 0.12  # Minimal left edge with proper alignment
#     y_pos = 0.5
    
#     # Create the full header text with concept in italics
#     # We need to construct this differently to maintain right alignment
#     header_parts = ['Tweet (', f'{concept.capitalize()}', ' Highlighted):']
    
#     # First, create a temporary text to measure total width
#     full_temp_text = f'Tweet ({concept.capitalize()} Highlighted):'
#     temp_text = ax.text(header_x, y_pos, full_temp_text, transform=ax.transAxes, 
#                         fontsize=11, fontweight='bold', va='center', ha='right')
#     renderer = fig.canvas.get_renderer()
#     bbox = temp_text.get_window_extent(renderer=renderer)
#     inv = ax.transAxes.inverted()
#     bbox_data = inv.transform(bbox)
#     temp_text.remove()
    
#     # Calculate starting position for left-aligned rendering
#     text_width = bbox_data[1][0] - bbox_data[0][0]
#     start_x = header_x - text_width
    
#     # Add "Tweet (" 
#     ax.text(start_x, y_pos, 'Tweet (', transform=ax.transAxes, 
#             fontsize=11, fontweight='bold', va='center')
    
#     # Measure "Tweet (" width
#     temp_text = ax.text(start_x, y_pos, 'Tweet (', transform=ax.transAxes, 
#                         fontsize=11, fontweight='bold', va='center')
#     bbox = temp_text.get_window_extent(renderer=renderer)
#     bbox_data = inv.transform(bbox)
#     concept_x = bbox_data[1][0]
#     temp_text.remove()
    
#     # Add concept in italics
#     ax.text(concept_x, y_pos, f'{concept.capitalize()}', transform=ax.transAxes, 
#             fontsize=11, fontweight='bold', style='italic', va='center')
    
#     # Measure concept width
#     temp_text = ax.text(concept_x, y_pos, f'{concept.capitalize()}', transform=ax.transAxes, 
#                         fontsize=11, fontweight='bold', style='italic', va='center')
#     bbox = temp_text.get_window_extent(renderer=renderer)
#     bbox_data = inv.transform(bbox)
#     end_x = bbox_data[1][0]
#     temp_text.remove()
    
#     # Add " Highlighted):"
#     ax.text(end_x, y_pos, ' Highlighted):', transform=ax.transAxes, 
#             fontsize=11, fontweight='bold', va='center')
    
#     # Fixed x-position for all sentence text (aligned across all rows)
#     text_start_x = 0.14  # Minimal left edge with proper alignment
#     x_pos = text_start_x
#     y_pos = 0.5
    
#     # Get renderer for text measurement
#     renderer = fig.canvas.get_renderer()
#     inv = ax.transAxes.inverted()
    
#     # Just display the original text cleanly
#     ax.text(x_pos, y_pos, display_text, transform=ax.transAxes, 
#            fontsize=10, va='center')
    
#     # Skip the complex highlighting for now
#     if False:  # Placeholder for highlighting logic
#         last_pos = 0
#         for start, end, token_idx in token_positions:
#             # Add unhighlighted text before this token
#             if start > last_pos:
#                 text_part = display_text[last_pos:start]
#                 if text_part:
#                 t = ax.text(x_pos, y_pos, text_part, transform=ax.transAxes, 
#                            fontsize=10, va='center')
#                 bbox = t.get_window_extent(renderer=renderer)
#                 bbox_data = inv.transform(bbox)
#                 x_pos = bbox_data[1][0]
        
#             # Add token with or without highlight
#             token_text = display_text[start:end]
#         if token_idx < len(gt_token_mask) and gt_token_mask[token_idx]:
#             # Highlighted token with yellow background
#             t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
#                        fontsize=10, va='center', color='black', 
#                        bbox=dict(boxstyle='square,pad=0', facecolor='yellow', alpha=0.7, edgecolor='none'))
#         else:
#             # Normal token
#             t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
#                        fontsize=10, va='center')
        
#         bbox = t.get_window_extent(renderer=renderer)
#         bbox_data = inv.transform(bbox)
#         x_pos = bbox_data[1][0]
#         last_pos = end
        
#         # Wrap to next line if needed
#         if x_pos > 0.95:
#             x_pos = text_start_x
#             y_pos -= 0.25
    
#     # Add remaining text
#     if last_pos < len(display_text):
#         text_part = display_text[last_pos:]
#         if text_part:
#             ax.text(x_pos, y_pos, text_part, transform=ax.transAxes, 
#                    fontsize=10, va='center')
    
#     # 2. Raw heatmap
#     ax = axes[1]
#     ax.axis('off')
    
#     # Normalize colors
#     norm = Normalize(vmin=vmin, vmax=vmax)
#     cmap = plt.cm.get_cmap(cmap_name)
    
#     # Display on same line
#     # Right-aligned header position
#     header_x = 0.12  # Minimal left edge with proper alignment
#     y_pos = 0.5
    
#     # Add label - right aligned
#     ax.text(header_x, y_pos, 'Concept Alignment Heatmap:', transform=ax.transAxes, 
#             fontsize=11, fontweight='bold', va='center', ha='right')
    
#     # Fixed x-position for all sentence text (aligned across all rows)
#     text_start_x = 0.30  # Same as other sections
#     x_pos = text_start_x
#     y_pos = 0.5
    
#     # Get renderer for text measurement
#     renderer = fig.canvas.get_renderer()
#     inv = ax.transAxes.inverted()
    
#     # Add the sentence with color-coded activations
#     last_pos = 0
#     for start, end, token_idx in token_positions:
#         # Add unhighlighted text before this token
#             if start > last_pos:
#                 text_part = display_text[last_pos:start]
#                 if text_part:
#                 t = ax.text(x_pos, y_pos, text_part, transform=ax.transAxes, 
#                            fontsize=10, va='center')
#                 bbox = t.get_window_extent(renderer=renderer)
#                 bbox_data = inv.transform(bbox)
#                 x_pos = bbox_data[1][0]
        
#         # Add token with heatmap background color
#         token_text = display_text[start:end]
#         if token_idx < len(concept_acts):
#             color = cmap(norm(concept_acts[token_idx]))
#             # Convert matplotlib color to hex for background
#             import matplotlib.colors as mcolors
#             hex_color = mcolors.rgb2hex(color[:3])
#             t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
#                        fontsize=10, va='center', color='black',
#                        bbox=dict(boxstyle='square,pad=0', facecolor=hex_color, alpha=0.8, edgecolor='none'))
#         else:
#             t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
#                        fontsize=10, va='center')
        
#         bbox = t.get_window_extent(renderer=renderer)
#         bbox_data = inv.transform(bbox)
#         x_pos = bbox_data[1][0]
#         last_pos = end
        
#         # Wrap to next line if needed
#         if x_pos > 0.95:
#             x_pos = text_start_x
#             y_pos -= 0.25
    
#     # Add remaining text
#     if last_pos < len(display_text):
#         text_part = display_text[last_pos:]
#         if text_part:
#             ax.text(x_pos, y_pos, text_part, transform=ax.transAxes, 
#                    fontsize=10, va='center')
    
    
#     # 3. Binarized versions at different thresholds
#     for i, percentile in enumerate(sorted_percentiles):
#         ax = axes[2 + i]
#         ax.axis('off')
        
#         threshold = thresholds_dict[percentile]
        
#         # Display label
#         # Right-aligned header position
#         header_x = 0.12  # Minimal left edge with proper alignment
#         y_pos = 0.5
        
#         # Add label - right aligned
#         ax.text(header_x, y_pos, f'{percentile*100:.0f}%:', transform=ax.transAxes, 
#                 fontsize=11, fontweight='bold', va='center', ha='right')
        
#         # Fixed x-position for all sentence text (aligned across all rows)
#         text_start_x = 0.14  # Same as other sections
#         x_pos = text_start_x
        
#         # Get renderer for text measurement
#         renderer = fig.canvas.get_renderer()
#         inv = ax.transAxes.inverted()
        
#         # Add the sentence with binary highlighting
#         last_pos = 0
#         for start, end, token_idx in token_positions:
#             # Add unhighlighted text before this token
#                 if start > last_pos:
#                 text_part = display_text[last_pos:start]
#                 if text_part:
#                     t = ax.text(x_pos, y_pos, text_part, transform=ax.transAxes, 
#                                fontsize=10, va='center')
#                     bbox = t.get_window_extent(renderer=renderer)
#                     bbox_data = inv.transform(bbox)
#                     x_pos = bbox_data[1][0]
            
#             # Add token with binary background coloring
#             token_text = display_text[start:end]
#             if token_idx < len(concept_acts):
#                 if concept_acts[token_idx] >= threshold:
#                     # Active token - white background with black text
#                     t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
#                                fontsize=10, va='center', color='black',
#                                bbox=dict(boxstyle='square,pad=0', facecolor='white', alpha=0.9, edgecolor='none'))
#                 else:
#                     # Inactive token - completely blacked out
#                     t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
#                                fontsize=10, va='center', color='black',
#                                bbox=dict(boxstyle='square,pad=0', facecolor='black', alpha=1.0, edgecolor='none'))
#             else:
#                 t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
#                            fontsize=10, va='center')
            
#             bbox = t.get_window_extent(renderer=renderer)
#             bbox_data = inv.transform(bbox)
#             x_pos = bbox_data[1][0]
#             last_pos = end
            
#             # Wrap to next line if needed
#             if x_pos > 0.98:
#                 x_pos = 0.02
#                 y_pos -= 0.3
        
#         # Add remaining text
#         if last_pos < len(original_text):
#             text_part = original_text[last_pos:]
#             if text_part:
#                 ax.text(x_pos, y_pos, text_part, transform=ax.transAxes, 
#                        fontsize=10, va='center')
    
#     plt.tight_layout()
    
#     if save_file:
#         plt.savefig(save_file, dpi=500, format='pdf', bbox_inches='tight')
    
#     plt.show()


# def plot_multi_concept_heatmaps(
#     sentence_idx,
#     main_concept,
#     additional_concepts,
#     act_loader,
#     tokens_list,
#     dataset_name,
#     thresholds_dict=None,
#     gt_samples_per_concept=None,
#     cmap_name="magma",
#     vmin=None,
#     vmax=None,
#     save_file=None,
#     use_matplotlib=True,
#     show_colorbar_ticks=True,
#     metric_type="Concept Activation",
#     figsize=None
# ):
#     """
#     Creates a visualization showing multiple concept heatmaps for the same sentence.
#     """
#     from IPython.display import HTML, display
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as mpatches
#     from matplotlib.colors import Normalize
#     import re
    
#     def clean_token(token):
#         # Strip GPT/RoBERTa word boundary marker but keep the token
#         cleaned = token.replace("Ġ", "")
#         return cleaned if cleaned else "[EMPTY]"
    
#     # Get tokens and activations for this sentence
#     raw_tokens = tokens_list[sentence_idx]
#     tokens = [clean_token(tok) for tok in raw_tokens]
#     start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list)
    
#     # Get all concepts to process
#     all_concepts = [main_concept] + additional_concepts
    
#     # Load activations for this sentence and get activations for all concepts
#     sentence_acts = act_loader.load_tensor_range(start_idx, end_idx).cpu().numpy()
#     concept_activations = {}
    
#     for concept in all_concepts:
#         concept_idx = act_loader.get_concept_index(concept)
#         concept_activations[concept] = sentence_acts[:, concept_idx]
    
#     # Determine global vmin/vmax across all concepts if not provided
#     if vmin is None or vmax is None:
#         all_acts = [acts for acts in concept_activations.values()]
#         if vmin is None:
#             vmin = min(acts.min() for acts in all_acts)
#         if vmax is None:
#             vmax = max(acts.max() for acts in all_acts)
    
#     # Get ground truth indices for this sentence if available
#     gt_token_mask = [False] * len(tokens)
#     if gt_samples_per_concept is not None and main_concept in gt_samples_per_concept:
#         for idx in gt_samples_per_concept[main_concept]:
#             if start_idx <= idx < end_idx:
#                 gt_token_mask[idx - start_idx] = True
    
#     # Get original text from metadata
#     original_text = retrieve_sentence(sentence_idx, dataset_name)
#     if not original_text:
#         original_text = " ".join(tokens)
    
#     # Print BEFORE cleaning
#     print(f"[BEFORE CLEANING] Original text:")
#     print(f"  Text: {repr(original_text)}")
#     print(f"  Length: {len(original_text)}")
    
#     # Get token alignments to original text FIRST
#     alignments = align_tokens_to_text(raw_tokens, original_text)
    
#     # Clean up text AFTER alignment - remove double spaces and spaces before punctuation
#     cleaned_text = re.sub(r'\s+', ' ', original_text)  # Multiple spaces -> single space
#     cleaned_text = re.sub(r'\s+([.!?,:;])', r'\1', cleaned_text)  # Remove spaces before punctuation
#     cleaned_text = cleaned_text.strip()  # Remove leading/trailing spaces
    
#     # Print AFTER cleaning
#     print(f"[AFTER CLEANING] Cleaned text:")
#     print(f"  Text: {repr(cleaned_text)}")
#     print(f"  Length: {len(cleaned_text)}")
#     print(f"  Changes made: {original_text != cleaned_text}")
    
#     # Print alignment info
#     if alignments:
#         print(f"[ALIGNMENTS] Found {len(alignments)} token alignments")
#         if len(alignments) <= 5:
#             print(f"  Alignments: {alignments}")
#         else:
#             print(f"  First 5 alignments: {alignments[:5]}")
#     else:
#         print("[ALIGNMENTS] No alignments found!")
    
#     # Create figure
#     num_rows = 1 + len(all_concepts)  # Original + heatmap for each concept
#     if figsize is None:
#         fig_width = 12.5
#         fig_height = 1.0 + len(all_concepts) * 1.0
#         figsize = (fig_width, fig_height)
    
#     fig, axes = plt.subplots(num_rows, 1, figsize=figsize, 
#                             gridspec_kw={'height_ratios': [0.8] + [0.8]*len(all_concepts),
#                                        'hspace': 0.02})
    
#     # Handle single axis case
#     if num_rows == 1:
#         axes = [axes]
    
#     # Set white background
#     fig.patch.set_facecolor('white')
    
#     # 1. Original sentence with GT highlights
#     ax = axes[0]
#     ax.axis('off')
    
#     # Header
#     header_x = 0.12
#     ax.text(header_x, 0.6, 'Tweet', transform=ax.transAxes, 
#             fontsize=11, fontweight='bold', va='center', ha='right')
#     ax.text(header_x, 0.4, f'({main_concept.capitalize()} Highlighted)', transform=ax.transAxes, 
#             fontsize=11, fontweight='bold', va='center', ha='right')
    
#     # Display text with GT highlighting
#     text_start_x = 0.14
#     ax.text(text_start_x, 0.5, cleaned_text, transform=ax.transAxes, 
#            fontsize=10, va='center')
    
#     # 2. Heatmaps for all concepts
#     for i, concept in enumerate(all_concepts):
#         ax = axes[1 + i]
#         ax.axis('off')
        
#         concept_acts = concept_activations[concept]
        
#         # Header
#         ax.text(header_x, 0.5, f'{concept.capitalize()}:', transform=ax.transAxes, 
#                 fontsize=11, fontweight='bold', style='italic', va='center', ha='right')
        
#         # Display text (will add heatmap highlighting later)
#         ax.text(text_start_x, 0.5, cleaned_text, transform=ax.transAxes, 
#                fontsize=10, va='center')
    
#     # Add colorbar
#     fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.2, wspace=0.1, hspace=0.02)
#     cbar_width = 0.4
#     cbar_center = 0.5 - cbar_width / 2
#     cbar_ax = fig.add_axes([cbar_center, 0.08, cbar_width, 0.02])
    
#     from matplotlib.cm import ScalarMappable
#     norm = Normalize(vmin=vmin, vmax=vmax)
#     cmap = plt.cm.get_cmap(cmap_name)
#     colorbar_im = ScalarMappable(norm=norm, cmap=cmap)
    
#     cbar = fig.colorbar(colorbar_im, cax=cbar_ax, orientation='horizontal', label=metric_type)
    
#     if not show_colorbar_ticks:
#         cbar_ax.set_xticks([])
    
#     if save_file:
#         plt.savefig(save_file, dpi=500, format='pdf', bbox_inches='tight')
    
#     plt.show()


# def plot_sentence_similarity_heatmap(sentence_idx, embeddings, tokens_list):
#     """
#     Computes and visualizes pairwise cosine similarities between tokens in a sentence.

#     Args:
#         sentence_idx (int): Index of the sentence to visualize.
#         embeddings (torch.Tensor): Tensor of token embeddings.
#         tokens_list (List[List[str]]): List of sentences, where each sentence is a list of tokens.

#     Returns:
#         None: Displays a heatmap using seaborn.
#     """
#     # Get global token indices for the sentence
#     start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list)
    
#     # Extract embeddings and tokens for the sentence
#     sentence_embeddings = embeddings[start_idx:end_idx]
#     tokens = tokens_list[sentence_idx]
#     cleaned_tokens = remove_leading_token(tokens)
    
#     # Compute pairwise cosine similarities
#     n_tokens = len(tokens)
#     sim_matrix = torch.zeros(n_tokens, n_tokens)
    
#     for i in range(n_tokens):
#         for j in range(n_tokens):
#             sim = cosine_similarity(sentence_embeddings[i].unsqueeze(0), sentence_embeddings[j].unsqueeze(0))
#             sim_matrix[i, j] = sim.item()
    
#     # Create heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(sim_matrix.numpy(), 
#                 xticklabels=cleaned_tokens, 
#                 yticklabels=cleaned_tokens, 
#                 cmap='coolwarm', 
#                 center=0, 
#                 cbar_kws={'label': 'Cosine Similarity'})
#     plt.title(f'Token Similarity Matrix for Sentence {sentence_idx}')
#     plt.tight_layout()
#     plt.show()


# def plot_most_aligned_tokens(
#     concept, 
#     acts_loader, 
#     tokens_list, 
#     dataset_name,
#     k=20
# ):
#     """
#     Plots the most aligned tokens for a given concept.
#     Uses ChunkedActivationLoader for efficient memory usage.
    
#     Args:
#         concept (str): The concept to visualize
#         acts_loader: ChunkedActivationLoader instance
#         tokens_list: List of token lists
#         dataset_name (str): Name of the dataset
#         k (int): Number of top tokens to show
#     """
#     # Get concept index
#     concept_idx = acts_loader.get_concept_index(concept)
    
#     # Get all activations for this concept efficiently
#     all_acts = []
#     token_indices = []
    
#     # Process in chunks
#     for start_idx in range(0, acts_loader.total_samples, acts_loader.chunk_size):
#         end_idx = min(start_idx + acts_loader.chunk_size, acts_loader.total_samples)
#         chunk_acts = acts_loader.load_tensor_range(start_idx, end_idx)
#         concept_acts = chunk_acts[:, concept_idx].cpu().numpy()
#         all_acts.extend(concept_acts.tolist())
#         token_indices.extend(range(start_idx, end_idx))
    
#     # Get top-k tokens
#     top_indices = sorted(range(len(all_acts)), key=lambda i: all_acts[i], reverse=True)[:k]
    
#     # Get the actual tokens
#     flat_tokens_list = flatten_token_list(tokens_list)
#     top_tokens = [(flat_tokens_list[idx], all_acts[idx]) for idx in top_indices]
    
#     # Display results
#     print(f"\nTop {k} tokens most aligned with concept '{concept}':")
#     for i, (token, score) in enumerate(top_tokens):
#         print(f"{i+1}. '{token}': {score:.4f}")


# def plot_most_aligned_sentences(
#     concept,
#     dataset_name,
#     acts_loader,
#     tokens_list, 
#     k=10,
#     split='test',
#     aggregate='max'
# ):
#     """
#     Plots the top-k most aligned sentences (CLS embeddings) for a given concept.
    
#     Args:
#         concept (str): The concept to find aligned sentences for.
#         dataset_name (str): Name of the dataset.
#         acts_loader: ChunkedActivationLoader instance.
#         tokens_list: List of token lists.
#         k (int): Number of top sentences to display.
#         split (str): Which split to use ('train', 'val', 'test').
#         aggregate (str): How to aggregate token scores ('mean' or 'max').
#     """
#     # Get top sentences
#     top_sentences = get_sentences_by_metric(
#         dataset_name=dataset_name,
#         concept=concept,
#         acts_loader=acts_loader,
#         tokens_list=tokens_list,
#         split=split,
#         k=k,
#         aggregate=aggregate,
#         highest=True
#     )
    
#     print(f"\nTop {k} sentences most aligned with concept '{concept}' ({split} split, {aggregate} aggregation):")
#     print("=" * 80)
    
#     for rank, (sent_idx, score) in enumerate(top_sentences):
#         # Get the sentence tokens
#         tokens = tokens_list[sent_idx]
#         cleaned_tokens = remove_leading_token(tokens)
#         sentence = ' '.join(cleaned_tokens)
        
#         # Get sentence category if available
#         try:
#             category = get_sentence_category(sent_idx, dataset_name)
#             category_str = f" [{category}]"
#         except:
#             category_str = ""
        
#         print(f"\n{rank+1}. Sentence {sent_idx}{category_str} (Score: {score:.4f}):")
#         print(f"   {sentence}")
        
#         # Optionally show original text if available
#         try:
#             original = retrieve_sentence(sent_idx, dataset_name)
#             if original and original != sentence:
#                 print(f"   Original: {original}")
#         except:
#             pass


# def plot_tokens_in_context_byconcept(
#     sentence_idx,
#     concept,
#     acts_loader,
#     tokens_list,
#     dataset_name,
#     context_window=2,
#     show_scores=True
# ):
#     """
#     Visualize tokens in context colored by concept activation scores.
    
#     Args:
#         sentence_idx: Index of the sentence to visualize
#         concept: The concept to visualize
#         acts_loader: ChunkedActivationLoader instance
#         tokens_list: List of token lists
#         dataset_name: Name of the dataset
#         context_window: Number of sentences before/after to show
#         show_scores: Whether to show activation scores
#     """
#     from IPython.display import display, HTML
    
#     # Get concept index
#     concept_idx = acts_loader.get_concept_index(concept)
    
#     # Determine sentence range
#     start_sent = max(0, sentence_idx - context_window)
#     end_sent = min(len(tokens_list), sentence_idx + context_window + 1)
    
#     # Get global token indices
#     global_start_idx = sum(len(tokens_list[i]) for i in range(start_sent))
#     global_end_idx = sum(len(tokens_list[i]) for i in range(end_sent))
    
#     # Load activations
#     acts = acts_loader.load_tensor_range(global_start_idx, global_end_idx)
#     concept_acts = acts[:, concept_idx].cpu().numpy()
    
#     # Get min/max for coloring
#     vmin, vmax = concept_acts.min(), concept_acts.max()
    
#     # Create colormap
#     cmap = plt.cm.get_cmap('coolwarm')
#     norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
#     # Build HTML visualization
#     html_parts = []
#     html_parts.append(f"<h3>Concept: {concept}</h3>")
    
#     current_idx = 0
#     for sent_idx in range(start_sent, end_sent):
#         tokens = tokens_list[sent_idx]
#         cleaned_tokens = remove_leading_token(tokens)
        
#         # Get activations for this sentence
#         sent_acts = concept_acts[current_idx:current_idx + len(tokens)]
        
#         # Style for current sentence
#         if sent_idx == sentence_idx:
#             html_parts.append('<div style="background-color: #f0f0f0; padding: 5px; margin: 5px 0;">')
#             html_parts.append(f"<strong>Sentence {sent_idx}:</strong> ")
#         else:
#             html_parts.append(f"<div>Sentence {sent_idx}: ")
        
#         # Color each token
#         for token, score in zip(cleaned_tokens, sent_acts):
#             color = get_colormap_color(score, cmap, norm)
#             tooltip = f"{score:.3f}" if show_scores else ""
#             html_parts.append(
#                 f'<span style="background-color: {color}; padding: 2px 4px; '
#                 f'margin: 1px; border-radius: 3px;" title="{tooltip}">{token}</span>'
#             )
        
#         html_parts.append('</div>')
#         current_idx += len(tokens)
    
#     # Add colorbar
#     html_parts.append(plot_colorbar(vmin, vmax, 'coolwarm'))
    
#     # Display
#     display(HTML(''.join(html_parts)))


# def plot_tokens_by_activation_and_gt(
#     concept,
#     acts_loader,
#     tokens_list,
#     dataset_name,
#     gt_samples_per_concept,
#     n_examples=3,
#     context_words=10,
#     split='test'
# ):
#     """
#     Plot tokens showing paragraphs with most positive/negative/near-zero activations,
#     split by whether the paragraph contains ground truth tokens for the concept.
    
#     Args:
#         concept: The concept to analyze
#         acts_loader: ChunkedActivationLoader instance
#         tokens_list: List of token lists
#         dataset_name: Name of the dataset
#         gt_samples_per_concept: Dict mapping concepts to GT token indices
#         n_examples: Number of examples per category
#         context_words: Number of tokens to show as context
#         split: Which split to analyze
#     """
#     from IPython.display import display, HTML
#     from utils.general_utils import get_split_df
    
#     # Get concept index
#     concept_idx = acts_loader.get_concept_index(concept)
    
#     # Get split information
#     split_df = get_split_df(dataset_name)
#     test_indices = split_df[split_df == split].index.tolist()
    
#     # Get GT indices for this concept
#     gt_indices = set(gt_samples_per_concept.get(concept, []))
    
#     # Compute max activation per paragraph and track which have GT
#     paragraph_data = []
    
#     for sent_idx in test_indices:
#         if sent_idx >= len(tokens_list):
#             continue
            
#         # Get global token indices
#         start_idx = sum(len(tokens_list[i]) for i in range(sent_idx))
#         end_idx = start_idx + len(tokens_list[sent_idx])
        
#         # Load activations
#         acts = acts_loader.load_tensor_range(start_idx, end_idx)
#         concept_acts = acts[:, concept_idx].cpu().numpy()
#         max_act = concept_acts.max()
#         max_idx = concept_acts.argmax()
        
#         # Check if any token in this paragraph is GT
#         paragraph_token_indices = list(range(start_idx, end_idx))
#         has_gt = any(idx in gt_indices for idx in paragraph_token_indices)
        
#         paragraph_data.append({
#             'sent_idx': sent_idx,
#             'max_activation': max_act,
#             'max_token_idx': max_idx,
#             'has_gt': has_gt
#         })
    
#     # Split by GT status
#     gt_true = [p for p in paragraph_data if p['has_gt']]
#     gt_false = [p for p in paragraph_data if not p['has_gt']]
    
#     # Sort and get examples
#     gt_true_sorted = sorted(gt_true, key=lambda x: x['max_activation'], reverse=True)
#     gt_false_sorted = sorted(gt_false, key=lambda x: x['max_activation'], reverse=True)
    
#     # Get examples for each category
#     categories = {
#         'GT True - Most Positive': gt_true_sorted[:n_examples],
#         'GT True - Most Negative': gt_true_sorted[-n_examples:] if len(gt_true_sorted) >= n_examples else gt_true_sorted,
#         'GT True - Near Zero': sorted(gt_true, key=lambda x: abs(x['max_activation']))[:n_examples],
#         'GT False - Most Positive': gt_false_sorted[:n_examples],
#         'GT False - Most Negative': gt_false_sorted[-n_examples:] if len(gt_false_sorted) >= n_examples else gt_false_sorted,
#         'GT False - Near Zero': sorted(gt_false, key=lambda x: abs(x['max_activation']))[:n_examples]
#     }
    
#     # Create colormap
#     all_acts = []
#     for cat_data in categories.values():
#         for p in cat_data:
#             sent_idx = p['sent_idx']
#             start_idx = sum(len(tokens_list[i]) for i in range(sent_idx))
#             end_idx = start_idx + len(tokens_list[sent_idx])
#             acts = acts_loader.load_tensor_range(start_idx, end_idx)[:, concept_idx]
#             all_acts.extend(acts.cpu().numpy().tolist())
    
#     if all_acts:
#         vmin, vmax = min(all_acts), max(all_acts)
#     else:
#         vmin, vmax = -1, 1
    
#     cmap = plt.cm.get_cmap('coolwarm')
#     norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
#     # Display results
#     html_parts = []
#     html_parts.append(f"<h2>Token Activations for Concept: {concept}</h2>")
    
#     for category, examples in categories.items():
#         if not examples:
#             continue
            
#         html_parts.append(f"<h3>{category}</h3>")
        
#         for i, p in enumerate(examples):
#             sent_idx = p['sent_idx']
#             tokens = tokens_list[sent_idx]
#             cleaned_tokens = remove_leading_token(tokens)
            
#             # Get activations
#             start_idx = sum(len(tokens_list[j]) for j in range(sent_idx))
#             acts = acts_loader.load_tensor_range(start_idx, start_idx + len(tokens))
#             concept_acts = acts[:, concept_idx].cpu().numpy()
            
#             # Find context window around max activation
#             max_idx = p['max_token_idx']
#             context_start = max(0, max_idx - context_words)
#             context_end = min(len(tokens), max_idx + context_words + 1)
            
#             html_parts.append(f"<div style='margin: 10px 0; padding: 10px; background: #f5f5f5;'>")
#             html_parts.append(f"<strong>Sentence {sent_idx}</strong> (Max: {p['max_activation']:.3f})<br>")
            
#             # Show tokens with colors
#             for j in range(context_start, context_end):
#                 token = cleaned_tokens[j]
#                 score = concept_acts[j]
#                 color = get_colormap_color(score, cmap, norm)
                
#                 # Check if this is a GT token
#                 global_idx = start_idx + j
#                 is_gt = global_idx in gt_indices
#                 border = "2px solid gold" if is_gt else "none"
                
#                 # Highlight max token
#                 if j == max_idx:
#                     html_parts.append(
#                         f'<span style="background-color: {color}; padding: 2px 4px; '
#                         f'margin: 1px; border-radius: 3px; border: {border}; '
#                         f'font-weight: bold; text-decoration: underline;">{token}</span>'
#                     )
#                 else:
#                     html_parts.append(
#                         f'<span style="background-color: {color}; padding: 2px 4px; '
#                         f'margin: 1px; border-radius: 3px; border: {border};">{token}</span>'
#                     )
            
#             html_parts.append("</div>")
    
#     # Add colorbar
#     html_parts.append(plot_colorbar(vmin, vmax, 'coolwarm'))
    
#     # Add legend
#     html_parts.append('<div style="margin-top: 20px;"><strong>Legend:</strong> ')
#     html_parts.append('<span style="border: 2px solid gold; padding: 2px;">GT Token</span> ')
#     html_parts.append('<span style="text-decoration: underline;">Max Activation Token</span></div>')
    
#     # Display
#     display(HTML(''.join(html_parts)))


# # def plot_binarized_multiconcept_text(
# #     sentence_idx,
# #     concepts,
# #     act_loader,
# #     tokens_list,
# #     dataset_name,
# #     thresholds_dict,
# #     gt_samples_per_concept=None,
# #     cmap_name="coolwarm",
# #     vmin=None,
# #     vmax=None,
# #     save_file=None,
# #     show_colorbar_ticks=True,
# #     percentiles_to_show=None,
# #     use_matplotlib=True
# # ):
# #     """
# #     Creates a visualization showing multiple concepts at different threshold percentiles for a single sentence.
# #     Similar to plot_binarized_patchsims_with_raw_heatmaps but for text.
    
# #     Args:
# #         sentence_idx: Index of the sentence to visualize
# #         concepts: List of concept names to visualize (only first one will be highlighted)
# #         act_loader: Activation loader with concept activations
# #         tokens_list: List of token lists for all sentences
# #         dataset_name: Name of the dataset
# #         thresholds_dict: Dictionary mapping percentiles to threshold dictionaries
# #                         e.g., {0.9: {concept1: threshold1, concept2: threshold2}, ...}
# #         gt_samples_per_concept: Optional dict mapping concepts to GT token indices
# #         cmap_name: Colormap name for the heatmap
# #         vmin/vmax: Min/max values for colormap scaling (computed if None)
# #         save_file: Optional path to save the figure
# #         show_colorbar_ticks: Whether to show colorbar tick labels
# #         percentiles_to_show: List of percentiles to display (default: all in thresholds_dict)
# #         use_matplotlib: If True, use matplotlib. If False, use HTML display (default).
# #     """
# #     from IPython.display import HTML, display
# #     import matplotlib.pyplot as plt
# #     import matplotlib.patches as mpatches
# #     from matplotlib.colors import Normalize
# #     import matplotlib.gridspec as gridspec
    
# #     # Get tokens for this sentence
# #     raw_tokens = tokens_list[sentence_idx]
# #     tokens = [tok.replace("Ġ", "") for tok in raw_tokens]
# #     start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list)
    
# #     # Get original text
# #     original_text = retrieve_sentence(sentence_idx, dataset_name)
# #     use_original = original_text is not None
# #     if not use_original:
# #         # Fallback to tokens
# #         original_text = " ".join(tokens)
    
# #     # Load all activations for this sentence
# #     sentence_acts = act_loader.load_tensor_range(start_idx, end_idx).cpu().numpy()
    
# #     # Get activations for each concept
# #     concept_activations = {}
# #     for concept in concepts:
# #         concept_idx = act_loader.get_concept_index(concept)
# #         concept_activations[concept] = sentence_acts[:, concept_idx]
    
# #     # Determine vmin/vmax if not provided
# #     if vmin is None or vmax is None:
# #         all_acts = []
# #         for acts in concept_activations.values():
# #             all_acts.extend(acts)
# #         if vmin is None:
# #             vmin = min(all_acts)
# #         if vmax is None:
# #             vmax = max(all_acts)
    
# #     # Get sentence category
# #     sentence_class = get_sentence_category(sentence_idx, dataset_name)
    
# #     # Determine which percentiles to show
# #     if percentiles_to_show is None:
# #         percentiles_to_show = sorted(thresholds_dict.keys(), reverse=True)
    
# #     # Only use the first concept for highlighting
# #     primary_concept = concepts[0] if concepts else None
    
# #     # If using HTML display (default) and not saving to file
# #     if not use_matplotlib and not save_file:
# #         # Get colormap
# #         cmap = plt.cm.get_cmap(cmap_name)
# #         norm = Normalize(vmin=vmin, vmax=vmax)
        
# #         html_blocks = []
        
# #         # Header - match original format
# #         if primary_concept:
# #             html_blocks.append(f"<h3>Comment (<i>{primary_concept}</i> highlighted):</h3>")
# #         else:
# #             html_blocks.append(f"<h3>Comment:</h3>")
        
# #         # Function to create highlighted text using original text alignment
# #         def create_highlighted_text(text, tokens, highlights, concept=None, style_type='gt', threshold=None):
# #             """Map tokens back to original text and apply highlighting."""
# #             alignments = align_tokens_to_text(raw_tokens, text)
            
# #             result = []
# #             last_pos = 0
            
# #             for start, end, token_idx in token_positions:
# #                 # Add unhighlighted text before this token
# #                 if start > last_pos:
# #                     result.append(text[last_pos:start])
                
# #                 token_text = text[start:end]
                
# #                 if style_type == 'gt':
# #                     if token_idx < len(highlights) and highlights[token_idx]:
# #                         result.append(f'<span style="background-color: yellow; padding: 2px; border-radius: 3px;">{token_text}</span>')
# #                     else:
# #                         result.append(token_text)
                
# #                 elif style_type == 'heatmap':
# #                     if token_idx < len(highlights):
# #                         color = get_color_for_sim(highlights[token_idx], vmin, vmax, cmap)
# #                         result.append(f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;">{token_text}</span>')
# #                     else:
# #                         result.append(token_text)
                
# #                 elif style_type == 'binary':
# #                     if token_idx < len(highlights):
# #                         act_value = highlights[token_idx]
                        
# #                         if act_value >= threshold:
# #                             result.append(f'<span style="background-color: white; color: black; padding: 2px; border-radius: 3px; border: 1px solid #ccc;">{token_text}</span>')
# #                         else:
# #                             result.append(f'<span style="background-color: black; color: black; padding: 2px; border-radius: 3px;">{token_text}</span>')
# #                     else:
# #                         result.append(token_text)
                
# #                 last_pos = end
            
# #             # Add remaining text
# #             if last_pos < len(text):
# #                 result.append(text[last_pos:])
            
# #             return ''.join(result)
        
# #         # Only show for the primary concept
# #         if primary_concept:
# #             # 1. Ground truth annotations (if available)
# #             if gt_samples_per_concept and primary_concept in gt_samples_per_concept:
# #                 # Create GT mask for this concept
# #                 gt_mask = [False] * len(tokens)
# #                 for idx in gt_samples_per_concept[primary_concept]:
# #                     if start_idx <= idx < end_idx:
# #                         gt_mask[idx - start_idx] = True
                
# #                 gt_highlighted = create_highlighted_text(original_text, tokens, gt_mask, primary_concept, 'gt')
# #                 html_blocks.append(f"<div style='margin-bottom: 15px;'><b>Original Sentence with <i>{primary_concept}</i> highlighted:</b> {gt_highlighted}</div>")
            
# #             # 2. Raw activation heatmap
# #             heatmap_highlighted = create_highlighted_text(original_text, tokens, concept_activations[primary_concept], primary_concept, 'heatmap')
# #             html_blocks.append(f"<div style='margin-bottom: 15px;'><b>Raw Concept Alignment Heatmap:</b> {heatmap_highlighted}</div>")
            
# #             # 3. Binarized versions at different thresholds
# #             for percentile in percentiles_to_show:
# #                 # Get threshold
# #                 if isinstance(thresholds_dict[percentile], dict):
# #                     threshold_value = thresholds_dict[percentile].get(primary_concept, 0)
# #                     if isinstance(threshold_value, tuple):
# #                         threshold = threshold_value[0]
# #                     else:
# #                         threshold = threshold_value
# #                 else:
# #                     threshold = thresholds_dict[percentile][primary_concept][0]
                
# #                 binary_highlighted = create_highlighted_text(original_text, tokens, concept_activations[primary_concept], primary_concept, 'binary', threshold)
# #                 html_blocks.append(f"<div style='margin-bottom: 15px;'><b>{percentile*100:.0f}%:</b> {binary_highlighted}</div>")
        
# #         # Display everything
# #         full_html = f"""
# #         <div style='padding: 20px; background: #f9f9f9; border-radius: 10px;'>
# #             {''.join(html_blocks)}
# #         </div>
# #         """
# #         display(HTML(full_html))
        
# #         # Save HTML if requested
# #         if save_file:
# #             if save_file.endswith('.html'):
# #                 save_html_to_file(full_html, save_file)
# #             elif save_file.endswith(('.png', '.pdf')):
# #                 save_html_as_image(full_html, save_file)
# #             else:
# #                 print(f"Warning: save_file '{save_file}' not .html, .png, or .pdf. Use use_matplotlib=True for other formats.")
        
# #         return
    
# #     # Match the exact structure of plot_binarized_token_activations_with_raw_heatmap
# #     if primary_concept is None:
# #         print("No concept provided for visualization")
# #         return
        
# #     # Get data for the primary concept only
# #     concept_acts = concept_activations[primary_concept]
    
# #     # Get ground truth indices for this sentence if available
# #     gt_token_mask = [False] * len(tokens)
# #     if gt_samples_per_concept is not None and primary_concept in gt_samples_per_concept:
# #         for idx in gt_samples_per_concept[primary_concept]:
# #             if start_idx <= idx < end_idx:
# #                 gt_token_mask[idx - start_idx] = True
    
# #     # Sort thresholds by percentile (high to low)
# #     sorted_percentiles = sorted(percentiles_to_show, reverse=True)
    
# #     # Create figure with better spacing - match original exactly
# #     num_rows = 2 + len(sorted_percentiles)  # Original + heatmap + threshold rows
# #     fig_height = 1.0 + 1.0 + len(sorted_percentiles) * 0.8  # Height for single-line headers
# #     fig, axes = plt.subplots(num_rows, 1, figsize=(12.5, fig_height), 
# #                             gridspec_kw={'height_ratios': [0.8, 0.8] + [0.8]*len(sorted_percentiles),
# #                                        'hspace': 0.02})  # Minimal spacing between rows
    
# #     # Set white background
# #     fig.patch.set_facecolor('white')
    
# #     # Get token alignments
# #     raw_tokens = tokens_list[sentence_idx]
# #     alignments = align_tokens_to_text(raw_tokens, original_text)
    
# #     # 1. Original sentence with GT highlights - match exact formatting
# #     ax = axes[0]
# #     ax.axis('off')
    
# #     # Right-aligned header position (same as other headers)
# #     header_x = 0.12  # Minimal left edge with proper alignment
# #     y_pos = 0.5
    
# #     # Create the header text with concept in italics: "Comment (concept highlighted):"
# #     header_parts = ['Comment (', f'{primary_concept.capitalize()}', ' Highlighted):']
    
# #     # First, create a temporary text to measure total width
# #     full_temp_text = f'Comment ({primary_concept.capitalize()} Highlighted):'
# #     temp_text = ax.text(header_x, y_pos, full_temp_text, transform=ax.transAxes, 
# #                         fontsize=11, fontweight='bold', va='center', ha='right')
# #     renderer = fig.canvas.get_renderer()
# #     bbox = temp_text.get_window_extent(renderer=renderer)
# #     inv = ax.transAxes.inverted()
# #     bbox_data = inv.transform(bbox)
# #     temp_text.remove()
    
# #     # Calculate starting position for left-aligned rendering
# #     text_width = bbox_data[1][0] - bbox_data[0][0]
# #     start_x = header_x - text_width
    
# #     # Add "Comment (" 
# #     ax.text(start_x, y_pos, 'Comment (', transform=ax.transAxes, 
# #             fontsize=11, fontweight='bold', va='center')
    
# #     # Measure "Comment (" width
# #     temp_text = ax.text(start_x, y_pos, 'Comment (', transform=ax.transAxes, 
# #                         fontsize=11, fontweight='bold', va='center')
# #     bbox = temp_text.get_window_extent(renderer=renderer)
# #     bbox_data = inv.transform(bbox)
# #     concept_x = bbox_data[1][0]
# #     temp_text.remove()
    
# #     # Add concept in italics
# #     ax.text(concept_x, y_pos, f'{primary_concept.capitalize()}', transform=ax.transAxes, 
# #             fontsize=11, fontweight='bold', style='italic', va='center')
    
# #     # Measure concept width
# #     temp_text = ax.text(concept_x, y_pos, f'{primary_concept.capitalize()}', transform=ax.transAxes, 
# #                         fontsize=11, fontweight='bold', style='italic', va='center')
# #     bbox = temp_text.get_window_extent(renderer=renderer)
# #     bbox_data = inv.transform(bbox)
# #     end_x = bbox_data[1][0]
# #     temp_text.remove()
    
# #     # Add " Highlighted):"
# #     ax.text(end_x, y_pos, ' Highlighted):', transform=ax.transAxes, 
# #             fontsize=11, fontweight='bold', va='center')
    
# #     # Fixed x-position for all sentence text (aligned across all rows)
# #     text_start_x = 0.14  # Minimal left edge with proper alignment
# #     x_pos = text_start_x
# #     y_pos = 0.5
    
# #     # Get renderer for text measurement
# #     renderer = fig.canvas.get_renderer()
# #     inv = ax.transAxes.inverted()
    
# #     # Just display the original text cleanly
# #     ax.text(x_pos, y_pos, display_text, transform=ax.transAxes, 
# #            fontsize=10, va='center')
    
# #     # Skip the complex highlighting for now
# #     if False:  # Placeholder for highlighting logic
# #         last_pos = 0
# #         for start, end, token_idx in token_positions:
# #             # Add unhighlighted text before this token
# #             if start > last_pos:
# #                 text_part = display_text[last_pos:start]
# #                 if text_part:
# #                 t = ax.text(x_pos, y_pos, text_part, transform=ax.transAxes, 
# #                            fontsize=10, va='center')
# #                 bbox = t.get_window_extent(renderer=renderer)
# #                 bbox_data = inv.transform(bbox)
# #                 x_pos = bbox_data[1][0]
        
# #             # Add token with or without highlight
# #             token_text = display_text[start:end]
# #         if token_idx < len(gt_token_mask) and gt_token_mask[token_idx]:
# #             # Highlighted token with yellow background
# #             t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
# #                        fontsize=10, va='center', color='black', 
# #                        bbox=dict(boxstyle='square,pad=0', facecolor='yellow', alpha=0.7, edgecolor='none'))
# #         else:
# #             # Normal token
# #             t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
# #                        fontsize=10, va='center')
        
# #         bbox = t.get_window_extent(renderer=renderer)
# #         bbox_data = inv.transform(bbox)
# #         x_pos = bbox_data[1][0]
# #         last_pos = end
        
# #         # Wrap to next line if needed
# #         if x_pos > 0.95:
# #             x_pos = text_start_x
# #             y_pos -= 0.25
    
# #     # Add remaining text
# #     if last_pos < len(display_text):
# #         text_part = display_text[last_pos:]
# #         if text_part:
# #             ax.text(x_pos, y_pos, text_part, transform=ax.transAxes, 
# #                    fontsize=10, va='center')
    
# #     # 2. Raw heatmap - match exact formatting
# #     ax = axes[1]
# #     ax.axis('off')
    
# #     # Normalize colors
# #     norm = Normalize(vmin=vmin, vmax=vmax)
# #     cmap = plt.cm.get_cmap(cmap_name)
    
# #     # Display on same line
# #     # Right-aligned header position
# #     header_x = 0.12  # Minimal left edge with proper alignment
# #     y_pos = 0.5
    
# #     # Add label - right aligned
# #     ax.text(header_x, y_pos, 'Concept Alignment Heatmap:', transform=ax.transAxes, 
# #             fontsize=11, fontweight='bold', va='center', ha='right')
    
# #     # Fixed x-position for all sentence text (aligned across all rows)
# #     text_start_x = 0.30  # Same as other sections
# #     x_pos = text_start_x
# #     y_pos = 0.5
    
# #     # Get renderer for text measurement
# #     renderer = fig.canvas.get_renderer()
# #     inv = ax.transAxes.inverted()
    
# #     # Add the sentence with color-coded activations
# #     last_pos = 0
# #     for start, end, token_idx in token_positions:
# #         # Add unhighlighted text before this token
# #             if start > last_pos:
# #                 text_part = display_text[last_pos:start]
# #                 if text_part:
# #                 t = ax.text(x_pos, y_pos, text_part, transform=ax.transAxes, 
# #                            fontsize=10, va='center')
# #                 bbox = t.get_window_extent(renderer=renderer)
# #                 bbox_data = inv.transform(bbox)
# #                 x_pos = bbox_data[1][0]
        
# #         # Add token with heatmap background color
# #         token_text = display_text[start:end]
# #         if token_idx < len(concept_acts):
# #             color = cmap(norm(concept_acts[token_idx]))
# #             # Convert matplotlib color to hex for background
# #             import matplotlib.colors as mcolors
# #             hex_color = mcolors.rgb2hex(color[:3])
# #             t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
# #                        fontsize=10, va='center', color='black',
# #                        bbox=dict(boxstyle='square,pad=0', facecolor=hex_color, alpha=0.8, edgecolor='none'))
# #         else:
# #             t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
# #                        fontsize=10, va='center')
        
# #         bbox = t.get_window_extent(renderer=renderer)
# #         bbox_data = inv.transform(bbox)
# #         x_pos = bbox_data[1][0]
# #         last_pos = end
        
# #         # Wrap to next line if needed
# #         if x_pos > 0.95:
# #             x_pos = text_start_x
# #             y_pos -= 0.25
    
# #     # Add remaining text
# #     if last_pos < len(display_text):
# #         text_part = display_text[last_pos:]
# #         if text_part:
# #             ax.text(x_pos, y_pos, text_part, transform=ax.transAxes, 
# #                    fontsize=10, va='center')
    
# #     # 3. Binarized versions at different thresholds - match exact formatting
# #     for i, percentile in enumerate(sorted_percentiles):
# #         ax = axes[2 + i]
# #         ax.axis('off')
        
# #         # Get threshold
# #         if isinstance(thresholds_dict[percentile], dict):
# #             threshold_value = thresholds_dict[percentile].get(primary_concept, 0)
# #             if isinstance(threshold_value, tuple):
# #                 threshold = threshold_value[0]
# #             else:
# #                 threshold = threshold_value
# #         else:
# #             threshold = thresholds_dict[percentile][primary_concept][0]
        
# #         # Display label
# #         # Right-aligned header position
# #         header_x = 0.12  # Minimal left edge with proper alignment
# #         y_pos = 0.5
        
# #         # Add label - right aligned
# #         ax.text(header_x, y_pos, f'{percentile*100:.0f}%:', transform=ax.transAxes, 
# #                 fontsize=11, fontweight='bold', va='center', ha='right')
        
# #         # Fixed x-position for all sentence text (aligned across all rows)
# #         text_start_x = 0.14  # Same as other sections
# #         x_pos = text_start_x
        
# #         # Get renderer for text measurement
# #         renderer = fig.canvas.get_renderer()
# #         inv = ax.transAxes.inverted()
        
# #         # Add the sentence with binary highlighting
# #         last_pos = 0
# #         for start, end, token_idx in token_positions:
# #             # Add unhighlighted text before this token
# #                 if start > last_pos:
# #                 text_part = display_text[last_pos:start]
# #                 if text_part:
# #                     t = ax.text(x_pos, y_pos, text_part, transform=ax.transAxes, 
# #                                fontsize=10, va='center')
# #                     bbox = t.get_window_extent(renderer=renderer)
# #                     bbox_data = inv.transform(bbox)
# #                     x_pos = bbox_data[1][0]
            
# #             # Add token with binary background coloring
# #             token_text = display_text[start:end]
# #             if token_idx < len(concept_acts):
# #                 if concept_acts[token_idx] >= threshold:
# #                     # Active token - white background with black text
# #                     t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
# #                                fontsize=10, va='center', color='black',
# #                                bbox=dict(boxstyle='square,pad=0', facecolor='white', alpha=0.9, edgecolor='none'))
# #                 else:
# #                     # Inactive token - completely blacked out
# #                     t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
# #                                fontsize=10, va='center', color='black',
# #                                bbox=dict(boxstyle='square,pad=0', facecolor='black', alpha=1.0, edgecolor='none'))
# #             else:
# #                 t = ax.text(x_pos, y_pos, token_text, transform=ax.transAxes, 
# #                            fontsize=10, va='center')
            
# #             bbox = t.get_window_extent(renderer=renderer)
# #             bbox_data = inv.transform(bbox)
# #             x_pos = bbox_data[1][0]
# #             last_pos = end
            
# #             # Wrap to next line if needed
# #             if x_pos > 0.98:
# #                 x_pos = text_start_x
# #                 y_pos -= 0.25
        
# #         # Add remaining text
# #         if last_pos < len(original_text):
# #             text_part = original_text[last_pos:]
# #             if text_part:
# #                 ax.text(x_pos, y_pos, text_part, transform=ax.transAxes, 
# #                        fontsize=10, va='center')
    
# #     plt.tight_layout()
    
# #     if save_file:
# #         plt.savefig(save_file, dpi=500, format='pdf', bbox_inches='tight')
    
# #     plt.show()




# def plot_binarized_multiconcept_text_html(
#     sentence_idx,
#     concepts,
#     act_loader,
#     tokens_list,
#     dataset_name,
#     thresholds_dict,
#     gt_samples_per_concept=None,
#     cmap_name="coolwarm",
#     vmin=None,
#     vmax=None,
#     percentiles_to_show=None
# ):
#     """
#     HTML version of plot_binarized_multiconcept_text that uses original text with proper alignment.
    
#     Args:
#         sentence_idx: Index of the sentence to visualize
#         concepts: List of concept names to visualize
#         act_loader: Activation loader with concept activations
#         tokens_list: List of token lists for all sentences
#         dataset_name: Name of the dataset
#         thresholds_dict: Dictionary mapping percentiles to threshold dictionaries
#         gt_samples_per_concept: Optional dict mapping concepts to GT token indices
#         cmap_name: Colormap name for the heatmap
#         vmin/vmax: Min/max values for colormap scaling
#         percentiles_to_show: List of percentiles to display
#     """
#     from IPython.display import HTML, display
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import Normalize
    
#     # Get tokens and original text
#     raw_tokens = tokens_list[sentence_idx]
#     tokens = raw_tokens  # Keep raw tokens for alignment
#     start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list)
    
#     # Get original text
#     original_text = retrieve_sentence(sentence_idx, dataset_name)
#     if not original_text:
#         # Fallback
#         original_text = " ".join([t.replace("Ġ", "") for t in tokens if t.replace("Ġ", "") and t.replace("Ġ", "") != "[EMPTY]"])
    
#     # Get token alignments
#     alignments = align_tokens_to_text(tokens, original_text)
    
#     # Load all activations for this sentence
#     sentence_acts = act_loader.load_tensor_range(start_idx, end_idx).cpu().numpy()
    
#     # Get activations for each concept
#     concept_activations = {}
#     for concept in concepts:
#         concept_idx = act_loader.get_concept_index(concept)
#         concept_activations[concept] = sentence_acts[:, concept_idx]
    
#     # Determine vmin/vmax
#     if vmin is None or vmax is None:
#         all_acts = []
#         for acts in concept_activations.values():
#             all_acts.extend(acts)
#         if vmin is None:
#             vmin = min(all_acts)
#         if vmax is None:
#             vmax = max(all_acts)
    
#     # Get sentence category
#     sentence_class = get_sentence_category(sentence_idx, dataset_name)
    
#     # Determine which percentiles to show
#     if percentiles_to_show is None:
#         percentiles_to_show = sorted(thresholds_dict.keys(), reverse=True)
    
#     # Create colormap
#     cmap = plt.cm.get_cmap(cmap_name)
#     norm = Normalize(vmin=vmin, vmax=vmax)
    
#     # Helper function to create highlighted text
#     def create_concept_highlighted_text(text, concept, style='raw', threshold=None):
#         """Create highlighted text for a specific concept."""
#         acts = concept_activations[concept]
        
#         if not alignments:
#             # Fallback to token display
#             parts = []
#             for i, token in enumerate(tokens):
#                 cleaned = token.replace("Ġ", "")
#                 if not cleaned or cleaned == "[EMPTY]":
#                     continue
                    
#                 if i >= len(acts):
#                     parts.append(cleaned)
#                     continue
                
#                 if style == 'raw':
#                     color = get_color_for_sim(acts[i], vmin, vmax, cmap)
#                     parts.append(f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;">{cleaned}</span>')
#                 elif style == 'binary':
#                     if acts[i] >= threshold:
#                         parts.append(f'<span style="background-color: white; color: black; padding: 2px; border-radius: 3px;">{cleaned}</span>')
#                     else:
#                         parts.append(f'<span style="background-color: black; color: #666; padding: 2px; border-radius: 3px;">{cleaned}</span>')
#                 elif style == 'gt':
#                     global_idx = start_idx + i
#                     if gt_samples_per_concept and global_idx in gt_samples_per_concept.get(concept, []):
#                         parts.append(f'<span style="background-color: yellow; padding: 2px; border-radius: 3px;">{cleaned}</span>')
#                     else:
#                         parts.append(cleaned)
            
#             return ' '.join(parts)
        
#         # Use alignments to preserve original text
#         result = []
#         last_pos = 0
        
#         for start, end, token_idx in token_positions:
#             # Add unhighlighted text
#                 if start > last_pos:
#                 result.append(text[last_pos:start])
            
#             token_text = text[start:end]
            
#             if token_idx >= len(acts):
#                 result.append(token_text)
#                 last_pos = end
#                 continue
            
#             if style == 'raw':
#                 color = get_color_for_sim(acts[token_idx], vmin, vmax, cmap)
#                 result.append(f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;">{token_text}</span>')
#             elif style == 'binary':
#                 if acts[token_idx] >= threshold:
#                     result.append(f'<span style="background-color: white; color: black; padding: 2px; border-radius: 3px;">{token_text}</span>')
#                 else:
#                     result.append(f'<span style="background-color: black; color: #666; padding: 2px; border-radius: 3px;">{token_text}</span>')
#             elif style == 'gt':
#                 global_idx = start_idx + token_idx
#                 if gt_samples_per_concept and global_idx in gt_samples_per_concept.get(concept, []):
#                     result.append(f'<span style="background-color: yellow; padding: 2px; border-radius: 3px;">{token_text}</span>')
#                 else:
#                     result.append(token_text)
            
#             last_pos = end
        
#         # Add remaining text
#         if last_pos < len(text):
#             result.append(text[last_pos:])
        
#         return ''.join(result)
    
#     # Build HTML
#     html_parts = []
#     html_parts.append(f'<div style="font-family: Arial, sans-serif; padding: 20px; background: #f9f9f9; border-radius: 10px;">')
#     html_parts.append(f'<h2>Multi-Concept Analysis: Sentence {sentence_idx} ({sentence_class})</h2>')
    
#     # 1. Ground truth highlights (if available)
#     if gt_samples_per_concept:
#         html_parts.append('<div style="margin-bottom: 20px;">')
#         html_parts.append('<h3>Ground Truth Annotations</h3>')
#         for concept in concepts:
#             html_parts.append(f'<div style="margin-bottom: 10px;">')
#             html_parts.append(f'<strong>{concept}:</strong><br>')
#             html_parts.append(create_concept_highlighted_text(original_text, concept, style='gt'))
#             html_parts.append('</div>')
#         html_parts.append('</div>')
    
#     # 2. Raw activation heatmaps
#     html_parts.append('<div style="margin-bottom: 20px;">')
#     html_parts.append(f'<h3>Raw Activation Heatmaps (Range: [{vmin:.3f}, {vmax:.3f}])</h3>')
#     for concept in concepts:
#         html_parts.append(f'<div style="margin-bottom: 10px;">')
#         html_parts.append(f'<strong>{concept}:</strong><br>')
#         html_parts.append(create_concept_highlighted_text(original_text, concept, style='raw'))
#         html_parts.append('</div>')
#     html_parts.append('</div>')
    
#     # 3. Binarized at different thresholds
#     for percentile in percentiles_to_show:
#         html_parts.append(f'<div style="margin-bottom: 20px;">')
#         html_parts.append(f'<h3>{percentile*100:.0f}% Threshold</h3>')
        
#         for concept in concepts:
#             # Get threshold
#             if isinstance(thresholds_dict[percentile], dict):
#                 threshold_value = thresholds_dict[percentile].get(concept, 0)
#                 # Handle case where the value might be a tuple
#                 if isinstance(threshold_value, tuple):
#                     threshold = threshold_value[0]
#                 else:
#                     threshold = threshold_value
#             else:
#                 threshold = thresholds_dict[percentile][concept][0]
            
#             html_parts.append(f'<div style="margin-bottom: 10px;">')
#             html_parts.append(f'<strong>{concept} (threshold: {threshold:.3f}):</strong><br>')
#             html_parts.append(create_concept_highlighted_text(original_text, concept, style='binary', threshold=threshold))
#             html_parts.append('</div>')
        
#         html_parts.append('</div>')
    
#     # Add colorbar
#     colorbar_html = make_colorbar_image(vmin, vmax, cmap_name)
#     html_parts.append(f'<div style="margin-top: 20px;">{colorbar_html}</div>')
    
#     html_parts.append('</div>')
    
#     # Display
#     display(HTML(''.join(html_parts)))


# def filter_and_print_concept_texts(
#     metadata_path,
#     required_concepts,
#     dataset_name,
#     tokens_list,
#     gt_samples_per_concept=None,
#     chosen_split='test',
#     start_idx=0,
#     n_texts=5
# ):
#     """
#     Filters metadata for texts with specified concepts and prints them with highlighted tokens.
    
#     Args:
#         metadata_path (str): Path to metadata CSV.
#         required_concepts (list): Concept column names required to be 1 (e.g., ['sarcastic']).
#         dataset_name (str): Name of the dataset.
#         tokens_list (list): List of token lists for all sentences.
#         gt_samples_per_concept (dict): Optional dict mapping concepts to GT token indices.
#         chosen_split (str): Split to filter on ('train', 'test', etc.).
#         start_idx (int): Starting index in filtered results.
#         n_texts (int): Number of matching texts to display.
#     """
#     # Load metadata
#     metadata_df = pd.read_csv(metadata_path)
    
#     # Get unique file indices for sentences with required concepts
#     mask = metadata_df['split'] == chosen_split
#     for concept in required_concepts:
#         if concept not in metadata_df.columns:
#             print(f"Warning: Concept '{concept}' not found!")
#             return
#         mask &= metadata_df[concept] == 1
    
#     # Get unique sentence indices using file_idx
#     filtered_df = metadata_df[mask]
    
#     if 'file_idx' in filtered_df.columns:
#         sentence_indices = sorted(filtered_df['file_idx'].unique())
#     else:
#         # Fallback: get unique filenames and use their order
#         if 'sample_filename' in filtered_df.columns:
#             file_col = 'sample_filename'
#         else:
#             file_col = 'text_path'
        
#         # Get all unique files in order
#         all_files = metadata_df[file_col].unique()
#         file_to_idx = {f: i for i, f in enumerate(all_files)}
        
#         # Get indices for our filtered files
#         filtered_files = filtered_df[file_col].unique()
#         sentence_indices = sorted([file_to_idx[f] for f in filtered_files if f in file_to_idx])
    
#     total_matches = len(sentence_indices)
    
#     print(f"Found {total_matches} texts matching concepts: {required_concepts}")
#     print(f"Showing {min(n_texts, total_matches - start_idx)} texts from index {start_idx}")
    
#     # Debug: Check if gt_samples_per_concept has the requested concepts
#     if gt_samples_per_concept:
#         print(f"\nDebug - Available concepts in gt_samples_per_concept: {list(gt_samples_per_concept.keys())}")
#         for concept in required_concepts:
#             if concept in gt_samples_per_concept:
#                 print(f"  - '{concept}' has {len(gt_samples_per_concept[concept])} GT token indices")
#             else:
#                 print(f"  - '{concept}' NOT FOUND in gt_samples_per_concept")
#     else:
#         print("\nDebug - gt_samples_per_concept is None or empty")
    
#     print("=" * 80)
    
#     # Collect all HTML for display
#     html_blocks = []
    
#     # Display subset
#     display_indices = sentence_indices[start_idx:start_idx + n_texts]
    
#     for sent_idx in display_indices:
#         # Get tokens
#         if sent_idx >= len(tokens_list):
#             continue
        
#         # Try to get original text from metadata
#         original_text = None
#         if 'file_idx' in metadata_df.columns:
#             sent_metadata = metadata_df[metadata_df['file_idx'] == sent_idx].iloc[0] if not metadata_df[metadata_df['file_idx'] == sent_idx].empty else None
#         else:
#             sent_metadata = None
            
#         if sent_metadata is not None and 'sample_text' in sent_metadata:
#             original_text = sent_metadata['sample_text']
            
#         tokens = tokens_list[sent_idx]
        
#         # Get GT indices for highlighting
#         start_tok_idx, end_tok_idx = get_glob_tok_indices_from_sent_idx(sent_idx, tokens_list)
        
#         # Check which tokens should be highlighted
#         highlighted_token_indices = []
#         if gt_samples_per_concept:
#             for i in range(len(tokens)):
#                 global_tok_idx = start_tok_idx + i
#                 for concept in required_concepts:
#                     if concept in gt_samples_per_concept:
#                         if global_tok_idx in gt_samples_per_concept[concept]:
#                             highlighted_token_indices.append(i)
#                             break
        
#         # Debug: Check for tokens containing the concept word
#         debug_found_concept_tokens = []
#         for i, token in enumerate(tokens):
#             token_lower = token.replace("Ġ", "").lower()
#             for concept in required_concepts:
#                 if concept.lower() in token_lower:
#                     debug_found_concept_tokens.append((i, token, start_tok_idx + i))
        
#         if debug_found_concept_tokens and not highlighted_token_indices:
#             print(f"\nDebug - Sentence {sent_idx}: Found tokens containing concepts but not highlighted:")
#             for token_idx, token, global_idx in debug_found_concept_tokens:
#                 print(f"  - Token '{token}' at local index {token_idx}, global index {global_idx}")
        
#         # Create HTML for this sentence
#         if original_text and highlighted_token_indices:
#             # Use original text with alignment
#             token_positions = align_tokens_to_text(tokens, original_text)
            
#             html_parts = []
#             last_pos = 0
            
#             for token_idx in highlighted_token_indices:
#                 if token_idx < len(token_positions):
#                     start, end = token_positions[token_idx]
                    
#                     # Add text before this token
#                         if start > last_pos:
#                         html_parts.append(original_text[last_pos:start])
                    
#                     # Add highlighted token
#                     token_text = display_text[start:end]
#                     html_parts.append(f'<span style="background-color: yellow; padding: 2px; border-radius: 3px;">{token_text}</span>')
                    
#                     last_pos = end
            
#             # Add any remaining text
#             if last_pos < len(original_text):
#                 html_parts.append(original_text[last_pos:])
            
#             highlighted_text = ''.join(html_parts)
#         else:
#             # Fallback to token-based display
#             highlighted_parts = []
#             cleaned_tokens = [tok.replace("Ġ", "") for tok in tokens]
            
#             for i, token in enumerate(cleaned_tokens):
#                 if i in highlighted_token_indices:
#                     highlighted_parts.append(f'<span style="background-color: yellow; padding: 2px; border-radius: 3px;">{token}</span>')
#                 else:
#                     highlighted_parts.append(token)
            
#             highlighted_text = ' '.join(highlighted_parts)
        
#         html_blocks.append(f'<div style="margin-bottom: 15px;"><b>Sentence {sent_idx}:</b><br>{highlighted_text}</div>')
    
#     # Display all as HTML
#     full_html = f"""
#     <div style="padding: 20px; background: #f9f9f9; border-radius: 10px;">
#         {''.join(html_blocks)}
#     </div>
#     """
#     display(HTML(full_html))
    
#     print("\n" + "=" * 80)


# def save_html_to_file(html_content, filepath):
#     """
#     Save HTML content to a file with proper styling.
    
#     Args:
#         html_content: The HTML string to save
#         filepath: Path to save the HTML file
#     """
#     # Create a complete HTML document
#     full_html = f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <meta charset="UTF-8">
#         <title>Visualization</title>
#         <style>
#             body {{
#                 font-family: Arial, sans-serif;
#                 margin: 20px;
#                 background-color: white;
#             }}
#         </style>
#     </head>
#     <body>
#         {html_content}
#     </body>
#     </html>
#     """
    
#     with open(filepath, 'w', encoding='utf-8') as f:
#         f.write(full_html)
#     print(f"HTML saved to: {filepath}")


# def save_html_as_image(html_content, filepath, width=1200):
#     """
#     Convert HTML to an image file (PNG or PDF).
#     Requires: pip install playwright; playwright install chromium
    
#     Args:
#         html_content: The HTML string to convert
#         filepath: Path to save the image (e.g., 'output.png' or 'output.pdf')
#         width: Width of the viewport in pixels
#     """
#     try:
#         from playwright.sync_api import sync_playwright
#         import tempfile
#         import os
        
#         # Create temporary HTML file
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp:
#             tmp.write(f"""
#             <!DOCTYPE html>
#             <html>
#             <head>
#                 <meta charset="UTF-8">
#                 <style>
#                     body {{
#                         font-family: Arial, sans-serif;
#                         margin: 0;
#                         padding: 20px;
#                         background-color: white;
#                     }}
#                 </style>
#             </head>
#             <body>
#                 {html_content}
#             </body>
#             </html>
#             """)
#             tmp_path = tmp.name
        
#         # Convert to image using Playwright
#         with sync_playwright() as p:
#             browser = p.chromium.launch()
#             page = browser.new_page(viewport={'width': width, 'height': 800})
#             page.goto(f'file://{tmp_path}')
            
#             # Get the full height of the content
#             height = page.evaluate('document.body.scrollHeight')
#             page.set_viewport_size({'width': width, 'height': height})
            
#             # Save as image or PDF
#             if filepath.lower().endswith('.pdf'):
#                 page.pdf(path=filepath, format='A4')
#             else:
#                 page.screenshot(path=filepath, full_page=True)
            
#             browser.close()
        
#         # Clean up temp file
#         os.unlink(tmp_path)
#         print(f"Image saved to: {filepath}")
        
#     except ImportError:
#         print("Error: playwright not installed. Install with: pip install playwright && playwright install chromium")
#         print("Alternatively, you can save as HTML and open in a browser to save as PDF.")
        

# def plot_highlighted_text_for_pdf(
#     sentence_idx,
#     concept,
#     act_loader,
#     tokens_list,
#     dataset_name,
#     thresholds_dict,
#     gt_samples_per_concept=None,
#     cmap_name="coolwarm",
#     vmin=None,
#     vmax=None,
#     save_file=None,
#     dpi=300
# ):
#     """
#     Create a clean matplotlib visualization optimized for PDF export.
#     Uses a more modern, publication-ready style.
#     """
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as mpatches
#     from matplotlib.colors import Normalize
#     import textwrap
    
#     # Get tokens and activations
#     raw_tokens = tokens_list[sentence_idx]
#     tokens = [tok.replace("Ġ", " ") if tok.startswith("Ġ") else tok for tok in raw_tokens]
#     start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list)
    
#     # Get original text
#     original_text = retrieve_sentence(sentence_idx, dataset_name)
#     if not original_text:
#         original_text = "".join(tokens)
    
#     # Get concept activations
#     concept_idx = act_loader.get_concept_index(concept)
#     sentence_acts = act_loader.load_tensor_range(start_idx, end_idx).cpu().numpy()
#     concept_acts = sentence_acts[:, concept_idx]
    
#     # Auto-scale if needed
#     if vmin is None:
#         vmin = concept_acts.min()
#     if vmax is None:
#         vmax = concept_acts.max()
    
#     # Get GT mask
#     gt_mask = [False] * len(tokens)
#     if gt_samples_per_concept and concept in gt_samples_per_concept:
#         for idx in gt_samples_per_concept[concept]:
#             if start_idx <= idx < end_idx:
#                 gt_mask[idx - start_idx] = True
    
#     # Get sentence info
#     sentence_class = get_sentence_category(sentence_idx, dataset_name)
    
#     # Sort thresholds
#     sorted_percentiles = sorted(thresholds_dict.keys(), reverse=True)
    
#     # Create figure
#     n_sections = 2 + len(sorted_percentiles)
#     fig = plt.figure(figsize=(10, 2 + n_sections * 0.8), dpi=dpi)
#     fig.patch.set_facecolor('white')
    
#     # Title
#     fig.suptitle(f'Sentence {sentence_idx} ({sentence_class}) - Concept: {concept}', 
#                  fontsize=14, fontweight='bold', y=0.98)
    
#     # Use GridSpec for better control
#     import matplotlib.gridspec as gridspec
#     gs = gridspec.GridSpec(n_sections, 1, figure=fig, hspace=0.5, 
#                           top=0.94, bottom=0.02, left=0.05, right=0.95)
    
#     # Wrap text for display
#     wrapped_text = textwrap.fill(original_text, width=100)
    
#     # 1. Ground Truth Section
#     ax1 = fig.add_subplot(gs[0])
#     ax1.axis('off')
#     ax1.text(0, 0.8, "Ground Truth Annotations:", fontweight='bold', fontsize=12, 
#              transform=ax1.transAxes)
    
#     # Show wrapped text with GT highlights
#     y_pos = 0.5
#     x_pos = 0
#     for line in wrapped_text.split('\n'):
#         ax1.text(x_pos, y_pos, line, fontsize=10, transform=ax1.transAxes,
#                 fontfamily='monospace', wrap=True)
#         y_pos -= 0.25
    
#     # 2. Raw Heatmap Section
#     ax2 = fig.add_subplot(gs[1])
#     ax2.axis('off')
#     ax2.text(0, 0.8, f"Raw Activation Heatmap (Range: [{vmin:.3f}, {vmax:.3f}]):", 
#              fontweight='bold', fontsize=12, transform=ax2.transAxes)
    
#     # Create color boxes for each token
#     cmap = plt.cm.get_cmap(cmap_name)
#     norm = Normalize(vmin=vmin, vmax=vmax)
    
#     x_offset = 0
#     y_offset = 0.4
#     box_height = 0.15
    
#     for i, (token, act_val) in enumerate(zip(tokens, concept_acts)):
#         if x_offset > 0.9:  # Wrap to next line
#             x_offset = 0
#             y_offset -= 0.2
        
#         # Create colored rectangle
#         color = cmap(norm(act_val))
#         rect = mpatches.Rectangle((x_offset, y_offset), 0.08, box_height,
#                                  facecolor=color, edgecolor='none',
#                                  transform=ax2.transAxes)
#         ax2.add_patch(rect)
        
#         # Add token text
#         ax2.text(x_offset + 0.04, y_offset + box_height/2, token[:8], 
#                 fontsize=8, ha='center', va='center',
#                 transform=ax2.transAxes)
        
#         x_offset += 0.09
    
#     # 3. Binary threshold sections
#     for idx, percentile in enumerate(sorted_percentiles):
#         ax = fig.add_subplot(gs[2 + idx])
#         ax.axis('off')
        
#         threshold = thresholds_dict[percentile]
#         if isinstance(threshold, tuple):
#             threshold = threshold[0]
        
#         ax.text(0, 0.8, f"{percentile*100:.0f}% Threshold: {threshold:.3f}", 
#                 fontweight='bold', fontsize=12, transform=ax.transAxes)
        
#         # Binary visualization
#         x_offset = 0
#         y_offset = 0.4
        
#         for i, (token, act_val) in enumerate(zip(tokens, concept_acts)):
#             if x_offset > 0.9:
#                 x_offset = 0
#                 y_offset -= 0.2
            
#             # Binary coloring
#             if act_val >= threshold:
#                 bgcolor = 'white'
#                 fgcolor = 'black'
#                 edge = 'black'
#             else:
#                 bgcolor = 'black'
#                 fgcolor = 'gray'
#                 edge = 'gray'
            
#             rect = mpatches.Rectangle((x_offset, y_offset), 0.08, box_height,
#                                      facecolor=bgcolor, edgecolor=edge,
#                                      linewidth=0.5, transform=ax.transAxes)
#             ax.add_patch(rect)
            
#             ax.text(x_offset + 0.04, y_offset + box_height/2, token[:8], 
#                    fontsize=8, ha='center', va='center', color=fgcolor,
#                    transform=ax.transAxes)
            
#             x_offset += 0.09
    
#     # Add colorbar
#     cbar_ax = fig.add_axes([0.96, 0.3, 0.02, 0.4])
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, cax=cbar_ax)
#     cbar.set_label('Activation', fontsize=10)
    
#     plt.tight_layout()
    
#     if save_file:
#         plt.savefig(save_file, dpi=dpi, bbox_inches='tight', facecolor='white')
#         print(f"Saved to: {save_file}")
    
#     plt.show()


# def get_colormap_color(score, cmap, norm):
#     return matplotlib.colors.rgb2hex(cmap(norm(score)))


# def plot_colorbar(vmin=0.0, vmax=1.0, cmap_name="coolwarm", orientation="vertical"):
#     if orientation not in {"vertical", "horizontal"}:
#         raise ValueError("orientation must be 'vertical' or 'horizontal'")

#     figsize = (1.5, 0.5) if orientation == "vertical" else (5, 0.4)

#     fig, ax = plt.subplots(figsize=figsize)
#     cmap = matplotlib.colormaps.get_cmap(cmap_name)
#     norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

#     fig.subplots_adjust(left=0.2 if orientation == "vertical" else 0.3)
#     cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=orientation)
#     cb.set_label("Score")

#     # Convert plot to base64 for embedding
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", bbox_inches="tight")
#     buf.seek(0)
#     b64_img = base64.b64encode(buf.read()).decode("utf-8")
#     plt.close(fig)

#     return f'<img src="data:image/png;base64,{b64_img}" style="margin-top:10px; max-width:100%;" />'


# def highlight_tokens_with_legend(tokens, scores, cmap_name="coolwarm", vmin=None, vmax=None, include_colorbar=True):
#     if vmin is None:
#         vmin = min(scores)
#     if vmax is None:
#         vmax = max(scores)         
        
#     # Normalize scores and get colormap
#     norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
#     cmap = matplotlib.colormaps.get_cmap(cmap_name)
    
#     tokens = remove_leading_token(tokens)

#     html = ""
#     for token, score in zip(tokens, scores):
#         color = get_colormap_color(score, cmap, norm)
#         html += f'<span style="background-color:{color}; padding:2px 4px; margin:2px; border-radius:3px;">{token}</span> '

#     if include_colorbar:
#         colorbar_html = plot_colorbar(vmin=vmin, vmax=vmax, cmap_name=cmap_name)
#         html_block = f"""
#         <div style="display: flex; align-items: flex-start;">
#             <div style="flex: 1;">{html}</div>
#             <div style="padding-left: 20px;">{colorbar_html}</div>
#         </div>
#         """
#     else:
#         html_block = f"<div>{html}</div>"

#     return HTML(html_block)


# def plot_sentence_similarity_heatmap(sentence_idx, tokens_list, embeds, max_tokens=None):
#     """
#     Compute pairwise cosine similarities between all tokens in a specific sentence.

#     Args:
#         sentence_idx (int): The index of the sentence in tokens_list.
#         tokens_list (List[List[str]]): List of tokenized sentences.
#         embeds (torch.Tensor): Flattened tensor of shape (n_tokens, hidden_dim), aligned with tokens_list.
#         max_tokens (int, optional): Maximum number of tokens to include in the heatmap.
#     """
#     # Step 1: Determine the start and end index in the flattened embedding tensor
#     start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list)

#     sentence_tokens = remove_leading_token(tokens_list[sentence_idx])
#     sentence_embeds = embeds[start_idx:end_idx]

#     # Trim if max_tokens is set
#     if max_tokens is not None and len(sentence_tokens) > max_tokens:
#         sentence_tokens = sentence_tokens[:max_tokens]
#         sentence_embeds = sentence_embeds[:max_tokens]

#     # Normalize embeddings
#     sentence_embeds = torch.nn.functional.normalize(sentence_embeds, p=2, dim=1)

#     # Compute cosine similarity matrix
#     sim_matrix = torch.matmul(sentence_embeds, sentence_embeds.T).cpu().numpy()

#     # Plot
#     plt.figure(figsize=(len(sentence_tokens) * 0.5 + 6, len(sentence_tokens) * 0.5 + 6))
#     ax = sns.heatmap(sim_matrix, xticklabels=sentence_tokens, yticklabels=sentence_tokens,
#                      cmap="coolwarm", center=0, annot=True, fmt=".2f",
#                      linewidths=0.5, square=True, cbar_kws={"label": "Cosine Similarity"})
#     plt.title(f'Heatmap for Sentence {sentence_idx}')
#     plt.xticks(rotation=45, ha='right')
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.show()


# def plot_most_aligned_tokens(act_loader, tokens_list, dataset_name, concept=None, top_k=5):
#     """
#     Plot the most aligned tokens for a concept using act_loader.
    
#     Args:
#         act_loader: ChunkedActivationLoader instance
#         tokens_list: List of tokenized sentences
#         dataset_name: Name of dataset
#         concept: Concept to visualize (optional)
#         top_k: Number of top tokens to show
#     """
#     # Step 1: If no concept is passed, prompt the user
#     if concept is None:
#         concept = user_select_concept(act_loader.columns)

#     # Step 2: Get top token indices
#     top_token_indices = get_top_token_indices_for_concept(act_loader, tokens_list, concept, dataset_name, top_k)
    
#     # Step 3: Map token indices back to token strings
#     top_tokens = get_word_from_indices(top_token_indices, tokens_list)
    
#     # Step 4: Get the activation scores for these indices
#     concept_acts = act_loader.load_concept_activations_for_indices(concept, top_token_indices)
    
#     # Step 5: Plot tokens with similarity scores
#     display(highlight_tokens_with_legend(top_tokens, concept_acts.cpu().numpy(), vmin=0))


# def plot_most_aligned_sentences(act_loader, all_texts, dataset_name, concept=None, top_k=5, split='test'):
#     """
#     Plots the top-k most aligned sentences (CLS embeddings) for a given concept.

#     Args:
#         act_loader: ChunkedActivationLoader instance
#         all_texts (List[str]): Original sentences, aligned row-wise with activations
#         dataset_name (str): Dataset name for loading split info
#         concept (str): Concept to visualize
#         top_k (int): Number of top aligned sentences to return
#         split (str): One of 'train', 'test', 'cal', or 'both'
#     """
#     # Step 1: Choose concept if not passed
#     if concept is None:
#         concept = user_select_concept(act_loader.columns)

#     # Step 2: Get concept column index
#     concept_idx = act_loader.get_concept_index(concept)
    
#     # Step 3: Filter by split
#     if split != 'both':
#         split_df = get_split_df(dataset_name)
#         valid_indices = split_df[split_df == split].index.tolist()
#     else:
#         valid_indices = None
    
#     # Step 4: Find top-k sentences
#     import heapq
#     top_k_heap = []
#     chunk_size = 10000
    
#     for chunk_start in range(0, len(act_loader), chunk_size):
#         chunk_end = min(chunk_start + chunk_size, len(act_loader))
        
#         # Load chunk
#         chunk_tensor = act_loader.load_tensor_range(chunk_start, chunk_end)
#         concept_acts = chunk_tensor[:, concept_idx]
        
#         # Process each activation in the chunk
#         for i, activation in enumerate(concept_acts):
#             global_idx = chunk_start + i
            
#             # Skip if not in valid split
#             if valid_indices is not None and global_idx not in valid_indices:
#                 continue
            
#             # Use min heap to track top k values
#             if len(top_k_heap) < top_k:
#                 heapq.heappush(top_k_heap, (activation.item(), global_idx))
#             elif activation > top_k_heap[0][0]:
#                 heapq.heapreplace(top_k_heap, (activation.item(), global_idx))
        
#         # Clean up
#         del chunk_tensor, concept_acts
#         torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
#     # Step 5: Extract and sort results
#     top_items = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
    
#     # Step 6: Display sentences and scores
#     print(f"\nTop {top_k} sentences most aligned with concept '{concept}':\n")
#     for rank, (score, idx) in enumerate(top_items):
#         sentence = all_texts[idx]
#         print(f"[{rank+1}] Score: {score:.4f}")
#         print(f"     {sentence}\n")


# def plot_tokens_in_context_byconcept(
#     act_loader,
#     tokens_list,
#     dataset_name,
#     concept=None,
#     top_k=5,
#     top=True,
#     aggr_method='avg',
#     cmap_name="coolwarm",
#     split='test'
# ):
#     is_sentence_level = dataset_name == "Stanford-Tree-Bank"
#     unit_type = "sentence" if is_sentence_level else "paragraph"

#     if concept is None:
#         concept = user_select_concept(act_loader.columns)

#     samples = get_sentences_by_metric(act_loader, tokens_list, dataset_name, concept, top_k, top, aggr_method, split)

#     if top:
#         print(f"\nPlotting {unit_type}s MOST activated by {concept} ({aggr_method} over tokens)\n")
#     else:
#         print(f"\nPlotting {unit_type}s LEAST activated by {concept} ({aggr_method} over tokens)\n")

#     # Get concept column index
#     concept_idx = act_loader.get_concept_index(concept)
    
#     # Collect all scores to determine color scale
#     all_scores = []
#     for idx, _ in samples:
#         start_idx, end_idx = get_glob_tok_indices_from_sent_idx(idx, tokens_list)
#         sentence_acts = act_loader.load_tensor_range(start_idx, end_idx)
#         all_scores.extend(sentence_acts[:, concept_idx].cpu().numpy().tolist())
#     vmin, vmax = min(all_scores), max(all_scores)

#     html_blocks = []
#     for i, (idx, metric) in enumerate(samples):
#         category = get_sentence_category(idx, dataset_name)
#         start_idx, end_idx = get_glob_tok_indices_from_sent_idx(idx, tokens_list)
#         tokens = tokens_list[idx]
        
#         # Load activations for this sentence
#         sentence_acts = act_loader.load_tensor_range(start_idx, end_idx)
#         sims = sentence_acts[:, concept_idx].cpu().numpy().tolist()

#         # Title for each text unit with its concept score
#         title = f"<h4>Rank {i+1} : {unit_type.capitalize()} {idx} -- {category.capitalize()} ({aggr_method}={metric:.2f})</h4>"

#         html = highlight_tokens_with_legend(tokens, sims, cmap_name=cmap_name, vmin=vmin, vmax=vmax, include_colorbar=False)
#         html_blocks.append(f"{title}{html.data}")

#     # Append shared colorbar
#     colorbar_html = plot_colorbar(vmin=vmin, vmax=vmax, cmap_name=cmap_name, orientation="horizontal")

#     full_html = f"""
#     <div>
#         {''.join(html_blocks)}
#         <div style="margin-top: 10px;">{colorbar_html}</div>
#     </div>
#     """
#     display(HTML(full_html))


# def plot_tokens_by_activation_and_gt(
#     act_loader,
#     tokens_list,
#     dataset_name,
#     model_input_size,
#     concept=None,
#     n_examples=3,
#     cmap_name="coolwarm"
# ):
#     """
#     Plots tokens in context showing paragraphs with the most positive, negative, and near-zero 
#     maximum token activations, split by ground truth labels. Only includes paragraphs from the test split.
    
#     Args:
#         act_loader: ChunkedActivationLoader instance
#         tokens_list (List[List[str]]): List of tokenized sentences
#         dataset_name (str): Name of dataset
#         model_input_size (tuple): Model input size for loading GT samples
#         concept (str): Concept to visualize
#         n_examples (int): Number of examples per category (default: 3)
#         cmap_name (str): Colormap name for visualization
#     """
#     # Import needed functions
#     from collections import defaultdict
#     import os
    
#     # Select concept if not provided
#     if concept is None:
#         concept = user_select_concept(act_loader.columns)
    
#     # Get concept column index
#     concept_idx = act_loader.get_concept_index(concept)
    
#     # Load ground truth token indices (patches for text datasets)
#     gt_path = f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt'
#     if not os.path.exists(gt_path):
#         # Try alternative path for text datasets
#         gt_path = f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_text.pt'
    
#     if os.path.exists(gt_path):
#         gt_patches_per_concept = torch.load(gt_path, weights_only=False)
#         gt_indices = set(gt_patches_per_concept.get(concept, []))
#     else:
#         print(f"Warning: Could not find GT samples file at {gt_path}")
#         gt_indices = set()
    
#     # Get test split indices
#     split_df = get_split_df(dataset_name)
#     test_sentence_indices = split_df[split_df == 'test'].index.tolist()
    
#     # Compute max activation per paragraph and track which paragraphs have GT tokens
#     paragraph_data = []
#     idx = 0
#     for sent_idx, tokens in enumerate(tokens_list):
#         if sent_idx in test_sentence_indices:
#             start_idx = idx
#             end_idx = idx + len(tokens)
            
#             # Get activations for all tokens in this paragraph
#             paragraph_acts = act_loader.load_tensor_range(start_idx, end_idx)
#             concept_acts = paragraph_acts[:, concept_idx].cpu().numpy()
#             max_activation = concept_acts.max()
            
#             # Check if any token in this paragraph is GT
#             paragraph_token_indices = list(range(start_idx, end_idx))
#             has_gt = any(token_idx in gt_indices for token_idx in paragraph_token_indices)
            
#             paragraph_data.append({
#                 'sent_idx': sent_idx,
#                 'max_activation': max_activation,
#                 'has_gt': has_gt
#             })
#         idx += len(tokens)
    
#     # Split by ground truth
#     gt_true_paragraphs = [p for p in paragraph_data if p['has_gt']]
#     gt_false_paragraphs = [p for p in paragraph_data if not p['has_gt']]
    
#     # Sort by max activation
#     gt_true_sorted = sorted(gt_true_paragraphs, key=lambda x: x['max_activation'], reverse=True)
#     gt_false_sorted = sorted(gt_false_paragraphs, key=lambda x: x['max_activation'], reverse=True)
    
#     # Get examples for each category
#     categories = {}
    
#     # GT True categories
#     if len(gt_true_sorted) >= n_examples:
#         categories["GT True - Most Positive"] = gt_true_sorted[:n_examples]
#         categories["GT True - Most Negative"] = gt_true_sorted[-n_examples:]
#         # For near zero, sort by absolute value of max activation
#         gt_true_by_abs = sorted(gt_true_paragraphs, key=lambda x: abs(x['max_activation']))
#         categories["GT True - Near Zero"] = gt_true_by_abs[:n_examples]
#     else:
#         categories["GT True - Most Positive"] = gt_true_sorted
#         categories["GT True - Most Negative"] = []
#         categories["GT True - Near Zero"] = []
    
#     # GT False categories
#     if len(gt_false_sorted) >= n_examples:
#         categories["GT False - Most Positive"] = gt_false_sorted[:n_examples]
#         categories["GT False - Most Negative"] = gt_false_sorted[-n_examples:]
#         # For near zero, sort by absolute value of max activation
#         gt_false_by_abs = sorted(gt_false_paragraphs, key=lambda x: abs(x['max_activation']))
#         categories["GT False - Near Zero"] = gt_false_by_abs[:n_examples]
#     else:
#         categories["GT False - Most Positive"] = gt_false_sorted
#         categories["GT False - Most Negative"] = []
#         categories["GT False - Near Zero"] = []
    
#     # Collect all scores for consistent colormap scale
#     all_scores = []
#     for category_paragraphs in categories.values():
#         for paragraph in category_paragraphs:
#             sent_idx = paragraph['sent_idx']
#             start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sent_idx, tokens_list)
#             paragraph_acts = act_loader.load_tensor_range(start_idx, end_idx)
#             all_scores.extend(paragraph_acts[:, concept_idx].cpu().numpy().tolist())
    
#     if all_scores:
#         vmin, vmax = min(all_scores), max(all_scores)
#     else:
#         vmin, vmax = -1, 1  # Default range if no scores
    
#     # Create visualization
#     html_blocks = []
    
#     for category_name, paragraphs in categories.items():
#         if paragraphs:  # Only show categories with examples
#             html_blocks.append(f"<h3>{category_name}</h3>")
            
#             for i, paragraph in enumerate(paragraphs):
#                 sent_idx = paragraph['sent_idx']
#                 max_activation = paragraph['max_activation']
#                 start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sent_idx, tokens_list)
                
#                 tokens = tokens_list[sent_idx]
#                 paragraph_acts = act_loader.load_tensor_range(start_idx, end_idx)
#                 sims = paragraph_acts[:, concept_idx].cpu().numpy().tolist()
                
#                 # Find which token has the max activation
#                 max_token_idx = np.argmax(sims)
#                 max_token = remove_leading_token([tokens[max_token_idx]])[0]
                
#                 # Get sentence category
#                 category = get_sentence_category(sent_idx, dataset_name)
                
#                 # Create title with paragraph info
#                 title = f"<h4>Example {i+1}: Paragraph {sent_idx} - {category} (max token: '{max_token}' = {max_activation:.3f})</h4>"
                
#                 # Create highlighted sentence
#                 html = highlight_tokens_with_legend(tokens, sims, cmap_name=cmap_name, vmin=vmin, vmax=vmax, include_colorbar=False)
                
#                 # Add marker for the max token
#                 html_with_marker = html.data.replace(
#                     f'>{max_token}</span>',
#                     f' style="border: 3px solid black; font-weight: bold;">{max_token}</span>'
#                 )
                
#                 html_blocks.append(f"{title}{html_with_marker}")
    
#     # Add shared colorbar
#     colorbar_html = plot_colorbar(vmin=vmin, vmax=vmax, cmap_name=cmap_name, orientation="horizontal")
    
#     # Print summary statistics
#     print(f"\nConcept: {concept}")
#     print(f"GT True paragraphs in test set: {len(gt_true_paragraphs)}")
#     print(f"GT False paragraphs in test set: {len(gt_false_paragraphs)}")
#     print(f"Activation range: [{vmin:.3f}, {vmax:.3f}]")
    
#     full_html = f"""
#     <div>
#         {''.join(html_blocks)}
#         <div style="margin-top: 20px;">{colorbar_html}</div>
#     </div>
#     """
#     display(HTML(full_html))



def plot_multi_concept_heatmaps(
    sentence_idx,
    main_concept,
    additional_concepts,
    act_loader,
    tokens_list,
    dataset_name,
    thresholds_dict=None,
    gt_samples_per_concept=None,
    cmap_name="magma",
    vmin=None,
    vmax=None,
    save_file=None,
    show_colorbar_ticks=True,
    metric_type="Concept Activation",
    figsize=None
):
    """
    Creates a matplotlib visualization showing multiple concept heatmaps for the same sentence.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    import re
    from utils.general_utils import get_paper_plotting_style
    
    # Apply paper plotting style
    plt.rcParams.update(get_paper_plotting_style())
    
    # Get tokens and activations for this sentence
    raw_tokens = tokens_list[sentence_idx]
    start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list)
    
    # Get all concepts to process
    all_concepts = [main_concept] + additional_concepts
    
    # Load activations for this sentence and get activations for all concepts
    sentence_acts = act_loader.load_tensor_range(start_idx, end_idx).cpu().numpy()
    concept_activations = {}
    
    for concept in all_concepts:
        concept_idx = act_loader.get_concept_index(concept)
        concept_activations[concept] = sentence_acts[:, concept_idx]
    
    # Determine global vmin/vmax across all concepts if not provided
    if vmin is None or vmax is None:
        all_acts = [acts for acts in concept_activations.values()]
        if vmin is None:
            vmin = min(acts.min() for acts in all_acts)
        if vmax is None:
            vmax = max(acts.max() for acts in all_acts)
    
    # Get ground truth indices for this sentence if available
    gt_token_mask = [False] * len(raw_tokens)
    if gt_samples_per_concept is not None and main_concept in gt_samples_per_concept:
        for idx in gt_samples_per_concept[main_concept]:
            if start_idx <= idx < end_idx:
                local_idx = idx - start_idx
                gt_token_mask[local_idx] = True
    
    # Build text from tokens and create position mapping
    # Step 1: Properly reconstruct text handling subword tokens
    # The Ġ marker indicates the start of a new word (space before)
    # Tokens without Ġ are continuations of the previous token
    cleaned_text = ""
    for i, token in enumerate(raw_tokens):
        if i == 0:
            # First token - just remove Ġ if present
            cleaned_text = token.replace("Ġ", "")
        else:
            if token.startswith("Ġ"):
                # This token starts a new word - add space and remove Ġ
                cleaned_text += " " + token[1:]  # Remove the Ġ marker
            else:
                # This token continues the previous word - no space
                cleaned_text += token
    
    # Step 2: Final cleanup
    # Fix spaces before punctuation
    cleaned_text = re.sub(r"\s+([\\.!?,:;])", r"\1", cleaned_text)
    # Fix multiple spaces (shouldn't happen with our logic, but just in case)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    # Trim
    cleaned_text = cleaned_text.strip()
    
    print(f"[AFTER CLEANING] Cleaned text:")
    print(f"  Text: {repr(cleaned_text)}")
    print(f"  Length: {len(cleaned_text)}")
    
    # Step 3: Create token position mapping for the cleaned text
    # Since cleaned_text may have had punctuation adjustments, we need to find
    # where each token actually ended up in the final text
    token_positions = []
    
    # For each token, find its position in the cleaned text
    # We'll search sequentially to maintain order
    search_pos = 0
    
    for i, token in enumerate(raw_tokens):
        # Remove Ġ marker to get the actual token text
        clean_token = token.replace("Ġ", "")
        
        if clean_token:  # Skip empty tokens
            # Find this token in the cleaned text starting from our current position
            token_start = cleaned_text.find(clean_token, search_pos)
            
            if token_start != -1:
                token_end = token_start + len(clean_token)
                token_positions.append((token_start, token_end, i))
                # Update search position to after this token
                search_pos = token_end
            else:
                # Token not found - this shouldn't happen with our reconstruction
                print(f"[WARNING] Could not find token {i}: \"{clean_token}\" in cleaned text")
                # Use current position as a fallback
                token_positions.append((search_pos, search_pos, i))
    
    print(f"[TOKEN POSITIONS] Found {len(token_positions)} token positions")
    if len(token_positions) <= 10:
        print(f"  Positions: {token_positions}")
    else:
        print(f"  First 10 positions: {token_positions[:10]}")
    
    # Create figure
    num_rows = 1 + len(all_concepts)
    if figsize is None:
        fig_width = 12.5
        fig_height = 1.0 + len(all_concepts) * 1.0
        figsize = (fig_width, fig_height)
    
    fig, axes = plt.subplots(num_rows, 1, figsize=figsize, 
                            gridspec_kw={"height_ratios": [0.8] + [0.8]*len(all_concepts),
                                       "hspace": 0.02})
    
    if num_rows == 1:
        axes = [axes]
    
    fig.patch.set_facecolor("white")
    
    # Constants for layout
    header_x = 0.12
    text_start_x = 0.14
    
    # Get renderer for text measurement
    renderer = fig.canvas.get_renderer()
    
    # 1. Original sentence with GT highlights
    ax = axes[0]
    ax.axis("off")
    
    # Header
    ax.text(header_x, 0.6, "Tweet", transform=ax.transAxes, 
            fontsize=11, fontweight="bold", va="center", ha="right")
    ax.text(header_x, 0.4, f"({main_concept.capitalize()} Highlighted)", transform=ax.transAxes, 
            fontsize=11, fontweight="bold", va="center", ha="right")
    
    # Display text with GT highlighting
    x_pos = text_start_x
    y_pos = 0.5
    last_pos = 0
    
    
    # Simple approach: just display the cleaned text as a single string
    # For highlighting, we'll use a monospace font and add colored backgrounds
    
    # First, let's build a list of text segments with their properties
    segments = []
    current_pos = 0
    
    # Sort positions to process in order
    sorted_positions = sorted(token_positions, key=lambda x: x[0])
    
    for start, end, token_idx in sorted_positions:
        # Add any text before this token
        if start > current_pos:
            segments.append({
                'text': cleaned_text[current_pos:start],
                'highlight': False
            })
        
        # Add the token
        segments.append({
            'text': cleaned_text[start:end],
            'highlight': token_idx < len(gt_token_mask) and gt_token_mask[token_idx]
        })
        current_pos = end
    
    # Add any remaining text
    if current_pos < len(cleaned_text):
        segments.append({
            'text': cleaned_text[current_pos:],
            'highlight': False
        })
    
    # Now display all segments
    for segment in segments:
        if segment['highlight']:
            t = ax.text(x_pos, y_pos, segment['text'], transform=ax.transAxes, 
                       fontsize=10, va="center", family='monospace',
                       bbox=dict(boxstyle="square,pad=0.1", facecolor="yellow", alpha=0.7, edgecolor="none"))
        else:
            t = ax.text(x_pos, y_pos, segment['text'], transform=ax.transAxes, 
                       fontsize=10, va="center", family='monospace')
        
        # Move x position for next segment
        # Use monospace assumption: each character has same width
        char_width = 0.007  # Approximate width per character in axes coordinates
        x_pos += len(segment['text']) * char_width
    
    # 2. Heatmaps for all concepts
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap(cmap_name)
    
    for i, concept in enumerate(all_concepts):
        ax = axes[1 + i]
        ax.axis("off")
        
        concept_acts = concept_activations[concept]
        
        # Header
        ax.text(header_x, 0.5, f"{concept.capitalize()}:", transform=ax.transAxes, 
                fontsize=11, fontweight="bold", style="italic", va="center", ha="right")
        
        # Display text with heatmap coloring
        x_pos = text_start_x
        y_pos = 0.5
        last_pos = 0
        
        # Build segments with heatmap colors
        import matplotlib.colors as mcolors
        segments = []
        current_pos = 0
        
        # Sort token positions by start index
        sorted_positions = sorted(token_positions, key=lambda x: x[0])
        
        for start, end, token_idx in sorted_positions:
            # Add any text before this token
            if start > current_pos:
                segments.append({
                    'text': cleaned_text[current_pos:start],
                    'color': None,
                    'superdetector': False
                })
            
            # Add the token with its color
            if token_idx < len(concept_acts):
                color = cmap(norm(concept_acts[token_idx]))
                hex_color = mcolors.rgb2hex(color[:3])
                is_superdetector = (thresholds_dict is not None and 
                                  concept in thresholds_dict and 
                                  concept_acts[token_idx] >= thresholds_dict[concept])
                segments.append({
                    'text': cleaned_text[start:end],
                    'color': hex_color,
                    'superdetector': is_superdetector
                })
            else:
                segments.append({
                    'text': cleaned_text[start:end],
                    'color': None,
                    'superdetector': False
                })
            current_pos = end
        
        # Add any remaining text
        if current_pos < len(cleaned_text):
            segments.append({
                'text': cleaned_text[current_pos:],
                'color': None,
                'superdetector': False
            })
        
        # Display all segments
        for segment in segments:
            if segment['color']:
                if segment['superdetector']:
                    t = ax.text(x_pos, y_pos, segment['text'], transform=ax.transAxes, 
                               fontsize=10, va="center", family='monospace', color="black",
                               bbox=dict(boxstyle="square,pad=0.1", facecolor=segment['color'], 
                                       alpha=0.8, edgecolor="deepskyblue", linewidth=2))
                else:
                    t = ax.text(x_pos, y_pos, segment['text'], transform=ax.transAxes, 
                               fontsize=10, va="center", family='monospace', color="black",
                               bbox=dict(boxstyle="square,pad=0.1", facecolor=segment['color'], 
                                       alpha=0.8, edgecolor="none"))
            else:
                t = ax.text(x_pos, y_pos, segment['text'], transform=ax.transAxes, 
                           fontsize=10, va="center", family='monospace')
            
            # Move x position for next segment
            x_pos += len(segment['text']) * 0.007
    
    # Add colorbar
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.2, wspace=0.1, hspace=0.02)
    cbar_width = 0.4
    cbar_center = 0.5 - cbar_width / 2
    cbar_ax = fig.add_axes([cbar_center, 0.08, cbar_width, 0.02])
    
    from matplotlib.cm import ScalarMappable
    colorbar_im = ScalarMappable(norm=norm, cmap=cmap)
    
    cbar = fig.colorbar(colorbar_im, cax=cbar_ax, orientation="horizontal", label=metric_type)
    
    if not show_colorbar_ticks:
        cbar_ax.set_xticks([])
    
    if save_file:
        plt.savefig(save_file, dpi=500, format="pdf", bbox_inches="tight")
    
    plt.show()


def filter_and_print_concept_texts(
    metadata_path,
    required_concepts,
    dataset_name,
    tokens_list,
    gt_samples_per_concept=None,
    chosen_split='test',
    start_idx=0,
    n_texts=5
):
    """
    Filters metadata for texts with specified concepts and prints them with highlighted tokens.
    
    Args:
        metadata_path (str): Path to metadata CSV.
        required_concepts (list): Concept column names required to be 1 (e.g., ['sarcasm']).
        dataset_name (str): Name of the dataset.
        tokens_list (list): List of token lists for all sentences.
        gt_samples_per_concept (dict): Optional dict mapping concepts to GT token indices.
        chosen_split (str): Split to filter on ('train', 'test', etc.).
        start_idx (int): Starting index in filtered results.
        n_texts (int): Number of matching texts to display.
    """
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    
    # Get unique file indices for sentences with required concepts
    mask = metadata_df['split'] == chosen_split
    for concept in required_concepts:
        if concept not in metadata_df.columns:
            print(f"Warning: Concept '{concept}' not found!")
            return
        mask &= metadata_df[concept] == 1
    
    # Get unique sentence indices using file_idx
    filtered_df = metadata_df[mask]
    
    if 'file_idx' in filtered_df.columns:
        sentence_indices = sorted(filtered_df['file_idx'].unique())
    else:
        # Fallback: get unique filenames and use their order
        if 'sample_filename' in filtered_df.columns:
            file_col = 'sample_filename'
        else:
            file_col = 'text_path'
        
        # Get all unique files in order
        all_files = metadata_df[file_col].unique()
        file_to_idx = {f: i for i, f in enumerate(all_files)}
        
        # Get indices for our filtered files
        filtered_files = filtered_df[file_col].unique()
        sentence_indices = sorted([file_to_idx[f] for f in filtered_files if f in file_to_idx])
    
    total_matches = len(sentence_indices)
    
    print(f"Found {total_matches} texts matching concepts: {required_concepts}")
    print(f"Showing {min(n_texts, total_matches - start_idx)} texts from index {start_idx}")
    
    # Debug: Check if gt_samples_per_concept has the requested concepts
    if gt_samples_per_concept:
        print(f"\nDebug - Available concepts in gt_samples_per_concept: {list(gt_samples_per_concept.keys())}")
        for concept in required_concepts:
            if concept in gt_samples_per_concept:
                print(f"  - '{concept}' has {len(gt_samples_per_concept[concept])} GT token indices")
            else:
                print(f"  - '{concept}' NOT FOUND in gt_samples_per_concept")
    else:
        print("\nDebug - gt_samples_per_concept is None or empty")
    
    print("=" * 80)
    
    # Collect all HTML for display
    html_blocks = []
    
    # Display subset
    display_indices = sentence_indices[start_idx:start_idx + n_texts]
    
    for sent_idx in display_indices:
        # Get tokens
        if sent_idx >= len(tokens_list):
            continue
        
        # Try to get original text from metadata
        original_text = None
        if 'file_idx' in metadata_df.columns:
            sent_metadata = metadata_df[metadata_df['file_idx'] == sent_idx].iloc[0] if not metadata_df[metadata_df['file_idx'] == sent_idx].empty else None
        else:
            sent_metadata = None
            
        if sent_metadata is not None and 'sample_text' in sent_metadata:
            original_text = sent_metadata['sample_text']
            
        tokens = tokens_list[sent_idx]
        
        # Get GT indices for highlighting
        start_tok_idx, end_tok_idx = get_glob_tok_indices_from_sent_idx(sent_idx, tokens_list)
        
        # Check which tokens should be highlighted
        highlighted_token_indices = []
        if gt_samples_per_concept:
            for i in range(len(tokens)):
                global_tok_idx = start_tok_idx + i
                for concept in required_concepts:
                    if concept in gt_samples_per_concept:
                        if global_tok_idx in gt_samples_per_concept[concept]:
                            highlighted_token_indices.append(i)
                            break
        
        # Debug: Check for tokens containing the concept word
        debug_found_concept_tokens = []
        for i, token in enumerate(tokens):
            token_lower = token.replace("Ġ", "").lower()
            for concept in required_concepts:
                if concept.lower() in token_lower:
                    debug_found_concept_tokens.append((i, token, start_tok_idx + i))
        
        if debug_found_concept_tokens and not highlighted_token_indices:
            print(f"\nDebug - Sentence {sent_idx}: Found tokens containing concepts but not highlighted:")
            for token_idx, token, global_idx in debug_found_concept_tokens:
                print(f"  - Token '{token}' at local index {token_idx}, global index {global_idx}")
        
        # Create HTML for this sentence
        if original_text and highlighted_token_indices:
            # Use original text with alignment
            token_positions = align_tokens_to_text(tokens, original_text)
            
            html_parts = []
            last_pos = 0
            
            for token_idx in highlighted_token_indices:
                if token_idx < len(token_positions):
                    start, end = token_positions[token_idx]
                    
                    # Add text before this token
                    if start > last_pos:
                        html_parts.append(original_text[last_pos:start])
                    
                    # Add highlighted token
                    token_text = original_text[start:end]
                    html_parts.append(f'<span style="background-color: yellow; padding: 2px; border-radius: 3px;">{token_text}</span>')
                    
                    last_pos = end
            
            # Add any remaining text
            if last_pos < len(original_text):
                html_parts.append(original_text[last_pos:])
            
            highlighted_text = ''.join(html_parts)
        else:
            # Fallback to token-based display
            highlighted_parts = []
            cleaned_tokens = [tok.replace("Ġ", "") for tok in tokens]
            
            for i, token in enumerate(cleaned_tokens):
                if i in highlighted_token_indices:
                    highlighted_parts.append(f'<span style="background-color: yellow; padding: 2px; border-radius: 3px;">{token}</span>')
                else:
                    highlighted_parts.append(token)
            
            highlighted_text = ' '.join(highlighted_parts)
        
        html_blocks.append(f'<div style="margin-bottom: 15px;"><b>Sentence {sent_idx}:</b><br>{highlighted_text}</div>')
    
    # Display all as HTML
    full_html = f"""
    <div style="padding: 20px; background: #f9f9f9; border-radius: 10px;">
        {''.join(html_blocks)}
    </div>
    """
    display(HTML(full_html))
    
    print("\n" + "=" * 80)
