"""Utils for Computing Concepts"""

from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd
import random
import numpy as np
import math
from collections import defaultdict
import copy
import gc
import sys
import json

import torch
from torchvision import transforms
from sklearn.metrics import mean_squared_error, f1_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import resample
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# import cupy as cp 
# from cuml.cluster import KMeans as cuml_kmeans 
from fast_pytorch_kmeans import KMeans
import faiss


import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, clear_output

import importlib
import utils.general_utils
import utils.patch_alignment_utils
importlib.reload(utils.general_utils)
importlib.reload(utils.patch_alignment_utils)

from utils.general_utils import retrieve_image, load_images, get_split_df, create_binary_labels, filter_coco_concepts
from utils.patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence
from utils.embedding_utils import ChunkedEmbeddingDataset


### For Computing Concept Vectors ###
def assign_labels_to_centers(embeddings, cluster_centers, device):
    embeddings = embeddings.to(device)
    cluster_centers = cluster_centers.to(device)
    dists = torch.cdist(embeddings, cluster_centers, p=2)
    labels = torch.argmin(dists, dim=1)
    return labels.cpu()

def run_fast_pytorch_kmeans(n_clusters, train_embeddings, test_embeddings, cal_embeddings, device):
    """
    Run KMeans clustering with fast_pytorch_kmeans as primary option and faiss-gpu as fallback.
    
    Args:
        n_clusters: Number of clusters
        train_embeddings: Training embeddings tensor
        test_embeddings: Test embeddings tensor  
        cal_embeddings: Calibration embeddings tensor
        device: Device to run on (cuda/cpu)
        
    Returns:
        train_labels, test_labels, cal_labels, cluster_centers
    """
    try:
        # Try fast_pytorch_kmeans first
        print(f"Attempting fast_pytorch_kmeans with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=True, max_iter=10000, tol=1e-6)
        
        # Fit k-means on training data
        train_labels = kmeans.fit_predict(train_embeddings.to(device))
        cluster_centers = kmeans.centroids.detach().cpu()
        
        # Assign labels to test and cal sets
        test_labels = assign_labels_to_centers(test_embeddings, cluster_centers, device)
        cal_labels = assign_labels_to_centers(cal_embeddings, cluster_centers, device)
        
        print("Successfully completed clustering with fast_pytorch_kmeans")
        return train_labels, test_labels, cal_labels, cluster_centers
        
    except Exception as e:
        print(f"fast_pytorch_kmeans failed with error: {e}")
        print("Falling back to faiss-gpu...")
        
        # Ensure inputs are float32 and on CPU for faiss
        train_embeddings = train_embeddings.cpu().float()
        test_embeddings = test_embeddings.cpu().float()
        cal_embeddings = cal_embeddings.cpu().float()
        
        # Convert to numpy
        train_np = train_embeddings.numpy()
        test_np = test_embeddings.numpy()
        cal_np = cal_embeddings.numpy()
        d = train_np.shape[1]
        
        # Initialize faiss kmeans
        gpu_available = device == 'cuda' and faiss.get_num_gpus() > 0
        kmeans = faiss.Kmeans(d=d, k=n_clusters, niter=300, verbose=True, gpu=gpu_available)
        
        print(f"Fitting FAISS KMeans with {n_clusters} clusters on {len(train_np)} samples...")
        kmeans.train(train_np)
        
        print("Assigning labels to clusters...")
        train_labels = kmeans.index.search(train_np, 1)[1].squeeze()
        test_labels = kmeans.index.search(test_np, 1)[1].squeeze()
        cal_labels = kmeans.index.search(cal_np, 1)[1].squeeze()
        
        # Convert back to torch tensors
        train_labels = torch.from_numpy(train_labels).long()
        test_labels = torch.from_numpy(test_labels).long()
        cal_labels = torch.from_numpy(cal_labels).long()
        cluster_centers = torch.from_numpy(kmeans.centroids)
        
        print("Successfully completed clustering with faiss-gpu")
        return train_labels, test_labels, cal_labels, cluster_centers


# def run_fast_pytorch_kmeans(n_clusters, train_embeddings, test_embeddings, cal_embeddings, device='cuda', max_iter=300):
#     """
#     Memory-efficient FAISS-based KMeans that avoids CPU copies.

#     Args:
#         n_clusters (int): Number of clusters.
#         train_embeddings (torch.Tensor): [N, D], float32, on CPU.
#         test_embeddings (torch.Tensor): [M, D], float32, on CPU.
#         device (str): 'cuda' or 'cpu'.
#         max_iter (int): Max iterations for KMeans.

#     Returns:
#         train_labels (torch.LongTensor)
#         test_labels (torch.LongTensor)
#         cluster_centers (torch.Tensor)  # [k, d]
#     """
#     # Ensure float32 and CPU, no graph
#     assert not train_embeddings.requires_grad and not test_embeddings.requires_grad
#     assert train_embeddings.dtype == torch.float32 and test_embeddings.dtype == torch.float32
#     assert train_embeddings.device.type == 'cpu' and test_embeddings.device.type == 'cpu'

#     train_np = train_embeddings.numpy()
#     test_np = test_embeddings.numpy()
#     cal_np = cal_embeddings.numpy()
#     d = train_np.shape[1]

#     kmeans = faiss.Kmeans(d=d, k=n_clusters, niter=max_iter, verbose=True, gpu=(device == 'cuda'))
#     print(f"Fitting FAISS KMeans with {n_clusters} clusters on {len(train_np)} samples...")
#     kmeans.train(train_np)

#     print("Assigning Labels to Kmeans Clusters...")
#     train_labels = kmeans.index.search(train_np, 1)[1].squeeze()
#     test_labels = kmeans.index.search(test_np, 1)[1].squeeze()
#     cal_labels = kmeans.index.search(cal_np, 1)[1].squeeze()

#     # Convert results back to torch
#     train_labels = torch.from_numpy(train_labels).long()
#     test_labels = torch.from_numpy(test_labels).long()
#     cal_labels = torch.from_numpy(cal_labels).long()
#     cluster_centers = torch.from_numpy(kmeans.centroids)

#     return train_labels, test_labels, cal_labels, cluster_centers



def map_samples_to_clusters(train_image_indices, train_labels, test_image_indices, test_labels, 
                            cal_image_indices, cal_labels, dataset_name, concepts_filename=None):
    # Initialize cluster mappings
    train_cluster_to_samples = defaultdict(list)
    test_cluster_to_samples = defaultdict(list)
    cal_cluster_to_samples = defaultdict(list)

    # Map training samples if available
    if len(train_labels) > 0:
        for i, train_idx in enumerate(train_image_indices):
            cluster_label = str(train_labels[i].item())
            train_cluster_to_samples[cluster_label].append(train_idx)
    else:
        print("No training labels provided — skipping train mapping.")

    # Map test samples if available
    if len(test_labels) > 0:
        for i, test_idx in enumerate(test_image_indices):
            cluster_label = str(test_labels[i].item())
            test_cluster_to_samples[cluster_label].append(test_idx)
    else:
        print("No test labels provided — skipping test mapping.")
        
    # Map cal samples if available
    if len(cal_labels) > 0:
        for i, cal_idx in enumerate(cal_image_indices):
            cluster_label = str(cal_labels[i].item())
            cal_cluster_to_samples[cluster_label].append(cal_idx)
    else:
        print("No cal labels provided — skipping test mapping.")

    # Convert to regular dicts and sort keys
    train_cluster_to_samples = dict(sorted(train_cluster_to_samples.items()))
    test_cluster_to_samples = dict(sorted(test_cluster_to_samples.items()))
    cal_cluster_to_samples = dict(sorted(cal_cluster_to_samples.items()))

    # Optionally save to file
    if concepts_filename:
        torch.save(train_cluster_to_samples, f'Concepts/{dataset_name}/train_samples_{concepts_filename}')
        torch.save(test_cluster_to_samples, f'Concepts/{dataset_name}/test_samples_{concepts_filename}')
        torch.save(cal_cluster_to_samples, f'Concepts/{dataset_name}/cal_samples_{concepts_filename}')
        print(f"Saved mapped cluster indices to Concepts/{dataset_name}/train_samples_{concepts_filename} :)")

    return train_cluster_to_samples, test_cluster_to_samples, cal_cluster_to_samples


def gpu_kmeans(n_clusters, embeddings_path, dataset_name, device, model_input_size, concepts_filename=None, sample_type='patch', map_samples=True):
    """
    Memory-efficient GPU-accelerated KMeans clustering that loads embeddings from chunks as needed.
    Only loads the split (train/test/cal) that's currently being processed.
    
    Args:
        n_clusters (int): Number of clusters for KMeans.
        embeddings_path (str): Path to embeddings file (chunked or not).
        dataset_name (str): Name of the dataset.
        device: Device to run KMeans on.
        model_input_size: Model input size.
        concepts_filename (str, optional): Filename to save the concepts.
        sample_type (str): 'patch' or 'cls'
        map_samples (bool): Whether to map samples to clusters.
        
    Returns:
        Same as gpu_kmeans
    """
    from utils.memory_management_utils import ChunkedEmbeddingLoader
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get split information
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name, model_input_size)
        all_indices = split_df.index
        relevant_indices = filter_patches_by_image_presence(all_indices, dataset_name, model_input_size).tolist()
    else:
        split_df = get_split_df(dataset_name)
        all_indices = split_df.index
        relevant_indices = list(all_indices)
    
    # Get indices for each split
    # Convert to set for O(1) membership checking
    relevant_indices_set = set(relevant_indices)
    train_indices = [idx for idx in split_df[split_df == 'train'].index if idx in relevant_indices_set]
    test_indices = [idx for idx in split_df[split_df == 'test'].index if idx in relevant_indices_set]
    cal_indices = [idx for idx in split_df[split_df == 'cal'].index if idx in relevant_indices_set]
    
    # Parse dataset name and file from path for ChunkedEmbeddingLoader
    # Expected format: {scratch_dir}Embeddings/{dataset_name}/{embeddings_file}
    path_parts = embeddings_path.split('/')
    embeddings_file = path_parts[-1]
    dataset_name_from_path = path_parts[-2]
    scratch_dir_from_path = '/'.join(path_parts[:-3]) + '/' if len(path_parts) > 3 else ''
    
    # Load only train embeddings for KMeans
    print("Loading train embeddings for KMeans...")
    loader = ChunkedEmbeddingLoader(dataset_name_from_path, embeddings_file, scratch_dir_from_path, device='cpu')
    train_embeddings = loader.load_specific_embeddings(train_indices)
    
    # Check if this is COCO, Broden-Pascal, or Broden-OpenSurfaces with Llama - if so, use faiss-gpu directly
    if dataset_name.lower() in ['coco', 'broden-pascal', 'broden-opensurfaces'] and 'llama' in embeddings_file.lower():
        print(f"Using faiss-gpu for {dataset_name} Llama combination...")
        
        # Ensure float32 and on CPU for faiss
        train_embeddings = train_embeddings.cpu().float()
        
        # Convert to numpy
        train_np = train_embeddings.numpy()
        d = train_np.shape[1]
        
        # Initialize faiss kmeans with GPU
        gpu_available = device == 'cuda' and faiss.get_num_gpus() > 0
        kmeans = faiss.Kmeans(d=d, k=n_clusters, niter=300, verbose=True, gpu=gpu_available)
        
        print(f"Fitting FAISS KMeans with {n_clusters} clusters on {len(train_np)} samples...")
        kmeans.train(train_np)
        
        # Get cluster assignments for train data
        train_labels = kmeans.index.search(train_np, 1)[1].squeeze()
        train_labels = torch.from_numpy(train_labels).long()
        cluster_centers = torch.from_numpy(kmeans.centroids)
        
        # Clear train embeddings
        del train_embeddings, train_np
        torch.cuda.empty_cache()
        gc.collect()
        
        # Process test embeddings
        print("Loading test embeddings for cluster assignment...")
        test_embeddings = loader.load_specific_embeddings(test_indices).cpu().float()
        test_np = test_embeddings.numpy()
        test_labels = kmeans.index.search(test_np, 1)[1].squeeze()
        test_labels = torch.from_numpy(test_labels).long()
        
        # Clear test embeddings
        del test_embeddings, test_np
        torch.cuda.empty_cache()
        gc.collect()
        
        # Process cal embeddings
        print("Loading cal embeddings for cluster assignment...")
        cal_embeddings = loader.load_specific_embeddings(cal_indices).cpu().float()
        cal_np = cal_embeddings.numpy()
        cal_labels = kmeans.index.search(cal_np, 1)[1].squeeze()
        cal_labels = torch.from_numpy(cal_labels).long()
        
        # Clear cal embeddings
        del cal_embeddings, cal_np
        
    else:
        # Use original fast_pytorch_kmeans for other cases
        train_embeddings = train_embeddings.to(device)
        
        # Run KMeans on train embeddings
        print(f"Running KMeans with {n_clusters} clusters on {len(train_embeddings)} train samples...")
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        train_labels = kmeans.fit_predict(train_embeddings)
        cluster_centers = kmeans.centroids
        
        # Clear train embeddings from GPU
        del train_embeddings
        torch.cuda.empty_cache()
        gc.collect()
        
        # Assign test embeddings to clusters
        print("Loading test embeddings for cluster assignment...")
        test_embeddings = loader.load_specific_embeddings(test_indices).to(device)
        test_labels = kmeans.predict(test_embeddings)
        
        # Clear test embeddings
        del test_embeddings
        torch.cuda.empty_cache()
        gc.collect()
        
        # Assign cal embeddings to clusters
        print("Loading cal embeddings for cluster assignment...")
        cal_embeddings = loader.load_specific_embeddings(cal_indices).to(device)
        cal_labels = kmeans.predict(cal_embeddings)
        
        # Clear cal embeddings
        del cal_embeddings
        torch.cuda.empty_cache()
        gc.collect()
    
    # Map cluster labels to cluster centers
    label_to_center = {label: center.cpu() for label, center in enumerate(cluster_centers)}
    label_to_center = dict(sorted(label_to_center.items()))
    label_to_center = {str(label): center for label, center in label_to_center.items()}
    
    if concepts_filename:
        torch.save(label_to_center, f'Concepts/{dataset_name}/{concepts_filename}')
        torch.save(cluster_centers, f'Concepts/{dataset_name}/cluster_centers_{concepts_filename}')
        torch.save(train_labels, f'Concepts/{dataset_name}/train_labels_{concepts_filename}')
        torch.save(test_labels, f'Concepts/{dataset_name}/test_labels_{concepts_filename}')
        torch.save(cal_labels, f'Concepts/{dataset_name}/cal_labels_{concepts_filename}')
        print(f"Saved cluster centers and labels to Concepts/{dataset_name}/{concepts_filename} :)")
    
    train_cluster_to_samples, test_cluster_to_samples, cal_cluster_to_samples = [], [], []
    if map_samples:
        train_cluster_to_samples, test_cluster_to_samples, cal_cluster_to_samples = map_samples_to_clusters(
            train_indices, train_labels,
            test_indices, test_labels,
            cal_indices, cal_labels,
            dataset_name, concepts_filename)
    
    return label_to_center, train_cluster_to_samples, test_cluster_to_samples, cal_cluster_to_samples



def aggregate_concept_vectors(concept_embeddings, dataset_name, save_file=None):
    """
    Aggregates concept embeddings by computing their mean, centers, and normalizes the concept vectors.

    Args:
        concept_embeddings (dict): A dictionary where keys are concept names and values are lists 
            of embeddings corresponding to those concepts.
        dataset_name (str): The name of the dataset. 
        save_file (str, optional): File name to save the aggregated concept vectors. If None, the vectors 
            will not be saved. Defaults to None.

    Returns:
        dict: A dictionary of processed concept vectors, where each vector is mean-centered and normalized.
    """
    # Compute mean vectors for each concept
    concept_vectors = {concept: torch.stack(embeddings).mean(dim=0) 
                       for concept, embeddings in concept_embeddings.items()}
    
    # Stack all mean vectors to compute global statistics
    all_vectors = torch.stack(list(concept_vectors.values()))
    
    # Compute global mean and L2 norm
    global_mean = all_vectors.mean(dim=0)
    global_l2_norm = all_vectors.norm(dim=1, keepdim=True).mean()  # Mean L2 norm of all concept vectors
    
    # Normalize all concept vectors based on global L2 norm
    for concept, vector in concept_vectors.items():
        # Normalize by the global L2 norm 
        normalized_vector = vector / global_l2_norm
        concept_vectors[concept] = normalized_vector

    # Save to file if specified
    if save_file:
        save_path = f'Concepts/{dataset_name}/{save_file}'
        torch.save(concept_vectors, save_path)
        print(f"Concept vectors saved to {save_path}")
    
    return concept_vectors




def compute_avg_concept_vectors(gt_samples_per_concept_train, loader, dataset_name=None, output_file=None):
    """
    Computes average concept vectors from chunked embeddings by loading only the necessary samples.
    
    Args:
        gt_samples_per_concept_train (dict): Concept names -> lists of global sample indices
        loader (ChunkedEmbeddingLoader): Loader for chunked embeddings
        dataset_name (str): Dataset name for saving
        output_file (str): Output file name
        
    Returns:
        dict: Concept names -> average concept vectors
    """
    concepts = {}
    
    for concept, sample_indices in tqdm(gt_samples_per_concept_train.items(), desc="Computing avg concepts"):
        if len(sample_indices) == 0:
            print(f"Warning: No samples for concept {concept}")
            continue
            
        # Load only the embeddings we need for this concept
        concept_embeddings = loader.load_specific_embeddings(sample_indices)
        
        # Compute average
        avg_vector = torch.mean(concept_embeddings, dim=0)
        concepts[concept] = avg_vector
        
        # Clear memory
        del concept_embeddings
        torch.cuda.empty_cache()
        gc.collect()
    
    if output_file:
        Path('Concepts', dataset_name).mkdir(parents=True, exist_ok=True)
        torch.save(concepts, f'Concepts/{dataset_name}/{output_file}')
        print(f'Concepts saved to Concepts/{dataset_name}/{output_file} :)')
    
    return concepts



def sort_data_by_split(embeds, all_concept_labels, split, dataset_name, model_input_size, sample_type):
    """
    Extract embeddings and labels corresponding to a specified split. Doesn't include padding.
    
    Args:
        embeds (torch.Tensor): Tensor containing embeddings.
        labels (torch.Tensor): Tensor containing binary labels.
        split_df (pd.DataFrame): DataFrame with split information ('train'/'test').
        split (str): The split to extract (e.g., 'train' or 'test').
    
    Returns:
        tuple: A tuple (split_embeds, split_labels) corresponding to the specified split.
    """
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name, model_input_size=model_input_size)
        relevant_indices = filter_patches_by_image_presence(split_df.index, dataset_name, model_input_size).tolist()
    elif sample_type == 'cls':
        split_df = get_split_df(dataset_name)
        relevant_indices = split_df.index
    
    split_indices = (split_df[split_df == split]).index
    overlapping_indices = list(set(split_indices).intersection(relevant_indices))
    
    split_embeds = embeds[overlapping_indices]
    split_all_concept_labels = {}
    for concept, labels in all_concept_labels.items():
        split_labels = labels[overlapping_indices]
        split_all_concept_labels[concept] = split_labels
    return split_embeds, split_all_concept_labels


def balance_dataset(embeds, labels, seed=42, max_samples=100000):
    """
    Balance the dataset for positive and negative examples, optionally capping the number of samples per class.

    Args:
        embeds (torch.Tensor): Tensor of embeddings (N, D).
        labels (torch.Tensor): Binary labels (N,).
        seed (int): Random seed.
        max_samples (int or None): Optional cap for the number of pos/neg samples to include.

    Returns:
        tuple: (balanced_embeds, balanced_labels) or (None, None) if not enough data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
    neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        print(f"Skipping due to insufficient data: {len(pos_indices)} pos, {len(neg_indices)} neg")
        return None, None

    # Choose the number of samples per class
    num_samples = min(len(pos_indices), len(neg_indices))
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    # Resample to balance
    pos_indices = resample(pos_indices.cpu().numpy(), n_samples=num_samples, replace=False, random_state=seed)
    neg_indices = resample(neg_indices.cpu().numpy(), n_samples=num_samples, replace=False, random_state=seed)

    balanced_indices = torch.tensor(np.concatenate([pos_indices, neg_indices]), dtype=torch.long)

    balanced_embeds = embeds[balanced_indices]
    balanced_labels = labels[balanced_indices]

    return balanced_embeds, balanced_labels


def balance_dataset_evenly_for_ooc(embeds, all_concept_labels, target_concept, seed=42):
    """
    Balance the dataset so that 50% of the samples are in-concept (target_concept) and 50% are out-of-concept. Out of
    the 50% that are out of concept, they are distributed evenly across other concepts.

    Args:
        embeds (torch.Tensor): Tensor of embeddings (N x D).
        all_concept_labels (dict): Dictionary mapping each concept to a binary tensor (N,) indicating concept presence.
        target_concept (str): The concept for which we are training a classifier.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (balanced_embeds, balanced_labels) after undersampling.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    labels = all_concept_labels[target_concept]
    pos_indices = labels.nonzero(as_tuple=True)[0]
    neg_indices = (labels == 0).nonzero(as_tuple=True)[0]
    num_pos = len(pos_indices)

    non_target_concepts = [c for c in all_concept_labels if c != target_concept]

    neg_indices_np = neg_indices.cpu().numpy()
    candidate_matrix = np.stack([
        all_concept_labels[c][neg_indices].cpu().numpy() for c in non_target_concepts
    ], axis=0)

    def choose_candidate(col, seed=42):
        np.random.seed(seed)  # Set seed for reproducibility
        candidates = np.nonzero(col)[0]
        if candidates.size == 0:
            return -1
        else:
            return np.random.choice(candidates)

    chosen_candidates = np.apply_along_axis(choose_candidate, 0, candidate_matrix)

    group_assignments = {c: [] for c in non_target_concepts}
    group_assignments["none"] = []

    for i, candidate in enumerate(chosen_candidates):
        idx = int(neg_indices_np[i])
        if candidate == -1:
            group_assignments["none"].append(idx)
        else:
            chosen_concept = non_target_concepts[int(candidate)]
            group_assignments[chosen_concept].append(idx)

    if len(group_assignments["none"]) == 0:
        del group_assignments["none"]

    target_total_negatives = num_pos
    num_groups = len(group_assignments)
    equal_share = target_total_negatives // num_groups

    selected_negatives = {}
    total_selected = 0

    for group, indices in group_assignments.items():
        if len(indices) <= equal_share:
            selected_negatives[group] = indices[:]
        else:
            sampled = resample(indices, n_samples=equal_share, replace=False, random_state=seed)
            selected_negatives[group] = sampled
        total_selected += len(selected_negatives[group])

    remaining_needed = target_total_negatives - total_selected

    extra_counts = {group: len(set(group_assignments[group]) - set(selected_negatives.get(group, [])))
                    for group in group_assignments}

    total_extras = sum(extra_counts.values())

    if remaining_needed > 0 and total_extras > 0:
        allocated = {group: int(round((extra_counts[group] / total_extras) * remaining_needed))
                     for group in extra_counts}

        allocated_total = sum(allocated.values())
        diff = remaining_needed - allocated_total

        if diff:
            sorted_groups = sorted(extra_counts.items(), key=lambda x: x[1], reverse=True)
            idx = 0
            while diff:
                grp = sorted_groups[idx % len(sorted_groups)][0]
                allocated[grp] += 1 if diff > 0 else -1
                diff += -1 if diff > 0 else 1
                idx += 1

        for group, extra_alloc in allocated.items():
            if extra_alloc > 0:
                available = list(set(group_assignments[group]) - set(selected_negatives.get(group, [])))
                if len(available) >= extra_alloc:
                    sampled = resample(available, n_samples=extra_alloc, replace=False, random_state=seed)
                else:
                    sampled = available
                selected_negatives[group].extend(sampled)

    all_selected_negatives = []
    for group in selected_negatives:
        all_selected_negatives.extend(selected_negatives[group])

    if len(pos_indices) > len(all_selected_negatives):
        pos_indices = pos_indices[torch.randperm(len(pos_indices), generator=torch.Generator().manual_seed(seed))[:len(all_selected_negatives)]]

    balanced_indices = torch.cat([pos_indices, torch.tensor(all_selected_negatives, dtype=torch.long)])
    balanced_embeds = embeds[balanced_indices]
    balanced_labels = labels[balanced_indices]

    return balanced_embeds, balanced_labels


def print_balancing_results(train_labels, test_labels):
    """
    Print the count of positive and negative samples for both training and test sets.
    
    Args:
        train_labels (torch.Tensor): Tensor of training labels.
        test_labels (torch.Tensor): Tensor of test labels.
    
    Returns:
        None
    """
    num_train_pos = (train_labels == 1).sum().item()
    num_train_neg = (train_labels == 0).sum().item()

    num_test_pos = (test_labels == 1).sum().item()
    num_test_neg = (test_labels == 0).sum().item()

    print(f"Resampled to {len(train_labels)} train samples "
          f"({num_train_pos} positive, {num_train_neg} negative); "
          f"{len(test_labels)} test samples "
          f"({num_test_pos} positive, {num_test_neg} negative)")
    
    
def create_dataloader(embeds, labels, batch_size, shuffle=True):
    """
    Create a DataLoader from embeddings and labels.
    
    Args:
        embeds (torch.Tensor): Tensor of embeddings.
        labels (torch.Tensor): Tensor of labels.
        batch_size (int): Number of samples per batch.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
    
    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = TensorDataset(embeds, labels.float())
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dl
    
    
def evaluate_model(model, dl, criterion, device):
    """
    Evaluate the model on data provided by the dataloader.
    
    Args:
        model (nn.Module): The model to evaluate.
        dl (DataLoader): DataLoader for the evaluation dataset.
        criterion (torch.nn.Module): Loss function.
        device (str): The device on which to perform computations.
    
    Returns:
        tuple: A tuple (avg_loss, accuracy, f1) containing the average loss, accuracy, and F1 score.
    """
    model.eval()
    with torch.no_grad():
        sum_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        for batch_features, batch_labels in dl:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features.float()).view(-1)
            loss = criterion(outputs, batch_labels)

            sum_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

        avg_loss = sum_loss / len(dl)
        acc = correct / total
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1

def log_progress(logs, avg_train_loss, train_acc, train_f1, avg_test_loss, test_acc, test_f1, epoch, epochs):
    """
    Log and print training and testing metrics for the current epoch.
    
    Args:
        logs (dict): Dictionary to store metrics over epochs.
        avg_train_loss (float): Average training loss for the epoch.
        train_acc (float): Training accuracy for the epoch.
        train_f1 (float): Training F1 score for the epoch.
        avg_test_loss (float): Average test loss for the epoch.
        test_acc (float): Test accuracy for the epoch.
        test_f1 (float): Test F1 score for the epoch.
        epoch (int): Current epoch index.
        epochs (int): Total number of epochs.
    
    Returns:
        dict: The updated logs dictionary.
    """
    logs['train_loss'].append(avg_train_loss)
    logs['train_accuracy'].append(train_acc)
    logs['train_f1'].append(train_f1)
    logs['test_loss'].append(avg_test_loss)
    logs['test_accuracy'].append(test_acc)
    logs['test_f1'].append(test_f1)
    
    msg = (f"Epoch [{epoch+1}/{epochs}] - "
           f"Train Loss: {avg_train_loss:.6f} - Train Acc: {train_acc:.4f} - Train F1: {train_f1:.4f} | "
           f"Test Loss: {avg_test_loss:.4f} - Test Acc: {test_acc:.4f} - Test F1: {test_f1:.4f}")
    
    if epoch+1 == epochs:
        print(msg)
    else:
        print(msg, end="\r")
        
    return logs
    
def create_linear_model(D, device, weights=None):
    """
    Creates a single-layer linear model (no bias), optionally initializing with given weights.

    Args:
        D (int): Input dimensionality (feature size).
        device (str or torch.device): Device to move the model to.
        weights (torch.Tensor, optional): Tensor of shape (1, D) or (D,) to initialize model weights.

    Returns:
        nn.Linear: A linear model with weights optionally preloaded.
    """
    model = nn.Linear(D, 1, bias=False).to(device)

    if weights is not None:
        # Ensure weights shape is (1, D)
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)
        elif weights.shape != (1, D):
            raise ValueError(f"Expected weights of shape (1, {D}) or ({D},), got {weights.shape}")
        
        model.weight.data.copy_(weights.to(device))

    return model
    
def train_model(train_dl, test_dl, epochs, lr, weight_decay, lr_step_size, lr_gamma, patience, tolerance, device, model=None):
    """
    Train a linear model using the provided training and test dataloaders.
    
    Args:
        train_dl (DataLoader): DataLoader for the training dataset.
        test_dl (DataLoader): DataLoader for the test dataset.
        epochs (int): Maximum number of training epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay coefficient for the optimizer.
        lr_step_size (int): Number of epochs between each learning rate decay step.
        lr_gamma (float): Factor by which the learning rate is decayed.
        patience (int): Number of epochs with insufficient improvement before early stopping.
        tolerance (float): Minimum improvement required to reset the early stopping counter.
        device (str): The device on which to train the model.
    
    Returns:
        tuple: A tuple (model_weights, logs) where model_weights is the learned weight vector,
               and logs is a dictionary containing the training metrics.
    """
    if model is None:
        D = len(train_dl.dataset[0][0])
        model = create_linear_model(D, device)
    
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    # Tracking metrics
    logs = {'train_loss': [], 'train_accuracy': [], 'train_f1': [], 'test_loss': [], 'test_accuracy': [], 'test_f1': []}
    best_loss = float("inf")
    patience_counter = 0  
    
    for epoch in range(epochs):
        #training 
        model.train()
        
        for batch_features, batch_labels in train_dl:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features.float()).view(-1)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        #evaluation
        train_avg_loss, train_acc, train_f1 = evaluate_model(model, train_dl, criterion, device)
        test_avg_loss, test_acc, test_f1 = evaluate_model(model, test_dl, criterion, device)
        
        logs = log_progress(logs, train_avg_loss, train_acc, train_f1, test_avg_loss, test_acc, test_f1, epoch, epochs)
        
        #Potential early stopping
        if logs['train_f1'][-1] >= 0.99:
            print(f"    Early stopping at epoch {epoch + 1} (train_f1 >= 0.99)")
            break
        if epoch > 0 and (best_loss - logs['train_loss'][-1]) < tolerance:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch + 1} (patience exhausted)")
                break
        else:
            patience_counter = 0  # Reset if improvement is sufficient

        best_loss = min(best_loss, logs['train_loss'][-1])
        
        scheduler.step()
        
    return model.weight.detach().squeeze(0).cpu(), logs


def create_dataloaders(concept, train_embeds, train_all_concept_labels, test_embeds, test_all_concept_labels, 
                       batch_size, balance_data, balance_negatives):
    # Balance dataset potentially (undersampling)
    if balance_data:
        if balance_negatives:
            print("Balancing Overall Data and Negatives")
            train_embeds, train_labels = balance_dataset_evenly_for_ooc(train_embeds, train_all_concept_labels, concept)
            test_embeds, test_labels = balance_dataset_evenly_for_ooc(test_embeds, test_all_concept_labels, concept)
        else:
            print("Balancing Just Overall Data")
            train_embeds, train_labels = balance_dataset(train_embeds, train_all_concept_labels[concept])
            test_embeds, test_labels = balance_dataset(test_embeds, test_all_concept_labels[concept])
    else:
        train_labels = train_all_concept_labels[concept]
        test_labels = test_all_concept_labels[concept]
   
    if train_embeds is None or test_embeds is None:
        return None, None
        
    print_balancing_results(train_labels, test_labels)
    
    #Create Dataloaders
    train_dl = create_dataloader(train_embeds, train_labels, batch_size, shuffle=True)
    test_dl = create_dataloader(test_embeds, test_labels, batch_size, shuffle=False)
    return train_dl, test_dl


def compute_a_linear_separator(
    concept, train_embeds, test_embeds, train_all_concept_labels, test_all_concept_labels,
    lr=0.01, epochs=100, patience=15, tolerance=0.001, batch_size=32, weight_decay=1e-2, 
    lr_step_size=10, lr_gamma=0.5, device='cuda', balance_data=True, balance_negatives=False
):
    """
    Compute a linear separator for a given concept with dataset balancing, weight decay, and LR scheduling.

    Args:
        concept (str): Concept name.
        embeds (torch.Tensor): (N, D) tensor of embeddings.
        concept_gt_patches (set): Indices where the concept is present.
        lr (float): Initial learning rate.
        epochs (int): Max training epochs.
        patience (int): Early stopping patience.
        batch_size (int): Training batch size.
        weight_decay (float): Weight decay for Adam optimizer.
        lr_step_size (int): Step size for LR scheduler.
        lr_gamma (float): Decay factor for LR scheduler.
        device (str): Compute device.
        undersampling_ratio (float): Ratio of in-concept to out-of-concept samples for undersampling in training.

    Returns:
        torch.Tensor: Learned concept weight vector.
        dict: Training & test metrics.
    """
    print(f"Training linear classifier for concept {concept}")
    train_dl, test_dl = create_dataloaders(concept, train_embeds, train_all_concept_labels, test_embeds, test_all_concept_labels, 
                       batch_size, balance_data, balance_negatives)
    
    if train_dl is not None and test_dl is not None:
        #Train Model
        model_weights, logs = train_model(train_dl, test_dl, epochs, lr, weight_decay, lr_step_size, lr_gamma, patience, tolerance, device)
    else:
        #no patches in cluster, just make random model
        model = create_linear_model(train_embeds.shape[1], device)
        model_weights = model.weight.detach().squeeze(0).cpu()
        logs = []
    
    return model_weights, logs




def preload_balanced_embeddings(embeddings_path: str, indices: list, labels: torch.Tensor, 
                               loader) -> tuple:
    """
    Efficiently preload only the required embeddings from chunks.
    
    Args:
        embeddings_path: Path to chunked embeddings
        indices: List of global indices to load
        labels: Corresponding labels for the indices
        loader: ChunkedEmbeddingLoader instance
        
    Returns:
        tuple: (embeddings_tensor, labels_tensor) both in memory
    """
    # Map global indices to chunks
    chunk_map = loader.global_indices_to_chunk_map(indices)
    
    # Preallocate result tensor
    embedding_dim = loader.embedding_dim
    embeddings = torch.zeros((len(indices), embedding_dim), dtype=torch.float32)
    
    # Create index mapping for efficient assignment
    global_to_result_idx = {global_idx: i for i, global_idx in enumerate(indices)}
    
    # Load each chunk once and extract needed embeddings
    for chunk_num, chunk_indices in chunk_map.items():
        chunk_file = loader.chunk_info['chunks'][chunk_num]['file']
        chunk_path = os.path.join(loader.chunks_dir, chunk_file)
        
        # Load chunk
        chunk_data = torch.load(chunk_path, map_location='cpu', weights_only=False)
        if isinstance(chunk_data, dict):
            chunk_embeddings = chunk_data.get('normalized_embeddings', chunk_data.get('embeddings'))
        else:
            chunk_embeddings = chunk_data
            
        # Extract needed embeddings from this chunk
        for global_idx, local_idx in chunk_indices:
            result_idx = global_to_result_idx[global_idx]
            embeddings[result_idx] = chunk_embeddings[local_idx]
            
        # Free chunk memory
        del chunk_data, chunk_embeddings
    
    return embeddings, labels[indices]


def preload_train_test_embeddings(embeddings_path: str, train_indices: list, test_indices: list,
                                 labels: torch.Tensor, loader) -> tuple:
    """
    Efficiently preload both train and test embeddings in a single pass through chunks.
    
    Args:
        embeddings_path: Path to chunked embeddings
        train_indices: List of global indices for training
        test_indices: List of global indices for testing
        labels: Corresponding labels for all indices
        loader: ChunkedEmbeddingLoader instance
        
    Returns:
        tuple: (train_embeddings, train_labels, test_embeddings, test_labels)
    """
    # Combine all indices and track which are train vs test
    all_indices = train_indices + test_indices
    is_train = [True] * len(train_indices) + [False] * len(test_indices)
    
    # Map global indices to chunks
    chunk_map = loader.global_indices_to_chunk_map(all_indices)
    
    # Preallocate result tensors
    embedding_dim = loader.embedding_dim
    train_embeddings = torch.zeros((len(train_indices), embedding_dim), dtype=torch.float32)
    test_embeddings = torch.zeros((len(test_indices), embedding_dim), dtype=torch.float32)
    
    # Create index mappings
    train_counter = 0
    test_counter = 0
    global_to_result_idx = {}
    
    for i, (global_idx, is_train_sample) in enumerate(zip(all_indices, is_train)):
        if is_train_sample:
            global_to_result_idx[global_idx] = ('train', train_counter)
            train_counter += 1
        else:
            global_to_result_idx[global_idx] = ('test', test_counter)
            test_counter += 1
    
    # Load each chunk once and extract needed embeddings
    for chunk_num, chunk_indices in chunk_map.items():
        chunk_file = loader.chunk_info['chunks'][chunk_num]['file']
        chunk_path = os.path.join(loader.chunks_dir, chunk_file)
        
        # Load chunk
        chunk_data = torch.load(chunk_path, map_location='cpu', weights_only=False)
        if isinstance(chunk_data, dict):
            chunk_embeddings = chunk_data.get('normalized_embeddings', chunk_data.get('embeddings'))
        else:
            chunk_embeddings = chunk_data
            
        # Extract needed embeddings from this chunk
        for global_idx, local_idx in chunk_indices:
            split, result_idx = global_to_result_idx[global_idx]
            if split == 'train':
                train_embeddings[result_idx] = chunk_embeddings[local_idx]
            else:
                test_embeddings[result_idx] = chunk_embeddings[local_idx]
            
        # Free chunk memory
        del chunk_data, chunk_embeddings
    
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    
    return train_embeddings, train_labels, test_embeddings, test_labels


def _train_single_concept_parallel(args):
    """
    Helper function for parallel training of a single concept.
    Must be defined at module level to be picklable.
    """
    (concept_name, concept_idx, concept_labels, train_indices, test_indices, 
     embeddings_path, embeddings_file, dataset_name_from_path, scratch_dir_from_path,
     embedding_dim, balance_data, batch_size, lr, epochs, weight_decay, 
     lr_step_size, lr_gamma, patience, tolerance, num_gpus) = args
    
    from utils.memory_management_utils import ChunkedEmbeddingLoader
    
    if num_gpus > 1:
        # Multiple GPUs: distribute across GPUs
        device_id = concept_idx % num_gpus
    else:
        # Single GPU or CPU: all use the same device
        device_id = 0
    
    device_str = f'cuda:{device_id}' if num_gpus > 0 else 'cpu'
    
    # Set device for this process
    if torch.cuda.is_available() and num_gpus > 0:
        torch.cuda.set_device(device_id)
    
    try:
        # Balance dataset if needed
        if balance_data:
            # Convert to tensor for faster indexing
            train_indices_tensor = torch.tensor(train_indices, dtype=torch.long)
            train_labels = concept_labels[train_indices_tensor]
            
            # Use tensor operations for finding positive/negative indices
            pos_indices_tensor = torch.where(train_labels == 1)[0]
            neg_indices_tensor = torch.where(train_labels == 0)[0]
            pos_indices = pos_indices_tensor.tolist()
            neg_indices = neg_indices_tensor.tolist()
            
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                # Balance by undersampling
                n_samples = min(len(pos_indices), len(neg_indices))
                balanced_pos = random.sample(pos_indices, n_samples)
                balanced_neg = random.sample(neg_indices, n_samples)
                balanced_indices = balanced_pos + balanced_neg
                balanced_train_indices = [train_indices[i] for i in balanced_indices]
                
                # Now balance test set
                test_indices_tensor = torch.tensor(test_indices, dtype=torch.long)
                test_labels = concept_labels[test_indices_tensor]
                test_pos_indices = torch.where(test_labels == 1)[0].tolist()
                test_neg_indices = torch.where(test_labels == 0)[0].tolist()
                
                if len(test_pos_indices) > 0 and len(test_neg_indices) > 0:
                    n_test_samples = min(len(test_pos_indices), len(test_neg_indices))
                    balanced_test_pos = random.sample(test_pos_indices, n_test_samples)
                    balanced_test_neg = random.sample(test_neg_indices, n_test_samples)
                    balanced_test_indices = balanced_test_pos + balanced_test_neg
                    balanced_test_indices_global = [test_indices[i] for i in balanced_test_indices]
                    
                    # Create new loader for this process
                    process_loader = ChunkedEmbeddingLoader(dataset_name_from_path, embeddings_file, 
                                                           scratch_dir_from_path, device='cpu')
                    
                    # Preload both train and test embeddings together
                    train_embeddings, train_labels_balanced, test_embeddings, test_labels_balanced = preload_train_test_embeddings(
                        embeddings_path,
                        balanced_train_indices,
                        balanced_test_indices_global,
                        concept_labels,
                        process_loader
                    )
                    
                    from torch.utils.data import TensorDataset
                    train_dataset = TensorDataset(train_embeddings, train_labels_balanced)
                    test_dataset = TensorDataset(test_embeddings, test_labels_balanced)
                else:
                    # No test samples
                    test_dataset = None
                    train_dataset = None
            else:
                # No positive or negative samples
                train_dataset = None
                test_dataset = None
        
        else:
            # Not balancing - preload all train and test indices together
            process_loader = ChunkedEmbeddingLoader(dataset_name_from_path, embeddings_file, 
                                                   scratch_dir_from_path, device='cpu')
            train_embeddings, train_labels_all, test_embeddings, test_labels_all = preload_train_test_embeddings(
                embeddings_path,
                train_indices,
                test_indices,
                concept_labels,
                process_loader
            )
            
            from torch.utils.data import TensorDataset
            train_dataset = TensorDataset(train_embeddings, train_labels_all)
            test_dataset = TensorDataset(test_embeddings, test_labels_all)
        
        # Create dataloaders and train
        if train_dataset is not None and len(train_dataset) > 0 and test_dataset is not None and len(test_dataset) > 0:
            train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            model_weights, concept_logs = train_model(train_dl, test_dl, epochs, lr, weight_decay, 
                                                    lr_step_size, lr_gamma, patience, tolerance, device_str)
        else:
            # No samples, create random model
            model = create_linear_model(embedding_dim, device_str)
            model_weights = model.weight.detach().squeeze(0).cpu()
            concept_logs = []
        
        # Cleanup
        del train_embeddings, test_embeddings
        if train_dataset:
            del train_dataset
        if test_dataset:
            del test_dataset
        torch.cuda.empty_cache()
        gc.collect()
        
        return concept_name, model_weights, concept_logs
        
    except Exception as e:
        print(f"Error training concept {concept_name}: {str(e)}")
        # Return random model on error
        model = create_linear_model(embedding_dim, 'cpu')
        model_weights = model.weight.detach().squeeze(0).cpu()
        return concept_name, model_weights, []


def compute_linear_separators(embeddings_path, gt_samples_per_concept, dataset_name, sample_type, model_input_size,
                                    device='cuda', output_file=None, lr=0.01, epochs=100, batch_size=32, patience=15,
                                    tolerance=3, weight_decay=1e-4, lr_step_size=10, lr_gamma=0.5, balance_data=True,
                                    balance_negatives=False, use_parallel=True, num_workers=None, use_batched=False,
                                    use_onepass=False, onepass_kwargs=None):
    """
    Computes linear separators using chunked embeddings loaded on-demand.
    Automatically uses parallel processing when multiple concepts are available.
    
    Args:
        embeddings_path: Path to embeddings file (chunked or not)
        gt_samples_per_concept: Dictionary mapping concept names to lists of positive sample indices
        use_parallel: Whether to use parallel processing (default: True)
        num_workers: Number of parallel workers (default: number of GPUs)
        use_batched: Whether to use batched training with single D×K model (default: False)
        use_onepass: Whether to use one-pass accumulation method (default: False)
        onepass_kwargs: Additional kwargs for one-pass method (lambda_reg, chunk_size, etc.)
        Other args same as compute_linear_separators
        
    Returns:
        Dictionary containing learned linear separators and logs
    """
    # Use one-pass method if requested
    if use_onepass:
        from utils.compute_concepts_utils_onepass import compute_linear_separators_onepass
        
        # Set default onepass kwargs if not provided
        if onepass_kwargs is None:
            onepass_kwargs = {}
        
        # Set defaults for onepass method
        onepass_defaults = {
            'lambda_reg': weight_decay if weight_decay > 0 else 1e-3,
            'chunk_size': 100000,
            'use_fp16_chunks': True,
            'normalize_cavs': True,
            'sign_align': True
        }
        
        # Merge defaults with provided kwargs
        for key, value in onepass_defaults.items():
            if key not in onepass_kwargs:
                onepass_kwargs[key] = value
        
        print(f"Using one-pass accumulation method for {len(gt_samples_per_concept)} concepts")
        return compute_linear_separators_onepass(
            embeddings_path=embeddings_path,
            gt_samples_per_concept=gt_samples_per_concept,
            dataset_name=dataset_name,
            sample_type=sample_type,
            model_input_size=model_input_size,
            device=device,
            output_file=output_file,
            **onepass_kwargs
        )
    
    # Use batched training if requested and makes sense (many concepts)
    if use_batched and len(gt_samples_per_concept) > 100:
        from utils.compute_concepts_utils_batched_optimized import train_batched_linear_separators_optimized as train_batched_linear_separators
        print(f"Using batched training for {len(gt_samples_per_concept)} concepts")
        return train_batched_linear_separators(
            embeddings_path=embeddings_path,
            gt_samples_per_concept=gt_samples_per_concept,
            dataset_name=dataset_name,
            sample_type=sample_type,
            model_input_size=model_input_size,
            device=device,
            output_file=output_file,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            tolerance=tolerance,
            weight_decay=weight_decay,
            lr_step_size=lr_step_size,
            lr_gamma=lr_gamma,
            sample_ratio=1.0,
            min_samples=10,
            use_memmap=True,
            concepts_per_chunk=50
        )
    from utils.memory_management_utils import ChunkedEmbeddingLoader
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Parse dataset name and file from path for ChunkedEmbeddingLoader
    # Expected format: {scratch_dir}Embeddings/{dataset_name}/{embeddings_file}
    path_parts = embeddings_path.split('/')
    embeddings_file = path_parts[-1]
    dataset_name_from_path = path_parts[-2]
    scratch_dir_from_path = '/'.join(path_parts[:-3]) + '/' if len(path_parts) > 3 else ''
    
    # Get embedding info
    loader = ChunkedEmbeddingLoader(dataset_name_from_path, embeddings_file, scratch_dir_from_path, device='cpu')
    total_samples = loader.total_samples
    embedding_dim = loader.embedding_dim
    
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name, model_input_size=model_input_size)
    elif sample_type == 'cls':
        split_df = get_split_df(dataset_name)
    
    concept_names = list(gt_samples_per_concept.keys())
    concept_representations = {}
    logs = {}
    
    # Compute labels
    print("Computing labels")
    all_concept_labels = create_binary_labels(total_samples, gt_samples_per_concept)
    
    # Get indices for train and test splits
    if sample_type == 'patch':
        nonpadding_indices = filter_patches_by_image_presence(split_df.index, dataset_name, model_input_size).tolist()
        nonpadding_set = set(nonpadding_indices)
        
        train_split_indices = split_df[split_df == 'train'].index
        test_split_indices = split_df[split_df == 'test'].index
        
        train_indices = [idx for idx in train_split_indices if idx in nonpadding_set]
        test_indices = [idx for idx in test_split_indices if idx in nonpadding_set]
    else:
        train_indices = list(split_df[split_df == 'train'].index)
        test_indices = list(split_df[split_df == 'test'].index)
    
    # Decide whether to use parallel processing
    num_concepts = len(concept_names)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Use parallel processing if we have multiple concepts and either multiple GPUs or explicit request
    # For single GPU, we can still parallelize since linear separators use minimal memory
    if use_parallel and num_concepts > 1:
        if num_workers is None:
            if num_gpus > 1:
                num_workers = min(num_gpus, num_concepts, 4)  # Multi-GPU: one worker per GPU
            else:
                # Single GPU: use multiple workers since each concept uses minimal memory
                num_workers = min(4, num_concepts)  # Default to 4 workers on single GPU
        
        print(f"Using parallel processing with {num_workers} workers for {num_concepts} concepts")
        
        # Prepare arguments for parallel processing
        args_list = []
        for idx, concept_name in enumerate(concept_names):
            concept_labels = all_concept_labels[concept_name]
            args = (
                concept_name, idx, concept_labels, train_indices, test_indices,
                embeddings_path, embeddings_file, dataset_name_from_path, scratch_dir_from_path,
                embedding_dim, balance_data, batch_size, lr, epochs, weight_decay,
                lr_step_size, lr_gamma, patience, tolerance, num_gpus
            )
            args_list.append(args)
        
        # Run parallel training
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_concept = {
                executor.submit(_train_single_concept_parallel, args): args[0]  # args[0] is concept_name
                for args in args_list
            }
            
            # Process results with progress bar
            with tqdm(total=len(concept_names), desc="Training concepts (parallel)") as pbar:
                for future in as_completed(future_to_concept):
                    concept_name, model_weights, concept_logs = future.result()
                    concept_representations[concept_name] = model_weights
                    logs[concept_name] = concept_logs
                    pbar.update(1)
    
    else:
        # Sequential processing (original code)
        print(f"Using sequential processing for {num_concepts} concepts")
        
        # Process each concept
        for concept_name in tqdm(concept_names):
            print(f"Training linear classifier for concept {concept_name}")
            
            # Get labels for this concept
            concept_labels = all_concept_labels[concept_name]
            
            # Balance dataset if needed
            if balance_data:
                # Convert to tensor for faster indexing
                train_indices_tensor = torch.tensor(train_indices, dtype=torch.long)
                train_labels = concept_labels[train_indices_tensor]
                
                # Use tensor operations for finding positive/negative indices
                pos_indices_tensor = torch.where(train_labels == 1)[0]
                neg_indices_tensor = torch.where(train_labels == 0)[0]
                pos_indices = pos_indices_tensor.tolist()
                neg_indices = neg_indices_tensor.tolist()
                
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    # Balance by undersampling
                    n_samples = min(len(pos_indices), len(neg_indices))
                    balanced_pos = random.sample(pos_indices, n_samples)
                    balanced_neg = random.sample(neg_indices, n_samples)
                    balanced_indices = balanced_pos + balanced_neg
                    balanced_train_indices = [train_indices[i] for i in balanced_indices]
                    
                    # Now balance test set
                    test_indices_tensor = torch.tensor(test_indices, dtype=torch.long)
                    test_labels = concept_labels[test_indices_tensor]
                    test_pos_indices = torch.where(test_labels == 1)[0].tolist()
                    test_neg_indices = torch.where(test_labels == 0)[0].tolist()
                    
                    if len(test_pos_indices) > 0 and len(test_neg_indices) > 0:
                        n_test_samples = min(len(test_pos_indices), len(test_neg_indices))
                        balanced_test_pos = random.sample(test_pos_indices, n_test_samples)
                        balanced_test_neg = random.sample(test_neg_indices, n_test_samples)
                        balanced_test_indices = balanced_test_pos + balanced_test_neg
                        balanced_test_indices_global = [test_indices[i] for i in balanced_test_indices]
                        
                        # Preload both train and test embeddings together
                        train_embeddings, train_labels_balanced, test_embeddings, test_labels_balanced = preload_train_test_embeddings(
                            embeddings_path,
                            balanced_train_indices,
                            balanced_test_indices_global,
                            concept_labels,
                            loader
                        )
                        
                        from torch.utils.data import TensorDataset
                        train_dataset = TensorDataset(train_embeddings, train_labels_balanced)
                        test_dataset = TensorDataset(test_embeddings, test_labels_balanced)
                    else:
                        # No test samples
                        test_dataset = None
                        train_dataset = None
                        print(f"  Warning: No positive or negative test samples for concept {concept_name}")
                else:
                    # No positive or negative samples
                    train_dataset = None
                    test_dataset = None
            
            else:
                # Not balancing - preload all train and test indices together
                train_embeddings, train_labels_all, test_embeddings, test_labels_all = preload_train_test_embeddings(
                    embeddings_path,
                    train_indices,
                    test_indices,
                    concept_labels,
                    loader
                )
                
                from torch.utils.data import TensorDataset
                train_dataset = TensorDataset(train_embeddings, train_labels_all)
                test_dataset = TensorDataset(test_embeddings, test_labels_all)
            
            # Create dataloaders
            if train_dataset is not None and len(train_dataset) > 0 and test_dataset is not None and len(test_dataset) > 0:
                train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                model_weights, concept_logs = train_model(train_dl, test_dl, epochs, lr, weight_decay, 
                                                        lr_step_size, lr_gamma, patience, tolerance, device)
            else:
                # No samples, create random model
                model = create_linear_model(embedding_dim, device)
                model_weights = model.weight.detach().squeeze(0).cpu()
                concept_logs = []
                if train_dataset is None or len(train_dataset) == 0:
                    print(f"  No train samples for concept {concept_name}, using random model")
                elif test_dataset is None or len(test_dataset) == 0:
                    print(f"  No test samples for concept {concept_name}, using random model")
                # Initialize empty tensors for cleanup
                train_embeddings = torch.tensor([])
                test_embeddings = torch.tensor([])
            
            concept_representations[concept_name] = model_weights
            logs[concept_name] = concept_logs
            
            # Memory cleanup
            del train_embeddings, test_embeddings
            if train_dataset:
                del train_dataset
            if test_dataset:
                del test_dataset
            torch.cuda.empty_cache()
            gc.collect()
    
    if output_file:
        torch.save(concept_representations, f'Concepts/{dataset_name}/{output_file}')
        print(f"Concepts saved to Concepts/{dataset_name}/{output_file} :)")
        torch.save(logs, f'Concepts/{dataset_name}/logs_{output_file}')
        print(f"Logs saved to Concepts/{dataset_name}/logs_{output_file}")
    
    return concept_representations, logs
  



def filter_embeddings_by_patch_activations(embeddings, act_metrics, gt_samples_per_concept, top_percent, split, dataset_name,
                                           model_input_size, use_gt_labels=True,
                                           impose_negatives=False):
    """
    For each concept, select the top n% of *split* samples based on cosine similarity and
    randomly sample the same number from the rest. Return filtered embeddings and binary labels.

    Args:
        embeddings (torch.Tensor): Tensor of shape (n_samples, hidden_dim)
        act_metrics (pd.DataFrame): DataFrame of shape (n_samples, n_concepts) with cosine similarities
        top_percent (float): Percentage (0 < top_percent < 1) of top samples to select
        split (str): 'train' or 'test'
        dataset_name (str): Name of dataset (used to get train/test split)
        model_input_size (tuple): Needed for indexing the split
        impose_negatives (bool): If True, selects most negative activations instead of random negatives.

    Returns:
        dict: Dictionary mapping each concept to a tuple (filtered_embeddings, labels)
    """
    assert 0 < top_percent < 1, "top_percent must be between 0 and 1"
    np.random.seed(42)

    concept_names = act_metrics.columns
    selected_embeddings, selected_labels = {}, {}
    
    if use_gt_labels: #use actual labels
        all_concept_labels = create_binary_labels(embeddings.shape[0], gt_samples_per_concept)
    else: #consider only superpatches as 'positive' examples
        all_concept_labels = {concept:None for concept in concept_names}
        
    for concept in concept_names:
        # concept_embeds, labels = filter_concept_by_patch_activations(
        #     embeddings, act_metrics[concept].to_numpy(), top_percent, all_concept_labels[concept], impose_negatives
        # )
        concept_embeds, labels = filter_concept_by_patch_activations(split, embeddings, act_metrics[concept], top_percent,
                                        all_concept_labels[concept], dataset_name, model_input_size, impose_negatives)
        selected_embeddings[concept] = concept_embeds
        selected_labels[concept] = labels

    print(f"Selecting top {top_percent*100}% {split} patches ({len(labels) // 2})")

    return selected_embeddings, selected_labels



def compute_linear_separators_w_superpatches(top_per, embeds, original_dists, gt_samples_per_concept, dataset_name,
                                             model_input_size, 
                                  device='cuda', output_file=None, lr=0.01, epochs=100, batch_size=32, patience=15, 
                                  tolerance=3, weight_decay=1e-4, lr_step_size=10, lr_gamma=0.5, balance_data=True, 
                                  balance_negatives=False, impose_negatives=False):
    """
    Computes linear separators for concepts.
    
    Args:
        embeds: Dictionary mapping concept names to embedding tensors.
        gt_samples_per_concept: Dictionary mapping concept names to lists of positive sample indices.
        dataset_name: Name of the dataset.
        sample_type: Type of sampling method.
        model_input_size: Input size for the model.
        device: Compute device (default: 'cuda').
        output_file: Path to save results (default: None).
        lr, epochs, batch_size, patience, tolerance, weight_decay, lr_step_size, lr_gamma: Training hyperparameters.
        balance_data: Whether to balance positive and negative samples.
        balance_negatives: Whether to balance negative samples across concepts.

    Returns:
        Dictionary containing learned linear separators and logs.
    """  
    # Might have removeed concepts originally
    curr_concepts = list(original_dists.columns)
    gt_samples_per_concept = {c: samples for c, samples in gt_samples_per_concept.items() if c in curr_concepts}
    
    concept_representations = {}
    logs = {}
    
    # Separate train and test_data (filtering out patches that don't correspond to any image locations)
    train_all_concept_embeds, train_all_concept_labels = filter_embeddings_by_patch_activations(embeds, original_dists, top_per, 
                                                                         'train', dataset_name, model_input_size,
                                                                        impose_negatives=impose_negatives)
    test_all_concept_embeds, test_all_concept_labels = filter_embeddings_by_patch_activations(embeds, original_dists, top_per, 
                                                                         'test', dataset_name, model_input_size,
                                                                        impose_negatives=impose_negatives)
    
    for concept_name in tqdm(curr_concepts):
        train_embeds = train_all_concept_embeds[concept_name]
        test_embeds = test_all_concept_embeds[concept_name]
        linear_separator, concept_logs = compute_a_linear_separator(concept=concept_name, 
                                                                    train_embeds=train_embeds, test_embeds=test_embeds,
                                                                    train_all_concept_labels=train_all_concept_labels,
                                                                    test_all_concept_labels=test_all_concept_labels,
                                                                    lr=lr, epochs=epochs, patience=patience, 
                                                                    tolerance=tolerance, batch_size=batch_size,
                                                                    weight_decay=weight_decay, lr_step_size=lr_step_size,
                                                                    lr_gamma=lr_gamma, device=device,
                                                                    balance_data=balance_data,
                                                                    balance_negatives=balance_negatives)
        concept_representations[concept_name] = linear_separator
        logs[concept_name] = concept_logs
    
    if output_file:
        torch.save(concept_representations, f'Concepts/{dataset_name}/{output_file}')
        print(f"Concepts saved to Concepts/{dataset_name}/{output_file} :)")
        torch.save(logs, f'Concepts/{dataset_name}/logs_{output_file}')
        print(f"Logs saved to Concepts/{dataset_name}/logs_{output_file}")
    
    return concept_representations, logs


def compute_linear_separators_w_superpatches_across_pers(top_pers, embeds, original_dists, gt_samples_per_concept, 
                                                         dataset_name, model_input_size, device='cuda', output_file=None,
                                                         lr=0.01, epochs=100, batch_size=32, patience=15, 
                                  tolerance=3, weight_decay=1e-4, lr_step_size=10, lr_gamma=0.5, balance_data=True, 
                                  balance_negatives=False, impose_negatives=False):
    for top_per in top_pers:
        if impose_negatives:
            per_output_file = f'imposeneg_per_{top_per}_{output_file}'
        else:
            per_output_file = f'per_{top_per}_{output_file}'
        print(f"Computing classifiers using {top_per* 100}% superpatches")
        # if per_output_file in os.listdir(f'Concepts/{dataset_name}'): #skip if already computed
        #     continue
        # else:
        compute_linear_separators_w_superpatches(top_per, embeds, original_dists, gt_samples_per_concept, dataset_name, 
                                                 model_input_size, device=device, output_file=per_output_file, lr=lr, 
                                                 epochs=epochs, batch_size=batch_size, patience=patience, 
                                                  tolerance=tolerance, weight_decay=weight_decay, 
                                                 lr_step_size=lr_step_size, lr_gamma=lr_gamma,
                                                  balance_data=balance_data, 
                                                  balance_negatives=balance_negatives,
                                                  impose_negatives=impose_negatives)
        


### Functions for Visualizing Patch Methods ###

def plot_similar_patches_to_given_patch(image_index, patch_index_in_image, embeddings, images, save_path, 
                                        patch_size=14, top_k=5, model_input_size=(224, 224)):
    """
    Given a patch index, this function plots the given patch and the most similar patches based on cosine similarity.
    
    Args:
        image_index (int): The index of the selected image.
        patch_index_in_image (int): The image of the selected patch in the image.
        embeddings (torch.Tensor): A tensor containing the embeddings for each patch (shape: n_patches x embed_dim).
        images (list of PIL.Image): A list of images corresponding to the patches.
        patch_size (int): Size of each patch.
        top_k (int): Number of top similar patches to display.
        model_input_size (tuple): The input size to which the image should be resized (width, height).
    
    Returns:
        None: Displays the patches and their respective images with highlighted locations.
    """
    image = images[image_index]
    patch_idx = get_global_patch_idx(image_index, patch_index_in_image, images, 
                           patch_size=patch_size, model_input_size=model_input_size)
    
    left, top, right, bottom = calculate_patch_location(image, patch_idx, patch_size)
    
    make_image_with_highlighted_patch(image, left, top, right, bottom, plot_image_title=f'Image {image_index}: Patch {patch_index_in_image}')
    
    # Get the embedding of the selected patch
    patch_embedding = embeddings[patch_idx]

    # Compute cosine similarities between the patch embedding and all patch embeddings
    cos_sims = cosine_similarity(patch_embedding.unsqueeze(0).cpu().numpy(), 
                                 embeddings.cpu().numpy()).flatten()

    # Sort by similarity and get the top k similar patches
    top_k_patch_indices = cos_sims.argsort()[::-1][:top_k]
    
    overall_title = f'{top_k} Patches Most Similar to Image {image_index}, Patch {patch_index_in_image}'
    plot_patches_w_corr_images(top_k_patch_indices, cos_sims, images, overall_title, save_path=save_path, patch_size=patch_size, model_input_size=model_input_size)
    
    
def find_top_k_concepts_for_patch(patch_idx, embeddings, concepts, top_k=5):
    """
    Find the top k concepts that are most similar to the embedding of the given patch.
    
    Args:
        patch_idx (int): The index of the selected patch in the embeddings tensor.
        embeddings (torch.Tensor): A tensor containing the embeddings for each patch (shape: n_patches x embed_dim).
        concepts (dict): A dictionary where the key is the concept label and the value is the concept embedding tensor 
                         (shape: n_concepts x embed_dim).
        top_k (int): The number of top concepts to return based on similarity (default is 5).
    
    Returns:
        top_k_concepts (list): List of the concept labels corresponding to the top k most similar concepts.
        top_k_sims (list): List of cosine similarity values corresponding to the top k most similar concepts.
    """
    # Get the embedding of the selected patch
    patch_embedding = embeddings[patch_idx]

    # Compute cosine similarities between the patch embedding and all concept embeddings
    cos_sims = []
    for concept_label, concept_tensor in concepts.items():
        sim = cosine_similarity(patch_embedding.unsqueeze(0).cpu().numpy(), 
                                concept_tensor.unsqueeze(0).cpu().numpy()).flatten()
        cos_sims.append((concept_label, sim))

    # Sort by similarity and get the top k concepts
    cos_sims.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity value (descending order)
    top_k_concepts = [x[0] for x in cos_sims[:top_k]]
    top_k_sims = [x[1][0] for x in cos_sims[:top_k]]
    
    return top_k_concepts, top_k_sims


def plot_concepts_most_aligned_w_chosen_patch(image_index, patch_index_in_image, images, embeddings, concepts, 
                                              cos_sims, save_dir, overall_label, patch_size=14, k_concepts=5,
                                              n_examples_per_concept=5, model_input_size=(224, 224)):
    """
    Allow the user to manually select a patch from an image, find the top k most aligned concepts to the patch, 
    and display the corresponding patches of these top k concepts.

    Args:
        image_index (int): The index of the selected image.
        patch_index_in_image (int): The index of the selected patch within the image.
        images (list of PIL.Image): A list of images to choose from.
        embeddings (torch.Tensor): A tensor containing the embeddings for each patch (shape: n_patches x embed_dim).
        concepts (dict): A dictionary where the key is the concept label and the value is the concept embedding tensor 
        cos_sims (pd.Dataframe) : cosine similarities between each patch 
        save_dir (str) : Dir to see if the plot is already saved.
        overall_label (str): Label to help find correct images.
        patch_size (int): The size of each patch (default is 14).
        k_concepts (int): The number of top concepts to return based on similarity (default is 5).
        n_examples_per_concept (int): The number of aligning patches per concept to display.
    
    Returns:
        None: Displays the chosen patch and the patches for the top aligned concepts.
    """
    # Plot the selected patch with the corresponding image
    image = images[image_index]
    patch_idx = get_global_patch_idx(image_index, patch_index_in_image, images, 
                           patch_size=patch_size, model_input_size=model_input_size)
    
    left, top, right, bottom = calculate_patch_location(image, patch_idx, patch_size)
    make_image_with_highlighted_patch(image, left, top, right, bottom, plot_image_title=f'Image {image_index}: Patch {patch_index_in_image}')

    # Find the top k concepts most similar to the selected patch
    top_k_concepts, top_k_sims = find_top_k_concepts_for_patch(patch_idx, embeddings, concepts, k_concepts)

    # Plot the patches for each of the top k concepts
    for i, concept_label in enumerate(top_k_concepts):
        print(f'Rank {i+1}: Concept {concept_label} (Sim: {top_k_sims[i]:.2f})')
        print(f"Plotting top patches for concept {concept_label}")
        
        save_path = f'{save_dir}/{n_examples_per_concept}_patches_simto_concept_{concept_label}__{overall_label}'
        plot_top_patches_for_concept(str(concept_label), cos_sims, images, save_path, top_n=n_examples_per_concept, patch_size=patch_size, model_input_size=model_input_size)

    
### Other ###
def calculate_wcss(embeddings, labels, cluster_centers):
    """
    Calculates the Within-Cluster Sum of Squares (WCSS).
    
    Args:
        embeddings (torch.Tensor): Tensor of shape (n_samples, n_features) containing the data points.
        labels (torch.Tensor or array-like): Cluster labels for each sample.
        cluster_centers (torch.Tensor): Tensor of shape (n_clusters, n_features) representing cluster centroids.
    
    Returns:
        float: The WCSS value.
    """
    # Ensure labels are a tensor on the same device as embeddings
    embeddings = embeddings.to(labels.device)
    
    # For each data point, select its corresponding cluster center.
    # This uses advanced indexing: cluster_centers[labels] returns a tensor of shape (n_samples, n_features)
    centers_for_samples = cluster_centers[labels]
    
    # Compute the squared Euclidean distance between each sample and its assigned center
    squared_diffs = (embeddings - centers_for_samples) ** 2
    squared_distances = squared_diffs.sum(dim=1)
    
    # WCSS is the sum of these squared distances over all samples
    wcss = squared_distances.sum().item()
    
    # Average WCSS over the number of samples
    avg_wcss = wcss / embeddings.shape[0]
    return avg_wcss

def calculate_davies_bouldin(embeds, cluster_labels):
    """
    Compute the Davies-Bouldin Index.
    
    Args:
        embeds (torch.Tensor): Tensor of shape (N, D) where N is the number of samples and D is the embedding dimension.
        cluster_labels (torch.Tensor): Tensor of shape (N,) where each element is the cluster label assigned to the corresponding sample.
    
    Returns:
        float: The Davies-Bouldin Index score.
    """
    # Convert embeddings and labels to numpy arrays (required by scikit-learn)
    embeds_np = embeds.cpu().numpy()
    cluster_labels_np = cluster_labels.cpu().numpy()

    # Compute the Davies-Bouldin Index using scikit-learn's function
    db_index = davies_bouldin_score(embeds_np, cluster_labels_np)
    return db_index


def compute_calinski_score(embeds, cluster_labels):
    """
    Compute the Calinski-Harabasz Index.
    
    Args:
        embeds (torch.Tensor): Tensor of shape (N, D) where N is the number of samples and D is the embedding dimension.
        cluster_labels (torch.Tensor): Tensor of shape (N,) where each element is the cluster label assigned to the corresponding sample.
    
    Returns:
        float: The Calinski-Harabasz Index score.
    """
    # Convert embeddings and labels to numpy arrays (required by scikit-learn)
    embeds_np = embeds.cpu().numpy()
    cluster_labels_np = cluster_labels.cpu().numpy()

    # Compute the Calinski-Harabasz Index using scikit-learn's function
    ch_index = calinski_harabasz_score(embeds_np, cluster_labels_np)
    return ch_index

def evaluate_clustering_metrics(n_clusters_list, embeddings, dataset_name, device, model_input_size, concepts_filenames=None, sample_type='patch'):
    """
    Evaluates clustering performance using multiple metrics and updates a live plot.
    
    Args:
        n_clusters_list (list): List of cluster numbers to evaluate.
        embeddings (torch.Tensor or np.array): Embeddings to be clustered.
        dataset_name (str): Name of the dataset.
        device (str): Compute device (e.g., 'cuda' or 'cpu').
    
    Returns:
        dict: Dictionary containing metric values.
    """
    #separate embeddings into test and train
    relevant_indices = torch.arange(embeddings.shape[0])
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name)
        # Filter patches that are 'padding' given the preprocessing schemes
        relevant_indices = filter_patches_by_image_presence(relevant_indices, dataset_name, model_input_size).tolist()
        
    elif sample_type == 'cls':
        split_df = get_split_df(dataset_name)
        relevant_indices = split_df.index
    
    
    # Get train and test image indices from split_df
    train_image_indices = split_df[split_df == 'train'].index
    test_image_indices = split_df[split_df == 'test'].index
    train_relevant_indices = [idx for idx in relevant_indices if idx in train_image_indices]
    test_relevant_indices = [idx for idx in relevant_indices if idx in test_image_indices]

    train_embeddings = embeddings[train_relevant_indices]
    test_embeddings = embeddings[test_relevant_indices]
    
    metrics = {
        'Train WCSS': [],
        'Test WCSS': [],
        'Train Davies-Bouldin Index': [],
        'Test Davies-Bouldin Index': [],
        'Train Calinski Score': [],
        'Test Calinski Score': []
    }

    # Initialize plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    metric_names = ['WCSS', 'Davies-Bouldin Index', 'Calinski Score']
    colors = ['blue', 'orange']
    lines = {}

    for i, metric in enumerate(metric_names):
        for j, split in enumerate(['Train', 'Test']):
            label = f"{split} {metric}"
            line, = axes[i].plot([], [], label=label, color=colors[j], marker='o' if j == 0 else 'x')
            lines[label] = line
        axes[i].set_title(f'{metric} vs Number of Clusters')
        axes[i].set_xlabel('Number of Clusters')
        axes[i].legend()

    plt.tight_layout()
    display(fig)

    for i, n_clusters in enumerate(n_clusters_list):
        # Get the clustering results
        
        #no saving
        # train_labels, test_labels, cluster_centers = run_fast_pytorch_kmeans(n_clusters, train_embeddings, test_embeddings, device)
        
        #saving
        concepts_filename = concepts_filenames[i]
        if not os.path.exists(f'Concepts/{dataset_name}/cluster_centers_{concepts_filename}'):
            gpu_kmeans(n_clusters, embeddings, dataset_name, device, model_input_size, concepts_filename, sample_type=sample_type, map_samples=False)
        cluster_centers = torch.load(f'Concepts/{dataset_name}/cluster_centers_{concepts_filename}', weights_only=False)
        train_labels = torch.load(f'Concepts/{dataset_name}/train_labels_{concepts_filename}', weights_only=False)
        test_labels = torch.load(f'Concepts/{dataset_name}/test_labels_{concepts_filename}', weights_only=False)
        

        # Compute metrics
        train_wcss = calculate_wcss(train_embeddings, train_labels, cluster_centers)
        test_wcss = calculate_wcss(test_embeddings, test_labels, cluster_centers)
        train_db = calculate_davies_bouldin(train_embeddings, train_labels)
        test_db = calculate_davies_bouldin(test_embeddings, test_labels)
        train_ch = compute_calinski_score(train_embeddings, train_labels)
        test_ch = compute_calinski_score(test_embeddings, test_labels)

        # Append metrics
        metrics['Train WCSS'].append(train_wcss)
        metrics['Test WCSS'].append(test_wcss)
        metrics['Train Davies-Bouldin Index'].append(train_db)
        metrics['Test Davies-Bouldin Index'].append(test_db)
        metrics['Train Calinski Score'].append(train_ch)
        metrics['Test Calinski Score'].append(test_ch)

        # Update plot data
        for i, metric in enumerate(metric_names):
            for split in ['Train', 'Test']:
                label = f"{split} {metric}"
                lines[label].set_xdata(n_clusters_list[:len(metrics[label])])
                lines[label].set_ydata(metrics[label])

            axes[i].relim()  # Recalculate limits
            axes[i].autoscale_view()  # Autoscale

        # Refresh the plot
        clear_output(wait=True)
        display(fig)

    # Ensure the final plot remains visible
    plt.close(fig) 
    
    
def plot_train_history(train_history, metric_type, concepts=None):
    """
    Plots the train and test metrics over epochs for multiple concepts with different colors.

    Args:
        train_history (dict): Dictionary where keys are concept names and values are dictionaries 
                               containing 'train_*' and 'test_*' metric lists for each concept over epochs.
        metric_type (str): The metric to plot, e.g., 'loss', 'accuracy', or 'f1'.
    """
    if concepts is None:
        num_concepts = len(train_history)
    else:
        num_concepts = len(concepts)
        train_history = {k:v for k, v in train_history.items() if k in concepts}
        
    num_cols = 3  # 3 plots per row
    num_rows = math.ceil(num_concepts / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    if num_rows == 1:
        axes = np.array(axes)  # Ensure it's an array for consistent indexing

    axes = axes.flatten()  # Flatten in case of fewer concepts than grid slots

    for i, (concept, metrics) in enumerate(train_history.items()):
        # Retrieve the relevant metrics
        train_metric = metrics[f'train_{metric_type}']
        test_metric = metrics[f'test_{metric_type}']
        
        axes[i].plot(train_metric, label=f"Train {metric_type}", color='blue', marker='o')
        axes[i].plot(test_metric, label=f"Test {metric_type}", color='red', marker='o')
        
        axes[i].set_title(f"{metric_type} for {concept}")
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel(metric_type)
        axes[i].legend()

    # Hide empty subplots if the number of concepts isn't a multiple of 3
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    

def plot_avg_train_history(train_history, metric_type, concepts=None):
    """
    Plots the average train and test metrics over epochs across all (or selected) concepts.
    Shorter concept histories are padded by repeating the final value.
    
    Args:
        train_history (dict): concept -> {'train_*': [...], 'test_*': [...]}
        metric_type (str): Metric name like 'loss', 'accuracy', or 'f1'.
        concepts (list of str, optional): Which concepts to include. Defaults to all.
    """
    if concepts is not None:
        train_history = {k: v for k, v in train_history.items() if k in concepts}

    # Get the max number of epochs across all concepts
    max_epochs = max(len(v[f'train_{metric_type}']) for v in train_history.values())

    def pad_to_length(arr, length):
        """Pad array with last value to a fixed length"""
        if len(arr) == length:
            return arr
        return arr + [arr[-1]] * (length - len(arr))

    # Pad each metric list to max_epochs
    train_metrics = np.array([
        pad_to_length(v[f'train_{metric_type}'], max_epochs)
        for v in train_history.values()
    ])
    test_metrics = np.array([
        pad_to_length(v[f'test_{metric_type}'], max_epochs)
        for v in train_history.values()
    ])

    # Compute means and stds
    avg_train = train_metrics.mean(axis=0)
    std_train = train_metrics.std(axis=0)
    avg_test = test_metrics.mean(axis=0)
    std_test = test_metrics.std(axis=0)

    # Plot
    epochs = np.arange(max_epochs)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, avg_train, label=f"Train {metric_type}", color='blue')
    plt.fill_between(epochs, avg_train - std_train, avg_train + std_train, color='blue', alpha=0.2)
    plt.plot(epochs, avg_test, label=f"Test {metric_type}", color='red')
    plt.fill_between(epochs, avg_test - std_test, avg_test + std_test, color='red', alpha=0.2)

    plt.xlabel("Epochs")
    plt.ylabel(f"Average {metric_type}")
    plt.title(f"Average {metric_type} Over Epochs (across concepts)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    
    
def plot_train_history_justtrain(train_history, metric_type, concepts=None):
    """
    Plots the train and test metrics over epochs for multiple concepts with different colors.

    Args:
        train_history (dict): Dictionary where keys are concept names and values are dictionaries 
                               containing 'train_*' and 'test_*' metric lists for each concept over epochs.
        metric_type (str): The metric to plot, e.g., 'loss', 'accuracy', or 'f1'.
    """
    if concepts is None:
        num_concepts = len(train_history)
    else:
        num_concepts = len(concepts)
        train_history = {k:v for k, v in train_history.items() if k in concepts}
        
    num_cols = 3  # 3 plots per row
    num_rows = math.ceil(num_concepts / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    if num_rows == 1:
        axes = np.array(axes)  # Ensure it's an array for consistent indexing

    axes = axes.flatten()  # Flatten in case of fewer concepts than grid slots

    for i, (concept, metrics) in enumerate(train_history.items()):
        # Retrieve the relevant metrics
        train_metric = metrics[f'{metric_type}']
        
        axes[i].plot(train_metric, color='blue')
        
        axes[i].set_title(f"{metric_type} for {concept}")
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel(metric_type)

    # Hide empty subplots if the number of concepts isn't a multiple of 3
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def update_embedding_stats_json(dataset_name, embeddings_file, embeds_dic, model_input_size):
    """
    Update the embedding statistics JSON file when new embeddings are saved.
    
    Args:
        dataset_name: Name of the dataset
        embeddings_file: Filename of the embeddings (e.g., 'Llama_patch_embeddings_percentthrumodel_100.pt')
        embeds_dic: Dictionary containing embeddings data
        model_input_size: Input size used for the model
    """
    stats_file = 'Embeddings/embedding_stats.json'
    
    # Parse filename to extract model name, embedding type, and percent
    filename_parts = embeddings_file.replace('.pt', '').split('_')
    
    # Extract model name (first part)
    model_name = filename_parts[0]
    
    # Extract embedding type (patch or cls)
    if 'patch' in embeddings_file:
        embedding_type = 'patch'
    elif 'cls' in embeddings_file:
        embedding_type = 'cls'
    else:
        print(f"Warning: Could not determine embedding type from filename: {embeddings_file}")
        return
    
    # Extract percent through model
    if 'percentthrumodel' in embeddings_file:
        idx = filename_parts.index('percentthrumodel')
        if idx + 1 < len(filename_parts):
            percent_thru_model = filename_parts[idx + 1]
        else:
            print(f"Warning: Could not extract percent from filename: {embeddings_file}")
            return
    else:
        print(f"Warning: 'percentthrumodel' not found in filename: {embeddings_file}")
        return
    
    # Load existing stats or create new structure
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            all_stats = json.load(f)
    else:
        all_stats = {}
    
    # Create nested structure if needed
    if dataset_name not in all_stats:
        all_stats[dataset_name] = {}
    if model_name not in all_stats[dataset_name]:
        all_stats[dataset_name][model_name] = {}
    if embedding_type not in all_stats[dataset_name][model_name]:
        all_stats[dataset_name][model_name][embedding_type] = {}
    
    # Extract statistics from the embeddings dictionary
    stats = {
        'filename': embeddings_file,
        'filepath': f'Embeddings/{dataset_name}/{embeddings_file}',
        'file_size_mb': os.path.getsize(f'Embeddings/{dataset_name}/{embeddings_file}') / (1024 * 1024)
    }
    
    # Add mean embedding info
    if 'mean_train_embedding' in embeds_dic:
        mean_emb = embeds_dic['mean_train_embedding']
        stats['mean_embedding_shape'] = list(mean_emb.shape)
        stats['mean_embedding_dim'] = mean_emb.shape[0]
        # Convert to list for JSON serialization
        stats['mean_embedding'] = mean_emb.cpu().numpy().tolist()
    
    # Add norm info
    if 'train_norm' in embeds_dic:
        norm = embeds_dic['train_norm']
        if isinstance(norm, torch.Tensor):
            stats['train_norm'] = float(norm.item())
        else:
            stats['train_norm'] = float(norm)
    
    # Add embedding shape info
    if 'normalized_embeddings' in embeds_dic:
        emb = embeds_dic['normalized_embeddings']
        stats['num_embeddings'] = emb.shape[0]
        stats['embedding_dim'] = emb.shape[1]
    
    # Update the stats for this specific configuration
    all_stats[dataset_name][model_name][embedding_type][percent_thru_model] = stats
    
    # Save updated stats
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    # Also update the lite version (without embeddings)
    stats_lite_file = stats_file.replace('.json', '_lite.json')
    
    # Create lite version by excluding mean_embedding
    stats_lite = {k: v for k, v in stats.items() if k != 'mean_embedding'}
    
    # Load existing lite stats or create new
    if os.path.exists(stats_lite_file):
        with open(stats_lite_file, 'r') as f:
            all_stats_lite = json.load(f)
    else:
        all_stats_lite = {}
    
    # Update lite stats
    if dataset_name not in all_stats_lite:
        all_stats_lite[dataset_name] = {}
    if model_name not in all_stats_lite[dataset_name]:
        all_stats_lite[dataset_name][model_name] = {}
    if embedding_type not in all_stats_lite[dataset_name][model_name]:
        all_stats_lite[dataset_name][model_name][embedding_type] = {}
    
    all_stats_lite[dataset_name][model_name][embedding_type][percent_thru_model] = stats_lite
    
    # Save lite stats
    with open(stats_lite_file, 'w') as f:
        json.dump(all_stats_lite, f, indent=2)
    
    print(f"Updated embedding stats for {dataset_name}/{model_name}/{embedding_type}/{percent_thru_model}")
