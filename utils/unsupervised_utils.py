import torch
from tqdm import tqdm
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
from torch.nn.utils.rnn import pad_sequence
import  csv
import torch.nn.functional as F
import ast
import gc
import pyarrow 
from glob import glob

from utils.quant_concept_evals_utils import compute_concept_metrics, filter_patches_by_image_presence, compute_stats_from_counts
from utils.patch_alignment_utils import get_patch_split_df
from utils.general_utils import create_binary_labels, get_split_df, filter_coco_concepts


### Helper Quant Functions ###
def find_closest_cluster_per_concept(clusters, gt_concepts, dataset_name, con_label, device='cpu'):
    """
    For each semantic concept, find the closest cluster vector by cosine similarity.

    Args:
        clusters (dict): cluster_label -> vector tensor [D]
        gt_concepts (dict): concept_name -> vector tensor [D]
        device (str): Device for computation

    Returns:
        dict: concept_name -> (cluster_label, cosine_similarity)
    """
    cluster_labels = list(clusters.keys())
    cluster_matrix = torch.stack([clusters[label] for label in cluster_labels]).to(device)  # [C, D]

    results = {}

    for concept, vector in gt_concepts.items():
        vector = vector.to(device).unsqueeze(0)  # [1, D]
        sims = F.cosine_similarity(cluster_matrix, vector)  # [C]
        best_sim, best_idx = torch.max(sims, dim=0)
        results[concept] = (cluster_labels[best_idx.item()], round(best_sim.item(), 4))

    torch.save(results, f'Unsupervised_Matches/{dataset_name}/{con_label}.pt')
    # print(f"Alignment results saved at Unsupervised_Matches/{dataset_name}/{con_label}.pt :)")
    return results


def find_topk_clusters_per_concept(clusters, gt_concepts, k=5, device='cpu', metric='cosine'):
    """
    For each semantic concept, find the top-k most similar cluster vectors by cosine similarity or Euclidean distance.

    Args:
        clusters (dict): cluster_label -> vector tensor [D]
        gt_concepts (dict): concept_name -> vector tensor [D]
        k (int): Number of top clusters to return
        device (str): Compute device ('cpu' or 'cuda')
        metric (str): 'cosine' or 'euclidean'

    Returns:
        dict: concept_name -> list of (cluster_label, similarity/distance)
    """
    cluster_labels = list(clusters.keys())
    cluster_matrix = torch.stack([clusters[label] for label in cluster_labels]).to(device)  # [C, D]

    if metric == 'cosine':
        cluster_matrix = F.normalize(cluster_matrix, dim=1)

    results = {}

    for concept, vector in gt_concepts.items():
        vector = vector.to(device).unsqueeze(0)  # [1, D]
        if metric == 'cosine':
            vector = F.normalize(vector, dim=1)
            sims = F.cosine_similarity(cluster_matrix, vector)  # [C]
            topk_sims, topk_indices = torch.topk(sims, k=k)
            topk_results = [(cluster_labels[i], round(topk_sims[j].item(), 4)) 
                            for j, i in enumerate(topk_indices)]
        elif metric == 'euclidean':
            # Compute squared Euclidean distance
            dists = torch.norm(cluster_matrix - vector, dim=1)  # [C]
            topk_dists, topk_indices = torch.topk(-dists, k=k)  # negative for closest
            topk_results = [(cluster_labels[i], round(-topk_dists[j].item(), 4))
                            for j, i in enumerate(topk_indices)]
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        results[concept] = topk_results

    return results


def compute_similarity_to_supervised(gt_concepts, concepts, best_clusters_by_detect):
    """
    Computes cosine similarity, Euclidean distance, and similarity/distance rank of each best-matching cluster.

    Args:
        gt_concepts (dict): concept_name -> GT embedding tensor
        concepts (dict): cluster_id -> cluster embedding tensor
        best_clusters_by_detect (dict): concept_name -> {'best_cluster': cluster_id, ...}

    Returns:
        dict: concept_name -> {
            'best_cluster': cluster_id,
            'cosine_sim': float,
            'cosine_rank': int,
            'euclidean_dist': float,
            'euclidean_rank': int
        }
    """
    cluster_labels = list(concepts.keys())
    cluster_matrix = torch.stack([concepts[c] for c in cluster_labels])  # [C, D]
    cluster_matrix = F.normalize(cluster_matrix, dim=1)  # normalize once for cosine

    results = {}

    for gt_concept, matching_info in best_clusters_by_detect.items():
        best_cluster = matching_info['best_cluster']
        best_cluster_idx = cluster_labels.index(best_cluster)

        gt_vec = F.normalize(gt_concepts[gt_concept], dim=0).unsqueeze(0)  # [1, D]

        # Cosine similarity: [C]
        cos_sims = F.cosine_similarity(cluster_matrix, gt_vec)
        cosine_sim = cos_sims[best_cluster_idx].item()
        cosine_rank = (cos_sims > cos_sims[best_cluster_idx]).sum().item() + 1

        # Euclidean distance: [C]
        dists = torch.norm(cluster_matrix - gt_vec, dim=1)
        euclidean_dist = dists[best_cluster_idx].item()
        euclidean_rank = (dists < dists[best_cluster_idx]).sum().item() + 1

        results[gt_concept] = {
            'best_cluster': best_cluster,
            'cosine_sim': cosine_sim,
            'cosine_rank': cosine_rank,
            'euclidean_dist': euclidean_dist,
            'euclidean_rank': euclidean_rank
        }
    return results



def evaluate_topk_cluster_matches(gt_concepts, concepts, topk_clusters_per_concept):
    """
    For each GT concept, evaluate cosine similarity and Euclidean distance for each top-k matching cluster.

    Args:
        gt_concepts (dict): Concept name → torch.Tensor (GT vector)
        concepts (dict): Cluster ID (as string) → torch.Tensor (cluster vector)
        topk_clusters_per_concept (dict): Concept name → list of (cluster_id, cosine_sim)

    Returns:
        dict: Concept → list of dicts with 'cluster', 'cosine_sim', 'euclidean_dist'
    """
    results = {}

    for concept, topk_matches in topk_clusters_per_concept.items():
        gt_vector = F.normalize(gt_concepts[concept], dim=0)
        cluster_results = []

        for cluster_id, _ in topk_matches:
            cluster_vector = F.normalize(concepts[cluster_id], dim=0)

            cosine_sim = torch.dot(gt_vector, cluster_vector).item()
            euclidean_dist = torch.norm(gt_vector - cluster_vector, p=2).item()

            cluster_results.append({
                'cluster': cluster_id,
                'cosine_sim': cosine_sim,
                'euclidean_dist': euclidean_dist
            })

        results[concept] = cluster_results

    return results


# def write_single_concept_cossim_batch(writer, batch_embeddings, concept_vector):
#     """
#     Writes a single column of cosine similarities for a batch to CSV.

#     Args:
#         writer (csv.writer): CSV writer for a single concept
#         batch_sims (torch.Tensor): [B] tensor of similarities
#     """
#     sims = F.cosine_similarity(batch_embeddings, concept_vector.unsqueeze(0), dim=1)
#     for sim in sims.cpu().tolist():
#         writer.writerow([sim])
        
# def compute_cosine_sims_per_concept(embeddings, concepts, dataset_name, device, output_file, scratch_dir, batch_size=32):
#     """
#     Computes cosine similarities between all embeddings and each concept vector separately.
#     Writes one CSV file per concept under Cosine_Similarities/{dataset_name}/{concept_name}.csv.
#     """
#     concept_keys = list(concepts.keys())

#     base_path = f'{scratch_dir}/Cosine_Similarities/{dataset_name}/kmeans'
#     os.makedirs(base_path, exist_ok=True)

#     with torch.no_grad():
#         # counter = 0
#         for concept_name in tqdm(concept_keys, desc="Computing per-concept cosine similarities"):
#             concept_vector = concepts[concept_name].to(device)
            
#             output_path = os.path.join(base_path, f"{concept_name}_{output_file}")
#             if os.path.exists(output_path):
#                 print(f"Skipping {concept_name} — already exists at {output_path}")
#                 continue

#             with open(output_path, mode='w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([concept_name])  # header

#                 for i in range(0, embeddings.shape[0], batch_size):
#                     batch_embeddings = embeddings[i:i+batch_size].to(device)

#                     write_single_concept_cossim_batch(writer, batch_embeddings, concept_vector)

#                     del batch_embeddings
#                     torch.cuda.empty_cache()
#             # counter += 1
#             # if counter > 10:
#             #     break

#             print(f"Saved: {output_path}")
def write_multi_concept_cossim_batch(writer, batch_embeddings, concept_names, concept_matrix, device):
    """
    Computes and writes cosine similarities for a batch across multiple concepts.

    Args:
        writer (csv.DictWriter): Writer object for streaming output.
        batch_embeddings (torch.Tensor): [B, D] batch.
        concept_names (list[str]): Concept column names.
        concept_matrix (torch.Tensor): [C, D] concept vectors.
        device (str): Torch device.
    """
    batch = batch_embeddings.to(device)  # [B, D]
    concept_matrix = F.normalize(concept_matrix.to(device), dim=1)  # [C, D]
    batch = F.normalize(batch, dim=1)  # normalize embeddings too

    sims = batch @ concept_matrix.T  # [B, C]
    sims = sims.cpu().tolist()

    for row in sims:
        writer.writerow(dict(zip(concept_names, row)))

    del batch, sims
    torch.cuda.empty_cache()
    

def compute_cosine_sims_allpairs(embeddings, concepts, dataset_name, device, output_file, scratch_dir, batch_size=32):
    """
    Computes cosine similarities between all embeddings and all concept vectors, writing to one CSV.

    Args:
        embeddings (torch.Tensor): [N, D] patch embeddings.
        concepts (dict[str, torch.Tensor]): Concept name -> [D] concept vector.
        dataset_name (str): Dataset name.
        device (str): CUDA or CPU device.
        output_file (str): Final merged CSV name.
        scratch_dir (str): Base path for output.
        batch_size (int): Number of embeddings to process at once.
    """
    concept_names = list(concepts.keys())
    concept_matrix = torch.stack([concepts[c] for c in concept_names])

    output_path = os.path.join(scratch_dir, "Cosine_Similarities", dataset_name, "kmeans", output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[→] Writing all similarities to: {output_path}")

    with open(output_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=concept_names)
        writer.writeheader()

        for i in tqdm(range(0, embeddings.shape[0], batch_size), desc="Computing similarities"):
            batch_embeddings = embeddings[i:i+batch_size]
            write_multi_concept_cossim_batch(writer, batch_embeddings, concept_names, concept_matrix, device)

    print(f"[✔] Finished writing similarities to {output_path}")

            
            
# def write_single_concept_signed_dist_batch(writer, batch_embeddings, concept_vector):
#     """
#     Computes and writes signed distances for a single concept.
    
#     Args:
#         writer (csv.writer): Writer for a single concept's CSV file.
#         batch_embeddings (torch.Tensor): [B, D] batch.
#         concept_vector (torch.Tensor): [D] concept vector.
#     """
#     norm_weight = torch.norm(concept_vector, p=2)
#     sims = (batch_embeddings @ concept_vector) / norm_weight
#     for sim in sims.cpu().tolist():
#         writer.writerow([sim])


# def compute_signed_distances_per_concept(embeds, concept_weights, dataset_name, device, output_file, scratch_dir, batch_size=100):
#     """
#     Computes signed distances between each embedding and each concept vector.
#     Writes one CSV per concept to Distances/{dataset_name}/.
#     """
#     concept_names = list(concept_weights.keys())
#     base_path = os.path.join(scratch_dir, "Distances", dataset_name)
#     os.makedirs(base_path, exist_ok=True)

#     with torch.no_grad():
#         for concept in tqdm(concept_names, desc="Computing signed distances per concept"):
#             concept_vector = concept_weights[concept].to(device)

#             output_path = os.path.join(base_path, f"{concept}_{output_file}")
#             if os.path.exists(output_path):
#                 print(f"Skipping {concept} — already exists at {output_path}")
#                 continue

#             with open(output_path, mode='w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([concept])  # header

#                 for i in range(0, embeds.shape[0], batch_size):
#                     batch_embeddings = embeds[i:i+batch_size].to(device)
#                     write_single_concept_signed_dist_batch(writer, batch_embeddings, concept_vector)
#                     del batch_embeddings
#                     torch.cuda.empty_cache()

#             print(f"Saved: {output_path}")


def compute_and_write_batch_multi_concept(embeds, batch_size, i, concept_names, concept_matrix, writer, device):
    """
    Computes and writes a single batch of signed distances for multiple concepts.
    Each row is a sample, and each column is a concept.
    """
    batch = embeds[i:i + batch_size].to(device)  # [B, D]
    norm_matrix = torch.norm(concept_matrix, p=2, dim=1, keepdim=True)  # [C, 1]
    sims = (batch @ concept_matrix.T) / norm_matrix.T  # [B, C]
    sims = sims.cpu().tolist()  # convert to nested list: [[sim1, sim2, ...], ...]

    for row in sims:
        row_dict = {concept: sim for concept, sim in zip(concept_names, row)}
        writer.writerow(row_dict)

    del batch
    del sims
    gc.collect()
    torch.cuda.empty_cache()



def compute_signed_distances_streaming_multi(
    embeds, concept_weights, dataset_name, device, output_file, scratch_dir,
    batch_size=100
):
    """
    Computes signed distances for all concepts at once, and writes rows (samples) to CSV.
    Each row = one sample. Each column = one concept.
    """
    output_path = f'{scratch_dir}/Distances/{dataset_name}/kmeans/{output_file}'
    print(f"Writing to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_concepts = list(concept_weights.keys())
    concept_matrix = torch.stack([concept_weights[c] for c in all_concepts]).to(device)  # [C, D]
    norm_matrix = torch.norm(concept_matrix, p=2, dim=1, keepdim=True)  # [C, 1]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_concepts)
        writer.writeheader()

        for i in tqdm(range(0, embeds.shape[0], batch_size), desc="Writing batches"):
            compute_and_write_batch_multi_concept(embeds, batch_size, i, all_concepts, concept_matrix, writer, device)

    print(f"[✔] Done writing full CSV: {output_path}")


# def compute_concept_thresholds_over_percentiles_all_pairs(gt_samples_per_concept, cos_sims, percentiles, device, dataset_name, con_label, n_vectors=5, n_concepts_to_print=0):
#     """
#     Computes thresholds for multiple percentiles, pretending any cluster could match any concept.
    
#     Args:
#         gt_samples_per_concept (dict): Mapping of GT concept name -> list of patch indices
#         cos_sims (pd.DataFrame): Cosine similarity matrix (rows: patches, columns: cluster labels)
#         percentiles (list): List of percentile values to compute thresholds for
#         device (str): Compute device (e.g., "cuda")
#         dataset_name (str): Name of dataset for cache file
#         con_label (str): Label for cache file
#         n_vectors (int): Number of random vectors (unused in current implementation)
#         n_concepts_to_print (int): Number of concepts to print for debugging
        
#     Returns:
#         dict: Mapping from percentile -> (concept, cluster) -> threshold
#     """
#     cache_file = f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt'
    
#     try:
#         all_thresholds = torch.load(cache_file)
#         existing_percentiles = set(all_thresholds.keys())
#         new_percentiles = set(percentiles) - existing_percentiles
        
#         if not new_percentiles:
#             return {p: all_thresholds[p] for p in percentiles}
            
#     except FileNotFoundError:
#         all_thresholds = {}
#         new_percentiles = set(percentiles)
    
#     if new_percentiles:
#         cos_sims_tensor = torch.tensor(cos_sims.values.astype(np.float32), device=device)
#         cluster_labels = list(cos_sims.columns)
#         concept_names = list(gt_samples_per_concept.keys())

#         # Create (concept, cluster) → sims mapping
#         sims_list = []
#         concept_cluster_pairs = []

#         for concept_name in concept_names:
#             sample_indices = gt_samples_per_concept[concept_name]
#             sample_indices_tensor = torch.tensor(sample_indices, device=device)


#             for cluster_label in cluster_labels:
#                 col_idx = cos_sims.columns.get_loc(cluster_label)
#                 sims = cos_sims_tensor[sample_indices_tensor, col_idx]
#                 sims_list.append(sims)
#                 concept_cluster_pairs.append((concept_name, cluster_label))

#         # Pad sequences for batch processing
#         padded_sims = pad_sequence(sims_list, batch_first=True, padding_value=float('nan'))
        
#         percentiles_tensor = torch.tensor([(1 - p) for p in new_percentiles], device=device)
#         thresholds_tensor = torch.nanquantile(padded_sims, percentiles_tensor, dim=1)
        
#         for p_idx, percentile in enumerate(new_percentiles):
#             pair_thresholds = {}
#             for c_idx, (concept_name, cluster_label) in enumerate(concept_cluster_pairs):
#                 threshold_val = thresholds_tensor[p_idx, c_idx].item()
#                 pair_thresholds[(concept_name, cluster_label)] = (threshold_val, np.nan)
            
#             all_thresholds[percentile] = pair_thresholds

#             if n_concepts_to_print > 0:
#                 print(f"\nThresholds at {percentile*100:.1f}%:")
#                 for i, ((concept_name, cluster_label), (threshold, _)) in enumerate(pair_thresholds.items()):
#                     if i >= n_concepts_to_print:
#                         break
#                     print(f"Concept '{concept_name}' vs Cluster '{cluster_label}': {threshold:.4f}")

#         torch.save(all_thresholds, cache_file)

#         del cos_sims_tensor, padded_sims, sims_list, thresholds_tensor

#     return {p: all_thresholds[p] for p in percentiles}

# def compute_cluster_threshold(cluster_label, sample_indices_tensor, new_percentiles, dataset_name, cossim_file, scratch_dir, device):
#     # Load just the cluster column using pandas
#     filepath = f'{scratch_dir}/Cosine_Similarities/{dataset_name}/kmeans/{cluster_label}_{cossim_file}'
#     df = pd.read_csv(filepath)
#     sims_tensor = torch.tensor(df[cluster_label].values, dtype=torch.float32).to(device)

#     # Subselect and clean
#     sims = sims_tensor[sample_indices_tensor]
#     sims = sims[~torch.isnan(sims)]

#     if sims.numel() == 0:
#         return None

#     thresholds = [torch.quantile(sims, 1 - p).item() for p in new_percentiles]
#     return thresholds


# def compute_concept_thresholds_over_percentiles_all_pairs(gt_samples_per_concept, percentiles, device, dataset_name, con_label, cossim_file, scratch_dir, n_vectors=5, n_concepts_to_print=0):
#     """
#     Computes thresholds for multiple percentiles, pretending any cluster could match any concept.
#     """
#     cache_file = f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt'
    
#     try:
#         all_thresholds = torch.load(cache_file)
#         existing_percentiles = set(all_thresholds.keys())
#         new_percentiles = set(percentiles) - existing_percentiles

#         if not new_percentiles:
#             return {p: all_thresholds[p] for p in percentiles}
        
#     except FileNotFoundError:
#         all_thresholds = {}
#         new_percentiles = set(percentiles)

#     if new_percentiles:
#         cluster_labels = [str(i) for i in range(1000)]
#         concept_names = list(gt_samples_per_concept.keys())

#         thresholds_per_percentile = {p: {} for p in new_percentiles}

#         for concept_name in tqdm(concept_names, desc="Computing thresholds for concepts"):
#             sample_indices = gt_samples_per_concept[concept_name]
#             if len(sample_indices) == 0:
#                 continue  # skip if no samples
#             sample_indices_tensor = torch.tensor(sample_indices, device=device)

#             for cluster_label in cluster_labels:
#                 thresholds = compute_cluster_threshold(cluster_label, 
#                                                        sample_indices_tensor, 
#                                                        new_percentiles, dataset_name, cossim_file,
#                                                        scratch_dir, device)
#                 if thresholds is None:
#                     continue

#                 for p, threshold_val in zip(new_percentiles, thresholds):
#                     thresholds_per_percentile[p][(concept_name, cluster_label)] = (threshold_val, np.nan)

#         # Save the computed thresholds into all_thresholds
#         for p in new_percentiles:
#             all_thresholds[p] = thresholds_per_percentile[p]

#         torch.save(all_thresholds, cache_file)

#     return {p: all_thresholds[p] for p in percentiles}


# def compute_thresholds_for_cluster(cluster_label, concept_indices, new_percentiles,
#                                    thresholds_per_percentile, dataset_name, cossim_file,
#                                    scratch_dir, device, cache_file):
#     """
#     Computes thresholds for all concepts for a single cluster and updates thresholds_per_percentile in-place.
#     Also writes out the updated thresholds to cache file for persistence.
#     """
#     filepath = f'{scratch_dir}/Cosine_Similarities/{dataset_name}/kmeans/{cluster_label}_{cossim_file}'
#     try:
#         df = pd.read_csv(filepath)
#     except Exception as e:
#         print(f"Failed to load {filepath}: {e}")
#         return

#     if str(cluster_label) not in df.columns:
#         print(f"Missing cluster column {cluster_label} in {filepath}")
#         return

#     sims_tensor = torch.tensor(df[str(cluster_label)].values, dtype=torch.float32).to(device)

#     for concept, sample_indices_tensor in concept_indices.items():
#         sims = sims_tensor[sample_indices_tensor]
#         sims = sims[~torch.isnan(sims)]

#         for p in new_percentiles:
#             thresholds_per_percentile[p][(concept, str(cluster_label))] = (torch.quantile(sims, 1 - p).item(), float('nan'))
#     torch.save(thresholds_per_percentile, cache_file)


# def compute_concept_thresholds_over_percentiles_all_pairs(gt_samples_per_concept, percentiles, device,
#                                                           dataset_name, con_label, cossim_file,
#                                                           scratch_dir, n_vectors=5, n_concepts_to_print=0):
#     """
#     Computes thresholds for multiple percentiles, pretending any cluster could match any concept.
#     Saves progress incrementally to disk after each cluster to handle interruptions.
#     """
#     cache_file = f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt'
#     os.makedirs(os.path.dirname(cache_file), exist_ok=True)

#     # Load existing thresholds if present
#     try:
#         all_thresholds = torch.load(cache_file)
#         existing_percentiles = set(all_thresholds.keys())
#         new_percentiles = set(percentiles) - existing_percentiles
#         print(f"Loaded cached thresholds. Existing: {existing_percentiles}, New: {new_percentiles}")
#     except FileNotFoundError:
#         all_thresholds = {}
#         new_percentiles = set(percentiles)
#         print(f"No cache found. Will compute for all percentiles: {new_percentiles}")

#         if not new_percentiles:
#             return {p: all_thresholds[p] for p in percentiles}

#         # Initialize threshold storage
#         for p in new_percentiles:
#             all_thresholds[p] = {}

#         cluster_labels = [str(i) for i in range(1000)]
#         concept_names = list(gt_samples_per_concept.keys())
#         thresholds_per_percentile = {p: all_thresholds[p] for p in new_percentiles}

#         # Precompute concept -> sample indices tensor
#         concept_indices = {
#             concept: torch.tensor(indices, device=device)
#             for concept, indices in gt_samples_per_concept.items()
#             if len(indices) > 0
#         }

#         for cluster_label in tqdm(cluster_labels, desc="Computing thresholds per cluster"):
#             compute_thresholds_for_cluster(cluster_label, concept_indices, new_percentiles,
#                                            thresholds_per_percentile, dataset_name, cossim_file,
#                                            scratch_dir, device, cache_file)


# def get_patch_detection_tensor_all_pairs(act_metrics, detect_thresholds, model_input_size, dataset_name, device, patch_size=14):
#     """
#     Computes detection masks for each (concept, cluster) pair (image-level detection, patch-level expansion).
#     """
#     if model_input_size[0] == 'text':
#         raise NotImplementedError("Text mode for all-pairs detection not implemented yet.")
#     else:
#         num_patches_per_image = (model_input_size[0] // patch_size) ** 2
#         num_images = len(act_metrics) // num_patches_per_image
#         sample_indices = torch.arange(len(act_metrics), device=device) // num_patches_per_image

#     detected_patch_masks = {}

#     # Precompute cosine similarities tensor
#     cos_sims_tensor = torch.tensor(act_metrics.values.astype(np.float32), device=device)

#     for (concept, cluster), (threshold, _) in detect_thresholds.items():
#         cluster_idx = act_metrics.columns.get_loc(str(cluster))
#         sims_cluster = cos_sims_tensor[:, cluster_idx]  # (n_patches,)

#         # Check which patches are individually above threshold
#         patch_above_thresh = sims_cluster >= threshold  # (n_patches,) bool

#         # Reshape to (num_images, num_patches_per_image)
#         patch_above_thresh_per_image = patch_above_thresh.view(num_images, num_patches_per_image)

#         # For each image, check if ANY patch is above threshold
#         detected_images = patch_above_thresh_per_image.any(dim=1)  # (num_images,) bool

#         # Expand back to patches
#         detected_mask = detected_images[sample_indices]  # (n_patches,) bool

#         detected_patch_masks[(concept, cluster)] = detected_mask  # Already on device

#     return detected_patch_masks

def get_patch_detection_tensor_all_pairs(act_metrics, detect_thresholds, model_input_size, dataset_name, device, patch_size=14):
    """
    Computes detection masks for each (concept, cluster) pair (image-level detection, patch-level expansion).
    """
    if model_input_size[0] == 'text':
        raise NotImplementedError("Text mode for all-pairs detection not implemented yet.")
    else:
        num_patches_per_image = (model_input_size[0] // patch_size) ** 2
        num_images = len(act_metrics) // num_patches_per_image
        sample_indices = torch.arange(len(act_metrics)) // num_patches_per_image  # <-- CPU tensor

    detected_patch_masks = {}

    for (concept, cluster), (threshold, _) in tqdm(detect_thresholds.items(), desc="Building detection masks"):
        # Fetch sims for this cluster only (stay on CPU)
        sims_cluster_cpu = torch.from_numpy(
            act_metrics[str(cluster)].values.astype(np.float32)
        )  # (n_patches,) CPU tensor

        # Thresholding on CPU
        patch_above_thresh = sims_cluster_cpu >= threshold  # (n_patches,) bool

        # Reshape to (num_images, num_patches_per_image)
        patch_above_thresh_per_image = patch_above_thresh.view(num_images, num_patches_per_image)

        # For each image, check if ANY patch is above threshold
        detected_images = patch_above_thresh_per_image.any(dim=1)  # (num_images,) bool

        # Expand back to patches
        detected_mask = detected_images[sample_indices]  # (n_patches,) bool

        # Finally move to GPU
        detected_patch_masks[(concept, cluster)] = detected_mask

    return detected_patch_masks




### Detection ####
# def find_activated_images_bypatch_allpairs(curr_thresholds, model_input_size, dataset_name, patch_size=14):
#     """
#     Find activated images for each (concept, cluster) pair over patches.
#     """

#     split_df = get_split_df(dataset_name)
#     patches_per_image = (model_input_size[0] // patch_size) ** 2
#     num_images = len(cos_sims) // patches_per_image

#     cos_sims_tensor = torch.tensor(cos_sims.values)
#     cluster_labels = list(cos_sims.columns)

#     reshaped_sims = cos_sims_tensor.reshape(num_images, patches_per_image, -1)  # [num_images, patches_per_image, n_clusters]

#     max_activations = torch.max(reshaped_sims, dim=1)[0]  # [num_images, n_clusters]

#     split_array = np.array(split_df)
#     train_mask = torch.tensor(split_array == 'train')
#     test_mask = torch.tensor(split_array == 'test')

#     activated_images_train = defaultdict(set)
#     activated_images_test = defaultdict(set)

#     concept_names = list({k[0] for k in curr_thresholds.keys()})

#     for (concept, cluster_label) in curr_thresholds:
#         cluster_idx = cluster_labels.index(cluster_label)
#         threshold_val = curr_thresholds[(concept, cluster_label)][0]

#         # Activation logic
#         cluster_activations = max_activations[:, cluster_idx] >= threshold_val

#         # Which images activated
#         train_indices = torch.where(cluster_activations & train_mask)[0].tolist()
#         test_indices = torch.where(cluster_activations & test_mask)[0].tolist()

#         activated_images_train[(concept, cluster_label)].update(train_indices)
#         activated_images_test[(concept, cluster_label)].update(test_indices)

#     return activated_images_train, activated_images_test, activated_images_cal


# def find_activated_images_bypatch_per_cluster(cluster_label, concept_list, curr_thresholds, patches_per_image, dataset_name,
#                                               train_mask, test_mask, cossim_file, scratch_dir, device):
#     cluster_sims = pd.read_csv(f'{scratch_dir}/Cosine_Similarities/{dataset_name}/kmeans/{cluster_label}_{cossim_file}')

#     # Compute max activation over patches
#     num_images = len(cluster_sims) // patches_per_image
#     cluster_tensor = torch.tensor(cluster_sims.values, dtype=torch.float32, device=device)
#     reshaped = cluster_tensor.reshape(num_images, patches_per_image)
#     max_activations = torch.max(reshaped, dim=1)[0]  # [num_images]

#     train_indices_per_concept = {}
#     test_indices_per_concept = {}
#     for concept in concept_list:
#         threshold_val = curr_thresholds[(concept, cluster_label)][0]

#         activated = max_activations >= threshold_val
#         train_indices = torch.where(activated & train_mask[:len(activated)])[0].tolist()
#         test_indices = torch.where(activated & test_mask[:len(activated)])[0].tolist()
        
        
#         train_indices_per_concept[concept] = train_indices
#         test_indices_per_concept[concept] = test_indices
#     return train_indices_per_concept, test_indices_per_concept

# def find_activated_images_bypatch_per_cluster(cluster_label, concept_list, curr_thresholds, patches_per_image, dataset_name,
#                                               train_mask, test_mask, cossim_file, scratch_dir, device):
#     filepath = f'{scratch_dir}/Cosine_Similarities/{dataset_name}/kmeans/{cluster_label}_{cossim_file}'
#     sims = pd.read_csv(filepath, usecols=[cluster_label])  # Only load the needed column
#     sims_tensor = torch.tensor(sims[cluster_label].values, dtype=torch.float32, device=device)

#     num_images = len(sims_tensor) // patches_per_image
#     reshaped = sims_tensor.view(num_images, patches_per_image)
#     max_activations = torch.max(reshaped, dim=1)[0]  # [num_images]

#     # Stack thresholds into a tensor for vectorized comparison
#     thresholds = torch.tensor([curr_thresholds[(concept, cluster_label)][0] for concept in concept_list], device=device).view(-1, 1)
#     max_activations = max_activations.view(1, -1)  # [1, num_images]

#     # Vectorized thresholding: [n_concepts, n_images]
#     activated = (max_activations >= thresholds)  # boolean tensor

#     # Apply split masks (only keep train/test images)
#     train_acts = activated[:, :len(train_mask)] & train_mask.view(1, -1)
#     test_acts = activated[:, :len(test_mask)] & test_mask.view(1, -1)

#     # Convert to concept->list
#     train_indices_per_concept = {
#         concept: torch.where(train_acts[i])[0].tolist()
#         for i, concept in enumerate(concept_list)
#     }
#     test_indices_per_concept = {
#         concept: torch.where(test_acts[i])[0].tolist()
#         for i, concept in enumerate(concept_list)
#     }

#     return train_indices_per_concept, test_indices_per_concept

        

# def find_activated_images_bypatch_allpairs(curr_thresholds, model_input_size, dataset_name,
#                                           cossim_file, scratch_dir, patch_size=14, device='cuda'):
#     """
#     Finds activated images for each (concept, cluster) pair using per-cluster cosine similarity files.

#     Args:
#         curr_thresholds (dict): {(concept, cluster_label) -> (threshold_val, ...)}
#         model_input_size (tuple): (W, H) of input image
#         dataset_name (str): Dataset name (used for splits)
#         cluster_csv_dir (str): Directory with per-cluster cosine similarity CSV files
#         patch_size (int): Patch resolution
#         device (str): Device for tensor operations

#     Returns:
#         (activated_images_train, activated_images_test): dicts mapping (concept, cluster) -> set(image indices)
#     """
#     split_df = get_split_df(dataset_name)
#     split_array = np.array(split_df)
#     train_mask = torch.tensor(split_array == 'train', device=device)
#     test_mask = torch.tensor(split_array == 'test', device=device)

#     patches_per_image = (model_input_size[0] // patch_size) ** 2

#     # Group thresholds by cluster for efficiency
#     cluster_to_concepts = defaultdict(list)
#     for concept, cluster in curr_thresholds:
#         cluster_to_concepts[cluster].append(concept)

#     activated_images_train = defaultdict(set)
#     activated_images_test = defaultdict(set)

#     for cluster_label, concept_list in tqdm(cluster_to_concepts.items()):
#         train_indices_per_concept, test_indices_per_concept = find_activated_images_bypatch_per_cluster(cluster_label, concept_list,
#                                                                                                         curr_thresholds,
#                                                                                                         patches_per_image, 
#                                                                                                         dataset_name, train_mask,
#                                                                                                         test_mask, cossim_file,
#                                                                                                         scratch_dir, device)

#         for concept in concept_list:
#             activated_images_train[(concept, cluster_label)].update(train_indices_per_concept[concept])
#             activated_images_test[(concept, cluster_label)].update(test_indices_per_concept[concept])

#         torch.cuda.empty_cache()

#     return activated_images_train, activated_images_test, activated_images_cal

# def load_sims_to_gpu(filepath, cluster_labels, device):
#     sims = pd.read_csv(filepath, usecols=cluster_labels, engine='pyarrow')
#     sims_tensor = torch.tensor(sims.values, dtype=torch.float32, device=device)  # [n_patches, n_clusters]
#     return sims_tensor

# def load_sims_to_gpu(filepath, cluster_labels, device, chunk_size=100000):
#     chunk_iter = pd.read_csv(filepath, usecols=cluster_labels, chunksize=chunk_size)
#     all_chunks = []

#     for chunk in tqdm(chunk_iter):
#         tensor_chunk = torch.tensor(chunk.values, dtype=torch.float32, device=device)
#         all_chunks.append(tensor_chunk)

#     return torch.cat(all_chunks, dim=0)

# def find_activated_images_bypatch_per_cluster_batch(cluster_labels, cluster_to_concepts, curr_thresholds, 
#                                                      patches_per_image, dataset_name,
#                                                      train_mask, test_mask, cossim_file, scratch_dir, device):
#     """
#     Process a batch of cluster labels at once from a single multi-cluster cosine similarity CSV.
#     """
#     #filepath = f'{scratch_dir}/Cosine_Similarities/{dataset_name}/kmeans/{cossim_file}'
#     filepath = f'Cosine_Similarities/{dataset_name}/{cossim_file}'
#     sims_tensor = load_sims_to_gpu(filepath, cluster_labels, device)

#     num_images = sims_tensor.shape[0] // patches_per_image
#     reshaped = sims_tensor.view(num_images, patches_per_image, len(cluster_labels))  # [n_images, n_patches, n_clusters]
#     max_activations = torch.max(reshaped, dim=1)[0].T  # [n_clusters, n_images]

#     train_indices_per_concept = defaultdict(list)
#     test_indices_per_concept = defaultdict(list)
    
#     for idx, cluster_label in enumerate(cluster_labels):
#         max_act = max_activations[idx].view(1, -1)  # [1, n_images]
#         concepts = cluster_to_concepts[cluster_label]
#         thresholds = torch.tensor(
#             [curr_thresholds[(concept, cluster_label)][0] for concept in concepts], device=device
#         ).view(-1, 1)  # [n_concepts, 1]

#         activated = (max_act >= thresholds)  # [n_concepts, n_images]
#         train_acts = activated[:, :len(train_mask)] & train_mask.view(1, -1)
#         test_acts = activated[:, :len(test_mask)] & test_mask.view(1, -1)

#         for i, concept in enumerate(concepts):
#             train_indices_per_concept[(concept, cluster_label)] = torch.where(train_acts[i])[0].tolist()
#             test_indices_per_concept[(concept, cluster_label)] = torch.where(test_acts[i])[0].tolist()

#     return train_indices_per_concept, test_indices_per_concept


# def find_activated_images_bypatch_allpairs(curr_thresholds, model_input_size, dataset_name,
#                                           cossim_file, scratch_dir, patch_size=14, device='cuda', cluster_batch_size=10):
#     split_df = get_split_df(dataset_name)
#     split_array = np.array(split_df)
#     train_mask = torch.tensor(split_array == 'train', device=device)
#     test_mask = torch.tensor(split_array == 'test', device=device)

#     patches_per_image = (model_input_size[0] // patch_size) ** 2

#     # Group thresholds by cluster
#     cluster_to_concepts = defaultdict(list)
#     for concept, cluster in curr_thresholds:
#         cluster_to_concepts[cluster].append(concept)

#     all_cluster_labels = list(cluster_to_concepts.keys())
#     activated_images_train = defaultdict(set)
#     activated_images_test = defaultdict(set)

#     for i in tqdm(range(0, len(all_cluster_labels), cluster_batch_size), desc="Cluster batches"):
#         cluster_batch = all_cluster_labels[i:i + cluster_batch_size]
#         train_indices, test_indices = find_activated_images_bypatch_per_cluster_batch(
#             cluster_batch, cluster_to_concepts, curr_thresholds,
#             patches_per_image, dataset_name, train_mask, test_mask,
#             cossim_file, scratch_dir, device
#         )

#         for key in train_indices:
#             activated_images_train[key].update(train_indices[key])
#             activated_images_test[key].update(test_indices[key])

#         torch.cuda.empty_cache()

#     return activated_images_train, activated_images_test, activated_images_cal



# def find_activated_images_byimage_allpairs(cos_sims, curr_thresholds, dataset_name, concepts):
#     """Find activated images for each (GT concept, Cluster) pair using image-level activations."""
#     split_df = get_patch_split_df(dataset_name, model_input_size=None)

#     cluster_labels = list(cos_sims.columns)
#     thresholds = {c: curr_thresholds[c][0] for c in cluster_labels}

#     cos_sims_tensor = torch.tensor(cos_sims.values)
#     threshold_tensor = torch.tensor([thresholds[c] for c in cluster_labels])

#     activated_clusters = cos_sims_tensor >= threshold_tensor.unsqueeze(0)

#     split_array = np.array(split_df)
#     train_mask = torch.tensor(split_array == 'train')
#     test_mask = torch.tensor(split_array == 'test')

#     activated_images_train = defaultdict(set)
#     activated_images_test = defaultdict(set)

#     for cluster_idx, cluster_label in enumerate(cluster_labels):
#         for concept_name in concepts:
#             train_indices = torch.where(activated_clusters[:, cluster_idx] & train_mask)[0].tolist()
#             test_indices = torch.where(activated_clusters[:, cluster_idx] & test_mask)[0].tolist()
#             activated_images_train[(concept_name, cluster_label)].update(train_indices)
#             activated_images_test[(concept_name, cluster_label)].update(test_indices)

#     return activated_images_train, activated_images_test, activated_images_cal


# def get_cluster_max_activation(filepath, cluster_label, patches_per_image, num_images, device):
#     sims = pd.read_csv(filepath, usecols=[cluster_label])
#     sims_tensor = torch.tensor(sims[cluster_label].values, dtype=torch.float32, device=device)

#     num_images = len(sims_tensor) // patches_per_image
#     reshaped = sims_tensor.view(num_images, patches_per_image)
#     max_activations = torch.max(reshaped, dim=1)[0]  # [num_images]
#     return max_activations


# def preload_max_activations_per_cluster(cluster_labels, dataset_name, cossim_file, scratch_dir, model_input_size, patch_size, device, num_images):
#     patches_per_image = (model_input_size[0] // patch_size) ** 2
#     cluster_max_activations = {}

#     for cluster_label in tqdm(cluster_labels, desc="Loading max activations"):
#         filepath = f'{scratch_dir}/Cosine_Similarities/{dataset_name}/kmeans/{cluster_label}_{cossim_file}'
#         max_activations = get_cluster_max_activation(filepath, cluster_label, patches_per_image, num_images, device)
#         cluster_max_activations[cluster_label] = max_activations

#     return cluster_max_activations

# def find_activated_images_from_max_activations(thresholds, cluster_max_activations, train_mask, test_mask):
#     """
#     Determines activated images for each (concept, cluster) pair using precomputed max activations.

#     Args:
#         thresholds (dict): {(concept, cluster): (threshold_value, _)}
#         cluster_max_activations (dict): {cluster_label: [num_images] tensor of max activations}
#         train_mask (torch.BoolTensor): [num_images] indicating training split
#         test_mask (torch.BoolTensor): [num_images] indicating test split

#     Returns:
#         activated_images_train, activated_images_test: {(concept, cluster): set(image indices)}
#     """
#     activated_images_train = defaultdict(set)
#     activated_images_test = defaultdict(set)

#     for (concept, cluster), (threshold, _) in thresholds.items():
#         if cluster not in cluster_max_activations:
#             continue  # Skip if max activations weren't loaded for this cluster

#         max_acts = cluster_max_activations[cluster]  # [num_images]
#         activated = max_acts >= threshold  # [num_images], bool

#         # Split by train/test
#         train_indices = torch.where(activated & train_mask)[0].tolist()
#         test_indices = torch.where(activated & test_mask)[0].tolist()

#         activated_images_train[(concept, cluster)].update(train_indices)
#         activated_images_test[(concept, cluster)].update(test_indices)

#     return activated_images_train, activated_images_test, activated_images_cal


# def compute_detection_metrics_over_percentiles_allpairs(percentiles, gt_samples_per_concept_test, 
#                                                         gt_images_per_concept_test, 
#                                                dataset_name, model_input_size, device, 
#                                                con_label, cossim_file, scratch_dir, sample_type='patch',
#                                                     cluster_batch_size=10, patch_size=14, n_clusters=1000):
#     """
#     Computes detection metrics over multiple percentiles.

#     Args:
#         percentiles: List of percentiles
#         gt_samples_per_concept_test: {concept: patch indices}
#         gt_images_per_concept_test: {concept: image indices}
#         sim_metrics: Cosine similarities
#         dataset_name: Dataset name
#         model_input_size: (width, height) tuple
#         device: CUDA/CPU device
#         con_label: Label for saving
#         sample_type: 'patch' or 'cls'
#         patch_size: Patch size
#     Returns:
#         all_metrics: dict mapping per -> metrics_df
#     """
#     split_df = get_patch_split_df(dataset_name, model_input_size=model_input_size)
#     train_mask = torch.tensor(split_df.values == 'train', device=device)
#     test_mask = torch.tensor(split_df.values == 'test', device=device)
#     relevant_indices = set(filter_patches_by_image_presence(split_df.index, dataset_name, model_input_size).tolist())
    
#     compute_concept_thresholds_over_percentiles_all_pairs(gt_samples_per_concept_test,
#                                                                        percentiles, device, dataset_name, con_label,
#                                                                        cossim_file, scratch_dir, n_vectors=1, 
#                                                                        n_concepts_to_print=0)
#     thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt')
    
#     # cluster_labels = set([cluster for _, cluster in thresholds[percentiles[0]].keys()])
#     # cluster_max_activations = preload_max_activations_per_cluster(
#     #     cluster_labels, dataset_name, cossim_file, scratch_dir, model_input_size, patch_size, device, split_df.shape[0]
#     # )

#     for per in tqdm(percentiles, desc="Computing metrics for each percentile"):
#         save_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{per}_{con_label}.csv'
#         # if os.path.exists(save_path):
#         #     print(f"Skipping per {per}")
#         #     continue
#         # else:
#         # === Thresholds for current percentile
#         curr_thresholds = thresholds[per]

#         # === Activation
#         if sample_type == 'patch':
#             _, activated_images_test = find_activated_images_bypatch_allpairs(curr_thresholds, 
#                                                                               model_input_size, dataset_name,
#                                                                               cossim_file, scratch_dir, 
#                                                                               patch_size=14, device='cuda', 
#                                                                               cluster_batch_size=cluster_batch_size)
#         #     _, activated_images_test = find_activated_images_from_max_activations(
#         #                                     curr_thresholds, cluster_max_activations, train_mask, test_mask
#         #                                 )
#         # # elif sample_type == 'cls':
#         #     _, activated_images_test = find_activated_images_byimage_allpairs(
#         #         sim_metrics, curr_thresholds, model_input_size, dataset_name
#         #     )
#         else:
#             raise ValueError(f"Unknown sample_type: {sample_type}")

#         # === Compute TP, FP, TN, FN for each (concept, cluster) pair ===
#         fp_count, tp_count, tn_count, fn_count = {}, {}, {}, {}

#         for concept in gt_images_per_concept_test.keys():
#             gt_images = set(gt_images_per_concept_test[concept]) & relevant_indices

#             for cluster in range(n_clusters):
#                 cluster = str(cluster)
#                 activated_images = activated_images_test.get((concept, cluster), set()) & relevant_indices

#                 tp = len(gt_images & activated_images)
#                 fp = len(activated_images - gt_images)
#                 fn = len(gt_images - activated_images)
#                 tn = len(relevant_indices) - (tp + fp + fn)

#                 key = (concept, cluster)
#                 tp_count[key] = tp
#                 fp_count[key] = fp
#                 fn_count[key] = fn
#                 tn_count[key] = tn

#         metrics = compute_stats_from_counts(tp_count, fp_count, tn_count, fn_count)

#         # Save
#         metrics.to_csv(save_path, index=False)


def compute_concept_thresholds_over_percentiles_all_pairs(loader, gt_samples_per_concept_cal, percentiles, device,
                                            dataset_name, con_label):
    """
    Computes activation thresholds for every (concept, cluster) pair using chunked activation data.
    Only loads the calibration samples needed for threshold computation.

    Args:
        loader (ChunkedActivationLoader): Loader for chunked activation files
        gt_samples_per_concept_cal (dict): Mapping from concept name to list of cal sample indices.
        percentiles (list): List of percentile thresholds to compute (e.g., [0.9, 0.95]).
        device (str): CUDA or CPU device.
        dataset_name (str): Dataset name (for saving cache).
        con_label (str): Label to include in cache filename.

    Returns:
        dict: Mapping percentile -> {(concept, cluster): (threshold, nan)}
    """
    from utils.memory_management_utils import ChunkedActivationLoader
    
    cache_dir = f'Thresholds/{dataset_name}'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'all_percentiles_allpairs_{con_label}.pt')

    all_thresholds = {}
    new_percentiles = set(percentiles)
    print(f"Will compute for all percentiles: {new_percentiles}")

    if not new_percentiles:
        return {p: all_thresholds[p] for p in percentiles}

    for p in new_percentiles:
        all_thresholds[p] = {}

    # Get all unique calibration indices we need
    all_cal_indices = set()
    for concept, indices in gt_samples_per_concept_cal.items():
        all_cal_indices.update(indices)
    all_cal_indices = sorted(list(all_cal_indices))
    
    if len(all_cal_indices) == 0:
        return {p: all_thresholds[p] for p in percentiles}
    
    print(f"   Loading activations for {len(all_cal_indices)} calibration samples...")
    
    # Get cluster labels from loader
    loader_info = loader.get_activation_info() if hasattr(loader, 'get_activation_info') else loader.get_info()
    cluster_labels = loader_info['concept_names']
    
    # Load calibration data in chunks to avoid memory issues
    chunk_size = 100000  # Process 100k samples at a time
    num_chunks = (len(all_cal_indices) + chunk_size - 1) // chunk_size
    
    for chunk_start in tqdm(range(0, len(all_cal_indices), chunk_size), desc=f"Processing {len(all_cal_indices):,} calibration samples", leave=False):
        chunk_end = min(chunk_start + chunk_size, len(all_cal_indices))
        chunk_indices = all_cal_indices[chunk_start:chunk_end]
        
        # Load this chunk efficiently 
        min_idx = min(chunk_indices)
        max_idx = max(chunk_indices)
        range_size = max_idx - min_idx + 1
        
        # Always load the full range - sparse loading is unnecessarily slow
        # Load the range for this chunk
        chunk_range = loader.load_chunk_range(min_idx, max_idx + 1)
        local_positions = [idx - min_idx for idx in chunk_indices]
        chunk_acts = chunk_range[local_positions].to(device)
        del chunk_range
        
        # Create mapping from global index to position in chunk_acts
        chunk_idx_to_pos = {idx: pos for pos, idx in enumerate(chunk_indices)}
        
        # Process all concepts for this chunk
        for concept, indices in gt_samples_per_concept_cal.items():
            if len(indices) == 0:
                continue
                
            # Find which of this concept's indices are in the current chunk
            concept_positions = []
            for idx in indices:
                if idx in chunk_idx_to_pos:
                    concept_positions.append(chunk_idx_to_pos[idx])
            
            if len(concept_positions) == 0:
                continue
                
            # Get activations for this concept in this chunk
            concept_chunk_acts = chunk_acts[concept_positions]  # Shape: [n_concept_samples, n_clusters]
            
            # Compute percentiles for each cluster
            for cluster_idx, cluster_label in enumerate(cluster_labels):
                cluster_sims = concept_chunk_acts[:, cluster_idx]
                
                if len(cluster_sims) > 0:
                    # For SAE, only compute percentiles on positive activations
                    if 'sae' in con_label.lower():
                        positive_sims = cluster_sims[cluster_sims > 0]
                        if len(positive_sims) > 0:
                            percentiles_tensor = torch.tensor([1 - p for p in new_percentiles], device=device)
                            thresholds = torch.quantile(positive_sims, percentiles_tensor, interpolation='linear')
                        else:
                            # No positive activations - use zeros
                            thresholds = torch.zeros(len(new_percentiles), device=device)
                    else:
                        # Regular computation for non-SAE
                        percentiles_tensor = torch.tensor([1 - p for p in new_percentiles], device=device)
                        thresholds = torch.quantile(cluster_sims, percentiles_tensor, interpolation='linear')
                    
                    # Accumulate results (we'll combine across chunks later)
                    for p_idx, p in enumerate(new_percentiles):
                        # For SAE, remove 'sae_unit_' prefix to match detection metrics expectations
                        if 'sae_unit_' in cluster_label and cluster_label.startswith('sae_unit_'):
                            clean_label = cluster_label.replace('sae_unit_', '')
                        else:
                            clean_label = cluster_label
                        key = (concept, clean_label)
                        if key not in all_thresholds.get(p, {}):
                            if p not in all_thresholds:
                                all_thresholds[p] = {}
                            all_thresholds[p][key] = []
                        all_thresholds[p][key].append(thresholds[p_idx].item())
        
        # Clean up chunk data
        del chunk_acts
        torch.cuda.empty_cache() if device == 'cuda' else None
    
    # Combine thresholds across chunks (take median of all chunk thresholds)
    for p in all_thresholds:
        for key in all_thresholds[p]:
            if isinstance(all_thresholds[p][key], list):
                # Convert list of thresholds to single value (median)
                threshold_list = all_thresholds[p][key]
                combined_threshold = float(torch.tensor(threshold_list).median().item())
                all_thresholds[p][key] = (combined_threshold, float('nan'))

    # Save computed thresholds
    torch.save(all_thresholds, cache_file)
    print(f"Saved thresholds to {cache_file}")
    
    # Final memory cleanup
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {p: all_thresholds[p] for p in percentiles}


# REMOVED DUPLICATE FUNCTION: compute_concept_thresholds_over_percentiles_all_pairs
# This version (taking cos_sims_df) was removed in favor of the memory-efficient version
# that uses ChunkedActivationLoader (defined at line 1084)

        

def find_activated_images_bypatch_allpairs(curr_thresholds, loader, model_input_size, dataset_name, patch_size=14, device='cuda'):
    """
    Computes per-image activations for all (concept, cluster) pairs using chunked data.
    """
    from utils.memory_management_utils import ChunkedActivationLoader
    
    split_df = get_split_df(dataset_name)
    patches_per_image = (model_input_size[0] // patch_size) ** 2
    
    info = loader.get_activation_info()
    total_patches = info['total_samples']
    num_images = total_patches // patches_per_image
    
    # Initialize result dictionaries
    activated_images_train = defaultdict(set)
    activated_images_test = defaultdict(set)
    activated_images_cal = defaultdict(set)
    
    # Process images in chunks
    images_per_chunk = 500  # Process 500 images at a time for all-pairs
    
    for img_start_idx in range(0, num_images, images_per_chunk):
        img_end_idx = min(img_start_idx + images_per_chunk, num_images)
        
        # Calculate patch range for these images
        patch_start_idx = img_start_idx * patches_per_image
        patch_end_idx = img_end_idx * patches_per_image
        
        # Load activation chunk (already a tensor)
        chunk_tensor = loader.load_chunk_range(patch_start_idx, patch_end_idx)
        num_images_in_chunk = img_end_idx - img_start_idx
        
        # Move to device if needed
        if device == 'cuda' and not chunk_tensor.is_cuda:
            chunk_tensor = chunk_tensor.cuda()
        
        # Reshape: [num_images_in_chunk, patches_per_image, num_clusters]
        reshaped_sims = chunk_tensor.reshape(num_images_in_chunk, patches_per_image, -1)
        
        # Max over patches → [num_images_in_chunk, num_clusters]
        max_activations = torch.max(reshaped_sims, dim=1)[0]
        
        # Get cluster labels from loader
        loader_info = loader.get_activation_info() if hasattr(loader, 'get_activation_info') else loader.get_info()
        cluster_labels = loader_info['concept_names']
        
        # Process each (concept, cluster) pair
        for (concept, cluster), (threshold, _) in curr_thresholds.items():
            cluster_idx = cluster_labels.index(str(cluster)) if str(cluster) in cluster_labels else -1
            if cluster_idx == -1:
                continue
            
            # Find activated images for this threshold
            activated = max_activations[:, cluster_idx] >= threshold
            
            # Process each image in chunk
            for local_img_idx in range(num_images_in_chunk):
                if activated[local_img_idx]:
                    global_img_idx = img_start_idx + local_img_idx
                    
                    if global_img_idx < len(split_df):
                        split = split_df.iloc[global_img_idx]
                        
                        if split == 'train':
                            activated_images_train[(concept, cluster)].add(global_img_idx)
                        elif split == 'test':
                            activated_images_test[(concept, cluster)].add(global_img_idx)
                        elif split == 'cal':
                            activated_images_cal[(concept, cluster)].add(global_img_idx)
        
        # Clear memory
        del chunk_tensor, reshaped_sims, max_activations
        gc.collect()
    
    return activated_images_train, activated_images_test, activated_images_cal




# REMOVED DUPLICATE FUNCTION: find_activated_sentences_bytoken_allpairs
# This wrapper version was removed. The full implementation is kept at line 1476


# DUPLICATE FUNCTION - Commented out to avoid conflict
# def find_activated_images_bypatch_allpairs(curr_thresholds, loader, model_input_size, dataset_name, patch_size=14):
#     """
#     Find activated images for each (concept, cluster) pair over patches.
#     Memory-efficient version that processes one split at a time.
#     
#     Args:
#         curr_thresholds: Dictionary of (concept, cluster) -> (threshold, _) pairs
#         loader: ChunkedActivationLoader instance
#         model_input_size: Model input size tuple
#         dataset_name: Dataset name
#         patch_size: Patch size (default 14)
#     
#     Returns:
#         Tuple of (activated_images_train, activated_images_test, activated_images_cal)
#     """
#     from utils.memory_management_utils import ChunkedActivationLoader
#     
#     split_df = get_split_df(dataset_name)
#     patches_per_image = (model_input_size[0] // patch_size) ** 2
#     
#     activated_images_train = defaultdict(set)
#     activated_images_test = defaultdict(set)
#     activated_images_cal = defaultdict(set)
#     
#     cluster_labels = loader.columns
#     
#     # Process each split separately to save memory
#     for split_name, activated_dict in [('train', activated_images_train), 
#                                        ('test', activated_images_test), 
#                                        ('cal', activated_images_cal)]:
#         
#         # Load only the data for this split
#         split_tensor = loader.load_split_tensor(split_name, dataset_name, model_input_size, patch_size)
#         
#         # Get the indices for this split
#         split_indices = split_df[split_df == split_name].index.tolist()
#         num_split_images = len(split_indices)
#         
#         # Reshape to image-level max activations
#         reshaped_sims = split_tensor.reshape(num_split_images, patches_per_image, -1)
#         max_activations = torch.max(reshaped_sims, dim=1)[0]  # [num_split_images, n_clusters]
#         
#         # Process each concept-cluster pair
#         for (concept, cluster_label) in curr_thresholds:
#             cluster_idx = cluster_labels.index(cluster_label)
#             threshold_val = curr_thresholds[(concept, cluster_label)][0]
#             
#             # Find activated images in this split
#             cluster_activations = max_activations[:, cluster_idx] >= threshold_val
#             activated_indices = torch.where(cluster_activations)[0].tolist()
#             
#             # Map back to global image indices
#             global_indices = [split_indices[i] for i in activated_indices]
#             activated_dict[(concept, cluster_label)].update(global_indices)
#         
#         # Clean up memory for this split
#         del split_tensor, reshaped_sims, max_activations
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#     
#     return activated_images_train, activated_images_test, activated_images_cal


def find_activated_images_byimage_allpairs(loader, curr_thresholds, dataset_name):
    """
    Find activated images for each (concept, cluster) pair using image-level activations.
    Memory-efficient version that processes one split at a time.

    Args:
        loader: ChunkedActivationLoader instance
        curr_thresholds (dict): {(concept, cluster): (threshold, _)}
        dataset_name (str): For retrieving train/test split

    Returns:
        Tuple of dicts: 
            activated_images_train[(concept, cluster)] = set(image indices)
            activated_images_test[(concept, cluster)] = set(image indices)
            activated_images_cal[(concept, cluster)] = set(image indices)
    """
    from utils.memory_management_utils import ChunkedActivationLoader
    
    split_df = get_split_df(dataset_name)
    cluster_labels = loader.columns
    
    # Map cluster label → column index
    cluster_to_index = {label: idx for idx, label in enumerate(cluster_labels)}
    
    activated_images_train = defaultdict(set)
    activated_images_test = defaultdict(set)
    activated_images_cal = defaultdict(set)
    
    # Process each split separately to save memory
    for split_name, activated_dict in [('train', activated_images_train), 
                                       ('test', activated_images_test), 
                                       ('cal', activated_images_cal)]:
        
        # Get indices for this split
        split_indices = split_df[split_df == split_name].index.tolist()
        if not split_indices:
            continue
            
        # Load only the data for this split
        split_tensor = loader.load_split_tensor(split_name, dataset_name, ('cls', 'cls'))
        
        # Process each concept-cluster pair
        for (concept, cluster), (threshold, _) in curr_thresholds.items():
            if cluster not in cluster_to_index:
                continue
                
            cluster_idx = cluster_to_index[cluster]
            sims = split_tensor[:, cluster_idx]  # [num_split_images]
            
            # Find activated images in this split
            activated = sims >= threshold
            activated_indices = torch.where(activated)[0].tolist()
            
            # Map back to global image indices
            global_indices = [split_indices[i] for i in activated_indices]
            activated_dict[(concept, cluster)].update(global_indices)
        
        # Clean up memory for this split
        del split_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return activated_images_train, activated_images_test, activated_images_cal


def find_activated_sentences_bytoken_allpairs(act_metrics, curr_thresholds, model_input_size, dataset_name):
    """
    Find activated sentences for each (concept, cluster) pair over tokens.
    """
    split_df = get_split_df(dataset_name)
    token_counts_per_sentence = torch.load(f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt', weights_only=False)  # List[List[int]]
    
    # Compute sentence start and end indices in the flat activation tensor
    token_counts_flat = torch.tensor([sum(x) for x in token_counts_per_sentence])
    sentence_starts = torch.cat([torch.tensor([0]), token_counts_flat.cumsum(0)[:-1]])
    sentence_ends = token_counts_flat.cumsum(0)
    
    # Flatten activation tensor and get cluster keys
    metrics_tensor = torch.tensor(act_metrics.values)
    all_clusters = list(act_metrics.columns)  # Should be (concept, cluster) string keys
    concept_cluster_keys = list(curr_thresholds.keys())

    # Build mapping from (concept, cluster) to column index
    col_index_map = {key: all_clusters.index(key[1]) for key in concept_cluster_keys}

    # Pre-compute max activation per sentence for each cluster
    max_activations = torch.stack([
        metrics_tensor[start:end].amax(dim=0) for start, end in zip(sentence_starts, sentence_ends)
    ])  # shape: [num_sentences, num_clusters]

    # Convert split to boolean mask
    split_array = np.array(split_df)
    train_mask = torch.tensor(split_array == 'train')
    test_mask = torch.tensor(split_array == 'test')
    cal_mask = torch.tensor(split_array == 'cal')

    activated_sentences_train = defaultdict(set)
    activated_sentences_test = defaultdict(set)
    activated_sentences_cal = defaultdict(set)

    for (concept, cluster_label), threshold in curr_thresholds.items():
        cluster_idx = col_index_map[(concept, cluster_label)]
        threshold_val = threshold[0]

        cluster_activations = max_activations[:, cluster_idx] >= threshold_val

        train_indices = torch.where(cluster_activations & train_mask)[0].tolist()
        test_indices = torch.where(cluster_activations & test_mask)[0].tolist()
        cal_indices = torch.where(cluster_activations & cal_mask)[0].tolist()

        activated_sentences_train[(concept, cluster_label)].update(train_indices)
        activated_sentences_test[(concept, cluster_label)].update(test_indices)
        activated_sentences_cal[(concept, cluster_label)].update(cal_indices)

    return activated_sentences_train, activated_sentences_test, activated_sentences_cal



def compute_detection_metrics_over_percentiles_allpairs(percentiles, gt_images_per_concept_split, 
                                               dataset_name, model_input_size, device, 
                                               con_label, loader, scratch_dir, sample_type='patch',
                                                    cluster_batch_size=10, patch_size=14, n_clusters=1000):
    """
    Loads activations only once per split and processes all percentiles.
    
    Args:
        percentiles: List of percentiles
        gt_images_per_concept_split: {concept: image indices} for the split being evaluated
        dataset_name: Dataset name
        model_input_size: (width, height) tuple
        device: CUDA/CPU device
        con_label: Label for saving
        loader: ChunkedActivationLoader
        scratch_dir: Scratch directory
        sample_type: 'patch' or 'cls'
        cluster_batch_size: Batch size for processing clusters
        patch_size: Patch size
        n_clusters: Number of clusters
    Returns:
        all_metrics: dict mapping per -> metrics_df
    """
    from collections import defaultdict
    import gc
    
    # Ensure we're using GPU if available
    device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
    # print(f"Using device: {device}")
    
    # Get actual number of clusters from the loader
    info = loader.get_activation_info()
    n_clusters = info['num_concepts']
    
    # Load thresholds for all percentiles
    threshold_label = con_label.replace("_cal", "") if con_label.endswith("_cal") else con_label
    thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{threshold_label}.pt', weights_only=False)
    
    # Get split info
    split_df = get_split_df(dataset_name)
    if con_label.endswith("_cal"):
        eval_split = 'cal'
    else:
        eval_split = 'test'
    
    # Create masks for different splits on GPU
    split_array = np.array(split_df)
    eval_mask = torch.tensor(split_array == eval_split, device=device)
    eval_indices_set = set(torch.where(eval_mask)[0].cpu().numpy())
    
    # Precompute max activations for ALL images at once
    # print(f"Precomputing max activations for {eval_split} split on {device}...")
    
    if sample_type == 'patch':
        # Check if this is a text dataset
        if isinstance(model_input_size[0], str) and model_input_size[0] == 'text':
            # For text datasets, load token counts to determine tokens per paragraph
            token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
            
            if not os.path.exists(token_counts_file):
                raise FileNotFoundError(f"Token counts file not found: {token_counts_file}")
            
            token_counts = torch.load(token_counts_file, weights_only=False)
            
            # Convert to tokens per sentence/paragraph
            tokens_per_paragraph = [sum(sent_tokens) if isinstance(sent_tokens, list) else sent_tokens 
                                  for sent_tokens in token_counts]
            
            info = loader.get_activation_info()
            total_tokens = info['total_samples']
            num_paragraphs = len(tokens_per_paragraph)
            
            # For text, we'll use variable-size handling below
            patches_per_image = None  # Variable for text
            num_images = num_paragraphs
        else:
            # For image datasets, use fixed patch grid
            patches_per_image = (model_input_size[0] // patch_size) ** 2
            info = loader.get_activation_info()
            total_patches = info['total_samples']
            num_images = total_patches // patches_per_image
        
        # Allocate on GPU if possible
        if device == 'cuda':
            # Check available GPU memory
            torch.cuda.empty_cache()
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            required_memory = num_images * n_clusters * 4  # float32
            
            if required_memory < available_memory * 0.8:  # Use 80% of available memory
                # print(f"Allocating {required_memory / 1e9:.2f}GB on GPU")
                max_activations_all = torch.zeros(num_images, n_clusters, dtype=torch.float32, device=device)
                keep_on_gpu = True
            else:
                # print(f"Not enough GPU memory ({required_memory / 1e9:.2f}GB required), using CPU for storage")
                max_activations_all = torch.zeros(num_images, n_clusters, dtype=torch.float32)
                keep_on_gpu = False
        else:
            max_activations_all = torch.zeros(num_images, n_clusters, dtype=torch.float32)
            keep_on_gpu = False
        
        # Process in chunks
        if patches_per_image is not None:
            # Fixed patch size (image datasets)
            images_per_chunk = 500 if device == 'cuda' else 1000
            for img_start_idx in tqdm(range(0, num_images, images_per_chunk), desc="Loading activations"):
                img_end_idx = min(img_start_idx + images_per_chunk, num_images)
                
                # Calculate patch range
                patch_start_idx = img_start_idx * patches_per_image
                patch_end_idx = img_end_idx * patches_per_image
                
                # Load chunk
                chunk_tensor = loader.load_chunk_range(patch_start_idx, patch_end_idx)
                num_images_in_chunk = img_end_idx - img_start_idx
                
                # Move to GPU if needed
                if device == 'cuda' and not chunk_tensor.is_cuda:
                    chunk_tensor = chunk_tensor.cuda()
                
                # Reshape and compute max on GPU
                reshaped = chunk_tensor.reshape(num_images_in_chunk, patches_per_image, -1)
                max_acts = torch.max(reshaped, dim=1)[0]  # [num_images_in_chunk, n_clusters]
                
                # Store (keep on GPU if possible)
                if keep_on_gpu:
                    max_activations_all[img_start_idx:img_end_idx] = max_acts
                else:
                    max_activations_all[img_start_idx:img_end_idx] = max_acts.cpu()
                
                # Clean up
                del chunk_tensor, reshaped, max_acts
                if device == 'cuda':
                    torch.cuda.empty_cache()
        else:
            # Variable token size (text datasets)
            paragraphs_per_chunk = 100 if device == 'cuda' else 200
            
            # Calculate cumulative token indices
            cumulative_tokens = [0]
            for tokens in tokens_per_paragraph:
                cumulative_tokens.append(cumulative_tokens[-1] + tokens)
            
            for para_start_idx in tqdm(range(0, num_paragraphs, paragraphs_per_chunk), desc="Loading activations"):
                para_end_idx = min(para_start_idx + paragraphs_per_chunk, num_paragraphs)
                
                # Calculate token range for this chunk of paragraphs
                token_start_idx = cumulative_tokens[para_start_idx]
                token_end_idx = cumulative_tokens[para_end_idx]
                
                # Load chunk
                chunk_tensor = loader.load_chunk_range(token_start_idx, token_end_idx)
                
                # Move to GPU if needed
                if device == 'cuda' and not chunk_tensor.is_cuda:
                    chunk_tensor = chunk_tensor.cuda()
                
                # Process each paragraph separately due to variable lengths
                chunk_start = 0
                for para_idx in range(para_start_idx, para_end_idx):
                    para_tokens = tokens_per_paragraph[para_idx]
                    para_acts = chunk_tensor[chunk_start:chunk_start + para_tokens]
                    
                    # Compute max activation for this paragraph
                    max_act = torch.max(para_acts, dim=0)[0]
                    
                    # Store results
                    if keep_on_gpu:
                        max_activations_all[para_idx] = max_act
                    else:
                        max_activations_all[para_idx] = max_act.cpu()
                    
                    chunk_start += para_tokens
                
                # Clean up
                del chunk_tensor
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
    else:  # cls
        # For CLS, activations are already at image level
        # print("Loading image-level activations...")
        max_activations_all = loader.load_full_tensor()
        if device == 'cuda' and max_activations_all.numel() * 4 < torch.cuda.get_device_properties(0).total_memory * 0.8:
            max_activations_all = max_activations_all.cuda()
            keep_on_gpu = True
        else:
            max_activations_all = max_activations_all.cpu()
            keep_on_gpu = False
    
    # print(f"Max activations shape: {max_activations_all.shape}, device: {max_activations_all.device}")
    
    # Get cluster labels
    loader_info = loader.get_activation_info() if hasattr(loader, 'get_activation_info') else loader.get_info()
    cluster_labels = loader_info['concept_names']
    
    # Precompute GT image sets as GPU tensors for faster intersection
    gt_image_tensors = {}
    num_images = max_activations_all.shape[0]
    
    for concept, image_indices in gt_images_per_concept_split.items():
        # Create a boolean mask for GT images
        gt_mask = torch.zeros(num_images, dtype=torch.bool, device=device if keep_on_gpu else 'cpu')
        if image_indices:
            valid_indices = [idx for idx in image_indices if idx < num_images]
            if valid_indices:
                gt_mask[valid_indices] = True
        gt_image_tensors[concept] = gt_mask
    
    # Now process each percentile using the precomputed max activations
    for per in tqdm(percentiles, desc="Computing metrics for each percentile"):
        save_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{per}_{con_label}.csv'
        
        curr_thresholds = thresholds[per]
        
        # Vectorized threshold processing
        fp_count, tp_count, tn_count, fn_count = {}, {}, {}, {}
        
        # Process in batches if on GPU
        if keep_on_gpu and device == 'cuda':
            # Process concepts in batches for better GPU utilization
            for concept in gt_images_per_concept_split.keys():
                gt_mask = gt_image_tensors[concept]
                
                # Process all clusters for this concept
                for cluster in range(n_clusters):
                    cluster_str = str(cluster)
                    key = (concept, cluster_str)
                    
                    if key in curr_thresholds:
                        threshold = curr_thresholds[key][0]
                        
                        # Vectorized activation check on GPU
                        cluster_idx = cluster_labels.index(cluster_str) if cluster_str in cluster_labels else -1
                        if cluster_idx == -1:
                            continue
                        
                        # GPU operations
                        activated_mask = max_activations_all[:, cluster_idx] >= threshold
                        activated_in_split = activated_mask & eval_mask
                        
                        # Compute metrics using GPU operations
                        tp = (activated_in_split & gt_mask).sum().item()
                        fp = (activated_in_split & ~gt_mask).sum().item()
                        fn = (~activated_in_split & gt_mask & eval_mask).sum().item()
                        tn = (~activated_in_split & ~gt_mask & eval_mask).sum().item()
                        
                        tp_count[key] = tp
                        fp_count[key] = fp
                        fn_count[key] = fn
                        tn_count[key] = tn
        else:
            # CPU path - use original logic but with some optimizations
            # Find activated images for this percentile
            activated_images_split = defaultdict(set)
            
            # Process each (concept, cluster) pair
            for (concept, cluster), (threshold, _) in curr_thresholds.items():
                cluster_idx = cluster_labels.index(str(cluster)) if str(cluster) in cluster_labels else -1
                if cluster_idx == -1:
                    continue
                    
                # Find which images are activated for this threshold
                activated_mask = max_activations_all[:, cluster_idx] >= threshold
                
                # Get indices of activated images
                activated_indices = torch.where(activated_mask)[0].tolist()
                
                # Filter by split
                for img_idx in activated_indices:
                    if img_idx in eval_indices_set:
                        activated_images_split[(concept, cluster)].add(img_idx)
            
            # Compute TP, FP, TN, FN
            for concept in gt_images_per_concept_split.keys():
                gt_images = set(gt_images_per_concept_split[concept])
                
                for cluster in range(n_clusters):
                    cluster = str(cluster)
                    activated_images = activated_images_split.get((concept, cluster), set())
                    
                    tp = len(gt_images & activated_images)
                    fp = len(activated_images - gt_images)
                    fn = len(gt_images - activated_images)
                    tn = len(eval_indices_set) - (tp + fp + fn)
                    
                    key = (concept, cluster)
                    tp_count[key] = tp
                    fp_count[key] = fp
                    fn_count[key] = fn
                    tn_count[key] = tn
        
        metrics = compute_stats_from_counts(tp_count, fp_count, tn_count, fn_count)
        metrics.to_csv(save_path, index=False)
    
    # Clean up
    del max_activations_all
    if gt_image_tensors:
        del gt_image_tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_best_clusters_per_concept_from_detectionmetrics(dataset_name, 
                                                          model_name,
                                                          sample_type, 
                                                          metric_type, 
                                                          percentiles,
                                                          con_label):
    """
    For each semantic concept, finds the best cluster + percentile that maximizes the given metric.
    
    Args:
        dataset_name: Name of dataset.
        model_name: Name of model (CLIP, Llama, etc.).
        sample_type: 'patch' or 'cls'.
        metric_type: Metric to maximize ('f1', 'tpr', 'fpr').
        percentiles: List of percentiles to search over.
        
    Returns:
        best_cluster_per_concept: dict mapping concept -> (best_cluster, best_score, best_percentile)
    """

    # Where detection metrics are saved
    base_dir = f'Quant_Results/{dataset_name}'

    best_cluster_per_concept = {}

    for per in tqdm(percentiles, desc="Searching over percentiles"):
        # Load detection metrics at this percentile
        detectionmetrics_path = f"{base_dir}/detectionmetrics_allpairs_per_{per}_{con_label}.csv"
        
        try:
            metrics_df = pd.read_csv(detectionmetrics_path)
        except:
            print(f"Warning: Missing {detectionmetrics_path}, skipping.")
            continue

        # Filter by concepts you care about (optional, if needed)

        for idx, row in metrics_df.iterrows():
            concept, cluster = ast.literal_eval(row['concept']) 
            cluster = str(cluster).strip("'\"") 
            score = row[metric_type]

            # Only update if score is better
            if concept not in best_cluster_per_concept or score > best_cluster_per_concept[concept]['best_score']:
                best_cluster_per_concept[concept] = {
                    'best_cluster': cluster,
                    'best_score': score,
                    'best_percentile': per
                }
                
    torch.save(best_cluster_per_concept, f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt')
    # print(f"Best matches saved to Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt :)")
    return best_cluster_per_concept


def plot_weighted_detection_curve_for_best_clusters(best_cluster_per_concept, 
                                                     dataset_name, model_name, sample_type, 
                                                     metric_type, percentiles, con_label):
    """
    Plots weighted average detection metric curve across percentiles for best cluster per concept.

    Args:
        best_cluster_per_concept: dict mapping concept -> dict(best_cluster, best_score, best_percentile)
        dataset_name: Dataset name
        model_name: Model name (e.g., CLIP)
        sample_type: 'patch' or 'cls'
        metric_type: 'f1', 'tpr', 'fpr'
        percentiles: List of detection percentiles
        split: 'test' or 'train'
    """
    # === Load GT samples per concept ===
    if model_name == 'CLIP':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(224, 224).pt', weights_only=False)
    elif model_name == 'Llama':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(560, 560).pt', weights_only=False)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    base_dir = f'Quant_Results/{dataset_name}'
    weighted_scores = []

    for per in percentiles:
        try:
            metrics_df = pd.read_csv(f"{base_dir}/detectionmetrics_allpairs_per_{per}_{con_label}.csv")
        except:
            print(f"Warning: Missing detection metrics for percentile {per}")
            weighted_scores.append(np.nan)
            continue
        weighted_sum = 0
        total_samples = 0

        for concept, info in best_cluster_per_concept.items():
            cluster = info['best_cluster']
            sample_count = len(gt_samples_per_concept[concept])

            if sample_count == 0:
                continue  # Skip if no GT samples
                
            match = metrics_df[
                                metrics_df['concept'].apply(
                                    lambda x: isinstance(x, str) and
                                              x.strip("()").split(", ")[0].strip("'\"") == concept and
                                              x.strip("()").split(", ")[1].strip("'\"") == str(cluster)
                                )
                                ]

            if not match.empty:
                metric_val = match.iloc[0][metric_type]
                weighted_sum += metric_val * sample_count
                total_samples += sample_count

        if total_samples > 0:
            weighted_avg = weighted_sum / total_samples
        else:
            weighted_avg = np.nan

        weighted_scores.append(weighted_avg)

    # === Plot ===
    plt.figure(figsize=(10, 6))
    plt.plot(percentiles, weighted_scores, marker='o', color='blue', label='Weighted Average across Concepts')

    plt.xlabel("Chosen Detection Percentile", fontsize=14)
    plt.ylabel(f"{metric_type.upper()} Score", fontsize=14)
    plt.title(f"Weighted Average {metric_type.upper()} Detection Curve\n{model_name} on {dataset_name}", fontsize=16)
    plt.xticks(np.linspace(0, 1.0, 11), [f"{int(x*100)}%" for x in np.linspace(0, 1.0, 11)])
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_detection_curves_for_best_clusters(best_cluster_per_concept,
                                             dataset_name, model_name, sample_type, 
                                             metric_type, percentiles, con_label, concepts):
    """
    Plots detection metric across percentiles for the best cluster per semantic concept.

    Args:
        best_cluster_per_concept: dict mapping concept -> dict(best_cluster, best_score, best_percentile)
        dataset_name: Dataset name
        model_name: Model name (e.g., CLIP)
        sample_type: 'patch' or 'cls'
        metric_type: 'f1', 'tpr', 'fpr'
        percentiles: List of percentiles (thresholds evaluated)
        split: 'test' or 'train'
    """

    base_dir = f'Quant_Results/{dataset_name}'

    plt.figure(figsize=(14, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, len(best_cluster_per_concept)))

    for i, (concept, info) in enumerate(best_cluster_per_concept.items()):
        if concept not in concepts:
            continue
            
        cluster = info['best_cluster']
        scores = []

        for per in percentiles:
            try:
                metrics_df = pd.read_csv(f"{base_dir}/detectionmetrics_allpairs_per_{per}_{con_label}.csv")
            except:
                print(f"Warning: Missing percentile {per} for {concept}")
                scores.append(np.nan)
                continue

            # Find the (concept, cluster) pair
            match = metrics_df[
                                metrics_df['concept'].apply(
                                    lambda x: isinstance(x, str) and
                                              x.strip("()").split(", ")[0].strip("'\"") == concept and
                                              x.strip("()").split(", ")[1].strip("'\"") == str(cluster)
                                )
                                ]

            if not match.empty:
                scores.append(match.iloc[0][metric_type])
            else:
                scores.append(np.nan)

        label = f"Cluster {cluster} (Concept {concept})"
        plt.plot(percentiles, scores, marker='o', label=label, color=colors[i])

    plt.xlabel("Detection Percentile", fontsize=14)
    plt.ylabel(f"{metric_type.upper()} Score", fontsize=14)
    plt.title(f"{metric_type.upper()} Curve Across Percentiles (Best Cluster per Concept)\n{model_name} on {dataset_name}", fontsize=16)
    plt.xticks(np.linspace(0, 1.0, 11), [f"{int(x*100)}%" for x in np.linspace(0, 1.0, 11)])
    plt.ylim(0, 1.05)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title="Concepts", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    
        
### GETTING MOST CORRELATED CONCEPTS ####
def get_top_percent_patches_per_concept(concepts, cos_sims, per=0.2):
    """
    For each concept (cluster), selects the top n% of patches with highest cosine similarity.

    Args:
        concepts (dict): Mapping from cluster label (str) to centroid (torch.Tensor).
        cos_sims (pd.DataFrame): DataFrame of cosine similarities (rows = patches, columns = cluster labels).
        per (float): Fraction of top patches to select (e.g., 0.2 for top 20%).

    Returns:
        dict: Mapping from concept (cluster label) to list of selected patch indices (row indices from cos_sims).
    """
    top_patches_per_concept = {}

    for concept_label in tqdm(concepts.keys()):
        sims = cos_sims[concept_label]
        n_select = max(1, int(len(sims) * per))

        # Get indices of top n% patches (highest cosine similarity)
        top_indices = sims.nlargest(n_select).index.tolist()
        top_patches_per_concept[concept_label] = top_indices

    return top_patches_per_concept


def find_cluster_concept_correlations(gt_samples_per_concept, concepts, cos_sims, per=0.2):
    """
    Finds correlations between semantic concepts and clusters based on patch indices (using binary IoU).

    Args:
        gt_samples_per_concept (dict): Mapping from concept name to list of patch indices.
        concepts (dict): Mapping from cluster id to centroid.
        cos_sims (pd.DataFrame): Cosine similarity dataframe.
        per (float): Top % of patches to consider for each cluster.

    Returns:
        dict: Mapping from concept name to sorted list of (cluster_id, correlation_score) tuples.
    """
    concept_to_cluster_corrs = {}

    # Get top per patches per cluster
    cluster_to_samples = get_top_percent_patches_per_concept(concepts, cos_sims, per=per)

    # Convert lists to sets for fast intersection/union
    concept_sets = {concept: set(indices) for concept, indices in gt_samples_per_concept.items()}
    cluster_sets = {cluster: set(indices) for cluster, indices in cluster_to_samples.items()}

    for concept, concept_indices in concept_sets.items():
        cluster_corrs = []
        for cluster, cluster_indices in cluster_sets.items():
            intersection = concept_indices.intersection(cluster_indices)
            union = concept_indices.union(cluster_indices)

            # Binary IoU: size of intersection / size of union
            score = len(intersection) / len(union) if len(union) > 0 else 0.0
            cluster_corrs.append((cluster, score))
        
        # Sort by descending IoU
        cluster_corrs.sort(key=lambda x: x[1], reverse=True)
        concept_to_cluster_corrs[concept] = cluster_corrs

    return concept_to_cluster_corrs


def get_topn_aligning_clusters_per_concept(n, gt_samples_per_concept, concepts, cos_sims, per=0.2):
    """
    For each concept, finds the top-n aligning clusters based on binary IoU.

    Args:
        n (int): Number of top clusters to select.
        gt_samples_per_concept (dict): Mapping from concept to patch indices.
        concepts (dict): Mapping from cluster label to centroid.
        cos_sims (pd.DataFrame): Cosine similarity dataframe.
        per (float): Top % of patches to consider for each cluster.

    Returns:
        dict: Mapping from concept to list of top-n cluster ids (as strings).
    """
    concept_to_cluster_corrs = find_cluster_concept_correlations(gt_samples_per_concept, concepts, cos_sims, per=per)

    topn_clusters_per_concept = {}

    for concept, cluster_corrs in concept_to_cluster_corrs.items():
        # Get top-n cluster ids (keep only cluster names, discard scores)
        top_clusters = [cluster_id for cluster_id, score in cluster_corrs[:n]]
        topn_clusters_per_concept[concept] = top_clusters

    return topn_clusters_per_concept


# def compute_soft_concept_cluster_correlations(gt_samples_per_concept, cos_sims):
#     """
#     Computes soft correlation (cosine similarity) between concept presence vectors and cluster similarity vectors.

#     Args:
#         gt_samples_per_concept (dict): Mapping from concept name to list of patch indices.
#         cos_sims (pd.DataFrame): DataFrame of cosine similarities (rows = patches, columns = cluster labels).

#     Returns:
#         dict: Mapping from concept name to sorted list of (cluster_id, correlation_score) tuples.
#     """
#     concept_to_cluster_corrs = {}
#     total_patches = len(cos_sims)

#     # Create binary vector for each concept
#     concept_vectors = {
#         concept: torch.tensor([
#             1.0 if i in patch_indices else 0.0
#             for i in range(total_patches)
#         ])
#         for concept, patch_indices in gt_samples_per_concept.items()
#     }

#     # Convert cos_sims to tensor [N, K]
#     cos_sims_tensor = torch.tensor(cos_sims.values)  # [num_patches, num_clusters]

#     for concept, bin_vec in concept_vectors.items():
#         bin_vec = bin_vec.float()

#         # Normalize concept vector
#         bin_vec_norm = bin_vec / (bin_vec.norm(p=2) + 1e-8)

#         cluster_corrs = []
#         for i, cluster_label in enumerate(cos_sims.columns):
#             sim_vec = cos_sims_tensor[:, i]
#             sim_vec_norm = sim_vec / (sim_vec.norm(p=2) + 1e-8)

#             # Cosine similarity as soft correlation
#             score = torch.dot(sim_vec_norm, bin_vec_norm).item()
#             cluster_corrs.append((cluster_label, score))

#         cluster_corrs.sort(key=lambda x: x[1], reverse=True)
#         concept_to_cluster_corrs[concept] = cluster_corrs

#     return concept_to_cluster_corrs



# def get_topn_aligning_clusters_per_concept(n, gt_samples_per_concept, cos_sims):
#     """
#     For each concept, returns the top-n clusters with highest soft alignment score.

#     Args:
#         n (int): Number of top clusters to return.
#         gt_samples_per_concept (dict): Mapping from concept name to list of patch indices.
#         cos_sims (pd.DataFrame): DataFrame of cosine similarities (rows = patches, columns = cluster labels).

#     Returns:
#         dict: Mapping from concept name to list of top-n cluster ids.
#     """
#     concept_to_cluster_corrs = compute_soft_concept_cluster_correlations(gt_samples_per_concept, cos_sims)

#     topn_clusters_per_concept = {
#         concept: [cluster_id for cluster_id, _ in corrs[:n]]
#         for concept, corrs in concept_to_cluster_corrs.items()
#     }

#     return topn_clusters_per_concept



def get_most_aligning_cluster_per_concept(gt_samples_per_concept, concepts, cos_sims, per):
    """
    For each concept, finds the single cluster with the highest IoU score.

    Args:
        gt_samples_per_concept (dict): Mapping from concept to patch indices.
        concepts (dict): Mapping from cluster label to centroid.
        cos_sims (pd.DataFrame): Cosine similarity dataframe.
        per (float): Top % of patches to consider for each cluster.

    Returns:
        dict: Mapping from concept -> best matching cluster id (string).
    """
    concept_to_cluster_corrs = find_cluster_concept_correlations(gt_samples_per_concept, concepts, cos_sims, per=per)
    concept_to_cluster = {concept: cluster_corrs[0][0] for concept, cluster_corrs in concept_to_cluster_corrs.items()}
    return concept_to_cluster


###COMPUTING METRICS FOR ALL CONCEPTS ####

def detect_then_invert_metrics_all_pairs(
    detect_percentile, invert_percentiles, act_metrics, concepts, 
    gt_samples_per_concept, gt_samples_per_concept_test, relevant_indices,
    all_concept_labels, device, dataset_name, 
    model_input_size, con_label, all_object_patches=None, patch_size=14):
    """
    Performs two-stage concept detection over all (concept, cluster) pairs using cached thresholds.
    Computes metrics for multiple invert percentiles while processing concepts in parallel.
    """

    all_percentiles = [detect_percentile] + list(invert_percentiles)
    thresholds = compute_concept_thresholds_over_percentiles_all_pairs(
        gt_samples_per_concept_test, 
        act_metrics,
        all_percentiles,
        device=device,
        dataset_name=f"{dataset_name}-Cal",
        con_label=con_label,
        n_vectors=1,
        n_concepts_to_print=0
    )

    detect_thresholds = thresholds[detect_percentile]
    detected_patch_masks = get_patch_detection_tensor_all_pairs(
        act_metrics, detect_thresholds, model_input_size, dataset_name, device
    )

    concept_names = list(gt_samples_per_concept.keys())
    cluster_names = list(concepts.keys())

    if all_object_patches is not None:
        relevant_indices = torch.tensor(
            [int(idx.item()) for idx in relevant_indices if int(idx.item()) in all_object_patches],
            device=device
        )

    n_samples = relevant_indices.shape[0]
    n_clusters = len(cluster_names)
    n_concepts = len(concept_names)

    # Build act_metrics tensor: (n_samples, n_clusters)
    # act_metrics_gpu = torch.tensor(
    #     act_metrics.iloc[relevant_indices.tolist()].values.astype(np.float32),
    #     device=device
    # )
    act_metrics_cpu = act_metrics.iloc[relevant_indices.tolist()].astype(np.float32)
    act_metrics_gpu = None

    # Build gt_masks_all: (n_samples, n_concepts)
    gt_masks_all = torch.zeros((n_samples, n_concepts), device=device, dtype=torch.bool)
    for i, concept in enumerate(concept_names):
        gt_masks_all[:, i] = torch.tensor(
            all_concept_labels[concept][relevant_indices].cpu().numpy() == 1,
            device=device
        )

    # Build detected_patch_masks_all ONCE (n_samples, n_concepts, n_clusters)
    detected_patch_masks_all = torch.stack([
        torch.stack([detected_patch_masks[(concept, cluster)][relevant_indices] for cluster in cluster_names], dim=1)
        for concept in concept_names
    ], dim=1).to(device)  # (n_samples, n_concepts, n_clusters)

    metrics_dfs = {p: {
        'tp_count': {}, 'fp_count': {}, 'tn_count': {}, 'fn_count': {}
    } for p in invert_percentiles}

    # ----------------------
    # Now main invert loop
    # ----------------------
#     for invert_percentile in invert_percentiles:
#         invert_thresholds = thresholds[invert_percentile]

#         # Build a tensor of inversion thresholds for all (concept, cluster) pairs
#         invert_thresholds_tensor = torch.zeros((len(concept_names), len(cluster_names)), device=device)
#         for concept_idx, concept in enumerate(concept_names):
#             for cluster_idx, cluster in enumerate(cluster_names):
#                 invert_thresholds_tensor[concept_idx, cluster_idx] = invert_thresholds[(concept, cluster)][0]

#         # Expand act_metrics to match (n_samples, n_concepts, n_clusters)
#         act_vals_expanded = act_metrics_gpu.unsqueeze(1).expand(-1, len(concept_names), -1)

#         # Expand invert thresholds
#         invert_thresholds_expanded = invert_thresholds_tensor.unsqueeze(0)  # (1, n_concepts, n_clusters)

#         # Determine activation based on detection + inversion thresholds
#         above_invert_threshold = act_vals_expanded >= invert_thresholds_expanded
#         activated = detected_patch_masks_all & above_invert_threshold

#         # Expand ground truth mask
#         gt_masks_expanded = gt_masks_all.unsqueeze(2)  # (n_samples, n_concepts, 1)

#         # Compute confusion matrix elements (vectorized!)
#         tp = (activated & gt_masks_expanded).sum(dim=0)  # (n_concepts, n_clusters)
#         fn = ((~activated) & gt_masks_expanded).sum(dim=0)
#         fp = (activated & (~gt_masks_expanded)).sum(dim=0)
#         tn = ((~activated) & (~gt_masks_expanded)).sum(dim=0)

#         # Save to metrics_dfs
#         for concept_idx, concept in enumerate(concept_names):
#             for cluster_idx, cluster in enumerate(cluster_names):
#                 key = (concept, cluster)
#                 metrics_dfs[invert_percentile]['tp_count'][key] = tp[concept_idx, cluster_idx].item()
#                 metrics_dfs[invert_percentile]['fn_count'][key] = fn[concept_idx, cluster_idx].item()
#                 metrics_dfs[invert_percentile]['fp_count'][key] = fp[concept_idx, cluster_idx].item()
#                 metrics_dfs[invert_percentile]['tn_count'][key] = tn[concept_idx, cluster_idx].item()
    for invert_percentile in tqdm(invert_percentiles):
        invert_thresholds = thresholds[invert_percentile]

        # Build invert thresholds
        invert_thresholds_tensor = torch.zeros((n_concepts, n_clusters), device=device)
        for concept_idx, concept in enumerate(concept_names):
            for cluster_idx, cluster in enumerate(cluster_names):
                invert_thresholds_tensor[concept_idx, cluster_idx] = invert_thresholds[(concept, cluster)][0]

        # -- Instead of expanding full act_metrics_gpu, batch over clusters --
        tp = torch.zeros((n_concepts, n_clusters), device=device)
        fn = torch.zeros((n_concepts, n_clusters), device=device)
        fp = torch.zeros((n_concepts, n_clusters), device=device)
        tn = torch.zeros((n_concepts, n_clusters), device=device)

        batch_size = 50  # <- Adjustable, how many clusters to process at once

        for batch_start in range(0, n_clusters, batch_size):
            batch_end = min(batch_start + batch_size, n_clusters)
            cluster_batch = cluster_names[batch_start:batch_end]

            # Slice act_metrics only for cluster batch
            act_metrics_batch = torch.tensor(
                act_metrics_cpu[[str(c) for c in cluster_batch]].values,
                device=device
            )  # (n_samples, batch_size)

            # Expand dimensions
            act_vals_expanded = act_metrics_batch.unsqueeze(1).expand(-1, n_concepts, -1)  # (n_samples, n_concepts, batch_size)
            invert_thresholds_expanded = invert_thresholds_tensor[:, batch_start:batch_end].unsqueeze(0)  # (1, n_concepts, batch_size)
            detected_patch_masks_batch = torch.stack([
                torch.stack([detected_patch_masks[(concept, cluster)][relevant_indices] for cluster in cluster_batch], dim=1)
                for concept in concept_names
            ], dim=1).to(device)  # (n_samples, n_concepts, batch_size)

            above_invert_threshold = act_vals_expanded >= invert_thresholds_expanded
            activated = detected_patch_masks_batch & above_invert_threshold
            gt_masks_expanded = gt_masks_all.unsqueeze(2)  # (n_samples, n_concepts, 1)

            tp[:, batch_start:batch_end] = (activated & gt_masks_expanded).sum(dim=0)
            fn[:, batch_start:batch_end] = ((~activated) & gt_masks_expanded).sum(dim=0)
            fp[:, batch_start:batch_end] = (activated & (~gt_masks_expanded)).sum(dim=0)
            tn[:, batch_start:batch_end] = ((~activated) & (~gt_masks_expanded)).sum(dim=0)

        # Save to metrics_dfs
        for concept_idx, concept in enumerate(concept_names):
            for cluster_idx, cluster in enumerate(cluster_names):
                key = (concept, cluster)
                metrics_dfs[invert_percentile]['tp_count'][key] = tp[concept_idx, cluster_idx].item()
                metrics_dfs[invert_percentile]['fn_count'][key] = fn[concept_idx, cluster_idx].item()
                metrics_dfs[invert_percentile]['fp_count'][key] = fp[concept_idx, cluster_idx].item()
                metrics_dfs[invert_percentile]['tn_count'][key] = tn[concept_idx, cluster_idx].item()


    # ----------------------
    # Finish: organize final metrics
    # ----------------------
    final_metrics = {}
    for invert_percentile in invert_percentiles:
        counts = metrics_dfs[invert_percentile]
        metrics_df = compute_concept_metrics(
            counts['fp_count'], counts['fn_count'], 
            counts['tp_count'], counts['tn_count'], 
            counts['tp_count'].keys(), dataset_name, con_label, 
            just_obj=(all_object_patches is not None),
            invert_percentile=invert_percentile, 
            detect_percentile=detect_percentile
        )
        final_metrics[invert_percentile] = metrics_df

    return final_metrics


def detect_then_invert_metrics_over_percentiles_all_pairs(detect_percentiles, invert_percentiles, 
                                                          act_metrics, concepts, gt_samples_per_concept, 
                                                          gt_samples_per_concept_test, device, dataset_name, 
                                                          model_input_size, con_label, all_object_patches=None,
                                                          patch_size=14):
    """
    Evaluates metrics across all detect percentile combinations using cached threshold computation.
    More efficient version that handles multiple invert percentiles at once.
    """
    # Compute all thresholds at once for caching
    all_percentiles = sorted(list(set(detect_percentiles) | set(invert_percentiles)))
    thresholds = compute_concept_thresholds_over_percentiles_all_pairs(gt_samples_per_concept_test, 
                                                                       act_metrics, all_percentiles, 
                                                                       device, f"{dataset_name}-Cal",
                                                                       con_label, n_vectors=1, 
                                                                       n_concepts_to_print=0)
    
    total_iters = len(detect_percentiles)  # Now we only iterate over detect percentiles
    pbar = tqdm(total=total_iters, desc="Evaluating thresholds")

    # Get the split dataframe and indices
    split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
    test_indices = torch.tensor(split_df.index[split_df == 'test'].tolist())
    relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size)

    # Get ground truth labels
    all_concept_labels = create_binary_labels(len(split_df), gt_samples_per_concept)
    
    for detect_percentile in detect_percentiles:
        # Get valid invert percentiles for this detect percentile
        valid_invert_percentiles = [p for p in invert_percentiles if p >= detect_percentile]
        
        if not valid_invert_percentiles:
            continue

        # Compute metrics for all valid invert percentiles at once
        metrics = detect_then_invert_metrics_all_pairs(
            detect_percentile, valid_invert_percentiles,
            act_metrics, concepts,
            gt_samples_per_concept, gt_samples_per_concept_test,
            relevant_indices, all_concept_labels,
            device, dataset_name, model_input_size, con_label,
            all_object_patches=None,
            patch_size=patch_size
        )
        
        
#     # Compute metrics with object patches for all valid invert percentiles
#     if all_object_patches is not None:
#         metrics = detect_then_invert_metrics(
#             detect_percentile, valid_invert_percentiles,
#             act_metrics, concepts,
#             gt_samples_per_concept, gt_samples_per_concept_test,
#             device, dataset_name, model_input_size, con_label,
#             all_object_patches=all_object_patches, n_trials=n_trials,
#             balance_dataset=balance_dataset, patch_size=patch_size
#         )
        
        pbar.update(1)
    
    pbar.close()

def find_best_clusters_per_concept(metric_name, gt_samples_per_concept_test, dataset_name, con_label, 
                                  detect_percentiles, invert_percentiles, just_obj=False):
    """
    For each concept, finds the (detect, invert, cluster) triplet with the highest metric value.
    
    Returns:
        dict: { concept_name: { 'cluster': cluster_id, 'invert': invert_percentile, 'detect': detect_percentile, 'metric': metric_value } }
    """
    prefix = "" if not just_obj else "justobj_"
    invert_percentiles = sorted(invert_percentiles, reverse=True)
    detect_percentiles = sorted(detect_percentiles)

    all_concepts = list(gt_samples_per_concept_test.keys())
    best_info = {concept: {'cluster': None, 'invert': None, 'detect': None, metric_name: -np.inf} for concept in all_concepts}

    for invert_p in invert_percentiles:
        for detect_p in detect_percentiles:
            if invert_p >= detect_p:
                filename = f"Quant_Results/{dataset_name}/{prefix}detectfirst_{detect_p*100}_per_{invert_p*100}_{con_label}.csv"
                try:
                    df = pd.read_csv(filename)
                    
                    if isinstance(df['concept'].iloc[0], str):
                        df['concept'] = df['concept'].apply(ast.literal_eval)
                    
                    for concept in all_concepts:
                        sub_df = df[df['concept'].apply(lambda x: x[0] == concept)]
                        if not sub_df.empty and metric_name in sub_df.columns:
                            best_row = sub_df.loc[sub_df[metric_name].idxmax()]
                            best_metric = best_row[metric_name]
                            best_cluster = best_row['concept'][1]

                            if best_metric > best_info[concept][metric_name]:
                                best_info[concept] = {
                                    'cluster': best_cluster,
                                    'invert': invert_p,
                                    'detect': detect_p,
                                    metric_name: best_metric
                                }
                except FileNotFoundError:
                    print(f"Missing file: {filename}")
                    continue

    return best_info
       

def plot_best_cluster_heatmap_per_concept(concept_to_best_clusters, metric_name, dataset_name, con_label,
                                          detect_percentiles, invert_percentiles, just_obj=False):
    """
    Plots detect/invert heatmaps for all concepts, arranged 3 per row.
    """
    prefix = "" if not just_obj else "justobj_"
    invert_percentiles = sorted(invert_percentiles, reverse=True)
    detect_percentiles = sorted(detect_percentiles)

    all_concepts = list(concept_to_best_clusters.keys())
    n_concepts = len(all_concepts)
    n_cols = 3
    n_rows = (n_concepts + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()

    for idx, concept in enumerate(all_concepts):
        heatmap_data = []
        mask_data = []

        best_info = concept_to_best_clusters[concept]
        best_cluster = best_info['cluster']

        for invert_p in invert_percentiles:
            row = []
            mask_row = []
            for detect_p in detect_percentiles:
                if invert_p >= detect_p:
                    filename = f"Quant_Results/{dataset_name}/{prefix}detectfirst_{detect_p*100}_per_{invert_p*100}_{con_label}.csv"
                    
                    if os.path.exists(filename):
                        df = pd.read_csv(filename)
                        if isinstance(df['concept'].iloc[0], str):
                            df['concept'] = df['concept'].apply(ast.literal_eval)

                        # Filter for (concept, cluster)
                        sub_df = df[(df['concept'].apply(lambda x: x[0] == concept and str(x[1]) == str(best_cluster)))]
                        
                        if not sub_df.empty and metric_name in sub_df.columns:
                            value = sub_df.iloc[0][metric_name]
                        else:
                            value = np.nan
                    else:
                        print(f"Missing file: {filename}")
                        value = np.nan
                    
                    row.append(value)
                    mask_row.append(False)
                else:
                    row.append(np.nan)
                    mask_row.append(True)

            heatmap_data.append(row)
            mask_data.append(mask_row)

        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=[f"{p:.2f}" for p in invert_percentiles],
            columns=[f"{p:.2f}" for p in detect_percentiles]
        )
        mask = np.array(mask_data)

        ax = axes[idx]
        sns.heatmap(
            heatmap_df,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="plasma",
            cbar_kws={"label": metric_name},
            mask=mask,
            vmin=0,
            vmax=1
        )
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        max_val = np.nanmax(heatmap_df.values)
        if not np.isnan(max_val):
            max_idx = np.unravel_index(np.nanargmax(heatmap_df.values), heatmap_df.shape)
            max_detect = heatmap_df.columns[max_idx[1]]
            max_invert = heatmap_df.index[max_idx[0]]
            max_label = f" (Max: {max_val:.2f} @ detect={max_detect}, invert={max_invert}, cluster={best_cluster})"
        else:
            max_label = ""

        title_prefix = "Just Obj Patches" if just_obj else ""
        ax.set_title(f"{concept}\n{max_label}", pad=6, fontsize=10)

        ax.set_xlabel("Detect %", fontsize=8)
        ax.set_ylabel("Invert %", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide any extra subplots if concepts < grid
    for j in range(len(all_concepts), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

        
        
def plot_cluster_heatmap_per_concept(concept_to_cluster, metric_name, dataset_name, con_label,
                                     detect_percentiles, invert_percentiles, just_obj=False):
    """
    For each (concept, aligned cluster) pair, plot a grid of heatmaps showing the metric
    across all concepts vs that cluster.

    Args:
        concept_to_cluster (dict): Mapping from concept name -> aligned cluster id (str or int).
        metric_name (str): Metric name to plot (e.g., 'f1', 'accuracy').
        dataset_name (str): Dataset name.
        con_label (str): Concept label (e.g., 'color', 'shape').
        detect_percentiles (list): List of detection percentiles.
        invert_percentiles (list): List of inversion percentiles.
        just_obj (bool): Whether to add "justobj_" prefix for file paths.
    """
    prefix = "" if not just_obj else "justobj_"
    invert_percentiles = sorted(invert_percentiles, reverse=True)
    detect_percentiles = sorted(detect_percentiles)

    all_concepts = list(concept_to_cluster.keys())
    n_concepts = len(all_concepts)
    n_cols = 3
    n_rows = (n_concepts + n_cols - 1) // n_cols

    for anchor_concept, cluster_id in concept_to_cluster.items():
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()

        for idx, concept in enumerate(all_concepts):
            heatmap_data = []
            mask_data = []

            for invert_p in invert_percentiles:
                row = []
                mask_row = []
                for detect_p in detect_percentiles:
                    if invert_p >= detect_p:
                        filename = f"Quant_Results/{dataset_name}/{prefix}detectfirst_{detect_p*100}_per_{invert_p*100}_{con_label}.csv"

                        if os.path.exists(filename):
                            df = pd.read_csv(filename)
                            if isinstance(df['concept'].iloc[0], str):
                                df['concept'] = df['concept'].apply(ast.literal_eval)

                            # Filter for (current concept, fixed cluster_id)
                            sub_df = df[(df['concept'].apply(lambda x: x[0] == concept and str(x[1]) == str(cluster_id)))]

                            if not sub_df.empty and metric_name in sub_df.columns:
                                value = sub_df.iloc[0][metric_name]
                            else:
                                value = np.nan
                        else:
                            value = np.nan

                        row.append(value)
                        mask_row.append(False)
                    else:
                        row.append(np.nan)
                        mask_row.append(True)

                heatmap_data.append(row)
                mask_data.append(mask_row)

            heatmap_df = pd.DataFrame(
                heatmap_data,
                index=[f"{p:.2f}" for p in invert_percentiles],
                columns=[f"{p:.2f}" for p in detect_percentiles]
            )
            mask = np.array(mask_data)

            ax = axes[idx]
            sns.heatmap(
                heatmap_df,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap="plasma",
                cbar=True,
                mask=mask,
                vmin=0,
                vmax=1
            )

            ax.set_title(concept, fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")

        # Hide any unused subplots
        for idx in range(n_concepts, len(axes)):
            axes[idx].axis("off")

        title_prefix = "Just Obj Patches" if just_obj else ""
        fig.suptitle(
            f"Cluster {cluster_id} (aligned with {anchor_concept}) | Metric: {metric_name} {title_prefix}",
            fontsize=16, y=1.02
        )
        plt.tight_layout()
        plt.show()
      
    
def filter_and_save_best_clusters(dataset_name, con_label):
    """
    Filters detection metrics CSVs to only include best cluster per concept.

    Args:
        dataset_name (str): Name of dataset, used to match files.
        con_label (str): Concept group label (used in filenames).
    """
    metrics_dir = f"Quant_Results/{dataset_name}"
    
    # Try to load best clusters info
    try:
        best_clusters_by_detect = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)
    except FileNotFoundError:
        print(f"Warning: No best clusters file found for {con_label}, skipping filter_and_save_best_clusters")
        return
    
    # Pattern for all matching detection metric CSV files
    pattern = os.path.join(metrics_dir, f"detectionmetrics_allpairs_per_*_{con_label}.csv")
    
    metric_files = glob(pattern)
    if not metric_files:
        print(f"Warning: No detection metric files found matching pattern: {pattern}")
        return

    for metric_file in metric_files:
        try:
            # Check if file is empty
            if os.path.getsize(metric_file) == 0:
                print(f"Warning: Empty file {metric_file}, skipping")
                continue
                
            # Load the detection metrics CSV
            df = pd.read_csv(metric_file)
            
            if df.empty:
                print(f"Warning: No data in {metric_file}, skipping")
                continue
            
            # Filter to only include best clusters
            filtered_rows = []
            
            for idx, row in df.iterrows():
                # Parse the concept tuple
                concept, cluster = ast.literal_eval(row['concept'])
                
                # Check if this is the best cluster for this concept
                if concept in best_clusters_by_detect:
                    best_info = best_clusters_by_detect[concept]
                    if str(cluster) == str(best_info['best_cluster']):
                        # Update the concept column to just be the concept name
                        row['concept'] = concept
                        filtered_rows.append(row)
            
            # Create filtered dataframe and save
            if filtered_rows:
                filtered_df = pd.DataFrame(filtered_rows)
                # Remove '_allpairs' from filename
                output_filename = os.path.basename(metric_file).replace("_allpairs", "")
                save_path = os.path.join(metrics_dir, output_filename)
                filtered_df.to_csv(save_path, index=False)
                print(f"   Saved filtered metrics: {output_filename} ({len(filtered_df)} concepts)")
            else:
                print(f"Warning: No matching data found in {metric_file}")
                
        except pd.errors.EmptyDataError:
            print(f"Error: Empty CSV file {metric_file} - detection metrics may not have been computed")
            continue
        except Exception as e:
            print(f"Error processing {metric_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
def match_thresholds_across_percentiles(thresholds, dataset_name, con_label):
    alignment_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
    alignment_results = torch.load(alignment_path, weights_only=False)

    cluster_to_concept = {
        info['best_cluster']: concept for concept, info in alignment_results.items()
    }

    new_thresholds = defaultdict(dict)
    for per, threshold_dic in thresholds.items():
        for cluster, concept in cluster_to_concept.items():
            new_thresholds[per][cluster] = threshold_dic[(concept, cluster)]

    return dict(new_thresholds)
    
            
def get_matched_concepts_and_data(
    dataset_name,
    con_label,
    act_metrics,
    gt_samples_per_concept_cal=None,
    gt_samples_per_concept_test=None,
    gt_samples_per_concept=None,
    concepts=None,
    scratch_dir='',
    acts_file=None
):
    """
    Loads best alignment results and creates MatchedConceptActivationLoader for concept-aligned activation access.
    Always uses ChunkedActivationLoader internally for memory efficiency.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'CLEVR').
        con_label (str): Concept label (e.g., 'CLIP_kmeans_1000_patch...').
        act_metrics: IGNORED - will always use ChunkedActivationLoader instead.
        gt_patches_per_concept_cal (dict): Concept → list of calibration patch indices.
        gt_patches_per_concept_test (dict): Concept → list of test patch indices.
        gt_patches_per_concept (dict): Concept → list of train patch indices.
        concepts (dict): Cluster ID → concept embedding.
        scratch_dir (str): Scratch directory path

    Returns:
        matched_acts_loader (MatchedConceptActivationLoader): Memory-efficient concept-matched activation loader.
        matched_gt_cal (dict): Cluster ID → list of calibration patches.
        matched_gt_test (dict): Cluster ID → list of test patches.
        matched_gt (dict): Cluster ID → list of train patches.
        matched_concepts (dict): Cluster ID → concept vector.
    """
    from utils.memory_management_utils import ChunkedActivationLoader, MatchedConceptActivationLoader
    
    alignment_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
    alignment_results = torch.load(alignment_path, weights_only=False)

    matching_cluster_ids = [info['best_cluster'] for info in alignment_results.values()]
    
    matched_acts_loader, matched_gt_cal, matched_gt_test, matched_gt, matched_concepts = None, None, None, None, None
    
    # Create cluster_id -> concept_name mapping
    cluster_to_concept = {info['best_cluster']: concept_name for concept_name, info in alignment_results.items()}
    
    # Always use ChunkedActivationLoader for activation metrics
    # print("Creating MatchedConceptActivationLoader for memory-efficient activation access...")
    try:
        # Use provided acts_file if available, otherwise construct from con_label
        if acts_file is None:
            # Determine activation file name based on con_label (strip _cal suffix if present)
            base_con_label = con_label.replace('_cal', '') if con_label.endswith('_cal') else con_label
            if 'linsep' in con_label and 'kmeans' in con_label:
                # For kmeans linsep, need to extract parts and reconstruct
                parts = base_con_label.split('_')
                model_idx = parts.index('kmeans') - 1
                model_name = parts[model_idx]
                n_clusters = parts[parts.index('kmeans') + 1]
                sample_type = parts[parts.index('linsep') + 1]
                percent_thru_model = parts[-1]
                acts_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}"
            elif 'linsep' in con_label:
                acts_file = f"dists_{base_con_label}"
            elif 'kmeans' in con_label:
                # For regular kmeans, need to extract parts and add "concepts"
                parts = base_con_label.split('_')
                model_idx = 0
                model_name = parts[model_idx]
                n_clusters = parts[parts.index('kmeans') + 1]
                
                # Find sample_type - it's after n_clusters and before 'embeddings'
                embeddings_idx = parts.index('embeddings')
                sample_type = parts[embeddings_idx - 1]
                
                percent_thru_model = parts[-1]
                acts_file = f"cosine_similarities_kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}"
            elif 'sae' in con_label:
                # For SAE dense activations
                parts = base_con_label.split('_')
                model_name = parts[0]  # CLIP or Gemma
                sample_type = parts[2]  # patch or cls
                if model_name == 'CLIP':
                    acts_file = f"clipscope_{sample_type}_dense"
                else:  # Gemma
                    acts_file = f"gemmascope_{sample_type}_dense"
            else:
                acts_file = f"cosine_similarities_{base_con_label}"
        else:
            # Use the provided acts_file, removing .pt extension if present since ChunkedActivationLoader adds it
            if acts_file.endswith('.pt'):
                acts_file = acts_file[:-3]
        
        # Try to load filtered version first if it's a kmeans method
        if 'kmeans' in con_label:
            filtered_acts_file = acts_file.replace('.pt', '_filtered.pt')
            # Check if filtered version exists
            import os
            acts_dir = 'Distances' if 'dists_' in acts_file else 'Cosine_Similarities'
            filtered_path = os.path.join(scratch_dir, acts_dir, dataset_name, filtered_acts_file + '.pt')
            if os.path.exists(filtered_path) or os.path.exists(filtered_path.replace('.pt', '_chunks_info.json')):
                acts_file = filtered_acts_file
        
        activation_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir)
        
        # Create the matched concept activation loader
        matched_acts_loader = MatchedConceptActivationLoader(activation_loader, cluster_to_concept)
        
        # NOTE: Not saving CSV automatically to avoid loading all data at once
        # If you need the CSV for backward compatibility, call: matched_acts_loader.to_csv(path)
        # print(f"Created MatchedConceptActivationLoader with {len(matched_acts_loader.columns)} matching clusters")
            
    except Exception as e:
        print(f"DEBUG: Could not load chunked activation files: {e}")
        print(f"DEBUG: Exception type: {type(e)}")
        matched_acts_loader = None
    
    if gt_samples_per_concept_cal is not None:
        # Map concept names to ground truth patches (for consistency with supervised methods)
        matched_gt_cal = {concept_name: gt_samples_per_concept_cal[concept_name]
                          for concept_name, info in alignment_results.items()
                          if concept_name in gt_samples_per_concept_cal}
        torch.save(matched_gt_cal, f'Unsupervised_Matches/{dataset_name}/gt_samples_per_concept_cal_{con_label}.pt')
    
    if gt_samples_per_concept_test is not None:
        # Map concept names to ground truth patches (for consistency with supervised methods)
        matched_gt_test = {concept_name: gt_samples_per_concept_test[concept_name]
                           for concept_name, info in alignment_results.items()
                           if concept_name in gt_samples_per_concept_test}
        torch.save(matched_gt_test, f'Unsupervised_Matches/{dataset_name}/gt_samples_per_concept_test_{con_label}.pt')
        
    if gt_samples_per_concept is not None:
        # Map concept names to ground truth patches (for consistency with supervised methods)
        matched_gt = {concept_name: gt_samples_per_concept[concept_name]
                      for concept_name, info in alignment_results.items()
                      if concept_name in gt_samples_per_concept}
        torch.save(matched_gt, f'Unsupervised_Matches/{dataset_name}/gt_samples_per_concept_{con_label}.pt')
        
    if concepts is not None:
        # Map concept names to concept vectors (using the matched cluster vectors)
        # This ensures consistency with supervised methods where concept names are keys
        matched_concepts = {concept_name: concepts[info['best_cluster']] 
                           for concept_name, info in alignment_results.items() 
                           if info['best_cluster'] in concepts}
        torch.save(matched_concepts, f'Unsupervised_Matches/{dataset_name}/concepts_{con_label}.pt') 

    return matched_acts_loader, matched_gt_cal, matched_gt_test, matched_gt, matched_concepts


# REMOVED DUPLICATE FUNCTION: get_matched_concepts_and_data
# This simpler version with fewer parameters was removed in favor of the more
# comprehensive version at line 2575 that supports optional parameters


def plot_cluster_f1_distributions_auto(
    dataset_name,
    model_name,
    n_clusters,
    gt_concepts_to_show=None,
    top_n=None,
    figsize_per_concept=(10, 4),
    save_dir=None,
    percentiles_to_search=[0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    metric='f1'
):
    """
    Automatically finds the best percentile based on average calibration performance across all concepts,
    then plots the TEST F1 scores for clusters at that single best percentile, sorted by test performance.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'CLEVR')
        model_name (str): Model name (e.g., 'CLIP')
        n_clusters (int): Number of clusters (e.g., 1000)
        gt_concepts_to_show (list): List of specific concepts to plot. If None, plots all.
        top_n (int): If specified, only show the top N highest F1 scoring clusters
        figsize_per_concept (tuple): Figure size for each concept's plot
        save_dir (str): Directory to save plots. If None, doesn't save.
        percentiles_to_search (list): Percentiles to search for best calibration performance
        metric (str): Metric to optimize ('f1', 'precision', 'recall')
        
    Returns:
        dict: Mapping of concept -> list of (cluster_id, test_f1_score) tuples
    """
    from tqdm import tqdm
    
    # Construct concept label based on model type
    if 'patch' in model_name.lower() or 'CLIP' in model_name:
        sample_type = 'patch'
        con_label = f'{model_name}_kmeans_{n_clusters}_patch_embeddings_kmeans_percentthrumodel_100'
    else:
        sample_type = 'cls'
        con_label = f'{model_name}_kmeans_{n_clusters}_cls_embeddings_kmeans_percentthrumodel_100'
    
    print(f"Using concept label: {con_label}")
    
    # Find the single best percentile based on average calibration performance
    print("Finding best percentile from calibration data...")
    best_percentile = None
    best_avg_metric = -1
    best_matches_cal = None
    
    for per in tqdm(percentiles_to_search, desc="Searching calibration percentiles"):
        # Load CALIBRATION metrics - look for _cal suffix in same directory
        cal_metrics_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{per}_{con_label}_cal.csv'
        
        try:
            cal_df = pd.read_csv(cal_metrics_path)
        except FileNotFoundError:
            print(f"Warning: Missing calibration file {cal_metrics_path}")
            continue
            
        # Parse concept column to extract GT concept and cluster ID
        parsed_data = []
        for idx, row in cal_df.iterrows():
            concept_tuple = ast.literal_eval(row['concept'])
            gt_concept = concept_tuple[0]
            cluster_id = concept_tuple[1]
            
            if gt_concepts_to_show and gt_concept not in gt_concepts_to_show:
                continue
                
            parsed_data.append({
                'gt_concept': gt_concept,
                'cluster_id': cluster_id,
                metric: row[metric] if metric in row else 0.0
            })
        
        if not parsed_data:
            continue
            
        parsed_df = pd.DataFrame(parsed_data)
        
        # Find best cluster for each GT concept
        concept_best_metrics = []
        matches_at_this_percentile = {}
        
        for concept in parsed_df['gt_concept'].unique():
            concept_df = parsed_df[parsed_df['gt_concept'] == concept]
            if not concept_df.empty and concept_df[metric].notna().any():
                best_row = concept_df.loc[concept_df[metric].idxmax()]
                concept_best_metrics.append(best_row[metric])
                matches_at_this_percentile[concept] = (best_row['cluster_id'], best_row[metric])
        
        # Calculate average of best metrics across concepts
        if concept_best_metrics:
            avg_metric = np.mean(concept_best_metrics)
            
            if avg_metric > best_avg_metric:
                best_avg_metric = avg_metric
                best_percentile = per
                best_matches_cal = matches_at_this_percentile
    
    if best_percentile is None:
        print("No valid calibration data found!")
        return {}
    
    print(f"Best percentile: {best_percentile} (avg best {metric}={best_avg_metric:.4f})")
    
    # Now load TEST data at the single best percentile
    print(f"\nLoading test data at best percentile {best_percentile}...")
    test_metrics_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{best_percentile}_{con_label}.csv'
    
    try:
        test_df = pd.read_csv(test_metrics_path)
    except FileNotFoundError:
        print(f"Error: Missing test file {test_metrics_path}")
        return {}
    
    # Extract all cluster scores for each concept
    concept_test_scores = defaultdict(list)
    
    for idx, row in test_df.iterrows():
        concept_tuple = ast.literal_eval(row['concept'])
        gt_concept = concept_tuple[0]
        cluster_id = str(concept_tuple[1]).strip("'\"")
        
        # Filter by requested concepts if specified
        if gt_concepts_to_show and gt_concept not in gt_concepts_to_show:
            continue
            
        metric_value = row[metric] if metric in row else 0.0
        
        if pd.notna(metric_value):
            concept_test_scores[gt_concept].append({
                'cluster': cluster_id,
                metric: metric_value
            })
    
    if not concept_test_scores:
        print("No concepts found to plot!")
        return {}
    
    print(f"\nPlotting TEST {metric.upper()} distributions for {len(concept_test_scores)} concepts at percentile {best_percentile}...")
    
    # Process and plot each concept
    all_concept_data = {}
    
    for concept in sorted(concept_test_scores.keys()):
        scores_data = concept_test_scores[concept]
        
        if not scores_data:
            print(f"No test scores found for concept: {concept}")
            continue
        
        # Get calibration info for this concept
        cal_cluster, cal_metric = best_matches_cal.get(concept, ('Unknown', 0))
        
        # Convert to list of tuples and sort by metric score
        cluster_metric_pairs = [(d['cluster'], d[metric]) for d in scores_data]
        cluster_metric_pairs.sort(key=lambda x: x[1])  # Sort by metric ascending
        
        # Apply top_n filter if specified
        if top_n and len(cluster_metric_pairs) > top_n:
            cluster_metric_pairs = cluster_metric_pairs[-top_n:]  # Keep top N highest
            
        all_concept_data[concept] = cluster_metric_pairs
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=figsize_per_concept)
        
        # Extract data for plotting
        clusters = [x[0] for x in cluster_metric_pairs]
        metric_scores = [x[1] for x in cluster_metric_pairs]
        
        # Create bar plot
        x_positions = np.arange(len(clusters))
        bars = ax.bar(x_positions, metric_scores, alpha=0.7)
        
        # Color bars by metric score
        colors = plt.cm.viridis(np.array(metric_scores) / max(metric_scores) if max(metric_scores) > 0 else np.ones_like(metric_scores))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Customize plot
        title = f"TEST {metric.upper()} Scores for Clusters Aligned with: {concept}"
        if top_n:
            title += f" (Top {top_n})"
        title += f"\n[Best percentile: {best_percentile}, Cal best cluster: {cal_cluster} ({metric}={cal_metric:.3f})]"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_xlabel(f"Cluster Rank (by TEST {metric} score)", fontsize=12)
        ax.set_ylabel(f"TEST {metric.upper()} Score", fontsize=12)
        ax.set_ylim(0, 1.05)
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars if not too many
        if len(clusters) <= 30:
            for i, score in enumerate(metric_scores):
                ax.text(i, score + 0.01, f"{score:.3f}", ha='center', va='bottom', fontsize=8)
        
        # Add statistics
        if metric_scores:
            mean_score = np.mean(metric_scores)
            median_score = np.median(metric_scores)
            max_score = max(metric_scores)
            
            stats_text = f"TEST Stats - Max: {max_score:.3f} | Mean: {mean_score:.3f} | Median: {median_score:.3f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        # Save if directory specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{save_dir}/test_f1_distribution_{concept}_{con_label}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filename}")
        
        plt.show()
        plt.close()
    
    return all_concept_data