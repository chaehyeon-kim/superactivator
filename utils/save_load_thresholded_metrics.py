import os
import torch
import json
import pickle
from typing import Dict, Optional, Union
from datetime import datetime


def save_thresholded_metrics_results(
    results: Dict,
    save_path: Optional[str] = None,
    save_dir: str = "Saved_Results/thresholded_metrics",
    filename_prefix: str = "thresholded_metrics",
    format: str = "pt",
    include_metadata: bool = True
) -> str:
    """
    Save the results from compute_overlap_across_layers_data to disk.
    
    Args:
        results: Results dictionary from compute_overlap_across_layers_data
        save_path: Full path to save file (overrides save_dir and filename_prefix)
        save_dir: Directory to save results (default: "Saved_Results/thresholded_metrics")
        filename_prefix: Prefix for the filename (default: "thresholded_metrics")
        format: Save format - "pt" (PyTorch), "pkl" (pickle), or "json" (JSON) (default: "pt")
        include_metadata: Whether to include metadata like timestamp and data summary
        
    Returns:
        str: Path where the results were saved
        
    Note:
        - PyTorch format (.pt) is recommended for best compatibility and performance
        - JSON format will convert numpy arrays and tensors to lists (larger file size)
        - Pickle format works but may have compatibility issues across Python versions
    """
    # Create save directory if it doesn't exist
    if save_path is None:
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include dataset and model info in filename
        datasets = set()
        models = set()
        concept_types = set()
        
        for key, data in results.items():
            datasets.add(data['dataset'])
            models.add(data['model'])
            concept_types.add(data['concept_type'])
        
        # Create informative filename
        datasets_str = "-".join(sorted(datasets))[:50]  # Limit length
        models_str = "-".join(sorted(models))[:30]
        
        filename = f"{filename_prefix}_{datasets_str}_{models_str}_{timestamp}.{format}"
        save_path = os.path.join(save_dir, filename)
    else:
        # Ensure directory exists for provided path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Add metadata if requested
    if include_metadata:
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'n_configurations': len(results),
            'datasets': sorted(list(set(data['dataset'] for data in results.values()))),
            'models': sorted(list(set(data['model'] for data in results.values()))),
            'concept_types': sorted(list(set(data['concept_type'] for data in results.values()))),
            'background_percentiles': sorted(list(set(data.get('background_percentile', 0.99) for data in results.values()))),
            'keys': list(results.keys())
        }
        
        save_data = {
            'metadata': metadata,
            'results': results
        }
    else:
        save_data = results
    
    # Save based on format
    if format == "pt":
        torch.save(save_data, save_path)
    elif format == "pkl":
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
    elif format == "json":
        # Convert numpy arrays and tensors to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (torch.Tensor, np.ndarray)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        import numpy as np
        serializable_data = convert_to_serializable(save_data)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}. Choose from 'pt', 'pkl', or 'json'")
    
    print(f"Results saved to: {save_path}")
    
    # Print summary
    if include_metadata:
        print(f"  - {metadata['n_configurations']} configurations")
        print(f"  - Datasets: {', '.join(metadata['datasets'])}")
        print(f"  - Models: {', '.join(metadata['models'])}")
        print(f"  - Concept types: {', '.join(metadata['concept_types'])}")
    
    return save_path


def load_thresholded_metrics_results(
    load_path: str,
    format: Optional[str] = None
) -> Union[Dict, tuple]:
    """
    Load previously saved results from compute_overlap_across_layers_data.
    
    Args:
        load_path: Path to the saved file
        format: Format of the file - "pt", "pkl", or "json" (default: auto-detect from extension)
        
    Returns:
        If file contains metadata: tuple of (results_dict, metadata_dict)
        Otherwise: results_dict
        
    Example:
        # Load and separate metadata
        results, metadata = load_thresholded_metrics_results("path/to/results.pt")
        
        # Or if no metadata
        results = load_thresholded_metrics_results("path/to/results.pt")
    """
    # Auto-detect format from file extension if not provided
    if format is None:
        ext = os.path.splitext(load_path)[1].lower()
        if ext == '.pt':
            format = 'pt'
        elif ext == '.pkl':
            format = 'pkl'
        elif ext == '.json':
            format = 'json'
        else:
            raise ValueError(f"Cannot auto-detect format from extension: {ext}")
    
    # Load based on format
    if format == "pt":
        data = torch.load(load_path, map_location='cpu')
    elif format == "pkl":
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
    elif format == "json":
        with open(load_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Check if data contains metadata
    if isinstance(data, dict) and 'metadata' in data and 'results' in data:
        print(f"Loaded results from: {load_path}")
        metadata = data['metadata']
        print(f"  - Timestamp: {metadata.get('timestamp', 'Unknown')}")
        print(f"  - {metadata.get('n_configurations', 'Unknown')} configurations")
        print(f"  - Datasets: {', '.join(metadata.get('datasets', []))}")
        print(f"  - Models: {', '.join(metadata.get('models', []))}")
        return data['results'], metadata
    else:
        print(f"Loaded results from: {load_path}")
        return data


def save_thresholded_metrics_summary(
    results: Dict,
    save_path: Optional[str] = None,
    save_dir: str = "Saved_Results/thresholded_metrics",
    filename: str = "summary.csv"
) -> str:
    """
    Save a CSV summary of the thresholded metrics results.
    
    Creates a table with columns:
    - dataset, model, concept_type, layer, gt_mass_mean, gt_mass_std, detection_rate_mean, detection_rate_std
    
    Args:
        results: Results dictionary from compute_overlap_across_layers_data
        save_path: Full path to save file (overrides save_dir and filename)
        save_dir: Directory to save summary
        filename: Filename for the summary
        
    Returns:
        str: Path where the summary was saved
    """
    import pandas as pd
    import numpy as np
    
    if save_path is None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create summary data
    summary_rows = []
    
    for key, data in results.items():
        dataset = data['dataset']
        model = data['model']
        concept_type = data['concept_type']
        percentthrus = data['percentthrus']
        gt_mass_values = data['gt_mass_above_threshold']
        detection_rates = data['detection_rates']
        
        # Get per-concept data for computing std
        gt_mass_per_concept = data.get('gt_mass_per_concept', {})
        detection_per_concept = data.get('detection_rates_per_concept', {})
        
        for i, percentthru in enumerate(percentthrus):
            if i < len(gt_mass_values) and i < len(detection_rates):
                # Get values for this layer across concepts
                layer_gt_masses = []
                layer_detections = []
                
                for concept, values in gt_mass_per_concept.items():
                    if i < len(values) and not np.isnan(values[i]):
                        layer_gt_masses.append(values[i])
                
                for concept, values in detection_per_concept.items():
                    if i < len(values) and not np.isnan(values[i]):
                        layer_detections.append(values[i])
                
                # Compute statistics
                gt_mass_mean = gt_mass_values[i] if not np.isnan(gt_mass_values[i]) else None
                gt_mass_std = np.std(layer_gt_masses) if layer_gt_masses else None
                detection_mean = detection_rates[i] if not np.isnan(detection_rates[i]) else None
                detection_std = np.std(layer_detections) if layer_detections else None
                
                summary_rows.append({
                    'dataset': dataset,
                    'model': model,
                    'concept_type': concept_type,
                    'percentthru': percentthru,
                    'gt_mass_mean': gt_mass_mean,
                    'gt_mass_std': gt_mass_std,
                    'detection_rate_mean': detection_mean,
                    'detection_rate_std': detection_std,
                    'n_concepts': len(layer_gt_masses)
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(summary_rows)
    df.to_csv(save_path, index=False)
    
    print(f"Summary saved to: {save_path}")
    print(f"  - {len(df)} rows")
    print(f"  - {len(df['dataset'].unique())} datasets")
    print(f"  - {len(df['model'].unique())} models")
    
    return save_path