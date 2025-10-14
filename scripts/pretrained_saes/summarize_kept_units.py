#!/usr/bin/env python3
"""
Summarize which SAE units were kept after pruning for each dataset/model combination.
"""

import json
import os
import glob

def summarize_kept_units():
    """Find and summarize all unit mapping files"""
    
    filtered_dir = 'SCRATCH_DIR/SAE_Activations_Filtered'
    
    print("SAE Unit Pruning Summary")
    print("=" * 80)
    
    total_summary = {}
    
    # Find all unit mapping files
    mapping_files = glob.glob(f"{filtered_dir}/*/*unit_mapping.json", recursive=True)
    
    if not mapping_files:
        print(f"No unit mapping files found in {filtered_dir}")
        return
    
    # Process each file
    for mapping_file in sorted(mapping_files):
        # Parse dataset, model, and sample type from path
        parts = mapping_file.split('/')
        dataset = parts[-3]
        filename = parts[-1]
        
        # Extract model and sample type from filename
        if 'clipscope' in filename:
            model = 'CLIP'
            if 'patch' in filename:
                sample_type = 'patch'
            else:
                sample_type = 'cls'
        elif 'gemmascope' in filename:
            model = 'Gemma'
            if 'patch' in filename or 'token' in filename:
                sample_type = 'patch/token'
            else:
                sample_type = 'cls'
        else:
            continue
            
        # Load mapping
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        n_original = mapping['n_original']
        n_filtered = mapping['n_filtered']
        kept_units = mapping['kept_units']
        
        key = f"{dataset} - {model} - {sample_type}"
        total_summary[key] = {
            'n_original': n_original,
            'n_kept': n_filtered,
            'reduction': f"{(1 - n_filtered/n_original)*100:.1f}%",
            'kept_units': kept_units[:10],  # Show first 10
            'file': mapping_file
        }
        
        print(f"\n{key}:")
        print(f"  Original units: {n_original:,}")
        print(f"  Kept units: {n_filtered:,}")
        print(f"  Reduction: {(1 - n_filtered/n_original)*100:.1f}%")
        print(f"  First 10 kept units: {kept_units[:10]}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    
    clip_patch_kept = []
    clip_cls_kept = []
    gemma_patch_kept = []
    gemma_cls_kept = []
    
    for key, info in total_summary.items():
        if 'CLIP' in key and 'patch' in key:
            clip_patch_kept.append(info['n_kept'])
        elif 'CLIP' in key and 'cls' in key:
            clip_cls_kept.append(info['n_kept'])
        elif 'Gemma' in key and 'patch' in key:
            gemma_patch_kept.append(info['n_kept'])
        elif 'Gemma' in key and 'cls' in key:
            gemma_cls_kept.append(info['n_kept'])
    
    if clip_patch_kept:
        avg = sum(clip_patch_kept) / len(clip_patch_kept)
        print(f"\nCLIP patch: avg {avg:.0f} units kept across {len(clip_patch_kept)} datasets")
        
    if clip_cls_kept:
        avg = sum(clip_cls_kept) / len(clip_cls_kept)
        print(f"CLIP cls: avg {avg:.0f} units kept across {len(clip_cls_kept)} datasets")
        
    if gemma_patch_kept:
        avg = sum(gemma_patch_kept) / len(gemma_patch_kept)
        print(f"Gemma patch/token: avg {avg:.0f} units kept across {len(gemma_patch_kept)} datasets")
        
    if gemma_cls_kept:
        avg = sum(gemma_cls_kept) / len(gemma_cls_kept)
        print(f"Gemma cls: avg {avg:.0f} units kept across {len(gemma_cls_kept)} datasets")
    
    # Save full mapping to file
    output_file = '/workspace/Experiments/scripts/pretrained_saes/kept_units_summary.json'
    with open(output_file, 'w') as f:
        # Save full unit lists
        full_summary = {}
        for key, info in total_summary.items():
            with open(info['file'], 'r') as mf:
                full_mapping = json.load(mf)
            full_summary[key] = {
                'n_original': info['n_original'],
                'n_kept': info['n_kept'],
                'kept_units': full_mapping['kept_units']
            }
        json.dump(full_summary, f, indent=2)
    
    print(f"\nFull unit lists saved to: {output_file}")

if __name__ == "__main__":
    summarize_kept_units()