# Pretrained SAE Evaluation Pipeline

This directory contains scripts for evaluating pretrained Sparse Autoencoders (SAEs) on concept detection tasks. The pipeline treats SAE units as potential concept detectors and evaluates their performance.

## Overview

The SAE evaluation pipeline follows the same structure as the regular unsupervised concept detection pipeline, but works with SAE activations instead of computing new concept vectors. The key insight is that SAE units may already encode meaningful concepts, so we align them with ground truth concepts and evaluate their performance.

## Pipeline Steps

### 1. Generate SAE Activations
First, run the pretrained SAE script to generate activations:
```bash
cd ../
python pretrained_saes.py --models CLIP --datasets CLEVR --sample-types patch
```

This creates SAE activation files in `SCRATCH_DIR/SAE_Acts/{dataset}/`

### 2. Run Full Evaluation Pipeline
```bash
python run_full_sae_pipeline.py --models CLIP --datasets CLEVR --sample-types patch
```

This runs all evaluation steps in sequence:

#### Step 1: Compute SAE Alignment (`compute_sae_alignment.py`)
- Computes detection metrics for all SAE unit × concept pairs
- Finds the best SAE unit for each ground truth concept
- Saves alignment mapping SAE units to concepts

#### Step 2: Compute Validation Thresholds (`sae_validation_thresholds.py`)
- Uses calibration data to find optimal activation thresholds
- Only processes aligned SAE units (not all 8192)
- Saves thresholds for detection

#### Step 3: Compute Detection Statistics (`sae_detection_stats.py`)
- Evaluates detection performance on test data
- Computes precision, recall, and F1 scores
- Saves detection metrics

#### Step 4: Compute Inversion Statistics (`sae_inversion_stats.py`)
- Finds superdetector patches for SAE units
- Performs concept inversion to visualize what units detect
- Creates summary visualizations

## Individual Script Usage

You can also run scripts individually:

```bash
# Just compute alignment
python compute_sae_alignment.py --models CLIP --datasets CLEVR --sample-types patch

# Just compute thresholds (requires alignment)
python sae_validation_thresholds.py --models CLIP --datasets CLEVR --sample-types patch

# Just compute detection stats (requires alignment and thresholds)
python sae_detection_stats.py --models CLIP --datasets CLEVR --sample-types patch

# Just compute inversions (requires alignment)
python sae_inversion_stats.py --models CLIP --datasets CLEVR --sample-types patch
```

## Pipeline Options

The full pipeline script supports skipping steps:

```bash
# Skip alignment if already computed
python run_full_sae_pipeline.py --skip-alignment

# Skip multiple steps
python run_full_sae_pipeline.py --skip-alignment --skip-thresholds

# Run only specific models/datasets
python run_full_sae_pipeline.py --models CLIP --datasets CLEVR COCO --sample-types patch
```

## Output Files

- **Alignment**: `SCRATCH_DIR/Concepts/{dataset}/sae_alignment_{model}_{sae}_{sample_type}_*.pt`
- **Thresholds**: `Metrics/{dataset}/concept_thresholds_{model}_{sae}_{sample_type}_*.pt`
- **Detection Metrics**: `Metrics/{dataset}/detection_metrics_{model}_{sae}_{sample_type}_*.pt`
- **Inversions**: `Inversions/{dataset}/{model}_{sae}_{sample_type}_*/`

## Key Differences from Regular Pipeline

1. **No concept learning**: SAE units are the concepts
2. **Alignment step**: Maps SAE units to ground truth concepts
3. **Sparse activations**: SAE outputs are already sparse (ReLU)
4. **Large number of units**: Typically 8192 SAE units vs 1000 k-means clusters

## Requirements

- Completed SAE activation generation (`pretrained_saes.py`)
- Ground truth concept annotations
- Sufficient GPU memory for processing

## Notes

- Currently supports CLIP patch embeddings with PatchSAE
- Llama SAE support removed (pretrained models not working)
- Processing can be memory intensive due to large number of SAE units