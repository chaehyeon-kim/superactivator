import sys
import os
import torch
import argparse

from transformers import CLIPModel, AutoProcessor, MllamaForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.general_utils import load_images
from utils.compute_concepts_utils import compute_avg_concept_vectors
from utils.activation_utils import compute_cosine_sims
from utils.embedding_utils import compute_batch_embeddings
from utils.default_percentthrumodels import CLIP_PERCENTTHRUMODELS, LLAMA_VISION_PERCENTTHRUMODELS


PERCENT_THRU_MODEL = 100  # Default value, can be overridden by command line
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAMES = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces']
scratch_dir='' 


#for clip
clip_model_name = "openai/clip-vit-large-patch14"
CLIP_PROCESSOR = AutoProcessor.from_pretrained(clip_model_name)
CLIP_MODEL = CLIPModel.from_pretrained(clip_model_name).to(DEVICE)
CLIP_MODEL.eval()
CLIP_INPUT_IMAGE_SIZE = (224, 224)

# #for llama
llama_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LLAMA_MODEL = MllamaForConditionalGeneration.from_pretrained(llama_model_id, torch_dtype=torch.float16).to(DEVICE)
LLAMA_PROCESSOR = AutoProcessor.from_pretrained(llama_model_id)
LLAMA_MODEL.eval()
LLAMA_INPUT_IMAGE_SIZE = (560, 560)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embed image datasets using CLIP and Llama models')
    parser.add_argument('--model', type=str, choices=['CLIP', 'Llama', 'all'], default='all',
                        help='Model to use for embedding (default: all)')
    parser.add_argument('--models', nargs='+', choices=['CLIP', 'Llama'],
                        help='Multiple models to use for embedding')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process')
    parser.add_argument('--datasets', nargs='+', help='Multiple datasets to process')
    parser.add_argument('--percentthrumodels', nargs='+', type=int,
                        help='List of percentages through model layers to use (default: model-specific every 2 layers)')
    
    args = parser.parse_args()
    
    # Determine which model(s) to process
    models_to_process = []
    if args.models:
        models_to_process = args.models
    elif args.model == 'CLIP':
        models_to_process = ['CLIP']
    elif args.model == 'Llama':
        models_to_process = ['Llama']
    else:  # 'all'
        models_to_process = ['CLIP', 'Llama']
    
    # Set default percentthrumodels based on selected model(s)
    if args.percentthrumodels:
        percentthrumodels = args.percentthrumodels
    else:
        # Use model-specific defaults
        if len(models_to_process) == 1:
            if models_to_process[0] == 'CLIP':
                percentthrumodels = CLIP_PERCENTTHRUMODELS
            else:  # Llama
                percentthrumodels = LLAMA_VISION_PERCENTTHRUMODELS
        else:  # Multiple models
            # When processing multiple models, use union of both defaults
            percentthrumodels = sorted(list(set(CLIP_PERCENTTHRUMODELS + LLAMA_VISION_PERCENTTHRUMODELS)))
    
    print(f"Models to process: {models_to_process}")
    print(f"Percentthrumodels ({len(percentthrumodels)} values): {percentthrumodels}")
    
    # Determine which datasets to process
    if args.dataset:
        datasets_to_process = [args.dataset]
    elif args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = DATASET_NAMES
    
    # Loop through all percentthrumodels
    for PERCENT_THRU_MODEL in percentthrumodels:
        print(f"\n{'='*60}")
        print(f"Processing with PERCENT_THRU_MODEL = {PERCENT_THRU_MODEL}")
        print(f"{'='*60}\n")
        
        for DATASET_NAME in datasets_to_process:
            print(f"Dataset: {DATASET_NAME}, PercentThruModel: {PERCENT_THRU_MODEL}")
            
            # Process CLIP model if selected
            if 'CLIP' in models_to_process:
                print("\n=== Processing CLIP Model ===")
                clip_images, _, _ = load_images(dataset_name=DATASET_NAME, model_input_size=CLIP_INPUT_IMAGE_SIZE)
                
                print(f"Computing both CLS and patch embeddings for CLIP")
                compute_batch_embeddings(clip_images, CLIP_MODEL, CLIP_PROCESSOR, DEVICE, 
                                        percent_thru_model=PERCENT_THRU_MODEL, dataset_name=DATASET_NAME,
                                        model_input_size=CLIP_INPUT_IMAGE_SIZE,
                                        batch_size=100, scratch_dir=scratch_dir)
                
                torch.cuda.empty_cache()            
                torch.cuda.ipc_collect()
            
            # Process Llama model if selected
            if 'Llama' in models_to_process:
                print("\n=== Processing Llama Model ===")
                llama_images, _, _ = load_images(dataset_name=DATASET_NAME, model_input_size=LLAMA_INPUT_IMAGE_SIZE)
                
                print(f"Computing both CLS and patch embeddings for Llama")
                compute_batch_embeddings(llama_images, LLAMA_MODEL, LLAMA_PROCESSOR, DEVICE, 
                                        percent_thru_model=PERCENT_THRU_MODEL, dataset_name=DATASET_NAME,
                                        model_input_size=LLAMA_INPUT_IMAGE_SIZE,
                                        batch_size=2, scratch_dir=scratch_dir)  # Reduced batch size to avoid OOM
                
                torch.cuda.empty_cache()            
                torch.cuda.ipc_collect()