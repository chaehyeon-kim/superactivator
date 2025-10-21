import sys
import torch
import argparse

from transformers import MllamaForConditionalGeneration, AutoModel
from transformers import AutoProcessor, AutoTokenizer

from utils.embedding_utils import compute_batch_embeddings
from utils.general_utils import load_text
from utils.gt_concept_segmentation_utils import compute_attention_masks, map_sentence_to_concept_gt
from utils.default_percentthrumodels import LLAMA_TEXT_PERCENTTHRUMODELS, GEMMA_TEXT_PERCENTTHRUMODELS, QWEN_TEXT_PERCENTTHRUMODELS, get_model_default_percentthrumodels


PERCENT_THRU_MODEL = 100  # Default value, can be overridden by command line
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAMES = ['Sarcasm', 'iSarcasm', 'GoEmotions']
scratch_dir=''


# Model configurations
MODEL_CONFIGS = [
    {
        'name': 'Llama',
        'model_id': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
        'model_class': MllamaForConditionalGeneration,
        'processor_class': AutoProcessor,
        'model_input_size': ('text', 'text')
    },
    {
        'name': 'Gemma',
        'model_id': 'google/gemma-2-9b',
        'model_class': AutoModel,
        'processor_class': AutoTokenizer,
        'model_input_size': ('text', 'text2')
    },
    {
        'name': 'Qwen',
        'model_id': 'Qwen/Qwen3-Embedding-4B',
        'model_class': AutoModel,
        'processor_class': AutoTokenizer,
        'model_input_size': ('text', 'text3')
    }
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embed text datasets using various language models')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process')
    parser.add_argument('--datasets', nargs='+', help='Multiple datasets to process')
    parser.add_argument('--model', type=str, choices=['Llama', 'Gemma', 'Qwen'], help='Specific model to use')
    parser.add_argument('--models', nargs='+', choices=['Llama', 'Gemma', 'Qwen'], help='Multiple models to use')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets and exit')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing (default: 32)')
    parser.add_argument('--percentthrumodels', nargs='+', type=int,
                        help='List of percentages through model layers to use (default: model-specific values)')
    
    args = parser.parse_args()
    
    # List available datasets if requested
    if args.list_datasets:
        print("Available datasets:")
        all_datasets = ['Sarcasm', 'iSarcasm', 'GoEmotions', 'IMDB', 'StanfordTreeBank']
        for dataset in all_datasets:
            print(f"  - {dataset}")
        sys.exit(0)
    
    # List available models if requested
    if args.list_models:
        print("Available models:")
        for config in MODEL_CONFIGS:
            print(f"  - {config['name']}: {config['model_id']}")
        sys.exit(0)
    
    
    # Determine which datasets to process
    if args.dataset:
        datasets_to_process = [args.dataset]
    elif args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = DATASET_NAMES
    
    # Determine which models to process
    if args.models:
        models_to_process = [config for config in MODEL_CONFIGS if config['name'] in args.models]
        if len(models_to_process) != len(args.models):
            found_models = [m['name'] for m in models_to_process]
            missing_models = [m for m in args.models if m not in found_models]
            print(f"Error: Model(s) not found: {missing_models}")
            sys.exit(1)
    elif args.model:
        models_to_process = [config for config in MODEL_CONFIGS if config['name'] == args.model]
        if not models_to_process:
            print(f"Error: Model '{args.model}' not found")
            sys.exit(1)
    else:
        models_to_process = MODEL_CONFIGS
    
    # Loop through selected models first
    for model_config in models_to_process:
        # Get model-specific percentthrumodel values
        model_percentthrumodels = get_model_default_percentthrumodels(model_config['name'], model_config['model_input_size'])
        
        # If user specified percentthrumodels, use those instead of defaults
        if args.percentthrumodels:
            model_percentthrumodels = args.percentthrumodels
        
        print(f"\n{'='*60}")
        print(f"Processing {model_config['name']} model with percentthrumodels: {model_percentthrumodels}")
        print(f"{'='*60}\n")
        
        # Load model and processor once per model
        print(f"\nLoading {model_config['name']} model...")
        if model_config['name'] == 'Qwen':
            # Load Qwen embedding model
            MODEL = model_config['model_class'].from_pretrained(
                model_config['model_id'], 
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif model_config['name'] == 'Gemma':
            # Load Gemma model without device_map to avoid parallel style issues
            MODEL = model_config['model_class'].from_pretrained(
                model_config['model_id'], 
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            MODEL = MODEL.to(DEVICE)
        else:
            MODEL = model_config['model_class'].from_pretrained(
                model_config['model_id'], 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
        
        if model_config['name'] == 'Gemma':
            PROCESSOR = model_config['processor_class'].from_pretrained(model_config['model_id'], trust_remote_code=True)
        else:
            PROCESSOR = model_config['processor_class'].from_pretrained(model_config['model_id'])
        MODEL_NAME = model_config['name']
        
        # Loop through model-specific percentthrumodels
        for PERCENT_THRU_MODEL in model_percentthrumodels:
            print(f"\nProcessing with PERCENT_THRU_MODEL = {PERCENT_THRU_MODEL}")
            
            # Process datasets
            for DATASET_NAME in datasets_to_process:
                print(f"Dataset: {DATASET_NAME}, Model: {model_config['name']}, PercentThruModel: {PERCENT_THRU_MODEL}")
                all_text, train_text, test_text, cal_text = load_text(DATASET_NAME)
                
                # Compute list of list of tokens
                compute_attention_masks(all_text, PROCESSOR, DATASET_NAME, model_config['model_input_size'])
                
                # Compute gt
                map_sentence_to_concept_gt(DATASET_NAME, model_config['model_input_size'], one_indexed=True)
                
                # Compute both cls and patch embeddings in one pass
                print(f"Computing both CLS and patch embeddings for {MODEL_NAME}")
                compute_batch_embeddings(all_text, MODEL, PROCESSOR, DEVICE, 
                                        percent_thru_model=PERCENT_THRU_MODEL, dataset_name=DATASET_NAME, 
                                        model_input_size=model_config['model_input_size'],
                                        batch_size=args.batch_size, scratch_dir=scratch_dir, model_name=MODEL_NAME)
        
        # Clear model from memory before loading next model
        del MODEL
        del PROCESSOR
        torch.cuda.empty_cache()