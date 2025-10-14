"""
Default percentthrumodel values for each model type.
These values correspond to every 5 layers, always including the final layer.
"""

# Vision models
CLIP_PERCENTTHRUMODELS = [4, 25, 46, 67, 88, 100]
LLAMA_VISION_PERCENTTHRUMODELS = [2, 15, 28, 40, 52, 65, 78, 90, 100]

# Text models
LLAMA_TEXT_PERCENTTHRUMODELS = [3, 19, 34, 50, 66, 81, 97, 100]
GEMMA_TEXT_PERCENTTHRUMODELS = [4, 21, 39, 57, 75, 93, 100]
QWEN_TEXT_PERCENTTHRUMODELS = [3, 19, 34, 50, 66, 81, 97, 100]

# For scripts that process all models, combine all unique values
ALL_PERCENTTHRUMODELS = sorted(list(set(
    CLIP_PERCENTTHRUMODELS + 
    LLAMA_VISION_PERCENTTHRUMODELS + 
    LLAMA_TEXT_PERCENTTHRUMODELS + 
    GEMMA_TEXT_PERCENTTHRUMODELS + 
    QWEN_TEXT_PERCENTTHRUMODELS
)))

def get_model_default_percentthrumodels(model_name, input_size):
    """
    Get the default percentthrumodel values for a specific model.
    
    Args:
        model_name: Name of the model (e.g., 'CLIP', 'Llama', 'Gemma', 'Qwen')
        input_size: Input size tuple (e.g., (224, 224) for vision, ('text', 'text') for text)
    
    Returns:
        List of percentthrumodel values
    """
    if model_name == 'CLIP':
        return CLIP_PERCENTTHRUMODELS
    elif model_name == 'Llama' or model_name == 'Llama-Vision':
        if isinstance(input_size, tuple) and input_size[0] == 'text':
            return LLAMA_TEXT_PERCENTTHRUMODELS
        else:
            return LLAMA_VISION_PERCENTTHRUMODELS
    elif model_name == 'Llama-Text':
        return LLAMA_TEXT_PERCENTTHRUMODELS
    elif model_name == 'Gemma':
        return GEMMA_TEXT_PERCENTTHRUMODELS
    elif model_name == 'Qwen':
        return QWEN_TEXT_PERCENTTHRUMODELS
    else:
        # Default to all values if model not recognized
        return ALL_PERCENTTHRUMODELS