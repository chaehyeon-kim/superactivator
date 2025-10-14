DATASET_TO_CONCEPTS = {
    'Coco': [
        'accessory', 'animal', 'appliance', 'bench', 'book', 'bottle', 'bowl', 'bus', 'car', 
        'chair', 'couch', 'cup', 'dining table', 'electronic', 'food', 'furniture', 'indoor', 
        'kitchen', 'motorcycle', 'outdoor', 'person', 'pizza', 'potted plant', 'sports', 
        'train', 'truck', 'tv', 'umbrella', 'vehicle'
    ],
    'Broden-Pascal': [
        'object::airplane', 'object::bicycle', 'object::bird', 'object::boat', 'object::body', 'object::book',
        'object::building', 'object::bus', 'object::cap', 'object::car', 'object::cat', 'object::cup',
        'object::dog', 'object::door', 'object::ear', 'object::engine', 'object::grass', 'object::hair',
        'object::horse', 'object::leg', 'object::mirror', 'object::motorbike', 'object::mountain',
        'object::painting', 'object::person', 'object::pottedplant', 'object::saddle', 'object::screen',
        'object::sky', 'object::sofa', 'object::table', 'object::track', 'object::train', 'object::tvmonitor',
        'object::wheel', 'object::wood',
        'part::arm', 'part::bag', 'part::beak', 'part::bottle', 'part::box', 'part::cabinet', 'part::ceiling',
        'part::chain wheel', 'part::chair', 'part::coach', 'part::curtain', 'part::eye', 'part::eyebrow',
        'part::fabric', 'part::fence', 'part::floor', 'part::foot', 'part::ground', 'part::hand',
        'part::handle bar', 'part::head', 'part::headlight', 'part::light', 'part::mouth', 'part::muzzle',
        'part::neck', 'part::nose', 'part::paw', 'part::plant', 'part::plate', 'part::plaything', 'part::pole',
        'part::pot', 'part::road', 'part::rock', 'part::rope', 'part::shelves', 'part::sidewalk',
        'part::signboard', 'part::stern', 'part::tail', 'part::torso', 'part::tree', 'part::wall',
        'part::water', 'part::windowpane', 'part::wing'
    ],
    'Broden-OpenSurfaces': [
        'material::brick', 'material::cardboard', 'material::carpet', 'material::ceramic',
        'material::concrete', 'material::fabric', 'material::food', 'material::fur', 'material::glass',
        'material::granite', 'material::hair', 'material::laminate', 'material::leather', 'material::metal',
        'material::mirror', 'material::painted', 'material::paper', 'material::plastic-clear',
        'material::plastic-opaque', 'material::rock', 'material::rubber', 'material::skin', 'material::tile',
        'material::wallpaper', 'material::wicker', 'material::wood'
    ],
    'CLEVR': [
        'color::blue', 'color::green', 'color::red',
        'shape::cube', 'shape::cylinder', 'shape::sphere'
    ],
    'Sarcasm': ['sarcasm'],
    # 'iSarcasm': ['sarcasm', 'sarcastic'],
    'iSarcasm': ['sarcastic'],
    'GoEmotions': ['confusion', 'joy', 'sadness', 'anger', 'love', 'caring', 'optimism', 'amusement', 'curiosity', 'disapproval', 'approval', 'annoyance', 'gratitude', 'admiration']
}

def filter_concept_dict(concept_dict, dataset_name):
    """
    Filters a concept dictionary so that only keys corresponding to the given dataset's concepts are retained.

    Args:
        concept_dict (dict): Dictionary with concept names as keys.
        dataset_name (str): Name of the dataset (must be a key in concept_map).

    Returns:
        dict: Filtered concept dictionary.
    """
    allowed_concepts = set(DATASET_TO_CONCEPTS.get(dataset_name, []))
    if len(allowed_concepts) == 0:
        return concept_dict
    return {k: v for k, v in concept_dict.items() if k in allowed_concepts}

def filter_concept_list(concept_list, dataset_name):
    """
    Filters a concept dictionary so that only keys corresponding to the given dataset's concepts are retained.

    Args:
        concept_dict (dict): Dictionary with concept names as keys.
        dataset_name (str): Name of the dataset (must be a key in concept_map).

    Returns:
        dict: Filtered concept dictionary.
    """
    allowed_concepts = set(DATASET_TO_CONCEPTS.get(dataset_name, []))
    if len(allowed_concepts) == 0:
        return concept_dict
    return [c for c in concept_list if c in allowed_concepts]


