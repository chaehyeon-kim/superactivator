import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM

from utils import data_path, repo_path

# from utils.datasets import ImageDataset  # Not used - using FixedImageDataset instead
from utils.inversion_methods import prompt_inversion
from utils.prompt_concepts import LLMNet, RawInput


class FixedImageDataset:
    """Fixed version of ImageDataset that works with absolute paths"""
    def __init__(self, root, dataset_name, split="test", transform=None):
        self.root = root
        self.dataset_name = dataset_name
        self.transform = transform
        self.split = split

        # Load metadata to get concept information
        self.metadata = pd.read_csv(data_path(dataset_name, "metadata.csv"))

        # Select images based on the split
        if split == "train":
            self.metadata = self.metadata[self.metadata["split"] == "train"].reset_index(drop=True)
        elif split == "test":
            self.metadata = self.metadata[self.metadata["split"] == "test"].reset_index(drop=True)

        # Define specific concepts for each dataset
        if dataset_name == 'Coco':
            target_concepts = ['bed', 'surfboard', 'bicycle', 'spoon', 'pizza', 'fork', 'train', 'motorcycle', 'tennis racket', 'sports ball', 'potted plant', 'umbrella', 'dog', 'knife', 'laptop', 'cat', 'sink', 'bus', 'traffic light', 'couch', 'clock', 'tv', 'cell phone', 'backpack', 'book', 'bench', 'truck', 'handbag', 'bowl', 'appliance', 'bottle', 'cup', 'dining table', 'car', 'outdoor', 'chair', 'electronic', 'indoor', 'food', 'accessory', 'kitchen', 'sports', 'animal', 'vehicle', 'furniture', 'person']
        elif dataset_name == 'Broden-Pascal':
            target_concepts = ['color::black-c', 'color::blue-c', 'color::brown-c', 'color::green-c', 'color::grey-c', 'color::orange-c', 'color::pink-c', 'color::purple-c', 'color::red-c', 'color::white-c', 'color::yellow-c', 'object::airplane', 'object::bicycle', 'object::bird', 'object::boat', 'object::body', 'object::book', 'object::building', 'object::bus', 'object::cap', 'object::car', 'object::cat', 'object::cup', 'object::dog', 'object::door', 'object::ear', 'object::engine', 'object::grass', 'object::hair', 'object::horse', 'object::leg', 'object::mirror', 'object::motorbike', 'object::mountain', 'object::painting', 'object::person', 'object::pottedplant', 'object::saddle', 'object::screen', 'object::sky', 'object::sofa', 'object::table', 'object::track', 'object::train', 'object::tvmonitor', 'object::wheel', 'object::wood', 'part::arm', 'part::bag', 'part::beak', 'part::bottle', 'part::box', 'part::cabinet', 'part::ceiling', 'part::chain wheel', 'part::chair', 'part::coach', 'part::curtain', 'part::eye', 'part::eyebrow', 'part::fabric', 'part::fence', 'part::floor', 'part::foot', 'part::ground', 'part::hand', 'part::handle bar', 'part::head', 'part::headlight', 'part::light', 'part::mouth', 'part::muzzle', 'part::neck', 'part::nose', 'part::paw', 'part::plant', 'part::plate', 'part::plaything', 'part::pole', 'part::pot', 'part::road', 'part::rock', 'part::rope', 'part::shelves', 'part::sidewalk', 'part::signboard', 'part::stern', 'part::tail', 'part::torso', 'part::tree', 'part::wall', 'part::water', 'part::windowpane', 'part::wing']
        elif dataset_name == 'Broden-OpenSurfaces':
            target_concepts = ['color::black-c', 'color::blue-c', 'color::brown-c', 'color::green-c', 'color::grey-c', 'color::orange-c', 'color::pink-c', 'color::purple-c', 'color::red-c', 'color::white-c', 'color::yellow-c', 'material::brick', 'material::cardboard', 'material::carpet', 'material::ceramic', 'material::concrete', 'material::fabric', 'material::food', 'material::fur', 'material::glass', 'material::granite', 'material::hair', 'material::laminate', 'material::leather', 'material::metal', 'material::mirror', 'material::painted', 'material::paper', 'material::plastic-clear', 'material::plastic-opaque', 'material::rock', 'material::rubber', 'material::skin', 'material::tile', 'material::wallpaper', 'material::wicker', 'material::wood']
        elif dataset_name == 'CLEVR':
            target_concepts = ['color::blue', 'color::green', 'color::red', 'shape::cube', 'shape::cylinder', 'shape::sphere']
        elif dataset_name == 'Sarcasm':
            target_concepts = ['sarcasm']
        elif dataset_name == 'iSarcasm':
            target_concepts = ['sarcasm', 'irony', 'satire', 'understatement', 'overstatement', 'rhetorical_question', 'sarcastic']
        elif dataset_name == 'GoEmotions':
            target_concepts = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'disappointment', 'disapproval', 'disgust', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'realization', 'sadness', 'surprise']
        else:
            # Fallback to all concepts if dataset not recognized
            target_concepts = [col for col in self.metadata.columns if col not in ["image_path", "split", "class", "text_path"]]
        
        # Filter to only target concepts that exist in metadata
        self.concept_columns = [col for col in target_concepts if col in self.metadata.columns]

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Get an image/text and its metadata by index.
        """
        # Check if this is a text dataset
        if 'text_path' in self.metadata.columns:
            # Load text
            text_path = data_path(self.dataset_name, self.metadata.iloc[idx]['text_path'])
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            data = text
        else:
            # Load image
            from PIL import Image
            image_path = data_path(self.dataset_name, self.metadata.iloc[idx]['image_path'])
            image = Image.open(image_path).convert("RGB")
            
            # Apply transforms if available
            if self.transform:
                image = self.transform(image)
            data = image

        # Get concepts as a tensor
        concepts = (self.metadata.loc[idx, self.concept_columns].values,)
        
        return data, concepts

    def get_concept_names(self):
        """Return list of concept names"""
        return self.concept_columns


def concept_inversion(args):
    # Create prompt_results directory if it doesn't exist
    prompt_results_dir = os.path.join(args.output_dir, "prompt_results")
    os.makedirs(prompt_results_dir, exist_ok=True)
    
    # load model
    model = LLM(
        model=args.model,
        max_model_len=12288,
        limit_mm_per_prompt={"image": 10},
        max_num_seqs=1,
        enforce_eager=True if "llama" in args.model.lower() else False,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )

    # load dataset
    data = FixedImageDataset(
        root=repo_path(), dataset_name=args.dataset, split="test"
    )
    concept_names = data.get_concept_names()

    if "class" in concept_names:
        concept_names.remove("class")

    # load detected concepts
    concepts_file = os.path.join(prompt_results_dir, f"{args.dataset}_{args.model.split('/')[1]}_concepts.txt")
    with open(concepts_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        detected_concepts = [
            list(map(lambda x: 1 if "yes" in x.lower() else 0, row[1:])) for row in reader
        ]
    detected_concepts = np.array(detected_concepts)

    inversion_results = []
    for i in tqdm(range(len(data))):
        image, _ = data[i]

        # get concept inversion for only present concepts
        concept_inversion = {}
        for j, concept in enumerate(concept_names):
            if detected_concepts[i][j] == 1:
                inversion = prompt_inversion(model, concept, image)
                print("Inversion:", inversion)
                concept_inversion[concept] = inversion
        inversion_results.append(concept_inversion)

    # Save inversion results to a file
    output_file = f"{prompt_results_dir}/{args.dataset}_{args.model.split('/')[1]}_inversion.txt"
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Index"] + concept_names)
        for idx, concepts in enumerate(inversion_results):
            row = [idx]
            for concept in concept_names:
                if concept in concepts:
                    row.append(concepts[concept])
                else:
                    row.append("No inversion")
            writer.writerow(row)


def main(args):
    # Create prompt_results directory if it doesn't exist
    prompt_results_dir = os.path.join(args.output_dir, "prompt_results", args.dataset)
    os.makedirs(prompt_results_dir, exist_ok=True)
    
    # load model
    model = LLM(
        model=args.model,
        max_model_len=12288,
        limit_mm_per_prompt={"image": 10},
        max_num_seqs=1,
        enforce_eager=True if "llama" in args.model.lower() else False,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )

    # load dataset
    data = FixedImageDataset(
        root=repo_path(), dataset_name=args.dataset, split="test"
    )
    concept_names = data.get_concept_names()

    if "class" in concept_names:
        concept_names.remove("class")

    concept_extractors = []
    print("Concepts:", concept_names)
    
    # Check if this is a text dataset
    is_text_dataset = 'text_path' in data.metadata.columns
    
    for concept in concept_names:
        if is_text_dataset:
            extractor = LLMNet(
                model,
                input_desc=f"a text passage which may contain concepts from the list {concept_names}",
                output_desc=f"the word 'Yes' if the text contains {concept}, otherwise 'No'",
                image_before_prompt=False,
            )
        else:
            extractor = LLMNet(
                model,
                input_desc=f"an image which may contain concepts from the list {concept_names}",
                output_desc=f"the word 'Yes' if the image contains {concept}, otherwise 'No'",
                image_before_prompt=True,
            )
        concept_extractors.append(extractor)

    extracted_concepts = []
    inversion_results = []
    for i in tqdm(range(len(data))):
        data_item, _ = data[i]

        # extract concepts
        concept_outputs = []
        concept_inversion = {}
        for j, extractor in enumerate(concept_extractors):
            if is_text_dataset:
                output = extractor.forward(RawInput(image_input=None, text_input=data_item))
            else:
                output = extractor.forward(RawInput(image_input=data_item, text_input=None))
            
            if "Yes" in output:
                output = 1
            else:
                output = 0
            concept_outputs.append(output)

            # get inversion if concept present
            if output == 1:
                if is_text_dataset:
                    inversion = prompt_inversion(model, concept_names[j], data_item, is_text=True)
                else:
                    inversion = prompt_inversion(model, concept_names[j], data_item)
                print("Inversion:", inversion)
                concept_inversion[concept_names[j]] = inversion
        inversion_results.append(concept_inversion)

        print("Extracted:", concept_outputs)
        extracted_concepts.append(concept_outputs)

    # Save extracted concepts to a file
    output_file = f"{prompt_results_dir}/{args.dataset}_{args.model.split('/')[1]}_concepts.txt"
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Index"] + concept_names)
        for idx, concepts in enumerate(extracted_concepts):
            writer.writerow([idx] + concepts)

    # Save inversion results to a file
    output_file = f"{prompt_results_dir}/{args.dataset}_{args.model.split('/')[1]}_inversion.txt"
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Index"] + concept_names)
        for idx, concepts in enumerate(inversion_results):
            row = [idx]
            for concept in concept_names:
                if concept in concepts:
                    row.append(concepts[concept])
                else:
                    row.append("No inversion")
            writer.writerow(row)


def eval(args):
    # Create prompt_results directory if it doesn't exist
    prompt_results_dir = os.path.join(args.output_dir, "prompt_results", args.dataset)
    os.makedirs(prompt_results_dir, exist_ok=True)
    
    # load dataset
    data = FixedImageDataset(
        root=repo_path(), dataset_name=args.dataset, split="test"
    )
    concept_names = data.get_concept_names()

    # get gt
    gt_labels = []
    for i in range(len(data)):
        _, label = data[i]
        label = label[0]
        gt_labels.append([int(label[j]) for j in range(len(label)) if concept_names[j] != "class"])
    gt_labels_array = np.array(gt_labels)

    print("GT labels size:", gt_labels_array.shape)

    if "class" in concept_names:
        concept_names.remove("class")

    # Load extracted concepts from the file
    output_file = f"{prompt_results_dir}/{args.dataset}_{args.model.split('/')[1]}_concepts.txt"
    with open(output_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        extracted_concepts = [list(map(int, row[1:])) for row in reader]

    extracted_concepts_array = np.array(extracted_concepts)

    # Evaluate the extracted concepts
    f1_results = []
    for idx, concepts in enumerate(concept_names):
        gt = gt_labels_array[:, idx]
        pred = extracted_concepts_array[:, idx]
        
        # Calculate confusion matrix components
        tp = np.sum((gt == 1) & (pred == 1))
        fp = np.sum((gt == 0) & (pred == 1))
        fn = np.sum((gt == 1) & (pred == 0))
        tn = np.sum((gt == 0) & (pred == 0))
        
        # Calculate rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
        
        # Calculate performance metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr  # Same as TPR
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        print(f"F1 for {concepts}: {f1_score:.4f}")
        f1_results.append({
            'concept': concepts,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'tn': tn,
            'fpr': fpr,
            'tpr': tpr,
            'fnr': fnr,
            'tnr': tnr,
            'f1': f1_score,
            'accuracy': accuracy
        })

    # Save F1 results to CSV file with specified column order
    f1_results_file = f"{prompt_results_dir}/{args.dataset}_{args.model.split('/')[1]}_f1_scores.csv"
    f1_df = pd.DataFrame(f1_results)
    
    # Reorder columns to match your specification: concept, fp, fn, tp, tn, fpr, tpr, fnr, tnr, f1, accuracy
    column_order = ['concept', 'fp', 'fn', 'tp', 'tn', 'fpr', 'tpr', 'fnr', 'tnr', 'f1', 'accuracy']
    f1_df = f1_df[column_order]
    f1_df.to_csv(f1_results_file, index=False)
    print(f"F1 scores saved to: {f1_results_file}")
    
    # Calculate and save average F1 score
    avg_f1 = f1_df['f1'].mean()
    print(f"Average F1 Score: {avg_f1:.4f}")
    
    # Save summary statistics
    summary_file = f"{prompt_results_dir}/{args.dataset}_{args.model.split('/')[1]}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Prompt-based Concept Extraction Results\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Number of images: {len(extracted_concepts_array)}\n")
        f.write(f"Number of concepts: {len(concept_names)}\n")
        f.write(f"Average F1 Score: {avg_f1:.4f}\n\n")
        f.write("Per-concept results:\n")
        for result in f1_results:
            f.write(f"{result['concept']:20s} - F1: {result['f1']:.4f} (Acc: {result['accuracy']:.4f})\n")
    print(f"Summary saved to: {summary_file}")

    # create a bar plot of the F1 scores sorted by F1 score
    f1_scores = f1_df['f1'].values
    f1_scores = np.array(f1_scores)
    sorted_indices = np.argsort(f1_scores)[::-1]
    sorted_f1_scores = f1_scores[sorted_indices]
    sorted_concept_names = np.array(concept_names)[sorted_indices]

    plt.figure(figsize=(10, 15))
    plt.barh(sorted_concept_names, sorted_f1_scores)
    plt.xlabel("F1 Score")
    plt.title("F1 Scores for Extracted Concepts")

    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # add a line at 0.5
    plt.axvline(x=0.5, color="r", linestyle="--", label="Threshold")

    # save the plot
    plot_file = f"{prompt_results_dir}/{args.dataset}_{args.model.split('/')[1]}_f1_scores.png"
    plt.savefig(plot_file)
    print(f"F1 scores plot saved to: {plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use prompting to extract concepts from a dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to extract concepts from. Options: [CLEVR, Coco].",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2-11",
        help="The model to use for extraction. Options: [llama3.2-11].",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(repo_path("Experiments")),
        help="The output directory for extracted concepts.",
    )
    parser.add_argument("--eval", action="store_true", help="Whether to evaluate the concepts.")
    parser.add_argument("--inversion", action="store_true", help="Whether to perform inversion.")
    args = parser.parse_args()

    if args.model == "llama3.2-11":
        args.model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    elif args.model == "qwen2.5-vl-3":
        args.model = "Qwen/Qwen2.5-VL-3B-Instruct"

    if args.eval:
        eval(args)
    elif args.inversion:
        concept_inversion(args)
    else:
        main(args)