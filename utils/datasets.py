"""Dataset classes for loading and processing images."""

import pandas as pd
from torch.utils.data import Dataset

from .general_utils import load_images, retrieve_present_concepts


class ImageDataset(Dataset):
    """
    A PyTorch Dataset that uses the load_images function to load images.

    This dataset loads all images at initialization time and keeps them in memory.
    """

    def __init__(self, root=".", dataset_name="CLEVR", transform=None, split=None):
        """
        Initialize the ImageDataset.

        Args:
            root (str): Root directory where data is stored.
            dataset_name (str): Name of the dataset to load.
            transform (callable, optional): Optional transform to be applied to images.
            split (str, optional): If provided, only use 'train' or 'test' images.
        """
        self.root = root
        self.dataset_name = dataset_name
        self.transform = transform
        self.split = split

        # Load all images using the load_images function
        all_images, train_images, test_images = load_images(root=root, dataset_name=dataset_name)

        # Load metadata to get concept information
        self.metadata = pd.read_csv(f"{root}/Data/{dataset_name}/metadata.csv")

        # Select images based on the split
        if split == "train":
            self.images = train_images
            self.metadata = self.metadata[self.metadata["split"] == "train"].reset_index(drop=True)
        elif split == "test":
            self.images = test_images
            self.metadata = self.metadata[self.metadata["split"] == "test"].reset_index(drop=True)
        else:
            self.images = all_images

        # Get concept columns (excluding non-concept columns)
        self.concept_columns = [
            col for col in self.metadata.columns if col not in ["image_path", "split"]
        ]

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get an image and its metadata by index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: (image, concepts) where image is a tensor and concepts is a tensor
                  of binary values indicating presence of concepts.
        """
        image = self.images[idx]

        # Apply transforms if available
        if self.transform:
            image = self.transform(image)

        # Get concepts as a tensor
        concepts = (self.metadata.loc[idx, self.concept_columns].values,)

        return image, concepts

    def get_concept_names(self):
        """Return the list of concept names in the dataset."""
        return self.concept_columns

    def get_present_concepts(self, idx):
        """
        Get a list of concepts present in the image at the given index.

        Args:
            idx (int): Index of the image.

        Returns:
            list: Names of concepts present in the image.
        """
        return retrieve_present_concepts(idx, self.dataset_name, self.root)


def main():
    root = "/workspace/"
    dataset = ImageDataset(root=root, split="train")

    print(f"Number of images in dataset: {len(dataset)}")
    print(f"Concept names: {dataset.get_concept_names()}")

    # Test getting an item
    image, concepts = dataset[0]
    print(f"First image shape: {image.size}")
    print(f"First image concepts: {concepts}")
    print(f"Present concepts: {dataset.get_present_concepts(0)}")

    print()

    root = "/workspace/"
    dataset = ImageDataset(root=root, dataset_name="Coco", split="train")

    print(f"Number of images in dataset: {len(dataset)}")
    print(f"Concept names: {dataset.get_concept_names()}")

    # Test getting an item
    image, concepts = dataset[0]
    print(f"First image shape: {image.size}")
    print(f"First image concepts: {concepts}")
    print(f"Present concepts: {dataset.get_present_concepts(0)}")


if __name__ == "__main__":
    main()