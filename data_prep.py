import os
from torch.utils.data import Dataset
from PIL import Image

def prepare_dataset(dataset):
    """
    Prepares the PlacePulse 2.0 dataset for model training. Transforms every input into two separate inputs with either a positive or negative label.

    Args:
        dataset: The dataset to be prepared. Includes three things:
            - 'selected': Either 'left', 'right', or 'equal' for each image pair.
            - 'left_image': The left image in the pair (filepath).
            - 'right_image': The right image in the pair (filepath).

    Returns:
        A list of dicts ready for training. Each dict includes:
            - 'weight': The weight of the image, positive or negative.
            - 'image': The image filepath to be used for training.
    """
    training_dataset = []
    for obj in dataset:
        if obj['selected'] == 'left':
            training_dataset.append({'weight': 1, 'image': obj['left_image']})
            training_dataset.append({'weight': -1, 'image': obj['right_image']})
        elif obj['selected'] == 'right':
            training_dataset.append({'weight': 1, 'image': obj['right_image']})
            training_dataset.append({'weight': -1, 'image': obj['left_image']})
        elif obj['selected'] == 'equal':
            training_dataset.append({'weight': 0.5, 'image': obj['left_image']})
            training_dataset.append({'weight': 0.5, 'image': obj['right_image']})
    return training_dataset

class PlacePulseDataset(Dataset):
    """
    PyTorch Dataset for PlacePulse 2.0, expects a list of dicts with 'image' and 'weight'.
    """
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = item['weight']
        return {"pixel_values": image, "labels": label}

def load_placepulse_json(json_path):
    """
    Loads PlacePulse 2.0 data from a JSON file.
    Each entry should have 'selected', 'left_image', 'right_image'.
    """
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def prepare_placepulse_dataset(json_path, transform=None):
    """
    Loads PlacePulse 2.0 data from a JSON file and prepares a PyTorch Dataset.
    """
    raw_data = load_placepulse_json(json_path)
    converted_data = prepare_dataset(raw_data)
    return PlacePulseDataset(converted_data, transform=transform)