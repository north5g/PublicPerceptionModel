import os
from torch.utils.data import Dataset
from PIL import Image

labels = {
    'safe': ['safe', 'unsafe', 'neutral'],
    'lively': ['lively', 'quiet', 'neutral'],
    'clean': ['clean', 'dirty', 'neutral'],
    'wealthy': ['wealthy', 'poor', 'neutral'],
    'depressing': ['depressing', 'pleasant', 'neutral'],
    'beautiful': ['beautiful', 'ugly', 'neutral']
}

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
        study = obj.get('study')
        if study not in labels:
            continue  # skip unknown study types
        if obj['selected'] == 'left':
            training_dataset.append({'image': obj['left_image'], 'label': labels[study][0]})
            training_dataset.append({'image': obj['right_image'], 'label': labels[study][1]})
        elif obj['selected'] == 'right':
            training_dataset.append({'image': obj['right_image'], 'label': labels[study][0]})
            training_dataset.append({'image': obj['left_image'], 'label': labels[study][1]})
        elif obj['selected'] == 'equal':
            training_dataset.append({'image': obj['left_image'], 'label': labels[study][2]})
            training_dataset.append({'image': obj['right_image'], 'label': labels[study][2]})
    return training_dataset

class PlacePulseDataset(Dataset):
    """
    PyTorch Dataset for PlacePulse 2.0, expects a list of dicts with 'image' and 'label'.
    Converts label strings to text embeddings.
    """
    def __init__(self, data, tokenizer, text_encoder, transform=None, device="gpu"):
        self.data = data
        self.transform = transform
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        # Precompute label embeddings for efficiency
        self.label_embeddings = {}
        for item in data:
            label = item['label']
            if label not in self.label_embeddings:
                inputs = self.tokenizer([label], return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    emb = self.text_encoder(**inputs).last_hidden_state.mean(dim=1)
                self.label_embeddings[label] = emb.squeeze(0).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item['image']).convert('RGB')
        except Exception as e:
            # If image is missing/corrupt, return a black image
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        label_embedding = self.label_embeddings[item['label']]
        return {"pixel_values": image, "labels": label_embedding}

def load_placepulse_json(json_path):
    """
    Loads PlacePulse 2.0 data from a JSON file.
    Each entry should have 'selected', 'left_image', 'right_image'.
    """
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def prepare_placepulse_dataset(json_path, tokenizer, text_encoder, transform=None, device="cpu"):
    """
    Loads PlacePulse 2.0 data from a JSON file and prepares a PyTorch Dataset.
    """
    raw_data = load_placepulse_json(json_path)
    converted_data = prepare_dataset(raw_data)
    return PlacePulseDataset(converted_data, tokenizer=tokenizer, text_encoder=text_encoder, transform=transform, device=device)