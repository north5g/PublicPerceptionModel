import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms
import pandas as pd
from torchvision.transforms.functional import crop

import os
from os import listdir
from os.path import isfile, join

import zipfile
from tqdm import tqdm
import requests
import shutil

from sklearn.model_selection import train_test_split

labels = {
    'safe': ['safe', 'unsafe', 'neutral'],
    'lively': ['lively', 'quiet', 'neutral'],
    'clean': ['clean', 'dirty', 'neutral'],
    'wealthy': ['wealthy', 'poor', 'neutral'],
    'depressing': ['depressing', 'pleasant', 'neutral'],
    'beautiful': ['beautiful', 'ugly', 'neutral']
}

def crop_google_logo(img):
    return crop(img, 0, 0, img.size[1]- 25, img.size[0])

transform_cnn = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(crop_google_logo),
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
])

transform_transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(crop_google_logo),
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

class PlacePulseDatasetText(Dataset):
    def __init__(self, dataframe=None, votes_tsv_path=None,
                 transform=None, img_dir='place-pulse-2.0/images/',
                 return_location_id=False, study_id=None, study_type=None,
                 transform_only_image=True, split=None,
                 tokenizer=None, text_encoder=None, device="cpu"):

        if votes_tsv_path and dataframe is not None:
            raise ValueError("Please provide either 'votes_tsv_path' or 'dataframe', not both.")

        self.transform = transform
        self.dataset_folder_path = img_dir
        self.return_location_id = return_location_id
        self.transform_only_image = transform_only_image
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.study_types = {
            'safe': '50a68a51fdc9f05596000002',
            'lively': '50f62c41a84ea7c5fdd2e454',
            'clean': '50f62c68a84ea7c5fdd2e456',
            'wealthy': '50f62cb7a84ea7c5fdd2e458',
            'depressing': '50f62ccfa84ea7c5fdd2e459',
            'beautiful': '5217c351ad93a7d3e7b07a64'
        }
        self.study_ids_to_type = {v: k for k, v in self.study_types.items()}

        # Load and process data
        if votes_tsv_path:
            votes_df = pd.read_csv(votes_tsv_path, sep='\t')
            if study_type and study_type != "all":
                study_id = self.study_types[study_type]
                votes_df = votes_df[votes_df['study_id'] == study_id]
            # Always add study_type column
            votes_df['study_type'] = votes_df['study_id'].map(self.study_ids_to_type)
            self.dataframe = votes_to_single_image_labels(votes_df)
        elif dataframe is not None:
            self.dataframe = dataframe
            # If study_type is "all" or not provided, ensure study_type column exists
            if 'study_type' not in self.dataframe.columns and 'study_id' in self.dataframe.columns:
                self.dataframe['study_type'] = self.dataframe['study_id'].map(self.study_ids_to_type)
        else:
            raise ValueError("Must provide either votes_tsv_path or dataframe.")

        # Filter by study_type if not "all"
        if study_type and study_type != "all":
            self.dataframe = self.dataframe[self.dataframe['study_type'] == study_type]

        # Optionally filter by study_id (if provided)
        if study_id:
            self.dataframe = self.dataframe[self.dataframe['study_id'] == study_id]

        # Split if requested
        if split:
            train_df, val_df = train_test_split(self.dataframe, test_size=0.4, random_state=42, stratify=self.dataframe['study_type'])
            if split == 'train':
                self.dataframe = train_df
            elif split == 'val':
                self.dataframe = val_df

        # Precompute label embeddings for all unique labels
        self.label_embeddings = {}
        if self.tokenizer is not None and self.text_encoder is not None:
            unique_labels = self.dataframe['label'].unique()
            for label in unique_labels:
                inputs = self.tokenizer([label], return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    emb = self.text_encoder(**inputs).last_hidden_state.mean(dim=1)
                self.label_embeddings[label] = emb.squeeze(0).to(self.device)

    def __len__(self) -> int:
        return len(self.dataframe)


    def __getitem__(self, idx):
        location_id = self.dataframe.iloc[idx]['location_id']
        img = self.get_img_by_location_id(location_id)
        label = self.dataframe.iloc[idx]['label']

        if self.transform:
            img = self.transform(img)

        # Convert label string to embedding tensor
        if self.tokenizer is not None and self.text_encoder is not None:
            label_emb = self.label_embeddings[label]
        else:
            label_emb = label  # fallback for debugging

        return {"pixel_values": img, "labels": label_emb}

    def get_img_by_location_id(self, location_id):
        extension = '.jpg'
        img_name = f"{location_id}{extension}"
        img = io.imread(f'{self.dataset_folder_path}{img_name}')

        if self.transform and self.transform_only_image:
            img = self.transform(img)

        return img

    def get_sample_by_location_id(self, location_id):
        img = self.get_img_by_location_id(location_id)
        row = self.dataframe[self.dataframe['location_id'] == location_id]
        rating = row['trueskill.score'].values[0]

        if self.return_location_id:
            return img, rating, location_id

        return img, rating

    @staticmethod
    def get_q_score_only_for_files_in_folder(q_scores: pd.DataFrame, folder_path):
        """
        Filters the given 'q_scores' dataframe to only include the files that are present in the specified folder.

        Args:
            q_scores (pd.DataFrame): The dataframe containing the q scores.
            folder_path (str): The path to the folder containing the files.

        Returns:
            pd.DataFrame: The filtered dataframe containing the q scores for the files in the folder.
        """
        file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        location_ids_from_existing_files = [os.path.splitext(file_name)[0] for file_name in file_names]
        q_scores_clean = q_scores[q_scores['location_id'].isin(location_ids_from_existing_files)]

        return q_scores_clean

    @staticmethod
    def clean_qscores():
        qscores_tsv_path = 'place-pulse-2.0/qscores.tsv'
        qscores_df = pd.read_csv(qscores_tsv_path, sep='\t')
        qscores_clean = PlacePulseDatasetText.get_q_score_only_for_files_in_folder(qscores_df, 'place-pulse-2.0/images/')
        qscores_clean.to_csv(qscores_tsv_path, sep='\t', index=False)

    @staticmethod
    def download_archive():
        url = "https://www.dropbox.com/s/grzoiwsaeqrmc1l/place-pulse-2.0.zip?dl=1"
        response = requests.get(url, stream=True)

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, desc='Downloading archive', unit='iB', unit_scale=True)

        with open("place-pulse-2.0.zip", "wb") as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong")

    @staticmethod
    def extract_archive(zip_file_path='place-pulse-2.0.zip', destination_folder='data') -> None:
        """
        Extracts the specified zip file to the destination folder.

        Args:
            zip_file_path (str, optional): The path to the zip file. Defaults to 'place-pulse-2.0.zip'.
            destination_folder (str, optional): The path to the destination folder. Defaults to 'data'.
        """
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_list = [f for f in zip_ref.namelist() if not f.startswith('__MACOSX/')]

            with tqdm(total=len(file_list), desc='Extracting files', unit='file') as pbar:
                for file in file_list:
                    zip_ref.extract(file, destination_folder)

                    pbar.update()

    @staticmethod
    def preprocess() -> None:
        """
        Preprocesses the images by copying them from the source directory to the destination directory,
        renaming them with a unique file ID, and then removing the original images folder.
        """
        source_dir = 'place-pulse-2.0/images/'
        file_names = os.listdir(source_dir)

        destination_dir = 'place-pulse-2.0/images_preprocessed/'
        os.makedirs(destination_dir, exist_ok=True)

        for file_name in tqdm(file_names, desc='Preprocessing images', unit='file'):
            # This id seems to be unique. Checked for duplicates with:
            # find . -type f -exec basename {} \; | sort | uniq -D
            unique_file_id = file_name.split("_")[2]
            new_file_name = f'{unique_file_id}.jpg'

            file_path = os.path.join(source_dir, file_name)
            destination_path = os.path.join(destination_dir, new_file_name)

            shutil.copyfile(file_path, destination_path)

        print('Cleaning up.')
        shutil.rmtree(source_dir)
        os.rename(destination_dir, source_dir)

        print('Removing samples where image is missing.')
        PlacePulseDatasetText.clean_qscores()

    @staticmethod
    def load() -> None:
        """
        Loads the dataset by downloading, extracting, preprocessing, and deleting the archive.
        """
        if os.path.exists('place-pulse-2.0'):
            print('Error: The "place-pulse-2.0" folder already exists.')
            return

        PlacePulseDatasetText.download_archive()
        zip_file_path='place-pulse-2.0.zip'
        PlacePulseDatasetText.extract_archive(zip_file_path=zip_file_path)
        print('Deleting archive.')
        os.remove(zip_file_path)
        PlacePulseDatasetText.preprocess()

def split_dataset_to_train_test(dataframe=None, qscores_tsv_path='place-pulse-2.0/qscores.tsv',
                                train_size=1000, random_state=42, output_dir='place-pulse-2.0/'):
        """
        Splits the PlacePulse dataset into training and test sets, and saves them as TSV files.

        Args:
            dataframe (pd.DataFrame, optional): The DataFrame containing the dataset.
                If None, loads from qscores_tsv_path. Defaults to None.
            qscores_tsv_path (str, optional): Path to the TSV file if dataframe is not provided.
                Defaults to 'data/qscores.tsv'.
            train_size (int, optional): Number of samples in the training set. Defaults to 1000.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            output_dir (str, optional): Directory to save the output files. Defaults to 'data/'.

        Returns:
            tuple: (train_df, test_df) DataFrames for training and test sets.
        """
        # Load the data if no DataFrame is provided
        if dataframe is None:
            dataframe = pd.read_csv(qscores_tsv_path, sep='\t')

        # Split the data
        train_df, test_df = train_test_split(
            dataframe,
            train_size=train_size,
            random_state=random_state,
            shuffle=True
        )

        # Save to TSV files
        train_path = f"{output_dir}train_data.tsv"
        test_path = f"{output_dir}test_data.tsv"

        train_df.to_csv(train_path, sep='\t', index=False)
        test_df.to_csv(test_path, sep='\t', index=False)

        print(f"Saved training data ({len(train_df)} samples) to {train_path}")
        print(f"Saved test data ({len(test_df)} samples) to {test_path}")

        return train_df, test_df

def votes_to_single_image_labels(votes_df):
    """
    Converts a votes DataFrame (from votes.tsv) into a DataFrame of single-image, single-label samples.
    Each vote creates two samples: one for each image, with the appropriate label.
    """
    rows = []
    for _, row in votes_df.iterrows():
        study = row['study_type'] if 'study_type' in row else row['study']
        choice = row['choice']
        left = row['left']
        right = row['right']
        # Use your labels mapping
        if study not in labels:
            continue
        if choice == 'left':
            rows.append({'location_id': left, 'label': labels[study][0], 'study_type': study})
            rows.append({'location_id': right, 'label': labels[study][1], 'study_type': study})
        elif choice == 'right':
            rows.append({'location_id': right, 'label': labels[study][0], 'study_type': study})
            rows.append({'location_id': left, 'label': labels[study][1], 'study_type': study})
        elif choice == 'equal':
            rows.append({'location_id': left, 'label': labels[study][2], 'study_type': study})
            rows.append({'location_id': right, 'label': labels[study][2], 'study_type': study})
    return pd.DataFrame(rows)

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_train_test_split(dataset, output_dir='place-pulse-2.0/splits', train_size=1000, random_state=42):
    """
    Performs stratified sampling on the PlacePulseDataset to create balanced train/test splits.

    Args:
        dataset (PlacePulseDataset): The dataset to split
        output_dir (str): Directory to save the split files
        train_size (int): Number of training samples
        random_state (int): Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the dataframe from the dataset
    df = dataset.dataframe.copy()

    # Verify we have the study_type column
    if 'study_type' not in df.columns:
        study_type_map = {v: k for k, v in dataset.study_types.items()}
        df['study_type'] = df['study_id'].map(study_type_map)

    # Verify we have enough samples
    if len(df) < train_size:
        raise ValueError(f"Dataset only has {len(df)} samples, but requested train_size={train_size}")

    # Perform stratified split
    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df['study_type'],
        random_state=random_state
    )

    # Save the splits
    train_path = os.path.join(output_dir, 'train_data.tsv')
    test_path = os.path.join(output_dir, 'test_data.tsv')

    train_df.to_csv(train_path, sep='\t', index=False)
    test_df.to_csv(test_path, sep='\t', index=False)

    print(f"Successfully created splits:")
    print(f"- Training set: {train_path} ({len(train_df)} samples)")
    print(f"- Test set: {test_path} ({len(test_df)} samples)")

    # Print class distributions
    print("\nClass distribution in training set:")
    print(train_df['study_type'].value_counts(normalize=True))
    print("\nClass distribution in test set:")
    print(test_df['study_type'].value_counts(normalize=True))

if __name__ == "__main__":
    try:
        # Initialize dataset - adjust paths as needed
        dataset = PlacePulseDatasetText(
            qscores_tsv_path='place-pulse-2.0/qscores.tsv',
            img_dir='place-pulse-2.0/images/',
            study_type='beautiful'
        )

        # Create stratified splits
        stratified_train_test_split(dataset)

    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Verify 'place-pulse-2.0/qscores.tsv' exists and is properly formatted")
        print("2. Check that 'place-pulse-2.0/images/' contains the image files")
        print("3. Ensure all required columns are present in qscores.tsv")
        print("4. Make sure PlacePulseDataset.py is in your Python path")
