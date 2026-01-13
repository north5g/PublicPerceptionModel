from numpy import random
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil
import zipfile
import requests
from os import listdir
from tqdm import tqdm
from os.path import isfile, join
import numpy as np

from transformations import multi_transformation

class PlacePulseDataset(Dataset):
    def __init__(self, transform_data, instances = 1, csv_path='place-pulse-2.0/qscores.tsv', image_folder='place-pulse-2.0/images_preprocessed/', study_type_filter="all", fraction: float = 1.0, noise = 0, random_state: int = 42):
        """
        Args:
            csv_path (str): Path to the CSV file with 'image_path' and 'score' columns.
            image_folder (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        base_df = pd.read_csv(csv_path, sep='\t')
        study_types = {
            '50a68a51fdc9f05596000002': 'safe',
            '50f62c41a84ea7c5fdd2e454': 'lively',
            '50f62c68a84ea7c5fdd2e456': 'clean',
            '50f62cb7a84ea7c5fdd2e458': 'wealthy',
            '50f62ccfa84ea7c5fdd2e459': 'depressing',
            '5217c351ad93a7d3e7b07a64': 'beautiful'
        }
        base_df['study_type'] = base_df['study_id'].map(study_types)
        if fraction < 1.0:
            base_df = base_df.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
        
                

        dataframes = []
        self.transform_map = {}
        available_transforms = ["none", "zoomed", "greyscale", "contrast", "lowresolution", "flipped"]

        print("Available transformations: " + ", ".join(available_transforms))
        for i in range(instances):
            while True:
                transforms = input(
                    f"Setting up dataset instance {i+1}. "
                    "What transformations do you want to apply to the dataset? (comma-separated, spaces optional)\n"
                )
                # Clean up input and split
                transform_list = transforms.replace(" ", "").split(",")

                # Check if all transforms are valid
                invalid = [t for t in transform_list if t not in available_transforms]
                if invalid:
                    print(f"Invalid transformations: {', '.join(invalid)}. Please try again.")
                else:
                    self.transform_map[i] = transform_list
                    break
            
        for i in range(instances):
            
            df = base_df.copy()
            df["instances"] = i
            
            noise = 0.01 * np.random.uniform(-noise, noise, size=len(df)) + 1
            df['score'] = df['trueskill.score'].astype(float) * noise

            if study_type_filter != "all":
                if isinstance(study_type_filter, str):
                    study_type_filter = [study_type_filter]
                df = df[df['study_type'].isin(study_type_filter)]
                df['normalized_score'] = df.groupby('study_type')['score'].transform(
                    lambda x: ((x - x.min()) / (x.max() - x.min())) * 100
                )
            else:
                df['normalized_score'] = ((df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())) * 100
            dataframes.append(df)

        self.df = pd.concat(dataframes).reset_index(drop=True)
        
        self.image_folder = image_folder
        self.label_encoder = LabelEncoder()
        self.df['study_type_id'] = self.label_encoder.fit_transform(self.df['study_type'])
        self.transform_data = transform_data

        # Filter out any missing files
        self.df = self.df[self.df['location_id'].apply(
            lambda loc_id: os.path.exists(os.path.join(self.image_folder, f"{loc_id}.jpg")))]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_folder, f"{row['location_id']}.jpg")
        image = Image.open(image_path).convert("RGB")

        transform = self.transform_map[row["instances"]]
        transformation = multi_transformation(image, transform, self.transform_data)
        image = transformation(image)
        normalized_score = float(row['normalized_score'])

        return {
          'pixel_values': image,
          'labels': normalized_score,
          'study_type_ids': int(row['study_type_id']),
        }

    def split(self, eval=0.2, test=0.1, random_state=42):
        """
        Splits the dataset into training and evaluation sets.

        Args:
            eval (float): Proportion of the dataset to include in the evaluation split.
            random_state (int): Seed for reproducibility.

        Returns:
            Tuple[PlacePulseDataset, PlacePulseDataset]: train_dataset, eval_dataset
        """
        train_df, eval_df = train_test_split(self.df, test_size=eval, random_state=random_state)
        train_df, test_df = train_test_split(train_df, test_size=test, random_state=random_state)

        train_dataset = PlacePulseDataset.__from_split__(train_df, self.image_folder, self.transform, self.label_encoder)
        eval_dataset = PlacePulseDataset.__from_split__(eval_df, self.image_folder, self.transform, self.label_encoder)
        test_dataset = PlacePulseDataset.__from_split__(test_df, self.image_folder, self.transform, self.label_encoder)

        return train_dataset, eval_dataset, test_dataset

    @classmethod
    def __from_split__(cls, df, image_folder, transform, label_encoder):
        """Helper method to create a dataset from a split DataFrame."""
        obj = cls.__new__(cls)
        obj.df = df.reset_index(drop=True)
        obj.image_folder = image_folder
        obj.transform = transform
        obj.label_encoder = label_encoder
        return obj

    @staticmethod
    def download_and_extract():
        url = "https://www.dropbox.com/s/grzoiwsaeqrmc1l/place-pulse-2.0.zip?dl=1"
        zip_path = 'place-pulse-2.0.zip'

        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(total=total_size, desc='Downloading', unit='B', unit_scale=True) as pbar:
            for chunk in r.iter_content(1024):
                f.write(chunk)
                pbar.update(len(chunk))

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('place-pulse-2.0')

        os.remove(zip_path)

    @staticmethod
    def preprocess_images() -> None:
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

        #print('Cleaning up.')
        #shutil.rmtree(source_dir)
        #os.rename(destination_dir, source_dir)

        print('Removing samples where image is missing.')
        PlacePulseDataset.clean_qscores()

    @staticmethod
    def get_q_score_only_for_files_in_folder(q_scores: pd.DataFrame, folder_path):
        file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        location_ids_from_existing_files = [os.path.splitext(file_name)[0] for file_name in file_names]
        return q_scores[q_scores['location_id'].isin(location_ids_from_existing_files)]

    @staticmethod
    def clean_qscores(folder_path='place-pulse-2.0/images_preprocessed/'):
        qscores_tsv_path = 'place-pulse-2.0/qscores.tsv'
        qscores_df = pd.read_csv(qscores_tsv_path, sep='\t')
        qscores_clean = PlacePulseDataset.get_q_score_only_for_files_in_folder(qscores_df, folder_path)
        qscores_clean.to_csv(qscores_tsv_path, sep='\t', index=False)

    @staticmethod
    def setup():
        if os.path.exists('place-pulse-2.0'):
            print("Dataset already exists. Cleaning qscores...")
            PlacePulseDataset.preprocess_images()
            PlacePulseDataset.clean_qscores('place-pulse-2.0/images_preprocessed/')
            print("Setup complete.")
        else:
            PlacePulseDataset.download_and_extract()
            PlacePulseDataset.preprocess_images()
            PlacePulseDataset.clean_qscores('place-pulse-2.0/images_preprocessed/')
            print("Setup complete.")