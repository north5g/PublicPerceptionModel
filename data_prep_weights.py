import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms
import pandas as pd
from torchvision.transforms.functional import crop
from PIL import Image
import numpy as np

import os
from os import listdir
from os.path import isfile, join

import zipfile
from tqdm import tqdm
import requests
import shutil

from sklearn.model_selection import train_test_split

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

class PlacePulseDatasetWeight(Dataset):
    """
    A PyTorch dataset class for the Place Pulse 2.0 dataset, returning image and weight.

    Args:
        dataframe (pd.DataFrame, optional): The dataframe containing the dataset information. If not provided, it will be loaded from the 'qscores_tsv_path'. Defaults to None.
        qscores_tsv_path (str, optional): The path to the tsv file containing the dataset information. Defaults to 'place-pulse-2.0/qscores.tsv'.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Defaults to None.
        img_dir (str, optional): The directory path where the dataset images are stored. Defaults to 'place-pulse-2.0/images/'.
        return_location_id (bool, optional): Whether to return the location ID along with the image and weight. Defaults to False.
        study_id (int, optional): The ID of the study to filter the dataset. Defaults to None.
        study_type (str, optional): The type of the study to filter the dataset. Defaults to None.
        transform_only_image (bool, optional): Whether to apply the transformation only to the image. Defaults to True.
        split (str, optional): The split of the dataset to use. Can be 'train' or 'val'. Defaults to None.
    """
    def __init__(self, dataframe=None, qscores_tsv_path='place-pulse-2.0/qscores.tsv',
                 transform=None, img_dir='place-pulse-2.0/images_preprocessed/',
                 train_size=None, return_location_id=False, study_id=None, study_type=None,
                 transform_only_image=True, split=None):

        if qscores_tsv_path and dataframe is not None:
            raise ValueError("Please provide either 'qscores_tsv_path' or 'dataframe', not both.")

        self.transform = transform
        self.dataset_folder_path = img_dir
        self.return_location_id = return_location_id
        self.transform_only_image = transform_only_image
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
        if qscores_tsv_path:
            dataframe = pd.read_csv(qscores_tsv_path, sep='\t')
        if qscores_tsv_path or isinstance(dataframe, pd.DataFrame):
            self.dataframe = dataframe
            # Rename trueskill.score to weight for clarity
            self.dataframe['weight'] = torch.FloatTensor(self.dataframe['trueskill.score'].values)
            # Always add study_type column if possible
            if 'study_type' not in self.dataframe.columns and 'study_id' in self.dataframe.columns:
                self.dataframe['study_type'] = self.dataframe['study_id'].map(self.study_ids_to_type)
        else:
            raise ValueError("Must provide either qscores_tsv_path or dataframe.")

        # Filter by study_type if not "all"
        if study_type and study_type != "all":
            self.dataframe = self.dataframe[self.dataframe['study_type'] == study_type]

        # Optionally filter by study_id (if provided)
        if study_id:
            self.dataframe = self.dataframe[self.dataframe['study_id'] == study_id]

        # Split if requested
        if split:
            if train_size is None:
                train_df, val_df = train_test_split(self.dataframe, test_size=0.2, random_state=42, stratify=self.dataframe['study_type'])
            else:
                train_df, val_df = train_test_split(self.dataframe, train_size=train_size, test_size=0.2, random_state=42, stratify=self.dataframe['study_type'])
            if split == 'train':
                self.dataframe = train_df
            elif split == 'val':
                self.dataframe = val_df

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):
        location_id = self.dataframe.iloc[idx]['location_id']
        img = self.get_img_by_location_id(location_id)  # Always returns PIL.Image
        weight = self.dataframe.iloc[idx]['weight']

        # Apply transform if provided (should expect PIL.Image)
        if self.transform:
            img = self.transform(img)

        sample = {"pixel_values": img, "labels": weight}
        if self.return_location_id:
            sample["location_id"] = location_id
        return sample

    def get_img_by_location_id(self, location_id):
        extension = '.jpg'
        img_name = f"{location_id}{extension}"
        img_path = f'{self.dataset_folder_path}{img_name}'
        img = io.imread(img_path)
        # Always convert to PIL.Image
        if isinstance(img, np.ndarray):
            if img.ndim == 2:  # grayscale to RGB
                img = np.stack([img]*3, axis=-1)
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            # Fallback: try to open with PIL directly
            img = Image.open(img_path).convert("RGB")
        return img

    def get_sample_by_location_id(self, location_id):
        """
        Returns a sample dictionary for a given location_id, matching the format of __getitem__.
        """
        img = self.get_img_by_location_id(location_id)
        row = self.dataframe[self.dataframe['location_id'] == location_id]
        if row.empty:
            raise ValueError(f"location_id {location_id} not found in dataframe.")
        weight = row['weight'].values[0]

        if self.transform:
            img = self.transform(img)

        sample = {"pixel_values": img, "weight": weight}
        if self.return_location_id:
            sample["location_id"] = location_id
        return sample

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
        qscores_clean = PlacePulseDatasetWeight.get_q_score_only_for_files_in_folder(qscores_df, 'place-pulse-2.0/images_preprocessed/')
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
    def extract_archive(zip_file_path='place-pulse-2.0.zip', destination_folder='place-pulse-2.0') -> None:
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

        #print('Cleaning up.')
        #shutil.rmtree(source_dir)
        #os.rename(destination_dir, source_dir)

        print('Removing samples where image is missing.')
        PlacePulseDatasetWeight.clean_qscores()

    @staticmethod
    def load() -> None:
        """
        Loads the dataset by downloading, extracting, preprocessing, and deleting the archive.
        """
        if os.path.exists('place-pulse-2.0'):
            print('Error: The "place-pulse-2.0" folder already exists.')
            return

        PlacePulseDatasetWeight.download_archive()
        zip_file_path='place-pulse-2.0.zip'
        PlacePulseDatasetWeight.extract_archive(zip_file_path=zip_file_path)
        print('Deleting archive.')
        os.remove(zip_file_path)
        #PlacePulseDatasetWeight.preprocess()

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
            output_dir (str, optional): Directory to save the output files. Defaults to 'place-pulse-2.0/'.

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
        dataset = PlacePulseDatasetWeight(
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

# import pandas as pd
# from sklearn.model_selection import train_test_split
# import numpy as np

# def prepare_placepulse_data(input_path):
#     """Load and prepare the PlacePulse dataset"""
#     df = pd.read_csv(input_path)

#     # Create balanced 'choice' column (left/right/equal)
#     df['choice'] = np.where(df['left'] < df['right'], 'left',
#                            np.where(df['left'] > df['right'], 'right', 'equal'))

#     # Create category column based on left/right/equal
#     df['category'] = df['choice']

#     return df

# def balanced_stratified_sample(df, test_size=0.8, random_state=42):
#     """Perform stratified sampling with perfectly balanced left/right choices"""
#     # Separate equal choices first (these will be split proportionally)
#     df_equal = df[df['choice'] == 'equal']
#     df_unequal = df[df['choice'] != 'equal']

#     # Split equal choices (20% train, 80% test)
#     equal_train, equal_test = train_test_split(
#         df_equal,
#         test_size=test_size,
#         random_state=random_state
#     )

#     # For unequal choices, we need perfect balance between left/right
#     df_left = df_unequal[df_unequal['choice'] == 'left']
#     df_right = df_unequal[df_unequal['choice'] == 'right']

#     # Determine the minimum number of samples between left and right
#     min_samples = min(len(df_left), len(df_right))

#     # Take equal numbers from left and right for perfect balance
#     df_left = df_left.sample(min_samples, random_state=random_state)
#     df_right = df_right.sample(min_samples, random_state=random_state)

#     # Now split these balanced samples into train/test
#     left_train, left_test = train_test_split(
#         df_left,
#         test_size=test_size,
#         random_state=random_state,
#         stratify=df_left['choice']  # Though redundant here since all are 'left'
#     )

#     right_train, right_test = train_test_split(
#         df_right,
#         test_size=test_size,
#         random_state=random_state,
#         stratify=df_right['choice']  # Though redundant here since all are 'right'
#     )

#     # Combine all samples
#     train_df = pd.concat([equal_train, left_train, right_train])
#     test_df = pd.concat([equal_test, left_test, right_test])

#     # Add set identifier
#     train_df['set'] = 'train'
#     test_df['set'] = 'test'

#     # Reset index to get row numbers
#     train_df = train_df.reset_index(drop=True).reset_index().rename(columns={'index': 'row_number'})
#     test_df = test_df.reset_index(drop=True).reset_index().rename(columns={'index': 'row_number'})

#     return train_df, test_df

# def save_datasets(train_df, test_df, output_dir='output'):
#     """Save the train and test datasets"""
#     import os
#     os.makedirs(output_dir, exist_ok=True)

#     train_path = f"{output_dir}/safety_train_df.csv"
#     test_path = f"{output_dir}/safety_test_df.csv"

#     # Select and order the required columns
#     output_cols = ['row_number', 'choice', 'left', 'right', 'set', 'category']
#     train_df = train_df[output_cols + [c for c in train_df.columns if c not in output_cols]]
#     test_df = test_df[output_cols + [c for c in test_df.columns if c not in output_cols]]

#     train_df.to_csv(train_path, index=False)
#     test_df.to_csv(test_path, index=False)

#     print(f"Saved training data to {train_path} ({len(train_df)} samples)")
#     print(f"Saved test data to {test_path} ({len(test_df)} samples)")

#     return train_path, test_path

# # Example usage
# if __name__ == "__main__":
#     # Load your dataset (adjust path)
#     input_path = "safety_train_df.csv"  # Or your actual file path
#     df = prepare_placepulse_data(input_path)

#     # Perform stratified sampling with perfect left/right balance
#     train_df, test_df = balanced_stratified_sample(df)

#     # Save results
#     save_datasets(train_df, test_df)

#     # Print class distributions
#     print("\nTraining set distribution:")
#     print(train_df['choice'].value_counts())
#     print("\nTest set distribution:")
#     print(test_df['choice'].value_counts())

#     # Verify balance
#     print("\nBalance verification:")
#     print("Train - Left vs Right:", sum(train_df['choice'] == 'left'), "vs", sum(train_df['choice'] == 'right'))
#     print("Test - Left vs Right:", sum(test_df['choice'] == 'left'), "vs", sum(test_df['choice'] == 'right'))