import os
from enum import Enum
import functools

import torch
from torch.utils.data import Dataset
from torchvision import io
from torchvision.io.image import ImageReadMode
from torchvision import transforms
import torchvision.transforms.functional as F

import pandas as pd
import numpy as np
from sklearn import model_selection


class OCTDLClass(Enum):
    AMD = 0
    DME = 1
    ERM = 2
    NO = 3
    RAO = 4
    RVO = 5
    VID = 6


class OCTDLDataset(Dataset):
    """
    A PyTorch Dataset for the OCTDL dataset.

    Parameters:
        data (list[tuple]): 
            List of tuples containing an image-path and a label.
        classes (list[str]): 
            List of classes in the dataset. 
            The categorical labels will be assigned in the order of the list,
            e.g. [OCTDLClass.AMD, OCTDLClass.NO] -> AMD: 0, NO: 1.
        transform (callable, optional): Optional transform to be applied to a sample.
    """

    def __init__(self, data: list[tuple[str, str]], classes: list[OCTDLClass], transform=None):
        self.data = data
        self.transform = transform
        self.classes = [cls.name for cls in classes]
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}

        print("OCTDL Dataset initialized, "
              f"labels: {', '.join([f'{cls}: {self.class_to_index[cls]}' for cls in self.classes])}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_file, label = self.data[idx]
        image = io.read_image(image_file, mode=ImageReadMode.RGB)
        # Convert to grayscale with three output channels
        # for transfer-learning compatability
        image = F.rgb_to_grayscale(image, num_output_channels=3)
        image = F.convert_image_dtype(image)
        if self.transform:
            image = self.transform(image)
        encoded_label = self.class_to_index[label]

        return image, encoded_label


def _get_image_label_pairs(
    ids: list[np.int64],
    id_to_images: dict[np.int64, list[(str, str)]]
) -> list[(str, str)]:
    """
    Get a list of image-label pairs from a list of IDs and a dictionary mapping IDs to image-label pairs.

    Parameters:

        ids (list[np.int64]): 
            A list of IDs.

        id_to_images (dict[np.int64, list[(str, str)]]): 
            A dictionary where keys are IDs and values are lists of tuples with image-path and label.

    Returns:
        list[(str, str)]: A list of tuples of image-path and label.
    """
    return functools.reduce(
        lambda pairs, pid: pairs + id_to_images[pid],
        ids, []
    )


def load_octdl_data(
    classes: list[OCTDLClass],
    ds_dir: str = './OCTDL',
    labels_file: str = 'OCTDL_labels.csv'
):
    """
    Load OCTDL dataset containing the given classes,
    split into train, validation, and test sets, 
    and compute balancing weights.
    The data is split such that each patient's data is
    present in only one of the sets.

    Parameters:
        classes (list[OCTDLClass]): 
            The classes to load.
        ds_dir (str): 
            Directory containing the dataset. Default is './OCTDL'.
        labels_file (str): 
            Path to the labels CSV file. Default is './OCTDL_labels.csv'.

    Returns:
        Tuple containing: (train_data, val_data, test_data, balancing_weights)
    """
    labels = [cls.name for cls in classes]

    labels_df = pd.read_csv(os.path.join(ds_dir, labels_file))
    labels_df = labels_df.query('disease in @labels')

    # map patient_id to (image_path, label) list
    patient_to_images: dict[np.int64, list[(str, str)]] = {}
    i = 0
    for label in labels:
        label_dir = os.path.join(ds_dir, label)
        if os.path.isdir(label_dir):
            for image_file in sorted(os.listdir(label_dir)):
                image_path = os.path.join(label_dir, image_file)
                patient_id = labels_df.iloc[i]['patient_id']

                if patient_id not in patient_to_images:
                    patient_to_images[patient_id] = []

                patient_to_images[patient_id].append((image_path, label))
                i = i+1

    patient_ids = list(patient_to_images.keys())

    # Split patients into train, val and test sets
    train_ids, val_test_ids = model_selection.train_test_split(
        patient_ids, test_size=0.3, random_state=42
    )
    val_ids, test_ids = model_selection.train_test_split(
        val_test_ids, test_size=0.5, random_state=42
    )

    train_data = _get_image_label_pairs(train_ids, patient_to_images)
    val_data = _get_image_label_pairs(val_ids, patient_to_images)
    test_data = _get_image_label_pairs(test_ids, patient_to_images)

    # Compute balancing weights according to the distribution in the whole dataset
    all_labels = labels_df['disease'].to_list()
    balancing_weights = []
    for cls in classes:
        num_cls_labels = len([l for l in all_labels if l == cls.name])
        balancing_weights.append(len(all_labels) / num_cls_labels)

    balancing_weights = torch.Tensor(balancing_weights)

    return train_data, val_data, test_data, balancing_weights


def get_transforms(img_target_size: int):
    """
    Get base and augmentation transforms for image preprocessing.
    Images will be of size (img_target_size, img_target_size) after the transforms
    and their pixel values normalized between -1 and 1.

    Parameters:
        img_target_size (int): Target size for image resizing.

    Returns:
        tuple: (base_transform, augment_transform)
    """
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    base_transform = transforms.Compose([
        transforms.Resize((img_target_size, img_target_size)),
        transforms.Normalize(mean=mean, std=std)
    ])

    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_target_size, scale=(0.8, 1.0)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=(3, 5)),
        transforms.Normalize(mean=mean, std=std),
    ])

    return base_transform, augment_transform
