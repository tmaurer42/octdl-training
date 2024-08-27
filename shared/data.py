import os
from enum import Enum
import functools

import torch
from torch.utils.data import Dataset, DataLoader
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_file, label = self.data[idx]
        image = io.read_image(image_file, mode=ImageReadMode.RGB)
        # Convert to grayscale with three output channels
        # for transfer-learning compatability
        image = F.convert_image_dtype(image)
        image = F.rgb_to_grayscale(image, num_output_channels=3)
        if self.transform:
            image = self.transform(image)
        encoded_label = self.class_to_index[label]

        return image, encoded_label


def _get_image_label_pairs(
    ids: list[np.int64],
    id_to_images: dict[np.int64, list[(str, str)]]
) -> list[tuple[str, str]]:
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


def prepare_dataset(
    classes: list[OCTDLClass],
    augmentation: bool,
    batch_size: int,
    validation_batch_size: int,
    img_target_size=224,
    ds_dir: str = './OCTDL',
    labels_file: str = 'OCTDL_labels.csv'
):
    train_loaders, val_loaders, test_loader = prepare_dataset_partitioned(
        classes=classes,
        augmentation=augmentation,
        batch_size=batch_size,
        n_partitions=1,
        img_target_size=img_target_size,
        ds_dir=ds_dir,
        labels_file=labels_file,
        validation_batch_size=validation_batch_size
    )
    train_loader, val_loader = train_loaders[0], val_loaders[0]

    return train_loader, val_loader, test_loader


def get_partitioned_data(
    classes: list[OCTDLClass],
    augmentation: bool,
    n_partitions: int,
    img_target_size=224,
    ds_dir: str = './OCTDL',
    labels_file: str = 'OCTDL_labels.csv'
) -> tuple[list[OCTDLDataset], list[OCTDLDataset], OCTDLDataset]:
    partitions, test_data = load_octdl_data(
        classes, ds_dir, labels_file, n_partitions)

    base_transform, train_transform = get_transforms(img_target_size)

    train_datasets = []
    val_datasets = []
    for partition in partitions:
        (train_data, val_data) = partition
        train_ds = OCTDLDataset(
            train_data,
            classes,
            transform=train_transform if augmentation else base_transform
        )
        val_ds = OCTDLDataset(val_data, classes, transform=base_transform)

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    test_ds = OCTDLDataset(test_data, classes, transform=base_transform)

    return train_datasets, val_datasets, test_ds


def prepare_dataset_partitioned(
    classes: list[OCTDLClass],
    augmentation: bool,
    batch_size: int,
    validation_batch_size: int,
    n_partitions: int,
    img_target_size=224,
    ds_dir: str = './OCTDL',
    labels_file: str = 'OCTDL_labels.csv',
    n_workers: int = 0
) -> tuple[list[DataLoader], list[DataLoader], DataLoader]:
    train_datasets, val_datasets, test_dataset = get_partitioned_data(
        classes,
        augmentation,
        n_partitions,
        img_target_size,
        ds_dir,
        labels_file
    )

    train_loaders = [DataLoader(train_ds, batch_size, shuffle=True, num_workers=n_workers)
                    for train_ds in train_datasets]
    val_loaders = [DataLoader(val_ds, validation_batch_size, shuffle=False)
                    for val_ds in val_datasets]
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loaders, val_loaders, test_loader


def load_octdl_data(
    classes: list[OCTDLClass],
    ds_dir: str = './OCTDL',
    labels_file: str = 'OCTDL_labels.csv',
    n_partitions: int = 1
):
    """
    Load OCTDL dataset containing the given classes,
    split into train, validation, and test sets.
    The data is split such that each patient's data is
    present in only one of the sets.

    Parameters:
        classes (list[OCTDLClass]): 
            The classes to load.
        ds_dir (str): 
            Directory containing the dataset. Default is './OCTDL'.
        labels_file (str): 
            Path to the labels CSV file. Default is './OCTDL_labels.csv'.
        n_partitions (int):
            The number of partitions to split the train and validation data into.

    Returns:
        Tuple containing:
        - test_data: List of (image_path, label) for the test set.
        - List of tuples: Each tuple contains (train_data, val_data) for one partition.
    """
    labels = [cls.name for cls in classes]
    labels_df = pd.read_csv(os.path.join(ds_dir, labels_file))
    labels_df = labels_df.query('disease in @labels')

    # map patient_id to (image_path, label) list
    patient_to_images: dict[np.int64, list[(str, str)]] = {}
    patient_classes: dict[np.int64, str] = {}
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
                patient_classes[patient_id] = label
                i += 1

    patient_ids = list(patient_to_images.keys())
    patient_labels = [patient_classes[pid] for pid in patient_ids]

    # First, split off the test set
    train_val_ids, test_ids, _ , _ = model_selection.train_test_split(
        patient_ids, patient_labels, test_size=0.15, random_state=42, stratify=patient_labels
    )

    test_data = _get_image_label_pairs(test_ids, patient_to_images)

    all_partitions = []
    partition_size = len(train_val_ids) // n_partitions
    for i in range(n_partitions):
        if i == n_partitions - 1:
            partition_ids = train_val_ids[i*partition_size:]
        else:
            partition_ids = train_val_ids[i*partition_size:(i+1)*partition_size]
            
        partition_samples = _get_image_label_pairs(partition_ids, patient_to_images)
        partition_x = [sample[0] for sample in partition_samples]
        partition_y = [sample[1] for sample in partition_samples]

        # Stratified split into train and val sets within each partition
        train_data, val_data, train_labels, val_labels = model_selection.train_test_split(
            partition_x, partition_y, test_size=0.20, random_state=42, stratify=partition_y
        )

        train_data = list(zip(train_data, train_labels))
        val_data = list(zip(val_data, val_labels))

        all_partitions.append((train_data, val_data))

    return all_partitions, test_data


def get_balancing_weights(
    classes: list[OCTDLClass],
    ds_dir: str = './OCTDL',
    labels_file: str = 'OCTDL_labels.csv'
):
    labels = [cls.name for cls in classes]
    labels_df = pd.read_csv(os.path.join(ds_dir, labels_file))
    labels_df = labels_df.query('disease in @labels')

    all_labels = labels_df['disease'].to_list()
    balancing_weights = []
    for cls in classes:
        num_cls_labels = len([l for l in all_labels if l == cls.name])
        balancing_weights.append(len(all_labels) / num_cls_labels)

    balancing_weights = torch.Tensor(balancing_weights)

    return balancing_weights


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
