import os
from enum import Enum
import functools
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torchvision import io, transforms
from torchvision.io.image import ImageReadMode
import torchvision.transforms.functional as F

import pandas as pd
import numpy as np
from sklearn import model_selection


class OCTDLClass(Enum):
    AMD = 0
    DME = 1
    ERM = 2
    NO  = 3
    RAO = 4
    RVO = 5
    VID = 6


class OCTDLDataset(Dataset):
    def __init__(self, data, classes: List[str], transform=None):
        self.data = data
        self.transform = transform
        self.classes = classes
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}

        print("OCTDL Dataset initialized, "
              f"labels: {', '.join([f'{cls}: {self.class_to_index[cls]}' for cls in self.classes])}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_file, label = self.data[idx]
        # Convert to grayscale
        image = io.read_image(image_file, mode=ImageReadMode.RGB)
        image = F.rgb_to_grayscale(image, num_output_channels=3)
        image = F.convert_image_dtype(image)
        if self.transform:
            image = self.transform(image)
        encoded_label = self.class_to_index[label]

        return image, encoded_label
    

def get_image_label_pairs(
        patient_ids: List[np.int64], 
        patient_to_images: Dict[np.int64, List[Tuple[str, str]]]
    ):
    return functools.reduce(
        lambda pairs, pid: pairs + patient_to_images[pid],
        patient_ids, []
    )

def load_octdl_dataset(
        classes: List[OCTDLClass], 
        train_transform: transforms.Compose,
        val_test_transform: transforms.Compose,
        ds_dir: str = './OCTDL', 
        labels_file: str = './OCTDL_labels.csv'
    ):
    labels = [cls.name for cls in classes]
    
    labels_df = pd.read_csv(labels_file)
    labels_df = labels_df.query('disease in @labels')

    # map patient_id to (image_path, label) list
    patient_to_images: Dict[np.int64, List[(str, str)]] = {} 
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

    train_data = get_image_label_pairs(train_ids, patient_to_images)
    val_data = get_image_label_pairs(val_ids, patient_to_images)
    test_data = get_image_label_pairs(test_ids, patient_to_images)

    train_dataset = OCTDLDataset(train_data, labels, transform=train_transform)
    val_dataset = OCTDLDataset(val_data, labels, transform=val_test_transform)
    test_dataset = OCTDLDataset(test_data, labels, transform=val_test_transform)

    all_labels = labels_df['disease'].to_list()

    balancing_weights = []
    for cls in classes:
        num_cls_labels = len([l for l in all_labels if l == cls.name])
        balancing_weights.append(len(all_labels) / num_cls_labels)

    balancing_weights = torch.Tensor(balancing_weights)

    return train_dataset, val_dataset, test_dataset, balancing_weights