import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as t
from tqdm import tqdm
from PIL import Image

from constants import (
    IMAGES_PATH,
    embedded_metadata,
    TRAIN_IMAGES,
    TEST_IMAGES,
    TRAIN_IMAGES_WE,
    TEST_IMAGES_WE
)

'''
Data preprocessing
'''


def split_data(df=embedded_metadata, test_percent=0.2, without_empty=False):
    '''
    Split data into a train and test set
    :return: two different lists of image ids - train and test
    '''
    train_image_ids = []
    test_image_ids = []

    modulo = 1 / test_percent
    n_empty_images = 0
    n_abnormal_images = 0
    n_abnormal_images_train = 0
    n_abnormal_images_test = 0

    for i, image_id in tqdm(enumerate(df["image_id"].unique())):
        bboxes = df[df["image_id"] == image_id].reset_index(drop=True)
        no_anomalies_img = len(bboxes) == 3 and bboxes.loc[0, "class_id"] == 0

        # this condition is used to split the data when saving the background/class image ration
        if no_anomalies_img:
            # Here we save only the image with an anomaly detected
            if without_empty:
                continue
            if n_empty_images % modulo == 0:
                test_image_ids.append(image_id)
            else:
                train_image_ids.append(image_id)
            n_empty_images += 1
        else:
            if n_abnormal_images % modulo == 0:
                test_image_ids.append(image_id)
                n_abnormal_images_test += 1
            else:
                train_image_ids.append(image_id)
                n_abnormal_images_train += 1
            n_abnormal_images += 1

    return train_image_ids, test_image_ids


class xray_dataset(Dataset):
    def __init__(
            self, image_ids, df=embedded_metadata, augmentation=False, save_path=None
    ):
        self.df = df
        self.image_ids = image_ids

        # Data augmentation - for each image that actually have an anomaly,
        # we will duplicate it with one of the 3 distortions [brightness, contrast, saturation] randomly.
        augmented_images_ids = []
        if augmentation:
            for i, image_id in tqdm(enumerate(image_ids)):
                bboxes = self.df[self.df["image_id"] == image_id].reset_index(drop=True)
                if not (len(bboxes) == 3 and bboxes.loc[0, "class_id"] == 0):
                    augmented_images_ids.append(
                        f'{image_id}_{random.choice(["brightness", "contrast", "saturation"])}'
                    )
            self.image_ids.extend(augmented_images_ids)

        if save_path:
            with open(save_path, "a") as f:
                for id in self.image_ids:
                    f.write(f"{id}\n")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        splitted_image_id = image_id.split("_")
        image_id = splitted_image_id[0]

        # Get the bounding boxes
        bboxes = self.df[self.df["image_id"] == image_id]
        bboxes = bboxes.reset_index(drop=True)

        image = cv2.imread(IMAGES_PATH + "/" + image_id + ".png", 0)

        if len(splitted_image_id) > 1:
            # apply the data modification
            image = transform_img(image, splitted_image_id[1])

        # Normalize the image
        image = image.astype("float32")
        image = image - image.min()
        image = image / image.max()
        image = torch.from_numpy(image)

        # If no anomaly are found in the image, we want to see the corresponding bbox [0, 0, 0, 0] only once
        # and not 3 times because of the 3 radiologists
        if bboxes.loc[0, "class_id"] == 0:
            bboxes = bboxes.loc[[0], :]

        boxes = bboxes[["x_min", "y_min", "x_max", "y_max"]].values

        labels = torch.tensor(bboxes["class_id"].values, dtype=torch.int64)

        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32), "labels": labels}

        # return the image and the bboxes
        return image, target


def collate_fn(batch):
    images = list()
    boxes = list()
    labels = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1]["boxes"])
        labels.append(b[1]["labels"])

    images = torch.stack(images, dim=0)
    images = torch.unsqueeze(images, dim=1)

    return images, boxes, labels


def get_data(batch_size=8, without_empty=False):
    train_path = TRAIN_IMAGES_WE if without_empty else TRAIN_IMAGES
    test_path = TEST_IMAGES_WE if without_empty else TEST_IMAGES

    # Get the images list from a predefined file to save compute time
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        with open(train_path, "r") as f:
            lines = f.readlines()
            train_image_ids = [l[:-1] for l in lines]
        with open(test_path, "r") as f:
            lines = f.readlines()
            test_image_ids = [l[:-1] for l in lines]
        train_dataset = xray_dataset(train_image_ids)
        test_dataset = xray_dataset(test_image_ids)
    else:
        train, test = split_data(without_empty=without_empty)
        train_dataset = xray_dataset(train, augmentation=True, save_path=train_path)
        test_dataset = xray_dataset(test, save_path=test_path)

    # Create and return the train and the test set
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4
    )
    return train_loader, test_loader


def transform_img(image, distortion):
    """
    Distort the image according to the required distortion
    """
    new_image = Image.fromarray(image)
    distortions = {
        "brightness": t.adjust_brightness,
        "contrast": t.adjust_contrast,
        "saturation": t.adjust_saturation,
    }

    new_image = distortions[distortion](new_image, random.uniform(0.5, 1.5))

    return np.array(new_image)
