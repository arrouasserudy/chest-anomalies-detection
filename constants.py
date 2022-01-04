import pandas as pd
import torch
import os

'''
Constants 
'''

colab = os.path.isdir("drive")

if colab:
    IMAGES_PATH = "data/train"
    DATA_PATH = "data/train.csv"
    METADATA_PATH = "data/train_meta.csv"
    EMBEDDED_METADATA_PATH = "data/embedded_metadata.csv"
    CHECKPOINT_PATH = "drive/MyDrive/object_detection_project/checkpoints"
    TRAIN_IMAGES = "drive/MyDrive/object_detection_project/train_images.txt"
    TEST_IMAGES = "drive/MyDrive/object_detection_project/test_images.txt"
    TRAIN_IMAGES_WE = "drive/MyDrive/object_detection_project/train_images_we.txt"
    TEST_IMAGES_WE = "drive/MyDrive/object_detection_project/test_images_we.txt"
    SAVED_MODELS = "drive/MyDrive/object_detection_project/checkpoints"

else:
    IMAGES_PATH = "./data/train"
    DATA_PATH = "./data/train.csv"
    METADATA_PATH = "./data/train_meta.csv"
    EMBEDDED_METADATA_PATH = "data/embedded_metadata.csv"
    CHECKPOINT_PATH = "./checkpoints"
    TRAIN_IMAGES = "./data/train_images.txt"
    TEST_IMAGES = "./data/test_images.txt"
    TRAIN_IMAGES_WE = "./data/train_images_we.txt"
    TEST_IMAGES_WE = "./data/test_images_we.txt"
    SAVED_MODELS = "checkpoints"

# The classes used fo the model
CLASSES = [
    "Background",
    "Aortic enlargement",
    "Cardiomegaly",
    "Pleural thickening",
    "Pulmonary fibrosis",
]

n_classes = len(CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Original data - contains image ids and annotated objects
train_data = pd.read_csv(DATA_PATH)

# Original size of each image
train_metadata = pd.read_csv(METADATA_PATH)

# Data with bounding boxes rescaled and transformed to a [x_center, y_center, width, height] format
embedded_metadata = (
    pd.read_csv(EMBEDDED_METADATA_PATH)
    if os.path.isfile(EMBEDDED_METADATA_PATH)
    else None
)

# Different configuration of priors
priors_config = [
    {
        "fmap_dims": {
            "conv4_3": 32,
            "conv7": 16,
            "conv8_2": 8,
            "conv9_2": 4,
            "conv10_2": 2,
        },
        "obj_scales": {
            "conv4_3": 0.1,
            "conv7": 0.2,
            "conv8_2": 0.375,
            "conv9_2": 0.55,
            "conv10_2": 0.725,
        },
        "aspect_ratios": {
            "conv4_3": [1.0, 2.0, 0.5],
            "conv7": [1.0, 2.0, 3.0, 0.5, 0.333],
            "conv8_2": [1.0, 2.0, 3.0, 0.5, 0.333],
            "conv9_2": [1.0, 2.0, 3.0, 0.5, 0.333],
            "conv10_2": [1.0, 2.0, 0.5],
        },
        "n_boxes": {
            "conv4_3": 4,
            "conv7": 6,
            "conv8_2": 6,
            "conv9_2": 6,
            "conv10_2": 4,
        },
    },
    {
        "fmap_dims": {
            "conv4_3": 32,
            "conv7": 16,
            "conv8_2": 8,
            "conv9_2": 4,
            "conv10_2": 2,
        },
        "obj_scales": {
            "conv4_3": 0.05,
            "conv7": 0.2,
            "conv8_2": 0.375,
            "conv9_2": 0.55,
            "conv10_2": 0.725,
        },
        "aspect_ratios": {
            "conv4_3": [1.0, 2.0, 3.0, 0.5, 0.333],
            "conv7": [1.0, 2.0, 3.0, 0.5, 0.333],
            "conv8_2": [1.0, 2.0, 3.0, 0.5, 0.333],
            "conv9_2": [1.0, 2.0, 3.0, 0.5, 0.333],
            "conv10_2": [1.0, 2.0, 0.5],
        },
        "n_boxes": {
            "conv4_3": 6,
            "conv7": 6,
            "conv8_2": 6,
            "conv9_2": 6,
            "conv10_2": 4,
        },
    },
]
