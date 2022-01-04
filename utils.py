import glob
import os
from math import sqrt

import pandas as pd
import torch
from tqdm import tqdm
from constants import (
    EMBEDDED_METADATA_PATH,
    train_data,
    train_metadata,
    device,
    priors_config, SAVED_MODELS,
)


def create_embedded_metadata(reduced=True):
    """
    Merge the different input files in order to
    create a new embedded csv file
    """
    if os.path.isfile(EMBEDDED_METADATA_PATH):
        return

    image_id_array = []
    class_id_array = []
    class_name_array = []
    x_min_array = []
    x_max_array = []
    y_min_array = []
    y_max_array = []
    x_center_array = []
    y_center_array = []
    width_array = []
    height_array = []

    # To use only the 5 biggest classes
    if reduced:
        map = {
            0: 1,
            3: 2,
            11: 3,
            13: 4,
            14: 0,
        }
    else:
        map = {
            0: 14,
            14: 0,
        }

    for index, row in tqdm(train_data.iterrows()):
        # Keep only the wanted classes and remap them
        if row["class_id"] in map:
            class_id = map[row["class_id"]]
        elif reduced:
            continue
        else:
            class_id = row["class_id"]

        image_id = row["image_id"]

        img_metadata = train_metadata.loc[train_metadata["image_id"] == image_id]
        dim0, dim1 = int(img_metadata["dim0"]), int(img_metadata["dim1"])

        # Rescale the bounding boxes according to the original image size
        x_min = (row["x_min"] / dim1) if str(row["x_min"]) != "nan" else 0
        x_max = (row["x_max"] / dim1) if str(row["x_max"]) != "nan" else 0
        y_min = (row["y_min"] / dim0) if str(row["y_min"]) != "nan" else 0
        y_max = (row["y_max"] / dim0) if str(row["y_max"]) != "nan" else 0

        # Compute the center x,y, the width and the height of the bounding boxes
        x_center = (((x_max - x_min) / 2) + x_min) if str(row["x_min"]) != "nan" else 0
        y_center = (((y_max - y_min) / 2) + y_min) if str(row["y_min"]) != "nan" else 0
        width = x_max - x_min
        height = y_max - y_min

        image_id_array.append(image_id)
        class_id_array.append(class_id)
        class_name_array.append(row["class_name"])
        x_min_array.append(x_min)
        x_max_array.append(x_max)
        y_min_array.append(y_min)
        y_max_array.append(y_max)
        x_center_array.append(x_center)
        y_center_array.append(y_center)
        width_array.append(width)
        height_array.append(height)

    d = {
        "image_id": image_id_array,
        "class_id": class_id_array,
        "class_name": class_name_array,
        "x_min": x_min_array,
        "y_min": y_min_array,
        "x_max": x_max_array,
        "y_max": y_max_array,
        "x_center": x_center_array,
        "y_center": y_center_array,
        "width": width_array,
        "height": height_array,
    }
    df = pd.DataFrame(data=d)

    # Create the new embedded file
    df.to_csv(EMBEDDED_METADATA_PATH)


def create_prior_boxes(config=priors_config[0]):
    """
    Create the prior boxes for the SSD model according to the defined priors_config
    """

    fmap_dims = config["fmap_dims"]
    obj_scales = config["obj_scales"]
    aspect_ratios = config["aspect_ratios"]

    fmaps = list(fmap_dims.keys())

    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append(
                        [
                            cx,
                            cy,
                            obj_scales[fmap] * sqrt(ratio),
                            obj_scales[fmap] / sqrt(ratio),
                        ]
                    )

                    # Compute the additional prior
                    if ratio == 1.0:
                        try:
                            additional_scale = sqrt(
                                obj_scales[fmap] * obj_scales[fmaps[k + 1]]
                            )
                        # For the last feature map, there is no "next" feature map
                        except IndexError:
                            additional_scale = 1.0
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])

    prior_boxes = torch.FloatTensor(prior_boxes).to(device)

    # Ensure the values between 0 and 1
    prior_boxes.clamp_(0, 1)

    return prior_boxes


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes.
    """

    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes.
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])

    # Find the union
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection

    return intersection / union


def get_models_to_evaluate():
    """
    return the list of the saved models
    """
    models = glob.glob(f"{SAVED_MODELS}/*.tar")
    models = [model.split("/")[-1].split(".")[0] for model in models]
    return models

