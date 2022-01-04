import torch
from constants import CLASSES, device, n_classes
from utils import find_jaccard_overlap


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    :param det_boxes: detected bounding boxes
    :param det_labels: detected labels
    :param det_scores: detected labels' scores
    :param true_boxes: true objects bounding boxes
    :param true_labels: true objects labels
    """

    # Store true objects
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))

    true_images = torch.LongTensor(true_images).to(device)
    true_boxes = torch.cat(true_boxes, dim=0)
    true_labels = torch.cat(true_labels, dim=0)

    # Store detected objects
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)
    det_boxes = torch.cat(det_boxes, dim=0)
    det_labels = torch.cat(det_labels, dim=0)
    det_scores = torch.cat(det_scores, dim=0)

    # Store APs into this variable
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)

    # Calculate APs for each class (except background)
    for c in range(1, n_classes):
        # Find objects from the current class
        true_class_images = true_images[true_labels == c]
        true_class_boxes = true_boxes[true_labels == c]

        true_class_objects = true_labels[true_labels == c]

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]
        det_class_boxes = det_boxes[det_labels == c]
        det_class_scores = det_scores[det_labels == c]
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)
        det_class_images = det_class_images[sort_ind]
        det_class_boxes = det_class_boxes[sort_ind]

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)

        for d in range(n_class_detections):
            detection_box = det_class_boxes[d].unsqueeze(0)
            detected_image = det_class_images[d]

            # Find objects in the same image with this class
            object_boxes = true_class_boxes[true_class_images == detected_image]

            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(detection_box, object_boxes)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                true_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)
        cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)
        cumul_recall = cumul_true_positives / len(true_class_objects)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=0.1).tolist()
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.0
        average_precisions[c - 1] = precisions.mean()

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {CLASSES[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


def get_losses_from_file(path="./checkpoints/checkpoint_losses.txt"):
    losses = []
    mAP = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(" ")
            losses.append(float(splitted[4]))
            if len(splitted) >= 8:
                mAP.append(float(splitted[7]))
    return losses, mAP
