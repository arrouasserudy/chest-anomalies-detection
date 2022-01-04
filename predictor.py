import torch
from torch.nn import Softmax

from constants import device
from loss_function import cxcy_to_xy, gcxgcy_to_cxcy
from utils import find_jaccard_overlap


class Predictor:
    def __init__(self, model, n_classes, priors_cxcy={}):

        self.model = model
        self.n_classes = n_classes
        self.priors_cxcy = priors_cxcy

    def detect_objects(
        self,
        predicted_locs,
        predicted_scores,
        min_score=None,
        max_overlap=None,
        top_k=200,
    ):
        """
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted bounding boxes
        :param predicted_scores: class scores for each predicted bounding boxe
        :param min_score: minimum threshold for a box to be considered a match
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: ikeep only the top 'k' results
        :return: detections (boxes, labels, scores)
        """
        max_overlap = max_overlap or 0.5
        min_score = min_score or 0.3
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        sf = Softmax(dim=2)
        predicted_scores = sf(predicted_scores)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]
                score_above_min_score = (class_scores > min_score)
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]
                class_decoded_locs = decoded_locs[score_above_min_score]

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)

                #################################
                # Non-Maximum Suppression (NMS) #
                #################################

                # A tensor to keep the boxes to suppress
                suppress = torch.zeros(n_above_min_score, dtype=torch.uint8).to(device)

                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps are greater than maximum overlap
                    suppress = torch.max(suppress, overlap[box] > max_overlap)

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.0]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes)
            image_labels = torch.cat(image_labels)
            image_scores = torch.cat(image_scores)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]
                image_boxes = image_boxes[sort_ind][:top_k]
                image_labels = image_labels[sort_ind][:top_k]

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores

    def detect(self, image):
        """
        Detect objects in an image with a trained SSD300, and visualize the results.
        """
        self.model.eval()

        # Transform the image
        image = image.astype("float32")
        image = image - image.min()
        image = image / image.max()
        image = torch.from_numpy(image)
        image = image.to(device)

        # Pass through the model
        predicted_locs, predicted_scores = self.model(image.unsqueeze(0).unsqueeze(0))

        # Detect objects from model output
        det_boxes, det_labels, det_scores = self.detect_objects(predicted_locs, predicted_scores)

        # Move detections to the CPU
        det_boxes = det_boxes[0].to("cpu")

        # Transform to original image dimensions
        original_dims = torch.FloatTensor([image.shape[0], image.shape[1], image.shape[0], image.shape[1]]).unsqueeze(0)
        det_boxes = det_boxes * original_dims

        return det_boxes, det_labels, det_scores
