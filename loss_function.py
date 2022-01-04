import torch
from torch import nn
from constants import device
from utils import find_jaccard_overlap


class MultiBoxLoss(nn.Module):
    """
    This is a combination of the predicted bounding boxes and their prediction score
    """

    def __init__(self, priors_cxcy=None, threshold=0.5, neg_pos_ratio=3):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(self.priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)

        for i in range(batch_size):
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (6128)

            # Set priors whose overlaps with objects are less than the threshold to be background
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)

        # Identify priors that are positive
        positive_priors = true_classes != 0

        # LOCALIZATION LOSS is computed only over positive priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])

        # CONFIDENCE LOSS is computed over positive priors and the most difficult (hardest) negative priors in each
        # image Hard Negative Mining

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)
        n_hard_negatives = self.neg_pos_ratio * n_positives

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]

        # Next, find which priors are hard-negative
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.0  # positive priors are ignored
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()

        # TOTAL LOSS
        alpha = 1.0
        return conf_loss + alpha * loc_loss


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max)
    to center-size coordinates (center_x, center_y, width, height).
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2, xy[:, 2:] - xy[:, :2]], 1)


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (center_x, center_y, width, height)
    to boundary coordinates (x_min, y_min, x_max, y_max).
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form)

    For the center coordinates, find the offset and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5, ], 1)


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model
    This is the inverse of the function above.
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)
