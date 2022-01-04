import torch
from torch import nn
import torch.nn.functional as F

from constants import priors_config


class BaseModel(nn.Module):
    """
    BaseModel convolutions to produce lower-level feature maps.
    """

    def __init__(self):
        super(BaseModel, self).__init__()

        # VGG16 model modified
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    def forward(self, image):
        out = F.relu(self.conv1_1(image))  # (N, 64, 256, 256)
        out = F.relu(self.conv1_2(out))  # (N, 64, 256, 256)
        out = self.pool1(out)  # (N, 64, 128, 128)

        out = F.relu(self.conv2_1(out))  # (N, 128, 128, 128)
        out = F.relu(self.conv2_2(out))  # (N, 128, 128, 128)
        out = self.pool2(out)  # (N, 128, 64, 64)

        out = F.relu(self.conv3_1(out))  # (N, 256, 64, 64)
        out = F.relu(self.conv3_2(out))  # (N, 256, 64, 64)
        out = F.relu(self.conv3_3(out))  # (N, 256, 64, 64)
        out = self.pool3(out)  # (N, 256, 32, 32)

        out = F.relu(self.conv4_1(out))  # (N, 512, 32, 32)
        out = F.relu(self.conv4_2(out))  # (N, 512, 32, 32)
        out = F.relu(self.conv4_3(out))  # (N, 512, 32, 32)
        conv4_3_feats = out  # (N, 512, 32, 32)
        out = self.pool4(out)  # (N, 512, 16, 16)

        out = F.relu(self.conv5_1(out))  # (N, 512, 16, 16)
        out = F.relu(self.conv5_2(out))  # (N, 512, 16, 16)
        out = F.relu(self.conv5_3(out))  # (N, 512, 16, 16)
        out = self.pool5(out)  # (N, 512, 16, 16)

        out = F.relu(self.conv6(out))  # (N, 1024, 16, 16)

        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 16, 16)

        # Lower-level feature maps
        return conv4_3_feats, conv7_feats


class AuxiliaryConvolutions(nn.Module):
    """
    Auxiliary convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

    def forward(self, conv7_feats):
        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 16, 16)
        out = F.relu(self.conv8_2(out))  # (N, 512, 8, 8)
        conv8_2_feats = out  # (N, 512, 8, 8)

        out = F.relu(self.conv9_1(out))  # (N, 128, 8, 8)
        out = F.relu(self.conv9_2(out))  # (N, 256, 4, 4)
        conv9_2_feats = out  # (N, 256, 4, 4)

        out = F.relu(self.conv10_1(out))  # (N, 128, 4, 4)
        out = F.relu(self.conv10_2(out))  # (N, 256, 2, 2)
        conv10_2_feats = out  # (N, 256, 2, 2)

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    """

    def __init__(self, n_classes, priors):
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = priors["n_boxes"]

        # Localization prediction convolutions
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes["conv4_3"] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes["conv7"] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes["conv8_2"] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes["conv9_2"] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes["conv10_2"] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes["conv4_3"] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes["conv7"] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes["conv8_2"] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes["conv9_2"] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes["conv10_2"] * n_classes, kernel_size=3, padding=1)

    def forward(
        self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats
    ):
        batch_size = conv4_3_feats.size(0)

        # Predict localization bounding boxes
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 32, 32)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 32, 32, 16), to match prior-box order
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 4096, 4)

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 16, 16)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 16, 16, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 1536, 4)

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 8, 8)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 8, 8, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 384, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 4, 4)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 4, 4, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 96, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 2, 2)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 2, 2, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 16, 4)

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 32, 32)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 32, 32, 4 * n_classes)
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)  # (N, 4096, n_classes)

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 16, 16)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 16, 16, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)  # (N, 1536, n_classes)

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 8, 8)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 8, 8, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 384, n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 4, 4)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 4, 4, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 96, n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 2, 2)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 2, 2, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 16, n_classes)

        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2], dim=1)  # (N, 6128, 4)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2], dim=1)  # (N, 6128, n_classes)

        # Return a total of 6128 boxes
        return locs, classes_scores


def init_conv2d_layers(model):
    """
    Help function to initialize convolution parameters.
    """
    for c in model.children():
        if isinstance(c, nn.Conv2d):
            nn.init.xavier_uniform_(c.weight)
            nn.init.constant_(c.bias, 0.0)


class SSD(nn.Module):
    """
    The full network
    """

    def __init__(self, n_classes, priors=priors_config[0]):
        super(SSD, self).__init__()

        self.n_classes = n_classes

        self.base = BaseModel()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes, priors)

        # Initialize convolutions' parameters
        for model in [self.base, self.aux_convs, self.pred_convs]:
            init_conv2d_layers(model)

    def forward(self, image):
        # Run base network convolutions
        conv4_3_feats, conv7_feats = self.base(image)

        # Run auxiliary convolutions
        conv8_2_feats, conv9_2_feats, conv10_2_feats = self.aux_convs(conv7_feats)

        # Run prediction convolutions
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats)

        return locs, classes_scores
