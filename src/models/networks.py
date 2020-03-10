""" Network architectures """

from torch import nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """ Basic convnet for feature extraction """

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.BatchNorm2d(32), nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5), nn.BatchNorm2d(64), nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512), nn.BatchNorm1d(512), nn.PReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.PReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        out = self.convnet(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

    def get_features(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    """
    Baseline classification net: add a fully-connected layer with the number of classes and
    train the feature extractor for classification with softmax and cross-entropy.
    """
    def __init__(self, feature_extractor, feature_dim, n_classes):
        super(ClassificationNet, self).__init__()
        self.extractor = feature_extractor
        self.n_classes = n_classes
        self.activation = nn.PReLU()
        self.linear = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        out = self.extractor(x)
        out = self.activation(out)
        out = self.linear(out)
        # out = F.log_softmax(out, dim=-1)
        return out,

    # extract 2-dim features from penultimate layer
    def get_features(self, x):
        self.activation(self.extractor(x))
















