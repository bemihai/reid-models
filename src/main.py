""" CIFAR-10 classification """

import torch
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.backends import cudnn

from src.networks import ClassificationNet
from src.trainer import fit
from src.osnet import OSBlock, OSNet
from src.metrics import AccumulatedAccuracy

cudnn.benchmark = True
batch_size = 256
lr = 0.001
momentum = 0.8
n_epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

train_transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.247, 0.2435, 0.2616)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.247, 0.2435, 0.2616)),
])

cifar_training = datasets.CIFAR10('../data/', train=True, download=True, transform=train_transform)
cifar_testing = datasets.CIFAR10('../data/', train=False, download=True, transform=train_transform)

train_loader = DataLoader(cifar_training, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
val_loader = DataLoader(cifar_testing, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)


def fit_cross_entropy():
    feature_extractor = OSNet(
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[32, 128, 192, 256],
        feature_dim=256
    )
    model = ClassificationNet(feature_extractor, feature_dim=256, n_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss = torch.nn.NLLLoss()

    fit(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=loss,
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device,
        metrics=[AccumulatedAccuracy()],
    )

    torch.save(model.state_dict(), '../models/cross_entropy_osnet.pt')


if __name__ == "__main__":
    fit_cross_entropy()


