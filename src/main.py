""" ImageNet classification """

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
do_learn = True
batch_size = 128
lr = 0.01
n_epochs = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_training = datasets.CIFAR10('../data/', train=True, download=True, transform=train_transform)
cifar_testing = datasets.CIFAR10('../data/', train=False, download=True, transform=train_transform)

def fit_cross_entropy():
    feature_extractor = OSNet(blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[16, 64, 96, 128])
    model = ClassificationNet(feature_extractor, feature_dim=512, n_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(cifar_training, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(cifar_testing, batch_size=batch_size, shuffle=False, **kwargs)

    fit(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=torch.nn.NLLLoss(),
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device,
        log_interval=10,
        metrics=[AccumulatedAccuracy()],
    )

    torch.save(model.state_dict(), '../models/cross_entropy_osnet.pt')

if __name__ == '__main__':
    fit_cross_entropy()



