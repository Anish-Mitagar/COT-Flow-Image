from torchvision import transforms, datasets
import numpy as np
import torch
import torch.fft
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

class MNISTWithFourier(torch.utils.data.Dataset):
    def __init__(self, root, noise_level = 0.01, train=True, transform=None, download=False):
        self.mnist = datasets.MNIST(root=root, train=train, transform=transform, download=download)
        self.transform = transform
        self.noise_level = noise_level

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        img, target = self.mnist[index]
        #print(img.shape, 'img.shape in mnist') #1,28,28
        fourier_img = torch.fft.fftshift(torch.fft.fft2(img)) #torch.fft.fft2(img)
        noise = self.noise_level * torch.randn_like(fourier_img)
        fourier_img = fourier_img + noise
        return img, fourier_img, target

def getLoader(name, batch, test_batch, augment=False, hasGPU=False, conditional=-1):
    if name == 'mnist':
        val_size = 1.0 / 6.0
        random_seed = 0

        # define transforms
        normalize = transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

        kwargs = {'num_workers': 0, 'pin_memory': True} if hasGPU else {}

        # load the dataset
        data = MNISTWithFourier(root='../data', train=True, transform=train_transform, download=True)
        test_data = MNISTWithFourier(root='../data', train=False, transform=val_transform, download=True)

        if conditional >= 0 and conditional <= 9:
            idx = data.mnist.targets == conditional
            data.mnist.data = data.mnist.data[idx, :]
            data.mnist.targets = data.mnist.targets[idx]
            nTot = torch.sum(idx).item()
            nTrain = int((5.0 / 6.0) * nTot)
            nVal = nTot - nTrain
            train_data, valid_data = random_split(data, [nTrain, nVal])

            idx = test_data.mnist.targets == conditional
            test_data.mnist.data = test_data.mnist.data[idx, :]
            test_data.mnist.targets = test_data.mnist.targets[idx]
        else:
            train_data, valid_data = random_split(data, [50000, 10000])

        train_loader = DataLoader(train_data, batch_size=batch, shuffle=True, **kwargs)
        val_loader = DataLoader(valid_data, batch_size=test_batch, shuffle=False, **kwargs)
        test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader
    else:
        raise ValueError('Unknown dataset')
