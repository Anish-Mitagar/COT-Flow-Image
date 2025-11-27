import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.utils as utils
import os
from src.plotter import *

# Define the encoder-decoder architecture with conditioning
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.d = encoding_dim

        self.encoder_orig = nn.Linear(28 * 28, encoding_dim)
        self.encoder_fourier = nn.Linear(28 * 28, encoding_dim)
        self.decoder = nn.Linear(2 * encoding_dim, 28 * 28)
        self.register_buffer('mu', torch.zeros(2 * encoding_dim))
        self.register_buffer('std', torch.ones(2 * encoding_dim))

    def forward(self, x, condition):
        x_encoded = self.encode_orig(x)
        condition_encoded = self.encode_fourier(condition)
        combined = torch.cat((x_encoded, condition_encoded), dim=1)
        x_decoded = self.decode(combined)
        return x_decoded

    def encode_orig(self, x):
        return F.relu(self.encoder_orig(x))

    def encode_fourier(self, x):
        return F.relu(self.encoder_fourier(x))

    def decode(self, x):
        return torch.sigmoid(self.decoder(x))


def trainAE(net, train_loader, val_loader, saveDir, sStartTime, argType=torch.float32, device=torch.device('cuda')):
    print("Training conditional autoencoder")

    cvt = lambda x: x.type(argType).to(device, non_blocking=True)
    utils.makedirs(saveDir)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    best_loss = float('inf')
    bestParams = None

    n_epochs = 100

    for epoch in range(1, n_epochs + 1):
        net.train()
        train_loss = 0.0
        for data in train_loader:
            images, fourier_images, _ = data
            images = images.view(images.size(0), -1)
            fourier_images = fourier_images.view(fourier_images.size(0), -1)
            images = cvt(images)
            fourier_images = cvt(fourier_images)

            optimizer.zero_grad()
            outputs = net(images, fourier_images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        net.eval()
        val_loss = 0.0
        for data in val_loader:
            images, fourier_images, _ = data
            images = images.view(images.size(0), -1)
            fourier_images = fourier_images.view(fourier_images.size(0), -1)
            images = cvt(images)
            fourier_images = cvt(fourier_images)

            outputs = net(images, fourier_images)
            loss = criterion(outputs, images)
            val_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {val_loss:.6f}')

        if val_loss < best_loss:
            best_loss = val_loss
            bestParams = net.state_dict()

        if epoch % 20 == 0:
            net.eval()
            sSavePath = os.path.join(saveDir, 'figs', sStartTime + f'_autoencoder{epoch}.png')
            xRecreate = net(images, fourier_images)
            
            images = images.view(images.size(0), -1)
            xRecreate = xRecreate.view(xRecreate.size(0), -1)
            print(xRecreate.shape, 'xRecreate shape', images.shape, 'images shape95')
            xRecreate = xRecreate[:, :28*28]  # Only the original image part
            print(xRecreate.shape, 'xRecreate shape', images.shape, 'images shape97')
            plotAutoEnc(images, xRecreate, sSavePath)

        if epoch % 150 == 0:
            for p in optimizer.param_groups:
                p['lr'] /= 10.0
            print("lr: ", p['lr'])

    d = net.d
    mu = torch.zeros((1, 2 * d), dtype=argType, device=device)
    musqrd = torch.zeros((1, 2 * d), dtype=argType, device=device)
    totImages = 0

    net.load_state_dict(bestParams)

    i = 0
    net.eval()
    with torch.no_grad():
        for data in train_loader:
            images, fourier_images, _ = data
            images = images.view(images.size(0), -1)
            fourier_images = fourier_images.view(fourier_images.size(0), -1)
            images = cvt(images)
            fourier_images = cvt(fourier_images)

            combined_images = torch.cat((images, fourier_images), dim=1)
            outputs = net.encode_orig(images)
            condition_outputs = net.encode_fourier(fourier_images)
            encoded = torch.cat((outputs, condition_outputs), dim=1)

            nImages = encoded.shape[0]
            totImages += nImages
            mu += torch.mean(encoded, dim=0, keepdims=True)
            musqrd += torch.mean(encoded ** 2, dim=0, keepdims=True)

            if i == 0:
                sSavePath = os.path.join(saveDir, 'figs', sStartTime + '_autoencoder.png')
                outputs = (encoded - 2.34) / 0.005
                xRecreate = net.decode(outputs * 0.005 + 2.34)
                xRecreate = xRecreate.view(xRecreate.size(0), -1)
                print(xRecreate.shape, 'xRecreate shape', images.shape, 'images shape134')
                xRecreate = xRecreate[:, :28*28]  # Only the original image part
                images = images.view(images.size(0), -1)
                plotAutoEnc(images, xRecreate, sSavePath)

            i += 1

        mu = mu / i
        musqrd = musqrd / i
        std = torch.sqrt(torch.abs(mu ** 2 - musqrd))

        mu.requires_grad = False
        std.requires_grad = False
        net.mu = mu
        net.std = std

        torch.save({
            'state_dict': net.state_dict(),
        }, os.path.join(saveDir, sStartTime + '_autoenc_checkpt.pth'))

        return net
