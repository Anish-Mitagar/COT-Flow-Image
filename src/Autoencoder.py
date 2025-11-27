import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.utils as utils
import os
from src.plotter import *

# Define the encoder-decoder architecture
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.d = encoding_dim

        # Encoder layers
        self.layer1 = nn.Linear(2 * 28 * 28, encoding_dim)  

        # Decoder layers
        self.layer2 = nn.Linear(encoding_dim, 2 * 28 * 28)

        # Register these as buffers
        self.register_buffer('mu', None)
        self.register_buffer('std', None)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Add layer with ReLU activation function
        return F.relu(self.layer1(x))

    def decode(self, x):
        # Output layer with sigmoid activation for scaling from 0 to 1
        return torch.sigmoid(self.layer2(x))

def trainAE(net, train_loader, val_loader, saveDir, sStartTime, argType=torch.float32, device=torch.device('cuda')):
    print("Training autoencoder")

    cvt = lambda x: x.type(argType).to(device, non_blocking=True)
    utils.makedirs(saveDir)

    # Specify loss function
    criterion = nn.MSELoss()

    # Specify optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    best_loss = float('inf')
    bestParams = None

    # Number of epochs to train the model
    n_epochs = 200

    for epoch in range(1, n_epochs + 1):
        net.train()
        train_loss = 0.0
        for data in train_loader:
            images, fourier_images, _ = data
            images = torch.cat((images, fourier_images), dim=1)  # Concatenate along the channel dimension
            images = images.view(images.size(0), -1)
            images = cvt(images)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        net.eval()
        val_loss = 0.0
        for data in val_loader:
            images, fourier_images, _ = data
            combined_images = torch.cat((images, fourier_images), dim=1)
            combined_images = combined_images.view(combined_images.size(0), -1)
            combined_images = cvt(combined_images)

            outputs = net(combined_images)
            loss = criterion(outputs, combined_images)
            val_loss += loss.item() * combined_images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {val_loss:.6f}')

        if val_loss < best_loss:
            best_loss = val_loss
            bestParams = net.state_dict()

        if epoch % 20 == 0:
            net.eval()
            sSavePath = os.path.join(saveDir, 'figs', sStartTime + f'_autoencoder{epoch}.png')
            xRecreate = net(combined_images)
            
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
    mu = torch.zeros((1, d), dtype=argType, device=device)
    musqrd = torch.zeros((1, d), dtype=argType, device=device)
    totImages = 0

    net.load_state_dict(bestParams)

    i = 0
    net.eval()
    with torch.no_grad():
        for data in train_loader:
            images, fourier_images, _ = data
            combined_images = torch.cat((images, fourier_images), dim=1)
            combined_images = combined_images.view(combined_images.size(0), -1)
            combined_images = cvt(combined_images)
            outputs = net.encode(combined_images)
            nImages = outputs.shape[0]
            totImages += nImages
            mu += torch.mean(outputs, dim=0, keepdims=True)
            musqrd += torch.mean(outputs ** 2, dim=0, keepdims=True)

            if i == 0:
                sSavePath = os.path.join(saveDir, 'figs', sStartTime + '_autoencoder.png')
                outputs = (net.encode(combined_images) - 2.34) / 0.005
                xRecreate = net.decode(outputs * 0.005 + 2.34)
                # xRecreate = xRecreate.view(-1, 2, 28, 28)[:, 0, :, :]  # Only the original image part
                # print(xRecreate.shape, 'xRecreate shape', images.shape, 'images shape 129')
                xRecreate = xRecreate.view(xRecreate.size(0), -1)
                print(xRecreate.shape, 'xRecreate shape', images.shape, 'images shape134')
                xRecreate = xRecreate[:, :28*28]  # Only the original image part
                images = images.view(images.size(0), -1)
                plotAutoEnc(images, xRecreate, sSavePath)

                # sSavePath = os.path.join(saveDir, 'figs', sStartTime + '_noise_autoencoder.png')
                # noisy_outputs = outputs + 1.0 * torch.randn_like(outputs)
                # xRecreate_noisy = net.decode(noisy_outputs)
                # xRecreate_noisy = xRecreate_noisy.view(-1, 2, 28, 28)[:, 0, :, :]  # Only the original image part
                # print(xRecreate.shape, 'xRecreate shape', images.shape, 'images shape136')
                # plotAutoEnc(images[:, 0, :, :], xRecreate_noisy, sSavePath)

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

