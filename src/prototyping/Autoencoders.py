import torch 
import torch.nn as nn
import torch.nn.functional as F
import lib.utils as utils
from src.prototyping.CustomPlotter import *

class LinearAutoencoder(nn.Module):
    def __init__(self, original_dim, encoding_dim):
        super(LinearAutoencoder, self).__init__()
        self.d = encoding_dim
        self.layer1 = nn.Linear(original_dim, encoding_dim)
        self.layer2 = nn.Linear(encoding_dim, original_dim)

        self.register_buffer('mu', None)
        self.register_buffer('std', None)

        self.type_net = "linear"

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self,x):
        return F.relu(self.layer1(x))

    def decode(self,x):
        return torch.sigmoid(self.layer2(x))
    
class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, original_dim, encoding_dim):
        super(ConvolutionalAutoencoder, self).__init__()
        self.d = encoding_dim
        self.og_d = original_dim
        
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 1x28x28 -> 16x14x14
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 16x14x14 -> 32x7x7
        self.conv3 = nn.Conv2d(32, 64, 7)  # 32x7x7 -> 64x1x1
        self.fc1 = nn.Linear(64, encoding_dim)

        self.fc2 = nn.Linear(encoding_dim, 64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 7)  # 64x1x1 -> 32x7x7
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)  # 32x7x7 -> 16x14x14
        self.deconv3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)  # 16x14x14 -> 1x28x28

        self.type_net = "conv"

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def decode(self, x):
        x = self.fc2(x)
        x = x.view(x.size(0), 64, 1, 1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x
    
# Not to be used with train_auto_enc
class DualLinearAutoencoder(nn.Module):
    def __init__(self, original_dim, encoding_dim):
        super(DualLinearAutoencoder, self).__init__()
        self.d = encoding_dim

        self.encoder_orig = nn.Linear(original_dim, encoding_dim)
        self.encoder_fourier = nn.Linear(original_dim, encoding_dim)
        self.decoder = nn.Linear(2 * encoding_dim, original_dim)

        self.register_buffer('mu', torch.zeros(2 * encoding_dim))
        self.register_buffer('std', torch.ones(2 * encoding_dim))

        self.type_net_1 = "linear"
        self.type_net_2 = "linear"

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

# Not to be used with train_auto_enc  
class DualLinearAutoencoder2(nn.Module):
    def __init__(self, original_dim, encoding_dim):
        super(DualLinearAutoencoder2, self).__init__()
        self.d = encoding_dim

        self.encoder_orig_0 = nn.Linear(original_dim, 512)
        self.encoder_orig_1 = nn.Linear(512, 256)
        self.encoder_orig_2 = nn.Linear(256, encoding_dim)
        self.encoder_fourier_0 = nn.Linear(original_dim, 512)
        self.encoder_fourier_1 = nn.Linear(512, 256)
        self.encoder_fourier_2 = nn.Linear(256, encoding_dim)
        self.decoder_0 = nn.Linear(2 * encoding_dim, 512)
        self.decoder_1 = nn.Linear(512, original_dim)

        self.register_buffer('mu', torch.zeros(2 * encoding_dim))
        self.register_buffer('std', torch.ones(2 * encoding_dim))

        self.type_net_1 = "linear"
        self.type_net_2 = "linear"

    def forward(self, x, condition):
        x_encoded = self.encode_orig(x)
        condition_encoded = self.encode_fourier(condition)
        combined = torch.cat((x_encoded, condition_encoded), dim=1)
        x_decoded = self.decode(combined)
        return x_decoded
    
    def encode_orig(self, x):
        out = F.relu(self.encoder_orig_0(x))
        out = F.relu(self.encoder_orig_1(out))
        return F.relu(self.encoder_orig_2(out))

    def encode_fourier(self, x):
        out = F.relu(self.encoder_fourier_0(x))
        out = F.relu(self.encoder_fourier_1(out))
        return F.relu(self.encoder_fourier_2(out))

    def decode(self, x):
        out = F.relu(self.decoder_0(x))
        return torch.sigmoid(self.decoder_1(out))
    
def train_auto_enc(net, train_loader, val_loader, n_epochs, lr, custom_name, save_dir='experiments/cnf/large', img_shape = None, argType=torch.float32, device=torch.device('cpu'), logger = None):
    print("Training Autoencoder")
    cvt = lambda x: x.type(argType).to(device, non_blocking=True)
    utils.makedirs(save_dir)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_losses = list()
    val_losses = list()

    for epoch in range(1, n_epochs + 1):

        net.train()
        train_loss = 0.0
        for data in train_loader:
            images, *_ = data

            if net.type_net == "linear":
                images = images.view(images.size(0), -1)
            else:
                images = images.view(images.size(0), 1, img_shape[0], img_shape[1])

            images = cvt(images)
            outputs = net(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        if torch.isnan(loss):
                print("Loss is nan")
                return -1

        net.eval()
        val_loss = 0.0
        for data in val_loader:
            images, *_ = data

            if net.type_net == "linear":
                images = images.view(images.size(0), -1)
            else:
                images = images.view(images.size(0), 1, img_shape[0], img_shape[1])

            images = cvt(images)
            outputs = net(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            val_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if torch.isnan(loss):
                print("Loss is nan")
                return -1

        print('Epoch: {} \tTraining Loss: {:.6f} \t Validation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            val_loss
        ))

        # plot
        if epoch % 20 == 0:
            net.eval()
            sSavePath = os.path.join(save_dir, 'figs', custom_name + '_autoencoder{:d}.png'.format(epoch))
            xRecreate = net(images)
            conv = net.type_net == "conv"
            plot_auto_enc(images, xRecreate, sSavePath, conv=conv)

        if epoch % 150 == 0:
            for p in optimizer.param_groups:
                p['lr'] /= 10.0
            print("lr: ", p['lr'])
    
    plot_and_save_loss_values([train_losses, val_losses], ["Train Loss", "Validation Loss"], save_dir + str("/") + custom_name + "_loss_graph.png")
    return 0

def train_dual_auto_enc(net, train_loader, val_loader, n_epochs, lr, custom_name, save_dir='experiments/cnf/large', img_shape = None, cond_shape = None, argType=torch.float32, device=torch.device('cpu'), logger = None):
    print("Training Autoencoder")
    cvt = lambda x: x.type(argType).to(device, non_blocking=True)
    utils.makedirs(save_dir)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_losses = list()
    val_losses = list()

    for epoch in range(1, n_epochs + 1):

        net.train()
        train_loss = 0.0
        for data in train_loader:
            images, conditions, *_ = data

            if net.type_net_1 == "linear":
                images = images.view(images.size(0), -1)
            else:
                images = images.view(images.size(0), 1, img_shape[0], img_shape[1])

            if net.type_net_2 == "linear":
                conditions = conditions.view(images.size(0), -1)
            else:
                conditions = conditions.view(images.size(0), 1, cond_shape[0], cond_shape[1])

            images = cvt(images)
            conditions = cvt(conditions)

            outputs = net(images, conditions)

            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        if torch.isnan(loss):
                print("Loss is nan")
                return -1

        net.eval()
        val_loss = 0.0
        for data in val_loader:
            images, conditions, *_ = data

            if net.type_net_1 == "linear":
                images = images.view(images.size(0), -1)
            else:
                images = images.view(images.size(0), 1, img_shape[0], img_shape[1])

            if net.type_net_2 == "linear":
                conditions = conditions.view(images.size(0), -1)
            else:
                conditions = conditions.view(images.size(0), 1, cond_shape[0], cond_shape[1])

            images = cvt(images)
            conditions = cvt(conditions)

            outputs = net(images, conditions)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            val_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if torch.isnan(loss):
                print("Loss is nan")
                return -1

        print('Epoch: {} \tTraining Loss: {:.6f} \t Validation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            val_loss
        ))

        # plot
        if epoch % 20 == 0:
            net.eval()
            sSavePath = os.path.join(save_dir, 'figs', custom_name + '_autoencoder{:d}.png'.format(epoch))
            xRecreate = net(images, conditions)
            conv = net.type_net_1 == "conv"
            plot_auto_enc(images, xRecreate, sSavePath, conv=conv)

        if epoch % 150 == 0:
            for p in optimizer.param_groups:
                p['lr'] /= 10.0
            print("lr: ", p['lr'])
    
    plot_and_save_loss_values([train_losses, val_losses], ["Train Loss", "Validation Loss"], save_dir + str("/") + custom_name + "_loss_graph.png")
    return 0