import torch 
import torch.nn as nn
import torch.nn.functional as F
import lib.utils as utils
from src.prototyping.CustomPlotter import *
from src.OTFlowProblem import *
from src.prototyping.LossTerms import *
from src.mmd import *

class MaskAutoEncCNF(nn.Module):
    def __init__(self, original_dim, encoding_dim, mask, autoenc, Phi, nt, eps):
        super(MaskAutoEncCNF, self).__init__()
        self.original_dim = original_dim
        self.encoding_dim = encoding_dim
        self.autoenc = autoenc
        self.type_net_1 = self.autoenc.type_net_1
        self.type_net_2 = self.autoenc.type_net_2
        self.mask = mask
        self.Phi = Phi
        self.nt = nt
        self.eps = eps

    def forward(self, x, condition):
        masked_condition, mask = self.mask(condition)
        x_encoded = self.autoenc.encode_orig(x)
        condition_encoded = self.autoenc.encode_fourier(masked_condition)
        combined = torch.cat((x_encoded, condition_encoded), dim=1)
        x_decoded = self.autoenc.decode(combined)
        normalized_combined = (combined - self.autoenc.mu) / (self.autoenc.std + self.eps)
        x0, y = normalized_combined.chunk(2, dim=1)
        Jc , costs = OTFlowProblem(x0, y, self.Phi, [0,1], nt=self.nt, stepper="rk4", alph=self.Phi.alph)
        return x_decoded, Jc, costs, torch.mean(combined, dim=0, keepdims=True), torch.mean(combined ** 2, dim=0, keepdims=True), mask

def trainMaskAutoEncCNF(net, train_loader, val_loader, test_loader, n_epochs, lr, l1_mask_reg, custom_name, save_dir='experiments/cnf/large', img_shape = None, cond_shape = None, argType=torch.float32, device=torch.device('cpu'), logger = None):
    print("Training Masked Dual Autoencoder")
    cvt = lambda x: x.type(argType).to(device, non_blocking=True)
    utils.makedirs(save_dir)
    criterion = nn.MSELoss()
    optimizer_autoenc = torch.optim.Adam(net.autoenc.parameters(), lr=lr)
    optimizer_cnf = torch.optim.Adam(net.Phi.parameters(), lr=lr)
    optimizer_mask = torch.optim.Adam(net.mask.parameters(), lr=lr)

    clampMax = 2.0
    clampMin = -2.0

    train_losses = list()
    val_losses = list()
    train_losses_mse = list()
    val_losses_mse = list()
    train_losses_l1 = list()
    val_losses_l1 = list()
    train_ot_flow_losses = list()
    val_ot_flow_losses = list()

    mmds = list()


    d = net.autoenc.d
    mu = None
    musqrd = None

    for epoch in range(1, n_epochs + 1):
        print("Currently on epoch " + str(epoch) + "!")

        mu = torch.zeros((1, 2 * d), dtype=argType, device=device)
        musqrd = torch.zeros((1, 2 * d), dtype=argType, device=device)
        print("Training!")
        net.train()

        train_loss = 0.0
        train_mse_loss_total = 0.0
        train_l1_loss_total = 0.0
        train_ot_flow_loss_total = 0.0

        val_loss = 0.0
        val_mse_loss_total = 0.0
        val_l1_loss_total = 0.0
        val_ot_flow_loss_total = 0.0

        cnt = 0
        cnt_val = 0

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
            optimizer_autoenc.zero_grad()
            optimizer_cnf.zero_grad()
            optimizer_mask.zero_grad()

            for p in net.Phi.parameters():
                p.data = torch.clamp(p.data, clampMin, clampMax)

            decoded_images, ot_flow_loss_train, _, mu_i, musqrd_i, _ = net(images, conditions)
            mse_loss_train = criterion(decoded_images, images)
            l1_loss_train = l1_mask_reg * l1_loss(net.mask.logits)
            loss = mse_loss_train + ot_flow_loss_train + l1_loss_train
            mu += mu_i
            musqrd += musqrd_i
            loss.backward()
            optimizer_autoenc.step()
            optimizer_cnf.step()
            optimizer_mask.step()

            train_mse_loss_total += mse_loss_train.item() * images.size(0)
            train_l1_loss_total += l1_loss_train.item() 

            train_ot_flow_loss_total += ot_flow_loss_train.item() 
            train_loss += ((mse_loss_train.item() * images.size(0)) + l1_loss_train.item() + ot_flow_loss_train.item())

            cnt+=1

        mu = mu / cnt
        musqrd = musqrd / cnt
        std = torch.sqrt(torch.abs(mu ** 2 - musqrd))

   
        net.autoenc.mu = mu.clone().detach()
        net.autoenc.std = std.clone().detach()

        if torch.isnan(ot_flow_loss_train):
                print("Loss train is nan")
                return -1
        
        with torch.no_grad():
            net.eval()
            print("Validating!")
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

                for p in net.Phi.parameters():
                    p.data = torch.clamp(p.data, clampMin, clampMax)

                decoded_images, ot_flow_loss_val, _, mu_i, musqrd_i, _ = net(images, conditions)
                mse_loss_val = criterion(decoded_images, images)
                
                l1_loss_val = l1_mask_reg * l1_loss(net.mask.logits)
                loss = mse_loss_val + ot_flow_loss_val + l1_loss_val

                val_mse_loss_total += mse_loss_val.item() * images.size(0)
                val_l1_loss_total += l1_loss_val.item() 
                val_ot_flow_loss_total += ot_flow_loss_val.item() 
                val_loss += ((mse_loss_val.item() * images.size(0)) + l1_loss_val.item() + ot_flow_loss_val.item())

                cnt_val+=1

        if torch.isnan(ot_flow_loss_val):
                print("Loss val is nan")
                return -1

        
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        train_loss_mse = train_mse_loss_total / len(train_loader)
        train_losses_mse.append(train_loss_mse)
        train_loss_l1 = train_l1_loss_total / len(train_loader)
        train_losses_l1.append(train_loss_l1)
        train_ot_flow_loss = train_ot_flow_loss_total / len(train_loader)
        train_ot_flow_losses.append(train_ot_flow_loss)

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_loss_mse = val_mse_loss_total / len(val_loader)
        val_losses_mse.append(val_loss_mse)
        val_loss_l1 = val_l1_loss_total / len(val_loader)
        val_losses_l1.append(val_loss_l1)
        val_ot_flow_loss = val_ot_flow_loss_total / len(val_loader)
        val_ot_flow_losses.append(val_ot_flow_loss)



        output ='Epoch: {} \tTraining Loss: {:.6f} \t  Training MSE Loss: {:.6f} \t  Training l1 Loss: {:.6f} \t  Training OT_Flow Loss: {:.6f}'.format(
            epoch,
            train_loss,
            train_loss_mse,
            train_loss_l1,
            train_ot_flow_loss,
        )
        print(output)
        with open(f'experiments/cnf/large/txt_train_logs/training_log_{custom_name}.txt', 'a') as file:
            file.write(output + '\n')

        output ='Epoch: {} \tValidation Loss: {:.6f} \t  Validation MSE Loss: {:.6f} \t  Validation l1 Loss: {:.6f} \t  Validation OT_Flow Loss: {:.6f}'.format(
            epoch,
            val_loss,
            val_loss_mse,
            val_loss_l1,
            val_ot_flow_loss,
        )
        print(output)
        with open(f'experiments/cnf/large/txt_train_logs/training_log_{custom_name}.txt', 'a') as file:
            file.write(output + '\n')

        #plot masks
        if epoch % 10 == 0:
            with torch.no_grad():
                mask_image_2 = (torch.sigmoid(net.mask.logits) > 0.5).float().cpu().numpy().reshape(28, 28)
                mask_image = torch.sigmoid(net.mask.logits).float().cpu().numpy().reshape(28, 28)
                plt.imsave(f'mask/{custom_name}_mask_epoch_{epoch}.png', mask_image, cmap='gray')
                plt.imsave(f'mask/{custom_name}_maskbi_epoch_{epoch}.png', mask_image_2, cmap='gray')

        # plot autoencoder output
        if epoch % 10 == 0:
            net.eval()
            sSavePath = os.path.join(save_dir, 'figs', custom_name + '_autoencoder{:d}.png'.format(epoch))
            xRecreate, *_ = net(images, conditions)
            conv = net.type_net_1 == "conv"
            plot_auto_enc(images, xRecreate, sSavePath, conv=conv)

        # plot ccnf output
        if epoch % 10 == 0:
            with torch.no_grad():
                    net.eval()
                    test_data = next(iter(test_loader))
                    images, fourier_images, *_ = test_data
                    conditions = None
                    if net.type_net_1 == "linear":
                        images = images.view(images.size(0), -1)
                    else:
                        images = images.view(images.size(0), 1, img_shape[0], img_shape[1])

                    if net.type_net_2 == "linear":
                        conditions = fourier_images.view(images.size(0), -1)
                    else:
                        conditions = fourier_images.view(images.size(0), 1, cond_shape[0], cond_shape[1])
                    
                    images = cvt(images)
                    conditions = cvt(conditions)
                    
                    masked_conditions, _ = net.mask(conditions)
                    x0 = net.autoenc.encode_orig(images)
                    y = net.autoenc.encode_fourier(masked_conditions)
                    combined_encoded_tensor = torch.cat((x0, y), dim=1)
                    normalized_tensor = (combined_encoded_tensor - net.autoenc.mu) / (net.autoenc.std + net.eps)
                    p_samples, y = normalized_tensor.chunk(2, dim=1)

                    nSamples = p_samples.shape[0]
                    dx= net.encoding_dim
                    z = cvt(torch.randn(nSamples, dx))  # Sampling from the standard normal (rho_1)

                    nSamples = 48  
                    y_samples = fourier_images[0:nSamples, 0, :, :]
          
                    indir = masked_conditions[0:nSamples]
                    
                    ifft_transformed = torch.fft.ifft2(y_samples).real
                    combined_images = torch.cat((ifft_transformed, y_samples), dim=1)
                    combined_images = combined_images.view(combined_images.size(0), -1).real
                    print(y_samples.shape, 'fourier', ifft_transformed.shape, 'iff', combined_images.shape, 'combined', combined_images.dtype, 'IFFT dtype')
                    p_samples = p_samples[0:nSamples, :]
                    z = z[0:nSamples, :]
                    y = y[0:nSamples, :]
              
    
                    sPath = os.path.join(save_dir, 'figs', custom_name + '_model_generations{:d}.png'.format(epoch))
                    sPath2 = os.path.join(save_dir, 'figs', custom_name + '_gray_model_generations{:d}.png'.format(epoch))
                    genModel = integrate(z, y, net.Phi, [1.0, 0.0], net.nt, stepper="rk4", alph=net.Phi.alph)

                    genModel = genModel[:, 0:dx]
                    combined_gen = torch.cat((genModel, y), dim=1)
                    #combined_p_samples = torch.cat((p_samples, y), dim=1)

                    genDecoded = net.autoenc.decode(combined_gen * (net.autoenc.std + net.eps) + net.autoenc.mu) 
                    #pDecoded = net.autoenc.decode(combined_p_samples * (net.autoenc.std + net.eps) + net.autoenc.mu) 
            
                    #plot_auto_enc(combined_images[:, :28*28], genDecoded[:, :28*28], sPath)
                    plot_gens(indir, combined_images[:, :28*28], genDecoded[:, :28*28], sPath)
                    plot_gens(indir, combined_images[:, :28*28], genDecoded[:, :28*28], sPath2, grayscale=True)
                    #print(mmd(combined_images[:, :28*28].detach().cpu().numpy(), genDecoded[:, :28*28].detach().cpu().numpy()))
                    
        
        # mmd
        if epoch % 1 == 0:
            with torch.no_grad():
                    net.eval()
                    test_data = next(iter(test_loader))
                    images, fourier_images, *_ = test_data
                    conditions = None
                    if net.type_net_1 == "linear":
                        images = images.view(images.size(0), -1)
                    else:
                        images = images.view(images.size(0), 1, img_shape[0], img_shape[1])

                    if net.type_net_2 == "linear":
                        conditions = fourier_images.view(images.size(0), -1)
                    else:
                        conditions = fourier_images.view(images.size(0), 1, cond_shape[0], cond_shape[1])
                    
                    images = cvt(images)
                    conditions = cvt(conditions)
                    
                    masked_conditions, _ = net.mask(conditions)
                    x0 = net.autoenc.encode_orig(images)
                    y = net.autoenc.encode_fourier(masked_conditions)
                    combined_encoded_tensor = torch.cat((x0, y), dim=1)
                    normalized_tensor = (combined_encoded_tensor - net.autoenc.mu) / (net.autoenc.std + net.eps)
                    p_samples, y = normalized_tensor.chunk(2, dim=1)

                    nSamples = p_samples.shape[0]
                    dx= net.encoding_dim
                    z = cvt(torch.randn(nSamples, dx))  # Sampling from the standard normal (rho_1)

                    nSamples = 800  
                    y_samples = fourier_images[0:nSamples, 0, :, :]
          
                    indir = masked_conditions[0:nSamples]
                    
                    ifft_transformed = torch.fft.ifft2(y_samples).real
                    combined_images = torch.cat((ifft_transformed, y_samples), dim=1)
                    combined_images = combined_images.view(combined_images.size(0), -1).real
                    print(y_samples.shape, 'fourier', ifft_transformed.shape, 'iff', combined_images.shape, 'combined', combined_images.dtype, 'IFFT dtype')
                    p_samples = p_samples[0:nSamples, :]
                    z = z[0:nSamples, :]
                    y = y[0:nSamples, :]
            
                    genModel = integrate(z, y, net.Phi, [1.0, 0.0], net.nt, stepper="rk4", alph=net.Phi.alph)
                    genModel = genModel[:, 0:dx]
                    combined_gen = torch.cat((genModel, y), dim=1)
                    genDecoded = net.autoenc.decode(combined_gen * (net.autoenc.std + net.eps) + net.autoenc.mu) 
                   
                    mmd_val = mmd(combined_images[:, :28*28].detach().cpu().numpy(), genDecoded[:, :28*28].detach().cpu().numpy()) 
                    mmds.append(mmd_val)

                    
                    output ='Epoch: {} \tTest Batch MMD: {}'.format(
                        epoch,
                        mmd_val
                    )
                    print(output)
                    with open(f'experiments/cnf/large/txt_train_logs/training_log_{custom_name}.txt', 'a') as file:
                        file.write(output + '\n')
                    

        if epoch % 150 == 0:
            for p in optimizer_autoenc.param_groups:
                p['lr'] /= 10.0
            print("lr: ", p['lr'])
        if epoch % 150 == 0:
            for p in optimizer_cnf.param_groups:
                p['lr'] /= 10.0
            print("lr: ", p['lr'])
        if epoch % 150 == 0:
            for p in optimizer_mask.param_groups:
                p['lr'] /= 10.0
            print("lr: ", p['lr'])

    train_losses = np.array(train_losses)
    train_losses_mse = np.array(train_losses_mse)
    train_losses_l1 = np.array(train_losses_l1)
    train_ot_flow_losses = np.array(train_ot_flow_losses)

    output ='Minimum Training Total Loss: {} at epoch {} \n Minimum Training OTFlow Loss: {} at epoch {} \n Minimum Training MSE Loss: {} at epoch {} \n Minimum Training L1 Loss: {} at epoch {}'.format(
            np.min(train_losses),
            np.argmin(train_losses)+1,
            np.min(train_ot_flow_losses),
            np.argmin(train_ot_flow_losses)+1,
            np.min(train_losses_mse),
            np.argmin(train_losses_mse)+1,
            np.min(train_losses_l1),
            np.argmin(train_losses_l1)+1,
        )
    print(output)
    with open(f'experiments/cnf/large/txt_train_logs/training_log_{custom_name}.txt', 'a') as file:
        file.write(output + '\n')
    
    output ='Maximum Training Total Loss: {} at epoch {} \n Maximum Training OTFlow Loss: {} at epoch {} \n Maximum Training MSE Loss: {} at epoch {} \n Maximum Training L1 Loss: {} at epoch {}'.format(
            np.max(train_losses),
            np.argmax(train_losses)+1,
            np.max(train_ot_flow_losses),
            np.argmax(train_ot_flow_losses)+1,
            np.max(train_losses_mse),
            np.argmax(train_losses_mse)+1,
            np.max(train_losses_l1),
            np.argmax(train_losses_l1)+1,
        )
    print(output)
    with open(f'experiments/cnf/large/txt_train_logs/training_log_{custom_name}.txt', 'a') as file:
        file.write(output + '\n')

    val_losses = np.array(val_losses)
    val_losses_mse = np.array(val_losses_mse)
    val_losses_l1 = np.array(val_losses_l1)
    val_ot_flow_losses = np.array(val_ot_flow_losses)

    output ='Minimum Validation Total Loss: {} at epoch {} \n Minimum Validation OTFlow Loss: {} at epoch {} \n Minimum Validation MSE Loss: {} at epoch {} \n Minimum Validation L1 Loss: {} at epoch {}'.format(
            np.min(val_losses),
            np.argmin(val_losses)+1,
            np.min(val_ot_flow_losses),
            np.argmin(val_ot_flow_losses)+1,
            np.min(val_losses_mse),
            np.argmin(val_losses_mse)+1,
            np.min(val_losses_l1),
            np.argmin(val_losses_l1)+1,
        )
    print(output)
    with open(f'experiments/cnf/large/txt_train_logs/training_log_{custom_name}.txt', 'a') as file:
        file.write(output + '\n')
    
    output ='Maximum Validation Total Loss: {} at epoch {} \n Maximum Validation OTFlow Loss: {} at epoch {} \n Maximum Validation MSE Loss: {} at epoch {} \n Maximum Validation L1 Loss: {} at epoch {}'.format(
            np.max(val_losses),
            np.argmax(val_losses)+1,
            np.max(val_ot_flow_losses),
            np.argmax(val_ot_flow_losses)+1,
            np.max(val_losses_mse),
            np.argmax(val_losses_mse)+1,
            np.max(val_losses_l1),
            np.argmax(val_losses_l1)+1,
        )
    print(output)
    with open(f'experiments/cnf/large/txt_train_logs/training_log_{custom_name}.txt', 'a') as file:
        file.write(output + '\n')

    try:
        mmds = np.array(mmds)
        plot_single_loss_array(mmds, f"MMD of Test Batch Size 800 | Minimum: {min(mmds)}", save_dir + str("/") + custom_name + "_test_mdd_graph.png")
    except:
        print("plot 1 failed")
    try: 
        plot_loss_values_4_grid(train_losses, 
                                train_losses_mse, 
                                train_losses_l1, 
                                train_ot_flow_losses, 
                                "Total Training Loss",
                                "MSE Autoencoder Training Loss",
                                "L1 Mask Training Loss",
                                "CCNF Training Loss",
                                "Training Losses",
                                save_dir + str("/") + custom_name + "_train_graph.png")
    except:
        print("plot 2 failed")
    try:
        plot_loss_values_4_grid(val_losses, 
                                val_losses_mse, 
                                val_losses_l1, 
                                val_ot_flow_losses, 
                                "Total Validation Loss",
                                "MSE Autoencoder Validation Loss",
                                "L1 Mask Validation Loss",
                                "CCNF Validation Loss",
                                "Validition Losses",
                                save_dir + str("/") + custom_name + "_validation_graph.png")
    except:
        print("plot 3 failed")
    try:
        plot_loss_values_4_grid_train_val(train_losses, 
                                        val_losses,
                                        train_losses_mse, 
                                        val_losses_mse, 
                                        train_losses_l1,
                                        val_losses_l1,  
                                        train_ot_flow_losses,
                                        val_ot_flow_losses, 
                                        "Total Loss",
                                        "MSE Autoencoder Loss",
                                        "L1 Mask Loss",
                                        "CCNF Loss",
                                        "Training and Validation Losses",
                                        save_dir + str("/") + custom_name + "_train_and_validation_graph.png")
    except:
        print("plot 4 failed")
    
    return 0      