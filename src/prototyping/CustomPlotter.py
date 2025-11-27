import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import math 
import numpy as np

def plot_and_save_loss_values(loss_values, names, filename):
    
    epochs = range(1, len(loss_values[0]) + 1)  

    for i, loss in enumerate(loss_values):
        plt.plot(epochs, loss, label=names[i])

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss Values ({names[i]}) Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename + f"_{names[i]}.png")
        plt.close() 

def plot_auto_enc(x, xRecreate, sPath, conv = False):

    # assume square image
    if conv:
        s = int(x.shape[2])
    else:
        s = int(math.sqrt(x.shape[1]))

    nex = 8

    fig, axs = plt.subplots(4, nex//2)
    fig.set_size_inches(9, 9)
    fig.suptitle("first 2 rows originals. Rows 3 and 4 are generations.")

    for i in range(nex//2):
        axs[0, i].imshow(x[i,:].reshape(s,s).detach().cpu().numpy())
        axs[1, i].imshow(x[ nex//2 + i , : ].reshape(s,s).detach().cpu().numpy())
        axs[2, i].imshow(xRecreate[i,:].reshape(s,s).detach().cpu().numpy())
        axs[3, i].imshow(xRecreate[ nex//2 + i , : ].reshape(s, s).detach().cpu().numpy())

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

def plot_gens(indir, x, xRecreate, sPath, conv = False, grayscale = False):

    # assume square image
    if conv:
        s = int(x.shape[2])
    else:
        s = int(math.sqrt(x.shape[1]))

    nex = 48

    fig, axs = plt.subplots(nex//4, nex//4)
    fig.set_size_inches(27, 27)
    fig.suptitle("First 4 rows: Test Ground Truths \n Next 4 rows: Corresponding Observations \n Last 4 rows: Corresponding Recontructions from Observations", fontsize=52)

    if grayscale:
        for i in range(nex//4):
            axs[0, i].imshow(x[i,:].reshape(s,s).detach().cpu().numpy(), cmap='gray')
            axs[1, i].imshow(x[ nex//4 + i , : ].reshape(s,s).detach().cpu().numpy(), cmap='gray')
            axs[2, i].imshow(x[ nex//4 + nex//4 + i , : ].reshape(s,s).detach().cpu().numpy(), cmap='gray')
            axs[3, i].imshow(x[ nex//4 + nex//4 + nex//4 + i , : ].reshape(s,s).detach().cpu().numpy(), cmap='gray')
            axs[4, i].imshow(indir[ i , : ].reshape(s,s).detach().cpu().numpy(), cmap='gray')
            axs[5, i].imshow(indir[ nex//4 + i , : ].reshape(s,s).detach().cpu().numpy(), cmap='gray')
            axs[6, i].imshow(indir[ nex//4 + nex//4 + i , : ].reshape(s,s).detach().cpu().numpy(), cmap='gray')
            axs[7, i].imshow(indir[ nex//4 + nex//4 + nex//4 + i , : ].reshape(s,s).detach().cpu().numpy(), cmap='gray')
            axs[8, i].imshow(xRecreate[i,:].reshape(s,s).detach().cpu().numpy(), cmap='gray')
            axs[9, i].imshow(xRecreate[ nex//4 + i , : ].reshape(s, s).detach().cpu().numpy(), cmap='gray')
            axs[10, i].imshow(xRecreate[ nex//4 + nex//4 + i ,:].reshape(s,s).detach().cpu().numpy(), cmap='gray')
            axs[11, i].imshow(xRecreate[ nex//4 + nex//4 + nex//4 + i , : ].reshape(s, s).detach().cpu().numpy(), cmap='gray')
    else:
        for i in range(nex//4):
            axs[0, i].imshow(x[i,:].reshape(s,s).detach().cpu().numpy())
            axs[1, i].imshow(x[ nex//4 + i , : ].reshape(s,s).detach().cpu().numpy())
            axs[2, i].imshow(x[ nex//4 + nex//4 + i , : ].reshape(s,s).detach().cpu().numpy())
            axs[3, i].imshow(x[ nex//4 + nex//4 + nex//4 + i , : ].reshape(s,s).detach().cpu().numpy())
            axs[4, i].imshow(indir[ i , : ].reshape(s,s).detach().cpu().numpy())
            axs[5, i].imshow(indir[ nex//4 + i , : ].reshape(s,s).detach().cpu().numpy())
            axs[6, i].imshow(indir[ nex//4 + nex//4 + i , : ].reshape(s,s).detach().cpu().numpy())
            axs[7, i].imshow(indir[ nex//4 + nex//4 + nex//4 + i , : ].reshape(s,s).detach().cpu().numpy())
            axs[8, i].imshow(xRecreate[i,:].reshape(s,s).detach().cpu().numpy())
            axs[9, i].imshow(xRecreate[ nex//4 + i , : ].reshape(s, s).detach().cpu().numpy())
            axs[10, i].imshow(xRecreate[ nex//4 + nex//4 + i ,:].reshape(s,s).detach().cpu().numpy())
            axs[11, i].imshow(xRecreate[ nex//4 + nex//4 + nex//4 + i , : ].reshape(s, s).detach().cpu().numpy())
        

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

def plot_loss_values_4_grid_train_val(arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, tit1, tit2, tit3, tit4, big_tit, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    array_pairs = [(arr1, arr2), (arr3, arr4), (arr5, arr6), (arr7, arr8)]
    titles = [tit1, tit2, tit3, tit4]

    for ax, (arr_a, arr_b), title in zip(axes.flatten(), array_pairs, titles):
        epochs = np.arange(1, len(arr_a) + 1)
        line_a, = ax.plot(epochs, arr_a, label='Training')
        # color_a = line_a.get_color()
        # min_idx_a = np.argmin(arr_a)
        # min_val_a = arr_a[min_idx_a]
        # ax.plot(epochs[min_idx_a], min_val_a, 'o', color=color_a) 
        # ax.annotate(f'Epoch: {min_idx_a+1}\nMin: {min_val_a:.2f}', xy=(epochs[min_idx_a], min_val_a), xytext=(epochs[min_idx_a], min_val_a + 0.05),
        #             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, ha='center')
        # max_idx_a = np.argmax(arr_a)
        # max_val_a = arr_a[max_idx_a]
        # ax.plot(epochs[max_idx_a], max_val_a, 'o', color=color_a)
        # ax.annotate(f'Epoch: {max_idx_a+1}\nMax: {max_val_a:.2f}', xy=(epochs[max_idx_a], max_val_a), xytext=(epochs[max_idx_a], max_val_a - 0.05),
        #             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, ha='center')
        line_b, = ax.plot(epochs, arr_b, label='Validation')
        # color_b = line_b.get_color()
        # min_idx_b = np.argmin(arr_b)
        # min_val_b = arr_b[min_idx_b]
        # ax.plot(epochs[min_idx_b], min_val_b, 'o', color=color_b) 
        # ax.annotate(f'Epoch: {min_idx_b+1}\nMin: {min_val_b:.2f}', xy=(epochs[min_idx_b], min_val_b), xytext=(epochs[min_idx_b], min_val_b + 0.05),
        #             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, ha='center')
        # max_idx_b = np.argmax(arr_b)
        # max_val_b = arr_b[max_idx_b]
        # ax.plot(epochs[max_idx_b], max_val_b, 'o', color=color_b) 
        # ax.annotate(f'Epoch: {max_idx_b+1}\nMax: {max_val_b:.2f}', xy=(epochs[max_idx_b], max_val_b), xytext=(epochs[max_idx_b], max_val_b - 0.05),
        #             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, ha='center')
        ax.set_title(title)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
    fig.suptitle(big_tit, fontsize=20)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_loss_values_4_grid(arr1, arr2, arr3, arr4, tit1, tit2, tit3, tit4, big_tit, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    arrays = [arr1, arr2, arr3, arr4]
    titles = [tit1, tit2, tit3, tit4]

    for ax, arr, title in zip(axes.flatten(), arrays, titles):
        epochs = np.arange(1, len(arr) + 1)
        line, = ax.plot(epochs, arr, label='Loss')
        # color = line.get_color()
        # min_idx = np.argmin(arr)
        # min_val = arr[min_idx]
        # ax.plot(epochs[min_idx], min_val, 'o', color=color)  
        # ax.annotate(f'Epoch: {min_idx+1}\nMin: {min_val:.2f}', xy=(epochs[min_idx], min_val), xytext=(epochs[min_idx], min_val + 0.05),
        #             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, ha='center')
        # max_idx = np.argmax(arr)
        # max_val = arr[max_idx]
        # ax.plot(epochs[max_idx], max_val, 'o', color=color)  
        # ax.annotate(f'Epoch: {max_idx+1}\nMax: {max_val:.2f}', xy=(epochs[max_idx], max_val), xytext=(epochs[max_idx], max_val - 0.05),
        #             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, ha='center')
        ax.set_title(title)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
    fig.suptitle(big_tit, fontsize=20)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_single_loss_array(arr, big_tit, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.arange(1, len(arr) + 1)
    line, = ax.plot(epochs, arr, label='Loss')
    ax.set_title(big_tit, fontsize=16)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()