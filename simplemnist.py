# trainMnistOTflow.py
# train the MNIST model with the encoder-decoder structure
import argparse
import os
import time
import datetime
import torch.optim as optim
import math
from lib import dataloader as dl
import lib.utils as utils
from lib.utils import count_parameters
import datasets
from datasets.mnist import getLoader
from src.plotter import *
from src.OTFlowProblem import *
import config

cf = config.getconfig()

if cf.gpu:
    def_viz_freq = 100
    def_batch    = 200
    def_niters   = 5000
    def_m        = 128
    def_val_freq = 20
else: # if no gpu on platform, assume debugging on a local cpu
    def_viz_freq = 4
    def_batch    = 20
    def_niters   = 40
    def_val_freq = 1
    def_m        = 16

parser = argparse.ArgumentParser('OT-Flow')
parser.add_argument(
    '--data', choices=['mnist'], type=str, default='mnist'
)
parser.add_argument("--nt"    , type=int, default=8, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=16, help="number of time steps for validation")
parser.add_argument('--alph'  , type=str, default='1.0,80.0,500.0')
parser.add_argument('--m'     , type=int, default=def_m)
parser.add_argument('--d'     , type=int, default=128) # encoded dimension

parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr'          , type=float, default=0.008)
parser.add_argument('--drop_freq'   , type=int,   default=5000, help="how often to decrease learning rate")
parser.add_argument('--lr_drop'     , type=float, default=10.0**(0.5), help="how much to decrease learning rate (divide by)")
parser.add_argument('--eps'         , type=float, default=10**-6)

parser.add_argument('--niters'     , type=int, default=def_niters)
parser.add_argument('--batch_size' , type=int, default=def_batch)
parser.add_argument('--val_batch_size', type=int, default=def_batch)
parser.add_argument('--resume'     , type=str, default=None)
parser.add_argument('--save'       , type=str, default='experiments/cnf/large')
parser.add_argument('--viz_freq'   , type=int, default=def_viz_freq)
parser.add_argument('--val_freq'   , type=int, default=def_val_freq)
parser.add_argument('--gpu'        , type=int, default=0)
parser.add_argument('--conditional', type=int, default=-1) # -1 means unconditioned
args = parser.parse_args()

args.alph = [float(item) for item in args.alph.split(',')]

# add timestamp to save path
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info("start time: " + start_time)
logger.info(args)

val_batch_size = args.val_batch_size if args.val_batch_size else args.batch_size
# device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_loss(net, x, y, nt): 
    Jc , costs = OTFlowProblem(x, y, net, [0,1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, costs

def generate_noisy_fourier_data(x):
    """
    Generate y as the noisy Fourier transform of x.
    """
    x_np = x.cpu().numpy()
    x_fft = np.fft.fft2(x_np)
    noise = np.random.normal(0, 0.01, x_fft.shape)
    y = np.real(x_fft + noise)
    y = torch.from_numpy(y).to(x.device)
    return y

if __name__ == '__main__':

    prec = torch.float64
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)
    print("device: ", device)

    train_loader, val_loader, _ = getLoader(args.data, args.batch_size, args.val_batch_size, augment=False, hasGPU=cf.gpu, conditional=args.conditional)

    d = args.d # encoded dimensions
    dx = args.d # encoded dimensions
    dy = args.d # encoded dimensions
    nt  = args.nt
    nt_val = args.nt_val
    nTh = 2
    m   = args.m

    net = Phi(nTh=nTh, m=m, dx=dx,dy=dy, alph=args.alph) # the phi aka the value function
    net = net.to(prec).to(device)

    if args.val_freq == 0:
        # if val_freq set to 0, then validate after every epoch....assume mnist train 50000
        args.val_freq = math.ceil( 50000 /args.batch_size)

    # ADAM optimizer
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.info(net)
    logger.info("-------------------------")
    logger.info("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d,m,nTh,net.alph))
    logger.info("nt={:}   nt_val={:}".format(nt,nt_val))
    logger.info("Number of trainable parameters: {}".format(count_parameters(net)))
    logger.info("-------------------------")
    logger.info(str(optim)) # optimizer info
    logger.info("data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu))
    logger.info("maxIters={:} val_freq={:} viz_freq={:}".format(args.niters, args.val_freq, args.viz_freq))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("-------------------------\n")

    begin = time.time()
    end   = begin
    best_loss   = float('inf')
    best_costs  = [0.0]*3
    best_params = None

    log_msg = (
        '{:5s}  {:6s}  {:7s}   {:9s}  {:9s}  {:9s}  {:9s}     {:9s}  {:9s}  {:9s}  {:9s} '.format(
            'iter', ' time','lr','loss', 'L (L_2)', 'C (loss)', 'R (HJB)', 'valLoss', 'valL', 'valC', 'valR'
        )
    )
    logger.info(log_msg)

    timeMeter = utils.AverageMeter()
    clampMax = 2.0
    clampMin = -2.0

    net.train()
    itr = 1
    while itr < args.niters:
        # train
        for data in train_loader:
            images, _ = data
            # flatten images
            x0 = images.view(images.size(0), -1)
            # y = generate_noisy_fourier_data(images)
            y = images.view(images.size(0), -1)
            # visualize y
            import matplotlib.pyplot as plt
            plt.imshow(images[0,0].cpu().numpy(), cmap='gray')
            plt.savefig('output.png')

            # y = y.view(y.size(0), -1)
            x0 = cvt(x0)
            y = cvt(y)
            print("x0 shape: ", x0.shape, "y shape: ", y.shape)
            optim.zero_grad()

            # clip parameters
            for p in net.parameters():
                p.data = torch.clamp(p.data, clampMin, clampMax)

            loss, costs  = compute_loss(net, x0, y, nt=nt)
            loss.backward()
            optim.step()
            timeMeter.update(time.time() - end)

            log_message = (
                '{:05d}  {:6.3f}  {:7.1e}   {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} '.format(
                    itr, timeMeter.val, optim.param_groups[0]['lr'], loss, costs[0], costs[1], costs[2]
                )
            )

            if torch.isnan(loss):
                logger.info(log_message)
                logger.info("NaN encountered....exiting prematurely")
                logger.info("Training Time: {:} seconds".format(timeMeter.sum))
                logger.info('File: ' + start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(
                        args.data, int(net.alph[1]), int(net.alph[2]), m)
                )
                exit(1)

            # validation
            if itr == 1 or itr % args.val_freq == 0 or itr == args.niters:
                net.eval()
                with torch.no_grad():

                    valLossMeter = utils.AverageMeter()
                    valAlphMeterL = utils.AverageMeter()
                    valAlphMeterC = utils.AverageMeter()
                    valAlphMeterR = utils.AverageMeter()

                    for data in val_loader:
                        images, _ = data
                        # flatten images
                        x0 = images.view(images.size(0), -1)
                        x0 = cvt(x0)

                        nex = x0.shape[0]
                        val_loss, val_costs = compute_loss(net, x0, y, nt=nt_val)
                        valLossMeter.update(val_loss.item(), nex)
                        valAlphMeterL.update(val_costs[0].item(), nex)
                        valAlphMeterC.update(val_costs[1].item(), nex)
                        valAlphMeterR.update(val_costs[2].item(), nex)

                        if not cf.gpu:  # for debugging
                            break

                    # add to print message
                    log_message += '    {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} '.format(
                        valLossMeter.avg, valAlphMeterL.avg, valAlphMeterC.avg, valAlphMeterR.avg
                    )

                    # save best set of parameters
                    if valLossMeter.avg < best_loss:
                        logger.info('saving new best')
                        best_loss = valLossMeter.avg
                        best_costs = [  valAlphMeterL.avg, valAlphMeterC.avg, valAlphMeterR.avg ]
                        utils.makedirs(args.save)
                        best_params = net.state_dict()
                        torch.save({
                            'args': args,
                            'state_dict': best_params,
                        }, os.path.join(args.save, start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(net.alph[1]),int(net.alph[2]),m)))
                    net.train()

            logger.info(log_message) # print iteration

            # create plots
            if itr % args.viz_freq == 0:
                with torch.no_grad():
                    net.eval()
                    currState = net.state_dict()
                    net.load_state_dict(best_params)

                    # plot one batch in R^d space
                    p_samples = next(iter(val_loader))[0] #mnist images

                    print("p_samples: ", p_samples.shape)
                    p_samples = p_samples.view(p_samples.size(0), -1)
                    p_samples = cvt(p_samples)

                    nSamples = p_samples.shape[0]
                    print("nSamples: ", p_samples.shape)

                    x_samples = p_samples[:,0:dx].view(-1,dx)
                    y_samples = p_samples[:,0:dx].view(-1,dx) #p_samples[:,dx:].view(-1,dy)

                    print(x_samples.shape, y_samples.shape, 'x_samples, y_samples')
                    z = cvt(torch.randn(nSamples,dx)) # sampling from the standard normal (rho_1)
                    
                    # y = cvt(torch.randn(nSamples,d)) # sampling from rho_1



                    sPath = os.path.join(args.save, 'figs', start_time + '_{:04d}.png'.format(itr))
                    plot4(net, p_samples, y, z, nt_val, sPath, sTitle='loss {:.2f}  ,  C {:.2f}'.format(best_loss, best_costs[1] ))

                    # # plot the Mnist images
                    nSamples = 8 # overwrite
                    p_samples = p_samples[0:nSamples,0:d]
                    z = z[0:nSamples,:]
                    y = y_samples[0:nSamples,:]

                    sPath = os.path.join(args.save, 'figs', start_time + '_class{:d}_imshow{:04d}.png'.format(args.conditional, itr))
                    genModel = integrate(z[:, 0:d], y[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph)
                    # print(z[:, 0:d].shape, y[:, 0:d].shape, 'z, y')
                    # print("genModel: ", genModel.shape)

                    plotAutoEnc(p_samples, genModel[:, 0:d], sPath)
                    net.load_state_dict(currState)
                    net.train()

            # shrink step size
            if itr % args.drop_freq == 0:
                for p in optim.param_groups:
                    p['lr'] /= args.lr_drop  # 10.0**(0.5)
                print("lr: ", p['lr'])

            itr += 1
            end = time.time()
            # end batch_iter

    logger.info("Training Time: {:} seconds".format(timeMeter.sum))
    logger.info('Training has finished.  ' + start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(net.alph[1]),int(net.alph[2]),m))
