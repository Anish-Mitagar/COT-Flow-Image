# trainToyOTflow.py
# training driver for the two-dimensional toy problems
import argparse
import os
import time
import datetime
import torch.optim as optim
import numpy as np
import math
import lib.toy_data as toy_data
import lib.utils as utils
from lib.utils import count_parameters
from src.plotter import plot4, plot4mnist
from src.OTFlowProblem import *
import config

cf = config.getconfig()

if cf.gpu: # if gpu on platform
    def_viz_freq = 100
    def_batch    = 4096
    def_niter    = 1500
else:  # if no gpu on platform, assume debugging on a local cpu
    def_viz_freq = 100
    def_batch    = 2048
    def_niter    = 1000

parser = argparse.ArgumentParser('OT-Flow')
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='8gaussians'
)

parser.add_argument("--nt"    , type=int, default=8, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=8, help="number of time steps for validation")
parser.add_argument('--alph'  , type=str, default='1.0,100.0,5.0')
parser.add_argument('--m'     , type=int, default=32)
parser.add_argument('--nTh'   , type=int, default=2)

parser.add_argument('--niters'        , type=int  , default=def_niter)
parser.add_argument('--batch_size'    , type=int  , default=def_batch)
parser.add_argument('--val_batch_size', type=int  , default=def_batch)

parser.add_argument('--lr'          , type=float, default=0.1)
parser.add_argument("--drop_freq"   , type=int  , default=100, help="how often to decrease learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr_drop'     , type=float, default=2.0)
parser.add_argument('--optim'       , type=str  , default='adam', choices=['adam'])
parser.add_argument('--prec'        , type=str  , default='single', choices=['single','double'], help="single or double precision")

parser.add_argument('--save'    , type=str, default='experiments/cnf/toy')
parser.add_argument('--viz_freq', type=int, default=def_viz_freq)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--gpu'     , type=int, default=0)
parser.add_argument('--sample_freq', type=int, default=25)


args = parser.parse_args()

args.alph = [float(item) for item in args.alph.split(',')]

# get precision type
if args.prec =='double':
    prec = torch.float64
else:
    prec = torch.float32

# get timestamp for saving models
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info("start time: " + start_time)
logger.info(args)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


def compute_loss(net, x, y, nt):
    Jc , cs = OTFlowProblem(x, y, net, [0,1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs


def generate_noisy_fourier_data(data, batch_size):
    if data == "8gaussians":
        rng = np.random.RandomState()
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414

        # Compute noisy Fourier transform
        dataset_fft = np.fft.fft2(dataset, axes=(0, 1))
        noise = rng.normal(0, 1, dataset_fft.shape)
        noisy_fft = dataset_fft + noise
        y = np.real(np.fft.ifft2(noisy_fft, axes=(0, 1)))

        return dataset, y

if __name__ == '__main__':

    torch.set_default_dtype(prec)
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    # neural network for the potential function Phi
    dx     = 2  # Updated to match the actual data dimension
    dy     = 2  # Updated to match the noisy Fourier transformed data dimension
    alph   = args.alph
    nt     = args.nt
    nt_val = args.nt_val
    nTh    = args.nTh
    m      = args.m
    net = Phi(nTh=nTh, m=args.m, dx=dx, dy=dy, alph=alph)
    net = net.to(prec).to(device)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay ) # lr=0.04 good

    logger.info(net)
    logger.info("-------------------------")
    logger.info("dx={:} dy={:}  m={:}  nTh={:}   alpha={:}".format(dx,dy,m,nTh,alph))
    logger.info("nt={:}   nt_val={:}".format(nt,nt_val))
    logger.info("Number of trainable parameters: {}".format(count_parameters(net)))
    logger.info("-------------------------")
    logger.info(str(optim)) # optimizer info
    logger.info("data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu))
    logger.info("maxIters={:} val_freq={:} viz_freq={:}".format(args.niters, args.val_freq, args.viz_freq))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("-------------------------\n")

    end = time.time()
    best_loss = float('inf')
    bestParams = None

    # setup data [nSamples, d]
    # use one batch as the entire data set
    x, y = generate_noisy_fourier_data(args.data, batch_size=args.batch_size)
    x = cvt(torch.from_numpy(x))
    y = cvt(torch.from_numpy(y))
    print(x.shape)
    print(y.shape)
    # plot x and y
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], label='x (8 Gaussians)')
    plt.title('x (8 Gaussians)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y.cpu().numpy()[:, 0], y.cpu().numpy()[:, 1], label='y (Noisy FFT(x))')
    plt.title('y (Noisy FFT(x))')
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.legend()

    plt.tight_layout()
    plt.savefig('x_y_visualization.jpg')
    plt.show()

    xval, yval = generate_noisy_fourier_data(args.data, batch_size=args.val_batch_size)
    xval = cvt(torch.from_numpy(xval))
    yval = cvt(torch.from_numpy(yval))

    log_msg = (
        '{:5s}  {:6s}   {:9s}  {:9s}  {:9s}  {:9s}      {:9s}  {:9s}  {:9s}  {:9s}  '.format(
            'iter', ' time','loss', 'L (L_2)', 'C (loss)', 'R (HJB)', 'valLoss', 'valL', 'valC', 'valR'
        )
    )
    logger.info(log_msg)

    time_meter = utils.AverageMeter()

    net.train()
    for itr in range(1, args.niters + 1):
        # train
        optim.zero_grad()
        loss, costs  = compute_loss(net, x, y , nt=nt)
        loss.backward()
        optim.step()

        time_meter.update(time.time() - end)

        log_message = (
            '{:05d}  {:6.3f}   {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e}  '.format(
                itr, time_meter.val , loss, costs[0], costs[1], costs[2]
            )
        )

        # validate
        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                net.eval()
                test_loss, test_costs = compute_loss(net, xval, yval, nt=nt_val)

                # add to print message
                log_message += '    {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} '.format(
                    test_loss, test_costs[0], test_costs[1], test_costs[2]
                )

                # save best set of parameters
                if test_loss.item() < best_loss:
                    best_loss   = test_loss.item()
                    best_costs = test_costs
                    utils.makedirs(args.save)
                    best_params = net.state_dict()
                    torch.save({
                        'args': args,
                        'state_dict': best_params,
                    }, os.path.join(args.save, start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m)))
                    net.train()

        logger.info(log_message) # print iteration

        # create plots
        if itr % args.viz_freq == 0:
            with torch.no_grad():
                net.eval()
                curr_state = net.state_dict()
                net.load_state_dict(best_params)

                nSamples = 20000
                p_samples = cvt(torch.Tensor( toy_data.inf_train_gen(args.data, batch_size=nSamples) ))
                x_samples = p_samples[:,0:dx].view(-1,dx)
                y_samples = p_samples[:,dx:].view(-1,dy)
                z = cvt(torch.randn(nSamples,dx)) # sampling from the standard normal (rho_1)
                print(x_samples.shape, 'x_samples')
                sPath = os.path.join(args.save, 'figs', start_time + '_{:04d}.png'.format(itr))
                plot4mnist(x_samples, sPath, sTitle='')

                net.load_state_dict(curr_state)
                net.train()

        # shrink step size
        if itr % args.drop_freq == 0:
            for p in optim.param_groups:
                p['lr'] /= args.lr_drop
            print("lr: ", p['lr'])

        # resample data
        if itr % args.sample_freq == 0:
            # resample data [nSamples, d+1]
            logger.info("resampling")
            x, y = generate_noisy_fourier_data(args.data, batch_size=args.batch_size)
            x = cvt(torch.from_numpy(x))
            y = cvt(torch.from_numpy(y))


        end = time.time()

    logger.info("Training Time: {:} seconds".format(time_meter.sum))
    logger.info('Training has finished.  ' + start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m))
