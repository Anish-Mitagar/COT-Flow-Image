import argparse
import os
import datetime
import lib.utils as utils

from datasets.mnist  import getLoader
from src.plotter import *
from src.OTFlowProblem import *
from src.prototyping.Autoencoders import *
from src.prototyping.ProbabilisticMask import *
from src.prototyping.MaskAutoEncCNF import *
from src.prototyping.MaskDesigns import *
import config

cf = config.getconfig()

if cf.gpu:
    def_viz_freq = 100
    def_batch    = 800
    def_niters   = 1000 # changed from 50000
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
parser.add_argument('--autoenc'    , type=str, default=None)
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
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    prec = torch.float64
    print("device: ", device)

    train_loader, val_loader, test_loader = getLoader(args.data, args.batch_size, args.val_batch_size, augment=False, hasGPU=cf.gpu, conditional=args.conditional)

    d = args.d # encoded dimensions
    dx = 128
    dy = 128
    # -----------AutoEncoder ------------------------------------------------------------
    if args.autoenc is None: # if no trained encoder-decoder is provided, then train one
        auto_enc = DualLinearAutoencoder(original_dim=784, encoding_dim=128)
        mask = ProbabilisticMask(input_dim=784, image_shape=(28, 28), mask=RandomMask(784))
        phi = Phi(nTh=2, m=args.m, dx=dx,dy=dy, alph=args.alph)
        model = MaskAutoEncCNF(original_dim = 784, encoding_dim = 128, mask = mask, autoenc = auto_enc, Phi = phi, nt = 2, eps =args.eps)
        
        model = model.type(prec).to(device)
        print(model)
        mask_type='random'
        n_epochs = 100
        lr=0.0001
        auto_enc_design = 'DualLinearAutoencoder'
        l1_mask_reg = 0.01
        custom_name_str = mask_type + "_mask_" + "n_epochs_" + str(n_epochs) + "_lr_" + str(lr) + "_aed_" + auto_enc_design + "_l1_mask_reg_" + str(l1_mask_reg) + "_updated_model_run"
        result = trainMaskAutoEncCNF(model, train_loader, val_loader, test_loader, n_epochs = n_epochs, lr=lr, l1_mask_reg = l1_mask_reg, img_shape = (28, 28), cond_shape= (28, 28), custom_name=custom_name_str, argType=prec, device=device)

