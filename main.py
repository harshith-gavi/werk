import argparse
import math
import h5py

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import update_prob_estimates
from snn_models_LIF4_save4_l2 import * # 2 layers

from torchvision import datasets, transforms
import tonic

def data_mod(X, y, batch_size, step_size, input_size, max_time, shuffle=False):
    '''
    This function generates batches of sparse data from the SHD dataset
    '''
    labels = np.array(y, int)
    nb_batches = len(labels)//batch_size
    sample_index = np.arange(len(labels))

    firing_times = X['times']
    units_fired = X['units']

    time_bins = np.linspace(0, max_time, num=step_size)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    mod_data = []
    while counter<nb_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]

            coo[0].extend(batch)
            coo[2].extend(units)
            coo[1].extend(times)

        i = torch.LongTensor(coo).to(device_2)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device_2)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,step_size,input_size])).to(device_2)
        y_batch = torch.tensor(labels[batch_index], device = device_2)

        mod_data.append((X_batch.to(device_2), y_batch.to(device_2)))

        counter += 1

    return mod_data

def data_generator(dataset, batch_size, dataroot, shuffle=True):
    datapath = '../data/'

    if dataset == 'SHD':
        shd_train = h5py.File(datapath + 'train_data/SHD/shd_train.h5', 'r')
        shd_test = h5py.File(datapath + 'test_data/SHD/shd_test.h5', 'r')

        shd_train = data_mod(shd_train['spikes'], shd_train['labels'], batch_size = batch_size, step_size = 100, input_size = 700, max_time = 1.37)
        shd_test = data_mod(shd_test['spikes'], shd_test['labels'], batch_size = 1, step_size = 100, input_size = 700, max_time = 1.37)
        
        train_loader = shd_train
        test_loader = shd_test
        n_classes = 20
        seq_length = 100
        input_channels = 700
        
    elif dataset == 'MNIST-10':
        train_set = datasets.MNIST(root=dataroot, train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ]))
        test_set = datasets.MNIST(root=dataroot, train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor()
                                  ]))

        train_set = train_set

        train_loader = torch.utils.data.DataLoader(train_set, shuffle=shuffle, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=batch_size)
        n_classes = 10
        seq_length = 28*28
        input_channels = 1 

    else:
        print('Please provide a valid dataset name.')
        exit(1)
    return train_loader, test_loader, seq_length, input_channels, n_classes

def get_stats_named_params( model ):
    named_params = {}
    for name, param in model.named_parameters():
        sm, lm, dm = param.detach().clone(), 0.0*param.detach().clone(), 0.0*param.detach().clone()
        named_params[name] = (param, sm, lm, dm)
    return named_params


def post_optimizer_updates( named_params, args, epoch ):
    alpha = args.alpha
    beta = args.beta
    rho = args.rho
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        lm.data.add_( -alpha * (param - sm) )
        sm.data.mul_( (1.0-beta) )
        sm.data.add_( beta * param - (beta/alpha) * lm )

def get_regularizer_named_params( named_params, args, _lambda=1.0 ):
    alpha = args.alpha
    rho = args.rho
    regularization = torch.zeros( [], device=args.device )
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        regularization += (rho-1.) * torch.sum( param * lm )
        r_p = _lambda * 0.5 * alpha * torch.sum( torch.square(param - sm) )
        regularization += r_p
            # print(name,r_p)
    return regularization 

def reset_named_params(named_params, args):
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param.data.copy_(sm.data)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.to_dense()
        # data = data.view(-1, input_channels, seq_length)#[:,:,:700]

        with torch.no_grad():
            model.eval()

            hidden = model.init_hidden(data.size(0))
            
            outputs, hidden, recon_loss = model(data, hidden) 
            
            output = outputs[-1]
            test_loss += F.nll_loss(output, target, reduction='sum').data.item()
            pred = output.data.max(1, keepdim=True)[1]
        
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader)
    
    return test_loss, 100. * correct / len(test_loader)


def train(epoch, args, train_loader, n_classes, model, named_params,k):
    global steps
    global estimate_class_distribution

    batch_size = args.batch_size
    alpha = args.alpha
    beta = args.beta

    PARTS = args.parts
    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    total_oracle_loss = 0
    model.train()
    
    T = seq_length
    #entropy = EntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.to_dense()
        # data = data.view(-1, input_channels, seq_length)
  
        B = target.size()[0]
        step = model.network.step
        xdata = data.clone()
        pdata = data.clone()
        
        # T = inputs.size()[0]
 
        Delta = torch.zeros(B, dtype=xdata.dtype, device=xdata.device)
        
        _PARTS = PARTS
        # if (PARTS * step < T):
        #     _PARTS += 1
        h = model.init_hidden(xdata.size(0))
      
        p_range = range(_PARTS)
        # for p in p_range:
        #     # x = data[:,0,p:p+1].view(-1,1,1)
        #     x = data[:, p, :].view(1, 1, -1)
        #     print(x.shape)
        data = torch.split(data, split_size_or_sections=1, dim=1)
        for p in range(len(data)):
            x = data[p]
            if p==p_range[0]:
                h = model.init_hidden(xdata.size(0))
            else:
                h = tuple(v.detach() for v in h)
            
            if p<PARTS-1:
                if epoch <0:
                    if args.per_ex_stats:
                        oracle_prob = estimatedDistribution[batch_idx*batch_size:(batch_idx+1)*batch_size, p]
                    else:
                        oracle_prob = 0*estimate_class_distribution[target, p] + (1.0/n_classes)
                else:
                    oracle_prob = estimate_class_distribution[target, p]
            else:
                oracle_prob = F.one_hot(target).float() 

            
            o, h,hs = model.network.forward(x, h ,p)
            # print(os[-1].shape,h[-1].shape,hs[-1][-1].shape)
            # print(h[-1],os[-1])
            # print(x.shape)
            # print(os.shape)

            prob_out = F.softmax(h[-1], dim=1)
            output = F.log_softmax(h[-1], dim=1) 

            if p<PARTS-1:
                with torch.no_grad():
                    filled_class = [0]*n_classes
                    n_filled = 0
                    for j in range(B):
                        if n_filled==n_classes: break

                        y = target[j].item()
                        if filled_class[y] == 0 and (torch.argmax(prob_out[j]) != target[j]):
                            filled_class[y] = 1
                            estimate_class_distribution[y, p] = prob_out[j].detach()
                            n_filled += 1

            if p%k==0 or p==p_range[-1]:
                optimizer.zero_grad()
                
                # clf_loss = (p+1)/(_PARTS)*F.nll_loss(output, target,reduction='none')
                # nll_loss = 0.9*F.nll_loss(output, target,reduction='none')-0.1*output.mean(dim=-1)
                nll_loss = F.nll_loss(output, target,reduction='none')
                # clf_loss = (p+1)/(_PARTS)*nll_loss
                # clf_loss = (p+1)/(_PARTS)*nll_loss*data[:,0,max(p-10,0):p].sum(-1).gt(.1)
                clf_loss = (p+1)/(_PARTS)*nll_loss#*data[:,0,:p].sum(-1).gt(1.)
                clf_loss = clf_loss.mean()
                # clf_loss = (p+1)/(_PARTS)*F.cross_entropy(output, target)
                # oracle_loss = (1 - (p+1)/(_PARTS)) * 1.0 *torch.mean( -oracle_prob * output )
                oracle_loss = (1-(p+1)/(_PARTS)) * 1.0 *torch.mean( -oracle_prob * output)
                    
                regularizer = get_regularizer_named_params( named_params, args, _lambda=1.0 ) 
                # if p>600:     
                #     loss = clf_loss + regularizer  + oracle_loss#+ model.network.fr*0.5
                # else:
                #     loss = clf_loss + regularizer 
                loss = clf_loss + regularizer + oracle_loss#
   
                # loss.backward(retain_graph=True)
                loss.backward()

                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    
                optimizer.step()
                post_optimizer_updates( named_params, args,epoch )
            
                train_loss += loss.item()
                total_clf_loss += clf_loss.item()
                total_regularizaton_loss += regularizer #.item()
                total_oracle_loss += oracle_loss.item()
        
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            
            # print(model.network.fr)
            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_oracle_loss = 0

    # print(model.network.layer1_x.weight.grad, model.network.tau_m_r1.grad)
    # print( model.network.tau_m_r1.grad)

parser = argparse.ArgumentParser(description='Sequential Decision Making..')

parser.add_argument('--alpha', type=float, default=.1, help='Alpha')
parser.add_argument('--beta', type=float, default=0.5, help='Beta')
parser.add_argument('--rho', type=float, default=0.0, help='Rho')
parser.add_argument('--lmbda', type=float, default=2.0, help='Lambda')

parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=256, help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default= 2,
                    help='number of layers')
parser.add_argument('--bptt', type=int, default=300, #35,
                    help='sequence length')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')

parser.add_argument('--n_experts', type=int, default=15,
                    help='PTB-Word n_experts')
parser.add_argument('--nhid', type=int, default=256,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=620,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--clip', type=float, default=1., #0.5,
                    help='gradient clipping')

parser.add_argument('--epochs', type=int, default=250,
                    help='upper epoch limit (default: 200)')
parser.add_argument('--parts', type=int, default=100,
                    help='Parts to split the sequential input into (default: 10)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--small_batch_size', type=int, default=-1, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='batch size')

parser.add_argument('--wnorm', action='store_false',
                    help='use weight normalization (default: True)')
parser.add_argument('--wdecay', type=float, default=0.,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use')
parser.add_argument('--when', nargs='+', type=int, default=[10,30,50,75,90],#[30,70,120],#[10,20,50, 75, 90],
                    help='When to decay the learning rate')
parser.add_argument('--load', type=str, default='',
                    help='path to load the model')
parser.add_argument('--save', type=str, default='./models/',
                    help='path to load the model')

parser.add_argument('--per_ex_stats', action='store_true',
                    help='Use per example stats to compute the KL loss (default: False)')
parser.add_argument('--dataset', type=str, default='MNIST-10',
                    help='dataset to use')
parser.add_argument('--dataroot', type=str, 
                    default='./data/',
                    help='root location of the dataset')
args = parser.parse_args()


args.cuda = True

k = 1


exp_name = args.dataset + '-nhid-' + str(args.nhid) + '-parts-' + str(args.parts) + '-optim-' + args.optim
exp_name += '-B-' + str(args.batch_size) + '-E-' + str(args.epochs)
exp_name += '-alpha-' + str(args.alpha) + '-beta-' + str(args.beta) + '-k-' + str(k) + '-V2'
# exp_name += 

if args.per_ex_stats:
    exp_name += '-per-ex-stats-'

print('args.per_ex_stats: ', args.per_ex_stats)
prefix = args.save + exp_name


torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_2 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_3 = torch.device('cpu')
args.device = device

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed(args.seed)



steps = 0
if args.dataset in ['CIFAR-10', 'MNIST-10', 'SHD']:
    train_loader, test_loader, seq_length, input_channels, n_classes = data_generator(args.dataset, 
                                                                     batch_size=args.batch_size,
                                                                     dataroot=args.dataroot, 
                                                                     shuffle=(not args.per_ex_stats))
    estimate_class_distribution = torch.zeros(n_classes, args.parts, n_classes, dtype=torch.float)
    estimatedDistribution = None
    if args.per_ex_stats:
        estimatedDistribution = torch.zeros(len(train_loader)*args.batch_size, args.parts, n_classes, dtype=torch.float)
else:
    exit(1)

optimizer = None
lr = args.lr

model = SeqModel(ninp = input_channels,
                    nhid = args.nhid,
                    nout = n_classes,
                    wnorm = args.wnorm,
                    n_timesteps = seq_length, 
                    parts = args.parts)

if len(args.load) > 0:
    model_ckp = torch.load(args.load)
    model.load_state_dict(model_ckp['state_dict'])
    print('best acc of loaded model: ',model_ckp['best_acc1'])

print('Model: ', model)
if args.cuda:
    model.cuda()

if optimizer is None:
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.wdecay)
    if args.optim == 'SGD':
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.wdecay)
        

all__losses = []
epochs = args.epochs

best_acc1 = 0.0
best_val_loss = None
first_update = False
named_params = get_stats_named_params(model)

# k =5
# python train_mnist_snn.py --dataset MNIST-10 --parts 784 --batch_size 256 --nhid 512 --alpha 0.5 --optim Adamax --lr 5e-3 --beta 0.5
# python train_mnist_snn.py --dataset MNIST-10 --parts 784 --batch_size 128 --nhid 512 --alpha 0.5 --optim Adam --lr 1e-3 --beta 0.1 --load ./models/MNIST-10-nhid-512-parts-784-optim-Adamax-B-256-E-200-K-1-alpha-0.5-beta-0.5_snn_model_sota_best1.pth.tar


for epoch in range(1, epochs + 1):
    print('Epoch ', epoch)
    if args.dataset in ['MNIST-10', 'SHD']:
        if args.per_ex_stats and epoch%5 == 1 :
            first_update = update_prob_estimates( model, args, train_loader, estimatedDistribution, estimate_class_distribution, first_update )
            
        print('TRAINING...')
        train(epoch, args, train_loader, n_classes, model, named_params,k)   
        #train_oracle(epoch)

        reset_named_params(named_params, args)

        print('TESTING...')
        test_loss, acc1 = test(model, test_loader)
      

        if epoch in args.when :
            # Scheduled learning rate decay
            lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
            
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
            
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                # 'oracle_state_dict': oracle.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                # 'oracle_optimizer' : oracle_optim.state_dict(),
            }, is_best, prefix=prefix)
 
        all_test_losses.append(test_loss)