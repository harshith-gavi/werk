import math
import h5py
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import tonic
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from snn_models_LIF4_save4_l2 import *

if torch.cuda.is_available():
    print('USING CUDA...')
    # torch.backends.cudnn.benchmark = True
    device_1 = torch.device('cuda:0')
    device_2 = torch.device('cuda:1')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print('Use a device with CUDA!!')

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

def data_generator(dataset, batch_size, datapath, shuffle=True):
    if dataset == 'SHD':
        shd_train = h5py.File(datapath + 'train_data/SHD/shd_train.h5', 'r')
        shd_test = h5py.File(datapath + 'test_data/SHD/shd_test.h5', 'r')

        shd_train = data_mod(shd_train['spikes'], shd_train['labels'], batch_size = batch_size, step_size = 100, input_size = 700, max_time = 1.37, shuffle = shuffle)
        shd_test = data_mod(shd_test['spikes'], shd_test['labels'], batch_size = 1, step_size = 100, input_size = 700, max_time = 1.37, shuffle = shuffle)
        
        train_loader = shd_train
        test_loader = shd_test
        n_classes = 20
        seq_length = 100
        input_channels = 700

    else:
        print('Dataset not included! Use a different dataset.')
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

def get_regularizer_named_params( named_params, args):
    alpha = args.alpha
    rho = args.rho
    _lambda = args.lmda 
    regularization = torch.zeros([])
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        regularization += (rho-1.) * torch.sum( param * lm ).cpu()
        r_p = _lambda * 0.5 * alpha * torch.sum( torch.square(param - sm) ).cpu()
        regularization += r_p
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

    test_loss /= (len(test_loader) * data.shape[0])
    
    return test_loss, 100. * correct / (len(test_loader) * data.shape[0])


def train(epoch, args, train_loader, n_classes, model, named_params, k, progress_bar):
    global steps
    global estimate_class_distribution
    global optimizer
    global seq_length

    # estimate_class_distribution.to(device_1)
    batch_size = args.batch_size
    alpha = args.alpha
    beta = args.beta

    PARTS = args.parts
    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    total_oracle_loss = 0
    model.train()
    
    # T = seq_length
    #entropy = EntropyLoss()
   
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device_1), target.to(device_1)
        data = data.to_dense()
        # data = data.view(-1, input_channels, seq_length)
  
        B = target.size()[0]
        step = model.network.step
        xdata = data.clone()
        pdata = data.clone()
        
        # T = inputs.size()[0]
 
        # Delta = torch.zeros(B, dtype=xdata.dtype, device=xdata.device)
        
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
                    oracle_prob = estimate_class_distribution[target.cpu(), p]
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
                
                oracle_loss = (1-(p+1)/(_PARTS)) * 1.0 * torch.mean( -oracle_prob.cpu() * output.cpu())
                    
                regularizer = get_regularizer_named_params(named_params, args) 
                # if p>600:     
                #     loss = clf_loss + regularizer  + oracle_loss#+ model.network.fr*0.5
                # else:
                #     loss = clf_loss + regularizer 
                loss = clf_loss + regularizer + oracle_loss
   
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

        progress_bar.update(1)

def main():
    global estimate_class_distribution, optimizer, steps, seq_length
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='SHD', help='Dataset')
    parser.add_argument('--datapath', type=str, default= '../data/', help='path to the dataset')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='Batch size')
    parser.add_argument('--parts', type=int, default=100, help='Parts to split the sequential input into')

    parser.add_argument('--nlayers', type=int, default=2, help='Number of layers')
    parser.add_argument('--nhid', type=int, default=256, help='Number of Hidden units')
    parser.add_argument('--epochs', type=int, default=100, help='Number of Epochs')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--when', nargs='+', type=int, default=[25, 50, 75], help='Epochs where Learning rate decays')
    parser.add_argument('--optim', type=str, default='Adam', help='Optimiser')
    parser.add_argument('--wnorm', action='store_false', help='Weight normalization (default: True)')
    parser.add_argument('--wdecay', type=float, default=0., help='Weight decay')
    parser.add_argument('--clip', type=float, default=1., help='Gradient Clipping')
    parser.add_argument('--alpha', type=float, default=.1, help='Alpha')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta')
    parser.add_argument('--rho', type=float, default=0.0, help='Rho')
    parser.add_argument('--lmda', type=float, default=1.0, help='Lambda')
                        
    parser.add_argument('--seed', type=int, default=1111, help='Random seed')
    parser.add_argument('--load', type=str, default='', help='Path to load the model')
    parser.add_argument('--save', type=str, default='./models/', help='Path to save the model')
    parser.add_argument('--per_ex_stats', action='store_true', help='Use per example stats to compute the KL loss (default: False)')


    print('PARSING ARGUMENTS...')           
    args = parser.parse_args()
    
    exp_name = 'optim-' + args.optim + '-B-' + str(args.batch_size) + '-alpha-' + str(args.alpha) + '-beta-' + str(args.beta)
    if args.per_ex_stats: exp_name += '-per-ex-stats-'    
    print('args.per_ex_stats: ', args.per_ex_stats)
    prefix = args.save + exp_name
    torch.cuda.manual_seed(args.seed)

    print('PREPROCESSING DATA...')
    train_loader, test_loader, seq_length, input_channels, n_classes = data_generator(args.dataset, batch_size = args.batch_size, datapath = args.datapath, shuffle = (not args.per_ex_stats))
    estimate_class_distribution = torch.zeros(n_classes, args.parts, n_classes, dtype=torch.float)
    estimatedDistribution = None
    if args.per_ex_stats:
        estimatedDistribution = torch.zeros(len(train_loader)*args.batch_size, args.parts, n_classes, dtype=torch.float)

    
    print('CREATING A MODEL...')
    model = SeqModel(ninp = input_channels,
                     nhid = args.nhid,
                     nout = n_classes,
                     wnorm = args.wnorm,
                     n_timesteps = seq_length, 
                     parts = args.parts) 
    model = nn.DataParallel(model, device_ids=[0, 1])
    print('Model: ', model)

    # if len(args.load) > 0:
    #     print('LOADING THE MODEL...')
    #     model_ckp = torch.load(args.load)
    #     model.load_state_dict(model_ckp['state_dict'])
    #     print('best acc of loaded model: ',model_ckp['best_acc'])

    optimizer = None
    if optimizer is None:
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        if args.optim == 'SGD':
            optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay)
            
    
    all_test_losses = []
    epochs = args.epochs
    best_acc = 0.0
    lr = args.lr
    steps = 0
    first_update = False
    named_params = get_stats_named_params(model)
    
    for epoch in range(1, epochs + 1):  
        if args.dataset in ['SHD']:
            if args.per_ex_stats and epoch%5 == 1 :
                first_update = update_prob_estimates( model, args, train_loader, estimatedDistribution, estimate_class_distribution, first_update )
    
            progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
            k = 1
            train(epoch, args, train_loader, n_classes, model, named_params, k, progress_bar)  
            progress_bar.close()
            #train_oracle(epoch)
    
            reset_named_params(named_params, args)
    
            test_loss, acc = test(model, train_loader)
            print('Loss:', test_loss, end = '\t')
            print('Accuracy:', acc.item())
          
            if epoch in args.when :
                # Scheduled learning rate decay
                lr *= 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
                
            # remember best acc@1 and save checkpoint
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
                
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    # 'oracle_state_dict': oracle.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                    # 'oracle_optimizer' : oracle_optim.state_dict(),
                }, is_best, prefix=prefix)
     
            all_test_losses.append(test_loss)

if __name__ == "__main__":
    main()
