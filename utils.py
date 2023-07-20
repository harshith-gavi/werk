import torch
from torch import nn
import torch.nn.functional as F

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

def get_xt(p, step, T, inputs):
    start = p * step
    end = (p + 1) * step
    if end >= T:
        end = T

    x = inputs[start:end]
    return x, start, end

def update_prob_estimates(model, args, train_loader, estimatedDistribution, estimate_class_distribution, first_update=False):
    PARTS = args.parts
    model.eval()

    print('Find current distribution for each image...')
    batch_size = args.batch_size
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        B = target.size()[0]
        step = model.network.step
        xdata = data.to_dense()
        T = data.size()[0]

        for p in range(PARTS):
            x, start, end = get_xt(p, step, T, data)

            with torch.no_grad():
                if p == 0:
                    h = model.init_hidden(xdata.size(0))
                else:
                    h = (h[0].detach(), h[1].detach())

                o, h = model.network[0].rnn(x, h)
                out = F.dropout(model.linear2(model.linear1((h[0]))), model.dropout)
                out = out.squeeze(dim=0)
                prob_out = F.softmax(out, dim=1)

                if first_update == False:
                    estimatedDistribution[batch_idx * batch_size: (batch_idx + 1) * batch_size, p] = prob_out
                else:
                    A = estimatedDistribution[batch_idx * batch_size: (batch_idx + 1) * batch_size, p]
                    B = prob_out
                    estimatedDistribution[batch_idx * batch_size: (batch_idx + 1) * batch_size, p] = 0.6 * A + 0.4 * B

    print('Find best for each class...')
    estimate_class_distribution_tensor = torch.tensor(estimate_class_distribution)
    for batch_idx, (data, target) in enumerate(train_loader):
        target_np = target.cpu().numpy()
        for idx, y in enumerate(target_np):
            for p in range(PARTS):
                current_distribution = estimatedDistribution[batch_idx * batch_size + idx, p]
                argmax_current_distribution = torch.argmax(current_distribution)

                if argmax_current_distribution != y:
                    estimatedDistribution[batch_idx * batch_size + idx, p] = estimate_class_distribution_tensor[y, p]

    first_update = True
    return first_update
