from collections import Counter, defaultdict
import os
import matplotlib.pyplot as pltb
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from sklearn.manifold import TSNE
import pickle
from struct import unpack
import argparse
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
import torch.multiprocessing as mp
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

class GloveDataset(IterableDataset):
    
    def __init__(self, coocurr_file, start=0, end=None):
        self.coocurr_file = open(coocurr_file, 'rb')
        self.coocurr_file.seek(0, 2)
        file_size = self.coocurr_file.tell()
        if file_size % 16 != 0:
            raise Exception('Unproper file. The file need to be the output coocurrence.bin or coocurrence.shuf.bin of offical glove.')
        if end is None:
            self.end = file_size // 16
        else:
            self.end = end
        self.start = start
    
    def __len__(self):
        return self.end
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # single-process data loading, reaturn the full iterator
            iter_start = self.start
            iter_end = self.end
        else: # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        for p in range(iter_start, iter_end):
            self.coocurr_file.seek(p*16)
            chunk = self.coocurr_file.read(16)
            w1, w2, v = unpack('iid', chunk)
            yield v, w1, w2


def get_init_emb_weight(vocab_size, emb_size):
    init_width = 0.5 / emb_size
    init_weight = np.random.uniform(low=-init_width, high=init_width, size=(vocab_size, emb_size))
    return init_weight


class GloveModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(vocab_size, embedding_dim)
        self.wi.weight.data.copy_(torch.from_numpy(get_init_emb_weight(vocab_size, embedding_dim)))
        self.wj = nn.Embedding(vocab_size, embedding_dim)
        self.wj.weight.data.copy_(torch.from_numpy(get_init_emb_weight(vocab_size, embedding_dim)))
        self.bi = nn.Embedding(vocab_size, 1)
        self.bi.weight.data.copy_(torch.from_numpy(get_init_emb_weight(vocab_size, 1)))
        self.bj = nn.Embedding(vocab_size, 1)
        self.bj.weight.data.copy_(torch.from_numpy(get_init_emb_weight(vocab_size, 1)))        
        
    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
        
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
        
        return x

def weight_func(x, x_max, alpha):
    wx = (x/x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx

def wmse_loss(weights, inputs, targets):
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss)

def train(rank, args, model, device, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)
    start = (args.num_lines // args.threads) * rank
    end = start + args.lines_per_thread[rank]
    glove_dataset = GloveDataset(args.coocurr_file, start=start, end=end)
    train_loader = DataLoader(glove_dataset, batch_size=args.batch_size, **dataloader_kwargs)
    optimizer = optim.Adagrad(model.parameters(), lr=args.eta)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)

def train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    for batch_idx, (x_ij, i_idx, j_idx) in enumerate(data_loader):
        x_ij = x_ij.float().to(device)
        i_idx = i_idx.long().to(device)
        j_idx = j_idx.long().to(device)
        optimizer.zero_grad()
        outputs = model(i_idx, j_idx)
        weights_x = weight_func(x_ij, args.x_max, args.alpha)
        loss = wmse_loss(weights_x, outputs, torch.log(x_ij))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}
    
    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')

    vocab_file = open(args.vocab_file, 'r')
    vocab_size = sum(1 for line in vocab_file)
    vocab_file.close()
    print(f'read {vocab_size} words.')
    glove_dataset = GloveDataset(args.coocurr_file)
    EMBED_DIM = args.emb_size
    model = GloveModel(vocab_size + 1, EMBED_DIM).to(device)
    model.share_memory()
    
    if args.threads == 0:
        train(0, args, model, device, dataloader_kwargs)
    else:
        num_lines = len(glove_dataset)
        del glove_dataset
        num_threads = args.threads
        lines_per_thread = []
        for a in range(num_threads-1):
            lines_per_thread.append(num_lines // num_threads)
        lines_per_thread.append(num_lines // num_threads + num_lines % num_threads)
        args.num_lines = num_lines
        args.lines_per_thread = lines_per_thread
    
        processes = []
    
        for rank in range(args.threads):
            p = mp.Process(target=train, args=(rank, args, model, device, dataloader_kwargs))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    
    torch.save(model.state_dict(), args.model_name)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab_file', type=str, help='path to the vocab file', required=True)
    parser.add_argument('-c', '--coocurr_file', type=str, help='path to the coocurr file you can get it by offical glove', required=True)
    parser.add_argument('--emb_size', type=int, help='embedding size', default=200)
    # batch_size, eta tuned for this implementation and jpwiki
    parser.add_argument('--batch_size', type=int, help='batch size', default=1)
    parser.add_argument('--eta', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.75)
    # xmax same to the offical glove example (not their default 100),
    # this leads to the similar performance in the sim_eval task with gensim skip-gram
    parser.add_argument('--xmax', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('-m', '--model_name', type=str, help='saving model path', required=True)
    parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    args = parser.parse_args()
    main(args)
