from collections import Counter, defaultdict
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

# loss function from https://nlpython.com/implementing-glove-model-with-pytorch/
    
def weight_func(x, x_max, alpha):
    wx = (x/x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    # return wx.cuda()  
    return wx

def wmse_loss(weights, inputs, targets):
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    # return torch.mean(loss).cuda()
    return torch.mean(loss)

def main(args):
    vocab_file = open(args.vocab_file, 'r')
    vocab_size = sum(1 for line in vocab_file)
    vocab_file.close()
    print(f'read {vocab_size} words.')
    EMBED_DIM = args.emb_size
    glove = GloveModel(vocab_size + 1, EMBED_DIM)
    # glove.cuda()
    print(glove)
    # adagrad the same as offical but converge slow
    optimizer = optim.Adagrad(glove.parameters(), lr=args.eta)
    # Adam diverge after some epochs
    # optimizer = optim.Adam(glove.parameters(), lr=args.eta)
    glove_dataset = GloveDataset(args.coocurr_file)
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    X_MAX = args.xmax
    ALPHA = args.alpha
    n_batches = len(glove_dataset) // BATCH_SIZE
    dataloader = DataLoader(glove_dataset, batch_size=BATCH_SIZE, num_workers=args.threads, pin_memory=True)

    loss_values = []   
    # loss_check_point = [i * (n_batches//5) for i in range(5)]
    loss_history_output = open('loss_history.log', 'w')
    for epoch in range(1, N_EPOCHS+1):
        batch_i = 0
        for x_ij, i_idx, j_idx in tqdm(dataloader, desc=f'Training on {n_batches} batches...'):
            batch_i += 1
            # x_ij = x_ij.float().cuda()
            # i_idx = i_idx.long().cuda()
            # j_idx = j_idx.long().cuda()
            x_ij = x_ij.float()
            i_idx = i_idx.long()
            j_idx = j_idx.long()
            optimizer.zero_grad()
            outputs = glove(i_idx, j_idx)
            weights_x = weight_func(x_ij, X_MAX, ALPHA)
            loss = wmse_loss(weights_x, outputs, torch.log(x_ij))
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())
            # if batch_i % 100 == 0:
            #     print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(epoch, N_EPOCHS, batch_i, n_batches, np.mean(loss_values[-100:])))
        epoch_loss = np.mean(loss_values[-n_batches:])
        log_text = f"{datetime.now()} Epoch: {epoch}/{N_EPOCHS} \t Loss: {np.mean(loss_values[-n_batches:])}"
        print(log_text)
        loss_history_output.write(log_text + '\n')
        loss_history_output.flush()
        print("Saving checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': glove.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            }, args.model_name + '.e' + str(epoch))
    loss_history_output.close()
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab_file', type=str, help='path to the vocab file', required=True)
    parser.add_argument('-c', '--coocurr_file', type=str, help='path to the coocurr file you can get it by offical glove', required=True)
    parser.add_argument('--emb_size', type=int, help='embedding size', default=200)
    # batch_size, eta tuned for this implementation and jpwiki
    parser.add_argument('--batch_size', type=int, help='batch size', default=262144)
    parser.add_argument('--eta', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.75)
    # xmax same to the offical glove example (not their default 100),
    # this leads to the similar performance in the sim_eval task with gensim skip-gram
    parser.add_argument('--xmax', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('-m', '--model_name', type=str, help='saving model path', required=True)
    args = parser.parse_args()
    main(args)
