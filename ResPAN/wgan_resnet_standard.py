import loompy
from numba import jit
from collections import Counter

import random, os
import numpy as np
import pandas as pd
import scanpy as sc
from skmisc.loess import loess
import sklearn.preprocessing as preprocessing
from sklearn.neighbors import NearestNeighbors, KDTree

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_s 
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

from annoy import AnnoyIndex

import matplotlib.pyplot as plt

# ACProp
from adabelief_pytorch import AdaBelief
import math
import torch
from torch.optim.optimizer import Optimizer
from tabulate import tabulate
from colorama import Fore, Back, Style

version_higher = ( torch.__version__ >= "1.5.0" )

class ACProp(Optimizer):
    r"""Implements ACProp algorithm. Modified from AdaBelief in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-16)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: True) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: True) If set as True, then perform the rectified
            update similar to RAdam
        degenerated_to_sgd (boolean, optional) (default:True) If set as True, then perform SGD update
            when variance of gradient is high
        print_change_log (boolean, optional) (default: True) If set as True, print the modifcation to
            default hyper-parameters
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False, rectify=True,
                 degenerated_to_sgd=True, momentum_update=False, bias_correction = False):

        
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, buffer=[[None, None, None] for _ in range(10)])
        super(ACProp, self).__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        self.momentum_update = momentum_update
        if self.weight_decouple:
            print('Weight decoupling enabled in AdaBelief')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaBelief')
        if amsgrad:
            print('AMSGrad enabled in AdaBelief')

    def __setstate__(self, state):
        super(ACProp, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'ACProp does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                            if version_higher else torch.zeros_like(p.data)
                
                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** (state['step']-1)

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group['eps']), out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    if state['step']>1:
                        denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # update
                if state['step']<=1:
                    p.data.add_(exp_avg/bias_correction1, alpha=-group['lr'])

                elif not self.rectify:
                    numerator = exp_avg if self.momentum_update else grad * bias_correction1
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_( numerator, denom, value=-step_size)

                else:  # Rectified update, forked from RAdam
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        # more conservative since it's an approximated value
                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if N_sma >= 5:
                        numerator = exp_avg if self.momentum_update else grad * bias_correction1
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p.data.addcdiv_(numerator, denom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        p.data.add_( exp_avg, alpha=-step_size * group['lr'])

                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_( grad_residual, grad_residual, value=1 - beta2)

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half() 

        return loss

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x*torch.tanh(F.softplus(x))

#WGAN model, and it does not need to use bath normalization based on WGAN paper.
class discriminator(nn.Module):
    def __init__(self, N = 2000):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(N, 1024),  
            Mish(),
            nn.Linear(1024, 512),  
            Mish(),
            nn.Linear(512, 256),  
            Mish(),
            nn.Linear(256, 128),  
            Mish(),
            nn.Linear(128, 1)

        )

    def forward(self, x):
        x = self.dis(x)
        return x
 
 
# WGAN generator
# Require batch normalization    
class generator(nn.Module):
    def __init__(self, N=2000):
        super(generator, self).__init__()
        self.relu_f = nn.ReLU(True)
        self.mish_f = Mish()
        self.gen1 = nn.Sequential(
                    nn.Linear(N, 1024),
                    nn.BatchNorm1d(1024),
                    Mish())
        self.gen2 = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    Mish())
        self.gen3 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    Mish())
        self.gen4 = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.BatchNorm1d(512),
                    )
        self.gen5 = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.BatchNorm1d(1024),
                    )
        self.gen6 = nn.Linear(1024, N)

    def forward(self, x):
        x1 = self.gen1(x)
        x2 = self.gen2(x1)
        x3 = self.gen3(x2)
        x4 = self.mish_f(self.gen4(x3) + x2)
        x5 = self.mish_f(self.gen5(x4) + x1)
        gre = self.gen6(x5)
        
        return self.relu_f(gre+x)    #residual network

# calculate gradient penalty
def calculate_gradient_penalty(real_data, fake_data, D, center=1, p=2): 
    eta = torch.FloatTensor(real_data.size(0),1).uniform_(0,1) 
    eta = eta.expand(real_data.size(0), real_data.size(1)) 
    cuda = True if torch.cuda.is_available() else False 
    if cuda: 
        eta = eta.cuda() 
    else: 
        eta = eta 
    interpolated = eta * real_data + ((1 - eta) * fake_data) 
    if cuda: 
        interpolated = interpolated.cuda() 
    else: 
        interpolated = interpolated 
    # define it to calculate gradient 
    interpolated = Variable(interpolated, requires_grad=True) 
    # calculate probability of interpolated examples 
    prob_interpolated = D(interpolated) 
    # calculate gradients of probabilities with respect to examples 
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated, 
                                grad_outputs=torch.ones( 
                                  prob_interpolated.size()).cuda() if cuda else torch.ones( 
                                  prob_interpolated.size()), 
                                create_graph=True, retain_graph=True)[0] 
    grad_penalty = ((gradients.norm(2, dim=1) - center) ** p).mean() 
    return grad_penalty

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

from sklearn.utils import shuffle
def WGAN_train(label_data, train_data, epoch, batch, lambda_1, query_data, n_critic=100, 
               b1=0.5, b2=0.9, lr=0.0001, opt='AdamW'):

    D = discriminator(N=2000)
    G = generator(N=2000)

    if torch.cuda.is_available():
        D = D.cuda()
        G = G.cuda()

    lr = lr
#     D.apply(weights_init_normal)
#     G.apply(weights_init_normal)
    
    if opt == 'ACProp':
        d_optimizer = ACProp(D.parameters(), lr=lr, betas=(b1, b2))
        g_optimizer = ACProp(G.parameters(), lr=lr, betas=(b1, b2)) 
    elif opt == 'AdaBelief':
        d_optimizer = AdaBelief(D.parameters(), lr=lr, betas=(b1, b2))
        g_optimizer = AdaBelief(G.parameters(), lr=lr, betas=(b1, b2))
    else:
        d_optimizer = torch.optim.AdamW(D.parameters(), lr=lr, betas=(b1, b2))
        g_optimizer = torch.optim.AdamW(G.parameters(), lr=lr, betas=(b1, b2))  
    G.train()
    D.train()
    

    for epoch in range(epoch):
        
        sample_index = np.random.choice(len(label_data), size=min(10*1024, len(label_data)), replace=False)
        label_data_sample = label_data[sample_index]
        train_data_sample = train_data[sample_index]
        

        label_data_sample = torch.FloatTensor(label_data_sample).cuda()
        train_data_sample = torch.FloatTensor(train_data_sample).cuda()
        
        training_set = data_utils.TensorDataset(train_data_sample, label_data_sample)
        dataloader = DataLoader(training_set, batch_size=batch)
        
        for i, (false_data, true_data) in enumerate(dataloader):

            # train D

            d_optimizer.zero_grad()

            real_out = D(true_data)
            real_loss = -torch.mean(real_out)

            fake_data = G(false_data)
            fake_out = D(fake_data)
            fake_loss = torch.mean(fake_out)

            div = calculate_gradient_penalty(true_data, fake_data, D, center = 1)    # sample from G(fake)

            D_loss = real_loss + fake_loss + div/lambda_1
            D_loss.backward()

            # err_D.append(label_loss.cpu().item())

            d_optimizer.step()

            # train G
            
            if i % n_critic == 0:
#                 print('Train G for batch ', i)

                fake_data = G(false_data)
                fake_out = D(fake_data)
                
                G_loss = -torch.mean(fake_out)
                    
                # err_G.append(real_loss1.cpu().item())

                g_optimizer.zero_grad()
                G_loss.backward()
                g_optimizer.step()

        if epoch % 50 == 0:
            print("This is epoch: ", epoch)
            print("d step loss: ", D_loss)
            print("g step loss: ", G_loss)
#         label_data, train_data = shuffle(label_data, train_data)

    print("Train step finished")
    G.eval()
    test_data = torch.FloatTensor(query_data).cuda()
    test_list = G(test_data).detach().cpu().numpy()    
    return test_list, G

# dimensional reduction tools
from sklearn import preprocessing
def cca_seurat(X, Y, n_components=20, normalization=True):
    X = preprocessing.scale(X, axis=0)
    Y = preprocessing.scale(Y, axis=0)
    X = preprocessing.scale(X, axis=1)
    Y = preprocessing.scale(Y, axis=1)
    mat = X @ Y.T
#     if normalization:
#         mat = preprocessing.scale(mat)
    k = n_components
    u,sig,v = np.linalg.svd(mat,full_matrices=False)
    sigma = np.diag(sig)
    return np.dot(u[:,:k],np.sqrt(sigma[:k,:k])), np.dot(v.T[:,:k], np.sqrt(sigma[:k,:k]))

from sklearn.decomposition import PCA #same as probabilistc pca
def pca_reduction(X, Y, n_components=20, normalization=True):
    mat = np.vstack([X,Y])
    L1 = len(X)
    if normalization:
        mat = preprocessing.scale(mat)
    model = PCA(n_components=n_components)
    pca_fit = model.fit_transform(mat)
    return pca_fit[0:L1,:],pca_fit[L1:,:]

# from sklearn.manifold import Isomap
# def isomap_reduction(X, Y, n_components=20, normalization=True):
#     mat = np.vstack([X,Y])
#     L1 = len(X)
#     if normalization:
#         mat = preprocessing.scale(mat)
#     isomap = Isomap()
#     isodata = isomap.fit_transform(mat)
#     return isodata[0:L1,:], isodata[L1:,:]

from sklearn.decomposition import KernelPCA
def kpca_reduction(X, Y, n_components=20, kernel='cosine', normalization=True):
    mat = np.vstack([X,Y])
    L1 = len(X)
    if normalization:
        mat = preprocessing.scale(mat)
    kpca = KernelPCA(n_components = n_components, kernel=kernel)
    kpca_fit = kpca.fit_transform(mat)
    return kpca_fit[0:L1,:],kpca_fit[L1:,:]

def rpca_reduction(X, Y, n_components=20, normalization=True):
    if normalization:
        X = preprocessing.scale(X)
        Y = preprocessing.scale(Y)
    U1,S1,V1 = np.linalg.svd(X)
    U2,S2,V2 = np.linalg.svd(Y)
    results1 = []
    results2 = []
    # Project X onto X's PCA space
    results1.append(X@V1[0:n_components,:].T)
    # Project X onto Y's PCA space
    results1.append(X@V2[0:n_components,:].T)
    # Project Y onto X's PCA space
    results2.append(Y@V1[0:n_components,:].T)
    # Project Y onto Y's PCA space
    results2.append(Y@V2[0:n_components,:].T)
    return results1, results2



def create_pairs_dict(pairs):
    pairs_dict = {}
    for x,y in pairs:
        if x not in pairs_dict.keys():
            pairs_dict[x] = [y]
        else:
            pairs_dict[x].append(y)
    return pairs_dict


def acquire_rwmnn_pairs(X, Y, k, metric):
    X = preprocessing.normalize(X,axis=1)
    Y = preprocessing.normalize(Y,axis=1)
    t1 = KDTree(X)
    t2 = KDTree(Y)
    mnn_mat = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array(t2.query(X,k,return_distance=False))
    for i in range(len(sorted_mat)):
        mnn_mat[i,sorted_mat[i]] = True
    _ = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array(t1.query(Y,k,return_distance=False))
    for i in range(len(sorted_mat)):
        _[sorted_mat[i],i] = True
    mnn_mat = np.logical_and(_, mnn_mat)
    pairs = [(x, y) for x, y in zip(*np.where(mnn_mat>0))]
    return pairs

def top_features(X, Y, n_components=20, features_per_dim=10, normalization=True):
    mat = np.vstack([X,Y])
    if normalization:
        mat = preprocessing.scale(mat)
    model = PCA(n_components=n_components)
    pca_fit = model.fit_transform(mat)
    feature_loadings = model.components_[:n_components]
    top_features_index = np.unique((-np.abs(feature_loadings)).argsort(axis=1)[:,:features_per_dim])
    return top_features_index

def top_features_cca(X, Y, X_cca, Y_cca, n_components=20, features_per_dim=10, normalization=True):
    mat_cca = np.vstack([X_cca, Y_cca])
    if normalization:
        mat = np.vstack([preprocessing.scale(X),preprocessing.scale(Y)])
    else:
        mat = np.vstack([X,Y])
    feature_loadings = mat_cca.T @ mat
    top_features_index = np.unique((-np.abs(feature_loadings)).argsort(axis=1)[:,:features_per_dim])
    return top_features_index

def filter_pairs(ref, query, pairs, k=200, n_components=20, features_per_dim=10, reduction='pca', 
                ref_reduced = None, query_reduced=None):
    ref_index = [x for x,y in pairs]
    query_index = [y for x,y in pairs]
    if reduction == 'cca':
        top_features_index = top_features_cca(ref, query, ref_reduced, query_reduced,
                                              n_components=n_components, features_per_dim=features_per_dim)
    else:
        top_features_index = top_features(ref, query, n_components=n_components, features_per_dim=features_per_dim)
    ref = ref[:,top_features_index]
    query = query[:,top_features_index]
    ref = preprocessing.normalize(ref, axis=1)
    query = preprocessing.normalize(query, axis=1)
    t1 = KDTree(ref)
    sorted_mat = np.array(t1.query(query[query_index],k,return_distance=False))
    
    ref_index_filtered = []
    query_index_filtered = []
    pairs_filtered = []
    for i, j in zip(range(len(sorted_mat)), ref_index):
        if j in sorted_mat[i]:
            pairs_filtered.append((j, query_index[i]))

    return pairs_filtered

def calculate_rwmnn_pairs(X, Y, ref_data, query_data, metric='angular', k1 = None, k2 = None, 
                          filtering = True, k_filter=100, reduction='pca'):

    if k2 is None:
        k2 = max(int(min(len(ref_data), len(query_data))/100), 1)
    if k1 is None:
        k1 = max(int(k2/2), 1)

    print('Calculating Anchor Pairs...')
    anchor_pairs = acquire_rwmnn_pairs(X,X, k2, metric)
    print('Calculating Query Pairs...')
    query_pairs = acquire_rwmnn_pairs(Y,Y, k2, metric)
    print('Calculating KNN Pairs...')
    pairs = acquire_rwmnn_pairs(X,Y, k1, metric)
    print('Number of MNN pairs is %d' % len(pairs))
    
    print('Calculating Random Walk Pairs...')
    anchor_pairs_dict = create_pairs_dict(anchor_pairs)
    query_pairs_dict = create_pairs_dict(query_pairs)
    pair_plus = []
    for x, y in pairs:
        start = (x, y)
        for i in range(50):
            pair_plus.append(start)
            start = (np.random.choice(anchor_pairs_dict[start[0]]), np.random.choice(query_pairs_dict[start[1]]))
    
    if filtering and reduction is not None:
        print('Start filtering of selected pairs')
        pair_plus = filter_pairs(ref_data, query_data, pair_plus, k=k_filter, reduction=reduction, 
                             ref_reduced=X, query_reduced=Y)
        print('Number of rwMNN pairs after filtering is %d.' % len(pair_plus))
        
#     query_result = query_data[[y for x,y in pair_plus], :]
#     reference_result = ref_data[[x for x,y in pair_plus], :]
    ref_index = [x for x,y in pair_plus]
    query_index = [y for x,y in pair_plus]
    print('Done.')
    return ref_index, query_index

def training_set_generator(ref, query, reduction=None, mode='MNN', metric='angular', subsample=None,
                           k=None, k1=None, k2=None, norm=True, filtering=True):
    ref_reduced, query_reduced = None,None
    if subsample:
        len_ref = len(ref)
        len_query = len(query)
        thre = max(len_ref, len_query)
        if thre>subsample:
            tmp = np.arange(len_ref)
            np.random.shuffle(tmp)
            tmp2 = np.arange(len_query)
            np.random.shuffle(tmp2)
            ref = ref[tmp][:subsample]
            query = query[tmp2][:subsample]
        
    if reduction == 'cca':
        ref_reduced, query_reduced = cca_seurat(ref, query,normalization=norm)
    elif reduction == 'pca':
        ref_reduced, query_reduced = pca_reduction(ref, query,normalization=norm)
    elif reduction == 'kpca':
        ref_reduced, query_reduced = kpca_reduction(ref, query,normalization=norm)
    elif reduction == 'rpca':
        ref_reduced, query_reduced = rpca_reduction(ref, query,normalization=norm)
    elif reduction is None:
        ref_reduced, query_reduced = ref, query
        
    if mode == 'rwMNN':
        ref_index, query_index = calculate_rwmnn_pairs(ref_reduced, query_reduced, ref, query, k1=k1, k2=k2, 
                                                       filtering=filtering)
    print('Dimension of paired data is %d.' % len(ref_index))
        
    return ref[ref_index], query[query_index]

def order_selection(adata, key='batch', orders=None):
    if orders == None:
        batch_list = list(set(adata.obs[key].values))
        adata_values = [np.array(adata.X[adata.obs[key] == batch]) for batch in batch_list]
        std_ = [np.sum(np.std(item, axis=0)) for item in adata_values]
        orders = np.argsort(std_)[::-1]
        return [batch_list[i] for i in orders] 
    else:
        return orders

def generate_target_dataset(adata, batch_list):
    adata0 = adata[adata.obs['batch'] == batch_list[0]]
    for i in batch_list[1:]:
        adata0 = adata0.concatenate(adata[adata.obs['batch'] == i], batch_key='batch_key', index_unique=None)
    return adata0

def sequencing_train(adata, key='batch', order=None, epoch=300, batch=1024,
                     lambda_1=1/10, mode='MNN', metric='angular', 
                     reduction='pca', subsample=3000, k=None, k1=None, k2=None,
                     n_critic=100, seed=999, b1=0.9, b2=0.999, lr=0.0001, opt='AdamW',
                     filtering=False):
    
    # Set a seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # for CNN
    
    if not order:
        order = order_selection(adata, key=key)
    print('The sequential mapping order will be: ', order)
        
    adata = generate_target_dataset(adata, order)
    ref_data_ori = adata[adata.obs['batch'] == order[0]].X

    for bat in order[1:]:
        print("##########################Training %s#####################"%(bat))
        batch_data_ori = adata[adata.obs['batch'] == bat].X
        label_data, train_data = training_set_generator(ref_data_ori, batch_data_ori, reduction=reduction,
                                                       mode=mode, metric=metric, subsample=subsample, 
                                                        k=k, k1=k1, k2=k2, filtering=filtering)
        print("#################Finish Pair finding##########################")
        remove_batch_data, G_tar = WGAN_train(label_data, train_data, epoch, batch, lambda_1, batch_data_ori,
                                             n_critic=n_critic, b1=b1, b2=b2, lr=lr, opt=opt)
        ref_data_ori = np.vstack([ref_data_ori, remove_batch_data])
    print("###################### Finish Training ###########################")
    
    adata_new = sc.AnnData(ref_data_ori)
    adata_new.obs['batch'] = list(adata.obs['batch'])
    if adata.obs.columns.str.contains('celltype').any():
        adata_new.obs['celltype'] = list(adata.obs['celltype'])
    adata_new.var_names = adata.var_names
    adata_new.var_names.name = 'Gene'
    adata_new.obs_names = adata.obs_names
    adata_new.obs_names.name = 'CellID'
    
    return adata_new


# adata

# adata = data_preprocess(adata,'batch')

# data_correct = sequencing_train(adata, mode='MNN', k=30, reduction=None, epoch=300, 
#                                 recon=True, sim=True, knn_sim=True, k_sim=10)

# adata_correct.write_h5ad('CL_wgan.h5ad')
