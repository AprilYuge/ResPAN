import random, os
import numpy as np
import scanpy as sc
import sklearn.preprocessing as preprocessing
from sklearn.neighbors import KDTree

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

class Mish(nn.Module):
    """A class implementation for the state-of-the-art activation function, Mish.
    """
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x*torch.tanh(F.softplus(x))

# The discriminator model, and it does not need to use bath normalization based on WGAN-GP paper.
class discriminator(nn.Module):
    """The discrimimator structure of our AWGAN. 
        Layer: N(2000)->1024->512->256->128->1
    """
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
 
 
# The generator model that requires batch normalization    
class generator(nn.Module):
    """The generator structure of our AWGAN. 
    Layer: 2000->1024->512->256->512->1024->2000
    with skip connection
    """
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
    """Function utilized to calculate gradient penalty. This term is used to construct the loss function. 
    Args:
       real_data: Tensor of reference (new reference) batch data
       fake_data: Tensor of query batch data
       D: The discriminator network 
       center: K of Lipschitz condition 
       p: Dimensions of distance 
    Output:
        result(list): A list containing index pair for NNPs.
    """
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

def train(label_data, train_data, epoch, batch, lambda_1, query_data, n_critic=100, 
          b1=0.5, b2=0.9, lr=0.0001, opt='AdamW'):
    """Function utilized to train ResPAN
        Args:
            label_data: Tensor of reference (new reference) batch data
            train_data: Tensor of query batch data
            epoch: The number of iteraiton steps
            batch: Input batch 
            lambda_1: A hyperparameter used to control the weights of gradient penalty
            query_data: Gene level expression matrix of another batch
            n_critic: A step value used to determine the training times of generator 
            b1,b2: Hyperparameters used in AdamW optimizer
            lr: Learning rate
            opt: The name of our optimizer
        Output:
            test_list: Results for query batch after batch correction
            G: The generator of ResPAN
    """
    D = discriminator(N=2000)
    G = generator(N=2000)

    if torch.cuda.is_available():
        D = D.cuda()
        G = G.cuda()

    d_optimizer = torch.optim.AdamW(D.parameters(), lr=lr, betas=(b1, b2))
    g_optimizer = torch.optim.AdamW(G.parameters(), lr=lr, betas=(b1, b2))  
    G.train()
    D.train()
    
    print('Start adversarial training...')
    
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

            div = calculate_gradient_penalty(true_data, fake_data, D, center = 1)

            D_loss = real_loss + fake_loss + lambda_1*div
            D_loss.backward()
            d_optimizer.step()

            # train G
            
            if i % n_critic == 0:

                fake_data = G(false_data)
                fake_out = D(fake_data)
                
                G_loss = -torch.mean(fake_out)
                g_optimizer.zero_grad()
                G_loss.backward()
                g_optimizer.step()

        if epoch % 50 == 0:
            print("This is epoch: ", epoch)
            print("d step loss: ", D_loss)
            print("g step loss: ", G_loss)

    print("Train step finished.")
    print("Generate corrected data...")
    G.eval()
    test_data = torch.FloatTensor(query_data).cuda()
    test_list = G(test_data).detach().cpu().numpy()    
    return test_list, G

# dimensional reduction tools
from sklearn import preprocessing
def cca_seurat(X, Y, n_components=20, normalization=True):
    """Function used to perform dimension reduction based on CCA.
        Args:
            X: Gene expression matrix of the reference batch
            Y: Gene expression matrix of the query batch
            n_components: The components we intend to preserve after dimension reduction
            normalization: A bool value used to determine whether to scale the data or not
        Output:
            W1: Embeddings of the reference data
            W2: Embeddings of the query data
    """
    if normalization:
        X = preprocessing.scale(X, axis=0)
        Y = preprocessing.scale(Y, axis=0)
    X = preprocessing.scale(X, axis=1)
    Y = preprocessing.scale(Y, axis=1)
    mat = X @ Y.T
    k = n_components
    u,sig,v = np.linalg.svd(mat,full_matrices=False)
    sigma = np.diag(sig)
    W1 = np.dot(u[:,:k],np.sqrt(sigma[:k,:k]))
    W2 = np.dot(v.T[:,:k], np.sqrt(sigma[:k,:k]))
    return W1, W2

from sklearn.decomposition import PCA
def pca_reduction(X, Y, n_components=20, normalization=True):
    """Function used to perform dimension reduction based on PCA.
        Args:
            X: Gene expression matrix of the reference batch
            Y: Gene expression matrix of the query batch
            n_components: The components we intend to preserve after dimension reduction
            normalization: A bool value used to determine whether to scale the data or not
        Output:
            W1: Embeddings of the reference data
            W2: Embeddings of the query data
    """
    mat = np.vstack([X,Y])
    L1 = len(X)
    if normalization:
        mat = preprocessing.scale(mat)
    model = PCA(n_components=n_components)
    pca_fit = model.fit_transform(mat)
    W1 = pca_fit[0:L1,:]
    W2 = pca_fit[L1:,:]
    return W1, W2

from sklearn.decomposition import KernelPCA
def kpca_reduction(X, Y, n_components=20, kernel='cosine', normalization=True):
    """Function used to perform dimension reduction based on kernelPCA.
        Args:
            X: Gene expression matrix of the reference batch
            Y: Gene expression matrix of the query batch
            n_components: The components we intend to preserve after dimension reduction
            normalization: A bool value used to determine whether to scale the data or not
        Output:
            W1: Embeddings of the reference data
            W2: Embeddings of the query data
    """
    mat = np.vstack([X,Y])
    L1 = len(X)
    if normalization:
        mat = preprocessing.scale(mat)
    kpca = KernelPCA(n_components = n_components, kernel=kernel)
    kpca_fit = kpca.fit_transform(mat)
    W1 = kpca_fit[0:L1,:]
    W2 = kpca_fit[L1:,:]
    return W1, W2


# features selection
def top_features(X, Y, n_components=20, features_per_dim=10, normalization=True):
    """Function used to select the top components after PCA dimension reduction
        Args:
            X: Gene expression matrix of the reference batch
            Y: Gene expression matrix of the query batch
            n_components: The components we intend to preserve after dimension reduction
            features_pre_dim: The number of features being selected for differnet dimensions
            normalization: A bool value used to determine whether to scale the data or not
        Output:
            top_features_index: The index array of top features

    """
    mat = np.vstack([X,Y])
    if normalization:
        mat = preprocessing.scale(mat)
    model = PCA(n_components=n_components)
    pca_fit = model.fit_transform(mat)
    feature_loadings = model.components_[:n_components]
    top_features_index = np.unique((-np.abs(feature_loadings)).argsort(axis=1)[:,:features_per_dim])
    return top_features_index

def top_features_cca(X, Y, X_cca, Y_cca, n_components=20, features_per_dim=10, normalization=True):
    """Function used to select the top components after CCA dimension reduction
        Args:
            X: Gene expression matrix of the reference batch
            Y: Gene expression matrix of the query batch
            n_components: The components we intend to preserve after dimension reduction
            features_pre_dim: The number of features being selected for differnet dimensions
            normalization: A bool value used to determine whether to scale the data or not
        Output:
            top_features_index: The index array of top features

    """
    mat_cca = np.vstack([X_cca, Y_cca])
    if normalization:
        mat = np.vstack([preprocessing.scale(X),preprocessing.scale(Y)])
    else:
        mat = np.vstack([X,Y])
    feature_loadings = mat_cca.T @ mat
    top_features_index = np.unique((-np.abs(feature_loadings)).argsort(axis=1)[:,:features_per_dim])
    return top_features_index

def filter_pairs(ref, query, pairs, k=200, n_components=20, features_per_dim=10, reduction='pca', 
                ref_reduced=None, query_reduced=None):

    """Function used to filter the rwMNN pairs we found
        Args:
            ref: Gene expression matrix of the reference batch 
            query: Gene expression matrix of the query batch
            pairs: rwMNN pairs we found
            k: The number of nearest neighbors we used
            n_components: The components we used to select top features
            features_per_dim: The number of features being selected for differnet dimensions
            reduction: The dimension reduction tool we used here. The options of this parameter include: 'pca' and 'cca'
            ref_reduced: Dimension reduction results of the reference batch 
            query_reduced: Dimension reduction results of the query batch 
        Output:
            pairs_filtered: rwMNN pairs after filtering

    """
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

def create_pairs_dict(pairs):
    """A simple function used to generate a dictionary for random walk selection based on MNN seeds.
    """
    pairs_dict = {}
    for x,y in pairs:
        if x not in pairs_dict.keys():
            pairs_dict[x] = [y]
        else:
            pairs_dict[x].append(y)
    return pairs_dict

def acquire_rwmnn_pairs(X, Y, k):
    """Function used to generate the MNN pairs (used for rwMNN pairs finding)
        Args:
            X: Embeddings of the reference data
            Y: Embeddings of the query data
            k: The number of nearest neighbors we chose
        Output:
             pairs: The MNN pairs we found
    """
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

def calculate_rwmnn_pairs(X, Y, ref_data, query_data, k1 = None, k2 = None, 
                          filtering = False, k_filter = 100, reduction = 'pca'):
    """Function used to generate the random walk MNN pairs
        Args:
            X:  Gene expression matrix of the reference batch
            Y:  Gene expression matrix of the query batch
            ref_data: Dimension reduction results of the reference batch 
            query_data: Dimension reduction results of the query batch 
            k1: The number of nearest neighbors we used across different batches
            k2: The number of nearest neighbors we used inner one batch
            filtering: A bool value used to determine whether we used top features selection or not
            k_filter: The number of nearest neighbors we used in the filter step
            reduction = The dimension reduction tool we used here. The options of this parameter include: 'cca', 'pca' and 'kpca'
        Output:
            ref_index:  Partial rwMNN pairs index used for the reference dataset
            query_index: Partial rwMNN pairs index used for the query dataset

    """
    if k2 is None:
        k2 = max(int(min(len(ref_data), len(query_data))/100), 1)
    if k1 is None:
        k1 = max(int(k2/2), 1)

    print('Calculating reference pairs...')
    anchor_pairs = acquire_rwmnn_pairs(X, X, k2)
    print('Calculating query pairs...')
    query_pairs = acquire_rwmnn_pairs(Y, Y, k2)
    print('Calculating kNN pairs...')
    pairs = acquire_rwmnn_pairs(X, Y, k1)
    print('Number of MNN pairs is %d' % len(pairs))
    
    print('Calculating random walk MNN (rwMNN) Pairs...')
    anchor_pairs_dict = create_pairs_dict(anchor_pairs)
    query_pairs_dict = create_pairs_dict(query_pairs)
    pair_plus = []
    for x, y in pairs:
        start = (x, y)
        for i in range(50):
            pair_plus.append(start)
            start = (np.random.choice(anchor_pairs_dict[start[0]]), np.random.choice(query_pairs_dict[start[1]]))
    print('Number of rwMNN pairs is %d.' % len(pair_plus))
    
    if filtering and reduction is not None:
        print('Start filtering of selected pairs...')
        pair_plus = filter_pairs(ref_data, query_data, pair_plus, k=k_filter, reduction=reduction, 
                                 ref_reduced=X, query_reduced=Y)
        print('Number of rwMNN pairs after filtering is %d.' % len(pair_plus))
        
    ref_index = [x for x,y in pair_plus]
    query_index = [y for x,y in pair_plus]
    print('Done.')
    return ref_index, query_index

def training_set_generator(ref, query, reduction=None, subsample=None,
                           k1=None, k2=None, norm=True, filtering=False):
    """Function used to generate the training dataset
        Args:
            ref: Gene expression matrix of the reference batch
            query: Gene expression matrix of the query batch
            reduction: The dimension reduction tool we used here. The options of this parameter include: None (raw space), 'cca', 'pca' and 'kpca' 
            subsample: The size of our data used to generate training dataset
            k1: The number of nearest neighbors we used across different batches
            k2: The number of nearest neighbors we used inner one batch
            norm: A bool value used to determine whether we need to normalize the given dataset when searching for the pairs or not
            filtering: A bool value used to determine whether we used top features selection or not
        Output:
            ref[ref_index]: The reference part of the training dataset
            query[query_index]: The query part of the training dataset

    """
    ref_reduced, query_reduced = None, None
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
    elif reduction is None:
        ref_reduced, query_reduced = ref, query
        
    ref_index, query_index = calculate_rwmnn_pairs(ref_reduced, query_reduced, ref, query, k1=k1, k2=k2, filtering=filtering)
    print('Dimension of paired data is %d.' % len(ref_index))
        
    return ref[ref_index], query[query_index]

def order_selection(adata, key='batch', orders=None):
    '''Function used to determine the training sequence based on the variance across genes.
        Args:
            adata: The given dataset in AnnData form
            key: The index name of batch information in the given dataset
            orders: The batch sequence for training or none
        Output:
            orders: The batch sequence for training
    '''
    if orders == None:
        batch_list = list(set(adata.obs[key].values))
        adata_values = [np.array(adata.X[adata.obs[key] == batch]) for batch in batch_list]
        std_ = [np.sum(np.std(item, axis=0)) for item in adata_values]
        orders = np.argsort(std_)[::-1]
        return [batch_list[i] for i in orders] 
    else:
        return orders

def generate_target_dataset(adata, batch_list):
    """A simple function used to rearrange the batch index of our given dataset.
    """
    adata0 = adata[adata.obs['batch'] == batch_list[0]]
    for i in batch_list[1:]:
        adata0 = adata0.concatenate(adata[adata.obs['batch'] == i], batch_key='batch_key', index_unique=None)
    return adata0

def run_respan(adata, batch_key='batch', order=None, epoch=300, batch=1024, lambda_1=10., 
               reduction='pca', subsample=3000, k1=None, k2=None, filtering=False,
               n_critic=10, seed=999, b1=0.9, b2=0.999, lr=0.0001, opt='AdamW'):
    '''The main entry of our model
        Args:
            adata: Data with batch effect
            batch_key:  The index name of batch information in the given dataset
            order: The batch sequence for training or none
            epoch: The number of iteraiton steps
            batch: Input batch 
            lambda_1: A hyperparameter used to control the weights of gradient penalty
            reduction: Method we used for dimension reduction, including None, 'cca', 'pca' and 'kpca' 
            subsample: The size of our data used to generate training dataset
            k1: The number of nearest neighbors we used across different batches
            k2: The number of nearest neighbors we used inner one batch
            filtering: A bool value used to determine whether we used top features selection or not
            n_critic: A step value used to determine the training times of generator 
            seed: Random seed we used in our training process
            b1,b2: Hyperparameters used in AdamW optimizer
            lr: Learning rate
            opt: The name of our optimizer
        Output:
            adata_new: Results after batch correction
    '''
    # Set a seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    if not order:
        order = order_selection(adata, key=batch_key)
    print('The sequential mapping order will be: ', '->'.join(order))
        
    adata = generate_target_dataset(adata, order)
    ref_data_ori = adata[adata.obs['batch'] == order[0]].X

    for bat in order[1:]:
        print("########################## Mapping %s to the reference data #####################"%(bat))
        batch_data_ori = adata[adata.obs['batch'] == bat].X
        label_data, train_data = training_set_generator(ref_data_ori, batch_data_ori, reduction=reduction,
                                                        subsample=subsample, k1=k1, k2=k2, filtering=filtering)
        print("######################## Finish pair finding ########################")
        remove_batch_data, G_tar = train(label_data, train_data, epoch, batch, lambda_1, batch_data_ori,
                                         n_critic=n_critic, b1=b1, b2=b2, lr=lr, opt=opt)
        ref_data_ori = np.vstack([ref_data_ori, remove_batch_data])
    print("######################## Finish all batch correction ########################")
    
    adata_new = sc.AnnData(ref_data_ori)
    adata_new.obs['batch'] = list(adata.obs['batch'])
    if adata.obs.columns.str.contains('celltype').any():
        adata_new.obs['celltype'] = list(adata.obs['celltype'])
    adata_new.var_names = adata.var_names
    adata_new.var_names.name = 'Gene'
    adata_new.obs_names = adata.obs_names
    adata_new.obs_names.name = 'CellID'
    
    return adata_new
