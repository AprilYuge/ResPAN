# ResPAN
This reporsity contains the codes used in the paper _ResPAN: a powerful batch correction model for scRNA-seq data through residual adversarial networks_.

This tool is a light structured **Res**idual autoencoder and mutual nearest neighbor **P**aring guided **A**dversarial **N**etwork for batch effect correction.


# Download 
To download and install this tool, please use this instruction:
```
git clone https://github.com/AprilYuge/ResPAN.git
```
# Brief tutorial
This section contains the steps of applying our tool on real datasets.

To run our method, the first thing is to import necessary packages:
```
import awgan
import scprep
import numpy as np
import pandas as pd
import graphtools as gt
import os
import scanpy as sc
from skmisc.loess import loess

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F  

import sklearn.preprocessing as preprocessing
import seaborn as sns
from sklearn.neighbors import NearestNeighbors, KDTree

```
Then we need to load the scRNA-seq data with batch information:
```
# data loading
adata = sc.read_loom('CL_raw.loom', sparse=False) 
# pre-processing
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='batch')
adata = adata[:, adata.var['highly_variable']]

```
Now we can directly load the codes of our method and run them to get the output results as an AnnData file:
```
adata_new = sequencing_train(adata, key='batch', epoch=300, batch=1024, reduction='pca', mode='rwMNN', metric='angular', subsample=3000, filtering=False, opt='AdamW', lr=0.0001)
```

To visualize our results, we can use these commands:
```
sc.tl.pca(adata_new, 20, svd_solver='arpack')
sc.pp.neighbors(adata_new)
sc.tl.umap(adata_new)
sc.set_figure_params(figsize=(5,5),fontsize=12)
sc.pl.umap(adata_new, color=['batch', 'celltype'], frameon=False, show=False)
```

For the batch correction on simulation datasets and benchmarking analysis, please refer this path of our project:
https://github.com/AprilYuge/ResPAN/tree/main/tutorials

# Code reference

To implement our tool, we referred [WGAN-GP](https://github.com/Zeleni9/pytorch-wgan) for the structure of Generative Adversarial Network and [iMAP](https://github.com/Svvord/iMAP) for random walk mutual nearest neighbors method. Many thanks to their open-source treasure.

# Citation
To be continued...


