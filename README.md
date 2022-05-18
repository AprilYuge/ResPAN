# ResPAN: a light structured **Res**idual autoencoder and mutual nearest neighbor **P**aring guided **A**dversarial **N**etwork for scRNA-seq batch correction

This reporsity contains code and information of data used in the paper “*ResPAN: a powerful batch correction model for scRNA-seq data through residual adversarial networks*”. Source code for ResPAN are in the [ResPAN](https://github.com/AprilYuge/ResPAN/tree/main/ResPAN) folder, scipts for reproducing benchmarking results are in the [scripts](https://github.com/AprilYuge/ResPAN/tree/main/scripts) folder, and data information can be found in the [data](https://github.com/AprilYuge/ResPAN/tree/main/data) folder.

The workflow of ResPAN contains three key steps: generation of training data, adversarial training of the neural network, and generation of corrected data without batch effect. A figure summary is shown below.

![alt text](https://github.com/AprilYuge/ResPAN/blob/main/images/workflow.png).

More details about ResPAN can be found in our [manuscript](https://www.biorxiv.org/content/10.1101/2021.11.08.467781v3.full).

### Package requirement

ResPAN is implemented in Python. Before downloading and installing ResPAN, some packages need to be installed first. To use ResPAN, PyTorch is required. Other packages required for ResPAN and their versions used in our manuscript are listed below.

### Download 

To download and install ResPAN, please copy and paste the following line to your terminal:
```
git clone https://github.com/AprilYuge/ResPAN.git
```

### Brief tutorial

A brief tutorial of using ResPAN can be found below and under the folder [tutorials](https://github.com/AprilYuge/ResPAN/tree/main/tutorials).

To run our method, the first thing is to import necessary packages:
```
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from ResPAN import run_respan
```

Then we need to load the scRNA-seq data with batch information and preprocess it before running ResPAN:
```
# data loading
adata = sc.read_loom('CL_raw.loom', sparse=False) 
# pre-processing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='batch')
adata = adata[:, adata.var['highly_variable']]
```

Now we can run ResPAN on the preprocessed data for batch correction. The output result is an AnnData object:
```
adata_new = run_respan(adata, batch_key='batch', epoch=300, batch=1024, reduction='pca', subsample=3000)
```
As indicated in our manuscipt, we use PCA for dimensionality reduction, kPCA (`reduction='kpca'`) and CCA (`reduction='cca'`) are also implemented, but their performance were not as good as PCA. Meanwhile, we subsampled cells in each batch to 3,000 before finding random walk MNN pairs [1].

To visualize our results, we can use the following commands:
```
sc.tl.pca(adata_new, 20, svd_solver='arpack')
sc.pp.neighbors(adata_new)
sc.tl.umap(adata_new)
sc.set_figure_params(figsize=(5,5),fontsize=12)
sc.pl.umap(adata_new, color=['batch', 'celltype'], frameon=False, show=False)
```

### Code referencew

For the implementation of ResPAN, we referred to [WGAN-GP](https://github.com/Zeleni9/pytorch-wgan) for the structure of Generative Adversarial Network and [iMAP](https://github.com/Svvord/iMAP) for the random walk mutual nearest neighbor method. Many thanks to their open-source treasure.

# Citations
[1] Wang, Dongfang, et al. "iMAP: integration of multiple single-cell datasets by adversarial paired transfer networks." Genome biology 22.1 (2021): 1-24.


