Tutorial
==================================

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
# check if data is in sparse format
if isinstance(adata.X, scipy.sparse.csr.csr_matrix): 
    adata_new = sc.AnnData(adata.X.todense())
    adata_new.obs = adata.obs.copy()
    adata_new.obs_names = adata.obs_names
    adata_new.var_names = adata.var_names
    adata_new.obs_names.name = 'CellID'
    adata_new.var_names.name = 'Gene'
    del adata
    adata = adata_new
```

Now we can run ResPAN on the preprocessed data for batch correction. The output result is an AnnData object:
```
adata_new = run_respan(adata, batch_key='batch', epoch=300, batch=1024, reduction='pca', subsample=3000, seed=999)
```
As indicated in our manuscipt, we use PCA for dimensionality reduction, kPCA (`reduction='kpca'`) and CCA (`reduction='cca'`) are also implemented, but their performance were not as good as PCA. Meanwhile, we subsampled cells in each batch to 3,000 before finding random walk MNN pairs [1].

To visualize our results, we can use the following commands:
```
adata_new.raw = adata_new
sc.pp.scale(adata_new, max_value=10)
sc.tl.pca(adata_new, 20, svd_solver='arpack')
sc.pp.neighbors(adata_new)
sc.tl.umap(adata_new)
sc.set_figure_params(figsize=(5,5),fontsize=12)
sc.pl.umap(adata_new, color=['batch', 'celltype'], frameon=False, show=False)
```
