import sys
import scanpy as sc
import pandas as pd
import numpy as np
from metrics import calculate_metrics
import scipy
import re

# datanames = ['CL', 'DC', 'Pancrm', 'PBMC368k', 'HumanPBMC', 'MHSP', 'MCA', 'Lung', 'MouseRetina', 'HCA', 'MouseBrain']
# methods = ['raw', 'imap', 'mnn', 'scvi', 'harmony', 'bbknn', 'liger', 'seurat']

dataname = str(sys.argv[1]) 
method = str(sys.argv[2])
if bool(re.match('Sim[0-9]*.', dataname)):
    folder = str(sys.argv[3])
elif len(sys.argv) == 3+1:
# elif dataname == 'HCA' or dataname == 'MouseBrain':
    seed = int(sys.argv[3])

npcs = 20
is_raw = False
is_embed = False
embed = 'X_pca'
if dataname.startswith('Sim'):
    is_simul = True
else:
    is_simul = False
if dataname in ['MCA', 'MouseRetina', 'MouseBrain', 'MHSP']:
    org = 'mouse'
else:
    org = 'human'

# Read in raw data
if 'folder' in locals():
    if folder == 'baseline':
        adata_raw = sc.read_loom('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/%s/%s_raw.loom' % (dataname, dataname), 
                             sparse=False)  #Load cell line dataset(-> count data). 
    else:
        adata_raw = sc.read_loom('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/%s/%s/%s_raw.loom' % (dataname, folder, dataname), 
                             sparse=False)  #Load cell line dataset(-> count data). 
else:
    if dataname == 'HCA' or dataname == 'MouseBrain':
        adata_raw = sc.read_h5ad('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_raw.h5ad' % (dataname, dataname))
    else:
        adata_raw = sc.read_loom('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_raw.loom' % (dataname, dataname), 
                             sparse=False)  #Load cell line dataset(-> count data). 
# sc.pp.filter_cells(adata_raw, min_genes=200)
# sc.pp.filter_genes(adata_raw, min_cells=3)
sc.pp.normalize_per_cell(adata_raw, counts_per_cell_after=1e4)
sc.pp.log1p(adata_raw)
#sc.pp.highly_variable_genes(adata_raw, n_top_genes=2000, batch_key='batch')
#adata_raw = adata_raw[:, adata.var['highly_variable']]
adata_raw.raw = adata_raw
sc.pp.scale(adata_raw, max_value=10)
if method == 'seurat':
    adata_raw.var_names = ['-'.join(gene.split('_')) for gene in adata_raw.var_names]

# Read in corrected data
if method == 'raw':
    is_raw = True
    adata = adata_raw
else:
    if 'folder' in locals():
        if folder == 'baseline':
            adata = sc.read_h5ad('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/%s/%s_%s.h5ad' % (dataname, dataname, method))
        else:
            adata = sc.read_h5ad('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/%s/%s/%s_%s.h5ad' % (dataname, folder, dataname, method))
    elif method == 'seurat' or method == 'mnn':
        adata = sc.read_h5ad('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_%s.h5ad' % (dataname, dataname, method))
    else:
        adata = sc.read_h5ad('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_%s-seed%d.h5ad' % (dataname, dataname, method, seed))
    if method == 'mnn':
        adata.obs_names = ['-'.join(g.split('-')[:-1]) for g in adata.obs_names]
    if method in ['scvi', 'harmony']:
        adata.obsm['X_embed'] = adata.obsm['X_pca']
        is_embed = True
        embed = 'X_embed'
    if method == 'liger':
        if isinstance(adata.X, scipy.sparse.csr.csr_matrix):
            adata.obsm['X_pca'] = adata.X.todense().copy()
        else:
            adata.obsm['X_pca'] = adata.X.copy()
        adata.obsm['X_embed'] = adata.obsm['X_pca']
        is_embed = True
        embed = 'X_embed'
    adata = adata[adata_raw.obs_names, :].copy()
if dataname == 'HCA':
    adata.obs['celltype'] = 'NA'
    adata.obs['celltype'] = adata.obs['celltype'].astype('category')

print(dataname, method)

if 'X_pca' not in adata.obsm.keys():
    adata.raw = adata
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, 20, svd_solver='arpack')

if 'X_pca' not in adata_raw.obsm.keys():
    try:
        adata_raw.obsm['X_pca'] = np.load('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_raw_20pc.npy' % (dataname, dataname))
    except:
        sc.tl.pca(adata_raw, 20, svd_solver='arpack')
#     adata_new = sc.AnnData(adata_raw.obsm['X_pca'])
#     adata_new.obsm['X_pca'] = adata_raw.obsm['X_pca']
#     adata_new.obs = adata_raw.obs.copy()
#     del adata_raw
#     adata = adata_new
#     adata_raw = adata_new
    
DEGpath = None
if dataname == 'HCA':
    if method == 'seurat' or method == 'mnn':
        savepath = '/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_%s_%d_metrics.csv' % (dataname, dataname, method, seed)
    else:
        savepath = '/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_%s-seed%d_%d_metrics.csv' % (dataname, dataname, method, seed, seed)
    calculate_metrics(adata, adata_raw=adata_raw, celltype_key=None, savepath=savepath, is_raw=is_raw, is_embed=is_embed,
                      embed=embed, is_simul=is_simul, org=org, dataname=dataname, subsample=0.1, seed=seed)
elif dataname == 'MouseBrain':
    if method == 'seurat' or method == 'mnn':
        savepath = '/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_%s_%d_metrics.csv' % (dataname, dataname, method, seed)
    else:
        savepath = '/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_%s-seed%d_%d_metrics.csv' % (dataname, dataname, method, seed, seed)
    calculate_metrics(adata, adata_raw=adata_raw, savepath=savepath, is_raw=is_raw, is_embed=is_embed, embed=embed,
                      is_simul=is_simul, org=org, dataname=dataname, subsample=0.1, seed=seed)
elif 'folder' in locals():
    if folder == 'baseline':
        savepath = '/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/%s/%s_%s_metrics.csv' % (dataname, dataname, method)
    else:
        savepath = '/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/%s/%s/%s_%s_metrics.csv' % (dataname, folder, dataname, method)
    DEGpath = '/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/%s/geneinfo.txt' % (dataname)
    calculate_metrics(adata, adata_raw=adata_raw, savepath=savepath, is_raw=is_raw, is_embed=is_embed, embed=embed,
                      is_simul=is_simul, org=org, dataname=dataname, DEGpath=DEGpath)
else:
    if method == 'seurat' or method == 'mnn':
        savepath = '/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_%s_metrics.csv' % (dataname, dataname, method)
    else:
        savepath = '/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_%s-seed%d_metrics.csv' % (dataname, dataname, method, seed)
    if is_simul:
        DEGpath = '/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/geneinfo.txt' % (dataname)
    calculate_metrics(adata, adata_raw=adata_raw, savepath=savepath, is_raw=is_raw, is_embed=is_embed, embed=embed,
                      is_simul=is_simul, org=org, dataname=dataname, DEGpath=DEGpath)

