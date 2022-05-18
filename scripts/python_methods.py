import sys
import scanpy as sc
import pandas as pd
import numpy as np
import time
import scipy

# Parameters
npcs = 20
dataname = str(sys.argv[1]) # ['CL', 'DC', 'Pancrm', 'PBMC368k', 'HumanPBMC', 'MHSP', 'MCA', 'Lung', 'MouseRetina', 'HCA', 'MouseBrain']
method = str(sys.argv[2]) # ['imap', 'mnn', 'scvi', 'harmony', 'bbknn']
only_pca = False
#return_pca = False

if len(sys.argv) == 3+1:
    folder = str(sys.argv[3])
    if folder.startswith('common'):
        epoch = 50
    elif folder.startswith('batch'):
        epoch = 100
    else:
        epoch = 300
    if folder == 'baseline':
        adata = sc.read_loom('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/%s/%s_raw.loom' % (dataname, dataname), 
                             sparse=False)  #Load cell line dataset(-> count data). 
    else:
        adata = sc.read_loom('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/%s/%s/%s_raw.loom' % (dataname, folder, dataname), 
                             sparse=False)  #Load cell line dataset(-> count data). 
else:
    # Read data and preprocess
    if dataname == 'HCA' or dataname == 'MouseBrain':
        adata = sc.read_h5ad('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_raw.h5ad' % (dataname, dataname))
    else:
        adata = sc.read_loom('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_raw.loom' % (dataname, dataname), 
                             sparse=False)  #Load cell line dataset(-> count data). 
     

# # Check if there is a counts layer
# if 'counts' in adata.layers.keys():
#     adata.X = adata.layers['counts'].copy()   
start = time.time()
# sc.pp.filter_cells(adata, min_genes=200)
# sc.pp.filter_genes(adata, min_cells=3)
# sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', batch_key='batch')
if method == 'scvi':
    adata.layers['counts'] = adata.X.copy()  
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='batch')
adata = adata[:, adata.var['highly_variable']]

if method == 'imap':
    if isinstance(adata.X, scipy.sparse.csr.csr_matrix): 
        adata_new = sc.AnnData(adata.X.todense())
        adata_new.obs = adata.obs.copy()
        adata_new.obs_names = adata.obs_names
        adata_new.var_names = adata.var_names
        adata_new.obs_names.name = 'CellID'
        adata_new.var_names.name = 'Gene'
        del adata
        adata = adata_new
    import imap
    ### Stage I
    EC, ec_data = imap.stage1.iMAP_fast(adata, key="batch", seed=0) 
    ### Stage II
    if min(adata.obs.batch.value_counts()) < 100:
        k1 = k2 = 1
    else:
        k1 = k2 = None
    output_results = imap.stage2.integrate_data(adata, ec_data, k1=k1, k2=k2, seed=0)
    adata_correct = adata
    adata_correct.X = output_results
elif method == 'respan':
    if isinstance(adata.X, scipy.sparse.csr.csr_matrix): 
        adata_new = sc.AnnData(adata.X.todense())
        adata_new.obs = adata.obs.copy()
        adata_new.obs_names = adata.obs_names
        adata_new.var_names = adata.var_names
        adata_new.obs_names.name = 'CellID'
        adata_new.var_names.name = 'Gene'
        del adata
        adata = adata_new
    from ResPAN import run_respan
    adata_correct = run_respan(adata, batch_key='batch', epoch=300, batch=1024, reduction='pca', subsample=3000)
elif method == 'mnn':
    import mnnpy
    temp = [adata[adata.obs['batch'] == batch] for batch in list(set(adata.obs['batch']))]
    adata_correct = mnnpy.mnn_correct(*temp, batch_key = 'batch', batch_categories=list(set(adata.obs['batch'])))[0] 
elif method == 'scvi':
    import scvi
    scvi.settings.seed = 0
    adata.layers['counts'] = np.array(adata.layers['counts'])
    scvi.model.SCVI.setup_anndata(adata, batch_key='batch', layer="counts")
    model = scvi.model.SCVI(adata, n_latent=npcs)
    model.train()
    adata.obsm['X_pca'] = model.get_latent_representation()
elif method == 'harmony':
    import harmonypy as harmony
    adata.raw = adata
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, npcs, svd_solver='arpack')
    output_results = harmony.run_harmony(adata.obsm['X_pca'], adata.obs, vars_use=['batch'], random_state=0)
    adata.obsm['X_pca'] = output_results.Z_corr.T
elif method == 'bbknn':
    import bbknn
    adata.raw = adata
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, npcs, svd_solver='arpack')
    bbknn.bbknn(adata)
    
end = time.time()
print('Running time on %s for %s is %.2f sec' % (dataname, method, end-start))

# Save results
if method == 'scvi' or method == 'harmony' or method == 'bbknn':
    adata_correct = adata
    # For scvi and harmony, no need to save the data in original space because the two
    # methods do not operate or correct the data on that space
    if only_pca:
        if not method == 'bbknn':
            adata_correct.X = adata_correct.obsm['X_pca']
            del adata_correct.obsm['X_pca']
else:
    adata_correct.raw = adata_correct
    sc.pp.scale(adata_correct, max_value=10)
    sc.tl.pca(adata_correct, npcs, svd_solver='arpack')

if len(sys.argv) == 3+1:
    if folder == 'baseline':
        adata_correct.write_h5ad('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/%s/%s_%s_k1020.h5ad' % (dataname, dataname, method))
    else:
        adata_correct.write_h5ad('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/%s/%s/%s_%s_k1020.h5ad' % (dataname, folder, 
                                                                                                    dataname, method)) 
else:
    adata_correct.write_h5ad('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/%s/%s_%s.h5ad' % (dataname, dataname, method))

# # Plot UMAP
# if method != 'bbknn':
#     sc.pp.neighbors(adata_correct)
# sc.tl.umap(adata_correct)
# sc.set_figure_params(figsize=(5,5),fontsize=12)
# sc.pl.umap(adata_correct, color=['batch'], save='_batch_%s_%s.h5ad' % (dataname, method))
# sc.pl.umap(adata_correct, color=['celltype'], save='_celltype_%s_%s.h5ad' % (dataname, method))

