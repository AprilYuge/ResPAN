
import scanpy as sc

def data_preprocessing(adata):
    """Function used to preprocess our data with batch effect
    """
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
    return adata