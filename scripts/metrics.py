import scanpy as sc
import scib 
import numpy as np 
import pandas as pd
from sklearn.metrics import silhouette_score
import random
from sklearn.neighbors import NearestNeighbors 
from sklearn.neighbors import KDTree
from sklearn import preprocessing
import scipy

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()

rscript = '''
library(kBET)
library(lisi)
'''
robjects.r(rscript)
#kbet = robjects.r('kBET')
lisi = robjects.r['compute_lisi']

def calculate_ASW(adata, labels=None, total_cells=None, percent_extract=0.8, batch_key='batch', celltype_key='celltype', verbose=True):
    random.seed(0)
    np.random.seed(0)
    min_val = -1
    max_val = 1
    asw_f1 = []
    asw_b = []
    asw_c = []
    if total_cells:
        total_cells_ = total_cells
    
    for i in range(20):
        
        if celltype_key is None:
            # Get a subset of data
            rand_idx = np.random.choice(adata.obs_names, size=int(len(adata.obs_names) * percent_extract), replace=False)
            adata_sub = adata[rand_idx]

            # Calculate 1-bASW
            asw_batch = silhouette_score(adata_sub.obsm['X_pca'], adata_sub.obs[batch_key])
            asw_batch_norm = (asw_batch - min_val) / (max_val - min_val)
            asw_b.append(1-asw_batch_norm)
            # Print scores
            asw_b = np.array(asw_b).mean()
            asw_c = asw_f1 = np.nan
        else:
            # Get a subset of data
            rand_idx = np.random.choice(adata.obs_names, size=int(len(adata.obs_names) * percent_extract), replace=False)
            adata_sub = adata[rand_idx]

            # Calculate cASW
            asw_celltype = silhouette_score(adata_sub.obsm['X_pca'], adata_sub.obs[celltype_key])
            asw_celltype_norm = (asw_celltype - min_val) / (max_val - min_val)
            asw_c.append(asw_celltype_norm)

            # Calculate 1-bASW
            temp = 0
            total_cells = total_cells_
            for label in labels:
                adata_sub_c = adata_sub[adata_sub.obs[celltype_key] == label]
                if len(set(adata_sub_c.obs[batch_key])) == 1 or adata_sub_c.shape[0] < 10:
                    total_cells -= adata_sub_c.shape[0]
                else:
                    asw_batch = silhouette_score(adata_sub_c.obsm['X_pca'], adata_sub_c.obs[batch_key])
                    asw_batch_norm = (asw_batch - min_val) / (max_val - min_val)
                    temp += (1-asw_batch_norm)*adata_sub_c.shape[0]
            temp /= total_cells
            asw_b.append(temp)

            # Calcualte F1 score
            asw_fscore = (2 * (temp)*(asw_celltype_norm))/(temp + asw_celltype_norm)
            asw_f1.append(asw_fscore)
 
            # Print scores
            asw_c = np.array(asw_c).mean()
            asw_b = np.array(asw_b).mean()
            asw_f1 = np.array(asw_f1).mean()
            
        if verbose:
            print('cASW: %.4f, 1-bASW: %.4f, F1 ASW: %.4f' % (asw_c, asw_b, asw_f1))
        return asw_c, asw_b, asw_f1

# kBET
def calculate_kBET(adata, labels=None, total_cells=None, batch_key='batch', celltype_key='celltype', verbose=True):
    value = 0
    if celltype_key is None:
        quarter_mean = np.floor(np.mean(adata.obs[batch_key].value_counts())*0.25).astype('int')
        k0 = np.min([100, np.max([10, quarter_mean])])
        robjects.globalenv['data_pca'] = adata.obsm['X_pca']
        robjects.globalenv['batch_label'] = np.array(adata.obs[batch_key])
        robjects.globalenv['k0'] = k0
        rscript = '''
        set.seed(0)
        batch_est <- kBET(data_pca, batch_label, k0=k0,
        plot=FALSE, do.pca=FALSE, heuristic=FALSE, adapt=FALSE)
        '''
        robjects.r(rscript)
        kbet_re = robjects.r("batch_est$summary$kBET.observed[1]")[0]
        value += (1-kbet_re)
    else:
        for label in labels:
            adata_sub = adata[adata.obs[celltype_key] == label]
            quarter_mean = np.floor(np.mean(adata_sub.obs[batch_key].value_counts())*0.25).astype('int')
            k0 = np.min([100, np.max([10, quarter_mean])])
            robjects.globalenv['data_pca'] = adata_sub.obsm['X_pca']
            robjects.globalenv['batch_label'] = np.array(adata_sub.obs[batch_key])
            robjects.globalenv['k0'] = k0
            rscript = '''
            set.seed(0)
            batch_est <- kBET(data_pca, batch_label, k0=k0,
            plot=FALSE, do.pca=FALSE, heuristic=FALSE, adapt=FALSE)
            '''
            robjects.r(rscript)
            kbet_re = robjects.r("batch_est$summary$kBET.observed[1]")[0]
            value += (1-kbet_re)*adata_sub.shape[0]
        value /= total_cells
 
    # Print scores
    if verbose:
        print('kBET: %.4f' % (value))
    
    return value

# LISI
def calculate_LISI(adata, labels=None, total_cells=None, batch_key='batch', celltype_key='celltype', verbose=True):
    if celltype_key is None:
        # Calculate bLISI
        lisi_b = 0
        lisi_c = lisi_f1 = np.nan
        if adata.shape[0] < 90:
            perplexity = int(adata.shape[0]/6)
        else:
            perplexity = 30
        lisi_res = lisi(adata.obsm['X_pca'], adata.obs, batch_key, perplexity=perplexity)
        lisi_b = (np.mean(lisi_res)[batch_key]-1.)/(len(set(adata.obs[batch_key]))-1.)
    else:
        # Calculate 1-cLISI
        lisi_res = lisi(adata.obsm['X_pca'], adata.obs, celltype_key)
        lisi_c = 1 - (np.mean(lisi_res)[celltype_key]-1.)/(len(set(adata.obs[celltype_key]))-1.)

        # Calculate bLISI
        lisi_b = 0
        for label in labels:
            adata_sub = adata[adata.obs[celltype_key] == label]
            if adata_sub.shape[0] < 90:
                perplexity = int(adata_sub.shape[0]/6)
            else:
                perplexity = 30
            lisi_res = lisi(adata_sub.obsm['X_pca'], adata_sub.obs, batch_key, perplexity=perplexity)
            lisi_batch = (np.mean(lisi_res)[batch_key]-1.)/(len(set(adata_sub.obs[batch_key]))-1.)
            lisi_b += lisi_batch*adata_sub.shape[0]
        lisi_b /= total_cells

        # Calcualte F1 score
        lisi_f1 = (2*lisi_c*lisi_b)/(lisi_c + lisi_b)
    
    # Print scores
    if verbose:
        print('1-clisi: %.4f, blisi: %.4f, F1 lisi: %.4f' % (lisi_c, lisi_b, lisi_f1))
    
    return lisi_c, lisi_b, lisi_f1

# Preservation of kNN graph
def calculate_knn_sim(adata_pre, adata_post, batch_key='batch', k=None, distance='cosine', use_raw=False, embed='X_pca'):
    knn_sim = []
    for b in set(adata_pre.obs[batch_key]):
        if use_raw:
            if isinstance(adata_pre.X, scipy.sparse.csr.csr_matrix):
                data_pre = adata_pre[adata_pre.obs[batch_key] == b].X.todense()
            else:
                data_pre = adata_pre[adata_pre.obs[batch_key] == b].X
            if isinstance(adata_post.X, scipy.sparse.csr.csr_matrix):
                data_post = adata_post[adata_post.obs[batch_key] == b].X.todense()
            else:
                data_post = adata_post[adata_post.obs[batch_key] == b].X
        else:
            data_pre = adata_pre[adata_pre.obs[batch_key] == b].obsm[embed]
            data_post = adata_post[adata_post.obs[batch_key] == b].obsm[embed]
        if k is None:
            k = int(max(5, min(50, (data_pre.shape[0]*0.01))))
            print(k)
        if distance == 'cosine':
            data_pre = data_pre / np.linalg.norm(data_pre, axis=1, keepdims=True)
            data_post = data_post / np.linalg.norm(data_post, axis=1, keepdims=True)
        neigh_pre = NearestNeighbors(n_neighbors=k+1).fit(data_pre)
        neigh_post = NearestNeighbors(n_neighbors=k+1).fit(data_post)
        knns_pre = neigh_pre.kneighbors(data_pre, return_distance=False)[:,1:]
        knns_post = neigh_post.kneighbors(data_post, return_distance=False)[:,1:]
        value = 0
        for i in range(knns_pre.shape[0]):
            inter = len(set(knns_pre[i]).intersection(knns_post[i]))
            value += inter/(2*k-inter) # Calculate Jaccard similarity
        value /= knns_pre.shape[0]
        knn_sim.append(value)
    knn_sim = np.array(knn_sim).mean()
    return knn_sim

# Preservation of cell-cell similarity
def calculate_CC_sim(adata_pre, adata_post, batch_key='batch', rate = 1, use_raw=False, embed='X_pca'):
    CC_sim = []
    for b in set(adata_pre.obs[batch_key]):
        if use_raw:
            if isinstance(adata_pre.X, scipy.sparse.csr.csr_matrix):
                data_pre = adata_pre[adata_pre.obs[batch_key] == b].X.todense()
            else:
                data_pre = adata_pre[adata_pre.obs[batch_key] == b].X
            if isinstance(adata_post.X, scipy.sparse.csr.csr_matrix):
                data_post = adata_post[adata_post.obs[batch_key] == b].X.todense()
            else:
                data_post = adata_post[adata_post.obs[batch_key] == b].X
        else:
            data_pre = adata_pre[adata_pre.obs[batch_key] == b].obsm[embed]
            data_post = adata_post[adata_post.obs[batch_key] == b].obsm[embed]
        data_pre = data_pre / np.linalg.norm(data_pre, axis=1, keepdims=True)
        data_post = data_post / np.linalg.norm(data_post, axis=1, keepdims=True)
        sim_pre = data_pre @ data_pre.T
        sim_post = data_post @ data_post.T
        CC_sim.append(np.exp(-rate*np.abs((sim_pre-sim_post)).mean()))
    CC_sim = np.array(CC_sim).mean()
    return CC_sim

# Similarity between expression data matrix before and after correction
def calculate_exp_sim(adata_pre, adata_post, batch_key='batch', direction='cell', use_raw=False, embed='X_pca'):
    if direction == 'cell':
        if use_raw:
            if isinstance(adata_pre.X, scipy.sparse.csr.csr_matrix):
                data_pre = adata_pre.X.todense()
            else:
                data_pre = adata_pre.X
            if isinstance(adata_post.X, scipy.sparse.csr.csr_matrix):
                data_post = adata_post.X.todense()
            else:
                data_post = adata_post.X
        else:
            data_pre = adata_pre.obsm[embed]
            data_post = adata_post.obsm[embed]
        data_pre = data_pre / np.linalg.norm(data_pre, axis=1, keepdims=True)
        data_post = data_post / np.linalg.norm(data_post, axis=1, keepdims=True)
        exp_sim = (np.array(data_pre) * np.array(data_post)).sum(1).mean()
    elif direction == 'gene':
        exp_sim = []
        for b in set(adata_pre.obs[batch_key]):
            if use_raw:
                if isinstance(adata_pre.X, scipy.sparse.csr.csr_matrix):
                    data_pre = adata_pre[adata_pre.obs[batch_key] == b].X.todense()
                else:
                    data_pre = adata_pre[adata_pre.obs[batch_key] == b].X
                if isinstance(adata_post.X, scipy.sparse.csr.csr_matrix):
                    data_post = adata_post[adata_post.obs[batch_key] == b].X.todense()
                else:
                    data_post = adata_post[adata_post.obs[batch_key] == b].X
            else:
                data_pre = adata_pre[adata_pre.obs[batch_key] == b].obsm[embed]
                data_post = adata_post[adata_post.obs[batch_key] == b].obsm[embed]
            data_pre = data_pre / np.linalg.norm(data_pre, axis=0, keepdims=True)
            data_post = data_post / np.linalg.norm(data_post, axis=0, keepdims=True)
            exp_sim.append((data_pre * data_post).sum(0).mean())
        exp_sim = np.array(exp_sim).mean()
    return exp_sim

# Positive and true positive cells defined in iMAP
def positive_true_positive(adata, batch_key='batch', celltype_key='celltype', use_raw=False,
                           k1=20, k2=100, tp_thr=3., distance='cosine', embed='X_pca'):
    celltype_list = adata.obs[celltype_key]
    batch_list = adata.obs[batch_key]

    temp_c = adata.obs[celltype_key].value_counts()
    temp_b = pd.crosstab(adata.obs[celltype_key], adata.obs[batch_key])
    temp_b_prob = temp_b.divide(temp_b.sum(1), axis=0)
    
    if use_raw:
        if isinstance(adata.X, scipy.sparse.csr.csr_matrix):
            X = adata.X.todense()
        else:
            X = adata.X
    else:
        X = adata.obsm[embed]
    if distance == 'cosine':
        X = preprocessing.normalize(X, axis=1)

    t1 = KDTree(X)

    p_list = []
    tp_list = []

    for cell in range(len(X)):

        # Discriminate positive cells
        neig1 = min(k1, temp_c[celltype_list[cell]])
        NNs = t1.query(X[cell].reshape(1,-1), neig1+1, return_distance=False)[0, 1:]
        c_NN = celltype_list[NNs]
        true_rate = sum(c_NN == celltype_list[cell])/neig1
        if true_rate > 0.5:
            p_list.append(True)
        else:
            p_list.append(False)

        # Discriminate true positive cells
        if p_list[cell] == True:
            neig2 = min(k2, temp_c[celltype_list[cell]])
            NNs = t1.query(X[cell].reshape(1,-1), neig2, return_distance=False)[0]
            NNs_c = celltype_list[NNs]
            NNs_i = NNs_c == celltype_list[cell]
            NNs = NNs[NNs_i] # get local neighbors that are from the same cell type
            neig2 = len(NNs)
            NNs_b = batch_list[NNs]

            max_b = 0
            b_prob = temp_b_prob.loc[celltype_list[cell]]
            for b in set(batch_list):
                if b_prob[b] > 0 and b_prob[b] < 1:
                    p_b = sum(NNs_b == b)
                    stat_b = abs(p_b - neig2*b_prob[b]) / np.sqrt(neig2*b_prob[b]*(1-b_prob[b]))
                    max_b = max(max_b, stat_b)
            if max_b <= tp_thr:
                tp_list.append(True)
            else:
                tp_list.append(False)
        else:
            tp_list.append(False)

    pos_rate = sum(p_list)/len(p_list)
    truepos_rate = sum(tp_list)/len(tp_list)
    return pos_rate, truepos_rate

def generate_true_tags(genelist, groups):
    gene_list = []
    group_list = []
    for i in groups:
        gene_i = genelist[genelist['DEFac'+i]!=1]
        gene_list += list(gene_i['Gene'])
        group_list += [i for _ in range(len(gene_i))]
        
    true_degs = pd.DataFrame()
    true_degs['gene'] = gene_list 
    true_degs['group'] = group_list
    return true_degs

def access_degs(adata, group_list, p_val=0.05, celltype_key='celltype'):
    adata = adata[adata.obs[celltype_key].isin(group_list)]
    sc.tl.rank_genes_groups(adata, groupby='celltype', method='wilcoxon')
    group_genelist = []
    group_grouplist = []
    group_pvallist = []
    
    df_new_degs = pd.DataFrame()
    result = adata.uns['rank_genes_groups']
    for i in group_list:
        temp = pd.DataFrame(
            {key: result[key][i]
            for key in ['names', 'pvals_adj']})
        temp['group'] = i
        temp.rename(columns={'names': 'gene'}, inplace=True)
        df_new_degs = pd.concat([df_new_degs, temp])
    
    df_new_degs = df_new_degs[df_new_degs['pvals_adj'] < p_val]
    
    return df_new_degs

def calculate_deg(ground_truth, correct_result):
    TP,TN,FP,FN = 0,0,0,0 
    TP = len(ground_truth.intersection(correct_result))
    FP = len(correct_result - ground_truth.intersection(correct_result))
    FN = len(ground_truth - ground_truth.intersection(correct_result))
    TN = 0 #NOT USED

    precision = TP/(TP+FP) if (TP+FP) != 0 else 0
    recall = TP/(TP+FN) if (TP+FN) != 0 else 0
    if precision == 0 or recall ==0:
        Fscore = 0
    else:
        Fscore = 2*precision*recall/(precision+recall)
    return precision, recall, Fscore

def calculate_degs_score(df_new_degs, true_deg, group_list):
    
    precision, Recall, Fscore = 0,0,0
    for i in group_list:
        df_now = df_new_degs[df_new_degs['group'] == i]
        df_pre = true_deg[true_deg['group'] == i]
        temp1,temp2,temp3 = calculate_deg(set(df_pre['gene']), set(df_now['gene']))
        precision += temp1
        Recall += temp2
        Fscore += temp3
        
    return precision/len(group_list), Recall/len(group_list), Fscore/len(group_list)
    
def calculate_degs_score_real(adata, adata_raw, batch_key='batch', celltype_key='celltype'):
    precision = 0
    recall = 0
    F1 = 0
    for i, b in enumerate(set(adata.obs[batch_key])):
        adata_b = adata[adata.obs[batch_key] == b]
        adata_raw_b = adata_raw[adata_raw.obs[batch_key] == b]
        group_list = adata_b.obs[celltype_key].value_counts()
        group_list = sorted(list(group_list[group_list >= 20].index))
        df_raw_degs = access_degs(adata_raw_b, group_list, celltype_key=celltype_key)
        df_new_degs = access_degs(adata_b, group_list, celltype_key=celltype_key)
        DEG_precision, DEG_recall, DEG_F1 = calculate_degs_score(df_new_degs, df_raw_degs, group_list)
        precision += DEG_precision
        recall += DEG_recall
        F1 += DEG_F1
    precision /= (i+1)
    recall /= (i+1)
    F1 /= (i+1)
    return precision, recall, F1
    
def calculate_degs_score_sim(adata, DEGpath, batch_key='batch', celltype_key='celltype'):
    precision = 0
    recall = 0
    F1 = 0
    genelist = pd.read_table(DEGpath)
    for i, b in enumerate(set(adata.obs[batch_key])):
        adata_b = adata[adata.obs[batch_key] == b]
        group_list = adata_b.obs[celltype_key].value_counts()
        group_list = sorted(list(group_list[group_list >= 20].index))
        df_true_degs = generate_true_tags(genelist, group_list)
        df_new_degs = access_degs(adata_b, group_list, celltype_key=celltype_key)
        DEG_precision, DEG_recall, DEG_F1 = calculate_degs_score(df_new_degs, df_true_degs, group_list)   
        precision += DEG_precision
        recall += DEG_recall
        F1 += DEG_F1
    precision /= (i+1)
    recall /= (i+1)
    F1 /= (i+1)
    return precision, recall, F1
    
def calculate_metrics(adata, adata_raw, batch_key='batch', celltype_key='celltype', percent_extract=0.8, 
                      verbose=True, savepath=None, n_neighbors=5, subsample=None, seed=0, CC_npcs=20,
                      org='human', is_raw=False, use_raw=False, is_embed=False, embed='X_pca',
                      CC_rate=1., exp_dir='cell', tp_thr=3., is_simul=False, dataname=None,
                      DEGpath=None):
#     if not is_embed:
#         use_raw = True
    if subsample:
        random.seed(seed)
        np.random.seed(seed)
        sample_idx = np.random.choice(adata.shape[0], int(subsample*adata.shape[0]), replace=False)
        adata = adata[sample_idx].copy()
        if not is_raw:
            adata_raw = adata_raw[sample_idx].copy()
        del sample_idx
        print('Data size:', adata.shape)
    
    if celltype_key is None:
        if verbose:
            print('--------------------Start calculating graph connectivity--------------------')
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_pca')
        graph_conn = scib.metrics.graph_connectivity(adata, 'celltype')
        if verbose:
            print('graph_conn: %.4f' % graph_conn)
        if verbose:
            print('--------------------Start calculating ASW--------------------')
        _, bASW, _ = calculate_ASW(adata, batch_key=batch_key, celltype_key=celltype_key, verbose=verbose)
        if verbose:
            print('--------------------Start calculating LISI--------------------')
        _, bLISI, _ = calculate_LISI(adata, batch_key=batch_key, celltype_key=celltype_key, verbose=verbose)
        if verbose:
            print('--------------------Start calculating kBET--------------------')
        kBET_acc = calculate_kBET(adata, batch_key=batch_key, celltype_key=celltype_key, verbose=verbose)
        if verbose:
            print('--------------------Start calculating CC score--------------------')
        if is_simul:
            CC_sim = np.nan
        elif is_raw:
            CC_score = 1
        else:
            CC_score = scib.me.cell_cycle(adata_raw, adata, batch_key=batch_key, organism=org, n_comps=CC_npcs, embed=embed)
        if verbose:
            print('CC score %.4f' % (CC_score))
        if verbose:
            print('--------------------Start calculating knn similarity--------------------')
        if is_raw:
            knn_sim = 1
        else:
            knn_sim = calculate_knn_sim(adata_raw, adata, batch_key=batch_key, use_raw=use_raw, embed='X_pca')
        if verbose:
            print('knn similarity %.4f' % (knn_sim))
        if verbose:
            print('--------------------Start calculating cell-cell similarity--------------------')
        if is_raw:
            CC_sim = 1
        else:
            CC_sim = calculate_CC_sim(adata_raw, adata, batch_key=batch_key, 
                                      rate=CC_rate, use_raw=use_raw, embed='X_pca')
        if verbose:
            print('Similarity of raw and corrected cell-cell similarity %.4f' % (CC_sim))
        if verbose:
            print('--------------------Start calculating expression similarity--------------------')
        if is_raw:
            exp_sim = 1
        elif is_embed:
            exp_sim = np.nan
        else:
            adata_raw.X = adata_raw.raw.X.copy()
            adata_raw = adata_raw[:, adata.var_names].copy()
            adata.X = adata.raw.X.copy()
            exp_sim = calculate_exp_sim(adata_raw, adata, batch_key=batch_key, 
                                        direction=exp_dir, use_raw=True)
        if verbose:
            print('exp similarity %.4f' % (exp_sim))
        if verbose:
            print('--------------------Start calculating HVG score--------------------')
        if is_raw:
            HVG_score = 1
        elif is_embed:
            HVG_score = np.nan
        else:
            HVG_score = scib.me.hvg_overlap(adata_raw, adata, batch=batch_key)
        if verbose:
            print('HVG score %.4f' % (HVG_score))
        if verbose:
            print('--------------------Combining results--------------------')
        df = pd.DataFrame({'1-bASW': [bASW],
                            'bLISI': [bLISI],
                            'kBET Accept Rate': [kBET_acc],
                            'graph connectivity': [graph_conn],
                            'CC score': [CC_score],
                            'HVG score': [HVG_score],
                            'knn sim': [knn_sim],
                            'cell-cell sim preservation': [CC_sim],
                            'exp sim': [exp_sim]})
    else:
        # Detect batch-specific cell types
        labels = set(adata.obs[celltype_key])
        labels_ = labels.copy()
        total_cells = adata.shape[0]
        for label in labels_:
            adata_sub = adata[adata.obs[celltype_key] == label]
            if len(set(adata_sub.obs[batch_key])) == 1 or adata_sub.shape[0] < 10:
                print('Cell cluster %s contains only one batch or has less than 10 cells. Skip.' % label)
                total_cells -= adata_sub.shape[0]
                labels.remove(label)

        if verbose:
            print('--------------------Start calculating ASW--------------------')
        cASW, bASW, ASW_F1 = calculate_ASW(adata, batch_key=batch_key, celltype_key=celltype_key, 
                                           labels=labels, total_cells=total_cells, verbose=verbose)
        if verbose:
            print('--------------------Start calculating LISI--------------------')
        cLISI, bLISI, LISI_F1 = calculate_LISI(adata, batch_key=batch_key, celltype_key=celltype_key, 
                                               labels=labels, total_cells=total_cells, verbose=verbose)
        if verbose:
            print('--------------------Start calculating kBET--------------------')
        kBET_acc = calculate_kBET(adata, batch_key=batch_key, celltype_key=celltype_key, 
                                  labels=labels, total_cells=total_cells, verbose=verbose)
        if verbose:
            print('--------------------Start calculating graph connectivity--------------------')
        adata.obs[celltype_key] = adata.obs[celltype_key].astype('category')
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_pca')
        graph_conn = scib.metrics.graph_connectivity(adata, celltype_key)
        if verbose:
            print('graph_conn: %.4f' % graph_conn)
        if verbose:
            print('--------------------Start calculating NMI--------------------')
        scib.cl.opt_louvain(
            adata,
            label_key=celltype_key,
            cluster_key='cluster',
            plot=False,
            inplace=True,
            force=True,
            verbose=False
        )
        NMI = scib.me.nmi(adata, group1='cluster', group2=celltype_key)
        if verbose:
            print('NMI %.4f' % (NMI))
        if verbose:
            print('--------------------Start calculating ARI--------------------')
        ARI = scib.me.ari(adata, group1='cluster', group2=celltype_key)
        if verbose:
            print('ARI %.4f' % (ARI))
        if verbose:
            print('--------------------Start calculating positive and true positive rate--------------------')
        pos_rate, truepos_rate = positive_true_positive(adata, batch_key=batch_key, celltype_key=celltype_key, 
                                                        k1=20, k2=100, tp_thr=tp_thr, embed='X_pca')
        if verbose:
            print('pos rate: %.4f, true pos rate: %.4f' % (pos_rate, truepos_rate))
        if verbose:
            print('--------------------Start calculating CC score--------------------')
        if is_simul:
            CC_score = np.nan
        elif is_raw:
            CC_score = 1
        else:
            CC_score = scib.me.cell_cycle(adata_raw, adata, batch_key=batch_key, organism=org, n_comps=CC_npcs, embed=embed)
        if verbose:
            print('CC score %.4f' % (CC_score))
        if verbose:
            print('--------------------Start calculating knn similarity--------------------')
        if is_raw:
            knn_sim = 1
        else:
            knn_sim = calculate_knn_sim(adata_raw, adata, batch_key=batch_key, use_raw=use_raw, embed='X_pca')
        if verbose:
            print('knn similarity %.4f' % (knn_sim))
        if verbose:
            print('--------------------Start calculating cell-cell similarity--------------------')
        if is_raw:
            CC_sim = 1
        else:
            CC_sim = calculate_CC_sim(adata_raw, adata, batch_key=batch_key, 
                                      rate=CC_rate, use_raw=use_raw, embed='X_pca')
        if verbose:
            print('Similarity of raw and corrected cell-cell similarity %.4f' % (CC_sim))
        if verbose:
            print('--------------------Start calculating expression similarity--------------------')
        if is_raw:
            exp_sim = 1
        elif is_embed:
            exp_sim = np.nan
        else:
            adata_raw.X = adata_raw.raw.X.copy()
            adata_raw = adata_raw[:, adata.var_names].copy()
            adata.X = adata.raw.X.copy()
            exp_sim = calculate_exp_sim(adata_raw, adata, batch_key=batch_key, 
                                        direction=exp_dir, use_raw=True)
        if verbose:
            print('exp similarity %.4f' % (exp_sim))
        if verbose:
            print('--------------------Start calculating HVG score--------------------')
        if is_raw:
            HVG_score = 1
        elif is_embed:
            HVG_score = np.nan
        else:
            HVG_score = scib.me.hvg_overlap(adata_raw, adata, batch=batch_key)
        if verbose:
            print('HVG score %.4f' % (HVG_score))
        if not is_embed:
            if verbose:
                print('--------------------Start calculating metrics for DEG tests--------------------')
            if is_raw:
                DEG_precision = DEG_recall = DEG_F1 = 1.
            elif is_simul:
                DEG_precision, DEG_recall, DEG_F1 = calculate_degs_score_sim(adata, DEGpath, 
                                                                             batch_key, celltype_key)
            else:
                DEG_precision, DEG_recall, DEG_F1 = calculate_degs_score_real(adata, adata_raw, 
                                                                             batch_key, celltype_key)
            if verbose:
                print('DEG precision: %.4f, recall: %.4f, and F1: %.4f' % (DEG_precision, DEG_recall, DEG_F1))
        else:
            DEG_precision = DEG_recall = DEG_F1 = np.nan
        if verbose:
            print('--------------------Combining results--------------------')
        df = pd.DataFrame({'1-bASW': [bASW],
                            'cASW': [cASW],
                            'F1 ASW': [ASW_F1],
                            'bLISI': [bLISI],
                            '1-cLISI': [cLISI],
                            'F1 LISI': [LISI_F1],
                            'kBET Accept Rate': [kBET_acc],
                            'graph connectivity': [graph_conn],
                            'ARI': [ARI],
                            'NMI': [NMI],
                            'CC score': [CC_score],
                            'HVG score': [HVG_score],
                            'knn sim': [knn_sim],
                            'cell-cell sim preservation': [CC_sim],
                            'exp sim': [exp_sim],
                            'pos rate': [pos_rate],
                            'true pos rate': [truepos_rate],
                            'DEG precision': [DEG_precision],
                            'DEG recall': [DEG_recall],
                            'DEG F1': [DEG_F1]})
    if savepath:
        df.to_csv(savepath)
    if verbose:
        print('--------------------Finished--------------------')
    return df
