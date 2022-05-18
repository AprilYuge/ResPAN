# Analysis code

### Overview

This folder contains analysis code for reproducing benchmarking results shown in our manuscript. We benchmarked ResPAN with seven other batch correction methods, including five implemented in Python (iMAP, scVI, Harmony, MNN and BBKNN) and two implemented in R (Seurat v4 and Liger).

File `python_methods.py` contains code for running all Python methods and file `r_methods.R` includes code for running the two R methods. File `cal_metrics.py` calls functions in `metrics.py` that contains code for calculating 16 quantitative metrics on corrected scRNA-seq data. These 16 metrics are further separated into metrics that quantify the level of batch mixing (1-bASW, bLISI, graph connectivity, kBET, and true positive rate) and metrics that quantify the conservation of biological information from different aspects (cASW, 1-cLISI, ARI, NMI, positive rate, cell-cycle score, HVG score, kNN similarity, cell-cell similarity score, expression similarity, and DEG F1 score). Details can be found in our manuscript Section 2.3 and Supplementary Note S1.3.

### Package requirement

Python and R packages required for running our benchmarking piepline are listed below (packages listed in the README file of the parent folder are not shown again).

|                 | Package       | Version    |
|-----------------|---------------|------------|
| Python packages | scib          | 1.0.0      |
|                 | imap          | 1.0.0      |
|                 | scvi-tools    | 0.15.0     |
|                 | bbknn         | 1.5.1      |
|                 | mnnpy         | 0.1.9.5    |
|                 | harmonypy     | 0.0.6      |
|                 | scikit-learn  | 1.0.2      |
|                 | rpy2          | 3.4.5      |
| R packages      | kBET          | 0.99.6     |
|                 | lisi          | 1.0.0      |
|                 | rliger        | 1.0.0      |
|                 | Seurat        | 4.1.0      |
|                 | SeuratDisk    | 0.0.0.9019 |

