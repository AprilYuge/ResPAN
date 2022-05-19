# Information of data used in ResPAN

This folder contains detailed information of sources for the 11 real datasets and simulated data used in our manuscript.

### Simulation study

For the simulation study, we generated synthetic scRNA-seq data using Splatter [1], which is based on negative binomial distributions through a hierarchical Gamma-Poisson model. Parameters were the same as those used in Kotliar et al. [2], which were estimated by Splatter from 8,000 cells of the organoid dataset in Quadrato et al. [3]. We simulated two batches and seven cell types. The number of cells in each batch was 2,000, and the number of genes was 10,000. Ten baseline datasets with balanced settings using different seeds were first simulated. Then, we considered three general scenarios, which were unbalanced batch size (scenario 1), rare cell types (scenario 2), and batch-specific cell types (scenario 3). For each scenario, we further considered three sub-levels. For scenario 1, cells in batch 1 were downsampled to 50% (batch1-0.5), 25% (batch1-0.25), and 12.5% (batch1-0.125); for scenario 2, cells labeled as Group1 in each batch were downsampled to 50% (rare1-0.5), 20% (rare1-0.2), and 10% (rare1-0.1); for scenario 3, the number of common cell types were reduced to five (common-5), three (common-3), and one (common-1) separately. Therefore, nine other datasets were generated based on each baseline dataset, and there were in total 10 sub-scenarios, and each of them contained 10 random repeats.

Code for generating all simulated data can be found in `simulate.R` and `simulate_post_process.ipynb`.

### Real data

We collected 10 real datasets for benchmarking different methods, including eight small-to-moderate-scale datasets and two large-scale datasets. The eight small-to-moderate scale datasets are pure cell lines (CL), human pancreas, human peripheral blood mononuclear cells (Human PBMC), PBMC 3&68K, mouse hematopoietic stem and progenitor cells (MHSP), Mouse Cell Atlas (MCA),  Mouse Retina, and a multi-batch human lung dataset. The two large-scale datasets contain over half a million cells, and they are a Mouse Brain dataset and a dataset from Human Cell Atlas (HCA). In addition, we used a dataset from patients with colorectal cancer (CRC) to demonstrate the ability of ResPAN on downstream analysis. 

The sources of these datasets can be found in `STable3.xlsx` and the cell type composition of each dataset can be found in `STable4.xlsx`.


### References
* [1] Zappia, Luke, Belinda Phipson, and Alicia Oshlack. "Splatter: simulation of single-cell RNA sequencing data." Genome biology 18.1 (2017): 1-15.
* [2] Kotliar, Dylan, et al. "Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq." Elife 8 (2019): e43803.
* [3] Quadrato, Giorgia, et al. "Cell diversity and network dynamics in photosensitive human brain organoids." Nature 545.7652 (2017): 48-53.
