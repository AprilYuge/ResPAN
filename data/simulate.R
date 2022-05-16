library(splatter)

base_params <- newSplatParams(
#                         nGenes = 2000,
                        mean.shape = 0.34,
                        mean.rate = 7.68,
                        lib.loc = 7.64,
                        lib.scale = 0.78,
                        out.prob = 0.00286,
                        out.facLoc = 6.15,
                        out.facScale = 0.49,
                        bcv.common = 0.448,
                        bcv.df = 22.087)

for (seed in 1:10){
    
    sim_params <- setParams(
        base_params,
#         lib.loc        = 12,
        batchCells     = c(2000, 2000),
#         batchCells     = c(3000),
#         batch.facLoc   = c(0.10, 0.15),
        batch.facLoc   = c(0.50, 0.5),
#         batch.facScale = c(0.10, 0.08),
        batch.facScale = c(0.50, 1.),
        # Groups with equal probabilities
        group.prob     = rep(1, 7) / 7,
        # Differential expression by group
        de.prob        = c(0.10, 0.12, 0.08, 0.20, 0.12, 0.10, 0.16),
#         de.prob      = rep(0.1, 7),
#         de.facLoc      = c(0.10, 0.08, 0.12, 0.18, 0.06, 0.20, 0.14),
        de.facLoc      = rep(0.5, 7),
#         de.facScale    = c(0.40, 0.30, 0.45, 0.48, 0.42, 0.38, 0.36),
        de.facScale    = rep(1.0, 7),
        # Seed
        seed           = seed,
        dropout.type   = 'experiment'
    )

    # Simulate the full dataset that we will downsample
    sim <- splatSimulateGroups(sim_params)
    
    counts <- as.data.frame(t(as.array(counts(sim))))
    # truecounts <- as.data.frame(t(assays(sim)$TrueCounts))
    cellinfo <- as.data.frame(colData(sim))
    geneinfo <- as.data.frame(rowData(sim))
    
    base_name <- paste0('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim_new/Sim', seed)
    write.table(counts, file = paste0(base_name,"/counts.txt"), sep = "\t", row.names = TRUE, col.names = TRUE)
    # write.table(truecounts, file = paste0(base_name,"/truecounts.txt"), sep = "\t", row.names = TRUE, col.names = TRUE)
    write.table(geneinfo, file = paste0(base_name,"/geneinfo.txt"), sep = "\t", row.names = TRUE, col.names = TRUE)
    write.table(cellinfo, file = paste0(base_name,"/cellinfo.txt"), sep = "\t", row.names = TRUE, col.names = TRUE)
}