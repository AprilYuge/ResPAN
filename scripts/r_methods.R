# necessary pacakges
library(SeuratDisk)
library(Seurat)
library(rliger)

# Parameters
npcs = 20
args = commandArgs(trailingOnly=TRUE)
dataname = args[1] # ['CL', 'DC', 'Panc', 'PBMC368k', 'HumanPBMC', 'MHSP', 'MCA', 'Lung', 'MouseRetina', 'HCA', 'MouseBrain']
method = args[2] # ['liger', 'seurat']
folder = args[3]

chunk <- function(x,n){
    vect <- c(1:x)
    num <- ceiling(x/n)
    split(vect,rep(1:num,each=n,len=x))
}

# Load data
# if (dataname == 'HCA'){
#     filepath <- paste0('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/', dataname, '/', dataname, '_raw.h5seurat')
#     data = LoadH5Seurat(filepath)
#     data@meta.data$batch <- as.character(data@meta.data$batch)
#     if ('celltype' %in% colnames(data@meta.data)){
#         data@meta.data$cell_type <- as.character(data@meta.data$cell_type)
#     }
# } else{
#     filepath <- paste0('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/', dataname, '/', dataname, '_raw.loom')
#     data <- LoadLoom(filepath, mode='r')
# }

if (folder == 'baseline'){
    filepath <- paste0('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/', dataname, '/', dataname, '_raw.loom')
    data <- LoadLoom(filepath, mode='r')
} else{
    filepath <- paste0('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/', dataname, '/', folder, '/', dataname, '_raw.loom')
    data <- LoadLoom(filepath, mode='r')
}

if (dataname == 'HCA' | dataname == 'MouseBrain'){
    options(future.globals.maxSize= 1e10)
}

if (method == 'liger'){
    batches <- unique(data@meta.data$batch)
    liger_list <- list()
    liger_names <- list()
    i <- 1
    for (b in batches){
        liger_list <- append(liger_list, subset(x = data, subset = batch == b)[["RNA"]]@counts)
        liger_names <- append(liger_names, paste0('l', i))
        i <- i+1
    }
    names(liger_list) <- liger_names
    data_liger <- createLiger(liger_list)
    data_liger <- normalize(data_liger)
    data_liger <- selectGenes(data_liger)
    if (dataname == 'HCA' | dataname == 'MouseBrain'){
        data_liger@scale.data <- lapply(1:length(data_liger@norm.data),
                                     function(i) {rliger:::scaleNotCenterFast(t(data_liger@norm.data[[i]][data_liger@var.genes,]))})
        data_liger@scale.data <- lapply(data_liger@scale.data,function(l){
            l2 <- lapply(chunk(nrow(l),2000), function(i){as.matrix(l[i,])})
            res <- do.call(rbind,l2)
            return(res)
        })
        names(data_liger@scale.data) <- names(data_liger@norm.data)
        for (i in 1:length(data_liger@scale.data)) {
          data_liger@scale.data[[i]][is.na(data_liger@scale.data[[i]])] <- 0
          rownames(data_liger@scale.data[[i]]) <- colnames(data_liger@raw.data[[i]])
          colnames(data_liger@scale.data[[i]]) <- data_liger@var.genes
        }
        data_liger <-rliger:::removeMissingObs(data_liger, slot.use = "scale.data", use.cols = F)
        gc()
    } else {data_liger <- scaleNotCenter(data_liger)}
    start <- Sys.time()
    data_liger <- optimizeALS(data_liger, k=npcs, rand.seed=1)
    data_liger <- quantile_norm(data_liger, rand.seed=1)
    result <- as.data.frame(t(data_liger@H.norm))
    data_correct <- CreateSeuratObject(as.data.frame(t(data_liger@H.norm)), meta.data=data@meta.data)
} else if (method == 'seurat'){
    min_batch <- min(table(data@meta.data$batch))
    data.list <- SplitObject(data, split.by = "batch")
    rm(data)
    gc()
    data.list <- lapply(X = data.list, FUN = function(x) {
      x <- NormalizeData(x)
      x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
    })
    start <- Sys.time()
    features <- SelectIntegrationFeatures(object.list = data.list)
    data.anchors <- FindIntegrationAnchors(object.list = data.list, anchor.features=features)
    rm(features)
    gc()
    if (min_batch < 100){
        data_correct <- IntegrateData(anchorset = data.anchors, k.weight=min_batch)
    }else{
        data_correct <- IntegrateData(anchorset = data.anchors)
    }
    rm(data.list)
    gc()
    DefaultAssay(data_correct) <- "integrated"
    data_correct[['RNA']] <- NULL
}

end <- Sys.time()
time_elapse <- as.numeric(difftime(end, start, units = "secs"))
print(paste0('Running time on ', dataname, ' for ', method, ' is ', time_elapse, ' sec.'))

# Save results
# savepath <- paste0('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/', dataname, '/', dataname, '_', method, '.loom')
if (folder == 'baseline'){
    savepath <- paste0('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/', dataname, '/', dataname, '_', method, '.loom')
} else{
    savepath <- paste0('/gpfs/gibbs/pi/zhao/yw599/AWGAN/datasets/Sim/', dataname, '/', folder, '/', dataname, '_', method, '.loom')
}
cl.loom <- as.loom(data_correct, filename = savepath, verbose = FALSE)
cl.loom$close_all()
