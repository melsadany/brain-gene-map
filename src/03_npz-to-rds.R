################################################################################
################################################################################
rm(list = ls()); gc()
device <- ifelse(grepl("/LSS/", system("cd &pwd", intern = T)), "IDAS", "argon")
source(paste0(ifelse(device == "IDAS", "~/LSS", "/Dedicated"),"/jmichaelson-wdata/msmuhammad/workbench/customized-functions/correct_path.R"))
source(correct_path("/Dedicated/jmichaelson-wdata/msmuhammad/msmuhammad-source.R"))
library(reticulate)
################################################################################
################################################################################
project.dir <- correct_path("/Dedicated/jmichaelson-wdata/msmuhammad/projects/brain-gene-map")
setwd(project.dir)
################################################################################
################################################################################
np <- import("numpy")

# Load the .npz file
data <- np$load("src/02_model/predictions/2.0mm/predictions_2.0mm.npz", allow_pickle = TRUE)

# Extract arrays
predictions <- data$f[["predictions"]]  # (N_voxels, 5000) matrix
coords <- data$f[["coords"]]            # (N_voxels, 3) matrix
gene_names <- data$f[["gene_names"]]    # (5000,) character vector
shape <- data$f[["shape"]]              # (nx, ny, nz) dimensions

# Convert gene_names from numpy array to R character vector
gene_names <- as.character(gene_names)

# Check dimensions
dim(predictions)  # Should be ~900000 x 5000
dim(coords)       # Should be ~900000 x 3
length(gene_names)  # Should be 5000

# Get specific gene
gene_idx <- which(gene_names == "SLC17A7")
slc17a7_expr <- predictions[, gene_idx]

# Create data frame for easier manipulation
brain_data <- data.frame(x = coords[, 1], y = coords[, 2], z = coords[, 3])
# Add gene expression columns (for selected genes)
selected_genes <- gene_names
for (gene in selected_genes) {
  gene_idx <- which(gene_names == gene)
  if (length(gene_idx) > 0) {
    brain_data[[gene]] <- predictions[, gene_idx]
  }
}
brain_data[1:10,1:10]
pdssave(brain_data, file="data/derivatives/predicted-5k-gex-2.0mm.rds")

################################################################################
################################################################################
## save nifti maps
brain_data <- pdsload("data/derivatives/predicted-5k-gex-2.0mm.rds.pxz")
mni.labs2 <- read_rds("/wdata/msmuhammad/refs/labeled-MNI/2mm/voxels-w-labels-R.rds") %>% select(-ends_with("_id"))
source("/wdata/msmuhammad/workbench/customized-functions/make-array-to-nifti.R")
gene.of.int <- colnames(brain_data)[-c(1:3)]

registerDoMC(30)
foreach(g = 1:length(gene.of.int)) %dopar% {
  gene <- gene.of.int[g]
  df <- inner_join(mni.labs2[,1:6], brain_data %>% select(mni_x=x,mni_y=y,mni_z=z,gene) %>% rename(intensity = 4)) %>% 
    select(-starts_with("mni")) %>% mutate(intensity_mm = ((intensity-min(intensity))/(max(intensity)-min(intensity)))* 2 - 1)
  summary(df)
  make_nifti_from_df_V2(df, value_col = "intensity_mm",
                        out_path = paste0("data/derivatives/gene-nifti-maps/",gene,".nii.gz"))
}

