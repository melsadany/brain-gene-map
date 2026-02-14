################################################################################
################################################################################
rm(list = ls()); gc()
device <- ifelse(grepl("/LSS/", system("cd &pwd", intern = T)), "IDAS", "argon")
source(paste0(ifelse(device == "IDAS", "~/LSS", "/Dedicated"),"/jmichaelson-wdata/msmuhammad/workbench/customized-functions/correct_path.R"))
source(correct_path("/Dedicated/jmichaelson-wdata/msmuhammad/msmuhammad-source.R"))
################################################################################
################################################################################
project.dir <- correct_path("/Dedicated/jmichaelson-wdata/msmuhammad/projects/brain-gene-map")
setwd(project.dir)
################################################################################
################################################################################
# build metadata
donors.meta <- tribble(~donor, ~nii_donor, ~sex, ~age, ~ethnicity, ~PMI,
                       9861, "H0351.2001","M",24,"BAA",23,
                       10021, "H0351.2002","M",39,"BAA",10,
                       12876, "H0351.1009","M",57,"WC",26,
                       14380, "H0351.1012","M",31,"WC",17,
                       15496, "H0351.1015","F",49,"H",30,
                       15697, "H0351.1016","M",55,"WC",18)
write_tsv(donors.meta, "data/raw/donors-metadata.tsv")
################################################################################
# read gene expression data
data.dir <- correct_path("/Dedicated/jmichaelson-wdata/msmuhammad/data/Allen/Imaging/")
donors.gex <- list()
donors.annot <- list()
for (donor in donors.meta$nii_donor) {
  print(donor)
  # annot well-id are unique across samples and donors
  annot <- read_csv(paste0(data.dir, donor, "/SampleAnnot.csv"))
  probes <- read_csv(paste0(data.dir, donor, "/Probes.csv"))
  brain.exp <- read_csv(paste0(data.dir, donor, "/MicroarrayExpression.csv"), col_names = c("probe", annot$well_id))
  tmp <- cbind(gene = probes$gene_symbol[which(probes$probe_id %in% brain.exp$probe)], brain.exp[,-1])
  brain.gex <- tmp %>% group_by(gene) %>% summarise_all(.funs=sum) %>%
    ungroup() %>% column_to_rownames("gene") %>% t()
  donors.gex[[as.character(donor)]] <- brain.gex
  donors.annot[[as.character(donor)]] <- annot %>% mutate(nii_donor = donor)
}
donors.gex <- do.call(rbind, donors.gex) %>% as.data.frame() %>% rownames_to_column("well_id") %>% select(-na)
donors.annot <- do.call(rbind, donors.annot) %>% as.data.frame() %>% mutate(well_id =as.character(well_id))
write_rds(list(annot = donors.annot, gex = donors.gex, meta = donors.meta), 
          "data/raw/gex-and-annotation.rds", compress = "gz")
ll <- read_rds("data/raw/gex-and-annotation.rds")
ll$annot %>% mutate(hemisphere = case_when(grepl("right",structure_name)~"right",
                                           grepl("left",structure_name)~"left",
                                           T~"")) %>%
  group_by(nii_donor, hemisphere) %>% summarise(c=n()) %>% 
  group_by(nii_donor) %>% mutate(tot = sum(c), percentage = (c/tot)*100) %>%
  left_join(ll$meta)
donors.annot = ll$annot; donors.gex = ll$gex; donors.meta = ll$meta
################################################################################
################################################################################
## keep gray matter only
# These are not gray matter
# Corpus callosum → major commissural white-matter tract.
# Cingulum bundle (left/right) → long association white-matter tract.
# Central glial substance → central spinal/brainstem tissue rich in glia and fibers, usually treated as non-cortical gray and often excluded in cortical GM analyses.
# Choroid plexus of the lateral ventricle → vascular/ependymal structure in ventricles, not neuronal gray matter.
# Pineal gland → endocrine gland, not typical neuronal gray matter.

donors.annot.gm <- donors.annot %>%
  filter(!grepl(paste(c("corpus","cingulum","central glial","choroid","pineal"),collapse = "|"), 
                structure_name))
################################################################################
################################################################################
##############
##### QC #####
##############


##############
### sample-based
qc_metrics <- function(donor){
  donor %>% left_join(donors.annot %>% select(starts_with("mni"),well_id)) %>% 
    mutate(gene_cols = select(., -c(well_id)),
           total_expr = rowSums(gene_cols),
           n_expressed = rowSums(gene_cols > -2),
           expr_variance = apply(gene_cols, 1, var))
}
donors.gex.qc <- lapply(unique(donors.annot$nii_donor), function(donor.n){
  qc_metrics(donors.gex%>% filter(well_id %in% donors.annot$well_id[donors.annot$nii_donor==donor.n]))
})
# IQR-based outlier removal
remove_outliers <- function(x, k = 1.5) {
  q1 <- quantile(x, 0.25); q3 <- quantile(x, 0.75); iqr <- q3 - q1
  x >= (q1 - k * iqr) & x <= (q3 + k * iqr)
}
donors.gex.clean <- do.call(rbind, lapply(donors.gex.qc, function(df0){
  df0 %>% as.data.frame() %>%
    filter(remove_outliers(total_expr),n_expressed > 5000, !is.infinite(expr_variance)) %>%
    select(-c(total_expr, n_expressed, expr_variance))
})) %>% left_join(donors.annot %>% select(well_id, donor_id=nii_donor))

##############
### gene-based (stable expression across participants)

# Calculate inter-donor correlation for each gene
gene_cols.0 <- setdiff(colnames(donors.gex.clean), c("donor_id", "mni_x", "mni_y", "mni_z"))

### drop non protein-coding genes
biomart.genes <- read_rds(correct_path("/wdata/msmuhammad/data/genomics/bioMart-protein-coding-genes.rds"))
table(gene_cols.0 %in% biomart.genes$external_gene_name)
gene_cols <- gene_cols.0[(gene_cols.0 %in% biomart.genes$external_gene_name)]

# Compute mean expression per region per donor for each gene
donors.gex.region.means <- donors.gex.clean %>% 
  left_join(donors.annot %>% select(well_id, donor_id=nii_donor, structure_id)) %>%
  group_by(structure_id, donor_id) %>%
  summarize(across(all_of(gene_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop")

inter.donor.cor <- map_dbl(gene_cols, function(gene) {
  # Region × donor matrix for this gene
  mat <- donors.gex.region.means %>%
    select(structure_id, donor_id, !!sym(gene)) %>%
    pivot_wider(names_from = donor_id, values_from = !!sym(gene)) %>%
    select(-structure_id) %>%
    as.matrix()
  # Remove regions where all donors are NA
  keep <- rowSums(!is.na(mat)) > 1
  mat <- mat[keep, , drop = FALSE]
  if (nrow(mat) < 5 || ncol(mat) < 2) return(NA_real_)
  # Correlation matrix across donors
  cm <- stats::cor(mat, use = "pairwise.complete.obs")
  # Mean of upper triangle (all donor–donor correlations)
  mean(cm[upper.tri(cm)], na.rm = TRUE)
})

names(inter.donor.cor) <- gene_cols
summary(inter.donor.cor)
# Filter genes with r > 0.4 (standard threshold) (Markello et al. 2021)
stable.genes <- names(inter.donor.cor)[!is.na(inter.donor.cor) & inter.donor.cor > 0.4]
cat(sprintf("Genes passing inter-donor consistency (r > 0.4): %d / %d (%.1f%%)\n",
            length(stable.genes), length(gene_cols),
            100 * length(stable.genes) / length(gene_cols)))
# 10401 / 15256 (68.2%)
# Filter dataset
donors.gex.stable <- donors.gex.clean %>%
  select(mni_x, mni_y, mni_z, donor_id, well_id, all_of(stable.genes))

##############
## gene-based (remove low-variance genes)
# calculate variance for each gene
gene.vars <- donors.gex.stable %>%
  select(-mni_x, -mni_y, -mni_z, -donor_id) %>%
  summarize(across(everything(), var, na.rm = TRUE)) %>%
  pivot_longer(everything(), names_to = "gene", values_to = "variance")
gene.vars %>% ggplot(aes(x = log10(variance))) +
  geom_histogram(bins = 50) +
  geom_vline(xintercept = log10(0.05), color = "red", linetype = 2) +
  labs(title = "Gene Expression Variance Distribution", x = "log10(Variance)", y = "Count")+
  bw.theme
ggsave2("figs/QC/gene-variance-dist.png")

# Filter genes with variance > 0.01
variable.genes <- gene.vars %>% filter(variance > 0.05) %>% pull(gene)

donors.gex.filtered <- donors.gex.stable %>%
  select(mni_x, mni_y, mni_z, donor_id, all_of(variable.genes))

cat(sprintf("Genes with sufficient variance: %d / %d\n",
            length(variable.genes), nrow(gene.vars)))

################################################################################
################################################################################
#### batch correction

library(sva) # BiocManager package
expr.matrix <- donors.gex.filtered %>% select(-c(starts_with("mni"),donor_id, well_id)) %>%
  as.matrix() %>% t()
gc()
batch <- donors.gex.filtered$donor_id
mod <- model.matrix(~ 1, data = data.frame(sample = 1:ncol(expr.matrix)))

# Run ComBat
expr.combat <- ComBat(dat = expr.matrix, batch = batch,
                      mod = mod, par.prior = TRUE, prior.plots = FALSE)

# Transpose back and recombine with coordinates
donors.gex.corrected <- donors.gex.filtered %>%
  select(mni_x, mni_y, mni_z, donor_id, well_id) %>%
  bind_cols(as_tibble(t(expr.combat)))

################################################################################
################################################################################
################################################################################
## save
list(meta = donors.meta, annot = donors.annot, gex_clean = donors.gex.clean,
     gex_filtered = donors.gex.filtered, gex_corrected = donors.gex.corrected) %>%
  write_rds("data/derivatives/gex-cleaned-and-corrected.rds",compress="gz")

## add a structure column: CX:cortex, BS:brainstem, CB:cerebellum
write_csv(inner_join(donors.gex.corrected,donors.annot %>% select(mni_x,mni_y,mni_z,donor_id=nii_donor,structure=slab_type)) %>% 
            select(-well_id) %>% relocate(structure, .after=donor_id), "data/derivatives/gex-corrected.csv",
          num_threads = 30)
## normalize
df <- read_csv("data/derivatives/gex-corrected.csv")
df_normalized <- df %>% group_by(structure) %>%
  mutate(across(.cols = colnames(donors.gex.corrected)[-c(1:5)],
                .fns = ~ {
                  m <- mean(.x, na.rm = TRUE)
                  s <- sd(.x, na.rm = TRUE)
                  if (s > 0) {
                    (.x - m) / s  # z-score
                  } else {
                      rep(0, length(.x))  # constant gene → map to 0
                  }
                },.names = "{.col}")) %>%
  ungroup()  %>%
  mutate(across(.cols = colnames(donors.gex.corrected)[-c(1:5)],
                .fns = ~ pmax(pmin(.x, 3), -3)  # clip to [-3, 3] to clip extreme values
                ))
df_normalized %>%
  select(structure, all_of(colnames(donors.gex.corrected)[-c(1:5)])) %>%
  group_by(structure) %>%
  summarize(across(everything(), list(mean = mean, sd = sd), .names = "{.col}_{.fn}"))

write_csv(df_normalized, "data/derivatives/gex-corrected-normalized.csv",num_threads = 30)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
