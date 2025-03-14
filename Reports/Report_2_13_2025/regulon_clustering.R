
library(ggplot2)
library(data.table)
library(corrplot)
library(pheatmap)
library(RColorBrewer)
library(NMI)

# read in the regulon activity data
regulon_activity = as.data.frame(fread(file = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Applications/Regulon_DMEM_organoid.csv'))
colnames(regulon_activity) = c('gene', colnames(regulon_activity)[-1])

# read in the raw crop seq
sc_seq = as.data.frame(fread(file = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Applications/Crop Liver/expression_crop_liver.csv'))

# read in the groups that Jun Cao provided
cao_groups = read.csv('C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Applications/Regulon_DM_groups.csv')


# extract just the DM cells
DMcells = regulon_activity[, c(1, which(grepl('DM', colnames(regulon_activity), fixed = T)))]
# pull out the tf names and stip the "(+)" symbol from the end of each name 
tfs = unlist(lapply(regulon_activity[, 1], function(x) unlist(strsplit(x, split = '(+)', fixed = T)[1])))
DMcells$gene = tfs
# get the sorting indices so that the regulon activity data rows are sorted according to Jun Cao's data
ix = na.omit(match(cao_groups$TF_gene, tfs))


DMcells_T = t(DMcells[ix, -1])
colnames(DMcells_T) = DMcells$gene[ix]
# get the spearman correlation matrix
rgc = cor(DMcells_T, method = 'spearman')

#plot the heatmap of correlations using pheatmap as they did in the organoid paper
pheatmap(rgc, color = colorRampPalette(brewer.pal(n = 10, name ="PRGn"))(100), cluster_cols = F, cluster_rows = F)

result = pheatmap(rgc, color = colorRampPalette(brewer.pal(n = 10, name ="PRGn"))(100), 
                  cluster_cols = F, 
                  cluster_rows = T, 
                  kmeans_k = 5)


Cluster = result$kmeans$cluster
# Create annotation data frames
row_annotation = data.frame(Regulon = paste(paste('TF Group', Cluster), ' '))
rownames(row_annotation) = rownames(cm)  # This is the critical fix


#clust.cols = colorRampPalette(brewer.pal(n = 10, name =""))
clust.cols = pal_jco()(length(unique(Cluster)))
names(clust.cols) = paste(paste('TF Group', 1:length(unique(Cluster))), ' ')
# Define colors for clusters
annotation_colors = list(Regulon = clust.cols)

# Create heatmap with annotations
pheatmap(
  rgc,
  cluster_rows = FALSE, 
  cluster_cols = FALSE,
  color = colorRampPalette(brewer.pal(n = 10, name ="PRGn"))(100),  # Fixed color palette syntax
  annotation_row = row_annotation,
  annotation_col = row_annotation,
  annotation_colors = annotation_colors,
  annotation_names_col = FALSE,
  annotation_names_row = FALSE,
  show_rownames = FALSE,  # Hide row labels
  show_colnames = FALSE,  # Hide column labels
  fontsize = 30,
  annotation_legend = T,
  border_color = NA,
  annotation_legend_side = "left",
  width = 16, 
  height = 14,
  angle_col = 90,
  fontsize_col = 18,
  fontsize_row = 12,
  main = 'Regulon Activity Correlation Matrix'
)


nmi = NMI(cao_groups[, c(3,1)], cbind.data.frame(gene = DMcells$gene[ix], group = result$kmeans$cluster))

print(nmi)



result2 = kmeans(rgc, 5, nstart = 1000)

nmi2 = NMI(cao_groups[, c(3,1)], cbind.data.frame(gene = DMcells$gene[ix], group = result$kmeans$cluster))

print(nmi2)
















