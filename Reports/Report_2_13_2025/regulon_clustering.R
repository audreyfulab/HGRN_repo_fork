
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


nmi = NMI(cao_groups[, c(3,1)], cbind.data.frame(gene = DMcells$gene[ix], group = result$kmeans$cluster))

print(nmi)



result2 = kmeans(rgc, 5, nstart = 1000)

nmi2 = NMI(cao_groups[, c(3,1)], cbind.data.frame(gene = DMcells$gene[ix], group = result$kmeans$cluster))

print(nmi2)
















