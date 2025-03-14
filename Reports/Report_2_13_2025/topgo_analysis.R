# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# 
# BiocManager::install("topGO")
# 

library(gprofiler2)
library(ggplot2)
library(tm)
library(wordcloud)

datapath = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Reports/Report_2_13_2025/Output/applications/REGULON_DM/'
savepath = 'C:/Users/Bruin/OneDrive/Documents/GitHub/Doctoral_Thesis/Figures/hcd/topgo/'

gl = read.csv(paste0(datapath, 'gene_data.csv'))

#stdata.corr = read.csv(paste0(path_raa2, 'data_top_sorted.csv'))

topc = unique(gl$Top.Assignment)
midc = unique(gl$Middle.Assignment)

groups_top = list()
go_top = list()
go_top_plots = list()
go_top_tables = list()
go_top_highlighted = list()

groups_middle = list()
go_middle = list()
go_middle_plots = list()
go_middle_tables = list()
go_middle_highlighted = list()


for(i in 1:length(unique(gl$Top.Assignment))){
  
  groups_top[[i]] = gl$TF_gene[which(gl$Top.Assignment == topc[i])]
  go_top[[i]] = gost(query = groups_top[[i]], 
                     organism = "hsapiens", 
                     exclude_iea = FALSE, 
                     correction_method = "bonferroni", 
                     highlight = TRUE)
  
  label = paste0('Regulon group ', i, ' Enrichments: ', sum(go_top[[i]]$result$significant))
  go_top[[i]]$result$query = rep(label, nrow(go_top[[i]]$result))
  
  go_top_highlighted[[i]] = go_top[[i]]$result$term_name[go_top[[i]]$result$highlighted]

  p = gostplot(go_top[[i]], capped = TRUE, interactive = F)
  A = p+theme(
    legend.position = 'top', legend.text = element_text(size = 16),axis.text = element_text(size = 15),
    legend.title = element_text(size = 16),
    axis.title.y = element_text(margin = margin(t = 0, r = 15, b = 0, l = 0), size = 16),
    axis.title.x = element_text(margin = margin(t = 0, r = 35, b = 0, l = 0), size = 16),
    strip.text.x = element_text(size = 16),
    strip.text.y = element_text(size = 16),
    plot.title = element_text(size = 16, hjust = 0.5))
  
  
  go_top_plots[[i]] = A
  
  pdf(paste0(savepath,'top groups/regulon_group_', i, '.pdf'), height = 10, width = 10)
  plot(A)
  dev.off()
  
  
  final_label = paste0('Regulon group ', i)
  go_top[[i]]$result$query = rep(final_label, nrow(go_top[[i]]$result))
  
  go_top_tables[[i]] = go_top[[i]]$result[,-14]
  
}



for(j in 1:length(unique(gl$Middle.Assignment))){
  
  groups_middle[[j]] = gl$TF_gene[which(gl$Middle.Assignment == midc[j])]
  go_middle[[j]] = gost(query = groups_middle[[j]], organism = "hsapiens", exclude_iea = FALSE, correction_method = "bonferroni", highlight = TRUE)
  
  label = paste0('Regulon sub-group ', j, '', 'Genes ',  ' Enrichments: ', sum(go_middle[[j]]$result$significant))
  go_middle[[j]]$result$query = rep(label, nrow(go_middle[[j]]$result))
  
  go_middle_highlighted[[j]] = go_middle[[j]]$result$term_name[go_middle[[j]]$result$highlighted]
  
  p = gostplot(go_middle[[j]], capped = TRUE, interactive = F)
  
  B = p+theme(
    legend.position = 'top', legend.text = element_text(size = 16),axis.text = element_text(size = 15),
    legend.title = element_text(size = 16),
    axis.title.y = element_text(margin = margin(t = 0, r = 15, b = 0, l = 0), size = 16),
    axis.title.x = element_text(margin = margin(t = 0, r = 35, b = 0, l = 0), size = 16),
    strip.text.x = element_text(size = 16),
    strip.text.y = element_text(size = 16),
    plot.title = element_text(size = 16, hjust = 0.5))
  
  
  go_middle_plots[[j]] = B
  
  
  pdf(paste0(savepath,'mid groups/regulon_group_', j, '.pdf'), height = 10, width = 10)
  plot(p)
  dev.off()
  
  final_label_mid = paste0('Regulon sub-group ', j)
  go_middle[[j]]$result$query = rep(final_label_mid, nrow(go_middle[[j]]$result))
  
  go_middle_tables[[j]] = go_middle[[j]]$result[,-14]
    
}


final_top_table = do.call('rbind', go_top_tables)
final_mid_table = do.call('rbind', go_middle_tables)

write.csv(final_top_table, 
          file = paste0(savepath, 'top groups/go_table_results_top.csv'))


write.csv(final_mid_table, 
          file = paste0(savepath, 'mid groups/go_table_results_middle.csv'))














































