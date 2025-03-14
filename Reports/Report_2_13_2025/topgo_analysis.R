# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# 
# BiocManager::install("topGO")
# 

library(gprofiler2)
library(ggplot2)
library(ggpubr)
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
  
  label = paste0('Regulon group ', i, ' Genes: ', length(groups_top[[i]]), '\n Enrichments: ', sum(go_top[[i]]$result$significant))
  go_top[[i]]$result$query = rep(label, nrow(go_top[[i]]$result))
  
  go_top_highlighted[[i]] = go_top[[i]]$result$term_name[go_top[[i]]$result$highlighted]

  p = gostplot(go_top[[i]], capped = TRUE, interactive = F)
  A = p+theme(
    legend.position = 'top', legend.text = element_text(size = 15),axis.text = element_text(size = 15),
    legend.title = element_text(size = 15),
    axis.title.y = element_text(margin = margin(t = 0, r = 15, b = 0, l = 0), size = 15),
    axis.title.x = element_text(margin = margin(t = 0, r = 35, b = 0, l = 0), size = 15),
    strip.text.x = element_text(size = 15),
    strip.text.y = element_text(size = 15),
    plot.title = element_text(size = 15))
  
  
  go_top_plots[[i]] = A
  
  pdf(paste0(savepath,'top groups/regulon_group_', i, '.pdf'), height = 10, width = 10)
  plot(A)
  dev.off()
  
  
  final_label = paste0('Regulon group ', i)
  go_top[[i]]$result$query = rep(final_label, nrow(go_top[[i]]$result))
  
  go_top_tables[[i]] = go_top[[i]]$result[,-14]
  
}

combplotA = ggarrange(go_top_plots[[1]]+rremove('ylab'), go_top_plots[[2]]+rremove('ylab'),
                      go_top_plots[[3]]+rremove('ylab'), go_top_plots[[4]]+rremove('ylab'),
                      go_top_plots[[5]]+rremove('ylab'), nrow =3, ncol = 2)


combA_annot = annotate_figure(combplotA,
                              left = text_grob("-log10(p-adjustment)", rot = 90, size = 16))
pdf(paste0(savepath,'top groups/regulon_top_combined.pdf'), height = 14, width = 12)
plot(combA_annot)
dev.off()





for(j in 1:length(unique(gl$Middle.Assignment))){
  
  groups_middle[[j]] = gl$TF_gene[which(gl$Middle.Assignment == midc[j])]
  go_middle[[j]] = gost(query = groups_middle[[j]], organism = "hsapiens", exclude_iea = FALSE, correction_method = "bonferroni", highlight = TRUE)
  
  label = paste0('Regulon sub-group ', j, ' Genes: ', length(groups_middle[[j]]), '\n Enrichments: ', sum(go_middle[[j]]$result$significant))
  go_middle[[j]]$result$query = rep(label, nrow(go_middle[[j]]$result))
  
  go_middle_highlighted[[j]] = go_middle[[j]]$result$term_name[go_middle[[j]]$result$highlighted]
  
  p = gostplot(go_middle[[j]], capped = TRUE, interactive = F)
  
  B = p+theme(
    legend.position = 'top', legend.text = element_text(size = 14),axis.text = element_text(size = 14),
    legend.title = element_text(size = 14),
    axis.title.y = element_text(margin = margin(t = 0, r = 15, b = 0, l = 0), size = 14),
    axis.title.x = element_text(margin = margin(t = 0, r = 35, b = 0, l = 0), size = 14),
    strip.text.x = element_text(size = 14),
    strip.text.y = element_text(size = 14),
    plot.title = element_text(size = 14))
  
  
  go_middle_plots[[j]] = B
  
  
  pdf(paste0(savepath,'mid groups/regulon_group_', j, '.pdf'), height = 10, width = 10)
  plot(B)
  dev.off()
  
  final_label_mid = paste0('Regulon sub-group ', j)
  go_middle[[j]]$result$query = rep(final_label_mid, nrow(go_middle[[j]]$result))
  
  go_middle_tables[[j]] = go_middle[[j]]$result[,-14]
    
}


combplotB = ggarrange(go_middle_plots[[1]]+rremove('ylab'), go_middle_plots[[2]]+rremove('ylab'),
                      go_middle_plots[[3]]+rremove('ylab'), go_middle_plots[[4]]+rremove('ylab'),
                      go_middle_plots[[5]]+rremove('ylab'), go_middle_plots[[6]]+rremove('ylab'),
                      go_middle_plots[[7]]+rremove('ylab'), go_middle_plots[[8]]+rremove('ylab'),
                      go_middle_plots[[9]]+rremove('ylab'), go_middle_plots[[10]]+rremove('ylab'),
                      go_middle_plots[[11]]+rremove('ylab'), go_middle_plots[[12]]+rremove('ylab'), 
                      go_middle_plots[[13]]+rremove('ylab'), go_middle_plots[[14]]+rremove('ylab'),
                      go_middle_plots[[15]]+rremove('ylab'),
                      nrow =5, ncol = 3)


combB_annot = annotate_figure(combplotB,
                              left = text_grob("-log10(p-adjustment)", rot = 90, size = 16))
pdf(paste0(savepath,'mid groups/regulon_middle_combined.pdf'), height = 16, width = 12)
plot(combB_annot)
dev.off()




final_top_table = do.call('rbind', go_top_tables)
final_mid_table = do.call('rbind', go_middle_tables)

write.csv(final_top_table, 
          file = paste0(savepath, 'top groups/go_table_results_top.csv'))


write.csv(final_mid_table, 
          file = paste0(savepath, 'mid groups/go_table_results_middle.csv'))





# word_freqs = summary(as.factor(unlist(go_top_highlighted)))
# df <- data.frame(word = names(word_freqs), freq = word_freqs)
# 
# 
# wordcloud(words = df$word, freq = df$freq, min.freq = 1, 
#           max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Dark2"))











for(i in 1:5){
  idx = sort(go_top[[i]]$result$p_value, decreasing = F, index.return = T)$ix
  
  
  x = paste0('Group ', i, ' ', paste(go_top[[i]]$result[idx,]$term_name[1:5], collapse = ','))
  y = paste0('Group ', i, ' ', paste(subset(go_top[[i]]$result[idx,], source == 'KEGG')$term_name[1:5], collapse = ','))
  print(y)
}
























