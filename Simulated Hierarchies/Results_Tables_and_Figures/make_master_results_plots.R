
library(ggpubr)
tab = read.csv('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/Results_Tables_and_Figures/MASTER_results.csv')
tab$input_graph[tab$input_graph == 'A_ingraph_true'] = 'Input graph: True graph'
tab$input_graph[tab$input_graph == 'A_corr_no_cutoff'] = 'Input graph: Corr. matrix'
tab$input_graph[tab$input_graph == 'A_ingraph02'] = 'Input graph: r > 0.2'
tab$input_graph[tab$input_graph == 'A_ingraph05'] = 'Input graph: r > 0.5'
tab$input_graph[tab$input_graph == 'A_ingraph07'] = 'Input graph: r > 0.7'
tab$Connection_prob[tab$Connection_prob == 'disc'] = 'Disconnected Top Layer'
tab$Connection_prob[tab$Connection_prob == 'full'] = 'Fully Connected Top Layer'
tab$Layers[tab$Layers == 2] = '2 Layers'
tab$Layers[tab$Layers == 3] = '3 Layers'
# ggplot(data = tab, aes(x = Network_type, y = Modularity_top, fill = as.factor(Resolution_top)))+
#   geom_boxplot()
# 
# 
# 
# subtab = subset(tab, Resolution_top == 1)
# ggplot(data = subtab, aes(x = True_mod_top, y = Modularity_top))+geom_point()
# 
# 
# tab$Resolution_top = as.factor(tab$Resolution_top)
# tab$Resolution_middle = as.factor(tab$Resolution_middle)
# ggplot(data = tab, aes(x = Network_type, y = Top_homogeneity, fill = Resolution_top))+
#   geom_boxplot()+
#   facet_wrap(~Gamma)
# 
# ggplot(data = tab, aes(x = Network_type, y = Top_homogeneity, fill = Resolution_middle))+
#   geom_boxplot()+
#   facet_wrap(~Gamma)


#tab$True_mod_top = as.factor(round(tab$True_mod_top, 3))
#network performance vs true modularity:
tabsub = tab
tabsub$True_mod_top = round(tabsub$True_mod_top, 2)
A = ggplot(data = tabsub, aes(y = Top_NMI, x = as.factor(True_mod_top),
                              color = Network_type, linetype = Layers, fill = Network_type))+
  geom_boxplot(color ='black', size = 1)+#geom_point(size = 3)+
  facet_wrap(~Connection_prob, scales = 'free_x')+
  theme_bw()+
  theme(legend.position = 'top')+
  xlab('True Modularity Of Top Level Communities')+
  ylab('NMI Top Layer (HCD)')+theme(axis.title.x = element_blank(), 
                                    axis.text.x = element_text(size = 11),
                                    legend.text = element_text(size = 11))

B = ggplot(data = tabsub, aes(y = Louvain_NMI_top, x = as.factor(True_mod_top),
                              color = Network_type, linetype = Layers, fill = Network_type))+
  geom_boxplot(color ='black', size = 1)+#geom_point(size = 3)+
  facet_wrap(~Connection_prob, scales = 'free_x')+
  theme_bw()+
  theme(legend.position = 'top')+
  xlab('True Modularity Of Top Level Communities')+
  ylab('NMI Top Layer (Louvain)')+theme(axis.title.x = element_blank(), 
                                    axis.text.x = element_text(size = 11),
                                    legend.text = element_text(size = 11))


tabsub2 = tabsub[-which(is.na(tabsub$True_mod_middle)),]
tabsub2$True_mod_middle = round(tabsub2$True_mod_middle, 2)
A2 = ggplot(data = tabsub2, aes(y = Middle_NMI,  x = as.factor(True_mod_middle),
                            color = Network_type, fill = Network_type))+
  geom_boxplot(color ='black', size = 1)+#geom_point(size = 3)+
  facet_wrap(~Connection_prob, scales = 'free_x')+
  theme_bw()+
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25))+
  theme(legend.position = 'top', axis.title.x = element_blank(),
        axis.text.x = element_text(size = 11), legend.text = element_text(size = 11))+
  xlab('True Modularity Of Middle Level Communities')+
  ylab('NMI Middle Layer (HCD)')

B2 = ggplot(data = tabsub2, aes(y = Louvain_NMI_middle,  x = as.factor(True_mod_middle),
                            color = Network_type, fill = Network_type))+
  geom_boxplot(color ='black', size = 1)+#geom_point(size = 3)+
  facet_wrap(~Connection_prob, scales = 'free_x')+
  theme_bw()+
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25))+
  theme(legend.position = 'top',
        axis.text.x = element_text(size = 11), legend.text = element_text(size = 11))+
  xlab('True Modularity Of Middle Level Communities')+
  ylab('NMI Middle Layer (Louvain)')

ggarrange(A, B, nrow = 2)
ggarrange(A2, B2, nrow = 2)

png('C:/Users/Bruin/Desktop/all_nets_scatter_top.png', res = 800, height = 6, width = 12, units = 'in')
ggarrange(A, B, nrow = 2)
dev.off()

png('C:/Users/Bruin/Desktop/all_nets_scatter_middle.png', res = 800, height = 6, width = 12, units = 'in')
ggarrange(A2, B2, nrow = 2)
dev.off()







# same idea as above but y-axis is completeness
#network performance vs true modularity:

A = ggplot(data = tabsub, aes(y = Top_completeness, x = as.factor(True_mod_top),
                              color = Network_type, linetype = Layers, fill = Network_type))+
  geom_boxplot(color ='black', size = 1)+#geom_point(size = 3)+
  facet_wrap(~Connection_prob, scales = 'free_x')+
  theme_bw()+
  theme(legend.position = 'top')+
  xlab('True Modularity Of Top Level Communities')+
  ylab('NMI Top Layer (HCD)')+theme(axis.title.x = element_blank(), 
                                    axis.text.x = element_text(size = 11),
                                    legend.text = element_text(size = 11))

B = ggplot(data = tabsub, aes(y = Louvain_completeness_top, x = as.factor(True_mod_top),
                              color = Network_type, linetype = Layers, fill = Network_type))+
  geom_boxplot(color ='black', size = 1)+#geom_point(size = 3)+
  facet_wrap(~Connection_prob, scales = 'free_x')+
  theme_bw()+
  theme(legend.position = 'top')+
  xlab('True Modularity Of Top Level Communities')+
  ylab('NMI Top Layer (Louvain)')+theme(axis.title.x = element_blank(), 
                                        axis.text.x = element_text(size = 11),
                                        legend.text = element_text(size = 11))


A2 = ggplot(data = tabsub2, aes(y = Middle_Completeness,  x = as.factor(True_mod_middle),
                                color = Network_type, fill = Network_type))+
  geom_boxplot(color ='black', size = 1)+#geom_point(size = 3)+
  facet_wrap(~Connection_prob, scales = 'free_x')+
  theme_bw()+
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25))+
  theme(legend.position = 'top', axis.title.x = element_blank(),
        axis.text.x = element_text(size = 11), legend.text = element_text(size = 11))+
  xlab('True Modularity Of Middle Level Communities')+
  ylab('NMI Middle Layer (HCD)')

B2 = ggplot(data = tabsub2, aes(y = Louvain_completeness_middle,  x = as.factor(True_mod_middle),
                                color = Network_type, fill = Network_type))+
  geom_boxplot(color ='black', size = 1)+#geom_point(size = 3)+
  facet_wrap(~Connection_prob, scales = 'free_x')+
  theme_bw()+
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25))+
  theme(legend.position = 'top',
        axis.text.x = element_text(size = 11), legend.text = element_text(size = 11))+
  xlab('True Modularity Of Middle Level Communities')+
  ylab('NMI Middle Layer (Louvain)')

ggarrange(A, B, nrow = 2)
ggarrange(A2, B2, nrow = 2)

png('C:/Users/Bruin/Desktop/all_nets_scatter.png', res = 800, height = 6, width = 12, units = 'in')
ggarrange(A, A2, B, B2, nrow = 2, ncol =2,  common.legend = T)
dev.off()






#============================================================
#---------HCD vs Louvain scatterplots-top--------------------
#============================================================
#results separated by resolution (disconnected networks only)
tab2 = subset(tab, Connection_prob == 'disc' & input_graph == 'A_ingraph_true')
A = ggplot(data = tab2, aes(y = Top_NMI, x = True_mod_top, 
                            color= Network_type, shape = Layers))+
  geom_point()+
  facet_wrap(~Resolution_top)+
  theme_bw()+
  theme(legend.position = 'top')+
  xlab('True Modularity Of Top Level Communities')+
  ylab('NMI Top Layer (HCD)')

B = ggplot(data = tab2, aes(y = Louvain_NMI_top, x = True_mod_top, 
                            color= Network_type, shape = Layers))+
  geom_point()+
  facet_wrap(~Resolution_top)+
  theme_bw()+
  theme(legend.position = 'top')+
  xlab('True Modularity Of Top Level Communities')+
  ylab('NMI Top Layer (Louvain)')

ggarrange(A, B, nrow = 2, common.legend = T)




#results separated by resolution (fully connected networks only)
tab3 = subset(tab, Connection_prob == 'full')
A = ggplot(data = tab3, aes(y = Top_NMI, x = True_mod_top, 
                            color= Network_type, shape = Layers))+
  geom_point()+
  facet_wrap(~Resolution_top)+
  theme_bw()+
  theme(legend.position = 'top')+
  xlab('True Modularity Of Top Level Communities')+
  ylab('NMI Top Layer (HCD)')

B = ggplot(data = tab3, aes(y = Louvain_NMI_top, x = True_mod_top, 
                            color= Network_type, shape = Layers))+
  geom_point()+
  facet_wrap(~Resolution_top)+
  theme_bw()+
  theme(legend.position = 'top')+
  xlab('True Modularity Of Top Level Communities')+
  ylab('NMI Top Layer (Louvain)')

ggarrange(A, B, nrow = 2, common.legend = T)





#============================================================
#---------HCD vs Louvain scatterplots------------------------
#============================================================
#results separated by resolution (disconnected networks only)
tab2 = subset(tab, Connection_prob == 'disc')
A = ggplot(data = tab2, aes(y = Middle_NMI, x = True_mod_middle, 
                            color= Network_type, shape = Layers))+
  geom_point()+
  facet_wrap(~Resolution_top)+
  theme_bw()+
  theme(legend.position = 'top')+
  xlab('True Modularity Of Top Level Communities')+
  ylab('NMI Middle Layer (HCD)')

B = ggplot(data = tab2, aes(y = Louvain_NMI_middle, x = True_mod_middle, 
                            color= Network_type, shape = Layers))+
  geom_point()+
  facet_wrap(~Resolution_top)+
  theme_bw()+
  theme(legend.position = 'top')+
  xlab('True Modularity Of Top Level Communities')+
  ylab('NMI Middle Layer (Louvain)')

ggarrange(A, B, nrow = 2, common.legend = T)




#results separated by resolution (fully connected networks only)
tab3 = subset(tab, Connection_prob == 'full')
A = ggplot(data = tab3, aes(y = Middle_NMI, x = True_mod_middle, 
                            color= Network_type, shape = Layers))+
  geom_point()+
  facet_wrap(~Resolution_top)+
  theme_bw()+
  theme(legend.position = 'top')+
  xlab('True Modularity Of Top Level Communities')+
  ylab('NMI Middle Layer (HCD)')

B = ggplot(data = tab3, aes(y = Louvain_NMI_middle, x = True_mod_middle, 
                            color= Network_type, shape = Layers))+
  geom_point()+
  facet_wrap(~Resolution_top)+
  theme_bw()+
  theme(legend.position = 'top')+
  xlab('True Modularity Of Top Level Communities')+
  ylab('NMI Middle Layer (Louvain)')

ggarrange(A, B, nrow = 2, common.legend = T)






longform = cbind.data.frame(metric = as.factor(c(rep('homogeneity', dim(tab)[1]),
                              rep('completeness', dim(tab)[1]),
                              rep('NMI', dim(tab)[1]))),
                            connect = as.factor(rep(tab$Connection_prob, 3)),
                            layers = rep(tab$Layers, 3),
                            network = as.factor(rep(tab$Network_type, 3)),
                            input_graph = as.factor(rep(tab$input_graph, 3)),
                            gamma = as.factor(rep(tab$Gamma, 3)),
                            resolution_top = as.factor(rep(tab$Resolution_top, 3)),
                            resolution_middle = as.factor(rep(tab$Resolution_middle, 3)),
                            Top.stats = c(tab$Top_homogeneity, tab$Top_completeness, tab$Top_NMI),
                            Mid.stats = c(tab$Middle_homogeneity, tab$Middle_Completeness, tab$Middle_NMI))


longform.louv = cbind.data.frame(metric = as.factor(c(rep('homogeneity', dim(tab)[1]),
                                                      rep('completeness', dim(tab)[1]),
                                                      rep('NMI', dim(tab)[1]))),
                                 connect = as.factor(rep(tab$Connection_prob, 3)),
                                 layers = rep(tab$Layers, 3),
                                 network = as.factor(rep(tab$Network_type, 3)),
                                 input_graph = as.factor(rep(tab$input_graph, 3)),
                                 gamma = as.factor(rep(tab$Gamma, 3)),
                                 resolution_top = as.factor(rep(tab$Resolution_top, 3)),
                                 resolution_middle = as.factor(rep(tab$Resolution_middle, 3)),
                                 Top.stats = c(tab$Louvain_homogenity_top, 
                                                 tab$Louvain_completeness_top, 
                                                 tab$Louvain_NMI_top),
                                 Mid.stats = c(tab$Louvain_homogenity_middle, 
                                                    tab$Louvain_completeness_middle,
                                                    tab$Louvain_NMI_middle))


# lf.final = subset(cbind.data.frame(Method = c(rep('HCD', dim(tab)[1]*3), 
#                                        rep('Louvain', dim(tab)[1]*3)), 
#                             rbind.data.frame(longform, longform.louv)),
#                   input_graph == 'Input graph: True graph')

lf.final = cbind.data.frame(Method = c(rep('HCD', dim(tab)[1]*3), 
                                              rep('Louvain', dim(tab)[1]*3)), 
                                   rbind.data.frame(longform, longform.louv))


#across connectivity
A = ggplot(data = lf.final, aes(x = Method, y = Top.stats, fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Performance Top Layer')+
  facet_wrap(~connect)+
  theme_hc()+
  theme(legend.position = 'top', 
        axis.text = element_text(size = 16),
        axis.title = element_text(size = 18),
        strip.text = element_text(size = 16),
        legend.text = element_text(size = 16))

B = ggplot(data = lf.final, aes(x = Method, y = Mid.stats, fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Performance Middle Layer')+
  facet_wrap(~connect)+
  theme_hc()+
  theme(legend.position = 'top', 
        axis.text = element_text(size = 16),
        axis.title = element_text(size = 18),
        strip.text = element_text(size = 16),
        legend.text = element_text(size = 16))

 png('C:/Users/Bruin/Desktop/BCB_student_HL_talk_2024/Boxplots_summary_connectivity.png', 
     res = 500, height = 8, width = 10, units = 'in')
#pdf('C:/Users/Bruin/Desktop/BCB_student_HL_talk_2024/Boxplots_summary_connectivity.pdf', 
#    height = 8, width = 10)
ggarrange(A, B, common.legend = T, nrow = 2)
dev.off()

#Across number of layers
A = ggplot(data = lf.final, aes(x = Method, y = Top.stats, fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Performance Top Layer')+
  facet_wrap(~layers)+
  theme_hc()+
  theme(legend.position = 'top', 
        axis.text = element_text(size = 16),
        axis.title = element_text(size = 18),
        strip.text = element_text(size = 16),
        legend.text = element_text(size = 16))

B = ggplot(data = subset(lf.final, layers == '3 Layers'), 
           aes(x = Method, y = Mid.stats, fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Performance Middle Layer')+
  facet_wrap(~layers)+
  theme_hc()+
  theme(legend.position = 'top', 
        axis.text = element_text(size = 16),
        axis.title = element_text(size = 18),
        strip.text = element_text(size = 16),
        legend.text = element_text(size = 16))

 png('C:/Users/Bruin/Desktop/BCB_student_HL_talk_2024/Boxplots_summary_number_of_layers.png', 
     res = 500, height = 8, width = 10, units = 'in')
#pdf('C:/Users/Bruin/Desktop/BCB_student_HL_talk_2024/Boxplots_summary_number_of_layers.pdf', 
#    height = 8, width = 10)
ggarrange(A, B, common.legend = T, nrow = 2)
dev.off()



#Across Types of networks
A = ggplot(data = lf.final, aes(x = Method, y = Top.stats, fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Performance Top Layer')+
  facet_wrap(~network)+
  theme_hc()+
  theme(legend.position = 'top', 
        axis.text = element_text(size = 16),
        axis.title = element_text(size = 18),
        strip.text = element_text(size = 16),
        legend.text = element_text(size = 16))

B = ggplot(data = lf.final, 
           aes(x = Method, y = Mid.stats, fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Performance Middle Layer')+
  facet_wrap(~network)+
  theme_hc()+
  theme(legend.position = 'top', 
        axis.text = element_text(size = 16),
        axis.title = element_text(size = 18),
        strip.text = element_text(size = 16),
        legend.text = element_text(size = 16))

png('C:/Users/Bruin/Desktop/BCB_student_HL_talk_2024/Boxplots_summary_types_of_network.png',
    res = 500, height = 8, width = 12, units = 'in')
# pdf('C:/Users/Bruin/Desktop/BCB_student_HL_talk_2024/Boxplots_summary_types_of_network.pdf', 
#     height = 8, width = 10)
ggarrange(A, B, common.legend = T, nrow = 2)
dev.off()







#Across input graph
A = ggplot(data = lf.final, aes(x = Method, y = Top.stats, fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Performance Top Layer')+
  facet_wrap(~input_graph,nrow = 3)+
  theme_hc()+
  theme(legend.position = 'top', 
        axis.text = element_text(size = 16),
        axis.title = element_text(size = 18),
        strip.text = element_text(size = 16),
        legend.text = element_text(size = 16))

B = ggplot(data = lf.final, 
           aes(x = Method, y = Mid.stats, fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Performance Middle Layer')+
  facet_wrap(~input_graph, nrow = 3)+
  theme_hc()+
  theme(legend.position = 'top', 
        axis.text = element_text(size = 16),
        axis.title = element_text(size = 18),
        strip.text = element_text(size = 16),
        legend.text = element_text(size = 16))

png('C:/Users/Bruin/Desktop/BCB_student_HL_talk_2024/Boxplots_summary_types_of_ingraphs.png',
    res = 500, height = 8, width = 12, units = 'in')
# pdf('C:/Users/Bruin/Desktop/BCB_student_HL_talk_2024/Boxplots_summary_types_of_network.pdf', 
#     height = 8, width = 10)
ggarrange(A, B, common.legend = T, ncol = 2)
dev.off()



# ggplot(data = longform, aes(x = Top.stats, y = Mid.stats, color = metric))+
#   geom_point()+
#   geom_smooth(method = 'lm')+
#   xlab('Peformance Top Layer')+
#   ylab('Performance Middle Layer')+
#   facet_wrap(~input_graph)+
#   theme_classic2()+
#   theme(legend.position = 'top')
#   
#   
# ggplot(data = longform, aes(x = Top.stats, y = Mid.stats, color = metric))+
#   geom_point()+
#   geom_smooth(method = 'lm')+
#   xlab('Peformance Top Layer')+
#   ylab('Performance Middle Layer')+
#   facet_wrap(~network)+
#   theme_classic2()+
#   theme(legend.position = 'top')


#plotted over gamma parameter
ggplot(data = longform, aes(x = gamma, y = Top.stats, fill = metric))+
  geom_boxplot()+
  xlab('Weight Applied to Reconstruction Loss')+
  ylab('Performance Top Layer')+
  facet_wrap(~network)+
  theme_classic2()+
  theme(legend.position = 'top')


ggplot(data = longform, aes(x = gamma, y = Mid.stats, fill = metric))+
  geom_boxplot()+
  xlab('Weight Applied to Reconstruction Loss')+
  ylab('Performance Middle Layer')+
  facet_wrap(~network)+
  theme_classic2()+
  theme(legend.position = 'top')

#plotted over resolution parameter
ggplot(data = longform, aes(x = resolution_top, y = Top.stats, fill = metric))+
  geom_boxplot()+
  xlab('Resolution Value')+
  ylab('Performance Top Layer')+
  facet_wrap(~network)+
  theme_classic2()+
  theme(legend.position = 'top')


ggplot(data = longform, aes(x = resolution_middle, y = Mid.stats, fill = metric))+
  geom_boxplot()+
  xlab('Resolution Value')+
  ylab('Performance Middle Layer')+
  facet_wrap(~network)+
  theme_classic2()+
  theme(legend.position = 'top')














#Louvain Perfomance
A = ggplot(data = longform, aes(x = input_graph, y = Louvain.top, fill = metric))+
  geom_boxplot()+
  xlab('Input Graph')+
  ylab('Performance Top Layer')+
  facet_wrap(~network)+
  theme_classic2()+
  theme(legend.position = 'top', axis.text.x = element_text(angle = 90))


B = ggplot(data = longform, aes(x = input_graph, y = Louvain.middle, fill = metric))+
  geom_boxplot()+
  xlab('Resolution Value')+
  ylab('Performance Middle Layer')+
  facet_wrap(~network)+
  theme_classic2()+
  theme(legend.position = 'top', axis.text.x = element_text(angle = 90))


#HCD Perfomance
A2 = ggplot(data = longform, aes(x = input_graph, y = Top.stats, fill = metric))+
  geom_boxplot()+
  xlab('Input Graph')+
  ylab('Performance Top Layer')+
  facet_wrap(~network)+
  theme_classic2()+
  theme(legend.position = 'top', axis.text.x = element_text(angle = 90))


B2 = ggplot(data = longform, aes(x = input_graph, y = Mid.stats, fill = metric))+
  geom_boxplot()+
  xlab('Resolution Value')+
  ylab('Performance Middle Layer')+
  facet_wrap(~network)+
  theme_classic2()+
  theme(legend.position = 'top', axis.text.x = element_text(angle = 90))



ggarrange(A, A2, labels = c('Louvain', 'HCD'), ncol = 1, common.legend = T)

ggarrange(B, B2, labels = c('Louvain', 'HCD'), ncol = 1, common.legend = T)





#Louvain Perfomance
A = ggplot(data = longform, aes(x = input_graph, y = Louvain.top, fill = metric))+
  geom_boxplot()+
  xlab('Input Graph')+
  ylab('Performance Top Layer (Louvain)')+
  facet_wrap(~network)+
  theme_classic2()+
  theme(legend.position = 'top', axis.text.x = element_text(angle = 90))


B = ggplot(data = longform, aes(x = input_graph, y = Louvain.middle, fill = metric))+
  geom_boxplot()+
  xlab('Resolution Value')+
  ylab('Performance Middle Layer (Louvain)')+
  facet_wrap(~network)+
  theme_classic2()+
  theme(legend.position = 'top', axis.text.x = element_text(angle = 90))


#HCD Perfomance
A2 = ggplot(data = longform, aes(x = input_graph, y = Top.stats, fill = metric))+
  geom_boxplot()+
  xlab('Input Graph')+
  ylab('Performance Top Layer (HCD)')+
  facet_wrap(~network)+
  theme_classic2()+
  theme(legend.position = 'top', axis.text.x = element_text(angle = 90))


B2 = ggplot(data = longform, aes(x = input_graph, y = Mid.stats, fill = metric))+
  geom_boxplot()+
  xlab('Resolution Value')+
  ylab('Performance Middle Layer (HCD)')+
  facet_wrap(~network)+
  theme_classic2()+
  theme(legend.position = 'top', axis.text.x = element_text(angle = 90))



ggarrange(A, A2, labels = c('Louvain', 'HCD'), ncol = 1, common.legend = T)

ggarrange(B, B2, labels = c('Louvain', 'HCD'), ncol = 1, common.legend = T)
















longform2 = cbind.data.frame(method = as.factor(c(rep('HCD', dim(longform)[1]), 
                                        rep('Louvain', dim(longform)[1]))),
                             metric = as.factor(rep(c(rep('homogeneity', dim(tab)[1]),
                                         rep('completeness', dim(tab)[1]),
                                         rep('NMI', dim(tab)[1])), 2)),
                  network = as.factor(rep(tab$Network_type, 6)),
                  input_graph = as.factor(rep(tab$input_graph, 6)),
                  gamma = as.factor(rep(tab$Gamma, 6)),
                  resolution_top = as.factor(rep(tab$Resolution_top, 6)),
                  resolution_middle = as.factor(rep(tab$Resolution_middle, 6)),
                  stats.top = c(longform$Top.stats, longform$Louvain.top),
                  stats.mid = c(longform$Mid.stats, longform$Louvain.middle))




A = ggplot(data = subset(longform2, method == 'HCD'), 
       aes(x = network, y = stats.top, 
                         fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Top Layer Performance')+
  facet_wrap(~input_graph, nrow = 3)+
  theme_hc()+
  theme(legend.position = 'none',axis.text = element_text(size = 11))

B = ggplot(data = subset(longform2, method == 'Louvain'), 
           aes(x = network, y = stats.top, 
               fill = metric))+
  geom_boxplot()+
  scale_y_continuous()+
  xlab('Method')+
  ylab('Top Layer Performance')+
  facet_wrap(~input_graph, nrow = 3)+
  theme_hc()+
  theme(legend.position = 'none',axis.text = element_text(size = 11))

ggarrange(A, B)



A = ggplot(data = subset(longform2, method == 'HCD'), 
           aes(x = network, y = stats.mid, 
               fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Middle Layer Performance')+
  facet_wrap(~input_graph, nrow = 3)+
  theme_hc()+
  theme(legend.position = 'none',axis.text = element_text(size = 11))

B = ggplot(data = subset(longform2, method == 'Louvain'), 
           aes(x = network, y = stats.mid, 
               fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Middle Layer Performance')+
  facet_wrap(~input_graph, nrow = 3)+
  theme_hc()+
  theme(legend.position = 'none',axis.text = element_text(size = 11))

ggarrange(A, B)



#comparing Louvain and HCD:

#Top level comparison
#smallworld network
lf1.1 = subset(longform2, network == 'small world')
#png('C:/Users/Bruin/Desktop/small_world_top_allperf.png', res = 500, height = 6, width = 7, units = 'in')
p1 = ggplot(data = lf1.1, aes(x = method, y = stats.top, 
                             fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Top Layer Performance')+
  facet_wrap(~input_graph, nrow = 3)+
  theme_classic2()+
  theme(legend.position = 'none',axis.text = element_text(size = 11))
#dev.off()
#scale free
lf1.2 = subset(longform2, network == 'scale free')
p11 = ggplot(data = lf1.2, aes(x = method, y = stats.top, 
                             fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Top Layer Performance')+
  facet_wrap(~input_graph, nrow = 3)+
  theme_classic2()+
  theme(legend.position = 'top',  axis.text = element_text(size = 11))

#random graph
lf1.3 = subset(longform2, network == 'random graph')
ggplot(data = lf1.3, aes(x = method, y = stats.top, 
                             fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Top Layer Performance')+
  facet_wrap(~input_graph)+
  theme_classic2()+
  theme(legend.position = 'top', axis.text.x = element_text(angle = 90))



#comparing Louvain and HCD:
#comparing on middle level
#smallworld network
lf1.1 = subset(longform2, network == 'small world')
#png('C:/Users/Bruin/Desktop/small_world_middle_allperf.png', res = 500, height = 6, width = 7, unit = 'in')
p2 = ggplot(data = lf1.1, aes(x = method, y = stats.mid, 
                         fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Middle Layer Performance')+
  facet_wrap(~input_graph, nrow = 3)+
  theme_classic2()+
  theme(legend.position = 'top', axis.text = element_text(size = 11))
#dev.off()
png('C:/Users/Bruin/Desktop/small_world_top_allperf.png', res = 500, height = 6, width = 12, units = 'in')
ggarrange(p2, p1, nrow = 1, common.legend = T)
dev.off()
#scale free
lf1.2 = subset(longform2, network == 'scale free')
p22 = ggplot(data = lf1.2, aes(x = method, y = stats.mid, 
                         fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Middle Layer Performance')+
  facet_wrap(~input_graph, nrow =3)+
  theme_classic2()+
  theme(legend.position = 'top',  axis.text = element_text(size = 11))

png('C:/Users/Bruin/Desktop/scale_free_top_allperf.png', res = 500, height = 6, width = 12, units = 'in')
ggarrange(p22, p11, nrow = 1, common.legend = T)
dev.off()
#random graph
lf1.3 = subset(longform2, network == 'random graph')
ggplot(data = lf1.3, aes(x = method, y = stats.mid, 
                         fill = metric))+
  geom_boxplot()+
  xlab('Method')+
  ylab('Middle Layer Performance')+
  facet_wrap(~input_graph)+
  theme_classic2()+
  theme(legend.position = 'top', axis.text.x = element_text(angle = 90))


















longform3 = cbind.data.frame(method = as.factor(c(rep('HCD_middle', dim(tab)[1]),
                                                  rep('HCD_top', dim(tab)[1]),
                                                  rep('Louvain', dim(tab)[1]))),
                             predicted = c(tab$Comms_predicted_middle, 
                                           tab$Comms_predicted_top,
                                           tab$Louvain_predicted))
ggplot(data = longform3, aes(x = method, y = predicted))+geom_boxplot()


summary(subset(tab, input_graph == 'A_corr_no_cutoff')$Top_homogeneity)
summary(subset(tab, input_graph == 'A_corr_no_cutoff')$Top_completeness)
summary(subset(tab, input_graph == 'A_corr_no_cutoff')$Top_NMI)







sub.table = function(data, net, graph, connect, layers, stat = 'mean'){
  sub = subset(data, Connection_prob == connect & Network_type == net & input_graph == graph & Layers == layers)
  
  if(stat == 'mean'){
    hcd.top = c(mean(sub$Top_completeness, na.rm = T),
                mean(sub$Top_homogeneity, na.rm = T),
                mean(sub$Top_NMI, na.rm = T))
    
    hcd.middle = c(mean(sub$Middle_Completeness, na.rm = T),
                   mean(sub$Middle_homogeneity, na.rm = T),
                   mean(sub$Middle_NMI, na.rm = T))
    
    louv.top = c(mean(sub$Louvain_completeness_top, na.rm = T),
                 mean(sub$Louvain_homogenity_top, na.rm = T),
                 mean(sub$Louvain_NMI_top, na.rm = T))
    
    louv.middle = c(mean(sub$Louvain_completeness_middle, na.rm = T),
                    mean(sub$Louvain_homogenity_middle, na.rm = T),
                    mean(sub$Louvain_NMI_middle, na.rm = T))
  }else{
    hcd.top = c(sd(sub$Top_completeness, na.rm = T),
                sd(sub$Top_homogeneity, na.rm = T),
                sd(sub$Top_NMI, na.rm = T))
    
    hcd.middle = c(sd(sub$Middle_Completeness, na.rm = T),
                   sd(sub$Middle_homogeneity, na.rm = T),
                   sd(sub$Middle_NMI, na.rm = T))
    
    louv.top = c(sd(sub$Louvain_completeness_top, na.rm = T),
                 sd(sub$Louvain_homogenity_top, na.rm = T),
                 sd(sub$Louvain_NMI_top, na.rm = T))
    
    louv.middle = c(sd(sub$Louvain_completeness_middle, na.rm = T),
                 sd(sub$Louvain_homogenity_middle, na.rm = T),
                 sd(sub$Louvain_NMI_middle, na.rm = T))
  }
  
  return(list(hcd.top = hcd.top, hcd.middle = hcd.middle, 
              louv.top = louv.top, louv.middle = louv.middle))

}






create_summ_table = function(tab){
  
  net_types = unique(tab$Network_type)
  #graph_types = unique(tab$input_graph)
  graph_types = "Input graph: True graph"
  connect_types = unique(tab$Connection_prob)
  lays = unique(tab$Layers)
  #HCD
  mean.hcd.top = NULL
  mean.hcd.middle = NULL
  sd.hcd.top = NULL
  sd.hcd.middle = NULL
  #louvain
  mean.louv.top = NULL
  mean.louv.middle = NULL
  sd.louv.top = NULL
  sd.louv.middle = NULL
  #graph and net
  Conn = NULL
  Net = NULL
  Graph = NULL
  Num_Layers = NULL
  for(j in 1:length(net_types)){
    for(i in 1:length(connect_types)){
      for(l in 1:length(lays)){
        for(k in 1:length(graph_types)){
          Conn = append(Conn, connect_types[i])
          Net = append(Net, net_types[j])
          Graph = append(Graph, graph_types[k])
          Num_Layers = append(Num_Layers, lays[l])
          #HCD
          mean.hcd.top = append(mean.hcd.top, sub.table(tab, 
                                                        connect = connect_types[i], 
                                                        net = net_types[j], 
                                                        graph = graph_types[k], 
                                                        layers = lays[l])$hcd.top)
          mean.hcd.middle = append(mean.hcd.top, sub.table(tab, 
                                                           connect = connect_types[i], 
                                                           net = net_types[j], 
                                                           graph = graph_types[k], 
                                                           layers = lays[l])$hcd.middle)
          sd.hcd.top = append(mean.hcd.top, sub.table(tab, 
                                                      connect = connect_types[i], 
                                                      net = net_types[j], 
                                                      graph = graph_types[k], 
                                                      layers = lays[l],
                                                      stat = 'sd')$hcd.top)
          sd.hcd.middle = append(mean.hcd.top, sub.table(tab, 
                                                         connect = connect_types[i], 
                                                         net = net_types[j], 
                                                         graph = graph_types[k], 
                                                         layers = lays[l],
                                                         stat = 'sd')$hcd.middle)
          
          
          #Louv
          mean.louv.top = append(mean.hcd.top, sub.table(tab, 
                                                         connect = connect_types[i], 
                                                         net = net_types[j], 
                                                         graph = graph_types[k], 
                                                         layers = lays[l])$louv.top)
          mean.louv.middle = append(mean.hcd.top, sub.table(tab, 
                                                            connect = connect_types[i], 
                                                            net = net_types[j], 
                                                            graph = graph_types[k], 
                                                            layers = lays[l])$louv.middle)
          sd.louv.top = append(mean.hcd.top, sub.table(tab, 
                                                       connect = connect_types[i], 
                                                       net = net_types[j], 
                                                       graph = graph_types[k], 
                                                       layers = lays[l],
                                                       stat = 'sd')$louv.top)
          sd.louv.middle = append(mean.hcd.top, sub.table(tab, 
                                                          connect = connect_types[i], 
                                                          net = net_types[j], 
                                                          graph = graph_types[k], 
                                                          layers = lays[l],
                                                          stat = 'sd')$louv.middle)
        }
      }
    }
  }
  
  HCD.top.stats = cbind.data.frame(Net, Conn, Num_Layers, Graph, 
                                   as.data.frame(matrix(mean.hcd.top, nrow = 12, ncol = 3, byrow = T)))
  Louv.top.stats = cbind.data.frame(Net, Conn, Num_Layers, Graph, 
                                   as.data.frame(matrix(mean.louv.top, nrow = 12, ncol = 3, byrow = T)))
  colnames(HCD.top.stats) = colnames(Louv.top.stats) = c("Network", "Connection", "Num_Layers", "Input Graph",
                              "Completeness", "Homogeneity", 'NMI')
  return(list(HCD.top.stats, Louv.top.stats))
}
out = create_summ_table(tab)























ggplot(data = tab, aes(x = ))