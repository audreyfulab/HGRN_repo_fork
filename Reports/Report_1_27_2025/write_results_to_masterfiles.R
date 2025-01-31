
library(dplyr)

#fp = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Reports/Report_1_3_2025/Output/Intermediate_applications/SET1/linear_layers_unk_k_1_9_2025/'
#fp = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Reports/Report_1_3_2025/Output/Intermediate_applications/SET2/GATCONV_v2_unk_k_1_9_2025/'


fp1 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/Linear_15_5_opt_clusts_False/'
fp2 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/Linear_64_5_opt_clusts_False/'
fp3 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/Linear_64_64_opt_clusts_False/'
fp4 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/None_15_5_opt_clusts_False/'
fp5 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/None_64_5_opt_clusts_False/'
fp6 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/None_64_64_opt_clusts_False/'

flist = c(fp1, fp2, fp3, fp4, fp5, fp6)
#flist = c(fp1)

for(sim in 1:length(flist)){
  
  fp = flist[sim]
  graphs = c('small_world', 'scale_free', 'random_graph')
  connect = c('disc', 'full')
  case = 'Case'
  sd = c('01', '05')
  case.num = as.character(c(0:24))
  
  fns = expand.grid(graphs, connect, sd, case, case.num)
  
  fn = apply(fns, MARGIN = 1, FUN = function(x) paste(x, collapse = '_'))
  
  
  tables_mid = vector('list', length = length(fn))
  tables_top = vector('list', length = length(fn))
  netstats = vector('list', len = length(fn))
  
  read_file = function(filename){
    file = read.csv(filename, row.names = 1)
    return(file)
  }
  
  
  read_file_stats = function(filename, case){
    file = read.csv(filename, row.names = 1)
    file$CaseName = case
    return(file)
  }
  
  
  for(i in 1:length(fn)){
    
    filename1 = paste0(fp, fn[i], '.csv')
    filename2 = paste0(fp, fn[i], '.csv')
    filename3 = paste0(fp, fn[i], '/Simulation_Results.csv')
    
    tables_mid[[i]] = tryCatch({
      read_file(filename1)[c(1,3,5),]
    }, error = function(msg){
      return(NA)
    })
    
    
    tables_top[[i]] = tryCatch({
      read_file(filename2)[c(2,4,6),]
    }, error = function(msg){
      return(NA)
    })
    
    
    netstats[[i]] = tryCatch({
      read_file(filename3, fn[i])
    }, error = function(msg){
      return(NA)
    })
  }
  
  data_mid = do.call('rbind', tables_mid)
  data_top = do.call('rbind', tables_top)
  
  
  data_top$subgraph_type = as.factor(data_top$subgraph_type)
  data_top$Method = as.factor(data_top$Method)
  data_top$Method <- recode(data_top$Method,
                            "Louvain Top" = "Louvain",
                            "HCD Top" = "HCD",
                            "HC Top" = "HC" )
  
  
  
  data_mid$subgraph_type = as.factor(data_mid$subgraph_type)
  data_mid$Method = as.factor(data_mid$Method)
  data_mid$Method <- recode(data_mid$Method,
                            "Louvain Middle" = "Louvain",
                            "HCD Middle" = "HCD",
                            "HC Middle" = "HC" )
  
  data_top$StDev = as.factor(data_top$StDev)
  data_mid$StDev = as.factor(data_mid$StDev)
  
  netstats_final = do.call('rbind', netstats)
  
  
  write.csv(data_top, file = paste0(fp, 'combined_all_results_top.csv'))
  write.csv(data_mid, file = paste0(fp, 'combined_all_results_middle.csv'))
  
  write.csv(netstats_final, paste0(fp, 'combined_sim_results.csv'))
}



