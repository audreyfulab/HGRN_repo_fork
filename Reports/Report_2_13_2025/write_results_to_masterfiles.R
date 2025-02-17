
library(dplyr)

#fp = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Reports/Report_1_3_2025/Output/Intermediate_applications/SET1/linear_layers_unk_k_1_9_2025/'
#fp = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Reports/Report_1_3_2025/Output/Intermediate_applications/SET2/GATCONV_v2_unk_k_1_9_2025/'


fp1 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/Linear_15_5_opt_clusts_False/'
fp2 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/Linear_64_5_opt_clusts_False/'
fp3 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/Linear_64_5_opt_clusts_bethe_hessian/'
fp4 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/Linear_64_5_opt_clusts_silouette/'


fp5 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/SAGEConv_15_5_opt_clusts_False/'
fp6 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/SAGEConv_64_5_opt_clusts_False/'
fp7 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/SAGEConv_64_5_opt_clusts_bethe_hessian/'
fp8 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/SAGEConv_64_5_opt_clusts_silouette/'

fp9 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/None_15_5_opt_clusts_False/'
fp10 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/None_64_5_opt_clusts_False/'
fp11 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/None_64_5_opt_clusts_bethe_hessian/'
fp12 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/None_64_5_opt_clusts_silouette/'

fp13 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/GATv2Conv_15_5_opt_clusts_False/'
fp14 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/GATv2Conv_64_5_opt_clusts_False/'
fp15 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/GATv2Conv_64_5_opt_clusts_bethe_hessian/'
fp16 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/GATv2Conv_64_5_opt_clusts_silouette/'


fp17 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/Kmeans_None_15_5_opt_clusts_False/'
fp18 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/Kmeans_None_64_5_opt_clusts_False/'
fp19 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/Kmeans_None_64_5_opt_clusts_bethe_hessian/'
fp20 = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER/Kmeans_None_64_5_opt_clusts_silouette/'


flist = c(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, fp10, fp11, fp12, fp13, fp14, fp15, fp16,
          fp17, fp18, fp19, fp20)


read_file = function(filename){
  file = read.csv(filename, row.names = 1)
  return(file)
}


read_file_stats = function(filename, case){
  file = read.csv(filename, row.names = 1)
  file$CaseName = case
  return(file)
}

process.results = function(PATH, ol, scenario, checker){
  
    
  fp = PATH
  graphs = c('small_world', 'scale_free', 'random_graph')
  connect = c('disc', 'full')
  case = 'Case'
  sd = c('01', '05')
  case.num = as.character(c(0:24))
  
  fns = expand.grid(graphs, connect, sd, case, case.num)
  
  fn = apply(fns, MARGIN = 1, FUN = function(x) paste(x, collapse = '_'))
  
  tables_mid = vector('list', length = length(fn))
  tables_top = vector('list', length = length(fn))
  netstats = vector('list', length = length(fn))
  
  for(i in 1:length(fn)){
    
    
    filename1 = paste0(fp, fn[i], '.csv')
    filename2 = paste0(fp, fn[i], '.csv')
    filename3 = paste0(fp, fn[i], '/Simulation_Results.csv')
    
    
    
    if(checker > 1){
      tables_mid[[i]] = tryCatch({
        read_file(filename1)[5,]
      }, error = function(msg){
        return(NA)
      })
      
      
      tables_top[[i]] = tryCatch({
        read_file(filename2)[6,]
      }, error = function(msg){
        return(NA)
      })
    }else{
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
    }
    
    
    
    netstats[[i]] = tryCatch({
      read_file(filename3, fn[i])
    }, error = function(msg){
      return(NA)
    })
  }
    
    
  data_mid = do.call('rbind', tables_mid)
  data_top = do.call('rbind', tables_top)
  
  data_mid = do.call('rbind', tables_mid)
  data_top = do.call('rbind', tables_top)
  
  
  data_top$subgraph_type = as.factor(data_top$subgraph_type)
  data_top$Method = as.factor(data_top$Method)
  data_top$Method <- recode(data_top$Method,
                            "Louvain Top" = "Louvain",
                            "HCD Top" = paste0("HCD-", ol),
                            "HC Top" = "HC" )
  
  
  
  data_mid$subgraph_type = as.factor(data_mid$subgraph_type)
  data_mid$Method = as.factor(data_mid$Method)
  data_mid$Method <- recode(data_mid$Method,
                            "Louvain Middle" = "Louvain",
                            "HCD Middle" = paste0("HCD-", ol),
                            "HC Middle" = "HC" )
  
  data_top$StDev = as.factor(data_top$StDev)
  data_mid$StDev = as.factor(data_mid$StDev)
  
  netstats_final = do.call('rbind', netstats)
  write.csv(data_top, file = paste0(fp, 'combined_all_results_top.csv'))
  write.csv(data_mid, file = paste0(fp, 'combined_all_results_middle.csv'))
  write.csv(netstats_final, paste0(fp, 'combined_sim_results.csv'))
  
  comb_temp = rbind.data.frame(data_top, data_mid)
  ft = cbind.data.frame(comb_temp, 
                        Simulation = c(fn, fn),
                        Layer = c(rep('Top Layer', dim(data_top)[1]), 
                                  rep('Middle Layer', dim(data_mid)[1])),
                        Scenario = rep(scenario, dim(comb_temp)[1]),
                        output_layer = rep(ol, dim(comb_temp)[1]))
  
  
  return(list(final.table = ft, tab.with.stats = netstats, data.middle = data_mid, data.top = data_top))
    
}


reslist = vector('list', length = length(flist))
outlayer = c('Linear', 'Linear', 'Linear', 'Linear',
             'SAGE', 'SAGE', 'SAGE', 'SAGE',
             'NOL', 'NOL', 'NOL', 'NOL',
             'GATv2', 'GATv2', 'GATv2','GATv2',
             'Kmeans-NOL', 'Kmeans-NOL', 'Kmeans-NOL','Kmeans-NOL')
scenarios = c('GT', '64-5', 'BH', 'Silouette',
              'GT', '64-5', 'BH', 'Silouette',
              'GT', '64-5', 'BH', 'Silouette',
              'GT', '64-5', 'BH', 'Silouette',
              'GT', '64-5', 'BH', 'Silouette')
for(sim in 1:length(flist)){
  output = process.results(PATH = flist[sim], ol = outlayer[sim], scenario = scenarios[sim], checker = 0)
  
  reslist[[sim]] = output$final.table
}

final.combined.all = do.call('rbind', reslist)

write.csv(final.combined.all, 
          file = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/combined_results_all_simulations.csv')
