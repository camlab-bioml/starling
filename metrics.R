rm(list=ls())
library(tidyr)
library(dplyr)
library(ggplot2)
#library(rmote)

## doublet performance
syth_perf <- function() {
  
  fns <- list.files(pattern = "doub_perf*")
  
  mat <- NULL
  for (i in 1:length(fns)) {
    tmp <- t(as.matrix(read.csv(fns[i])[,2]))
    mat <- rbind(mat, c(fns[i], tmp[,-c(1:8)]))
  }

  parsed_fn <- do.call(rbind, strsplit(mat[,1], "_"))[,-(1:2)]
  
  nc <- substr(parsed_fn[,1],3,100)
  ia <- substr(parsed_fn[,2],3,100)
  cf <- substr(parsed_fn[,3],3,100)
  nm <- substr(parsed_fn[,4],3,100)
  so <- substr(parsed_fn[,5],3,100)
  lv <- substr(parsed_fn[,6],3,100)
  rr <- substr(parsed_fn[,7],3,100)
  r <- substr(parsed_fn[,8],2,2)
  
  amat <- as.data.frame(cbind(nc, ia, cf, nm, so, lv, rr, r, mat[,-1]))
  
  colnames(amat) <- c('nc', 'ia', 'cf', 'nm', 'so', 'lv', 'rr', 'r', 
                      'val_acc', 'val_f1', 'val_auroc', 'test_acc', 'test_f1', 'test_auroc')
  
  res <- pivot_longer(amat, cols=9:14, names_to='metric', values_to ="value")
  out <- res %>% mutate(score = as.numeric(as.character(value)))

  return(as.data.frame(out[,-10]))
}

####
####

new_corr_metric <- function(df, anti, pairs, thresholds) {
  
  out <- NULL
  for ( i in 1:dim(pairs)[1] ) {
    
    x_name <- pairs[i,1]; y_name <- pairs[i,2]
    x1 <- df[x_name]; y1 <- df[y_name]
    
    if (anti == 1) {
      ## case 1: negative
      ns <- sum((x1 < thresholds[x_name][1,1]) | (y1 < thresholds[y_name][1,1]))/dim(x1)[1]
      out <- rbind(out, cbind(paste0(x_name, '_', y_name), ns, -1))
    } else {
      ## case 2: positive
      ps <- sum((x1 < thresholds[x_name][1,1]) | (y1 > thresholds[y_name][2,1]))/dim(x1)[1]
      out <- rbind(out, cbind(paste0(x_name, '_', y_name), ps, 1))
    }
  }
  res <- as.data.frame(out); colnames(res) <- c('pair', 'metric', 'type')
  return(res)
}

plausibility_score <- function(cohort) {
  
  if (cohort == 'eddy') {
    anti_pairs <- rbind(
      c('CD3','CD20'), c('CD3','CD31'), c('CD3','CD68'), c('CD3','ECadherin'),
      c('CD4','CD8'), c('CD4','CD20'), c('CD4','CD31'), c('CD4','ECadherin'),
      c('CD8','CD20'), c('CD8','CD31'), c('CD8','CD68'), c('CD8','ECadherin'),
      c('CD20','CD31'), c('CD20','CD68'), c('CD20','ECadherin'),
      c('CD31','CD68'), c('CD31','ECadherin'), c('CD68','ECadherin'))
    
    co_pairs <- rbind(
      c('CD3','CD4'), c('CD3','CD8'), c('CD45','CD3'), c('CD45','CD4'),
      c('CD45','CD8'), c('CD45','CD20'), c('CD45','CD68'))
    
  } else {
    anti_pairs <- rbind(
      c('CD3', 'CD20'), c('CD3', 'CD68'), c('CD3', 'pCK'), c('CD3', 'CD31'),
      c('CD20', 'CD68'), c('CD20', 'pCK'), c('CD20', 'CD31'), c('CD45', 'pCK'), 
      c('pCK', 'Vimentin'))
    
    co_pairs <- rbind(
      c('CD45', 'CD3'), c('CD45', 'CD20'), c('CD45', 'CD68'), c('pCK', 'CK5'),
      c('pCK', 'CK7'), c('pCK', 'CK8n18'), c('pCK', 'CK19'))
  }

  afns <- list.files(pattern = "damm_cen*")
  
  #trY <- read.csv(paste0("/home/campbell/yulee/DAMM/new/res/", cohort, "/", ss, "k/tr_ex2_", substr(parsed_fn[3], 3, 3), ".csv"))[,-1]
  trY <- read.csv(paste0("/home/campbell/yulee/DAMM/new/data/", cohort, "/", cohort, "_dc_samples.csv"))[,-c(1:8)]
  
  mat <- NULL
  for (i in 1:length(afns)) {
    
    print(i)
    #i = 1
    parsed_fn <- strsplit(afns[i], "_")[[1]][-(1:2)]
    
    bfn <- paste0("init_cen_", paste0(parsed_fn, collapse = "_"))
    
    if ( file.exists(bfn) == FALSE ) { next } 
    
    nc <- substr(parsed_fn[1],3,100)
    ia <- substr(parsed_fn[2],3,100)
    cf <- substr(parsed_fn[3],3,100)
    nm <- substr(parsed_fn[4],3,100)
    so <- substr(parsed_fn[5],3,100)
    lv <- substr(parsed_fn[6],3,100)
    rr <- substr(parsed_fn[7],3,100)
    r <- substr(parsed_fn[8],2,2)
    
    if ( as.integer(cf) > 0 ) {
      thresholds <- as.data.frame( sapply(asinh(trY / as.integer(cf)), function(x) quantile(x, c(0.25, 0.75))) )
    } else {
      thresholds <- as.data.frame(sapply(trY, function(x) quantile(x, c(0.25, 0.75))))  
    }
    
    bmat <- read.csv(bfn)[,-1]
    amat <- read.csv(afns[i])[,-1]
    
    bneg <- new_corr_metric(bmat, 1, anti_pairs, thresholds)
    bpos <- new_corr_metric(bmat, 0, co_pairs, thresholds)
    
    aneg <- new_corr_metric(amat, 1, anti_pairs, thresholds)
    apos <- new_corr_metric(amat, 0, co_pairs, thresholds)
    
    mat <- rbind(mat, cbind(nc, ia, cf, nm, so, lv, rr, r, before=rbind(bneg, bpos)[,-3], after=rbind(aneg, apos)[,2]))
  }
  as.data.frame(mat)
}

plausibility_score <- function(trY, po_pairs, ne_pairs) {

  afns <- list.files(pattern = "damm_cen*")

  pos_pairs <- do.call(rbind, po_pairs)
  neg_pairs <- do.call(rbind, ne_pairs)
  
  mat <- NULL
  for (i in 1:length(afns)) {
    
    print(i)
    parsed_fn <- strsplit(afns[i], "_")[[1]][-(1:2)]
    
    bfn <- paste0("init_cen_", paste0(parsed_fn, collapse = "_"))
    
    if ( file.exists(bfn) == FALSE ) { next } 
    
    nc <- substr(parsed_fn[1],3,100)
    ia <- substr(parsed_fn[2],3,100)
    cf <- substr(parsed_fn[3],3,100)
    nm <- substr(parsed_fn[4],3,100)
    so <- substr(parsed_fn[5],3,100)
    lv <- substr(parsed_fn[6],3,100)
    rr <- substr(parsed_fn[7],3,100)
    r <- substr(parsed_fn[8],2,2)
    
    if ( as.integer(cf) > 0 ) {
      thresholds <- as.data.frame( sapply(asinh(trY / as.integer(cf)), function(x) quantile(x, c(0.25, 0.75))) )
    } else {
      thresholds <- as.data.frame(sapply(trY, function(x) quantile(x, c(0.25, 0.75))))  
    }
    
    bmat <- read.csv(bfn)[,-1]
    amat <- read.csv(afns[i])[,-1]
    
    bneg <- new_corr_metric(bmat, 1, neg_pairs, thresholds)
    bpos <- new_corr_metric(bmat, 0, pos_pairs, thresholds)
    
    aneg <- new_corr_metric(amat, 1, neg_pairs, thresholds)
    apos <- new_corr_metric(amat, 0, pos_pairs, thresholds)
    
    mat <- rbind(mat, cbind(nc, ia, cf, nm, so, lv, rr, r, before=rbind(bneg, bpos)[,-3], after=rbind(aneg, apos)[,2]))
  }
  as.data.frame(mat)
}

####
####

## Morphological score
morphological_score <- function(init_algo) {
  
  fns <- list.files(pattern = "algo_label*")

  mat <- NULL
  for (i in 1:length(fns)) {
    
    #i = 1
    print(i)
    label_mat <- read.csv(fns[i])
    
    tmp <- NULL
    for (j in 1:length(init_algo)) {
      
      singlet_set <- filter(label_mat, !!sym(paste0('damm_', init_algo[j], '_tr_label')) != 'doublets')
      doublet_set <- filter(label_mat, !!sym(paste0('damm_', init_algo[j], '_tr_label')) == 'doublets')
      
      df1 <- singlet_set['area_convex']/singlet_set['area']
      df2 <- doublet_set['area_convex']/doublet_set['area']
      
      ttres <- t.test(df1, df2)
      
      tmp <- rbind(tmp, cbind(ia=init_algo[j], Size=ttres$estimate[2] - ttres$estimate[1], Pval=ttres$p.value))
    }
    
    parsed_fn <- strsplit(fns[i], "_")[[1]][-(1:2)]
    nc <- substr(parsed_fn[1],3,100)
    #ia <- substr(parsed_fn[2],3,100)
    cf <- substr(parsed_fn[3],3,100)
    nm <- substr(parsed_fn[4],3,100)
    so <- substr(parsed_fn[5],3,100)
    lv <- substr(parsed_fn[6],3,100)
    rr <- substr(parsed_fn[7],3,100)
    r <- substr(parsed_fn[8],2,2)
    
    mat <- rbind(mat, cbind(nc, cf, nm, so, lv, rr, r, tmp))
  }
  as.data.frame(mat)
}

### R version
library(yaml)
library(argparse)

parser <- ArgumentParser(description = "run STARLING metrics")
parser$add_argument('--cohort', type = 'character', help = 'which cohort to use?')
args <- parser$parse_args()

cohort <- args$cohort
print(cohort)

config <- read_yaml("/home/campbell/yulee/DAMM/new/code/pro/local/config.yml")

working_dir <- paste0(config['res_path'], cohort, '/', config['sample_size'], 'k/exp', config['experiment_id'], '/')
setwd(working_dir)

## synthetic performances
sp_res <- syth_perf()
write.csv(sp_res, file = paste0(working_dir, 'sp_res.csv')) 

## plausibility score
trY <- read.csv(paste0(config['data_path'], cohort, "/", cohort, "_dc_samples.csv"))[,-c(1:7)]
ps_res <- plausibility_score(trY, config$plausibility_markers[[cohort]]$positive, config$plausibility_markers[[cohort]]$negative)
write.csv(ps_res, file = paste0(working_dir, 'ps_res.csv')) 

## morphological score
ms_res <- morphological_score(config$initial_algo)
write.csv(ms_res, file = paste0(working_dir, 'ms_res.csv')) 

#myargs = commandArgs(trailingOnly=TRUE)
#print(myargs[2])

#cohort <- myargs[1]
#expNum <- myargs[2]
#ss <- myargs[3]

#print(cohort)
#setwd(paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/'))
  
#sp_res <- syth_perf()
#write.csv(sp_res, file = paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/sp_res.csv')) 
  
#ps_res <- plausibility_score(cohort)
#write.csv(ps_res, file = paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/ps_res.csv')) 
  
#ms_res <- morphological_score()
#write.csv(ms_res, file = paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/ms_res.csv')) 

###Rscript metrics.R eddy 1 50 > eddy_exp1_50k

#expNum <- 1
#ss <- 50
#cohorts <- c('eddy', 'basel', 'meta')

#for (i in 1:length((cohorts))) {
#    print(cohorts[i])
#    setwd(paste0('/home/campbell/yulee/DAMM/new/res/', cohorts[i], '/', ss, 'k/exp', expNum, '/'))
  
#    sp_res <- syth_perf()
#    write.csv(sp_res, file = paste0('/home/campbell/yulee/DAMM/new/res/', cohorts[i], '/', ss, 'k/exp', expNum, '/sp_res.csv')) 
  
#    ps_res <- plausibility_score(cohorts[i])
#    write.csv(ps_res, file = paste0('/home/campbell/yulee/DAMM/new/res/', cohorts[i], '/', ss, 'k/exp', expNum, '/ps_res.csv')) 
  
#    ms_res <- morphological_score()
#    write.csv(ms_res, file = paste0('/home/campbell/yulee/DAMM/new/res/', cohorts[i], '/', ss, 'k/exp', expNum, '/ms_res.csv')) 
#}