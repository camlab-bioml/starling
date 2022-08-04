rm(list=ls())
library(ComplexHeatmap)


## original data
trY <- read.csv(paste0("/home/campbell/yulee/DAMM/new/data/eddy/eddy_dc_samples.csv"))[,-c(1:7)]
#ori_cor <- cor(trY)

cutoff <- 0.25
cutoffs <- as.data.frame(sapply(trY, function(x) quantile(x, c(cutoff, 1-cutoff))))

#out <- NULL
#pairs <- combn(dim(cutoffs)[2], 2)
pos_out <- matrix(NA, nrow=dim(cutoffs)[2], ncol=dim(cutoffs)[2])
neg_out <- matrix(NA, nrow=dim(cutoffs)[2], ncol=dim(cutoffs)[2])
row.names(neg_out) <- colnames(neg_out) <- row.names(pos_out) <- colnames(pos_out) <- names(cutoffs)

for ( i in 1:dim(cutoffs)[2] ) {
  for ( j in 1:dim(cutoffs)[2] ) {
  
    x_name <- names(cutoffs)[i]
    y_name <- names(cutoffs)[j]
  
    score1 = sum((trY[,x_name] < cutoffs[x_name][1,1]) | (trY[,y_name] < cutoffs[y_name][1,1]))/dim(trY)[1]
    neg_out[x_name, y_name] <- score1
  
    score2 = sum((trY[,x_name] < cutoffs[x_name][1,1]) | (trY[,y_name] > cutoffs[y_name][1,1]))/dim(trY)[1]
    pos_out[x_name, y_name] <- score2
  }
}

png("~/ps_neg_raw.png")
ht <- Heatmap(neg_out, name = "Score", cluster_rows = TRUE)
draw(ht)
dev.off()

png("~/ps_pos_raw.png")
ht <- Heatmap(pos_out, name = "Score", cluster_rows = TRUE)
draw(ht)
dev.off()


## sampled data (load dataset & correlation)
trYs <- read.csv(paste0("/home/campbell/yulee/DAMM/new/res/eddy/50k/exp3/fs_nc23_iaGMM_cf1_nm2_so0.0_lv1.0_rr1_r1.csv"))[,-c(1,26)]
colnames(trYs) <- c('SMA', 'ECadherin', 'Cytokeratin', 'HLADR', 'Vimentin', 'CD28', 'CD15', 'CD45RA', 'CD66b', 
                   'CD20', 'CD68', 'CD4', 'CD8', 'CD11c', 'CD45RO', 'CD3', 'IFNg', 'TCF1', 'CD14', 'CD56',
                   'PD1', 'CD45', 'PNAd', 'CD31')

cutoff <- 0.25
cutoffs <- as.data.frame(sapply(trYs, function(x) quantile(x, c(cutoff, 1-cutoff))))

#out <- NULL
#pairs <- combn(dim(cutoffs)[2], 2)
pos_out <- matrix(NA, nrow=dim(cutoffs)[2], ncol=dim(cutoffs)[2])
neg_out <- matrix(NA, nrow=dim(cutoffs)[2], ncol=dim(cutoffs)[2])
row.names(neg_out) <- colnames(neg_out) <- row.names(pos_out) <- colnames(pos_out) <- names(cutoffs)
for ( i in 1:dim(cutoffs)[2] ) {
  for ( j in 1:dim(cutoffs)[2] ) {
    
    x_name <- names(cutoffs)[i]
    y_name <- names(cutoffs)[j]
    
    score1 = sum((trYs[,x_name] < cutoffs[x_name][1,1]) | (trYs[,y_name] < cutoffs[y_name][1,1]))/dim(trYs)[1]
    neg_out[x_name, y_name] <- score1
    
    score2 = sum((trYs[,x_name] < cutoffs[x_name][1,1]) | (trYs[,y_name] > cutoffs[y_name][1,1]))/dim(trYs)[1]
    pos_out[x_name, y_name] <- score2
  }
}

png("~/ps_neg_50k.png")
ht <- Heatmap(neg_out, name = "Score", cluster_rows = TRUE)
draw(ht)
dev.off()

png("~/ps_pos_50k.png")
ht <- Heatmap(pos_out, name = "Score", cluster_rows = TRUE)
draw(ht)
dev.off()


## singlet label data (load labels, subset sampled data, correlation)
labels <- read.csv("/home/campbell/yulee/DAMM/new/res/eddy/50k/exp3/algo_label_nc23_iaFS_cf1_nm2_so0.0_lv1.0_rr1_r1.csv")
trYss <- trY[labels[,'damm_KM_tr_label'] != 'doublets',]

cutoff <- 0.25
cutoffs <- as.data.frame(sapply(trYss, function(x) quantile(x, c(cutoff, 1-cutoff))))

#out <- NULL
#pairs <- combn(dim(cutoffs)[2], 2)
pos_out <- matrix(NA, nrow=dim(cutoffs)[2], ncol=dim(cutoffs)[2])
neg_out <- matrix(NA, nrow=dim(cutoffs)[2], ncol=dim(cutoffs)[2])
row.names(neg_out) <- colnames(neg_out) <- row.names(pos_out) <- colnames(pos_out) <- names(cutoffs)
for ( i in 1:dim(cutoffs)[2] ) {
  for ( j in 1:dim(cutoffs)[2] ) {
    
    x_name <- names(cutoffs)[i]
    y_name <- names(cutoffs)[j]
    
    score1 = sum((trYss[,x_name] < cutoffs[x_name][1,1]) | (trYss[,y_name] < cutoffs[y_name][1,1]))/dim(trY)[1]
    neg_out[x_name, y_name] <- score1
    
    score2 = sum((trYss[,x_name] < cutoffs[x_name][1,1]) | (trYss[,y_name] > cutoffs[y_name][1,1]))/dim(trY)[1]
    pos_out[x_name, y_name] <- score2
  }
}

png("~/ps_neg_50ksinglet.png")
ht <- Heatmap(neg_out, name = "Score", cluster_rows = TRUE)
draw(ht)
dev.off()

png("~/ps_pos_50ksinglet.png")
ht <- Heatmap(pos_out, name = "Score", cluster_rows = TRUE)
draw(ht)
dev.off()

##
#rm(list=ls())
#library(ComplexHeatmap)

mat1 <- read.csv("/home/campbell/yulee/DAMM/new/res/eddy/50k/exp3/damm_cen_nc23_iaKM_cf1_nm2_so0.0_lv1.0_rr1_r1.csv")[,-1]
mat2 <- read.csv("/home/campbell/yulee/DAMM/new/res/eddy/50k/exp3/init_cen_nc23_iaPG_cf1_nm2_so0.0_lv1.0_rr1_r1.csv")[,-1]
mat3 <- read.csv("/home/campbell/yulee/DAMM/new/res/eddy/50k/exp3/init_cen_nc23_iaKM_cf1_nm2_so0.0_lv1.0_rr1_r1.csv")[,-1]
mat4 <- read.csv("/home/campbell/yulee/DAMM/new/res/eddy/50k/exp3/init_cen_nc23_iaGMM_cf1_nm2_so0.0_lv1.0_rr1_r1.csv")[,-1]
mat5 <- read.csv("/home/campbell/yulee/DAMM/new/res/eddy/50k/exp3/init_cen_nc23_iaFS_cf1_nm2_so0.0_lv1.0_rr1_r1.csv")[,-1]

nc = dim(mat1)[1]
p = dim(mat1)[2]
X <- cbind(trYs, label=labels[,'damm_KM_tr_label'])

cutoff = 0.25
cutoffs <- as.data.frame(sapply(trYs, function(x) quantile(x, c(cutoff, 1-cutoff))))

cen_means <- NULL #matrix(0, nc, p-1); colnames(cen_means) <- colnames(trY)
for ( k in 0:nc ) {
  
  idx <- X[,'label'] == (k+1)
  if (dim(X[idx,-25])[1] < 1) { 
    cen_means <- rbind(cen_means, matrix(mat1[k+1,-25], nrow=1))
    next 
  } else {
    cen_means <- rbind(cen_means, matrix(abs(mat1[k+1,-25] - colMeans(X[idx,-25])), nrow=1))
  }
  
  pos_out <- matrix(NA, nrow=dim(cutoffs)[2], ncol=dim(cutoffs)[2])
  neg_out <- matrix(NA, nrow=dim(cutoffs)[2], ncol=dim(cutoffs)[2])
  row.names(neg_out) <- colnames(neg_out) <- row.names(pos_out) <- colnames(pos_out) <- names(cutoffs)
  for ( i in 1:dim(cutoffs)[2] ) {
    for ( j in 1:dim(cutoffs)[2] ) {
      
      x_name <- names(cutoffs)[i]
      y_name <- names(cutoffs)[j]
      
      score1 = sum((X[idx,-25][,x_name] < cutoffs[x_name][1,1]) | (X[idx,-25][,y_name] < cutoffs[y_name][1,1]))/dim(X[idx,-25])[1]
      neg_out[x_name, y_name] <- score1
      
      score2 = sum((X[idx,-25][,x_name] < cutoffs[x_name][1,1]) | (X[idx,-25][,y_name] > cutoffs[y_name][1,1]))/dim(X[idx,-25])[1]
      pos_out[x_name, y_name] <- score2
    }
  }
  png(paste0("~/ps_neg_50ksinglet_nc", k, ".png"))
  ht <- Heatmap(neg_out, name = "Score", cluster_rows = TRUE)
  draw(ht)
  dev.off()
  
  png(paste0("~/ps_pos_50ksinglet_nc", k, ".png"))
  ht <- Heatmap(pos_out, name = "Score", cluster_rows = TRUE)
  draw(ht)
  dev.off()
}

out <- matrix(as.numeric(cen_means), nrow=24)
colnames(out) <- colnames(trYs)
row.names(out) <- 1:24

png(paste0("~/cen_diff.png"))
ht <- Heatmap(out[-24,], name = "Dif.", cluster_rows = FALSE)
draw(ht)
dev.off()