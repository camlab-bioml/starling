

rm(list=ls())

library(tidyr)
library(dplyr)
library(ggplot2)
library(glue)
library(rmote)
library(stringr)
start_rmote(port=4322)

## doublet performance
doublet_pref_table <- function() {
  
  fns <- list.files(pattern = paste0("doub_perf"))
  
  mat <- NULL
  for (i in 1:length(fns)) {
    
    parsed_fn <- strsplit(fns[i], "_")[[1]]
    nc <- substr(parsed_fn[3], 3, 100)
    ia <- substr(parsed_fn[4], 3, 100)
    cf <- substr(parsed_fn[5], 3, 100) 
    nm <- substr(parsed_fn[6], 3, 100)
    so <- substr(parsed_fn[7], 3, 100)
    lv <- substr(parsed_fn[8], 3, 100)
    cs <- substr(parsed_fn[9], 3, 100)
    rr <- substr(parsed_fn[10], 3, 100)
    r <- substr(parsed_fn[11], 2, 2)
    
    tmp <- t(as.matrix(read.csv(fns[i])[,2]))
    #mat <- rbind(mat, tmp)
    mat <- rbind(mat, c(nc, ia, cf, nm, so, lv, cs, rr, r, tmp[,10:16]))
    #mat <- rbind(mat, c(fns[i], tmp[,-c(1:8)]))
  }
  
  colnames(mat) <- c('nc', 'ia', 'cf', 'nm', 'so', 'lv', 'cs', 'rr', 'r', 'rf_f1', 'va_acc', 'va_f1', 'va_auroc', 'te_acc', 'te_f1', 'te_auroc')

  amat <- as.data.frame(mat[,1:13])
  
  res <- pivot_longer(amat, cols=10:13, names_to='metric', values_to ="value")
  out <- res %>% mutate(score = as.numeric(as.character(value)))
  
  return(as.data.frame(out[,-11]))
}

#dp_table <- doublet_pref_table()

####
####

new_corr_metric <- function(mat1, anti, pairs, thresholds) {
  
  #mat1=bmat
  #anti=1
  #pairs=anti_pairs
  
  out = NULL
  for ( i in 1:dim(pairs)[1] ) {
  
    #print(i)
    x_name = pairs[i,1]; y_name = pairs[i,2]
    x1 = mat1[x_name]; y1 = mat1[y_name]
    
    if (anti == 1) {
      ## case 1: negative
      score1 = sum((x1 < thresholds[x_name][1,1]) | (y1 < thresholds[y_name][1,1]))/dim(x1)[1]
      out <- rbind(out, cbind(paste0(x_name, '_', y_name), score1, -1))
    } else {
      ## case 2: positive
      score1 = sum((x1 < thresholds[x_name][1,1]) | (y1 > thresholds[y_name][2,1]))/dim(x1)[1]
      out <- rbind(out, cbind(paste0(x_name, '_', y_name), score1, 1))
    }
  }
  res <- as.data.frame(out); colnames(res) <- c('pair', 'metric', 'type')
  return(res)
}

correlation_table <- function() {
  
  #cohort = 'eddy'
  #ss = 10
  #expNum = 5
  
  if (cohort == 'eddy') {
    anti_pairs = rbind(
      c('CD3','CD20'), c('CD3','CD31'), c('CD3','CD68'), c('CD3','ECadherin'),
      c('CD4','CD8'), c('CD4','CD20'), c('CD4','CD31'), c('CD4','ECadherin'),
      c('CD8','CD20'), c('CD8','CD31'), c('CD8','CD68'), c('CD8','ECadherin'),
      c('CD20','CD31'), c('CD20','CD68'), c('CD20','ECadherin'),
      c('CD31','CD68'), c('CD31','ECadherin'), c('CD68','ECadherin'))
    
    co_pairs = rbind(
      c('CD3','CD4'), c('CD3','CD8'), c('CD45','CD3'), c('CD45','CD4'),
      c('CD45','CD8'), c('CD45','CD20'), c('CD45','CD68'))
    
  } else {
    anti_pairs = rbind(
      c('CD3', 'CD20'), c('CD3', 'CD68'), c('CD3', 'pCK'), c('CD3', 'CD31'),
      c('CD20', 'CD68'), c('CD20', 'pCK'), c('CD20', 'CD31'), c('CD45', 'pCK'), 
      c('pCK', 'Vimentin'))
    
    co_pairs = rbind(
      c('CD45', 'CD3'), c('CD45', 'CD20'), c('CD45', 'CD68'), c('pCK', 'CK5'),
      c('pCK', 'CK7'), c('pCK', 'CK8n18'), c('pCK', 'CK19'))
  }
  
  fnss <- list.files()
  fns <- fnss[grepl("init_cen_nc[0-9]*", fnss)]
  
  mat <- NULL
  for (i in 1:length(fns)) {
    
    print(i)
    parsed_fn <- strsplit(fns[i], "_")[[1]]
    trY <- read.csv(paste0("/home/campbell/yulee/DAMM/new/res/", cohort, "/", ss, "k/tr_ex2_", substr(parsed_fn[5], 3, 3), ".csv"))[,-1]
    thresholds <- as.data.frame(sapply(trY, function(x) quantile(x, c(0.25, 0.75))))
    colnames(thresholds)[which(colnames(thresholds) == 'panCK')] = 'pCK'
    colnames(thresholds)[which(colnames(thresholds) == 'CK8_18')] = 'CK8n18'
    
    bmat <- read.csv(paste0(fns[i]))[,-1]
    
    afn <- paste0("damm_cen_", paste0(parsed_fn[-c(1:2)], collapse = "_"))
    
    if ( file.exists(afn) == FALSE ) { next } 
    
    amat <- read.csv(afn)[,-1]
    
    bneg <- new_corr_metric(bmat, 1, anti_pairs, thresholds)
    bpos <- new_corr_metric(bmat, 0, co_pairs, thresholds)
    
    aneg <- new_corr_metric(amat, 1, anti_pairs, thresholds)
    apos <- new_corr_metric(amat, 0, co_pairs, thresholds)
    
    nc <- substr(parsed_fn[3], 3, 100)
    ia <- substr(parsed_fn[4], 3, 100)
    cf <- substr(parsed_fn[5], 3, 100) 
    nm <- substr(parsed_fn[6], 3, 100)
    so <- substr(parsed_fn[7], 3, 100)
    lv <- substr(parsed_fn[8], 3, 100)
    cs <- substr(parsed_fn[9], 3, 100)
    rr <- substr(parsed_fn[10], 3, 100)
    r <- substr(parsed_fn[11], 2, 2)
    
    rfn <- paste0("RF_cen_", paste0(parsed_fn[-c(1:2)], collapse = "_"))
    rmat <- read.csv(paste0(rfn))[,-1]
    
    rneg <- new_corr_metric(rmat, 1, anti_pairs, thresholds)
    rpos <- new_corr_metric(rmat, 0, co_pairs, thresholds)
    
    mat <- rbind(mat, cbind(nc, ia, cf, nm, so, lv, cs, rr, r, 
                            rbind(bneg, bpos)[,-3], after=rbind(aneg, apos)[,2], rf=rbind(rneg, rpos)[,2]))
  }
  as.data.frame(mat)
}

#cor_table <- correlation_table(ss, 'eddy', expNum)

####
####

## Pinch Effect
pinch_effect_table <- function() {
  
  fns <- list.files()
  label_fns <- fns[grepl("algo_label", fns)]
  
  mat <- NULL
  for (i in 1:length(label_fns)) {
    
    #i = 1
    print(i)
    
    parsed_fn <- strsplit(label_fns[i], "_")[[1]]
    df <- read.csv(label_fns[i])
    
    #ia <- substr(parsed_fn[4], 3, 100)
    
    tmp <- NULL
    for (nc in c('10','20','30')) {
        
      both_sets <- select(df, c(starts_with('area'), contains(nc)))
      
      for (ia in c('KM', 'GMM', 'FS')) {
        
        singlet_set <- filter(both_sets, !!sym(paste0('damm_', ia, '_label_nc', nc)) != 'doublets')[,1:2]
        doublet_set <- filter(both_sets, !!sym(paste0('damm_', ia, '_label_nc', nc)) == 'doublets')[,1:2]
        
        if (dim(doublet_set)[1] > 1) {
          df1 <- singlet_set['area_convex']/singlet_set['area']
          df2 <- doublet_set['area_convex']/doublet_set['area']
          
          ttres <- t.test(df1, df2)
          tmp <- rbind(tmp, cbind(ia=ia, nc=nc, Size=ttres$estimate[2] - ttres$estimate[1], Pval=ttres$p.value))
        } else {
          tmp <- rbind(tmp, cbind(ia=ia, nc=nc, Size=NA, Pval=NA))
        }
      }
    }
    
    cf <- substr(parsed_fn[3], 3, 100) 
    nm <- substr(parsed_fn[4], 3, 100) 
    so <- substr(parsed_fn[5], 3, 100)
    lv <- substr(parsed_fn[6], 3, 100)
    cs <- substr(parsed_fn[7], 3, 100)
    rr <- substr(parsed_fn[8], 3, 100)
    r <- substr(parsed_fn[9], 2, 2)
    
    mat <- rbind(mat, cbind(cf, nm, so, lv, cs, rr, r, tmp))
  }
  as.data.frame(mat)
}

##
##

expNum <- 1
ss <- 10
cohorts <- c('meta', 'basel', 'eddy')

for (cohort in cohorts) {
  
  cohort = 'eddy'
  print(cohort)
  setwd(paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/'))

  #cor_table <- correlation_table()
  #write.csv(cor_table, file = paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/cor_df.csv')) 
  
  #dp_table <- doublet_pref_table()
  #write.csv(dp_table, file = paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/perf_df.csv')) 
  
  #pe_table <- pinch_effect_table()
  #write.csv(pe_table, file = paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/pinch_df.csv')) 
}

##

rm(list=ls())

expNum <- 1
ss <- 10
cohort <- 'eddy'

## plausibility score
correlation <- read.csv(paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/cor_df.csv'))[,-1]
correlation[,'Cluster'] <- as.factor(correlation[,'nc'])
correlation[,'Initial'] <- as.factor(correlation[,'ia'])
correlation[,'Cofactor'] <- as.factor(correlation[,'cf'])

correlation[,'Dist'] <- as.factor(correlation[,'nm'])
correlation[,'so'] <- as.factor(correlation[,'so'])
correlation[,'Lambda'] <- as.factor(correlation[,'lv'])
correlation[,'Cell'] <- as.factor(correlation[,'cs'])
correlation[,'Relax'] <- as.factor(correlation[,'rr'])
correlation[,'Run'] <- as.factor(correlation[,'r'])
str(correlation)

correlation[,'Cell'] <- ifelse(correlation[,'Cell'] == 1, 'Yes', 'No')
correlation[,'Relax'] <- ifelse(correlation[,'Relax'] == 1, 'Yes', 'No')

df11 <- correlation %>% filter(Dist == 2) %>% group_by(Cluster, Initial, Cofactor, Lambda, Cell, Relax, Run) %>% 
  summarise('At initialization'=sum(metric), 'After training'=sum(after), 'Two step'=sum(rf), .groups = 'drop') %>%
  pivot_longer(cols = 8:10, names_to = 'Metric', values_to = 'score') %>%
  group_by(Initial, Cofactor, Lambda, Cell, Relax, Metric) %>% summarise(Correlation=mean(score), .groups = 'drop')

df1 <- df11 %>% filter(Metric == "At initialization") 
df2 <- df11 %>% filter(Metric == "After training")
df3 <- df11 %>% filter(Metric == "Two step")

p1 <- ggplot(df2, aes(x = Initial, y = Correlation, shape=Metric)) +
  #geom_point() +
  stat_summary(fun=mean, geom="line", aes(group = Lambda, col = Lambda)) +
  stat_summary(fun=mean, geom="point",  aes(group = Lambda, col = Lambda))

p2 <- p1 + geom_smooth(data=df1, aes(x = Initial, y = Correlation)) + 
  stat_summary(data=df1, fun=mean, geom="line", aes(group = 1), linetype = "dotted") + #linetype = 2,
  stat_summary(data=df1, fun=mean, geom="point")

p3 <- p2 + geom_smooth(data=df3, aes(x = Initial, y = Correlation)) + 
  stat_summary(data=df3, fun=mean, geom="line", aes(group = 1), linetype = "dashed") + #linetype = 3,
  stat_summary(data=df3, fun=mean, geom="point")

p3 + facet_grid(glue("CF: {Cofactor}")~glue("'Model cell size': {Cell}")+glue("'Model cell overlap': {Relax}"), labeller = label_parsed) + 
  labs(x = "Cluster initialization algorithm", y = "Cluster plausibility score") +
  #ylim(8, 16) + 
  scale_colour_brewer(palette = "Set2") +
  cowplot::theme_cowplot() +
  theme(legend.position="bottom", 
        #legend.text=element_text(size=8),
        #legend.title = element_text(size=10), #element_blank(),
        #axis.text.x = element_text(size = 10),
        strip.background = element_rect(fill='white'), 
        strip.text = element_text(face='bold'))
ggsave(paste0("/home/campbell/yulee/DAMM/new/res/", cohort, "/", ss, "k/exp", expNum, "/plot/ps.pdf"), height = 10, width = 10)

##

rm(list=ls())
ss = 10
expNum = 1
cohort <- 'eddy'

doublet_perf <- read.csv(paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/perf_df.csv'))[,-1]
doublet_perf[,'Cluster'] <- as.factor(doublet_perf[,'nc'])
doublet_perf[,'Initial'] <- as.factor(doublet_perf[,'ia'])
doublet_perf[,'Cofactor'] <- as.numeric(doublet_perf[,'cf'])

doublet_perf[,'Dist'] <- as.factor(doublet_perf[,'nm'])
doublet_perf[,'so'] <- as.factor(doublet_perf[,'so'])
doublet_perf[,'Lambda'] <- as.factor(doublet_perf[,'lv'])
doublet_perf[,'Cell'] <- as.factor(doublet_perf[,'cs'])
doublet_perf[,'Relax'] <- as.factor(doublet_perf[,'rr'])
doublet_perf[,'Run'] <- as.factor(doublet_perf[,'r'])
str(doublet_perf)

doublet_perf[,'Metric'] <- doublet_perf[,'metric']
doublet_perf[,'Metric'] <- as.character(doublet_perf[,'Metric'])
#doublet_perf[,'Metric'][doublet_perf[,'Metric'] == 'va_acc'] = "Accuracy"
#doublet_perf[,'Metric'][doublet_perf[,'Metric'] == 'va_auroc'] = "ROC"
doublet_perf[,'Metric'][doublet_perf[,'Metric'] == 'va_f1'] = "Starling"
doublet_perf[,'Metric'][doublet_perf[,'Metric'] == 'rf_f1'] = "Random Forest"

df21 <- doublet_perf %>% filter(Dist == 2 & (Metric == "Starling" | Metric == "Random Forest")) %>%
  group_by(Cluster, Initial, Cofactor, Lambda, Cell, Relax, Metric) %>% 
  summarise(F1=mean(score), .groups = 'drop')

df1 <- df21 %>% filter(Metric == "Starling")
df2 <- df21 %>% filter(Metric == "Random Forest") 

p1 <- ggplot(df1, aes(x = Initial, y = F1, shape=Metric)) +
  #geom_point() +
  stat_summary(fun=mean, geom="line", aes(group = Lambda, col = Lambda)) +
  stat_summary(fun=mean, geom="point",  aes(group = Lambda, col = Lambda))

p2 <- p1 + geom_smooth(data=df2, aes(x = Initial, y = F1)) + 
  stat_summary(data=df2, fun=mean, geom="line", linetype = 2, aes(group = 1)) +
  stat_summary(data=df2, fun=mean, geom="point")

p2 + facet_grid(glue("CF: {Cofactor}")~glue("Cell: {Cell}")+glue("RR: {Relax}"), labeller = label_parsed) + 
  labs(x = "Cluster initialization algorithm", y = "Cluster plausibility score") +
  #ylim(8, 16) + 
  scale_colour_brewer(palette = "Set2") +
  cowplot::theme_cowplot() +
  theme(legend.position="bottom", 
        strip.background = element_rect(fill='white'), 
        strip.text = element_text(face='bold'))
ggsave(paste0("/home/campbell/yulee/DAMM/new/res/", cohort, "/", ss, "k/exp", expNum, "/plot/sp.png"), height = 8, width = 10)

## pinch effect scores

cohort <- 'eddy'

pinch_effect <- read.csv(paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/pinch_df.csv'))[,-1]
pinch_effect[,'Cluster'] <- as.factor(pinch_effect[,'nc'])
pinch_effect[,'Initial'] <- as.factor(pinch_effect[,'ia'])
pinch_effect[,'Cofactor'] <- as.factor(pinch_effect[,'cf'])

pinch_effect[,'Dist'] <- as.factor(pinch_effect[,'nm'])
pinch_effect[,'so'] <- as.factor(pinch_effect[,'so'])
pinch_effect[,'Lambda'] <- as.factor(pinch_effect[,'lv'])
pinch_effect[,'Cell'] <- as.factor(pinch_effect[,'cs'])
pinch_effect[,'Relax'] <- as.factor(pinch_effect[,'rr'])
pinch_effect[,'Run'] <- as.factor(pinch_effect[,'r'])
str(pinch_effect)

pinch_effect %>% filter(Dist == 2) %>%
  ggplot(aes(x = Initial, y = Size, col = Run)) + 
  #geom_boxplot() +
  #facet_grid(Lambda~Cofactor, labeller = label_both) +
  facet_grid(glue('lambda*": {Lambda}"') + glue("Cell: {Cell}") ~ glue("CF: {Cofactor}") + glue("RR: {Relax}"), labeller = label_parsed) +
  cowplot::theme_cowplot(font_size = 12) +
  theme(legend.position="bottom", 
        axis.text.x = element_text(size = 5),
        strip.background = element_rect(fill='white'), 
        strip.text = element_text(face='bold')) + 
  #guides(fill=guide_legend()) +
  labs(x = "Cluster initialization algorithm", y = "Difference in pinch effect ratios") +
  #scale_color_discrete(name="Run") +
  stat_summary(fun=median, geom="line", aes(group = 1)) +
  stat_summary(fun=median, geom="point", aes(group = 1))
#ggsave(paste0("/home/campbell/yulee/DAMM/new/res/", cohort, "/", ss, "k/exp", expNum, "/plot/msr.pdf"))

##

df1 <- pinch_effect %>% filter(Dist == 2) %>% 
  group_by(Initial, Cofactor, Lambda, Cell, Relax) %>% 
  summarise(Size=mean(Size), .groups = 'drop') %>%
  pivot_longer(cols = 6, names_to = 'Metric', values_to = 'Size')

p1 <- ggplot(df1, aes(x = Initial, y = Size, shape=Metric)) +
  stat_summary(fun=mean, geom="line", aes(group = Lambda, col = Lambda)) +
  stat_summary(fun=mean, geom="point",  aes(group = Lambda, col = Lambda))

p1 + facet_grid(glue("CF: {Cofactor}")~glue("Cell: {Cell}")+glue("RR: {Relax}"), labeller = label_parsed) + 
  labs(x = "Cluster initialization algorithm", y = "Size") +
  scale_colour_brewer(palette = "Set2") +
  cowplot::theme_cowplot() +
  theme(legend.position="bottom", 
        #legend.text=element_text(size=8),
        #legend.title = element_text(size=10), #element_blank(),
        #axis.text.x = element_text(size = 10),
        strip.background = element_rect(fill='white'), 
        strip.text = element_text(face='bold'))

df1 <- pinch_effect %>% filter(Dist == 2) %>% 
  group_by(Initial, Cofactor, Lambda, Cell, Relax) %>% 
  summarise(Pval=-log10(mean(Pval)), .groups = 'drop') %>%
  pivot_longer(cols = 6, names_to = 'Metric', values_to = 'Pval')

p1 <- ggplot(df1, aes(x = Initial, y = Pval, shape=Metric)) +
  stat_summary(fun=mean, geom="line", aes(group = Lambda, col = Lambda)) +
  stat_summary(fun=mean, geom="point",  aes(group = Lambda, col = Lambda))

p1 + facet_grid(glue("CF: {Cofactor}")~glue("Cell: {Cell}")+glue("RR: {Relax}"), labeller = label_parsed) + 
  labs(x = "Cluster initialization algorithm", y = "-log(P-value)") +
  scale_colour_brewer(palette = "Set2") +
  cowplot::theme_cowplot() +
  theme(legend.position="bottom", 
        #legend.text=element_text(size=8),
        #legend.title = element_text(size=10), #element_blank(),
        #axis.text.x = element_text(size = 10),
        strip.background = element_rect(fill='white'), 
        strip.text = element_text(face='bold'))

###
###

rm(list=ls())
library(grid)
library(circlize)
library(ComplexHeatmap)
## heatmap (centroid) plots (first select the best parameter combinations)

minmax <- function(x, na.rm = TRUE) {
  return((x- min(x, na.rm = TRUE)) /(max(x, na.rm = TRUE)-min(x, na.rm = TRUE)))
}

#nc10_iaKM_cf1_nm2_so0_lv1.0_cs1_rr1
#basel
#1.462073137	0.530743118	1.992816255	0.900816321	0.81190002	0.813633204
#1.690784315	0.557678087	2.248462401	0.89338851	0.804099977	0.805674076
#1.750304603	0.561631312	2.311935915	0.893593431	0.808899999	0.810848296
#1.364079383	0.568428644	1.932508027	0.895699263	0.810199976	0.811444521 x
#1.738019682	0.546489685	2.284509366	0.896685898	0.815500021	0.817921638

#meta
#-2.05039642	0.638998331	-1.411398089	0.880679131	0.794200003	0.792707503
#-1.60798559	0.575894836	-1.032090754	0.888708115	0.798200011	0.799602807
#-2.054848056	0.688404965	-1.366443092	0.87244612	0.782299995	0.778422415 x
#-1.832388829	0.601456061	-1.230932768	0.882147074	0.787100017	0.787588596
#-1.962541058	0.561629015	-1.400912043	0.891438842	0.807399988	0.809495568

#tonsil
#11.95134674	0.358009164	12.3093559	0.937784195	0.860300004	0.864958882 x
#12.10586989	0.387961405	12.49383129	0.930640578	0.852800012	0.8573367
#12.1669316	0.377040669	12.54397227	0.933645844	0.856400013	0.861683667
#12.11411012	0.390830605	12.50494072	0.928976297	0.846700013	0.851439059
#12.14935268	0.359469593	12.50882228	0.938842118	0.857900023	0.863352299

#nc20_iaKM_cf1_nm2_so0_lv1.0_cs1_rr1
#basel
-0.630502007					-0.115380925				 -0.28635482	0.920467615	0.840399981	0.842479348 x
-0.149305377					 0.496457058					0.00013957	0.908798218	0.830799997	0.827628374
 0.168439646					 0.354483298					0.285509843	0.903936923	0.823300004	0.822143912
-0.434767426					-0.017756318					0.232618076	0.929226816	0.850600004	0.854016006
-0.16792544						 0.007898832					0.056755999	0.918807864	0.837599993	0.840126038

#meta
-3.497470935						-3.089899683							-3.260986515	0.901586473	0.819599986	0.820425987
-3.664927194						-3.251141749							-3.194869056	0.905677319	0.824800014	0.824729919
-3.084799002						-3.326942449							-3.194084912	0.908721268	0.828599989	0.829078615
-3.151222432						-3.053953564							-3.054681638	0.892396748	0.811399996	0.807196856
-4.178493403						-3.54366335							  -3.419646895	0.912814319	0.83160001	0.833399355 x 

#tonsil
10.80132609						11.00247948							10.98683789	0.939907789	0.863699973	0.868143559
10.79699096						10.90530822							10.91201623	0.945551038	0.867699981	0.872506559
10.78398974						10.95393944							10.93468284	0.943617165	0.872399986	0.87714231
10.79150312						10.89346266							10.89107676	0.944509864	0.872900009	0.876205385
10.87410452						10.85604376							10.94543424	0.941938102	0.866100013	0.871311903 x

10.83475769						10.97867085							10.92863787	0.947201669	0.872799993	0.87762171
10.75254297						11.02599742							10.89455103	0.943900704	0.874000013	0.877288699
10.73408206						10.90747665							10.918797	0.947680175	0.871399999	0.875892639
10.81154822						10.84735878							11.0083671	0.944301963	0.869700015	0.875059903
10.94475963						11.01680997							10.99406784	0.949287474	0.879000008	0.883136928

#nc30_iaKM_cf1_nm2_so0_lv1.0_cs1_rr1
#basel
#-1.407376173	0.505391965	-0.901984209	0.912946403	0.829400003	0.827012777
#-1.391678209	0.556514159	-0.83516405	0.898603678	0.814100027	0.809937656
#-1.551009088	0.525980479	-1.02502861	0.905670345	0.825100005	0.823351145 x
#-1.524879111	0.501706627	-1.023172484	0.912433803	0.834999979	0.834170878
#-1.473636828	0.463854166	-1.009782662	0.919368684	0.838500023	0.838224947

#meta
-4.591534293	0.5075931	-4.083941193	0.90534246	0.824400008	0.824715555
-4.477242535	0.523269753	-3.953972782	0.907860339	0.832300007	0.831372499
-4.710202196	0.493806276	-4.21639592	0.91196537	0.837899983	0.839330018
-4.6266828	0.507807865	-4.118874935	0.90850085	0.831499994	0.83084029
-4.721233548	0.518354695	-4.202878853	0.909644127	0.832599998	0.832633436 x

#tonsil
9.949470611	0.33631812	10.28578873	0.943760812	0.872399986	0.876212656
9.911467128	0.316159116	10.22762624	0.949375272	0.875400007	0.878911614
9.90198236	0.337836843	10.2398192	0.944500923	0.875199974	0.878646374
9.889051931	0.314876603	10.20392853	0.94931227	0.882799983	0.886125147 x
9.902167738	0.337651787	10.23981952	0.9441486	0.874800026	0.87919724

10.76608273						10.99659814							10.91911252	0.939383566	0.864600003	0.868823886
10.93501856						10.85564024							11.05928251	0.94535929	0.873600006	0.877376735
10.870683015430900		10.962546768013400			11.031673893356100	0.9454889297485350	0.8715000152587890	0.8756651878356930

run <- 2
nc <- 20
ss <- 10
expNum <- 1
cohort <- 'eddy'

mlabels <- c("KM_label", "GMM_label", "FS_label", "damm_KM_label_nc")
path <- paste0("/home/campbell/yulee/DAMM/new/res/", cohort, "/", ss, "k/exp", expNum, "/")
selected_parameters = "_cf1_nm2_so0_lv1.0_cs1_rr1_r"

samp_df <- read.csv(paste0(path, "fs_nc30_iaFS", selected_parameters, run, ".csv"))
samp_lb <- read.csv(paste0(path, "algo_label",  selected_parameters, run, ".csv"))[,-1]

m1 <- m2 <- m3 <- NULL
count <- NULL
for (c in 0:(nc-1)) {
  
  not_in_n_are_in  <- samp_lb[,paste0("KM_label_nc", nc)] != c & samp_lb[,paste0("damm_KM_label_nc", nc)] == c ## werent in & are in
  were_in_n_are_in <- samp_lb[,paste0("KM_label_nc", nc)] == c & samp_lb[,paste0("damm_KM_label_nc", nc)] == c ## were in & are in
  were_in_n_not_in <- samp_lb[,paste0("KM_label_nc", nc)] == c & samp_lb[,paste0("damm_KM_label_nc", nc)] != c ## were in & not in
  
  #m1 <- rbind(m1, apply(samp_df[which(not_in_n_are_in == 1),-1], 2, mean))
  #m2 <- rbind(m2, apply(samp_df[which(were_in_n_are_in == 1),-1], 2, mean))
  #m3 <- rbind(m3, apply(samp_df[which(were_in_n_not_in == 1),-1], 2, mean))
  count <- rbind(count, c(sum(not_in_n_are_in), sum(were_in_n_are_in), sum(were_in_n_not_in), 
                          sum(samp_lb[,paste0("KM_label_nc", nc)] == c),
                          sum(samp_lb[,paste0("damm_KM_label_nc", nc)] == c),
                          sum(samp_lb[,paste0("RF_KM_label_nc", nc)] == c)))
}

m4 <- read.csv(paste0(path, "init_cen_nc", nc, "_iaKM", selected_parameters, run, ".csv"))[,-1]
m5 <- read.csv(paste0(path, "damm_cen_nc", nc, "_iaKM", selected_parameters, run, ".csv"))[,-1]
m6 <- read.csv(paste0(path, "RF_cen_nc", nc, "_iaKM", selected_parameters, run, ".csv"))[,-1]

dmat <- apply(rbind(m4, m5, m6), 2, minmax)

i=0
ma1 <- as.matrix(dmat[(i*nc+1):((i+1)*nc),])
rownames(ma1) <- 1:nc
haa1 = rowAnnotation(" " = anno_barplot(as.integer(count[,4]), axis = TRUE, ylim = c(0,1000)))
htt1 <- Heatmap(ma1, right_annotation = haa1, column_title = "Initialization (Kmeans)", column_title_side = 'top', #column_names_rot = 0,
               cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE)

i=1
ma2 <- as.matrix(dmat[(i*nc+1):((i+1)*nc),])
rownames(ma2) <- 1:nc
haa2 = rowAnnotation(" " = anno_barplot(as.integer(count[,5]), axis = TRUE, ylim = c(0,1000)))
htt2 <- Heatmap(ma2, right_annotation = haa2, column_title = "STARLING", column_title_side = 'top',
               cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE)

i=2
ma3 <- as.matrix(dmat[(i*nc+1):((i+1)*nc),])
rownames(ma3) <- 1:nc
haa3 = rowAnnotation(" " = anno_barplot(as.integer(count[,6]), axis = TRUE, ylim = c(0,1000)))
htt3 <- Heatmap(ma3, right_annotation = haa3, column_title = "Kmeans after remving doublets via RF", column_title_side = 'top',
                cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE)

col_fun = colorRamp2(c(0, 0.5, 1), c("blue", "white", "red"))
lgd = Legend(col_fun = col_fun, title = "Exp.")

pdf(paste0(path, "plot/", "starling_", nc, "_", run, ".pdf"), width=18, height=6)
draw(htt3 + htt1 + htt2, annotation_legend_list = lgd, row_title = "Cluster", column_title = "Marker", column_title_side = 'bottom')
dev.off()

###

#mat <- apply(rbind(m1, m2, m3, as.matrix(m4)), 2, minmax)
mat <- rbind(m1, m2, m3)
mat[is.nan(mat)] <- NA
mat <- apply(mat, 2, minmax)
#mat <- apply(rbind(m1, m2, m3), 2, minmax)
colnames(mat) <- colnames(m4)

i=0
mat1 <- as.matrix(mat[(i*nc+1):((i+1)*nc),])
rownames(mat1) <- 1:nc
#ha1 = rowAnnotation(" " = anno_barplot(as.integer(count[,1]), axis = TRUE, ylim = c(0,750),
#                                       axis_param = list(at = c(0,250,500,750), labels = c(0,250,500,750))))
ht1 <- Heatmap(mat1, column_title = "weren't in & are in", column_title_side = 'top', #column_names_rot = 0,
               cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE)

i=1
mat2 <- as.matrix(mat[(i*nc+1):((i+1)*nc),])
rownames(mat2) <- 1:nc
ht2 <- Heatmap(mat2, column_title = "were in & are in", column_title_side = 'top',
               cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE)

i=2
mat3 <- as.matrix(mat[(i*nc+1):((i+1)*nc),])
rownames(mat3) <- 1:nc
ha3 = rowAnnotation("Count" = anno_barplot(count[,1:3], axis = TRUE, gp = gpar(fill = 2:4)))
ht3 <- Heatmap(mat3, right_annotation = ha3, column_title = "were in & not in", column_title_side = 'top',
               cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE)

ht = ht1 + ht2 + ht3

lgd_c = Legend(title = 'Condition', labels = c("weren't in & are in", "were in & are in", "were in & not in"), legend_gp = gpar(fill = 2:4))
pdf(paste0(path, "plot/", "cluster_", nc, "_", run, ".pdf"), width=24, height=6)
draw(ht, annotation_legend_list = list(lgd_c, lgd), row_title = "Cluster", column_title = "Marker", column_title_side = 'bottom')
dev.off()

##

ht1 <- Heatmap(cor(t(mat1), t(mat2)), cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE,
               column_title = "(weren't in & are in) vs (were in & are in)", column_title_side = 'top')
ht2 <- Heatmap(cor(t(mat1), t(mat3)), cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE,
               column_title = "(weren't in & are in) vs (were in & not in)", column_title_side = 'top')
ht3 <- Heatmap(cor(t(mat2), t(mat3)), cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE,
               column_title = "(were in & are in) vs (were in & not in)", column_title_side = 'top')
ht = ht1 + ht2 + ht3

col_fun = colorRamp2(c(-1, 0, 1), c("blue", "white", "red"))
lgd = Legend(col_fun = col_fun, title = "Cor.")

pdf(paste0(path, "plot/", "cluster_cor_", nc, "_", run, ".pdf"), width=24, height=6)
draw(ht, annotation_legend_list = lgd, column_title = "Cluster Correlation", column_title_side = 'bottom')
dev.off()

ht1 <- Heatmap(cor(mat1, mat2, use = "na.or.complete"), cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE,
               column_title = "(weren't in & are in) vs (were in & are in)", column_title_side = 'top')
ht2 <- Heatmap(cor(mat1, mat3, use = "na.or.complete"), cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE,
               column_title = "(weren't in & are in) vs (were in & not in)", column_title_side = 'top')
ht3 <- Heatmap(cor(mat2, mat3, use = "na.or.complete"), cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE,
               column_title = "(were in & are in) vs (were in & not in)", column_title_side = 'top')
ht = ht1 + ht2 + ht3

pdf(paste0(path, "plot/", "marker_cor_", nc, "_", run, ".pdf"), width=24, height=10)
draw(ht, annotation_legend_list = lgd, column_title = "Marker Correlation", column_title_side = 'bottom')
dev.off()