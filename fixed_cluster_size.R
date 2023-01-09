rm(list=ls())

library(tidyr)
library(dplyr)
library(ggplot2)
library(glue)
library(rmote)
library(stringr)

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
    mat <- rbind(mat, c(nc, ia, cf, nm, so, lv, cs, rr, r, tmp[,10:15]))
    #mat <- rbind(mat, c(fns[i], tmp[,-c(1:8)]))
  }
  
  colnames(mat) <- c('nc', 'ia', 'cf', 'nm', 'so', 'lv', 'cs', 'rr', 'r', 'va_acc', 'va_f1', 'va_auroc', 'te_acc', 'te_f1', 'te_auroc')

  amat <- as.data.frame(mat[,1:12])
  
  res <- pivot_longer(amat, cols=10:12, names_to='metric', values_to ="value")
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
    
    mat <- rbind(mat, cbind(nc, ia, cf, nm, so, lv, cs, rr, r, rbind(bneg, bpos)[,-3], after=rbind(aneg, apos)[,2]))
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
    
    #i = 1132
    print(i)
    
    parsed_fn <- strsplit(label_fns[i], "_")[[1]]
    df <- read.csv(label_fns[i])
    
    ia <- substr(parsed_fn[4], 3, 100)
    
    tmp <- NULL
    for (nc in c('10','20','30')) {
        
      both_sets <- select(df, c(starts_with('area'), contains(nc)))
      singlet_set <- filter(both_sets, !!sym(paste0('damm_', ia, '_nc', nc, '_tr_label')) != 'doublets')[,1:2]
      doublet_set <- filter(both_sets, !!sym(paste0('damm_', ia, '_nc', nc, '_tr_label')) == 'doublets')[,1:2]
        
      #singlet_set <- filter(df, !!sym(paste0('damm_', methods[j], '_label_nc', nc)) != 'doublets')
      #doublet_set <- filter(df, !!sym(paste0('damm_', methods[j], '_label_nc', nc)) == 'doublets')
      
      if (dim(doublet_set)[1] > 1) {
        df1 <- singlet_set['area_convex']/singlet_set['area']
        df2 <- doublet_set['area_convex']/doublet_set['area']
        
        ttres <- t.test(df1, df2)
        tmp <- rbind(tmp, cbind(ia=ia, nc=nc, Size=ttres$estimate[2] - ttres$estimate[1], Pval=ttres$p.value))
      } else {
        tmp <- rbind(tmp, cbind(ia=ia, nc=nc, Size=NA, Pval=NA))
      }
    }
    
    cf <- substr(parsed_fn[5], 3, 100) 
    nm <- substr(parsed_fn[6], 3, 100) 
    so <- substr(parsed_fn[7], 3, 100)
    lv <- substr(parsed_fn[8], 3, 100)
    cs <- substr(parsed_fn[9], 3, 100)
    rr <- substr(parsed_fn[10], 3, 100)
    r <- substr(parsed_fn[11], 2, 2)
    
    mat <- rbind(mat, cbind(cf, nm, so, lv, cs, rr, r, tmp))
  }
  as.data.frame(mat)
}


##
##

expNum <- 5
ss <- 10
cohorts <- c('meta', 'basel', 'eddy')

for (cohort in cohorts) {
  
  print(cohort)
  setwd(paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/'))

  #cor_table <- correlation_table()
  #write.csv(cor_table, file = paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/correlation_table.csv')) 
  
  dp_table <- doublet_pref_table()
  write.csv(dp_table, file = paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/doublet_perf_table.csv')) 
  
  #pe_table <- pinch_effect_table()
  #write.csv(pe_table, file = paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/pinch_effect_table.csv')) 
}

##

rm(list=ls())

expNum <- 5
ss <- 10
cohort <- 'eddy'

## plausibility score
correlation <- read.csv(paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/correlation_table.csv'))[,-1]
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

df11 <- correlation %>% filter(Dist == 2) %>% group_by(Cluster, Initial, Cofactor, Lambda, Cell, Relax, Run) %>% 
  summarise('At initialization'=sum(metric), 'After training'=sum(after), .groups = 'drop') %>%
  pivot_longer(cols = 8:9, names_to = 'Metric', values_to = 'score') %>%
  group_by(Initial, Cofactor, Lambda, Cell, Relax, Metric) %>% summarise(Correlation=mean(score), .groups = 'drop')

df1 <- df11 %>% filter(Metric == "At initialization") 
df2 <- df11 %>% filter(Metric == "After training") 

p1 <- ggplot(df2, aes(x = Initial, y = Correlation, shape=Metric)) +
  #geom_point() +
  stat_summary(fun=mean, geom="line", aes(group = Lambda, col = Lambda)) +
  stat_summary(fun=mean, geom="point",  aes(group = Lambda, col = Lambda))

p2 <- p1 + geom_smooth(data=df1, aes(x = Initial, y = Correlation)) + 
  stat_summary(data=df1, fun=mean, geom="line", linetype = 2, aes(group = 1)) +
  stat_summary(data=df1, fun=mean, geom="point", aes(group = 1))

p2 + facet_grid(glue("CF: {Cofactor}")~glue("Cell: {Cell}")+glue("RR: {Relax}"), labeller = label_parsed) + 
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
#ggsave(paste0("/home/campbell/yulee/DAMM/new/res/", cohort, "/", ss, "k/exp", expNum, "/plot/ps2.png"), height = 4, width = 6)

##

cohort <- 'meta'

doublet_perf <- read.csv(paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/doublet_perf_table.csv'))[,-1]
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
doublet_perf[,'Metric'][doublet_perf[,'Metric'] == 'va_acc'] = "Accuracy"
doublet_perf[,'Metric'][doublet_perf[,'Metric'] == 'va_auroc'] = "ROC"
doublet_perf[,'Metric'][doublet_perf[,'Metric'] == 'va_f1'] = "F1"

df1 <- doublet_perf %>% filter(Dist == 2 & Metric == "F1") %>% 
  group_by(Initial, Cofactor, Lambda, Cell, Relax) %>% 
  summarise(F1=mean(score), .groups = 'drop') %>%
  pivot_longer(cols = 6, names_to = 'Metric', values_to = 'score')

p1 <- ggplot(df1, aes(x = Initial, y = score, shape=Metric)) +
  stat_summary(fun=mean, geom="line", aes(group = Lambda, col = Lambda)) +
  stat_summary(fun=mean, geom="point",  aes(group = Lambda, col = Lambda))

p1 + facet_grid(glue("CF: {Cofactor}")~glue("Cell: {Cell}")+glue("RR: {Relax}"), labeller = label_parsed) + 
  labs(x = "Cluster initialization algorithm", y = "F1") +
  scale_colour_brewer(palette = "Set2") +
  #ylim(0.3, 1.0) + 
  cowplot::theme_cowplot() +
  theme(legend.position="bottom", 
        #legend.text=element_text(size=8),
        #legend.title = element_text(size=10), #element_blank(),
        #axis.text.x = element_text(size = 10),
        strip.background = element_rect(fill='white'), 
        strip.text = element_text(face='bold'))
#ggsave(paste0("/home/campbell/yulee/DAMM/new/res/", cohort, "/", ss, "k/exp", expNum, "/plot/sp.pdf"))

## pinch effect scores

cohort <- 'eddy'

pinch_effect <- read.csv(paste0('/home/campbell/yulee/DAMM/new/res/', cohort, '/', ss, 'k/exp', expNum, '/pinch_effect_table.csv'))[,-1]
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
ggsave(paste0("/home/campbell/yulee/DAMM/new/res/", cohort, "/", ss, "k/exp", expNum, "/plot/msr.pdf"))

##

df1 <- pinch_effect %>% filter(Dist == 0) %>% 
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

df1 <- pinch_effect %>% filter(Dist == 0) %>% 
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
  return((x- min(x)) /(max(x)-min(x)))
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
#-0.43357142	0.435920918	0.002349498	0.923088133	0.845399976	0.848905385
#-0.388634348	0.446675759	0.058041411	0.920670569	0.844799995	0.847574115
#-0.554700572	0.479093109	-0.075607463	0.914070129	0.832799971	0.834028184
#-0.608186812	0.431051587	-0.177135225	0.924441457	0.846400023	0.84944129 x
#-0.387226355	0.461259403	0.074033048	0.918630242	0.842599988	0.844065726

#meta
#-4.131905567	0.482602064	-3.649303503	0.914368451	0.831399977	0.832038224 x
#-3.443263944	0.509215389	-2.934048555	0.90486002	0.819000006	0.817724049
#-4.016065033	0.457442302	-3.558622731	0.918577731	0.840900004	0.842272222
#-3.872294279	0.53099159	-3.34130269	0.904693305	0.826499999	0.825960457
#-3.982537153	0.535045834	-3.447491319	0.901901364	0.822099984	0.821797073

#tonsil
#10.49585312	0.333840972	10.82969409	0.944893599	0.875199974	0.878386319 x
#10.74073744	0.355315609	11.09605305	0.938229144	0.866100013	0.869480491
#10.71948701	0.342754366	11.06224138	0.942935586	0.874199986	0.879270613
#10.66827195	0.321608856	10.9898808	0.947731316	0.872600019	0.876741409
#10.66147777	0.322573169	10.98405094	0.948491812	0.87470001	0.878385007

#nc30_iaKM_cf1_nm2_so0_lv1.0_cs1_rr1
#basel
#-1.407376173	0.505391965	-0.901984209	0.912946403	0.829400003	0.827012777
#-1.391678209	0.556514159	-0.83516405	0.898603678	0.814100027	0.809937656
#-1.551009088	0.525980479	-1.02502861	0.905670345	0.825100005	0.823351145 x
#-1.524879111	0.501706627	-1.023172484	0.912433803	0.834999979	0.834170878
#-1.473636828	0.463854166	-1.009782662	0.919368684	0.838500023	0.838224947

#meta
#-4.5759756	0.541377511	-4.034598089	0.89937067	0.82160002	0.819067001
#-4.721709244	0.548301104	-4.173408139	0.900144696	0.821200013	0.818771541
#-5.171370048	0.446734616	-4.724635433	0.920615971	0.847500026	0.847667515 x
#-4.610629256	0.547961873	-4.062667383	0.901819944	0.824100018	0.823553085
#-4.887908091	0.478164676	-4.409743416	0.914569497	0.835900009	0.835817933

#tonsil
#9.810074743	0.318356089	10.12843083	0.948963344	0.880999982	0.88399297 x
#10.034341	0.335774418	10.37011542	0.943565607	0.873899996	0.877298832
#10.07082106	0.327606735	10.39842779	0.945986331	0.875199974	0.87855202
#9.981798704	0.332313574	10.31411228	0.945672691	0.874599993	0.878558934
#9.931683769	0.329383708	10.26106748	0.947175086	0.874499977	0.87797755


run <- 4
ss <- 10
nc <- 20
expNum <- 5
cohort <- 'basel'

mlabels <- c("KM_label", "GMM_label", "FS_label", "damm_KM_tr_label")
path <- paste0("/home/campbell/yulee/DAMM/new/res/", cohort, "/", ss, "k/exp", expNum, "/")
#selected_parameters = "_cf1_nm0_so0_lv1.0_cs1_rr1_r"
selected_parameters = "_cf1_nm2_so0_lv1.0_cs1_rr1_r"

samp_df <- read.csv(paste0(path, "fs_nc", 30, "_iaGMM", selected_parameters, run, ".csv"))
samp_lb <- read.csv(paste0(path, "algo_label_nc", 30, "_iaKM",  selected_parameters, run, ".csv"))[,-1]

m1 <- m2 <- m3 <- NULL
count <- NULL
for (c in 0:(nc-1)) {
  
  not_in_n_are_in  <- samp_lb[,"KM_label_nc20"] != c & samp_lb[,"damm_KM_nc20_tr_label"] == c ## werent in & are in
  were_in_n_are_in <- samp_lb[,"KM_label_nc20"] == c & samp_lb[,"damm_KM_nc20_tr_label"] == c ## were in & are in
  were_in_n_not_in <- samp_lb[,"KM_label_nc20"] == c & samp_lb[,"damm_KM_nc20_tr_label"] != c ## were in & not in
  
  m1 <- rbind(m1, apply(samp_df[which(not_in_n_are_in == 1),-1], 2, mean))
  m2 <- rbind(m2, apply(samp_df[which(were_in_n_are_in == 1),-1], 2, mean))
  m3 <- rbind(m3, apply(samp_df[which(were_in_n_not_in == 1),-1], 2, mean))
  count <- rbind(count, c(sum(not_in_n_are_in), sum(were_in_n_are_in), sum(were_in_n_not_in)))
}

m4 <- read.csv(paste0(path, "damm_cen_nc", nc, "_iaKM", selected_parameters, run, ".csv"))[,-1]
#mat <- apply(rbind(m1, m2, m3, as.matrix(m4)), 2, minmax)
#mat <- rbind(m1, m2, m3)
#mat[is.nan(mat)] <- NA
mat <- apply(rbind(m1, m2, m3), 2, minmax)
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
ha3 = rowAnnotation("Count" = anno_barplot(count, axis = TRUE, gp = gpar(fill = 2:4)))
ht3 <- Heatmap(mat3, right_annotation = ha3, column_title = "were in & not in", column_title_side = 'top',
               cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE)

ht = ht1 + ht2 + ht3
col_fun = colorRamp2(c(0, 0.5, 1), c("blue", "white", "red"))
lgd = Legend(col_fun = col_fun, title = "Expression")
lgd_c = Legend(title = 'Condition', labels = c("weren't in & are in", "were in & are in", "were in & not in"), legend_gp = gpar(fill = 2:4))

pdf(paste0(path, "plot/", "heatmap.pdf"), width=24, height=6)
draw(ht, annotation_legend_list = list(lgd_c, lgd), row_title = "Cluster", column_title = "Marker", column_title_side = 'bottom')
dev.off()

##

#mat1[is.na(mat1)] <- 0
ht1 <- Heatmap(cor(t(mat1), t(mat2)), cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE,
               column_title = "(weren't in & are in) vs (were in & are in)", column_title_side = 'top')
ht2 <- Heatmap(cor(t(mat1), t(mat3)), cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE,
               column_title = "(weren't in & are in) vs (were in & not in)", column_title_side = 'top')
ht3 <- Heatmap(cor(t(mat2), t(mat3)), cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE,
               column_title = "(were in & are in) vs (were in & not in)", column_title_side = 'top')
ht = ht1 + ht2 + ht3

col_fun = colorRamp2(c(-1, 0, 1), c("blue", "white", "red"))
lgd = Legend(col_fun = col_fun, title = "Exp.")

pdf(paste0(path, "plot/", "cor_cluster.pdf"), width=24, height=6)
draw(ht, annotation_legend_list = lgd, column_title = "Cluster Correlation", column_title_side = 'bottom')
dev.off()

ht1 <- Heatmap(cor(mat1, mat2), cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE,
               column_title = "(weren't in & are in) vs (were in & are in)", column_title_side = 'top')
ht2 <- Heatmap(cor(mat1, mat3), cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE,
               column_title = "(weren't in & are in) vs (were in & not in)", column_title_side = 'top')
ht3 <- Heatmap(cor(mat2, mat3), cluster_rows = FALSE, cluster_columns = FALSE, show_heatmap_legend = FALSE,
               column_title = "(were in & are in) vs (were in & not in)", column_title_side = 'top')
ht = ht1 + ht2 + ht3

pdf(paste0(path, "plot/", "cor_marker.pdf"), width=24, height=10)
draw(ht, annotation_legend_list = lgd, column_title = "Marker Correlation", column_title_side = 'bottom')
dev.off()