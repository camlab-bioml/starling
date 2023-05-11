import os
import argparse

import numpy as np
import pandas as pd
#import scanpy as sc
from anndata import AnnData
from scipy.stats.mstats import winsorize

import torchmetrics
#from torchmetrics import F1Score
#from torchmetrics import PrecisionRecallCurve

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch
#import torch.nn as nn
#import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import scanpy.external as sce

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from flowsom import flowsom as flowsom
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

from sklearn.ensemble import RandomForestClassifier

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 512 if AVAIL_GPUS else 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_p_y_given_z(Y, Theta, dist_option): ## singlet case given expressions
    
    """ Returns # of obs x # of cluster matrix - p(y_n | z_n = c) """
  
    mu = torch.exp(Theta['log_mu'])
    sigma = torch.exp(torch.clamp(Theta['log_sigma'], min=-12, max=14))
    
    if dist_option == 0:
        dist_Y = torch.distributions.Normal(loc = mu, scale = sigma)
    elif dist_option == 2:
        dist_Y = torch.distributions.StudentT(df = dist_option, loc = mu, scale = sigma)
        
    return dist_Y.log_prob(Y.reshape(-1, 1, Y.shape[1])).sum(2) # <- sum because IID over G

def compute_p_s_given_z(S, Theta, dist_option): ## singlet case given cell sizes
    
    """ Returns # of obs x # of cluster matrix - p(s_n | z_n = c) """
    
    psi = torch.exp(Theta['log_psi'])
    omega = torch.exp(torch.clamp(Theta['log_omega'], min=-12, max=14)) #+ 1e-6

    if dist_option == 0:
        dist_S = torch.distributions.Normal(loc = psi, scale = omega)
    elif dist_option == 2:
        dist_S = torch.distributions.StudentT(df = dist_option, loc = psi, scale = omega)
        
    return dist_S.log_prob(S.reshape(-1,1))

def compute_p_y_given_gamma(Y, Theta, dist_option): ## doublet case given expressions
    
    """ Returns # of obs x # of cluster x # of cluster matrix - p(y_n | gamma_n = [c,c']) """

    mu = torch.exp(Theta['log_mu'])
    sigma = torch.exp(torch.clamp(Theta['log_sigma'], min=-12, max=14))

    mu2 = mu.reshape(1, mu.shape[0], mu.shape[1])
    mu2 = (mu2 + mu2.permute(1,0,2)) / 2.0 # C x C x G matrix 

    sigma2 = sigma.reshape(1, mu.shape[0], mu.shape[1])
    sigma2 = (sigma2 + sigma2.permute(1,0,2)) / 2.0

    dist_Y2 = torch.distributions.Normal(mu2, sigma2)

    if dist_option == 0:
        dist_Y2 = torch.distributions.Normal(loc = mu2, scale = sigma2)
    elif dist_option == 2:
        dist_Y2 = torch.distributions.StudentT(df = dist_option, loc = mu2, scale = sigma2)

    return dist_Y2.log_prob(Y.reshape(-1, 1, 1, mu.shape[1])).sum(3) # <- sum because IID over G

def compute_p_s_given_gamma(S, Theta, dist_option): ## singlet case given cell size
    
    """ Returns # of obs x # of cluster x # of cluster matrix - p(s_n | gamma_n = [c,c']) """

    psi = torch.exp(Theta['log_psi'])
    omega = torch.exp(torch.clamp(Theta['log_omega'], min=-12, max=14))

    psi2 = psi.reshape(-1,1)
    psi2 = psi2 + psi2.T

    omega2 = omega.reshape(-1,1)
    omega2 = omega2 + omega2.T #+ 1e-6
    
    if dist_option == 0:
        dist_S2 = torch.distributions.Normal(loc = psi2, scale = omega2)
    elif dist_option == 2:
        dist_S2 = torch.distributions.StudentT(df = dist_option, loc = psi2, scale = omega2)
    return dist_S2.log_prob(S.reshape(-1, 1, 1))

def compute_p_s_given_gamma_model_overlap(S, Theta):
    
    """ Returns # of obs x # of cluster x # of cluster matrix - p(s_n | gamma_n = [c,c']) """

    psi = torch.exp(Theta['log_psi'])
    omega = torch.exp(torch.clamp(Theta['log_omega'], min=-12, max=14))

    psi2 = psi.reshape(-1,1)
    psi2 = psi2 + psi2.T

    omega2 = omega.reshape(-1,1)
    omega2 = omega2 + omega2.T
    
    ## for v
    ccmax = torch.combinations(psi).max(1).values
    mat = torch.zeros(len(psi), len(psi), dtype=torch.float64).to(DEVICE)
    mat[np.triu_indices(len(psi), 1)] = ccmax
    mat += mat.clone().T
    mat += torch.eye(len(psi)).to(DEVICE) * psi
    
    ## for s
    c = 1 / (np.sqrt(2) * omega2)
    q = psi2 - S.reshape(-1,1,1)
    p = mat - S.reshape(-1,1,1)
    
    const = 1/(2 * (psi2 - mat))
    lbp = torch.special.erf(p * c)
    ubp = torch.special.erf(q * c)
    prob = torch.clamp(const * (ubp - lbp), min=1e-6, max=1.)
    
    return prob.log()

def compute_posteriors(Y, S, Theta, dist_option, model_overlap):
    
    ## priors
    log_pi = torch.nn.functional.log_softmax(Theta['is_pi'], dim=0) ## C
    log_tau = torch.nn.functional.log_softmax(Theta['is_tau'].reshape(-1), dim=0).reshape(log_pi.shape[0], log_pi.shape[0]) ## CxC
    #log_delta = torch.nn.functional.log_softmax(torch.tensor(np.log([0.95, 0.05])), dim=0) ## 2
    log_delta = torch.nn.functional.log_softmax(Theta['is_delta'], dim=0) ## 2
    
    prob_y_given_z = compute_p_y_given_z(Y, Theta, dist_option) ## log p(y_n|z=c) -> NxC
    prob_data_given_z_d0 = prob_y_given_z + log_pi ## log p(y_n|z=c) + log p(z=c) -> NxC + C -> NxC
  
    if S is not None:
        prob_s_given_z = compute_p_s_given_z(S, Theta, dist_option) ## log p(s_n|z=c) -> NxC
        prob_data_given_z_d0 += prob_s_given_z ## log p(y_n|z=c) + log p(s_n|z=c) -> NxC
  
    prob_y_given_gamma = compute_p_y_given_gamma(Y, Theta, dist_option) ## log p(y_n|g=[c,c']) -> NxCxC
    prob_data_given_gamma_d1 = prob_y_given_gamma + log_tau ## log p(y_n|g=[c,c']) + log p(g=[c,c']) -> NxCxC
    
    if S is not None:
        if model_overlap == 1:
            prob_s_given_gamma = compute_p_s_given_gamma_model_overlap(S, Theta) ## log p(s_n|g=[c,c']) -> NxCxC
        else:
            prob_s_given_gamma = compute_p_s_given_gamma(S, Theta, dist_option) ## log p(s_n|g=[c,c']) -> NxCxC
    
        prob_data_given_gamma_d1 += prob_s_given_gamma ## log p(y_n|g=[c,c']) + log p(s_n|g=[c,c']) -> NxCxC

    prob_data = torch.hstack([prob_data_given_z_d0 + log_delta[0], prob_data_given_gamma_d1.reshape(Y.shape[0], -1) + log_delta[1]]) 
    prob_data = torch.logsumexp(prob_data, dim=1) ## N
    ## log p(data) = 
    #case 1:
    #log p(y_n|z=c) + log p(d_n=0) + 
    #log p(y_n|g=[c,c']) + log p(d_n=1)
    #case 2:
    #log p(y_n,s_n|z=c) + log p(d_n=0) + 
    #log p(y_n,s_n|g=[c,c']) + log p(d_n=1)
    
    ## average negative likelihood scores
    cost = -prob_data.mean() ## a value
    
    ## integrate out z
    prob_data_given_d0 = torch.logsumexp(prob_data_given_z_d0, dim=1) ## p(data_n|d=0)_N
    prob_singlet = torch.clamp(torch.exp(prob_data_given_d0 + log_delta[0] - prob_data), min=0., max=1.)

    ## assignments
    r = prob_data_given_z_d0.T + log_delta[0] - prob_data ## p(d=0,z=c|data)
    v = prob_data_given_gamma_d1.T + log_delta[1] - prob_data ## p(d=1,gamma=[c,c']|data)
    
    return r.T, v.T, cost, prob_singlet

def simulate_data(Y, S=None, model_overlap=None): ## use real data to simulate singlets/doublets

    ''' return same number of cells as in Y/S, half of them are singlets and another half are doublets '''
    
    sample_size = int(Y.shape[0]/2)
    idx_singlet = np.random.choice(Y.shape[0], size = sample_size, replace=True)
    Y_singlet = Y[idx_singlet,:] ## expression

    idx_doublet = [np.random.choice(Y.shape[0], size = sample_size), np.random.choice(Y.shape[0], size = sample_size)]
    Y_doublet = (Y[idx_doublet[0],:] + Y[idx_doublet[1],:])/2.

    fake_Y = torch.vstack([Y_singlet, Y_doublet])
    fake_label = torch.concat([torch.ones(sample_size, dtype=torch.int), torch.zeros(sample_size, dtype=torch.int)])

    if S is None:
        return fake_Y, None, fake_label
    else:
        S_singlet = S[idx_singlet]
        if model_overlap:
            dmax = torch.vstack([S[idx_doublet[0]], S[idx_doublet[1]]]).max(0).values
            dsum = S[idx_doublet[0]] + S[idx_doublet[1]]
            rr_dist = torch.distributions.Uniform(dmax.type(torch.float64), dsum.type(torch.float64))
            S_doublet = rr_dist.sample() 
        else:
            S_doublet = S[idx_doublet[0]] + S[idx_doublet[1]]  
        fake_S = torch.hstack([S_singlet, S_doublet])
        return fake_Y, fake_S, fake_label ## singlet == 1, doublet == 0

def predict(dataLoader, model_params, dist_option, model_cell_size, model_overlap, threshold=0.5):
    singlet_prob_list = []
    singlet_assig_prob_list = []
    singlet_assig_label_list = []

    with torch.no_grad():
        for i, bat in enumerate(dataLoader):
            if model_cell_size:
                singlet_assig_prob, _, _, singlet_prob = compute_posteriors(bat[0].to(DEVICE), bat[1].to(DEVICE), model_params, dist_option, model_overlap)
            else:
                singlet_assig_prob, _, _, singlet_prob = compute_posteriors(bat.to(DEVICE), None, model_params, dist_option, model_overlap)

            singlet_prob_list.append(singlet_prob.cpu())
            singlet_assig_prob_list.append(singlet_assig_prob.exp().cpu())

            batch_pred = singlet_assig_prob.exp().max(1).indices
            batch_pred[singlet_prob <= threshold] = -1
            singlet_assig_label_list.append(batch_pred.cpu())

    singlet_prob = torch.cat(singlet_prob_list)
    singlet_assig_prob = torch.cat(singlet_assig_prob_list)
    singlet_assig_label = torch.cat(singlet_assig_label_list)
    
    return singlet_prob, singlet_assig_prob, singlet_assig_label

def model_paramters(init_c, init_v, init_s=None, init_sv=None):

    nc = init_c.shape[0]
    pi = np.ones(nc) / nc
    tau = np.ones((nc, nc))
    tau = tau / tau.sum()

    model_params = {
    'is_pi': np.log(pi + 1e-6),
    'is_tau': np.log(tau + 1e-6),
    'is_delta': np.log([0.95, 0.05])
    }

    model_params['log_mu'] = np.log(init_c + 1e-6)
    
    init_v[np.isnan(init_v)] = 1e-6
    model_params['log_sigma'] = np.log(init_v + 1e-6)
    
    if init_s:
        model_params['log_psi'] = np.log(np.array(init_s).astype(float) + 1e-6)

        init_sv[np.isnan(init_sv)] = 1e-6
        model_params['log_omega'] = np.log(np.array(init_sv).astype(float) ** 0.5 + 1e-6)
    
    return model_params

def init_clustering(X, initial_clustering_method, k, training_file_fn=None):
    
    if initial_clustering_method == 'KM':
        kms = KMeans(k).fit(X)
        init_l = kms.labels_
        init_label_class = np.unique(init_l)

        init_c = kms.cluster_centers_
        init_v = np.array([np.array(X)[init_l == c,:].var(0) for c in init_label_class])
        
    elif initial_clustering_method == 'GMM':
        gmm = GaussianMixture(n_components = k, covariance_type = 'diag').fit(X)
        init_l = gmm.predict(X)

        init_c = gmm.means_
        init_v = gmm.covariances_
                 
    elif initial_clustering_method == 'PG':
        init_l, _, _ = sce.tl.phenograph(X)
        
        ## save phenograph centers
        nc = len(np.unique(init_l))
        init_v = np.zeros((nc, X.shape[1]))
        init_c = np.zeros((nc, X.shape[1]))
        for c in range(nc):
            init_v[c,:] = X[init_l==c].var(0)
            init_c[c,:] = X[init_l==c].mean(0)
    elif initial_clustering_method == 'FS':

        ## needs to output to csv first
        #ofn = OPATH + "fs_" + ONAME + ".csv"
        #pd.DataFrame(mat.numpy()).to_csv(training_file_fn)
        fsom = flowsom(training_file_fn, if_fcs=False, if_drop=True, drop_col=['Unnamed: 0'])
            
        fsom.som_mapping(50, # x_n: e.g. 100, the dimension of expected map
            50, # y_n: e.g. 100, the dimension of expected map
            fsom.df.shape[1],
            1, # sigma: e.g 1, the standard deviation of initialized weights
            0.5, # lr: e.g 0.5, learning rate
            1000, # batch_size: 1000, iteration times
            tf_str=None, # string, e.g. hlog', None, etc - the transform algorithm
            if_fcs=False # bool, whethe the imput file is fcs file. If not, it should be a csv file
            # seed = 10, for reproducing
        )
        start = k; fsom_num_cluster = 0
        while fsom_num_cluster < k:
            #print(nc, start, fsom_nc)
            fsom.meta_clustering(AgglomerativeClustering, min_n=start, max_n=start, verbose=False, iter_n=10) # train the meta clustering for cluster in range(40,45)  

            fsom.labeling()
            #fsom.bestk # the best number of clusters within the range of (min_n, max_n)
            fsom_class = np.unique(fsom.df['category'])
            fsom_num_cluster = len(fsom_class)
            start += 1
    
        fsom_labels = np.array(fsom.df['category'])

        i = 0
        init_l = np.zeros(fsom.df.shape[0])
        init_vars = np.zeros((len(fsom_class), fsom.df.shape[1]))
        init_centers = np.zeros((len(fsom_class), fsom.df.shape[1]))
        for row in fsom_class:
            init_l[fsom_labels==row] = i
            init_vars[i,:] = fsom.df[fsom_labels==row].var(0)
            init_centers[i,:] = fsom.df[fsom_labels==row].mean(0)
            i += 1

        init_v = init_vars[:,:-1]
        init_c = init_centers[:,:-1]
        #os.remove(ofn)
        
    return init_c, init_v, init_l.astype(int)

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
def construct_annData(data):
    
    #data = df[df.loc[:,'sample'] == sample]
    wCols = data.columns
    df = data.loc[:,wCols]

    cell_info = df['sample'].astype(str) + '_' + df['id'].astype(str)
    df = df.set_index(cell_info)

    obs = df[ ['sample', 'id', 'x', 'y', 'area', 'area_convex', 'neighbor'] ]    

    df = df.drop("sample", axis=1)
    df = df.drop("id", axis=1)
    df = df.drop("x", axis=1)
    df = df.drop("y", axis = 1)
    df = df.drop("area", axis=1)
    df = df.drop("area_convex", axis=1)
    df = df.drop("neighbor", axis=1)

    adata = AnnData(df.values, obs)
    adata.var_names = wCols[7:]
    
    return adata
    #adata.write("/home/campbell/yulee/DAMM/new/data/{}/{}_{}_samples_ex1.h5ad".format(cohort, cohort, segA)) # h5ad file 

class STL(pl.LightningModule):
    def __init__(self, adata,
                 model_cell_size=1, 
                 cofactor=1,
                 initial_clustering_method='PG', 
                 k=20, 
                 dist_option=2, 
                 model_overlap=1, 
                 model_regularizer=1, 
                 ofn=None,
                 learning_rate=1e-3,
                 ):
        super().__init__()
        
        #self.save_hyperparameters()
        
        self.adata = adata
        self.initial_clustering_method = initial_clustering_method
        self.k = k
        self.ofn = ofn

        self.cofactor = cofactor
        self.dist_option = dist_option
        self.model_cell_size = model_cell_size
        self.model_overlap = model_overlap
        self.model_regularizer = model_regularizer
        self.learning_rate = learning_rate
        self.indx = 4 if self.model_cell_size == 1 else 2
        
        # add metrics
        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1Score(number_classes=2, average="micro")
        self.train_auroc = torchmetrics.AUROC(number_classes=2, average="micro")
        
    def forward(self, batch):
        
        if self.model_cell_size == 1:
            y, s, fy, fs, fl = batch
            _, _, model_nll, _ = compute_posteriors(y, s, self.model_params, self.dist_option, self.model_overlap)        
            _, _, _, p_fake_singlet = compute_posteriors(fy, fs, self.model_params, self.dist_option, self.model_overlap)
        else:
            y, fy, fl = batch
            _, _, model_nll, _ = compute_posteriors(y, None, self.model_params, self.dist_option, self.model_overlap)        
            _, _, _, p_fake_singlet = compute_posteriors(fy, None, self.model_params, self.dist_option, self.model_overlap)

        fake_loss = torch.nn.BCELoss()(p_fake_singlet, fl.to(torch.double))
        
        return model_nll, fake_loss, p_fake_singlet
        
    def training_step(self, batch, batch_idx):
        
        #y, s, fy, fs, fl = batch
        model_nll, fake_loss, p_fake_singlet = self(batch)
        
        # total loss
        loss = model_nll + self.model_regularizer * fake_loss

        # accumulate and return metrics for logging
        self.train_auroc(p_fake_singlet, batch[self.indx])
        self.train_acc(p_fake_singlet, batch[self.indx])
        self.train_f1(p_fake_singlet, batch[self.indx])
        
        self.log("train_auroc", self.train_auroc)                
        self.log("train_accuracy", self.train_acc)
        self.log("train_f1", self.train_f1)
        
        self.log("train_nll", model_nll)
        self.log("train_bce", fake_loss)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True) #, logger=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model_params.values(), lr=self.learning_rate)
        return optimizer
    
    def prepare_data(self):
        
        ## preprocessing
        #if self.model_cell_size:
        #    transform_mat = np.hstack((self.adata.X, self.adata.obs['area'].values.reshape(-1,1)))
        #else:
        #    transform_mat = self.adata.X
            
        #if self.cofactor > 0:
        #    transform_mat = np.arcsinh(transform_mat / self.cofactor)
        #else:
        #    transform_mat = transform_mat
            
        #self.X = torch.tensor(np.array(transform_mat))
        
        self.X = torch.tensor(self.adata.X)

        ## simulate data
        if self.model_cell_size == 1:
            self.S = torch.tensor(self.adata.obs['area'])
            tr_fy, tr_fs, tr_fl = simulate_data(self.X, self.S, self.model_overlap)
            self.train_df = ConcatDataset(self.X, self.S, tr_fy, tr_fs, tr_fl)        
        else:
            self.S = None
            tr_fy, _, tr_fl = simulate_data(self.X)
            self.train_df = ConcatDataset(self.X, tr_fy, tr_fl)
    
        self.init_c, self.init_v, self.init_label = init_clustering(self.X, self.initial_clustering_method, self.k, self.ofn)
        if self.model_cell_size == 1:
            self.init_s = []; self.init_sv = []
            for nc in np.unique(self.init_label):
                self.init_s.append(self.adata.obs.iloc[np.where(self.init_label == nc)]['area'].mean(0))
                self.init_sv.append(self.adata.obs.iloc[np.where(self.init_label == nc)]['area'].var(0))
            
            #self.init_sv[np.isnan(self.init_sv)] = 0
            model_params = model_paramters(self.init_c, self.init_v, self.init_s, torch.tensor(np.array(self.init_sv)))
        else:
            self.init_s = None; self.init_sv = None
            model_params = model_paramters(self.init_c, self.init_v)

        self.model_params = {k: torch.from_numpy(val).to(DEVICE).requires_grad_(True) for (k,val) in model_params.items()}
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_df, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    
    def result(self, threshold=0.5):

        if self.model_cell_size == 1:
            model_pred_loader = torch.utils.data.DataLoader(ConcatDataset(self.X, self.S), batch_size = 1000, shuffle = False)
        else:
            model_pred_loader = torch.utils.data.DataLoader(self.X, batch_size = 1000, shuffle = False)

        self.singlet_prob, self.singlet_assig_prob, singlet_assig_label = predict(model_pred_loader, self.model_params, self.dist_option, self.model_cell_size, self.model_overlap, threshold)
        self.label = singlet_assig_label.numpy().astype('str')
        self.label[self.label == '-1'] = 'doublet'

        if self.model_cell_size == 1:
            pretty_printing = np.hstack((self.adata.var_names, 'area'))
            c = torch.hstack([self.model_params['log_mu'], self.model_params['log_psi'].reshape(-1,1)]).detach().exp().cpu().numpy()
            #v = torch.hstack([self.model_params['log_sigma'], self.model_params['log_omega'].reshape(-1,1)]).detach().exp().cpu().numpy()
            self.c = pd.DataFrame(c, columns=pretty_printing)
        else:
            c = self.model_params['log_mu'].detach().exp().cpu().numpy()
            #v = self.model_params['log_sigma'].cpu().detach().exp().cpu().numpy()
            self.c = pd.DataFrame(c, columns=self.adata.var_names)
        
        self.init_c = pd.DataFrame(self.init_c, columns = self.adata.var_names) #.to_csv(code_dir + "/output/init_centroids.csv")            
        ## starling centroids
        #if self.model_cell_size == 1:
        #    st_c, _, self.st_label, self.prob_singlet, self.prob_cluster_assig = predict(self.X.to(DEVICE), self.S.to(DEVICE), self.model_params, self.dist_option, self.model_cell_size, self.model_overlap, threshold)
        #    pretty_printing = np.hstack((self.adata.var_names, 'area'))
        #    self.init_c = pd.DataFrame(np.hstack([self.init_c, np.array(self.init_s).reshape(-1,1)]), columns = pretty_printing) #.to_csv(code_dir + "/output/init_centroids.csv")
        #else:
        #    st_c, _, self.st_label, self.prob_singlet, self.prob_cluster_assig = predict(self.X.to(DEVICE), None, self.model_params, self.dist_option, self.model_cell_size, self.model_overlap, threshold)
        #    pretty_printing = self.adata.var_names
        #    self.init_c = pd.DataFrame(self.init_c, columns = pretty_printing) #.to_csv(code_dir + "/output/init_centroids.csv")
        #self.star_c = pd.DataFrame(st_c, columns = pretty_printing) #.to_csv(code_dir + "/output/star_centroids.csv")

def init_clustering2(X, initial_clustering_method, k, training_file_fn=None):
    
    if initial_clustering_method == 'KM':
        kms = KMeans(k).fit(X)
        init_l = kms.labels_
        init_label_class = np.unique(init_l)

        init_c = kms.cluster_centers_
        init_v = np.array([np.array(X)[init_l == c,:].var(0) for c in init_label_class])
        init_m = kms

    elif initial_clustering_method == 'GMM':
        gmm = GaussianMixture(n_components = k, covariance_type = 'diag').fit(X)
        init_l = gmm.predict(X)

        init_c = gmm.means_
        init_v = gmm.covariances_
        init_m = gmm

    elif initial_clustering_method == 'PG':
        init_l, _, _ = sce.tl.phenograph(X, k = 300)
        
        ## save phenograph centers
        nc = len(np.unique(init_l))
        init_v = np.zeros((nc, X.shape[1]))
        init_c = np.zeros((nc, X.shape[1]))
        init_m = None
        for c in range(nc):
            init_v[c,:] = X[init_l==c].var(0)
            init_c[c,:] = X[init_l==c].mean(0)
    elif initial_clustering_method == 'FS':

        ## needs to output to csv first
        #ofn = OPATH + "fs_" + ONAME + ".csv"
        #pd.DataFrame(mat.numpy()).to_csv(training_file_fn)
        fsom = flowsom(training_file_fn, if_fcs=False, if_drop=True, drop_col=['Unnamed: 0'])
            
        fsom.som_mapping(50, # x_n: e.g. 100, the dimension of expected map
            50, # y_n: e.g. 100, the dimension of expected map
            fsom.df.shape[1],
            1, # sigma: e.g 1, the standard deviation of initialized weights
            0.5, # lr: e.g 0.5, learning rate
            1000, # batch_size: 1000, iteration times
            tf_str=None, # string, e.g. hlog', None, etc - the transform algorithm
            if_fcs=False # bool, whethe the imput file is fcs file. If not, it should be a csv file
            # seed = 10, for reproducing
        )
        start = k; fsom_num_cluster = 0
        while fsom_num_cluster < k:
            #print(nc, start, fsom_nc)
            fsom.meta_clustering(AgglomerativeClustering, min_n=start, max_n=start, verbose=False, iter_n=10) # train the meta clustering for cluster in range(40,45)  

            fsom.labeling()
            #fsom.bestk # the best number of clusters within the range of (min_n, max_n)
            fsom_class = np.unique(fsom.df['category'])
            fsom_num_cluster = len(fsom_class)
            start += 1
    
        fsom_labels = np.array(fsom.df['category'])

        i = 0
        init_l = np.zeros(fsom.df.shape[0])
        init_vars = np.zeros((len(fsom_class), fsom.df.shape[1]))
        init_centers = np.zeros((len(fsom_class), fsom.df.shape[1]))
        for row in fsom_class:
            init_l[fsom_labels==row] = i
            init_vars[i,:] = fsom.df[fsom_labels==row].var(0)
            init_centers[i,:] = fsom.df[fsom_labels==row].mean(0)
            i += 1

        init_v = init_vars[:,:-1]
        init_c = init_centers[:,:-1]
        init_m = fsom

        #os.remove(ofn)
        
    return init_m, init_c, init_v, init_l.astype(int)

class STL2(pl.LightningModule):
    def __init__(self, adata,
                 model_cell_size=1, 
                 cofactor=1,
                 initial_clustering_method='PG', 
                 k=20, 
                 dist_option=2, 
                 model_overlap=1, 
                 model_regularizer=1, 
                 ofn=None,
                 learning_rate=1e-3,
                 ):
        super().__init__()
        
        #self.save_hyperparameters()
        
        self.adata = adata
        self.initial_clustering_method = initial_clustering_method
        self.k = k
        self.ofn = ofn

        self.cofactor = cofactor
        self.dist_option = dist_option
        self.model_cell_size = model_cell_size
        self.model_overlap = model_overlap
        self.model_regularizer = model_regularizer
        self.learning_rate = learning_rate
        self.indx = 4 if self.model_cell_size == 1 else 2
        
        # add metrics
        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1Score(number_classes=2, average="micro")
        self.train_auroc = torchmetrics.AUROC(number_classes=2, average="micro")
        
    def forward(self, batch):
        
        if self.model_cell_size == 1:
            y, s, fy, fs, fl = batch
            _, _, model_nll, _ = compute_posteriors(y, s, self.model_params, self.dist_option, self.model_overlap)        
            _, _, _, p_fake_singlet = compute_posteriors(fy, fs, self.model_params, self.dist_option, self.model_overlap)
        else:
            y, fy, fl = batch
            _, _, model_nll, _ = compute_posteriors(y, None, self.model_params, self.dist_option, self.model_overlap)        
            _, _, _, p_fake_singlet = compute_posteriors(fy, None, self.model_params, self.dist_option, self.model_overlap)

        fake_loss = torch.nn.BCELoss()(p_fake_singlet, fl.to(torch.double))
        
        return model_nll, fake_loss, p_fake_singlet
        
    def training_step(self, batch, batch_idx):
        
        #y, s, fy, fs, fl = batch
        model_nll, fake_loss, p_fake_singlet = self(batch)
        
        # total loss
        loss = model_nll + self.model_regularizer * fake_loss

        # accumulate and return metrics for logging
        self.train_auroc(p_fake_singlet, batch[self.indx])
        self.train_acc(p_fake_singlet, batch[self.indx])
        self.train_f1(p_fake_singlet, batch[self.indx])
        
        self.log("train_auroc", self.train_auroc)                
        self.log("train_accuracy", self.train_acc)
        self.log("train_f1", self.train_f1)
        
        self.log("train_nll", model_nll)
        self.log("train_bce", fake_loss)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True) #, logger=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model_params.values(), lr=self.learning_rate)
        return optimizer
    
    def prepare_data(self):
        
        ## preprocessing
        #if self.model_cell_size:
        #    transform_mat = np.hstack((self.adata.X, self.adata.obs['area'].values.reshape(-1,1)))
        #else:
        #    transform_mat = self.adata.X
            
        #if self.cofactor > 0:
        #    transform_mat = np.arcsinh(transform_mat / self.cofactor)
        #else:
        #    transform_mat = transform_mat
            
        #self.X = torch.tensor(np.array(transform_mat))
        
        self.X = torch.tensor(self.adata.X)

        ## simulate data
        if self.model_cell_size == 1:
            self.S = torch.tensor(self.adata.obs['area'])
            tr_fy, tr_fs, tr_fl = simulate_data(self.X, self.S, self.model_overlap)
            self.train_df = ConcatDataset(self.X, self.S, tr_fy, tr_fs, tr_fl)        
        else:
            self.S = None
            tr_fy, _, tr_fl = simulate_data(self.X)
            self.train_df = ConcatDataset(self.X, tr_fy, tr_fl)
    
        self.init_m, self.init_c, self.init_v, self.init_label = init_clustering2(self.X, self.initial_clustering_method, self.k, self.ofn)
        if self.model_cell_size == 1:
            self.init_s = []; self.init_sv = []
            for nc in np.unique(self.init_label):
                self.init_s.append(self.adata.obs.iloc[np.where(self.init_label == nc)]['area'].mean(0))
                self.init_sv.append(self.adata.obs.iloc[np.where(self.init_label == nc)]['area'].var(0))
            
            #self.init_sv[np.isnan(self.init_sv)] = 0
            model_params = model_paramters(self.init_c, self.init_v, self.init_s, torch.tensor(np.array(self.init_sv)))
        else:
            self.init_s = None; self.init_sv = None
            model_params = model_paramters(self.init_c, self.init_v)

        self.model_params = {k: torch.from_numpy(val).to(DEVICE).requires_grad_(True) for (k,val) in model_params.items()}
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_df, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    
    def result(self, threshold=0.5):

        if self.model_cell_size == 1:
            model_pred_loader = torch.utils.data.DataLoader(ConcatDataset(self.X, self.S), batch_size = 1000, shuffle = False)
        else:
            model_pred_loader = torch.utils.data.DataLoader(self.X, batch_size = 1000, shuffle = False)

        self.singlet_prob, self.singlet_assig_prob, singlet_assig_label = predict(model_pred_loader, self.model_params, self.dist_option, self.model_cell_size, self.model_overlap, threshold)
        self.label = singlet_assig_label.numpy().astype('str')
        self.label[self.label == '-1'] = 'doublet'

        if self.model_cell_size == 1:
            pretty_printing = np.hstack((self.adata.var_names, 'area'))
            c = torch.hstack([self.model_params['log_mu'], self.model_params['log_psi'].reshape(-1,1)]).detach().exp().cpu().numpy()
            #v = torch.hstack([self.model_params['log_sigma'], self.model_params['log_omega'].reshape(-1,1)]).detach().exp().cpu().numpy()
            self.c = pd.DataFrame(c, columns=pretty_printing)
        else:
            c = self.model_params['log_mu'].detach().exp().cpu().numpy()
            #v = self.model_params['log_sigma'].cpu().detach().exp().cpu().numpy()
            self.c = pd.DataFrame(c, columns=self.adata.var_names)
        
        self.init_c = pd.DataFrame(self.init_c, columns = self.adata.var_names) #.to_csv(code_dir + "/output/init_centroids.csv")            
        ## starling centroids
        #if self.model_cell_size == 1:
        #    st_c, _, self.st_label, self.prob_singlet, self.prob_cluster_assig = predict(self.X.to(DEVICE), self.S.to(DEVICE), self.model_params, self.dist_option, self.model_cell_size, self.model_overlap, threshold)
        #    pretty_printing = np.hstack((self.adata.var_names, 'area'))
        #    self.init_c = pd.DataFrame(np.hstack([self.init_c, np.array(self.init_s).reshape(-1,1)]), columns = pretty_printing) #.to_csv(code_dir + "/output/init_centroids.csv")
        #else:
        #    st_c, _, self.st_label, self.prob_singlet, self.prob_cluster_assig = predict(self.X.to(DEVICE), None, self.model_params, self.dist_option, self.model_cell_size, self.model_overlap, threshold)
        #    pretty_printing = self.adata.var_names
        #    self.init_c = pd.DataFrame(self.init_c, columns = pretty_printing) #.to_csv(code_dir + "/output/init_centroids.csv")
        #self.star_c = pd.DataFrame(st_c, columns = pretty_printing) #.to_csv(code_dir + "/output/star_centroids.csv")