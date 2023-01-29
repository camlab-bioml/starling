import os
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats.mstats import winsorize

import torchmetrics
from torchmetrics import F1Score
from torchmetrics import PrecisionRecallCurve

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch
import torch.nn as nn
import torch.optim as optim
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
BATCH_SIZE = 256 if AVAIL_GPUS else 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_p_y_given_z(Y, Theta, dist_option):
  """ Returns NxC
  p(y_n | z_n = c)
  """
  #mu = torch.exp(torch.clamp(Theta['log_mu'], min=-12))
  mu = torch.exp(Theta['log_mu'])
  sigma = torch.exp(torch.clamp(Theta['log_sigma'], min=-12, max=14))
    
  if dist_option == 0:
    dist_Y = D.Normal(loc = mu, scale = sigma)
  elif dist_option == 2:
    dist_Y = D.StudentT(df = dist_option, loc = mu, scale = sigma)
  else:
    return "Please input the correct noise distribution"
  
  return dist_Y.log_prob(Y.reshape(-1, 1, Y.shape[1])).sum(2) # <- sum because IID over G

def compute_p_s_given_z(S, Theta, dist_option):
  """ Returns NxC
  p(s_n | z_n = c)
  """
  #print(Theta['log_psi'])
  psi = torch.exp(Theta['log_psi'])
  #psi = torch.exp(torch.clamp(Theta['log_psi'], min=-12))
  omega = torch.exp(torch.clamp(Theta['log_omega'], min=-12, max=14)) #+ 1e-6

  if dist_option == 0:
    dist_S = D.Normal(loc = psi, scale = omega)
  elif dist_option == 2:
    dist_S = D.StudentT(df = dist_option, loc = psi, scale = omega)
  else:
    return "Please input the correct noise distribution"

  return dist_S.log_prob(S.reshape(-1,1))

def compute_p_y_given_gamma(Y, Theta, dist_option):
  """ NxCxC
  p(y_n | gamma_n = [c,c'])
  """
  mu = torch.exp(Theta['log_mu'])
  #mu = torch.exp(torch.clamp(Theta['log_mu'], min=-12))
  sigma = torch.exp(torch.clamp(Theta['log_sigma'], min=-12, max=14))

  mu2 = mu.reshape(1, mu.shape[0], mu.shape[1])
  mu2 = (mu2 + mu2.permute(1,0,2)) / 2.0 # C x C x G matrix 

  sigma2 = sigma.reshape(1, mu.shape[0], mu.shape[1])
  sigma2 = (sigma2 + sigma2.permute(1,0,2)) / 2.0 #+ 1e-6

  dist_Y2 = D.Normal(mu2, sigma2)

  if dist_option == 0:
    dist_Y2 = D.Normal(loc = mu2, scale = sigma2)
  elif dist_option == 2:
    dist_Y2 = D.StudentT(df = dist_option, loc = mu2, scale = sigma2)
  else:
    return "Please input the correct noise distribution"

  return dist_Y2.log_prob(Y.reshape(-1, 1, 1, mu.shape[1])).sum(3) # <- sum because IID over G

def compute_p_s_given_gamma(S, Theta, dist_option):

    psi = torch.exp(Theta['log_psi'])
    #psi = torch.exp(torch.clamp(Theta['log_psi'], min=-12))
    omega = torch.exp(torch.clamp(Theta['log_omega'], min=-12, max=14))

    psi2 = psi.reshape(-1,1)
    psi2 = psi2 + psi2.T

    omega2 = omega.reshape(-1,1)
    omega2 = omega2 + omega2.T #+ 1e-6
    
    if dist_option == 0:
        dist_S2 = D.Normal(loc = psi2, scale = omega2)
    elif dist_option == 2:
        dist_S2 = D.StudentT(df = dist_option, loc = psi2, scale = omega2)
    return dist_S2.log_prob(S.reshape(-1, 1, 1))

def compute_p_s_given_gamma1(S, Theta):

    psi = torch.exp(Theta['log_psi'])
    #psi = torch.exp(torch.clamp(Theta['log_psi'], min=-12))
    omega = torch.exp(torch.clamp(Theta['log_omega'], min=-12, max=14))

    psi2 = psi.reshape(-1,1)
    psi2 = psi2 + psi2.T

    omega2 = omega.reshape(-1,1)
    omega2 = omega2 + omega2.T #+ 1e-6
    
    ## for v
    ccmax = torch.combinations(psi).max(1).values
    mat = torch.zeros(len(psi), len(psi), dtype=torch.float64).to(DEVICE)
    mat[np.triu_indices(len(psi), 1)] = ccmax
    mat += mat.clone().T
    #mat = mat + mat.T
    #mat += torch.eye(len(psi)).cuda() * psi
    mat += torch.eye(len(psi)).to(DEVICE) * psi
    #v = D.Uniform(mat, psi2).sample()
    
    ## for s
    #c = (1 / (2 * (omega2 ** 2)))
    c = 1 / (np.sqrt(2) * omega2)
    q = psi2 - S.reshape(-1,1,1)
    p = mat - S.reshape(-1,1,1)
    
    const = 1/(2 * (psi2 - mat))
    #ubp = torch.special.erf(q * torch.sqrt(c))
    #lbp = torch.special.erf(p * torch.sqrt(c))
    lbp = torch.special.erf(p * c)
    ubp = torch.special.erf(q * c)
    prob = torch.clamp(const * (ubp - lbp), min=1e-6, max=1.)
    
    return prob.log()

def compute_posteriors_nll_p_singlet(Y, S, Theta, dist_option, model_overlap):

    log_pi = F.log_softmax(Theta['is_pi'], dim=0)
    log_tau = F.log_softmax(Theta['is_tau'].reshape(-1), dim=0).reshape(log_pi.shape[0], log_pi.shape[0])
    #log_delta = F.log_softmax(Theta['is_delta'], dim=0)
    log_delta = F.log_softmax(torch.tensor(np.log([0.95, 0.05])), dim=0)

    prob_y_given_z = compute_p_y_given_z(Y, Theta, dist_option) ## p(y_n|z=c)
    prob_data_given_z_d0 = prob_y_given_z + log_pi ## p(y_n|z=c)p(pi)
  
    if S is not None:
        prob_s_given_z = compute_p_s_given_z(S, Theta, dist_option) ## p(s_n|z=c)
        prob_data_given_z_d0 += prob_s_given_z ## p(y_n,s_n|z=c) -> NxC
  
    prob_y_given_gamma = compute_p_y_given_gamma(Y, Theta, dist_option) ## p(y_n|g=[c,c']) -> NxCxC
    prob_data_given_gamma_d1 = prob_y_given_gamma + log_tau ## p(y_n|g=[c,c'])p(tau)
    
    if S is not None:
        if model_overlap:
            prob_s_given_gamma = compute_p_s_given_gamma1(S, Theta) ## p(s_n|g=[c,c']) -> NxCxC
        else:
            prob_s_given_gamma = compute_p_s_given_gamma(S, Theta, dist_option) ## p(s_n|g=[c,c']) -> NxCxC
    
        prob_data_given_gamma_d1 += prob_s_given_gamma ## p(y_n,s_n|g=[c,c']) -> NxCxC

    #p_data = torch.cat([prob_data_given_z_d0 + log_delta[0], prob_data_given_gamma_d1.reshape(X.shape[0], -1) + log_delta[1]], dim=1)
    prob_data = torch.hstack([prob_data_given_z_d0 + log_delta[0], prob_data_given_gamma_d1.reshape(Y.shape[0], -1) + log_delta[1]]) 
    prob_data = torch.logsumexp(prob_data, dim=1)
    ## p(data) = p(y_n,s_n|g=[c,c'])p(d_n=0) + p(y_n,s_n|g=[c,c'])p(d_n=1)
    
    ## average negative likelihood scores
    #cost = -prob_data.sum()
    cost = -prob_data.mean()
    
    ## integrate out z
    prob_data_given_d0 = torch.logsumexp(prob_data_given_z_d0, dim=1) ## p(data_n|d=0)_N
    prob_singlet = torch.clamp(torch.exp(prob_data_given_d0 + log_delta[0] - prob_data), min=0., max=1.)

    ## assignments
    r = prob_data_given_z_d0.T + log_delta[0] - prob_data ## p(d=0,z=c|data)
    v = prob_data_given_gamma_d1.T + log_delta[1] - prob_data ## p(gamma=[c,c']|data)
    
    return r.T, v.T, cost, prob_singlet

def simulate_data(Y, S=None, model_overlap=None): ## use real data to simulate singlets/doublets
  
  ''' return same number of cells as in Y/S, half of them are singlets and another half are doublets '''

  #N_training = 5000
  sample_size = int(Y.shape[0]/2)
  idx_singlet = np.random.choice(Y.shape[0], size = sample_size, replace=True)
  Y_singlet = Y[idx_singlet,:] ## expression
  
  idx_doublet = [np.random.choice(Y.shape[0], size = sample_size), np.random.choice(Y.shape[0], size = sample_size)]
  Y_doublet = (Y[idx_doublet[0],:] + Y[idx_doublet[1],:])/2.
  
  #fake_Y = torch.tensor(np.vstack([Y_singlet, Y_doublet]))
  #fake_label = torch.tensor(np.concatenate([np.ones(sample_size), np.zeros(sample_size)]))
    
  fake_Y = torch.vstack([Y_singlet, Y_doublet])
  fake_label = torch.concat([torch.ones(sample_size, dtype=torch.int), torch.zeros(sample_size, dtype=torch.int)])

  if S is None:
    return fake_Y, None, fake_label
  else:
    S_singlet = S[idx_singlet]
    if model_overlap:
        #dmax = torch.tensor(np.vstack([S[idx_doublet[0]], S[idx_doublet[1]]])).max(0).values        
        dmax = torch.vstack([S[idx_doublet[0]], S[idx_doublet[1]]]).max(0).values
        dsum = S[idx_doublet[0]] + S[idx_doublet[1]]
        rr_dist = D.Uniform(dmax.type(torch.float64), dsum.type(torch.float64))
        S_doublet = rr_dist.sample()
    else:
        S_doublet = S[idx_doublet[0]] + S[idx_doublet[1]]  
    #fake_S = torch.tensor(np.hstack([S_singlet, S_doublet]))
    fake_S = torch.hstack([S_singlet, S_doublet])
    return fake_Y, fake_S, fake_label ## singlet == 1, doublet == 0

class ConcatDataset(torch.utils.data.Dataset):
  def __init__(self, *datasets):
    self.datasets = datasets

  def __getitem__(self, i):
    return tuple(d[i] for d in self.datasets)

  def __len__(self):
    return min(len(d) for d in self.datasets)
    
def construct_annData(data, channels):
    wCols = np.hstack(['sample', 'id', 'x', 'y', 'area', 'area_convex', channels])
    df = data.loc[:,wCols]

    cell_info = df['sample'].astype(str) + '_' + df['id'].astype(str)
    df = df.set_index(cell_info)

    obs = df[ ['sample', 'id', 'x', 'y', 'area', 'area_convex'] ]    

    df = df.drop("sample", axis=1)
    df = df.drop("id", axis=1)
    df = df.drop("x", axis=1)
    df = df.drop("y", axis = 1)
    df = df.drop("area", axis=1)
    df = df.drop("area_convex", axis=1)

    #obs['area'] = winsorize(obs['area'], limits=[0, 0.01]).data ## winsorize cell sizes

    #for c in df.columns:
    #    df[c] = winsorize(df[c], limits=[0, 0.01]).data ## winsorize cell expressions

    adata = AnnData(df.values, obs)
    adata.var_names = channels
    
    return adata
    #adata.write("/home/campbell/yulee/DAMM/new/data/{}/{}_{}_samples_ex1.h5ad".format(cohort, cohort, segA)) # h5ad file

class sampleData(pl.LightningDataModule):
    def __init__(self, input_fn, channels, sample_size, cofactor, cell_size, overlap):
        super().__init__()
        self.sample_size = sample_size * 1000
        
        self.input_mat = pd.read_csv(input_fn, index_col=0)
        
        if channels == None:
            channels = self.input_mat.columns[6:]

        if cofactor > 0:
            self.transform_mat = pd.concat([self.input_mat.iloc[:,:6], np.arcsinh(self.input_mat.loc[:,channels] / cofactor)], axis=1)
        else:
            self.transform_mat = self.input_mat

        idx = np.random.choice(self.transform_mat.shape[0], size = 2 * self.sample_size, replace = False)            
        self.tr_idx = idx[:self.sample_size]
        self.va_idx = idx[self.sample_size:(2*self.sample_size)]
        #self.te_idx = idx[(2*self.sample_size):]

        self.tr_h5ad = construct_annData(self.transform_mat.iloc[self.tr_idx,:], channels)
        self.va_h5ad = construct_annData(self.transform_mat.iloc[self.va_idx,:], channels)
        #self.te_h5ad = construct_annData(self.transform_mat.iloc[self.te_idx,:], channels)

        if cell_size:
            self.subset_mat = self.transform_mat.loc[:,np.hstack([channels, 'area'])]
        else:
            self.subset_mat = self.transform_mat.loc[:,channels]
        
        self.tr_mat = torch.tensor(np.array(self.subset_mat.iloc[self.tr_idx,:]))
        self.va_mat = torch.tensor(np.array(self.subset_mat.iloc[self.va_idx,:]))
        #self.te_mat = torch.tensor(np.array(self.subset_mat.loc[self.te_idx,:]))

        #self.exp = construct_annData(self.input_mat, channels)
        #self.tr_h5ad = self.exp[idx[:self.sample_size]]
        #self.va_h5ad = self.exp[idx[self.sample_size:(2*self.sample_size)]]
        #self.te_h5ad = self.exp[idx[(2*self.sample_size):]]

        if cell_size:
            tr_fy, tr_fs, tr_fl = simulate_data(self.tr_mat[:,:-1], self.tr_mat[:,-1], overlap)
            self.train_df = ConcatDataset(self.tr_mat[:,:-1], self.tr_mat[:,-1], tr_fy, tr_fs, tr_fl)
        
            va_fy, va_fs, va_fl = simulate_data(self.va_mat[:,:-1], self.va_mat[:,-1], overlap)
            self.val_df = ConcatDataset(self.va_mat[:,:-1], self.va_mat[:,-1], va_fy, va_fs, va_fl)

            #te_fy, te_fs, te_fl = simulate_data(self.te_mat[:,:-1], self.te_mat[:,-1], overlap)
            #self.test_df = ConcatDataset(self.te_mat[:,:-1], self.te_mat[:,-1], te_fy, te_fs, te_fl)
        else:
            tr_fy, _, tr_fl = simulate_data(self.tr_mat)
            self.train_df = ConcatDataset(self.tr_mat, tr_fy, tr_fl)
        
            va_fy, _, va_fl = simulate_data(self.va_mat)
            self.val_df = ConcatDataset(self.va_mat, va_fy, va_fl)

            #te_fy, _, te_fl = simulate_data(self.te_mat)
            #self.test_df = ConcatDataset(self.te_mat, te_fy, te_fl)

def init_clustering(mat, initial_clustering_method, k, training_file_fn=None):
    
    if initial_clustering_method == 'KM':
        kms = KMeans(k).fit(mat)
        init_labels = kms.labels_
        init_label_class = np.unique(init_labels)

        init_centers = kms.cluster_centers_
        init_vars = np.array([np.array(mat)[init_labels == c,:].var(0) for c in init_label_class])
        
    elif initial_clustering_method == 'GMM':
        gmm = GaussianMixture(n_components=k, covariance_type = 'diag').fit(mat)
        init_labels = gmm.predict(mat)

        init_centers = gmm.means_
        init_vars = gmm.covariances_
                 
    elif initial_clustering_method == 'PG':
        init_labels, _, _ = sce.tl.phenograph(mat)
        
        ## save phenograph centers
        nc = len(np.unique(init_labels))
        init_vars = np.zeros((nc, mat.shape[1]))
        init_centers = np.zeros((nc, mat.shape[1]))
        for k in range(nc):
            init_vars[k,:] = mat[init_labels==k].var(0)
            init_centers[k,:] = mat[init_labels==k].mean(0)
    
    elif initial_clustering_method == 'FS':

        ## needs to output to csv first
        #ofn = OPATH + "fs_" + ONAME + ".csv"
        pd.DataFrame(mat.numpy()).to_csv(training_file_fn)
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
        init_labels = np.zeros(fsom.df.shape[0])
        init_vars = np.zeros((len(fsom_class), fsom.df.shape[1]))
        init_centers = np.zeros((len(fsom_class), fsom.df.shape[1]))
        for row in fsom_class:
            init_labels[fsom_labels==row] = i
            init_vars[i,:] = fsom.df[fsom_labels==row].var(0)
            init_centers[i,:] = fsom.df[fsom_labels==row].mean(0)
            i += 1

        init_vars = init_vars[:,:-1]
        init_centers = init_centers[:,:-1]
        #os.remove(ofn)
        
    return init_centers, init_vars, init_labels.astype(str)

def model_paramters(cell_size, init_cen, init_var):
    
    nc = init_cen.shape[0]
    init_var[np.isnan(init_var)] = 0
    #pi_init = np.array([np.mean(init_labels == c) for c in init_label_class])
    pi_init = np.ones(nc) / nc
    tau_init = np.ones((nc, nc))
    tau_init = tau_init / tau_init.sum()

    model_params = {
    'is_pi': np.log(pi_init + 1e-6),
    'is_tau': np.log(tau_init + 1e-6),
    #'is_delta': np.log([0.95, 0.05])
    }

    if cell_size:
        mu_init = init_cen[:,:-1]
        psi_init = init_cen[:,-1]

        sigma_init = init_var[:,:-1]
        omega_init = init_var[:,-1] ** 0.5

        model_params['log_mu'] = np.log(mu_init + 1e-6)
        model_params['log_sigma'] = np.log(sigma_init + 1e-6)

        model_params['log_psi'] = np.log(psi_init + 1e-6)
        model_params['log_omega'] = np.log(omega_init + 1e-6)
    else:    
        model_params['log_mu'] = np.log(init_cen + 1e-6)
        model_params['log_sigma'] = np.log(init_var + 1e-6)
    
    return model_params

class init_setup(pl.LightningDataModule):
    def __init__(self, init_params, tr_df, va_df): #, te_df
        super().__init__()
        self.tr_df = tr_df
        self.va_df = va_df
        #self.te_df = te_df
        self.model_params = {k: torch.from_numpy(v).to(DEVICE).requires_grad_(True) for (k,v) in init_params.items()}

    def train_dataloader(self):
        return DataLoader(self.tr_df, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.va_df, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    #def test_dataloader(self):
    #    return DataLoader(self.te_df, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

class model(pl.LightningModule):
    def __init__(self, model_params, model_dist, model_cell_size, model_overlap, reg_val=1, learning_rate=1e-3):
        super(model, self).__init__()
        self.save_hyperparameters()
        self.model_dist = model_dist
        self.cell_size = model_cell_size
        self.overlap = model_overlap
        self.regularizer = reg_val
        self.learning_rate = learning_rate
        self.indx = 4 if self.cell_size == 1 else 2
        self.model_params = model_params

        # add metrics
        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1Score(number_classes=2, average="micro")
        self.train_auroc = torchmetrics.AUROC(number_classes=2, average="micro")
        
        self.val_acc = torchmetrics.Accuracy()
        self.val_f1 = torchmetrics.F1Score(number_classes=2, average="micro")
        self.val_auroc = torchmetrics.AUROC(number_classes=2, average="micro")
        
        self.test_acc = torchmetrics.Accuracy()
        self.test_f1 = torchmetrics.F1Score(number_classes=2, average="micro")
        self.test_auroc = torchmetrics.AUROC(number_classes=2, average="micro")
        
    def forward(self, batch):
        
        if self.cell_size:
            y, s, fy, fs, fl = batch
            _, _, model_nll, _ = compute_posteriors_nll_p_singlet(y, s, self.model_params, self.model_dist, self.overlap)        
            _, _, _, p_fake_singlet = compute_posteriors_nll_p_singlet(fy, fs, self.model_params, self.model_dist, self.overlap)
        else:
            y, fy, fl = batch
            _, _, model_nll, _ = compute_posteriors_nll_p_singlet(y, None, self.model_params, self.model_dist, self.overlap)        
            _, _, _, p_fake_singlet = compute_posteriors_nll_p_singlet(fy, None, self.model_params, self.model_dist, self.overlap)

        fake_loss = nn.BCELoss()(p_fake_singlet, fl.to(torch.double))
        
        return model_nll, fake_loss, p_fake_singlet
        
    def training_step(self, batch, batch_idx):
        
        #y, s, fy, fs, fl = batch
        model_nll, fake_loss, p_fake_singlet = self(batch)
        
        # total loss
        loss = model_nll + self.regularizer * fake_loss

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
    
    def validation_step(self, batch, batch_idx):
        
        model_nll, fake_loss, p_fake_singlet = self(batch)
        #y, s, fy, fs, fl = batch
        #_, _, model_nll, _ = compute_posteriors_nll_p_singlet(y, s, self.Theta, self.model_dist, self.relax_rule)        
        #_, _, _, p_fake_singlet = compute_posteriors_nll_p_singlet(fy, fs, self.Theta, self.model_dist, self.relax_rule)
        #fake_loss = nn.BCELoss()(p_fake_singlet, fl.to(torch.double))
        
        # total loss
        loss = model_nll + self.regularizer * fake_loss
        #self.log("val_loss", loss)#, prog_bar=True, on_step=False, on_epoch=True) #, logger=True)
        #self.log("val_loss", loss)
        
        # accumulate and return metrics for logging
        self.val_auroc.update(p_fake_singlet, batch[self.indx])
        self.val_acc.update(p_fake_singlet, batch[self.indx])
        self.val_f1.update(p_fake_singlet, batch[self.indx])
        
        self.log("val_auroc", self.val_auroc)                
        self.log("val_accuracy", self.val_acc)
        self.log("val_f1", self.val_f1)
        
        self.log("val_nll", model_nll)
        self.log("val_bce", fake_loss)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True) #, logger=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        
        model_nll, fake_loss, p_fake_singlet = self(batch)
           
        # total loss
        loss = model_nll + self.regularizer * fake_loss
        
        self.log("test_nll", model_nll)
        self.log("test_bce", fake_loss)
        self.log("test_loss", loss, on_step=False, on_epoch=True) #, logger=True)
        
        self.test_auroc.update(p_fake_singlet, batch[self.indx])
        self.test_acc.update(p_fake_singlet, batch[self.indx])
        self.test_f1.update(p_fake_singlet, batch[self.indx])
        
        self.log("test_auroc", self.test_auroc)                
        self.log("test_accuracy", self.test_acc)
        self.log("test_f1", self.test_f1)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model_params.values(), lr=self.learning_rate)
        return optimizer

def starling(df, model_params, dist_option, model_cell_size, model_overlap):
    
    df = df.to(DEVICE)

    with torch.no_grad():
        if model_cell_size:
            singlet_label, singlet_cluster_assig_label, _, prob_singlet, prob_cluster_assig = batch_label(df[:,:-1], df[:,-1], model_params, dist_option, model_cell_size, model_overlap)
        else:
            singlet_label, singlet_cluster_assig_label, _, prob_singlet, prob_cluster_assig = batch_label(df, None, model_params, dist_option, model_cell_size, model_overlap)
            
        ## save damm labels
        labels = np.zeros(df.shape[0], dtype=int) - 1
        labels[singlet_label.cpu() == 0] = singlet_cluster_assig_label.cpu()
        labels = labels.astype(str)
        labels[labels == '-1'] = 'doublets'
        
        if model_cell_size:
            ## DAMM centroids/vars
            centers = torch.hstack([model_params['log_mu'], model_params['log_psi'].reshape(-1,1)]).cpu().detach().exp().numpy()
            vars = torch.hstack([model_params['log_sigma'], model_params['log_omega'].reshape(-1,1)]).cpu().detach().exp().numpy()
        else:
            ## DAMM centroids/vars
            centers = model_params['log_mu'].cpu().detach().exp().numpy()
            vars = model_params['log_sigma'].cpu().detach().exp().numpy()

    return centers, vars, labels, prob_singlet, prob_cluster_assig

def batch_label(Y, S, model_params, dist_option, model_cell_size, model_overlap, batch=5000):
    
    if model_cell_size:
        loader = torch.utils.data.DataLoader(ConcatDataset(Y, S), batch_size = batch, shuffle = False)
    else:
        loader = torch.utils.data.DataLoader(Y, batch_size = batch, shuffle = False)
    
    prob_singlet = []
    prob_singlet_cluster_assig = []
    #prob_doublet_cluster_assig = []
    
    singlet_label = []
    singlet_cluster_assig_label = []
    doublet_cluster_assig_label = []
    with torch.no_grad():
        for i, bat in enumerate(loader):
            if model_cell_size:
                bpsa, bpda, _, bps = compute_posteriors_nll_p_singlet(bat[0], bat[1], model_params, dist_option, model_overlap)
            else:
                bpsa, bpda, _, bps = compute_posteriors_nll_p_singlet(bat, None, model_params, dist_option, model_overlap)
            
            prob_singlet.append(bps)
            prob_singlet_cluster_assig.append(bpsa)
            #prob_doublet_cluster_assig.append(bpda)

            mat = torch.zeros(len(bps)).to(DEVICE); mat[bps <= 0.5] = 1
            singlet_label.append(mat)

            b_singlet_cell_label = bps > 0.5 ## probability being a singlet

            b_pred_singlet_assig_mat = bpsa[b_singlet_cell_label].exp()
            b_pred_singlet_cluster_info = b_pred_singlet_assig_mat.max(1)
            singlet_cluster_assig_label.append(b_pred_singlet_cluster_info.indices)

            b_pred_doublet_assig_mat = bpda[~b_singlet_cell_label].exp() # np.where(p_singlet <= 0.5)[0]
            b_pred_doublet_cluster_info = b_pred_doublet_assig_mat.reshape(-1,bpda.shape[1]**2).max(1)

            for i in range(b_pred_doublet_assig_mat.shape[0]):
                idx = np.argwhere(b_pred_doublet_assig_mat[i].cpu() == b_pred_doublet_cluster_info.values[i].cpu())[0]
                if len(idx) == 1:
                    doublet_cluster_assig_label.append(torch.tensor([idx.item(), idx.item()]))
                else:
                    doublet_cluster_assig_label.append(idx)

    return torch.hstack(singlet_label), torch.hstack(singlet_cluster_assig_label), torch.hstack(doublet_cluster_assig_label), torch.hstack(prob_singlet), torch.vstack(prob_singlet_cluster_assig)

def rf_clustering(input, prob, initial_clustering_method, k, training_file_fn=None):

    mat = input[prob > 0.5] ## cells are singlets (subset)
    labels = np.zeros(input.shape[0], dtype=int) - 1

    if initial_clustering_method == 'KM':
        kms = KMeans(k).fit(mat)
        kms_labels = kms.labels_
        kms_label_class = np.unique(kms_labels)

        labels[np.where(prob > 0.5)[0]] = kms_labels
        labels = labels.astype(str)
        labels[labels == '-1'] = 'doublets'

        cens = kms.cluster_centers_
        vars = np.array([np.array(mat)[kms_labels == c,:].var(0) for c in kms_label_class])

    elif initial_clustering_method == 'GMM':
        gmm = GaussianMixture(n_components=k, covariance_type = 'diag').fit(mat)
        gmm_labels = gmm.predict(mat)
        
        labels[np.where(prob > 0.5)[0]] = gmm_labels
        labels = labels.astype(str)
        labels[labels == '-1'] = 'doublets'        

        cens = gmm.means_
        vars = gmm.covariances_

    elif initial_clustering_method == 'PG':

        pg_labels, _, _ = sce.tl.phenograph(mat)
        
        ## save phenograph labels (in anndata object)
        labels[np.where(prob > 0.5)[0]] = pg_labels
        labels = labels.astype(str)
        labels[labels == '-1'] = 'doublets'

        ## save phenograph centers
        nc = len(np.unique(pg_labels))
        vars = np.zeros((nc, mat.shape[1]))
        cens = np.zeros((nc, mat.shape[1]))
        for k in range(nc):
            vars[k,:] = mat[pg_labels==k].var(0)
            cens[k,:] = mat[pg_labels==k].mean(0)
                
    elif initial_clustering_method == 'FS':
        ## needs to output to csv first
        #ofn = OPATH + "fs_rf_" + ONAME + ".csv"
        pd.DataFrame(mat.numpy()).to_csv(training_file_fn)
        fsom = flowsom(training_file_fn, if_fcs=False, if_drop=True, drop_col=['Unnamed: 0'])
        #fsom = flowsom(fs_fn, if_fcs=False, if_drop=True, drop_col=['Unnamed: 0'])

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
        fs_labels = np.zeros(mat.shape[0], dtype=int) - 1
        cens = np.zeros((len(fsom_class), fsom.df.shape[1]))
        vars = np.zeros((len(fsom_class), fsom.df.shape[1]))
        for row in fsom_class:
            fs_labels[fsom_labels==row] = i
            cens[i,:] = fsom.df[fsom_labels==row].mean(0)
            vars[i,:] = fsom.df[fsom_labels==row].var(0)
            i += 1

        labels[np.where(prob > 0.5)[0]] = fs_labels
        labels = labels.astype(str)
        labels[labels == '-1'] = 'doublets'

        cens = cens[:,:-1]
        vars = vars[:,:-1]

    return cens, vars, labels.astype(str)