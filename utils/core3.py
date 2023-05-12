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

class STL4(pl.LightningModule):
    def __init__(self, adata,
                 init_c,
                 init_v,
                 init_l,
                 keep_c,
                 model_cell_size=1, 
                 cofactor=1,
                 dist_option=2, 
                 model_overlap=1, 
                 lambda1=1,
                 lambda2=1, 
                 ofn=None,
                 learning_rate=1e-3,
                 ):
        super().__init__()
        
        #self.save_hyperparameters()
        
        self.adata = adata
        self.init_c = init_c
        self.init_v = init_v
        self.init_l = init_l
        self.keep_c = keep_c
        self.cofactor = cofactor
        self.dist_option = dist_option
        self.model_cell_size = model_cell_size
        self.model_overlap = model_overlap
        self.lambda1 = lambda1
        self.lambda2 = lambda2
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
        loss = model_nll + self.lambda1 * fake_loss + self.lambda2 * torch.mean(torch.abs(torch.exp(self.model_params['log_mu'])))

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
    
        if self.model_cell_size == 1:
            self.init_s = []; self.init_sv = []
            for nc in np.unique(self.init_l):
                self.init_s.append(self.adata.obs.iloc[np.where(self.init_l == nc)]['area'].mean(0))
                self.init_sv.append(self.adata.obs.iloc[np.where(self.init_l == nc)]['area'].var(0))

            #print(self.init_s)
            #print(self.init_sv)            
            self.init_s = np.array(self.init_s)
            self.init_sv = np.array(self.init_sv)
            self.init_c = np.array(self.init_c)
            self.init_v = np.array(self.init_v)

            #print(self.init_c)
            #print(self.init_v)
            #print(self.init_s)
            #print(self.init_sv)

            #print(self.init_c.shape)
            #print(self.init_v.shape)
            #print(self.init_s.shape)
            #print(self.init_sv.shape)

            #self.init_sv[np.isnan(self.init_sv)] = 0
            #model_params = model_paramters(self.init_c, self.init_v, self.init_s, torch.tensor(np.array(self.init_sv)))
            model_params = model_paramters(self.init_c, self.init_v, self.init_s, torch.tensor(self.init_sv))
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