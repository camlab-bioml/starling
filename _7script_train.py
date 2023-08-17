import os

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
#from anndata import AnnData

import torchmetrics
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

import core
from sklearn.mixture import GaussianMixture

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:30000"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

k = 20
dist_option = 2
model_overlap = 1
model_cell_size = 1
model_regularizer = 1

## single-cell matrix (h5ad)
tr_adata = ad.read_h5ad('sample_input.h5ad')
X = tr_adata.X.copy()
tr_adata.X = np.arcsinh(tr_adata.X)

## clustering on expressions
gmm = GaussianMixture(n_components = k, covariance_type = 'diag').fit(tr_adata.X)
init_l = gmm.predict(tr_adata.X).astype(int)
init_c = gmm.means_
init_v = gmm.covariances_

## get cell size averge/variance
s = []; sv = [] 
for ii in range(k):
    tmp = tr_adata[np.where(init_l == ii)[0]].obs.area
    s.append(np.mean(tmp))
    sv.append(np.var(tmp))

## setup starling
st = core.ST(tr_adata, np.array(init_c), np.array(init_v), np.array(s), np.array(sv),                  
             model_cell_size, dist_option, model_overlap, model_regularizer)

cb_progress = RichProgressBar()
cb_early_stopping = EarlyStopping(monitor = 'train_loss', mode = 'min', verbose = False)
log_tb = TensorBoardLogger(save_dir='log')
trainer = pl.Trainer(max_epochs = 200, accelerator = 'auto', devices = 'auto', callbacks = [cb_progress, cb_early_stopping], logger=[log_tb])
trainer.fit(st)
st.result()

## save init label and expression in ST object
st.init_l = init_l
st.init_X = X

## output model parameters
torch.save(st, 'model.pt')