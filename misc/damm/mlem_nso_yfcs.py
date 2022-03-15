import wandb

import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F

#from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader, random_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from scipy.stats.mstats import winsorize

import random

def compute_p_y_given_z(Y, Theta, dist='normal', reg=1e-6):
  
  """ Returns NxC
  p(y_n | z_n = c)
  """
  
  mu = torch.exp(Theta['log_mu'])
  sigma = torch.exp(Theta['log_sigma']) + reg

  if dist == 'normal':
    dist_Y = D.Normal(mu, sigma)
  else:
    dist_Y = D.StudentT(mu, sigma)

  return dist_Y.log_prob(Y.reshape(Y.shape[0], 1, NF)).sum(2) # <- sum because IID over G

def compute_p_s_given_z(S, Theta, dist='normal', reg=1e-6):
  
  """ Returns NxC
  p(s_n | z_n = c)
  """
  
  psi = torch.exp(Theta['log_psi'])
  omega = torch.exp(Theta['log_omega']) + reg

  if dist == 'normal':
    dist_S = D.Normal(psi, omega)
  else:
    dist_S = D.StudentT(psi, omega)

  return dist_S.log_prob(S.reshape(-1,1)) 

def compute_p_y_given_gamma(Y, Theta, dist='normal', reg=1e-6):
  
  """ NxCxC
  p(y_n | gamma_n = [c,c'])
  """

  mu = torch.exp(Theta['log_mu'])
  sigma = torch.exp(Theta['log_sigma']) + reg

  mu2 = mu.reshape(1, NC, NF)
  mu2 = (mu2 + mu2.permute(1, 0, 2)) / 2.0 # C x C x G matrix 

  sigma2 = sigma.reshape(1, NC, NF)
  sigma2 = (sigma2 + sigma2.permute(1,0,2)) / 2.0

  if dist == 'normal':
    dist_Y2 = D.Normal(mu2, sigma2)
  else:
    dist_Y2 = D.StudentT(mu2, sigma2)

  return  dist_Y2.log_prob(Y.reshape(-1, 1, 1, NF)).sum(3) # <- sum because IID over G

def compute_p_s_given_gamma(S, Theta, dist='normal', reg=1e-6):
  
  """ NxCxC
  p(s_n | gamma_n = [c,c'])
  """
  
  psi = torch.exp(Theta['log_psi'])
  omega = torch.exp(Theta['log_omega']) + reg

  psi2 = psi.reshape(-1,1)
  psi2 = psi2 + psi2.T

  omega2 = omega.reshape(-1,1)
  omega2 = omega2 + omega2.T

  if dist == 'normal':
    dist_S2 = D.Normal(psi2, omega2)
  else:
    dist_S2 = D.StudentT(psi2, omega2)

  return dist_S2.log_prob(S.reshape(-1, 1, 1))

def _ics(logL, n_obs, n_features, n_clusters): #, n, p, c
  #params = ( (((n_features * n_features) - n_features)/2 + 2 * n_features + 3) * (((n_clusters * n_clusters) - n_clusters)/2 + 2 * n_clusters) ) - 1
  param_mu = n_clusters * n_features
  param_sigma = n_clusters * n_features
  param_psi = n_clusters
  param_omega = n_clusters
  param_delta = 1
  param_pi = n_clusters - 1
  param_tau = ((n_clusters * n_clusters) - n_clusters)/2 + n_clusters - 1
  params = param_mu + param_sigma + param_psi + param_omega + param_delta + param_pi + param_tau
  return 2 * (params - logL), -2 * logL + params * np.log(n_obs)

def ll(Y, S, Theta, dist):
  
  """compute
  p(gamma = [c,c'], d= 1 | Y,S)
  p(z = c, d=0 | Y,S)
  """

  log_pi = F.log_softmax(Theta['is_pi'], 0)
  log_tau = F.log_softmax(Theta['is_tau'].reshape(-1), 0).reshape(NC,NC)
  log_delta = F.log_softmax(Theta['is_delta'], 0)

  p_y_given_z = compute_p_y_given_z(Y, Theta, dist)
  p_s_given_z = compute_p_s_given_z(S, Theta, dist)

  p_data_given_z_d0 = p_y_given_z + p_s_given_z + log_pi
  p_data_given_d0 = torch.logsumexp(p_data_given_z_d0, dim=1) # this is p(data|d=0)

  p_y_given_gamma = compute_p_y_given_gamma(Y, Theta, dist)
  p_s_given_gamma = compute_p_s_given_gamma(S, Theta, dist)

  p_data_given_gamma_d1 = (p_y_given_gamma + p_s_given_gamma + log_tau).reshape(Y.shape[0], -1)

  # p_data_given_d1 = torch.logsumexp(p_data_given_gamma_d1, dim=1)

  p_data = torch.cat([p_data_given_z_d0 + log_delta[0], p_data_given_gamma_d1 + log_delta[1]], dim=1)
  #p_data = torch.logsumexp(p_data, dim=1)

  return torch.logsumexp(p_data, dim=1).sum()

def compute_r_v_2(Y, S, Theta, dist):
  
  """Need to compute
  p(gamma = [c,c'], d= 1 | Y,S)
  p(z = c, d=0 | Y,S)
  """
  
  #lookups = np.triu_indices(nc) # wanted indices

  log_pi = F.log_softmax(Theta['is_pi'], 0)
  log_tau = F.log_softmax(Theta['is_tau'].reshape(-1), 0).reshape(NC,NC)
  log_delta = F.log_softmax(Theta['is_delta'], 0)

  p_y_given_z = compute_p_y_given_z(Y, Theta, dist)
  p_s_given_z = compute_p_s_given_z(S, Theta, dist)

  p_data_given_z_d0 = p_y_given_z + p_s_given_z + log_pi
  p_data_given_d0 = torch.logsumexp(p_data_given_z_d0, dim=1) # this is p(data|d=0)

  p_y_given_gamma = compute_p_y_given_gamma(Y, Theta, dist)
  p_s_given_gamma = compute_p_s_given_gamma(S, Theta, dist)

  p_data_given_gamma_d1 = (p_y_given_gamma + p_s_given_gamma + log_tau).reshape(Y.shape[0], -1)

  # p_data_given_d1 = torch.logsumexp(p_data_given_gamma_d1, dim=1)

  p_data = torch.cat([p_data_given_z_d0 + log_delta[0], p_data_given_gamma_d1 + log_delta[1]], dim=1)
  p_data = torch.logsumexp(p_data, dim=1)

  r = p_data_given_z_d0.T + log_delta[0] - p_data
  v = p_data_given_gamma_d1.T + log_delta[1] - p_data

  p_singlet = torch.exp(p_data_given_d0 + log_delta[0] - p_data)

  return r.T, v.T.reshape(-1,NC,NC), p_data, p_singlet #, p_assign, p_assign1

def torch_mle_minibatch(Y, S, Theta, dist):
  
  wandb.init(project="mlem_{}_nc{}".format(PROJECT_NAME, NC))
  
  lookups = np.triu_indices(NC) # wanted indices
  uwanted = np.tril_indices(NC, -1)
  
  opt = optim.Adam(Theta.values(), lr=LEARNING_RATE)

  XX = torch.hstack((Y, S.reshape(-1,1))).float()
  trainloader = DataLoader(torch.tensor(XX), batch_size=BATCH_SIZE, shuffle=True)
  #validloader = DataLoader(valid, batch_size=1280, shuffle=False)
  #testloader = DataLoader(test, batch_size=1280, shuffle=False)
    
  loss = []
  for epoch in range(N_ITER * N_ITER_OPT):
    
    nlls = 0
    for j, train_batch in enumerate(trainloader):
      
      bY = train_batch[:,:NF]
      bS = train_batch[:,NF]
      
      opt.zero_grad()  
      nll = -ll(bY, bS, Theta, dist)
      nll.backward()
      opt.step()
            
      nlls += nll

    with torch.no_grad():
      
      aic, bic = _ics(-nlls, Y.shape[0], NF, NC) #, n, p, c

      wandb.log({
        'nll': nlls, 
        'AIC': aic,
        'BIC': bic,
      })

      if epoch > 15 and abs(np.mean(loss[-10:]) - np.mean(loss[-11:-1])) < TOL:
        print(nlls)
        print(F.log_softmax(Theta['is_delta'], 0).exp())
        print(Theta['log_psi'].exp())
        break
            
      loss.append(nlls)
    
  with torch.no_grad():
    
    r, v, L, p_singlet = compute_r_v_2(Y, S, Theta, dist)

    ugt = torch.tensor(v[:,lookups[0], lookups[1]]).exp()
    lt = torch.tensor(v[:,uwanted[0], uwanted[1]]).exp()
    ugt[:,lookups[0] != lookups[1]] = ugt[:,lookups[0] != lookups[1]] + lt             
    p_cluster = torch.hstack((ugt, torch.tensor(r).exp()))

  #return {'theta': Theta, 'p_singlet': p_singlet}
  return p_singlet, p_cluster

parser = argparse.ArgumentParser(description='Pass in array id')
parser.add_argument("--trial_id", type=int, help="number of trial for this model")

args = parser.parse_args()
print(args.trial_id)

#PATH = '/Users/jettlee/Desktop/DAMM/'
PATH = '/home/campbell/yulee/DAMM/'

import scanpy as sc

'''
#adata = sc.read_h5ad("{}data/mouse_1000.h5ad".format(PATH))
#adata = sc.read_h5ad("{}data/mouse_5000.h5ad".format(PATH))
adata = sc.read_h5ad("{}data/mouse_single_cell_expression.h5ad".format(PATH))
included_names = ['B220', 'CCR7', 'CD11b', 'CD11c', 'CD19', 'CD28', 'CD3', 'CD31', 'CD4',
 'CD45', 'CD49b', 'CD68', 'CD73', 'CD8', 'CTLA4', 'FOXP3', 'GATA3', 'GFP', 
 'GranzymeB', 'HA', 'ICOS', 'IL7Ra', 'Ly6G', 'MHCII', 'PD1', 'PDL1', 'PNAd', 
 'Perforin', 'RFP', 'S100A8-9', 'TBET', 'TCF1', 'YAP', 'iNOS']

'''

#adata = sc.read_h5ad("{}data/human_single_cell_expression.h5ad".format(PATH))
#adata = sc.read_h5ad("{}data/basel_zuri_subsample.h5ad".format(PATH))
adata = sc.read_h5ad("{}data/basel_zuri.h5ad".format(PATH))
included_names = ['EGFR', 'ECadherin', 'ER', 'GATA3', 'Histone_H3_1', 'Ki67', 'SMA', 
'Vimentin', 'cleaved_Parp', 'Her2', 'p53', 'panCytokeratin', 'CD19', 'PR', 'Myc', 
'Fibronectin', 'CK14', 'Slug', 'CD20', 'vWF', 'Histone_H3_2', 'CK5', 'CD44', 'CD45', 
'CD68', 'CD3', 'CAIX', 'CK8/18', 'CK7', 'phospho Histone', 'phospho S6', 'phospho mTOR']

adata = adata[:,included_names]

YY = adata.X
YY = np.array(np.arcsinh(YY / 5.))

NO, NF = YY.shape #number obs & features 

for i in range(NF):
  YY[:,i] = winsorize(YY[:,i], limits=[0, 0.01]).data

#SS = adata.obs['size']
SS = adata.obs['Area']
SS = winsorize(SS, limits=[0, 0.01]).data

#PROJECT_NAME = 'mouse_nso_ycs_stuT'
#PROJECT_NAME = 'mouse_nso_ycs_norm'

PROJECT_NAME = 'human_nso_ycs_stuT'
#PROJECT_NAME = 'human_nso_ycs_norm'

NC = 35 # number of clusters
LEARNING_RATE = 1e-3
BATCH_SIZE = 1280
N_ITER = 10000
N_ITER_OPT = 500
TOL = 1e-3 #converagence criterion

## w&b api key
wandb.login(key='4117bb00bef94e0904c16afed79f1888e0839eb9')

Y = torch.tensor(YY)
S = torch.tensor(SS)

kms = KMeans(NC).fit(Y)
init_labels = kms.labels_
init_label_class = np.unique(init_labels)

mu_init = np.array([YY[init_labels == c,:].mean(0) for c in init_label_class])
sigma_init = np.array([YY[init_labels == c,:].std(0) for c in init_label_class])

psi_init = np.array([SS[init_labels == c].mean() for c in init_label_class])
omega_init = np.array([SS[init_labels == c].std() for c in init_label_class])

pi_init = np.array([np.mean(init_labels == c) for c in init_label_class])
tau_init = np.ones((NC,NC))
tau_init = tau_init / tau_init.sum()

Theta = {
    'log_mu': np.log(mu_init + 1e-6),
    'log_sigma': np.log(sigma_init + 1e-6), #np.zeros_like(sigma_init),
    'log_psi': np.log(psi_init + 1e-6),
    'log_omega': np.log(omega_init + 1e-6),
    'is_delta': np.array([0.5, 0.5]),
    'is_pi': pi_init,
    'is_tau': tau_init,
}

Theta = {k: torch.tensor(v, requires_grad=True) for (k,v) in Theta.items()}
#Theta['log_psi'].requires_grad = False

torch.save(Theta, "{}res/mlesm/theta0_{}_nc{}_{}".format(PATH, PROJECT_NAME, NC, args.trial_id))
mle1 = torch_mle_minibatch(Y, S, Theta, dist='student')
#mle1 = torch_mle_minibatch(Y, S, Theta, dist='normal')
torch.save(mle1, "{}res/mlesm/p_doublet_{}_nc{}_{}".format(PATH, PROJECT_NAME, NC, args.trial_id)) ##torch.load
torch.save(Theta, "{}res/mlesm/theta1_{}_nc{}_{}".format(PATH, PROJECT_NAME, NC, args.trial_id))