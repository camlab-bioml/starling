#!/usr/bin/env python3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F

nc = 5; no = 1000; nf = 20; P = nf + 1

## generate toy data
def generateData(n_clusters = 3, n_obs = 10000, n_features = 2):

  #n_clusters = 3; n_obs = 100; n_features = 2
  
  ## set truth expression means/covariances (multivariate) ##
  mu = np.random.rand(n_clusters, n_features)
  # mu = np.sort(mu, 0) ## sort expressions
  sigma = 0.001 * np.identity(n_features) ## variance-covariance matrix

  ## set truth cell size means/variances (univariate) ##
  psi = [np.random.normal(100, 25) for i in range(n_clusters)]
  #psi = np.arange(90, 90 + 5 * n_clusters, 5)
  psi = np.sort(psi, 0)
  omega = 1 ## standard deviation
  ###

  ## set latent variables distributions ##
  lambda_arr = np.random.binomial(1, .95, n_obs) # p=.95 (a cell belongs to singlet or doublet) 

  n_singlet = np.sum(lambda_arr == 1) ## number of cells in singlet clusters
  n_doublet = np.sum(lambda_arr == 0) ## number of cells in doublet clusters
  
  lambda0_arr = n_singlet / n_obs ## proportion of cells belong to singlet
  lambda1_arr = n_doublet / n_obs ## proportion of cells belong to doublet

  #pi_arr = np.sort(np.random.sample(n_clusters))
  pi_arr = np.sort(np.random.rand(n_clusters))
  pi_arr /= pi_arr.sum()

  n_doublet_clusters = int((n_clusters * n_clusters - n_clusters)/2 + n_clusters)
  #tau_arr = np.sort(np.random.sample(n_doublet_clusters))
  tau_arr = np.sort(np.random.rand(n_doublet_clusters))
  tau_arr /= tau_arr.sum()

  ## draw cells based on defined parameters theta1 = (mu, sigma, psi, omega) & theta2 = (lambda, pi, tau)
  x = np.zeros((n_singlet, n_features+5))
  for i in range(n_singlet):
    selected_cluster = np.random.choice(n_clusters, size = 1, p = pi_arr)[0] ## select a single cell cluster
    x[i] = np.append(np.random.multivariate_normal(mu[selected_cluster], sigma),
                     [np.random.normal(psi[selected_cluster], omega), 
                      0, selected_cluster, 0, selected_cluster + n_doublet_clusters])
  
  x[x < 0] = 1e-4
  lookups = np.triu_indices(n_clusters) # wanted indices
  xx = np.zeros((n_doublet, n_features+5))
  for i in range(n_doublet):
    selected_cluster = np.random.choice(n_doublet_clusters, p = tau_arr)

    indx1 = lookups[0][selected_cluster]
    indx2 = lookups[1][selected_cluster]

    xx[i] = np.append(np.random.multivariate_normal( (mu[indx1] + mu[indx2])/2, (sigma + sigma)/2 ),
                     [np.random.normal( (psi[indx1] + psi[indx2]), omega+omega ), 
                      1, indx1, indx2, selected_cluster])
  xx[xx < 0] = 1e-4
  xxx = np.append(x, xx).reshape(n_obs, n_features+5)

  truth_theta = {
    'log_mu': np.log(mu),
    'log_sigma': np.log(sigma),
    'log_psi': np.log(psi),
    'log_omega': np.log(omega),
    "log_lambda0": np.log(lambda0_arr),
    'log_pi': np.log(pi_arr),
    'log_tau': np.log(tau_arr)
  }
  
  print(lambda0_arr)
  print(pi_arr)
  print(tau_arr)


  return xxx[:,:n_features], xxx[:,n_features], xxx, truth_theta

  #return torch.tensor(xxx[:,:n_features]), torch.tensor(xxx[:,n_features]), torch.tensor(xxx), [mu, sigma, psi, omega], [lambda0_arr, pi_arr, tau_arr]

def compute_p_y_given_z(Y, Theta):
  """ Returns NxC
  p(y_n | z_n = c)
  """
  mu = torch.exp(Theta['log_mu'])
  sigma = torch.exp(Theta['log_sigma'])

  dist_Y = D.Normal(mu, sigma)
  return dist_Y.log_prob(Y.reshape(Y.shape[0], 1, nf)).sum(2) # <- sum because IID over G

def compute_p_s_given_z(S, Theta):
  """ Returns NxC
  p(s_n | z_n = c)
  """
  psi = torch.exp(Theta['log_psi'])
  omega = torch.exp(Theta['log_omega'])

  dist_S = D.Normal(psi, omega)
  return dist_S.log_prob(S.reshape(-1,1)) 

def compute_p_y_given_gamma(Y, Theta):
  """ NxCxC
  p(y_n | gamma_n = [c,c'])
  """

  mu = torch.exp(Theta['log_mu'])
  sigma = torch.exp(Theta['log_sigma'])

  mu2 = mu.reshape(1, nc, nf)
  mu2 = (mu2 + mu2.permute(1, 0, 2)) / 2.0 # C x C x G matrix 

  sigma2 = sigma.reshape(1, nc, nf)
  sigma2 = (sigma2 + sigma2.permute(1,0,2)) / 2.0

  dist_Y2 = D.Normal(mu2, sigma2)
  return  dist_Y2.log_prob(Y.reshape(-1, 1, 1, nf)).sum(3) # <- sum because IID over G

def compute_p_s_given_gamma(S, Theta):
  """ NxCxC
  p(s_n | gamma_n = [c,c'])
  """
  psi = torch.exp(Theta['log_psi'])
  omega = torch.exp(Theta['log_omega'])

  psi2 = psi.reshape(-1,1)
  psi2 = psi2 + psi2.T

  omega2 = omega.reshape(-1,1)
  omega2 = omega2 + omega2.T

  dist_S2 = D.Normal(psi2, omega2)
  return dist_S2.log_prob(S.reshape(-1, 1, 1))

def _ics(logL, n_obs, n_features, n_clusters): #, n, p, c
  params = n_clusters * (n_features**2 + n_features + 4) + (n_clusters * n_clusters - n_clusters)/2
  return 2 * (params - logL), -2 * logL + params * np.log(n_obs)

## for MLE version
def ll(Y, S, Theta):
  """compute
  p(gamma = [c,c'], d= 1 | Y,S)
  p(z = c, d=0 | Y,S)
  """
  log_pi = F.log_softmax(Theta['is_pi'], 0)
  log_tau = F.log_softmax(Theta['is_tau'].reshape(-1), 0).reshape(nc,nc)
  log_delta = F.log_softmax(Theta['is_delta'], 0)

  p_y_given_z = compute_p_y_given_z(Y, Theta)
  p_s_given_z = compute_p_s_given_z(S, Theta)

  p_data_given_z_d0 = p_y_given_z + p_s_given_z + log_pi
  p_data_given_d0 = torch.logsumexp(p_data_given_z_d0, dim=1) # this is p(data|d=0)

  p_y_given_gamma = compute_p_y_given_gamma(Y, Theta)
  p_s_given_gamma = compute_p_s_given_gamma(S, Theta)

  p_data_given_gamma_d1 = (p_y_given_gamma + p_s_given_gamma + log_tau).reshape(Y.shape[0], -1)

  # p_data_given_d1 = torch.logsumexp(p_data_given_gamma_d1, dim=1)

  p_data = torch.cat([p_data_given_z_d0 + log_delta[0], p_data_given_gamma_d1 + log_delta[1]], dim=1)
  #p_data = torch.logsumexp(p_data, dim=1)

  #r = p_data_given_z_d0.T + log_delta[0] - p_data
  #v = p_data_given_gamma_d1.T + log_delta[1] - p_data

  #p_singlet = torch.exp(p_data_given_d0 + log_delta[0] - p_data)

  #return r.T, v.T.reshape(-1,nc,nc), -p_data, p_singlet

  return torch.logsumexp(p_data, dim=1).sum()

## for EM version
def compute_r_v_2(Y, S, Theta):
  """Need to compute
  p(gamma = [c,c'], d= 1 | Y,S)
  p(z = c, d=0 | Y,S)
  """
  
  #lookups = np.triu_indices(nc) # wanted indices

  log_pi = F.log_softmax(Theta['is_pi'], 0)
  log_tau = F.log_softmax(Theta['is_tau'].reshape(-1), 0).reshape(nc,nc)
  log_delta = F.log_softmax(Theta['is_delta'], 0)

  p_y_given_z = compute_p_y_given_z(Y, Theta)
  p_s_given_z = compute_p_s_given_z(S, Theta)

  p_data_given_z_d0 = p_y_given_z + p_s_given_z + log_pi
  p_data_given_d0 = torch.logsumexp(p_data_given_z_d0, dim=1) # this is p(data|d=0)

  p_y_given_gamma = compute_p_y_given_gamma(Y, Theta)
  p_s_given_gamma = compute_p_s_given_gamma(S, Theta)

  p_data_given_gamma_d1 = (p_y_given_gamma + p_s_given_gamma + log_tau).reshape(Y.shape[0], -1)

  # p_data_given_d1 = torch.logsumexp(p_data_given_gamma_d1, dim=1)

  p_data = torch.cat([p_data_given_z_d0 + log_delta[0], p_data_given_gamma_d1 + log_delta[1]], dim=1)
  p_data = torch.logsumexp(p_data, dim=1)

  r = p_data_given_z_d0.T + log_delta[0] - p_data
  v = p_data_given_gamma_d1.T + log_delta[1] - p_data

  p_singlet = torch.exp(p_data_given_d0 + log_delta[0] - p_data)

  return r.T, v.T.reshape(-1,nc,nc), p_data, p_singlet #, p_assign, p_assign1

def Q(Theta, Y, S, r, v, ignored_indices):

  log_pi = F.log_softmax(Theta['is_pi'], 0)
  log_tau = F.log_softmax(Theta['is_tau'].reshape(-1), 0).reshape(nc,nc)
  log_delta = F.log_softmax(Theta['is_delta'], 0)

  p_y_given_z = compute_p_y_given_z(Y, Theta)
  p_s_given_z = compute_p_s_given_z(S, Theta)

  log_rd0z = p_s_given_z + p_y_given_z + log_pi + log_delta[0]

  p_y_given_gamma = compute_p_y_given_gamma(Y, Theta)
  p_s_given_gamma = compute_p_s_given_gamma(S, Theta)

  log_rd1g = p_y_given_gamma + p_s_given_gamma + log_tau + log_delta[1] # can use torch.triu to get upper triangle

  #remove_indices = np.tril_indices(nc, -1) ## remove indices
  #log_rd1g[:, remove_indices[0], remove_indices[1]] = float("NaN")

  q1 = log_rd0z * r.exp() #; q1[torch.isnan(q1)] = 0.0
  q2 = log_rd1g * v.exp() #; q2[torch.isnan(q2)] = 0.0

  #print("{} {} {}".format(log_rd1g.shape, v.shape, q2.shape))
  return q1.sum() + q2.sum()

## for nn version
class BasicForwardNet(nn.Module):
  """Encoder for when data is input without any encoding"""
  def __init__(self, input_dim, output_dim, hidden_dim = 20, hideen_layer = 10):
    super().__init__()
    
    self.input = nn.Linear(input_dim, hidden_dim)
    #self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        
    self.linear1 = nn.ModuleList(
        [nn.Linear(hidden_dim, hidden_dim) for i in range(hideen_layer)]
    )

    self.output = nn.Linear(hidden_dim, output_dim)
    
  def forward(self, x):
    out = F.relu(self.input(x))
    
    #out = F.relu(self.linear1(out))

    for net in self.linear1:
      out = F.relu(net(out))
    
    out = self.output(out)
        
    return F.softmax(out, dim=1), F.log_softmax(out, dim=1) ## r/v/d log_r/log_v/log_d

def compute_joint_probs(Theta, Y, S):

  log_pi = F.log_softmax(Theta['is_pi'], 0)
  log_tau = F.log_softmax(Theta['is_tau'].reshape(-1), 0).reshape(nc,nc)
  log_delta = F.log_softmax(Theta['is_delta'], 0)
  
  p_y_given_z = compute_p_y_given_z(Y, Theta)
  p_s_given_z = compute_p_s_given_z(S, Theta)

  log_rzd0 = p_s_given_z + p_y_given_z + log_pi + log_delta[0]

  p_y_given_gamma = compute_p_y_given_gamma(Y, Theta)
  p_s_given_gamma = compute_p_s_given_gamma(S, Theta)

  log_vgd1 = p_y_given_gamma + p_s_given_gamma + log_tau + log_delta[1]

  #remove_indices = np.tril_indices(nc, -1) ## remove indices
  #log_rd1g[:, remove_indices[0], remove_indices[1]] = float("NaN")

  #q1 = r.exp() * log_rd0z #; q1[torch.isnan(q1)] = 0.0
  #q2 = v.exp() * log_rd1g #; q2[torch.isnan(q2)] = 0.0

  return log_rzd0, log_vgd1.reshape(Y.shape[0], nc*nc)