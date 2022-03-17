#!/usr/bin/env python3

from func_class import *

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F

import wandb
wandb.login(key='4117bb00bef94e0904c16afed79f1888e0839eb9')

#from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader, random_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

LR = 1e-3
TOL = 1e-3
N_ITER = 100
N_ITER_OPT = 1000

def torch_em(XX, Theta):
    
    lookups = np.triu_indices(nc) # wanted indices
    uwanted = np.tril_indices(nc, -1)
            
    wandb.init(project='torch_em5_20')
    config = wandb.config
    config.lr = LR
    config.tol = TOL
    config.data_type = 'toy'
      
    opt = optim.Adam(Theta.values(), lr=LR)
    
    #Y = torch.tensor(Y)
    #S = torch.tensor(S)
   
    Y = torch.tensor(XX[:,:nf])
    S = torch.tensor(XX[:,nf])
    D = torch.tensor(XX[:,-4]) ##singlet or doublet label (true)
    
    #print('EM trial: {}'.format(trial + 1))

    ls = []
    #qs = []
    for i in range(N_ITER):    
        #print(i)
    
        # E Step:
        with torch.no_grad():
            r, v, L, p_singlet = compute_r_v_2(Y, S, Theta)
            
        # M step (i.e. maximizing Q):
        for j in range(N_ITER_OPT):
        
            opt.zero_grad()
            q = -Q(Theta, Y, S, r, v, None)
            q.backward()
            opt.step()
 
        with torch.no_grad():

            if i % (10 - 1) == 0:
                print("L: {}; {}; {}".format(L.sum(), F.log_softmax(Theta['is_delta'].detach(), 0).exp(), F.log_softmax(Theta['is_pi'].detach(), 0).exp()))
                
            if i > 0 and abs(ls[-1] - L.sum()) < TOL:
                print(L.sum())
                print(F.log_softmax(Theta['is_delta'].detach(), 0).exp())
                break
    
            #qs.append(-q.detach())
            ls.append(L.sum())
    
            #with torch.no_grad():
            tn, fp, fn, tp = confusion_matrix(D, np.where(p_singlet > 0.5, 0, 1)).ravel()
    
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            sensitivity = tp / (tp + fn) ##true positive rate/recall
            specificity = tn / (tn + fp) ##true negative rate
            precision = tp / (tp + fp)
            
            ugt = v[:,lookups[0], lookups[1]].exp()
            lt = v[:,uwanted[0], uwanted[1]].exp()
            ugt[:,lookups[0] != lookups[1]] = ugt[:,lookups[0] != lookups[1]] + lt 
            p_cluster = torch.hstack((ugt, r.exp()))
            cas = np.array(p_cluster.max(1).indices)
                        
            wandb.log({
            'll': L.detach().sum(), 
            'Q': -q.detach().sum(),
            'acc': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'amis': adjusted_mutual_info_score(cas, XX[:,-1]),
            'ars': adjusted_rand_score(cas, XX[:,-1])
            })
    
    #print(Theta0)
        
    #return {'theta': Theta, 'p_singlet': p_singlet}
    return p_singlet, p_cluster
#print("L: {}; pi: {}".format(L.detach().sum(), F.log_softmax(Theta0['is_pi'].detach()).exp()))


def torch_mle(XX, Theta):

    lookups = np.triu_indices(nc) # wanted indices
    uwanted = np.tril_indices(nc, -1)
    
    wandb.init(project='torch_mle5_20')
    config = wandb.config
    config.lr = LR
    config.tol = TOL
    config.data_type = 'toy'
    
    opt = optim.Adam(Theta.values(), lr=LR)
        
    Y = torch.tensor(XX[:,:nf])
    S = torch.tensor(XX[:,nf])
    D = torch.tensor(XX[:,-4]) ##singlet or doublet label (true)
            
    loss = []
    for epoch in range(N_ITER * N_ITER_OPT):
        #print(i)
    
        opt.zero_grad()
        nlls = -ll(Y, S, Theta) #nll
        nlls.backward()
        opt.step()
    
        with torch.no_grad():

            if epoch % (100 - 1) == 0:
                print("L: {}; {}; {}".format(nlls.sum(), F.log_softmax(Theta['is_delta'].detach(), 0).exp(), F.log_softmax(Theta['is_pi'].detach(), 0).exp()))

            if epoch > 0 and abs(loss[-1] - nlls.sum()) < TOL:
                break
        
            loss.append(nlls.sum())
        
            r, v, L, p_singlet = compute_r_v_2(Y, S, Theta)
        
            tn, fp, fn, tp = confusion_matrix(D, np.where(p_singlet > 0.5, 0, 1)).ravel()
    
            accuracy = (tn + tp) / (tn + fp + fn + tp)  
            sensitivity = tp / (tp + fn) ##true positive rate/recall
            specificity = tn / (tn + fp) ##true negative rate
            precision = tp / (tp + fp)   
            
            ugt = v[:,lookups[0], lookups[1]].exp()
            lt = v[:,uwanted[0], uwanted[1]].exp()
            ugt[:,lookups[0] != lookups[1]] = ugt[:,lookups[0] != lookups[1]] + lt 
            p_cluster = torch.hstack((ugt, r.exp()))
            cas = np.array(p_cluster.max(1).indices)            
            #cas = np.array(torch.hstack((ugt, r.exp())).max(1).indices)
            
            wandb.log({
                'nll': nlls.sum(), 
                'acc': accuracy,
                'precision': precision,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'amis': adjusted_mutual_info_score(cas, XX[:,-1]),
                'ars': adjusted_rand_score(cas, XX[:,-1])
                })
        
    #return {'theta': Theta, 'p_singlet': p_singlet}
    return p_singlet, p_cluster

def torch_mle_minibatch(XX, Theta):
    
    lookups = np.triu_indices(nc) # wanted indices
    uwanted = np.tril_indices(nc, -1)
    
    wandb.init(project='torch_mle_mini5_20')
    config = wandb.config
    config.data_type = 'toy'
        
    opt = optim.Adam(Theta.values(), lr=LR)

    #part1 = int(0.7 * XX.shape[0])
    #part2 = int((XX.shape[0] - part1) / 2)
    #part3 = XX.shape[0] - part1 - part2

    #train, valid, test = random_split(torch.tensor(XX), [part1, part2, part3], generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(torch.tensor(XX), batch_size=500, shuffle=True)
    #validloader = DataLoader(valid, batch_size=1280, shuffle=False)
    #testloader = DataLoader(test, batch_size=1280, shuffle=False)
    
    loss = []
    for epoch in range(N_ITER * N_ITER_OPT):
        
        nlls = 0
        tn = 0; fp = 0; fn = 0; tp = 0
        r = []; v = []
        for j, train_batch in enumerate(trainloader):
        
            bY = train_batch[:,:nf]
            bS = train_batch[:,nf]
            bD = train_batch[:,-4]
            bC = train_batch[:,-1]
            
            #bYS = torch.hstack((bY, bS.reshape(-1,1))).float()

            opt.zero_grad()  
            nll = -ll(bY, bS, Theta)
            nll.backward()
            opt.step()
            
            with torch.no_grad():
            
                nlls += nll.sum()
                
                br, bv, bL, p_singlet = compute_r_v_2(bY, bS, Theta)
            
                ## problem is XX p_singlet 256
                #print(confusion_matrix(bD, np.where(p_singlet > 0.5, 0, 1)))
                ctn, cfp, cfn, ctp = confusion_matrix(bD, np.where(p_singlet > 0.5, 0, 1)).ravel()
                tn += ctn; fp += cfp; fn += cfn; tp += ctp
                
                if len(r) == 0 and len(v) == 0:
                    r = np.array(br)
                    v = np.array(bv)
                    c = np.array(bC)
                else:
                    r = np.append(r, np.array(br), 0)
                    v = np.append(v, np.array(bv), 0)
                    c = np.append(c, np.array(bC), 0)

        with torch.no_grad():
            
            if epoch % (100 - 1) == 0:
                print("L: {}; {}; {}".format(nlls.sum(), F.log_softmax(Theta['is_delta'].detach(), 0).exp(), F.log_softmax(Theta['is_pi'].detach(), 0).exp()))

            if epoch > 0 and abs(loss[-1] - nlls.detach()) < TOL:
                break
            
            loss.append(nlls.detach())

            accuracy = (tn + tp) / (tn + fp + fn + tp)
            sensitivity = tp / (tp + fn) ##true positive rate/recall
            specificity = tn / (tn + fp) ##true negative rate
            precision = tp / (tp + fp)   

            ugt = torch.tensor(v[:,lookups[0], lookups[1]]).exp()
            lt = torch.tensor(v[:,uwanted[0], uwanted[1]]).exp()
            ugt[:,lookups[0] != lookups[1]] = ugt[:,lookups[0] != lookups[1]] + lt 
            
            p_cluster = torch.hstack((ugt, torch.tensor(r).exp()))
            cas = np.array(p_cluster.max(1).indices)
            
            #cas = np.array(torch.hstack((ugt, torch.tensor(r).exp())).max(1).indices)
            
            wandb.log({
            'nll': nlls.detach().sum(), 
            'acc': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'amis': adjusted_mutual_info_score(cas, c),
            'ars': adjusted_rand_score(cas, c)
            })

    #return {'theta': Theta, 'p_singlet': p_singlet}
    return p_singlet, p_cluster

def torch_vi(XX, Theta):

    lookups = np.triu_indices(nc) # wanted indices
    uwanted = np.tril_indices(nc, -1)
    
    wandb.init(project='torch_vi5_20')
    config = wandb.config
    config.lr = LR
    config.tol = TOL
    config.data_type = 'toy'
    
    r_net = BasicForwardNet(P, nc, 10, 3)
    v_net = BasicForwardNet(P, nc ** 2, 100, 9)
    d_net = BasicForwardNet(P, 2, 5, 2)

    params = list(Theta.values()) + list(r_net.parameters()) + list(v_net.parameters()) + list(d_net.parameters())
    opt = optim.AdamW(params, lr=LR)

    D = torch.tensor(XX[:,-4]) ##singlet or doublet label (true)

    Y = torch.tensor(XX[:,:nf])
    S = torch.tensor(XX[:,nf])
    YS = torch.hstack((Y,S.reshape(-1,1))).float()
    YS1 = (YS - YS.mean(0)) / YS.std(0)

    loss = []
    for epoch in range(N_ITER * N_ITER_OPT):
    
        opt.zero_grad()
        r, log_r = r_net(YS1)
        v, log_v = v_net(YS1)
        d, log_d = d_net(YS1)
  
        ## row sums to 1 (from neural net)
        log_q0 = log_d[:,0].reshape(-1,1) + log_r ## like r in em version
        log_q1 = log_d[:,1].reshape(-1,1) + log_v ## like v in em version
    
        log_rzd0, log_vgd1 = compute_joint_probs(Theta, Y, S)

        entro = (d * log_d).sum() + (r * log_r).sum() + (v * log_v).sum()
        recon = (log_q0.exp() * log_rzd0).sum() + (log_q1.exp() * log_vgd1).sum()
        nelbo = entro - recon
        nelbo.backward()
        opt.step()
    
        with torch.no_grad():
            
            #if epoch % (100 - 1) == 0:
            #    print("nelbo: {}; {}; {}".format(nelbo.sum(), F.log_softmax(Theta['is_delta'].detach(), 0).exp(), F.log_softmax(Theta['is_pi'].detach(), 0).exp()))
  
            if epoch > 0 and abs(loss[-1] - nelbo.detach().sum()) < TOL:
                print(nelbo.sum())
                print(F.log_softmax(Theta['is_delta'], 0).exp())
                break
           
            loss.append(nelbo.detach())
            
            tn, fp, fn, tp = confusion_matrix(D, np.where(d.T[0] > 0.5, 0, 1)).ravel()
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            sensitivity = tp / (tp + fn) ##true positive rate/recall
            specificity = tn / (tn + fp) ##true negative rate
            precision = tp / (tp + fp)   
            #print(precision)
            
            dr = d[:,0].reshape(-1,1) * r
            dv = (d[:,1].reshape(-1,1) * v).reshape(no, nc, nc)
    
            ugt = dv[:,lookups[0], lookups[1]]
            lt = dv[:,uwanted[0], uwanted[1]]
            ugt[:,lookups[0] != lookups[1]] = ugt[:,lookups[0] != lookups[1]] + lt
            p_cluster = torch.hstack((ugt, dr))
            cas = np.array(p_cluster.max(1).indices)

            wandb.log({
            'entropy': entro.detach(), 
            'reconstruction_loss': recon.detach(),
            'nelbo': nelbo.detach(),
            'acc': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'amis': adjusted_mutual_info_score(cas, XX[:,-1]),
            'ars': adjusted_rand_score(cas, XX[:,-1])
            })        

    #plt.figure()
    #plt.hist(np.array(d.T[0].detach().numpy()))
    #plt.title('d=0 nelbo={}'.format(nelbo.detach()))
    #plt.xlabel('prob')
    #plt.ylabel('obs')
    #plt.show()
    
    #return {'theta': Theta, 'p_singlet': d.T[0]}

    return d, p_cluster

def torch_vi_minibatch(XX, Theta):
    
    lookups = np.triu_indices(nc) # wanted indices
    uwanted = np.tril_indices(nc, -1)
    
    wandb.init(project='torch_vi_mini5_20')
    config = wandb.config
    config.data_type = 'toy'
    
    r_net = BasicForwardNet(P, nc, 10, 3)
    v_net = BasicForwardNet(P, nc ** 2, 100, 9)
    d_net = BasicForwardNet(P, 2, 5, 2)
    
    params = list(Theta.values()) + list(r_net.parameters()) + list(v_net.parameters()) + list(d_net.parameters())
    opt = optim.AdamW(params, lr=LR)
    
    #part1 = int(0.7 * XX.shape[0])
    #part2 = int((XX.shape[0] - part1) / 2)
    #part3 = XX.shape[0] - part1 - part2

    #train, valid, test = random_split(torch.Tensor(XX), [part1, part2, part3], generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(torch.tensor(XX), batch_size=500, shuffle=True)
    #validloader = DataLoader(valid, batch_size=1280, shuffle=False)
    #testloader = DataLoader(test, batch_size=1280, shuffle=False)
    
    loss = []
    for epoch in range(N_ITER * N_ITER_OPT):
        
        DD = []
        nelbo = 0; entro = 0; recon = 0
        tn = 0; fp = 0; fn = 0; tp = 0
        dr = []; dv = []
        for j, train_batch in enumerate(trainloader):
        
            bD = train_batch[:,-4] ##singlet or doublet label (true)
            bC = train_batch[:,-1]

            bY = torch.tensor(train_batch[:,:nf])
            bS = torch.tensor(train_batch[:,nf])
            bYS = torch.hstack((bY, bS.reshape(-1,1))).float()
            bYS1 = (bYS - bYS.mean(0)) / bYS.std(0) ##standardize
            
            opt.zero_grad()
            r, log_r = r_net(bYS1)
            v, log_v = v_net(bYS1)
            d, log_d = d_net(bYS1)
            
            ## row sums to 1 (from neural net)
            log_q0 = log_d[:,0].reshape(-1,1) + log_r ## like r in em version
            log_q1 = log_d[:,1].reshape(-1,1) + log_v ## like v in em version
    
            log_rzd0, log_vgd1 = compute_joint_probs(Theta, bY, bS)
            
            centro = (d * log_d).sum() + (r * log_r).sum() + (v * log_v).sum()
            crecon = (log_q0.exp() * log_rzd0).sum() + (log_q1.exp() * log_vgd1).sum()
            cnelbo = centro - crecon
            cnelbo.backward()
            opt.step()
            
            with torch.no_grad():
                
                nelbo += cnelbo.detach()
                entro += centro.detach()
                recon += crecon.detach()

                ctn, cfp, cfn, ctp = confusion_matrix(bD, np.where(d.T[0] > 0.5, 0, 1)).ravel()
                tn += ctn; fp += cfp; fn += cfn; tp += ctp
                
                DD.append(d.T[0])
                
                tmp1 = np.array(d[:,0].reshape(-1,1) * r)
                tmp2 = np.array((d[:,1].reshape(-1,1) * v).reshape(-1, nc, nc))
                
                #print(tmp1.shape)
                #print(tmp2.shape)
                
                if len(dr) == 0 and len(dv) == 0:    
                    dr = tmp1
                    dv = tmp2
                    c = np.array(bC)
                else:
                    dr = np.append(dr, tmp1, 0)
                    dv = np.append(dv, tmp2, 0)
                    c = np.append(c, np.array(bC))
                    
        with torch.no_grad():
            
            if epoch % (10 - 1) == 0:
                print("nelbo: {}; {}; {}".format(nelbo.sum(), F.log_softmax(Theta['is_delta'], 0).exp(), F.log_softmax(Theta['is_pi'], 0).exp()))
  
            if epoch > 0 and abs(loss[-1] - nelbo.detach().sum()) < TOL:
                print(nelbo.sum())
                print(F.log_softmax(Theta['is_delta'], 0).exp())
                break
           
            loss.append(nelbo.detach())
            
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            sensitivity = tp / (tp + fn) ##true positive rate/recall
            specificity = tn / (tn + fp) ##true negative rate
            precision = tp / (tp + fp)   
            
            ugt = torch.tensor(dv[:,lookups[0], lookups[1]])
            lt = torch.tensor(dv[:,uwanted[0], uwanted[1]])
            ugt[:,lookups[0] != lookups[1]] = ugt[:,lookups[0] != lookups[1]] + lt 
            p_cluster = torch.hstack((ugt, torch.tensor(dr)))
            cas = np.array(p_cluster.max(1).indices)
            
            wandb.log({
            'entropy': entro, 
            'reconstruction_loss': recon,
            'nelbo': nelbo,
            'acc': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'amis': adjusted_mutual_info_score(cas, c),
            'ars': adjusted_rand_score(cas, c)
            })   
        
    #plt.figure()
    #plt.hist(np.array(d.T[0].detach().numpy()))
    #plt.title('d=0 nelbo={}'.format(nelbo.detach()))
    #plt.xlabel('prob')
    #plt.ylabel('obs')
    #plt.show()
    
    #return {'theta': Theta, 'p_singlet': d.T[0]}
    return DD, p_cluster #dr, dv