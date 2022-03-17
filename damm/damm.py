import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class ConcatDataset(torch.utils.data.Dataset):
  def __init__(self, *datasets):
    self.datasets = datasets

  def __getitem__(self, i):
    return tuple(d[i] for d in self.datasets)

  def __len__(self):
    return min(len(d) for d in self.datasets)

class damm:

    def __init__(self, h5ad_dset, soMat=None, numOfCluster=15, incCellSize=None, initMethod='gmm', noiseModel='student', 
    spilloverRate=0, relaxRule=True, regularizer=10, batchSize=128, learnRate=1e-3, maxEpoch=150):

        self.h5ad_dset = h5ad_dset
        self.incCellSize = incCellSize
        #self.trY = None
        #self.trS = None
        #self.trFY = None
        #self.trFS = None
        #self.trFL = None
        #self.train = None
        self.soMat = soMat

        self.numOfCluster = numOfCluster
        self.batchSize = batchSize
        self.learnRate = learnRate
        self.maxEpoch = maxEpoch

        self.initMethod = initMethod
        self.noiseModel = noiseModel
        self.relaxRule = relaxRule
        self.regularizer = regularizer
        self.spilloverRate = spilloverRate

    def prepData(self):
    
        Y = torch.tensor(self.h5ad_dset.X.copy()) ## expressions
    
        if self.incCellSize:
            S = torch.tensor(self.h5ad_dset.obs['area'])
            return Y, S
        else:
            return Y, None
    
    def simulate_data(self): ## use real data to simulate singlets/doublets
  
        ''' return same number of cells as in Y/S, half of them are singlets and another half are doublets '''

        #N_training = 5000
        sample_size = int(self.trY.shape[0]/2)
        idx_singlet = np.random.choice(self.trY.shape[0], size = sample_size, replace=True)
        Y_singlet = self.trY[idx_singlet,:] ## expression
        
        idx_doublet = [np.random.choice(self.trY.shape[0], size = sample_size), np.random.choice(self.trY.shape[0], size = sample_size)]
        Y_doublet = (self.trY[idx_doublet[0],:] + self.trY[idx_doublet[1],:])/2.
        
        fake_Y = torch.tensor(np.vstack([Y_singlet, Y_doublet]))
        fake_label = torch.tensor(np.concatenate([np.ones(sample_size), np.zeros(sample_size)]))

        if self.trS is None:
            return fake_Y, None, fake_label
        else:
            S_singlet = self.trS[idx_singlet]
            if self.relaxRule:
                dmax = torch.vstack([self.trS[idx_doublet[0]], self.trS[idx_doublet[1]]]).max(0).values
                dsum = self.trS[idx_doublet[0]] + self.trS[idx_doublet[1]]
                rr_dist = D.Uniform(dmax.type(torch.float64), dsum.type(torch.float64))
                S_doublet = rr_dist.sample()
            else:
                S_doublet = self.trS[idx_doublet[0]] + self.trS[idx_doublet[1]]  
            fake_S = torch.tensor(np.hstack([S_singlet, S_doublet]))
            return fake_Y, fake_S, fake_label ## have cell size and create fake cell size

    def damm_init(self):

        self.trY, self.trS = DAMM.prepData(self)
        trMat = np.hstack((self.trY, self.trS.reshape(-1,1)))
        
        self.trFY, self.trFS, self.trFL = DAMM.simulate_data(self)
        #trFMat = np.hstack((trFY, trFS.reshape(-1,1)))
    
        if self.initMethod == 'kmeans':
            kms = KMeans(self.numOfCluster).fit(trMat)
            init_labels = kms.labels_
            init_label_class = np.unique(init_labels)
            
            init_centers = kms.cluster_centers_
            init_var = np.array([np.array(trMat)[init_labels == c,:].var(0) for c in init_label_class])        
        elif self.initMethod == 'gmm':
            gmm = GaussianMixture(n_components = self.numOfCluster, covariance_type = 'diag').fit(trMat)
            init_labels = gmm.predict(trMat)
            init_label_class = np.unique(init_labels)
            
            init_centers = gmm.means_
            init_var = gmm.covariances_
            
        #pi_init = np.array([np.mean(init_labels == c) for c in init_label_class])
        pi_init = np.ones(self.numOfCluster) / self.numOfCluster
        tau_init = np.ones((self.numOfCluster, self.numOfCluster))
        tau_init = tau_init / tau_init.sum()

        Theta = {
        'is_pi': np.log(pi_init + 1e-6),
        'is_tau': np.log(tau_init + 1e-6),
        }

        if self.incCellSize:
            mu_init = init_centers[:,:-1]
            psi_init = init_centers[:,-1]

            sigma_init = init_var[:,:-1]
            omega_init = init_var[:,-1] ** 0.5

            Theta['log_mu'] = np.log(mu_init + 1e-6)
            Theta['log_sigma'] = np.log(sigma_init + 1e-6)

            Theta['log_psi'] = np.log(psi_init + 1e-6)
            Theta['log_omega'] = np.log(omega_init + 1e-6)
        else:    
            Theta['log_mu'] = np.log(init_centers + 1e-6)
            Theta['log_sigma'] = np.log(init_var + 1e-6)

        if self.spilloverRate != 0:
            Theta['is_so'] = np.random.rand(trMat.shape[0])

        Theta = {k: torch.from_numpy(v).requires_grad_(True) for (k,v) in Theta.items()}
        
        if self.spilloverRate != 0:
            Theta['mnu'] = torch.tensor(mu_init)
            Theta['mnu'].requires_grad = False
        
        self.Theta = Theta
        return init_labels, init_centers

    def compute_p_y_given_z(self, Y, sRate):
  
        if sRate == 0:
            mu = torch.exp(self.Theta['log_mu'])
        else:       
            l = torch.sigmoid(self.Theta['is_so']) * sRate
            first_half = (1 - l).reshape(-1, 1, 1) * torch.exp(self.Theta['log_mu'])
            second_half = (l.reshape(-1, 1) * self.soMat).reshape(-1, 1, self.trY.shape[1])
            mu = (first_half + second_half).mean(0)
            self.Theta['mnu'] = mu

        sigma = torch.exp(self.Theta['log_sigma']) + 1e-6

        if self.noiseModel == 'normal':
            dist_Y = D.Normal(loc = mu, scale = sigma)
        elif self.noiseModel == 'student':
            dist_Y = D.StudentT(df = 2.0, loc = mu, scale = sigma)
        else:
            return "Please input the correct noise distribution"
        
        return dist_Y.log_prob(Y.reshape(-1, 1, Y.shape[1])).sum(2) # <- sum because IID over G

    def compute_p_s_given_z(self, S):
        """ Returns NxC
        p(s_n | z_n = c)
        """
        #print(Theta['log_psi'])
        psi = torch.exp(self.Theta['log_psi'])
        omega = torch.exp(self.Theta['log_omega']) + 1e-6

        if self.noiseModel == 'normal':
            dist_S = D.Normal(loc = psi, scale = omega)
        elif self.noiseModel == 'student':
            dist_S = D.StudentT(df = 2.0, loc = psi, scale = omega)
        else:
            return "Please input the correct noise distribution"

        return dist_S.log_prob(S.reshape(-1,1)) 

    def compute_p_y_given_gamma(self, Y, sRate):
        """ NxCxC
        p(y_n | gamma_n = [c,c'])
        """
        if sRate == 0:
            mu = torch.exp(self.Theta['log_mu'])
        else:
            mu = self.Theta['mnu']

        sigma = torch.exp(self.Theta['log_sigma'])

        mu2 = mu.reshape(1, mu.shape[0], mu.shape[1])
        mu2 = (mu2 + mu2.permute(1,0,2)) / 2.0 # C x C x G matrix 

        sigma2 = sigma.reshape(1, mu.shape[0], mu.shape[1])
        sigma2 = (sigma2 + sigma2.permute(1,0,2)) / 2.0 + 1e-6

        dist_Y2 = D.Normal(mu2, sigma2)

        if self.noiseModel == 'normal':
            dist_Y2 = D.Normal(loc = mu2, scale = sigma2)
        elif self.noiseModel == 'student':
            dist_Y2 = D.StudentT(df = 2.0, loc = mu2, scale = sigma2)
        else:
            return "Please input the correct noise distribution"

        return dist_Y2.log_prob(Y.reshape(-1, 1, 1, mu.shape[1])).sum(3) # <- sum because IID over G

    def compute_p_s_given_gamma(self, S):

        psi = torch.exp(self.Theta['log_psi'])
        omega = torch.exp(self.Theta['log_omega'])

        psi2 = psi.reshape(-1,1)
        psi2 = psi2 + psi2.T

        omega2 = omega.reshape(-1,1)
        omega2 = omega2 + omega2.T + 1e-6
        
        if self.relaxRule:
            ## for v
            ccmax = torch.combinations(psi).max(1).values
            mat = torch.zeros(len(psi), len(psi), dtype=torch.float64)    
            mat[np.triu_indices(len(psi), 1)] = ccmax
            mat += mat.clone().T
            #mat = mat + mat.T
            mat += torch.eye(len(psi)) * psi
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
            prob = 1e-6 + (const * (ubp - lbp))
            return prob.log()
        else:
            if self.noiseModel == 'normal':
                dist_S2 = D.Normal(loc = psi2, scale = omega2)
            elif self.noiseModel == 'student':
                dist_S2 = D.StudentT(df = 2.0, loc = psi2, scale = omega2)
            return dist_S2.log_prob(S.reshape(-1, 1, 1))

    def compute_posteriors_nll_p_singlet(self, Y, S, sRate):

        log_pi = F.log_softmax(self.Theta['is_pi'], dim=0)
        log_tau = F.log_softmax(self.Theta['is_tau'].reshape(-1), dim=0).reshape(log_pi.shape[0], log_pi.shape[0])
        #log_delta = F.log_softmax(Theta['is_delta'], dim=0)
        log_delta = F.log_softmax(torch.tensor(np.log([0.95, 0.05])), dim=0)

        prob_y_given_z = DAMM.compute_p_y_given_z(self, Y, sRate) ## p(y_n|z=c)
        prob_data_given_z_d0 = prob_y_given_z + log_pi
        
        if S is not None:
            prob_s_given_z = DAMM.compute_p_s_given_z(self, S) ## p(data_n|z=c)
            prob_data_given_z_d0 += prob_s_given_z ## p(data_n|z=c,d=0) -> NxC
        
        prob_y_given_gamma = DAMM.compute_p_y_given_gamma(self, Y, sRate) ## p(y_n|g=[c,c']) -> NxCxC
        prob_data_given_gamma_d1 = prob_y_given_gamma + log_tau
        
        if S is not None:
            prob_s_given_gamma = DAMM.compute_p_s_given_gamma(self, S) ## p(s_n|g=[c,c']) -> NxCxC
            prob_data_given_gamma_d1 += prob_s_given_gamma ## p(data_n|d=1) -> NxCxC

        #p_data = torch.cat([prob_data_given_z_d0 + log_delta[0], prob_data_given_gamma_d1.reshape(X.shape[0], -1) + log_delta[1]], dim=1)
        prob_data = torch.hstack([prob_data_given_z_d0 + log_delta[0], prob_data_given_gamma_d1.reshape(Y.shape[0], -1) + log_delta[1]]) ## p(data)
        prob_data = torch.logsumexp(prob_data, dim=1)

        ## average negative likelihood scores
        #cost = -prob_data.sum()
        cost = -prob_data.mean()

        r = prob_data_given_z_d0.T + log_delta[0] - prob_data ## p(d=0,z=c|data)
        v = prob_data_given_gamma_d1.T + log_delta[1] - prob_data ## p(gamma=[c,c']|data)

        ## normalize
        prob_data_given_d0 = torch.logsumexp(prob_data_given_z_d0, dim=1) ## p(data_n|d=0)_N
        prob_singlet = torch.exp(prob_data_given_d0 + log_delta[0] - prob_data)

        return r.T, v.T, cost, prob_singlet

    def fit(self):
    
        opt = optim.Adam(self.Theta.values(), lr = self.learnRate)
        df = ConcatDataset(self.trY, self.trS, self.trFY, self.trFS, self.trFL)
        trainloader = torch.utils.data.DataLoader(df, batch_size = self.batchSize, shuffle = True)
        
        loss = []
        for epoch in range(self.maxEpoch):
        
            #print(epoch)
            rnlls = 0; fnlls = 0; floss = 0; tloss = 0
            for j, bat in enumerate(trainloader):
                
                #bFY = bat[2], bFS = bat[3], bFL = bat[4]
                opt.zero_grad()
                
                _, _, real_nll, _ = DAMM.compute_posteriors_nll_p_singlet(self, bat[0], bat[1], self.spilloverRate)
                _, _, fake_nll, p_fake_singlet = DAMM.compute_posteriors_nll_p_singlet(self, bat[2], bat[3], 0)
                
                fake_loss = nn.BCELoss()(p_fake_singlet, bat[4]) ## want to min
            
                closs = real_nll + self.regularizer * fake_loss
                closs.backward()
                opt.step()
                
                tloss += closs.item()
                rnlls += real_nll.item()
                fnlls += fake_nll.item()
                floss += fake_loss.item()
            
            with torch.no_grad():
                
                loss.append([-tloss, -rnlls, -fnlls, floss])
                            
                #if epoch % 100 == 0:
                #  print('Epoch: {}: total ll: {} real ll: {} fake ll: {} fake loss: {}'.format(epoch, loss[-1][0], loss[-1][1], loss[-1][2], loss[-1][3]))
                #if epoch > 10 and abs(np.mean(loss[-5:]) - np.mean(loss[-6:-1])) < TOL:
                #  break
        return loss, self.Theta

    def get_labels(self): # #Y, S, Theta, noiseModel, rRule, sRate, sMat):
    
        df = ConcatDataset(self.trY, self.trS)    
        pred_loader = torch.utils.data.DataLoader(df, batch_size = 5000, shuffle = False)

        pred_singlet_label = []
        pred_singlet_cluster_assig_label = []
        pred_doublet_cluster_assig_label = []
        with torch.no_grad():
            for i, bat in enumerate(pred_loader):
                bpsa, bpda, _, bps = DAMM.compute_posteriors_nll_p_singlet(self, bat[0], bat[1], self.spilloverRate)
                
                temp_mat = np.zeros(len(bps)); temp_mat[np.where(bps <= 0.5)[0]] = 1
                pred_singlet_label.append(temp_mat)

                b_singlet_cell_label = np.where(bps > 0.5)[0]
                
                b_pred_singlet_assig_mat = bpsa[b_singlet_cell_label].exp()
                b_pred_singlet_cluster_info = b_pred_singlet_assig_mat.max(1)
                #print(np.unique(b_pred_singlet_cluster_info.indices))
                pred_singlet_cluster_assig_label.append(b_pred_singlet_cluster_info.indices)
                
                b_pred_doublet_assig_mat = bpda[~b_singlet_cell_label].exp() # np.where(p_singlet <= 0.5)[0]
                b_pred_doublet_cluster_info = b_pred_doublet_assig_mat.reshape(-1,bpda.shape[1]**2).max(1)
                
                for i in range(b_pred_doublet_assig_mat.shape[0]):
                    idx = np.argwhere(b_pred_doublet_assig_mat[i] == b_pred_doublet_cluster_info.values[i])[0]
                    if len(idx) == 1:
                        pred_doublet_cluster_assig_label.append(torch.tensor([idx.item(), idx.item()]))
                    else:
                        pred_doublet_cluster_assig_label.append(idx)
                
        return np.hstack(pred_singlet_label), np.hstack(pred_singlet_cluster_assig_label), np.hstack(pred_doublet_cluster_assig_label)

