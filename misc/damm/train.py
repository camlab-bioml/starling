#!/usr/bin/env python3

from algo import *
from func_class import *

#from matplotlib.backends.backend_pdf import PdfPages
#import seaborn as sns

#import sklearn.datasets as datasets

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

#nc = 5; no = 1000; nf = 20
#Y, S, XX, theta_true = generateData(n_clusters = nc, n_obs = no, n_features = nf)

path = '/Users/jettlee/Desktop/DAMM/'
#path = '/home/campbell/yulee/DAMM/'

## load data
Y = np.loadtxt('{}data/toy_expressions5_20.csv'.format(path), delimiter=',')
S = np.loadtxt('{}data/toy_cellsizes5_20.csv'.format(path), delimiter=',')
XX = np.loadtxt('{}data/toy_all5_20.csv'.format(path), delimiter=',')

#pip install scanpy
#import scanpy as sc
#adata = sc.read_h5ad("{}data/basel_zuri_subsample.h5ad".format(path))
#adata = sc.read_h5ad("{}data/basel_zuri.h5ad".format(path))

#Y = np.arcsinh(adata.X / 5.)
#S = adata.obs.Area
#no, nf = Y.shape
#nc = 5

nt = 10

em_dicts = []
mle1_dicts = []
mle2_dicts = []
vi1_dicts = []
vi2_dicts = []
    
for trial in range(nt):
    
    Y = np.array(Y)
    S = np.array(S)    
    
    #kms = KMeans(nc).fit(np.hstack((Y,S.reshape(-1,1))))
    
    kms = KMeans(nc).fit(Y)
    init_labels = kms.labels_
    init_label_class = np.unique(init_labels)

    mu_init = np.array([Y[init_labels == c,:].mean(0) for c in init_label_class])
    sigma_init = np.array([Y[init_labels == c,:].std(0) for c in init_label_class])
    psi_init = np.array([S[init_labels == c].mean() for c in init_label_class])
    omega_init = np.array([S[init_labels == c].std() for c in init_label_class])
    pi_init = np.array([np.mean(init_labels == c) for c in init_label_class])
    tau_init = np.ones((nc,nc))
    tau_init = tau_init / tau_init.sum()

    Theta = {
    'log_mu': np.log(mu_init), #+ 0.05 * np.random.randn(mu_init.shape[0], mu_init.shape[1])
    'log_sigma': np.log(sigma_init), #np.zeros_like(sigma_init),
    'log_psi': np.log(psi_init),
    'log_omega': np.log(omega_init),
    "is_delta": F.log_softmax(torch.tensor([0.95, 1-0.95]), 0),
    'is_pi': F.log_softmax(torch.tensor(pi_init), 0),
    'is_tau': F.log_softmax(torch.tensor(tau_init), 0)
    }
    
    Theta0 = {k: torch.tensor(v, requires_grad=True) for (k,v) in Theta.items()}
    Theta1 = {k: torch.tensor(v, requires_grad=True) for (k,v) in Theta.items()}
    Theta2 = {k: torch.tensor(v, requires_grad=True) for (k,v) in Theta.items()}
    Theta3 = {k: torch.tensor(v, requires_grad=True) for (k,v) in Theta.items()}
    Theta4 = {k: torch.tensor(v, requires_grad=True) for (k,v) in Theta.items()}
    
    ## em
    print("EM")
    #em_dicts.append(torch_em(XX, Theta0))
    em = torch_em(XX, Theta0)
    torch.save(em, "{}res/EM_p_doublet{}5_20".format(path, trial)) ##torch.load
    torch.save(Theta0, "{}res/EM_theta{}5_20".format(path, trial))
    
    ## mle
    print("MLE1")
    #mle1_dicts.append(torch_mle(XX, Theta1))
    mle1 = torch_mle(XX, Theta1)
    torch.save(mle1, "{}res/MLE1_p_doublet{}5_20".format(path, trial)) ##torch.load
    torch.save(Theta1, "{}res/MLE1_theta{}5_20".format(path, trial))

    ## mle (minibatch)
    print("MLE2")
    #mle2_dicts.append(torch_mle_minibatch(XX, Theta2))
    mle2 = torch_mle_minibatch(XX, Theta2)
    torch.save(mle2, "{}res/MLE2_p_doublet{}5_20".format(path, trial)) ##torch.load
    torch.save(Theta2, "{}res/MLE2_theta{}5_20".format(path, trial))

    ## vi
    print("VI1")
    #vi1_dicts.append(torch_vi(XX, Theta3))
    vi1 = torch_vi(XX, Theta3)
    torch.save(vi1, "{}res/VI1_p_doublet{}5_20".format(path, trial)) ##torch.load
    torch.save(Theta3, "{}res/VI1_theta{}5_20".format(path, trial))

    ## vi (minibatch)
    print("VI2")
    #vi2_dicts.append(torch_vi_minibatch(XX, Theta4))
    vi2 = torch_vi_minibatch(XX, Theta4)
    torch.save(vi2, "{}res/VI2_p_doublet{}5_20".format(path, trial)) ##torch.load
    torch.save(Theta4, "{}res/VI2_theta{}5_20".format(path, trial))
    
    ## cell label (0 == singlet; 1 == doublet)
    #np.where(em_dicts[0]['p_singlet'] > 0.5, 0, 1) == XX[:,-4]
    
    ## cluster label    
    #XX[:,-1]
    #torch.max(em_dicts[0]['p_assign'], 1).indices
    
    #(np.array(torch.max(em_dicts[0]['p_assign'], 1).indices) == XX[:,-1]).sum()
    
    ## add roc/auc from r/v
    
    #ugt = v[:,lookups[0], lookups[1]].exp()
    #lt = v[:,uwanted[0], uwanted[1]].exp()
    #ugt[:,lookups[0] != lookups[1]] = ugt[:,lookups[0] != lookups[1]] + lt 
    #cas = np.array(torch.hstack((ugt, r.exp())).max(1).indices)
    
    
    #adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
    #adjusted_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])

    #p_assign = torch.hstack((ugt, r.exp()))
    #, 'p_assign': p_assign