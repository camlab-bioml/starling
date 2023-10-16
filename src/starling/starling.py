import utility

import torch
import numpy as np
import pandas as pd

import pytorch_lightning as pl

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 512 if AVAIL_GPUS else 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ST(pl.LightningModule):
    def __init__(self, 
                 adata,                       ## annDATA of the sample
                 dist_option = 'T',           ## T: Student-T (df=2); N: Normal (Gaussian)
                 model_cell_size = 'Y',       ## If STARLING incoporates cell size in the model (Y: Yes; N: No)
                 cell_size_col_name = 'area', ## The column name in AnnDATA object (anndata.obs)
                 model_zplane_overlap = 'Y',  ## If cell size is modelled, STARLING can model z-plane overlap (Y: Yes; N: No)
                 model_regularizer = 1,       ## Regularizier term impose on synethic doublet loss (BCE)
                 learning_rate = 1e-3,        ## Learning rate of ADAM optimizer for STARLING
                 ):
        super().__init__()
        
        #self.save_hyperparameters()
        
        self.adata = adata        
        self.dist_option = dist_option
        self.model_cell_size = model_cell_size
        self.cell_size_col_name = cell_size_col_name
        self.model_zplane_overlap = model_zplane_overlap
        self.model_regularizer = model_regularizer
        self.learning_rate = learning_rate
        
        self.X = torch.tensor(self.adata.X)
        self.S = torch.tensor(self.adata.obs[self.cell_size_col_name]) if self.model_cell_size == 'Y' else None
        
    def forward(self, batch):
        
        if self.model_cell_size == 'Y':
            y, s, fy, fs, fl = batch
            _, _, model_nll, _ = utility.compute_posteriors(y, s, self.model_params, self.dist_option, self.model_zplane_overlap)        
            _, _, _, p_fake_singlet = utility.compute_posteriors(fy, fs, self.model_params, self.dist_option, self.model_zplane_overlap)
        else:
            y, fy, fl = batch
            _, _, model_nll, _ = utility.compute_posteriors(y, None, self.model_params, self.dist_option, self.model_zplane_overlap)
            _, _, _, p_fake_singlet = utility.compute_posteriors(fy, None, self.model_params, self.dist_option, self.model_zplane_overlap)

        fake_loss = torch.nn.BCELoss()(p_fake_singlet, fl.to(torch.double))
        
        return model_nll, fake_loss, p_fake_singlet
        
    def training_step(self, batch, batch_idx):
                
        #y, s, fy, fs, fl = batch
        model_nll, fake_loss, p_fake_singlet = self(batch)
        
        # total loss
        loss = model_nll + self.model_regularizer * fake_loss
        
        self.log("train_nll", model_nll)
        self.log("train_bce", fake_loss)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model_params.values(), lr=self.learning_rate)
        return optimizer
    
    def prepare_data(self):
        
        tr_fy, tr_fs, tr_fl = utility.simulate_data(self.X, self.S, self.model_zplane_overlap)
        
        ## simulate data
        if self.model_cell_size == 'Y':            
            self.train_df = utility.ConcatDataset(self.X, self.S, tr_fy, tr_fs, tr_fl)
            ## get cell size averge/variance
            init_s = []; init_sv = [] 
            for ii in range(len(np.unique(self.adata.obs['init_label']))):
                tmp = self.adata[np.where(self.adata.obs['init_label'] == ii)[0]].obs[self.cell_size_col_name]
                init_s.append(np.mean(tmp))
                init_sv.append(np.var(tmp))
            #self.init_cell_size_centroids = np.array(init_s); self.init_cell_size_variances = np.array(init_sv)
            self.adata.uns['init_cell_size_centroids'] = np.array(init_s)
            self.adata.uns['init_cell_size_variances'] = np.array(init_sv)
        else:
            #init_cell_size_centroids = None; init_cell_size_variances = None
            self.adata.varm['init_cell_size_centroids'] = None
            self.adata.varm['init_cell_size_variances'] = None
            self.train_df = utility.ConcatDataset(self.X, tr_fy, tr_fl)

        #model_params = utility.model_paramters(self.init_e, self.init_v, self.init_s, self.init_sv)
        model_params = utility.model_paramters(self.adata)
        self.model_params = {k: torch.from_numpy(val).to(DEVICE).requires_grad_(True) for (k,val) in model_params.items()}
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_df, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)
    
    def result(self, threshold=0.5):

        if self.model_cell_size == 'Y':
            model_pred_loader = torch.utils.data.DataLoader(utility.ConcatDataset(self.X, self.S), batch_size = 1000, shuffle = False)
        else:
            model_pred_loader = torch.utils.data.DataLoader(self.X, batch_size = 1000, shuffle = False)

        singlet_prob, singlet_assig_prob = utility.predict(model_pred_loader, self.model_params, self.dist_option, self.model_cell_size, self.model_zplane_overlap, threshold)
        
        self.adata.obs['st_label'] = np.array(singlet_assig_prob.max(1).indices) ##p(z=c|d=1)
        self.adata.obs['doublet_prob'] = 1 - np.array(singlet_prob)
        self.adata.obs['doublet'] = 0
        self.adata.obs.loc[self.adata.obs['doublet_prob'] > 0.5,'doublet'] = 1
        self.adata.obs['max_assign_prob'] = np.array(singlet_assig_prob.max(1).values)
        self.adata.obsm['assignment_prob_matrix'] = np.array(singlet_assig_prob)

        #st_label = singlet_assig_label.numpy().astype('str')
        #st_label[st_label == '-1'] = 'doublet'
        #self.adata.obs['st_label'] = st_label

        #if self.model_cell_size == 'Y':
        #    pretty_printing = np.hstack((self.adata.var_names, self.cell_size_col_name))
        #    c = torch.hstack([self.model_params['log_mu'], self.model_params['log_psi'].reshape(-1,1)]).detach().exp().cpu().numpy()
            #v = torch.hstack([self.model_params['log_sigma'], self.model_params['log_omega'].reshape(-1,1)]).detach().exp().cpu().numpy()
        #    self.adata.uns['st_exp_centroids'] = pd.DataFrame(c, columns=pretty_printing)

        #else:
        c = self.model_params['log_mu'].detach().exp().cpu().numpy()
        #v = self.model_params['log_sigma'].cpu().detach().exp().cpu().numpy()
        self.adata.varm['st_exp_centroids'] = c.T #pd.DataFrame(c, columns=self.adata.var_names)
        
        if self.model_cell_size == 'Y':
            self.adata.uns['st_cell_size_centroids'] = self.model_params['log_psi'].reshape(-1,1).detach().exp().cpu().numpy().T

        #self.adata.varm['init_exp_centroids'] = pd.DataFrame(self.adata.varm['init_exp_centroids'], columns = self.adata.var_names) #.to_csv(code_dir + "/output/init_centroids.csv")
        #self.adata.varm['init_exp_variances'] = pd.DataFrame(self.adata.varm['init_exp_variances'], columns = self.adata.var_names) #.to_csv(code_dir + "/output/init_centroids.csv")
