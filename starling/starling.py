import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from anndata import AnnData

from starling import utility

BATCH_SIZE = 512
AVAIL_GPUS = min(1, torch.cuda.device_count())
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ST(pl.LightningModule):
    """The STARLING module

    :param adata: The sample to be analyzed, with clusters and annotations from :py:func:`
    :type adata: AnnData
    :param dist_option: The distribution to use
    :type dist_option: str, one of 'T' for Student-T (df=2) or 'N' for Normal (Gaussian), defaults to T
    :param singlet_prop: The proportion of anticipated segmentation error free cells
    :type singlet_prop: float, defaults to 0.6
    :param model_cell_size: Whether STARLING should incoporate cell size in the model
    :type model_cell_size: bool, defaults to True
    :param cell_size_col_name: The column name in ``AnnData`` (anndata.obs)
    :type cell_size_col_name: str, defaults to "area"
    :param model_zplane_overlap: If cell size is modelled, should STARLING model z-plane overlap
    :type model_zplane_overlap: bool, defaults to True
    :param model_regularizer: Regularizier term impose on synethic doublet loss (BCE)
    :type model_regularizer: int, defaults to 1
    :param learning_rate: Learning rate of ADAM optimizer for STARLING
    :type learning_rate: float, defaults to 1e-3

    """

    def __init__(
        self,
        adata: AnnData,
        dist_option=True,
        singlet_prop=0.6,
        model_cell_size=True,
        cell_size_col_name="area",
        model_zplane_overlap=True,
        model_regularizer=1,
        learning_rate=1e-3,
    ):
        super().__init__()

        # self.save_hyperparameters()

        self.adata = adata
        self.dist_option = dist_option
        self.singlet_prop = singlet_prop
        self.model_cell_size = model_cell_size
        self.cell_size_col_name = cell_size_col_name
        self.model_zplane_overlap = model_zplane_overlap
        self.model_regularizer = model_regularizer
        self.learning_rate = learning_rate

        self.X = torch.tensor(self.adata.X)
        self.S = (
            torch.tensor(self.adata.obs[self.cell_size_col_name])
            if self.model_cell_size
            else None
        )

    def forward(
        self, batch: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The module's forward pass

        :param batch: A list of tensors of size m x n
        :type batch: list

        :returns: Negative log loss, Binary Cross-Entropy Loss, singlet probability
        :rtype: tuple of type ``pytorch.Tensor``, NLL and BCE are scalars, while singlet probability has shape m
        """
        if self.model_cell_size:
            y, s, fy, fs, fl = batch
            _, _, model_nll, _ = utility.compute_posteriors(
                y, s, self.model_params, self.dist_option, self.model_zplane_overlap
            )
            _, _, _, p_fake_singlet = utility.compute_posteriors(
                fy, fs, self.model_params, self.dist_option, self.model_zplane_overlap
            )
        else:
            y, fy, fl = batch
            _, _, model_nll, _ = utility.compute_posteriors(
                y, None, self.model_params, self.dist_option, self.model_zplane_overlap
            )
            _, _, _, p_fake_singlet = utility.compute_posteriors(
                fy, None, self.model_params, self.dist_option, self.model_zplane_overlap
            )

        fake_loss = torch.nn.BCELoss()(p_fake_singlet, fl.to(torch.double))

        return model_nll, fake_loss, p_fake_singlet

    def training_step(self, batch) -> torch.Tensor:
        """Compute and return the training loss

        :param batch: A list of tensors of size m x n
        :type batch: list

        :returns: Total loss
        :rtype: torch.Tensor, scalar
        """
        # y, s, fy, fs, fl = batch
        model_nll, fake_loss, p_fake_singlet = self(batch)

        # total loss
        loss = model_nll + self.model_regularizer * fake_loss

        self.log("train_nll", model_nll)
        self.log("train_bce", fake_loss)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self) -> torch.optim.adam.Adam:
        """Configure the Adam optimizer.

        :returns: the optimizer
        :rtype: torch.optim.adam.Adam
        """
        optimizer = torch.optim.Adam(self.model_params.values(), lr=self.learning_rate)
        return optimizer

    def prepare_data(self) -> None:
        """Create training dataset and set model parameters

        :returns: None
        :rtype: None
        """
        tr_fy, tr_fs, tr_fl = utility.simulate_data(
            self.X, self.S, self.model_zplane_overlap
        )

        ## simulate data
        if self.model_cell_size:
            self.train_df = utility.ConcatDataset(self.X, self.S, tr_fy, tr_fs, tr_fl)
            ## get cell size averge/variance
            init_s = []
            init_sv = []
            for ii in range(len(np.unique(self.adata.obs["init_label"]))):
                tmp = self.adata[np.where(self.adata.obs["init_label"] == ii)[0]].obs[
                    self.cell_size_col_name
                ]
                init_s.append(np.mean(tmp))
                init_sv.append(np.var(tmp))
            # self.init_cell_size_centroids = np.array(init_s); self.init_cell_size_variances = np.array(init_sv)
            self.adata.uns["init_cell_size_centroids"] = np.array(init_s)
            self.adata.uns["init_cell_size_variances"] = np.array(init_sv)
        else:
            # init_cell_size_centroids = None; init_cell_size_variances = None
            self.adata.varm["init_cell_size_centroids"] = None
            self.adata.varm["init_cell_size_variances"] = None
            self.train_df = utility.ConcatDataset(self.X, tr_fy, tr_fl)

        # model_params = utility.model_paramters(self.init_e, self.init_v, self.init_s, self.init_sv)
        model_params = utility.model_parameters(self.adata, self.singlet_prop)
        self.model_params = {
            k: torch.from_numpy(val).to(DEVICE).requires_grad_(True)
            for (k, val) in model_params.items()
        }

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader

        :returns: the training DataLoader
        :rtype: torch.utils.data.DataLoader
        """
        return DataLoader(
            self.train_df, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
        )

    def result(self, threshold=0.5) -> None:
        """Retrieve the results and add them to ``self.adata``

        :param threshold: minimum for singlet probability (?) (currently unused)
        :type threshold: float, defaults to .05

        :returns: None
        :rtype: None
        """
        if self.model_cell_size:
            model_pred_loader = DataLoader(
                utility.ConcatDataset(self.X, self.S), batch_size=1000, shuffle=False
            )
        else:
            model_pred_loader = DataLoader(self.X, batch_size=1000, shuffle=False)

        singlet_prob, singlet_assig_prob, gamma_assig_prob = utility.predict(
            model_pred_loader,
            self.model_params,
            self.dist_option,
            self.model_cell_size,
            self.model_zplane_overlap,
            threshold,
        )

        self.adata.obs["st_label"] = np.array(
            singlet_assig_prob.max(1).indices
        )  ##p(z=c|d=1)
        self.adata.obs["doublet_prob"] = 1 - np.array(singlet_prob)
        self.adata.obs["doublet"] = 0
        self.adata.obs.loc[self.adata.obs["doublet_prob"] > 0.5, "doublet"] = 1
        self.adata.obs["max_assign_prob"] = np.array(singlet_assig_prob.max(1).values)

        self.adata.obsm["assignment_prob_matrix"] = np.array(singlet_assig_prob)
        self.adata.obsm["gamma_assignment_prob_matrix"] = np.array(gamma_assig_prob)

        # st_label = singlet_assig_label.numpy().astype('str')
        # st_label[st_label == '-1'] = 'doublet'
        # self.adata.obs['st_label'] = st_label

        # if self.model_cell_size == 'Y':
        #    pretty_printing = np.hstack((self.adata.var_names, self.cell_size_col_name))
        #    c = torch.hstack([self.model_params['log_mu'], self.model_params['log_psi'].reshape(-1,1)]).detach().exp().cpu().numpy()
        # v = torch.hstack([self.model_params['log_sigma'], self.model_params['log_omega'].reshape(-1,1)]).detach().exp().cpu().numpy()
        #    self.adata.uns['st_exp_centroids'] = pd.DataFrame(c, columns=pretty_printing)

        # else:
        c = self.model_params["log_mu"].detach().exp().cpu().numpy()
        # v = self.model_params['log_sigma'].cpu().detach().exp().cpu().numpy()
        self.adata.varm[
            "st_exp_centroids"
        ] = c.T  # pd.DataFrame(c, columns=self.adata.var_names)

        if self.model_cell_size:
            self.adata.uns["st_cell_size_centroids"] = (
                self.model_params["log_psi"]
                .reshape(-1, 1)
                .detach()
                .exp()
                .cpu()
                .numpy()
                .T
            )

        # self.adata.varm['init_exp_centroids'] = pd.DataFrame(self.adata.varm['init_exp_centroids'], columns = self.adata.var_names) #.to_csv(code_dir + "/output/init_centroids.csv")
        # self.adata.varm['init_exp_variances'] = pd.DataFrame(self.adata.varm['init_exp_variances'], columns = self.adata.var_names) #.to_csv(code_dir + "/output/init_centroids.csv")
