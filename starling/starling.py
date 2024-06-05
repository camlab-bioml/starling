from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from anndata import AnnData
from lightning_fabric.connector import _PRECISION_INPUT
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.plugins import PLUGIN_INPUT
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.trainer.connectors.accelerator_connector import _LITERAL_WARN
from torch.utils.data import DataLoader

from starling import utility

BATCH_SIZE = 512
AVAIL_GPUS = min(1, torch.cuda.device_count())
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ST(pl.LightningModule):
    """The STARLING module

    :param adata: The sample to be analyzed, with clusters and annotations from :py:func:`starling.uility.init_clustering`
    :param dist_option: The distribution to use, one of 'T' for Student-T (df=2) or 'N' for Normal (Gaussian), defaults to T
    :param singlet_prop: The proportion of anticipated segmentation error free cells
    :param model_cell_size: Whether STARLING should incoporate cell size in the model
    :param cell_size_col_name: The column name in ``AnnData`` (anndata.obs). Required only if ``model_cell_size`` is ``True``,
        otherwise ignored.
    :param model_zplane_overlap: If cell size is modelled, should STARLING model z-plane overlap
    :param model_regularizer: Regularizer term impose on synethic doublet loss (BCE)
    :param learning_rate: Learning rate of ADAM optimizer for STARLING

    """

    def __init__(
        self,
        adata: AnnData,
        dist_option: str = "T",
        singlet_prop: float = 0.6,
        model_cell_size: bool = True,
        cell_size_col_name: str = "area",
        model_zplane_overlap: bool = True,
        model_regularizer: float = 1.0,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        # self.save_hyperparameters()

        utility.validate_starling_arguments(
            adata,
            dist_option,
            singlet_prop,
            model_cell_size,
            cell_size_col_name,
            model_zplane_overlap,
            model_regularizer,
            learning_rate,
        )

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

        :param batch: A list of tensors

        :returns: Negative log loss, Binary Cross-Entropy Loss, singlet probability
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

    def training_step(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """Compute and return the training loss

        :param batch: A list of tensors of size m x n

        :returns: Total loss
        """
        # y, s, fy, fs, fl = batch
        model_nll, fake_loss, p_fake_singlet = self(batch)

        # total loss
        loss = model_nll + self.model_regularizer * fake_loss

        self.log("train_nll", model_nll)
        self.log("train_bce", fake_loss)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        """Configure the Adam optimizer.

        :returns: the optimizer
        """
        optimizer = torch.optim.Adam(self.model_params.values(), lr=self.learning_rate)
        return optimizer

    def prepare_data(self) -> None:
        """Create training dataset and set model parameters"""
        tr_fy, tr_fs, tr_fl = utility.simulate_data(
            self.X, self.S, self.model_zplane_overlap
        )

        ## simulate data
        if self.S is not None and tr_fs is not None:
            self.train_df = utility.ConcatDataset([self.X, self.S, tr_fy, tr_fs, tr_fl])
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
            self.adata.uns["init_cell_size_centroids"] = None
            self.adata.uns["init_cell_size_variances"] = None
            self.train_df = utility.ConcatDataset([self.X, tr_fy, tr_fl])

        # model_params = utility.model_paramters(self.init_e, self.init_v, self.init_s, self.init_sv)
        model_params = utility.model_parameters(self.adata, self.singlet_prop)
        self.model_params = {
            k: torch.from_numpy(val).to(DEVICE).requires_grad_(True)
            for (k, val) in model_params.items()
        }

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader

        :returns: the training DataLoader
        """
        return DataLoader(
            self.train_df, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
        )

    def train_and_fit(
        self,
        *,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: Optional[_PRECISION_INPUT] = None,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        fast_dev_run: Union[int, bool] = False,
        max_epochs: Optional[int] = 100,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        overfit_batches: Union[int, float] = 0.0,
        val_check_interval: Optional[Union[int, float]] = None,
        check_val_every_n_epoch: Optional[int] = 1,
        num_sanity_val_steps: Optional[int] = None,
        log_every_n_steps: Optional[int] = None,
        enable_checkpointing: Optional[bool] = None,
        enable_progress_bar: Optional[bool] = None,
        enable_model_summary: Optional[bool] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = True,
        benchmark: Optional[bool] = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Optional[Union[Profiler, str]] = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: Optional[_PATH] = None,
    ) -> None:
        """Train the model using lightning's trainer.
        Param annotations (with defaults altered as needed) taken from https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/trainer/trainer.html#Trainer.__init__

        :param accelerator: Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
            as well as custom accelerator instances. Defaults to ``"auto"``.
        :param strategy: Supports different training strategies with aliases as well custom strategies.
            Defaults to ``"auto"``.
        :param devices: The devices to use. Can be set to a positive number (int or str), a sequence of device indices
            (list or str), the value ``-1`` to indicate all available devices should be used, or ``"auto"`` for
            automatic selection based on the chosen accelerator. Defaults to ``"auto"``.
        :param num_nodes: Number of GPU nodes for distributed training.
            Defaults to ``1``.
        :param precision: Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
            16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
            Can be used on CPU, GPU, TPUs, HPUs or IPUs.
            Defaults to ``'32-true'``.
        :param logger: Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
            the default ``TensorBoardLogger`` if it is installed, otherwise ``CSVLogger``.
            ``False`` will disable logging. If multiple loggers are provided, local files
            (checkpoints, profiler traces, etc.) are saved in the ``log_dir`` of the first logger.
            Defaults to ``True``.
        :param callbacks: Add a callback or list of callbacks.
            Defaults to ``None``.
        :param fast_dev_run: Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
            of train, val and test to find any bugs (:param ie: a sort of unit test).
            Defaults to ``False``.
        :param max_epochs: Stop training once this number of epochs is reached. Disabled by default (None).
            If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 100``.
            To enable infinite training, set ``max_epochs = -1``.
        :param min_epochs: Force training for at least these many epochs. Disabled by default (None).
        :param max_steps: Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
            and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
            ``max_epochs`` to ``-1``.
        :param min_steps: Force training for at least these number of steps. Disabled by default (``None``).
        :param max_time: Stop training after this amount of time has passed. Disabled by default (``None``).
            The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
            :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
            :class:`datetime.timedelta`.
        :param limit_train_batches: How much of training dataset to check (float = fraction, int = num_batches).
            Defaults to ``1.0``.
        :param limit_val_batches: How much of validation dataset to check (float = fraction, int = num_batches).
            Defaults to ``1.0``.
        :param limit_test_batches: How much of test dataset to check (float = fraction, int = num_batches).
            Defaults to ``1.0``.
        :param limit_predict_batches: How much of prediction dataset to check (float = fraction, int = num_batches).
            Defaults to ``1.0``.
        :param overfit_batches: Overfit a fraction of training/validation data (float) or a set number of batches (int).
            Defaults to ``0.0``.
        :param val_check_interval: How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
            after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
            batches. An ``int`` value can only be higher than the number of training batches when
            ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
            across epochs or during iteration-based training.
            Defaults to ``1.0``.
        :param check_val_every_n_epoch: Perform a validation loop every after every `N` training epochs. If ``None``,
            validation will be done solely based on the number of training batches, requiring ``val_check_interval``
            to be an integer value.
            Defaults to ``1``.
        :param num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
            Set it to `-1` to run all batches in all validation dataloaders.
            Defaults to ``2``.
        :param log_every_n_steps: How often to log within steps.
            Defaults to ``50``.
        :param enable_checkpointing: If ``True``, enable checkpointing.
            It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
            `lightning.pytorch.trainer.trainer.Trainer.callbacks`.
            Defaults to ``True``.
        :param enable_progress_bar: Whether to enable to progress bar by default.
            Defaults to ``True``.
        :param enable_model_summary: Whether to enable model summarization by default.
            Defaults to ``True``.
        :param accumulate_grad_batches: Accumulates gradients over k batches before stepping the optimizer.
            Defaults to 1.
        :param gradient_clip_val: The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
            gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.
            Defaults to ``None``.
        :param gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
            to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
            be set to ``"norm"``.
        :param deterministic: If ``True``, sets whether PyTorch operations must use deterministic algorithms.
            Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations
            that don't support deterministic mode. If not set, defaults to ``False``. Defaults to ``True``.
        :param benchmark: The value (``True`` or ``False``) to set ``torch.backends.cudnn.benchmark`` to.
            The value for ``torch.backends.cudnn.benchmark`` set in the current session will be used
            (``False`` if not manually set). If `deterministic`
            is set to ``True``, this will default to ``False``. Override to manually set a different value.
            Defaults to ``None``.
        :param inference_mode: Whether to use `torch.inference_mode` or `torch.no_grad` during
            evaluation (``validate``/``test``/``predict``).
        :param use_distributed_sampler: Whether to wrap the DataLoader's sampler with
            :class:`torch.utils.data.DistributedSampler`. If not specified this is toggled automatically for
            strategies that require it. By default, it will add ``shuffle=True`` for the train sampler and
            ``shuffle=False`` for validation/test/predict samplers. If you want to disable this logic, you can pass
            ``False`` and add your own distributed sampler in the dataloader hooks. If ``True`` and a distributed
            sampler was already added, Lightning will not replace the existing one. For iterable-style datasets,
            we don't do this automatically.
        :param profiler: To profile individual steps during training and assist in identifying bottlenecks.
            Defaults to ``None``.
        :param detect_anomaly: Enable anomaly detection for the autograd engine.
            Defaults to ``False``.
        :param barebones: Whether to run in "barebones mode", where all features that may impact raw speed are
            disabled. This is meant for analyzing the Trainer overhead and is discouraged during regular training
            runs.
        :param plugins: Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
            Defaults to ``None``.
        :param sync_batchnorm: Synchronize batch norm layers between process groups/whole world.
            Defaults to ``False``.
        :param reload_dataloaders_every_n_epochs: Set to a positive integer to reload dataloaders every n epochs.
            Defaults to ``0``.
        :param default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
            Defaults to ``os.getcwd()``.
            Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'

        Raises:
            TypeError:
                If ``gradient_clip_val`` is not an int or float.

            MisconfigurationException:
                If ``gradient_clip_algorithm`` is invalid.
        """

        _locals = locals()

        _locals.pop("self")

        trainer = pl.Trainer(**_locals)

        trainer.fit(self)

    def result(self, threshold: float = 0.5) -> AnnData:
        """Retrieve the results and add them to ``self.adata``

        :param threshold: minimum threshold for singlet probability
        """
        if self.S is not None:
            model_pred_loader = DataLoader(
                utility.ConcatDataset([self.X, self.S]), batch_size=1000, shuffle=False
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

        return self.adata
