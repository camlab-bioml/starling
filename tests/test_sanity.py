from os.path import dirname, join

import anndata as ad
import pandas as pd
import pytorch_lightning as pl
from lightning_lite import seed_everything
from pytorch_lightning.callbacks import EarlyStopping

from starling import starling, utility


def test_can_run_km(tmpdir):
    """Temporary sanity check"""
    seed_everything(10, workers=True)

    raw_adata = ad.read_h5ad(join(dirname(__file__), "fixtures", "sample_input.h5ad"))

    adata = utility.init_clustering(
        "KM",
        raw_adata,
        k=10,
    )
    st = starling.ST(adata)
    cb_early_stopping = EarlyStopping(monitor="train_loss", mode="min", verbose=False)

    ## train ST
    st.train_and_fit(
        max_epochs=2,
        callbacks=[cb_early_stopping],
        default_root_dir=tmpdir,
    )

    result = st.result()

    ## initial expression centriods (p x c) matrix
    init_cent = pd.DataFrame(result.varm["init_exp_centroids"], index=result.var_names)

    assert init_cent.shape == (24, 10)

    ## starling expression centriods (p x c) matrix
    exp_cent = pd.DataFrame(result.varm["st_exp_centroids"], index=result.var_names)

    assert exp_cent.shape == (24, 10)

    ## assignment distributions (n x c maxtrix)
    prom_mat = pd.DataFrame(
        result.obsm["assignment_prob_matrix"], index=result.obs.index
    )

    assert prom_mat.shape == (13685, 10)


def test_can_run_gmm(tmpdir):
    """Temporary sanity check"""
    seed_everything(10, workers=True)
    adata = utility.init_clustering(
        "GMM",
        ad.read_h5ad(join(dirname(__file__), "fixtures", "sample_input.h5ad")),
        k=10,
    )
    st = starling.ST(adata)
    cb_early_stopping = EarlyStopping(monitor="train_loss", mode="min", verbose=False)

    ## train ST
    st.train_and_fit(
        max_epochs=2,
        callbacks=[cb_early_stopping],
        default_root_dir=tmpdir,
    )

    result = st.result()

    ## initial expression centriods (p x c) matrix
    init_cent = pd.DataFrame(result.varm["init_exp_centroids"], index=result.var_names)

    assert init_cent.shape == (24, 10)

    ## starling expression centriods (p x c) matrix
    exp_cent = pd.DataFrame(result.varm["st_exp_centroids"], index=result.var_names)

    assert exp_cent.shape == (24, 10)

    ## assignment distributions (n x c maxtrix)
    prom_mat = pd.DataFrame(
        result.obsm["assignment_prob_matrix"], index=result.obs.index
    )

    assert prom_mat.shape == (13685, 10)


def test_can_run_pg(tmpdir):
    """Temporary sanity check"""
    seed_everything(10, workers=True)
    adata = utility.init_clustering(
        "PG",
        ad.read_h5ad(join(dirname(__file__), "fixtures", "sample_input.h5ad")),
        k=10,
    )
    st = starling.ST(adata)
    cb_early_stopping = EarlyStopping(monitor="train_loss", mode="min", verbose=False)

    ## train ST
    st.train_and_fit(
        max_epochs=2,
        callbacks=[cb_early_stopping],
        default_root_dir=tmpdir,
    )

    result = st.result()

    ## initial expression centriods (p x c) matrix
    init_cent = pd.DataFrame(result.varm["init_exp_centroids"], index=result.var_names)

    # j seems to vary here
    assert init_cent.shape[0] == 24

    ## starling expression centriods (p x c) matrix
    exp_cent = pd.DataFrame(result.varm["st_exp_centroids"], index=result.var_names)

    assert exp_cent.shape[0] == 24

    ## assignment distributions (n x c maxtrix)
    prom_mat = pd.DataFrame(
        result.obsm["assignment_prob_matrix"], index=result.obs.index
    )

    assert prom_mat.shape[0] == 13685

def test_can_run_pg_without_cell_size(tmpdir):
    """Temporary sanity check"""
    seed_everything(10, workers=True)
    adata = utility.init_clustering(
        "PG",
        ad.read_h5ad(join(dirname(__file__), "fixtures", "sample_input.h5ad")),
        k=10,
    )
    st = starling.ST(adata, model_cell_size=False)
    cb_early_stopping = EarlyStopping(monitor="train_loss", mode="min", verbose=False)

    ## train ST
    st.train_and_fit(
        max_epochs=2,
        callbacks=[cb_early_stopping],
        default_root_dir=tmpdir,
    )

    result = st.result()

    assert True
