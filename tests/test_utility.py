import numpy as np
from anndata import AnnData

from starling.utility import init_clustering, model_parameters

""" TODO: numeric checks """


def assert_annotated(adata: AnnData, k):
    assert "init_exp_centroids" in adata.varm
    assert adata.varm["init_exp_centroids"].shape == (adata.X.shape[1], k)

    assert "init_exp_centroids" in adata.varm
    assert adata.varm["init_exp_variances"].shape == (adata.X.shape[1], k)

    assert "init_label" in adata.obs
    assert adata.obs["init_label"].shape == (adata.X.shape[0],)


def test_init_clustering_km(simple_adata):
    k = 3
    initialized = init_clustering(simple_adata, "KM", k)
    assert_annotated(initialized, k)


def test_init_clustering_gmm(simple_adata):
    k = 3
    initialized = init_clustering(simple_adata, "GMM", k)
    assert_annotated(initialized, k)


def test_init_clustering_pg(simple_adata):
    k = 2
    initialized = init_clustering(simple_adata, "PG", k)
    assert_annotated(initialized, k)
