import random

import numpy as np
import pytest
from anndata import AnnData

from starling.utility import init_clustering


@pytest.fixture
def simple_adata(size=True):
    adata = AnnData(np.arange(512).reshape(32, 16))
    if size:
        adata.obs["area"] = [random.randint(1, 5) for i in range(adata.shape[0])]
    return adata


@pytest.fixture
def simple_adata_with_size(simple_adata):
    simple_adata.obs["area"] = [
        random.randint(1, 5) for i in range(simple_adata.shape[0])
    ]
    return simple_adata


@pytest.fixture
def simple_adata_km_initialized(simple_adata):
    k = 3
    return init_clustering("KM", simple_adata, k)


@pytest.fixture
def simple_adata_km_initialized_with_size(simple_adata_with_size):
    k = 3
    return init_clustering("KM", simple_adata_with_size, k)
