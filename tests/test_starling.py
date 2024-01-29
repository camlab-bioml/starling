import torch

from starling.starling import ST


def test_can_instantiate(simple_adata_with_size):
    st = ST(simple_adata_with_size)
    assert type(st.X) == torch.Tensor
    assert type(st.S) == torch.Tensor

def test_can_instantiate_without_size(simple_adata):
    st = ST(simple_adata, model_cell_size=False)
    assert type(st.X) == torch.Tensor
    assert st.S is None

def test_prepare_data(simple_adata_km_initialized):
    st = ST(simple_adata_km_initialized, model_cell_size=True)
    assert getattr(st, 'model_params', None) is None
    st.prepare_data()
    assert st.model_params is not None
