import torch

from uniplbind import UniPLBind, VNet


def test_model_and_vnet_shapes():
    model = UniPLBind(input_dim=32, dropout=0.0)
    features = torch.randn(2, 9, 32)
    lengths = torch.tensor([9, 6])
    logits = model(features, lengths)
    assert logits.shape == (2, 9, 2)

    weights = VNet(hidden_dim=8)(torch.rand(12))
    assert weights.shape == (12,)
    assert torch.all((weights > 0) & (weights < 1))

