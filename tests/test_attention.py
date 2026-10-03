import torch
from pap_age.models.pap_lstm import PAPLSTM

def test_pap_lstm_shapes():
    model=PAPLSTM(input_size=8,hidden_size=16,attention_dim=8)
    x=torch.randn(2,4,8); lengths=torch.tensor([4,2]); mask=torch.tensor([[1,1,1,1],[1,1,0,0]],dtype=torch.bool)
    pred,w=model(x,lengths,mask)
    assert pred.shape==(2,4)
    assert w.shape==(2,4,4)
    assert torch.isfinite(pred).all()
