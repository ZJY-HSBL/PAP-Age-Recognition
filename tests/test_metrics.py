from pap_age.metrics import mae, cumulative_scores

def test_mae_and_cs():
    y=[10,20,30]; p=[10,22,27]
    assert abs(mae(y,p)-5/3)<1e-8
    cs=cumulative_scores(y,p,3)
    assert abs(cs[0] - 100/3) < 1e-10
    assert abs(cs[2] - 200/3) < 1e-10
    assert cs[3] == 100.0
