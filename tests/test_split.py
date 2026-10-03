import pandas as pd
from pap_age.data.metadata import split_by_identity

def test_identity_split_no_leakage():
    df=pd.DataFrame({"path":[str(i) for i in range(20)],"person_id":[f"p{i//2}" for i in range(20)],"age":[20+i for i in range(20)]})
    out=split_by_identity(df,0.8,0.0,42)
    train=set(out[out.split=="train"].person_id); test=set(out[out.split=="test"].person_id)
    assert train.isdisjoint(test)
