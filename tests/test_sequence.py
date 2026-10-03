import numpy as np
from pap_age.data.sequence_dataset import FeatureSequenceDataset, collate_sequences

def test_sequence_sort_and_collate():
    f=np.arange(20,dtype=np.float32).reshape(5,4); ages=np.array([30,20,40,10,15],dtype=np.float32)
    people=np.array(["a","a","a","b","b"]); splits=np.array(["train"]*5)
    ds=FeatureSequenceDataset(f,ages,people,splits,"train",2)
    assert len(ds)==2
    batch=collate_sequences([ds[0],ds[1]])
    assert batch["features"].shape==(2,3,4)
    assert batch["mask"].sum().item()==5
