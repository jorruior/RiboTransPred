"""Two-process CPU/Gloo verification of additive evaluation metrics."""
import os
import sys
import tempfile
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from training_metrics_v2 import ProfileMetrics

def worker(rank,uri):
    dist.init_process_group('gloo',init_method=uri,rank=rank,world_size=2)
    try:
        p=torch.arange(60,dtype=torch.float32).reshape(5,12)/20
        t=p*.7+torch.sin(p)
        mask=torch.ones_like(p)
        local=ProfileMetrics('cpu')
        local.update(p[rank::2],t[rank::2],mask[rank::2])
        reference=ProfileMetrics('cpu'); reference.update(p,t,mask)
        expected=reference.compute(sync=False)
        for _ in range(2):
            actual=local.compute()
            assert actual['n_samples']==5
            assert abs(actual['loss']-expected['loss'])<1e-6
            assert abs(actual['pcc']-expected['pcc'])<1e-10
        empty=ProfileMetrics('cpu')
        if rank==0: empty.update(p[:1],t[:1],mask[:1])
        assert empty.compute()['n_samples']==1
    finally: dist.destroy_process_group()

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as tmp:
        uri=(Path(tmp)/'gloo_store').as_uri()
        mp.spawn(worker,args=(uri,),nprocs=2,join=True)
    print('PASS: two ranks, unequal partitions, empty-rank track, repeated all-reduce')
