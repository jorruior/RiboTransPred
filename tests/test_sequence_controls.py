import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from sequence_controls import shuffle_five_prime_utr, training_sequences
import train_tissues

class Checks(unittest.TestCase):
    def setUp(self):
        self.seq="ACGT"*8+"NN"+"TGCA"*8+"ATG"+"CCC"*8+"TAA"+"G"*24
        self.annot="1"*66+"2"*30+"3"*24
    def test_boundaries_and_reproducibility(self):
        shuffled=shuffle_five_prime_utr(self.seq,self.annot,4,"mouse/tx",120)
        self.assertEqual(shuffled[66:],self.seq[66:])
        self.assertEqual(Counter(shuffled[:66]),Counter(self.seq[:66]))
        self.assertEqual(shuffled[32:34],"NN")
        self.assertNotEqual(shuffled[:66],self.seq[:66])
        self.assertEqual(shuffled,shuffle_five_prime_utr(self.seq,self.annot,4,"mouse/tx",120))
        self.assertNotEqual(shuffled,shuffle_five_prime_utr(self.seq,self.annot,5,"mouse/tx",120))
        for annot in ("0"*120,"2"*120,""):
            self.assertEqual(self.seq,shuffle_five_prime_utr(self.seq,annot))
        self.assertEqual(shuffle_five_prime_utr(self.seq,self.annot,4,"mouse/tx",12)[12:],self.seq[12:])
    def test_datasets_keep_cds_rna_target_mask(self):
        frame=pd.DataFrame([dict(sequence=self.seq,region=self.annot,id="tx",species="mouse")])
        args=SimpleNamespace(shuflledutr=True,nosequence=False,seed=4,shuffle_seed=None,region_len=120)
        transformed=training_sequences(frame,args)
        rna=np.arange(120,dtype=np.float32)[None,:];target=np.ones((1,40),dtype=np.float32)
        for cls,extra in [(train_tissues.TissueTranscriptDataset,dict(tissue_id=0,cond_id=0))]:
            kw=dict(index_array=np.array([0]),inp_mmap=rna,out_mmap=target,seq_len=120,nbins=40,**extra)
            a=cls(sequences=np.array([self.seq]),**kw)[0];b=cls(sequences=transformed,**kw)[0]
            torch.testing.assert_close(a["features"][66:],b["features"][66:])
            torch.testing.assert_close(a["features"][:,-1],b["features"][:,-1])
            for key in ("mask","target"):torch.testing.assert_close(a[key],b[key])
            self.assertGreater(a["mask"].sum(),0)
        with self.assertRaises(ValueError):training_sequences(frame.drop(columns="region"),args)
    def test_shuffled_training_forward_and_backward(self):
        torch.set_num_threads(2)
        old=sys.argv
        try:
            for module in (train_tissues,):
                sys.argv=["test","--shuflledutr"]
                args=module.parse_args();args.region_len=120;args.nBins=40
                model=module.TissueRiboModel(args,{"heart":0},{"adult":0})
                seq=shuffle_five_prime_utr(self.seq,self.annot,4,"mouse/tx",120)
                x=torch.from_numpy(np.concatenate([module._encode_seq_fast(seq,120),np.ones((120,1),dtype=np.float32)],axis=1))[None]
                pred=model(x,torch.tensor([0]),torch.tensor([0]))
                self.assertEqual(pred.shape,(1,40))
                loss=model._compute_loss(pred,torch.ones_like(pred),torch.ones_like(pred));loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(model.hparams["shuflledutr"])
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
        finally:sys.argv=old

    def test_flags_and_defaults(self):
        old=sys.argv
        try:
            for module in (train_tissues,):
                sys.argv=["test","--shuflledutr"];args=module.parse_args()
                self.assertEqual((args.region_len,args.nBins),(4500,1500));self.assertTrue(args.shuflledutr)
                sys.argv=["test","--shuffledutr","--shuffle-seed","9"]
                self.assertEqual(module.parse_args().shuffle_seed,9)
        finally:sys.argv=old

if __name__=="__main__":unittest.main()
