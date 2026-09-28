"""Masked profile losses and globally reducible sufficient statistics."""
import math
import numpy as np
import torch
import torch.distributed as dist

LOG_ZERO = math.log(0.0001)


def load_rna_eligibility(input_path, n_rows):
    path = str(input_path).replace('_log_rnaseq_final_v2.npy', '_eligible_v2.npy')
    if path == str(input_path):
        raise ValueError('Expected a v2 RNA feature path')
    eligible = np.load(path, allow_pickle=False)
    if eligible.shape != (n_rows,) or eligible.dtype != np.bool_:
        raise ValueError(f'Invalid RNA eligibility mask: {path}')
    return eligible


def loss_components(pred, target, mask, zero_w):
    """MSE numerator/weight and per-transcript PCC loss sum/count."""
    pred, target, mask = pred.float(), target.float(), mask.float()
    zero = torch.isclose(target, torch.full_like(target, LOG_ZERO), atol=1e-5, rtol=0)
    weight = torch.where(zero, zero_w, 1.0) * mask
    mse_num = ((pred - target).square() * weight).sum()
    mse_den = weight.sum()
    n = mask.sum(dim=1)
    denom_n = n.clamp_min(1).unsqueeze(1)
    pc = pred - (pred * mask).sum(dim=1, keepdim=True) / denom_n
    tc = target - (target * mask).sum(dim=1, keepdim=True) / denom_n
    vp = (pc.square() * mask).sum(dim=1)
    vt = (tc.square() * mask).sum(dim=1)
    cov = (pc * tc * mask).sum(dim=1)
    r = cov / (vp * vt).clamp_min(1e-8).sqrt()
    valid = (n >= 10) & (vt > 1e-8)
    pcc_sum = ((1 - r) * valid).sum()
    return torch.stack((mse_num, mse_den, pcc_sum, valid.sum().to(pred.dtype)))


def masked_profile_loss(pred, target, mask, zero_w, pcc_loss_w):
    terms = loss_components(pred, target, mask, zero_w)
    return terms[0] / terms[1].clamp_min(1e-8) + pcc_loss_w * terms[2] / terms[3].clamp_min(1)


class ProfileMetrics:
    """Accumulate globally additive states, never averages of batch averages.

    The same metric keys must be created on every rank, including empty tracks.
    compute() reduces a clone so repeated calls do not double-count state.
    """
    def __init__(self, device, zero_w=0.1, pcc_loss_w=0.2):
        self.state = torch.zeros(11, dtype=torch.float64, device=device)
        self.zero_w = zero_w
        self.pcc_loss_w = pcc_loss_w

    @torch.no_grad()
    def update(self, pred, target, mask):
        self.state[:4] += loss_components(pred, target, mask, self.zero_w).double()
        p, t, w = pred.double(), target.double(), mask.double()
        self.state[4:10] += torch.stack(((p*t*w).sum(), (p*w).sum(), (t*w).sum(),
                                       (p*p*w).sum(), (t*t*w).sum(), w.sum()))
        self.state[10] += (mask.sum(dim=1) > 0).sum()

    def compute(self, sync=True):
        values = self.state.clone()
        if sync and dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        loss = (values[0] / values[1].clamp_min(1e-8) +
                self.pcc_loss_w * values[2] / values[3].clamp_min(1))
        xy, x, y, xx, yy, n = values[4:10]
        n = n.clamp_min(1)
        cov = xy - x*y/n
        vx, vy = (xx - x*x/n).clamp_min(0), (yy - y*y/n).clamp_min(0)
        pcc = cov / (vx*vy).clamp_min(1e-16).sqrt()
        return {'loss': loss.item(), 'pcc': pcc.item(), 'n_samples': int(values[10].item())}

    def reset(self):
        self.state.zero_()


class EvaluationDataset(torch.utils.data.Dataset):
    """Mask sampler padding while keeping equal evaluation work on every GPU."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[0 if index < 0 else index]
        if index < 0:
            sample = dict(sample)
            sample['mask'] = torch.zeros_like(sample['mask'])
        return sample


class DistributedEvaluationSampler(torch.utils.data.DistributedSampler):
    """Equal-length rank shards, using masked sentinels instead of duplicates."""
    def __init__(self, dataset, num_replicas=None, rank=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False, drop_last=False)
        self.size = len(dataset)
        self.world = dist.get_world_size() if num_replicas is None else num_replicas
        self.rank = dist.get_rank() if rank is None else rank
        self.num_samples = math.ceil(self.size / self.world)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for index in range(self.rank, self.num_samples * self.world, self.world):
            yield index if index < self.size else -1

    def set_epoch(self, epoch):
        pass
