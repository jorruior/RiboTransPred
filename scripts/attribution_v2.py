"""Integrated gradients of the same linear-scale signal used in mutation scoring."""
import torch


def integrated_gradients_signal(model, features, target_mask, model_args=(),
                                n_steps=20, eps=0.0001, zero_threshold=0.0001):
    if n_steps < 1:
        raise ValueError('n_steps must be positive')
    if model.training:
        raise ValueError('Attribution requires model.eval() (dropout disabled)')
    features = features.detach()
    baseline = features.clone()
    # Preserve N/padding positions; interpolate only actual A/T/C/G bases.
    valid_bases = features[:, :, :4].abs().sum(dim=-1) > 0
    baseline[:, :, :5] = torch.where(valid_bases.unsqueeze(-1),
                                     torch.zeros_like(baseline[:, :, :5]),
                                     baseline[:, :, :5])
    difference = features[:, :, :5] - baseline[:, :, :5]
    accumulated = torch.zeros_like(difference)
    target_mask = torch.as_tensor(target_mask, dtype=features.dtype, device=features.device)
    for step in range(n_steps + 1):
        point = baseline.clone()
        point[:, :, :5] += (step / n_steps) * difference
        point.requires_grad_(True)
        pred_log = model(point, *model_args)
        signal = pred_log.clamp(-14, 14).exp() - eps
        signal = torch.where(signal > zero_threshold, signal, torch.zeros_like(signal))
        objective = (signal * target_mask.unsqueeze(0)).sum() / target_mask.sum().clamp_min(1)
        gradient, = torch.autograd.grad(objective, point)
        weight = 0.5 if step in (0, n_steps) else 1.0
        accumulated += weight * gradient[:, :, :5]
    return ((accumulated / n_steps) * difference).sum(dim=-1).squeeze(0).detach().cpu().numpy()
