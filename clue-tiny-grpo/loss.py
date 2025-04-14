from typing import Optional
import torch
import torch.nn as nn

from replay_buffer import Experience


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Monte-Carlo approximation of KL divergence, k3 estimator, see: http://joschu.net/blog/kl-approx.html
    """

    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    """GRPO actor loss"""

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Calculate loss using tensors directly instead of Experience object."""
        # Get the log_probs from reference model for KL divergence
        log_probs_ref = old_log_probs

        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        # PPO clipping objective
        # Initialize ratio tensor with same shape as log_probs
        ratio = torch.zeros_like(log_probs)
        mean_ratio = 0.0
        
        # Only compute ratio where action_mask is True to avoid NaN from padded tokens
        if action_mask.sum() > 0:  # Check if there are valid tokens
            valid_log_probs = log_probs[action_mask]
            valid_old_log_probs = old_log_probs[action_mask]
            valid_ratio = (valid_log_probs - valid_old_log_probs).exp()
            ratio[action_mask] = valid_ratio
            mean_ratio = valid_ratio.mean().item()

        # Expand advantages to match the sequence dimension if needed
        if advantages.dim() == 2 and advantages.shape[1] == 1 and log_probs.dim() == 2 and log_probs.shape[1] > 1:
            # advantages has shape [batch_size, 1] and log_probs has shape [batch_size, seq_len]
            # We need to expand advantages to [batch_size, seq_len]
            expanded_advantages = advantages.expand(-1, log_probs.shape[1])
        else:
            expanded_advantages = advantages

        # Apply action mask to both ratio and expanded advantages
        masked_ratio = ratio * action_mask.float()
        masked_advantages = expanded_advantages * action_mask.float()

        # Compute PPO surrogate objectives
        surr1 = masked_ratio * masked_advantages
        surr2 = masked_ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * masked_advantages
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl

        # Compute statistics for logging
        stats = {
            "loss": loss.mean().item(),
            "kl_div": kl.mean().item(),
            "policy_ratio": mean_ratio,
            "reward/mean": returns.mean().item(),
            "reward/std": returns.std().item() if returns.numel() > 1 else 0.0,
            "reward/max": returns.max().item() if returns.numel() > 0 else 0.0,
        }

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss, stats
