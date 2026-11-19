"""Q3 preference objectives for CS781 Assignment 5.

The notebook writes this module again during Run All; keep the source in its
``%%writefile alignment.py`` cell synchronized with this file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_logprobs(logits: torch.Tensor, input_ids: torch.Tensor,
                      response_mask: torch.Tensor) -> torch.Tensor:
    """Sum causal token log probabilities at response positions, shape (B,)."""
    if logits.ndim != 3 or input_ids.shape != logits.shape[:2] or response_mask.shape != input_ids.shape:
        raise ValueError("Expected logits (B,T,V), input_ids (B,T), and response_mask (B,T)")
    token_logps = F.log_softmax(logits[:, :-1], dim=-1).gather(
        -1, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * response_mask[:, 1:].to(token_logps.dtype)).sum(dim=-1)


def reward_model_loss(chosen_rewards, rejected_rewards):
    """Mean Bradley–Terry loss for chosen over rejected rewards."""
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def reinforce_policy_loss(sample_logprobs, rewards, baseline=None):
    """REINFORCE with a detached reward advantage."""
    if baseline is None:
        baseline = rewards.mean()
    advantage = (rewards - baseline).detach()
    return -(advantage * sample_logprobs).mean()


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Reference-regularized direct preference optimization."""
    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = ref_chosen_logps.detach() - ref_rejected_logps.detach()
    return -F.logsigmoid(beta * (policy_margin - reference_margin)).mean()


def _log_odds(logp):
    """GIVEN. Stable log(p / (1-p)) from a log-prob."""
    return logp - torch.log1p(-torch.exp(logp).clamp(max=1 - 1e-7))


def orpo_loss(policy_chosen_logps, policy_rejected_logps, chosen_nll, lam=0.1):
    """Chosen-response SFT plus the mean ORPO preference penalty."""
    log_odds_ratio = _log_odds(policy_chosen_logps) - _log_odds(policy_rejected_logps)
    return chosen_nll.mean() - lam * F.logsigmoid(log_odds_ratio).mean()


def grpo_advantages(rewards, eps=1e-6):
    """Standardize rewards across the final (sampled group) dimension."""
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    # The assignment checks the sample standard deviation. A one-response group
    # carries no relative preference signal and therefore has zero advantage.
    std = (rewards.std(dim=-1, unbiased=True, keepdim=True)
           if rewards.shape[-1] > 1 else torch.zeros_like(centered))
    return centered / (std + eps)


def grpo_loss(new_logprobs, old_logprobs, advantages, clip_eps=0.2):
    """Mean clipped group-relative policy surrogate."""
    ratio = torch.exp(new_logprobs - old_logprobs.detach())
    unclipped = ratio * advantages.detach()
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages.detach()
    return -torch.minimum(unclipped, clipped).mean()


@torch.no_grad()
def preference_margin(model, batch) -> float:
    """Mean chosen minus rejected sequence log-probability (GIVEN helper)."""
    cl = sequence_logprobs(model(batch["chosen_ids"]), batch["chosen_ids"], batch["chosen_mask"])
    rl = sequence_logprobs(model(batch["rejected_ids"]), batch["rejected_ids"], batch["rejected_mask"])
    return (cl - rl).mean().item()
