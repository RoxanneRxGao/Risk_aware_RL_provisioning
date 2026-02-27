"""
Step-Level CVaR PPO — Advantage Re-weighting for Intra-Episode Risk

Standard PPO maximises E[Σ rₜ].
This module adds a callback that, after each rollout, re-weights the
advantage estimates so that timesteps in the **tail** of the per-step
reward distribution receive amplified gradients.

Effect: the policy is pushed harder to improve performance at the
worst individual timesteps (= blocking events during traffic bursts).

Usage
-----
    from models.step_cvar_ppo import StepCVaRCallback

    callback = StepCVaRCallback(alpha=0.1, cvar_weight=0.5)
    model = MaskablePPO("MlpPolicy", env, ...)
    model.learn(total_timesteps=..., callback=callback)
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from stable_baselines3.common.callbacks import BaseCallback


class StepCVaRCallback(BaseCallback):
    """
    Callback that applies step-level CVaR advantage re-weighting.

    At the end of every rollout (before PPO's optimisation pass):
      1. Read per-step rewards from the rollout buffer.
      2. Compute VaR_α  =  α-quantile of rewards.
      3. Build weight vector
             w_tail = cvar_weight / α       (for r ≤ VaR_α)
             w_body = (1-cvar_weight) / (1-α) (for r > VaR_α)
         so that E[w] = 1  and tail steps are upweighted.
      4. Multiply ``buffer.advantages`` element-wise by w.

    Parameters
    ----------
    alpha : float
        Risk level ∈ (0, 1).  0.1 → worst 10 % of timesteps.
    cvar_weight : float
        Blend between pure mean (0) and pure CVaR (1).
        0.5 is a balanced starting point.
    verbose : int
        Verbosity.
    """

    def __init__(self, alpha: float = 0.1, cvar_weight: float = 0.5,
                 verbose: int = 0):
        super().__init__(verbose)
        assert 0 < alpha < 1, "alpha must be in (0, 1)"
        assert 0 <= cvar_weight <= 1, "cvar_weight must be in [0, 1]"
        self.alpha = alpha
        self.cvar_weight = cvar_weight

        # Tracking (for tensorboard / analysis)
        self.history: Dict[str, list] = {
            'step_var': [], 'step_cvar': [], 'mean_reward': [],
            'tail_frac': [], 'tail_mean_adv': [], 'body_mean_adv': [],
        }

    # ──────────────────────────────────────────────────────────
    def _on_step(self) -> bool:
        return True  # nothing per-step

    def _on_rollout_end(self) -> None:
        """Called after rollout collection, before PPO update."""
        buf = self.model.rollout_buffer

        # rewards shape: (n_steps, n_envs)
        rewards = buf.rewards.copy().flatten()
        n = len(rewards)
        if n == 0:
            return

        # ── Step 1: VaR and CVaR ─────────────────────────────
        var_threshold = float(np.quantile(rewards, self.alpha))
        tail_mask = rewards <= var_threshold
        n_tail = int(tail_mask.sum())
        n_body = n - n_tail

        if n_tail > 0:
            cvar_val = float(rewards[tail_mask].mean())
        else:
            cvar_val = var_threshold

        # ── Step 2: Build weights ────────────────────────────
        weights = np.ones(n, dtype=np.float32)
        if n_tail > 0 and n_body > 0 and self.cvar_weight > 0:
            w_tail = self.cvar_weight / max(self.alpha, 1e-8)
            w_body = (1.0 - self.cvar_weight) / max(1.0 - self.alpha, 1e-8)
            weights[tail_mask] = w_tail
            weights[~tail_mask] = w_body

        # Reshape to match buffer
        weights_2d = weights.reshape(buf.advantages.shape)

        # ── Step 3: Apply to advantages ──────────────────────
        old_adv = buf.advantages.copy()
        buf.advantages *= weights_2d

        # ── Logging ──────────────────────────────────────────
        self.history['step_var'].append(var_threshold)
        self.history['step_cvar'].append(cvar_val)
        self.history['mean_reward'].append(float(rewards.mean()))
        self.history['tail_frac'].append(n_tail / n)

        tail_adv = old_adv.flatten()[tail_mask]
        body_adv = old_adv.flatten()[~tail_mask]
        self.history['tail_mean_adv'].append(
            float(tail_adv.mean()) if len(tail_adv) > 0 else 0.0)
        self.history['body_mean_adv'].append(
            float(body_adv.mean()) if len(body_adv) > 0 else 0.0)

        if self.logger is not None:
            self.logger.record('cvar/step_var', var_threshold)
            self.logger.record('cvar/step_cvar', cvar_val)
            self.logger.record('cvar/mean_reward', float(rewards.mean()))
            self.logger.record('cvar/tail_fraction', n_tail / n)
            self.logger.record('cvar/weight_tail', float(weights[tail_mask].mean()) if n_tail else 0)
            self.logger.record('cvar/weight_body', float(weights[~tail_mask].mean()) if n_body else 0)


# ──────────────────────────────────────────────────────────────
# Episode-level CVaR callback (for comparison / ablation)
# ──────────────────────────────────────────────────────────────
class EpisodeCVaRCallback(BaseCallback):
    """
    Tracks episode-level CVaR (for monitoring / comparison only).
    Does NOT modify advantages.
    """

    def __init__(self, alpha: float = 0.1, verbose: int = 0):
        super().__init__(verbose)
        self.alpha = alpha
        self.episode_returns: List[float] = []
        self.history: Dict[str, list] = {
            'ep_mean': [], 'ep_var': [], 'ep_cvar': [],
        }

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_returns.append(info['episode']['r'])
        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_returns) == 0:
            return
        rets = np.array(self.episode_returns)
        var_val = float(np.quantile(rets, self.alpha))
        tail = rets[rets <= var_val]
        cvar_val = float(tail.mean()) if len(tail) else var_val

        self.history['ep_mean'].append(float(rets.mean()))
        self.history['ep_var'].append(var_val)
        self.history['ep_cvar'].append(cvar_val)

        if self.logger is not None:
            self.logger.record('cvar_ep/mean', float(rets.mean()))
            self.logger.record('cvar_ep/var', var_val)
            self.logger.record('cvar_ep/cvar', cvar_val)

        self.episode_returns = []


# ──────────────────────────────────────────────────────────────
# Re-export windowed CVaR from standalone utility
# ──────────────────────────────────────────────────────────────
from utils.windowed_cvar import compute_windowed_cvar  # noqa: F401
