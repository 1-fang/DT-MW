"""
DT-MW manager utilities.

Main paper-aligned route:
  UnifiedLLMPriorManager: context X_t -> instruction, lambda_prior, thresholds
  DualController:         real ORR/Fairness feedback -> corrected lambda
  lambda_to_beta:         corrected lambda -> beta used by DRS

Legacy beta-fusion classes are kept below for old checkpoints/scripts, but the
DT-MW main training path should use DualController rather than beta-space fusion.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def lambda_to_beta(lambda_vec: np.ndarray, max_lambda: float = 20.0) -> np.ndarray:
    """Map dual variables [lambda_orr, lambda_fair] to beta on Delta^2.

    The GMV coefficient is the primal objective weight 1.0, so
    beta = [1, lambda_orr, lambda_fair] / (1 + lambda_orr + lambda_fair).
    """
    lam = np.asarray(lambda_vec, dtype=np.float32).reshape(-1)
    if lam.size != 2:
        raise ValueError(f"lambda_vec must have shape (2,), got {lam.shape}")
    lam = np.clip(lam, 0.0, max_lambda)
    raw = np.array([1.0, lam[0], lam[1]], dtype=np.float32)
    total = float(raw.sum())
    if total <= 0:
        return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    return (raw / total).astype(np.float32)


def beta_to_lambda(beta: np.ndarray, max_lambda: float = 20.0,
                   eps: float = 1e-4) -> np.ndarray:
    """Convert an old beta prior to lambda_prior for backward compatibility."""
    b = np.asarray(beta, dtype=np.float32).reshape(-1)
    if b.size != 3:
        raise ValueError(f"beta must have shape (3,), got {b.shape}")
    b = np.clip(b, eps, 1.0)
    b = b / b.sum()
    lam = np.array([b[1] / b[0], b[2] / b[0]], dtype=np.float32)
    return np.clip(lam, 0.0, max_lambda)


def infer_thresholds_from_beta(beta: np.ndarray) -> Tuple[float, float]:
    """Fallback thresholds when the manager/scenario only provides beta.

    These thresholds are semantic priors, not guarantees. The DualController
    corrects lambda online using observed ORR/Fairness violations.
    """
    b = np.asarray(beta, dtype=np.float32).reshape(-1)
    if b.size != 3:
        b = np.array([0.4, 0.4, 0.2], dtype=np.float32)
    b = np.clip(b, 0.0, 1.0)
    total = float(b.sum())
    b = b / total if total > 0 else np.array([0.4, 0.4, 0.2], dtype=np.float32)
    rho_min = float(np.clip(0.45 + 0.45 * b[1], 0.45, 0.90))
    phi_min = float(np.clip(0.45 + 0.35 * b[2], 0.45, 0.80))
    return rho_min, phi_min


# ---------------------------------------------------------------------------
# UnifiedLLMPriorManager — wraps the existing UnifiedManager
# Produces: instruction_text, instruction_emb, lambda_prior, thresholds
# ---------------------------------------------------------------------------

class UnifiedLLMPriorManager:
    """
    LLM semantic prior: context -> instruction + lambda_prior + thresholds.

    Wraps the UnifiedManager (ScenarioPrompt or LLM API backend).
    This is the open-loop component — no learning from worker returns.
    """

    def __init__(self, unified_manager):
        self._manager = unified_manager

    def _normalize_output(self, out: Dict) -> Dict:
        beta = out.get('beta', out.get('beta_prior', [0.4, 0.4, 0.2]))
        beta = np.asarray(beta, dtype=np.float32)
        if beta.size != 3:
            beta = np.array([0.4, 0.4, 0.2], dtype=np.float32)
        beta = np.clip(beta, 0.0, 1.0)
        beta = beta / beta.sum() if beta.sum() > 0 else np.array([0.4, 0.4, 0.2], dtype=np.float32)

        lambda_prior = out.get('lambda_prior', out.get('lambda', None))
        if lambda_prior is None:
            lambda_prior = beta_to_lambda(beta)
        else:
            lambda_prior = np.asarray(lambda_prior, dtype=np.float32).reshape(-1)
            if lambda_prior.size != 2:
                lambda_prior = beta_to_lambda(beta)
            lambda_prior = np.clip(lambda_prior, 0.0, 20.0)

        thresholds = out.get('thresholds', {}) or {}
        rho_min = out.get('rho_min', thresholds.get('rho_min', None))
        phi_min = out.get('phi_min', thresholds.get('phi_min', None))
        if rho_min is None or phi_min is None:
            rho_min, phi_min = infer_thresholds_from_beta(beta)

        beta_prior = lambda_to_beta(lambda_prior)
        return {
            'instruction_text': out['instruction_text'],
            'instruction_emb':  out['instruction_emb'],
            'lambda_prior':     lambda_prior.astype(np.float32),
            'rho_min':          float(np.clip(rho_min, 0.0, 1.0)),
            'phi_min':          float(np.clip(phi_min, 0.0, 1.0)),
            'beta_prior':       beta_prior.astype(np.float32),
            # Backward-compatible aliases for old logging/scripts.
            'beta_llm':         beta_prior.astype(np.float32),
            'beta':             beta_prior.astype(np.float32).tolist(),
            'label':            out.get('label', 0),
        }

    def step(self, context: Dict, step_t: int) -> Dict:
        out = self._manager.step(context, step_t)
        return self._normalize_output(out)

    def random_switch(self) -> Dict:
        out = self._manager.random_switch()
        return self._normalize_output(out)

    def get_all_embeddings(self) -> np.ndarray:
        return self._manager.get_all_embeddings()

    def build_context(self, **kwargs) -> Dict:
        return self._manager.build_context(**kwargs)


class DualController:
    """Projected dual feedback controller for service/fairness constraints.

    Update:
      lambda_{t+1} = Pi_R+ [ lambda_t
        + xi * [rho_min - observed_orr, phi_min - observed_fair]
        - kappa * (lambda_t - lambda_prior) ]
    """

    def __init__(self, xi: float = 0.1, kappa: float = 0.05,
                 lambda_max: float = 20.0):
        self.xi = float(xi)
        self.kappa = float(kappa)
        self.lambda_max = float(lambda_max)
        self.lambda_t: Optional[np.ndarray] = None

    def reset(self, lambda_prior: Optional[np.ndarray] = None) -> np.ndarray:
        if lambda_prior is None:
            lambda_prior = np.array([1.0, 0.5], dtype=np.float32)
        lam = np.asarray(lambda_prior, dtype=np.float32).reshape(-1)
        if lam.size != 2:
            lam = np.array([1.0, 0.5], dtype=np.float32)
        self.lambda_t = np.clip(lam, 0.0, self.lambda_max).astype(np.float32)
        return self.lambda_t.copy()

    def step(self, lambda_prior: np.ndarray,
             rho_min: float,
             phi_min: float,
             observed_orr: float,
             observed_fair: float) -> Dict:
        prior = np.asarray(lambda_prior, dtype=np.float32).reshape(-1)
        if prior.size != 2:
            prior = np.array([1.0, 0.5], dtype=np.float32)
        prior = np.clip(prior, 0.0, self.lambda_max)

        if self.lambda_t is None:
            self.reset(prior)

        obs_orr = float(np.clip(observed_orr, 0.0, 1.0))
        obs_fair = float(np.clip(observed_fair, 0.0, 1.0))
        rho = float(np.clip(rho_min, 0.0, 1.0))
        phi = float(np.clip(phi_min, 0.0, 1.0))
        violation = np.array([rho - obs_orr, phi - obs_fair], dtype=np.float32)

        lam = (self.lambda_t
               + self.xi * violation
               - self.kappa * (self.lambda_t - prior))
        self.lambda_t = np.clip(lam, 0.0, self.lambda_max).astype(np.float32)
        beta = lambda_to_beta(self.lambda_t, max_lambda=self.lambda_max)
        return {
            'lambda': self.lambda_t.copy(),
            'lambda_prior': prior.astype(np.float32),
            'beta': beta,
            'violation': violation,
            'rho_min': rho,
            'phi_min': phi,
            'observed_orr': obs_orr,
            'observed_fair': obs_fair,
        }

    def get_beta(self) -> np.ndarray:
        if self.lambda_t is None:
            self.reset()
        return lambda_to_beta(self.lambda_t, max_lambda=self.lambda_max)


# ---------------------------------------------------------------------------
# LearnableManager — MLP that maps env_stats → beta_lm
# Closed-loop: trained on worker rollout returns
# ---------------------------------------------------------------------------

class LearnableManager(nn.Module):
    """
    Return-driven correction signal: (env_stats, beta_llm) → beta_lm.

    Conditioned on the LLM's semantic prior beta_llm so that the network
    learns "given what the LLM suggested and the current env state, what
    correction is needed?" rather than predicting beta_fused from scratch.

    Input:
      env_stats (8-d): [drivers_norm, orders_norm, orr, sd_ratio_norm,
                        time_sin, time_cos, hour_norm, minute_norm]
      beta_llm  (3-d): LLM prior preference vector
      Optional: weather_emb (sbert_dim) projected to 32-d

    Output: beta_lm ∈ Δ² (3-d simplex via softmax)
    """

    ENV_STATS_DIM = 8
    PRIOR_DIM = 3  # beta_llm dimension

    def __init__(self, hidden_dim=64, output_dim=3, sbert_dim=384):
        super().__init__()
        self.output_dim = output_dim
        self.sbert_dim = sbert_dim

        self.weather_proj = nn.Linear(sbert_dim, 32)
        # 8 (env_stats) + 3 (beta_llm) + 32 (weather_proj) = 43
        combined_dim = self.ENV_STATS_DIM + self.PRIOR_DIM + 32

        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.register_parameter(
            'temperature', nn.Parameter(torch.tensor(1.0)))

    def forward(self, env_stats: torch.Tensor,
                beta_llm: torch.Tensor,
                weather_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch = env_stats.shape[0]
        if weather_emb is None:
            weather_emb = torch.zeros(
                batch, self.sbert_dim, device=env_stats.device)
        elif weather_emb.dim() == 1:
            weather_emb = weather_emb.unsqueeze(0).expand(batch, -1)
        if beta_llm.dim() == 1:
            beta_llm = beta_llm.unsqueeze(0).expand(batch, -1)

        w_proj = F.relu(self.weather_proj(weather_emb))
        x = torch.cat([env_stats, beta_llm, w_proj], dim=-1)
        logits = self.net(x)
        return F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)

    def get_beta(self, env_stats_dict: Dict,
                 beta_llm: np.ndarray,
                 weather_emb: Optional[np.ndarray] = None) -> np.ndarray:
        """Numpy interface: dict + beta_llm + optional weather → beta_lm (3,)."""
        drivers = float(env_stats_dict.get('total_drivers', 2000))
        orders = float(env_stats_dict.get('total_orders', 1000))
        orr = float(env_stats_dict.get('avg_orr', 0.5))
        sd_ratio = drivers / max(orders, 1)

        step_t = int(env_stats_dict.get('step_t', 0))
        dispatch_interval = int(env_stats_dict.get('dispatch_interval', 10))
        time_min = (step_t * dispatch_interval) % 1440
        hour = time_min // 60
        minute = time_min % 60

        drivers_norm = min(drivers / 3000, 1.0)
        orders_norm = min(orders / 2000, 1.0)
        sd_ratio_norm = min(sd_ratio / 3.0, 1.0)
        time_sin = np.sin(2 * np.pi * (hour * 60 + minute) / 1440)
        time_cos = np.cos(2 * np.pi * (hour * 60 + minute) / 1440)

        device = next(self.parameters()).device
        stats_t = torch.FloatTensor([
            drivers_norm, orders_norm, orr, sd_ratio_norm,
            time_sin, time_cos, hour / 24.0, minute / 60.0
        ]).unsqueeze(0).to(device)

        beta_llm_t = torch.FloatTensor(beta_llm).unsqueeze(0).to(device)

        weather_t = None
        if weather_emb is not None:
            weather_t = torch.FloatTensor(
                weather_emb).unsqueeze(0).to(device)

        with torch.no_grad():
            beta = self.forward(
                stats_t, beta_llm_t, weather_t).squeeze(0).cpu().numpy()
        return beta.astype(np.float32)


# ---------------------------------------------------------------------------
# ManagerBuffer — stores rollout outcomes for LearnableManager training
# ---------------------------------------------------------------------------

class ManagerBuffer:
    """
    Stores (env_stats, beta_llm, beta_lm, beta_fused, episode_outcome)
    for closed-loop manager training.
    """

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.entries: List[Dict] = []
        self.ptr = 0

    def push(self, env_stats: Dict, beta_llm: np.ndarray,
             beta_lm: np.ndarray, beta_fused: np.ndarray,
             episode_return: float, episode_orr: float,
             episode_gmv: float, episode_gini: float,
             weather_emb: Optional[np.ndarray] = None):
        entry = {
            'env_stats': env_stats,
            'beta_llm': beta_llm.copy(),
            'beta_lm': beta_lm.copy(),
            'beta_fused': beta_fused.copy(),
            'episode_return': episode_return,
            'episode_orr': episode_orr,
            'episode_gmv': episode_gmv,
            'episode_gini': episode_gini,
            'weather_emb': weather_emb.copy() if weather_emb is not None else None,
        }
        if len(self.entries) < self.capacity:
            self.entries.append(entry)
        else:
            self.entries[self.ptr] = entry
        self.ptr = (self.ptr + 1) % self.capacity

    @property
    def size(self):
        return len(self.entries)

    def get_batch(self, device='cpu') -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Returns (env_stats, beta_llm, beta_fused, beta_lm, returns, weather) tensors."""
        env_tensors = []
        for e in self.entries:
            s = e['env_stats']
            drivers = float(s.get('total_drivers', 2000))
            orders = float(s.get('total_orders', 1000))
            orr = float(s.get('avg_orr', 0.5))
            sd_ratio = drivers / max(orders, 1)
            step_t = int(s.get('step_t', 0))
            di = int(s.get('dispatch_interval', 10))
            time_min = (step_t * di) % 1440
            hour = time_min // 60
            minute = time_min % 60

            env_tensors.append([
                min(drivers / 3000, 1.0),
                min(orders / 2000, 1.0),
                orr,
                min(sd_ratio / 3.0, 1.0),
                np.sin(2 * np.pi * (hour * 60 + minute) / 1440),
                np.cos(2 * np.pi * (hour * 60 + minute) / 1440),
                hour / 24.0,
                minute / 60.0,
            ])

        env_t = torch.FloatTensor(env_tensors).to(device)
        beta_llm_t = torch.FloatTensor(
            np.stack([e['beta_llm'] for e in self.entries])).to(device)
        beta_fused_t = torch.FloatTensor(
            np.stack([e['beta_fused'] for e in self.entries])).to(device)
        beta_lm_t = torch.FloatTensor(
            np.stack([e['beta_lm'] for e in self.entries])).to(device)
        returns_t = torch.FloatTensor(
            [e['episode_return'] for e in self.entries]).unsqueeze(1).to(device)

        weather_t = None
        if any(e['weather_emb'] is not None for e in self.entries):
            dim = next(e['weather_emb'].shape[0]
                       for e in self.entries if e['weather_emb'] is not None)
            wl = [e['weather_emb'] if e['weather_emb'] is not None
                  else np.zeros(dim, dtype=np.float32)
                  for e in self.entries]
            weather_t = torch.FloatTensor(np.stack(wl)).to(device)

        return env_t, beta_llm_t, beta_fused_t, beta_lm_t, returns_t, weather_t

    def clear(self):
        self.entries.clear()
        self.ptr = 0


def train_learnable_manager(manager: LearnableManager,
                            buffer: ManagerBuffer,
                            optimizer: torch.optim.Optimizer,
                            device='cpu',
                            num_epochs=5) -> float:
    """
    Train LearnableManager to produce beta_lm that improves beta_fused outcomes.

    Uses beta_fused (the actually-deployed beta) as the REINFORCE target,
    not beta_lm alone. This ensures the LM learns what correction direction
    leads to better fused outcomes, not just to reproduce its own history.

    Losses:
      1. REINFORCE: log π(beta_fused | env) weighted by advantage
      2. Imitation: MSE toward beta_fused from top-50% return episodes
    """
    if buffer.size == 0:
        return 0.0

    env_t, beta_llm_t, beta_fused_t, beta_lm_t, returns_t, weather_t = buffer.get_batch(device)
    n = env_t.size(0)

    returns_norm = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-6)
    returns_norm = returns_norm.squeeze(1)

    median_ret = returns_t.median()
    high_mask = (returns_t >= median_ret).squeeze(1)

    total_loss = 0.0
    for _ in range(num_epochs):
        optimizer.zero_grad()
        # LM sees env_stats + beta_llm: learns correction conditioned on prior
        pred_beta = manager(env_t, beta_llm_t, weather_t)

        # REINFORCE: use beta_fused as target (what actually produced the return)
        log_prob = torch.sum(
            beta_fused_t * torch.log(pred_beta + 1e-8), dim=1)
        pg_loss = -torch.mean(returns_norm * log_prob)

        # Imitation: push toward beta_fused of high-return episodes
        if high_mask.sum() > 0:
            imitation_loss = torch.mean(torch.sum(
                (pred_beta[high_mask] - beta_fused_t[high_mask]) ** 2, dim=1))
        else:
            imitation_loss = torch.tensor(0.0, device=device)

        loss = 0.5 * pg_loss + 0.5 * imitation_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(manager.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / num_epochs


# ---------------------------------------------------------------------------
# FusionManager — combines beta_llm and beta_lm into beta_fused
# ---------------------------------------------------------------------------

class FusionManager:
    """
    Fuses LLM semantic prior with learnable correction:
      beta_fused = λ * beta_llm + (1-λ) * beta_lm

    Then L1-normalizes and optionally clips each dimension.

    Ablation modes controlled by fusion_lambda:
      λ=1.0 → LLM-only
      λ=0.0 → LM-only
      0<λ<1 → Fusion
    """

    def __init__(self, fusion_lambda: float = 0.5,
                 beta_clip_min: float = 0.05,
                 beta_clip_max: float = 0.90):
        self.fusion_lambda = fusion_lambda
        self.beta_clip_min = beta_clip_min
        self.beta_clip_max = beta_clip_max

    def fuse(self, beta_llm: np.ndarray,
             beta_lm: np.ndarray) -> np.ndarray:
        lam = self.fusion_lambda
        # Pure pass-through: no clipping when only one source is active
        if lam >= 1.0:
            return beta_llm.copy().astype(np.float32)
        if lam <= 0.0:
            return beta_lm.copy().astype(np.float32)

        raw = lam * beta_llm + (1.0 - lam) * beta_lm

        clipped = np.clip(raw, self.beta_clip_min, self.beta_clip_max)
        total = clipped.sum()
        if total > 0:
            clipped = clipped / total
        else:
            clipped = np.ones_like(clipped) / len(clipped)

        return clipped.astype(np.float32)
