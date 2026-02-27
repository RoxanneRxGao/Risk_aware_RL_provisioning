"""
Configuration — Updated for Burst-Aware Risk RL

New additions:
  - BurstConfig / DiurnalConfig (re-exported from traffic module)
  - CVaRConfig  (step-level or window-level)
  - episode_duration is now in TrafficConfig
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

from traffic.burst_traffic import BurstConfig, DiurnalConfig


# ─────────────────────────────────────────────────────────────
@dataclass
class NetworkConfig:
    topology_name: str = "US24"
    topology_path: str = ""
    k_paths: int = 5
    bands: List[str] = field(default_factory=lambda: ['C', 'L', 'S'])
    slots_per_band: int = 400
    slot_bandwidth_ghz: float = 12.5
    guard_band_slots: int = 1
    gsnr_data_path: Optional[str] = None
    gsnr_channel_spacing_ghz: float = 50.0

    def __post_init__(self):
        if not self.topology_path:
            self.topology_path = f"config_files/topo_{self.topology_name.lower()}_txOnly.xlsx"


# ─────────────────────────────────────────────────────────────
@dataclass
class TrafficConfig:
    """
    All times are in **seconds**.

    Typical optical-network lightpath holding times range from minutes
    to hours.  With holding_time=600 s (10 min) and base_load=3000 Erl
    the implied arrival rate is  λ = 3000/600 = 5.0 arrivals/s
    → ≈ 18 000 arrivals per hour, ≈ 432 000 per 24-hour day.

    At this load, the US24 network experiences ~4 % baseline blocking,
    which rises significantly during burst events — exactly the regime
    where risk-aware RL is useful.
    """
    base_load: float = 3000.0                   # offered load (Erlangs)
    mean_service_holding_time: float = 600.0    # 10 min  (seconds)
    bit_rates: List[int] = field(default_factory=lambda: [100, 200, 400])
    bit_rate_probabilities: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])

    # Episode = wall-clock time  (seconds)
    episode_duration: float = 86400.0           # 24 h

    # Burst / diurnal
    burst_config: BurstConfig = field(default_factory=BurstConfig)
    diurnal_config: DiurnalConfig = field(default_factory=DiurnalConfig)
    burst_observable: bool = True               # True → known burst

    def get_load(self) -> float:
        return self.base_load

    def get_arrival_rate(self) -> float:
        """Mean arrival rate λ = load / holding_time  (arrivals/s)."""
        return self.base_load / self.mean_service_holding_time

    def get_arrivals_per_episode(self) -> float:
        """Approximate expected arrivals per episode (ignoring bursts)."""
        return self.get_arrival_rate() * self.episode_duration


# ─────────────────────────────────────────────────────────────
@dataclass
class EnvConfig:
    monitoring_window_s: float = 900.0      # 15 min sliding window
    use_mask: bool = True

    # Reward
    blocking_penalty_scale: float = 1.0
    bitrate_normalization: float = 400.0

    # Criticality
    criticality_method: str = 'betweenness'
    highrisk_quantile: float = 0.10

    # Encoder
    use_original_encoder: bool = True


# ─────────────────────────────────────────────────────────────
@dataclass
class CVaRConfig:
    enabled: bool = True
    alpha: float = 0.1                      # risk level (worst 10 %)
    cvar_weight: float = 0.5               # blend CVaR vs mean
    level: str = 'step'                     # 'step' or 'window'
    window_size_s: float = 900.0            # only for level='window'


# ─────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    n_envs: int = 1
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: List[Dict] = None
    activation_fn: str = 'relu'

    # CVaR
    cvar: CVaRConfig = field(default_factory=CVaRConfig)

    # Logging
    log_dir: str = "./logs/"
    save_freq: int = 10000
    eval_freq: int = 5000
    n_eval_episodes: int = 5

    device: str = "auto"
    seed: int = 42

    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [dict(pi=[256, 256], vf=[256, 256])]


# ─────────────────────────────────────────────────────────────
def get_default_config() -> Dict:
    return {
        'network': NetworkConfig(),
        'traffic': TrafficConfig(),
        'env': EnvConfig(),
        'training': TrainingConfig(),
    }


def print_config(cfg: Dict):
    print("=" * 70)
    for cat, obj in cfg.items():
        print(f"\n{cat.upper()}:")
        for k, v in obj.__dict__.items():
            if not k.startswith('_'):
                print(f"  {k:.<42} {v}")
    print("=" * 70)
