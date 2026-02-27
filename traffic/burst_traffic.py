"""
Burst-Aware Traffic Generation — HIGHLY OPTIMIZED VERSION

Inspired by the simple old TrafficGenerator but with burst support.

KEY OPTIMIZATIONS vs previous burst_traffic.py:
  1. No sliding window tracking in generator (moved to env if needed)
  2. Pre-computed burst lookup array for O(1) active burst check
  3. Simplified diurnal using direct math (no lookup tables)
  4. Minimal object allocation in hot path
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import bisect


@dataclass
class Service:
    """Represents a lightpath / service request."""
    service_id: int
    source: str
    source_id: int
    destination: str
    destination_id: int
    arrival_time: float
    holding_time: float
    bit_rate: float
    accepted: bool = False
    path: Optional[object] = None
    band_id: Optional[int] = None
    channels: Optional[np.ndarray] = None
    modulation: Optional[object] = None


@dataclass
class DiurnalConfig:
    """Diurnal (day/night) traffic profile."""
    enabled: bool = True
    amplitude: float = 0.3
    peak_hour: float = 14.0


@dataclass
class BurstConfig:
    """Burst event parameters."""
    enabled: bool = True
    burst_rate_per_day: float = 3.0
    burst_duration_min_s: float = 300.0
    burst_duration_max_s: float = 1800.0
    burst_factor_min: float = 2.0
    burst_factor_max: float = 5.0
    min_bursts_per_episode: int = 0


class BurstTrafficGenerator:
    """
    Traffic generator with burst events — HIGHLY OPTIMIZED.
    
    Design principles from old fast version:
      - Minimal per-service overhead
      - No sliding window tracking (env does this if needed)
      - Simple state, fast generate_service()
    """

    def __init__(
        self,
        nodes: List[str],
        base_load: float = 900.0,
        mean_holding_time: float = 10.0,
        bit_rates: List[int] = None,
        bit_rate_probs: List[float] = None,
        episode_duration: float = 86400.0,
        diurnal_cfg: DiurnalConfig = None,
        burst_cfg: BurstConfig = None,
        burst_observable: bool = True,
        random_start_hour: bool = False,
        seed: int = 42,
    ):
        # Sort nodes for consistent ordering (like old version)
        nodes = sorted(nodes, key=lambda x: int(x))
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.node_to_id = {n: i for i, n in enumerate(nodes)}

        self.base_load = base_load
        self.mean_holding_time = mean_holding_time
        self.episode_duration = episode_duration
        self.random_start_hour = random_start_hour

        # Bitrate setup - numpy arrays for fast sampling
        self.bit_rates = np.array(bit_rates or [100, 200, 400], dtype=np.float64)
        if bit_rate_probs is None:
            self.bit_rate_probs = np.ones(len(self.bit_rates)) / len(self.bit_rates)
        else:
            self.bit_rate_probs = np.array(bit_rate_probs, dtype=np.float64)

        # Configs
        self.diurnal_cfg = diurnal_cfg or DiurnalConfig()
        self.burst_cfg = burst_cfg or BurstConfig()
        self.burst_observable = burst_observable

        # RNG
        self.rng = np.random.default_rng(seed)

        # Baseline rate: λ = load / holding_time
        self._base_rate = base_load / mean_holding_time
        
        # Pre-compute constants for normalization
        max_diurnal = 1.0 + (self.diurnal_cfg.amplitude if self.diurnal_cfg.enabled else 0.0)
        max_burst = self.burst_cfg.burst_factor_max if self.burst_cfg.enabled else 1.0
        self._max_rate = self._base_rate * max_diurnal * max_burst
        
        # Pre-compute diurnal constants
        self._diurnal_enabled = self.diurnal_cfg.enabled
        self._diurnal_amp = self.diurnal_cfg.amplitude
        self._diurnal_peak = self.diurnal_cfg.peak_hour
        self._two_pi_over_24 = 2.0 * np.pi / 24.0

        # Episode state
        self.current_time: float = 0.0
        self.service_counter: int = 0
        self._wall_clock_offset: float = 0.0
        
        # Burst state - arrays for fast lookup
        self._burst_starts: np.ndarray = np.array([])
        self._burst_ends: np.ndarray = np.array([])
        self._burst_factors: np.ndarray = np.array([])
        self._num_bursts: int = 0
        
        # Cache for get_traffic_info (avoid recomputing)
        self._cached_time: float = -1.0
        self._cached_info: Dict = {}
        
        # ═══════════════════════════════════════════════════════
        # LIGHTWEIGHT blocking rate tracking (fixed-size arrays)
        # Much faster than sliding window with timestamp filtering
        # ═══════════════════════════════════════════════════════
        self._outcome_window_size = 100  # Track last 100 services
        self._outcome_buffer = np.zeros(self._outcome_window_size, dtype=np.int8)
        self._outcome_idx = 0
        self._outcome_count = 0

    def reset(self, seed: Optional[int] = None):
        """Reset for a new episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.current_time = 0.0
        self.service_counter = 0
        self._cached_time = -1.0
        
        # Reset outcome tracking
        self._outcome_buffer.fill(0)
        self._outcome_idx = 0
        self._outcome_count = 0

        if self.random_start_hour:
            self._wall_clock_offset = self.rng.uniform(0.0, 86400.0)
        else:
            self._wall_clock_offset = 0.0

        self._generate_burst_schedule()

    def _generate_burst_schedule(self):
        """Pre-generate burst events as numpy arrays for fast lookup."""
        if not self.burst_cfg.enabled:
            self._burst_starts = np.array([])
            self._burst_ends = np.array([])
            self._burst_factors = np.array([])
            self._num_bursts = 0
            return

        bcfg = self.burst_cfg
        avg_bursts = bcfg.burst_rate_per_day * (self.episode_duration / 86400.0)
        n_bursts = max(self.rng.poisson(avg_bursts), bcfg.min_bursts_per_episode)
        
        if n_bursts == 0:
            self._burst_starts = np.array([])
            self._burst_ends = np.array([])
            self._burst_factors = np.array([])
            self._num_bursts = 0
            return

        # Generate all bursts at once (vectorized)
        starts = self.rng.uniform(0.0, self.episode_duration * 0.9, size=n_bursts)
        durations = self.rng.uniform(bcfg.burst_duration_min_s, bcfg.burst_duration_max_s, size=n_bursts)
        factors = self.rng.uniform(bcfg.burst_factor_min, bcfg.burst_factor_max, size=n_bursts)
        
        ends = np.minimum(starts + durations, self.episode_duration)
        
        # Sort by start time
        order = np.argsort(starts)
        self._burst_starts = starts[order]
        self._burst_ends = ends[order]
        self._burst_factors = factors[order]
        self._num_bursts = n_bursts

    def _get_burst_factor(self, t: float) -> float:
        """Get burst multiplier at time t — O(log n) with binary search."""
        if self._num_bursts == 0:
            return 1.0
        
        # Find bursts that could be active (start <= t)
        # Using binary search for efficiency
        idx = bisect.bisect_right(self._burst_starts, t)
        
        if idx == 0:
            return 1.0
        
        # Check recent bursts (those that started before t)
        max_factor = 1.0
        for i in range(max(0, idx - 5), idx):  # Check last few (bursts can overlap)
            if self._burst_starts[i] <= t < self._burst_ends[i]:
                max_factor = max(max_factor, self._burst_factors[i])
        
        return max_factor

    def _current_rate(self, t: float) -> float:
        """Instantaneous arrival rate λ(t) — FAST."""
        rate = self._base_rate

        # Diurnal modulation (inline computation)
        if self._diurnal_enabled:
            wall_t = t + self._wall_clock_offset
            t_hours = (wall_t / 3600.0) % 24.0
            rate *= 1.0 + self._diurnal_amp * np.cos(self._two_pi_over_24 * (t_hours - self._diurnal_peak))

        # Burst multiplier
        rate *= self._get_burst_factor(t)

        return max(rate, 1e-6)

    def generate_service(self) -> Service:
        """Generate next service — MINIMAL OVERHEAD like old version."""
        # Inter-arrival from current rate
        lam = self._current_rate(self.current_time)
        interval = self.rng.exponential(1.0 / lam)
        self.current_time += interval

        # Holding time
        holding_time = self.rng.exponential(self.mean_holding_time)

        # Source/destination (avoid while loop for different nodes)
        src_idx = self.rng.integers(0, self.num_nodes)
        dst_idx = self.rng.integers(0, self.num_nodes - 1)
        if dst_idx >= src_idx:
            dst_idx += 1

        # Bitrate
        bit_rate = float(self.rng.choice(self.bit_rates, p=self.bit_rate_probs))

        svc = Service(
            service_id=self.service_counter,
            source=self.nodes[src_idx],
            source_id=src_idx,
            destination=self.nodes[dst_idx],
            destination_id=dst_idx,
            arrival_time=self.current_time,
            holding_time=holding_time,
            bit_rate=bit_rate,
        )
        self.service_counter += 1
        
        # Invalidate cache
        self._cached_time = -1.0

        return svc

    def record_outcome(self, accepted: bool):
        """Record outcome — O(1) circular buffer update."""
        # Store in circular buffer (1 = accepted, 0 = blocked)
        idx = self._outcome_idx % self._outcome_window_size
        self._outcome_buffer[idx] = 1 if accepted else 0
        self._outcome_idx += 1
        self._outcome_count = min(self._outcome_count + 1, self._outcome_window_size)
        
        # Invalidate cache since blocking rate changed
        self._cached_time = -1.0
    
    def _get_recent_blocking_rate(self) -> float:
        """Get blocking rate from last N services — O(1) with pre-summed buffer."""
        if self._outcome_count == 0:
            return 0.0
        # Count accepted in the buffer
        if self._outcome_count < self._outcome_window_size:
            accepted = np.sum(self._outcome_buffer[:self._outcome_count])
        else:
            accepted = np.sum(self._outcome_buffer)
        return 1.0 - (accepted / self._outcome_count)

    def get_traffic_info(self) -> Dict[str, float]:
        """Return traffic state dict — with caching."""
        # Return cached if time hasn't changed AND no new outcomes recorded
        if self.current_time == self._cached_time and self._cached_info:
            return self._cached_info
        
        t = self.current_time
        wall_t = t + self._wall_clock_offset
        tod = (wall_t % 86400.0) / 86400.0
        
        # Compute sin/cos once
        angle = 2.0 * np.pi * tod
        
        # Get recent blocking rate from circular buffer
        recent_bp = self._get_recent_blocking_rate()
        
        info = {
            'time_of_day': tod,
            'time_of_day_sin': np.sin(angle),
            'time_of_day_cos': np.cos(angle),
            'recent_rate_norm': self._current_rate(t) / self._max_rate,
            'recent_blocking_rate': recent_bp,  # ← Now properly tracked!
        }

        if self.burst_observable:
            burst_factor = self._get_burst_factor(t)
            info['current_rate_norm'] = self._current_rate(t) / self._max_rate
            info['burst_active'] = 1.0 if burst_factor > 1.0 else 0.0
            
            # Burst remaining (find active burst)
            remaining = 0.0
            if burst_factor > 1.0:
                for i in range(self._num_bursts):
                    if self._burst_starts[i] <= t < self._burst_ends[i]:
                        remaining = max(remaining, (self._burst_ends[i] - t) / self.burst_cfg.burst_duration_max_s)
            info['burst_remaining_norm'] = remaining
        else:
            info['current_rate_norm'] = 0.0
            info['burst_active'] = 0.0
            info['burst_remaining_norm'] = 0.0
        
        # Cache
        self._cached_time = t
        self._cached_info = info

        return info

    def get_current_time(self) -> float:
        return self.current_time

    def get_load(self) -> float:
        return self.base_load

    def get_burst_schedule(self) -> List[Dict]:
        """Return burst schedule for plotting."""
        return [
            {'start': self._burst_starts[i], 'end': self._burst_ends[i], 'factor': self._burst_factors[i]}
            for i in range(self._num_bursts)
        ]

    def get_start_hour(self) -> float:
        return (self._wall_clock_offset / 3600.0) % 24.0

    @property
    def num_temporal_features(self) -> int:
        return 8
