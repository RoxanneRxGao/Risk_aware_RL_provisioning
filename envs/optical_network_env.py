"""
Optical Network Environment — HIGHLY OPTIMIZED VERSION

Inspired by the fast old RiskAwareProvisioningEnv but with time-based episodes.

KEY OPTIMIZATIONS:
  1. Pre-cached link IDs (like we added before)
  2. REMOVED per-step window blocking calculation (huge overhead!)
  3. Simplified info dict (only compute stats at episode end)
  4. Minimal object creation in step()
  5. Reused arrays where possible
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import heapq
from typing import Dict, List, Tuple, Optional
import networkx as nx

from utils.topology import Path
from utils.spectrum import first_fit_allocation
from utils.qot import slots_needed, SimpleQoTProvider
from envs.state_encoder import StateEncoder, EncoderConfig
from traffic.burst_traffic import BurstTrafficGenerator, Service


class OpticalNetworkEnv(gym.Env):
    """
    Time-based optical-network provisioning environment — HIGHLY OPTIMIZED.
    
    Inspired by the old fast RiskAwareProvisioningEnv design.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        topology: nx.DiGraph,
        ksp_dict: Dict[Tuple[str, str], List[Path]],
        qot_provider,
        edge_criticality: np.ndarray,
        encoder: StateEncoder,
        traffic_generator: BurstTrafficGenerator,
        *,
        bands: List[int] = None,
        slots_per_band: int = 400,
        slot_bandwidth_ghz: float = 12.5,
        guard_slots: int = 1,
        K: int = 5,
        episode_duration: float = 86400.0,
        monitoring_window_s: float = 900.0,
        use_action_masking: bool = True,
        blocking_penalty_scale: float = 1.0,
        bitrate_normalization: float = 400.0,
        seed: int = 42,
    ):
        super().__init__()

        # ── Network ──────────────────────────────────────────
        self.topology = topology
        self.ksp_dict = ksp_dict
        self.nodes = list(topology.nodes())
        self.num_nodes = len(self.nodes)
        self.num_links = topology.number_of_edges()
        self.node_to_idx = {n: topology.nodes[n]['index'] for n in self.nodes}

        self.bands = bands if bands is not None else [0, 1, 2]
        self.num_bands = len(self.bands)
        self.slots_per_band = slots_per_band
        self.slot_bandwidth_ghz = slot_bandwidth_ghz
        self.guard_slots = guard_slots
        self.K = K

        self.qot = qot_provider
        self.edge_criticality = edge_criticality
        self.encoder = encoder
        self.traffic_gen = traffic_generator
        self.use_action_masking = use_action_masking

        # ── Episode parameters ───────────────────────────────
        self.episode_duration = episode_duration
        self.monitoring_window_s = monitoring_window_s

        # ── Reward ───────────────────────────────────────────
        self.blocking_penalty_scale = blocking_penalty_scale
        self.bitrate_normalization = bitrate_normalization

        # ── Spaces ───────────────────────────────────────────
        n_actions = self.K * self.num_bands
        self.action_space = spaces.Discrete(n_actions)

        base_dim = self.encoder.obs_dim()
        self._temporal_dim = self.traffic_gen.num_temporal_features + 2
        obs_dim = base_dim + self._temporal_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.rng = np.random.default_rng(seed)

        # ═══════════════════════════════════════════════════════
        # OPTIMIZATION 1: Pre-cache ALL path link IDs at init
        # ═══════════════════════════════════════════════════════
        self._path_link_cache: Dict[int, np.ndarray] = {}
        for (src, dst), paths in ksp_dict.items():
            for path in paths:
                link_ids = self._compute_link_ids(path)
                self._path_link_cache[path.path_id] = link_ids
                path.links_id = link_ids  # Store in path object too

        # ═══════════════════════════════════════════════════════
        # OPTIMIZATION 2: Pre-allocate arrays
        # ═══════════════════════════════════════════════════════
        self._temporal_features = np.zeros(self._temporal_dim, dtype=np.float32)

        # ── Per-episode state ─────────────────────────────────
        self.spectrum_occ = None
        self.current_service: Optional[Service] = None
        self.current_paths: List[Path] = []
        self.current_mask: Optional[np.ndarray] = None
        self.current_time: float = 0.0
        self.step_count: int = 0
        self.release_events: list = []

        # Stats (minimal tracking)
        self._prev_global_util: float = 0.0
        self._services_processed: int = 0
        self._services_accepted: int = 0
        self._bitrate_requested: float = 0.0
        self._bitrate_carried: float = 0.0
        self._total_reward: float = 0.0
        self._burst_blocking: int = 0
        self._burst_total: int = 0
        self.step_rewards: List[float] = []
        
        # ═══════════════════════════════════════════════════════
        # LIGHTWEIGHT window blocking rate (for monitoring/info)
        # Uses circular buffer - O(1) update, O(1) query
        # ═══════════════════════════════════════════════════════
        self._window_size = 200  # Track last 200 services
        self._window_buffer = np.zeros(self._window_size, dtype=np.int8)
        self._window_idx = 0
        self._window_count = 0

    def _compute_link_ids(self, path: Path) -> np.ndarray:
        """Compute link IDs for a path (called once during init)."""
        if path.links_id is not None:
            return path.links_id
        ids = []
        for i in range(len(path.node_list) - 1):
            u, v = path.node_list[i], path.node_list[i + 1]
            ids.append(self.topology[u][v]['index'])
        return np.array(ids, dtype=np.int32)

    # ══════════════════════════════════════════════════════════
    # RESET
    # ══════════════════════════════════════════════════════════
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Spectrum - contiguous arrays
        self.spectrum_occ = {
            band: np.zeros((self.num_links, self.slots_per_band), dtype=np.int8)
            for band in self.bands
        }

        self.current_time = 0.0
        self.step_count = 0
        self.release_events = []
        self._prev_global_util = 0.0
        
        # Reset stats
        self._services_processed = 0
        self._services_accepted = 0
        self._bitrate_requested = 0.0
        self._bitrate_carried = 0.0
        self._total_reward = 0.0
        self._burst_blocking = 0
        self._burst_total = 0
        self.step_rewards = []
        
        # Reset window tracking
        self._window_buffer.fill(0)
        self._window_idx = 0
        self._window_count = 0

        self.traffic_gen.reset(seed=seed)
        self._generate_next_service()

        obs = self._get_observation()
        return obs, {}

    # ══════════════════════════════════════════════════════════
    # STEP — OPTIMIZED: Minimal overhead per step
    # ══════════════════════════════════════════════════════════
    def step(self, action: int):
        self._process_releases()

        self.step_count += 1
        self._services_processed += 1
        bitrate = self.current_service.bit_rate
        self._bitrate_requested += bitrate

        # ── Attempt allocation ───────────────────────────────
        accepted = False
        if self.use_action_masking and not self.current_mask[action]:
            accepted = False
        else:
            k = action // self.num_bands
            b = action % self.num_bands
            band = self.bands[b]
            if k < len(self.current_paths):
                path = self.current_paths[k]
                link_ids = self._path_link_cache.get(path.path_id)
                if link_ids is not None and len(link_ids) > 0:
                    accepted = self._attempt_allocation(path, band, link_ids)

        # ── Reward (simple, like old version) ────────────────
        reward = self._compute_reward(accepted, bitrate)

        if accepted:
            self._services_accepted += 1
            self._bitrate_carried += bitrate
        self.current_service.accepted = accepted

        self._total_reward += reward
        self.step_rewards.append(reward)
        
        # ═══════════════════════════════════════════════════════
        # LIGHTWEIGHT window tracking — O(1) circular buffer
        # ═══════════════════════════════════════════════════════
        idx = self._window_idx % self._window_size
        self._window_buffer[idx] = 1 if accepted else 0
        self._window_idx += 1
        self._window_count = min(self._window_count + 1, self._window_size)

        # Track burst blocking (lightweight)
        tinfo = self.traffic_gen.get_traffic_info()
        if tinfo.get('burst_active', 0.0) > 0.5:
            self._burst_total += 1
            if not accepted:
                self._burst_blocking += 1

        # ── Termination ──────────────────────────────────────
        terminated = self.current_time >= self.episode_duration
        truncated = False

        # ══════════════════════════════════════════════════════
        # OPTIMIZATION 3: Minimal info dict during episode
        # Only compute full stats at episode end
        # ══════════════════════════════════════════════════════
        if terminated:
            sp = max(self._services_processed, 1)
            br = max(self._bitrate_requested, 1e-9)
            sbr = 1.0 - self._services_accepted / sp
            bbr = 1.0 - self._bitrate_carried / br
            bt = self._burst_total
            burst_bp = (self._burst_blocking / max(bt, 1)) if bt > 0 else 0.0

            info = {
                'accepted': accepted,
                'reward': reward,
                'step': self.step_count,
                'bitrate': bitrate,
                'current_time': self.current_time,
                'step_blocking': 0.0 if accepted else 1.0,
                'burst_active': tinfo.get('burst_active', 0.0),
                'episode': {
                    'r': self._total_reward,
                    'l': self._services_processed,
                    't': self.current_time,
                },
                'episode_service_blocking_rate': sbr,
                'episode_bit_rate_blocking_rate': bbr,
                'episode_services_processed': self._services_processed,
                'episode_services_accepted': self._services_accepted,
                'episode_total_reward': self._total_reward,
                'episode_burst_blocking_rate': burst_bp,
                'episode_burst_events': bt,
                'episode_rewards': np.array(self.step_rewards),
            }
        else:
            # OPTIMIZATION: Minimal info during episode
            info = {
                'accepted': accepted,
                'reward': reward,
                'step': self.step_count,
                'bitrate': bitrate,
                'current_time': self.current_time,
                'step_blocking': 0.0 if accepted else 1.0,
                'window_blocking_rate': self._get_window_blocking_rate(),  # ← Now tracked!
                'burst_active': tinfo.get('burst_active', 0.0),
            }

        # ── Next service ─────────────────────────────────────
        if not terminated:
            self._generate_next_service()
            obs = self._get_observation()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, terminated, truncated, info

    # ══════════════════════════════════════════════════════════
    # OBSERVATION
    # ══════════════════════════════════════════════════════════
    def _get_observation(self) -> np.ndarray:
        base_obs, mask = self.encoder.encode(
            service=self.current_service,
            paths=self.current_paths,
            occ_by_band_by_link=self.spectrum_occ,
            path_to_link_ids_fn=self._path_to_link_ids,
        )
        self.current_mask = mask

        # Temporal features
        tinfo = self.traffic_gen.get_traffic_info()
        g_util = self._global_spectrum_utilization()
        util_delta = g_util - self._prev_global_util
        self._prev_global_util = g_util

        # Reuse pre-allocated array
        self._temporal_features[0] = tinfo['time_of_day']
        self._temporal_features[1] = tinfo['time_of_day_sin']
        self._temporal_features[2] = tinfo['time_of_day_cos']
        self._temporal_features[3] = tinfo['current_rate_norm']
        self._temporal_features[4] = tinfo['burst_active']
        self._temporal_features[5] = tinfo['burst_remaining_norm']
        self._temporal_features[6] = tinfo['recent_rate_norm']
        self._temporal_features[7] = tinfo['recent_blocking_rate']
        self._temporal_features[8] = g_util
        self._temporal_features[9] = util_delta

        return np.concatenate([base_obs, self._temporal_features])

    def action_masks(self) -> np.ndarray:
        if self.current_mask is None:
            self._get_observation()
        return self.current_mask.astype(bool)

    # ══════════════════════════════════════════════════════════
    # REWARD (simple like old version)
    # ══════════════════════════════════════════════════════════
    def _compute_reward(self, accepted: bool, bitrate: float) -> float:
        br_norm = bitrate / self.bitrate_normalization
        if accepted:
            return br_norm
        else:
            return -br_norm * self.blocking_penalty_scale

    # ══════════════════════════════════════════════════════════
    # SPECTRUM ALLOCATION
    # ══════════════════════════════════════════════════════════
    def _attempt_allocation(self, path: Path, band: int,
                            link_ids: np.ndarray) -> bool:
        worst_mod = self.qot.get_best_modulation(path.path_id)
        if worst_mod is None:
            return False

        req_slots = slots_needed(
            self.current_service.bit_rate,
            worst_mod.spectral_efficiency,
            self.slot_bandwidth_ghz, self.guard_slots)

        occ_arrays = [self.spectrum_occ[band][lid] for lid in link_ids]
        start_slot = first_fit_allocation(occ_arrays, req_slots)
        if start_slot is None:
            return False

        # GSNR-aware refinement
        if hasattr(self.qot, 'get_slot_gsnr'):
            actual_gsnr = self.qot.get_slot_gsnr(
                path.path_id, start_slot, num_slots=req_slots)
            actual_mod = self._get_modulation_for_gsnr(actual_gsnr.min())
            if actual_mod is None:
                return False
            actual_slots = slots_needed(
                self.current_service.bit_rate,
                actual_mod.spectral_efficiency,
                self.slot_bandwidth_ghz, self.guard_slots)
            best_mod = actual_mod
            req_slots = actual_slots
        else:
            best_mod = worst_mod

        if start_slot + req_slots > self.slots_per_band:
            return False

        # Allocate
        end_slot = start_slot + req_slots
        for lid in link_ids:
            self.spectrum_occ[band][lid, start_slot:end_slot] = 1

        self.current_service.path = path
        self.current_service.band_id = band
        self.current_service.channels = np.arange(start_slot, end_slot, dtype=np.int32)
        self.current_service.modulation = best_mod

        release_time = self.current_time + self.current_service.holding_time
        heapq.heappush(self.release_events, (release_time, self.current_service))
        return True

    def _get_modulation_for_gsnr(self, gsnr_db: float):
        feasible = [m for m in self.qot.modulations if gsnr_db >= m.minimum_osnr]
        if not feasible:
            return None
        return max(feasible, key=lambda m: m.spectral_efficiency)

    # ══════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════
    def _generate_next_service(self):
        svc = self.traffic_gen.generate_service()
        self.current_time = svc.arrival_time
        key = (svc.source, svc.destination)
        self.current_paths = list(self.ksp_dict.get(key, []))[:self.K]
        self.current_service = svc

    def _process_releases(self):
        while self.release_events and self.release_events[0][0] <= self.current_time:
            _, svc = heapq.heappop(self.release_events)
            self._release_spectrum(svc)

    def _release_spectrum(self, svc: Service):
        if not svc.accepted or svc.path is None:
            return
        band = svc.band_id
        link_ids = self._path_link_cache.get(svc.path.path_id)
        if link_ids is None:
            return
        for lid in link_ids:
            self.spectrum_occ[band][lid, svc.channels] = 0

    def _path_to_link_ids(self, path: Path) -> np.ndarray:
        """Use cached link IDs."""
        if path.path_id in self._path_link_cache:
            return self._path_link_cache[path.path_id]
        if path.links_id is not None:
            return path.links_id
        ids = []
        for i in range(len(path.node_list) - 1):
            u, v = path.node_list[i], path.node_list[i + 1]
            ids.append(self.topology[u][v]['index'])
        return np.array(ids, dtype=np.int32)

    def _global_spectrum_utilization(self) -> float:
        """Compute global utilization."""
        total_occ = sum(np.sum(occ) for occ in self.spectrum_occ.values())
        total_cap = self.num_links * self.slots_per_band * self.num_bands
        return float(total_occ / max(total_cap, 1))
    
    def _get_window_blocking_rate(self) -> float:
        """Get blocking rate from circular buffer — O(1)."""
        if self._window_count == 0:
            return 0.0
        if self._window_count < self._window_size:
            accepted = np.sum(self._window_buffer[:self._window_count])
        else:
            accepted = np.sum(self._window_buffer)
        return 1.0 - (accepted / self._window_count)

    def render(self):
        pass

    def close(self):
        pass
