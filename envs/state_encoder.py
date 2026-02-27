"""
State Encoder for Risk-Aware Optical Network Provisioning — OPTIMIZED VERSION

OPTIMIZATIONS:
  1. Pre-allocated arrays for repeated computations
  2. Cached QoT lookups per path (these don't change during episode)
  3. Vectorized link feature extraction
  4. Reduced object creation in hot paths
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass

from utils.spectrum import (
    compute_spectrum_fragmentation,
    compute_largest_free_block,
    compute_spectrum_utilization,
    compute_spectrum_entropy
)
from utils.criticality import compute_path_risk_exposure
from utils.qot import Modulation


@dataclass
class EncoderConfig:
    """Configuration for the state encoder."""
    num_nodes: int
    bands: List[int]
    K: int
    H_max: int
    num_mods: int
    delta_norm_db: float = 10.0
    weight_by_class: Optional[Dict[int, float]] = None
    free_blocks_norm_cap: int = 50
    highrisk_q: float = 0.10


class StateEncoder:
    """
    Comprehensive state encoder — OPTIMIZED VERSION.
    """
    
    def __init__(self,
                 cfg: EncoderConfig,
                 num_links: int,
                 edge_criticality: np.ndarray,
                 qot_provider,
                 slots_needed_fn: Callable[[float, Modulation, int], int]):
        self.cfg = cfg
        self.num_links = num_links
        self.edge_criticality = edge_criticality.astype(np.float32)
        self.qot = qot_provider
        self.slots_needed_fn = slots_needed_fn
        self.num_bands = len(cfg.bands)
        
        # Compute high-risk threshold
        q = float(np.clip(cfg.highrisk_q, 0.0, 1.0))
        if len(self.edge_criticality) > 0 and q > 0.0:
            self._highrisk_threshold = float(np.quantile(self.edge_criticality, 1.0 - q))
        else:
            self._highrisk_threshold = 1.0
        
        # ═══════════════════════════════════════════════════════
        # OPTIMIZATION: Pre-allocate arrays
        # ═══════════════════════════════════════════════════════
        self._obs_dim = self.obs_dim()
        
        # Pre-allocate observation buffer
        self._obs_buffer = np.zeros(self._obs_dim, dtype=np.float32)
        
        # Pre-allocate component arrays
        self._src_onehot = np.zeros(cfg.num_nodes, dtype=np.float32)
        self._dst_onehot = np.zeros(cfg.num_nodes, dtype=np.float32)
        
        # Pre-allocate link features array [num_bands * num_links * 5]
        self._link_features = np.zeros(self.num_bands * num_links * 5, dtype=np.float32)
        
        # Pre-allocate candidate features [K * num_bands * 8]
        self._cand_features = np.zeros(cfg.K * self.num_bands * 8, dtype=np.float32)
        
        # Pre-allocate criticality features [K * 3]
        self._crit_features = np.zeros(cfg.K * 3, dtype=np.float32)
        
        # Pre-allocate action mask
        self._mask = np.zeros(cfg.K * self.num_bands, dtype=np.float32)
        
        # ═══════════════════════════════════════════════════════
        # OPTIMIZATION: Cache for QoT lookups (path_id, band) -> (feasible_mods, best_mod)
        # ═══════════════════════════════════════════════════════
        self._qot_cache: Dict[Tuple[int, int, float], Tuple[List, Optional[Modulation], float]] = {}
        
        # Normalization constants
        self._dist_norm_scale = 5000.0
    
    def obs_dim(self) -> int:
        """Calculate total observation dimension."""
        req_dim = self.cfg.num_nodes + self.cfg.num_nodes + 2
        link_feat_dim = 5
        per_link_dim = self.num_bands * self.num_links * link_feat_dim
        cand_feat_dim = 8
        cand_dim = self.cfg.K * self.num_bands * cand_feat_dim
        crit_dim = self.cfg.K * 3
        return req_dim + per_link_dim + cand_dim + crit_dim
    
    def clear_cache(self):
        """Clear QoT cache (call between episodes if needed)."""
        self._qot_cache.clear()
    
    def _get_qot_info(self, path_id: int, band_idx: int, bitrate: float) -> Tuple[List, Optional[Modulation], float]:
        """Get QoT info with caching."""
        cache_key = (path_id, band_idx, bitrate)
        
        if cache_key in self._qot_cache:
            return self._qot_cache[cache_key]
        
        feasible_mods = self.qot.feasible_mods(path_id, band_idx, bitrate)
        
        if len(feasible_mods) == 0:
            result = (feasible_mods, None, 0.0)
        else:
            best_mod = self.qot.best_mod(path_id, band_idx, bitrate)
            delta_db = self.qot.margin_db(path_id, band_idx, best_mod)
            result = (feasible_mods, best_mod, delta_db)
        
        self._qot_cache[cache_key] = result
        return result
    
    def encode(self,
               service,
               paths: List,
               occ_by_band_by_link: Dict[int, List[np.ndarray]],
               path_to_link_ids_fn: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode current state into observation vector and action mask — OPTIMIZED.
        """
        # ═════════════════════════════════════════════════════
        # PART 1: REQUEST FEATURES (reuse pre-allocated arrays)
        # ═════════════════════════════════════════════════════
        src_id = int(service.source_id)
        dst_id = int(service.destination_id)
        
        # Reset and set one-hot (faster than creating new arrays)
        self._src_onehot.fill(0)
        self._dst_onehot.fill(0)
        
        if 0 <= src_id < self.cfg.num_nodes:
            self._src_onehot[src_id] = 1.0
        if 0 <= dst_id < self.cfg.num_nodes:
            self._dst_onehot[dst_id] = 1.0
        
        bitrate = float(service.bit_rate)
        weight = self._get_service_weight(service)
        
        # ═════════════════════════════════════════════════════
        # PART 2: PER-LINK SPECTRUM FEATURES
        # ═════════════════════════════════════════════════════
        width_req_by_band = self._compute_required_widths(paths, bitrate)
        
        # Reset link features
        self._link_features.fill(0)
        
        feat_idx = 0
        for band_idx in self.cfg.bands:
            occ_band = occ_by_band_by_link[band_idx]
            req_width = width_req_by_band[band_idx]
            
            for link_id in range(self.num_links):
                occ = occ_band[link_id]
                
                # Inline feature extraction (avoid function call overhead)
                # 1. Utilization
                util = float(np.mean(occ))
                
                # 2. Fragmentation (simplified)
                total_free = int(np.sum(occ == 0))
                if total_free == 0:
                    frag = 1.0
                else:
                    # Count free blocks using diff
                    free_mask = (occ == 0).astype(np.int8)
                    transitions = np.diff(np.concatenate([[0], free_mask, [0]]))
                    free_blocks = int(np.sum(transitions == 1))
                    frag = min(free_blocks / max(1, total_free), 1.0)
                
                # 3. Largest free block
                if total_free == 0:
                    largest = 0
                else:
                    padded = np.concatenate([[0], free_mask, [0]])
                    diff = np.diff(padded)
                    starts = np.where(diff == 1)[0]
                    ends = np.where(diff == -1)[0]
                    if len(starts) > 0:
                        largest = int(np.max(ends - starts))
                    else:
                        largest = 0
                largest_norm = largest / max(1, len(occ))
                
                # 4. Entropy
                if total_free == 0 or total_free == len(occ):
                    entropy = 0.0
                else:
                    p_free = total_free / len(occ)
                    p_occ = 1.0 - p_free
                    entropy = -(p_occ * np.log2(p_occ) + p_free * np.log2(p_free))
                
                # 5. Availability
                available = 1.0 if largest >= req_width else 0.0
                
                self._link_features[feat_idx:feat_idx+5] = [util, frag, largest_norm, entropy, available]
                feat_idx += 5
        
        # ═════════════════════════════════════════════════════
        # PART 3: CANDIDATE PATH FEATURES + ACTION MASK
        # ═════════════════════════════════════════════════════
        self._mask.fill(0)
        self._cand_features.fill(0)
        self._crit_features.fill(0)
        
        # Pre-compute path link IDs
        paths_link_ids = []
        for k in range(min(self.cfg.K, len(paths))):
            path = paths[k]
            link_ids = path_to_link_ids_fn(path)
            paths_link_ids.append(list(link_ids) if isinstance(link_ids, np.ndarray) else link_ids)
        
        # Pad with empty lists
        while len(paths_link_ids) < self.cfg.K:
            paths_link_ids.append([])
        
        # Compute overlap statistics
        overlap_stats = self._compute_path_overlap(paths_link_ids)
        
        cand_idx = 0
        crit_idx = 0
        
        for k in range(self.cfg.K):
            if k < len(paths):
                path = paths[k]
                link_ids_k = paths_link_ids[k]
                node_list_k = path.node_list
            else:
                link_ids_k = []
                node_list_k = []
            
            # Criticality features
            if len(link_ids_k) != 0:
                link_ids_arr = np.array(link_ids_k, dtype=np.int32)
                risk_metrics = compute_path_risk_exposure(
                    link_ids_arr,
                    self.edge_criticality,
                    self._highrisk_threshold
                )
                mean_crit = risk_metrics['mean_criticality']
                max_crit = risk_metrics['max_criticality']
                highrisk_frac = risk_metrics['high_risk_fraction']
            else:
                mean_crit = max_crit = highrisk_frac = 0.0
            
            self._crit_features[crit_idx:crit_idx+3] = [mean_crit, max_crit, highrisk_frac]
            crit_idx += 3
            
            # Per-band features
            for band_enum_idx, band_idx in enumerate(self.cfg.bands):
                action_idx = k * self.num_bands + band_enum_idx
                
                if k >= len(paths) or len(link_ids_k) == 0:
                    cand_idx += 8
                    continue
                
                # Topology features
                hops = float(max(0, len(node_list_k) - 1))
                hops_norm = float(np.clip(hops / max(1.0, self.cfg.num_nodes - 1), 0.0, 1.0))
                distance_km = float(path.length)
                dist_norm = float(np.clip(distance_km / self._dist_norm_scale, 0.0, 1.0))
                
                # Spectrum availability
                occ_list = [occ_by_band_by_link[band_idx][lid] for lid in link_ids_k]
                
                free_fracs = [float(np.mean(occ == 0)) for occ in occ_list]
                min_free_frac = float(min(free_fracs)) if free_fracs else 0.0
                
                lcb = compute_largest_free_block(occ_list)
                lcb_norm = float(lcb) / float(max(1, len(occ_list[0])))
                
                # QoT features (cached)
                path_id = int(path.path_id)
                feasible_mods, best_mod, delta_db = self._get_qot_info(path_id, band_idx, bitrate)
                
                eta_norm = float(len(feasible_mods)) / float(max(1, self.cfg.num_mods))
                delta_norm = float(np.clip(delta_db / self.cfg.delta_norm_db, 0.0, 1.0)) if best_mod else 0.0
                
                # Diversity features
                overlap_avg = float(overlap_stats['avg'][k])
                overlap_max = float(overlap_stats['max'][k])
                
                # Action feasibility
                feasible_now = (eta_norm > 0.0) and (lcb > 0)
                self._mask[action_idx] = 1.0 if feasible_now else 0.0
                
                self._cand_features[cand_idx:cand_idx+8] = [
                    hops_norm, dist_norm, min_free_frac, lcb_norm,
                    eta_norm, delta_norm, overlap_avg, overlap_max
                ]
                cand_idx += 8
        
        # ═════════════════════════════════════════════════════
        # FINAL: CONCATENATE ALL PARTS
        # ═════════════════════════════════════════════════════
        obs = np.concatenate([
            self._src_onehot,
            self._dst_onehot,
            np.array([bitrate, weight], dtype=np.float32),
            self._link_features,
            self._cand_features,
            self._crit_features
        ])
        
        return obs, self._mask.copy()
    
    def _get_service_weight(self, service) -> float:
        """Get weight for service based on class."""
        if self.cfg.weight_by_class is None:
            return 1.0
        cls = int(service.service_class) if hasattr(service, 'service_class') and service.service_class is not None else 0
        return float(self.cfg.weight_by_class.get(cls, 1.0))
    
    def _compute_required_widths(self, paths: List, bitrate: float) -> Dict[int, int]:
        """Compute minimum required slots per band."""
        width_by_band = {}
        
        for band_idx in self.cfg.bands:
            best_slots = None
            
            for k in range(min(self.cfg.K, len(paths))):
                path_id = int(paths[k].path_id)
                
                feasible_mods, best_mod, _ = self._get_qot_info(path_id, band_idx, bitrate)
                if not feasible_mods or best_mod is None:
                    continue
                
                slots = int(self.slots_needed_fn(bitrate, best_mod, band_idx))
                best_slots = slots if best_slots is None else min(best_slots, slots)
            
            width_by_band[band_idx] = int(best_slots) if best_slots is not None else int(1e9)
        
        return width_by_band
    
    def _extract_link_features(self, occ: np.ndarray, req_width: int) -> np.ndarray:
        """Extract 5 features for a single link's spectrum state."""
        util = compute_spectrum_utilization(occ)
        frag = compute_spectrum_fragmentation(occ)
        largest = compute_largest_free_block([occ])
        largest_norm = float(largest) / float(max(1, len(occ)))
        entropy = compute_spectrum_entropy(occ, normalize=True)
        available = 1.0 if largest >= req_width else 0.0
        return np.array([util, frag, largest_norm, entropy, available], dtype=np.float32)
    
    def _compute_path_overlap(self, paths_link_ids: List[List[int]]) -> Dict:
        """Compute overlap statistics between candidate paths."""
        K = len(paths_link_ids)
        overlap_avg = np.zeros(K, dtype=np.float32)
        overlap_max = np.zeros(K, dtype=np.float32)
        
        # Pre-convert to sets
        path_sets = [set(links) for links in paths_link_ids]
        
        for k in range(K):
            if len(paths_link_ids[k]) == 0:
                continue
            
            set_k = path_sets[k]
            overlaps = []
            
            for j in range(K):
                if j == k or len(paths_link_ids[j]) == 0:
                    continue
                
                set_j = path_sets[j]
                intersection = len(set_k & set_j)
                union = len(set_k | set_j)
                
                if union > 0:
                    overlaps.append(intersection / union)
            
            if overlaps:
                overlap_avg[k] = np.mean(overlaps)
                overlap_max[k] = np.max(overlaps)
        
        return {'avg': overlap_avg, 'max': overlap_max}


def _path_distance_km(path) -> float:
    """Extract distance in km from Path object."""
    return float(getattr(path, 'length', 0.0))
