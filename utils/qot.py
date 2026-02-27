"""
QoT (Quality of Transmission) and Modulation Utilities

Handles modulation format selection, QoT evaluation,
and slots calculation for optical signals.

Updated to support precomputed GSNR tables with 50GHz channel spacing.
"""
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Modulation:
    """
    Modulation format specification.
    
    Attributes:
        name: Modulation name (e.g., "QPSK", "16QAM")
        maximum_length: Maximum transmission distance in km
        spectral_efficiency: Bits per Hz per second
        minimum_osnr: Minimum OSNR requirement in dB
        inband_xt: In-band crosstalk tolerance in dB
    """
    name: str
    maximum_length: float
    spectral_efficiency: float
    minimum_osnr: Optional[float] = None
    inband_xt: Optional[float] = None


def slots_needed(bitrate_gbps: float, 
                spectral_efficiency: float,
                slot_bandwidth_ghz: float,
                guard_slots: int = 1) -> int:
    """
    Calculate number of frequency slots needed for a service.
    
    Formula: ceil(bitrate / (SE * slot_BW)) + guard_slots
    
    Args:
        bitrate_gbps: Service bitrate in Gbps
        spectral_efficiency: Modulation spectral efficiency (bits/s/Hz)
        slot_bandwidth_ghz: Bandwidth of one slot in GHz
        guard_slots: Number of guard band slots
        
    Returns:
        Total number of slots required
        
    Example:
        >>> # For 400 Gbps with 16-QAM (SE=4) and 12.5 GHz slots:
        >>> slots = slots_needed(400, 4, 12.5, guard_slots=1)
        >>> # Returns ceil(400 / (4 * 12.5)) + 1 = 8 + 1 = 9 slots
    """
    if spectral_efficiency <= 0:
        return int(1e9)  # Invalid modulation
    
    # Calculate required slots
    required = np.ceil(bitrate_gbps / (spectral_efficiency * slot_bandwidth_ghz))
    
    return int(required) + guard_slots


class SimpleQoTProvider:
    """
    Simple QoT provider based on path length.
    
    Determines feasible modulation formats based on whether
    the path length is within the modulation's maximum reach.
    
    This is a simplified model. For production, use GSNRQoTProvider
    with precomputed GSNR tables.
    """
    
    def __init__(self, 
                 modulations: List[Modulation],
                 path_lengths: Dict[int, float]):
        """
        Initialize QoT provider.
        
        Args:
            modulations: List of available modulation formats
            path_lengths: Dictionary mapping path_id -> length_km
        """
        # Sort modulations by spectral efficiency (descending)
        self.modulations = sorted(
            modulations,
            key=lambda m: m.spectral_efficiency,
            reverse=True
        )
        self.path_lengths = path_lengths
    
    def get_feasible_modulations(self, path_id: int) -> List[Modulation]:
        """
        Get list of feasible modulation formats for a path.
        
        Args:
            path_id: Path identifier
            
        Returns:
            List of modulations where path_length <= max_length
            Sorted by spectral efficiency (highest first)
        """
        length = self.path_lengths.get(path_id, float('inf'))
        
        feasible = [
            m for m in self.modulations
            if length <= m.maximum_length
        ]
        
        return feasible
    
    def get_best_modulation(self, path_id: int) -> Optional[Modulation]:
        """
        Get best (highest spectral efficiency) feasible modulation.
        
        Args:
            path_id: Path identifier
            
        Returns:
            Best modulation or None if no modulation is feasible
        """
        # TODO: Or is it better to use modulation with least margin here? Review Teng's paper again.
        feasible = self.get_feasible_modulations(path_id)
        return feasible[0] if feasible else None
    
    def get_margin_db(self, 
                      path_id: int,
                      modulation: Modulation) -> float:
        """
        Get QoT margin in dB.
        
        For this simple provider, margin is based on distance slack:
        margin ∝ (max_length - actual_length)
        
        Args:
            path_id: Path identifier
            modulation: Modulation format
            
        Returns:
            Margin in dB (higher is better)
        """
        length = self.path_lengths.get(path_id, float('inf'))
        
        # Calculate slack distance
        slack_km = modulation.maximum_length - length
        
        # Convert to rough dB margin (1000 km → 10 dB approximation)
        margin_db = max(0.0, slack_km / 100.0)
        
        return float(margin_db)
    
    def is_feasible(self,
                   path_id: int,
                   modulation: Modulation) -> bool:
        """
        Check if a specific modulation is feasible for a path.
        
        Args:
            path_id: Path identifier
            modulation: Modulation format to check
            
        Returns:
            True if feasible, False otherwise
        """
        length = self.path_lengths.get(path_id, float('inf'))
        return length <= modulation.maximum_length
    
    # Compatibility methods for use with state encoder
    def feasible_mods(self, path_id: int, band_id: int, bitrate: float) -> List[Modulation]:
        """Alias for get_feasible_modulations (for compatibility)."""
        return self.get_feasible_modulations(path_id)
    
    def best_mod(self, path_id: int, band_id: int, bitrate: float) -> Optional[Modulation]:
        """Alias for get_best_modulation (for compatibility)."""
        return self.get_best_modulation(path_id)
    
    def margin_db(self, path_id: int, band_id: int, modulation: Modulation) -> float:
        """Alias for get_margin_db (for compatibility)."""
        return self.get_margin_db(path_id, modulation)


class GSNRQoTProvider:
    """
    Advanced QoT provider using precomputed per-channel GSNR values.
    
    This provider uses precomputed GSNR tables (50 GHz channel spacing)
    and maps them to 12.5 GHz slots for spectrum allocation.
    
    GSNR Data Format:
        {(src, dst): [
            {  # Path 0
                'path': [node_list],
                'frequency_hz': [freq1, freq2, ...],  # 50 GHz spacing
                'gsnr_db': [gsnr1, gsnr2, ...]
            },
            ...  # Paths 1-4
        ]}
    
    Slot Mapping:
        - Precomputed: 50 GHz channels (e.g., 300 channels)
        - Provisioning: 12.5 GHz slots (e.g., 1200 slots)
        - Mapping: Each 50 GHz channel → 4 slots (all get same GSNR)
    """
    
    def __init__(self,
                 gsnr_data_path: str,
                 modulations: List[Modulation],
                 ksp_dict: Dict,
                 channel_spacing_ghz: float = 50.0,
                 slot_bandwidth_ghz: float = 12.5):
        """
        Initialize GSNR-based QoT provider.
        
        Args:
            gsnr_data_path: Path to pickle file with precomputed GSNR
            modulations: List of modulation formats with OSNR requirements
            ksp_dict: Dictionary of K-shortest paths {(src, dst): [Path]}
            channel_spacing_ghz: Spacing of GSNR data (default 50 GHz)
            slot_bandwidth_ghz: Provisioning slot bandwidth (default 12.5 GHz)
        """
        # Load GSNR data
        with open(gsnr_data_path, 'rb') as f:
            self.raw_gsnr_data = pickle.load(f)
        
        self.modulations = sorted(
            modulations,
            key=lambda m: m.spectral_efficiency,
            reverse=True
        )
        
        self.channel_spacing = channel_spacing_ghz
        self.slot_bandwidth = slot_bandwidth_ghz
        
        # Calculate slots per channel (50 GHz / 12.5 GHz = 4)
        self.slots_per_channel = int(channel_spacing_ghz / slot_bandwidth_ghz)
        
        # Build path ID to (src, dst, path_idx) mapping
        self.path_id_map = {}
        for (src, dst), paths in ksp_dict.items():
            for path_idx, path in enumerate(paths):
                self.path_id_map[path.path_id] = (src, dst, path_idx)
        
        # Convert GSNR data to slot-level format
        self._build_slot_gsnr_table()
    
    def _build_slot_gsnr_table(self):
        """
        Convert 50 GHz channel GSNR to 12.5 GHz slot GSNR.
        
        Each 50 GHz channel maps to 4 slots, all with same GSNR value.
        
        Creates:
            self.slot_gsnr: {path_id: np.array of GSNR per slot}
        """
        self.slot_gsnr = {}
        
        for path_id, (src, dst, path_idx) in self.path_id_map.items():
            # Get corresponding GSNR data
            # Handle node naming: "roadm 1-A" format in GSNR vs "1" in topology
            # Try multiple key formats
            gsnr_key = None
            
            # Try direct match
            if (src, dst) in self.raw_gsnr_data:
                gsnr_key = (src, dst)
            else:
                # Try with "roadm X-Y" format
                src_roadm = f"roadm {src}"
                dst_roadm = f"roadm {dst}"
                
                # Search for matching keys
                for key in self.raw_gsnr_data.keys():
                    # Extract node numbers from "roadm 1-A" format
                    key_src_num = key[0].split()[1].split('-')[0] if 'roadm' in key[0] else key[0]
                    key_dst_num = key[1].split()[1].split('-')[0] if 'roadm' in key[1] else key[1]
                    
                    if key_src_num == src and key_dst_num == dst:
                        gsnr_key = key
                        break
            
            if gsnr_key is None or gsnr_key not in self.raw_gsnr_data:
                # No GSNR data for this pair, use default (low GSNR)
                # Assume 400 channels * 4 slots = 1600 slots
                self.slot_gsnr[path_id] = np.full(1600, 10.0, dtype=np.float32)
                continue
            
            paths_data = self.raw_gsnr_data[gsnr_key]
            
            if path_idx >= len(paths_data):
                # Path index out of range, use default
                self.slot_gsnr[path_id] = np.full(1600, 10.0, dtype=np.float32)
                continue
            
            path_data = paths_data[path_idx]
            channel_gsnr = np.array(path_data['gsnr_db'], dtype=np.float32)
            
            # Expand: each channel GSNR → 4 slots
            # E.g., 300 channels → 1200 slots
            def lin2db(x):
                return 10 * np.log10(x) if x > 0 else -100.0
            # TODO: use spacing and slot width later from outside
            slot_gsnr = np.repeat(channel_gsnr + lin2db(self.slots_per_channel), self.slots_per_channel)
            
            self.slot_gsnr[path_id] = slot_gsnr
    
    def get_slot_gsnr(self, path_id: int, start_slot: int, num_slots: int) -> np.ndarray:
        """
        Get GSNR values for a specific slot allocation.
        
        Args:
            path_id: Path identifier
            start_slot: Starting slot index
            num_slots: Number of slots
            
        Returns:
            Array of GSNR values (in dB) for each slot
        """
        if path_id not in self.slot_gsnr:
            return np.full(num_slots, 10.0, dtype=np.float32)
        
        gsnr_array = self.slot_gsnr[path_id]
        end_slot = start_slot + num_slots
        
        if end_slot > len(gsnr_array):
            # Out of range, return default
            return np.full(num_slots, 10.0, dtype=np.float32)
        
        return gsnr_array[start_slot:end_slot]
    
    def get_feasible_modulations(self, path_id: int, 
                                 start_slot: int, 
                                 num_slots: int) -> List[Modulation]:
        """
        Get feasible modulations for a specific allocation.
        
        Args:
            path_id: Path identifier
            start_slot: Starting slot index
            num_slots: Number of slots to use
            
        Returns:
            List of modulations where min_GSNR >= modulation OSNR requirement
        """
        # Get GSNR for this allocation
        gsnr_values = self.get_slot_gsnr(path_id, start_slot, num_slots)
        #
        
        # Use minimum GSNR across all slots (worst case)
        min_gsnr = float(np.min(gsnr_values))
        
        # Check which modulations are feasible
        feasible = []
        for mod in self.modulations:
            if mod.minimum_osnr is not None:
                if min_gsnr >= mod.minimum_osnr:
                    feasible.append(mod)
        
        return feasible
    
    def get_best_modulation(self, path_id: int,
                           start_slot: int = 0,
                           num_slots: int = 100) -> Optional[Modulation]:
        """
        Get best modulation for a path (uses representative slots).
        
        Args:
            path_id: Path identifier
            start_slot: Representative start slot (default middle of band)
            num_slots: Representative number of slots
            
        Returns:
            Best (highest SE) feasible modulation, or None
        """
        feasible = self.get_feasible_modulations(path_id, start_slot, num_slots)
        return feasible[0] if feasible else None
    
    def get_margin_db(self, path_id: int, modulation: Modulation,
                     start_slot: int = 0, num_slots: int = 100) -> float:
        """
        Get QoT margin for a specific modulation.
        
        Margin = min_GSNR - required_OSNR
        
        Args:
            path_id: Path identifier
            modulation: Modulation format
            start_slot: Starting slot
            num_slots: Number of slots
            
        Returns:
            Margin in dB (positive = feasible, negative = not feasible)
        """
        gsnr_values = self.get_slot_gsnr(path_id, start_slot, num_slots)
        min_gsnr = float(np.min(gsnr_values))
        
        if modulation.minimum_osnr is not None:
            return min_gsnr - modulation.minimum_osnr
        
        return 0.0
    
    # Compatibility methods for state encoder
    def feasible_mods(self, path_id: int, band_id: int, bitrate: float) -> List[Modulation]:
        # TODO: This is a bit hacky, but we need to provide some representative slot allocation for the state encoder.
        """
        Get feasible modulations (compatibility method).
        
        Uses representative allocation in middle of band.
        """
        # # Use middle of band as representative
        # start_slot = 200 + band_id * 400  # Offset by band
        start_slot = band_id * 400  # Offset by band
        return self.get_feasible_modulations(path_id, start_slot, 400)
    
    def best_mod(self, path_id: int, band_id: int, bitrate: float) -> Optional[Modulation]:
        """Get best modulation (compatibility method)."""
        start_slot = 200 + band_id * 400
        return self.get_best_modulation(path_id, start_slot, 100)
    
    def margin_db(self, path_id: int, band_id: int, modulation: Modulation) -> float:
        """Get margin (compatibility method)."""
        start_slot = 200 + band_id * 400
        return self.get_margin_db(path_id, modulation, start_slot, 100)


def get_default_modulations() -> List[Modulation]:
    """
    Get default modulation formats for C/L/S band systems.
    https://github.com/xiaoliangchenUCD/DeepRMSA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/K-SP-FF%20benchmark_NSFNET.py#L268
    
    Returns:
        List of standard modulation formats with OSNR requirements
        
    Note:
        OSNR values are typical for coherent systems.
        Adjust based on your system specifications.
    """
    return [
        Modulation("BPSK", 100_000, 1, 9, -14),
        Modulation("QPSK", 2_000, 2, 15.6, -17),
        Modulation("8QAM", 1_000, 3, 18.6, -20),
        Modulation("16QAM", 500, 4, 22.4, -23),
        Modulation("32QAM", 250, 5, 26.4, -26),
        Modulation("64QAM", 125, 6, 30.4, -29),
    ]
