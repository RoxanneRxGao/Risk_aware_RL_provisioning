"""
Spectrum Management Utilities — OPTIMIZED VERSION

Handles spectrum allocation, fragmentation computation,
and spectrum-related operations for optical networks.

OPTIMIZATIONS:
- Vectorized NumPy operations instead of Python loops
- Pre-allocated arrays where possible
- Numba JIT compilation for hot paths (optional)
"""
import numpy as np
from typing import List, Optional, Tuple

# Try to import numba for JIT compilation (optional but ~5-10x speedup)
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: define njit as a no-op decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args or callable(args[0]) else decorator


@njit(cache=True)
def _first_fit_numba(common_free: np.ndarray, required_slots: int) -> int:
    """Numba-accelerated first-fit search. Returns -1 if not found."""
    n = len(common_free)
    max_start = n - required_slots
    if max_start < 0:
        return -1
    
    run_length = 0
    start_idx = 0
    
    for i in range(n):
        if common_free[i]:
            if run_length == 0:
                start_idx = i
            run_length += 1
            if run_length >= required_slots and start_idx <= max_start:
                return start_idx
        else:
            run_length = 0
    
    return -1


def first_fit_allocation(occ_arrays: List[np.ndarray], 
                        required_slots: int) -> Optional[int]:
    """
    First-Fit spectrum allocation algorithm — OPTIMIZED.
    
    Finds the first contiguous block of free slots across all links.
    
    Args:
        occ_arrays: List of occupancy arrays, one per link
                   (1 = occupied, 0 = free)
        required_slots: Number of contiguous slots needed
        
    Returns:
        Starting slot index if allocation possible, None otherwise
    """
    if not occ_arrays or required_slots <= 0:
        return None
    
    num_slots = len(occ_arrays[0])
    if required_slots > num_slots:
        return None
    
    # OPTIMIZATION: Stack arrays and use vectorized AND
    # This is faster than iterative &= for multiple arrays
    stacked = np.vstack(occ_arrays)
    
    # All links must be free (0) → product of (1-occ) across links
    # Equivalent to: all links have occ=0
    common_free = np.all(stacked == 0, axis=0).astype(np.int8)
    
    # Use numba-accelerated search if available
    if HAS_NUMBA:
        result = _first_fit_numba(common_free, required_slots)
        return result if result >= 0 else None
    
    # Fallback: vectorized convolution approach
    # Convolve with kernel of ones to find runs of length >= required_slots
    if required_slots == 1:
        # Simple case: find first free slot
        free_indices = np.where(common_free)[0]
        return int(free_indices[0]) if len(free_indices) > 0 else None
    
    # Use cumsum trick for finding runs
    # This is O(n) and fully vectorized
    padded = np.concatenate([[0], common_free, [0]])
    diff = np.diff(padded)
    
    # Start of free blocks: diff == 1
    # End of free blocks: diff == -1
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # Length of each block
    lengths = ends - starts
    
    # Find first block with sufficient length
    valid_mask = lengths >= required_slots
    if not np.any(valid_mask):
        return None
    
    # Get first valid block's start
    first_valid_idx = np.argmax(valid_mask)
    start_slot = starts[first_valid_idx]
    
    # Verify bounds
    if start_slot + required_slots > num_slots:
        return None
    
    return int(start_slot)


def best_fit_allocation(occ_arrays: List[np.ndarray],
                       required_slots: int) -> Optional[int]:
    """
    Best-Fit spectrum allocation algorithm — OPTIMIZED.
    
    Finds the smallest contiguous block that can accommodate
    the required slots.
    """
    if not occ_arrays or required_slots <= 0:
        return None
    
    num_slots = len(occ_arrays[0])
    if required_slots > num_slots:
        return None
    
    # Stack and find common free
    stacked = np.vstack(occ_arrays)
    common_free = np.all(stacked == 0, axis=0).astype(np.int8)
    
    # Find all free blocks using diff
    padded = np.concatenate([[0], common_free, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts
    
    # Filter blocks that are large enough
    valid_mask = lengths >= required_slots
    if not np.any(valid_mask):
        return None
    
    valid_starts = starts[valid_mask]
    valid_lengths = lengths[valid_mask]
    
    # Find smallest sufficient block
    best_idx = np.argmin(valid_lengths)
    return int(valid_starts[best_idx])


@njit(cache=True)
def _compute_fragmentation_numba(occ_array: np.ndarray) -> Tuple[int, int]:
    """Numba-accelerated fragmentation computation."""
    n = len(occ_array)
    if n == 0:
        return 0, 0
    
    free_blocks = 0
    total_free = 0
    in_free_block = False
    
    for i in range(n):
        if occ_array[i] == 0:
            total_free += 1
            if not in_free_block:
                free_blocks += 1
                in_free_block = True
        else:
            in_free_block = False
    
    return free_blocks, total_free


def compute_spectrum_fragmentation(occ_array: np.ndarray) -> float:
    """
    Compute spectrum fragmentation metric — OPTIMIZED.
    
    Fragmentation = (number of free blocks) / (total free slots)
    """
    if len(occ_array) == 0:
        return 0.0
    
    if HAS_NUMBA:
        free_blocks, total_free = _compute_fragmentation_numba(occ_array)
    else:
        # Vectorized approach using diff
        free_mask = (occ_array == 0).astype(np.int8)
        total_free = int(np.sum(free_mask))
        
        if total_free == 0:
            return 1.0
        
        # Count transitions from occupied to free
        padded = np.concatenate([[0], free_mask, [0]])
        diff = np.diff(padded)
        free_blocks = int(np.sum(diff == 1))
    
    if total_free == 0:
        return 1.0
    
    return min(free_blocks / max(1, total_free), 1.0)


def compute_largest_free_block(occ_arrays: List[np.ndarray]) -> int:
    """
    Find the size of largest contiguous free block — OPTIMIZED.
    """
    if not occ_arrays:
        return 0
    
    # Stack and find common free
    stacked = np.vstack(occ_arrays)
    common_free = np.all(stacked == 0, axis=0).astype(np.int8)
    
    # Find blocks using diff
    padded = np.concatenate([[0], common_free, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0:
        return 0
    
    lengths = ends - starts
    return int(np.max(lengths))


def compute_spectrum_utilization(occ_array: np.ndarray) -> float:
    """Compute spectrum utilization (fraction of occupied slots)."""
    if len(occ_array) == 0:
        return 0.0
    return float(np.mean(occ_array))


def compute_free_blocks_distribution(occ_array: np.ndarray) -> np.ndarray:
    """
    Get array of all free block sizes — OPTIMIZED.
    
    Returns numpy array instead of list for better performance.
    """
    if len(occ_array) == 0:
        return np.array([], dtype=np.int32)
    
    free_mask = (occ_array == 0).astype(np.int8)
    padded = np.concatenate([[0], free_mask, [0]])
    diff = np.diff(padded)
    
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0:
        return np.array([], dtype=np.int32)
    
    return (ends - starts).astype(np.int32)


def compute_spectrum_entropy(occ_array: np.ndarray, 
                             normalize: bool = True) -> float:
    """
    Compute entropy of spectrum occupancy pattern — OPTIMIZED.
    """
    n = len(occ_array)
    if n == 0:
        return 0.0
    
    # Count using sum (faster than explicit counting)
    n_occ = int(np.sum(occ_array))
    n_free = n - n_occ
    
    if n_occ == 0 or n_free == 0:
        return 0.0
    
    # Compute probabilities
    p_occ = n_occ / n
    p_free = n_free / n
    
    # Shannon entropy with numerical stability
    entropy = -(p_occ * np.log2(p_occ) + p_free * np.log2(p_free))
    
    return float(entropy) if not normalize else float(entropy)


def allocate_spectrum(occ_arrays: List[np.ndarray],
                     start_slot: int,
                     num_slots: int) -> None:
    """Allocate (mark as occupied) spectrum slots."""
    end_slot = start_slot + num_slots
    for occ in occ_arrays:
        occ[start_slot:end_slot] = 1


def release_spectrum(occ_arrays: List[np.ndarray],
                    start_slot: int,
                    num_slots: int) -> None:
    """Release (mark as free) spectrum slots."""
    end_slot = start_slot + num_slots
    for occ in occ_arrays:
        occ[start_slot:end_slot] = 0


# =============================================================================
# BATCH OPERATIONS for vectorized processing
# =============================================================================

def batch_compute_link_features(occ_by_band: dict, 
                                num_links: int,
                                req_widths: dict) -> np.ndarray:
    """
    Compute features for ALL links and bands in one vectorized pass.
    
    Returns array of shape [num_bands, num_links, 5] with features:
    [util, frag, largest_norm, entropy, available]
    
    This replaces per-link iteration in the encoder.
    """
    num_bands = len(occ_by_band)
    bands = sorted(occ_by_band.keys())
    
    # Pre-allocate output
    features = np.zeros((num_bands, num_links, 5), dtype=np.float32)
    
    for b_idx, band in enumerate(bands):
        occ_band = occ_by_band[band]  # Shape: [num_links, num_slots]
        req_width = req_widths.get(band, 1)
        
        # 1. Utilization: mean occupancy per link
        features[b_idx, :, 0] = np.mean(occ_band, axis=1)
        
        # Process each link for remaining features
        for lid in range(num_links):
            occ = occ_band[lid]
            
            # 2. Fragmentation
            features[b_idx, lid, 1] = compute_spectrum_fragmentation(occ)
            
            # 3. Largest free block (normalized)
            largest = compute_largest_free_block([occ])
            features[b_idx, lid, 2] = largest / max(1, len(occ))
            
            # 4. Entropy
            features[b_idx, lid, 3] = compute_spectrum_entropy(occ)
            
            # 5. Availability
            features[b_idx, lid, 4] = 1.0 if largest >= req_width else 0.0
    
    return features
