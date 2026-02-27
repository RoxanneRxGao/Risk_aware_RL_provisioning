"""
Link Criticality Computation - Updated for Directed Graphs

Key Changes:
- compute_link_criticality_betweenness() now works with directed graphs
- Converts to undirected for betweenness computation
- Returns criticality for DIRECTED edges (86 values, not 43)
- Each direction gets the same criticality as its physical link
"""
import numpy as np
import networkx as nx
from typing import Optional, Dict


def compute_link_criticality_betweenness(G: nx.DiGraph,
                                        weight: str = 'length',
                                        normalize: bool = True) -> np.ndarray:
    """
    Compute link criticality using edge betweenness centrality.
    
    For DIRECTED graphs:
    1. Convert to undirected for betweenness computation
    2. Compute betweenness on physical links
    3. Map back to directed edges (both directions get same value)
    
    Args:
        G: NetworkX DiGraph (directed graph)
        weight: Edge attribute to use as weight
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Array of criticality values for DIRECTED edges
        Shape: [num_directed_edges] (e.g., 86 for 43 physical links)
        
    Example:
        >>> G = load_topology('topology.xlsx')  # DiGraph with 86 edges
        >>> criticality = compute_link_criticality_betweenness(G)
        >>> # Returns array of length 86
        >>> # Edge 0 (1→2) and Edge 1 (2→1) get same criticality
    """
    num_directed_edges = G.number_of_edges()
    
    # Convert to undirected for betweenness computation
    G_undirected = G.to_undirected()
    
    # Compute edge betweenness on undirected graph
    edge_betweenness = nx.edge_betweenness_centrality(G_undirected, weight=weight)
    
    # Map betweenness to directed edges
    criticality = np.zeros(num_directed_edges, dtype=np.float32)
    
    for u, v, data in G.edges(data=True):
        edge_idx = data['index']
        
        # Get betweenness from undirected graph
        # Try both (u,v) and (v,u) since undirected
        if (u, v) in edge_betweenness:
            bc = edge_betweenness[(u, v)]
        elif (v, u) in edge_betweenness:
            bc = edge_betweenness[(v, u)]
        else:
            bc = 0.0
        
        criticality[edge_idx] = bc
    
    # Normalize to [0, 1]
    if normalize and criticality.max() > 0:
        criticality = criticality / criticality.max()
    
    return criticality


def compute_link_criticality_traffic(G: nx.DiGraph,
                                     traffic_matrix: np.ndarray,
                                     ksp_dict: Dict,
                                     k_paths: int = 5) -> np.ndarray:
    """
    Compute link criticality based on traffic distribution (directed).
    
    Args:
        G: NetworkX DiGraph
        traffic_matrix: [num_nodes, num_nodes] traffic demand matrix
        ksp_dict: Dictionary of K-shortest paths
        k_paths: Number of paths to consider
        
    Returns:
        Array of traffic-weighted criticality for DIRECTED edges
    """
    num_directed_edges = G.number_of_edges()
    
    # Initialize link load for directed edges
    link_load = np.zeros(num_directed_edges, dtype=np.float32)
    
    # Accumulate traffic over all paths
    for (src, dst), paths in ksp_dict.items():
        src_idx = G.nodes[src]['index']
        dst_idx = G.nodes[dst]['index']
        
        # Get traffic demand
        demand = traffic_matrix[src_idx, dst_idx]
        
        if demand > 0 and len(paths) > 0:
            # Distribute traffic equally over k paths
            per_path_traffic = demand / min(k_paths, len(paths))
            
            for path in paths[:k_paths]:
                # Add traffic to each DIRECTED link on the path
                for link_id in path.links_id:
                    link_load[link_id] += per_path_traffic
    
    # Normalize to [0, 1]
    if link_load.max() > 0:
        criticality = link_load / link_load.max()
    else:
        criticality = link_load
    
    return criticality.astype(np.float32)


def compute_link_criticality_combined(G: nx.DiGraph,
                                      traffic_matrix: Optional[np.ndarray] = None,
                                      ksp_dict: Optional[Dict] = None,
                                      alpha: float = 0.5) -> np.ndarray:
    """
    Compute combined criticality using topology and traffic (directed).
    
    Args:
        G: NetworkX DiGraph
        traffic_matrix: Optional traffic demand matrix
        ksp_dict: Optional K-shortest paths dictionary
        alpha: Weight for topology component [0, 1]
        
    Returns:
        Combined criticality for DIRECTED edges
    """
    # Topology-based criticality
    topo_crit = compute_link_criticality_betweenness(G)
    
    # Traffic-based criticality (if available)
    if traffic_matrix is not None and ksp_dict is not None:
        traffic_crit = compute_link_criticality_traffic(G, traffic_matrix, ksp_dict)
        criticality = alpha * topo_crit + (1 - alpha) * traffic_crit
    else:
        criticality = topo_crit
    
    return criticality.astype(np.float32)


def compute_path_criticality(path_link_ids: np.ndarray,
                             edge_criticality: np.ndarray,
                             aggregation: str = 'mean') -> float:
    """
    Compute criticality score for a path.
    
    Works with DIRECTED edge IDs.
    
    Args:
        path_link_ids: Array of DIRECTED edge indices
        edge_criticality: Array of per-edge criticality (directed)
        aggregation: How to aggregate ('mean', 'max', 'sum')
        
    Returns:
        Path criticality score
    """
    if len(path_link_ids) == 0:
        return 0.0
    
    # Get criticality values for directed edges on path
    path_crit_values = edge_criticality[path_link_ids]
    
    # Aggregate
    if aggregation == 'mean':
        return float(np.mean(path_crit_values))
    elif aggregation == 'max':
        return float(np.max(path_crit_values))
    elif aggregation == 'sum':
        return float(np.sum(path_crit_values))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def identify_high_risk_links(edge_criticality: np.ndarray,
                             threshold_quantile: float = 0.90) -> np.ndarray:
    """
    Identify high-risk (high criticality) DIRECTED edges.
    
    Args:
        edge_criticality: Array of criticality values (directed edges)
        threshold_quantile: Quantile threshold
        
    Returns:
        Boolean array for directed edges
    """
    if len(edge_criticality) == 0:
        return np.array([], dtype=bool)
    
    threshold = np.quantile(edge_criticality, threshold_quantile)
    return edge_criticality >= threshold


def compute_path_risk_exposure(path_link_ids: np.ndarray,
                               edge_criticality: np.ndarray,
                               high_risk_threshold: float = 0.90) -> Dict:
    """
    Compute risk exposure metrics for a path (directed).
    
    Args:
        path_link_ids: Array of DIRECTED edge indices
        edge_criticality: Array of criticality values (directed)
        high_risk_threshold: Quantile threshold for high-risk
        
    Returns:
        Dictionary with risk metrics
    """
    if len(path_link_ids) == 0:
        return {
            'mean_criticality': 0.0,
            'max_criticality': 0.0,
            'high_risk_fraction': 0.0,
            'num_high_risk': 0
        }
    
    # Get criticality for path's directed edges
    path_crit = edge_criticality[path_link_ids]
    
    # Identify high-risk edges
    high_risk_mask = identify_high_risk_links(edge_criticality, high_risk_threshold)
    path_high_risk = high_risk_mask[path_link_ids]
    
    return {
        'mean_criticality': float(np.mean(path_crit)),
        'max_criticality': float(np.max(path_crit)),
        'high_risk_fraction': float(np.mean(path_high_risk)),
        'num_high_risk': int(np.sum(path_high_risk))
    }


# Test function
def test_directed_criticality():
    """Test criticality computation with directed graph."""
    from utils.topology import load_topology, compute_ksp_with_gsnr_fallback, add_link_ids_to_paths
    
    print("="*70)
    print("TESTING DIRECTED CRITICALITY")
    print("="*70)
    
    # Load directed topology
    print("\n[1/3] Loading directed topology...")
    G = load_topology("topologies/topo_us24_txOnly.xlsx")
    print(f"  Directed edges: {G.number_of_edges()}")
    print(f"  Physical links: {G.number_of_edges() // 2}")
    
    # Compute paths
    print("\n[2/3] Computing paths...")
    ksp = compute_ksp_with_gsnr_fallback(
        G, k=5, 
        gsnr_data_path="gsnr_data/btuk_roadm_all_pairs_ksp_gsnr.pkl"
    )
    add_link_ids_to_paths(G, ksp)
    
    # Compute criticality
    print("\n[3/3] Computing criticality...")
    criticality = compute_link_criticality_betweenness(G)
    
    print(f"  Criticality shape: {criticality.shape}")
    print(f"  Criticality range: [{criticality.min():.3f}, {criticality.max():.3f}]")
    
    # Check that opposite directions have same criticality
    print(f"\nChecking bidirectional consistency:")
    for i in range(0, min(10, len(criticality)), 2):
        if i+1 < len(criticality):
            print(f"  Edge {i} criticality: {criticality[i]:.3f}")
            print(f"  Edge {i+1} criticality: {criticality[i+1]:.3f}")
            print(f"  Match: {np.isclose(criticality[i], criticality[i+1])}")
    
    # Test path criticality
    print(f"\nExample path risk:")
    example_path = list(ksp.values())[0][0]
    risk = compute_path_risk_exposure(example_path.links_id, criticality)
    print(f"  Path: {' → '.join(example_path.node_list)}")
    print(f"  Directed links: {example_path.links_id}")
    print(f"  Mean criticality: {risk['mean_criticality']:.3f}")
    print(f"  Max criticality: {risk['max_criticality']:.3f}")
    
    print("\n" + "="*70)
    print("✓ Directed criticality test complete!")
    print("="*70)


if __name__ == "__main__":
    test_directed_criticality()
