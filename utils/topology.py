"""
Topology Processing Utilities - Updated with Directed Graph Support

Key Changes:
- load_topology() now creates DIRECTED graph (DiGraph)
- Each physical link becomes 2 directed edges (A→B and B→A)
- Separate indexing for directed edges (86 edges from 43 physical links)
- Path extraction and criticality computation updated accordingly
"""
import numpy as np
import networkx as nx
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Path:
    """Represents a path in the network."""
    path_id: int
    node_list: List[str]
    hops: int
    length: float
    links_id: Optional[np.ndarray] = None


def load_topology(topology_path: str, k_paths: int = 5) -> nx.DiGraph:
    """
    Load network topology as DIRECTED graph.
    
    Each physical link in the Excel file becomes TWO directed edges:
    - A → B (forward direction)
    - B → A (reverse direction)
    
    This is necessary because each direction can have different spectrum occupancy.
    
    Args:
        topology_path: Path to topology Excel file
        k_paths: Number of K-shortest paths (not used here)
        
    Returns:
        NetworkX DiGraph (directed graph) with indexed directed edges
        
    Example:
        >>> G = load_topology('topology.xlsx')
        >>> # Physical link: 1-Seattle <-> 2-Portland (800 km)
        >>> # Becomes:
        >>> # Edge 0: 1 → 2 (800 km)
        >>> # Edge 1: 2 → 1 (800 km)
    """
    # Read topology file
    df = pd.read_excel(topology_path, sheet_name='Links', header=3)
    df = df.iloc[1:].reset_index(drop=True)
    df = df.dropna(subset=[df.columns[0], df.columns[1]])
    
    source_col = df.columns[0]
    dest_col = df.columns[1]
    dist_col = df.columns[2]
    
    # Create DIRECTED graph
    G = nx.DiGraph()
    
    edge_index = 0
    
    for idx, row in df.iterrows():
        source = str(row[source_col]).strip()
        destination = str(row[dest_col]).strip()
        
        try:
            distance = float(row[dist_col])
        except (ValueError, TypeError):
            continue
        
        # Extract node numbers from "X-Name" format
        source_num = source.split('-')[0] if '-' in source else source
        dest_num = destination.split('-')[0] if '-' in destination else destination
        
        # Add BOTH directions as separate directed edges
        # Forward direction: source → destination
        G.add_edge(
            source_num, dest_num,
            index=edge_index,
            length=distance,
            weight=distance,
            source_full=source,
            dest_full=destination,
            physical_link_id=idx  # Reference to original physical link
        )
        edge_index += 1
        
        # Reverse direction: destination → source
        G.add_edge(
            dest_num, source_num,
            index=edge_index,
            length=distance,
            weight=distance,
            source_full=destination,
            dest_full=source,
            physical_link_id=idx  # Same physical link
        )
        edge_index += 1
    
    # Add node indices
    for idx, node in enumerate(sorted(G.nodes())):
        G.nodes[node]['index'] = idx
    
    print(f"  Loaded topology: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} directed edges "
          f"({G.number_of_edges()//2} physical links)")
    
    return G


def get_undirected_topology(G_directed: nx.DiGraph) -> nx.Graph:
    """
    Convert directed topology to undirected for criticality computation.
    
    Args:
        G_directed: Directed topology
        
    Returns:
        Undirected graph (for betweenness centrality computation)
    """
    G_undirected = nx.Graph()
    
    # Copy nodes
    for node, attrs in G_directed.nodes(data=True):
        G_undirected.add_node(node, **attrs)
    
    # Add edges (only once per physical link)
    processed_links = set()
    
    for u, v, attrs in G_directed.edges(data=True):
        physical_link = attrs.get('physical_link_id', None)
        
        if physical_link is not None and physical_link not in processed_links:
            # Add undirected edge with same attributes
            G_undirected.add_edge(u, v, **attrs)
            processed_links.add(physical_link)
    
    return G_undirected


def extract_ksp_from_gsnr(gsnr_data_path: str, topology: nx.DiGraph) -> Dict[Tuple[str, str], List[Path]]:
    """
    Extract K-shortest paths from precomputed GSNR data.
    
    Works with directed topology - paths are inherently directional.
    
    Args:
        gsnr_data_path: Path to GSNR pickle file
        topology: NetworkX DiGraph
        
    Returns:
        Dictionary {(src, dst): [Path objects]}
    """
    print(f"Extracting K-shortest paths from GSNR data...")
    
    with open(gsnr_data_path, 'rb') as f:
        gsnr_data = pickle.load(f)
    
    ksp_dict = {}
    path_id = 0
    
    def extract_node_number(node_name: str) -> str:
        """Extract node number from 'roadm X-Y' format."""
        if 'roadm' in node_name.lower():
            parts = node_name.split()
            if len(parts) >= 2:
                return parts[1].split('-')[0]
        return node_name
    
    for (src_full, dst_full), paths_data in gsnr_data.items():
        src = extract_node_number(src_full)
        dst = extract_node_number(dst_full)
        
        path_objects = []
        
        for path_data in paths_data:
            full_path = path_data['path']
            
            # Extract ROADM nodes only
            node_list = []
            for node in full_path:
                if ('roadm' in node.lower() and 
                    not any(x in node.lower() for x in ['booster', 'preamp', 'fiber'])):
                    node_num = extract_node_number(node)
                    if node_num not in node_list:
                        node_list.append(node_num)
            
            if len(node_list) < 2:
                continue
            
            # Compute path length from directed topology
            try:
                length = sum(
                    topology[node_list[i]][node_list[i+1]]['length']
                    for i in range(len(node_list) - 1)
                )
            except KeyError:
                continue
            
            path_obj = Path(
                path_id=path_id,
                node_list=node_list,
                hops=len(node_list) - 1,
                length=length
            )
            
            path_objects.append(path_obj)
            path_id += 1
        
        if path_objects:
            ksp_dict[(src, dst)] = path_objects
    
    total_paths = sum(len(paths) for paths in ksp_dict.values())
    print(f"  ✓ Extracted {total_paths} paths for {len(ksp_dict)} SD pairs")
    
    return ksp_dict


def compute_ksp(G: nx.DiGraph, k: int = 5, gsnr_data_path: str=None) -> Dict[Tuple[str, str], List[Path]]:
    """
    Compute K-shortest paths on directed graph.
    
    Args:
        G: NetworkX DiGraph (directed graph)
        k: Number of shortest paths
        
    Returns:
        Dictionary {(src, dst): [Path objects]}
    """
    if gsnr_data_path:
        # print(f"Using GSNR data for paths (fast, ensures correspondence)")
        return extract_ksp_from_gsnr(gsnr_data_path, G)

    ksp_dict = {}
    path_id = 0
    
    nodes = list(G.nodes())
    
    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            
            try:
                # NetworkX automatically handles directed graphs
                paths = list(nx.shortest_simple_paths(G, src, dst, weight='length'))[:k]
            except nx.NetworkXNoPath:
                continue
            
            path_objects = []
            for node_list in paths:
                length = sum(
                    G[node_list[i]][node_list[i+1]]['length'] 
                    for i in range(len(node_list)-1)
                )
                
                path_obj = Path(
                    path_id=path_id,
                    node_list=node_list,
                    hops=len(node_list) - 1,
                    length=length
                )
                
                path_objects.append(path_obj)
                path_id += 1
            
            ksp_dict[(src, dst)] = path_objects
    
    return ksp_dict



def path_to_link_ids(G: nx.DiGraph, node_list: List[str]) -> np.ndarray:
    """
    Convert node list to array of DIRECTED link indices.
    
    Args:
        G: NetworkX DiGraph
        node_list: List of nodes forming a path
        
    Returns:
        Array of directed edge indices
        
    Example:
        >>> # Path: [1, 2, 3]
        >>> # Returns: [index(1→2), index(2→3)]
        >>> # These are directed edge indices
    """
    link_ids = []
    for i in range(len(node_list) - 1):
        # Get directed edge index
        link_id = G[node_list[i]][node_list[i+1]]['index']
        link_ids.append(link_id)
    return np.array(link_ids, dtype=np.int32)


def add_link_ids_to_paths(G: nx.DiGraph, ksp_dict: Dict) -> None:
    """Add directed link_ids to all Path objects."""
    for paths in ksp_dict.values():
        for path in paths:
            path.links_id = path_to_link_ids(G, path.node_list)


def get_topology_stats(G: nx.DiGraph) -> Dict:
    """
    Get topology statistics.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Dictionary with statistics
    """
    # For stats like diameter, use undirected version
    G_undirected = G.to_undirected()
    
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_directed_edges': G.number_of_edges(),
        'num_physical_links': G.number_of_edges() // 2,
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'diameter': nx.diameter(G_undirected) if nx.is_connected(G_undirected) else None,
        'avg_path_length': nx.average_shortest_path_length(G_undirected) if nx.is_connected(G_undirected) else None
    }
    return stats


# Test function
def test_directed_topology(topology_path: str, gsnr_path: Optional[str] = None):
    """Test directed topology loading."""
    print("="*70)
    print("TESTING DIRECTED TOPOLOGY")
    print("="*70)
    
    print(f"\n[1/3] Loading topology as directed graph...")
    G = load_topology(topology_path)
    
    stats = get_topology_stats(G)
    print(f"\nTopology Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Directed edges: {stats['num_directed_edges']}")
    print(f"  Physical links: {stats['num_physical_links']}")
    print(f"  Avg degree: {stats['avg_degree']:.2f}")
    
    # Show example directed edges
    print(f"\nExample directed edges:")
    for i, (u, v, data) in enumerate(list(G.edges(data=True))[:6]):
        print(f"  Edge {data['index']:2d}: {u} → {v} ({data['length']:.0f} km)")
    
    # Test path extraction
    print(f"\n[2/3] Testing path extraction...")
    if gsnr_path:
        ksp = extract_ksp_from_gsnr(gsnr_path, G)
    else:
        print("  Computing K=3 shortest paths...")
        ksp = compute_ksp(G, k=3)
    
    # Add link IDs
    add_link_ids_to_paths(G, ksp)
    
    # Show example paths with directed link IDs
    print(f"\n[3/3] Example paths with directed link IDs:")
    example_key = list(ksp.keys())[0]
    paths = ksp[example_key]
    
    print(f"\nPaths from {example_key[0]} to {example_key[1]}:")
    for i, path in enumerate(paths[:3]):
        print(f"  Path {i}: {' → '.join(path.node_list)}")
        print(f"    Length: {path.length:.0f} km")
        print(f"    Directed links: {path.links_id}")
    
    print("\n" + "="*70)
    print("✓ Directed topology test complete!")
    print("="*70)
    
    return G, ksp


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        topo_path = sys.argv[1]
        gsnr_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        topo_path = "topologies/topo_us24_txOnly.xlsx"
        gsnr_path = "gsnr_data/btuk_roadm_all_pairs_ksp_gsnr.pkl"
    
    G, ksp = test_directed_topology(topo_path, gsnr_path)
