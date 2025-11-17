"""
Bangalore Road Network Graph Builder using OpenStreetMap data

Constructs the spatial graph structure that is critical for GNN models.
"""

import os
import pickle
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import networkx as nx

try:
    import osmnx as ox
    import geopandas as gpd
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("Warning: OSMnx not available. Graph building from OSM will not work.")

from ..utils.graph_utils import build_spatial_adjacency, compute_graph_features


class BangaloreGraphBuilder:
    """
    Build spatial graph for Bangalore traffic network

    This class:
    1. Downloads road network from OpenStreetMap using OSMnx
    2. Maps traffic sensor locations to graph nodes
    3. Computes adjacency matrix
    4. Computes graph-theoretic features (betweenness, PageRank, etc.)
    """

    def __init__(self,
                 place_name: str = "Bangalore, India",
                 network_type: str = "drive",
                 simplify: bool = True):
        """
        Initialize graph builder

        Args:
            place_name: Location to query in OSM
            network_type: Type of road network ('drive', 'walk', 'bike', 'all')
            simplify: Whether to simplify the graph (merge nodes)
        """
        if not OSMNX_AVAILABLE:
            raise ImportError("OSMnx is required for graph building. Install with: uv add osmnx")

        self.place_name = place_name
        self.network_type = network_type
        self.simplify = simplify
        self.graph = None
        self.node_mapping = None
        self.adjacency_matrix = None
        self.distance_matrix = None

    def download_graph(self, cache_dir: Optional[str] = None) -> nx.MultiDiGraph:
        """
        Download road network graph from OpenStreetMap

        Args:
            cache_dir: Directory to cache downloaded graph

        Returns:
            NetworkX graph of road network
        """
        cache_file = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{self.place_name.replace(' ', '_')}_graph.pkl")

            if os.path.exists(cache_file):
                print(f"Loading cached graph from {cache_file}")
                with open(cache_file, 'rb') as f:
                    self.graph = pickle.load(f)
                return self.graph

        print(f"Downloading road network for {self.place_name}...")

        # Download graph from OpenStreetMap
        self.graph = ox.graph_from_place(
            self.place_name,
            network_type=self.network_type,
            simplify=self.simplify
        )

        # Add edge speeds and travel times
        self.graph = ox.add_edge_speeds(self.graph)
        self.graph = ox.add_edge_travel_times(self.graph)

        # Cache if requested
        if cache_file:
            print(f"Caching graph to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.graph, f)

        return self.graph

    def map_locations_to_nodes(self,
                               locations: pd.DataFrame,
                               lat_col: str = 'latitude',
                               lon_col: str = 'longitude',
                               name_col: str = 'location_name') -> Dict[str, int]:
        """
        Map traffic sensor locations to nearest graph nodes

        Args:
            locations: DataFrame with sensor locations
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            name_col: Column name for location name

        Returns:
            Dictionary mapping location names to node IDs
        """
        if self.graph is None:
            raise ValueError("Graph not loaded. Call download_graph() first.")

        mapping = {}

        for idx, row in locations.iterrows():
            lat = row[lat_col]
            lon = row[lon_col]
            name = row[name_col]

            # Find nearest network node
            nearest_node = ox.distance.nearest_nodes(self.graph, lon, lat)
            mapping[name] = nearest_node

        self.node_mapping = mapping
        return mapping

    def map_locations_by_name(self,
                             location_names: List[str],
                             use_geocoding: bool = True) -> Dict[str, int]:
        """
        Map location names to graph nodes using geocoding

        Useful when lat/lon coordinates are not available

        Args:
            location_names: List of location names (e.g., "Koramangala, Bangalore")
            use_geocoding: Whether to use Nominatim geocoding

        Returns:
            Dictionary mapping location names to node IDs
        """
        if self.graph is None:
            raise ValueError("Graph not loaded. Call download_graph() first.")

        mapping = {}

        for name in location_names:
            try:
                # Geocode location name
                full_name = f"{name}, {self.place_name}" if "Bangalore" not in name else name
                location = ox.geocode(full_name)

                # Find nearest node
                nearest_node = ox.distance.nearest_nodes(
                    self.graph,
                    location[1],  # longitude
                    location[0]   # latitude
                )

                mapping[name] = nearest_node
                print(f"Mapped {name} -> node {nearest_node}")

            except Exception as e:
                print(f"Warning: Could not map location '{name}': {e}")

        self.node_mapping = mapping
        return mapping

    def build_adjacency_matrix(self,
                              nodes: Optional[List[int]] = None,
                              weight: str = 'length',
                              normalized: bool = False) -> np.ndarray:
        """
        Build adjacency matrix from graph

        Args:
            nodes: List of nodes to include (if None, use all nodes)
            weight: Edge attribute to use as weight ('length', 'travel_time', etc.)
            normalized: Whether to normalize the matrix

        Returns:
            Adjacency matrix (N x N)
        """
        if self.graph is None:
            raise ValueError("Graph not loaded. Call download_graph() first.")

        # Get subgraph if specific nodes requested
        if nodes is not None:
            subgraph = self.graph.subgraph(nodes)
        else:
            subgraph = self.graph

        # Convert to undirected simple graph
        G_undirected = nx.Graph(subgraph)

        # Get adjacency matrix
        if weight and weight != 'binary':
            # Weighted adjacency
            adj_matrix = nx.adjacency_matrix(G_undirected, weight=weight).toarray()
        else:
            # Binary adjacency
            adj_matrix = nx.adjacency_matrix(G_undirected).toarray()

        if normalized:
            # Inverse weight normalization (shorter distance = higher weight)
            adj_matrix = 1.0 / (adj_matrix + 1e-6)
            np.fill_diagonal(adj_matrix, 0)

        self.adjacency_matrix = adj_matrix
        return adj_matrix

    def compute_distance_matrix(self, nodes: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute pairwise distances between nodes

        Args:
            nodes: List of nodes (if None, use all nodes)

        Returns:
            Distance matrix (N x N) in kilometers
        """
        if self.graph is None:
            raise ValueError("Graph not loaded. Call download_graph() first.")

        if nodes is None:
            nodes = list(self.graph.nodes())

        n = len(nodes)
        distances = np.zeros((n, n))

        # Get node coordinates
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i != j:
                    try:
                        # Use Dijkstra to find shortest path distance
                        distance = nx.shortest_path_length(
                            self.graph,
                            node_i,
                            node_j,
                            weight='length'
                        )
                        distances[i, j] = distance / 1000.0  # Convert to km
                    except nx.NetworkXNoPath:
                        # No path exists, use large distance
                        distances[i, j] = 999.0

        self.distance_matrix = distances
        return distances

    def build_gaussian_adjacency(self,
                                 nodes: Optional[List[int]] = None,
                                 sigma: float = 10.0,
                                 epsilon: float = 0.5) -> np.ndarray:
        """
        Build adjacency matrix using Gaussian kernel on distances

        w_ij = exp(-(d_ij^2) / (2 * sigma^2))

        This creates a fully connected graph with distance-weighted edges.

        Args:
            nodes: List of nodes
            sigma: Gaussian kernel bandwidth (in km)
            epsilon: Threshold (weights < epsilon set to 0)

        Returns:
            Gaussian-weighted adjacency matrix
        """
        # Compute distance matrix if not already computed
        if self.distance_matrix is None or nodes is not None:
            distances = self.compute_distance_matrix(nodes)
        else:
            distances = self.distance_matrix

        # Build Gaussian adjacency
        adj_matrix = build_spatial_adjacency(distances, sigma, epsilon)

        self.adjacency_matrix = adj_matrix
        return adj_matrix

    def extract_graph_features(self, nodes: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Extract graph-theoretic features for nodes

        Features include:
        - Degree centrality
        - Betweenness centrality (critical junctions)
        - Closeness centrality
        - PageRank
        - Clustering coefficient

        Args:
            nodes: List of nodes to extract features for

        Returns:
            DataFrame with node features
        """
        if self.graph is None:
            raise ValueError("Graph not loaded. Call download_graph() first.")

        # Get subgraph if specific nodes requested
        if nodes is not None:
            subgraph = nx.Graph(self.graph.subgraph(nodes))
        else:
            subgraph = nx.Graph(self.graph)
            nodes = list(subgraph.nodes())

        # Compute features
        features = compute_graph_features(subgraph)

        # Convert to DataFrame
        feature_df = pd.DataFrame({
            'node_id': nodes,
            'degree_centrality': [features['degree_centrality'].get(n, 0) for n in nodes],
            'betweenness_centrality': [features['betweenness_centrality'].get(n, 0) for n in nodes],
            'closeness_centrality': [features['closeness_centrality'].get(n, 0) for n in nodes],
            'pagerank': [features['pagerank'].get(n, 0) for n in nodes],
            'clustering': [features['clustering'].get(n, 0) for n in nodes],
        })

        return feature_df

    def save(self, output_dir: str):
        """
        Save graph, adjacency matrix, and mappings

        Args:
            output_dir: Directory to save outputs
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save graph
        if self.graph is not None:
            graph_file = os.path.join(output_dir, "road_network.pkl")
            with open(graph_file, 'wb') as f:
                pickle.dump(self.graph, f)

        # Save adjacency matrix
        if self.adjacency_matrix is not None:
            adj_file = os.path.join(output_dir, "adjacency_matrix.npz")
            np.savez_compressed(adj_file, adjacency=self.adjacency_matrix)

        # Save distance matrix
        if self.distance_matrix is not None:
            dist_file = os.path.join(output_dir, "distance_matrix.npz")
            np.savez_compressed(dist_file, distances=self.distance_matrix)

        # Save node mapping
        if self.node_mapping is not None:
            mapping_file = os.path.join(output_dir, "node_mapping.pkl")
            with open(mapping_file, 'wb') as f:
                pickle.dump(self.node_mapping, f)

        print(f"Graph data saved to {output_dir}")

    @staticmethod
    def load(input_dir: str) -> Tuple[nx.Graph, np.ndarray, Dict]:
        """
        Load previously saved graph data

        Args:
            input_dir: Directory containing saved data

        Returns:
            Tuple of (graph, adjacency_matrix, node_mapping)
        """
        # Load graph
        graph_file = os.path.join(input_dir, "road_network.pkl")
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f)

        # Load adjacency matrix
        adj_file = os.path.join(input_dir, "adjacency_matrix.npz")
        adjacency_matrix = np.load(adj_file)['adjacency']

        # Load node mapping
        mapping_file = os.path.join(input_dir, "node_mapping.pkl")
        with open(mapping_file, 'rb') as f:
            node_mapping = pickle.load(f)

        return graph, adjacency_matrix, node_mapping
