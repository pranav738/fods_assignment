"""
Build Bangalore Road Network Graph

This script:
1. Downloads the road network from OpenStreetMap
2. Maps traffic sensor locations to graph nodes
3. Computes adjacency matrix
4. Extracts graph features
5. Saves everything for later use
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.gnn.data.graph_builder import BangaloreGraphBuilder


def main():
    """Build and save Bangalore road network graph"""

    print("="*70)
    print("BANGALORE ROAD NETWORK GRAPH BUILDER")
    print("="*70 + "\n")

    # Configuration
    place_name = "Bangalore, India"
    network_type = "drive"  # drivable roads only
    output_dir = "./datasets/processed/graph"
    cache_dir = "./datasets/cache"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize graph builder
    print("Initializing graph builder...")
    builder = BangaloreGraphBuilder(
        place_name=place_name,
        network_type=network_type,
        simplify=True
    )

    # Download graph from OpenStreetMap
    print(f"\nDownloading road network for {place_name}...")
    print("This may take a few minutes...")
    graph = builder.download_graph(cache_dir=cache_dir)

    print(f"\nGraph downloaded successfully!")
    print(f"  Nodes: {graph.number_of_nodes():,}")
    print(f"  Edges: {graph.number_of_edges():,}")

    # Load actual traffic sensor locations from dataset
    print("\nLoading actual traffic locations from dataset...")

    locations_file = "./datasets/processed/bangalore_locations.csv"

    if os.path.exists(locations_file):
        # Load real coordinates
        locations_df = pd.read_csv(locations_file)

        # Create unique location name from area + road
        locations_df['location_name'] = locations_df['area_name'] + ' - ' + locations_df['road_intersection_name']

        # Select relevant columns
        locations_df = locations_df[['location_name', 'latitude', 'longitude', 'area_name', 'road_intersection_name']]

        print(f"Loaded {len(locations_df)} actual traffic locations from dataset:")
        for idx, row in locations_df.iterrows():
            print(f"  • {row['location_name']}")
    else:
        print(f"Warning: {locations_file} not found. Using sample locations...")
        # Fallback to sample locations
        traffic_locations = [
            {"name": "Indiranagar - 100 Feet Road", "lat": 12.9727, "lon": 77.6410},
            {"name": "Koramangala - Sony World Junction", "lat": 12.9374, "lon": 77.6272},
            {"name": "Whitefield - Marathahalli Bridge", "lat": 12.9567, "lon": 77.7012},
        ]
        locations_df = pd.DataFrame(traffic_locations)
        locations_df.columns = ['location_name', 'latitude', 'longitude']

    # Map locations to graph nodes
    try:
        node_mapping = builder.map_locations_to_nodes(
            locations_df,
            lat_col='latitude',
            lon_col='longitude',
            name_col='location_name'
        )

        print(f"\nMapped {len(node_mapping)} locations to graph nodes")
        for location, node_id in list(node_mapping.items())[:5]:
            print(f"  {location}: Node {node_id}")

        # Get mapped nodes
        mapped_nodes = list(node_mapping.values())

    except Exception as e:
        print(f"Error mapping locations: {e}")
        print("Using sample nodes from graph...")
        import random
        mapped_nodes = random.sample(list(graph.nodes()), min(20, graph.number_of_nodes()))
        node_mapping = {f"Location_{i}": node for i, node in enumerate(mapped_nodes)}

    # Build adjacency matrix
    print("\nBuilding adjacency matrix...")
    adjacency_binary = builder.build_adjacency_matrix(
        nodes=mapped_nodes,
        weight='binary'
    )

    adjacency_weighted = builder.build_adjacency_matrix(
        nodes=mapped_nodes,
        weight='length'
    )

    print(f"Adjacency matrix shape: {adjacency_binary.shape}")
    print(f"Number of edges: {np.sum(adjacency_binary > 0)}")

    # Compute distance matrix
    print("\nComputing pairwise distances...")
    distances = builder.compute_distance_matrix(nodes=mapped_nodes)

    print(f"Average distance: {np.mean(distances[distances > 0]):.2f} km")
    print(f"Max distance: {np.max(distances):.2f} km")

    # Build Gaussian-weighted adjacency
    print("\nBuilding Gaussian-weighted adjacency matrix...")
    adjacency_gaussian = builder.build_gaussian_adjacency(
        nodes=mapped_nodes,
        sigma=10.0,  # 10 km bandwidth
        epsilon=0.1  # threshold
    )

    print(f"Gaussian adjacency density: {np.sum(adjacency_gaussian > 0) / adjacency_gaussian.size:.2%}")

    # Extract graph features
    print("\nExtracting graph-theoretic features...")
    graph_features = builder.extract_graph_features(nodes=mapped_nodes)

    print("\nGraph features preview:")
    print(graph_features.head())

    # Save everything
    print(f"\nSaving graph data to {output_dir}...")
    builder.save(output_dir)

    # Save additional matrices
    np.savez_compressed(
        os.path.join(output_dir, "adjacency_matrices.npz"),
        binary=adjacency_binary,
        weighted=adjacency_weighted,
        gaussian=adjacency_gaussian
    )

    # Save graph features
    graph_features.to_csv(
        os.path.join(output_dir, "graph_features.csv"),
        index=False
    )

    # Save location mapping
    location_df = pd.DataFrame([
        {"location_name": name, "node_id": node_id}
        for name, node_id in node_mapping.items()
    ])
    location_df.to_csv(
        os.path.join(output_dir, "location_mapping.csv"),
        index=False
    )

    print("\n" + "="*70)
    print("GRAPH BUILDING COMPLETE!")
    print("="*70)
    print(f"\nOutput files saved to: {output_dir}")
    print(f"  - road_network.pkl: Full road network graph")
    print(f"  - adjacency_matrix.npz: Binary adjacency matrix")
    print(f"  - adjacency_matrices.npz: All adjacency variants")
    print(f"  - distance_matrix.npz: Pairwise distances")
    print(f"  - graph_features.csv: Node-level graph features")
    print(f"  - location_mapping.csv: Sensor-to-node mapping")
    print(f"  - node_mapping.pkl: Node mapping dictionary")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
