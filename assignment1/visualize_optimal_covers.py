"""
Script to find and visualize optimal vertex covers for specific graphs
"""
import time
from GraphGenerator import load_graph
from Visualization import visualize_graph_with_cover
from Exhaustive import exhaustive_search_optimized

# Graph specifications
graphs_to_solve = [
    (15, 0.75, "15 Nodes, Density 0.75"),
    (20, 0.75, "20 Nodes, Density 0.75"),
    (25, 0.5, "25 Nodes, Density 0.5")
]

print("Finding optimal vertex covers and generating visualizations...")
print("=" * 70)
print()

for n, density, description in graphs_to_solve:
    # Load graph
    graph_file = f"Grafos/graph_n{n}_d{int(density*100)}.pkl"
    print(f"Processing: {description}")
    print(f"Loading {graph_file}...")

    try:
        G = load_graph(graph_file)

        # Print graph info
        print(f"  Vertices: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")

        # Find optimal vertex cover
        print(f"  Running exhaustive search (this may take a while for n={n})...")
        start_time = time.time()

        cover, weight, ops, configurations, elapsed = exhaustive_search_optimized(G)

        optimal_cover = cover
        optimal_weight = weight

        print(f"  ✓ Found optimal solution in {elapsed:.2f} seconds")
        print(f"  Optimal cover size: {len(optimal_cover)} vertices")
        print(f"  Optimal weight: {optimal_weight}")
        print(f"  Configurations tested: {configurations:,}")
        print(f"  Optimal vertices: {sorted(optimal_cover)}")

        # Generate visualization
        output_file = f"Outputs/graph_n{n}_d{int(density*100)}_optimal.png"
        visualize_graph_with_cover(
            G,
            vertex_cover=optimal_cover,
            title=f"Optimal Vertex Cover: {description}\nCover size: {len(optimal_cover)}, Weight: {optimal_weight}",
            output_file=output_file
        )
        print(f"  Saved visualization to: {output_file}")
        print()

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        print()

print("=" * 70)
print("Done!")
