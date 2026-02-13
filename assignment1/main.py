"""
Main Runner for Minimum Weight Vertex Cover Project
Executes the complete workflow: generate graphs, run experiments, visualize results
"""

import os
import sys
import time


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")


def main():
    """Main execution workflow."""
    print_header("Minimum Weight Vertex Cover - Project 1")
    print("Advanced Algorithms 2025/2026")
    print("University of Aveiro\n")

    # Step 1: Generate graphs
    print_header("Step 1: Generating Test Graphs")
    print("This will generate random graphs with varying sizes and edge densities...")
    print("Graphs will be saved to Grafos/ directory\n")

    input("Press Enter to continue...")

    from GraphGenerator import generate_all_graphs, STUDENT_NUMBER

    try:
        graphs = generate_all_graphs(
            min_vertices=4,
            max_vertices=15,
            densities=[0.125, 0.25, 0.5, 0.75],
            seed=STUDENT_NUMBER
        )
        print(f"\n✓ Successfully generated {len(graphs)} graphs")
    except Exception as e:
        print(f"\n✗ Error generating graphs: {e}")
        return

    # Step 2: Run experiments
    print_header("Step 2: Running Experiments")
    print("This will run both exhaustive search and greedy heuristics...")
    print("WARNING: This may take several minutes depending on graph sizes")
    print("Results will be saved to Outputs/ directory\n")

    response = input("Continue with experiments? (y/n): ")
    if response.lower() != 'y':
        print("Skipping experiments. You can run them later with: python Experiments.py")
        return

    from Experiments import run_experiments, save_results, generate_summary_table

    try:
        start_time = time.time()

        results = run_experiments(
            min_vertices=4,
            max_vertices=12,  # Conservative for exhaustive search
            densities=[0.125, 0.25, 0.5, 0.75],
            timeout=300
        )

        elapsed_time = time.time() - start_time

        # Save results
        output_file = f"Outputs/Results_{STUDENT_NUMBER}.txt"
        save_results(results, output_file)

        # Generate summary
        summary = generate_summary_table(results)
        print("\n" + summary)

        summary_file = f"Outputs/Summary_{STUDENT_NUMBER}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)

        print(f"\n✓ Experiments completed in {elapsed_time:.2f} seconds")
        print(f"✓ Results saved to {output_file}")
        print(f"✓ Summary saved to {summary_file}")

    except Exception as e:
        print(f"\n✗ Error during experiments: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Generate visualizations
    print_header("Step 3: Generating Visualizations")
    print("This will create plots and graphs for the report...")
    print("Figures will be saved to Outputs/Figures/ directory\n")

    response = input("Continue with visualization? (y/n): ")
    if response.lower() != 'y':
        print("Skipping visualization. You can generate them later with: python Visualization.py")
        print_header("Workflow Complete!")
        return

    from Visualization import generate_all_visualizations

    try:
        results_file = f"Outputs/Results_{STUDENT_NUMBER}.pkl"
        generate_all_visualizations(results_file)

        print("\n✓ All visualizations generated successfully")

    except Exception as e:
        print(f"\n✗ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print_header("Workflow Complete!")
    print("Generated files:")
    print(f"  - Graphs: Grafos/")
    print(f"  - Results: Outputs/Results_{STUDENT_NUMBER}.txt")
    print(f"  - Summary: Outputs/Summary_{STUDENT_NUMBER}.txt")
    print(f"  - Figures: Outputs/Figures/")
    print("\nNext steps:")
    print("  1. Review the results and summary tables")
    print("  2. Examine the visualizations in Outputs/Figures/")
    print("  3. Use this data to write your report")
    print("  4. Perform formal complexity analysis")
    print("  5. Compare experimental results with theoretical analysis\n")


if __name__ == "__main__":
    main()
