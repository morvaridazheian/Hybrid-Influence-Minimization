"""
Example_EMAIL-EU-CORE.py
----------
Example script demonstrating how to run the Hybrid Influence Minimization pipeline
on the SNAP Email-EU-Core network dataset.

Steps:
1. Download and extract the dataset
2. Build a directed graph using NetworkX
3. Randomly choose initial seed nodes
4. Run the influence_minimization_pipeline_auto()
5. Print baseline vs optimized influence spread
"""

import os
import gzip
import random
import urllib.request
import networkx as nx

# Import the main pipeline function 
from Influence-Minimization-Pipeline import influence_minimization_pipeline_auto


# ============================================================
#  DEMO USING SNAP EMAIL-EU-CORE
# ============================================================
if __name__ == '__main__':

    # SNAP dataset URL
    url = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
    gz_file = "email-Eu-core.txt.gz"
    txt_file = "email-Eu-core.txt"

    # ------------------------------------------------------------
    # Download dataset if needed
    # ------------------------------------------------------------
    if not os.path.exists(gz_file):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, gz_file)

    # ------------------------------------------------------------
    # Extract text data if needed
    # ------------------------------------------------------------
    if not os.path.exists(txt_file):
        print("Extracting dataset...")
        with gzip.open(gz_file, 'rt') as f_in, open(txt_file, 'w') as f_out:
            for line in f_in:
                f_out.write(line)

    # ------------------------------------------------------------
    # Load graph (directed)
    # ------------------------------------------------------------
    G = nx.read_edgelist(txt_file, nodetype=int, create_using=nx.DiGraph())
    print(f"Loaded Email-EU-Core with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # ------------------------------------------------------------
    # Random seed selection for influence simulation
    # ------------------------------------------------------------
    random.seed(42)
    seeds = random.sample(list(G.nodes()), k=10)

    # ------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------
    res = influence_minimization_pipeline_auto(
        G,
        seeds,
        community_top_k=3,      # Number of top nodes per community taken from TOPSIS
        budget=20,              # Max nodes that can be removed globally
        ic_p=0.01,              # Independent Cascade probability
        ic_trials=500,          # Monte-Carlo simulations
        ga_pop=40,              # GA population size
        ga_gens=80,             # GA generations
        random_state=42,        # Reproducibility
        ic_on_directed=True,    # IC runs on directed graph
        protect_seeds=True,     # Prevent removal of seed nodes
        verbose=True
    )

    # ------------------------------------------------------------
    # Report Results
    # ------------------------------------------------------------
    print("\n========== RESULTS ==========")
    print("Baseline spread :", res['baseline_spread'])
    print("Best spread     :", res['best_spread'])
    print("Spread reduction:", res['spread_reduction'])
    print("Candidate nodes :", res['candidates'])
    print("GA-selected     :", res['best_nodes_to_remove'])

    num_candidates = len(res['candidates'])
    num_selected = len(res['best_nodes_to_remove'])
    pct = (num_selected / num_candidates * 100) if num_candidates > 0 else 0

    print(f"Percent of candidate nodes chosen by GA: {pct:.2f}%")
