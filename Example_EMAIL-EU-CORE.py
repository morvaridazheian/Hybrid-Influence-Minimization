# ============================================================
#  DEMO USING SNAP EMAIL-EU-CORE (optional)
# ============================================================

if __name__ == '__main__':
    url = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
    gz = "email-Eu-core.txt.gz"
    txt = "email-Eu-core.txt"

    if not os.path.exists(gz):
        urllib.request.urlretrieve(url, gz)

    if not os.path.exists(txt):
        with gzip.open(gz, 'rt') as f_in, open(txt, 'w') as f_out:
            for line in f_in:
                f_out.write(line)

    G = nx.read_edgelist(txt, nodetype=int, create_using=nx.DiGraph())

    random.seed(42)
    seeds = random.sample(list(G.nodes()), k=10)

    res = influence_minimization_pipeline_auto(
        G,
        seeds,
        community_top_k=3,
        budget=20,
        ic_p=0.01,
        ic_trials=500,
        ga_pop=40,
        ga_gens=80,
        random_state=42,
        ic_on_directed=True,
        protect_seeds=True,
        verbose=True
    )

    print("\nRESULTS:")
    print("Baseline:", res['baseline_spread'])
    print("Best spread:", res['best_spread'])
    print("Spread reduction:", res['spread_reduction'])
    print("Candidates:", res['candidates'])
    print("Selected:", res['best_nodes_to_remove'])

    num_candidates = len(res['candidates'])
    num_selected = len(res['best_nodes_to_remove'])
    pct = (num_selected / num_candidates * 100) if num_candidates > 0 else 0
    print(f"Percent of candidate nodes selected by GA: {pct:.2f}%")
