# ------------------------------------------------------------
# Influence Minimization Pipeline (Directed → Infomap,
# Undirected → Louvain)
# Requires: networkx, numpy, pandas, python-louvain, infomap
# ------------------------------------------------------------

import random, math, os, gzip, urllib.request
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# ---------------- Imports for community detection ----------------
try:
    from infomap import Infomap
except Exception as e:
    raise ImportError("Install infomap: pip install infomap. Error: {}".format(e))

try:
    from community import community_louvain
except Exception as e:
    raise ImportError("Install community-louvain: pip install python-louvain. Error: {}".format(e))


# ============================================================
#  COMMUNITY DETECTION
# ============================================================

# ---- Infomap (directed) ----
def detect_communities_infomap(G: nx.Graph) -> Dict[int, int]:
    im = Infomap()
    # Add edges preserving directionality
    for u, v in G.edges():
        im.add_link(int(u), int(v))
    im.run()
    return {int(node.node_id): int(node.module_id) for node in im.nodes}


# ---- Louvain (undirected) ----
def detect_communities_louvain(G: nx.Graph) -> Dict[int, int]:
    # community_louvain returns node→community directly
    return community_louvain.best_partition(G)


# ---- Automatic selection based on graph direction ----
def detect_communities_auto(G: nx.Graph) -> Dict[int, int]:
    """
    - Directed  → Infomap
    - Undirected → Louvain
    """
    if G.is_directed():
        print("Detecting communities using INFOMAP (directed graph)...")
        return detect_communities_infomap(G)
    else:
        print("Detecting communities using LOUVAIN (undirected graph)...")
        return detect_communities_louvain(G)


# ============================================================
#  NODE FEATURES
# ============================================================

def compute_node_features(G: nx.Graph) -> pd.DataFrame:
    nodes = list(G.nodes())
    deg = dict(G.degree())
    bet = nx.betweenness_centrality(G, normalized=True)
    clo = nx.closeness_centrality(G)

    try:
        eig = nx.eigenvector_centrality(G, max_iter=500)
    except Exception:
        eig = _approx_eigenvector_centrality(G)

    df = pd.DataFrame({
        'node': nodes,
        'degree': [deg[n] for n in nodes],
        'betweenness': [bet[n] for n in nodes],
        'closeness': [clo[n] for n in nodes],
        'eigenvector': [eig[n] for n in nodes],
    })
    return df


def _approx_eigenvector_centrality(G: nx.Graph, max_iter=100, tol=1e-6) -> Dict:
    A = nx.to_numpy_array(G)
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for _ in range(max_iter):
        v_next = A.dot(v)
        norm = np.linalg.norm(v_next)
        if norm == 0:
            break
        v_next = v_next / norm
        if np.linalg.norm(v_next - v) < tol:
            v = v_next
            break
        v = v_next

    return {node: float(v[i]) for i, node in enumerate(G.nodes())}


# ============================================================
#  TOPSIS RANKING
# ============================================================

def topsis_score(df: pd.DataFrame, criteria: List[str], weights: List[float] = None, impacts: List[int] = None) -> pd.Series:
    X = df[criteria].astype(float).values
    m, n = X.shape

    weights = np.array(weights if weights else [1.0] * n, dtype=float)
    weights = weights / np.sum(weights)

    impacts = np.array(impacts if impacts else [1] * n, dtype=int)

    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    R = X / norm

    V = R * weights

    ideal = np.where(impacts == 1, V.max(axis=0), V.min(axis=0))
    nadir = np.where(impacts == 1, V.min(axis=0), V.max(axis=0))

    d_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - nadir) ** 2).sum(axis=1))

    return pd.Series(d_neg / (d_pos + d_neg + 1e-12), index=df.index)


def select_top_by_community(
    G: nx.Graph,
    partition: Dict[int, int],
    df_features: pd.DataFrame,
    top_k: int = 1,
    criteria: List[str] = None,
    weights: List[float] = None,
    impacts: List[int] = None
) -> Tuple[List[int], Dict]:

    if criteria is None:
        criteria = ['degree', 'betweenness', 'closeness', 'eigenvector']

    communities = defaultdict(list)
    for node, c in partition.items():
        communities[c].append(node)

    candidates = set()
    community_scores = {}

    for c, nodes in communities.items():
        sub_df = df_features[df_features['node'].isin(nodes)].copy()
        if sub_df.empty:
            continue

        scores = topsis_score(sub_df, criteria, weights, impacts)
        sub_df = sub_df.assign(topsis_score=scores.values).sort_values('topsis_score', ascending=False)
        sel = sub_df.head(top_k)['node'].tolist()

        candidates.update(sel)
        community_scores[c] = sub_df[['node', 'topsis_score']].reset_index(drop=True)

    return list(candidates), community_scores


# ============================================================
#  INDEPENDENT CASCADE (IC) MODEL
# ============================================================

def independent_cascade(
    G: nx.Graph,
    seeds: List[int],
    p: float = 0.01,
    trials: int = 100,
    removed_nodes: List[int] = None
) -> float:

    if removed_nodes is None:
        removed_nodes = []

    total = 0
    G2 = G.copy()
    G2.remove_nodes_from(removed_nodes)
    nodes_in_G2 = set(G2.nodes())

    for _ in range(trials):
        active = set([s for s in seeds if s in nodes_in_G2])
        newly = set(active)
        activated = set(active)

        while newly:
            newset = set()
            for u in newly:
                for v in G2.neighbors(u):
                    if v in activated:
                        continue
                    if random.random() <= p:
                        newset.add(v)
            newly = newset
            activated.update(newset)

        total += len(activated)

    return total / float(trials)


# ============================================================
#  GENETIC ALGORITHM
# ============================================================

def repair_solution(bitstring: np.ndarray, budget: int) -> np.ndarray:
    arr = bitstring.copy()
    ones = int(arr.sum())
    n = len(arr)

    if ones > budget:
        idxs = np.where(arr == 1)[0]
        to_flip = np.random.choice(idxs, size=(ones - budget), replace=False)
        arr[to_flip] = 0
    elif ones < budget:
        idxs = np.where(arr == 0)[0]
        if len(idxs) > 0:
            to_flip = np.random.choice(
                idxs, size=min(budget - ones, len(idxs)), replace=False
            )
            arr[to_flip] = 1

    return arr


def ga_minimize_spread(
    G: nx.Graph,
    candidates: List[int],
    seeds: List[int],
    budget: int,
    p: float = 0.01,
    trials: int = 200,
    population_size: int = 40,
    generations: int = 60,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.02,
    elitism: int = 2,
    random_state: int = None,
    protect_seeds: bool = True,
    verbose: bool = True
) -> Tuple[List[int], float]:

    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    if protect_seeds:
        candidates = [c for c in candidates if c not in seeds]

    m = len(candidates)
    if m == 0:
        return [], independent_cascade(G, seeds, p=p, trials=trials, removed_nodes=[])

    idx2node = {i: candidates[i] for i in range(m)}

    population = []
    for _ in range(population_size):
        bit = np.zeros(m, dtype=int)
        chosen = np.random.choice(range(m), size=min(budget, m), replace=False)
        bit[chosen] = 1
        population.append(bit)

    fitness_cache = {}

    def eval_individual(bit):
        key = tuple(bit.tolist())
        if key in fitness_cache:
            return fitness_cache[key]

        sel_nodes = [idx2node[i] for i in np.where(bit == 1)[0]]
        val = independent_cascade(G, seeds, p=p, trials=trials, removed_nodes=sel_nodes)
        fitness_cache[key] = val
        return val

    best = None
    best_f = float('inf')

    if verbose:
        print(
            "GA start:",
            "population", population_size,
            "generations", generations,
            "candidates", m,
            "budget", budget
        )

    for gen in range(generations):
        scores = [eval_individual(ind) for ind in population]

        for ind, sc in zip(population, scores):
            if sc < best_f:
                best_f = sc
                best = ind.copy()

        if verbose and (gen % max(1, generations // 10) == 0):
            print(f" Gen {gen}: best spread so far = {best_f:.4f}")

        new_pop = []
        sorted_idx = np.argsort(scores)

        for i_e in range(elitism):
            new_pop.append(population[int(sorted_idx[i_e])].copy())

        while len(new_pop) < population_size:
            a, b = np.random.choice(population_size, size=2, replace=False)
            parent1 = population[a].copy() if scores[a] < scores[b] else population[b].copy()

            a, b = np.random.choice(population_size, size=2, replace=False)
            parent2 = population[a].copy() if scores[a] < scores[b] else population[b].copy()

            if random.random() < crossover_rate:
                pt = random.randint(1, m - 1)
                child1 = np.concatenate([parent1[:pt], parent2[pt:]])
                child2 = np.concatenate([parent2[:pt], parent1[pt:]])
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            for child in (child1, child2):
                for i in range(m):
                    if random.random() < mutation_rate:
                        child[i] = 1 - child[i]
                child = repair_solution(child, budget)
                new_pop.append(child)
                if len(new_pop) >= population_size:
                    break

        population = new_pop

    best_nodes = [idx2node[i] for i in np.where(best == 1)[0]] if best is not None else []

    if verbose:
        print(f"GA finished. Best spread = {best_f:.4f}; Best nodes(selected) = {best_nodes}")

    return best_nodes, best_f


# ============================================================
#  FULL PIPELINE (WITH AUTO COMMUNITY DETECTION)
# ============================================================

def influence_minimization_pipeline_auto(
    G: nx.Graph,
    seeds: List[int],
    community_top_k: int = 2,
    budget: int = 5,
    ic_p: float = 0.01,
    ic_trials: int = 300,
    ga_pop: int = 40,
    ga_gens: int = 60,
    random_state: int = None,
    ic_on_directed: bool = True,
    protect_seeds: bool = True,
    verbose: bool = True
):
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # 1) Automatic community detection
    partition = detect_communities_auto(G)

    # 2) node features
    features = compute_node_features(G)

    # 3) TOPSIS selection inside each community
    candidates, community_scores = select_top_by_community(
        G,
        partition,
        features,
        top_k=community_top_k
    )

    if verbose:
        print("Communities found:", len(set(partition.values())))
        print("Candidates (TOPSIS):", candidates)
        for c, df in community_scores.items():
            print(f" Community {c} TOPSIS:")
            print(df.head().to_string(index=False))

    # 4) graph choice for IC
    G_ic = G if ic_on_directed else G.to_undirected()

    baseline = independent_cascade(
        G_ic, seeds, p=ic_p, trials=ic_trials, removed_nodes=[]
    )

    if verbose:
        mode = "directed" if ic_on_directed and G.is_directed() else "undirected"
        print(f"Baseline spread (IC on {mode} graph): {baseline:.4f}")

    # 5) GA optimization
    best_nodes, best_spread = ga_minimize_spread(
        G_ic,
        candidates,
        seeds,
        budget=budget,
        p=ic_p,
        trials=ic_trials,
        population_size=ga_pop,
        generations=ga_gens,
        random_state=random_state,
        protect_seeds=protect_seeds,
        verbose=verbose
    )

    return {
        'partition': partition,
        'candidates': candidates,
        'community_scores': community_scores,
        'baseline_spread': baseline,
        'best_nodes_to_remove': best_nodes,
        'best_spread': best_spread,
        'spread_reduction': baseline - best_spread
    }


