# Hybrid-Influence-Minimization
A hybrid framework for minimizing influence in networks using community detection (Infomap/Louvain), multi-criteria ranking (TOPSIS), and genetic optimization over Independent Cascade simulations.

# Influence Minimization in Networks
**Community Detection + TOPSIS (MADM) + Genetic Algorithm + Independent Cascade**

## 📌 Overview

This project implements an influence minimization framework for networks, primarily applied to social networks, communication graphs, collaboration networks, etc.

**Objective:**  
Identify a set of nodes whose removal leads to the lowest possible influence spread across the network.

The framework consists of four main stages:

### 1️⃣ Community Detection

The network is first divided into sub-networks (communities):

- **Directed graphs → Infomap**  
- **Undirected graphs → Louvain**

This makes the problem more scalable and allows influence minimization to be done locally inside communities before optimizing globally.

### 2️⃣ Node Ranking with Multi-Criteria Decision Making (TOPSIS)

Inside each detected community, nodes are evaluated based on multiple criteria:

- Degree  
- Betweenness  
- Closeness  
- Eigenvector centrality 

These features are fed into a **MADM ranking model (TOPSIS)** to select the most influential nodes from each community.
This produces a candidate set of nodes that represent the most structurally important actors in the network.

### 3️⃣ Optimization with a Genetic Algorithm (GA)

A Genetic Algorithm is executed over the candidate nodes to select a subset whose removal:

- ✔ Minimizes overall influence spread  
- ✔ Optimizes globally across all communities  
- ✔ Respects a user-defined removal budget 

GA parameters that can be tuned include:

- Population size  
- Number of generations  
- Crossover and mutation rates  
- Elitism  
- Optional: Seed protection (avoid removing source nodes if desired)

### 4️⃣ Influence Spread Evaluation (Independent Cascade)

Solution quality is measured using **Independent Cascade (IC) simulations**, where:

- Seeds begin the diffusion  
- Influence spreads with probability *p*  
- Each simulation is independent (Monte-Carlo)  
- Average spread across trials is used as the fitness score

### ⚠ Note on Parallelization

In the current code, IC trials are executed sequentially. However, since each simulation is independent, they can easily be parallelized later using:

- `multiprocessing`  
- `joblib`  
- `concurrent.futures`

No shared state or dependencies between trials exist, making the implementation “parallelizable by design.”

### 📊 Inputs & Outputs
#### **Inputs**
- NetworkX graph (directed or undirected)  
- Initial seed nodes  
- GA parameters  
- IC simulation configuration
  
#### **Outputs**
- Detected community structure  
- TOPSIS rankings per community  
- Final selected nodes for removal  
- Baseline vs optimized influence spread  
- Percent improvement
  
### 🧠 Why This Approach?

This framework combines the strengths of:

- **Community detection** to reduce computational complexity  
- **MADM methods** to prioritize structurally important nodes  
- **Evolutionary optimization** to search the global solution space  
- **Network simulation** to measure real-world impact

### 🚀 Running the Example

The repository includes a complete experiment based on the SNAP **Email-EU-Core** dataset.

Running the script will:

1. Load the dataset  
2. Detect communities  
3. Rank nodes using TOPSIS  
4. Optimize selections using GA  
5. Report the best nodes to remove and the achieved reduction in influence spread
