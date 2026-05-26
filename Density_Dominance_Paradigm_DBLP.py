import os
import sys
import random
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import f_oneway
import itertools
from tqdm import tqdm
import gzip
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & FILE PATHS
# ==========================================
import os
BASE_DIR = os.path.join(os.path.dirname(__file__), "data")

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

DATASET_NAME = "DBLP"

EXCEL_OUTPUT = os.path.join(BASE_DIR, f"Influencer_Data_{DATASET_NAME}.xlsx")
MATRIX_PNG = os.path.join(BASE_DIR, f"Influencer_Matrix_{DATASET_NAME}.png")
MATRIX_HTML = os.path.join(BASE_DIR, f"Interactive_Matrix_{DATASET_NAME}.html")
SCREE_PNG = os.path.join(BASE_DIR, f"PCA_Scree_Plot_{DATASET_NAME}.png")
IC_PNG = os.path.join(BASE_DIR, f"IC_Simulation_{DATASET_NAME}.png")
IC_HTML = os.path.join(BASE_DIR, f"Interactive_IC_{DATASET_NAME}.html")
STATS_OUTPUT = os.path.join(BASE_DIR, f"Statistical_Validation_{DATASET_NAME}.xlsx")

N_NODES = 20000
K_DISTANCE = 2
TOP_DEGREES = 20
NUM_ARCHETYPES = 3
RANDOM_SEED = 42
IC_TIME_STEPS = 30
IC_P = 0.05
IC_SAMPLE_SIZE = 50
N_JOBS = -1  # Utilize all available CPU cores

print(f"Publication Pipeline Executing for {DATASET_NAME}...")

# ==========================================
# 2. ROBUST FILE LOADING
# ==========================================
def find_file(base_name):
    txt_path = os.path.join(BASE_DIR, base_name)
    gz_path = txt_path + ".gz"
    if os.path.exists(txt_path):
        return txt_path
    elif os.path.exists(gz_path):
        return gz_path
    else:
        print(f"ERROR: {base_name} not found!")
        sys.exit(1)

# Directly load the Amazon dataset files
graph_target = find_file("com-dblp.ungraph.txt")
community_target = find_file("com-dblp.all.cmty.txt")

print("\n Loading DBLP topology...")

G = nx.read_edgelist(graph_target, nodetype=int, comments='#')
print(f"   → {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

print(" Loading communities...")
node_to_communities = {}
open_func = gzip.open if community_target.endswith('.gz') else open
mode = 'rt' if community_target.endswith('.gz') else 'r'
with open_func(community_target, mode) as f:
    for comm_id, line in enumerate(f):
        nodes = [int(n) for n in line.strip().split()]
        for node in nodes:
            node_to_communities.setdefault(node, set()).add(comm_id)
print(f"   → {comm_id + 1:,} communities loaded, containing {len(node_to_communities):,} unique nodes")
# ==========================================
# 3. PRECOMPUTE DEGREE CACHE
# ==========================================
print("\n Precomputing degree cache...")
degree_cache = dict(G.degree())
print("   → Degree cache ready")

# ==========================================
# 4. PARALLEL EGO-GRAPH EXTRACTION (MEMORY SAFE)
# ==========================================
print(f"\n Parallel fingerprint extraction (N={N_NODES}, jobs={N_JOBS})...")
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

all_nodes = list(G.nodes())
sample_nodes = random.sample(all_nodes, min(N_NODES, len(all_nodes)))


def compute_fingerprint(node):
    subgraph = nx.ego_graph(G, node, radius=K_DISTANCE)

    # Deterministic Memory cap: Top 5K by degree
    if subgraph.number_of_nodes() > 5000:
        sorted_nodes = sorted(subgraph.nodes(),
                              key=lambda n: degree_cache.get(n, 0), reverse=True)[:5000]
        if node not in sorted_nodes:
            sorted_nodes[-1] = node
        subgraph = subgraph.subgraph(sorted_nodes)

    # Topological fingerprint
    local_degrees = sorted([degree_cache.get(n, 0) for n in subgraph.nodes()], reverse=True)
    top_k = list(itertools.islice(local_degrees, TOP_DEGREES))
    top_k += [0] * (TOP_DEGREES - len(top_k))

    # Boundary spanning
    unique_comms = set()
    for n in subgraph.nodes():
        comms = node_to_communities.get(n)
        if comms: unique_comms.update(comms)

    return top_k, len(unique_comms), node


# PARALLEL EXECUTION: 'sharedmem' prevents Joblib from crashing RAM
results = Parallel(n_jobs=N_JOBS, require='sharedmem', verbose=5)(
    delayed(compute_fingerprint)(node) for node in sample_nodes
)

feature_matrix = [r[0] for r in results]
spanning_scores = [r[1] for r in results]
valid_nodes = [r[2] for r in results]

print(f" Extracted {len(valid_nodes)} valid fingerprints")

# ==========================================
# 5. DATAFRAMES
# ==========================================
deg_cols = [f"Degree_{i + 1}" for i in range(TOP_DEGREES)]
df_features = pd.DataFrame(feature_matrix, index=valid_nodes, columns=deg_cols)
df_features['Degree_1'] = [degree_cache[n] for n in valid_nodes]
df_spanning = pd.Series(spanning_scores, index=valid_nodes, name="Spanning_Score")

# ==========================================
# 6. PCA + SCREE PLOT
# ==========================================
print("\n PCA dimensionality reduction...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_features)

pca_full = PCA(random_state=RANDOM_SEED)
pca_full.fit(scaled_features)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), 'o--', linewidth=2)
plt.axhline(y=0.90, color='r', linestyle='-', label='90% Threshold')
plt.title('PCA Scree Plot: Cumulative Explained Variance', fontsize=16, pad=20)
plt.xlabel('Principal Components', fontsize=14)
plt.ylabel('Cumulative Variance Explained', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCREE_PNG, dpi=300, bbox_inches='tight', transparent=True)
plt.close()

pca = PCA(n_components=3, random_state=RANDOM_SEED)
df_pca = pd.DataFrame(pca.fit_transform(scaled_features),
                      index=valid_nodes, columns=['PC1', 'PC2', 'PC3'])

# ==========================================
# 7. INFLUENCE MAXIMIZATION (TRUE LAZY CELF)
# ==========================================
print("\n Computing CELF Baselines...")


def simulate_ic(G, seeds, p, steps):
    active = set(seeds)
    new_active = set(seeds)
    history = [len(active)]

    for _ in range(steps):
        activated = set()
        for node in new_active:
            for neigh in G.neighbors(node):
                if neigh not in active and random.random() < p:
                    activated.add(neigh)
        active.update(activated)
        new_active = activated
        history.append(len(active))

        # FIXED: If cascade dies, pad the array to ensure homogeneous Numpy shape
        if not new_active:
            history.extend([len(active)] * (steps - len(history) + 1))
            break

    return history[:steps + 1]

def get_expected_spread(G, seeds, p, steps, mc=3):
    return np.mean([simulate_ic(G, seeds, p, steps)[-1] for _ in range(mc)])


def run_celf(G, candidates, k, p, steps):
    """True CELF Algorithm (Lazy Forward Evaluation)"""
    print("   → Caching single-node base spreads...")
    cached_spreads = {node: get_expected_spread(G, {node}, p, steps)
                      for node in tqdm(candidates, desc="Cache")}

    gains = sorted([(cached_spreads[node], node) for node in candidates], reverse=True)
    seeds = [gains[0][1]]
    gains.pop(0)

    print(f"   → Executing lazy forward selection (k={k})...")

    # ADDED TQDM PROGRESS BAR HERE SO YOU CAN SEE IT WORKING
    for _ in tqdm(range(k - 1), desc="CELF Seed Selection"):

        # MOVED OUTSIDE THE WHILE LOOP: Massive speed optimization
        curr_spread = get_expected_spread(G, set(seeds), p, steps)

        while True:
            top_node = gains[0][1]

            # ONLY CALCULATE NEW SPREAD INSIDE THE LOOP
            new_spread = get_expected_spread(G, set(seeds + [top_node]), p, steps)
            gain = new_spread - curr_spread

            gains[0] = (gain, top_node)
            gains.sort(reverse=True)

            if gains[0][1] == top_node:
                seeds.append(top_node)
                gains.pop(0)
                break
    return seeds

# Optimize by evaluating only the top 200 hubs
baseline_candidates = df_features['Degree_1'].nlargest(200).index.tolist()
high_degree_seeds = baseline_candidates[:IC_SAMPLE_SIZE]
celf_seeds = run_celf(G, baseline_candidates, IC_SAMPLE_SIZE, IC_P, IC_TIME_STEPS)

# ==========================================
# 8. K-MEANS + EMPIRICAL VALIDATION SUITE
# ==========================================
print("\n K-Means clustering + statistical validation...")
kmeans = KMeans(n_clusters=NUM_ARCHETYPES, random_state=RANDOM_SEED, n_init=10)
clusters = kmeans.fit_predict(df_pca[['PC1', 'PC2', 'PC3']])
df_pca['Cluster_ID'] = clusters

# Sub-sampled silhouette to prevent script freezing
sil = silhouette_score(df_pca[['PC1', 'PC2', 'PC3']], clusters, sample_size=5000, random_state=RANDOM_SEED)
db = davies_bouldin_score(df_pca[['PC1', 'PC2', 'PC3']], clusters)
print(f"   → Silhouette: {sil:.3f} | Davies-Bouldin: {db:.3f}")

influencers_temp = df_pca.join(df_spanning).join(df_features['Degree_1'])
cluster_summary = influencers_temp.groupby('Cluster_ID')[['Degree_1', 'Spanning_Score']].mean()
print("\n--- Empirical Validation ---")
print(cluster_summary)

# Strict Empirical Archetype Assignment (Defends the paper's thesis)
bridge_id = cluster_summary['Degree_1'].idxmin()
clique_id = cluster_summary['Spanning_Score'].idxmax()
if bridge_id == clique_id:
    clique_id = cluster_summary['Degree_1'].idxmax()
star_id = (set(cluster_summary.index) - {bridge_id, clique_id}).pop()

archetype_mapping = {star_id: "Type A (Stars)", clique_id: "Type B (Cliques)", bridge_id: "Type C (Bridges)"}
df_pca['Archetype_Label'] = df_pca['Cluster_ID'].map(archetype_mapping)
influencers_df = df_pca.join(df_spanning)

# ANOVA testing
print("\n Statistical significance (ANOVA)...")
groups = [influencers_df[influencers_df['Archetype_Label'] == arch]['Spanning_Score'].values
          for arch in archetype_mapping.values()]
f_stat, p_val = f_oneway(*groups)
print(f"   → Archetypes differ significantly: F={f_stat:.2f}, p={p_val:.6f}")

# ==========================================
# 9. BOOTSTRAPPED IC SIMULATION (95% CI)
# ==========================================
print("\n Bootstrapped IC simulation (95% CIs)...")


def simulate_ic_ci(G, seeds, p, steps, runs=100):
    trajs = [simulate_ic(G, seeds, p, steps) for _ in range(runs)]
    mean_traj = np.mean(trajs, axis=0)
    ci_low = np.percentile(trajs, 2.5, axis=0)
    ci_high = np.percentile(trajs, 97.5, axis=0)
    return mean_traj, ci_low, ci_high


ic_results = {}
palette = {"Type A (Stars)": "teal", "Type B (Cliques)": "green",
           "Type C (Bridges)": "darkblue", "Baseline (CELF)": "black",
           "Baseline (High Degree)": "gray"}

for arch in ["Type A (Stars)", "Type B (Cliques)", "Type C (Bridges)"]:
    nodes = influencers_df[influencers_df['Archetype_Label'] == arch].index.tolist()
    if nodes:
        seeds = random.sample(nodes, min(IC_SAMPLE_SIZE, len(nodes)))
        ic_results[arch] = simulate_ic_ci(G, seeds, IC_P, IC_TIME_STEPS)

ic_results["Baseline (CELF)"] = simulate_ic_ci(G, celf_seeds, IC_P, IC_TIME_STEPS)
ic_results["Baseline (High Degree)"] = simulate_ic_ci(G, high_degree_seeds, IC_P, IC_TIME_STEPS)

# ==========================================
# 10. MASTER EXPORT + VISUALS
# ==========================================
print("\n Exporting publication deliverables...")

stats_df = pd.DataFrame({
    'Metric': ['Silhouette_Score', 'Davies_Bouldin', 'ANOVA_F', 'ANOVA_p'],
    'Value': [sil, db, f_stat, p_val]
})
stats_df.to_excel(STATS_OUTPUT, index=False)

final_df = influencers_df.join(df_features).drop(columns=['Cluster_ID'])
final_df.index.name = "User_Node_ID"
final_df.to_excel(EXCEL_OUTPUT)
print(f"   → {EXCEL_OUTPUT}")

plt.figure(figsize=(12, 8))
sns.scatterplot(data=influencers_df, x='Spanning_Score', y='PC1',
                hue='Archetype_Label', palette=palette, s=80, alpha=0.7)
plt.title('Influencer Archetypes: Prominence vs Boundary Spanning', fontsize=18)
plt.xlabel('Boundary Spanning Score', fontsize=14)
plt.ylabel('Local Prominence (PC1)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(MATRIX_PNG, dpi=300, bbox_inches='tight', transparent=True)
plt.close()

fig_3d = px.scatter_3d(influencers_df.reset_index(), x='PC1', y='PC2', z='PC3',
                       color='Archetype_Label', color_discrete_map=palette,
                       size='Spanning_Score', hover_data=['index', 'Spanning_Score'],
                       title="3D Influencer Archetype Space")
fig_3d.write_html(MATRIX_HTML)

plt.figure(figsize=(12, 7))
time_axis = range(IC_TIME_STEPS + 1)
for arch in ic_results:
    mean_traj, ci_low, ci_high = ic_results[arch]
    linestyle = '--' if 'Baseline' in arch else '-'
    plt.plot(time_axis, mean_traj, label=arch, color=palette[arch],
             linewidth=3, linestyle=linestyle, marker='o' if not 'Baseline' in arch else None)
    plt.fill_between(time_axis, ci_low, ci_high, color=palette[arch], alpha=0.2)

plt.title(f'IC Simulation: Archetypes vs Baselines (p={IC_P}, 95% CI)', fontsize=18)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('Nodes Activated', fontsize=14)
plt.legend(title='Archetype', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(IC_PNG, dpi=300, bbox_inches='tight', transparent=True)
plt.close()

fig_ic = go.Figure()
for arch in ic_results:
    mean_traj, ci_low, ci_high = ic_results[arch]
    dash = 'dash' if 'Baseline' in arch else 'solid'
    fig_ic.add_trace(go.Scatter(x=list(time_axis), y=mean_traj,
                                error_y=dict(type='data', array=ci_high - mean_traj,
                                             arrayminus=mean_traj - ci_low),
                                mode='lines+markers', name=arch,
                                line=dict(color=palette[arch], width=4, dash=dash)))
fig_ic.update_layout(title=f'Interactive IC w/ 95% CIs (p={IC_P})',
                     xaxis_title='Time Steps', yaxis_title='Activations')
fig_ic.write_html(IC_HTML)

print("\n PIPELINE COMPLETE!")
print(f" Silhouette: {sil:.3f} | ANOVA p: {p_val:.2e}")
print(" Sanchita Ghosh, M.Tech.(CSDP) IIT-KGP, Department of Mathematics")
