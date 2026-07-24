# -*- coding: utf-8 -*-
# Pipeline: Bootstrap -> PCA -> RandomForest -> Time-ordered BN Structure Learning (stability via bootstrap)
# Charts: matplotlib only, one figure per plot, no explicit colors.

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

# -------------------------
# User parameters
# -------------------------
FILE_PATH = "AllData.xlsx"         # Pfad zur Excel-Datei anpassen
SHEET_NAME = "Sheet2"

# -------------------------
# Out-Verzeichnis definieren
# -------------------------
OUT_DIR = "results_bn"
os.makedirs(OUT_DIR, exist_ok=True)

PCA_CURVE = os.path.join(OUT_DIR, "pca_cumulative_variance.png")
RF_PLOT = os.path.join(OUT_DIR, "rf_feature_importances.png")
HEATMAP_PNG = os.path.join(OUT_DIR, "edge_frequency_heatmap_bn_integrated.png")
BN_GRAPH_PNG = os.path.join(OUT_DIR, "bn_graph_integrated.png")
EDGES_CSV = os.path.join(OUT_DIR, "bn_edges_selected.csv")
PATHS_CSV = os.path.join(OUT_DIR, "bn_paths_to_transsmittance.csv")
RF_TABLE_CSV = os.path.join(OUT_DIR, "rf_feature_importances_table.csv")
PCA_TABLE_CSV = os.path.join(OUT_DIR, "pca_explained_variance_table.csv")

# Parameter
N_BOOTSTRAP_ROWS = 100         
JITTER_SCALE = 0.01            
RF_N_EST = 300                 
TOP_K_RF = 6                   
BN_B = 80                      
MAX_PARENTS = 2                
FREQ_THRESHOLD = 0.5           

# -------------------------
# 1) Daten einlesen & Spalten prüfen
# -------------------------
if not os.path.exists(FILE_PATH):
    print(f"[ERROR] Datei nicht gefunden: {FILE_PATH}. Bitte Pfad prüfen.")
    exit()

df_raw = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME).select_dtypes(include=[np.number]).copy()

# Prüfe Transmissions-Spaltennamen
possible_targets = [c for c in df_raw.columns if "Trans" in c or "trans" in c]
if possible_targets:
    TARGET = possible_targets[0]
    print(f"[INFO] Zielspalte automatisch identifiziert: '{TARGET}'")
else:
    TARGET = "Transsmittance(%*nm)(380-780)"

# -------------------------
# 2) Zeitorientierte Gruppen definieren
# -------------------------
g1 = ["Concentration", "concentration"]
g2 = ["porosity_px", "porosity"] 
g3 = [
    "std_pore_diam_px",
    "mean_pore_circularity",
    "mean_pore_aspect_ratio",
    "mean_pore_solidity",
    "skeleton_length_density",
    "skeleton_branch_points",
    "skeleton_end_points",
    "fractal_dimension",
]
g4 = [
    "density(mg/mm^3)",
    "Volume Shrinkage",
    "Modulus(MPa)",
    "STDEV.S",
    TARGET
]

groups_all = [g1, g2, g3, g4]
exist = set(df_raw.columns)
groups = [[c for c in grp if c in exist] for grp in groups_all]
ordered_all = [c for grp in groups for c in grp]
df = df_raw[ordered_all].dropna().copy()

# -------------------------
# 3) Bootstrap Data Augmentation
# -------------------------
df_aug = resample(df, replace=True, n_samples=N_BOOTSTRAP_ROWS, random_state=2025)
for col in df_aug.columns:
    std = df[col].std()
    if pd.notna(std) and std > 0:
        df_aug[col] += np.random.normal(0, JITTER_SCALE * std, size=len(df_aug))

# -------------------------
# 4) PCA Analyse
# -------------------------
feature_cols = [c for c in df_aug.columns if c != TARGET]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_aug[feature_cols].values)

p = X_scaled.shape[1]
pca = PCA(n_components=min(10, p))
X_pca = pca.fit_transform(X_scaled)
exp_ratio = pca.explained_variance_ratio_
cum_exp = np.cumsum(exp_ratio)
k90 = int(np.searchsorted(cum_exp, 0.90) + 1)
k90 = max(1, min(k90, len(exp_ratio)))

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_exp) + 1), cum_exp, marker='o')
plt.axhline(0.90, linestyle='--')
plt.axvline(k90, linestyle='--')
plt.title("PCA Cumulative Explained Variance (Bootstrap n=100)")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.tight_layout()
plt.savefig(PCA_CURVE, dpi=160)
plt.close()

pca_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(exp_ratio))],
    "ExplainedVariance": exp_ratio,
    "Cumulative": cum_exp
})
pca_df.to_csv(PCA_TABLE_CSV, index=False)

# -------------------------
# 5) Random Forest Feature Importance
# -------------------------
y = df_aug[TARGET].values
rf = RandomForestRegressor(n_estimators=RF_N_EST, random_state=42)
rf.fit(X_scaled, y)
fi = pd.DataFrame({"Feature": feature_cols, "Importance": rf.feature_importances_}) \
       .sort_values("Importance", ascending=False)
fi.to_csv(RF_TABLE_CSV, index=False)

topn = min(10, len(feature_cols))
plt.figure(figsize=(8, 5))
plt.barh(range(topn), fi["Importance"].values[:topn])
plt.yticks(ticks=range(topn), labels=fi["Feature"].values[:topn])
plt.gca().invert_yaxis()
plt.title("Random Forest Feature Importances (Top 10)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(RF_PLOT, dpi=160)
plt.close()

# Wähle Variablen für das Bayessche Netz
top_features = fi["Feature"].values[:min(TOP_K_RF, len(feature_cols))].tolist()
anchors = [c for c in ["Concentration", "concentration", "porosity_px", "porosity", TARGET] if c in df_aug.columns]
for a in anchors:
    if a not in top_features:
        top_features.append(a)

selected_vars = [v for v in ordered_all if v in top_features or v == TARGET]

groups_sel = []
for grp in groups:
    kept = [c for c in grp if c in selected_vars]
    if kept:
        groups_sel.append(kept)
ordered_vars = [c for grp in groups_sel for c in grp]
df_sel = df_aug[ordered_vars].copy()

# -------------------------
# 6) Zeitorientiertes Strukturlernen (Lasso + Bootstrap)
# -------------------------
group_index = {name: i for i, grp in enumerate(groups_sel) for name in grp}
allowed_parents = {var: list(itertools.chain.from_iterable(groups_sel[:group_index[var]])) for var in ordered_vars}

def learn_edges_once(data: pd.DataFrame, allowed_parents: dict, max_parents: int = 2, seed: int = None):
    edges = []
    Xstd = (data - data.mean()) / (data.std(ddof=0) + 1e-12)
    for child in ordered_vars:
        parents = allowed_parents.get(child, [])
        if not parents:
            continue
        X = Xstd[parents].values
        y = Xstd[child].values
        if X.shape[0] < 5 or X.shape[1] < 1:
            continue
        try:
            model = LassoCV(cv=min(5, max(2, X.shape[0] // 5)), random_state=seed).fit(X, y)
            coefs = model.coef_
        except Exception:
            XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
            Xty = X.T @ y
            coefs = np.linalg.solve(XtX, Xty)
        idx_sorted = np.argsort(np.abs(coefs))[::-1]
        sel = [i for i in idx_sorted if np.abs(coefs[i]) > 1e-10][:max_parents]
        for i in sel:
            edges.append((parents[i], child))
    return edges

rng = np.random.default_rng(12345)
edge_counts = {(a, b): 0 for a in ordered_vars for b in ordered_vars if a != b}

for b in range(BN_B):
    boot = resample(df_sel, replace=True, n_samples=len(df_sel), random_state=rng.integers(1, 1_000_000))
    edges_b = learn_edges_once(boot, allowed_parents, max_parents=MAX_PARENTS, seed=rng.integers(1, 1_000_000))
    for e in edges_b:
        edge_counts[e] = edge_counts.get(e, 0) + 1

n = len(ordered_vars)
var_index = {v: i for i, v in enumerate(ordered_vars)}
freq_mat = np.zeros((n, n))
edges_list = []
for (a, b), cnt in edge_counts.items():
    f = cnt / BN_B
    if a in var_index and b in var_index:
        i, j = var_index[a], var_index[b]
        freq_mat[i, j] = f
        if f > 0:
            edges_list.append({"Parent": a, "Child": b, "Frequency": f})

edges_df = pd.DataFrame(edges_list).sort_values(["Frequency", "Parent", "Child"], ascending=[False, True, True])
edges_df.to_csv(EDGES_CSV, index=False)

# -------------------------
# 7) Visualisierung Heatmap & DAG
# -------------------------
plt.figure(figsize=(8, 6))
plt.imshow(freq_mat, aspect='auto')
plt.xticks(ticks=np.arange(n), labels=ordered_vars, rotation=90)
plt.yticks(ticks=np.arange(n), labels=ordered_vars)
plt.title("Edge Frequency Heatmap (Time-ordered BN)")
plt.colorbar()
plt.tight_layout()
plt.savefig(HEATMAP_PNG, dpi=160)
plt.close()

final_edges = [(ordered_vars[i], ordered_vars[j]) for i in range(n) for j in range(n)
               if i != j and freq_mat[i, j] >= FREQ_THRESHOLD]

G = nx.DiGraph()
G.add_nodes_from(ordered_vars)
for u, v in final_edges:
    G.add_edge(u, v, weight=freq_mat[var_index[u], var_index[v]])

pos = nx.spring_layout(G, seed=7)
plt.figure(figsize=(9, 7))
node_sizes = [1800 if node == TARGET else 1200 for node in G.nodes()]
edge_widths = [3 * G[u][v]['weight'] for u, v in G.edges()]
nx.draw(G, pos, with_labels=True, node_size=node_sizes, width=edge_widths, arrows=True)
plt.title("Learned BN Structure (Transmittance Emphasized)")
plt.tight_layout()
plt.savefig(BN_GRAPH_PNG, dpi=160)
plt.close()

# -------------------------
# 8) Kausale Pfade berechnen
# -------------------------
adj = {}
for a, b in final_edges:
    adj.setdefault(a, []).append(b)

def find_paths(start, target, max_len=8):
    paths = []
    def dfs(node, path):
        if len(path) > max_len:
            return
        if node == target and len(path) > 1:
            paths.append(list(path))
            return
        for nxt in adj.get(node, []):
            if nxt in path:
                continue
            dfs(nxt, path + [nxt])
    dfs(start, [start])
    return paths

start_node = ordered_vars[0]
paths_to_T = find_paths(start_node, TARGET, max_len=8)
pd.DataFrame({"Path": [" -> ".join(p) for p in paths_to_T]}).to_csv(PATHS_CSV, index=False)

print("Processing complete! Saved files to folder:", OUT_DIR)
