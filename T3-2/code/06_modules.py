"""6단계(선택): 부담 보정 양의 간선(FDR<0.05)으로 Louvain 모듈 탐지 (SAP 5.5).

networkx 없이 numpy 로 구현 (노드 12개). 무작위 노드 순서 100회 중 모듈성 최대 분할, seed 2026.
출력: results/modules.csv, modules_summary.csv
"""
import numpy as np
import pandas as pd

from common import RESULTS_DIR, SEED, SITE_LABEL, load_sites

sites = load_sites()
K = len(sites)
pm = pd.read_csv(RESULTS_DIR / "pairs_main.csv")
A = np.zeros((K, K))
for _, r in pm[(pm.adj_fdr < 0.05) & (pm.adj_logor > 0)].iterrows():
    i, j = sites.index(r.site_i), sites.index(r.site_j)
    A[i, j] = A[j, i] = r.adj_logor


def modularity(W, comm):
    m2 = W.sum()
    k = W.sum(1)
    same = comm[:, None] == comm[None, :]
    return ((W - np.outer(k, k) / m2) * same).sum() / m2


def local_moves(W, rng):
    n = len(W)
    comm = np.arange(n)
    k = W.sum(1)
    m2 = W.sum()
    improved = True
    while improved:
        improved = False
        for v in rng.permutation(n):
            cur = comm[v]
            best_c, best_gain = cur, 0.0
            tot = {c: k[comm == c].sum() - (k[v] if c == cur else 0) for c in np.unique(comm)}
            k_in = {c: W[v, (comm == c) & (np.arange(n) != v)].sum() for c in np.unique(comm)}
            base = k_in[cur] - tot[cur] * k[v] / m2
            for c in np.unique(comm):
                gain = (k_in[c] - tot[c] * k[v] / m2) - base
                if gain > best_gain + 1e-12:
                    best_c, best_gain = c, gain
            if best_c != cur:
                comm[v] = best_c
                improved = True
    _, comm = np.unique(comm, return_inverse=True)
    return comm


def louvain(W, rng):
    node_comm = np.arange(len(W))
    Wc = W.copy()
    while True:
        c = local_moves(Wc, rng)
        if len(np.unique(c)) == len(Wc):
            break
        node_comm = c[node_comm]
        nc = c.max() + 1
        Wn = np.zeros((nc, nc))
        for a in range(nc):
            for b in range(nc):
                Wn[a, b] = Wc[np.ix_(c == a, c == b)].sum()
        Wc = Wn
    return node_comm


rng = np.random.default_rng(SEED)
isolated = A.sum(1) == 0
best = (-np.inf, None)
sols = {}
for _ in range(100):
    comm = louvain(A, rng)
    q = modularity(A, comm)
    key = tuple(pd.factorize(comm)[0])
    sols[key] = sols.get(key, 0) + 1
    if q > best[0] + 1e-12:
        best = (q, comm)
q, comm = best
out = pd.DataFrame({"site": sites, "label": [SITE_LABEL[s] for s in sites], "module": comm + 1,
                    "degree_pos": (A > 0).sum(1), "strength_pos": A.sum(1), "isolated": isolated})
out.to_csv(RESULTS_DIR / "modules.csv", index=False)
pd.DataFrame([dict(modularity=q, n_modules=len(np.unique(comm)), n_edges=int((A > 0).sum() / 2),
                   distinct_solutions=len(sols), best_solution_freq=sols[tuple(pd.factorize(comm)[0])])]).to_csv(
    RESULTS_DIR / "modules_summary.csv", index=False)
print("Q =", round(q, 3), "| 서로 다른 해", len(sols))
print(out.sort_values(["module", "strength_pos"], ascending=[True, False]).round(2).to_string(index=False))
