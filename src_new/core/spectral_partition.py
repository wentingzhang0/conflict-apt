"""
谱分区（创新点 2，3.2.4）

构造签名拉普拉斯矩阵 L̄ = D̄ − A_s（D̄ 取边权绝对值，Kunegis 等 2010）：所有 confirmed 边
（含证据-证据、q↔证据、传递性补全）按 sign·weight 全权计入。取最小的 k 个特征值对应的特征向量
（保留体现图不平衡的负特征值），应用 k-means，k 由 eigengap 启发式确定。
"""
import numpy as np
from sklearn.cluster import KMeans

from src_new.core.awm import AWM
from src_new.core.signed_graph import QUERY_NODE, edge_weight
from src_new.utils.settings import get_pipeline_param


def _node_order(awm: AWM):
    nodes = [QUERY_NODE] + [ev.doc_id for ev in awm.evidence_set]
    return nodes, {nd: i for i, nd in enumerate(nodes)}


def signed_laplacian(awm: AWM) -> tuple[np.ndarray, list[str]]:
    """构造签名拉普拉斯 L̄ = D̄ − A_s。A_s[i,j] = sign·weight（confirmed 边全权）。
    q↔证据边来自 signed_graph_edges（由 conflict.build_query_edges 产出，3.2.4）。"""
    nodes, idx = _node_order(awm)
    n = len(nodes)
    A = np.zeros((n, n))

    # confirmed 边（含 q↔证据、传递性补全），全权 sign·weight 计入
    # （w 经 signed_graph.edge_weight：REWEIGH 后含独立性折扣 ι，更新方案 §3.3）
    for e in awm.signed_graph_edges:
        if e.edge_class != "confirmed":
            continue
        if e.source not in idx or e.target not in idx:
            continue
        w = edge_weight(e)
        i, j = idx[e.source], idx[e.target]
        val = e.sign * w
        A[i][j] += val
        A[j][i] += val

    D = np.diag(np.abs(A).sum(axis=1))
    L = D - A
    return L, nodes


def _eigengap_k(eigvals: np.ndarray, k_max: int = 4, k_min: int = 1) -> int:
    """eigengap 启发式选 k：在 [k_min, k_max] 内取使相邻特征值差 |λ_k − λ_{k−1}| 最大的 k。"""
    eigvals = np.sort(eigvals)
    upper = min(k_max, len(eigvals) - 1)
    if upper < k_min:
        return k_min
    best_k, best_gap = k_min, -1.0
    for k in range(k_min, upper + 1):
        gap = abs(eigvals[k] - eigvals[k - 1])
        if gap > best_gap:
            best_gap, best_k = gap, k
    return best_k


def spectral_partition(awm: AWM) -> dict:
    """返回 {k, clusters:{cluster_id:[doc_id,...]}, labels}（仅证据节点，不含 query 锚点）。"""
    ev_ids = [ev.doc_id for ev in awm.evidence_set]
    if len(ev_ids) <= 1:
        return {"k": 1, "clusters": {0: list(ev_ids)}, "labels": {nd: 0 for nd in ev_ids}}

    L, nodes = signed_laplacian(awm)
    idx = {nd: i for i, nd in enumerate(nodes)}
    eigvals, eigvecs = np.linalg.eigh(L)

    k = _eigengap_k(eigvals, k_max=min(4, len(ev_ids)))
    k = max(1, min(k, len(ev_ids)))
    # 取最小的 k 个特征值对应特征向量
    feats = eigvecs[:, :k]

    # 仅对证据节点聚类（排除 query 锚点；q 的 faction 由其谱嵌入自然落入，但不进假设聚合）
    ev_rows = np.array([idx[nd] for nd in ev_ids])
    X = feats[ev_rows]

    if k == 1:
        labels = {nd: 0 for nd in ev_ids}
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        labels = {nd: int(l) for nd, l in zip(ev_ids, km.labels_)}

    clusters = {}
    for nd, l in labels.items():
        clusters.setdefault(l, []).append(nd)
    return {"k": k, "clusters": clusters, "labels": labels}
