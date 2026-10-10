"""
传递性推理补全（创新点 2，3.2.5）

仅对 edge_class==none（四维全 none、未建边）的配对执行；不为预标注或已判定的配对
重复推断。三条传递规则统一为符号乘法 sign(A−B)×sign(B−C)。

多路径加权聚合（吸收矛盾推断）：
    对每个共同邻居 Bᵢ：路径符号 sᵢ = sign(A−Bᵢ)×sign(Bᵢ−C)
                       路径权重 wᵢ = √(conf(A−Bᵢ)·conf(Bᵢ−C)) × decay
    P = Σ_{sᵢ=+1} wᵢ，N = Σ_{sᵢ=−1} wᵢ
    净符号 = sign(P−N)
    一致性 = |P−N|/(P+N) ∈ [0,1]
    净置信度 = mean(wᵢ) × 一致性
一致性过低则不建边（矛盾推断被吸收）。
"""
from src_new.core.awm import AWM, GraphEdge
from src_new.utils.settings import get_pipeline_param


def _signed_lookup(awm: AWM):
    """构建 (a,b)->sign 的对称查表，仅 confirmed 边参与（sign=±1）。"""
    table = {}
    for e in awm.signed_graph_edges:
        if e.edge_class != "confirmed" or e.sign == 0:
            continue
        table[(e.source, e.target)] = e.sign
        table[(e.target, e.source)] = e.sign
    return table


def _conf_lookup(awm: AWM):
    """(a,b)->confidence 对称查表（confirmed 边；w 经 signed_graph.edge_weight 含 ι）。"""
    from src_new.core.signed_graph import edge_weight
    table = {}
    for e in awm.signed_graph_edges:
        if e.edge_class != "confirmed":
            continue
        c = edge_weight(e)
        table[(e.source, e.target)] = c
        table[(e.target, e.source)] = c
    return table


def infer_neutral_pairs(awm: AWM) -> None:
    """对 none 配对做传递性补全，结果以 confirmed 边追加进 signed_graph_edges（原地修改）。"""
    sign_tbl = _signed_lookup(awm)
    conf_tbl = _conf_lookup(awm)
    decay = get_pipeline_param("adaptive.decay", 0.85)
    min_consistency = get_pipeline_param("refutation.min_consistency_build_edge", 0.3)

    doc_ids = [ev.doc_id for ev in awm.evidence_set]
    # 已有 confirmed 判定的配对集合（none 配对无边，留待补全）
    existing = set()
    for e in awm.signed_graph_edges:
        existing.add(frozenset((e.source, e.target)))

    # 候选：无任何边的配对（含原 none 配对）
    new_edges = []
    for a, c in _pairs(doc_ids):
        if frozenset((a, c)) in existing:
            continue
        # 枚举全部共同邻居 Bᵢ：与 a、c 均有 confirmed 符号边
        votes = []  # [(sᵢ, wᵢ)]
        for b in doc_ids:
            if b == a or b == c:
                continue
            sa = sign_tbl.get((a, b))
            sc = sign_tbl.get((b, c))
            if sa is None or sc is None or sa == 0 or sc == 0:
                continue
            ca = conf_tbl.get((a, b), 0.5)
            cc = conf_tbl.get((b, c), 0.5)
            w = ((ca * cc) ** 0.5) * decay
            votes.append((sa * sc, w))
        if not votes:
            continue

        P = sum(w for s, w in votes if s > 0)
        N = sum(w for s, w in votes if s < 0)
        total = P + N
        if total <= 0:
            continue
        consistency = abs(P - N) / total
        if consistency < min_consistency:
            continue  # 矛盾/弱一致性 → 不建边，吸收
        net_sign = 1 if P > N else -1
        net_conf = (sum(w for _, w in votes) / len(votes)) * consistency

        new_edges.append(GraphEdge(
            source=a, target=c, sign=net_sign, confidence=float(net_conf),
            edge_class="confirmed", conflict_dimension="",
            profile_modulation=1.0, reliability_factor=1.0,
            dim_verdicts={}, inferred=True, transitivity_consistency=float(consistency),
        ))

    awm.signed_graph_edges.extend(new_edges)


def _pairs(doc_ids):
    """无序对生成器"""
    n = len(doc_ids)
    for i in range(n):
        for j in range(i + 1, n):
            yield doc_ids[i], doc_ids[j]
