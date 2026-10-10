"""
查询锚点常量与 h_graph 图量化强度（创新点 3，3.3）

QUERY_NODE 为查询锚点节点标识；h_graph = |C_h|·ρ_h − λ·κ_h（ρ 簇内 support 加权密度、
κ 落在关键支撑集 S(h) 的 conflict 总权）。带符号图的边全部来自 awm.signed_graph_edges
（confirmed，含证据-证据、q↔证据、传递性补全）；q↔证据边由 conflict.build_query_edges 按规范键
匹配产出（可为 conflict，与证据-证据同机制，3.2.4）。
"""

from src_new.core.awm import AWM
from src_new.utils.settings import get_pipeline_param

QUERY_NODE = "query"


def edge_weight(e) -> float:
    """统一有效边权 w(e) = confidence × profile_modulation × reliability_factor × ι。

    创新点 4（更新方案 §3.3）：ι 为独立性折扣（REWEIGH 阶段写入 independence_factor；
    未启用欺骗感知时该字段缺省 1.0，w 与会议版逐字节等价）。谱分区、h_graph、S(h)
    遴选、传递性路径权重一律经此函数取权，保证重加权后全链路自动生效。
    """
    iota = getattr(e, "independence_factor", 1.0)
    if iota is None:
        iota = 1.0
    return e.confidence * e.profile_modulation * e.reliability_factor * iota


def key_support_set(awm: AWM, hypothesis, top_n: int = 3) -> set:
    """S(h)：h 所在簇内 support(+1) 边里权重最高者所连证据节点（方案 3.3 关键支撑集）。
    q 不属于任何 hypothesis.evidence_ids，故天然不进 S(h)。"""
    doc_ids = set(hypothesis.evidence_ids)
    support = {}
    for e in awm.signed_graph_edges:
        if e.edge_class != "confirmed" or e.sign != 1:
            continue
        if e.source in doc_ids and e.target in doc_ids:
            w = edge_weight(e)
            for eid in (e.source, e.target):
                if eid in doc_ids:
                    support[eid] = max(support.get(eid, 0.0), w)
    top = sorted(support.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return {eid for eid, _ in top}


def graph_strength_for_hypothesis(awm: AWM, hypothesis) -> dict:
    """归一化 h_graph ∈ [0,1]（与 llm_score 同量纲，供混合评分）。

    原(未归一化)公式 |C_h|·ρ_h − λ·κ_h 量纲不受控：κ 未归一化可使 h_graph 取很大
    负值，与 llm_score∈[0,1] 混合后完全主导/扭曲假设排序（多证据正确假设被压垮）。
    现改为三项归一化复合：
      size_score = n/(n+5)             规模奖励 ∈(0,1)
      ρ_h        = 簇内 support 密度    ∈[0,1]（n<2 时取中性 0.5）
      κ_norm     = κ/(κ+support_on_S)  关键支撑集上冲突占比 ∈[0,1]
      h_graph    = size_score · ρ_h · (1 − λ·κ_norm)   ∈[0,1]
    多证据+高内聚+低冲突压力 → 接近 1；高冲突压力或单证据 → 接近 0。
    """
    doc_ids = set(hypothesis.evidence_ids)
    n = len(doc_ids)

    # ρ_h：簇内 support 边加权密度（∈[0,1]：support_w_sum ≤ pairs·w_max，w_max≈1）
    support_w_sum = 0.0
    for e in awm.signed_graph_edges:
        if e.edge_class != "confirmed" or e.sign != 1:
            continue
        if e.source in doc_ids and e.target in doc_ids:
            support_w_sum += edge_weight(e)
    pairs = n * (n - 1) / 2.0
    rho = float(support_w_sum / pairs) if pairs > 0 else 0.5  # 单证据无密度，取中性

    # S(h) 节点上的 support 总权 与 conflict 总权（κ）
    s_h = key_support_set(awm, hypothesis)
    support_on_s = 0.0
    kappa = 0.0
    for e in awm.signed_graph_edges:
        if e.edge_class != "confirmed":
            continue
        if e.source not in s_h and e.target not in s_h:
            continue
        w = edge_weight(e)
        if e.sign == 1:
            support_on_s += w
        else:  # sign == -1
            kappa += w

    lam = float(get_pipeline_param("h_graph.pressure_lambda", 0.5))
    kappa_norm = kappa / (kappa + support_on_s + 1e-6)  # ∈[0,1]
    size_score = n / (n + 5.0)                           # ∈(0,1)
    h_graph_norm = size_score * rho * (1.0 - lam * kappa_norm)  # ∈[0,1]
    h_graph_raw = n * rho - lam * kappa                 # 原(未归一化)值，仅作可解释参考
    return {"rho": rho, "kappa": kappa, "kappa_norm": kappa_norm,
            "n_evidence": n, "s_h_size": len(s_h), "lambda": lam,
            "h_graph": float(h_graph_norm), "h_graph_raw": float(h_graph_raw)}

