"""
确定性冲突判定与符号聚合（创新点 2）

流水线：启发式预标注 → 逐维规范键匹配（canonical_match，零 LLM，两档）→ 符号聚合
（三态边，保守化规则见 _aggregate：归因冲突∧行为支持，或时间冲突∧归因支持 → −1；
无冲突且任意维支持 → +1；全 none → 不建边）→ 画像/可靠性边权推导 →
查询锚点 q↔证据按同机制建边。

输出 awm.signed_graph_edges：confirmed 边（sign=±1，含冲突维度、边权等；含 q↔证据边）。
全 none 的配对此处不建边，留待 core/transitivity.py 传递性补全。
"""
from itertools import combinations
import numpy as np

from src_new.core.awm import AWM, GraphEdge
from src_new.core.canonical_match import match_all
from src_new.core.signed_graph import QUERY_NODE
from src_new.utils.settings import get_pipeline_param


# 冲突维度优先级（记录最高优先级冲突维时用；behavior/campaign 不产 conflict，置末位）
_CONFLICT_DIM_PRIORITY = ["attribution", "timeline", "behavior", "campaign"]


def _is_same_source(ev_a, ev_b) -> bool:
    """启发式预标注：同源且发布时间相近 → consistent（+1, confirmed）"""
    if not ev_a.source or not ev_b.source:
        return False
    if ev_a.source.strip().lower() != ev_b.source.strip().lower():
        return False
    # 发布日期相近（同年或都缺）
    ya = (ev_a.publish_date or "")[:4]
    yb = (ev_b.publish_date or "")[:4]
    if ya and yb:
        try:
            return abs(int(ya) - int(yb)) <= 1
        except ValueError:
            return True
    return True


def compute_profile_modulation(similarity: float) -> float:
    """β = c + (1−c)·cos(p_i,p_j)；与 sign 无关"""
    c = get_pipeline_param("signed_graph.modulation.base_weight_c", 0.5)
    return c + (1 - c) * similarity


def _cosine(a: list, b: list) -> float:
    if not a or not b:
        return 0.0
    va, vb = np.array(a), np.array(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _aggregate(dim_verdicts: dict) -> tuple[int, str, str]:
    """三态符号聚合（方案 3.2.3，带 RAG 场景下的冲突保守化）。返回 (sign, edge_class, conflict_dimension)。

    保守化原则（修正"归因不相交即 conflict"的伪冲突）：在多源 RAG 检索场景下，检索回的
    相关报告天然来自不同组织/时期，它们归因或活跃区间不同是常态而非"冲突"。仅当能判定为
    "同一活动的真实矛盾"时才判 conflict：
      ① 归因不同(conflict) 且 行为重叠(support) → 真冲突（同行为被归给不同组织）→ sign=-1
      ② 时间矛盾(conflict) 且 同组织(attribution support) → 真冲突（同组织互斥活跃区间）→ sign=-1
      ③ 无上述真冲突、任意维 support → sign=+1
      ④ 四维全 none → sign=0, none（不建边，转传递性补全）
    其余归因/时间线的不相交不再单独构成 conflict，避免伪冲突虚高 Φ、碎片化分区。
    """
    attr = dim_verdicts.get("attribution")
    tl = dim_verdicts.get("timeline")
    beh = dim_verdicts.get("behavior")

    attr_conflict = (attr == "conflict" and beh == "support")
    tl_conflict = (tl == "conflict" and attr == "support")
    has_support = any(v == "support" for v in dim_verdicts.values())

    if attr_conflict or tl_conflict:
        dim = "attribution" if attr_conflict else "timeline"
        return -1, "confirmed", dim
    if has_support:
        return 1, "confirmed", ""
    return 0, "none", ""


def build_edges(awm: AWM) -> None:
    """对证据集两两判定 + 查询锚点 q↔证据判定，写回 signed_graph_edges（原地修改 awm）。

    confirmed 边(sign=±1)入 signed_graph_edges；none 配对不建边(留传递性补全)。
    同源预标注对跳过匹配直接判 consistent。q↔证据走与证据-证据相同的规范键匹配(3.2.4)。
    """
    awm.signed_graph_edges = []

    profile_by_doc = {p.doc_id: p for p in awm.behavior_profiles}
    evs = awm.evidence_set

    # ① 证据-证据两两判定
    for ev_a, ev_b in combinations(evs, 2):
        a, b = ev_a.doc_id, ev_b.doc_id
        ck_a = (profile_by_doc.get(a).canonical_keys if profile_by_doc.get(a) else {}) or {}
        ck_b = (profile_by_doc.get(b).canonical_keys if profile_by_doc.get(b) else {}) or {}

        # 画像调制
        pa, pb = profile_by_doc.get(a), profile_by_doc.get(b)
        cos = _cosine(pa.embedding if pa else [], pb.embedding if pb else [])
        beta = compute_profile_modulation(cos)
        rel = (ev_a.reliability + ev_b.reliability) / 2.0

        # 启发式预标注：同源且日期相近 → consistent
        if _is_same_source(ev_a, ev_b):
            awm.signed_graph_edges.append(GraphEdge(
                source=a, target=b, sign=1, confidence=1.0,
                edge_class="confirmed", conflict_dimension="",
                profile_modulation=beta, reliability_factor=rel,
                dim_verdicts={"pre_annotated": "consistent"}))
            continue

        # 逐维规范键匹配（零 LLM，两档）+ 符号聚合（三态边）
        dim_verdicts = match_all(ck_a, ck_b)
        sign, edge_class, conflict_dim = _aggregate(dim_verdicts)
        if edge_class == "confirmed":
            awm.signed_graph_edges.append(GraphEdge(
                source=a, target=b, sign=sign, confidence=0.9,
                edge_class="confirmed", conflict_dimension=conflict_dim,
                profile_modulation=beta, reliability_factor=rel,
                dim_verdicts=dim_verdicts))
        # edge_class == "none"：不建边，留传递性补全

    # ② 查询锚点 q↔证据（与证据-证据同机制，3.2.4）
    build_query_edges(awm)


def build_query_edges(awm: AWM) -> None:
    """查询锚点 q 与每条证据按规范键确定性匹配建边（3.2.4「与证据-证据边同机制」）。
    sign 由匹配给出（可为 conflict）；权重经画像调制。neutral(none) 不建边。"""
    if awm.query_profile is None:
        return
    qk = awm.query_profile.canonical_keys or {}
    if not qk:
        return
    q_emb = awm.query_profile.embedding or []
    for ev in awm.evidence_set:
        p = awm.get_profile_by_doc_id(ev.doc_id)
        ck = (p.canonical_keys if p else {}) or {}
        if not ck:
            continue
        dim_verdicts = match_all(qk, ck)
        sign, edge_class, conflict_dim = _aggregate(dim_verdicts)
        if edge_class != "confirmed":
            continue  # neutral → 不建边
        cos = _cosine(q_emb, p.embedding if p else [])
        beta = compute_profile_modulation(cos)
        rel = (1.0 + ev.reliability) / 2.0  # query 可靠性视为 1.0
        awm.signed_graph_edges.append(GraphEdge(
            source=QUERY_NODE, target=ev.doc_id, sign=sign, confidence=0.9,
            edge_class="confirmed", conflict_dimension=conflict_dim,
            profile_modulation=beta, reliability_factor=rel, dim_verdicts=dim_verdicts))
