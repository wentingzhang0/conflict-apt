"""
创新点 4（v2）：欺骗感知的可信度审查 —— CORE → ADJUDGE → QUARANTINE

v1（结构异常四特征 FLAG→REWEIGH）经真实投毒实验证伪：假旗报告与全图呈
support 连接（归因别名被 LLM 混入查询组织、时间线重叠），Φ≈0、k=1，
阵营内敌意/挫折贡献/行为疏离全部为零；且 k=1 时"只降权"无假设可重排，
poison 文本仍进假设与报告的 prompt。失败复盘见 improve_reference/更新方案.md §11。

v2 机制（本模块为确定性部分）：
- CORE-1 近重复互证检测：**跨来源**（source 不同）且内容嵌入 cos>τ_dup 的文档对。
  攻击者单枪匹马伪造"多源印证"时，多篇假报告出自同一生成器——内容近重复是
  协同假旗的结构指纹。同报告多 chunk（Weaviate 分块产物）有相同 source，
  预标注即排除，不误伤（实测语料 1413 个多 chunk 报告全部同 source）。
- CORE-2 归因-内容分歧检测（v3，确定性，零 LLM）：假旗的不变量是"内容必须复制
  真实事件才可检索、归因必须嫁祸替罪组织才成攻击"——内容锚在事件簇、归因锚在
  替罪羊簇，双向分歧可被嵌入几何直接观测。三个子信号：
  C1 盟友分歧：检索集内同归因跨源盟友存在时，内容与盟友质心 cos<τ_div；
  C2 组织史分歧：内容与该组织在干净语料的历史内容质心 μ_A cos<τ_actor
  （μ_A=BM25(组织名)于 APT class 取文嵌入质心，排除 poison）；
  C3 同归因共发带：跨来源同归因对 cos∈[τ_band_lo, τ_dup)——自适应攻击者把
  近重复压到 τ_dup 下，共发指纹仍在。补齐 CORE-1 的两个逃逸面（单篇注入、
  风格分化）与 ⑤ 的骨干校准依赖（四骨干实证：定罪率 23%~93% 随对齐质量波动）。
- ADJUDGE（⑤）语义豁免（credibility_agent）：骨干 LLM 只负责找无害解释
  （辛迪加转载/同系列/正当学术分歧/新战役），verdict!=credible 即隔离——
  定罪证据全部来自确定性层，语义层不承担定罪职责。
- QUARANTINE 隔离：deceptive 节点从证据组装（假设生成/报告/ABP 摘要）中剔除，
  图与权重保留（可解释、可回滚）。降权救不了 k=1 共识场景—— poisoning 的
  伤害路径是 prompt 组装，隔离必须发生在组装处。

确定性纪律不变：CORE/QUARANTINE 零 LLM 可单测；⑤ 只出类别与理由不出数字。
"""
from __future__ import annotations

import numpy as np

from src_new.core.awm import AWM, GraphEdge, Evidence
from src_new.core.signed_graph import QUERY_NODE
from src_new.utils.settings import get_pipeline_param

_EPS = 1e-9


def _cfg(key: str, default):
    return get_pipeline_param(f"deception.{key}", default)


# ============================================================
# CORE-1：跨来源近重复互证检测（确定性，零 LLM）
# ============================================================
def _content_cosine(awm: AWM, a: str, b: str) -> float:
    """两条证据的内容余弦。优先 content_embedding（正文嵌入，骨干无关——ABP 描述
    嵌入受抽取 LLM 风格影响，同一攻击语料在不同骨干下 cos 0.95→0.72），
    缺失时回退 ABP 描述嵌入。"""
    ea, eb = awm.get_evidence_by_doc_id(a), awm.get_evidence_by_doc_id(b)
    va = (ea.content_embedding if ea else []) or []
    vb = (eb.content_embedding if eb else []) or []
    if not va or not vb:
        pa, pb = awm.get_profile_by_doc_id(a), awm.get_profile_by_doc_id(b)
        va = (pa.embedding if pa else []) or []
        vb = (pb.embedding if pb else []) or []
    if not va or not vb:
        return 0.0
    v1, v2 = np.array(va, dtype=float), np.array(vb, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _same_source(evs: dict, a: str, b: str) -> bool:
    """与 conflict._is_same_source 同判据：同 report_title 且年份相近 → 同源。"""
    from src_new.core.conflict import _is_same_source
    ea, eb = evs.get(a), evs.get(b)
    if ea is None or eb is None:
        return False
    return _is_same_source(ea, eb)


def detect_collusion_pairs(awm: AWM) -> list[dict]:
    """CORE-1：返回全部跨来源近重复对 [{a, b, cos, same_source:false}, ...]。

    阈值 τ_dup 读 deception.dup_cos_threshold（默认 0.85）。
    同源对（预标注口径）跳过——同一报告的多 chunk 天然近重复，不是协同欺骗。
    消融 D4（deception.no_same_source_exclude）：不做同源排除（预期假阳性灾难）。
    消融 D2（deception.flag_all）：返回空（标记移除，由 Controller 走全量送裁）。
    """
    if _cfg("flag_all", False):
        return []  # D2：CORE-1 标记关闭，Controller 改走全量路径
    tau = float(_cfg("dup_cos_threshold", 0.85))
    no_same_exclude = _cfg("no_same_source_exclude", False)  # D4
    evs = {e.doc_id: e for e in awm.evidence_set}
    ids = [e.doc_id for e in awm.evidence_set if e.doc_id != QUERY_NODE]
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if not no_same_exclude and _same_source(evs, a, b):
                continue
            cos = _content_cosine(awm, a, b)
            if cos > tau:
                pairs.append({"a": a, "b": b, "cos": round(float(cos), 4)})
    return pairs


def collusion_flagged_nodes(awm: AWM, pairs: list[dict]) -> set:
    """参与近重复互证对的节点集合（CORE-1 的标记输出）。"""
    flagged = set()
    for p in pairs or []:
        flagged.add(p["a"])
        flagged.add(p["b"])
    return flagged


# ============================================================
# CORE-2：归因-内容分歧检测（v3，确定性，零 LLM，检索支撑）
# ============================================================
def _attribution_names(awm: AWM, doc_id: str) -> set[str]:
    """证据归因主张的名字集（canonical_name + aliases，小写归一）。"""
    prof = awm.get_profile_by_doc_id(doc_id)
    if not prof:
        return set()
    ck = (prof.attribution.get("canonical_key") or {})
    names = {str(ck.get("canonical_name") or "").lower().strip()}
    names |= {str(a).lower().strip() for a in (ck.get("aliases") or [])}
    return {n for n in names if n}


def _actor_groups(awm: AWM) -> list[tuple[str, list[str]]]:
    """把证据按归因主张并查集分组（名字集相交即同一组织主张）。
    返回 [(代表名(最小名字), [doc_ids]), ...]，无归因主张的证据不参与。"""
    evs = [e for e in awm.evidence_set if e.doc_id != QUERY_NODE]
    names_by_doc = {e.doc_id: _attribution_names(awm, e.doc_id) for e in evs}
    ids = [e.doc_id for e in evs if names_by_doc[e.doc_id]]
    parent = {d: d for d in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if names_by_doc[a] & names_by_doc[b]:
                parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for d in ids:
        groups.setdefault(find(d), []).append(d)
    return [(min(names_by_doc[members[0]]), sorted(members)) for members in groups.values()]


def _emb(awm: AWM, doc_id: str) -> np.ndarray | None:
    ev = awm.get_evidence_by_doc_id(doc_id)
    v = (ev.content_embedding if ev else []) or []
    return np.array(v, dtype=float) if len(v) else None


def _centroid(vecs: list[np.ndarray]) -> np.ndarray | None:
    if not vecs:
        return None
    c = np.mean(np.stack(vecs), axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else None


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


_ACTOR_CENTROID_CACHE: dict[str, np.ndarray | None] = {}


def default_actor_centroid(actor: str) -> np.ndarray | None:
    """组织 A 在干净语料的历史内容质心 μ_A：BM25(组织名) 于 APT class 取
    top-k 文档（排除 poison_ 前缀），BGE 嵌入求质心。命中 <3 篇返回 None
    （小众组织无组织史可对照，C2 弃权）。进程内缓存。"""
    key = actor.lower().strip()
    if key in _ACTOR_CENTROID_CACHE:
        return _ACTOR_CENTROID_CACHE[key]
    vec = None
    try:
        from src_new.retrieval import weaviate_retriever as retr
        from src_new.utils.embedding import get_embed_model, get_target_dimension
        idx = retr._bm25_index
        if not idx.docs:
            idx.build()
        hits = [idx.get_doc(i) for i, _ in idx.search(actor, top_k=12)]
        hits = [d for d in hits
                if not str(d.get("source_file", "")).startswith("poison_")
                and str(d.get("content", "")).strip()][:int(_cfg("core2_actor_corpus_k", 8))]
        if len(hits) >= 3:
            embs = get_embed_model().embed_documents(
                [str(d.get("content", ""))[:4000] for d in hits])
            vecs = [np.array(v[:get_target_dimension()], dtype=float) for v in embs]
            vec = _centroid(vecs)
    except Exception:
        vec = None
    _ACTOR_CENTROID_CACHE[key] = vec
    return vec


def detect_attribution_divergence(awm: AWM, actor_centroid_fn=None) -> list[dict]:
    """CORE-2：返回分歧标记 [{doc_id, signals, scores, detail}, ...]（确定性）。

    C1 盟友分歧：同归因组内（跨源），该文与**任一**同归因跨源成员的内容 cos
        都 <τ_div（最大佐证余弦判据——不用质心：组被毒污染时质心会拉偏，
        真实成员互相佐证即豁免，毒只被协同伙伴佐证而伙伴已被 CORE-1/C3 覆盖）
        ——声称同组织而无任何内容佐证（hard 档主信号）。
    C2 组织史分歧：内容与该组织干净语料质心 μ_A cos<τ_actor——内容锚在别处
        （r1 单篇/adaptive 风格分化后的主信号，攻击者去掉它就失去可检索性）。
    C3 同归因共发带：跨来源同归因对 content cos∈[τ_band_lo, τ_dup)——
        压到阈值下的协同共发指纹（adaptive 主信号）。
    core2_enable=false（消融 D6）返回空。同源对不参与（预标注排除）。
    """
    if not _cfg("core2_enable", True):
        return []
    tau_div = float(_cfg("core2_div_threshold", 0.60))
    tau_actor = float(_cfg("core2_actor_threshold", 0.45))
    band_lo = float(_cfg("core2_band_lo", 0.70))
    tau_dup = float(_cfg("dup_cos_threshold", 0.85))
    if actor_centroid_fn is None:
        actor_centroid_fn = default_actor_centroid
    evs = {e.doc_id: e for e in awm.evidence_set}
    flags: dict[str, dict] = {}

    def _flag(doc_id, signal, score, detail=""):
        f = flags.setdefault(doc_id, {"doc_id": doc_id, "signals": [],
                                      "scores": {}, "detail": []})
        f["signals"].append(signal)
        f["scores"][signal] = round(float(score), 4)
        if detail:
            f["detail"].append(detail)

    for actor, members in _actor_groups(awm):
        mu = actor_centroid_fn(actor)
        # C2：组织史分歧（组内每个成员）
        if mu is not None:
            for d in members:
                v = _emb(awm, d)
                if v is None:
                    continue
                c = _cos(v, mu)
                if c < tau_actor:
                    _flag(d, "C2_actor_divergence", c,
                          f"content cos {c:.3f} < τ_actor={tau_actor} vs "
                          f"corpus centroid of '{actor}'")
        # C1：盟友分歧（组内 ≥2 个跨源成员时，最大佐证余弦判据）
        if len(members) >= 2:
            vecs = {d: _emb(awm, d) for d in members}
            for d in members:
                coss = [(_cos(vecs[d], vecs[m]), m) for m in members
                        if m != d and vecs[m] is not None and vecs[d] is not None
                        and not _same_source(evs, d, m)]
                if coss:
                    c, partner = max(coss, key=lambda x: x[0])
                    if c < tau_div:
                        _flag(d, "C1_ally_divergence", c,
                              f"max content cos {c:.3f} < τ_div={tau_div} to ANY of "
                              f"{len(coss)} independent same-attribution members "
                              f"(attribution uncorroborated by content)")
        # C3：同归因共发带（组内跨源对）
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if _same_source(evs, a, b):
                    continue
                c = _content_cosine(awm, a, b)
                if band_lo <= c < tau_dup:
                    for d in (a, b):
                        _flag(d, "C3_copublication_band", c,
                              f"same-attribution cross-source pair cos {c:.3f} in "
                              f"[{band_lo}, {tau_dup})")
    return sorted(flags.values(), key=lambda f: min(f["scores"].values()))


def flagged_nodes(awm: AWM, core2_flags: list[dict] | None = None) -> set:
    """v3 统一标记集：CORE-1 近重复对端 ∪ CORE-2 分歧节点。"""
    flagged = collusion_flagged_nodes(awm, detect_collusion_pairs(awm))
    for f in core2_flags or []:
        flagged.add(f["doc_id"])
    return flagged


def audit_scores(awm: AWM, actor_centroid_fn=None) -> list[dict]:
    """CORE-2 诊断：无阈值全量分数转储（不判 flag，只记分布）。
    每条证据：归因组、μ_A 是否可用、C2 cos、C1 最大佐证 cos、C3 组内最大对 cos。"""
    if actor_centroid_fn is None:
        actor_centroid_fn = default_actor_centroid
    evs = {e.doc_id: e for e in awm.evidence_set}
    out = []
    for actor, members in _actor_groups(awm):
        mu = actor_centroid_fn(actor)
        # 组内跨源对余弦
        pair_cos = {}
        for i in range(len(members)):
            for j in range(len(members)):
                if i >= j:
                    continue
                a, b = members[i], members[j]
                if _same_source(evs, a, b):
                    continue
                c = _content_cosine(awm, a, b)
                pair_cos[(a, b)] = c
        for d in members:
            v = _emb(awm, d)
            c2 = _cos(v, mu) if (v is not None and mu is not None) else None
            c1 = max([c for (a, b), c in pair_cos.items() if d in (a, b)], default=None)
            out.append({"doc_id": d, "actor": actor,
                        "mu_available": mu is not None,
                        "c2_cos": None if c2 is None else round(float(c2), 4),
                        "c1_max_cos": None if c1 is None else round(float(c1), 4)})
    # 无归因主张的证据也记一笔（组外）
    grouped = {d for _, ms in _actor_groups(awm) for d in ms}
    for e in awm.evidence_set:
        if e.doc_id != QUERY_NODE and e.doc_id not in grouped:
            out.append({"doc_id": e.doc_id, "actor": None, "mu_available": False,
                        "c2_cos": None, "c1_max_cos": None})
    return out


# ============================================================
# CORE-2 证据包：归因-行为一致性视图（Controller 组装，纯数值）
# ============================================================
def build_evidence_package(awm: AWM, doc_id: str, pair: dict | None = None,
                           core2_flag: dict | None = None) -> dict:
    """为 ⑤ 组装单个被标记节点的裁决证据包（零 LLM）。

    内容：被标记报告的归因主张摘要 + 技术细节摘录、近重复对端信息（若有）、
    CORE-2 确定性分歧视图（若有）、归因-行为错配视图（归因盟友的聚合行为
    令牌 vs 该节点行为令牌）、私有令牌率、先验可靠性。
    """
    ev = awm.get_evidence_by_doc_id(doc_id)
    prof = awm.get_profile_by_doc_id(doc_id)

    # 归因盟友 = 直接边 attribution==support 的对端（排除近重复对端与查询锚点）
    allies = set()
    for e in awm.signed_graph_edges:
        if e.inferred or not e.dim_verdicts:
            continue
        if e.dim_verdicts.get("attribution") != "support":
            continue
        other = e.target if e.source == doc_id else (e.source if e.target == doc_id else None)
        if other and other not in (doc_id, QUERY_NODE):
            allies.add(other)
    dup_partner = None
    if pair:
        dup_partner = pair["b"] if pair["a"] == doc_id else pair["a"]
    allies.discard(dup_partner)

    def beh_tokens(d):
        p = awm.get_profile_by_doc_id(d)
        return {str(t).lower() for t in
                (((p.behavior.get("canonical_key") or {}).get("tokens")) or [])} if p else set()

    my_tokens = beh_tokens(doc_id)
    ally_tokens, ally_views = set(), []
    for u in allies:
        tu = beh_tokens(u)
        ally_tokens |= tu
        up = awm.get_profile_by_doc_id(u)
        ally_views.append({
            "ally": u,
            "attribution_desc": ((up.attribution.get("description") or "")[:200]) if up else "",
            "behavior_tokens": sorted(tu)[:15],
        })
    overlap = my_tokens & ally_tokens

    # 私有令牌率：低 = 该报告行为令牌大多已在其他报告中出现（派生内容指标）
    others_all = set()
    for e2 in awm.evidence_set:
        if e2.doc_id != doc_id:
            others_all |= beh_tokens(e2.doc_id)
    private_rate = (len(my_tokens - others_all) / len(my_tokens)) if my_tokens else 1.0

    pkg = {
        "doc_id": doc_id,
        "metadata": {
            "source": ev.source if ev else "",
            "publish_date": ev.publish_date if ev else "",
            "attribution_summary": (ev.attribution_summary if ev else "") or
                                   (ev.content[:600] if ev else ""),
            "content_excerpt": (ev.content[:1500] if ev else ""),
            "prior_reliability": ev.reliability if ev else 1.0,
        },
        "attribution_claim": {
            "canonical_name": ((prof.attribution.get("canonical_key") or {})
                               .get("canonical_name")) if prof else None,
            "aliases": ((prof.attribution.get("canonical_key") or {})
                        .get("aliases", [])) if prof else [],
            "description": ((prof.attribution.get("description") or "")[:400]) if prof else "",
        },
        "my_behavior_tokens": sorted(my_tokens)[:20],
        "attribution_allies": ally_views,
        "ally_behavior_tokens_union": sorted(ally_tokens)[:20],
        "behavior_overlap_ratio": round(len(overlap) / len(my_tokens), 4) if my_tokens else 0.0,
        "private_token_rate": round(float(private_rate), 4),
        "prior_reliability": ev.reliability if ev else 1.0,
    }
    if pair:
        pkg["near_duplicate_partner"] = {
            "doc_id": dup_partner,
            "content_cosine": pair["cos"],
            "note": ("cross-source near-duplicate (different publishers, cosine "
                     f"{pair['cos']:.3f} above threshold): coordinated corroboration "
                     "or honest syndication"),
        }
    if core2_flag:
        pkg["deterministic_divergence"] = {
            "signals": core2_flag["signals"],
            "scores": core2_flag["scores"],
            "evidence": core2_flag["detail"],
            "note": ("deterministic embedding-geometry flags computed without any LLM: "
                     "C1 content far from same-attribution allies; C2 content far from "
                     "the claimed actor's documented corpus centroid; C3 same-attribution "
                     "cross-source pair in the sub-threshold co-publication band"),
        }
    return pkg


# ============================================================
# QUARANTINE：隔离（确定性；无 Harmless Explanation 的节点退出证据组装）
# ============================================================
VALID_VERDICTS = ("credible", "suspicious", "deceptive")


def quarantine_set(awm: AWM, adjudications: list[dict],
                   core2_flags: list[dict] | None = None) -> set:
    """被隔离节点集合（图内保留，仅退出组装）。

    判定（v3）：CORE-1 ∪ CORE-2 标记（确定性结构证据）∧ ⑤ 未判 credible。
    分布依据（全量实测）：poison-pair content cos 中位数 0.867——主战场在
    0.85~0.90 区段；real-real 跨源对 0.85+ 亦存在（同系列报告/百科重复条目）。
    两分布重叠 → 静态阈值不可分，precision 由 ⑤ 的 credible 放行承担：
    suspicious/deceptive 均隔离（无害解释失败），credible 保留。定罪的结构
    证据全部来自确定性层（近重复互证 / 归因-内容分歧），语义层只负责排除
    ——延续"确定性控制与 LLM 推理相分离"。
    """
    flagged = flagged_nodes(awm, core2_flags)
    return {a.get("doc_id") for a in (adjudications or [])
            if isinstance(a, dict) and a.get("doc_id") in flagged
            and a.get("verdict") != "credible"}


def reweigh(awm: AWM, adjudications: list[dict]) -> dict:
    """v2 保留的降权通道（suspicious 仍只降权不隔离）：
    verdict→乘子 η 映射 + 钳位 + 全边 reliability 重算 + 跨来源近重复 support 边折扣 ι。

    deceptive 的主处置是 QUARANTINE（组装剔除），降权同时生效（双保险：
    若未来某任务绕过组装隔离直接读图，权重也已塌缩）。
    """
    evs = {e.doc_id: e for e in awm.evidence_set}
    mult = _cfg("reliability_multipliers",
                {"credible": 1.0, "suspicious": 0.7, "deceptive": 0.3})
    r_min = float(_cfg("reliability_clamp", [0.2, 1.0])[0])
    r_max = float(_cfg("reliability_clamp", [0.2, 1.0])[1])

    updates = []
    by_doc = {a.get("doc_id"): a.get("verdict", "") for a in (adjudications or [])
              if isinstance(a, dict)}
    for doc_id, verdict in by_doc.items():
        ev = evs.get(doc_id)
        if ev is None or verdict not in VALID_VERDICTS:
            continue
        prior = float(ev.reliability)
        post = min(max(float(mult.get(verdict, 1.0)) * prior, r_min), r_max)
        ev.reliability = post
        updates.append({"doc_id": doc_id, "prior": round(prior, 4),
                        "posterior": round(post, 4), "verdict": verdict})

    # 全边 reliability 重算（q 侧 1.0；非证据端点的反例边维持原值）
    dup_nodes = collusion_flagged_nodes(awm, detect_collusion_pairs(awm))
    discounted = 0
    for e in awm.signed_graph_edges:
        rs = 1.0 if e.source == QUERY_NODE else (
            evs[e.source].reliability if e.source in evs else None)
        rt = 1.0 if e.target == QUERY_NODE else (
            evs[e.target].reliability if e.target in evs else None)
        if rs is not None and rt is not None:
            e.reliability_factor = (rs + rt) / 2.0
        # ι：跨来源近重复 support 边折扣（truth discovery 复制检测的落法）；
        # 同源边（_same_source 同判据，含预标注对）固定折扣（同厂商两篇不算独立印证）；
        # conflict 边不折扣
        if e.sign == 1:
            if _same_source(evs, e.source, e.target):
                e.independence_factor = float(_cfg("same_source_independence", 0.7))
                discounted += 1
            elif e.source in dup_nodes and e.target in dup_nodes:
                e.independence_factor = float(_cfg("dup_independence", 0.3))
                discounted += 1
            else:
                e.independence_factor = 1.0
        else:
            e.independence_factor = 1.0
    return {"reliability_updates": updates, "discounted_edges": discounted}
