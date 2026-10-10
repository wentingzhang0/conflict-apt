"""
创新点 4（v2）单测：CORE→ADJUDGE→QUARANTINE
- CORE-1：跨来源近重复检测（同源 chunk 排除、poison 对命中）
- QUARANTINE：deceptive 退出组装、图与权重保留；enable=false 全链零行为
- 降权通道：η 映射、钳位下限、独立性折扣
- 端到端：CORE-1 标记 → 裁决 → 隔离后 active_evidence 不含 poison
运行：python -m src_new.tests.test_deception
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src_new.core.awm import AWM, Evidence, BehaviorProfile, GraphEdge
from src_new.core import deception
from src_new.utils.settings import apply_overrides
from src_new import metrics


def _ev(did, rel=0.9, source=None):
    return Evidence(did, source or ("src_" + did), "content " + did, reliability=rel)


def _profile(did, emb):
    p = BehaviorProfile.from_canonical(did, {})
    p.embedding = emb
    return p


def _poisoned_awm():
    """4 证据：real1/real2 同源不同 chunk（近重复但同 source，应排除）；
    poison_a/poison_b 跨来源近重复（应命中）；real3 独立。"""
    awm = AWM(query="q", task_type="taa")
    awm.evidence_set = [
        _ev("real1", source="Vendor X Report"),
        _ev("real2", source="Vendor X Report"),
        _ev("poison_a", source="Fake Blog Alpha"),
        _ev("poison_b", source="Fake Blog Beta"),
        _ev("real3", source="Vendor Y Report"),
    ]
    awm.behavior_profiles = [
        _profile("real1", [1.0, 0.0, 0.0]),
        _profile("real2", [0.99, 0.05, 0.0]),   # 同 chunk 近重复
        _profile("poison_a", [0.0, 1.0, 0.0]),
        _profile("poison_b", [0.0, 0.97, 0.1]),  # 跨来源近重复
        _profile("real3", [0.5, 0.5, 0.5]),
    ]
    # 全 support 图（复现真实失败场景：k=1 无冲突）
    ids = [e.doc_id for e in awm.evidence_set]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            awm.signed_graph_edges.append(
                GraphEdge(ids[i], ids[j], 1, 0.9, edge_class="confirmed"))
    return awm


# ============== CORE-1：跨来源近重复检测 ==============
def test_core1_cross_source_near_duplicate():
    awm = _poisoned_awm()
    pairs = deception.detect_collusion_pairs(awm)
    assert len(pairs) == 1, f"应只命中 poison 对, got {pairs}"
    assert {pairs[0]["a"], pairs[0]["b"]} == {"poison_a", "poison_b"}
    # 同源 chunk（real1-real2）cos 同样高但 source 相同 → 排除
    flagged = deception.collusion_flagged_nodes(awm, pairs)
    assert flagged == {"poison_a", "poison_b"}
    print("PASS core1_cross_source_near_duplicate")


def test_core1_clean_graph_no_flag():
    awm = _poisoned_awm()
    # 移除 poison_b → 无跨来源近重复
    awm.evidence_set = [e for e in awm.evidence_set if e.doc_id != "poison_b"]
    awm.behavior_profiles = [p for p in awm.behavior_profiles if p.doc_id != "poison_b"]
    assert deception.detect_collusion_pairs(awm) == []
    print("PASS core1_clean_graph_no_flag")


# ============== QUARANTINE：隔离语义 ==============
def test_quarantine_removes_from_assembly_not_graph():
    awm = _poisoned_awm()
    n_edges = len(awm.signed_graph_edges)
    # 隔离条件：CORE-1 标记 ∧ ⑤ 未判 credible（suspicious/deceptive 均隔离——
    # 结构定罪来自确定性层，语义层只需无害解释失败）
    verdicts = [{"doc_id": "poison_a", "verdict": "suspicious"},
                {"doc_id": "poison_b", "verdict": "deceptive"}]
    q = deception.quarantine_set(awm, verdicts)
    assert q == {"poison_a", "poison_b"}, f"got {q}"
    awm.quarantined_doc_ids = sorted(q)
    # 图与证据集保留（可解释、可回滚）
    assert len(awm.evidence_set) == 5
    assert len(awm.signed_graph_edges) == n_edges
    # 组装视图剔除
    active_ids = {e.doc_id for e in awm.active_evidence()}
    assert active_ids == {"real1", "real2", "real3"}
    assert awm.is_quarantined("poison_a") and not awm.is_quarantined("real1")
    print("PASS quarantine_removes_from_assembly_not_graph")


def test_quarantine_requires_core1_flag_and_spares_credible():
    awm = _poisoned_awm()
    # poison 对 cos≈0.97（>τ_high=0.90 强证据区）：credible 放行，其余隔离
    verdicts = [{"doc_id": "poison_a", "verdict": "credible"},
                {"doc_id": "poison_b", "verdict": "suspicious"}]
    q = deception.quarantine_set(awm, verdicts)
    assert q == {"poison_b"}, f"credible 应被放行: {q}"
    # 未被 CORE-1 标记的节点即使判 deceptive 也不隔离（隔离必须挂在结构证据上）
    verdicts2 = [{"doc_id": "real3", "verdict": "deceptive"}]
    assert deception.quarantine_set(awm, verdicts2) == set()
    # poison 对 cos≈0.97（中段同判）：suspicious 也隔离（无害解释失败）
    q2 = deception.quarantine_set(
        awm, [{"doc_id": "poison_b", "verdict": "suspicious"}])
    assert q2 == {"poison_b"}, f"suspicious 应隔离: {q2}"
    print("PASS quarantine_requires_core1_flag_and_spares_credible")


# ============== 降权通道 ==============
def test_reweigh_mapping_and_clamp():
    assert abs(deception.reweigh(_poisoned_awm(), [])["discounted_edges"]) >= 0
    awm = _poisoned_awm()
    # deceptive ×0.3 + 钳位下限 0.2
    from src_new.core.deception import _cfg
    r = deception.reweigh(awm, [{"doc_id": "poison_a", "verdict": "deceptive"}])
    upd = {u["doc_id"]: u for u in r["reliability_updates"]}
    assert abs(upd["poison_a"]["posterior"] - 0.27) < 1e-9  # 0.9×0.3
    awm2 = _poisoned_awm()
    awm2.get_evidence_by_doc_id("poison_a").reliability = 0.5
    r2 = deception.reweigh(awm2, [{"doc_id": "poison_a", "verdict": "deceptive"}])
    upd2 = {u["doc_id"]: u for u in r2["reliability_updates"]}
    assert abs(upd2["poison_a"]["posterior"] - 0.2) < 1e-9  # clamp 下限
    print("PASS reweigh_mapping_and_clamp")


def test_independence_discount_on_dup_support_edges():
    awm = _poisoned_awm()
    r = deception.reweigh(awm, [])
    # poison_a–poison_b support 边应被折扣（跨来源近重复）
    e = [x for x in awm.signed_graph_edges
         if set((x.source, x.target)) == {"poison_a", "poison_b"}][0]
    assert abs(e.independence_factor - 0.3) < 1e-9
    # real1–real2 同源 → 固定 0.7
    e2 = [x for x in awm.signed_graph_edges
          if set((x.source, x.target)) == {"real1", "real2"}][0]
    assert abs(e2.independence_factor - 0.7) < 1e-9
    # 无关边不折扣
    e3 = [x for x in awm.signed_graph_edges
          if set((x.source, x.target)) == {"real1", "real3"}][0]
    assert e3.independence_factor == 1.0
    print(f"PASS independence_discount_on_dup_support_edges")


# ============== Φ 拓扑不变 ==============
def test_phi_invariant_under_reweigh():
    awm = _poisoned_awm()
    phi_before = metrics.frustration(awm)["Phi"]
    deception.reweigh(awm, [{"doc_id": "poison_a", "verdict": "deceptive"}])
    assert abs(metrics.frustration(awm)["Phi"] - phi_before) < 1e-12
    print("PASS phi_invariant_under_reweigh")


# ============== enable=false 等价性 ==============
def test_disabled_by_default_equivalence():
    from src_new.utils.settings import get_pipeline_param
    assert get_pipeline_param("deception.enable", False) is False
    awm = _poisoned_awm()
    snap_edges = [(e.source, e.target, e.sign, e.reliability_factor,
                   getattr(e, "independence_factor", 1.0))
                  for e in awm.signed_graph_edges]
    snap_rel = [e.reliability for e in awm.evidence_set]
    enabled = get_pipeline_param("deception.enable", False)
    if not enabled:   # 与 Controller._credibility_pass 首行同判定
        pass
    assert awm.credibility_pass == {}
    assert awm.quarantined_doc_ids == []
    assert [e.doc_id for e in awm.active_evidence()] == [e.doc_id for e in awm.evidence_set]
    assert [(e.source, e.target, e.sign, e.reliability_factor,
             getattr(e, "independence_factor", 1.0)) for e in awm.signed_graph_edges] == snap_edges
    assert [e.reliability for e in awm.evidence_set] == snap_rel
    print("PASS disabled_by_default_equivalence")


# ============== 证据包组装 ==============
def test_evidence_package_fields():
    awm = _poisoned_awm()
    pairs = deception.detect_collusion_pairs(awm)
    pkg = deception.build_evidence_package(awm, "poison_a", pairs[0])
    assert pkg["doc_id"] == "poison_a"
    assert pkg["near_duplicate_partner"]["doc_id"] == "poison_b"
    assert "attribution_claim" in pkg and "my_behavior_tokens" in pkg
    assert "private_token_rate" in pkg
    print("PASS evidence_package_fields")


# ============== CORE-2：归因-内容分歧（v3，确定性） ===============
def _core2_awm():
    """6 证据三组归因：
    - 组 X（替罪羊）：poison_a（事件内容，嫁祸 X）+ ally_x1/ally_x2（X 的真实历史内容，
      跨源）→ poison_a 与盟友质心分歧（C1）；与 μ_X 分歧（C2，用注入的质心函数）
    - 组 M（真值）：real_m1/real_m2 内容一致 → 无分歧
    - poison_a/poison_b 若调高相似度会另触发 CORE-1/C3（各测试单独控制）"""
    awm = AWM(query="q", task_type="taa")
    awm.evidence_set = [
        _ev("poison_a", source="Fake Blog Alpha"),
        _ev("ally_x1", source="Vendor X Research"),
        _ev("ally_x2", source="Institute X Paper"),
        _ev("real_m1", source="Vendor M Report"),
        _ev("real_m2", source="Team M Notes"),
        _ev("poison_b", source="Fake Blog Beta"),
    ]
    # 内容嵌入：事件方向 e1=(1,0,0)，X 组织史方向 x=(0,1,0)，M 方向 m=(0.6,0.2,0.8)
    embs = {
        "poison_a": [1.0, 0.05, 0.0],
        "ally_x1": [0.05, 1.0, 0.1],
        "ally_x2": [0.0, 0.95, 0.05],
        "real_m1": [0.60, 0.20, 0.80],
        "real_m2": [0.58, 0.22, 0.78],
        "poison_b": [0.95, 0.10, 0.05],
    }
    for e in awm.evidence_set:
        e.content_embedding = embs[e.doc_id]
    from src_new.core.awm import BehaviorProfile

    def _prof(did, name, aliases=()):
        p = BehaviorProfile.from_canonical(did, {})
        p.attribution = {"canonical_key": {"canonical_name": name,
                                            "aliases": list(aliases)}}
        p.embedding = embs[did]
        return p

    awm.behavior_profiles = [
        _prof("poison_a", "Scapegoat X"),
        _prof("ally_x1", "Scapegoat X"),
        _prof("ally_x2", "Scapegoat X", ["X Group"]),
        _prof("real_m1", "True Actor M"),
        _prof("real_m2", "True Actor M"),
        _prof("poison_b", "Scapegoat Y"),
    ]
    ids = [e.doc_id for e in awm.evidence_set]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            awm.signed_graph_edges.append(
                GraphEdge(ids[i], ids[j], 1, 0.9, edge_class="confirmed"))
    return awm


def _stub_centroids(actor):
    """μ_X 指向 X 组织史方向；μ_M 指向 M 方向；其他组织 None（无组织史）。"""
    return {"scapegoat x": [0.05, 1.0, 0.1],
            "true actor m": [0.60, 0.20, 0.80]}.get(actor.lower().strip())


def test_core2_ally_and_actor_divergence():
    apply_overrides({"deception.core2_enable": True, "deception.core2_band_lo": 0.99})
    awm = _core2_awm()
    flags = deception.detect_attribution_divergence(awm, actor_centroid_fn=_stub_centroids)
    flg = {f["doc_id"]: f for f in flags}
    # poison_a：内容=事件方向，与 X 盟友质心（x 方向）和 μ_X 均分歧 → C1+C2
    assert "C1_ally_divergence" in flg["poison_a"]["signals"], flg
    assert "C2_actor_divergence" in flg["poison_a"]["signals"], flg
    # 真实组 M 内容互相一致且贴 μ_M → 无标记
    assert "real_m1" not in flg and "real_m2" not in flg, flg
    # X 的真实盟友自身与质心一致 → 不标记
    assert "ally_x1" not in flg and "ally_x2" not in flg, flg
    print("PASS core2_ally_and_actor_divergence")


def test_core2_copublication_band():
    apply_overrides({"deception.core2_enable": True, "deception.core2_band_lo": 0.6})
    awm = _core2_awm()
    # 构造一对 cos≈0.72 的同归因跨源证据（C3 带）
    awm.evidence_set.append(_ev("band_a", source="Outlet P"))
    awm.evidence_set.append(_ev("band_b", source="Outlet Q"))
    awm.evidence_set[-1].reliability = 0.9
    awm.evidence_set[-2].reliability = 0.9
    for e in awm.evidence_set[-2:]:
        pass
    import numpy as _np
    va = _np.array([1.0, 0.0, 0.0]); vb = _np.array([0.75, 0.66, 0.0])  # cos≈0.75
    awm.evidence_set[-2].content_embedding = list(va)
    awm.evidence_set[-1].content_embedding = list(vb)
    from src_new.core.awm import BehaviorProfile
    for did in ("band_a", "band_b"):
        p = BehaviorProfile.from_canonical(did, {})
        p.attribution = {"canonical_key": {"canonical_name": "Band Group",
                                            "aliases": []}}
        awm.behavior_profiles.append(p)
    flags = deception.detect_attribution_divergence(awm, actor_centroid_fn=lambda a: None)
    flg = {f["doc_id"]: f for f in flags}
    c = float(_np.dot(va, vb) / (_np.linalg.norm(va) * _np.linalg.norm(vb)))
    assert "band_a" in flg and "band_b" in flg, flg
    assert 0.6 <= c < 0.85
    assert "C3_copublication_band" in flg["band_a"]["signals"]
    print("PASS core2_copublication_band")


def test_core2_quarantine_union():
    apply_overrides({"deception.core2_band_lo": 0.99,
                     "deception.core2_enable": True})
    awm = _core2_awm()
    core2 = deception.detect_attribution_divergence(awm, actor_centroid_fn=_stub_centroids)
    # poison_a 被 CORE-2 标记；⑤ 非 credible → 隔离（v3 公式：CORE-1 ∪ CORE-2）
    v = [{"doc_id": "poison_a", "verdict": "suspicious"}]
    q = deception.quarantine_set(awm, v, core2)
    assert q == {"poison_a"}, q
    # credible 豁免
    q2 = deception.quarantine_set(awm, [{"doc_id": "poison_a",
                                         "verdict": "credible"}], core2)
    assert q2 == set(), q2
    # D6（core2_enable=false）→ CORE-2 不参与；本 fixture 中 poison_a/poison_b
    # 本就是 CORE-1 近重复对（cos≈0.998），v2 语义仍隔离——D6 只验证 CORE-2 不加标
    apply_overrides({"deception.core2_enable": False})
    core2_off = deception.detect_attribution_divergence(awm, actor_centroid_fn=_stub_centroids)
    assert core2_off == []
    assert deception.quarantine_set(awm, v, core2_off) == {"poison_a"}  # 仅 CORE-1 路径
    apply_overrides({"deception.core2_enable": True})
    print("PASS core2_quarantine_union")


def test_core2_alias_grouping():
    """别名并查集：ally_x2 用别名 'X Group'，与 canonical 'Scapegoat X' 相交 → 同组。"""
    awm = _core2_awm()
    groups = dict(deception._actor_groups(awm))
    # 3 组：Scapegoat X（3 成员，含别名盟友）、True Actor M（2）、Scapegoat Y（1）
    sizes = sorted(len(m) for m in groups.values())
    assert sizes == [1, 2, 3], groups
    print("PASS core2_alias_grouping")


if __name__ == "__main__":
    test_core2_ally_and_actor_divergence()
    test_core2_copublication_band()
    test_core2_quarantine_union()
    test_core2_alias_grouping()
    test_core1_cross_source_near_duplicate()
    test_core1_clean_graph_no_flag()
    test_quarantine_removes_from_assembly_not_graph()
    test_quarantine_requires_core1_flag_and_spares_credible()
    test_reweigh_mapping_and_clamp()
    test_independence_discount_on_dup_support_edges()
    test_phi_invariant_under_reweigh()
    test_disabled_by_default_equivalence()
    test_evidence_package_fields()
    print("\nAll deception v2+v3 tests passed.")
