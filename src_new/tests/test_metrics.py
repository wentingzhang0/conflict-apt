"""
metrics 单测（创新点 1/2/3 数值部分，新版方案）：
- 两档匹配：时间线相接→conflict、behavior/campaign 仅 support/none（无灰区）
- 查询锚点 q↔证据走规范键匹配（可 conflict）
- neutral 对被传递性补全；h_graph = |C_h|·ρ_h − λ·κ_h
- 同输入同输出（可复现）
运行：python -m src_new.tests.test_metrics
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src_new.core.awm import AWM, Evidence, BehaviorProfile, GraphEdge, Hypothesis
from src_new.core import conflict, transitivity, canonical_match
from src import metrics
from src_new.core.signed_graph import graph_strength_for_hypothesis


def _ev(did, rel=0.9):
    return Evidence(did, "src_" + did, "content " + did, reliability=rel)


def _profile(did, ck=None):
    return BehaviorProfile.from_canonical(did, ck or {})


# ============== 两档匹配（无灰区）==============
def test_two_tier_matching():
    # attribution：相交→support；不相交→conflict；一侧缺→none
    ck_a = {"attribution": {"canonical_name": "lazarus group", "aliases": ["lazarus"]}}
    ck_b = {"attribution": {"canonical_name": "apt28", "aliases": ["fancy bear"]}}
    assert canonical_match.match_attribution(ck_a, {"attribution": {"aliases": ["lazarus"]}}) == "support"
    assert canonical_match.match_attribution(ck_a, ck_b) == "conflict"
    assert canonical_match.match_attribution(ck_a, {"attribution": {"aliases": []}}) == "none"

    # timeline：真正重叠(共享区间)→support；不相交/仅相接无共享→conflict；不全→none
    ov1 = {"timeline": {"start": "2020", "end": "2022"}}
    ov2 = {"timeline": {"start": "2022", "end": "2024"}}   # 共享 2022 → support
    dis = {"timeline": {"start": "2023", "end": "2024"}}   # 与 ov1 不相交 → conflict
    miss = {"timeline": {"start": "2020", "end": ""}}      # 不全 → none
    assert canonical_match.match_timeline(ov1, ov2) == "support"
    assert canonical_match.match_timeline(ov1, dis) == "conflict"
    assert canonical_match.match_timeline(ov1, miss) == "none"

    # behavior：Jaccard≥上阈→support；否则 none（无 suspect）
    bh = {"behavior": {"tokens": ["rat", "spearphishing", "defense"]}}
    bn = {"behavior": {"tokens": ["wiper", "financial"]}}
    assert canonical_match.match_behavior(bh, {"behavior": {"tokens": ["rat", "spearphishing", "defense"]}}) == "support"
    assert canonical_match.match_behavior(bh, bn) == "none"
    # campaign：令牌重叠→support；否则 none
    assert canonical_match.match_campaign({"campaign": {"tokens": ["op x"]}},
                                          {"campaign": {"tokens": ["op x"]}}) == "support"
    assert canonical_match.match_campaign({"campaign": {"tokens": ["op x"]}},
                                          {"campaign": {"tokens": ["op y"]}}) == "none"

    # match_all 无 suspect 档
    dv = canonical_match.match_all(ck_a, ck_b)
    assert "suspect" not in dv.values()
    print("PASS two_tier_matching")


# ============== 查询锚点 q↔证据走规范键匹配（3.2.4）==============
def test_query_anchor_signed_edges():
    awm = AWM(query="q", task_type="taa")
    awm.evidence_set = [_ev("d1"), _ev("d2"), _ev("d3")]
    awm.behavior_profiles = [
        _profile("d1", {"attribution": {"canonical_name": "apt28", "aliases": ["fancy bear"]},
                        "behavior": {"tokens": ["rat", "spearphishing"]}}),
        _profile("d2", {"attribution": {"canonical_name": "lazarus", "aliases": ["lazarus"]},
                        "behavior": {"tokens": ["rat", "spearphishing"]}}),
        _profile("d3", {"attribution": {"canonical_name": "apt29", "aliases": ["apt29"]},
                        "behavior": {"tokens": ["wiper"]}}),
    ]
    awm.query_profile = _profile("query",
                                 {"attribution": {"canonical_name": "lazarus", "aliases": ["lazarus"]},
                                  "behavior": {"tokens": ["rat", "spearphishing"]}})
    conflict.build_edges(awm)
    q_edges = [e for e in awm.signed_graph_edges if e.source == "query" or e.target == "query"]
    signs = {e.target: e.sign for e in q_edges}
    assert signs.get("d2") == 1, "query↔d2 同组织 → support(+1)"
    # _aggregate 保守化（conflict.py：归因 conflict 须行为重叠 beh==support 才判 −1）：
    # q↔d1 归因不相交但行为重叠 → 真冲突；q↔d3 归因不相交且行为不重叠 → none 不建边
    assert signs.get("d1") == -1, "query↔d1 归因冲突且行为重叠 → conflict(−1)"
    assert signs.get("d3") is None, "query↔d3 归因不相交且行为不重叠 → none（不建边，防伪冲突）"
    # Φ 三角形语义：T⁻ 数符号乘积为负（奇数条负边）的三角形。本图唯一完全三角
    # (q,d1,d2) = (−1)×(+1)×(−1) 乘积为正 → 平衡三角，T_minus=0 是正确值；
    # query conflict 边的效应体现在它进入图并参与后续分区，而非必然制造挫折。
    phi = metrics.frustration(awm)
    assert phi["T"] == 1 and phi["T_minus"] == 0, \
        f"(q,d1,d2) 两条负边 → 平衡三角, got {phi}"
    print("PASS query_anchor_signed_edges")


# ============== neutral 对被传递性补全 ==============
def test_neutral_pair_transitivity():
    awm = AWM(query="q", task_type="taa")
    for d in ["d1", "d2", "d3"]:
        awm.evidence_set.append(_ev(d))
    awm.signed_graph_edges = [
        GraphEdge("d1", "d2", 1, 0.9, edge_class="confirmed"),
        GraphEdge("d2", "d3", 1, 0.9, edge_class="confirmed"),
    ]
    awm.behavior_profiles = [_profile(d) for d in ["d1", "d2", "d3"]]
    transitivity.infer_neutral_pairs(awm)
    inferred = [e for e in awm.signed_graph_edges
                if frozenset((e.source, e.target)) == frozenset(("d1", "d3"))]
    assert len(inferred) == 1, f"expected 1 inferred edge, got {len(inferred)}"
    assert inferred[0].sign == 1, "consistent transitivity should yield +1"
    assert inferred[0].inferred is True
    print("PASS neutral_pair_transitivity")


# ============== h_graph（归一化复合：size·ρ·(1−λ·κ_norm)）==============
def test_h_graph_formula():
    awm = AWM(query="q", task_type="taa")
    awm.evidence_set = [_ev("d1"), _ev("d2")]
    awm.behavior_profiles = [_profile(d) for d in ["d1", "d2"]]
    # 簇内一条 support 边（weight = 0.9×1×1 = 0.9）
    awm.signed_graph_edges = [
        GraphEdge("d1", "d2", 1, 0.9, edge_class="confirmed"),
    ]
    hyp = Hypothesis(id="h1", answer="X", evidence_ids=["d1", "d2"])
    m = graph_strength_for_hypothesis(awm, hyp)
    # n=2, pairs=1, ρ=0.9, S(h)={d1,d2}, 无簇外 conflict → κ=0
    assert abs(m["rho"] - 0.9) < 1e-9
    assert m["kappa"] == 0.0
    # 归一化复合（signed_graph.py 当前公式）：size=n/(n+5) 与 llm_score 同量纲 ∈[0,1]
    size = 2 / 7.0
    assert abs(m["h_graph"] - size * 0.9 * (1 - m["lambda"] * 0.0)) < 1e-9
    # 原始量纲 |C_h|·ρ_h − λ·κ_h 仅作可解释参考
    assert abs(m["h_graph_raw"] - (2 * 0.9 - m["lambda"] * 0.0)) < 1e-9
    print(f"PASS h_graph_formula (rho={m['rho']:.3f} h_graph={m['h_graph']:.3f})")


def test_h_graph_pressure_lowers_strength():
    """落在 S(h) 节点的 conflict 边抬高 κ、压低 h_graph。"""
    awm = AWM(query="q", task_type="taa")
    awm.evidence_set = [_ev("d1"), _ev("d2")]
    awm.behavior_profiles = [_profile(d) for d in ["d1", "d2"]]
    awm.signed_graph_edges = [
        GraphEdge("d1", "d2", 1, 0.9, edge_class="confirmed"),
        GraphEdge("d1", "dx", -1, 0.8, edge_class="confirmed", conflict_dimension="attribution"),
    ]
    hyp = Hypothesis(id="h1", answer="X", evidence_ids=["d1", "d2"])
    m = graph_strength_for_hypothesis(awm, hyp)
    assert m["kappa"] > 0.0, "S(h) 上的 conflict 边应计入 κ"
    print(f"PASS h_graph_pressure_lowers_strength (kappa={m['kappa']:.3f})")


# ============== 门控可复现 ==============
def test_reproducible_same_output():
    awm = AWM(query="q", task_type="ate")
    for d in ["d1", "d2", "d3", "d4"]:
        awm.evidence_set.append(_ev(d))
        awm.behavior_profiles.append(_profile(d))
    awm.signed_graph_edges = [
        GraphEdge("d1", "d2", -1, 0.9, edge_class="confirmed", conflict_dimension="attribution"),
        GraphEdge("d2", "d3", 1, 0.8, edge_class="confirmed"),
    ]
    g1 = metrics.gating(awm, "ate")
    g2 = metrics.gating(awm, "ate")
    assert g1 == g2, "gating must be deterministic"
    assert g1["full_path"] is True and g1["has_neg"] is True
    print("PASS reproducible_same_output")


# ============== DEPTH 分档（任务自适应门槛）==============
def test_depth_from_phi_tiers():
    # I_max=1：DEPTH 退化为 Φ<θ₁→0、否则→1 的二值（θ₂ 在 I_max=1 下为无效项）
    assert metrics.depth_from_phi(0.05) == 0   # < θ₁(0.15) → 不反驳
    assert metrics.depth_from_phi(0.20) == 1   # ≥ θ₁ → 一轮反驳（I_max=1）
    assert metrics.depth_from_phi(0.50) == 1   # ≥ θ₂ → 仍是 I_max=1
    print("PASS depth_from_phi_tiers")


# ============== CLI 与纯函数一致（frustration）==============
def test_cli_matches_pure_function():
    import json
    import subprocess
    awm = AWM(query="q", task_type="taa")
    for d in ["d1", "d2", "d3"]:
        awm.evidence_set.append(_ev(d))
        awm.behavior_profiles.append(_profile(d))
    awm.signed_graph_edges = [
        GraphEdge("d1", "d2", -1, 0.9, edge_class="confirmed", conflict_dimension="attribution"),
        GraphEdge("d2", "d3", 1, 0.9, edge_class="confirmed"),
        GraphEdge("d1", "d3", -1, 0.9, edge_class="confirmed", conflict_dimension="attribution"),
    ]
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(awm.to_json())
        path = f.name
    pf = metrics.frustration(awm)["Phi"]
    try:
        out = subprocess.run([sys.executable, "-m", "src_new.metrics", "frustration",
                              "--graph", path],
                             capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        cli = json.loads(out.stdout)["Phi"]
        assert abs(pf - cli) < 1e-9, f"CLI {cli} != pure {pf}"
        print("PASS cli_matches_pure_function")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_two_tier_matching()
    test_query_anchor_signed_edges()
    test_neutral_pair_transitivity()
    test_h_graph_formula()
    test_h_graph_pressure_lowers_strength()
    test_reproducible_same_output()
    test_depth_from_phi_tiers()
    test_cli_matches_pure_function()
    print("\nAll metrics tests passed.")
