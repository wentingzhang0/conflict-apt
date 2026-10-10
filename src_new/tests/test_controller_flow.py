"""
Controller 控制流端到端验证（任务 7.1–7.4）。

真实流水线需 Weaviate + LLM 端点（部署依赖）。此处用确定性 stub 替换网络层
（retrieve / chat / parallel_chat / *_json_with_retry / embedding），验证：
  7.1 无冲突样例 → 共识快路径（fast_path=True，仅 ②+③共识+④ 各一次）
  7.2 冲突样例 → 完整路径（fast_path=False，depth_I 由 Φ 分档）
  7.3 无负边 → full_path=False（PATH 早退）；neutral 对被传递性补全
  7.4 相同输入两次运行的控制决策一致

运行：python -m src_new.tests.test_controller_flow
"""
import os
import sys
import types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 在导入 src 之前 stub 掉部署依赖（weaviate/openai/langchain），使测试可在无网络环境运行
for _m in ["weaviate", "openai", "langchain_huggingface", "langchain_huggingface.embeddings"]:
    sys.modules.setdefault(_m, types.ModuleType(_m))
sys.modules["weaviate"].Client = lambda **k: None
sys.modules["openai"].OpenAI = lambda **k: None
class _FakeEmbed:
    def embed_documents(self, x): return [[0.0] * 8 for _ in x]
    def embed_query(self, x): return [0.0] * 8
sys.modules["langchain_huggingface.embeddings"].HuggingFaceEmbeddings = lambda **k: _FakeEmbed()

import src_new.retrieval.weaviate_retriever as retr
import src_new.utils.llm_client as llm
import src_new.core.abp as abp_mod
from src_new.core.awm import BehaviorProfile
from src_new.controller import Controller
from src_new.tasks import TAATask


# ---------- stub 状态（按场景配置） ----------
_STATE = {"abp_results": [], "query_keys": {}, "gen_answers": [], "eval_scores": []}


def _stub_retrieve(query, top_k=10):
    """返回 _STATE 中预置的文档（含 source_file/content）。"""
    return _STATE.get("docs", [])[:top_k]


def _stub_chat(prompt="", system="", temperature=0.0, max_tokens=None, **kw):
    # query_summarizer / reporter / consensus / refutation_query 都走 chat
    if "Summarize" in prompt or "summary" in prompt[:40].lower():
        return "stub summary"
    if "counter-example" in prompt.lower() or "retrieval query" in prompt.lower():
        return "behavior tooling counterexample query"
    return "<answer>Stub Answer</answer> CONFIDENCE: 0.8"


def _stub_parallel_chat(tasks):
    return [_stub_chat(**t) for t in tasks]


def _stub_chat_json_retry(prompt="", system="", temperature=0.0, max_tokens=None, retries=None, **kw):
    # 查询画像抽取
    return _STATE.get("query_profile_result", {})


def _stub_parallel_chat_json_retry(tasks):
    return list(_STATE.get("abp_results", []))


def _stub_embedding(profile: BehaviorProfile):
    # 确定性向量（基于 doc_id 哈希），保证可复现
    h = abs(hash(profile.doc_id)) % 16
    return [float((h >> i) & 1) for i in range(8)]


def _install_stubs():
    # 消费模块在 import 时已绑定名字，须逐一替换各模块的绑定名
    import src_new.controller as ctrl
    import src_new.agents.evidence_graph_agent as ega
    import src_new.agents.hypothesis_agent as hyp
    import src_new.agents.reporter_agent as rep
    import src_new.core.abp as abp

    retr.retrieve = _stub_retrieve
    ega.retrieve = _stub_retrieve
    ctrl.chat = _stub_chat
    rep.chat = _stub_chat
    hyp.chat = _stub_chat
    hyp.parallel_chat = _stub_parallel_chat
    ega.parallel_chat_json_with_retry = _stub_parallel_chat_json_retry
    abp.chat_json_with_retry = _stub_chat_json_retry
    abp.compute_abp_embedding = _stub_embedding
    ega.compute_abp_embedding = _stub_embedding


def _doc(doc_id, content, title=""):
    return {"source_file": doc_id, "report_title": title or doc_id,
            "section_title": "", "content": content}


def _abp(doc_id, canonical_name, aliases, behavior=None, campaign=None,
         timeline=None):
    """模拟 ABP LLM 输出：四维 {description, canonical_key}（3.2.1 新结构）。"""
    return {
        "attribution": {"description": f"actor {canonical_name}",
                        "canonical_key": {"canonical_name": canonical_name, "aliases": aliases}},
        "timeline": {"description": "active period",
                     "canonical_key": timeline or {"start": "2020", "end": "2024"}},
        "behavior": {"description": "ttps",
                     "canonical_key": {"tokens": behavior or ["spearphishing"]}},
        "campaign": {"description": "campaign",
                     "canonical_key": {"tokens": campaign or ["campaignx"]}},
        "attribution_summary": f"summary for {doc_id}",
    }


# ============== 场景 ==============
def setup_no_conflict():
    """三份证据归因同一组织（别名相交）→ 全 support，无 conflict/uncertain。"""
    _STATE.clear()
    _STATE["docs"] = [_doc("d1", "report one"), _doc("d2", "report two"), _doc("d3", "report three")]
    _STATE["abp_results"] = [
        _abp("d1", "lazarus group", ["lazarus", "hidden cobra"]),
        _abp("d2", "lazarus group", ["lazarus"]),
        _abp("d3", "lazarus group", ["hidden cobra"]),
    ]
    _STATE["query_profile_result"] = _abp("query", "lazarus group", ["lazarus"])


def setup_conflict():
    """三方归因两两不相交（apt28 / lazarus / apt29）→ 含挫折三角形(三方互斥)，
    Φ>0 → 完整路径 + depth_I≥1（演示 DEPTH 旋钮）。"""
    _STATE.clear()
    _STATE["docs"] = [_doc("d1", "report one"), _doc("d2", "report two"), _doc("d3", "report three")]
    _STATE["abp_results"] = [
        _abp("d1", "apt28", ["fancy bear", "apt28"], behavior=["rat", "spearphishing"]),
        _abp("d2", "lazarus group", ["lazarus"], behavior=["rat", "spearphishing"]),
        _abp("d3", "apt29", ["apt29"], behavior=["rat"]),
    ]
    _STATE["query_profile_result"] = _abp("query", "lazarus group", ["lazarus"])


# ============== 测试 ==============
def test_no_conflict_fast_path():
    _install_stubs()
    setup_no_conflict()
    awm = Controller(TAATask()).analyze("some query report")
    assert awm.fast_path is True, "no-conflict scenario must take fast path"
    assert len(awm.hypothesis_space) == 1, "fast path → single consensus hypothesis"
    assert awm.Phi == 0.0
    assert len(awm.get_negative_edges()) == 0
    print("PASS no_conflict_fast_path (7.1)")


def test_conflict_full_path():
    _install_stubs()
    setup_conflict()
    awm = Controller(TAATask()).analyze("some query report")
    assert awm.fast_path is False, "conflict scenario must take full path"
    assert awm.conflict_detected is True
    assert len(awm.get_negative_edges()) >= 1, "should have ≥1 confirmed negative edge"
    assert awm.Phi > 0.0, "Φ should reflect conflict"
    print(f"PASS conflict_full_path (7.2): fast_path={awm.fast_path} "
          f"Φ={awm.Phi:.3f} k={awm.k} depth_I={awm.depth_I}")


def test_reproducible_control():
    _install_stubs()
    setup_conflict()
    a1 = Controller(TAATask()).analyze("some query report")
    a2 = Controller(TAATask()).analyze("some query report")
    assert a1.fast_path == a2.fast_path
    assert a1.Phi == a2.Phi and a1.k == a2.k and a1.depth_I == a2.depth_I
    print("PASS reproducible_control (7.4)")


def test_no_neg_fast_path_and_transitivity():
    """直接构造 AWM：无 sign=−1 负边 → PATH 早退(full_path=False, Φ=0)；neutral 对被传递性补全。

    gating 语义（metrics.gating docstring"始终谱分区"）：早退需【真单簇 k==1 且无负边】。
    合成 AWM 无 query_profile 时查询锚点 q 在拉普拉斯中是孤立零节点，零特征值重数 2，
    eigengap 会误选 k=2——这是合成场景的图结构事实，非流水线缺陷（真实运行时
    build_query_edges 总会给 q 建边）。故本测试给 q 补一条 support 边消除孤立零空间，
    验证补全后证据侧真单簇 + 无负边 → PATH 早退。"""
    from src_new.core.awm import AWM, Evidence, BehaviorProfile, GraphEdge
    from src_new.core import transitivity
    from src_new import metrics

    awm = AWM(query="q", task_type="taa")
    for d in ["d1", "d2", "d3"]:
        awm.evidence_set.append(Evidence(d, "s", "c"))
        awm.behavior_profiles.append(BehaviorProfile(d))
    # q 与全部证据建 support 边（模拟真实运行时 build_query_edges 的产出），
    # 证据链 d1-d2-d3 经传递性补全 → 全图连通单簇
    awm.signed_graph_edges = [
        GraphEdge("query", "d1", 1, 0.9, edge_class="confirmed"),
        GraphEdge("query", "d2", 1, 0.9, edge_class="confirmed"),
        GraphEdge("query", "d3", 1, 0.9, edge_class="confirmed"),
        GraphEdge("d1", "d2", 1, 0.9, edge_class="confirmed"),
        GraphEdge("d2", "d3", 1, 0.9, edge_class="confirmed"),
    ]
    transitivity.infer_neutral_pairs(awm)
    inferred = [e for e in awm.signed_graph_edges if e.inferred]
    assert len(inferred) >= 1, "d1-d3 为 neutral，应被传递性补全为 +1"
    assert all(e.sign == 1 for e in inferred)
    # 无负边 + 补全后全图连通单簇 → PATH 早退
    g = metrics.gating(awm, "taa")
    assert g["has_neg"] is False, "无 sign=−1 负边"
    assert g["k"] == 1, f"补全后应为真单簇, got k={g['k']}"
    assert g["full_path"] is False, "真单簇且无负边 → 共识早退"
    assert g["Phi"] == 0.0
    print("PASS no_neg_fast_path_and_transitivity (7.3)")


if __name__ == "__main__":
    test_no_conflict_fast_path()
    test_conflict_full_path()
    test_reproducible_control()
    test_no_neg_fast_path_and_transitivity()
    print("\nAll controller flow tests passed.")
