"""
① 确定性编排器 Controller

非 LLM。所有数值（C/Φ/k/h_graph/clusters/门控）一律调 metrics.py；分支决策基于
图结构信号，同输入同路径。3 个 LLM 智能体经 Controller 中转，互不直连。

完整路径反驳协作链：
  ③(refute) 产出 counterexample_query 文本 → Controller 调 ②(mode=expand, 扩展池, 排除
  initial_doc_ids) → Controller 硬过滤归因别名集与 S(h₁) 相交者 → 增量建图（规范键抽取+
  确定性匹配）→ 图条件闸门（反例须与 S(h₁) 形成 confirmed conflict 边）→ metrics 重分区/
  重算 h_graph → Δh_graph>θ_H 标 unstable。depth_I 冻结、反例重算 Φ 不回写门控（仅进 trace）。
"""
from src_new.core.awm import AWM, Evidence, GraphEdge
from src_new.core import conflict, transitivity, deception
from src_new.core.signed_graph import QUERY_NODE
from src_new.core.abp import compute_abp_embedding
from src_new.core.canonical_match import _aliases
from src_new import metrics
from src_new.agents.evidence_graph_agent import EvidenceGraphAgent
from src_new.agents.hypothesis_agent import HypothesisAgent
from src_new.agents.reporter_agent import ReporterAgent
from src_new.agents.credibility_agent import CredibilityAgent
from src_new.tasks.base import TaskAdapter
from src_new.utils.llm_client import chat
from src_new.utils.settings import get_pipeline_param, get_task_prompt


class Controller:
    def __init__(self, task: TaskAdapter, top_k: int = None):
        self.task = task
        self.ega = EvidenceGraphAgent(top_k=top_k)
        self.hyp = HypothesisAgent()
        self.rep = ReporterAgent()
        self.cred = CredibilityAgent()

    # ============== 主流程 ==============
    def analyze(self, query: str) -> AWM:
        import time as _t
        _phase_t0 = _t.time()
        awm = AWM(query=query)
        awm.task_type = self.task.task_type
        awm.task_instruction = self.task.task_instruction
        awm.answer_format = self.task.answer_format
        awm.extra_context = self.task.extra_context

        # Phase 0: 查询摘要（一次普通 LLM，非智能体常驻）
        awm.query_summary = (query if awm.task_type == "mcq"
                             else self._summarize(query))
        awm.log("controller", "query_summary", f"len={len(awm.query_summary)} t={_t.time()-_phase_t0:.1f}s")
        _phase_t0 = _t.time()

        # Phase 1: 证据建图（②）
        awm = self.ega.run_full(awm)
        awm.log("controller", "phase1_done", f"edges={len(awm.signed_graph_edges)} t={_t.time()-_phase_t0:.1f}s")
        _phase_t0 = _t.time()
        if not awm.evidence_set:
            awm.final_conclusion = get_pipeline_param(
                "orchestrator.fallback_conclusion_no_evidence", "No relevant evidence found.")
            awm.log("controller", "fallback_no_evidence")
            return awm

        # Phase 1.5: 可信度审查（CORE → ADJUDGE → QUARANTINE），在门控之前对全体证据执行
        self._credibility_pass(awm)

        # Phase 2: 门控（PATH/DEPTH 单轴推理强度，确定性；原 ROUTING 已删除，Φ 唯一控制信号）
        g = metrics.gating(awm, awm.task_type)
        awm.Phi, awm.k = g["Phi"], g["k"]
        awm.clusters = g["clusters"]
        awm.depth_I = g["I"]
        awm.conflict_detected = g["full_path"]
        awm.log("controller", "gating",
                f"has_neg={g['has_neg']} Φ={awm.Phi:.3f} k={awm.k} I={awm.depth_I}")

        if not g["full_path"]:
            # PATH 早退：共识快路径（active_evidence 已剔除被隔离证据）
            awm.fast_path = True
            awm = self.hyp.generate_consensus(awm)
            awm.log("controller", "fast_path", "no sign=−1 negative edge → consensus")
        else:
            # 完整路径迭代（生成轮 + 至多 depth_I 轮反驳）
            awm.fast_path = False
            passes = awm.depth_I + 1
            for it in range(passes):
                awm.iteration = it
                awm = self.hyp.generate_evaluate(awm)
                # 混合评分（Controller 算 h_graph 部分）
                self._mix_scores(awm)
                if not awm.hypothesis_space:
                    break
                if it < passes - 1 and awm.depth_I > 0:
                    stop = self._refute_round(awm)
                    if stop:
                        break
                    # 清空假设、保留反例结构化反馈
                    awm.hypothesis_space = []
            awm.log("controller", "phase2_done", f"hyps={len(awm.hypothesis_space)} t={_t.time()-_phase_t0:.1f}s")
            _phase_t0 = _t.time()

        # Phase 3: 报告（④）
        awm = self.rep.run(awm)
        awm.log("controller", "phase3_done", f"report t={_t.time()-_phase_t0:.1f}s")
        return awm

    # ============== 可信度审查（CORE→ADJUDGE→QUARANTINE） ==============
    def _credibility_pass(self, awm: AWM):
        """CORE-1 近重复互证检测（确定性）→ ⑤ 语义裁决 → QUARANTINE 隔离 + 降权。

        - deception.enable=false（默认）时直接返回；
        - 单次执行，反驳阶段不触发二次审查；
        - Φ/PATH/DEPTH 不动（隔离不改图拓扑，仅影响下游 prompt 组装）；
        - 隔离以确定性标记为前提，CORE-1 标记本身不定罪。
        """
        if not get_pipeline_param("deception.enable", False):
            return
        awm.credibility_pass = {"activated": True, "collusion_pairs": [],
                                "core2_flags": [], "flagged": [], "adjudications": [],
                                "reliability_updates": [], "quarantined": [],
                                "discounted_edges": 0}
        # CORE-1：跨来源近重复互证（确定性，零 LLM）
        pairs = deception.detect_collusion_pairs(awm)
        # CORE-2：归因-内容分歧（确定性，零 LLM，默认关闭）
        core2 = deception.detect_attribution_divergence(awm)
        awm.credibility_pass["core2_flags"] = core2
        # 消融 D2（deception.flag_all）：移除结构标记——全部证据按检索序送 ⑤ 裁决，
        # 非 credible 按裁决隔离
        if get_pipeline_param("deception.flag_all", False):
            pairs = []
            core2 = []
            ordered = [e.doc_id for e in awm.evidence_set
                       if e.doc_id != QUERY_NODE]
            max_m = int(get_pipeline_param("deception.max_adjudications", 3))
            targets = ordered[:max_m * 2] if max_m else ordered
            awm.credibility_pass["flagged"] = ordered
            awm.log("controller", "credibility_d2_flag_all",
                    f"flagging all evidence (n={len(ordered)}, adjudicating {len(targets)})")
            packages = [deception.build_evidence_package(awm, d, None)
                        for d in targets]
            verdicts = self.cred.adjudicate_batch(awm, packages)
            awm.credibility_pass["adjudications"] = verdicts
            quarantined = sorted({v["doc_id"] for v in verdicts
                                  if v.get("verdict") != "credible"})
            if not get_pipeline_param("deception.no_quarantine", False):
                awm.quarantined_doc_ids = quarantined
                awm.credibility_pass["quarantined"] = quarantined
            result = deception.reweigh(awm, verdicts)
            awm.credibility_pass["reliability_updates"] = result["reliability_updates"]
            return
        core1_nodes = deception.collusion_flagged_nodes(awm, pairs)
        core2_by_doc = {f["doc_id"]: f for f in core2}
        flagged_all = core1_nodes | set(core2_by_doc)
        if not flagged_all:
            awm.log("controller", "credibility_core",
                    "no near-duplicate pair, no attribution divergence")
            return
        awm.credibility_pass["collusion_pairs"] = pairs
        awm.credibility_pass["flagged"] = sorted(flagged_all)
        awm.log("controller", "credibility_core",
                f"pairs={[(p['a'], p['b'], p['cos']) for p in pairs]} "
                f"core2={[(f['doc_id'], f['signals']) for f in core2]}")

        # 消融 D1（deception.bypass_adjudication）：跳过 ⑤，CORE-1 对端直接隔离
        if get_pipeline_param("deception.bypass_adjudication", False):
            direct = sorted({d for p in pairs for d in (p["a"], p["b"])})
            awm.quarantined_doc_ids = direct
            awm.credibility_pass["quarantined"] = direct
            result = deception.reweigh(
                awm, [{"doc_id": d, "verdict": "deceptive"} for d in direct])
            awm.credibility_pass["reliability_updates"] = result["reliability_updates"]
            awm.credibility_pass["discounted_edges"] = result["discounted_edges"]
            awm.log("controller", "credibility_d1_bypass",
                    f"direct quarantine={direct}")
            return

        # ADJUDGE：⑤ 豁免取证（每个标记节点一次，预算 ≤ 2M；CORE-1 结构最强
        # 优先，CORE-2 按分歧严重度排序补位）
        max_m = int(get_pipeline_param("deception.max_adjudications", 3))
        pair_by_doc = {}
        for p in pairs:
            pair_by_doc.setdefault(p["a"], p)
            pair_by_doc.setdefault(p["b"], p)
        ordered = sorted(core1_nodes) + [f["doc_id"] for f in core2
                                         if f["doc_id"] not in core1_nodes]
        targets = ordered[:max_m * 2] if max_m else ordered
        packages = [deception.build_evidence_package(
            awm, d, pair_by_doc.get(d), core2_by_doc.get(d))
            for d in targets]
        verdicts = self.cred.adjudicate_batch(awm, packages)
        awm.credibility_pass["adjudications"] = verdicts
        for v in verdicts:
            awm.log("controller", "credibility_verdict",
                    f"{v['doc_id']} → {v['verdict']}")

        # QUARANTINE + 降权（确定性）
        # 消融 D3（deception.no_quarantine）：只降权不隔离（验证组装层隔离的必要性）
        quarantined = deception.quarantine_set(awm, verdicts, core2)
        if not get_pipeline_param("deception.no_quarantine", False):
            awm.quarantined_doc_ids = sorted(quarantined)
            awm.credibility_pass["quarantined"] = sorted(quarantined)
        result = deception.reweigh(awm, verdicts)
        awm.credibility_pass["reliability_updates"] = result["reliability_updates"]
        awm.credibility_pass["discounted_edges"] = result["discounted_edges"]
        awm.log("controller", "credibility_quarantine",
                f"quarantined={awm.quarantined_doc_ids} "
                f"updates={result['reliability_updates']}")

    # ============== 反驳轮（图条件证伪协作链） ==============
    def _refute_round(self, awm: AWM) -> bool:
        """执行一轮图条件反驳。返回 True 表示应终止迭代。"""
        # ③ 产出 counterexample_query 文本（不检索、不判归因）
        awm = self.hyp.refute(awm)
        if not awm.counterexample_queries:
            awm.log("controller", "refute_no_query")
            return True

        h1 = awm.hypothesis_space[0]
        s_h1_ids = {eid for eid, _ in self.hyp._key_support_edges(awm, h1)}
        s_aliases = self._union_aliases(awm, s_h1_ids)

        # h_graph(前)
        h_before = metrics.h_graph(awm, h1)["h_graph"]

        legal_counterexamples = []
        added_edges: list[GraphEdge] = []
        for ce_query in awm.counterexample_queries:
            # ② 在扩展池检索，排除初始结果
            candidates = self.ega.run_expand(awm, ce_query, awm.initial_doc_ids)
            # Controller 硬过滤：剔除归因别名集与 S(h₁) 相交者（确定性，不靠 LLM）
            candidates = [c for c in candidates if not self._alias_disjoint_violation(awm, c, s_aliases)]
            if not candidates:
                continue
            # 增量建图：抽取规范键 + 与 S(h₁) 确定性匹配，裁定 confirmed conflict 闸门
            ce_edges = self._incremental_match(awm, candidates, s_h1_ids)
            legal = [e for e in ce_edges if e.edge_class == "confirmed" and e.sign == -1]
            if legal:
                # 反例证据【不并入 evidence_set】：它仅作为 conflict 边压测 h₁（其 conflict 边
                # 落在 S(h) 节点上计入 κ），避免撑大主簇、在第二轮假设生成与 reporter 中
                # 引入海量伪冲突。反例 doc_id 不在 evidence_set → 不进谱分区节点、不进假设聚合。
                legal_counterexamples.extend(legal)
                added_edges.extend(ce_edges)

        # 图条件闸门：无合法反例 → 终止
        if not legal_counterexamples:
            awm.log("controller", "refute_no_legal_counterexample",
                    "graph-gate: no confirmed conflict edge with S(h1)")
            return True

        # 增量建边 + 传递性补全 + 重分区
        awm.signed_graph_edges.extend(added_edges)
        transitivity.infer_neutral_pairs(awm)
        sp = metrics.partition(awm)
        awm.k = sp["k"]
        awm.clusters = sp["clusters"]

        # 反例重算 Φ 仅进 trace（depth_I 冻结，不回写门控）
        phi_new = metrics.frustration(awm)["Phi"]
        awm.log("controller", "refute_recompute",
                f"Φ_new={phi_new:.3f}(not written back) depth_I frozen={awm.depth_I}")

        # h_graph(后) + Δh_graph
        m_after = metrics.h_graph(awm, h1)
        h1.graph_metrics = m_after
        delta = metrics.delta_h_graph(h_before, m_after["h_graph"])
        theta_H = get_pipeline_param("refutation.theta_H", 0.1)
        if delta > theta_H:
            h1.unstable = True
            awm.log("controller", "h1_unstable", f"Δh_graph={delta:.3f} > θ_H={theta_H}")
        else:
            awm.log("controller", "refute_converged", f"Δh_graph={delta:.3f} ≤ θ_H → terminate")
            return True  # 图级强度不再下降 → 收敛终止

        # 结构化反馈注入下一轮
        self.hyp.build_refutation_feedback(
            awm, broken_support=list(s_h1_ids),
            conflict_edges=[(e.source, e.target) for e in legal_counterexamples])
        return False

    # ============== 混合评分（Controller 算 h_graph） ==============
    def _mix_scores(self, awm: AWM):
        alpha = get_pipeline_param("evaluation.alpha", 0.5)
        for h in awm.hypothesis_space:
            m = metrics.h_graph(awm, h)
            h.graph_metrics = m
            h.strength = alpha * m["h_graph"] + (1 - alpha) * h.llm_score
        awm.hypothesis_space.sort(key=lambda x: x.strength, reverse=True)

    # ============== 增量建图辅助 ==============
    def _incremental_match(self, awm: AWM, candidates: list[Evidence], s_h1_ids: set) -> list[GraphEdge]:
        """对反例候选与 S(h₁) 证据做规范键确定性匹配，产出 GraphEdge 列表。
        仅返回与 S(h₁) 中证据形成的边（用于图条件闸门裁定）。"""
        # 为反例候选补画像（含规范键，单次 LLM/候选）
        self._ensure_profiles(awm, candidates)
        edges = []
        prof_by_doc = {p.doc_id: p for p in awm.behavior_profiles}
        for c in candidates:
            pc = prof_by_doc.get(c.doc_id)
            ck_c = (pc.canonical_keys if pc else {}) or {}
            for sid in s_h1_ids:
                ps = prof_by_doc.get(sid)
                ck_s = (ps.canonical_keys if ps else {}) or {}
                if not ck_c or not ck_s:
                    continue
                # 复用 conflict._aggregate + canonical_match
                dv = __import__("src_new.core.canonical_match", fromlist=["match_all"]).match_all(ck_c, ck_s)
                sign, edge_class, cdim = conflict._aggregate(dv)
                if edge_class == "confirmed":
                    edges.append(GraphEdge(
                        source=c.doc_id, target=sid, sign=sign, confidence=0.9,
                        edge_class="confirmed", conflict_dimension=cdim,
                        profile_modulation=1.0, reliability_factor=(c.reliability + 1.0) / 2.0,
                        dim_verdicts=dv))
        return edges

    def _ensure_profiles(self, awm: AWM, candidates: list[Evidence]):
        """为尚无画像的反例候选抽取 ABP（规范键），并入 awm.behavior_profiles。"""
        from src_new.core.abp import extract_abp
        existing = {p.doc_id for p in awm.behavior_profiles}
        for c in candidates:
            if c.doc_id in existing:
                continue
            prof = extract_abp(c.doc_id, c.content[:4000])
            prof.embedding = compute_abp_embedding(prof)
            awm.behavior_profiles.append(prof)
            existing.add(c.doc_id)

    # ============== 归因硬过滤辅助 ==============
    def _union_aliases(self, awm: AWM, doc_ids: set) -> set:
        s = set()
        for did in doc_ids:
            p = awm.get_profile_by_doc_id(did)
            if p:
                s |= _aliases(p.canonical_keys or {})
        return s

    def _alias_disjoint_violation(self, awm, candidate: Evidence, s_aliases: set) -> bool:
        """候选归因别名集与 S(h₁) 相交 → 违反硬过滤（应剔除）。"""
        if not s_aliases:
            return False
        p = awm.get_profile_by_doc_id(candidate.doc_id)
        if not p:
            return False
        cand_aliases = _aliases(p.canonical_keys or {})
        return bool(cand_aliases & s_aliases)

    # ============== 查询摘要 ==============
    def _summarize(self, query: str) -> str:
        prompts = get_task_prompt("query_summarizer", self.task.task_type)
        report_max = get_pipeline_param("query_summarizer.report_max_length", 6000)
        return chat(prompt=prompts["user"].format(report=query[:report_max]),
                    system=prompts["system"], temperature=0.0).strip()
