"""
③ 假设推理智能体 Hypothesis Agent（创新点 2 假设生成 + 创新点 3 精化，方案 4.5）

两个 phase（由 Controller 调度）：
- generate_evaluate：簇内画像聚合 → 并行假设生成 → 多次采样 LLM 多维评估（取中位 llm_score）
  → 后处理（合并证据高度重叠的假设、淘汰弱假设）。h_graph 由 Controller 调 metrics 算，
  混合评分 α×h_graph+(1−α)×llm_score 也由 Controller 完成；本智能体只产出 llm_score。
- refute：取最强假设 h₁ 及其关键支撑边集 S(h₁)；把每条关键支撑证据转写为可投检索的
  行为/工具查询文本（counterexample_queries）。归因不相交的硬对抗过滤由 Controller 在
  检索后确定性执行，不由本智能体判定。本智能体不直接检索（反驳证据扩展经 Controller 中转）。

智能体间不直接通信；所有协作经 Controller 与 AWM。
"""
import re

from src_new.agents.base import BaseAgent
from src_new.core.awm import AWM, Hypothesis, CONFLICT_DIMENSIONS
from src_new.utils.llm_client import chat, parallel_chat
from src_new.utils.settings import get_task_prompt, get_pipeline_param, get_global, get_prompt


def _cfg(key, default=None):
    return get_pipeline_param(f"hypothesis.{key}", default)


def _clean_answer(a: str) -> str:
    """清理假设答案的前后噪声：markdown 强调符(* _ `)、引号、ANSWER/Final Answer 前缀。"""
    if not a:
        return a
    a = a.strip()
    a = re.sub(r'^(?:ANSWER|Final Answer)\s*[:\-]?\s*', '', a, flags=re.IGNORECASE)
    a = a.strip().strip('*_`"\'').strip()
    return a


def _extract_platform(awm: AWM):
    if awm.task_type == "ate":
        m = re.match(r'\[Platform:\s*(\w+)\]', awm.query)
        if m:
            return m.group(1)
    return "Enterprise"


def _format_gen_prompt(prompts, awm, **extra):
    kwargs = dict(
        query_summary=awm.query_summary,
        evidence_texts=extra.get("evidence_texts", ""),
        aggregated_profile=extra.get("aggregated_profile", "N/A"),
        refutation_context=extra.get("refutation_context", ""),
        task_instruction=awm.task_instruction,
        answer_format=awm.answer_format,
        platform=_extract_platform(awm),
    )
    return prompts["user"].format(**kwargs)


class HypothesisAgent(BaseAgent):
    name = "hypothesis_agent"

    # ============== phase: generate_evaluate ==============
    def generate_evaluate(self, awm: AWM) -> AWM:
        self.log(awm, "generate_evaluate_start")
        prompts = get_task_prompt("hypothesis_gen", awm.task_type)

        # 簇划分由 Controller 谱分区给出（awm.clusters）
        clusters = awm.clusters or {0: [e.doc_id for e in awm.evidence_set]}
        refutation_ctx = self._format_refutation_context(awm)

        tasks, meta = [], []
        for cluster_id, doc_ids in clusters.items():
            ev_text = self._cluster_evidence(awm, doc_ids)
            if not ev_text:
                continue
            agg = self._aggregate_profiles(awm, doc_ids)
            tasks.append({
                "prompt": _format_gen_prompt(prompts, awm, evidence_texts=ev_text,
                                             aggregated_profile=agg,
                                             refutation_context=refutation_ctx),
                "system": prompts["system"].format(task_instruction=awm.task_instruction),
                "temperature": _cfg("temperature", 0.0),
            })
            meta.append((cluster_id, doc_ids))

        responses = parallel_chat(tasks) if tasks else []
        awm.hypothesis_space = []
        for (cluster_id, doc_ids), resp in zip(meta, responses):
            awm.hypothesis_space.append(
                self._parse_hypothesis(resp, cluster_id, doc_ids, awm.iteration))

        # LLM 多维评估采样（取中位 llm_score）
        self._evaluate(awm)
        # 后处理：合并重叠、淘汰弱假设
        self._postprocess(awm)

        self.log(awm, "hypotheses_ready", f"count={len(awm.hypothesis_space)}")
        return awm

    def generate_consensus(self, awm: AWM) -> AWM:
        """共识快路径：所有证据视为一个簇，生成单一共识假设（不评估、不反驳）。"""
        prompts = get_task_prompt("hypothesis_gen", awm.task_type)
        all_ids = [e.doc_id for e in awm.evidence_set]
        ev_text = self._cluster_evidence(awm, all_ids)
        agg = self._aggregate_profiles(awm, all_ids)
        prompt = _format_gen_prompt(prompts, awm, evidence_texts=ev_text,
                                    aggregated_profile=agg)
        resp = chat(prompt=prompt,
                    system=prompts["system"].format(task_instruction=awm.task_instruction),
                    temperature=_cfg("temperature", 0.0))
        h = self._parse_hypothesis(resp, 0, all_ids, awm.iteration)
        # 快路径无冲突：共识假设的可信度即其置信度；h_graph/llm_score/strength 同步，
        # 避免 reporter 看到 strength=0 而无视假设、自行幻觉。
        h.llm_score = h.confidence
        h.strength = h.confidence
        awm.hypothesis_space = [h]
        self.log(awm, "consensus_hypothesis", f"answer={h.answer}")
        return awm

    def _evaluate(self, awm: AWM):
        prompts = get_task_prompt("hypothesis_eval", awm.task_type)
        samples = get_pipeline_param("evaluation.eval_samples", 3)
        # 全部 假设×采样 一次性并行（原为外层假设串行、仅假设内采样并行）
        flat_tasks, owner = [], []
        for hi, h in enumerate(awm.hypothesis_space):
            for _ in range(samples):
                flat_tasks.append({
                    "prompt": prompts["user"].format(
                        query_summary=awm.query_summary, answer=h.answer,
                        reasoning=h.reasoning[:500], evidence_count=len(h.evidence_ids),
                        task_instruction=awm.task_instruction),
                    "system": prompts["system"].format(task_instruction=awm.task_instruction),
                    "temperature": 0.2,
                })
                owner.append(hi)
        responses = parallel_chat(flat_tasks) if flat_tasks else []
        buckets = {}
        for hi, r in zip(owner, responses):
            s = self._parse_eval_score(r)
            if s is not None:
                buckets.setdefault(hi, []).append(s)
        for hi, h in enumerate(awm.hypothesis_space):
            overalls = buckets.get(hi, [])
            h.llm_score = float(sum(overalls) / len(overalls)) if overalls else 0.5

    @staticmethod
    def _parse_eval_score(response: str):
        m = re.search(r'OVERALL[\s:]*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
        if m:
            try:
                return max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                return None
        return None

    def _postprocess(self, awm: AWM):
        """合并证据高度重叠的假设、淘汰弱假设（llm_score 阈值）。"""
        merge_thresh = get_pipeline_param("refutation.overlap_merge", 0.7)
        kept = []
        for h in sorted(awm.hypothesis_space, key=lambda x: x.llm_score, reverse=True):
            dup = False
            for k in kept:
                inter = len(set(h.evidence_ids) & set(k.evidence_ids))
                union = len(set(h.evidence_ids) | set(k.evidence_ids)) or 1
                if inter / union >= merge_thresh:
                    dup = True
                    break
            if not dup:
                kept.append(h)
        awm.hypothesis_space = kept

    # ============== phase: refute ==============
    def refute(self, awm: AWM) -> AWM:
        """针对 h₁ 的关键支撑边集 S(h₁)，把支撑证据转写为行为/工具检索查询文本。
        不检索、不判归因（硬过滤与检索由 Controller 执行）。"""
        if not awm.hypothesis_space:
            return awm
        h1 = awm.hypothesis_space[0]
        s_h1 = self._key_support_edges(awm, h1)
        self.log(awm, "refute_start", f"h1={h1.answer} |S(h1)|={len(s_h1)}")

        prompts = get_prompt("refutation_query")
        # 对 S(h1) 的全部支撑证据一次性并行生成反例查询（原为串行 chat）
        prepared = []
        for ev_id, _w in s_h1:
            ev = awm.get_evidence_by_doc_id(ev_id)
            if not ev:
                continue
            prof = awm.get_profile_by_doc_id(ev_id)
            prof_text = prof.to_summary()[:500] if prof else ""
            prepared.append(prompts["user"].format(
                answer=h1.answer,
                supporting_evidence=(ev.attribution_summary or ev.content[:800]),
                supporting_profile=prof_text))
        tasks = [{"prompt": p, "system": prompts["system"], "temperature": 0.2}
                 for p in prepared]
        responses = parallel_chat(tasks) if tasks else []
        queries = [r.strip() for r in responses if r and r.strip()]

        awm.counterexample_queries = queries
        awm.refutation_log.append({
            "iteration": awm.iteration, "target_hypothesis": h1.id,
            "s_h1": [(eid, w) for eid, w in s_h1],
            "counterexample_queries": queries,
        })
        self.log(awm, "counterexample_queries", f"n={len(queries)}")
        return awm

    def build_refutation_feedback(self, awm: AWM, broken_support, conflict_edges) -> dict:
        """反例经 Controller 增量建图并裁定后，构造结构化反馈注入下一轮。"""
        h1 = awm.hypothesis_space[0] if awm.hypothesis_space else None
        fb = {
            "target_answer": h1.answer if h1 else "",
            "broken_support": broken_support,         # 被 confirmed conflict 边对冲的支撑
            "conflict_edges": conflict_edges,          # 反例与 S(h1) 形成的 conflict 边
            "counterexample_queries": awm.counterexample_queries,
        }
        awm.refutation_feedback = fb
        return fb

    # ============== 辅助 ==============
    def _key_support_edges(self, awm: AWM, hypothesis) -> list[tuple[str, float]]:
        """S(h₁)：h₁ 簇内 support 边里权重最高者所连的证据。
        返回 [(evidence_id, weight), ...]，按权重降序。"""
        from src_new.core.signed_graph import edge_weight
        doc_ids = set(hypothesis.evidence_ids)
        support = []  # [(other_id, weight)]
        for e in awm.signed_graph_edges:
            if e.edge_class != "confirmed" or e.sign != 1:
                continue
            if e.source in doc_ids and e.target in doc_ids:
                w = edge_weight(e)
                support.append((e.target, w))
                support.append((e.source, w))
        # 仅保留 h₁ 证据集内的对端，按权重降序去重
        seen = {}
        for eid, w in support:
            if eid in doc_ids:
                seen[eid] = max(seen.get(eid, 0.0), w)
        return sorted(seen.items(), key=lambda x: x[1], reverse=True)[:3]

    def _aggregate_profiles(self, awm: AWM, doc_ids: list) -> str:
        """簇内四维 ABP 描述聚合（去重拼接），作为假设生成锚点（隔离节点跳过）。"""
        dims = {d: [] for d in CONFLICT_DIMENSIONS}
        for p in awm.behavior_profiles:
            if p.doc_id in doc_ids and not awm.is_quarantined(p.doc_id):
                for d in CONFLICT_DIMENSIONS:
                    desc = ((getattr(p, d) or {}).get("description") or "").strip()
                    if desc:
                        dims[d].append(desc)
        if not any(dims.values()):
            return "N/A"
        labels = {"attribution": "Attribution", "timeline": "Timeline",
                  "behavior": "Behavior", "campaign": "Campaign"}
        parts = [f"{labels[d]}: {'; '.join(dims[d])}"
                 for d in CONFLICT_DIMENSIONS if dims[d]]
        return "\n".join(parts)

    def _cluster_evidence(self, awm: AWM, doc_ids: list) -> str:
        max_per = get_global("intermediate_evidence_max", 600)
        texts = []
        for e in awm.active_evidence():   # QUARANTINE：被隔离证据不进假设生成
            if e.doc_id in doc_ids:
                body = e.attribution_summary if e.attribution_summary else e.content[:max_per]
                texts.append(f"[{e.source} - {e.section_title}]: {body}")
        return "\n".join(texts)

    def _format_refutation_context(self, awm: AWM) -> str:
        if not awm.refutation_feedback:
            return ""
        fb = awm.refutation_feedback
        return (
            "\n[IMPORTANT - Previous refutation feedback]:\n"
            f"The previous top hypothesis ({fb.get('target_answer','')}) was challenged by a counter-example.\n"
            f"Broken support: {fb.get('broken_support', '')}\n"
            f"Generate an alternative hypothesis that can resist this counter-example."
        )

    def _parse_hypothesis(self, response: str, cluster_id: int, doc_ids: list,
                          iteration: int = 0) -> Hypothesis:
        answer = "Unknown"
        confidence = 0.5
        m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if m:
            answer = m.group(1).strip()
        else:
            m = re.search(r'ANSWER[\s:]*\s*(.+)', response, re.IGNORECASE)
            if m:
                answer = m.group(1).strip()
        answer = _clean_answer(answer)
        m = re.search(r'CONFIDENCE[\s:]*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
        if m:
            try:
                confidence = float(m.group(1))
            except ValueError:
                pass
        reasoning = response
        m_tag = re.search(r'<answer>|ANSWER', response, re.IGNORECASE)
        if m_tag:
            reasoning = response[:m_tag.start()].strip()
        return Hypothesis(
            id=f"h_{cluster_id}_iter{iteration}", answer=answer,
            evidence_ids=list(doc_ids), confidence=confidence,
            reasoning=reasoning, cluster_id=cluster_id)
