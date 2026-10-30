"""
④ 分析报告智能体 Reporter Agent（出口，固定双通道）

- 通道 1（主要）：原始报告全文（截断到字符上限）+ 完整证据文本（每条截断、
  总量受固定 token 预算约束，超出截断尾部证据）。
- 通道 2（辅助）：假设摘要、反驳日志、ABP 摘要、冲突统计、任务上下文。
token 预算：budget_chars = max_input_tokens × 3 − 报告长度 − 模板开销。
LLM 生成 → 答案提取（无效重试，带上限）→ 标准化（去别名/前缀）→ ATE 防退化。
"""
import re

from src_new.agents.base import BaseAgent
from src_new.core.awm import AWM
from src_new.core.apt_groups import candidate_groups_text
from src_new.utils.llm_client import chat
from src_new.utils.settings import get_task_prompt, get_pipeline_param, get_settings, get_prompt


class ReporterAgent(BaseAgent):
    name = "reporter_agent"

    def run(self, awm: AWM) -> AWM:
        # ATE 按平台选独立 reporter（企业/移动矩阵分离，避免移动样本看到企业表而混入企业技术）
        if awm.task_type == "ate":
            is_mobile = "[platform: mobile]" in (awm.query or "").lower()
            prompts = get_prompt("ate/reporter_mobile" if is_mobile else "ate/reporter_enterprise")
        else:
            prompts = get_task_prompt("reporter", awm.task_type)

        report_max = get_pipeline_param("reporter.report_max_chars", 6000)
        report_text = awm.query[:report_max]
        budget_chars = self._evidence_budget(awm, len(report_text))

        # 双通道消融（R1/R2）：R1 w/o 通道1=只给结构化分析（原始报告与证据文本移除，
        # 检验回锚必要性）；R2 w/o 通道2=只给原始证据（结构化摘要移除）
        dis_ch1 = get_pipeline_param("reporter.disable_channel1", False)
        dis_ch2 = get_pipeline_param("reporter.disable_channel2", False)

        channel1 = "" if dis_ch1 else self._build_channel1(awm, budget_chars)
        self.log(awm, "report_start",
                 f"fixed budget_chars={budget_chars} ch1_len={len(channel1)}"
                 f"{' [R1:ch1-off]' if dis_ch1 else ''}{' [R2:ch2-off]' if dis_ch2 else ''}")

        prompt = prompts["user"].format(
            original_report=("(channel 1 disabled)" if dis_ch1 else report_text),
            evidence_full_text=("(channel 1 disabled)" if dis_ch1 else channel1),
            abp_summary=("(channel 2 disabled)" if dis_ch2 else self._abp_summary(awm)),
            hypotheses_summary=("(channel 2 disabled)" if dis_ch2
                                else self._hypotheses_summary(awm)),
            refutation_summary=("(channel 2 disabled)" if dis_ch2
                                else self._refutation_summary(awm)),
            conflict_count=0 if dis_ch2 else len(awm.get_negative_edges()),
            conflict_dimensions=("none" if dis_ch2 else self._conflict_dims(awm)),
            candidate_groups=candidate_groups_text(),
        )
        system = prompts["system"]

        # 答案提取 + 重试（带上限）
        retries = get_pipeline_param("reporter.retries", 2)
        conclusion = ""
        for _ in range(retries + 1):
            resp = chat(prompt=prompt, system=system, temperature=0.0)
            conclusion = resp.strip()
            if self._valid_answer(conclusion, awm):
                break
        awm.final_conclusion = conclusion
        self.log(awm, "report_done", f"len={len(conclusion)}")
        return awm

    # ---------- 固定 token 预算（方案 8.2） ----------
    def _max_input_tokens(self) -> int:
        """全局 token 上限 = 模型最大上下文（不随冲突自适应）。"""
        return int(get_settings().get("models", {}).get("chat_model", {})
                   .get("max_context_tokens", 8192))

    def _evidence_budget(self, awm: AWM, report_len: int) -> int:
        """通道 1 证据文本可用字符预算：
        budget_chars = max_input_tokens × char_per_token − 报告长度 − 模板开销（方案 8.2）。"""
        max_tokens = get_pipeline_param("reporter.max_context_tokens", self._max_input_tokens())
        char_per_token = get_pipeline_param("reporter.char_per_token", 3)
        overhead = get_pipeline_param("reporter.template_overhead_chars", 1500)
        budget = int(max_tokens) * char_per_token - overhead - report_len
        return max(2000, budget)

    # ---------- 双通道构建 ----------
    def _build_channel1(self, awm: AWM, budget_chars: int) -> str:
        """完整证据文本，每条截断到字符上限，总量受固定 token 预算约束、超出截断尾部。
        被隔离证据（创新点 4 v2 QUARANTINE）不进入组装。"""
        per = get_pipeline_param("reporter.per_evidence_max", 1500)
        parts, total = [], 0
        for e in awm.active_evidence():
            body = (e.attribution_summary or e.content)[:per]
            block = f"[{e.source} - {e.section_title}]: {body}\n"
            if total + len(block) > budget_chars:
                break  # 超预算，截断尾部证据
            parts.append(block)
            total += len(block)
        return "".join(parts)

    def _build_channel2(self, awm: AWM) -> str:
        return (
            f"Hypotheses: {self._hypotheses_summary(awm)}\n"
            f"Refutation: {self._refutation_summary(awm)}\n"
            f"Conflicts: {len(awm.get_negative_edges())} negative edges; "
            f"Φ={awm.Phi:.3f} k={awm.k}\n"
            f"Task: {awm.task_instruction}"
        )

    def _abp_summary(self, awm: AWM) -> str:
        parts = []
        for p in awm.behavior_profiles[:8]:
            if awm.is_quarantined(p.doc_id):
                continue
            s = p.to_summary()
            if s:
                parts.append(s)
        return " || ".join(parts)[:1500]

    def _hypotheses_summary(self, awm: AWM) -> str:
        if not awm.hypothesis_space:
            return "N/A"
        return "; ".join(f"{h.answer} (llm={h.llm_score:.2f}, strength={h.strength:.2f})"
                         for h in awm.hypothesis_space[:5])

    def _refutation_summary(self, awm: AWM) -> str:
        if not awm.refutation_log:
            return "none"
        return f"{len(awm.refutation_log)} round(s); feedback={'yes' if awm.refutation_feedback else 'no'}"

    def _conflict_dims(self, awm: AWM) -> str:
        dims = sorted({e.conflict_dimension for e in awm.get_negative_edges()
                       if e.conflict_dimension})
        return ", ".join(dims) if dims else "none"

    # ---------- 答案有效性 + ATE 防退化 ----------
    def _valid_answer(self, conclusion: str, awm: AWM) -> bool:
        if "<answer>" in conclusion.lower() or "answer:" in conclusion.lower():
            return True
        # 兜底：非空即认为有效（评估阶段由 TaskAdapter.parse_output 标准化）
        return bool(conclusion.strip())
