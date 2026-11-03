"""
⑤ 可信度裁决智能体 Credibility Agent（创新点 4 v2 的语义部分）

对 CORE-1 标记的跨来源近重复文档做归因-行为一致性取证。v2 裁决任务聚焦：
"归因主张引用的画像要素（组织名/别名）与报告自身技术细节、与归因盟友的
被印证画像是否互洽"。输出契约（只出类别与理由，不出数字）：
  {doc_id, verdict ∈ credible|suspicious|deceptive, justification}
"""
import json

from src_new.agents.base import BaseAgent
from src_new.core.awm import AWM
from src_new.utils.llm_client import parallel_chat_json_with_retry
from src_new.utils.settings import get_prompt, get_pipeline_param


class CredibilityAgent(BaseAgent):
    name = "credibility_agent"

    def adjudicate_batch(self, awm: AWM, evidence_packages: list[dict]) -> list[dict]:
        """对一批证据包并行裁决。解析失败 → credible 兜底（定罪失败宁可放过，
        与"隔离是重处置"的保守纪律一致）。"""
        if not evidence_packages:
            return []
        prompts = get_prompt("credibility")
        retries = get_pipeline_param("deception.adjudication_retries", 2)

        tasks = [{"prompt": self._format(prompts["user"], pkg),
                  "system": prompts["system"], "temperature": 0.0, "retries": retries}
                 for pkg in evidence_packages]
        responses = parallel_chat_json_with_retry(tasks)

        verdicts = []
        for pkg, res in zip(evidence_packages, responses):
            verdicts.append(self._parse(pkg["doc_id"], res))
        self.log(awm, "adjudicate_batch",
                 f"n={len(verdicts)} verdicts=" +
                 ",".join(v["verdict"][:4] for v in verdicts))
        return verdicts

    # ---------------- prompt 组装与解析 ----------------
    @staticmethod
    def _format(user_template: str, pkg: dict) -> str:
        meta = pkg.get("metadata", {})
        ac = pkg.get("attribution_claim", {})
        return user_template.format(
            source=meta.get("source", ""),
            publish_date=meta.get("publish_date", ""),
            attribution_summary=meta.get("attribution_summary", ""),
            content_excerpt=meta.get("content_excerpt", ""),
            attribution_name=ac.get("canonical_name") or "(unattributed)",
            attribution_aliases=", ".join(ac.get("aliases", []) or []) or "(none)",
            attribution_description=ac.get("description", ""),
            my_behavior_tokens=json.dumps(pkg.get("my_behavior_tokens", [])),
            attribution_allies=json.dumps(pkg.get("attribution_allies", []),
                                          ensure_ascii=False),
            ally_behavior_union=json.dumps(pkg.get("ally_behavior_tokens_union", [])),
            behavior_overlap_ratio=pkg.get("behavior_overlap_ratio", ""),
            private_token_rate=pkg.get("private_token_rate", ""),
            near_duplicate_info=json.dumps(pkg.get("near_duplicate_partner") or {},
                                           ensure_ascii=False) or "(none)",
            divergence_info=json.dumps(pkg.get("deterministic_divergence") or {},
                                       ensure_ascii=False) or "(none)",
        )

    @staticmethod
    def _parse(doc_id: str, res: dict) -> dict:
        verdict = (res or {}).get("verdict", "")
        if verdict not in ("credible", "suspicious", "deceptive"):
            verdict = "credible"
        return {
            "doc_id": doc_id,
            "verdict": verdict,
            "justification": (res or {}).get("justification", ""),
        }
