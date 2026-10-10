"""
② 证据建图智能体 Evidence & Graph Agent（创新点 2 表征的推理部分，方案 4.4）

职责：把一份分析请求变成一张带符号证据图。
- mode=full：原始 query 检索 top_k；ABP 规范键抽取（含查询画像）；调度 core 做预标注 +
  确定性匹配（两档）+ 符号聚合（三态边）+ 查询锚点建边 + 传递性补全；写回
  evidence_set/behavior_profiles/signed_graph_edges/initial_doc_ids。
- mode=expand：反驳阶段，用 counterexample_query 在扩展池检索并显式排除
  excluded_doc_ids（初始已检索结果），保证反例是初始图尚未包含的新证据。

仅 ABP 抽取为 LLM 调用（O(k)）；检索、预标注、确定性匹配、符号聚合、传递性补全均为确定性。
"""
import re
import numpy as np

from src_new.agents.base import BaseAgent
from src_new.core.awm import AWM, Evidence, BehaviorProfile, CONFLICT_DIMENSIONS
from src_new.core.abp import extract_abp, compute_abp_embedding
from src_new.core import conflict, transitivity
from src_new.retrieval.weaviate_retriever import retrieve
from src_new.utils.llm_client import parallel_chat_json_with_retry
from src_new.utils.settings import (get_pipeline_param, get_global, get_prompt)


_HIGH_RELIABILITY_SOURCES = {
    "crowdstrike", "mandiant", "fireeye", "kaspersky", "symantec", "microsoft",
    "google", "palo alto", "unit42", "secureworks", "eset", "trend micro",
    "sophos", "checkpoint", "cisco", "talos", "mcafee", "bitdefender", "fortinet",
    "mitre", "cisa", "ncsc", "positive technologies", "group-ib", "f-secure",
    "sentinelone", "drweb",
}


class EvidenceGraphAgent(BaseAgent):
    name = "evidence_graph_agent"

    def __init__(self, top_k: int = None):
        self.top_k = top_k or get_global("top_k", 10)

    # ---------------- mode=full：首次建图 ----------------
    def run_full(self, awm: AWM) -> AWM:
        self.log(awm, "start_full", f"query len={len(awm.query)}")

        # TAA 用原始报告全文检索（行为/TTP 细节全，召回更贴近目标组织）；
        # ATE/MCQ/RCM 用 query_summary 检索（短描述更聚焦）。top_k / 检索模式等全局参数不变。
        if awm.task_type == "taa":
            retrieval_query = awm.query
        else:
            retrieval_query = (awm.query_summary or awm.query)
        results = retrieve(retrieval_query, top_k=self.top_k)
        self.log(awm, "retrieve", f"{len(results)} docs (taa=full, else=summary)")
        if not results:
            return awm

        for r in results:
            ev = Evidence(
                doc_id=r.get("source_file", "") or r.get("report_title", ""),
                source=r.get("report_title", ""),
                content=r.get("content", ""),
                section_title=r.get("section_title", ""),
            )
            meta = _assess_source(ev.source, ev.content)
            ev.reliability = meta["reliability"]
            ev.publish_date = meta["publish_date"]
            awm.evidence_set.append(ev)

        awm.initial_doc_ids = [e.doc_id for e in awm.evidence_set]

        # 正文内容嵌入（创新点 4 CORE-1 用：近重复检测的嵌入源。ABP 描述嵌入受
        # 抽取 LLM 风格影响大——同一攻击语料在不同骨干下 cos 0.95→0.72；content
        # 嵌入只依赖文本本身，骨干无关）
        try:
            from src_new.utils.embedding import get_embed_model, get_target_dimension
            embs = get_embed_model().embed_documents(
                [e.content[:4000] for e in awm.evidence_set])
            for e, v in zip(awm.evidence_set, embs):
                e.content_embedding = list(v[:get_target_dimension()])
        except Exception as ex:  # 嵌入服务不可用时 CORE-1 回退 ABP 嵌入
            self.log(awm, "content_embed_failed", str(ex)[:120])

        # ABP 规范键抽取（并行，单次 LLM/文档）
        self._extract_profiles(awm)
        # 查询画像（查询锚点 + 冲突判定）
        awm.query_profile = self._extract_query_profile(awm)

        # 确定性冲突判定 + 符号聚合 + 传递性补全（零 LLM）
        conflict.build_edges(awm)
        transitivity.infer_neutral_pairs(awm)
        awm.signed_graph_nodes = [e.doc_id for e in awm.evidence_set]

        self.log(awm, "graph_built",
                 f"edges={len(awm.signed_graph_edges)} negative={len(awm.get_negative_edges())}")
        return awm

    # ---------------- mode=expand：反例扩展池检索 ----------------
    def run_expand(self, awm: AWM, counterexample_query: str,
                   excluded_doc_ids: list[str]) -> list[Evidence]:
        """在扩展池检索反例候选（排除初始已检索结果）。
        返回新增证据列表（尚未建边；建边由 Controller 增量触发）。"""
        top_k_expand = get_pipeline_param("retrieval.top_k_expand", 20)
        self.log(awm, "start_expand",
                 f"excluded={len(excluded_doc_ids)} top_k_expand={top_k_expand}")

        results = retrieve(counterexample_query, top_k=top_k_expand)
        excluded = set(excluded_doc_ids or [])
        new_results = [r for r in results
                       if (r.get("source_file", "") or r.get("report_title", "")) not in excluded]
        self.log(awm, "expand_retrieve",
                 f"{len(results)} raw → {len(new_results)} after excluding initial")

        new_evidence = []
        for r in new_results:
            ev = Evidence(
                doc_id=r.get("source_file", "") or r.get("report_title", ""),
                source=r.get("report_title", ""),
                content=r.get("content", ""),
                section_title=r.get("section_title", ""),
            )
            meta = _assess_source(ev.source, ev.content)
            ev.reliability = meta["reliability"]
            ev.publish_date = meta["publish_date"]
            new_evidence.append(ev)
        return new_evidence

    # ---------------- ABP 抽取（证据 + 查询） ----------------
    def _extract_profiles(self, awm: AWM):
        prompts = get_prompt("abp")
        text_max = get_pipeline_param("abp_extraction.report_text_max_length", 4000)
        retries = get_pipeline_param("abp_extraction.retries", 3)

        tasks = [{"prompt": prompts["user"].format(text=e.content[:text_max]),
                  "system": prompts["system"], "temperature": 0.0, "retries": retries}
                 for e in awm.evidence_set]
        results = parallel_chat_json_with_retry(tasks) if tasks else []

        for ev, res in zip(awm.evidence_set, results):
            prof = BehaviorProfile(doc_id=ev.doc_id)
            if res:
                for d in CONFLICT_DIMENSIONS:  # attribution / timeline / behavior / campaign
                    raw = res.get(d, {})
                    dim = {}
                    if isinstance(raw, dict):
                        if "description" in raw:
                            dim["description"] = raw.get("description") or ""
                        if "canonical_key" in raw:
                            dim["canonical_key"] = raw.get("canonical_key") or {}
                    setattr(prof, d, dim)
                ev.attribution_summary = res.get("attribution_summary", "")
            prof.embedding = compute_abp_embedding(prof)
            awm.behavior_profiles.append(prof)

    def _extract_query_profile(self, awm: AWM) -> BehaviorProfile:
        text_max = get_pipeline_param("abp_extraction.report_text_max_length", 4000)
        text = awm.query_summary or awm.query
        prof = extract_abp("query", text[:text_max])
        prof.embedding = compute_abp_embedding(prof)
        return prof


def _assess_source(source_title: str, content: str) -> dict:
    """启发式来源评估：正则日期 + 厂商白名单（零 LLM）。

    创新点 4（更新方案 §5.1）：本函数产出降格为**先验**可靠性——只回答"谁发布的"，
    不回答"这篇内容与全图交叉印证结构是否自洽"。后验由可信度审查传递
    （FLAG→ADJUDGE→REWEIGH，core/deception.py）在带符号图上联合估计；逻辑本身不变。"""
    date_match = re.search(
        r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        content[:3000], re.IGNORECASE)
    publish_date = date_match.group(1) if date_match else ""
    text_lower = (source_title + " " + content[:1000]).lower()
    reliability = 0.7
    for vendor in _HIGH_RELIABILITY_SOURCES:
        if vendor in text_lower:
            reliability = 0.9
            break
    return {"reliability": reliability, "publish_date": publish_date}
