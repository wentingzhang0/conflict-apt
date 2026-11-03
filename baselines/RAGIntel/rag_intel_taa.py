"""
BL5: RAGIntel — 适配版
原方法核心: Hybrid Retrieval (dense + BM25) + Flashrank Reranking
适配: Weaviate 混合检索 → Flashrank 重排 → LLM 生成

参考: Abeer Alhuthali, "RAGIntel: A Retrieval-Augmented Generation Approach
      for Cyber Threat Intelligence", 2024
      https://github.com/AbeerAlhuthali/RAGIntel
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from baselines.shared import (
    chat,
    retrieve,
    format_context,
    get_task_config,
)


class RAGIntel:
    """RAGIntel baseline adapted for multiple CTI tasks.

    Pipeline:
        1. Hybrid retrieval from Weaviate (top_n candidates, dense+BM25 RRF)
        2. Flashrank cross-encoder reranking → keep top_k
        3. LLM generation with reranked context
    """

    def __init__(self, top_k: int = 10, top_n: int = 20, task_type: str = "taa"):
        """
        Args:
            top_k: final number of documents after reranking
            top_n: initial number of candidates from hybrid retrieval
                   (should be >= top_k to give reranker room to work)
            task_type: task type (taa, mcq, rcm)
        """
        self.top_k = top_k
        self.top_n = max(top_n, top_k * 2)
        self.task_type = task_type
        self._reranker = None

    def _get_reranker(self):
        """Lazy-load Flashrank reranker (downloads model on first call)."""
        if self._reranker is None:
            try:
                from flashrank import Ranker, RerankRequest

                self._reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
            except ImportError:
                print("[RAGIntel] flashrank not installed, skipping reranking step")
                self._reranker = False
        return self._reranker if self._reranker is not False else None

    def _rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """Rerank documents using Flashrank cross-encoder."""
        ranker = self._get_reranker()
        if ranker is None or not docs:
            return docs[: self.top_k]

        from flashrank import RerankRequest

        passages = []
        for i, doc in enumerate(docs):
            title = doc.get("report_title", "")
            section = doc.get("section_title", "")
            content = doc.get("content", "")
            passages.append({
                "id": i,
                "text": f"{title} ({section}): {content}" if section else f"{title}: {content}",
                "meta": doc,
            })

        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)

        reranked = []
        for item in results[: self.top_k]:
            meta = item.get("meta", {})
            if not meta and "id" in item:
                idx = item["id"]
                if idx < len(docs):
                    meta = docs[idx]
            if meta:
                reranked.append(meta)

        return reranked if reranked else docs[: self.top_k]

    def run(self, report_text: str) -> str:
        """Run RAGIntel pipeline on a single sample."""
        cfg = get_task_config(self.task_type)

        # 1) Hybrid retrieval: get more candidates for reranking
        docs = retrieve(report_text, top_k=self.top_n)

        if not docs:
            prompt = cfg["user"].format(report=report_text)
            response = chat(prompt, system=cfg["system"])
            return cfg["parse"](response)

        # 2) Rerank with Flashrank cross-encoder
        reranked_docs = self._rerank(report_text, docs)

        # 3) Format context and generate answer
        context = format_context(reranked_docs)
        prompt = cfg["rag_user"].format(context=context, report=report_text)
        response = chat(prompt, system=cfg["system"])

        return cfg["parse"](response)
