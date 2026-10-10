"""
BL1: NaiveRAG — 最简单的 RAG baseline
query → Weaviate top-k → LLM generate → parse answer
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from baselines.shared import (
    chat, retrieve, format_context, get_task_config
)


class NaiveRAG:
    def __init__(self, top_k: int = 10, task_type: str = "taa"):
        self.top_k = top_k
        self.task_type = task_type

    def run(self, report_text: str) -> str:
        """输入报告文本，返回预测结果"""
        cfg = get_task_config(self.task_type)

        # 1. 用报告文本检索相关文档
        docs = retrieve(report_text, top_k=self.top_k)

        if not docs:
            prompt = cfg["user"].format(report=report_text)
            response = chat(prompt, system=cfg["system"], temperature=0.0)
            return cfg["parse"](response)

        # 2. 格式化上下文
        context = format_context(docs)

        # 3. LLM 生成答案
        prompt = cfg["rag_user"].format(context=context, report=report_text)
        response = chat(prompt, system=cfg["system"], temperature=0.0)

        return cfg["parse"](response)
