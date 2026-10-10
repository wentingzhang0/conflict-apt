"""
BL2: ITER-RETGEN — 迭代检索增强生成
迭代: retrieve → generate → 用生成结果 refine query → retrieve → generate
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from baselines.shared import (
    chat, retrieve, format_context, get_task_config
)


class IterRetGen:
    def __init__(self, top_k: int = 10, iterations: int = 3, task_type: str = "taa"):
        self.top_k = top_k
        self.iterations = iterations
        self.task_type = task_type

    def run(self, report_text: str) -> str:
        """输入报告文本，返回预测结果"""
        cfg = get_task_config(self.task_type)
        system_prompt = cfg["system"] + (
            "\nYou will first generate an initial analysis, then refine it through "
            "multiple rounds of retrieval and generation."
        )

        all_docs = []
        current_query = report_text
        last_response = ""

        for t in range(self.iterations):
            # 1. Retrieve
            docs = retrieve(current_query, top_k=self.top_k)

            # 去重累积
            seen = {(d.get("source_file", ""), d.get("section_title", "")) for d in all_docs}
            for d in docs:
                key = (d.get("source_file", ""), d.get("section_title", ""))
                if key not in seen:
                    all_docs.append(d)
                    seen.add(key)

            # 2. Format context
            context = format_context(all_docs)

            # 3. Generate
            if all_docs:
                user_prompt = cfg["rag_user"].format(
                    context=context, report=report_text
                )
            else:
                user_prompt = cfg["user"].format(report=report_text)

            last_response = chat(user_prompt, system=system_prompt, temperature=0.0)

            # 4. Refine query
            if t < self.iterations - 1:
                current_query = f"{report_text}\n\nPrevious analysis: {last_response[:500]}"

        return cfg["parse"](last_response)
