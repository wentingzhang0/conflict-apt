"""
BL6: Search-o1 — 检索增强推理
LLM 通过特殊 token 触发检索，系统返回结果后继续推理
核心 prompt 使用 shared.py 的标准模板（与其他 baseline 一致），仅在 system prompt 追加搜索工具说明
"""
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from baselines.shared import (
    chat, retrieve, format_context, get_task_config,
)

# 搜索触发 token（使用 [BEGIN/END] 避免与 Qwen2.5 特殊 token 冲突）
BEGIN_SEARCH_QUERY = "[BEGIN_SEARCH_QUERY]"
END_SEARCH_QUERY = "[END_SEARCH_QUERY]"
BEGIN_SEARCH_RESULT = "[BEGIN_SEARCH_RESULT]"
END_SEARCH_RESULT = "[END_SEARCH_RESULT]"

# 搜索工具说明（追加到 system prompt，核心 prompt 来自 shared.py）
_SEARCH_TOOL_SUFFIX = (
    "\n\nYou have a special search tool:\n"
    f"- To search: write {BEGIN_SEARCH_QUERY} your query here {END_SEARCH_QUERY}\n"
    f"- The system will return relevant documents in the format "
    f"{BEGIN_SEARCH_RESULT} ...results... {END_SEARCH_RESULT}\n\n"
    "You can search multiple times to gather more information before providing your final answer."
)


def extract_between(text: str, start_tag: str, end_tag: str) -> str:
    """提取两个 tag 之间的文本"""
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


class SearchO1:
    def __init__(self, top_k: int = 10, max_searches: int = 5, max_turns: int = 10, task_type: str = "taa"):
        self.top_k = top_k
        self.max_searches = max_searches
        self.max_turns = max_turns
        self.task_type = task_type

    def run(self, report_text: str) -> str:
        """输入报告文本，返回预测结果"""
        cfg = get_task_config(self.task_type)

        # 核心 prompt 使用 shared.py 的标准模板（与其他 baseline 完全一致）
        system_prompt = cfg["system"] + _SEARCH_TOOL_SUFFIX
        user_prompt = cfg["user"].format(report=report_text)

        full_prompt = user_prompt
        output_text = ""
        search_count = 0
        executed_queries = set()

        for turn in range(self.max_turns):
            # LLM 生成
            response = chat(full_prompt, system=system_prompt, temperature=0.0)

            if response is None:
                break

            full_prompt += response
            output_text += response

            # 检查是否有搜索请求
            search_query = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

            if search_query and search_count < self.max_searches and search_query not in executed_queries:
                # 执行检索
                docs = retrieve(search_query, top_k=self.top_k)
                executed_queries.add(search_query)
                search_count += 1

                # 格式化检索结果
                result_text = f"\n{BEGIN_SEARCH_RESULT}\n"
                if docs:
                    result_text += format_context(docs) + "\n"
                else:
                    result_text += "No relevant results found.\n"
                result_text += f"{END_SEARCH_RESULT}\n"

                full_prompt += result_text
                output_text += result_text

            elif search_query and search_count >= self.max_searches:
                limit_msg = f"\n{BEGIN_SEARCH_RESULT}\nMaximum search limit reached. Please provide your final answer.\n{END_SEARCH_RESULT}\n"
                full_prompt += limit_msg
                output_text += limit_msg

            elif search_query and search_query in executed_queries:
                dup_msg = f"\n{BEGIN_SEARCH_RESULT}\nYou already searched this query. Refer to previous results.\n{END_SEARCH_RESULT}\n"
                full_prompt += dup_msg
                output_text += dup_msg

            else:
                # 没有搜索请求，说明 LLM 认为已经足够
                break

        return cfg["parse"](output_text)
