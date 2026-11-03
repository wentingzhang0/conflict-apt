"""
MCQ TaskAdapter — Multiple Choice Question 多选题任务
"""
import re
from src_new.tasks.base import TaskAdapter


class MCQTask(TaskAdapter):
    task_type = "mcq"
    task_instruction = (
        "Answer the following cybersecurity multiple-choice question. "
        "Select the correct option based on the collected evidence and behavior profiles."
    )
    answer_format = "option letter (A, B, C, or D)"

    def format_query(self, question: str, options: dict = None, **kwargs) -> str:
        """
        格式化 MCQ 输入。
        options 格式: {"A": "...", "B": "...", "C": "...", "D": "..."}
        """
        parts = [f"Question: {question}"]
        if options:
            for key in sorted(options.keys()):
                parts.append(f"{key}) {options[key]}")
        return "\n".join(parts)

    def parse_output(self, conclusion: str) -> str:
        """从最终结论中提取选项字母 (A/B/C/D)。

        关键：必须拒绝把实体名/技术 ID 内部的字母当作答案（如
        ``<answer>Lazarus Group</answer>`` 不能解析为 'A'）。只接受「独立的」
        单个 A-D 字母，与 baselines.shared.parse_mcq_prediction 对齐。
        """
        if not conclusion:
            return "A"
        # 1) <answer>X</answer>，X 为（近）单个字母 —— 与 MCQ reporter 契约一致
        m = re.search(r'<answer>\s*([A-D])\s*</answer>', conclusion, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # 2) <answer> 内显式给出选项，如 "Option B" / "B." / "B)"
        m = re.search(r'<answer>\s*(?:option\s*)?\(?([A-D])\b', conclusion, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # 3) 显式 "Answer: X" / "Final answer: X" / "ANSWER: X"
        m = re.search(r'\b(?:final\s+answer|answer)\s*[:\-]?\s*\(?([A-D])\b',
                      conclusion, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # 4) 兜底：最后一个独立的 A-D 字母（结论中通常只剩答案行）
        matches = re.findall(r'\b([A-D])\b', conclusion)
        if matches:
            return matches[-1].upper()
        return "A"
