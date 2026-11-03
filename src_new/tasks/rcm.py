"""
RCM TaskAdapter — CVE→CWE Root Cause Mapping 根因映射任务
"""
import re
from src_new.tasks.base import TaskAdapter


class RCMTask(TaskAdapter):
    task_type = "rcm"
    task_instruction = (
        "Analyze the following CVE vulnerability description and map it to the most appropriate "
        "CWE (Common Weakness Enumeration) category. Base your analysis on the root cause of the vulnerability."
    )
    answer_format = "CWE ID (e.g., CWE-79, CWE-416)"

    def format_query(self, description: str, **kwargs) -> str:
        return description

    def parse_output(self, conclusion: str) -> str:
        """从最终结论中提取 CWE 编号"""
        # 1. <answer>...</answer>
        m = re.search(r'<answer>(.*?)</answer>', conclusion, re.DOTALL)
        if m:
            return _extract_cwe(m.group(1))

        # 2. CWE-XXX 模式（找最后一个，通常是最确定的）
        matches = re.findall(r'CWE[- ](\d+)', conclusion, re.IGNORECASE)
        if matches:
            return f"CWE-{matches[-1]}"

        # 3. 最后一行
        lines = [l.strip() for l in conclusion.strip().split('\n') if l.strip()]
        if lines:
            return _extract_cwe(lines[-1])

        return ""


def _extract_cwe(text: str) -> str:
    """从文本中提取标准 CWE 编号"""
    m = re.search(r'CWE[- ](\d+)', text, re.IGNORECASE)
    if m:
        return f"CWE-{m.group(1)}"
    # 纯数字
    m = re.search(r'\b(\d{1,4})\b', text)
    if m:
        return f"CWE-{m.group(1)}"
    return text.strip()
