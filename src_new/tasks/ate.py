"""
ATE TaskAdapter — Attack Technique Extraction 攻击技术提取任务
从恶意软件描述中提取 MITRE ATT&CK 技术 ID
"""
import re
from src_new.tasks.base import TaskAdapter

# 平台特化截断上限：企业样本 GT 少（4-7），硬截断抑制枚举通用技术；移动 GT 多（可达19），放宽
_MAX_TIDS_ENTERPRISE = 8
_MAX_TIDS_MOBILE = 20


class ATETask(TaskAdapter):
    task_type = "ate"
    task_instruction = (
        "Analyze the following malware/threat tool description and extract all applicable "
        "MITRE ATT&CK technique IDs. Focus on the specific attack patterns and techniques "
        "described in the text."
    )
    answer_format = "Comma-separated MITRE technique IDs (e.g., T1071, T1573, T1083)"

    def __init__(self):
        self._platform = "Enterprise"

    def format_query(self, description: str, platform: str = "Enterprise", **kwargs) -> str:
        """拼接 platform 信息到 description 前面"""
        self._platform = (platform or "Enterprise").strip()
        return f"[Platform: {self._platform}]\n{description}"

    def parse_output(self, conclusion: str) -> str:
        """从最终结论中提取逗号分隔的 T-ID，按平台硬截断（企业≤9 抑制枚举，移动≤20）"""
        # 1. <answer>...</answer>
        m = re.search(r'<answer>(.*?)</answer>', conclusion, re.DOTALL)
        text = m.group(1) if m else conclusion
        if not m:
            # 2. 找最后一行含 T-XXX 模式的
            lines = [l.strip() for l in conclusion.strip().split('\n') if l.strip()]
            for line in reversed(lines or []):
                if re.findall(r'T\d{3,4}(?:\.\d{3})?', line):
                    text = line
                    break
        tids = _extract_tids(text)
        limit = _MAX_TIDS_MOBILE if self._platform.lower() == "mobile" else _MAX_TIDS_ENTERPRISE
        return ", ".join(tids[:limit])


def _extract_tids(text: str) -> list:
    """提取并标准化 T-ID 列表（去子技术，去重保序）"""
    matches = re.findall(r'(T\d{3,4})(?:\.\d{3})?', text)
    seen = set()
    result = []
    for tid in matches:
        tid = tid.upper()
        if tid not in seen:
            seen.add(tid)
            result.append(tid)
    return result
