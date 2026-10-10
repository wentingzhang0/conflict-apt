"""
TAA TaskAdapter — Threat Actor Attribution 归因任务
"""
import re
from src_new.tasks.base import TaskAdapter

# LLM 兜底提取的 prompt
_FALLBACK_EXTRACT_PROMPT = (
    "Extract ONLY the threat actor group name from the following text. "
    "Output just the name, nothing else.\n\n"
    "Text:\n{text}"
)


class TAATask(TaskAdapter):
    task_type = "taa"
    task_instruction = (
        "Identify the threat actor (APT group) responsible for the described cyber attack. "
        "Base your analysis on the collected evidence and behavior profiles."
    )
    answer_format = "threat actor name (use the most commonly recognized name, e.g. Lazarus Group, Turla, OilRig, Kimsuky)"

    def format_query(self, report_text: str, **kwargs) -> str:
        return report_text

    def parse_output(self, conclusion: str) -> str:
        """从最终结论中提取归因组织名，多层 fallback"""
        # 1. <answer>...</answer> 标签
        m = re.search(r'<answer>(.*?)</answer>', conclusion, re.DOTALL)
        if m and _valid(m.group(1)):
            return _standardize(m.group(1))

        # 2. ANSWER: xxx
        m = re.search(r'ANSWER:\s*(.+)', conclusion, re.IGNORECASE)
        if m and _valid(m.group(1)):
            return _standardize(m.group(1))

        # 3. Answer: xxx / Final Answer: xxx
        m = re.search(r'(?:Final\s+)?Answer[\s:]*\s*(.+)', conclusion, re.IGNORECASE)
        if m and _valid(m.group(1)):
            return _standardize(m.group(1))

        # 4. **BoldName** 形式的组织名（放宽匹配）
        m = re.search(r'\*\*([A-Z][A-Za-z0-9\- ]{2,40}?)\*\*', conclusion)
        if m and _valid(m.group(1)):
            return _standardize(m.group(1))

        # 5. LLM 兜底提取（最可靠但多一次调用）
        result = _llm_extract(conclusion)
        if _valid(result):
            return _standardize(result)

        # 6. 最后一行
        lines = [l.strip() for l in conclusion.strip().split('\n') if l.strip()]
        return _standardize(lines[-1]) if lines else "Unknown"


# ── 辅助函数 ──────────────────────────────────────────────────────────

_INVALID = {"unknown", "n/a", "none", "not found", "no relevant evidence",
            "no evidence", "unclear", "inconclusive", "cannot determine", ""}


def _valid(text: str) -> bool:
    """检查提取结果是否有效"""
    if not text:
        return False
    t = text.strip().lower().rstrip('.')
    return t not in _INVALID and len(t) > 1


def _standardize(raw: str) -> str:
    """标准化输出：去除括号别名、前缀、斜杠、多余后缀等"""
    s = raw.strip()
    # 去除 "APT group", "threat group" 等后缀
    s = re.sub(r'\s*(?:APT\s+)?(?:group|threat\s+group)\s*$', '', s, flags=re.IGNORECASE)
    # 去除括号及其中内容（如 "APT28 (Fancy Bear)" → "APT28"）
    s = re.sub(r'\s*\([^)]*\)', '', s)
    # 去除斜杠分隔的别名（如 "Cozy Bear/APT29" → "Cozy Bear"）
    s = s.split('/')[0].strip()
    # 去除常见前缀
    s = re.sub(r'^(?:ANSWER:\s*|<answer>\s*|Answer:\s*)', '', s, flags=re.IGNORECASE)
    return s.strip()


def _llm_extract(text: str) -> str:
    """用 LLM 从文本中提取威胁组织名"""
    from src_new.utils.llm_client import chat
    try:
        # 截断太长的文本
        snippet = text[:2000] if len(text) > 2000 else text
        result = chat(
            prompt=_FALLBACK_EXTRACT_PROMPT.format(text=snippet),
            system="You are a threat intelligence assistant. Output only the threat actor name.",
            temperature=0.0,
            max_tokens=50,
        )
        return result.strip()
    except Exception:
        return ""
