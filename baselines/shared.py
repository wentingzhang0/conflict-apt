"""
Baseline 共享工具 — 所有 baseline 统一使用项目基础设施
- LLM: src.utils.llm_client.chat()
- 检索: src.retrieval.weaviate_retriever.retrieve()
- 配置: src.utils.settings
"""
import sys
import os
import re

# 确保项目根目录可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.llm_client import chat
from src.retrieval import weaviate_retriever
from src.utils.settings import get_global_top_k, get_global


# ── TAA 任务 prompt ──────────────────────────────────────────────────
TAA_SYSTEM_PROMPT = (
    "You are a cybersecurity expert specializing in threat intelligence. "
    "Your task is to analyze APT threat reports and attribute them to known threat actor groups."
)

TAA_USER_TEMPLATE = (
    "You are given a threat report that describes a cyber incident. "
    "Any direct mentions of the threat actor group, specific campaign names, or malware names "
    "responsible have been replaced with [PLACEHOLDER]. Your task is to analyze the report and "
    "attribute the incident to a known threat actor based on the techniques, tactics, procedures (TTPs), "
    "and any other relevant information described. "
    "Please provide the name of the threat actor you believe is responsible.\n\n"
    "Threat Report:\n{report}\n\n"
    "Please output ONLY the threat actor name in the format: <answer>Threat Actor Name</answer>\n"
    "Do NOT include any explanation or additional text outside the <answer> tags."
)

# 带 RAG 上下文的 prompt
TAA_RAG_USER_TEMPLATE = (
    "You are given a threat report that describes a cyber incident. "
    "Any direct mentions of the threat actor group, specific campaign names, or malware names "
    "responsible have been replaced with [PLACEHOLDER]. Your task is to analyze the report and "
    "attribute the incident to a known threat actor based on the techniques, tactics, procedures (TTPs), "
    "and any other relevant information described.\n\n"
    "--- Retrieved Reference Reports ---\n{context}\n\n"
    "--- Threat Report ---\n{report}\n\n"
    "Based on both the retrieved reference reports and the threat report, "
    "please output ONLY the name of the threat actor you believe is responsible "
    "in the format: <answer>Threat Actor Name</answer>\n"
    "Do NOT include any explanation or additional text outside the <answer> tags."
)


def retrieve(query: str, top_k: int = None) -> list[dict]:
    """统一的检索接口（top_k 默认从全局配置读取）"""
    if top_k is None:
        top_k = get_global_top_k()
    return weaviate_retriever.retrieve(query, top_k=top_k)


def format_context(docs: list[dict], max_per_doc: int = None) -> str:
    """将检索结果格式化为上下文字符串（max_per_doc 默认从全局配置读取）"""
    if max_per_doc is None:
        max_per_doc = get_global("context_max_per_doc", 1500)
    parts = []
    for i, doc in enumerate(docs):
        title = doc.get("report_title", doc.get("title", "Unknown"))
        section = doc.get("section_title", "")
        content = doc.get("content", doc.get("text", ""))
        if len(content) > max_per_doc:
            content = content[:max_per_doc] + "..."
        parts.append(f"[{i+1}] {title} ({section}):\n{content}")
    return "\n\n".join(parts)


def parse_prediction(text: str) -> str:
    """从 LLM 输出中提取预测的威胁组织名"""
    # 尝试 <answer>...</answer>
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: Answer: XXX
    m = re.search(r'(?:Final\s+)?Answer[\s:]*\s*(.+)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 最后一行非空内容
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else "Unknown"


# ── MCQ 任务 prompt + 工具 ──────────────────────────────────────────

MCQ_SYSTEM_PROMPT = (
    "You are a cybersecurity expert specializing in threat intelligence. "
    "Your task is to answer multiple-choice questions about cybersecurity topics."
)

MCQ_USER_TEMPLATE = (
    "Answer the following cybersecurity multiple-choice question.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Choose the best option. Output ONLY the letter (A, B, C, or D) "
    "in the format: <answer>X</answer>"
)

MCQ_RAG_USER_TEMPLATE = (
    "Answer the following cybersecurity multiple-choice question "
    "using both the retrieved reference documents and your knowledge.\n\n"
    "--- Retrieved Reference Documents ---\n{context}\n\n"
    "--- Question ---\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Choose the best option. Output ONLY the letter (A, B, C, or D) "
    "in the format: <answer>X</answer>"
)


def format_mcq_options(row: dict) -> str:
    """从数据集行格式化为选项文本"""
    return "\n".join([
        f"A) {row['Option A']}",
        f"B) {row['Option B']}",
        f"C) {row['Option C']}",
        f"D) {row['Option D']}",
    ])


def parse_mcq_prediction(text: str) -> str:
    """从 LLM 输出中提取 MCQ 选项字母 (A/B/C/D)"""
    # 尝试 <answer>...</answer>
    m = re.search(r'<answer>\s*([A-D])\s*</answer>', text, re.DOTALL)
    if m:
        return m.group(1).upper()
    # fallback: 找最后一个独立的 A-D
    matches = re.findall(r'\b([A-D])\b', text.upper())
    if matches:
        return matches[-1]
    return "A"


# ── RCM 任务 prompt + 工具 ──────────────────────────────────────────

RCM_SYSTEM_PROMPT = (
    "You are a cybersecurity vulnerability analyst specializing in CWE (Common Weakness Enumeration) classification."
)

RCM_USER_TEMPLATE = (
    "Analyze the following CVE vulnerability description and determine the most appropriate CWE category.\n"
    "Focus on the ROOT CAUSE of the vulnerability, not the attack vector or impact.\n\n"
    "Vulnerability Description:\n{report}\n\n"
    "Common CWE categories:\n"
    "Memory Safety: CWE-119, CWE-120, CWE-125, CWE-787, CWE-416, CWE-476, CWE-190\n"
    "Injection: CWE-79, CWE-89, CWE-78, CWE-94, CWE-77\n"
    "Input Validation: CWE-20, CWE-22\n"
    "Auth: CWE-287, CWE-306, CWE-862, CWE-863, CWE-269, CWE-732\n"
    "Data Handling: CWE-502, CWE-400, CWE-257, CWE-200\n"
    "Other: CWE-352, CWE-264, CWE-261\n\n"
    "Output ONLY the CWE ID in the format: <answer>CWE-XXX</answer>"
)

RCM_RAG_USER_TEMPLATE = (
    "Analyze the following CVE vulnerability description and determine the most appropriate CWE category.\n"
    "Focus on the ROOT CAUSE of the vulnerability, not the attack vector or impact.\n\n"
    "--- Retrieved Reference Documents ---\n{context}\n\n"
    "--- Vulnerability Description ---\n{report}\n\n"
    "Common CWE categories:\n"
    "Memory Safety: CWE-119, CWE-120, CWE-125, CWE-787, CWE-416, CWE-476, CWE-190\n"
    "Injection: CWE-79, CWE-89, CWE-78, CWE-94, CWE-77\n"
    "Input Validation: CWE-20, CWE-22\n"
    "Auth: CWE-287, CWE-306, CWE-862, CWE-863, CWE-269, CWE-732\n"
    "Data Handling: CWE-502, CWE-400, CWE-257, CWE-200\n"
    "Other: CWE-352, CWE-264, CWE-261\n\n"
    "Based on both the retrieved reference documents and the vulnerability description, "
    "output ONLY the CWE ID in the format: <answer>CWE-XXX</answer>"
)


def parse_rcm_prediction(text: str) -> str:
    """从 LLM 输出中提取 CWE 编号"""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        return _extract_cwe(m.group(1))
    matches = re.findall(r'CWE[- ](\d+)', text, re.IGNORECASE)
    if matches:
        return f"CWE-{matches[-1]}"
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        return _extract_cwe(lines[-1])
    return ""


def _extract_cwe(text: str) -> str:
    m = re.search(r'CWE[- ](\d+)', text, re.IGNORECASE)
    if m:
        return f"CWE-{m.group(1)}"
    m = re.search(r'\b(\d{1,4})\b', text)
    if m:
        return f"CWE-{m.group(1)}"
    return text.strip()


# ── ATE 任务 prompt + 工具 ──────────────────────────────────────────
# ATE prompt：与主方法 src/agents/prompts/ate/reporter.yaml 对齐到同一套「答案生成
# 指令」（控制变量）——相同 system / precision-first 语气 / 平台检测 / Enterprise+
# Mobile 双技术 reference / Rules（显式描述才输出、不确定就 OMIT、Enterprise 4-7 且
# ≤9、Mobile ≤18）。与主方法的唯一差异：主方法多一个 Structured Analysis 上下文块
#（abp_summary / hypotheses_summary / refutation_summary / conflicts），baseline 仅
# 注入检索文档（{context}）或无上下文。
# {report} 由 run_ate.py 的 caller 拼上 [Platform: ...] 前缀（与主方法
# src/tasks/ate.py:format_query 一致）；后处理平台截断（Enterprise≤8 / Mobile≤20）
# 在 worker 层统一施加，与主方法 parse_output 对称。

ATE_SYSTEM_PROMPT = (
    "You are a cybersecurity threat intelligence analyst specializing in MITRE ATT&CK technique identification."
)

# 与主方法 reporter.yaml 共用的指令骨架。三段拼接为无 RAG / RAG 两个模板，
# 单份 reference 避免两版漂移再次破坏控制变量。
_ATE_PREAMBLE = (
    "Extract MITRE ATT&CK technique IDs from the TARGET malware/threat-tool description "
    '(the "Description" section at the bottom).\n'
    "Precision matters more than coverage: extract ONLY techniques whose behavior is explicitly described in the TARGET.\n\n"
    "**STEP 1 — Detect the platform.** Read the [Platform: ...] tag at the top of the TARGET Description.\n"
    "- If [Platform: Enterprise] (or Windows/Linux/macOS desktop/server): use the ENTERPRISE reference below.\n"
    "- If [Platform: Mobile] (Android/iOS): use the MOBILE reference below. Mobile malware maps to the\n"
    "  ATT&CK for Mobile matrix (IDs typically T14xx–T16xx). Do NOT output Enterprise/Desktop techniques\n"
    "  (T10xx–T13xx such as T1059, T1083, T1070) for a Mobile target.\n\n"
)

_ATE_REFERENCE = (
    "ENTERPRISE technique reference — use ONLY for Enterprise platform:\n"
    "HTTP/HTTPS communication → T1071 | Encrypted channel → T1573 | File discovery → T1083\n"
    "Indicator removal → T1070 | Command execution → T1059 | Process discovery → T1057\n"
    "System info discovery → T1082 | Process injection → T1055 | Credential dumping → T1003\n"
    "Scheduled task → T1053 | Screen capture → T1113 | Keylogging → T1056\n"
    "Remote services → T1021 | Phishing → T1566 | Registry modify → T1112\n"
    "Exfiltration → T1041 | Privilege escalation → T1068 | Data encoding → T1132\n"
    "Obfuscation → T1027 | Deobfuscation/decode → T1140 | Ingress tool transfer → T1105\n"
    "Data from local system → T1005 | Archive/compress data → T1560 | Boot/logon autostart → T1547\n"
    "Network config discovery → T1016 | Data staged → T1074 | Data from network share → T1039\n\n"
    "MOBILE technique reference (ATT&CK for Mobile) — use ONLY for [Platform: Mobile]:\n"
    "App-layer network communication → T1437 | Audio capture → T1406 | Screen/video capture → T1407\n"
    "Location tracking → T1430 | App credentials → T1418 | Credential access → T1426\n"
    "Clipboard data → T1512 | Input capture (keylog) → T1517 | Data from local device → T1533\n"
    "Kernel/root exploitation → T1474 | Kernel module → T1626 | Account discovery → T1629\n"
    "Notifications access → T1633 | Hidden artifacts → T1646 | Container/resource → T1623\n"
    "App auto-start → T1398 | Exploitation for privilege escalation → T1404 | System service → T1544\n\n"
)

_ATE_RULES = (
    "Rules:\n"
    "1. Output ONLY techniques whose behavior is EXPLICITLY described in the TARGET Description, mapped via the\n"
    "   reference matching the TARGET's platform. Generic capabilities (e.g. \"has various modules\", \"fully-featured\")\n"
    "   do NOT justify a technique — the TARGET must describe the concrete action.\n"
    "2. Do NOT copy techniques from the retrieved documents or analyst suggestions unless the TARGET actually\n"
    "   describes that behavior. When uncertain, OMIT it.\n"
    "3. Quantity by platform — Enterprise: output only clearly-described techniques, usually 4-7, NEVER more than 9.\n"
    "   Mobile: the target often describes many techniques; output all clearly described (up to ~18).\n"
    "   Prefer PRECISION over coverage; do NOT pad with generic techniques.\n"
    "4. Do not repeat IDs. No subtechnique IDs. Output comma-separated main IDs only.\n\n"
    "Output ONLY the comma-separated MITRE technique IDs in the format: <answer>T1XXX, T1XXX</answer>\n"
    "Do NOT include any explanation or additional text outside the <answer> tags."
)

# 无 RAG 版（{report} 由 caller 拼上 [Platform: ...] 前缀）
ATE_USER_TEMPLATE = (
    _ATE_PREAMBLE
    + _ATE_REFERENCE
    + "--- Description (the TARGET software — your ONLY source of truth) ---\n"
    "{report}\n\n"
    + _ATE_RULES
)

# RAG 版（{context}=检索文档；{report} 由 caller 拼上 [Platform: ...] 前缀）
ATE_RAG_USER_TEMPLATE = (
    _ATE_PREAMBLE
    + _ATE_REFERENCE
    + "--- Retrieved Reference Documents (related reports — mapping context only) ---\n"
    "{context}\n\n"
    + "--- Description (the TARGET software — your ONLY source of truth) ---\n"
    "{report}\n\n"
    + _ATE_RULES
)


def parse_ate_prediction(text: str) -> str:
    """从 LLM 输出中提取 ATT&CK 技术 ID 列表"""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        return _normalize_tids(m.group(1))
    # fallback: 从全文提取
    return _normalize_tids(text)


def _normalize_tids(text: str) -> str:
    """提取并标准化 T-ID 列表（去子技术，逗号分隔，去重保序）"""
    matches = re.findall(r'(T\d{3,4})(?:\.\d{3})?', text)
    seen = set()
    result = []
    for tid in matches:
        if tid not in seen:
            seen.add(tid)
            result.append(tid)
    return ", ".join(result)


# ── 任务 prompt 统一选择器 ──────────────────────────────────────────

_TASK_PROMPTS = {
    "taa": {
        "system": TAA_SYSTEM_PROMPT,
        "user": TAA_USER_TEMPLATE,
        "rag_user": TAA_RAG_USER_TEMPLATE,
        "parse": parse_prediction,
    },
    "mcq": {
        "system": MCQ_SYSTEM_PROMPT,
        "user": MCQ_USER_TEMPLATE,
        "rag_user": MCQ_RAG_USER_TEMPLATE,
        "parse": parse_mcq_prediction,
    },
    "rcm": {
        "system": RCM_SYSTEM_PROMPT,
        "user": RCM_USER_TEMPLATE,
        "rag_user": RCM_RAG_USER_TEMPLATE,
        "parse": parse_rcm_prediction,
    },
    "ate": {
        "system": ATE_SYSTEM_PROMPT,
        "user": ATE_USER_TEMPLATE,
        "rag_user": ATE_RAG_USER_TEMPLATE,
        "parse": parse_ate_prediction,
    },
}


def get_task_config(task_type: str = "taa") -> dict:
    """获取任务类型的 prompt 配置：system, user, rag_user, parse"""
    return _TASK_PROMPTS.get(task_type, _TASK_PROMPTS["taa"])
