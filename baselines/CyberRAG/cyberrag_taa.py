"""
BL4: CyberRAG — Ontology-Aware RAG for Cybersecurity QA (adapted for TAA)

Original: https://github.com/ChengshuaiZhao0/CyberRAG
Paper: "Ontology-Aware RAG for Improved Question-Answering in Cybersecurity Education" (IEEE BigData 2025)

Core algorithm:
  1. Retrieve top-k documents via embedding similarity
  2. Generate answer with LLM using retrieved context
  3. Ontology validation: LLM checks whether answer aligns with domain ontology
  4. If validation fails, regenerate with ontology-enriched prompt

Adaptations:
  - Retrieval: Weaviate (shared) instead of Contriever + CSV KB
  - LLM: project chat() (shared Qwen2.5-7B) instead of HuggingFace Llama-3-8B
  - Embedding: project BGE-M3 (shared) via Weaviate instead of Contriever
  - Output: TAA attribution format <answer>Actor</answer>
  - All shared params (top_k etc.) from global settings
"""
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from baselines.shared import (
    chat, retrieve, format_context, get_task_config,
)
from src.utils.settings import get_global_top_k, get_global


# ═══════════════════════════════════════════════════════════════
# Ontology Validation Prompt (CyberRAG's core contribution)
# ═══════════════════════════════════════════════════════════════

_VALIDATION_PROMPT = """You are a cybersecurity threat intelligence validator.

Question: {question}
Generated Answer: {answer}

Ontology context — known APT groups and their typical characteristics:
{ontology}

Task: Does the generated answer correctly identify a threat actor that is
consistent with the evidence and the known APT ontology?

Return JSON:
{{"pass": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

_REGENERATION_PROMPT = """You are a cybersecurity expert specializing in threat intelligence attribution.

PREVIOUS ANSWER (rejected by validation): {previous_answer}
VALIDATION REASON: {validation_reason}

--- Retrieved Reference Reports ---
{context}

--- Threat Report ---
{report}

The previous answer did not pass domain validation. Consider the validation
feedback and the evidence more carefully. Identify the threat actor.

Output format: <answer>Threat Actor Name</answer>"""

# Brief APT ontology for validation (subset of well-known groups)
_APT_ONTOLOGY = """
Known APT groups and their characteristics:
- APT28 (Fancy Bear): Russian GRU, targets government/military, uses Spear-phishing, X-Agent, X-Tunnel
- APT29 (Cozy Bear): Russian FSB/SVR, targets government/foreign policy, uses Hammmertoss, CloudDuke
- Lazarus Group: North Korean, targets financial/crypto, uses Manrai, Dreamjob, APPLEJUICE
- Kimsuky: North Korean, targets South Korea government/defense, uses BabyShark, HeatPixel
- MUSTANG PANDA (Stately Taurus): Chinese, targets SE Asia/Europe government, uses PlugX, HOPLIGHT
- APT41: Chinese, targets gaming/telecom, uses Cobalt Strike, PlugX
- MuddyWater (MERCURY): Iranian, targets Middle East government, uses POWERSTATS, PhonyC2
- OilRig (APT34): Iranian, targets Middle East finance/government, uses Helminth, Poisonton
- SideCopy: Pakistani, targets Indian government/military, uses Allakore RAT, Action RAT
- Gamaredon: Russian-linked, targets Ukrainian government, uses Pteranodon, USB spread
- Turla (Snake): Russian, targets government/embassies, uses Snake malware, Epic Turla
- UNC2452 (Nobelium): Russian SVR-linked, supply-chain attacks, uses SolarWinds backdoor
- APT33 (Elfin): Iranian, targets energy/aerospace, uses Shapeshift, TurnedUp
- APT37 (Reaper): North Korean, targets South Korea, uses RUCHIKA, DOGCALL
- APT19: Chinese, targets US law firms/tech, uses HTTPCUT, BlackCoffee
- Confucius: Indian, targets Pakistani government/military
- CharmingCypress (APT35/Charming Kitten): Iranian, targets academia/policy
- Dark Caracal: Lebanese, targets military/government, uses Pallas malware
- Bahamut: Indian, targets Middle East/South Asia, uses custom Android spyware
- COLDRIVER: Russian, targets NGOs/journalists, uses SPOLIED identities
- Bitter APT: South Asian, targets Chinese/Pakistani government
- Dragonfly: Russian, targets energy sector, uses HAVEX, BlackEnergy
- Andariel: North Korean (Lazarus sub-group), targets South Korea, uses DTrack
- CHRYSENE: Iranian-linked, targets Middle East
- Diamond Sleet: North Korean, targets defense/tech, uses Trojanized software
- Sharp Panda: Chinese-linked, targets SE Asian government
- Mint Sandstorm: Iranian, targets US critical infrastructure
- OilRig: Iranian, targets Middle East finance/government
- APT36 (Transparent Tribe): Pakistani, targets Indian government/military
- APT35 (Charming Kitten): Iranian, targets academia/policy/government
- APT31 (Hurricane Panda): Chinese, targets foreign policy entities
"""


# ═══════════════════════════════════════════════════════════════
# ATE Evidence-Based Validation Prompt
# ═══════════════════════════════════════════════════════════════

_ATE_VALIDATION_PROMPT = """You are validating ATT&CK technique extraction results.

Description:
{description}

Extracted Techniques: {tids}

For each technique, determine if it is supported by behavioral evidence in the description.
Also check if the description mentions any attack behaviors that were NOT extracted.

Known technique behaviors for reference:
- T1071: HTTP/HTTPS/application layer communication
- T1573: encrypted channel communication
- T1083: file and directory discovery
- T1070: indicator removal / log deletion / clearing tracks
- T1059: command-line / scripting interpreter execution
- T1057: process discovery
- T1082: system information discovery
- T1055: process injection / DLL injection
- T1003: OS credential dumping
- T1053: scheduled task / job
- T1113: screen capture
- T1056: input capture / keylogging
- T1021: remote services
- T1566: phishing
- T1112: modify registry
- T1041: exfiltration over C2 channel
- T1068: exploitation for privilege escalation
- T1132: data encoding
- T1027: obfuscated files / information
- T1140: deobfuscation / decode files
- T1105: ingress tool transfer / download files

Return JSON:
{{"pass": true/false, "unsupported": ["T1XXX", ...], "missing_behaviors": "description of uncaptured behaviors", "reason": "brief explanation"}}"""


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════

class CyberRAG:
    """CyberRAG baseline adapted for TAA task.

    Pipeline:
        1. Retrieve from Weaviate (shared top_k)
        2. Generate attribution answer with LLM + retrieved context
        3. Validate answer against APT ontology (CyberRAG's core step)
        4. If validation fails, regenerate with ontology feedback
    """

    def __init__(self, top_k: int = None, max_retries: int = 1, task_type: str = "taa"):
        self.top_k = top_k or get_global_top_k()
        self.max_retries = max_retries  # regeneration attempts on validation failure
        self.task_type = task_type

    def run(self, report_text: str) -> str:
        """Run CyberRAG pipeline."""
        cfg = get_task_config(self.task_type)

        # Step 1: Retrieve
        docs = retrieve(report_text, top_k=self.top_k)

        if not docs:
            response = chat(
                prompt=cfg["user"].format(report=report_text),
                system=cfg["system"], temperature=0.0,
            )
            return cfg["parse"](response)

        # Step 2: Generate initial answer
        context = format_context(docs)
        prompt = cfg["rag_user"].format(context=context, report=report_text)
        response = chat(prompt=prompt, system=cfg["system"], temperature=0.0)
        answer = cfg["parse"](response)

        # Step 3: Ontology validation (CyberRAG's core contribution)
        for retry in range(self.max_retries):
            validation = self._validate(report_text, answer)

            if validation.get("pass", False):
                return answer

            # Step 4: Regenerate with ontology feedback
            reason = validation.get("reason", "Answer inconsistent with domain validation")
            if self.task_type == "ate":
                # ATE 证据回溯重生成：基于验证反馈精炼提取结果
                unsupported = validation.get("unsupported", [])
                missing = validation.get("missing_behaviors", "")
                feedback_parts = []
                if unsupported:
                    feedback_parts.append(
                        f"The following T-IDs lack behavioral evidence and may be hallucinated: "
                        f"{', '.join(unsupported)}. Remove them unless you find clear evidence."
                    )
                if missing:
                    feedback_parts.append(
                        f"The description mentions behaviors not yet captured: {missing}. "
                        f"Try to identify corresponding ATT&CK techniques."
                    )
                regen_prompt = (
                    f"PREVIOUS ANSWER (needs revision): {answer}\n"
                    f"VALIDATION FEEDBACK: {'; '.join(feedback_parts) or reason}\n\n"
                    "Re-extract MITRE ATT&CK technique IDs based on the description and retrieved evidence.\n"
                    "Only include techniques with clear behavioral evidence. Be thorough but precise.\n\n"
                    + cfg["rag_user"].format(context=context, report=report_text[:6000])
                )
            elif self.task_type == "taa":
                regen_prompt = _REGENERATION_PROMPT.format(
                    previous_answer=answer,
                    validation_reason=reason,
                    context=context,
                    report=report_text[:6000],
                )
            else:
                # 非 TAA 任务：用任务自身的 rag_user prompt 重新生成
                regen_prompt = (
                    f"PREVIOUS ANSWER (rejected: {reason}):\n{answer}\n\n"
                    f"Try again. Be thorough and precise.\n\n"
                    + cfg["rag_user"].format(context=context, report=report_text[:6000])
                )
            response = chat(prompt=regen_prompt, system=cfg["system"], temperature=0.0)
            answer = cfg["parse"](response)

        return answer

    def _validate(self, question: str, answer: str) -> dict:
        """Validate answer against domain ontology (CyberRAG's Step 3)."""
        # ATE 任务：证据回溯验证（LLM 检查每个 T-ID 是否有行为依据）
        if self.task_type == "ate":
            tids = re.findall(r'T\d{3,4}', answer, re.IGNORECASE)
            if not tids:
                return {"pass": False, "confidence": 0.3, "reason": "No valid ATT&CK technique IDs found"}

            prompt = _ATE_VALIDATION_PROMPT.format(
                description=question[:3000],
                tids=", ".join(tids),
            )
            raw = chat(
                prompt=prompt,
                system="You are a cybersecurity technique extraction validator. Return only JSON.",
                temperature=0.0,
                max_tokens=200,
            )
            try:
                m = re.search(r'\{.*\}', raw, re.S)
                if m:
                    result = json.loads(m.group(0))
                    is_pass = bool(result.get("pass", False))
                    unsupported = result.get("unsupported", [])
                    missing = result.get("missing_behaviors", "")
                    reason_parts = []
                    if unsupported:
                        reason_parts.append(f"Unsupported T-IDs (no evidence): {', '.join(unsupported)}")
                    if missing:
                        reason_parts.append(f"Missing behaviors: {missing}")
                    return {
                        "pass": is_pass,
                        "confidence": float(result.get("confidence", 0.5)),
                        "unsupported": unsupported,
                        "missing_behaviors": missing,
                        "reason": "; ".join(reason_parts) if reason_parts else str(result.get("reason", "")),
                    }
            except Exception:
                pass

            return {"pass": True, "confidence": 0.5, "reason": "Validation parse failed, accepting as-is"}

        # RCM 任务：验证 CWE 格式
        if self.task_type == "rcm":
            cwes = re.findall(r'CWE[- ]?\d+', answer, re.IGNORECASE)
            if cwes:
                return {"pass": True, "confidence": 0.8, "reason": f"Found valid CWE ID: {cwes[0]}"}
            return {"pass": False, "confidence": 0.3, "reason": "No valid CWE ID found"}

        # TAA/MCQ 任务：LLM 本体验证 + APT 组织名 fallback
        prompt = _VALIDATION_PROMPT.format(
            question=question[:2000],
            answer=answer,
            ontology=_APT_ONTOLOGY,
        )

        raw = chat(
            prompt=prompt,
            system="You are a cybersecurity ontology validator. Return only JSON.",
            temperature=0.0,
            max_tokens=150,
        )

        try:
            m = re.search(r'\{.*\}', raw, re.S)
            if m:
                result = json.loads(m.group(0))  # noqa: F821
                return {
                    "pass": bool(result.get("pass", False)),
                    "confidence": float(result.get("confidence", 0.5)),
                    "reason": str(result.get("reason", "")),
                }
        except Exception:
            pass

        # Fallback: check if answer contains a known APT name
        answer_lower = answer.lower().strip()
        known_groups = [
            "apt28", "apt29", "apt33", "apt34", "apt35", "apt36", "apt37", "apt38", "apt41",
            "lazarus", "kimsuky", "mustang panda", "stately taurus", "turla", "gamaredon",
            "muddywater", "oilrig", "sidecopy", "unc2452", "confucius", "dark caracal",
            "bahamut", "coldriver", "bitter", "dragonfly", "andariel", "chrysene",
            "diamond sleet", "sharp panda", "mint sandstorm", "apt-c-36",
        ]
        for g in known_groups:
            if g in answer_lower:
                return {"pass": True, "confidence": 0.7, "reason": "Known APT group name"}

        return {"pass": False, "confidence": 0.3, "reason": "Answer not recognized as known APT group"}


import json  # needed for _validate JSON parsing
