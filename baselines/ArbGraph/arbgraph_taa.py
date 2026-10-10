"""
BL3: ArbGraph — Claim-level Evidence Arbitration (adapted for TAA)

Original: https://github.com/1212Judy/ArbGraph
Core algorithm: atomic claim extraction → alignment → evidence graph → credibility arbitration → answer

Adaptations:
  - Retrieval: Weaviate (shared) instead of Wikipedia
  - LLM: project chat() (shared Qwen2.5-7B) instead of HFAdapter
  - Embedding: project BGE-M3 (shared) instead of sentence-transformers
  - Output: TAA attribution format <answer>Actor</answer>
  - All shared params (top_k etc.) from global settings
"""
import sys
import os
import re
import json
import math
import uuid
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import networkx as nx

from baselines.shared import (
    chat, retrieve, format_context, get_task_config,
)
from src.utils.settings import get_global_top_k, get_global


# ═══════════════════════════════════════════════════════════════
# Step 1: Atomic Claim Extraction (from ArbGraph atomization.py)
# ═══════════════════════════════════════════════════════════════

def extract_claims(text: str, source_id: str) -> List[Dict[str, Any]]:
    """Extract atomic claims from a document text via LLM."""
    prompt = f"""Decompose the following threat intelligence text into atomic factual claims.

Requirements:
- Each claim must express exactly one factual proposition about the attack
- Focus on: malware names, attack techniques, target sectors, C2 infrastructure, tools, attribution indicators
- Keep wording faithful to the source text
- Each claim must be independently verifiable

Text:
\"\"\"{text}\"\"\"

Return ONLY a JSON list:
[
  {{
    "claim": "atomic factual statement about the attack",
    "evidence": "supporting span from the text"
  }}
]"""

    raw = chat(prompt=prompt, system="You are a cybersecurity analyst.", temperature=0.0)
    claims = _parse_json_list(raw)

    result = []
    for idx, item in enumerate(claims):
        if not isinstance(item, dict):
            continue
        claim_text = str(item.get("claim", "")).strip()
        if not claim_text:
            continue
        result.append({
            "id": f"claim_{re.sub(r'[^a-zA-Z0-9_]', '_', source_id)[:30]}_{idx}_{uuid.uuid4().hex[:6]}",
            "text": claim_text,
            "evidence": str(item.get("evidence", "")).strip(),
            "source_id": source_id,
        })
    return result


# ═══════════════════════════════════════════════════════════════
# Step 2: Claim Alignment (from ArbGraph claim_alignment.py)
# ═══════════════════════════════════════════════════════════════

def align_and_merge_claims(
    claims: List[Dict[str, Any]],
    similarity_threshold: float = 0.88,
) -> List[Dict[str, Any]]:
    """Merge semantically similar claims using BGE-M3 embeddings."""
    if len(claims) <= 1:
        return claims

    from src.utils.embedding import get_embed_model, get_target_dimension

    embed_model = get_embed_model()
    texts = [c["text"] for c in claims]
    embeddings = np.array([e[:get_target_dimension()] for e in embed_model.embed_documents(texts)])

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms

    # Pairwise cosine similarity
    sim_matrix = embeddings @ embeddings.T

    # Union-Find for merging
    parent = list(range(len(claims)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            if sim_matrix[i, j] >= similarity_threshold:
                if _is_mergeable(claims[i]["text"], claims[j]["text"]):
                    union(i, j)

    # Group by root
    groups: Dict[int, List[int]] = {}
    for i in range(len(claims)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged = []
    for root, members in groups.items():
        rep = claims[members[0]].copy()
        source_ids = list(set(claims[m]["source_id"] for m in members))
        rep["source_ids"] = source_ids
        if len(members) > 1:
            rep["text"] = claims[root]["text"]  # keep representative
        merged.append(rep)

    return merged


def _is_mergeable(text_a: str, text_b: str) -> bool:
    """Heuristic check: don't merge claims that clearly disagree on numbers."""
    nums_a = set(re.findall(r'\b\d{4}\b', text_a))
    nums_b = set(re.findall(r'\b\d{4}\b', text_b))
    if nums_a and nums_b and not nums_a & nums_b:
        return False
    return True


# ═══════════════════════════════════════════════════════════════
# Step 3: Evidence Graph Construction (from ArbGraph evidence_graph.py)
# ═══════════════════════════════════════════════════════════════

def build_evidence_graph(
    claims: List[Dict[str, Any]],
    query: str,
    query_relevance_threshold: float = 0.3,
    pair_similarity_threshold: float = 0.75,
    max_support_edges: int = 60,
) -> nx.DiGraph:
    """Build evidence graph with support/contradiction edges."""
    from src.utils.embedding import get_embed_model, get_target_dimension

    embed_model = get_embed_model()
    dim = get_target_dimension()

    # Filter by query relevance
    query_emb = np.array(embed_model.embed_documents([query])[0][:dim])
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

    claim_texts = [c["text"] for c in claims]
    claim_embs = np.array([e[:dim] for e in embed_model.embed_documents(claim_texts)])
    claim_norms = np.linalg.norm(claim_embs, axis=1, keepdims=True)
    claim_norms = np.where(claim_norms == 0, 1, claim_norms)
    claim_embs = claim_embs / claim_norms

    sims_to_query = claim_embs @ query_emb
    filtered_claims = [c for c, sim in zip(claims, sims_to_query) if sim >= query_relevance_threshold]

    graph = nx.DiGraph()
    for c in filtered_claims:
        cid = c["id"]
        graph.add_node(cid, **c)

    if len(filtered_claims) < 2:
        return graph

    # Select candidate pairs via embedding similarity
    filtered_texts = [c["text"] for c in filtered_claims]
    filtered_ids = [c["id"] for c in filtered_claims]
    f_embs = np.array([e[:dim] for e in embed_model.embed_documents(filtered_texts)])
    f_norms = np.linalg.norm(f_embs, axis=1, keepdims=True)
    f_norms = np.where(f_norms == 0, 1, f_norms)
    f_embs = f_embs / f_norms
    sim_matrix = f_embs @ f_embs.T

    candidate_pairs = []
    for i in range(len(filtered_ids)):
        for j in range(i + 1, len(filtered_ids)):
            score = sim_matrix[i, j]
            if score >= pair_similarity_threshold:
                candidate_pairs.append((filtered_ids[i], filtered_ids[j], float(score)))

    # Verify relations via LLM (batch)
    support_edges = []
    contradiction_edges = []
    for ci, cj, sim_score in candidate_pairs:
        relation = _verify_relation(graph.nodes[ci]["text"], graph.nodes[cj]["text"])
        if relation == "support":
            support_edges.append((ci, cj, sim_score))
        elif relation == "contradiction":
            contradiction_edges.append((ci, cj, sim_score))

    support_edges.sort(key=lambda x: x[2], reverse=True)
    support_edges = support_edges[:max_support_edges]

    for ci, cj, score in support_edges:
        graph.add_edge(ci, cj, type="support", score=score)
        graph.add_edge(cj, ci, type="support", score=score)

    for ci, cj, score in contradiction_edges:
        graph.add_edge(ci, cj, type="contradiction", score=score)
        graph.add_edge(cj, ci, type="contradiction", score=score)

    return graph


def _verify_relation(text_a: str, text_b: str) -> str:
    """LLM: determine support/contradiction/neutral between two claims."""
    prompt = f"""Determine the relationship between these two claims about a cyber attack.

Claim A: {text_a}
Claim B: {text_b}

Choose exactly one: support, contradiction, or neutral.
Return JSON: {{"label": "support|contradiction|neutral"}}"""

    raw = chat(prompt=prompt, temperature=0.0, max_tokens=50)
    try:
        m = re.search(r'\{.*\}', raw, re.S)
        if m:
            data = json.loads(m.group(0))
            label = data.get("label", "").strip().lower()
            if label in ("support", "contradiction", "neutral"):
                return label
    except Exception:
        pass

    text = raw.strip().lower()
    if "contradiction" in text:
        return "contradiction"
    if "support" in text:
        return "support"
    return "neutral"


# ═══════════════════════════════════════════════════════════════
# Step 4: Credibility Arbitration (from ArbGraph conflict_arbitration.py)
# ═══════════════════════════════════════════════════════════════

def arbitrate(
    graph: nx.DiGraph,
    query: str,
    max_rounds: int = 2,
    accept_threshold: float = 0.3,
    arbitration_budget: int = 3,
    eta: float = 0.8,
) -> Dict[str, Any]:
    """Iterative credibility arbitration (Algorithm 1 from the paper)."""
    logits = {cid: 0.0 for cid in graph.nodes}
    defeat_counts = {cid: 0 for cid in graph.nodes}

    for t in range(max_rounds):
        pairs = _select_conflicts(graph, logits, accept_threshold, arbitration_budget)
        if not pairs:
            break
        for ci, cj in pairs:
            result = _resolve_pair(graph, ci, cj, query, logits)
            if result["gate"] == 1:
                logits[result["winner"]] += eta
                logits[result["loser"]] -= eta
                defeat_counts[result["loser"]] += 1

    validated = []
    suppressed = []
    for cid in graph.nodes:
        p = 1 / (1 + math.exp(-logits[cid]))
        node = graph.nodes[cid]
        item = {"id": cid, "text": node.get("text", ""), "confidence": round(p, 3),
                "evidence": node.get("evidence", ""), "source_id": node.get("source_id", "")}
        if p >= accept_threshold:
            validated.append(item)
        else:
            suppressed.append(item)

    return {"validated": validated, "suppressed": suppressed}


def _select_conflicts(graph, logits, threshold, budget):
    candidates = []
    seen = set()
    for u, v, d in graph.edges(data=True):
        if d.get("type") != "contradiction":
            continue
        pair = tuple(sorted([u, v]))
        if pair in seen:
            continue
        seen.add(pair)
        pu = 1 / (1 + math.exp(-logits[u]))
        pv = 1 / (1 + math.exp(-logits[v]))
        if pu < threshold or pv < threshold:
            continue
        intensity = (pu + pv) / (1.0 + abs(pu - pv))
        candidates.append((u, v, intensity))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return [(u, v) for u, v, _ in candidates[:budget]]


def _resolve_pair(graph, ci, cj, query, logits):
    node_i = graph.nodes[ci]
    node_j = graph.nodes[cj]

    # Collect support context
    ctx_i = [graph.nodes[v].get("text", "") for u, v, d in graph.edges(ci, data=True) if d.get("type") == "support"]
    ctx_j = [graph.nodes[v].get("text", "") for u, v, d in graph.edges(cj, data=True) if d.get("type") == "support"]

    prompt = f"""Query: {query}

Claim A: {node_i.get("text")}
Support for A: {"; ".join(ctx_i[:3])}

Claim B: {node_j.get("text")}
Support for B: {"; ".join(ctx_j[:3])}

Which claim is more credible for answering the query?
Choose one: A_entails_B, B_entails_A, A_contradicts_B, B_contradicts_A, unknown
Return JSON: {{"label": "..."}}"""

    raw = chat(prompt=prompt, temperature=0.0, max_tokens=50)
    label = None
    try:
        m = re.search(r'\{.*\}', raw, re.S)
        if m:
            label = json.loads(m.group(0)).get("label", "").strip()
    except Exception:
        pass

    gate = 1 if label in {"A_entails_B", "A_contradicts_B", "B_entails_A", "B_contradicts_A"} else 0

    if label in {"A_entails_B", "A_contradicts_B"}:
        winner, loser = ci, cj
    elif label in {"B_entails_A", "B_contradicts_A"}:
        winner, loser = cj, ci
    else:
        pi = 1 / (1 + math.exp(-logits[ci]))
        pj = 1 / (1 + math.exp(-logits[cj]))
        winner, loser = (ci, cj) if pi >= pj else (cj, ci)

    return {"winner": winner, "loser": loser, "gate": gate, "label": label}


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════

class ArbGraph:
    """ArbGraph baseline adapted for TAA task.

    Pipeline:
        1. Retrieve from Weaviate (shared top_k)
        2. Extract atomic claims from each document
        3. Align & merge similar claims via BGE-M3 embeddings
        4. Build evidence graph (support/contradiction edges via LLM)
        5. Credibility arbitration (iterative, from paper Algorithm 1)
        6. Generate attribution answer from validated claims
    """

    def __init__(self, top_k: int = None, task_type: str = "taa"):
        self.top_k = top_k or get_global_top_k()
        self.task_type = task_type

    def run(self, report_text: str) -> str:
        """Run ArbGraph pipeline."""
        cfg = get_task_config(self.task_type)

        # Step 1: Retrieve
        docs = retrieve(report_text, top_k=self.top_k)
        if not docs:
            response = chat(
                prompt=cfg["user"].format(report=report_text),
                system=cfg["system"], temperature=0.0,
            )
            return cfg["parse"](response)

        # Step 2: Extract atomic claims from each document
        all_claims = []
        ev_max = get_global("intermediate_evidence_max", 600)
        for doc in docs:
            content = doc.get("content", "")
            source_file = doc.get("source_file", "unknown")
            if len(content) > ev_max * 2:
                content = content[:ev_max * 2]
            claims = extract_claims(content, source_file)
            all_claims.extend(claims)

        if not all_claims:
            # Fallback: no claims extracted, use standard RAG
            context = format_context(docs)
            response = chat(
                prompt=cfg["rag_user"].format(context=context, report=report_text),
                system=cfg["system"], temperature=0.0,
            )
            return cfg["parse"](response)

        # Step 3: Align & merge similar claims
        merged = align_and_merge_claims(all_claims)

        # Step 4: Build evidence graph
        graph = build_evidence_graph(merged, report_text)

        # Step 5: Credibility arbitration
        result = arbitrate(graph, report_text)
        validated = result["validated"]

        # Step 6: Generate answer from validated claims
        validated_text = "\n".join(
            f"- {c['text']} (confidence={c['confidence']}, source={c['source_id']})"
            for c in validated[:20]
        )
        context = format_context(docs)

        prompt = cfg["rag_user"].format(context=context, report=report_text)
        prompt += f"\n\n--- Validated Evidence Claims ({len(validated)} claims) ---\n{validated_text}"

        response = chat(prompt=prompt, system=cfg["system"], temperature=0.0)
        return cfg["parse"](response)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _parse_json_list(text: str) -> list:
    try:
        cleaned = re.sub(r"```json|```", "", text).strip()
        m = re.search(r"\[.*\]", cleaned, re.S)
        if m:
            data = json.loads(m.group(0))
        else:
            data = json.loads(cleaned)
        return data if isinstance(data, list) else []
    except Exception:
        return []
