"""防御基线过滤器（替换式接法，学长 4.2 设计）：挂在检索层，与 SCHISM 其余部分无关。

- trustrag：TrustRAG Stage-1 的忠实适配（原文核实 arXiv:2501.00879v3 §3.2/B.2）：
  K-means(k=2) 于检索文档内容嵌入，多注入场景剔除"少数且更紧"的簇
  （原文超参 k=2；ROUGE-L 0.25/cos 0.85 属单注入保护路径，本设定为多注入，不启用）
- ppl：困惑度过滤。骨干 LLM prompt_logprobs 计 PPL，阈值=干净语料样本 PPL 的
  95 分位（懒标定、按模型缓存）。TrustRAG 原文 §5.3 自己证明 clean/毒 PPL 分布
  大量重叠——此基线预期弱，本身是实验结论的一部分。

接入点：包住 weaviate_retriever._hybrid_search（retrieve 经模块全局查名调用，
补丁对其生效；ega 直接 from-import retrieve 不受影响、但其内部调用会走到补丁）。
被剔除文档记入 DROPPED（run_poison 落盘做误剔除统计）。src/ 与 src_new/ 通用。
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

DROPPED = []          # [(source_file, is_poison)]，每次 install 重置
_FILTER_STATE = {"mode": None, "orig": None, "retr": None}

_PPL_CACHE = {}       # model -> threshold


def install_retrieval_filter(mode: str, module="src_new"):
    """mode ∈ {trustrag, ppl}；module ∈ {src_new, src}。返回还原函数。"""
    import importlib
    retr = importlib.import_module(f"{module}.retrieval.weaviate_retriever")
    DROPPED.clear()
    _FILTER_STATE.update(mode=mode, orig=retr._hybrid_search, retr=retr)

    def filtered_hybrid(query_text, top_k, alpha):
        docs = _FILTER_STATE["orig"](query_text, top_k, alpha)
        if not docs:
            return docs
        try:
            keep = _apply(mode, docs, query_text)
        except Exception:
            keep = docs          # 过滤器故障不伤流水线（防御基线不应成为故障点）
        for d in docs:
            if d not in keep:
                DROPPED.append((str(d.get("source_file", "")),
                                 str(d.get("source_file", "")).startswith("poison_")))
        return keep

    retr._hybrid_search = filtered_hybrid

    def restore():
        if _FILTER_STATE["orig"] is not None:
            _FILTER_STATE["retr"]._hybrid_search = _FILTER_STATE["orig"]
            _FILTER_STATE.update(mode=None, orig=None)
    return restore


# ---------------- 共用 ----------------
def _embed_docs(docs):
    from src_new.utils.embedding import get_embed_model, get_target_dimension
    emb = get_embed_model()
    dim = get_target_dimension()
    vecs = emb.embed_documents([str(d.get("content", ""))[:4000] for d in docs])
    return np.array([v[:dim] for v in vecs], dtype=float)


# ---------------- TrustRAG Stage-1 ----------------
def _trustrag(docs):
    if len(docs) < 4:
        return docs
    from sklearn.cluster import KMeans
    X = _embed_docs(docs)
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
    labels = km.labels_
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return docs

    def intra(ii):
        if len(ii) < 2:
            return -1.0
        V = X[ii]
        sim = V @ V.T
        n = len(ii)
        return float((sim.sum() - n) / (n * (n - 1)))

    minority, majority = (idx0, idx1) if len(idx0) <= len(idx1) else (idx1, idx0)
    # 忠实原设定：多注入时剔除更紧簇；仅当该簇为严格少数（<n/2）才剔，
    # 5/5 平分（干净语料常见）不剔——对应原文 20% 毒率下的保护行为
    if len(minority) < len(docs) / 2 and intra(minority) > intra(majority):
        return [docs[i] for i in majority]
    return docs


# ---------------- 困惑度过滤 ----------------
# vLLM 服务端 prompt_logprobs 有 bug（bytes 序列化崩溃）→ 按学长 4.2 原设计
# 改用本地小语言模型（Qwen2.5-3B-Instruct，/opt/share 权重）计算 PPL，骨干无关。
_PPL_MODEL = None
_PPL_KEY = "qwen2.5-3b"


def _ppl_lm():
    global _PPL_MODEL
    if _PPL_MODEL is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        path = "/opt/share/models/Qwen/Qwen2.5-3B-Instruct"
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        _PPL_MODEL = (tok, model)
    return _PPL_MODEL


def _ppl_threshold():
    if _PPL_KEY in _PPL_CACHE:
        return _PPL_CACHE[_PPL_KEY]
    cache_path = f"/tmp/ppl_thr_{_PPL_KEY}.json"
    if os.path.exists(cache_path):
        thr = json.load(open(cache_path))["thr"]
        _PPL_CACHE[_PPL_KEY] = thr
        return thr
    # 干净语料样本标定：BM25 取 60 篇真实 chunk（排除 poison_）
    from src_new.retrieval import weaviate_retriever as retr
    idx = retr._bm25_index
    if not idx.docs:
        idx.build()
    docs = []
    for q in ("threat actor", "malware campaign", "vulnerability exploit",
              "ransomware attack", "phishing infrastructure"):
        for i, _ in idx.search(q, top_k=20):
            d = idx.get_doc(i)
            sf = str(d.get("source_file", ""))
            if not sf.startswith("poison_") and str(d.get("content", "")).strip():
                docs.append(d)
    docs = docs[:60]
    ppls = [p for p in (_doc_ppl(d) for d in docs) if p is not None]
    thr = float(np.quantile(ppls, 0.95)) if ppls else float("inf")
    json.dump({"thr": thr, "n": len(ppls)}, open(cache_path, "w"))
    _PPL_CACHE[_PPL_KEY] = thr
    return thr


def _doc_ppl(doc):
    """本地小 LM 的 PPL。失败返回 None（不作为剔除依据）。"""
    text = str(doc.get("content", "")).strip()[:2000]
    if not text:
        return None
    try:
        import torch
        tok, model = _ppl_lm()
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=1024).to(model.device)
        if enc.input_ids.shape[1] < 8:
            return None
        with torch.no_grad():
            logits = model(**enc).logits[:, :-1]
            labels = enc.input_ids[:, 1:]
            lp = torch.log_softmax(logits, -1).gather(
                -1, labels.unsqueeze(-1)).squeeze(-1)
            return float(torch.exp(-lp.mean()))
    except Exception:
        return None


def _ppl(docs):
    thr = _ppl_threshold()
    if not math.isfinite(thr):
        return docs
    keep = [p is None or p <= thr for p in (_doc_ppl(d) for d in docs)]
    return [d for d, k in zip(docs, keep) if k]


def _apply(mode, docs, query_text):
    if mode == "trustrag":
        return _trustrag(docs)
    if mode == "ppl":
        return _ppl(docs)
    return docs
