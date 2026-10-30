"""攻击者两难边界测量 v2（真实检索通道口径）：
每格语料真注入 Weaviate APT_POISON + BM25 扩展，走 src_new.retrieval.retrieve()
（hybrid RRF α=0.4，与流水线完全同通道），统计毒进入 top-10 的 query 比例。
配对余弦轴与 v1 相同。零 LLM 调用。
用法：CUDA_VISIBLE_DEVICES=6 python experiments/eval_frontier2.py
"""
import os
import sys
import json
import shutil
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_from_disk

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import run_poison as rp
from src_new.retrieval import weaviate_retriever as retr
from src_new.utils.embedding import get_embed_model, get_target_dimension

POISON_DIR = os.path.join(_HERE, "poison_corpus")
TOP_K = 10


def cos(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def main():
    dataset = load_from_disk(os.path.join(_HERE, "..", "data", "CTIBench-TAA"))
    emb = get_embed_model()
    dim = get_target_dimension()
    print(f"{'cell':8s} {'hit%':>6s} {'pair-cos med':>12s} {'<0.85%':>7s} {'inc-cos med':>11s}")
    for level in ("L1", "L2", "L3"):
        for keep in (75, 50):
            src = os.path.join(POISON_DIR, f"frontier_{level}_{keep}.json")
            if not os.path.exists(src):
                continue
            cell = f"{level}_{keep}"
            # 以 run_poison 的注入协议落库（false_flags_ 前缀）
            dst = os.path.join(POISON_DIR, f"false_flags_{cell}.json")
            shutil.copyfile(src, dst)
            rp.inject_poison(SimpleNamespace(tier=cell, n_per_query=2))
            restore = rp.install_poisoned_retrieval(variant_cap=2)

            hits = evade = 0
            pair_coss, inc_coss = [], []
            corpus = json.load(open(src))
            for key, reports in sorted(corpus.items(), key=lambda kv: int(kv[0])):
                if len(reports) < 2:
                    continue
                idx = int(key)
                query = dataset[idx]["Text"][:6000]
                docs = retr.retrieve(query, top_k=TOP_K)
                if any(str(d.get("source_file", "")).startswith("poison_") for d in docs):
                    hits += 1
                vv = emb.embed_documents([r["text"][:4000] for r in reports[:2]])
                vv = [v[:dim] for v in vv]
                pc = cos(vv[0], vv[1])
                pair_coss.append(pc)
                evade += int(pc < 0.85)
                qv = emb.embed_documents([query[:2000]])[0][:dim]
                inc_coss.append(max(cos(qv, v) for v in vv))
            restore()
            rp.clean_poison(SimpleNamespace())
            retr._bm25_index.build()  # BM25 索引回净（重拉 APT class）
            if pair_coss:
                pc_s = sorted(pair_coss)
                ic_s = sorted(inc_coss)
                n = len(pair_coss)
                print(f"{cell:8s} {hits/n:6.1%} {pc_s[n//2]:12.3f} "
                      f"{evade/n:7.1%} {ic_s[n//2]:11.3f}", flush=True)


if __name__ == "__main__":
    main()
