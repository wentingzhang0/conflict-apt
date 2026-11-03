"""攻击者两难边界测量（零 LLM 流水线）：对 frontier 语料逐格计算三轴
  ① 可检索性：变体进入 merged top-10 的 query 比例（与防御 off 时攻击可见度同口径）
  ② CORE-1 逃逸：变体对 content cos 分布 + <0.85 的对比例（协同指纹残留）
  ③ 嫁祸潜力代理：变体与事件文本的 cos（内容锚定强度）
用法：python experiments/eval_frontier.py   （需 Weaviate APT class、BGE GPU）
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_from_disk

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import weaviate
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
    client = weaviate.Client(url="http://localhost:8011")

    print(f"{'cell':10s} {'hit%':>6s} {'pair-cos med':>12s} {'<0.85%':>7s} "
          f"{'inc-cos med':>11s}")
    for level in ("L1", "L2", "L3"):
        for keep in (75, 50):
            path = os.path.join(POISON_DIR, f"frontier_{level}_{keep}.json")
            if not os.path.exists(path):
                continue
            corpus = json.load(open(path))
            hits = pairs = evade = 0
            pair_coss, inc_coss = [], []
            n_q = 0
            for key, reports in sorted(corpus.items(), key=lambda kv: int(kv[0])):
                if len(reports) < 2:
                    continue
                n_q += 1
                idx = int(key)
                query = dataset[idx]["Text"][:2000]
                qvec = emb.embed_documents([query])[0][:dim]
                # 变体嵌入
                vvecs = emb.embed_documents(
                    [r["text"][:4000] for r in reports[:2]])
                vvecs = [v[:dim] for v in vvecs]
                # 基线检索（APT class，near_vector）
                resp = (client.query.get("APT", ["source_file"])
                        .with_near_vector({"vector": qvec})
                        .with_limit(TOP_K).with_additional(["distance"]).do())
                base = resp["data"]["Get"].get("APT", [])
                base_d = [float((d.get("_additional") or {}).get("distance", 1.0))
                          for d in base]
                # 合并口径：变体凭距离竞争 top-10（与 install_poisoned_retrieval 同判据）
                pd = [1.0 - cos(qvec, v) for v in vvecs]  # cos→distance
                merged = sorted(base_d + pd)[:TOP_K]
                if any(d in merged for d in pd):
                    hits += 1
                pair_coss.append(cos(vvecs[0], vvecs[1]))
                if pair_coss[-1] < 0.85:
                    evade += 1
                pairs += 1
                inc_coss.append(max(cos(qvec, v) for v in vvecs))
            if n_q:
                pc = sorted(pair_coss)
                ic = sorted(inc_coss)
                print(f"{level}-{keep:<7d} {hits/n_q:6.1%} {pc[len(pc)//2]:12.3f} "
                      f"{evade/pairs:7.1%} {ic[len(ic)//2]:11.3f}")


if __name__ == "__main__":
    main()
