"""
诊断: RCM baseline 是否真的在跑各自代码 / 输出是否坍缩为相同预测
跑 naive_rag / iter_retgen / cyberrag 在前 N 个样本上，逐 index 对比预测，
并与已存的 arbgraph 预测对比。
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_from_disk
from baselines.NavieRAG.naive_rag import NaiveRAG
import importlib
from baselines.CyberRAG.cyberrag_taa import CyberRAG

N = 12
ds = load_from_disk("data/CTIBench-RCM")

# load arbgraph stored preds
arb = {}
try:
    with open("experiments/results/arbgraph_rcm_results.json") as f:
        for r in json.load(f):
            if r["status"] == "ok":
                arb[r["index"]] = r["prediction"].strip()
except Exception as e:
    print("no arbgraph file:", e)

IterRetGen = importlib.import_module("baselines.ITER-RETGEN.iter_retgen").IterRetGen

models = {
    "naive":   NaiveRAG(top_k=10, task_type="rcm"),
    "iter":    IterRetGen(top_k=10, iterations=3, task_type="rcm"),
    "cyber":   CyberRAG(top_k=10, task_type="rcm"),
}

preds = {m: [] for m in models}
gts = []
for i in range(N):
    desc = ds[i]["Description"]
    gts.append(ds[i]["GT"])
    for name, m in models.items():
        try:
            p = m.run(desc).strip()
        except Exception as e:
            p = f"ERR:{e}"
        preds[name].append(p)
    print(f"[{i}] done", flush=True)

# comparison table
print("\n=== per-index comparison ===")
print(f"{'i':>2} {'GT':<9} {'naive':<9} {'iter':<9} {'cyber':<9} {'arbgraph':<9} same?")
naive_eq_iter = 0; naive_eq_cyber = 0; naive_eq_arb = 0
for i in range(N):
    n, it, cy = preds["naive"][i], preds["iter"][i], preds["cyber"][i]
    ar = arb.get(i, "-")
    if n == it: naive_eq_iter += 1
    if n == cy: naive_eq_cyber += 1
    if n == ar: naive_eq_arb += 1
    same = "Y" if (n == it == cy) else ("~" if n == it or n == cy else "N")
    print(f"{i:>2} {gts[i]:<9} {n:<9} {it:<9} {cy:<9} {ar:<9} {same}")

print("\n=== collapse summary ===")
print(f"naive==iter   identical on {naive_eq_iter}/{N}")
print(f"naive==cyber  identical on {naive_eq_cyber}/{N}")
print(f"naive==arbgraph identical on {naive_eq_arb}/{N}")

# per-method accuracy
for name in ["naive","iter","cyber"] + ["arbgraph"]:
    if name == "arbgraph":
        ps = [arb.get(i,"") for i in range(N)]
    else:
        ps = preds[name]
    corr = sum(1 for i in range(N) if ps[i].upper().replace(" ","") == gts[i].upper().replace(" ",""))
    print(f"{name:>9}: {corr}/{N} correct")
