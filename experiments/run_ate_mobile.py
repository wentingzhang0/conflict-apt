"""
只跑 CTIBench-ATE 的 Mobile 平台样本，验证移动矩阵修复。
用法: python experiments/run_ate_mobile.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_from_disk
from src.controller import Controller
from src.tasks import ATETask
from run_ate import evaluate_ate


def main():
    ds = load_from_disk(os.path.join(os.path.dirname(__file__), "..", "data", "CTIBench-ATE"))
    if hasattr(ds, 'keys'):
        ds = ds['test']
    try:
        from src.retrieval.weaviate_retriever import _bm25_index
        _bm25_index.build()
    except Exception as e:
        print(f"[warn] BM25: {e}")

    task = ATETask()
    task.extra_context = ""
    mobile_idx = [i for i, r in enumerate(ds) if r.get("Platform") == "Mobile"]
    print(f"Mobile samples: {len(mobile_idx)} -> {mobile_idx}")

    results = []
    for idx in mobile_idx:
        row = ds[idx]
        gt = row["GT"]
        platform = row.get("Platform", "Enterprise")
        t0 = time.time()
        try:
            ctl = Controller(task=task)
            query = task.format_query(description=row["Description"], platform=platform)
            awm = ctl.analyze(query=query)
            pred = task.parse_output(awm.final_conclusion)
            ev = evaluate_ate(pred, gt)
            print(f"#{idx} GT={gt[:60]}")
            print(f"   PRED={pred}")
            print(f"   F1={ev['f1']*100:.1f} P={ev['precision']*100:.1f} R={ev['recall']*100:.1f} ({time.time()-t0:.0f}s)")
            results.append({"idx": idx, "f1": ev['f1'], "pred": pred, "gt": gt})
        except Exception as e:
            import traceback
            print(f"#{idx} ERROR: {e}\n{traceback.format_exc()[:300]}")
            results.append({"idx": idx, "f1": 0, "pred": "", "gt": gt})

    # 汇总（micro F1）
    tp = sum(len(set(r["pred"].split(',')) & set(r["gt"].split(','))) if r["pred"] else 0 for r in results)
    fp = sum(len(set((r["pred"] or "").split(',')) - set(r["gt"].split(','))) for r in results)
    fn = sum(len(set(r["gt"].split(',')) - set((r["pred"] or "").split(','))) for r in results)
    P = tp/(tp+fp) if tp+fp else 0
    R = tp/(tp+fn) if tp+fn else 0
    F1 = 2*P*R/(P+R) if P+R else 0
    mean_f1 = sum(r["f1"] for r in results)/len(results)
    print(f"\n=== Mobile {len(results)} samples: micro F1={F1*100:.1f} P={P*100:.1f} R={R*100:.1f} | mean F1={mean_f1*100:.1f}")


if __name__ == "__main__":
    main()
