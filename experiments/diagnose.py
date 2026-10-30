"""
诊断脚本：对 TAA/ATE 前 N 条跑完整 pipeline，打印关键中间状态，定位失败模式。
用法:
    python experiments/diagnose.py --task taa --n 5
    python experiments/diagnose.py --task ate --n 5
输出同时写 stdout 与 experiments/results/diag_<task>.txt
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_from_disk


def _log(fh, msg):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def diagnose_taa(n, fh):
    from src.controller import Controller
    from src.tasks import TAATask
    from run_taa import GROUND_TRUTH, load_taa_dataset

    # 预热 BM25
    try:
        from src.retrieval.weaviate_retriever import _bm25_index
        _bm25_index.build()
    except Exception as e:
        _log(fh, f"[warn] BM25 warm-up skipped: {e}")

    ds = load_taa_dataset()
    task = TAATask()
    for idx in range(n):
        row = ds[idx]
        gt = GROUND_TRUTH[idx]
        _log(fh, "\n" + "=" * 70)
        _log(fh, f"[TAA #{idx}] GT={gt}  url={row['URL'][:70]}")
        t0 = time.time()
        try:
            ctl = Controller(task=task)
            query = task.format_query(report_text=row["Text"])
            awm = ctl.analyze(query=query)
            _dump_awm(fh, awm, task, gt, elapsed=time.time() - t0)
        except Exception as e:
            import traceback
            _log(fh, f"ERROR: {e}\n{traceback.format_exc()[:800]}")


def diagnose_ate(n, fh):
    from src.controller import Controller
    from src.tasks import ATETask
    from run_ate import load_ate_dataset

    try:
        from src.retrieval.weaviate_retriever import _bm25_index
        _bm25_index.build()
    except Exception as e:
        _log(fh, f"[warn] BM25 warm-up skipped: {e}")

    ds = load_ate_dataset()
    task = ATETask()
    task.extra_context = ""
    for idx in range(n):
        row = ds[idx]
        gt = row["GT"]
        platform = row.get("Platform", "Enterprise")
        _log(fh, "\n" + "=" * 70)
        _log(fh, f"[ATE #{idx}] GT={gt}  platform={platform}")
        t0 = time.time()
        try:
            ctl = Controller(task=task)
            query = task.format_query(description=row["Description"], platform=platform)
            awm = ctl.analyze(query=query)
            _dump_awm(fh, awm, task, gt, elapsed=time.time() - t0)
        except Exception as e:
            import traceback
            _log(fh, f"ERROR: {e}\n{traceback.format_exc()[:800]}")


def _dump_awm(fh, awm, task, gt, elapsed):
    neg = awm.get_negative_edges()
    n_edges = len(awm.get_confirmed_edges())
    # 簇分布
    clusters = awm.clusters or {}
    cluster_sizes = {cid: len(v) for cid, v in clusters.items()}
    # 各证据 attribution aliases
    _log(fh, f"-- evidence: {len(awm.evidence_set)}, confirmed_edges: {n_edges}, "
             f"negative_edges: {len(neg)}, Φ={awm.Phi:.3f}, k={awm.k}, "
             f"depth_I={awm.depth_I}, fast_path={awm.fast_path}, "
             f"clusters={cluster_sizes}, elapsed={elapsed:.1f}s")
    # 证据归因一览
    attr_lines = []
    for p in awm.behavior_profiles[:12]:
        a = (p.attribution or {}).get("canonical_key") or {}
        cn = a.get("canonical_name")
        al = a.get("aliases")
        attr_lines.append(f"   {p.doc_id[:40]:<40} cn={cn} aliases={al}")
    _log(fh, "-- attribution per evidence:\n" + "\n".join(attr_lines))
    # 负边维度分布
    from collections import Counter
    cdim = Counter(e.conflict_dimension for e in neg)
    _log(fh, f"-- negative-edge dims: {dict(cdim)}")
    # 假设
    _log(fh, "-- hypotheses:")
    for h in awm.hypothesis_space:
        hg = (h.graph_metrics or {}).get("h_graph", 0.0)
        _log(fh, f"   [{h.cluster_id}] answer={h.answer!r} llm={h.llm_score:.2f} "
                 f"h_graph={hg:.3f} strength={h.strength:.2f} n_ev={len(h.evidence_ids)} unstable={h.unstable}")
    # 查询锚点归因
    if awm.query_profile:
        qa = (awm.query_profile.attribution or {}).get("canonical_key") or {}
        _log(fh, f"-- query attribution: cn={qa.get('canonical_name')} aliases={qa.get('aliases')}")
    # 最终结论 + pred
    _log(fh, f"-- final_conclusion[:400]: {awm.final_conclusion[:400]!r}")
    pred = task.parse_output(awm.final_conclusion)
    _log(fh, f">> GT={gt!r}  PRED={pred!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["taa", "ate"], required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or f"experiments/results/diag_{args.task}.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        if args.task == "taa":
            diagnose_taa(args.n, fh)
        else:
            diagnose_ate(args.n, fh)
    print(f"\n[done] diag written to {out_path}")
