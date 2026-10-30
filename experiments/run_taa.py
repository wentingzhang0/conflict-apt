"""
CTIBench-TAA 实验 — 用本方法 (Orchestrator + TAATask) 跑 50 条归因任务
用法:
    python experiments/run_taa.py                        # 跑全部 50 条
    python experiments/run_taa.py --start 0 --end 5      # 只跑前 5 条 (调试)
    python experiments/run_taa.py --resume               # 断点续跑
    python experiments/run_taa.py --timeout 600          # 单样本超时 600 秒
"""
import os
import sys
import json
import time
import signal
import argparse
import traceback
from multiprocessing import Process, Queue
from datasets import load_from_disk

# 保证项目根目录可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.utils.settings import get_global_top_k, get_global


def _log(msg):
    """带时间戳的即时输出"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── 官方 ground truth (来自 cti-bench evaluation/responses/cti-taa-responses.tsv) ──
GROUND_TRUTH = [
    "SideCopy",        # 0
    "MUSTANG PANDA",   # 1
    "MUSTANG PANDA",   # 2
    "MUMMY SPIDER",    # 3
    "Dark Caracal",    # 4
    "APT19",           # 5
    "APT19",           # 6
    "APT-C-36",        # 7
    "APT-C-36",        # 8
    "Turla",           # 9
    "Turla",           # 10
    "APT28",           # 11
    "APT28",           # 12
    "UNC2452",         # 13
    "UNC2452",         # 14
    "CHRYSENE",        # 15
    "CHRYSENE",        # 16
    "APT41",           # 17
    "lazarus",         # 18
    "Kimsuky",         # 19
    "Kimsuky",         # 20
    "APT36",           # 21
    "APT35",           # 22
    "MuddyWater",      # 23
    "MuddyWater",      # 24
    "CharmingCypress", # 25
    "Mint Sandstorm",  # 26
    "APT31",           # 27
    "Gamaredon",       # 28
    "Gamaredon",       # 29
    "Sharp Panda",     # 30
    "Bitter APT",      # 31
    "Confucius",       # 32
    "Dragonfly",       # 33
    "Andariel",        # 34
    "COLDRIVER",       # 35
    "Bahamut",         # 36
    "APT37",           # 37
    "APT33",           # 38
    "APT29",           # 39
    "APT29",           # 40
    "APT29",           # 41
    "Diamond Sleet",   # 42
    "Lazarus",         # 43
    "Lazarus",         # 44
    "Lazarus",         # 45
    "Lazarus",         # 46
    "OilRig",          # 47
    "MuddyWater",      # 48
    "Turla",           # 49
]


def load_taa_dataset(data_path: str = None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "CTIBench-TAA")
    return load_from_disk(data_path)


# ── 子进程 worker：真正跑 pipeline ──────────────────────────────────
def _worker(report_text, sample_idx, queue: Queue):
    """在子进程中运行 pipeline，结果通过 queue 返回"""
    from src.controller import Controller
    from src.tasks import TAATask

    task = TAATask()
    # Controller 只接收 task + top_k；其余（clusters/iterations/eval_samples）
    # 由 settings.yaml 的 pipeline 配置在内部控制，CLI 参数仅用于日志展示
    orchestrator = Controller(
        task=task,
        top_k=_WORKER_ARGS["top_k"],
    )

    t0 = time.time()
    try:
        _log(f"  [worker] Phase 1: Evidence collection + ABP extraction ...")
        query = task.format_query(report_text=report_text)
        awm = orchestrator.analyze(query=query)

        # 打印每个 phase 的进展
        for entry in awm.analysis_trace:
            _log(f"    [{entry['iteration']}] {entry['agent']}: {entry['action']}")

        pred = task.parse_output(awm.final_conclusion)
        elapsed = time.time() - t0

        result = {
            "index": sample_idx,
            "prediction": pred,
            "conclusion": awm.final_conclusion,
            "evidence_count": len(awm.evidence_set),
            "conflict_detected": awm.conflict_detected,
            "num_hypotheses": len(awm.hypothesis_space),
            "iterations": awm.iteration + 1,
            "signed_edges": len(awm.signed_graph_edges),
            "negative_edges": sum(1 for e in awm.signed_graph_edges if e.sign == -1),
            "elapsed_sec": round(elapsed, 1),
            "status": "ok",
        }
        queue.put(result)
    except Exception as e:
        elapsed = time.time() - t0
        queue.put({
            "index": sample_idx,
            "prediction": "",
            "conclusion": "",
            "elapsed_sec": round(elapsed, 1),
            "status": f"error: {e}",
            "traceback": traceback.format_exc(),
        })


# 全局变量用于传递参数给 worker 子进程
_WORKER_ARGS = {}


def run_single(orchestrator_args: dict, report_text: str,
               sample_idx: int, timeout: int) -> dict:
    """跑单条样本，子进程隔离 + 超时强杀"""
    global _WORKER_ARGS
    _WORKER_ARGS = orchestrator_args

    queue = Queue()
    p = Process(target=_worker, args=(report_text, sample_idx, queue))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        _log(f"  TIMEOUT after {timeout}s, killing subprocess ...")
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        return {
            "index": sample_idx,
            "prediction": "",
            "conclusion": "",
            "elapsed_sec": timeout,
            "status": f"timeout after {timeout}s",
        }

    if queue.empty():
        return {
            "index": sample_idx,
            "prediction": "",
            "conclusion": "",
            "elapsed_sec": 0,
            "status": "error: subprocess died without result",
        }

    return queue.get()


def run_experiment(args):
    dataset = load_taa_dataset(args.data_path)

    orchestrator_args = {
        "top_k": args.top_k,
        "num_clusters": args.num_clusters,
        "max_iterations": args.max_iterations,
        "eval_samples": args.eval_samples,
    }

    # 加载已有结果 (断点续跑)
    results = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        _log(f"Loaded {len(results)} existing results from {args.output}")

    done_indices = {r["index"] for r in results}
    start = args.start
    end = min(args.end, len(dataset))

    _log(f"CTIBench-TAA Experiment")
    _log(f"  Samples: [{start}, {end})")
    _log(f"  Already done: {len(done_indices)}")
    _log(f"  Pipeline: top_k={args.top_k}, clusters={args.num_clusters}, "
         f"iters={args.max_iterations}, samples={args.eval_samples}")
    _log(f"  Timeout: {args.timeout}s per sample")
    _log("=" * 60)

    # 一次性预热 BM25 检索索引：fork 后所有 worker 子进程继承已构建的索引，
    # 避免每条样本都重复从 Weaviate 拉全量文档冷启动（~22s/次）。
    try:
        from src.retrieval.weaviate_retriever import _bm25_index
        _log("Pre-warming BM25 index (one-time) ...")
        _bm25_index.build()  # 仅建 BM25(CPU+HTTP)，不触发 embedding/CUDA，避免破坏 fork
        _log(f"BM25 index ready ({_bm25_index.N} docs); workers reuse it.")
    except Exception as e:
        _log(f"BM25 warm-up skipped: {e}")

    for idx in range(start, end):
        if idx in done_indices:
            continue

        row = dataset[idx]
        _log(f"[{idx+1}/{end}] Processing (GT: {GROUND_TRUTH[idx]})...")
        _log(f"  Report: {row['URL'][:80]}")

        result = run_single(orchestrator_args, row["Text"], idx, args.timeout)

        # 附加 ground truth
        result["ground_truth"] = GROUND_TRUTH[idx]
        result["url"] = row["URL"]

        results.append(result)

        if result["status"] == "ok":
            status_icon = "+" if result["prediction"].strip().lower() == GROUND_TRUTH[idx].strip().lower() else "-"
            _log(f"  GT: {GROUND_TRUTH[idx]}  |  Pred: {result['prediction']}  [{status_icon}]")
            _log(f"  Evidence: {result.get('evidence_count', '?')}  "
                 f"Conflicts: {result.get('conflict_detected', '?')}  "
                 f"Hypotheses: {result.get('num_hypotheses', '?')}  "
                 f"Iters: {result.get('iterations', '?')}  "
                 f"Time: {result.get('elapsed_sec', '?')}s")
        else:
            _log(f"  FAILED: {result['status']}")
            if "traceback" in result:
                _log(f"  {result['traceback'][:300]}")

        # 每条保存一次，防止中断丢结果
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 汇总评估 ─────────────────────────────────────────────────────
    # total 固定为实验范围大小（数据集总数）；失败/超时样本预测为空 -> 计为 Incorrect
    in_range = [r for r in results if start <= r["index"] < end]
    total = end - start
    n_failed = sum(1 for r in in_range if r["status"] != "ok")

    from evaluate_taa import compute_taa_accuracy
    gts = [r["ground_truth"] for r in in_range]
    preds = [r["prediction"] for r in in_range]
    correct_acc, plausible_acc, details = compute_taa_accuracy(gts, preds)

    _log("\n" + "=" * 60)
    _log("EVALUATION RESULTS")
    _log("=" * 60)
    _log(f"Total: {total}  Valid: {total - n_failed}  Failed: {n_failed}")
    _log(f"Correct Accuracy:    {correct_acc:.1f}%")
    _log(f"Plausible Accuracy:  {plausible_acc:.1f}%")

    # 逐条详情（details 与 in_range 等长同序）
    _log(f"\n{'Idx':>3} {'GT':<20} {'Pred':<25} {'Result':<10}")
    _log("-" * 60)
    for i, d in enumerate(details):
        tag = {"C": "Correct", "P": "Plausible", "I": "Incorrect"}[d["result"]]
        _log(f"{in_range[i]['index']:>3} {d['gt']:<20} {d['pred']:<25} {tag:<10}")

    # 保存评估报告
    report_path = args.output.replace(".json", "_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "correct_accuracy": round(correct_acc, 2),
            "plausible_accuracy": round(plausible_acc, 2),
            "total": total,
            "num_valid": total - n_failed,
            "num_failed": n_failed,
            "details": details,
        }, f, indent=2, ensure_ascii=False)
    _log(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CTIBench-TAA experiment")
    parser.add_argument("--data-path", default=None, help="CTIBench-TAA dataset path")
    parser.add_argument("--output", default="experiments/results/taa_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    _default_top_k = get_global_top_k()
    parser.add_argument("--top-k", type=int, default=_default_top_k)
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--eval-samples", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=get_global("timeout", 600),
                        help="单样本超时秒数")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_experiment(args)
