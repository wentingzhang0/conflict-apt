"""
RCM 实验 — CTIBench-RCM CVE→CWE 根因映射
用法:
    python experiments/run_rcm.py                               # 主方法，跑全部1000条
    python experiments/run_rcm.py --start 0 --end 10            # 调试
    python experiments/run_rcm.py --resume                      # 断点续跑
"""
import os
import sys
import json
import time
import argparse
import traceback
from multiprocessing import Process, Queue
from datasets import load_from_disk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.utils.settings import get_global_top_k, get_global


def _log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_rcm_dataset(data_path=None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "CTIBench-RCM")
    return load_from_disk(data_path)


# ── 评估 ─────────────────────────────────────────────────────────────

def evaluate_rcm(prediction: str, ground_truth: str) -> dict:
    """评估 RCM 预测：精确匹配"""
    pred_clean = prediction.strip().upper().replace(" ", "")
    gt_clean = ground_truth.strip().upper().replace(" ", "")
    exact_match = pred_clean == gt_clean
    return {"exact_match": exact_match, "prediction_clean": pred_clean, "gt_clean": gt_clean}


# ── 子进程 worker（主方法） ──────────────────────────────────────────

_WORKER_ARGS = {}


def _worker_main(description, sample_idx, queue: Queue):
    from src.agents.orchestrator import Orchestrator
    from src.tasks import RCMTask

    task = RCMTask()
    orchestrator = Orchestrator(
        task=task,
        top_k=_WORKER_ARGS["top_k"],
        num_clusters=_WORKER_ARGS["num_clusters"],
        max_iterations=_WORKER_ARGS["max_iterations"],
        eval_samples=_WORKER_ARGS["eval_samples"],
    )

    t0 = time.time()
    try:
        query = task.format_query(description=description)
        awm = orchestrator.analyze(query=query)
        pred = task.parse_output(awm.final_conclusion)
        elapsed = time.time() - t0
        queue.put({"index": sample_idx, "prediction": pred, "elapsed_sec": round(elapsed, 1),
                    "status": "ok", "conclusion": awm.final_conclusion[:500]})
    except Exception as e:
        elapsed = time.time() - t0
        queue.put({"index": sample_idx, "prediction": "", "elapsed_sec": round(elapsed, 1),
                    "status": f"error: {e}", "traceback": traceback.format_exc()})


def _worker_baseline(description, sample_idx, queue: Queue):
    from experiments.run_taa_baselines import _METHODS
    method_name = _WORKER_ARGS.get("_method_name", "naive_rag")
    runner = _METHODS[method_name](**{k: v for k, v in _WORKER_ARGS.items() if not k.startswith("_")})
    t0 = time.time()
    try:
        pred = runner.predict(description)
        elapsed = time.time() - t0
        queue.put({"index": sample_idx, "prediction": pred, "elapsed_sec": round(elapsed, 1), "status": "ok"})
    except Exception as e:
        elapsed = time.time() - t0
        queue.put({"index": sample_idx, "prediction": "", "elapsed_sec": round(elapsed, 1),
                    "status": f"error: {e}", "traceback": traceback.format_exc()})


def run_single(worker_fn, args, description, sample_idx, timeout):
    global _WORKER_ARGS
    _WORKER_ARGS = args

    queue = Queue()
    p = Process(target=worker_fn, args=(description, sample_idx, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        return {"index": sample_idx, "prediction": "", "elapsed_sec": timeout,
                "status": f"timeout after {timeout}s"}
    if queue.empty():
        return {"index": sample_idx, "prediction": "", "elapsed_sec": 0,
                "status": "error: subprocess died"}
    return queue.get()


# ── 主流程 ───────────────────────────────────────────────────────────

def run_experiment(args, dataset, mode="main"):
    output_path = args.output
    orchestrator_args = {
        "top_k": args.top_k,
        "num_clusters": args.num_clusters,
        "max_iterations": args.max_iterations,
        "eval_samples": args.eval_samples,
    }

    results = []
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        _log(f"Loaded {len(results)} existing results")

    done_indices = {r["index"] for r in results}
    start, end = args.start, min(args.end, len(dataset))
    worker_fn = _worker_main if mode == "main" else _worker_baseline
    runner_args = orchestrator_args if mode == "main" else {
        "top_k": args.top_k, "iterations": args.iterations,
        "max_searches": args.max_searches, "max_turns": args.max_turns,
        "_method_name": args.method,
        "task_type": "rcm",
    }

    _log(f"RCM {mode} | Samples [{start}, {end}), done={len(done_indices)}, timeout={args.timeout}s")
    _log("=" * 60)

    for idx in range(start, end):
        if idx in done_indices:
            continue

        row = dataset[idx]
        description = row["Description"]
        gt = row["GT"]

        result = run_single(worker_fn, runner_args, description, idx, args.timeout)
        result["ground_truth"] = gt
        result["url"] = row.get("URL", "")
        results.append(result)

        if result["status"] == "ok":
            ev = evaluate_rcm(result["prediction"], gt)
            tag = "+" if ev["exact_match"] else "-"
            _log(f"[{idx+1}/{end}] GT={gt} Pred={ev['prediction_clean']} [{tag}] ({result['elapsed_sec']}s)")
        else:
            _log(f"[{idx+1}/{end}] FAILED: {result['status']}")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 汇总：total 固定为实验范围大小（数据集总数），失败/超时/无答案一律按错误计，不剔除
    in_range = [r for r in results if start <= r["index"] < end]
    total = end - start
    n_failed = sum(1 for r in in_range if r["status"] != "ok")
    correct = sum(
        1 for r in in_range
        if r["status"] == "ok" and evaluate_rcm(r["prediction"], r["ground_truth"])["exact_match"]
    )
    acc = correct / total * 100 if total else 0.0
    _log(f"\nAccuracy: {acc:.1f}% ({correct}/{total})  [failed/timed-out counted as wrong: {n_failed}]")

    report_path = output_path.replace(".json", "_report.json")
    with open(report_path, "w") as f:
        json.dump({"mode": mode, "accuracy": round(acc, 2), "correct": correct, "total": total,
                   "num_failed": n_failed}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RCM experiment")
    parser.add_argument("--mode", choices=["main", "baseline"], default="main")
    parser.add_argument("--method", default="naive_rag", help="baseline method (only for --mode baseline)")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1000)
    _top_k = get_global_top_k()
    parser.add_argument("--top-k", type=int, default=_top_k)
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--eval-samples", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=get_global("timeout", 600))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        suffix = "rcm_results.json" if args.mode == "main" else f"{args.method}_rcm_results.json"
        args.output = f"experiments/results/{suffix}"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dataset = load_rcm_dataset(args.data_path)
    _log(f"Loaded CTIBench-RCM: {len(dataset)} samples")

    if args.mode == "baseline":
        from experiments.run_taa_baselines import _METHODS
        methods = list(_METHODS.keys()) if args.method == "all" else [args.method]
        for m in methods:
            _log(f"\n{'='*60}\nRunning RCM baseline: {m}\n{'='*60}")
            args.method = m
            args.output = f"experiments/results/{m}_rcm_results.json"
            run_experiment(args, dataset, mode="baseline")
    else:
        run_experiment(args, dataset, mode=args.mode)
