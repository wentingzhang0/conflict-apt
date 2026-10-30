"""
CTIBench-MCQ 实验 — 用本方法 (Orchestrator + MCQTask) 跑 2500 道多选题
用法:
    python experiments/run_mcq.py                        # 跑全部 2500 条
    python experiments/run_mcq.py --start 0 --end 10     # 调试
    python experiments/run_mcq.py --resume               # 断点续跑
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


def load_mcq_dataset(data_path=None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "CTIBench-MCQ")
    return load_from_disk(data_path)


def _worker(question, options_a, options_b, options_c, options_d, sample_idx, queue: Queue):
    from src.agents.orchestrator import Orchestrator
    from src.tasks import MCQTask

    task = MCQTask()
    orchestrator = Orchestrator(
        task=task,
        top_k=_WORKER_ARGS["top_k"],
        num_clusters=_WORKER_ARGS["num_clusters"],
        max_iterations=_WORKER_ARGS["max_iterations"],
        eval_samples=_WORKER_ARGS["eval_samples"],
    )

    t0 = time.time()
    try:
        query = task.format_query(
            question=question,
            options={"A": options_a, "B": options_b, "C": options_c, "D": options_d},
        )
        awm = orchestrator.analyze(query=query)
        pred = task.parse_output(awm.final_conclusion)
        elapsed = time.time() - t0

        result = {
            "index": sample_idx,
            "prediction": pred,
            "conclusion": awm.final_conclusion[:500],
            "evidence_count": len(awm.evidence_set),
            "conflict_detected": awm.conflict_detected,
            "num_hypotheses": len(awm.hypothesis_space),
            "elapsed_sec": round(elapsed, 1),
            "status": "ok",
        }
        queue.put(result)
    except Exception as e:
        elapsed = time.time() - t0
        queue.put({
            "index": sample_idx, "prediction": "A",
            "conclusion": "", "elapsed_sec": round(elapsed, 1),
            "status": f"error: {e}",
            "traceback": traceback.format_exc(),
        })


_WORKER_ARGS = {}


def run_single(orchestrator_args, question, options_a, options_b, options_c, options_d,
               sample_idx, timeout):
    global _WORKER_ARGS
    _WORKER_ARGS = orchestrator_args

    queue = Queue()
    p = Process(target=_worker, args=(question, options_a, options_b, options_c, options_d,
                                       sample_idx, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        _log(f"  TIMEOUT after {timeout}s, killing...")
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        return {"index": sample_idx, "prediction": "A", "conclusion": "",
                "elapsed_sec": timeout, "status": f"timeout after {timeout}s"}

    if queue.empty():
        return {"index": sample_idx, "prediction": "A", "conclusion": "",
                "elapsed_sec": 0, "status": "error: subprocess died"}

    return queue.get()


def run_experiment(args):
    dataset = load_mcq_dataset(args.data_path)
    orchestrator_args = {
        "top_k": args.top_k,
        "num_clusters": args.num_clusters,
        "max_iterations": args.max_iterations,
        "eval_samples": args.eval_samples,
    }

    results = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        _log(f"Loaded {len(results)} existing results")

    done_indices = {r["index"] for r in results}
    start = args.start
    end = min(args.end, len(dataset))

    _log(f"CTIBench-MCQ Experiment")
    _log(f"  Samples: [{start}, {end}), done: {len(done_indices)}, timeout: {args.timeout}s")
    _log("=" * 60)

    for idx in range(start, end):
        if idx in done_indices:
            continue

        row = dataset[idx]
        _log(f"[{idx+1}/{end}] GT={row['GT']}  Q={row['Question'][:60]}...")

        result = run_single(
            orchestrator_args,
            row["Question"], row["Option A"], row["Option B"],
            row["Option C"], row["Option D"], idx, args.timeout,
        )
        result["ground_truth"] = row["GT"]
        results.append(result)

        if result["status"] == "ok":
            tag = "+" if result["prediction"] == row["GT"] else "-"
            _log(f"  Pred: {result['prediction']}  GT: {row['GT']}  [{tag}]  ({result['elapsed_sec']}s)")
        else:
            _log(f"  FAILED: {result['status']}")

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 汇总：total 固定为实验范围大小（数据集总数），失败/超时/无答案一律按错误计，不剔除
    # 注意：失败样本的 prediction 被默认置为 "A"，但 status != "ok" 时绝不计为正确
    in_range = [r for r in results if start <= r["index"] < end]
    total = end - start
    n_failed = sum(1 for r in in_range if r["status"] != "ok")
    correct = sum(1 for r in in_range if r["status"] == "ok" and r["prediction"] == r["ground_truth"])
    acc = correct / total * 100 if total else 0.0

    _log(f"\n{'='*60}")
    _log(f"MCQ Accuracy: {acc:.1f}% ({correct}/{total})  [failed/timed-out counted as wrong: {n_failed}]")

    report_path = args.output.replace(".json", "_report.json")
    with open(report_path, "w") as f:
        json.dump({"accuracy": round(acc, 2), "correct": correct, "total": total, "num_failed": n_failed},
                  f, indent=2)
    _log(f"Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CTIBench-MCQ experiment")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output", default="experiments/results/mcq_results.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2500)
    _default_top_k = get_global_top_k()
    parser.add_argument("--top-k", type=int, default=_default_top_k)
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=get_global("timeout", 600))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_experiment(args)
