"""
ATE 实验 — CTIBench-ATE 攻击技术提取
从恶意软件描述中提取 MITRE ATT&CK 技术 ID
评估指标：F1（多标签）

用法:
    python experiments/run_ate.py                               # 主方法，跑全部60条
    python experiments/run_ate.py --start 0 --end 5             # 调试
    python experiments/run_ate.py --resume                      # 断点续跑
    python experiments/run_ate.py --mode baseline --method naive_rag  # baseline
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


def load_ate_dataset(data_path=None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "CTIBench-ATE")
    ds = load_from_disk(data_path)
    # 兼容 DatasetDict 和 Dataset
    if hasattr(ds, 'keys'):
        return ds['test']
    return ds


# ── 评估 ────────────────────────────────────────────────────

def parse_tids(text: str) -> set:
    """从文本中提取 T-ID 集合（去子技术）"""
    import re
    matches = re.findall(r'(T\d{3,4})(?:\.\d{3})?', text)
    return set(tid.upper() for tid in matches)


def _truncate_pred_by_platform(prediction: str, platform: str) -> str:
    """按平台硬截断 T-ID 列表，与 src/tasks/ate.py:parse_output 对称（控制变量：
    baseline 与主方法走完全相同的截断——同一 _extract_tids + 同一平台上限）。"""
    from src.tasks.ate import _extract_tids, _MAX_TIDS_ENTERPRISE, _MAX_TIDS_MOBILE
    tids = _extract_tids(prediction)  # 有序 / 去重 / 去子技术 / upper
    limit = _MAX_TIDS_MOBILE if (platform or "").lower() == "mobile" else _MAX_TIDS_ENTERPRISE
    return ", ".join(tids[:limit])


def evaluate_ate(prediction: str, ground_truth: str) -> dict:
    """评估 ATE 预测（多标签）"""
    pred_set = parse_tids(prediction)
    gt_set = parse_tids(ground_truth)

    if not pred_set and not gt_set:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0,
                "pred_tids": sorted(pred_set), "gt_tids": sorted(gt_set)}

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"f1": f1, "precision": precision, "recall": recall,
            "pred_tids": sorted(pred_set), "gt_tids": sorted(gt_set)}


# ── 子进程 worker（主方法） ──────────────────────────────────────────

_WORKER_ARGS = {}


def _worker_main(description, sample_idx, queue: Queue):
    from src.agents.orchestrator import Orchestrator
    from src.tasks import ATETask

    task = ATETask()
    # 不传入 Prompt 字段的完整技术列表（~202个T-ID），
    # 让模型依靠检索证据 + 自身知识提取技术，确保评估公平。
    task.extra_context = ""
    platform = _WORKER_ARGS.get("_platform", "Enterprise")
    orchestrator = Orchestrator(
        task=task,
        top_k=_WORKER_ARGS["top_k"],
        num_clusters=_WORKER_ARGS["num_clusters"],
        max_iterations=_WORKER_ARGS["max_iterations"],
        eval_samples=_WORKER_ARGS["eval_samples"],
    )

    t0 = time.time()
    try:
        query = task.format_query(description=description, platform=platform)
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
    # 控制变量：与主方法 format_query 对齐——在描述前注入 [Platform: ...]，使 baseline
    # 同样能看到平台信息并据此选择 Enterprise / Mobile 技术 reference。
    platform = (_WORKER_ARGS.get("_platform") or "Enterprise").strip()
    platformed = f"[Platform: {platform}]\n{description}"
    t0 = time.time()
    try:
        pred = runner.predict(platformed)
        # 控制变量：与主方法 parse_output 对称——按平台截断，抑制枚举通用技术。
        pred = _truncate_pred_by_platform(pred, platform)
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
        "task_type": "ate",
    }

    _log(f"ATE {mode} | Samples [{start}, {end}), done={len(done_indices)}, timeout={args.timeout}s")
    _log("=" * 60)

    for idx in range(start, end):
        if idx in done_indices:
            continue

        row = dataset[idx]
        description = row["Description"]
        gt = row["GT"]

        # 注入 platform（不再注入 Prompt 的完整技术列表，确保评估公平）
        runner_args["_platform"] = row.get("Platform", "Enterprise")

        # 主方法和 baseline 统一用 Description 作为输入
        # Prompt 字段包含完整202个技术ID列表，属于答案候选集，使用它会造成不公平优势
        input_text = description

        result = run_single(worker_fn, runner_args, input_text, idx, args.timeout)
        result["ground_truth"] = gt
        result["url"] = row.get("URL", "")
        results.append(result)

        if result["status"] == "ok":
            ev = evaluate_ate(result["prediction"], gt)
            _log(f"[{idx+1}/{end}] GT={gt} Pred={result['prediction']} "
                  f"F1={ev['f1']:.2f} ({result['elapsed_sec']}s)")
        else:
            _log(f"[{idx+1}/{end}] FAILED: {result['status']}")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 汇总：total 固定为实验范围大小（数据集总数），失败/超时/无答案一律按错误计，不剔除
    # 失败样本 prediction 已为空 -> pred_set 为空 -> 不产生 TP，其 GT 全部计入 FN，拉低 recall
    in_range = [r for r in results if start <= r["index"] < end]
    total = end - start
    n_failed = sum(1 for r in in_range if r["status"] != "ok")
    evals = [evaluate_ate(r["prediction"], r["ground_truth"]) for r in in_range]

    # 全局聚合 TP/FP/FN
    total_tp = sum(len(set(e["pred_tids"]) & set(e["gt_tids"])) for e in evals)
    total_fp = sum(len(set(e["pred_tids"]) - set(e["gt_tids"])) for e in evals)
    total_fn = sum(len(set(e["gt_tids"]) - set(e["pred_tids"])) for e in evals)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    _log(f"\nResults ({total} samples, failed/timed-out as wrong: {n_failed}):")
    _log(f"  F1:          {f1*100:.1f}%")
    _log(f"  Precision:   {precision*100:.1f}%")
    _log(f"  Recall:      {recall*100:.1f}%")
    _log(f"  Total TP={total_tp} FP={total_fp} FN={total_fn}")

    report_path = output_path.replace(".json", "_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "mode": mode,
            "total": total,
            "num_failed": n_failed,
            "f1": round(f1 * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "per_sample": [{
                "index": in_range[i]["index"],
                "prediction": in_range[i]["prediction"],
                "ground_truth": in_range[i]["ground_truth"],
                "precision": round(evals[i]["precision"] * 100, 2),
                "recall": round(evals[i]["recall"] * 100, 2),
                "f1": round(evals[i]["f1"] * 100, 2),
                "pred_tids": evals[i]["pred_tids"],
                "gt_tids": evals[i]["gt_tids"],
            } for i in range(len(in_range))],
        }, f, indent=2, ensure_ascii=False)
    _log(f"Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ATE experiment")
    parser.add_argument("--mode", choices=["main", "baseline"], default="main")
    parser.add_argument("--method", default="naive_rag", help="baseline method (only for --mode baseline)")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=60)
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
        suffix = "ate_results.json" if args.mode == "main" else f"{args.method}_ate_results.json"
        args.output = f"experiments/results/{suffix}"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dataset = load_ate_dataset(args.data_path)
    _log(f"Loaded CTIBench-ATE: {len(dataset)} samples")

    if args.mode == "baseline":
        from experiments.run_taa_baselines import _METHODS
        methods = list(_METHODS.keys()) if args.method == "all" else [args.method]
        for m in methods:
            _log(f"\n{'='*60}\nRunning ATE baseline: {m}\n{'='*60}")
            args.method = m
            args.output = f"experiments/results/{m}_ate_results.json"
            run_experiment(args, dataset, mode="baseline")
    else:
        run_experiment(args, dataset, mode=args.mode)
