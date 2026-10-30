"""
Baseline TAA 实验 — 用 3 个 baseline 跑 CTIBench-TAA
用法:
    python experiments/run_taa_baselines.py --method naive_rag
    python experiments/run_taa_baselines.py --method iter_retgen --start 0 --end 5
    python experiments/run_taa_baselines.py --method search_o1 --resume
    python experiments/run_taa_baselines.py --method all              # 跑全部 baseline
"""
import os
import sys
import json
import time
import argparse
import importlib
import traceback
from multiprocessing import Process, Queue
from datasets import load_from_disk

# 保证项目根目录可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 复用 experiments/ 的 ground truth 和评估
sys.path.insert(0, os.path.dirname(__file__))
from run_taa import GROUND_TRUTH, _log
from evaluate_taa import compute_taa_accuracy
from src.utils.settings import get_global_top_k, get_pipeline_param, get_global

# ── Baseline 注册表 ──────────────────────────────────────────────────
_METHODS = {}


def register(name):
    def decorator(cls):
        _METHODS[name] = cls
        return cls
    return decorator


@register("naive_rag")
class NaiveRAGRunner:
    def __init__(self, **kwargs):
        from baselines.NavieRAG.naive_rag import NaiveRAG
        self.model = NaiveRAG(top_k=kwargs.get("top_k", 10),
                              task_type=kwargs.get("task_type", "taa"))

    def predict(self, report_text: str) -> str:
        return self.model.run(report_text)


@register("iter_retgen")
class IterRetGenRunner:
    def __init__(self, **kwargs):
        _mod = importlib.import_module("baselines.ITER-RETGEN.iter_retgen")
        IterRetGen = _mod.IterRetGen
        self.model = IterRetGen(
            top_k=kwargs.get("top_k", 10),
            iterations=kwargs.get("iterations", 3),
            task_type=kwargs.get("task_type", "taa"),
        )

    def predict(self, report_text: str) -> str:
        return self.model.run(report_text)


@register("arbgraph")
class ArbGraphRunner:
    def __init__(self, **kwargs):
        from baselines.ArbGraph.arbgraph_taa import ArbGraph
        self.model = ArbGraph(top_k=kwargs.get("top_k", 10),
                              task_type=kwargs.get("task_type", "taa"))

    def predict(self, report_text: str) -> str:
        return self.model.run(report_text)


@register("cyberrag")
class CyberRAGRunner:
    def __init__(self, **kwargs):
        from baselines.CyberRAG.cyberrag_taa import CyberRAG
        self.model = CyberRAG(top_k=kwargs.get("top_k", 10),
                              task_type=kwargs.get("task_type", "taa"))

    def predict(self, report_text: str) -> str:
        return self.model.run(report_text)


@register("rag_intel")
class RAGIntelRunner:
    def __init__(self, **kwargs):
        from baselines.RAGIntel.rag_intel_taa import RAGIntel
        top_n = get_pipeline_param("rag_intel.top_n", 20)
        self.model = RAGIntel(
            top_k=kwargs.get("top_k", 10),
            top_n=top_n,
            task_type=kwargs.get("task_type", "taa"),
        )

    def predict(self, report_text: str) -> str:
        return self.model.run(report_text)


@register("search_o1")
class SearchO1Runner:
    def __init__(self, **kwargs):
        _mod = importlib.import_module("baselines.Search-o1-main.search_o1_taa")
        SearchO1 = _mod.SearchO1
        self.model = SearchO1(
            top_k=kwargs.get("top_k", 10),
            max_searches=kwargs.get("max_searches", 5),
            max_turns=kwargs.get("max_turns", 10),
            task_type=kwargs.get("task_type", "taa"),
        )

    def predict(self, report_text: str) -> str:
        return self.model.run(report_text)


# ── 子进程 worker ────────────────────────────────────────────────────
_WORKER_ARGS = {}


def _worker(method_name, report_text, sample_idx, queue: Queue):
    runner = _METHODS[method_name](**_WORKER_ARGS)
    t0 = time.time()
    try:
        pred = runner.predict(report_text)
        elapsed = time.time() - t0
        queue.put({"index": sample_idx, "prediction": pred, "elapsed_sec": round(elapsed, 1), "status": "ok"})
    except Exception as e:
        elapsed = time.time() - t0
        queue.put({"index": sample_idx, "prediction": "", "elapsed_sec": round(elapsed, 1),
                    "status": f"error: {e}", "traceback": traceback.format_exc()})


def run_single(method_name, runner_args, report_text, sample_idx, timeout):
    global _WORKER_ARGS
    _WORKER_ARGS = runner_args

    queue = Queue()
    p = Process(target=_worker, args=(method_name, report_text, sample_idx, queue))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        _log(f"  TIMEOUT after {timeout}s, killing...")
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        return {"index": sample_idx, "prediction": "", "elapsed_sec": timeout,
                "status": f"timeout after {timeout}s"}

    if queue.empty():
        return {"index": sample_idx, "prediction": "", "elapsed_sec": 0,
                "status": "error: subprocess died without result"}

    return queue.get()


# ── 主流程 ───────────────────────────────────────────────────────────
def run_for_method(method_name, args, dataset):
    output_path = args.output.format(method=method_name)
    runner_args = {
        "top_k": args.top_k,
        "iterations": args.iterations,
        "max_searches": args.max_searches,
        "max_turns": args.max_turns,
    }

    results = []
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        _log(f"[{method_name}] Loaded {len(results)} existing results")

    done_indices = {r["index"] for r in results}
    start, end = args.start, min(args.end, len(dataset))

    _log(f"[{method_name}] Samples [{start}, {end}), done={len(done_indices)}, timeout={args.timeout}s")

    for idx in range(start, end):
        if idx in done_indices:
            continue

        row = dataset[idx]
        _log(f"[{method_name}] [{idx+1}/{end}] GT={GROUND_TRUTH[idx]}")

        result = run_single(method_name, runner_args, row["Text"], idx, args.timeout)
        result["ground_truth"] = GROUND_TRUTH[idx]
        result["url"] = row["URL"]
        results.append(result)

        if result["status"] == "ok":
            tag = "+" if result["prediction"].strip().lower() == GROUND_TRUTH[idx].strip().lower() else "-"
            _log(f"  Pred: {result['prediction']}  [{tag}]  ({result['elapsed_sec']}s)")
        else:
            _log(f"  FAILED: {result['status']}")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 评估：total 固定为实验范围大小（数据集总数）；失败/超时样本预测为空 -> 计为 Incorrect
    in_range = [r for r in results if start <= r["index"] < end]
    total = end - start
    n_failed = sum(1 for r in in_range if r["status"] != "ok")
    gts = [r["ground_truth"] for r in in_range]
    preds = [r["prediction"] for r in in_range]
    ca, pa, details = compute_taa_accuracy(gts, preds)
    _log(f"[{method_name}] Correct={ca:.1f}%  Plausible={pa:.1f}%  ({total - n_failed}/{total}, failed as wrong: {n_failed})")

    report_path = output_path.replace(".json", "_report.json")
    with open(report_path, "w") as f:
        json.dump({"method": method_name, "correct_accuracy": round(ca, 2), "plausible_accuracy": round(pa, 2),
                    "total": total, "num_valid": total - n_failed, "num_failed": n_failed,
                    "details": details}, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run baseline TAA experiments")
    parser.add_argument("--method", required=True,
                        choices=list(_METHODS.keys()) + ["all"],
                        help="Baseline method name or 'all'")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output", default="experiments/results/{method}_taa_results.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    _default_top_k = get_global_top_k()
    parser.add_argument("--top-k", type=int, default=_default_top_k)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=get_global("timeout", 600))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs("experiments/results", exist_ok=True)

    data_path = args.data_path or os.path.join(os.path.dirname(__file__), "..", "data", "CTIBench-TAA")
    dataset = load_from_disk(data_path)

    methods = list(_METHODS.keys()) if args.method == "all" else [args.method]
    for m in methods:
        _log(f"\n{'='*60}")
        _log(f"Running baseline: {m}")
        _log(f"{'='*60}")
        run_for_method(m, args, dataset)


if __name__ == "__main__":
    main()
