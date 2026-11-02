"""
消融实验 runner — ICASSP 论文消融表（TAA 任务，50 条）

三个变体，一一对应三创新点：
    full          完整方法（= 已有 taa_results.json，可 --reuse-full 直接引用不重跑）
    no_adaptive   恒完整路径：关 PATH 早退 + Φ 分档恒定最深（θ₁=θ₂=0 → I 恒 = I_max）
    no_refute     恒不反驳：I_max=0（关反例证伪）
    no_canonical  规范键关闭：冲突判定退化为纯嵌入相似度（B2）

用法:
    python experiments/run_ablation.py --ablation no_adaptive              # 单变体全量
    python experiments/run_ablation.py --ablation no_adaptive --start 0 --end 3   # 调试
    python experiments/run_ablation.py --ablation all                      # 顺序跑全部变体
输出: experiments/results/ablation_{variant}_taa_results.json (+_report.json)
每条结果含 llm_calls / llm_latency_sec（效率轴）。
"""
import os
import sys
import json
import time
import argparse
from multiprocessing import Process, Queue
from datasets import load_from_disk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src_new.utils.settings import apply_overrides, get_global, get_global_top_k

# ── 消融变体 → settings 覆盖项 ─────────────────────────────────────────
# 覆盖经 apply_overrides 注册，get_pipeline_param 优先读取（见 utils/settings.py）
ABLATION_VARIANTS = {
    "full": {},
    # A 组：恒完整路径。PATH 门控条件恒 False（视为有负边）、Φ 分档恒最深。
    # theta1=0 且 theta2=0 → depth_from_phi 恒返回 i_max；force_full_path 见 metrics.gating
    "no_adaptive": {
        "adaptive.force_full_path": True,
        "adaptive.frustration_rebuttal_gate": 0.0,
        "adaptive.frustration_depth2": 0.0,
        # per-task 门槛也要覆盖（settings.yaml 的 depth_thresholds 优先于全局值）
        "adaptive.depth_thresholds.taa.theta1": 0.0,
        "adaptive.depth_thresholds.taa.theta2": 0.0,
    },
    # C 组：恒不反驳（I_max=0，反例证伪关闭；建图/分区/多假设保留）
    "no_refute": {
        "adaptive.rebuttal_max_rounds": 0,
        "adaptive.depth_thresholds.taa.i_max": 0,
    },
    # B 组：规范键关闭 → 纯嵌入匹配（只能 support/none，判不出 conflict）
    "no_canonical": {
        "canonical_match.use_canonical_keys": False,
    },
}

# per-task 覆盖键的解析：metrics._depth_thresholds 读的是 dict 形式
# depth_thresholds.taa.theta1 这类点路径需要展开成 dict 注入 —— 在 _apply_variant 处理


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _expand_task_overrides(flat: dict) -> dict:
    """把 adaptive.depth_thresholds.{task}.{key} 展开为 settings 内同构 dict 覆盖。

    实现方式：直接改 get_settings() 单例中的对应嵌套值（子进程 fork 前调用，
    worker 继承修改后的配置；不污染磁盘上的 settings.yaml）。"""
    from src_new.utils.settings import get_settings
    cfg = get_settings()
    for path, val in flat.items():
        if not path.startswith("adaptive.depth_thresholds."):
            continue
        _, _, task, key = path.split(".", 3)
        cfg.setdefault("pipeline", {}).setdefault("adaptive", {}) \
            .setdefault("depth_thresholds", {}).setdefault(task, {})[key] = val
    # 剩余扁平覆盖走 apply_overrides
    rest = {k: v for k, v in flat.items()
            if not k.startswith("adaptive.depth_thresholds.")}
    apply_overrides(rest)


# ground truth 与 run_taa.py 保持一致
from run_taa import GROUND_TRUTH, load_taa_dataset, run_single  # noqa: E402

_WORKER_ARGS = {}


def _worker(report_text, sample_idx, queue: Queue):
    from src_new.controller import Controller
    from src_new.tasks import TAATask
    from src_new.utils import llm_client

    llm_client.reset_llm_stats()  # 每样本清零调用统计

    task = TAATask()
    orchestrator = Controller(task=task, top_k=_WORKER_ARGS["top_k"])

    t0 = time.time()
    try:
        query = task.format_query(report_text=report_text)
        awm = orchestrator.analyze(query=query)
        pred = task.parse_output(awm.final_conclusion)

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
            "elapsed_sec": round(time.time() - t0, 1),
            "status": "ok",
        }
        result.update(llm_client.snapshot_llm_stats())  # 效率轴
        queue.put(result)
    except Exception as e:
        import traceback
        queue.put({
            "index": sample_idx, "prediction": "", "conclusion": "",
            "elapsed_sec": round(time.time() - t0, 1),
            "status": f"error: {e}", "traceback": traceback.format_exc(),
        })


def run_single_ablation(report_text, sample_idx, timeout):
    """子进程隔离跑单条（覆盖变体在 fork 前已注入全局 settings/overrides）"""
    global _WORKER_ARGS
    _WORKER_ARGS = {"top_k": get_global_top_k()}

    queue = Queue()
    p = Process(target=_worker, args=(report_text, sample_idx, queue))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        _log(f"  TIMEOUT after {timeout}s, killing ...")
        p.terminate(); p.join(5)
        if p.is_alive():
            p.kill(); p.join()
        return {"index": sample_idx, "prediction": "", "conclusion": "",
                "elapsed_sec": timeout, "llm_calls": None,
                "status": f"timeout after {timeout}s"}
    if queue.empty():
        return {"index": sample_idx, "prediction": "", "conclusion": "",
                "elapsed_sec": 0, "llm_calls": None,
                "status": "error: subprocess died without result"}
    return queue.get()


def run_variant(variant: str, args):
    overrides = ABLATION_VARIANTS[variant]
    _expand_task_overrides(overrides)  # fork 前注入，worker 继承
    _log(f"Variant [{variant}] overrides: {overrides}")

    dataset = load_taa_dataset(args.data_path)
    output = args.output or f"experiments/results/ablation_{variant}_taa_results.json"

    results = []
    if args.resume and os.path.exists(output):
        with open(output) as f:
            results = json.load(f)
        _log(f"Loaded {len(results)} existing results from {output}")

    done = {r["index"] for r in results}
    end = min(args.end, len(dataset))

    # BM25 预热（同 run_taa）
    try:
        from src_new.retrieval.weaviate_retriever import _bm25_index
        _bm25_index.build()
        _log(f"BM25 index ready ({_bm25_index.N} docs)")
    except Exception as e:
        _log(f"BM25 warm-up skipped: {e}")

    for idx in range(args.start, end):
        if idx in done:
            continue
        row = dataset[idx]
        _log(f"[{variant}][{idx+1}/{end}] GT: {GROUND_TRUTH[idx]}")
        result = run_single_ablation(row["Text"], idx, args.timeout)
        result["ground_truth"] = GROUND_TRUTH[idx]
        result["url"] = row["URL"]
        result["variant"] = variant
        results.append(result)

        if result["status"] == "ok":
            ok = result["prediction"].strip().lower() == GROUND_TRUTH[idx].strip().lower()
            _log(f"  Pred: {result['prediction']} [{'+' if ok else '-'}] "
                 f"llm_calls={result.get('llm_calls')} "
                 f"{result.get('elapsed_sec')}s")
        else:
            _log(f"  FAILED: {result['status']}")

        with open(output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 汇总（口径与 run_taa 一致：total=范围总数，失败按错计） ──
    evaluate_variant(variant, results, args.start, end, output)


def evaluate_variant(variant, results, start, end, output):
    sys.path.insert(0, os.path.dirname(__file__))
    from evaluate_taa import compute_taa_accuracy
    in_range = [r for r in results if start <= r["index"] < end]
    if not in_range:
        _log(f"[{variant}] no results in range, skip evaluation")
        return
    gts = [r["ground_truth"] for r in in_range]
    preds = [r["prediction"] for r in in_range]
    correct_acc, plausible_acc, details = compute_taa_accuracy(gts, preds)

    n_ok = [r for r in in_range if r["status"] == "ok"]
    avg_calls = (sum(r["llm_calls"] for r in n_ok if r.get("llm_calls") is not None)
                 / max(len([r for r in n_ok if r.get("llm_calls") is not None]), 1))
    avg_latency = (sum(r["elapsed_sec"] for r in n_ok) / max(len(n_ok), 1))

    _log(f"\n[{variant}] Correct={correct_acc:.1f}%  Plausible={plausible_acc:.1f}%  "
         f"avg_llm_calls={avg_calls:.1f}  avg_wall={avg_latency:.1f}s")

    report_path = output.replace(".json", "_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "variant": variant,
            "correct_accuracy": round(correct_acc, 2),
            "plausible_accuracy": round(plausible_acc, 2),
            "total": end - start,
            "num_valid": len(n_ok),
            "avg_llm_calls": round(avg_calls, 1),
            "avg_wall_sec": round(avg_latency, 1),
            "details": details,
        }, f, indent=2, ensure_ascii=False)
    _log(f"Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TAA ablation experiments")
    parser.add_argument("--ablation", required=True,
                        choices=list(ABLATION_VARIANTS) + ["all"],
                        help="消融变体（all = 顺序跑 no_adaptive/no_refute/no_canonical）")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=get_global("timeout", 600))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs("experiments/results", exist_ok=True)
    variants = list(ABLATION_VARIANTS) if args.ablation == "all" else [args.ablation]
    if args.ablation == "all":
        variants = [v for v in variants if v != "full"]  # full 用已有主结果即可

    for v in variants:
        _log("=" * 60)
        run_variant(v, args)
