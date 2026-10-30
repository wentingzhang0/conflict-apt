"""
θ₂ 扫描实验 — 深入分析：Φ 分档门槛如何影响 反例深度 / 准确率 / 延迟

背景：主实验 TAA 用 θ₁=0.15, θ₂=0.35, i_max=1（二值档：Φ<θ₁→I=0，否则 I=1）。
本实验把 θ₁ 从 0 扫到 1（i_max=1 时 θ₁ 就是唯一的分档门槛，等效于扫描分档点），
每个 θ₁ 值跑全 50 条 TAA，记录：
    - correct/plausible accuracy（含冲突子集切分）
    - 平均反例轮数（触发反驳的样本比例）
    - 平均延迟 / llm_calls

产出: experiments/results/sweep_theta2/theta_{t}_taa_results.json (+_report.json)
分析: 冲突越重的样本越多进入反例检验 → 准确率-开销曲线（论文深入分析图）

用法:
    python experiments/run_theta2_sweep.py --theta 0.0        # 单个 θ 值全量
    python experiments/run_theta2_sweep.py --theta 0.0 --start 0 --end 3   # 调试
    python experiments/run_theta2_sweep.py --all              # 依次跑 0.0/0.15/0.35/0.55/0.75/1.01
"""
import os
import sys
import json
import argparse
from multiprocessing import Queue

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.utils.settings import apply_overrides, get_global, get_global_top_k

# θ₁ 扫描点：0.0 = 全部进入反例（恒最深）；1.01 = 全部不反驳（I 恒 0）≈ no_refute
# 注意 i_max=1 时 depth_from_phi 只有 0/1 两档：Φ<θ₁→0、Φ≥θ₁→1
THETA_GRID = [0.0, 0.15, 0.35, 0.55, 0.75, 1.01]


def _log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def variant_overrides(theta: float) -> dict:
    """θ₁=theta，θ₂ 保持 0.35（i_max=1 时无效），i_max=1 不变。"""
    return {
        "adaptive.frustration_rebuttal_gate": theta,
        "adaptive.depth_thresholds.taa.theta1": theta,
    }


def _expand_task_overrides(flat: dict) -> dict:
    """与 run_ablation 相同：展开 per-task 覆盖到 settings 单例。"""
    from src.utils.settings import get_settings
    cfg = get_settings()
    for path, val in flat.items():
        if not path.startswith("adaptive.depth_thresholds."):
            continue
        _, _, task, key = path.split(".", 3)
        cfg.setdefault("pipeline", {}).setdefault("adaptive", {}) \
            .setdefault("depth_thresholds", {}).setdefault(task, {})[key] = val
    return {k: v for k, v in flat.items()
            if not k.startswith("adaptive.depth_thresholds.")}


# 复用 run_ablation 的执行骨架（worker/超时/断点续跑/评测全部一致）
from run_ablation import run_variant as _run_variant  # noqa: E402
from run_taa import GROUND_TRUTH  # noqa: E402

# run_ablation.ABLATION_VARIANTS 是模块级 dict，被 run_variant 读取；
# 我们 monkey-patch 它以注入 θ 扫描覆盖，而不是复制整个 runner。
import run_ablation  # noqa: E402

# ---- 扩展 worker：额外记录 Phi/depth_I/fast_path（分档分析必需） ----
_orig_worker = run_ablation._worker


def _rich_worker(report_text, sample_idx, queue: Queue):
    from src.controller import Controller
    from src.tasks import TAATask
    from src.utils import llm_client

    llm_client.reset_llm_stats()
    task = TAATask()
    orchestrator = Controller(task=task, top_k=run_ablation._WORKER_ARGS["top_k"])

    import time as _t
    t0 = _t.time()
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
            "Phi": round(float(awm.Phi), 4),
            "depth_I": int(awm.depth_I),
            "fast_path": bool(awm.fast_path),
            "elapsed_sec": round(_t.time() - t0, 1),
            "status": "ok",
        }
        result.update(llm_client.snapshot_llm_stats())
        queue.put(result)
    except Exception as e:
        import traceback
        queue.put({
            "index": sample_idx, "prediction": "", "conclusion": "",
            "elapsed_sec": round(_t.time() - t0, 1),
            "status": f"error: {e}", "traceback": traceback.format_exc(),
        })


run_ablation._worker = _rich_worker


class _ThetaArg:
    """仿 run_ablation 的 args namespace。"""
    def __init__(self, output, start=0, end=50, timeout=None, resume=False,
                 data_path=None):
        self.output = output
        self.start = start
        self.end = end
        self.timeout = timeout or get_global("timeout", 600)
        self.resume = resume
        self.data_path = data_path


def run_theta(theta: float, args):
    tag = f"theta{theta:.2f}".replace(".", "p")     # e.g. theta0p35
    outdir = "experiments/results/sweep_theta2"
    os.makedirs(outdir, exist_ok=True)
    output = f"{outdir}/{tag}_taa_results.json"

    flat = variant_overrides(theta)
    rest = _expand_task_overrides(flat)
    apply_overrides(rest)
    run_ablation.ABLATION_VARIANTS = {tag: flat}     # run_variant 内部会再 _expand 一次（幂等）
    _log(f"θ₁={theta:.2f} overrides={flat}")

    ns = _ThetaArg(output=output, start=args.start, end=args.end,
                   timeout=args.timeout, resume=args.resume, data_path=args.data_path)
    _run_variant(tag, ns)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--theta", type=float, default=None, help="单个 θ₁ 值")
    p.add_argument("--all", action="store_true", help="跑完整网格")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=50)
    p.add_argument("--timeout", type=int, default=get_global("timeout", 600))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--data-path", default=None)
    args = p.parse_args()

    thetas = THETA_GRID if args.all else [args.theta]
    for t in thetas:
        if t is None:
            p.error("--theta 必须指定（或用 --all）")
        _log("=" * 60)
        run_theta(t, args)
