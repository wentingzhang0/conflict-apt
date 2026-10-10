"""
metrics.py —— 所有确定性计算集中于此（创新点 1 + 创新点 2 数值部分）

设计原则（方案 6.2）：纯数值、无 LLM；同输入同输出、可复现、可单测。

双入口：
  1) 纯函数 API：controller.py 直接 import 调用（零序列化开销）。
  2) CLI / JSON：供 Claude Code 团队模式复用同一份逻辑。
        python3 -m src_new.metrics <cmd> --graph awm.json [--task taa] [--hypothesis hyp.json]

实现说明：
  - Φ 的 T⁻ 数所有 sign=−1 的负边（edge_class=='confirmed' & sign==-1），含归因/时间线
    灰区判为 conflict 的边与 q↔证据 conflict 边。Φ 是本机制唯一的算力控制信号（驱动 I）。
  - h_graph 仅用 confirmed 边；公式 |C_h|·ρ_h − λ·κ_h（ρ 簇内 support 加权密度、κ 落在
    S(h) 节点的 conflict 总权，见 signed_graph.graph_strength_for_hypothesis）。
  - PATH 门控只看有无 sign=−1 负边（二值，无 uncertain）。
"""
from __future__ import annotations
import sys
import json
import argparse
from itertools import combinations

from src_new.core.awm import AWM
from src_new.core.signed_graph import QUERY_NODE, graph_strength_for_hypothesis
from src_new.core.spectral_partition import spectral_partition as _spectral_partition
from src_new.utils.settings import get_pipeline_param


# ---------- Φ：签名三角形挫折 ----------
def frustration(awm: AWM) -> dict:
    """Φ = |T⁻|/|T|，T⁻ 为含奇数条 confirmed 负边的三角形（三边符号乘积为负）。
    q↔证据边由 build_edges/build_query_edges 产出，已在 signed_graph_edges 中（3.2.4）。"""
    nodes = [QUERY_NODE] + [ev.doc_id for ev in awm.evidence_set]
    if len(nodes) < 3:
        return {"Phi": 0.0, "T": 0, "T_minus": 0}

    sign = {}
    for e in awm.signed_graph_edges:
        if e.edge_class == "confirmed" and e.sign != 0:
            sign[(e.source, e.target)] = e.sign
            sign[(e.target, e.source)] = e.sign

    total = 0
    minus = 0
    for a, b, c in combinations(nodes, 3):
        s_ab = sign.get((a, b), 0)
        s_bc = sign.get((b, c), 0)
        s_ac = sign.get((a, c), 0)
        if s_ab == 0 or s_bc == 0 or s_ac == 0:
            continue
        total += 1
        if s_ab * s_bc * s_ac < 0:
            minus += 1
    phi = float(minus / total) if total > 0 else 0.0
    return {"Phi": phi, "T": total, "T_minus": minus}


# ---------- 谱分区 ----------
def partition(awm: AWM) -> dict:
    return _spectral_partition(awm)


# ---------- h_graph ----------
def h_graph(awm: AWM, hypothesis) -> dict:
    """单假设图量化强度（仅 confirmed 边）。"""
    return graph_strength_for_hypothesis(awm, hypothesis)


def delta_h_graph(h_before: float, h_after: float) -> float:
    """Δh_graph = h_graph(前) − h_graph(后)。"""
    return float(h_before - h_after)


# ---------- 门控：PATH / DEPTH（单轴：推理强度）----------
def has_confirmed_negative(awm: AWM) -> bool:
    return any(e.edge_class == "confirmed" and e.sign == -1 for e in awm.signed_graph_edges)


def _depth_thresholds(task_type: str) -> tuple:
    """任务自适应深度门槛（方案 3.1.3/3.1.8）：θ₁/θ₂/I_max 按任务从
    adaptive.depth_thresholds 读取，缺省回退全局默认。"""
    theta1 = get_pipeline_param("adaptive.frustration_rebuttal_gate", 0.15)
    theta2 = get_pipeline_param("adaptive.frustration_depth2", 0.35)
    i_max = int(get_pipeline_param("adaptive.rebuttal_max_rounds", 2))
    cfg = get_pipeline_param("adaptive.depth_thresholds", None)
    if isinstance(cfg, dict) and task_type in cfg and isinstance(cfg[task_type], dict):
        t = cfg[task_type]
        theta1 = float(t.get("theta1", theta1))
        theta2 = float(t.get("theta2", theta2))
        i_max = int(t.get("i_max", i_max))
    return theta1, theta2, i_max


def depth_from_phi(phi: float, task_type: str = "") -> int:
    """DEPTH 分档（含 0 档，任务自适应门槛）：Φ<θ₁→0；θ₁≤Φ<θ₂→1；Φ≥θ₂→I_max。"""
    theta1, theta2, i_max = _depth_thresholds(task_type)
    if phi < theta1:
        return 0
    if phi < theta2:
        return 1
    return i_max


def gating(awm: AWM, task_type: str = "") -> dict:
    """PATH/DEPTH 门控一次性输出（单轴推理强度；原 ROUTING 旋钮已删除，见方案 3.1.3/8.2）。

    PATH 早退判定：方案原文"无 sign=−1 负边→全正图天然单簇 k=1"在多源 RAG 场景并不必然成立——
    证据来自不同组织，即使无 conflict 边，support 边也可能形成多个连通分量（多个候选组织）。
    故改为：始终做谱分区（小图开销可忽略），仅当【真单簇 k==1 且无负边】才走共识快路径；
    否则（有负边 或 k>1）走完整多假设路径，由假设评估选优，避免单一共识假设受 LLM 偏向
    （如总是猜最知名组织）影响。"""
    phi = frustration(awm)
    has_neg = has_confirmed_negative(awm)
    I = depth_from_phi(phi["Phi"], task_type)

    # 消融 A0/A1（adaptive.force_full_path=true）：恒完整路径——
    # 无视 PATH 早退条件与 Φ 分档，I 恒取 i_max（θ₁/θ₂ 已被覆盖为 0）
    if get_pipeline_param("adaptive.force_full_path", False):
        _, _, i_max = _depth_thresholds(task_type)
        return {
            "has_neg": has_neg,
            "full_path": True,
            "Phi": phi["Phi"],
            "T": phi["T"],
            "T_minus": phi["T_minus"],
            "k": partition(awm)["k"],
            "clusters": partition(awm)["clusters"],
            "labels": partition(awm).get("labels", {}),
            "I": i_max,
            "forced_full": True,
        }

    sp = partition(awm)                      # 始终分区（节点≤10，开销可忽略）
    k = sp["k"]
    full_path = has_neg or k > 1
    if not full_path:                        # 真单簇且无负边 → 共识早退
        sp = {"k": 1, "clusters": {}, "labels": {}}

    return {
        "has_neg": has_neg,
        "full_path": full_path,
        "Phi": phi["Phi"],
        "T": phi["T"],
        "T_minus": phi["T_minus"],
        "k": k,
        "clusters": sp["clusters"],
        "labels": sp.get("labels", {}),
        "I": I,
    }


# ===================== CLI / JSON 入口 =====================
def _load_awm(path: str) -> AWM:
    with open(path, "r", encoding="utf-8") as f:
        return AWM.from_json(f.read())


def _dump(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, default=str))


def _hypothesis_from_json(path: str):
    from src_new.core.awm import Hypothesis
    with open(path, "r", encoding="utf-8") as f:
        return Hypothesis(**json.loads(f.read()))


def main(argv=None):
    parser = argparse.ArgumentParser(description="确定性度量（纯数值，无 LLM）")
    parser.add_argument("command", choices=["frustration", "partition",
                                            "h_graph", "gating"])
    parser.add_argument("--graph", required=True, help="AWM JSON 文件路径")
    parser.add_argument("--task", default="", help="任务类型（taa/ate/mcq/rcm）")
    parser.add_argument("--hypothesis", default=None, help="假设 JSON（h_graph 用）")
    args = parser.parse_args(argv)

    awm = _load_awm(args.graph)
    if args.command == "frustration":
        _dump(frustration(awm))
    elif args.command == "partition":
        sp = partition(awm)
        _dump({"k": sp["k"], "clusters": sp["clusters"]})
    elif args.command == "h_graph":
        if not args.hypothesis:
            print(json.dumps({"error": "--hypothesis required"})); sys.exit(2)
        hyp = _hypothesis_from_json(args.hypothesis)
        _dump(h_graph(awm, hyp))
    elif args.command == "gating":
        _dump(gating(awm, args.task))


if __name__ == "__main__":
    main()
