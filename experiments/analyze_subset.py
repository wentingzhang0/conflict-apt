"""共识/冲突子集拆分 + 报告智能体双通道消融分析（学长 §4.8.1/4.8.2）

用法：python experiments/analyze_subset.py glm-4-9b [后缀=空]
拆分口径：fast_path=True = 共识路径（Φ=0 早退），False = 完整路径（冲突驱动）。
分别统计各组在两个子集上的 Correct，回答"增益来自哪里"：
R1(w/o 通道1)/R2(w/o 通道2) 与参照（clean off）在共识子集上的差 = 双通道贡献。
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
SL = sys.argv[1] if len(sys.argv) > 1 else "glm-4-9b"
SUF = sys.argv[2] if len(sys.argv) > 2 else ""


def load(tag):
    p = os.path.join(_HERE, "results", SL, f"srcnew_taa_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def acc(rows, subset):
    sys.path.insert(0, _HERE)
    from evaluate_taa import compute_taa_accuracy
    sel = [r for r in rows if r.get("fast_path") == (subset == "consensus")
           and r.get("status") == "ok"]
    gts = [r.get("ground_truth", "") for r in sel]
    preds = [r.get("prediction", "") for r in sel]
    correct, _, _ = compute_taa_accuracy(gts, preds)
    return len(sel), int(round(correct * len(sel) / 100.0))


def main():
    ref = load(f"clean{SUF}_off")
    if not ref:
        print("无参照 clean_off 数据"); return
    runs = [("参照（防御 off）", ref)]
    for tag, label in [(f"clean{SUF}_on", "防御 on"),
                       (f"clean_R1{SUF}_off", "R1 w/o 通道1"),
                       (f"clean_R2{SUF}_off", "R2 w/o 通道2")]:
        d = load(tag)
        if d:
            runs.append((label, d))
    print(f"{'组':16s} {'共识子集 n/对':>12s} {'完整子集 n/对':>12s} {'合计对':>6s}")
    for label, rows in runs:
        nc, oc = acc(rows, "consensus")
        nf, of = acc(rows, "full")
        print(f"{label:16s} {nc:>4d}/{oc:<4d}{'':>4s} {nf:>4d}/{of:<4d}{'':>4s} {oc+of:>4d}")
    # 增益归因（R1/R2 相对参照）
    _, r1 = next(((l, d) for l, d in runs if l.startswith("R1")), (None, None))
    _, r2 = next(((l, d) for l, d in runs if l.startswith("R2")), (None, None))
    for label, rows in [("R1", r1), ("R2", r2)]:
        if rows is None:
            continue
        for sub in ("consensus", "full"):
            n_ref, o_ref = acc(ref, sub)
            n_x, o_x = acc(rows, sub)
            if n_ref and n_x:
                print(f"{label} 在{sub}子集: 参照 {o_ref}/{n_ref} → {o_x}/{n_x} "
                      f"（{o_x - o_ref:+d}）")


if __name__ == "__main__":
    main()
