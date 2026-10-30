"""CORE-2 信号审计分析：毒/真实证据分离度、clean 误标率、C2 分数分布（τ 调参依据）

用法：python experiments/analyze_core2_audit.py [glm-4-9b]
"""
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
SL = sys.argv[1] if len(sys.argv) > 1 else "glm-4-9b"
BASE = os.path.join(_HERE, "results", SL)


def analyze(setting):
    path = os.path.join(BASE, f"srcnew_taa_{setting}_v3audit_on.json")
    if not os.path.exists(path):
        print(f"[{setting}] (missing)")
        return
    rows = json.load(open(path))
    per_signal = defaultdict(lambda: {"poison": 0, "real": 0})
    c2_scores = {"poison": [], "real": []}
    c1_scores = {"poison": [], "real": []}
    n_poison_ev = n_real_ev = 0
    n_quar_p = n_quar_r = 0
    queries_poison_flagged = queries_poison_present = 0
    for r in rows:
        is_p = lambda d: str(d).startswith("poison_")
        flagged_docs = {f["doc_id"] for f in r.get("core2_flags", [])}
        n_poison_ev += r.get("poison_in_topk", 0)
        n_real_ev += r.get("evidence_count", 0) - r.get("poison_in_topk", 0)
        poison_in_set = r.get("poison_in_topk", 0) > 0
        if poison_in_set:
            queries_poison_present += 1
        for f in r.get("core2_flags", []):
            side = "poison" if is_p(f["doc_id"]) else "real"
            for s in f["signals"]:
                per_signal[s][side] += 1
                if s == "C2_actor_divergence":
                    c2_scores[side].append(f["scores"][s])
                if s == "C1_ally_divergence":
                    c1_scores[side].append(f["scores"][s])
        if poison_in_set and any(is_p(d) for d in flagged_docs):
            queries_poison_flagged += 1
        for d in r.get("quarantined", []):
            if is_p(d):
                n_quar_p += 1
            else:
                n_quar_r += 1

    def q(xs):
        if not xs:
            return "—"
        xs = sorted(xs)
        return f"min={xs[0]:.3f} med={xs[len(xs)//2]:.3f} max={xs[-1]:.3f} n={len(xs)}"

    print(f"== {setting} (n={len(rows)} queries) ==")
    print(f"  证据：poison {n_poison_ev} 篇 / real {n_real_ev} 篇")
    for s in ("C1_ally_divergence", "C2_actor_divergence", "C3_copublication_band"):
        d = per_signal[s]
        tot = d["poison"] + d["real"]
        rate_p = d["poison"] / max(n_poison_ev, 1)
        rate_r = d["real"] / max(n_real_ev, 1)
        print(f"  {s:24s} poison {d['poison']:3d} ({rate_p:5.1%})  "
              f"real {d['real']:3d} ({rate_r:5.1%})")
    print(f"  C2 分数 poison: {q(c2_scores['poison'])}")
    print(f"  C2 分数 real  : {q(c2_scores['real'])}")
    print(f"  C1 分数 poison: {q(c1_scores['poison'])}")
    print(f"  C1 分数 real  : {q(c1_scores['real'])}")
    print(f"  含毒 query 被 CORE-2 命中：{queries_poison_flagged}/{queries_poison_present}")
    print(f"  隔离：poison {n_quar_p} / real {n_quar_r}")
    correct = sum(1 for r in rows
                  if r.get("prediction", "").lower() == r.get("ground_truth", "").lower())
    print(f"  Correct {correct}/{len(rows)}")


if __name__ == "__main__":
    for s in ("clean", "r1", "r2", "adapt"):
        analyze(s)
