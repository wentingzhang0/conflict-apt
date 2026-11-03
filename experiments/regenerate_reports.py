"""
从已保存的 *_results.json 重新生成 *_report.json。

修正后的口径（与 run_*.py 一致）：
  - total = 数据集总数（不是"跑成功了多少个"）
  - 失败 / 超时 / 无答案 一律按错误计，绝不从分母里剔除
  - 只对"已跑完全量"的结果文件重写报告；未跑完的文件会告警并跳过
    （与在线脚本一致：在线脚本只在主循环跑完 [start,end) 后才写报告）

用法:
    python experiments/regenerate_reports.py            # 重算 experiments/results 下全部
    python experiments/regenerate_reports.py --dir experiments/results/glm-9b
"""
import os
import re
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 每个任务的数据集总数（total 固定为此值）
DATASET_SIZE = {"rcm": 1000, "mcq": 2500, "taa": 50, "ate": 60}


def _task_of(path: str):
    name = os.path.basename(path).lower()
    for t in ("rcm", "mcq", "taa", "ate"):
        if t in name:
            return t
    return None


# ── 各任务的评估 ─────────────────────────────────────────────────────

def _rcm_correct(pred, gt):
    a = (pred or "").strip().upper().replace(" ", "")
    b = (gt or "").strip().upper().replace(" ", "")
    return a == b


def _mcq_correct(pred, gt):
    return (pred or "").strip().upper() == (gt or "").strip().upper()


_TID_RE = re.compile(r'(T\d{3,4})(?:\.\d{3})?')


def _parse_tids(text):
    return {m.upper() for m in _TID_RE.findall(text or "")}


def _ate_eval_all(rows):
    """micro-F1 over ALL rows (failed -> empty pred -> 全 GT 计入 FN)。"""
    tp = fp = fn = 0
    for r in rows:
        pred_set = _parse_tids(r.get("prediction", "") if r.get("status") == "ok" else "")
        gt_set = _parse_tids(r.get("ground_truth", ""))
        tp += len(pred_set & gt_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall, tp, fp, fn


def _taa_eval_all(rows):
    from evaluate_taa import compute_taa_accuracy
    gts = [r.get("ground_truth", "") for r in rows]
    preds = [r.get("prediction", "") if r.get("status") == "ok" else "" for r in rows]
    return compute_taa_accuracy(gts, preds)  # (correct_acc, plausible_acc, details)


# ── 报告重算 ─────────────────────────────────────────────────────────

def regenerate(results_path):
    task = _task_of(results_path)
    if task is None:
        return None
    total = DATASET_SIZE[task]

    with open(results_path) as f:
        results = json.load(f)

    in_range = [r for r in results if 0 <= r.get("index", -1) < total]
    n_attempted = len(in_range)
    if n_attempted < total:
        # 未跑完全量：与在线脚本一致，不写报告（避免用残缺数据误导）
        return {"task": task, "path": results_path, "skipped": True,
                "attempted": n_attempted, "total": total}

    n_failed = sum(1 for r in in_range if r.get("status") != "ok")
    report = {"task": task, "total": total, "num_failed": n_failed}

    if task == "rcm":
        correct = sum(1 for r in in_range
                      if r.get("status") == "ok" and _rcm_correct(r.get("prediction", ""), r.get("ground_truth", "")))
        report.update({"accuracy": round(correct / total * 100, 2), "correct": correct})
        mode = "baseline" if any(k in os.path.basename(results_path) for k in
                                 ("naive_rag", "iter_retgen", "arbgraph", "cyberrag", "search_o1", "rag_intel")) else "main"
        report["mode"] = mode
    elif task == "mcq":
        correct = sum(1 for r in in_range
                      if r.get("status") == "ok" and _mcq_correct(r.get("prediction", ""), r.get("ground_truth", "")))
        report.update({"accuracy": round(correct / total * 100, 2), "correct": correct})
        for key in ("naive_rag", "iter_retgen", "arbgraph", "cyberrag", "search_o1", "rag_intel"):
            if key in os.path.basename(results_path):
                report["method"] = key
                break
    elif task == "ate":
        f1, p, r, tp, fp, fn = _ate_eval_all(in_range)
        report.update({"f1": round(f1 * 100, 2), "precision": round(p * 100, 2),
                       "recall": round(r * 100, 2), "total_tp": tp, "total_fp": fp, "total_fn": fn})
    elif task == "taa":
        ca, pa, details = _taa_eval_all(in_range)
        report.update({"correct_accuracy": round(ca, 2), "plausible_accuracy": round(pa, 2),
                       "num_valid": total - n_failed, "details": details})
        for key in ("naive_rag", "iter_retgen", "arbgraph", "cyberrag", "search_o1", "rag_intel"):
            if key in os.path.basename(results_path):
                report["method"] = key
                break

    report_path = results_path.replace(".json", "_report.json")
    # 保留原 report 里可能存在的额外字段？不保留——以重算结果为准。
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return {"task": task, "path": report_path, "skipped": False, "report": report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="experiments/results",
                        help="扫描该目录（递归）下的 *_results.json")
    args = parser.parse_args()

    root = args.dir
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(__file__), "..", root)

    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith("_results.json") and not fn.endswith("_report.json"):
                files.append(os.path.join(dp, fn))
    files.sort()

    print(f"Found {len(files)} result files under {root}\n" + "=" * 60)
    for fp in files:
        res = regenerate(fp)
        if res is None:
            print(f"[skip] {fp}  (unknown task)")
        elif res.get("skipped"):
            print(f"[WARN] {fp}  INCOMPLETE: {res['attempted']}/{res['total']} attempted — report NOT written")
        else:
            r = res["report"]
            acc = r.get("accuracy") or r.get("correct_accuracy") or r.get("f1")
            print(f"[ok]   {res['path']}  total={r.get('total')} failed={r.get('num_failed')} "
                  f"metric={acc}")


if __name__ == "__main__":
    main()
