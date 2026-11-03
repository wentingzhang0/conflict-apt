"""
运行入口 —— 对一个分析请求完成冲突感知多智能体分析。

用法：
  python -m src_new.run --query-file path/to/report.txt --task taa
  python -m src_new.run --query "..." --task mcq
"""
import argparse
import json
import sys

from src_new.controller import Controller
from src_new.tasks import TAATask, ATETask, MCQTask, RCMTask

_TASKS = {"taa": TAATask, "ate": ATETask, "mcq": MCQTask, "rcm": RCMTask}


def awm_summary(awm) -> dict:
    return {
        "task_type": awm.task_type,
        "fast_path": awm.fast_path,
        "Phi": round(awm.Phi, 4),
        "k": awm.k,
        "depth_I": awm.depth_I,
        "iterations": awm.iteration,
        "n_evidence": len(awm.evidence_set),
        "n_confirmed_edges": len(awm.signed_graph_edges),
        "n_negative_edges": len(awm.get_negative_edges()),
        "n_refutation_rounds": len(awm.refutation_log),
        "final_conclusion": awm.final_conclusion,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="冲突感知多智能体 CTI 分析")
    p.add_argument("--query", default=None, help="分析请求全文")
    p.add_argument("--query-file", default=None, help="从文件读取分析请求")
    p.add_argument("--task", required=True, choices=list(_TASKS.keys()))
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--raw", action="store_true", help="输出原始 final_conclusion（不解析）")
    args = p.parse_args(argv)

    if args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as f:
            query = f.read()
    elif args.query:
        query = args.query
    else:
        print("error: provide --query or --query-file", file=sys.stderr)
        sys.exit(2)

    task = _TASKS[args.task]()
    awm = Controller(task, top_k=args.top_k).analyze(query)

    if args.raw:
        print(awm.final_conclusion)
    else:
        parsed = task.parse_output(awm.final_conclusion)
        summary = awm_summary(awm)
        summary["parsed_answer"] = parsed
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
