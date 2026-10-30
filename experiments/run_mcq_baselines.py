"""
Baseline MCQ 实验 — 用各 baseline 跑 CTIBench-MCQ 2500 道
用法:
    python experiments/run_mcq_baselines.py --method naive_rag
    python experiments/run_mcq_baselines.py --method naive_rag --start 0 --end 10
    python experiments/run_mcq_baselines.py --method all
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

def _log(msg):
    """带时间戳的即时输出"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── 共享的 MCQ baseline runner 逻辑 ──────────────────────────────────

def _mcq_worker(method_name, question, options_text, sample_idx, queue: Queue):
    """子进程 worker：构造完整 question 文本，调用 baseline"""
    runner = _METHODS[method_name](**_WORKER_ARGS)
    t0 = time.time()
    try:
        # 所有 baseline 统一用 question+options 作为输入
        full_query = question + "\n\n" + options_text
        pred = runner.predict(full_query)
        elapsed = time.time() - t0
        queue.put({"index": sample_idx, "prediction": pred, "elapsed_sec": round(elapsed, 1), "status": "ok"})
    except Exception as e:
        elapsed = time.time() - t0
        queue.put({"index": sample_idx, "prediction": "A", "elapsed_sec": round(elapsed, 1),
                    "status": f"error: {e}", "traceback": traceback.format_exc()})


_WORKER_ARGS = {}


def run_single_baseline(method_name, runner_args, question, options_text, sample_idx, timeout):
    global _WORKER_ARGS
    _WORKER_ARGS = runner_args

    queue = Queue()
    p = Process(target=_mcq_worker, args=(method_name, question, options_text, sample_idx, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        return {"index": sample_idx, "prediction": "A", "elapsed_sec": timeout,
                "status": f"timeout after {timeout}s"}

    if queue.empty():
        return {"index": sample_idx, "prediction": "A", "elapsed_sec": 0,
                "status": "error: subprocess died"}

    return queue.get()


# ── Baseline 注册表（MCQ 版） ──────────────────────────────────────────
_METHODS = {}


def register(name):
    def decorator(cls):
        _METHODS[name] = cls
        return cls
    return decorator


@register("naive_rag")
class NaiveRAGMCQ:
    def __init__(self, **kwargs):
        from baselines.NavieRAG.naive_rag import NaiveRAG
        self.model = NaiveRAG(top_k=kwargs.get("top_k", 10))

    def predict(self, query_text: str) -> str:
        """baseline 的 run() 按 report 文本检索；MCQ 用 question 检索即可"""
        from baselines.shared import retrieve, format_context, chat, parse_mcq_prediction
        from baselines.shared import MCQ_SYSTEM_PROMPT, MCQ_RAG_USER_TEMPLATE
        docs = retrieve(query_text, top_k=self.model.top_k)
        if not docs:
            resp = chat(prompt=query_text + "\n\nOutput: <answer>X</answer>",
                        system=MCQ_SYSTEM_PROMPT, temperature=0.0)
            return parse_mcq_prediction(resp)
        context = format_context(docs)
        # 从 query_text 中提取 question 和 options（由 _mcq_worker 拼好）
        prompt = MCQ_RAG_USER_TEMPLATE.format(context=context,
                                               question="see above", options="see above")
        # 简化：直接把检索结果 + 原始问题一起给 LLM
        prompt = f"{MCQ_SYSTEM_PROMPT}\n\n--- Retrieved Documents ---\n{context}\n\n--- Question ---\n{query_text}\n\nOutput: <answer>X</answer>"
        resp = chat(prompt=prompt, system=MCQ_SYSTEM_PROMPT, temperature=0.0)
        return parse_mcq_prediction(resp)


@register("iter_retgen")
class IterRetGenMCQ:
    def __init__(self, **kwargs):
        import importlib
        _mod = importlib.import_module("baselines.ITER-RETGEN.iter_retgen")
        IterRetGen = _mod.IterRetGen
        self.model = IterRetGen(top_k=kwargs.get("top_k", 10), iterations=kwargs.get("iterations", 3))

    def predict(self, query_text: str) -> str:
        from baselines.shared import retrieve, format_context, chat, parse_mcq_prediction
        from baselines.shared import MCQ_SYSTEM_PROMPT
        all_docs = []
        current_query = query_text
        last_response = ""

        for t in range(self.model.iterations):
            docs = retrieve(current_query, top_k=self.model.top_k)
            seen = {(d.get("source_file", ""), d.get("section_title", "")) for d in all_docs}
            for d in docs:
                key = (d.get("source_file", ""), d.get("section_title", ""))
                if key not in seen:
                    all_docs.append(d)
                    seen.add(key)

            context = format_context(all_docs) if all_docs else ""
            prompt = f"Answer this cybersecurity MCQ using the reference documents.\n\n"
            if context:
                prompt += f"--- Retrieved Documents ---\n{context}\n\n"
            prompt += f"--- Question ---\n{query_text}\n\nOutput: <answer>X</answer>"

            last_response = chat(prompt=prompt, system=MCQ_SYSTEM_PROMPT, temperature=0.0)
            if t < self.model.iterations - 1:
                current_query = f"{query_text}\n\nPrevious analysis: {last_response[:300]}"

        return parse_mcq_prediction(last_response)


@register("arbgraph")
class ArbGraphMCQ:
    def __init__(self, **kwargs):
        from baselines.ArbGraph.arbgraph_taa import ArbGraph
        self.model = ArbGraph(top_k=kwargs.get("top_k", 10))
        self.top_k = kwargs.get("top_k", 10)

    def predict(self, query_text: str) -> str:
        """ArbGraph pipeline with MCQ-specific prompts (consistent with other baselines)."""
        from baselines.ArbGraph.arbgraph_taa import (
            extract_claims, align_and_merge_claims,
            build_evidence_graph, arbitrate,
        )
        from baselines.shared import (
            retrieve, format_context, chat, parse_mcq_prediction,
            MCQ_SYSTEM_PROMPT,
        )
        from src.utils.settings import get_global

        docs = retrieve(query_text, top_k=self.top_k)
        if not docs:
            prompt = f"{query_text}\n\nOutput: <answer>X</answer>"
            resp = chat(prompt=prompt, system=MCQ_SYSTEM_PROMPT, temperature=0.0)
            return parse_mcq_prediction(resp)

        # ArbGraph pipeline: claim extraction → alignment → graph → arbitration
        all_claims = []
        ev_max = get_global("intermediate_evidence_max", 600)
        for doc in docs:
            content = doc.get("content", "")
            source_file = doc.get("source_file", "unknown")
            if len(content) > ev_max * 2:
                content = content[:ev_max * 2]
            claims = extract_claims(content, source_file)
            all_claims.extend(claims)

        context = format_context(docs)
        if not all_claims:
            prompt = (f"{MCQ_SYSTEM_PROMPT}\n\n--- Retrieved Documents ---\n{context}\n\n"
                      f"--- Question ---\n{query_text}\n\nOutput: <answer>X</answer>")
            resp = chat(prompt=prompt, system=MCQ_SYSTEM_PROMPT, temperature=0.0)
            return parse_mcq_prediction(resp)

        merged = align_and_merge_claims(all_claims)
        graph = build_evidence_graph(merged, query_text)
        result = arbitrate(graph, query_text)
        validated = result["validated"]

        validated_text = "\n".join(
            f"- {c['text']} (confidence={c['confidence']}, source={c['source_id']})"
            for c in validated[:20]
        )

        prompt = (f"{MCQ_SYSTEM_PROMPT}\n\n--- Retrieved Documents ---\n{context}\n\n"
                  f"--- Validated Evidence Claims ({len(validated)} claims) ---\n{validated_text}\n\n"
                  f"--- Question ---\n{query_text}\n\nOutput: <answer>X</answer>")
        resp = chat(prompt=prompt, system=MCQ_SYSTEM_PROMPT, temperature=0.0)
        return parse_mcq_prediction(resp)


@register("cyberrag")
class CyberRAGMCQ:
    def __init__(self, **kwargs):
        from baselines.CyberRAG.cyberrag_taa import CyberRAG
        self.model = CyberRAG(top_k=kwargs.get("top_k", 10))
        self.top_k = kwargs.get("top_k", 10)

    def predict(self, query_text: str) -> str:
        """CyberRAG pipeline with MCQ prompts: generate → validate → regenerate."""
        from baselines.shared import (
            retrieve, format_context, chat, parse_mcq_prediction,
            MCQ_SYSTEM_PROMPT,
        )
        import re

        docs = retrieve(query_text, top_k=self.top_k)
        if not docs:
            resp = chat(prompt=f"{query_text}\n\nOutput: <answer>X</answer>",
                        system=MCQ_SYSTEM_PROMPT, temperature=0.0)
            return parse_mcq_prediction(resp)

        context = format_context(docs)
        base_prompt = (f"--- Retrieved Documents ---\n{context}\n\n"
                       f"--- Question ---\n{query_text}\n\nOutput: <answer>X</answer>")

        # CyberRAG core: generate → validate → regenerate
        response = chat(prompt=base_prompt, system=MCQ_SYSTEM_PROMPT, temperature=0.0)
        answer = parse_mcq_prediction(response)

        for _ in range(self.model.max_retries):
            if re.search(r'<answer>\s*[A-D]\s*</answer>', response, re.IGNORECASE):
                return answer
            regen = (f"{base_prompt}\n\nPrevious response was invalid. "
                     "You MUST output <answer>X</answer> where X is A, B, C, or D.")
            response = chat(prompt=regen, system=MCQ_SYSTEM_PROMPT, temperature=0.0)
            answer = parse_mcq_prediction(response)

        return answer


@register("search_o1")
class SearchO1MCQ:
    def __init__(self, **kwargs):
        import importlib
        _mod = importlib.import_module("baselines.Search-o1-main.search_o1_taa")
        SearchO1 = _mod.SearchO1
        self.model = SearchO1(top_k=kwargs.get("top_k", 10),
                              max_searches=kwargs.get("max_searches", 3),
                              max_turns=kwargs.get("max_turns", 6))

    def predict(self, query_text: str) -> str:
        """Search-o1 iterative search with MCQ prompts."""
        from baselines.shared import (
            retrieve, format_context, chat, parse_mcq_prediction,
            MCQ_SYSTEM_PROMPT,
        )
        import importlib
        _so1 = importlib.import_module("baselines.Search-o1-main.search_o1_taa")
        extract_between = _so1.extract_between
        BEGIN_SEARCH_QUERY = _so1.BEGIN_SEARCH_QUERY
        END_SEARCH_QUERY = _so1.END_SEARCH_QUERY
        BEGIN_SEARCH_RESULT = _so1.BEGIN_SEARCH_RESULT
        END_SEARCH_RESULT = _so1.END_SEARCH_RESULT
        _SEARCH_TOOL_SUFFIX = _so1._SEARCH_TOOL_SUFFIX

        system_prompt = MCQ_SYSTEM_PROMPT + _SEARCH_TOOL_SUFFIX
        user_prompt = f"{query_text}\n\nOutput: <answer>X</answer>"

        full_prompt = user_prompt
        output_text = ""
        search_count = 0
        executed_queries = set()

        for turn in range(self.model.max_turns):
            response = chat(full_prompt, system=system_prompt, temperature=0.0)
            if response is None:
                break
            full_prompt += response
            output_text += response

            search_query = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

            if search_query and search_count < self.model.max_searches and search_query not in executed_queries:
                docs = retrieve(search_query, top_k=self.model.top_k)
                executed_queries.add(search_query)
                search_count += 1
                result_text = f"\n{BEGIN_SEARCH_RESULT}\n"
                result_text += (format_context(docs) + "\n") if docs else "No relevant results found.\n"
                result_text += f"{END_SEARCH_RESULT}\n"
                full_prompt += result_text
                output_text += result_text
            elif search_query and search_count >= self.model.max_searches:
                msg = (f"\n{BEGIN_SEARCH_RESULT}\nMaximum search limit reached. "
                       f"Please provide your final answer.\n{END_SEARCH_RESULT}\n")
                full_prompt += msg
                output_text += msg
            elif search_query and search_query in executed_queries:
                msg = (f"\n{BEGIN_SEARCH_RESULT}\nYou already searched this query. "
                       f"Refer to previous results.\n{END_SEARCH_RESULT}\n")
                full_prompt += msg
                output_text += msg
            else:
                break

        return parse_mcq_prediction(output_text)


@register("rag_intel")
class RAGIntelMCQ:
    def __init__(self, **kwargs):
        from baselines.RAGIntel.rag_intel_taa import RAGIntel
        from src.utils.settings import get_pipeline_param
        top_n = get_pipeline_param("rag_intel.top_n", 20)
        self.model = RAGIntel(top_k=kwargs.get("top_k", 10), top_n=top_n)

    def predict(self, query_text: str) -> str:
        """RAGIntel pipeline with MCQ prompts: hybrid retrieve → rerank → generate."""
        from baselines.shared import (
            retrieve, format_context, chat, parse_mcq_prediction,
            MCQ_SYSTEM_PROMPT,
        )

        docs = retrieve(query_text, top_k=self.model.top_n)
        if not docs:
            resp = chat(prompt=f"{query_text}\n\nOutput: <answer>X</answer>",
                        system=MCQ_SYSTEM_PROMPT, temperature=0.0)
            return parse_mcq_prediction(resp)

        # Flashrank reranking (RAGIntel's core contribution)
        reranked = self.model._rerank(query_text, docs)

        context = format_context(reranked)
        prompt = (f"--- Retrieved Documents ---\n{context}\n\n"
                  f"--- Question ---\n{query_text}\n\nOutput: <answer>X</answer>")
        resp = chat(prompt=prompt, system=MCQ_SYSTEM_PROMPT, temperature=0.0)
        return parse_mcq_prediction(resp)


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
        options_text = f"A) {row['Option A']}\nB) {row['Option B']}\nC) {row['Option C']}\nD) {row['Option D']}"
        question = row["Question"]
        full_query = f"Question: {question}\n\nOptions:\n{options_text}"

        result = run_single_baseline(method_name, runner_args, full_query, options_text, idx, args.timeout)
        result["ground_truth"] = row["GT"]
        results.append(result)

        if result["status"] == "ok":
            tag = "+" if result["prediction"] == row["GT"] else "-"
            _log(f"[{method_name}] [{idx+1}/{end}] GT={row['GT']} Pred={result['prediction']} [{tag}] ({result['elapsed_sec']}s)")
        else:
            _log(f"[{method_name}] [{idx+1}/{end}] FAILED: {result['status']}")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 评估：total 固定为实验范围大小（数据集总数），失败/超时/无答案一律按错误计，不剔除
    in_range = [r for r in results if start <= r["index"] < end]
    total = end - start
    n_failed = sum(1 for r in in_range if r["status"] != "ok")
    correct = sum(1 for r in in_range if r["status"] == "ok" and r["prediction"] == r["ground_truth"])
    acc = correct / total * 100 if total else 0.0
    _log(f"[{method_name}] Accuracy: {acc:.1f}% ({correct}/{total})  [failed/timed-out as wrong: {n_failed}]")

    report_path = output_path.replace(".json", "_report.json")
    with open(report_path, "w") as f:
        json.dump({"method": method_name, "accuracy": round(acc, 2), "correct": correct, "total": total,
                   "num_failed": n_failed}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=list(_METHODS.keys()) + ["all"])
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output", default="experiments/results/{method}_mcq_results.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2500)
    from src.utils.settings import get_global_top_k, get_global
    parser.add_argument("--top-k", type=int, default=get_global_top_k())
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=get_global("timeout", 600))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs("experiments/results", exist_ok=True)

    data_path = args.data_path or os.path.join(os.path.dirname(__file__), "..", "data", "CTIBench-MCQ")
    dataset = load_from_disk(data_path)

    methods = list(_METHODS.keys()) if args.method == "all" else [args.method]
    for m in methods:
        _log(f"\n{'='*60}\nRunning MCQ baseline: {m}\n{'='*60}")
        run_for_method(m, args, dataset)


if __name__ == "__main__":
    main()
