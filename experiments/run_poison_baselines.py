"""会议版 6 基线在投毒场景下的表现（学长 4.2：SCHISM 不开防御是更脆弱还是更稳健）

复用 run_taa_baselines.py 的方法注册表（构造参数与调用接口保持一致），
仅替换检索层为投毒检索（src 模块版 install）。
输出 Correct/Plausible/framed + 毒命中统计。
用法：python experiments/run_poison_baselines.py --method naive_rag --setting r2
"""
import os
import sys
import json
import time
import argparse
import traceback
from multiprocessing import Process, Queue

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_from_disk

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_taa import GROUND_TRUTH  # noqa: E402
from run_taa_baselines import _METHODS  # noqa: E402  注册表：构造/predict 接口同主实验
from run_poison import POISON_DIR, SETTINGS, _LOG  # noqa: E402


def install_src_poisoned_retrieval(variant_cap: int = 99):
    """src 模块版投毒检索（基线经 baselines/shared.py → src.retrieval）。"""
    import src.retrieval.weaviate_retriever as retr
    orig_semantic = retr._semantic_search
    _bm25_add_poison_src(retr, variant_cap)

    def merged(query_text, top_k):
        base_docs = orig_semantic(query_text, top_k)
        poison_docs = _semantic_search_poison_src(query_text, top_k, variant_cap)
        if not poison_docs:
            return base_docs
        out = list(base_docs) + list(poison_docs)
        out.sort(key=lambda d: float(
            (d.get("_additional") or {}).get("distance", 1.0)))
        return out

    retr._semantic_search = merged
    return lambda: setattr(retr, "_semantic_search", orig_semantic)


def _semantic_search_poison_src(query_text, top_k, variant_cap=99):
    import io as _io
    import sys as _sys
    import weaviate as _wv
    from src.retrieval import weaviate_retriever as retr
    from src.utils.settings import get_settings
    fields = ["content", "report_title", "section_title", "source_file"]
    qvec = retr.get_embed_model().embed_documents([query_text])[0][:retr.get_target_dimension()]
    client = _wv.Client(url=get_settings()["models"]["weaviate_model"]["url"])
    old = _sys.stdout
    _sys.stdout = _io.StringIO()
    try:
        resp = (client.query.get("APT_POISON", fields)
                .with_near_vector({"vector": qvec})
                .with_limit(top_k * 4).with_additional(["distance"]).do())
    finally:
        _sys.stdout = old
    if "errors" in resp or "data" not in resp:
        return []
    docs = resp["data"]["Get"].get("APT_POISON", [])
    kept = []
    for d in docs:
        try:
            var = int(d["source_file"].rsplit("_", 1)[1])
        except (ValueError, IndexError):
            var = 0
        if var < variant_cap:
            kept.append(d)
    return kept[:top_k]


def _bm25_add_poison_src(retr, variant_cap=99):
    """src 版 BM25 扩展（与 run_poison._bm25_add_poison 同逻辑）。"""
    import io as _io
    import sys as _sys
    import weaviate as _wv
    from src.utils.settings import get_settings
    idx = retr._bm25_index
    idx.build()
    keep = [i for i, d in enumerate(idx.docs)
            if not str(d.get("source_file", "")).startswith("poison_")]
    if len(keep) != len(idx.docs):
        idx.docs = [idx.docs[i] for i in keep]
        idx.tokenized = [idx.tokenized[i] for i in keep]
        idx.term_freqs = [idx.term_freqs[i] for i in keep]
        idx.doc_lengths = [idx.doc_lengths[i] for i in keep]
        idx.df = {}
        for toks in idx.tokenized:
            for t in set(toks):
                idx.df[t] = idx.df.get(t, 0) + 1
        idx.N = len(idx.docs)
        idx.avgdl = sum(idx.doc_lengths) / max(idx.N, 1)
    if any(str(d.get("source_file", "")).startswith("poison_") for d in idx.docs):
        return
    client = _wv.Client(url=get_settings()["models"]["weaviate_model"]["url"])
    old = _sys.stdout
    _sys.stdout = _io.StringIO()
    try:
        resp = (client.query.get("APT_POISON",
                                 ["content", "report_title", "section_title", "source_file"])
                .with_limit(10000).do())
    finally:
        _sys.stdout = old
    if "errors" in resp or "data" not in resp:
        _LOG("src BM25 poison fetch failed — skip")
        return
    docs = resp["data"]["Get"].get("APT_POISON", [])
    added = 0
    for doc in docs:
        try:
            var = int(doc["source_file"].rsplit("_", 1)[1])
        except (ValueError, IndexError):
            var = 0
        if var >= variant_cap:
            continue
        tokens = idx._tokenize(doc.get("content", "") + " " + doc.get("report_title", ""))
        idx.tokenized.append(tokens)
        tf_map = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1
        idx.term_freqs.append(tf_map)
        idx.doc_lengths.append(len(tokens))
        for t in set(tokens):
            idx.df[t] = idx.df.get(t, 0) + 1
        idx.docs.append(doc)
        added += 1
    idx.N = len(idx.docs)
    idx.avgdl = sum(idx.doc_lengths) / max(idx.N, 1)
    _LOG(f"src BM25 extended with {added} poison docs (cap={variant_cap})")


_WORKER_ARGS = {}


def _worker(method, runner_args, report_text, sample_idx, queue: Queue):
    try:
        if _WORKER_ARGS.get("poison_retrieval"):
            install_src_poisoned_retrieval(variant_cap=_WORKER_ARGS.get("variant_cap", 99))
        runner = _METHODS[method](**runner_args)
        t0 = time.time()
        pred = runner.predict(report_text)
        queue.put({"index": sample_idx, "prediction": str(pred),
                   "elapsed_sec": round(time.time() - t0, 1), "status": "ok"})
    except Exception as e:
        queue.put({"index": sample_idx, "prediction": "", "elapsed_sec": 0,
                   "status": f"error: {e}", "traceback": traceback.format_exc()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(_METHODS.keys()))
    ap.add_argument("--setting", default="r2", choices=list(SETTINGS.keys()))
    ap.add_argument("--suffix", default="poison")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--max-searches", type=int, default=5)
    ap.add_argument("--max-turns", type=int, default=10)
    args = ap.parse_args()

    dataset = load_from_disk(os.path.join(_HERE, "..", "data", "CTIBench-TAA"))
    tier, n_per_query = SETTINGS[args.setting]
    runner_args = {"top_k": args.top_k, "iterations": args.iterations,
                   "max_searches": args.max_searches, "max_turns": args.max_turns}

    out_dir = os.path.join(_HERE, "results", _model_slug())
    out = os.path.join(out_dir, f"baseline_{args.method}_{args.setting}_{args.suffix}.json")
    os.makedirs(out_dir, exist_ok=True)

    _WORKER_ARGS.update({"poison_retrieval": args.setting != "clean",
                         "variant_cap": n_per_query if args.setting.startswith("r") else 99})
    results = []
    if args.resume and os.path.exists(out):
        with open(out) as f:
            results = json.load(f)
    done = {r["index"] for r in results}

    from evaluate_taa import compute_taa_accuracy
    for idx in range(args.start, min(args.end, len(dataset))):
        if idx in done:
            continue
        q = Queue()
        p = Process(target=_worker, args=(args.method, runner_args,
                                          dataset[idx]["Text"], idx, q))
        p.start(); p.join(timeout=args.timeout)
        if p.is_alive():
            p.terminate(); p.join(5)
            if p.is_alive():
                p.kill(); p.join()
            r = {"index": idx, "prediction": "",
                 "status": f"timeout after {args.timeout}s"}
        else:
            r = q.get() if not q.empty() else {"index": idx, "prediction": "",
                                               "status": "error: subprocess died"}
        r["ground_truth"] = GROUND_TRUTH[idx]
        results.append(r)
        _LOG(f"[{os.path.basename(out)}] {idx+1} pred={r.get('prediction','')[:24]!r} "
             f"t={r.get('elapsed_sec','?')}s")
        with open(out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    gts = [r["ground_truth"] for r in results]
    preds = [r.get("prediction", "") for r in results]
    correct, plausible, _ = compute_taa_accuracy(gts, preds)
    corpus_map = {}
    if tier:
        path = os.path.join(POISON_DIR, f"false_flags_{tier}.json")
        if os.path.exists(path):
            corpus_map = json.load(open(path))
    framed = sum(1 for r, p in zip(results, preds)
                 if str(p).strip().lower() in [s["scapegoat"].lower()
                                               for s in corpus_map.get(str(r["index"]), [])])
    rep = {"method": args.method, "setting": args.setting, "n": len(results),
           "correct_acc": round(correct, 1), "plausible_acc": round(plausible, 1),
           "framed": framed}
    with open(out.replace(".json", "_report.json"), "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    _LOG(f"[{os.path.basename(out)}] C={correct:.1f} P={plausible:.1f} framed={framed}")


def _model_slug():
    from src.utils.llm_client import MODEL_NAME
    return str(MODEL_NAME).lower().replace("/", "-").replace(" ", "-")


if __name__ == "__main__":
    main()
