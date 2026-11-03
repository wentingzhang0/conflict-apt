"""运行留痕（重跑确认清单第 7 点）：每次运行把配置快照、commit、seed、库状态、
served-model 信息写进 <output>_run_meta.json，与逐题 JSON 一起存档，不再删除。

纯记录性模块——不参与、不改变任何方法逻辑。各任务 runner 在写出报告处调用
write_run_meta(output_path, setting=..., ...) 一次即可。
"""
import json
import os
import subprocess
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))


def _git_commit() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_HERE,
                           capture_output=True, text=True, timeout=10
                           ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=_HERE,
                               capture_output=True, text=True, timeout=10
                               ).stdout.strip()
        return f"{h}" + (" (dirty)" if dirty else "")
    except Exception:
        return "unknown"


def _settings_snapshot() -> dict:
    p = os.path.join(_HERE, "..", "src_new", "settings.yaml")
    try:
        return {"path": "src_new/settings.yaml",
                "content": open(p, encoding="utf-8").read()}
    except Exception as e:
        return {"error": str(e)}


def _served_model() -> dict:
    """从 vLLM /v1/models 取实际服务的模型 id 与 max_model_len（启动参数的
    服务端见证；完整启动脚本本身在仓库里）。失败返回错误，不阻塞运行。"""
    try:
        import urllib.request as _u
        from src_new.utils.settings import get_settings
        base = get_settings()["models"]["chat_model"]["api_base"].rstrip("/")
        with _u.urlopen(f"{base}/models", timeout=5) as r:
            data = json.loads(r.read().decode())
        out = []
        for m in data.get("data", []):
            out.append({"id": m.get("id"),
                        "max_model_len": m.get("max_model_len")})
        return {"api_base": base, "models": out}
    except Exception as e:
        return {"error": str(e)}


def _collection_counts() -> dict:
    try:
        from src_new.retrieval.weaviate_retriever import collection_counts
        return collection_counts()
    except Exception as e:
        return {"error": str(e)}


def write_run_meta(output_path: str, **extra) -> str:
    """把本次运行的完整环境快照写到 <output_path 去掉 .json>_run_meta.json。
    extra 由调用方补充任务语义（setting/defense/ablation/baseline/suffix 等）。"""
    from src_new.utils.llm_client import MODEL_NAME
    meta = {
        "commit": _git_commit(),
        "backbone_model": MODEL_NAME,
        "seed": _seed(),
        "served_model": _served_model(),
        "collections": _collection_counts(),
        "settings_snapshot": _settings_snapshot(),
        "extra": extra,
    }
    meta_path = output_path.replace(".json", "") + "_run_meta.json"
    d = os.path.dirname(meta_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta_path


def _seed():
    from src_new.utils.settings import get_settings
    return get_settings()["models"]["chat_model"].get("seed")
