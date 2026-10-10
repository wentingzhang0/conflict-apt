"""
统一配置加载器 — 全局单例，供所有模块共享
加载 settings.yaml + prompts/*.yaml
"""
import yaml
from pathlib import Path

# src 独立配置：settings.yaml 与 prompts 都位于 src/ 下，与旧 src/ 隔离
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_NEW_SRC_ROOT = Path(__file__).resolve().parent.parent
_SETTINGS_PATH = _NEW_SRC_ROOT / "settings.yaml"
_PROMPTS_DIR = _NEW_SRC_ROOT / "agents" / "prompts"

_settings = None
_prompts = {}
_overrides = {}  # 消融/实验覆盖项（点号路径 → 值），优先级最高


def apply_overrides(overrides: dict):
    """注册覆盖项（如 {"adaptive.rebuttal_max_rounds": 0}），随后 get_pipeline_param 优先返回。"""
    global _overrides
    _overrides = dict(overrides or {})


def get_overrides() -> dict:
    return dict(_overrides)


def get_settings() -> dict:
    """获取 settings.yaml 配置（全局单例）"""
    global _settings
    if _settings is None:
        with open(_SETTINGS_PATH, 'r', encoding='utf-8') as f:
            _settings = yaml.safe_load(f)
    return _settings


def get_prompt(agent_name: str) -> dict:
    """
    获取指定 agent 的 prompt 配置。
    返回 {"system": "...", "user": "..."} 格式。
    agent_name 对应 prompts/{agent_name}.yaml
    """
    if agent_name not in _prompts:
        path = _PROMPTS_DIR / f"{agent_name}.yaml"
        with open(path, 'r', encoding='utf-8') as f:
            _prompts[agent_name] = yaml.safe_load(f)
    return _prompts[agent_name]


def get_task_prompt(agent_name: str, task_type: str = "") -> dict:
    """
    获取任务特化的 prompt：优先加载 prompts/{task_type}/{agent}.yaml，
    不存在则 fallback 到 prompts/{agent}.yaml。
    task_type 为空或 "taa" 时直接用默认 prompt。
    """
    if task_type and task_type not in ("taa", ""):
        specific_key = f"{task_type}/{agent_name}"
        if specific_key not in _prompts:
            path = _PROMPTS_DIR / task_type / f"{agent_name}.yaml"
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    _prompts[specific_key] = yaml.safe_load(f)
            else:
                _prompts[specific_key] = get_prompt(agent_name)
        return _prompts[specific_key]
    return get_prompt(agent_name)


def get_pipeline_param(path: str, default=None):
    """
    从 pipeline 配置中取值，用点号分隔路径。
    例: get_pipeline_param("evidence_collector.top_k") → 5
    覆盖项（apply_overrides 注册，消融用）优先于 settings.yaml。
    """
    if path in _overrides:
        return _overrides[path]
    cfg = get_settings().get("pipeline", {})
    keys = path.split(".")
    for k in keys:
        if isinstance(cfg, dict) and k in cfg:
            cfg = cfg[k]
        else:
            return default
    return cfg


def get_global_top_k() -> int:
    """获取全局统一的 top_k（所有方法和基线共用）"""
    return get_settings().get("global", {}).get("top_k", 10)


def get_global(key: str, default=None):
    """获取 global 配置项（所有方法和基线共用）"""
    return get_settings().get("global", {}).get(key, default)
