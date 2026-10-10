"""
LLM 调用封装 —— 通过 vLLM 的 OpenAI 兼容端点调用本地大模型。

重要：当前 vLLM 加载的是 Qwen3-30B-A3B-**Base**（预训练基座，非 Instruct）。
基座模型的 tokenizer 不带 chat_template，/v1/chat/completions 无法为其正确套用
对话格式，模型会把输入当成无结构纯文本续写 —— 回显 "Assistant:/User:" 角色标记、
答完后继续编造下一轮对话、甚至跑偏复述训练数据里的指令模板。
因此 chat() 改走 /v1/completions 并手工拼接 Qwen 系列通用的 ChatML
（<|im_start|>system/user/assistant<|im_end|>），显式构造对话结构后，基座模型才能
稳定作答（Qwen 预训练阶段即以 ChatML 做 packing，故对 2.5/3 的 Base 与 Instruct 均兼容）。
"""
import json
import os as _os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from src_new.utils.settings import get_settings

_settings = get_settings()
_chat_config = _settings['models']['chat_model']

_client = OpenAI(
    api_key=_chat_config.get('GRAPHRAG_API_KEY', 'EMPTY'),
    base_url=_os.environ.get('LLM_API_BASE') or _chat_config['api_base'],
)

import os as _os
MODEL_NAME = _os.environ.get('LLM_MODEL_NAME') or _chat_config['model_name']  # 并行链环境覆盖
DEFAULT_MAX_TOKENS = _chat_config.get('max_tokens', 2048)
DEFAULT_RETRIES = _chat_config.get('default_retries', 3)
# 固定采样 seed（vLLM 请求级）：temperature>0 的调用（假设评估 0.2 等）可复现；
# None = 不传（旧行为）。P0 统一实验（回复学长 §3）：全表以 seed=42 重跑为准。
_REQUEST_SEED = _chat_config.get('seed', None)
# 注：原 enable_thinking / chat_template_kwargs 仅对 /v1/chat/completions 有意义；
# 当前改用 /v1/completions + 手工 ChatML（见 _format_chatml），基座模型无思考机制，不再读取该开关。
# 重复惩罚（>1 抑制已出现 token 的重复概率，缓解 ATE 等任务的自回归退化：
# 如 T-ID 重复 T1055×65、顺序枚举 T1060→T1127）。对所有方法统一生效（控制变量）。
DEFAULT_REPETITION_PENALTY = _chat_config.get('repetition_penalty', 1.1)

# 模型实际上下文窗口长度（从模型配置读取，fallback 8192）
_MODEL_MAX_CONTEXT = _chat_config.get('max_context_tokens', 8192)

# 保守估计：网络安全文本混合技术术语，实际约 1.7 chars/token，取 1.5 留安全余量
_CHARS_PER_TOKEN = 1.5


def _safe_max_tokens(prompt: str, system: str, max_tokens: int) -> tuple[int, str, str]:
    """确保 input_tokens + max_tokens <= 模型上下文上限。
    返回 (capped_max_tokens, possibly_truncated_prompt, possibly_truncated_system)
    """
    # 预留输出空间
    min_output = 256
    max_input_tokens = _MODEL_MAX_CONTEXT - min(max_tokens, _MODEL_MAX_CONTEXT // 2)
    max_input_chars = int(max_input_tokens * _CHARS_PER_TOKEN) - 100

    # 如果 prompt 过长，从中间截断（保留开头指令 + 结尾最新信息）
    truncated_prompt = prompt
    if len(prompt) + len(system) > max_input_chars:
        budget = max_input_chars - len(system)
        if budget < 500:
            budget = 500
        if len(prompt) > budget:
            head = budget // 3
            tail = budget - head - 20
            truncated_prompt = prompt[:head] + "\n...[truncated]...\n" + prompt[-tail:]

    # 重新计算可用输出空间
    est_input = int((len(truncated_prompt) + len(system)) / _CHARS_PER_TOKEN) + 100
    available = _MODEL_MAX_CONTEXT - est_input
    capped = min(max_tokens, max(available, min_output))

    return capped, truncated_prompt, system


# ---- ChatML 对话格式（手工拼接） ----
# 基座模型 tokenizer 无 chat_template，必须显式构造 ChatML 结构，
# 否则 vLLM 把 messages 当纯文本拼接，模型会失控续写。
# 思考模式适配（按模型族）：
# - Qwen3 系列：默认开思考，预填一对空 <think></think> 关闭；
# - GLM-4 系列：chat 格式天然带 <think> 段且预填会导致空输出——不预填。
def _wants_think_prefill() -> bool:
    return "qwen3" in str(MODEL_NAME).lower()


def _format_chatml(system: str, user: str) -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    if _wants_think_prefill():
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    else:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# 生成终止符：assistant 回合结束（<|im_end|>）；或基座模型答完直接续写下一轮
# （<|im_start|>）时截断，避免它继续编造后续对话。
_CHAT_STOP = ["<|im_end|>", "<|im_start|>"]

# ---- LLM I/O 日志 ----
_log_cfg = _settings.get('llm_log', {})
_LLM_LOG_ENABLED = _log_cfg.get('enabled', False)
_LLM_LOG_PATH = Path(_log_cfg.get('path', 'logs/llm_io.log'))

# ---- 调用统计（效率轴：LLM 调用数 / 累计延迟 / 字符量；reset_llm_stats 快照） ----
_LLM_CALLS = 0
_LLM_LATENCY_MS = 0.0
_LLM_PROMPT_CHARS = 0
_LLM_COMPLETION_CHARS = 0


def reset_llm_stats():
    """清零调用统计（每样本开始前调用）"""
    global _LLM_CALLS, _LLM_LATENCY_MS, _LLM_PROMPT_CHARS, _LLM_COMPLETION_CHARS
    _LLM_CALLS = 0
    _LLM_LATENCY_MS = 0.0
    _LLM_PROMPT_CHARS = 0
    _LLM_COMPLETION_CHARS = 0


def snapshot_llm_stats() -> dict:
    """返回当前调用统计快照（写进每条结果 JSON）"""
    return {
        "llm_calls": _LLM_CALLS,
        "llm_latency_sec": round(_LLM_LATENCY_MS / 1000.0, 1),
        "llm_prompt_chars": _LLM_PROMPT_CHARS,
        "llm_completion_chars": _LLM_COMPLETION_CHARS,
    }


def _log_llm_io(system: str, prompt: str, temperature: float, response: str, latency_ms: float):
    """将单次LLM调用的输入输出追加写入日志文件"""
    _LLM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LLM_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]  latency={latency_ms:.0f}ms  temp={temperature}\n")
        f.write(f"{'─'*80}\n")
        f.write(f"[SYSTEM]\n{system}\n") if system else None
        f.write(f"[USER]\n{prompt}\n")
        f.write(f"{'─'*80}\n")
        f.write(f"[ASSISTANT]\n{response}\n")


def chat(prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int = None,
         repetition_penalty: float = None) -> str:
    """调用LLM，返回文本响应"""
    global _LLM_CALLS, _LLM_LATENCY_MS, _LLM_PROMPT_CHARS, _LLM_COMPLETION_CHARS
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    if repetition_penalty is None:
        repetition_penalty = DEFAULT_REPETITION_PENALTY
    max_tokens, prompt, system = _safe_max_tokens(prompt, system, max_tokens)
    full_prompt = _format_chatml(system, prompt)

    t0 = time.perf_counter()
    extra = {"repetition_penalty": repetition_penalty}
    if _REQUEST_SEED is not None:
        extra["seed"] = int(_REQUEST_SEED)
    resp = _client.completions.create(
        model=MODEL_NAME,
        prompt=full_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=_CHAT_STOP,
        extra_body=extra,
    )
    content = resp.choices[0].text.strip()
    latency_ms = (time.perf_counter() - t0) * 1000

    # 调用统计（效率轴：LLM 调用数 / 延迟 / 字符量）
    _LLM_CALLS += 1
    _LLM_LATENCY_MS += latency_ms
    _LLM_PROMPT_CHARS += len(prompt) + len(system)
    _LLM_COMPLETION_CHARS += len(content)

    if _LLM_LOG_ENABLED:
        _log_llm_io(system, prompt, temperature, content, latency_ms)

    return content


def chat_json(prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int = None) -> dict:
    """调用LLM并解析JSON响应，失败返回空dict"""
    raw = chat(prompt, system, temperature, max_tokens)

    # 尝试从```json ... ```代码块中提取
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if match:
        json_str = match.group(1)
    else:
        json_str = raw

    # 去掉非JSON前后缀，找到第一个{到最后一个}
    start = json_str.find('{')
    end = json_str.rfind('}')
    if start != -1 and end != -1:
        json_str = json_str[start:end + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


def chat_json_with_retry(prompt: str, system: str = "", temperature: float = 0.0,
                         max_tokens: int = None, retries: int = None) -> dict:
    """带重试的JSON LLM调用，确保返回有效JSON"""
    if retries is None:
        retries = DEFAULT_RETRIES
    for i in range(retries):
        result = chat_json(prompt, system, temperature, max_tokens)
        if result:
            return result
    return {}


# ---- 并行调用 ----

# 限制最大线程数，避免本地LLM过载
_MAX_WORKERS = 8


def parallel_chat(tasks: list[dict]) -> list[str]:
    """并行调用 chat()。

    Args:
        tasks: 每个元素是传给 chat() 的参数字典，如:
               [{"prompt": ..., "system": ..., "temperature": ...}, ...]
               缺省字段使用 chat() 的默认值。

    Returns:
        与 tasks 等长的结果列表（保持顺序）。
    """
    if not tasks:
        return []
    if len(tasks) == 1:
        t = tasks[0]
        return [chat(**t)]

    results = [None] * len(tasks)

    def _call(idx, kwargs):
        return idx, chat(**kwargs)

    with ThreadPoolExecutor(max_workers=min(len(tasks), _MAX_WORKERS)) as pool:
        futures = {pool.submit(_call, i, t): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            idx, text = future.result()
            results[idx] = text

    return results


def parallel_chat_json_with_retry(tasks: list[dict]) -> list[dict]:
    """并行调用 chat_json_with_retry()。

    Args:
        tasks: 每个元素是传给 chat_json_with_retry() 的参数字典。

    Returns:
        与 tasks 等长的结果列表（保持顺序）。
    """
    if not tasks:
        return []
    if len(tasks) == 1:
        t = tasks[0]
        return [chat_json_with_retry(**t)]

    results = [None] * len(tasks)

    def _call(idx, kwargs):
        return idx, chat_json_with_retry(**kwargs)

    with ThreadPoolExecutor(max_workers=min(len(tasks), _MAX_WORKERS)) as pool:
        futures = {pool.submit(_call, i, t): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            idx, data = future.result()
            results[idx] = data

    return results
