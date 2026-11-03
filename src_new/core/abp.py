"""
Attack Behavior Profile (ABP) 抽取 —— 单一四维结构（描述 + 规范键）（创新点 2，3.2.1）

每个维度同时携带自然语言 description（喂假设锚点/报告）与 canonical_key（供确定性两两匹配）。
规范键由 LLM 用自身参数知识做别名归一/释义归一/区间解析，不引外部 KG。
四维：attribution / timeline / behavior / campaign（Diamond Model / STIX 2.1 锚定）。
"""
from src_new.core.awm import BehaviorProfile, CONFLICT_DIMENSIONS
from src_new.utils.llm_client import chat_json_with_retry
from src_new.utils.settings import get_prompt, get_pipeline_param


def _cfg(key, default=None):
    return get_pipeline_param(f"abp_extraction.{key}", default)


def _dim_dict(raw) -> dict:
    """归一化为 {description, canonical_key}；容忍 LLM 缺字段/类型不规范。"""
    if not isinstance(raw, dict):
        return {}
    d = {}
    if "description" in raw:
        d["description"] = raw.get("description") or ""
    if "canonical_key" in raw:
        d["canonical_key"] = raw.get("canonical_key") or {}
    return d


def extract_abp(doc_id: str, report_text: str) -> BehaviorProfile:
    """从报告文本抽取 ABP（单次 LLM，O(k)，可缓存）。"""
    prompts = get_prompt("abp")
    text_max = _cfg("report_text_max_length", 4000)
    prompt = prompts["user"].format(text=report_text[:text_max])

    result = chat_json_with_retry(
        prompt=prompt,
        system=prompts["system"],
        temperature=_cfg("temperature", 0.0),
        retries=_cfg("retries", 3),
    )
    profile = BehaviorProfile(doc_id=doc_id)
    if result:
        for d in CONFLICT_DIMENSIONS:  # attribution / timeline / behavior / campaign
            setattr(profile, d, _dim_dict(result.get(d, {})))
    return profile


def compute_abp_embedding(profile: BehaviorProfile) -> list:
    """四维描述摘要 → BGE-M3 向量（供边权调制与查询锚点）。"""
    from src_new.utils.embedding import get_embed_model, get_target_dimension
    summary = profile.to_summary()
    if not summary:
        return []
    model = get_embed_model()
    vec = model.embed_documents([summary])[0]
    return vec[:get_target_dimension()]
