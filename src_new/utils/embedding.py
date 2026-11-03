"""
共享嵌入模型初始化 — 供 weaviate_retriever 和 abp 共同使用
"""
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from src_new.utils.settings import get_settings

_settings = get_settings()
_embed_config = _settings['models']['embedding_model']

_embed_model_name = _embed_config['path']
_target_dimension = _embed_config['dimension']
_embed_device = _embed_config.get('device', 'cuda')

_embed_model = None


def get_embed_model():
    """获取或初始化 BGE-M3 嵌入模型（单例）"""
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbeddings(
            model_name=_embed_model_name,
            model_kwargs={"device": _embed_device}
        )
    return _embed_model


def get_target_dimension() -> int:
    """获取目标嵌入维度"""
    return _target_dimension
