"""
Weaviate 检索 + 本地 BM25 混合检索
- semantic: Weaviate 纯语义向量
- keyword:  本地 BM25
- hybrid:   语义 + BM25 通过 RRF 融合
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*weaviate.*")

import sys
import io
import re
import math
import logging
logging.getLogger("weaviate").setLevel(logging.ERROR)

import weaviate
from src_new.utils.embedding import get_embed_model, get_target_dimension
from src_new.utils.settings import get_settings, get_global

_settings = get_settings()
_weaviate_url = _settings['models']['weaviate_model']['url']
_class_name = _settings['models']['weaviate_model']['class_name']

_client = weaviate.Client(url=_weaviate_url)

_RETRIEVAL_MODE = get_global("retrieval_mode", "hybrid")
_HYBRID_ALPHA = get_global("hybrid_alpha", 0.4)


# ── 本地 BM25 索引（懒加载，从 Weaviate 拉全部文档）────────────────────
class BM25Index:
    """简易 BM25，从 Weaviate 加载全部文档，首次查询时构建"""

    def __init__(self):
        self.docs = []        # [{"content", "report_title", "section_title", "source_file", "_id"}]
        self.tokenized = []   # [ [token, ...], ... ]
        self.term_freqs = []   # [ {token: count, ...}, ... ]
        self.doc_lengths = []  # [ doc_token_count, ... ]
        self.df = {}          # token → 出现文档数
        self.avgdl = 0
        self.N = 0
        self._built = False

    def build(self):
        """从 Weaviate 拉全部文档并建索引"""
        if self._built:
            return

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            # 分批拉取（Weaviate 默认限制 10000，我们文档不多一次拉完）
            response = (
                _client.query
                .get(_class_name, ["content", "report_title", "section_title", "source_file"])
                .with_limit(10000)
                .do()
            )
        finally:
            sys.stdout = old_stdout

        if 'errors' in response:
            self._built = True
            return

        self.docs = response['data']['Get'].get(_class_name, [])
        self.N = len(self.docs)

        # 分词 + 统计 df
        for doc in self.docs:
            tokens = self._tokenize(doc.get("content", "") + " " + doc.get("report_title", ""))
            self.tokenized.append(tokens)
            tf_map = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            self.term_freqs.append(tf_map)
            self.doc_lengths.append(len(tokens))
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

        self.avgdl = sum(self.doc_lengths) / max(self.N, 1)
        self._built = True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单英文分词：小写 + 去标点 + 去停用词"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # 去掉太短的词
        return [t for t in tokens if len(t) > 2]

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """返回 [(doc_index, bm25_score), ...] 按分数降序"""
        self.build()
        if not self.docs:
            return []

        query_tokens = self._tokenize(query)
        k1 = 1.5
        b = 0.75

        scores = []
        for i, tf_map in enumerate(self.term_freqs):
            dl = self.doc_lengths[i]
            score = 0.0
            for qt in query_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                df_val = self.df.get(qt, 0)
                idf = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1)
                tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl))
                score += idf * tf_component

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_doc(self, idx: int) -> dict:
        return self.docs[idx]


# 全局 BM25 索引单例
_bm25_index = BM25Index()


# ── RRF 融合 ──────────────────────────────────────────────────────────
def _rrf_merge(semantic_results: list[dict], bm25_results: list[tuple[int, float]],
               alpha: float, top_k: int) -> list[dict]:
    """
    Reciprocal Rank Fusion: 合并语义和 BM25 排序结果
    alpha 控制语义权重: alpha * rrf_semantic + (1-alpha) * rrf_bm25
    """
    k = 60  # RRF 常数

    # 语义结果的 rank 分
    sem_scores = {}
    for rank, doc in enumerate(semantic_results):
        key = doc.get("source_file", "") + "|" + doc.get("section_title", "")
        sem_scores[key] = 1.0 / (k + rank + 1)

    # BM25 结果的 rank 分 + 文档内容
    bm25_scores = {}
    bm25_docs = {}
    for rank, (idx, _) in enumerate(bm25_results):
        doc = _bm25_index.get_doc(idx)
        key = doc.get("source_file", "") + "|" + doc.get("section_title", "")
        bm25_scores[key] = 1.0 / (k + rank + 1)
        bm25_docs[key] = doc

    # 合并所有候选
    all_keys = set(sem_scores.keys()) | set(bm25_scores.keys())
    final = []
    for key in all_keys:
        score = alpha * sem_scores.get(key, 0) + (1 - alpha) * bm25_scores.get(key, 0)
        # 优先用语义结果（有 distance），否则用 BM25 文档
        doc = None
        for d in semantic_results:
            d_key = d.get("source_file", "") + "|" + d.get("section_title", "")
            if d_key == key:
                doc = d
                break
        if doc is None:
            doc = bm25_docs.get(key, {})
        doc_copy = dict(doc)
        doc_copy["hybrid_score"] = score
        final.append(doc_copy)

    final.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
    return final[:top_k]


# ── 主检索函数 ────────────────────────────────────────────────────────
def retrieve(query_text: str, top_k: int = None, mode: str = None, alpha: float = None) -> list[dict]:
    if top_k is None:
        top_k = get_global("top_k", 10)
    if mode is None:
        mode = _RETRIEVAL_MODE
    if alpha is None:
        alpha = _HYBRID_ALPHA

    if mode == "semantic":
        return _semantic_search(query_text, top_k)
    elif mode == "keyword":
        return _bm25_search(query_text, top_k)
    else:
        return _hybrid_search(query_text, top_k, alpha)


def _semantic_search(query_text: str, top_k: int) -> list[dict]:
    """纯语义向量检索"""
    fields = ["content", "report_title", "section_title", "source_file"]
    embed_model = get_embed_model()
    query_vector = embed_model.embed_documents([query_text])[0][:get_target_dimension()]

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        response = (
            _client.query
            .get(_class_name, fields)
            .with_near_vector({"vector": query_vector})
            .with_limit(top_k)
            .with_additional(["distance"])
            .do()
        )
    finally:
        sys.stdout = old_stdout

    if 'errors' in response:
        return []
    return response['data']['Get'].get(_class_name, [])


def _bm25_search(query_text: str, top_k: int) -> list[dict]:
    """纯 BM25 检索"""
    results = _bm25_index.search(query_text, top_k=top_k)
    return [_bm25_index.get_doc(idx) for idx, _ in results]


def _hybrid_search(query_text: str, top_k: int, alpha: float) -> list[dict]:
    """
    混合检索: 语义 + BM25 通过 RRF 融合
    alpha=1 纯语义, alpha=0 纯 BM25, alpha=0.5 均衡
    """
    # 多拉一些候选用于融合
    n_candidates = min(top_k * 3, 20)
    semantic_results = _semantic_search(query_text, n_candidates)
    bm25_results = _bm25_index.search(query_text, top_k=n_candidates)
    return _rrf_merge(semantic_results, bm25_results, alpha, top_k)
