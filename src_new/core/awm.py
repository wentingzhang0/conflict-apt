"""
Analysis Working Memory (AWM) —— 多智能体共享的分析工作记忆（新版方案扩充）

相对旧 src/core/awm.py 的关键变化（对应 improve_reference/方案设计.md）：
- GraphEdge.edge_class ∈ {confirmed, none}（两态；none 配对不实例化为边，3.2.2/3.2.3）
- BehaviorProfile 改为四维结构 attribution/timeline/behavior/campaign，每维 {description, canonical_key}（3.2.1/5.1）
- AWM 新增 initial_doc_ids / clusters / Phi / k / depth_I /
  counterexample_queries 等控制流字段（3.1、3.3、5.1）；Φ 为唯一算力控制信号，无 C
- 提供 AWM↔JSON 序列化，供 metrics.py CLI 复用同一份确定性逻辑
"""
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


CONFLICT_DIMENSIONS = ["attribution", "timeline", "behavior", "campaign"]


@dataclass
class Evidence:
    doc_id: str           # 来源标识（source_file）
    source: str           # 报告标题
    content: str          # 段落正文
    section_title: str = ""
    reliability: float = 1.0    # 来源可靠性（0.0-1.0，启发式评估）
    confidence: float = 1.0     # 证据置信度
    publish_date: str = ""      # 发布时间
    attribution_summary: str = ""
    # 正文内容嵌入（创新点 4 CORE-1 近重复检测用；骨干无关的嵌入源，建图时算好）
    content_embedding: list = field(default_factory=list)


@dataclass
class BehaviorProfile:
    """攻击行为画像 ABP（方案 3.2.1 / 5.1）：单一四维结构，每维 = {description, canonical_key}。
    - description：自然语言叙述（喂假设锚点 / 分析报告的可解释增强）；
    - canonical_key：规范键（供确定性两两匹配 canonical_match）。
    四维（Diamond Model / STIX 2.1 锚定，3.2.1 表）：
      attribution : {canonical_name, aliases}        归因（Diamond adversary / STIX Intrusion Set）
      timeline    : {start, end}                      时间线（Diamond timestamp / STIX first/last_seen）
      behavior    : {tokens:[TTP/工具/目标行业]}       行为（Diamond capability/victim/infrastructure）
      campaign    : {tokens:[战役/事件令牌]}           战役事件（STIX Campaign SDO）
    其中 attribution/timeline 可 support+conflict；behavior/campaign 只 support/none（3.2.1 不对称性）。
    """
    doc_id: str
    attribution: dict = field(default_factory=dict)   # {"description": str, "canonical_key": {...}}
    timeline: dict = field(default_factory=dict)
    behavior: dict = field(default_factory=dict)
    campaign: dict = field(default_factory=dict)
    embedding: list = field(default_factory=list)      # BGE-M3 嵌入（四维描述摘要→向量）

    @property
    def canonical_keys(self) -> dict:
        """扁平规范键视图 {dim: canonical_key}，供 canonical_match.match_all（只读，由四维派生）。"""
        return {d: ((getattr(self, d) or {}).get("canonical_key") or {})
                for d in CONFLICT_DIMENSIONS}

    @classmethod
    def from_canonical(cls, doc_id: str, canonical_keys: dict) -> "BehaviorProfile":
        """从扁平规范键 {dim: canonical_key} 构造（描述留空），便于测试与查询锚点快速构造。"""
        p = cls(doc_id=doc_id)
        for d in CONFLICT_DIMENSIONS:
            setattr(p, d, {"description": "",
                           "canonical_key": (canonical_keys or {}).get(d, {}) or {}})
        return p

    def to_summary(self) -> str:
        """四维描述拼接为自然语言摘要，用于 embedding 与 prompt 注入"""
        labels = {"attribution": "Attribution", "timeline": "Timeline",
                  "behavior": "Behavior", "campaign": "Campaign"}
        parts = []
        for d in CONFLICT_DIMENSIONS:
            desc = ((getattr(self, d) or {}).get("description") or "").strip()
            if desc:
                parts.append(f"{labels[d]}: {desc}")
        return ". ".join(parts) if parts else ""


@dataclass
class Hypothesis:
    id: str
    answer: str
    evidence_ids: list = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    cluster_id: int = -1
    llm_score: float = 0.0          # LLM 多维评估得分（中位）
    strength: float = 0.0           # 混合评分 α×h_graph + (1−α)×llm_score
    graph_metrics: dict = field(default_factory=dict)  # {rho, kappa, n_evidence, h_graph}
    unstable: bool = False


@dataclass
class GraphEdge:
    source: str           # 源节点 doc_id
    target: str           # 目标节点 doc_id
    sign: int             # +1(支持) / -1(矛盾) / 0(neutral，不建图边)
    confidence: float
    # 两态（3.2.2/3.2.3）：
    #   confirmed：经规范键明确判定（support 或 conflict），建边 sign=±1
    #   none     ：四维全 none，不建边，留待传递性推理补全（不实例化为 GraphEdge）
    edge_class: str = "confirmed"
    conflict_dimension: str = ""        # 最高优先级冲突维（conflict 时记录）
    profile_modulation: float = 1.0     # 画像调制因子 β = c + (1−c)·cos(p_i,p_j)
    reliability_factor: float = 1.0
    # 逐维原始判定 {dim: support|conflict|none}（确定性两档匹配产出）
    dim_verdicts: dict = field(default_factory=dict)
    # 是否传递性推理补全产出（用于统计/可解释）
    inferred: bool = False
    transitivity_consistency: float = 0.0
    # 独立性折扣 ι（创新点 4 REWEIGH 阶段写入；缺省 1.0 = 未折扣，
    # 未启用欺骗感知时与会议版等价）
    independence_factor: float = 1.0


@dataclass
class AWM:
    """分析工作记忆 —— 所有 Agent 共享的状态，Controller 持有唯一读写权"""
    query: str = ""
    query_summary: str = ""
    query_profile: Optional[BehaviorProfile] = None   # 查询报告 ABP（查询锚点）
    task_type: str = ""
    task_instruction: str = ""
    answer_format: str = ""
    extra_context: str = ""

    evidence_set: list[Evidence] = field(default_factory=list)
    initial_doc_ids: list[str] = field(default_factory=list)   # 初始检索结果，供扩展池排除
    behavior_profiles: list[BehaviorProfile] = field(default_factory=list)
    signed_graph_nodes: list[str] = field(default_factory=list)
    signed_graph_edges: list[GraphEdge] = field(default_factory=list)   # confirmed 边（含 sign=±1、q↔证据、推断补全）

    # 控制流度量（Controller 写，metrics.py 算）；Φ 为唯一算力控制信号（无 C）
    Phi: float = 0.0               # 签名三角形挫折（所有 sign=−1 的负边入 T⁻）
    k: int = 0                     # 谱分区簇数（eigengap）
    clusters: dict = field(default_factory=dict)   # {cluster_id: [doc_id, ...]}
    depth_I: int = 0               # 反驳深度（Φ 分档；迭代期间冻结）
    conflict_detected: bool = False

    hypothesis_space: list[Hypothesis] = field(default_factory=list)
    counterexample_queries: list[str] = field(default_factory=list)   # 反驳阶段 ③ 产出
    refutation_log: list[dict] = field(default_factory=list)
    refutation_feedback: dict = field(default_factory=dict)

    # 可信度审查传递记录（创新点 4 v2，Controller/⑤ 写）
    credibility_pass: dict = field(default_factory=dict)
    #   activated: bool                          是否执行（deception.enable）
    #   collusion_pairs: [{a, b, cos}]           CORE-1 跨来源近重复对
    #   flagged: [doc_id]                        参与互证对的节点
    #   adjudications: [{doc_id, verdict, justification}]  ⑤ 输出
    #   reliability_updates / quarantined / discounted_edges
    quarantined_doc_ids: list = field(default_factory=list)   # 被隔离节点（退出证据组装，图内保留）

    analysis_trace: list[dict] = field(default_factory=list)
    final_conclusion: str = ""
    iteration: int = 0
    fast_path: bool = False        # 是否走了共识快路径

    def log(self, agent_name: str, action: str, detail: str = ""):
        self.analysis_trace.append({
            "iteration": self.iteration,
            "agent": agent_name,
            "action": action,
            "detail": detail,
        })

    def get_profile_by_doc_id(self, doc_id: str) -> Optional[BehaviorProfile]:
        for p in self.behavior_profiles:
            if p.doc_id == doc_id:
                return p
        return None

    def get_evidence_by_doc_id(self, doc_id: str) -> Optional[Evidence]:
        for e in self.evidence_set:
            if e.doc_id == doc_id:
                return e
        return None

    def get_confirmed_edges(self) -> list[GraphEdge]:
        """所有 edge_class==confirmed 的边（含 sign=±1 与传递性补全）"""
        return [e for e in self.signed_graph_edges if e.edge_class == "confirmed"]

    def active_evidence(self) -> list[Evidence]:
        """未被隔离的证据（创新点 4 v2 QUARANTINE：deceptive 节点退出组装，
        图与权重保留。未启用欺骗感知时 quarantined 为空 = 全体证据）。"""
        q = set(self.quarantined_doc_ids or [])
        return [e for e in self.evidence_set if e.doc_id not in q]

    def is_quarantined(self, doc_id: str) -> bool:
        return doc_id in set(self.quarantined_doc_ids or [])

    def get_negative_edges(self) -> list[GraphEdge]:
        """confirmed 负边（sign=-1）—— Φ/h_graph 只数这些"""
        return [e for e in self.signed_graph_edges
                if e.edge_class == "confirmed" and e.sign == -1]

    # ---------- JSON 序列化（供 metrics.py CLI 复用） ----------
    def to_json(self) -> str:
        return json.dumps(_awm_to_dict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "AWM":
        return _awm_from_dict(json.loads(s))


# ---- 序列化辅助（dataclass asdict + 字段裁剪） ----
_SIMPLE_FIELDS = (
    "query", "query_summary", "task_type", "task_instruction", "answer_format",
    "extra_context", "initial_doc_ids", "signed_graph_nodes", "Phi", "k",
    "clusters", "depth_I", "conflict_detected", "counterexample_queries",
    "refutation_log", "refutation_feedback", "credibility_pass",
    "quarantined_doc_ids",
    "analysis_trace", "final_conclusion", "iteration", "fast_path",
)


def _awm_to_dict(awm: AWM) -> dict:
    d = {f: getattr(awm, f) for f in _SIMPLE_FIELDS}
    d["evidence_set"] = [asdict(e) for e in awm.evidence_set]
    d["behavior_profiles"] = [asdict(p) for p in awm.behavior_profiles]
    d["signed_graph_edges"] = [asdict(e) for e in awm.signed_graph_edges]
    d["query_profile"] = asdict(awm.query_profile) if awm.query_profile else None
    d["hypothesis_space"] = [asdict(h) for h in awm.hypothesis_space]
    return d


def _awm_from_dict(d: dict) -> AWM:
    awm = AWM()
    for f in _SIMPLE_FIELDS:
        if f in d:
            setattr(awm, f, d[f])
    awm.evidence_set = [Evidence(**e) for e in d.get("evidence_set", [])]
    awm.behavior_profiles = [BehaviorProfile(**p) for p in d.get("behavior_profiles", [])]
    awm.signed_graph_edges = [GraphEdge(**e) for e in d.get("signed_graph_edges", [])]
    awm.hypothesis_space = [Hypothesis(**h) for h in d.get("hypothesis_space", [])]
    if d.get("query_profile"):
        awm.query_profile = BehaviorProfile(**d["query_profile"])
    return awm
