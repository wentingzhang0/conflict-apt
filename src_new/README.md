# new_src —— 冲突感知多智能体 CTI 分析方法（新版方案落地）

本目录是 `improve_reference/方案设计.md` 的**完整 Python 实现**，与旧版 `src/` 并存、互不依赖。
方法 = 1 个确定性编排器（Controller）+ 3 个 LLM 推理智能体，三项创新点闭合于同一张
带符号证据图：

- **创新点 1（控制，~35%）** 冲突结构感知的自适应推理：PATH/DEPTH 两段单轴控制（推理强度；原 ROUTING 旋钮已删除，分析报告改固定双通道）
- **创新点 2（表征，~40%）** ABP 规范键驱动的带符号证据图 + 谱分区 + 冲突度量（C/Φ）
- **创新点 3（精化，~25%）** 反例驱动的图条件假设证伪

## 目录结构

```
new_src/
├── controller.py              # ① 确定性编排器（非 LLM）：流程编排、门控、迭代、反驳协作链
├── metrics.py                 # 所有确定性计算（C/Φ/k/h_graph/谱分区/门控）；纯函数 API + CLI
├── run.py                     # 运行入口
├── settings.yaml              # 全局共享参数（公平性）+ 本方法专属阈值/权重
├── core/                      # 确定性数值与数据结构
│   ├── awm.py                 #   AWM + GraphEdge(edge_class 三档) + BehaviorProfile(canonical_keys)
│   ├── canonical_match.py     #   规范键逐维三档匹配（零 LLM）
│   ├── conflict.py            #   启发式预标注 + 三层符号聚合 + 边权推导
│   ├── transitivity.py        #   neutral 对多路径加权传递性补全
│   ├── signed_graph.py        #   带符号图构建（文档级节点 + 查询锚点）+ h_graph 分量
│   ├── spectral_partition.py  #   签名拉普拉斯（uncertain 降权）+ eigengap 选 k + k-means
│   └── abp.py                 #   ABP 抽取（五维描述 + 四维规范键）+ 嵌入
├── agents/                    # 3 个 LLM 智能体（星型拓扑，经 Controller 中转）
│   ├── evidence_graph_agent.py#   ② 检索 + ABP 规范键抽取 + 确定性建图（mode=full/expand）
│   ├── hypothesis_agent.py    #   ③ 生成+评估 / 图条件反驳查询生成
│   ├── reporter_agent.py      #   ④ 双通道回锚 + 标准化 + ATE 防退化
│   └── prompts/               #   prompt yaml（abp/query_summarizer/hypothesis_gen/hypothesis_eval/reporter/refutation_query）
├── tasks/                     # 四任务适配层（TAA/ATE/MCQ/RCM）
├── retrieval/                 # Weaviate + BM25 混合检索（复制自旧 src/）
├── utils/                     # llm_client / embedding / settings（复制自旧 src/，路径已指向 new_src）
└── tests/                     # 单测：test_metrics.py、test_controller_flow.py
```

## 运行

```bash
# 单条分析请求
python -m new_src.run --query-file path/to/report.txt --task taa
python -m new_src.run --query "..." --task mcq --raw

# 确定性度量 CLI（供 Claude Code 团队模式复用，方案 6.2）
python -m new_src.metrics gating --graph awm.json --task taa
python -m new_src.metrics frustration --graph awm.json
```

输出（默认 JSON）：`parsed_answer` + AWM 摘要（C/Φ/k/depth_I/fast_path/迭代轮数/各边计数）。

## 依赖（部署）

与旧 `src/` 相同：`numpy`、`networkx`、`scikit-learn`、`weaviate-client`、`langchain-huggingface`
（BGE-M3）、`openai`（兼容端点）。先填写 `new_src/settings.yaml` 中标注 `# TODO` 的部署字段
（LLM 端点、BGE-M3 路径、Weaviate 实例）。

## 与旧 src/ 的关键区别

| 维度 | 旧 src/ | 新 new_src/ |
|------|---------|-------------|
| 冲突检测 | LLM 两两四维对比 | ABP 规范键**确定性**匹配（零两两 LLM，O(k) 抽取） |
| 边分类 | sign ±1/0 | 三档 `edge_class`：confirmed / uncertain / none |
| PATH 门控 | `has_conflicts`（看 sign=-1） | 无 confirmed 负边**且**无 uncertain 边才早退（高召回） |
| Φ/C | 含所有负边 | **仅** confirmed 负边（uncertain 不污染） |
| neutral 对 | 不补全 | 多路径加权传递性补全（吸收矛盾） |
| 反驳 | LLM 软反驳 | 扩展池反例 + 归因不相交硬过滤 + confirmed conflict 图条件闸门 + Δh_graph 收敛 |
| 数值计算 | 散落 signed_graph/orchestrator | 集中 `metrics.py`（纯函数 + CLI），Controller 不自行估算 |

## 消融开关位置（对应方案 3.1.6 / 3.2.6 / 3.3.5）

| 消融 | 开关 |
|------|------|
| A0 自适应 vs 恒完整 | 强制 `Controller` 走完整路径（跳过 fast_path 分支） |
| A1 恒快路径 | 强制 `fast_path=True` |
| A2 DEPTH 分档（θ₁/θ₂/I_max，任务自适应） | `settings.yaml: adaptive.{frustration_rebuttal_gate, frustration_depth2, rebuttal_max_rounds, depth_thresholds}` |
| A3（可解释）任务自适应冲突画像 C 的 α_m 加权 | `adaptive.task_dim_weights.*` |
| B1 规范-确定性 vs LLM 两两 | 切换 `core/conflict.py` 的 `match_all` 为 LLM 变体 |
| B2 规范键 on/off | 关闭 ABP 的 canonical_keys 抽取（仅文本嵌入匹配） |
| C1 扩展池 vs 初始池 | `Controller._refute_round`：`retrieval.top_k_expand` 与排除 `initial_doc_ids` |
| C2 图条件闸门 on/off | `_refute_round` 的 `legal = [... sign==-1]` 过滤 |
| C3 θ_H 扫描 | `refutation.theta_H` |

## 测试

```bash
python -m new_src.tests.test_metrics          # 确定性数值：uncertain 不污染 C/Φ、传递性、CLI 一致
python -m new_src.tests.test_controller_flow  # 控制流：快路径/完整路径/反驳链/可复现（stub 网络层）
```
