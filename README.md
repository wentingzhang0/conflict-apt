# Conflict-Aware Multi-Agent CTI Analysis

**Conflict-aware multi-hypothesis analysis of multi-source cyber threat
intelligence, with deception-aware credibility review.**

When multiple cyber threat intelligence (CTI) reports describe the same attack
campaign, their conclusions occasionally conflict, and some reports may even
be deliberately deceptive (e.g. false-flag attribution). This repository
organizes the support and conflict relations among reports into a **signed
evidence graph** and distills the most credible conclusion instead of
collapsing onto a single answer too early:

1. **Signed evidence graph construction** — each report is distilled into a
   structured behavior profile with canonical keys; deterministic matching
   over canonical keys turns the support and conflict relations among reports
   into graph edges (confirmed / uncertain / none), and spectral partitioning
   carves out competing hypotheses aligned with the query.
2. **Adaptive scheduling** — a frustration-index-driven gate decides the
   reasoning depth: with no conflict a consensus conclusion is returned
   directly, and the heavier the conflict, the more rounds of reasoning are
   invested.
3. **Counterexample-driven hypothesis testing** — counterexamples that could
   overturn the leading hypothesis are sought on demand; the change of its
   strength on the graph decides whether the conclusion has stabilized.
4. **Deception-aware credibility review** — deterministic detectors flag
   coordinated false-flag reports (cross-source near-duplicates,
   attribution-content divergence); the LLM only seeks benign explanations
   for the flagged ones, and adjudicated deceptive nodes are quarantined from
   evidence assembly while the graph is kept intact for explainability.

The system is orchestrated by one deterministic controller plus four LLM
agents (evidence graph, hypothesis, reporter, credibility), all closed on the
same signed evidence graph.

## Repository layout

```
├── src_new/              # implementation of our method
│   ├── core/             # signed graph, spectral partition, conflict,
│   │                     # canonical matching, ABP, transitivity, deception
│   ├── agents/           # LLM agents and their prompts (YAML)
│   ├── tasks/            # task adapters (TAA / RCM / ATE / MCQ)
│   ├── retrieval/        # hybrid retriever over the CTI corpus
│   ├── utils/            # LLM client (vLLM OpenAI-compatible), settings
│   ├── tests/            # unit tests of the deterministic core
│   ├── controller.py     # deterministic orchestration loop
│   ├── metrics.py        # deterministic metrics (C, Φ, k, h_graph, gates)
│   └── run.py            # single-query entry point
├── baselines/            # compared methods: ArbGraph, CyberRAG,
│                         # ITER-RETGEN, NaiveRAG (NavieRAG/), RAGIntel,
│                         # Search-o1 (Search-o1-main/), adapted to CTIBench;
│                         # upstream sources of ArbGraph / CyberRAG are kept
│                         # in ArbGraph-src/ / CyberRAG-src/
└── data/                 # CTIBench task sets and the CTI report corpus
```

## Code structure

- `src_new/controller.py` — the deterministic orchestrator: pipeline staging,
  conflict gating, iteration, and the refutation chain; no LLM calls.
  (`src_new/agents/orchestrator.py` is a thin alias used by experiment
  runners.)
- `src_new/agents/` — the LLM agents (`evidence_graph_agent`,
  `hypothesis_agent`, `reporter_agent`, `credibility_agent`) plus prompt
  templates in `src_new/agents/prompts/` (YAML). Prompts default to the TAA
  wording; the `ate/`, `mcq/`, and `rcm/` subdirectories hold per-task
  overrides, loaded with a fallback to the default.
- `src_new/core/` — the deterministic core: `awm.py` (AWM data structures,
  `GraphEdge` with three edge classes, `BehaviorProfile` with canonical
  keys), `abp.py` (behavior profile extraction), `canonical_match.py`
  (four-dimension canonical-key matching that decides support / conflict /
  undecidable), `conflict.py` (heuristic pre-labeling and sign aggregation),
  `transitivity.py` (weighted transitive completion for neutral pairs),
  `signed_graph.py`, `spectral_partition.py` (signed Laplacian, eigengap,
  k-means), and `deception.py` (the deterministic deception detectors).
- `src_new/metrics.py` — all deterministic numeric computation (conflict
  level, frustration index, partition, hypothesis strength, gates) as pure
  functions, also usable as a CLI.
- `src_new/tasks/` — task adapters over a common `TaskAdapter` base
  (`taa.py`, `mcq.py`, `rcm.py`, `ate.py`).
- `src_new/retrieval/` — the Weaviate retriever, hybrid semantic + BM25
  search with RRF fusion.
- `src_new/utils/llm_client.py` — LLM calls through a vLLM OpenAI-compatible
  endpoint, with automatic truncation that guards against context overflow.

## Configuration

All shared settings live in `src_new/settings.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `global.top_k` | 10 | retrieval size |
| `global.retrieval_mode` | hybrid | `semantic` / `keyword` / `hybrid` |
| `global.hybrid_alpha` | 0.4 | semantic weight in RRF fusion |
| `global.timeout` | 1800 | per-sample timeout (s) |

Deployment fields marked `# TODO` (LLM endpoint, BGE-M3 path, Weaviate URL)
must be filled in before running. Pipeline-specific parameters (gates
θ₁/θ₂, per-task I_max, hypothesis-strength blending, counterexample pool
size, deception thresholds) are documented inline in the `pipeline:` section
of the same file.

## Setup

1. Serve a backbone LLM behind an OpenAI-compatible endpoint (e.g. vLLM) and
   point `src_new/settings.yaml` → `models.chat_model.api_base` at it; host
   the BGE-M3 embedding model (`models.embedding_model.path`); and serve the
   retrieval corpus in Weaviate (`models.weaviate_model.url`, class `APT`
   with fields `content`, `report_title`, `section_title`, `source_file`).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

```bash
# single analysis request
python -m src_new.run --query-file path/to/report.txt --task taa
python -m src_new.run --query "..." --task mcq --raw

# deterministic metrics over a dumped AWM graph (no LLM involved)
python -m src_new.metrics gating --graph awm.json --task taa
python -m src_new.metrics frustration --graph awm.json
python -m src_new.metrics partition --graph awm.json
python -m src_new.metrics h_graph --graph awm.json --hypothesis hyp.json

# unit tests of the deterministic core
python -m pytest src_new/tests
```

The default output is JSON: the parsed answer plus an AWM summary
(fast-path flag, frustration index Φ, number of hypotheses k, reasoning
depth, iteration count, edge statistics, refutation rounds).

## Baselines

`baselines/` contains the compared methods adapted to CTIBench: NaiveRAG,
ITER-RETGEN, Search-o1, RAGIntel, ArbGraph, and CyberRAG. Each directory
holds the CTIBench adapter (e.g. `ArbGraph/arbgraph_taa.py`,
`CyberRAG/cyberrag_taa.py`); the upstream sources of ArbGraph and CyberRAG
are kept in `ArbGraph-src/` and `CyberRAG-src/` with their original READMEs
and dependency files.

## Data sources

- The task sets (TAA/ATE/MCQ/RCM under `data/CTIBench-*`) come from the
  [CTIBench](https://github.com/IBM/CTIBench) benchmark;
  `data/cti-bench-official/` holds the official evaluation helpers.
- The retrieval corpus (`data/APT_md`) aggregates publicly available APT
  reports from three community-maintained GitHub collections:
  [blackorbird/APT_REPORT](https://github.com/blackorbird/APT_REPORT),
  [RedDrip7/APT_Digital_Weapon](https://github.com/RedDrip7/APT_Digital_Weapon),
  and
  [CyberMonitor/APT_CyberCriminal_Campagin_Collections](https://github.com/CyberMonitor/APT_CyberCriminal_Campagin_Collections).
  The original reports (PDF and web pages) are converted to plain text with
  DeepSeek-OCR and sectioned for retrieval.
