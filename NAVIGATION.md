# StatusNow — Navigation Guide

## Quick Start

| Goal | Command | Docs |
|---|---|---|
| Train best V5 model (fast) | `python scripts/experiments/v5_train_best.py` | [README.md](README.md#usage) |
| Full V5 benchmark (all models + CV) | `python scripts/experiments/v5_full_benchmark.py` | [README.md](README.md#usage) |
| Train with your own Overture releases | `python pipeline/run_pipeline.py` | [pipeline/README.md](pipeline/README.md) |
| Run the V6 agent (interactive) | `python scripts/agent/main.py` | [docs/v6_agent_architecture.md](docs/v6_agent_architecture.md) |
| Run the V6 agent (async/high-throughput) | `python scripts/agent/async_main.py` | [docs/v6_agent_architecture.md](docs/v6_agent_architecture.md) |

## Repository Structure

```
StatusNow/
├── README.md                          # Main project documentation (START HERE)
├── NAVIGATION.md                      # This file
│
├── data/                              # Dataset files
│   └── combined_truth_dataset_expanded.parquet  # Gold Standard V5 (123k rows, 12 cities)
│
├── overture_releases/                 # Drop Overture parquet releases here
│   └── README.md                      # File naming conventions
│
├── pipeline/                          # Contributor training pipeline
│   ├── run_pipeline.py                # Single command: build data → engineer features → train
│   └── README.md                      # Full pipeline guide
│
├── scripts/
│   ├── data_processing/               # Reusable data utilities
│   │   ├── fetch_overture_expanded.py
│   │   ├── build_truth_expanded.py
│   │   └── merge_cities.py
│   │
│   ├── experiments/                   # Runnable experiments
│   │   ├── v5_train_best.py           # Train best model, save artifacts, export predictions
│   │   ├── v5_full_benchmark.py       # Full CV + all models + ensemble search
│   │   └── v6_enrichment_experiment.py  # V5 + website crawling + public records
│   │
│   ├── agent/                         # V6 AI agent layer
│   │   ├── main.py                    # Sync CLI entrypoint (interactive, approval-gated)
│   │   ├── async_main.py              # Async entrypoint (high-throughput, live dashboard)
│   │   ├── config.py                  # API keys and thresholds
│   │   ├── llm/interface.py           # LLM abstraction (Groq + Gemini)
│   │   ├── ingest.py                  # Phase 1: confidence filtering
│   │   ├── planner.py                 # Phase 2: research plan generation
│   │   ├── executor.py                # Phase 4: Tavily search + LLM prediction
│   │   ├── schemas.py                 # Pydantic schemas
│   │   └── agent_tools.py             # Tool execution helpers
│   │
│   └── research/                      # Research history (V3 → V5 iterations, archived)
│       ├── README.md                  # Explains what each script did
│       └── v5_holdout_eval.py         # Best research result: 89.41%
│
└── docs/
    └── v6_agent_architecture.md       # V6 agent design and implementation details
```

## Key Documentation

- **[README.md](README.md)** — Project overview, V5 results, feature table, project history
- **[pipeline/README.md](pipeline/README.md)** — How to train with your own Overture releases
- **[overture_releases/README.md](overture_releases/README.md)** — File naming conventions for releases
- **[scripts/research/README.md](scripts/research/README.md)** — Research script history (V3 → V5)
- **[docs/v6_agent_architecture.md](docs/v6_agent_architecture.md)** — V6 agent architecture and async pipeline
