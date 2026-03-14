# StatusNow V6 Agent Architecture

## 1. Overview
StatusNow is a POI (point of interest) status classification system that determines whether a physical business or location is currently open or closed, based on digital footprint signals derived from Overture Maps releases.

The current V5 model achieves 89.41% balanced accuracy on a fully honest geographic hold-out evaluation, identifying 93.7% of all closed businesses correctly.

This document defines the next evolution (V6): an AI agent layer using the **Gemini API** that sits on top of the V5 model. When the model returns low-confidence predictions, the agent autonomously researches each POI using web search, presents a remediation plan to the user for approval, re-runs the classification pipeline with enriched data, and delivers structured output with updated confidence scores.

## 2. Problem Statement
### 2.1 The Low-Confidence Tail
The V5 model performs exceptionally on clear-cut cases but has a structural limitation: for the 10-15% of POIs where digital signals are ambiguous, confidence scores are low and the model is frequently wrong. These cases share common characteristics:
- Non-brand small businesses with sparse or stale digital presence
- Recent closures not yet reflected in Overture Maps data
- Places where delta features are zero by construction (2-release limitation)
- High-churn categories: coffee shops (15.2% error), pharmacies (22.2%), hotels (14.7%)

Manually reviewing hundreds of flagged POIs per data refresh cycle is expensive and does not scale. The current V5 pipeline has no automated path to resolve these ambiguous cases.

### 2.2 The Data Enrichment Gap
External data sources such as live web content, business listing status, and real-time review platforms contain signals that Overture Maps does not capture.

### 2.3 Opportunity
An agentic pipeline targeting the low-confidence tail eliminates manual review overhead while keeping humans in control. By constraining web search to ambiguous cases only, the system operates well within the 4,000 Tavily API credit budget and returns concrete improvements to the overall pipeline accuracy.

## 3. Goals and Non-Goals
### 3.1 Goals
- Automatically identify POIs where V5 model confidence falls below a configurable threshold.
- Generate a structured remediation plan grouping POIs by research strategy using **Gemini**.
- Require explicit human approval before any web search or pipeline re-run is executed.
- Perform targeted Tavily web searches to enrich low-confidence POIs with current data.
- Gemini independently evaluates the search results to output a final open/closed prediction and confidence score.
- Output a structured results file with the final agent predictions, confidence scores, and reasoning.
- Stay within 4,000 Tavily search credits per month at 1-3 searches per POI.

### 3.2 Non-Goals
- Replacing the V5 model or modifying its weights.
- Auto-submitting corrections back to Overture Maps.
- Handling real-time streaming data.
- Full automation without a human approval gate.

## 4. System Architecture
### 4.1 Agent Flow
The agent operates in five sequential phases. Human approval is a hard gate between phases 2 and 3.

| Phase | Description |
| --- | --- |
| **1. Ingest** | Load model output parquet, filter to rows where prediction confidence < threshold (default: 0.65). Parse POI data. |
| **2. Plan Generation** | **Gemini** generates a structured JSON plan grouping POIs by research strategy. Specifies ordered fallback search query templates, `max_results` (web pages) per query, and estimated credit cost. |
| **3. Human Approval Gate** | Plan is rendered as a readable summary. User approves as-is, edits group strategies, or rejects individual POIs. |
| **4. Execution** | For each approved POI, agent calls Tavily search tool. Results are parsed using **Gemini** to produce a final, independent label (Open/Closed) and reasoning. |
| **5. Output Generation** | Produces structured output file with original/enriched predictions, confidence deltas, and research provenance. |

### 4.2 Tech Stack
- **Python 3.11+**
- **Google Gemini API** (`google-genai` package)
    - `gemini-2.5-flash` or `gemini-1.5-pro` for Plan Generation.
    - `gemini-2.5-flash` for extraction and execution-phase tasks.
- **Tavily Python SDK** for Web Search.
- **FastAPI / Typer (CLI)** for the approval gate interface.
- **SQLite** for the review queue, tracking state, and audit logging.
- **Pydantic** for all schema definitions and Gemini structured output constraints.
- Existing V5 training code (`CatBoost` + `LightGBM` ensemble).

### 4.3 Directory Structure
To keep everything well-organized, the new agent logic will live under `scripts/agent/`:

```
scripts/agent/
├── README.md                 # Documentation for agent tools & usage
├── config.yaml               # Agent configuration (thresholds, prompts)
├── main.py                   # CLI entrypoint for orchestrating the cycle
├── ingest.py                 # Phase 1: Ingest and filtering module
├── planner.py                # Phase 2: Gemini planning logic
├── approval.py               # Phase 3: Interactive CLI/UI approval interface
├── executor.py               # Phase 4: Tavily integration & Gemini direct prediction
├── evaluator.py              # Phase 5: Output generation and metric comparison
├── schemas.py                # Pydantic schemas (Plan, PlanGroup, Result)
└── database.py               # SQLite audit log and state management
```

## 5. Functional Requirements
### 5.1 Confidence Threshold Filtering
- Must accept a `confidence_threshold` parameter (float, 0-1, default 0.65).
- System must flag POIs where `max(P(open), P(closed)) < threshold`.
- Must log how many POIs were flagged and what % of the batch this represents.

### 5.2 Plan Generation
- Plan must be machine-readable JSON (enforced via Gemini `response_schema`).
- Strategies drawn from a defined taxonomy: `web_search_name_address`, `web_search_business_status`, `web_search_category_specific`, `skip`.
- Total estimated credits must be displayed prior to approval.

### 5.3 Approval Gate
- Approval must be explicit via CLI/TUI.
- User can change strategy, remove POIs, or adjust `max_results` per search.
- Approved plans are immutable and logged to SQLite.

- Each POI category can be tuned to fetch between 1 and 5 web results per query (configurable up to 5, default 3). Use `1` for broad initial checks to conserve credits.
- The plan contains a list of `query_templates`. The agent runs the first query. If Gemini's returned confidence is too low (e.g. < 0.8), it dynamically falls back and loops to the second query.
- Search results are fed back into Gemini. Gemini generates an independent label (1 = Open, 0 = Closed), a confidence score, and text reasoning.
- Graceful error handling: retry Tavily once, log API failures.
- Real-time credit tracking; hard stop if budget exhausted.

### 5.5 Override Rules
- The agent's prediction replaces the original only if: (a) new confidence > old confidence + 0.05, or (b) new label differs with confidence > 0.75.

### 5.6 Output
- Output: Parquet (primary) and JSON summary.
- Required fields: `poi_id`, `original_label`, `original_confidence`, `enriched_label`, `enriched_confidence`, `confidence_delta`, `resolution_status`, `search_queries`, `sources`.

## 6. Key Data Schemas

### 6.1 Plan Schema (Pydantic)
```python
class PlanGroup(BaseModel):
    group_id: str
    strategy: str
    query_templates: list[str]
    poi_ids: list[str]
    estimated_credits_per_poi: int
    max_results: int

class agent_plan(BaseModel):
    plan_id: str
    created_at: datetime
    total_pois: int
    total_estimated_credits: int
    groups: list[PlanGroup]
```

### 6.2 Execution Result Schema
```python
class EnrichmentResult(BaseModel):
    poi_id: str
    original_label: int
    original_confidence: float
    enriched_label: Optional[int]
    enriched_confidence: Optional[float]
    confidence_delta: Optional[float]
    resolution_status: str
    search_queries: list[str]
    sources: list[str]
    agent_reasoning: str
```

## 7. Milestones
1. **M1 — Scaffold:** Project structure (`scripts/agent/`), Pydantic schemas, SQLite setup, confidence filter logic.
2. **M2 — Plan Generation:** Gemini integration with Structured Outputs for planning, human-readable summary renderer.
3. **M3 — Approval Gate:** TUI/CLI approval flow, edit support, audit sequence.
4. **M4 — Execution:** Tavily search integration, credit limit checking, Gemini direct prediction & reasoning output.
5. **M5 — Output Generation:** Saving final predictions and comparing against original metrics.
6. **M6 — Eval:** Run on held-out subset, measure metric lift.
