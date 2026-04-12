---
title: MSME Payment Dispute OpenEnv
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# MSME Payment Dispute — OpenEnv Environment

An AI agent training environment for handling MSME (Micro, Small and Medium Enterprise) payment disputes in India. Built for the OpenEnv Hackathon × Scaler Round 1.

## Problem Context

Millions of small businesses in India face delayed payments from buyers. Most MSME owners lack access to affordable legal drafting support. This environment trains AI agents to classify disputes, extract facts, and draft formal demand letters — practical tasks with real-world impact.

**Why this matters:** This is not a toy problem. Delayed payments are a systemic issue in India, and automating the legal drafting process (MSMED Act 2006 demands) directly solves a major bottleneck for small businesses.
**Environment Design:** The environment features a unique **Sequential Chaining Mode**, where an agent's errors in early tasks (e.g., misclassifying a dispute) directly degrade the context provided in later tasks (e.g., drafting the letter). It also supports **Multi-turn Revision Loops** via an LLM+Rule based judge that provides actionable feedback on missing legal clauses.

## Tasks

| Task | Difficulty | Description | Scoring |
|------|-----------|-------------|---------|
| 1 | Easy | Classify dispute type from email | Exact match (0.999), adjacent class (0.40), wrong (0.001) |
| 2 | Medium | Extract structured facts from formal notice | Weighted field accuracy across 5 fields, clamped to (0, 1) |
| 3 | Hard | Draft a formal demand letter | Rule-based + LLM-as-judge, clamped to (0, 1) |

## Action & Observation Spaces

**Task 1**
- Observation: `{email: {subject, body}, valid_labels: [...]}`
- Action: `{label: "delayed_payment" | "partial_payment" | "payment_denial"}`

**Task 2**
- Observation: `{email: {subject, body}}`
- Action: `{claimant: str, opponent: str, amount: int, due_date: str, days_overdue: int}`

**Task 3**
- Observation: `{context: {claimant, opponent, amount, invoice_no, due_date, days_overdue, dispute_type, evidence}}`
- Action: `{letter: str}` (minimum 150 words)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| POST | `/reset` | Start new episode `{task_id: 1-3, seed: int?}` |
| POST | `/step` | Submit action `{action: {...}}` |
| GET | `/state` | Current environment state |

## Setup

```bash
pip install -r requirements.txt
uvicorn env.server:app --host 0.0.0.0 --port 7860

# In another terminal:
python inference.py
```

## Baseline Scores

| Task | Model | Score | Agent Steps |
|------|-------|-------|-------------|
| 1. Classify | `gpt-4o-mini` | 0.400 | 1 |
| 2. Extract Facts | `gpt-4o-mini` | 0.950 | 1 |
| 3. Draft Letter | `gpt-4o-mini` | 0.375 | 3 (Multi-turn) |

## Sample Inference Output

```text
[START] task=draft_demand_letter env=msme-dispute model=gpt-4o-mini
[STEP] step=1 action={"letter": "..."} reward=0.25 done=false error=null
[STEP] step=2 action={"letter": "..."} reward=0.38 done=true error=null
[END] success=true steps=2 score=0.38 rewards=0.25,0.38
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | API key |
| `ENV_URL` | Environment server URL for inference.py |

## Reward Function

```
All task rewards are clamped strictly to (0, 1) for validator compatibility.
Task 1: exact_match=0.999, adjacent_class=0.40, wrong/invalid=0.001
Task 2: weighted field accuracy (claimant/opponent/amount/days_overdue/due_date), clamped to (0,1)
Task 3: 0.5*(completeness+legal+tone+length) + 0.5*llm_judge, clamped to (0,1)
```
