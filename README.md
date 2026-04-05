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

## Tasks

| Task | Difficulty | Description | Scoring |
|------|-----------|-------------|---------|
| 1 | Easy | Classify dispute type from email | Exact match (1.0), adjacent class (0.4), wrong (0.0) |
| 2 | Medium | Extract structured facts from formal notice | Weighted field accuracy across 5 fields |
| 3 | Hard | Draft a formal demand letter | Rule-based + LLM-as-judge (0.0–1.0) |

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

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | API key |
| `ENV_URL` | Environment server URL for inference.py |

## Reward Function

```
Task 1: exact_match=1.0, adjacent_class=0.4, wrong=0.0
Task 2: 0.25*claimant + 0.25*opponent + 0.25*amount + 0.15*days_overdue + 0.10*due_date
Task 3: 0.5*(completeness+legal+tone+length) + 0.5*llm_judge
```
