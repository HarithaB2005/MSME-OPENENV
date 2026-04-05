"""
server.py - FastAPI server with typed Pydantic models (OpenEnv spec compliant).
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from .environment import MSMEDisputeEnv
from .models import ResetRequest, StepRequest, ResetResponse, StepResponse

app = FastAPI(
    title="MSME Payment Dispute — OpenEnv",
    version="2.0.0",
    description="OpenEnv environment for MSME payment dispute resolution tasks."
)
env = MSMEDisputeEnv()

@app.get("/")
def root():
    return {"status": "ok", "env": "msme-dispute", "version": "2.0.0",
            "modes": ["single", "sequential"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    try:
        obs = env.reset(task_id=req.task_id, seed=req.seed, mode=req.mode)
        return ResetResponse(observation=obs, state=env.state())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    try:
        result = env.step(req.action)
        return StepResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    return env.state()

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"task_id": 1, "name": "classify_dispute",
             "difficulty": "easy",
             "description": "Classify a payment dispute email into delayed_payment, partial_payment, or payment_denial.",
             "action_schema": {"label": "str — one of [delayed_payment, partial_payment, payment_denial]"},
             "reward": "1.0 exact, 0.4 adjacent class, 0.0 wrong"},
            {"task_id": 2, "name": "extract_facts",
             "difficulty": "medium",
             "description": "Extract claimant, opponent, amount, due_date, days_overdue from a formal notice.",
             "action_schema": {"claimant": "str", "opponent": "str", "amount": "int", "due_date": "str", "days_overdue": "int"},
             "reward": "weighted field accuracy — 0.25+0.25+0.25+0.15+0.10"},
            {"task_id": 3, "name": "draft_demand_letter",
             "difficulty": "hard",
             "description": "Draft a professional MSME payment demand letter. Sequential mode chains Tasks 1+2 context.",
             "action_schema": {"letter": "str — min 150 words"},
             "reward": "0.5 * rule_based + 0.5 * llm_judge (temp=0)"},
        ],
        "modes": {
            "single":     "One task per episode. reset(task_id=1|2|3)",
            "sequential": "All 3 tasks chained. Agent answers flow into later tasks."
        }
    }
