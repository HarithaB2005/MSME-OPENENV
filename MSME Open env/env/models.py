"""
models.py - Typed Pydantic models for OpenEnv spec compliance.
Observation, Action, Reward — all strictly typed.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal

# ── Reward ────────────────────────────────────
class MSMEReward(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Reward in [0.0, 1.0]")
    breakdown: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

# ── Actions ───────────────────────────────────
class Task1Action(BaseModel):
    label: Literal["delayed_payment", "partial_payment", "payment_denial"]

class Task2Action(BaseModel):
    claimant:     str
    opponent:     str
    amount:       int   = Field(..., gt=0)
    due_date:     str
    days_overdue: int   = Field(..., ge=0)

class Task3Action(BaseModel):
    letter: str = Field(
        ...,
        min_length=500,
        description="Full formal demand letter text; target at least 150 words.",
    )

class MSMEAction(BaseModel):
    """Union action — agent fills whichever field matches current task."""
    label:        Optional[str] = None   # Task 1
    claimant:     Optional[str] = None   # Task 2
    opponent:     Optional[str] = None   # Task 2
    amount:       Optional[int] = None   # Task 2
    due_date:     Optional[str] = None   # Task 2
    days_overdue: Optional[int] = None   # Task 2
    letter:       Optional[str] = None   # Task 3

    def to_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}

# ── Observations ─────────────────────────────
class EmailObs(BaseModel):
    subject: str
    body:    str

class Task1Observation(BaseModel):
    task:         Literal["classify_dispute"]
    description:  str
    email:        EmailObs
    valid_labels: List[str]
    action_format: Dict[str, str]
    note:         str = ""

class Task2Observation(BaseModel):
    task:         Literal["extract_facts"]
    description:  str
    email:        EmailObs
    action_format: Dict[str, str]
    note:         str = ""

class DisputeContext(BaseModel):
    claimant:     str
    opponent:     str
    amount:       int
    invoice_no:   str
    invoice_date: str
    due_date:     str
    days_overdue: int
    dispute_type: str
    evidence:     List[str]

class Task3Observation(BaseModel):
    task:         Literal["draft_demand_letter"]
    description:  str
    context:      Dict[str, Any]
    action_format: Dict[str, str]
    note:         str = ""

# ── Episode state ─────────────────────────────
class EpisodeState(BaseModel):
    mode:          str
    seed:          Optional[int]
    step_count:    int
    done:          bool
    last_reward:   Optional[float]
    task_id:       Optional[int]   = None
    scenario_id:   Optional[str]   = None
    current_task:  Optional[Any]   = None
    episode_rewards: Dict[str, Any] = Field(default_factory=dict)
    average_reward:  Optional[float] = None
    observation:   Optional[Dict[str, Any]] = None

# ── API request/response models ───────────────
class ResetRequest(BaseModel):
    task_id: int  = Field(1, ge=1, le=3)
    seed:    Optional[int] = None
    mode:    Literal["single", "sequential"] = "single"

class StepRequest(BaseModel):
    action: Dict[str, Any]

class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    state:       Dict[str, Any]

class StepResponse(BaseModel):
    state:           Dict[str, Any]
    reward:          float = Field(..., ge=0.0, le=1.0)
    done:            bool
    info:            Dict[str, Any]
    task_completed:  Optional[int]  = None
    next_task:       Optional[int]  = None
    episode_avg:     Optional[float] = None
