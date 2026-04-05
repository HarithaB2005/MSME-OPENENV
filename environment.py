"""
environment.py - MSME Payment Dispute OpenEnv Environment
Implements reset(), step(), state() per OpenEnv spec.

FIX 1: Sequential episode mode — tasks chain together.
        Wrong Task 1 classification bleeds into Task 3 context,
        making this a true multi-step environment.
"""
from .tasks import get_scenario
from .graders import grade_task1, grade_task2, grade_task3, grade_task3_multiturn

class MSMEDisputeEnv:
    def __init__(self):
        self._mode       = "single"   # "single" or "sequential"
        self._task_id    = None
        self._scenario   = None
        self._done       = False
        self._step_count = 0
        self._seed       = None
        self._last_reward     = None
        self._episode_rewards = []

        # Sequential mode state
        self._seq_step        = 0     # which task we're on (1,2,3)
        self._seq_scenarios   = {}    # {1: scenario, 2: scenario, 3: scenario}
        self._seq_results     = {}    # {1: grader_result, 2: grader_result}
        self._agent_t1_label  = None  # agent's Task1 answer bleeds into Task3
        self._agent_t2_facts  = None  # agent's Task2 answer bleeds into Task3
        self._t1_fallback_used = False # True if agent gave invalid Task1 label

        # Multi-turn Task 3 state (up to 3 turns: draft → revise → final)
        self._t3_turn          = 1
        self._t3_scenario      = None
        self._t3_done          = False

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def reset(self, task_id: int = 1, seed: int = None, mode: str = "single") -> dict:
        """
        Start a fresh episode.

        mode="single"     — one task, one step (original behaviour)
        mode="sequential" — all 3 tasks chained, agent's earlier
                            answers affect later task context
        task_id           — used in single mode only
        seed              — fixed seed for reproducibility
        """
        self._seed            = seed
        self._done            = False
        self._step_count      = 0
        self._last_reward     = None
        self._episode_rewards = []
        self._agent_t1_label  = None
        self._agent_t2_facts  = None
        self._seq_results     = {}
        self._t1_fallback_used = False
        self._t3_turn          = 1
        self._t3_done          = False

        if mode == "sequential":
            self._mode      = "sequential"
            self._task_id   = None
            self._seq_step  = 1
            # Pre-load all 3 scenarios with same seed for consistency
            self._seq_scenarios = {
                1: get_scenario(1, seed),
                2: get_scenario(2, seed),
                3: get_scenario(3, seed),
            }
            self._scenario = self._seq_scenarios[1]
            return self._build_obs(task_id=1)

        else:
            if task_id not in (1, 2, 3):
                raise ValueError("task_id must be 1, 2, or 3")
            self._mode    = "single"
            self._task_id = task_id
            self._scenario = get_scenario(task_id, seed)
            return self._build_obs(task_id=task_id)

    def step(self, action: dict) -> dict:
        """
        Submit an action.

        Single mode:     one step → done.
        Sequential mode: three steps → done after Task 3.
                         Agent's Task1 label and Task2 facts are
                         injected into the Task3 observation so
                         a wrong classification genuinely hurts
                         the final letter quality score.
        """
        if self._scenario is None:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode done. Call reset() to start a new one.")

        if self._mode == "sequential":
            return self._step_sequential(action)
        else:
            return self._step_single(action)

    def state(self) -> dict:
        """Return current environment state."""
        if self._scenario is None:
            return {"status": "not_started"}

        base = {
            "mode":         self._mode,
            "seed":         self._seed,
            "step_count":   self._step_count,
            "done":         self._done,
            "last_reward":  self._last_reward,
        }

        if self._mode == "sequential":
            base.update({
                "current_task":    self._seq_step if not self._done else "done",
                "episode_rewards": self._seq_results,
                "average_reward":  (
                    round(sum(r["score"] for r in self._seq_results.values()) /
                          len(self._seq_results), 3)
                    if self._seq_results else None
                ),
                "observation": self._build_obs(self._seq_step) if not self._done else None,
            })
        else:
            base.update({
                "task_id":      self._task_id,
                "scenario_id":  self._scenario.get("id"),
                "observation":  self._build_obs(self._task_id),
            })

        return base

    # ─────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────

    def _step_single(self, action: dict) -> dict:
        result = self._grade(action, self._task_id, self._scenario)
        self._last_reward = result["score"]
        self._step_count += 1

        # Multi-turn: Task 3 allows up to 3 turns (draft → revise → final)
        if self._task_id == 3 and result.get("needs_revision") and self._t3_turn < 3:
            self._t3_turn += 1
            done = False  # not done yet — agent should revise
        else:
            self._t3_done = True
            done = True

        return {
            "state":    self.state(),
            "reward":   result["score"],
            "done":     done,
            "info":     result,
            "feedback": result.get("feedback", []),
            "message":  result.get("message", "")
        }

    def _step_sequential(self, action: dict) -> dict:
        current = self._seq_step
        scenario = self._seq_scenarios[current]

        result = self._grade(action, current, scenario)
        self._seq_results[current] = result
        self._last_reward = result["score"]
        self._step_count += 1

        # Store agent answers so they bleed into Task 3 context.
        # If Task 1 label is invalid/broken, fall back to "delayed_payment"
        # (most common type) so Task 3 context is always gradeable.
        VALID_LABELS = {"delayed_payment", "partial_payment", "payment_denial"}
        if current == 1:
            raw_label = str(action.get("label", "")).strip().lower()
            self._agent_t1_label = raw_label if raw_label in VALID_LABELS else "delayed_payment"
            if raw_label not in VALID_LABELS:
                # Record that a fallback was used — this is visible in state()
                self._t1_fallback_used = True
        elif current == 2:
            self._agent_t2_facts = action

        # Advance or finish
        if current < 3:
            self._seq_step += 1
            self._scenario = self._seq_scenarios[self._seq_step]
            done = False
        else:
            self._done = True
            done = True

        avg = round(
            sum(r["score"] for r in self._seq_results.values()) / len(self._seq_results), 3
        )

        return {
            "state":          self.state(),
            "reward":         result["score"],
            "done":           done,
            "task_completed": current,
            "next_task":      current + 1 if not done else None,
            "episode_avg":    avg if done else None,
            "info":           result,
        }

    def _build_obs(self, task_id: int) -> dict:
        if self._mode == "sequential":
            s = self._seq_scenarios.get(task_id, {})
        else:
            s = self._scenario

        if task_id == 1:
            return {
                "task":        "classify_dispute",
                "description": "Classify the dispute type from the email.",
                "email":       s["email"],
                "valid_labels": ["delayed_payment", "partial_payment", "payment_denial"],
                "action_format": {"label": "<one of the valid_labels>"},
                "note": "Your classification here will affect the context given in Task 3."
            }

        elif task_id == 2:
            return {
                "task":        "extract_facts",
                "description": "Extract structured facts from the formal notice.",
                "email":       s["email"],
                "action_format": {
                    "claimant":     "<company name>",
                    "opponent":     "<company name>",
                    "amount":       "<integer in rupees>",
                    "due_date":     "<date string>",
                    "days_overdue": "<integer>"
                },
                "note": "Your extracted facts will be used as context in Task 3."
            }

        elif task_id == 3:
            ctx = dict(s["context"])  # copy

            # FIX 1 CORE: Inject agent's earlier answers into Task 3 context.
            # If agent got Task 1 wrong, they get the wrong dispute_type injected.
            # This means a wrong Task 1 genuinely hurts Task 3 letter quality.
            if self._mode == "sequential":
                if self._agent_t1_label:
                    ctx["dispute_type_from_agent"] = self._agent_t1_label
                    ctx["dispute_type"] = self._agent_t1_label  # overwrites ground truth
                if self._agent_t2_facts:
                    for field in ["claimant", "opponent", "amount", "due_date", "days_overdue"]:
                        if field in self._agent_t2_facts:
                            ctx[field] = self._agent_t2_facts[field]

            return {
                "task":        "draft_demand_letter",
                "description": "Draft a formal MSME payment demand letter using the context below.",
                "context":     ctx,
                "action_format": {"letter": "<full demand letter text, minimum 150 words>"},
                "note": "Context was built from your Task 1 and Task 2 answers." if self._mode == "sequential" else ""
            }

    def _grade(self, action: dict, task_id: int, scenario: dict) -> dict:
        if task_id == 1:
            return grade_task1(action, {"label": scenario["label"]})
        elif task_id == 2:
            return grade_task2(action, scenario["ground_truth"])
        elif task_id == 3:
            # Build grading scenario with possibly agent-modified context
            grading_scenario = dict(scenario)
            if self._mode == "sequential" and self._agent_t1_label:
                ctx = dict(scenario["context"])
                ctx["dispute_type"] = self._agent_t1_label
                grading_scenario = {**scenario, "context": ctx}
            # Use multi-turn grader — tracks which revision turn we're on
            return grade_task3_multiturn(action, grading_scenario, turn=self._t3_turn)
