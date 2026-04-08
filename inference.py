"""
inference.py - MSME Payment Dispute OpenEnv Baseline Agent

STDOUT FORMAT (mandatory — auto-scorer reads this exactly):
  [START] task=<task_name> env=msme-dispute model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>

Required env vars:
  API_BASE_URL  — LLM endpoint
  MODEL_NAME    — model identifier
  HF_TOKEN      — API key
  ENV_URL       — environment server (default: http://localhost:7860)
"""
import os, sys, json, re, requests
from openai import OpenAI
from requests.exceptions import RequestException
import time

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",       "http://localhost:7860")
ENV_NAME     = "msme-dispute"

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
http = requests.Session()
http.trust_env = False
VALID_LABELS = ["delayed_payment", "partial_payment", "payment_denial"]

# ── Logging (mandatory format) ─────────────────
def log_start(task_name):
    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done, error=None):
    action_str = json.dumps(action, separators=(',', ':')) if isinstance(action, dict) else str(action)
    # Remove newlines from action string — must be single line
    action_str = action_str.replace('\n', ' ').replace('\r', '')[:200]
    err = error if error else "null"
    done_str = "true" if done else "false"
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={err}", flush=True)

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)

# ── Environment calls ──────────────────────────
def call_env(endpoint, payload=None, method="POST"):
    url = f"{ENV_URL}/{endpoint}"
    last_err = None
    for attempt in range(3):
        try:
            r = http.get(url, timeout=30) if method == "GET" else \
                http.post(url, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()
        except RequestException as e:
            last_err = e
            if attempt < 2:
                time.sleep(0.7 * (attempt + 1))
    raise last_err

# ── LLM helper ────────────────────────────────
def llm(prompt):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

def _heuristic_task1(email: dict) -> dict:
    text = f"{email.get('subject', '')} {email.get('body', '')}".lower()
    if any(k in text for k in ["partial", "shortfall", "remaining", "short payment", "paid only"]):
        return {"label": "partial_payment"}
    if any(k in text for k in ["deny", "refus", "not be processed", "not honour", "not honor", "reject"]):
        return {"label": "payment_denial"}
    return {"label": "delayed_payment"}

def _extract_amount(text: str) -> int:
    m = re.search(r"rs\.?\s*([0-9][0-9,]*)", text, re.IGNORECASE)
    return int(m.group(1).replace(",", "")) if m else 0

def _heuristic_task2(email: dict) -> dict:
    text = f"{email.get('subject', '')}\n{email.get('body', '')}"
    claimant = "Unknown Claimant"
    opponent = "Unknown Opponent"

    m = re.search(r"M/s\s+(.+?)\s+(?:hereby\s+serves.*?\s+)?(?:against|to)\s+M/s\s+(.+?)\s+for", text, re.IGNORECASE)
    if m:
        claimant = m.group(1).strip().rstrip(".,")
        opponent = m.group(2).strip().rstrip(".,")

    due_date = "Unknown"
    m = re.search(r"due\s+(?:by|date\s*[:\-]?)\s*([^.\n]+)", text, re.IGNORECASE)
    if m:
        due_date = m.group(1).strip().rstrip(".,")

    days_overdue = 0
    m = re.search(r"(\d+)\s+days\s+(?:overdue|past\s+due)", text, re.IGNORECASE)
    if m:
        days_overdue = int(m.group(1))

    return {
        "claimant": claimant,
        "opponent": opponent,
        "amount": _extract_amount(text),
        "due_date": due_date,
        "days_overdue": days_overdue,
    }

def _heuristic_task3(obs: dict) -> dict:
    ctx = obs.get("context", {})
    try:
        amt = f"Rs. {int(ctx.get('amount', 0)):,}"
    except Exception:
        amt = f"Rs. {ctx.get('amount', 0)}"
    evidence = ", ".join(ctx.get("evidence", []))
    letter = (
        f"Without prejudice, this is a formal demand notice issued by {ctx.get('claimant', 'the claimant')} "
        f"to {ctx.get('opponent', 'the opponent')} regarding Invoice #{ctx.get('invoice_no', 'N/A')} for {amt}. "
        f"The payment due date was {ctx.get('due_date', 'N/A')} and the amount is overdue by {ctx.get('days_overdue', 0)} days. "
        f"This matter concerns {ctx.get('dispute_type', 'payment default')}. Supporting records include: {evidence}. "
        f"You are called upon to clear the full outstanding amount within 15 days of receipt of this notice. "
        f"Failing payment, proceedings will be initiated before the MSME Facilitation Council and all legal remedies will be pursued. "
        f"Interest at the applicable rate under the MSMED Act, 2006, including the statutory higher rate for delayed payment, "
        f"will be claimed along with costs and other consequences in law."
    )
    return {"letter": letter}

# ── Task agents ───────────────────────────────
def agent_task1(obs):
    email = obs["email"]
    prompt = f"""Classify this MSME payment dispute email.

Subject: {email['subject']}
Body: {email['body']}

Choose exactly one label:
- delayed_payment  : full invoice unpaid and overdue
- partial_payment  : only part of invoice was paid
- payment_denial   : buyer refusing or denying payment entirely

Reply with ONLY the label, nothing else."""
    raw = llm(prompt)
    if raw:
        raw = raw.lower().strip()
        for v in VALID_LABELS:
            if v in raw:
                return {"label": v}
    return _heuristic_task1(email)

def agent_task2(obs):
    email = obs["email"]
    prompt = f"""Extract structured facts from this MSME payment dispute notice.

Subject: {email['subject']}
Body: {email['body']}

Return ONLY valid JSON (no markdown, no explanation):
{{"claimant":"company making claim","opponent":"company being claimed against","amount":80000,"due_date":"31st March 2024","days_overdue":14}}

Rules:
- claimant/opponent: exclude M/s prefix
- amount: integer rupees only
- days_overdue: integer only"""
    raw = llm(prompt)
    if not raw:
        return _heuristic_task2(email)
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                return _heuristic_task2(email)
        return _heuristic_task2(email)

def agent_task3(obs):
    ctx = obs["context"]
    try:
        amt = f"Rs. {int(ctx.get('amount', 0)):,}"
    except Exception:
        amt = f"Rs. {ctx.get('amount', 0)}"
    prompt = f"""Draft a formal MSME payment demand letter.

Claimant:     {ctx.get('claimant')}
Opponent:     {ctx.get('opponent')}
Invoice no:   {ctx.get('invoice_no', 'N/A')}
Invoice date: {ctx.get('invoice_date', 'N/A')}
Due date:     {ctx.get('due_date', 'N/A')}
Amount:       {amt}
Days overdue: {ctx.get('days_overdue')}
Dispute type: {ctx.get('dispute_type')}
Evidence:     {', '.join(ctx.get('evidence', []))}

Requirements:
1. Assertive legal tone — no begging or excessive politeness
2. Cite MSMED Act 2006 / MSME Delayed Payments Act
3. Demand full payment within 15 days
4. State compound interest at 3x RBI rate will apply
5. Reference invoice number and exact rupee amount
6. Minimum 180 words
7. End with signature block

Write ONLY the letter text."""
    raw = llm(prompt)
    if raw and len(raw.split()) >= 50:
        return {"letter": raw}
    return _heuristic_task3(obs)

AGENTS = {1: agent_task1, 2: agent_task2, 3: agent_task3}
TASK_NAMES = {1: "classify_dispute", 2: "extract_facts", 3: "draft_demand_letter"}

# ── Run one episode (single mode) ─────────────
def run_episode(task_id, seed=42):
    task_name = TASK_NAMES[task_id]
    log_start(task_name)

    rewards = []
    steps   = 0
    success = False
    last_error = None

    try:
        resp = call_env("reset", {"task_id": task_id, "seed": seed, "mode": "single"})
        obs  = resp["observation"]

        # Task 3 can be multi-turn in single mode; others are single-step.
        max_turns = 3 if task_id == 3 else 1
        done = False
        turn = 0

        while not done and turn < max_turns:
            action = AGENTS[task_id](obs)
            result = call_env("step", {"action": action})

            reward = float(result["reward"])
            done   = bool(result["done"])
            steps  += 1
            turn   += 1
            rewards.append(reward)

            log_step(steps, action, reward, done)

            # For multi-turn task3, continue using current observation if asked to revise.
            if not done and task_id == 3:
                obs = result.get("state", {}).get("observation", obs)

        success = done and (rewards[-1] > 0.0 if rewards else False)

    except Exception as e:
        last_error = str(e)
        # Keep reward in strict (0,1) even on transport/runtime failures.
        log_step(steps + 1, {}, 0.01, True, error=last_error)
        success = False

    log_end(success, steps, rewards)
    return rewards

# ── Run sequential episode ─────────────────────
def run_sequential_episode(seed=42):
    task_name = "sequential_all_tasks"
    log_start(task_name)

    rewards    = []
    steps      = 0
    success    = False
    last_error = None

    try:
        resp = call_env("reset", {"seed": seed, "mode": "sequential"})
        obs  = resp["observation"]

        for step_num in range(1, 4):
            task_id = step_num
            action  = AGENTS[task_id](obs)
            result  = call_env("step", {"action": action})

            reward = float(result["reward"])
            done   = bool(result["done"])
            steps += 1
            rewards.append(reward)

            log_step(steps, action, reward, done)

            if not done:
                obs = result["state"]["observation"]

        success = True

    except Exception as e:
        last_error = str(e)
        # Keep reward in strict (0,1) even on transport/runtime failures.
        log_step(steps + 1, {}, 0.01, True, error=last_error)
        success = False

    log_end(success, steps, rewards)
    return rewards

# ── Main ──────────────────────────────────────
def main():
    # Health check (stderr so it doesn't pollute stdout scorer)
    try:
        h = call_env("health", method="GET")
        print(f"# Environment: {ENV_URL} | status: {h.get('status')}", file=sys.stderr)
    except Exception as e:
        print(f"# ERROR: Cannot reach environment at {ENV_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    all_rewards = {}

    # Run all 3 tasks in single mode (required by spec)
    for task_id in [1, 2, 3]:
        print(f"\n# --- Task {task_id}: {TASK_NAMES[task_id]} ---", file=sys.stderr)
        rewards = run_episode(task_id, seed=42)
        all_rewards[f"task_{task_id}"] = rewards

    # Also run sequential mode (bonus — shows environment depth)
    print(f"\n# --- Sequential mode ---", file=sys.stderr)
    seq_rewards = run_sequential_episode(seed=42)
    all_rewards["sequential"] = seq_rewards

    # Summary to stderr (not stdout — keep stdout clean for scorer)
    print("\n# ══ SUMMARY ══", file=sys.stderr)
    for name, rws in all_rewards.items():
        avg = sum(rws) / len(rws) if rws else 0
        print(f"# {name}: avg={avg:.2f} rewards={[round(r,2) for r in rws]}", file=sys.stderr)

    os.makedirs("output", exist_ok=True)
    with open("output/inference_results.json", "w") as f:
        json.dump({"scores": all_rewards, "model": MODEL_NAME,
                   "env_url": ENV_URL}, f, indent=2)
    print("# Results saved to output/inference_results.json", file=sys.stderr)

if __name__ == "__main__":
    main()
