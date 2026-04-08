"""
graders.py - MSME Payment Dispute OpenEnv Graders
All graders return float in (0.0, 1.0) — strictly between, not at boundaries.
FIX 3: temperature=0 on LLM judge for deterministic, reproducible scores.
"""
import os
import re
from openai import OpenAI

VALID_LABELS = {"delayed_payment", "partial_payment", "payment_denial"}
ADJACENT = {
    "delayed_payment":  {"partial_payment"},
    "partial_payment":  {"delayed_payment", "payment_denial"},
    "payment_denial":   {"partial_payment"},
}

def _strict_score(score: float, eps: float = 0.001) -> float:
    """Keep scores strictly inside (0, 1) for validator compatibility."""
    return max(eps, min(1.0 - eps, float(score)))

# ── Task 1 ────────────────────────────────────
def grade_task1(action: dict, ground_truth: dict) -> dict:
    predicted = str(action.get("label", "")).strip().lower()
    expected  = str(ground_truth.get("label", "")).strip().lower()
    if predicted not in VALID_LABELS:
        return {"score": _strict_score(0.0), "reason": f"Invalid label '{predicted}'"}
    if predicted == expected:
        return {"score": _strict_score(1.0), "reason": "Exact match"}
    if predicted in ADJACENT.get(expected, set()):
        return {"score": _strict_score(0.4), "reason": f"Adjacent class (got '{predicted}', expected '{expected}')"}
    return {"score": _strict_score(0.0), "reason": f"Wrong class (got '{predicted}', expected '{expected}')"}

# ── Task 2 ────────────────────────────────────
def _normalise_name(name: str) -> str:
    name = name.lower().strip()
    for prefix in ["m/s ", "ms ", "mr ", "mrs ", "pvt ltd", "ltd", "co.", "co", "private limited"]:
        name = name.replace(prefix, "")
    return name.strip()

def _names_match(a: str, b: str) -> bool:
    a, b = _normalise_name(a), _normalise_name(b)
    return a in b or b in a

def _amount_match(predicted, expected) -> bool:
    try: return abs(int(predicted) - int(expected)) <= 100
    except: return False

def _days_match(predicted, expected) -> bool:
    try: return abs(int(predicted) - int(expected)) <= 3
    except: return False

# Robust date normaliser — handles LLM creative date formats
# e.g. "01/03/2024", "March 1 2024", "1st March 2024", "2024-03-01"
_MONTH_MAP = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    "january":1,"february":2,"march":3,"april":4,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}

def _normalise_date(s: str) -> str:
    """Return 'DD-MM-YYYY' or original string if can't parse."""
    import re as _re
    s = str(s).strip().lower()
    # Remove ordinal suffixes: 1st→1, 2nd→2, 3rd→3, 4th→4
    s = _re.sub(r'(\d+)(st|nd|rd|th)', r'\1', s)
    # Try YYYY-MM-DD or DD/MM/YYYY or DD-MM-YYYY
    m = _re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', s)
    if m: return f"{int(m.group(3)):02d}-{int(m.group(2)):02d}-{m.group(1)}"
    m = _re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', s)
    if m: return f"{int(m.group(1)):02d}-{int(m.group(2)):02d}-{m.group(3)}"
    # Try "15 march 2024" or "march 15 2024"
    m = _re.search(r'(\d{1,2})\s+([a-z]+)\s+(\d{4})', s)
    if m:
        mon = _MONTH_MAP.get(m.group(2)[:3])
        if mon: return f"{int(m.group(1)):02d}-{mon:02d}-{m.group(3)}"
    m = _re.search(r'([a-z]+)\s+(\d{1,2})\s+(\d{4})', s)
    if m:
        mon = _MONTH_MAP.get(m.group(1)[:3])
        if mon: return f"{int(m.group(2)):02d}-{mon:02d}-{m.group(3)}"
    return s  # fallback: return as-is

def _dates_match(a: str, b: str) -> bool:
    na, nb = _normalise_date(str(a)), _normalise_date(str(b))
    # Exact normalised match
    if na == nb: return True
    # Fallback: word overlap (original lenient check)
    al, bl = str(a).lower(), str(b).lower()
    return bool(al and any(w in al for w in bl.split()))

def grade_task2(action: dict, ground_truth: dict) -> dict:
    weights = {"claimant": 0.25, "opponent": 0.25, "amount": 0.25, "due_date": 0.10, "days_overdue": 0.15}
    breakdown = {}
    breakdown["claimant"]     = _strict_score(1.0 if _names_match(str(action.get("claimant","")), str(ground_truth["claimant"])) else 0.0)
    breakdown["opponent"]     = _strict_score(1.0 if _names_match(str(action.get("opponent","")), str(ground_truth["opponent"])) else 0.0)
    breakdown["amount"]       = _strict_score(1.0 if _amount_match(action.get("amount",-1), ground_truth["amount"]) else 0.0)
    breakdown["due_date"]     = _strict_score(1.0 if _dates_match(str(action.get("due_date","")), str(ground_truth["due_date"])) else 0.0)
    breakdown["days_overdue"] = _strict_score(1.0 if _days_match(action.get("days_overdue",-1), ground_truth["days_overdue"]) else 0.0)
    score = sum(weights[k] * breakdown[k] for k in weights)
    return {"score": round(_strict_score(score), 3), "breakdown": breakdown, "reason": f"Field accuracy: {score:.2f}"}

# ── Task 3 ────────────────────────────────────
def _rule_based_score(letter: str, criteria: dict) -> dict:
    letter_lower = letter.lower()
    must  = criteria.get("must_mention", [])
    legal = criteria.get("legal_elements", [])
    bad   = criteria.get("tone_keywords_bad", [])

    completeness = sum(1 for m in must  if m.lower() in letter_lower) / len(must)  if must  else 0.5
    legal_score  = sum(1 for l in legal if l.lower() in letter_lower) / len(legal) if legal else 0.5
    tone_score   = max(0.0, 1.0 - sum(1 for b in bad if b.lower() in letter_lower) * 0.3)
    length_score = min(1.0, len(letter.split()) / 150)

    return {
        "completeness":    round(_strict_score(completeness), 2),
        "legal_elements":  round(_strict_score(legal_score), 2),
        "tone":            round(_strict_score(tone_score), 2),
        "length":          round(_strict_score(length_score), 2),
    }

def grade_task3(action: dict, scenario: dict) -> dict:
    letter = str(action.get("letter", "")).strip()
    if not letter:
        return {"score": _strict_score(0.0), "breakdown": {}, "reason": "Empty letter"}

    criteria = scenario.get("grading_criteria", {})
    rb = _rule_based_score(letter, criteria)
    rb_combined = (rb["completeness"]*0.4 + rb["legal_elements"]*0.3 + rb["tone"]*0.2 + rb["length"]*0.1)

    # FIX 3: temperature=0 for deterministic, reproducible LLM judge scores
    llm_score = None
    try:
        api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        model    = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        api_key  = os.environ.get("HF_TOKEN", "")
        if api_key:
            client = OpenAI(base_url=api_base, api_key=api_key)
            ctx = scenario.get("context", {})
            prompt = f"""You are evaluating a formal MSME payment demand letter.

Context:
- Claimant: {ctx.get('claimant')}
- Opponent: {ctx.get('opponent')}
- Amount: Rs. {ctx.get('amount')}
- Days overdue: {ctx.get('days_overdue')}
- Dispute type: {ctx.get('dispute_type')}

Letter:
\"\"\"{letter[:1500]}\"\"\"

Score this letter 0.0 to 1.0 on:
- Legal assertiveness (not polite/begging)
- Factual completeness (all key facts present)
- Professional tone
- Mention of MSME Act or legal consequences

Respond with ONLY a single float. Example: 0.74"""
            resp = client.chat.completions.create(
                model=model,
                max_tokens=10,
                temperature=0,        # FIX 3: pinned to 0 for reproducibility
                messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.choices[0].message.content.strip()
            llm_score = float(re.findall(r"\d+\.?\d*", raw)[0])
            llm_score = _strict_score(max(0.0, min(1.0, llm_score)))
    except Exception:
        llm_score = None

    if llm_score is not None:
        final  = round(_strict_score(0.5 * rb_combined + 0.5 * llm_score), 3)
        method = "rule_based + llm_judge (temp=0)"
    else:
        final  = round(_strict_score(rb_combined), 3)
        method = "rule_based_only"

    return {
        "score":     final,
        "breakdown": {**rb, "llm_judge": llm_score},
        "reason":    f"Method: {method}. Score: {final:.3f}"
    }


# ── Task 3 Multi-turn grader ──────────────────
# Critic suggestion: make Task 3 interactive.
# Environment gives structured feedback after draft,
# agent can revise. Up to 2 revision rounds.

FEEDBACK_CHECKS = [
    ("MSMED Act 2006",      "msmed act",      "You did not cite the MSMED Act 2006. Add it explicitly."),
    ("interest rate",       "interest",       "You did not mention interest on overdue amount. Add compound interest at 3x RBI rate."),
    ("15-day ultimatum",    "15 day",         "You did not set a clear payment deadline. State: 'pay within 15 days'."),
    ("invoice reference",   "invoice",        "You did not reference the invoice number. Include it."),
    ("legal consequences",  "legal action",   "You did not state legal consequences. Mention MSME Facilitation Council or legal proceedings."),
]

def generate_feedback(letter: str) -> list:
    """
    Returns list of specific feedback items the agent must fix.
    Empty list means the letter is complete.
    """
    letter_lower = letter.lower()
    missing = []
    for label, keyword, feedback in FEEDBACK_CHECKS:
        if keyword not in letter_lower:
            missing.append({"missing": label, "feedback": feedback})
    return missing

def grade_task3_multiturn(action: dict, scenario: dict, turn: int = 1) -> dict:
    """
    Multi-turn aware grader.
    turn=1: draft — give feedback if incomplete
    turn=2: revision — score strictly, no more feedback
    turn=3: final — score as-is

    Returns grade_task3 result + feedback for next turn.
    """
    result    = grade_task3(action, scenario)
    letter    = str(action.get("letter", ""))
    feedback  = generate_feedback(letter) if turn < 3 else []
    needs_revision = len(feedback) > 0 and turn < 3

    return {
        **result,
        "turn":           turn,
        "feedback":       feedback,
        "needs_revision": needs_revision,
        "message": (
            f"Turn {turn}: {len(feedback)} issue(s) found. Revise and resubmit."
            if needs_revision else
            f"Turn {turn}: Letter accepted. Final score: {result['score']:.3f}"
        )
    }
