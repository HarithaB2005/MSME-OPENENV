"""
graders.py - MSME Payment Dispute OpenEnv Graders
All scores clamped to (0.05, 0.95).
CRITICAL: 0.999 rounds to 1.00 when printed as .2f — use 0.95 max instead.
"""
import os
import re

VALID_LABELS = {"delayed_payment", "partial_payment", "payment_denial"}
ADJACENT = {
    "delayed_payment":  {"partial_payment"},
    "partial_payment":  {"delayed_payment", "payment_denial"},
    "payment_denial":   {"partial_payment"},
}

def _c(score: float) -> float:
    """
    Clamp strictly to (0.05, 0.95).
    WHY 0.95 not 0.999: when printed as .2f, 0.999 rounds to 1.00 which
    the validator rejects. 0.95 prints as 0.95 safely.
    """
    return round(max(0.05, min(0.95, float(score))), 4)

# ── Task 1 ────────────────────────────────────
def grade_task1(action: dict, ground_truth: dict) -> dict:
    predicted = str(action.get("label", "")).strip().lower()
    expected  = str(ground_truth.get("label", "")).strip().lower()
    if predicted not in VALID_LABELS:
        return {"score": 0.05, "reason": f"Invalid label '{predicted}'"}
    if predicted == expected:
        return {"score": 0.95, "reason": "Exact match"}
    if predicted in ADJACENT.get(expected, set()):
        return {"score": 0.40, "reason": f"Adjacent class: got '{predicted}', expected '{expected}'"}
    return {"score": 0.05, "reason": f"Wrong class: got '{predicted}', expected '{expected}'"}

# ── Task 2 helpers ────────────────────────────
def _normalise_name(name: str) -> str:
    name = name.lower().strip()
    for prefix in ["m/s ", "ms ", "mr ", "mrs ", "pvt ltd", "pvt. ltd.", "ltd", "co.", "co", "private limited"]:
        name = name.replace(prefix, "")
    return name.strip()

def _names_match(a: str, b: str) -> bool:
    a, b = _normalise_name(a), _normalise_name(b)
    return a in b or b in a

def _amount_match(predicted, expected) -> bool:
    try: return abs(int(str(predicted).replace(",","")) - int(expected)) <= 500
    except: return False

def _days_match(predicted, expected) -> bool:
    try: return abs(int(predicted) - int(expected)) <= 5
    except: return False

_MONTH_MAP = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    "january":1,"february":2,"march":3,"april":4,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}

def _normalise_date(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', s)
    m = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', s)
    if m: return f"{int(m.group(3)):02d}-{int(m.group(2)):02d}-{m.group(1)}"
    m = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', s)
    if m: return f"{int(m.group(1)):02d}-{int(m.group(2)):02d}-{m.group(3)}"
    m = re.search(r'(\d{1,2})\s+([a-z]+)\s+(\d{4})', s)
    if m:
        mon = _MONTH_MAP.get(m.group(2)[:3])
        if mon: return f"{int(m.group(1)):02d}-{mon:02d}-{m.group(3)}"
    m = re.search(r'([a-z]+)\s+(\d{1,2})\s+(\d{4})', s)
    if m:
        mon = _MONTH_MAP.get(m.group(1)[:3])
        if mon: return f"{int(m.group(2)):02d}-{mon:02d}-{m.group(3)}"
    return s

def _dates_match(a: str, b: str) -> bool:
    if _normalise_date(a) == _normalise_date(b): return True
    al, bl = str(a).lower(), str(b).lower()
    return bool(al and any(w in al for w in bl.split()))

# ── Task 2 grader ─────────────────────────────
def grade_task2(action: dict, ground_truth: dict) -> dict:
    weights = {"claimant":0.25, "opponent":0.25, "amount":0.25, "due_date":0.10, "days_overdue":0.15}
    bd = {
        "claimant":     1.0 if _names_match(str(action.get("claimant","")),    str(ground_truth["claimant"])) else 0.0,
        "opponent":     1.0 if _names_match(str(action.get("opponent","")),    str(ground_truth["opponent"])) else 0.0,
        "amount":       1.0 if _amount_match(action.get("amount",-1),           ground_truth["amount"]) else 0.0,
        "due_date":     1.0 if _dates_match(str(action.get("due_date","")),    str(ground_truth["due_date"])) else 0.0,
        "days_overdue": 1.0 if _days_match(action.get("days_overdue",-1),      ground_truth["days_overdue"]) else 0.0,
    }
    raw = sum(weights[k] * bd[k] for k in weights)
    return {"score": _c(raw), "breakdown": bd, "reason": f"field_accuracy:{raw:.2f}"}

# ── Task 3 grader ─────────────────────────────
def _rule_based_score(letter: str, criteria: dict) -> dict:
    ll    = letter.lower()
    must  = criteria.get("must_mention", [])
    legal = criteria.get("legal_elements", [])
    bad   = criteria.get("tone_keywords_bad", [])

    completeness = sum(1 for m in must  if m.lower() in ll) / len(must)  if must  else 0.5
    legal_score  = sum(1 for l in legal if l.lower() in ll) / len(legal) if legal else 0.5
    tone_score   = max(0.05, 1.0 - sum(1 for b in bad if b.lower() in ll) * 0.3)
    length_score = min(0.95, len(letter.split()) / 150)

    return {
        "completeness":   round(_c(completeness), 3),
        "legal_elements": round(_c(legal_score), 3),
        "tone":           round(_c(tone_score), 3),
        "length":         round(_c(length_score), 3),
    }

def grade_task3(action: dict, scenario: dict) -> dict:
    letter = str(action.get("letter", "")).strip()
    if not letter:
        return {"score": 0.05, "breakdown": {}, "reason": "Empty letter"}

    criteria = scenario.get("grading_criteria", {})
    rb = _rule_based_score(letter, criteria)
    rb_combined = (rb["completeness"]*0.4 + rb["legal_elements"]*0.3 + rb["tone"]*0.2 + rb["length"]*0.1)

    llm_score = None
    try:
        api_key = os.environ.get("HF_TOKEN", "")
        if api_key and len(api_key) > 8:
            from openai import OpenAI
            client = OpenAI(
                base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
                api_key=api_key
            )
            ctx = scenario.get("context", {})
            resp = client.chat.completions.create(
                model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
                max_tokens=10, temperature=0,
                messages=[{"role": "user", "content":
                    f"Score this MSME demand letter 0.10 to 0.90.\n"
                    f"Claimant:{ctx.get('claimant')} Opponent:{ctx.get('opponent')} "
                    f"Amount:Rs.{ctx.get('amount')}\n"
                    f"\"\"\"{letter[:1000]}\"\"\"\n"
                    f"Output ONE float only, e.g. 0.72"
                }]
            )
            nums = re.findall(r"0\.\d+|\d+\.\d+", resp.choices[0].message.content)
            if nums:
                llm_score = max(0.10, min(0.90, float(nums[0])))
    except Exception:
        llm_score = None

    raw = (0.5 * rb_combined + 0.5 * llm_score) if llm_score is not None else rb_combined
    final = _c(raw)
    method = "rule+llm" if llm_score is not None else "rule_only"
    return {
        "score":     final,
        "breakdown": {**rb, "llm_judge": llm_score},
        "reason":    f"{method}:{final:.4f}"
    }

# ── Task 3 Multi-turn grader ──────────────────
FEEDBACK_CHECKS = [
    ("MSMED Act 2006",    "msmed act",    "Cite the MSMED Act 2006 explicitly."),
    ("interest rate",     "interest",     "Mention compound interest at 3x RBI rate."),
    ("15-day deadline",   "15 day",       "State: pay within 15 days."),
    ("invoice reference", "invoice",      "Reference the invoice number."),
    ("legal action",      "legal action", "Mention MSME Facilitation Council or legal proceedings."),
]

def generate_feedback(letter: str) -> list:
    ll = letter.lower()
    return [{"missing": lbl, "feedback": fb} for lbl, kw, fb in FEEDBACK_CHECKS if kw not in ll]

def grade_task3_multiturn(action: dict, scenario: dict, turn: int = 1) -> dict:
    result   = grade_task3(action, scenario)
    letter   = str(action.get("letter", ""))
    feedback = generate_feedback(letter) if turn < 3 else []
    needs    = len(feedback) > 0 and turn < 3
    return {
        **result, "turn": turn, "feedback": feedback, "needs_revision": needs,
        "message": (f"Turn {turn}: {len(feedback)} issue(s). Revise."
                    if needs else f"Turn {turn}: Accepted. Score:{result['score']:.4f}")
    }
