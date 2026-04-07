"""
validate.py - Pre-submission validator
Run this before submitting to catch any issues early.
Usage: python validate.py [ENV_URL]
"""
import sys
import json
import requests

ENV_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

def check(name, fn):
    try:
        ok, msg = fn()
        status = PASS if ok else FAIL
        results.append((status, name, msg))
        print(f"{status} {name}: {msg}")
        return ok
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"{FAIL} {name}: {e}")
        return False

print(f"\nValidating environment at: {ENV_URL}\n{'='*50}")

# 1. Root returns 200
def test_root():
    r = requests.get(f"{ENV_URL}/", timeout=10)
    return r.status_code == 200, f"status={r.status_code}"
check("Root endpoint returns 200", test_root)

# 2. Health endpoint
def test_health():
    r = requests.get(f"{ENV_URL}/health", timeout=10)
    return r.status_code == 200, f"status={r.status_code}"
check("Health endpoint returns 200", test_health)

# 3. Tasks endpoint lists 3 tasks
def test_tasks():
    r = requests.get(f"{ENV_URL}/tasks", timeout=10)
    data = r.json()
    n = len(data.get("tasks", []))
    return n >= 3, f"found {n} tasks (need >= 3)"
check("Tasks endpoint lists 3+ tasks", test_tasks)

# 4-6. reset() works for all 3 task IDs
for tid in [1, 2, 3]:
    def test_reset(t=tid):
        r = requests.post(f"{ENV_URL}/reset", json={"task_id": t, "seed": 42}, timeout=15)
        data = r.json()
        has_obs = "observation" in data
        return r.status_code == 200 and has_obs, f"status={r.status_code}, has_observation={has_obs}"
    check(f"reset(task_id={tid}) returns observation", test_reset)

# 7-9. step() returns reward in [0,1] for all tasks
dummy_actions = {
    1: {"label": "delayed_payment"},
    2: {"claimant": "Test Co", "opponent": "Other Co", "amount": 50000, "due_date": "31st March 2024", "days_overdue": 30},
    3: {"letter": "This is a formal demand letter from Test Company to Defendant Company. Invoice #001 for Rs. 50,000 is overdue. We demand payment under the MSME Act. Legal action and interest will apply if not paid within 15 days. We have all documentation to support this claim and will pursue all legal remedies available."}
}
for tid in [1, 2, 3]:
    def test_step(t=tid):
        requests.post(f"{ENV_URL}/reset", json={"task_id": t, "seed": 42}, timeout=15)
        r = requests.post(f"{ENV_URL}/step", json={"action": dummy_actions[t]}, timeout=30)
        data = r.json()
        reward = data.get("reward", -1)
        in_range = 0.0 <= float(reward) <= 1.0
        return r.status_code == 200 and in_range, f"reward={reward:.3f} (valid={in_range})"
    check(f"step() task {tid} returns reward in [0,1]", test_step)

# 10. state() endpoint
def test_state():
    r = requests.get(f"{ENV_URL}/state", timeout=10)
    return r.status_code == 200, f"status={r.status_code}"
check("state() endpoint works", test_state)

# 11. Reproducibility: same seed = same scenario
def test_reproducibility():
    r1 = requests.post(f"{ENV_URL}/reset", json={"task_id": 1, "seed": 99}, timeout=15).json()
    r2 = requests.post(f"{ENV_URL}/reset", json={"task_id": 1, "seed": 99}, timeout=15).json()
    s1 = r1["observation"]["email"]["subject"]
    s2 = r2["observation"]["email"]["subject"]
    return s1 == s2, f"same_subject={s1 == s2}"
check("Same seed produces same scenario (reproducible)", test_reproducibility)

# Summary
print(f"\n{'='*50}")
passed = sum(1 for s,_,_ in results if s == PASS)
total  = len(results)
print(f"Result: {passed}/{total} checks passed")
if passed == total:
    print("\nAll checks passed! Safe to submit.")
else:
    failed = [name for s,name,_ in results if s == FAIL]
    print(f"\nFailed checks:")
    for f in failed:
        print(f"  - {f}")
    print("\nFix the above before submitting.")
sys.exit(0 if passed == total else 1)
