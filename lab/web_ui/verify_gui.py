#!/usr/bin/env python3
"""Verify AARL simple GUI: file integrity, routing, live API, full research flow."""
import json, os, subprocess, sys, time, urllib.request, urllib.error

BASE = "http://127.0.0.1:5050"
HERE = os.path.dirname(os.path.abspath(__file__))          # lab/web_ui
ROOT = os.path.dirname(HERE)                                # repo root
JS_PATH = os.path.join(HERE, "simple.js")
API_PATH = os.path.join(ROOT, "lab", "web_api.py")

errors = []

def check(cond, msg):
    if not cond:
        errors.append(msg)

def get(path, timeout=10):
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            ct = r.headers.get("Content-Type", "")
            body = r.read()
            if "json" in ct:
                return r.status, json.loads(body.decode("utf-8"))
            return r.status, body.decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return None, str(e)

# ---- 1. simple.js file content integrity ----
check(os.path.exists(JS_PATH), "simple.js missing at %s" % JS_PATH)
js = open(JS_PATH, encoding="utf-8").read()
for name in ["renderIdle", "renderRunning", "updateRunning", "paintActivity",
             "bestSurvivor", "hypGlyph", "renderReport", "appendReportPaper",
             "engage", "tick", "loadReport", "ready"]:
    check(name + "(" in js, "simple.js missing function: " + name)
check(js.rstrip().endswith("})();\n") or js.rstrip().endswith("})();"),
      "simple.js does not close IIFE cleanly")
check(js.count("function "), "functions in simple.js: %d" % js.count("function "))
print("[1] simple.js: %d bytes, %d functions" % (len(js), js.count("function ")))

# ---- 2. routing: / serves simple.html ----
status, body = get("/")
check(status == 200, "GET / returned %s" % status)
check("simple.js" in body and "simple.css" in body,
      "GET / does not look like simple.html (has simple.js/css: %s)" % ("yes" if ("simple.js" in body and "simple.css" in body) else "no"))
print("[2] GET / -> %d, simple.html served" % status)

# ---- 3. simple.js served at /simple.js ----
status, body = get("/simple.js")
check(status == 200, "GET /simple.js returned %s" % status)
check(len(body) > 15000, "GET /simple.js too small: %d bytes" % len(body))
check("renderReport" in body, "GET /simple.js missing renderReport")
check(body.rstrip().endswith("})();\n") or body.rstrip().endswith("})();"),
      "GET /simple.js does not close IIFE cleanly")
print("[3] GET /simple.js -> %d, %d bytes" % (status, len(body)))

# ---- 4. create a research exploration ----
req_body = json.dumps({"problem": "Design a new advanced GPU architecture"}).encode()
req = urllib.request.Request(BASE + "/api/explorations",
                            data=req_body,
                            headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        run = json.loads(r.read().decode())
except urllib.error.HTTPError as e:
    run = json.loads(e.read().decode())
    check(False, "POST /api/explorations failed: %s" % json.dumps(run))
check("run_id" in run, "POST /api/explorations missing run_id: %s" % run)
run_id = run["run_id"]
check(run.get("status") in ("queued", "running", "done"),
      "POST /api/explorations unexpected status: %s" % run.get("status"))
print("[4] POST /api/explorations ->", run_id, run.get("status"))

# ---- 5. get exploration, events, report ----
for path in ("/api/explorations/%s" % run_id,
             "/api/observatory/events?run_id=%s" % run_id,
             "/api/observatory/report?run_id=%s" % run_id):
    st, b = get(path)
    check(st == 200, "%s -> %s" % (path, st))
print("[5] explored /api/explorations/%s, events, report OK" % run_id)

# ---- 6. poll briefly to prove tick() path would work ----
for i in range(3):
    st, b = get("/api/explorations/%s" % run_id)
    check(st == 200, "poll %d failed: %s" % (i, st))
    if b.get("status") in ("done", "failed"):
        print("[6] run reached %s after %d polls" % (b.get("status"), i+1))
        break
    time.sleep(1)
else:
    print("[6] poll loop: run still %s (expected during real research)" % (b.get("status")))

# ---- 7. report endpoint returns sections ----
st, report = get("/api/observatory/report?run_id=%s" % run_id)
check(st == 200, "report endpoint failed: %s" % st)
sections = report.get("sections", [])
check(len(sections) >= 1, "report had %d sections (expected >=1)" % len(sections))
for s in sections:
    check("title" in s or s.get("empty"), "section missing title/empty: %s" % (s.keys()))
print("[7] report: %d sections, run_id=%s" % (len(sections), report.get("run_id")))

# ---- 8. hypotheses endpoint ----
st, hyps = get("/api/observatory/hypotheses?run_id=%s" % run_id)
check(st == 200, "hypotheses endpoint failed: %s" % st)
print("[8] hypotheses: %d total" % (hyps.get("total_stored", len(hyps.get("hypotheses", [])))))

# ---- 9. papers endpoint ----
st, papers = get("/api/papers")
check(st == 200, "papers endpoint failed: %s" % st)
print("[9] papers: %d" % (len(papers.get("papers", [])) if papers else 0))

print("\n=== VERIFICATION %s ===" % ("PASSED" if not errors else "FAILED (%d issues)" % len(errors)))
if errors:
    for e in errors:
        print("  - " + e)
    sys.exit(1)
print("All checks passed. GUI wiring is sound.")
