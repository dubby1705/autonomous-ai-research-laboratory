import sys, os, json
sys.path.insert(0, os.getcwd())
from lab.web_api import create_app
c = create_app().test_client()
runs = c.get("/api/observatory/runs").get_json()
rid = runs.get("latest_run_id") or (runs.get("runs") or [{}])[0].get("run_id", "")
print("RUN_ID:", rid)
r = c.get("/api/observatory/report?run_id=" + rid).get_json()
print("REPORT KEYS:", sorted(r.keys()))
for s in r.get("sections", []):
    rows = s.get("rows") or s.get("items") or []
    print("SECTION:", s.get("title"), "| keys:", sorted(s.keys()), "| rows:", len(rows),
          "| first row:", json.dumps(rows[0], default=str)[:160] if rows else None)
st = c.get("/api/explorations/" + rid).get_json()
print("RUN STATE KEYS:", sorted(st.keys()))
print("stages:", [(x.get("key"), x.get("state")) for x in st.get("stages", [])])
print("counts:", st.get("counts"), "stats:", json.dumps(st.get("stats"), default=str)[:200])
p = c.get("/api/papers").get_json()
plist = p.get("papers") or p if isinstance(p, list) else p.get("papers", [])
print("PAPERS:", len(plist), "first:", json.dumps(plist[0], default=str)[:200] if plist else None)
f = c.get("/api/failures").get_json()
fl = f.get("failures") if isinstance(f, dict) else f
print("FAILURES:", len(fl) if fl else 0, "first:", json.dumps((fl or [None])[0], default=str)[:200])
print("DONE")
