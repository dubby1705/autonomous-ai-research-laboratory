import time

from lab.web_api import create_app

app = create_app()
c = app.test_client()
rid = "RUN-6A1EDE71"

print("=== polling RUN-6A1EDE71 ===")
for i in range(25):
    t = c.get(f"/api/explorations/{rid}").get_json()
    st = t.get("status", "??")
    print(f"{i:02d} {st:10s}  dirs={len(t.get('directions') or [])}  updated={t.get('updated_at', '')[-15:]}")
    time.sleep(2)

print()
print("=== FINAL ===")
print(json.dumps(c.get(f"/api/explorations/{rid}").get_json(), indent=2, default=str)[:2000])
