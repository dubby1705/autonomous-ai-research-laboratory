import json
from lab.memory import ResearchMemory

memory = ResearchMemory()
latest = None
best_n = 0
for name in memory.list_collection("explorations"):
    if name.startswith("RUN-D24B8F81"):
        best_n = max(best_n, best_n)
        latest = name

# Actually scan for highest rev
import re
directory = memory.collection_dir("explorations")
for name in os.listdir(directory):
    m = re.fullmatch(r"RUN-D24B8F81\.rev(\d+)\.json", name)
    if m and int(m.group(1)) > best_n:
        best_n = int(m.group(1))
        latest = name

print(f"Latest revision: {latest} (rev{best_n})")
print()

data = memory.get("explorations", latest.replace(".json", ""))
print("STATUS:", data.get("status"))
print("STAGES:", len(data.get("stages", [])))
print("DIRECTIONS:", len(data.get("directions", [])))
print("STATS keys:", list(data.get("stats", {}).keys()))
print("SPACE nodes:", len(data.get("space", {}).get("nodes", [])))
print("SPACE edges:", len(data.get("space", {}).get("edges", [])))
print()
print("Full status JSON (first 2500 chars):")
print(json.dumps(data, indent=2, default=str)[:2500])
