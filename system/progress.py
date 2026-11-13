"""Progress of a phase: current round per run and estimated time left.
Usage (from system/):  python progress.py 2      (or 1 / 3; for phase 3 set BEST_MU if != 0.01)"""
import glob, json, os, sys

GR = int(os.environ.get("GR", 200))
phase = sys.argv[1] if len(sys.argv) > 1 else "2"
best_mu = os.environ.get("BEST_MU", "")

if phase == "1":
    names = ["fedavg_seed0", "fedprox_mu0.01_seed0"]
elif phase == "2":
    names = [f"fedprox_mu{m}_seed0" for m in ("0.001", "0.1", "1.0")]
elif phase == "ext":
    names = [f"fedprox_mu0.1_seed{s}_clean" for s in (0, 1, 2)]
else:
    names = []
    for s in (1, 2):
        names += [f"fedavg_seed{s}", f"fedprox_mu0.01_seed{s}"]
        if best_mu and best_mu != "0.01":
            names.append(f"fedprox_mu{best_mu}_seed{s}")
    for s in (0, 1, 2):
        names += [f"fedavg_seed{s}_clean", f"fedprox_mu0.01_seed{s}_clean"]

sec_per_round, left_rounds = [], 0
for n in names:
    hits = sorted(glob.glob(f"../outputs/{n}_????????/metrics.jsonl"))
    if not hits:
        print(f"{n:32s} not started")
        left_rounds += GR
        continue
    rows = [json.loads(l) for l in open(hits[-1]) if l.strip()]
    r = rows[-1]["round"]
    t = [x["round_total_time_sec"] for x in rows[-10:] if x.get("round_total_time_sec")]
    if t:
        sec_per_round.append(sum(t) / len(t))
    left_rounds += GR - r
    print(f"{n:32s} round {r}/{GR}" + ("  DONE" if r >= GR else ""))

spr = sum(sec_per_round) / len(sec_per_round) if sec_per_round else 22.0
mins = left_rounds * spr / 60
print(f"\nrounds left: {left_rounds}  |  ~{spr:.1f} sec/round  |  time left ~{int(mins // 60)} hr {int(mins % 60)} min")
