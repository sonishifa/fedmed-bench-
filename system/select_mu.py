"""Frozen mu-selection rule (seed 0, validation only):
pick the mu with the highest mean global val acc over the last 10 rounds (191-200);
ties (equal at 4 decimals) go to the smaller mu. Pool includes mu=0.01 from Phase 1.
Prints validation metrics and the corrupted-subset diagnostic only - no test numbers."""
import glob, json

GR = 200
pool = ["0.001", "0.01", "0.1", "1.0"]
res = []
for m in pool:
    hits = glob.glob(f"../outputs/fedprox_mu{m}_seed0_????????/metrics.jsonl")
    assert len(hits) == 1, f"mu={m}: expected one run folder, found {hits}"
    rows = [json.loads(l) for l in open(hits[0]) if l.strip()]
    assert len(rows) == GR + 1 and rows[-1]["round"] == GR, f"mu={m}: run not finished ({len(rows)} rows)"
    last10 = [r["global_val_acc"] for r in rows[-10:]]
    mean10 = sum(last10) / len(last10)
    best = max(rows, key=lambda r: (r["global_val_acc"], -r["round"]))
    f = rows[-1]
    pn, po = f["client_0_corrupted_pred_noisy"], f["client_0_corrupted_pred_original"]
    excess = pn - (1 - po) / 8
    res.append((m, mean10))
    print(f"mu={m:6s} mean val r191-200 = {mean10:.4f} | final val {f['global_val_acc']:.4f} | "
          f"best val {best['global_val_acc']:.4f} @ r{best['round']} | "
          f"c0 pred_noisy {pn:.3f} pred_original {po:.3f} excess {excess:+.3f}")

chosen = sorted(res, key=lambda t: (-round(t[1], 4), float(t[0])))[0][0]
print(f"\nSELECTED mu = {chosen}")
print(f"next:  BEST_MU={chosen} caffeinate -i ./run_main.sh 3")
