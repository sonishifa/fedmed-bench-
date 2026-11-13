"""Checks after `./run_main.sh check` (3-round runs):
  A. FedProx mu=0 == FedAvg (metrics, RNG fingerprints, final parameters)  -> implementation check
  B. FedAvg noisy vs clean share RNG state every round (same batch order / init) and the same
     corruption subset (applied vs hypothetical)                         -> paired clean baseline
  C. New diagnostics present; evaluation RNG guard held (run would have crashed otherwise)."""
import csv, glob, json, math
import numpy as np, torch


def one(pattern):
    hits = glob.glob(f"../outputs/{pattern}_????????")
    assert len(hits) == 1, f"expected exactly one folder for {pattern}, found {hits}"
    return hits[0]


def rows(d):
    return list(csv.DictReader(open(f"{d}/metrics.csv")))


def f(v):
    return float("nan") if v == "" else float(v)


avg, prox, clean = one("fedavg_seed0_check3"), one("fedprox_mu0.0_seed0_check3"), one("fedavg_seed0_clean_check3")
ra, rp, rc = rows(avg), rows(prox), rows(clean)
assert [r["round"] for r in ra] == [r["round"] for r in rp] == [r["round"] for r in rc] == ["0", "1", "2", "3"]

# ---------- A. FedProx(mu=0) vs FedAvg ----------
skip = {"round_train_time_sec", "round_total_time_sec"}
worst = 0.0
for a, p in zip(ra, rp):
    assert a["rng_state_crc"] == p["rng_state_crc"], ("RNG differs", a["round"])
    for k in a:
        if k in skip or a[k] == "" and p[k] == "":
            continue
        x, y = f(a[k]), f(p[k])
        if math.isnan(x) and math.isnan(y):
            continue
        worst = max(worst, abs(x - y))
sa = torch.load(f"{avg}/final_model.pt", map_location="cpu")
sp = torch.load(f"{prox}/final_model.pt", map_location="cpu")
assert [v.shape for v in sa.values()] == [v.shape for v in sp.values()]
pdiff = max((x - y).abs().max().item() for x, y in zip(sa.values(), sp.values()))
print(f"A. FedProx(mu=0) vs FedAvg: RNG fingerprints equal every round; "
      f"max metric diff {worst:.2e}; max parameter diff {pdiff:.2e}")
assert worst < 1e-5 and pdiff < 1e-5, "FedProx(mu=0) should reproduce FedAvg"

# ---------- B. noisy vs clean pairing ----------
for a, c in zip(ra, rc):
    assert a["rng_state_crc"] == c["rng_state_crc"], ("noisy/clean RNG differs", a["round"])
assert ra[0]["global_test_acc"] == rc[0]["global_test_acc"], "same init -> same round-0 test acc"
na, nc = np.load(f"{avg}/noise_indices.npz"), np.load(f"{clean}/noise_indices.npz")
for k in ["indices", "original_labels", "replacement_labels"]:
    assert np.array_equal(na[k], nc[k]), k
assert bool(na["applied"]) and not bool(nc["applied"])
ca, cc = json.load(open(f"{avg}/config.json")), json.load(open(f"{clean}/config.json"))
assert ca["num_corrupted_labels"] == 9626 and cc["num_corrupted_labels"] == 0
print("B. noisy vs clean: RNG fingerprints equal every round (same init + batch order); "
      "same 9626-sample subset, applied=True vs False; num_corrupted_labels 9626 vs 0")

# ---------- C. diagnostics ----------
keys = ["client_0_corrupted_pred_noisy", "client_0_corrupted_pred_original", "client_0_corrupted_pred_other",
        "rng_state_crc", "best_val_acc_so_far", "best_val_round_so_far", "global_val_acc"]
for name, rs, d in [("noisy", ra, avg), ("clean", rc, clean)]:
    for k in keys:
        assert all(r[k] != "" for r in rs), (name, k)
    s = [f(rs[i]["client_0_corrupted_pred_noisy"]) + f(rs[i]["client_0_corrupted_pred_original"])
         + f(rs[i]["client_0_corrupted_pred_other"]) for i in range(4)]
    assert all(abs(x - 1) < 1e-9 for x in s)
    assert glob.glob(f"{d}/best_val_model.pt"), "best_val_model.pt missing"
    print(f"C. {name}: corrupted-subset pred_noisy by round =",
          [round(f(r["client_0_corrupted_pred_noisy"]), 4) for r in rs],
          "| pred_original =", [round(f(r["client_0_corrupted_pred_original"]), 4) for r in rs],
          "| best val round =", rs[-1]["best_val_round_so_far"])
print("\nall pairing checks passed")
