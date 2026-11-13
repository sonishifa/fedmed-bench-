import csv, glob, json, math
import numpy as np, torch
from flcore.trainmodel.models import FedAvgCNN
from utils.data_utils import read_client_data

def latest(prefix):
    return sorted(glob.glob(f"../outputs/{prefix}_*"), key=lambda p: __import__("os").path.getmtime(p))[-1]

def load_rows(d):
    return list(csv.DictReader(open(f"{d}/metrics.csv")))

def num(v):
    return float("nan") if v == "" else float(v)

runs = {"FedAvg": latest("fedavg_seed0_smoke"), "FedProx": latest("fedprox_mu0.01_seed0_smoke")}
eval_keys = ["global_test_acc", "global_test_loss", "global_train_loss"] + \
            [f"client_{i}_{m}" for i in range(5) for m in ["test_acc", "test_loss", "train_loss"]]
upd_keys = ["global_update_norm"] + \
           [f"client_{i}_{m}" for i in range(5) for m in ["update_norm", "cosine", "cosine_loo"]]

rows_by = {}
for algo, d in runs.items():
    rows = load_rows(d); rows_by[algo] = rows
    cfg = json.load(open(f"{d}/config.json"))
    print(f"\n{algo}: {d}")
    assert [r["round"] for r in rows] == ["0", "1", "2"], [r["round"] for r in rows]
    for r in rows:
        for k in eval_keys:
            assert math.isfinite(num(r[k])), (algo, r["round"], k, r[k])
        for k in upd_keys:
            if r["round"] == "0":
                assert r[k] == "", (algo, k, "should be empty at round 0")
            else:
                assert math.isfinite(num(r[k])), (algo, r["round"], k, r[k])
    print("  rounds 0-2 present, all metrics finite, update stats only on rounds 1-2")
    print("  num_corrupted_labels:", cfg["num_corrupted_labels"], "| commit:", cfg["pfllib_commit"][:10],
          "| split_seed:", cfg["split_seed"], "| partition_alpha:", cfg["partition_alpha"])

    # final saved model must reproduce the final logged metrics
    sd = torch.load(f"{d}/final_model.pt", map_location="cpu")
    model = FedAvgCNN(in_features=3, num_classes=9, dim=1024)
    keys = list(model.state_dict().keys())
    assert [v.shape for v in sd.values()] == [model.state_dict()[k].shape for k in keys]
    model.load_state_dict(dict(zip(keys, sd.values()))); model.eval()
    last = rows[-1]; tot_c = tot_n = 0
    for cid in range(5):
        test = read_client_data("PathMNIST", cid, is_train=False)
        X = torch.stack([x for x, _ in test]); Y = torch.stack([y for _, y in test])
        with torch.no_grad():
            c = (model(X).argmax(1) == Y).sum().item()
        tot_c += c; tot_n += len(Y)
        assert abs(c / len(Y) - num(last[f"client_{cid}_test_acc"])) < 1e-6, (algo, cid)
    assert abs(tot_c / tot_n - num(last["global_test_acc"])) < 1e-6
    print(f"  final_model.pt reproduces round-2 metrics (global acc {tot_c/tot_n:.4f})")

# paired inputs identical
a, b = [np.load(f"{runs[k]}/noise_indices.npz") for k in ["FedAvg", "FedProx"]]
for k in ["indices", "original_labels", "replacement_labels"]:
    assert np.array_equal(a[k], b[k]), k
print("\nnoise records identical across FedAvg/FedProx:", len(a["indices"]), "labels")
for k in eval_keys:
    assert rows_by["FedAvg"][0][k] == rows_by["FedProx"][0][k], k
print("round-0 metrics identical -> same partition, corruption, initial weights")
print("\nall Step 8 checks passed")
