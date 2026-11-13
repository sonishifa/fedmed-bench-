import copy, types
import torch
from flcore.trainmodel.models import FedAvgCNN
from flcore.servers.serverbase import Server

torch.manual_seed(0)
g = FedAvgCNN(in_features=3, num_classes=9, dim=1024)
flat = lambda m: torch.cat([p.detach().reshape(-1) for p in m.parameters()])
D = flat(g).numel()

base = torch.randn(D)
true_deltas = [base + 0.1 * torch.randn(D) for _ in range(4)] + [-base]   # client 4 points opposite
true_deltas[2] = torch.zeros(D)                                          # client 2 did not move
models = []
for d in true_deltas:
    m = copy.deepcopy(g); off = 0
    for p in m.parameters():
        k = p.numel(); p.data += d[off:off + k].view_as(p); off += k
    models.append(m)

n = torch.tensor([24067, 20963, 14883, 11055, 19028], dtype=torch.float)
fake = types.SimpleNamespace(global_model=g, uploaded_models=models,
                             uploaded_weights=(n / n.sum()).tolist(), uploaded_ids=[0, 1, 2, 3, 4])
m = Server.compute_update_stats(fake)

# reference: mean_delta must equal (FedAvg aggregate) - global
agg = sum(w * flat(mm) for w, mm in zip(fake.uploaded_weights, models))
ref_mean = agg - flat(g)
assert abs(m["global_update_norm"] - ref_mean.norm().item()) < 1e-3 * ref_mean.norm().item()
for i, d in enumerate(true_deltas):
    assert abs(m[f"client_{i}_update_norm"] - d.norm().item()) < 1e-3 * max(1, d.norm().item())
import math
assert math.isnan(m["client_2_cosine"]) and m["client_2_update_norm"] < 1e-3
assert m["client_0_cosine"] > 0.5 and m["client_4_cosine"] < 0 and m["client_4_cosine_loo"] < -0.9
for k, v in sorted(m.items()):
    print(f"{k:28s} {v: .4f}")
print("all Step 5 checks passed")
