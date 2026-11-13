import argparse
import numpy as np, torch
from flcore.trainmodel.models import FedAvgCNN
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data

args = argparse.Namespace(
    model=FedAvgCNN(in_features=3, num_classes=9, dim=1024), algorithm="FedAvg",
    dataset="PathMNIST", device="cpu", save_folder_name="items", num_classes=9,
    batch_size=32, local_learning_rate=0.01, local_epochs=1, few_shot=0,
    learning_rate_decay=False, learning_rate_decay_gamma=0.99,
    seed=0, noisy_client=0, noise_rate=0.4)

def labels(data): return np.array([int(y) for _, y in data])

for cid in range(5):
    c = Client(args, cid, 0, 0, train_slow=False, send_slow=False)
    clean = labels(read_client_data("PathMNIST", cid, is_train=True))
    noisy = labels(c.get_train_data())
    changed = int((clean != noisy).sum())
    print(f"client {cid}: n={len(clean)}  changed={changed}  expected={int(0.4*len(clean)) if cid == 0 else 0}")
    if cid == 0:
        r = c.noise_record
        assert changed == int(0.4 * len(clean))
        assert (r["original_labels"] != r["replacement_labels"]).all()
        assert (clean[r["indices"]] == r["original_labels"]).all()
        assert (noisy[r["indices"]] == r["replacement_labels"]).all()
        assert (labels(c.get_train_data()) == noisy).all()           # repeated load same
        ys = torch.cat([y for _, y in c.load_train_data()]).numpy()   # loader path uses cache
        assert len(ys) <= len(clean)
        c2 = Client(args, cid, 0, 0, train_slow=False, send_slow=False)
        assert (labels(c2.get_train_data()) == noisy).all()           # same seed -> same corruption
        print("  noise checks passed; replacement label histogram:",
              np.bincount(r["replacement_labels"], minlength=9).tolist())
    else:
        assert changed == 0
    test_file = labels(read_client_data("PathMNIST", cid, is_train=False))
    test_loader = torch.cat([y for _, y in c.load_test_data()]).numpy()
    assert (np.sort(test_file) == np.sort(test_loader)).all()
print("all Step 3 checks passed")
