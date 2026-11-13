import argparse
import numpy as np, torch
from flcore.trainmodel.models import FedAvgCNN
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data

torch.manual_seed(0)
args = argparse.Namespace(
    model=FedAvgCNN(in_features=3, num_classes=9, dim=1024), algorithm="FedAvg",
    dataset="PathMNIST", device="cpu", save_folder_name="items", num_classes=9,
    batch_size=32, local_learning_rate=0.01, local_epochs=1, few_shot=0,
    learning_rate_decay=False, learning_rate_decay_gamma=0.99,
    seed=0, noisy_client=0, noise_rate=0.4)

for cid in [0, 3]:
    c = Client(args, cid, 0, 0, train_slow=False, send_slow=False)
    correct, n, loss_sum = c.clean_test_metrics()
    test = read_client_data("PathMNIST", cid, is_train=False)
    X = torch.stack([x for x, _ in test]); Y = torch.stack([y for _, y in test])
    with torch.no_grad():
        out = c.model.eval()(X)
    ref_loss = torch.nn.functional.cross_entropy(out, Y, reduction="sum").item()
    ref_correct = (out.argmax(1) == Y).sum().item()
    assert n == len(test), (n, len(test))
    assert correct == ref_correct
    assert abs(loss_sum - ref_loss) < 1e-2 * max(1, abs(ref_loss)) / 100
    assert c.clean_test_metrics() == (correct, n, loss_sum)   # deterministic
    tr_c, tr_n, tr_l = c.train_loss_metrics()
    assert tr_n == len(c.get_train_data())                    # no dropped samples
    print(f"client {cid}: test n={n} acc={correct/n:.4f} loss={loss_sum/n:.4f} | "
          f"train n={tr_n} loss={tr_l/tr_n:.4f}")
print("ln(9) =", round(float(np.log(9)), 4), "(untrained model loss should be near this)")
print("all Step 4 checks passed")
