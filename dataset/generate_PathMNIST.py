import os
import ujson
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from medmnist import PathMNIST
from utils.dataset_utils import separate_data

num_clients = 5
num_classes = 9
alpha = 0.3
split_seed = 0
dir_path = "PathMNIST/"
plot_path = "../plots/client_class_distribution.png"
NORMALIZATION = "x/255.0 -> float32 in [0,1], CHW, no mean/std"


def convert(dataset):
    x = dataset.imgs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    y = dataset.labels.reshape(-1).astype(np.int64)
    return x, y


def allocate_test(train_counts, y_test, rng):
    """Har class ke official test examples ko clients mein usi fraction se baanto
    jo us class ke training examples ka hai (largest-remainder rounding)."""
    test_idx = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = np.where(y_test == c)[0]
        rng.shuffle(idx_c)
        frac = train_counts[:, c] / train_counts[:, c].sum()
        raw = frac * len(idx_c)
        counts = np.floor(raw).astype(int)
        leftover = len(idx_c) - counts.sum()
        counts[np.argsort(-(raw - counts))[:leftover]] += 1
        for k, part in enumerate(np.split(idx_c, np.cumsum(counts)[:-1])):
            test_idx[k].extend(part.tolist())
    return [np.sort(np.array(t, dtype=np.int64)) for t in test_idx]


def save_heatmap(train_counts, test_counts):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, counts, title in zip(axes, [train_counts, test_counts], ["Train", "Test"]):
        im = ax.imshow(counts, cmap="Blues", aspect="auto")
        for i in range(num_clients):
            for j in range(num_classes):
                v = counts[i, j]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="white" if v > counts.max() / 2 else "black")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_clients))
        ax.set_yticklabels([f"client {k}" for k in range(num_clients)])
        ax.set_xlabel("class")
        ax.set_title(f"{title} samples per client/class (Dir α={alpha})")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print("Saved", plot_path)


def generate():
    for sub in ["train", "test", "rawdata"]:
        os.makedirs(dir_path + sub, exist_ok=True)

    root = dir_path + "rawdata"
    x_train, y_train = convert(PathMNIST(split="train", download=True, size=28, root=root))
    x_val, y_val = convert(PathMNIST(split="val", download=True, size=28, root=root))
    x_test, y_test = convert(PathMNIST(split="test", download=True, size=28, root=root))
    print(f"train {x_train.shape}  val {x_val.shape}  test {x_test.shape}")

    # Dirichlet split sirf official TRAIN pe. Images ki jagah indices pass karte hain,
    # isliye separate_data seedha har client ke train indices lauta deta hai.
    np.random.seed(split_seed)  # separate_data global NumPy RNG use karta hai
    idx_lists, _, _ = separate_data(
        (np.arange(len(y_train)), y_train), num_clients, num_classes,
        niid=True, balance=False, partition="dir", alpha=alpha)
    train_idx = [np.sort(np.asarray(ix, dtype=np.int64)) for ix in idx_lists]

    train_client = np.full(len(y_train), -1, dtype=np.int64)
    for k, ix in enumerate(train_idx):
        assert (train_client[ix] == -1).all(), "train example assigned twice"
        train_client[ix] = k
    assert (train_client >= 0).all(), "some train example unassigned"

    train_counts = np.stack([np.bincount(y_train[ix], minlength=num_classes) for ix in train_idx])

    rng = np.random.default_rng(split_seed)
    test_idx = allocate_test(train_counts, y_test, rng)
    test_client = np.full(len(y_test), -1, dtype=np.int64)
    for k, ix in enumerate(test_idx):
        assert (test_client[ix] == -1).all(), "test example assigned twice"
        test_client[ix] = k
    assert (test_client >= 0).all(), "some test example unassigned"
    assert all(len(ix) > 0 for ix in test_idx), "a client has no test data"

    test_counts = np.stack([np.bincount(y_test[ix], minlength=num_classes) for ix in test_idx])

    # PFLlib file format: {k}.npz -> data={'x': ..., 'y': ...}
    for k in range(num_clients):
        with open(f"{dir_path}train/{k}.npz", "wb") as f:
            np.savez_compressed(f, data={"x": x_train[train_idx[k]], "y": y_train[train_idx[k]]})
        with open(f"{dir_path}test/{k}.npz", "wb") as f:
            np.savez_compressed(f, data={"x": x_test[test_idx[k]], "y": y_test[test_idx[k]]})

    np.savez(dir_path + "partition_indices.npz",
             train_client=train_client, test_client=test_client,
             **{f"train_idx_{k}": train_idx[k] for k in range(num_clients)},
             **{f"test_idx_{k}": test_idx[k] for k in range(num_clients)})

    config = {
        "dataset": "PathMNIST", "size": 28,
        "num_clients": num_clients, "num_classes": num_classes,
        "partition": "dir", "alpha": alpha, "split_seed": split_seed,
        "normalization": NORMALIZATION,
        "test_allocation": "per-class, proportional to client's share of that class in train",
        "official_split_sizes": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
        "validation": "official val split kept separate (not partitioned, not used in training)",
        "train_samples_per_client": [int(len(ix)) for ix in train_idx],
        "test_samples_per_client": [int(len(ix)) for ix in test_idx],
        "train_class_counts": train_counts.tolist(),
        "test_class_counts": test_counts.tolist(),
    }
    with open(dir_path + "config.json", "w") as f:
        ujson.dump(config, f, indent=2)

    for k in range(num_clients):
        print(f"client {k}: train {len(train_idx[k]):6d}  test {len(test_idx[k]):5d}  "
              f"train classes {train_counts[k].tolist()}")
    save_heatmap(train_counts, test_counts)


if __name__ == "__main__":
    generate()
