"""Figures and tables for the README.

Reads every finished run in ../outputs/<name>_<wandb id>/metrics.jsonl (201 rows), plus the lr-sweep runs
in ../outputs/exploration, and writes
  ../plots/fig*.png              figures
  ../plots/results_tables.md     all tables (markdown, pasted into the README)
  ../plots/results_summary.json  the numbers behind the tables
It prints a few sanity checks the write-up depends on (early-round cosine, class overlap client 0 vs 1).

Usage (from system/):  python plot_results.py
"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

OUT = "../plots"
os.makedirs(OUT, exist_ok=True)
GR, NC = 200, 5
ROUNDS = np.arange(GR + 1)

# ---------------------------------------------------------------- style
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
METHODS = [("FedAvg", BLUE), ("FedProx μ=0.01", ORANGE), ("FedProx μ=0.1", AQUA)]
CLIENT_COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
SWEEP_COLORS = {"0.001": "#9ec5f4", "0.01": "#5598e7", "0.1": "#1c5cab", "1.0": "#0d366b"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "normal", "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "font.size": 10,
    "text.color": INK, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def save(fig, name):
    path = f"{OUT}/{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- load runs
PAT = re.compile(r"^(fedavg|fedprox)(?:_mu([0-9.]+))?_seed(\d+)(_clean)?_[a-z0-9]{8}$")


def load_rows(d):
    rows = [json.loads(l) for l in open(f"{d}/metrics.jsonl") if l.strip()]
    return rows if len(rows) == GR + 1 and os.path.exists(f"{d}/final_model.pt") else None


runs = {}  # (method, clean) -> {seed: rows}
for d in sorted(glob.glob("../outputs/*_????????")):
    m = PAT.match(os.path.basename(d))
    rows = load_rows(d) if m else None
    if rows is None:
        continue
    algo, mu, seed, clean = m.groups()
    method = "FedAvg" if algo == "fedavg" else f"FedProx μ={mu}"
    runs.setdefault((method, bool(clean)), {})[int(seed)] = rows
for k, v in sorted(runs.items()):
    print(f"loaded {k[0]:16s} {'clean' if k[1] else 'noisy'}  seeds {sorted(v)}")
for name, _ in METHODS:
    assert sorted(runs[(name, False)]) == [0, 1, 2], f"{name}: noisy seeds missing"
    assert sorted(runs[(name, True)]) == [0, 1, 2], f"{name}: clean seeds missing"


def stack(method, key, clean=False):
    """(n_seeds, 201) array, seeds in ascending order."""
    g = runs[(method, clean)]
    # update stats do not exist at round 0 (no training yet) -> NaN, which matplotlib skips
    return np.array([[r.get(key, np.nan) for r in g[s]] for s in sorted(g)], dtype=float)


def mean_sd(a, axis=0):
    return a.mean(axis), (a.std(axis, ddof=1) if a.shape[axis] > 1 else np.zeros(a.shape[1 - axis] if a.ndim > 1 else ()))


def pm(m, s, scale=100, nd=1):
    return f"{scale * m:.{nd}f} ± {scale * s:.{nd}f}"


def band(ax, a, color, label, ls="-"):
    m, s = mean_sd(a)
    ax.plot(ROUNDS, m, color=color, label=label, ls=ls)
    ax.fill_between(ROUNDS, m - s, m + s, color=color, alpha=0.18, lw=0)


def legend_right(ax, handles=None, labels=None, **kw):
    if handles is None:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), **kw)
    else:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), **kw)


ds = json.load(open("../dataset/PathMNIST/config.json"))
train_counts = np.array(ds["train_class_counts"])
test_counts = np.array(ds["test_class_counts"])
n_train, n_test = train_counts.sum(1), test_counts.sum(1)

# ================================================================ fig0: heatmap
cmap = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
fig, ax = plt.subplots(figsize=(8.2, 4.2))
ax.grid(False)
im = ax.imshow(train_counts, cmap=cmap, aspect="auto")
for i in range(NC):
    for j in range(train_counts.shape[1]):
        v = train_counts[i, j]
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8.5,
                color="#ffffff" if v > train_counts.max() * 0.45 else INK)
ax.set_xticks(range(9)); ax.set_xlabel("class")
ax.set_yticks(range(NC)); ax.set_yticklabels([f"client {i}{'  (40% label noise)' if i == 0 else ''}" for i in range(NC)])
ax.set_title("Training samples per client and class (Dirichlet α=0.3, split seed 0)")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(color=MUTED, labelcolor=MUTED)
save(fig, "fig0_client_class_heatmap")

# ================================================================ fig1: global test accuracy
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1]})
for name, c in METHODS:
    band(axes[0], stack(name, "global_test_acc"), c, name)
    band(axes[1], stack(name, "global_test_acc"), c, name)
axes[0].set(xlabel="round", ylabel="global test accuracy", xlim=(0, GR), ylim=(0, 1), title="all rounds")
axes[1].set(xlabel="round", xlim=(100, GR), ylim=(0.55, 0.75), title="rounds 100–200 (zoomed y-axis)")
legend_right(axes[1])
fig.suptitle("Global test accuracy (sample-weighted), mean ± sd over 3 seeds", y=1.02)
save(fig, "fig1_global_test_acc")

# ================================================================ fig2: per-client test accuracy vs round
fig, axes = plt.subplots(1, NC, figsize=(15, 3.4), sharey=True)
for i, ax in enumerate(axes):
    for name, c in METHODS:
        band(ax, stack(name, f"client_{i}_test_acc"), c, name)
    ax.set(xlim=(0, GR), ylim=(0, 1), title=f"client {i}" + ("  (noisy labels)" if i == 0 else ""), xlabel="round")
axes[0].set_ylabel("test accuracy (clean labels)")
legend_right(axes[-1])
fig.suptitle("Per-client test accuracy of the global model, mean ± sd over 3 seeds", y=1.03)
save(fig, "fig2_per_client_test_acc")

# ================================================================ fig3: global losses
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for name, c in METHODS:
    band(axes[0], stack(name, "global_test_loss"), c, name)
    band(axes[1], stack(name, "global_train_loss"), c, name)
axes[0].set(title="Global test loss (CE, clean labels)", xlabel="round", ylabel="loss", xlim=(0, GR))
axes[1].set(title="Global train loss (CE on training labels; noisy for client 0)", xlabel="round", xlim=(0, GR))
legend_right(axes[1])
save(fig, "fig3_global_loss")

# ================================================================ fig4: per-client losses
fig, axes = plt.subplots(2, NC, figsize=(15, 6), sharex=True)
for i in range(NC):
    for name, c in METHODS:
        band(axes[0, i], stack(name, f"client_{i}_test_loss"), c, name)
        band(axes[1, i], stack(name, f"client_{i}_train_loss"), c, name)
    axes[0, i].set_title(f"client {i}" + ("  (noisy labels)" if i == 0 else ""))
    axes[1, i].set_xlabel("round"); axes[1, i].set_xlim(0, GR)
axes[0, 0].set_ylabel("test loss (clean labels)")
axes[1, 0].set_ylabel("train loss (on training labels)")
legend_right(axes[0, -1])
fig.suptitle("Per-client losses, mean ± sd over 3 seeds. Client 0's train loss is measured on its 40%-corrupted labels.", y=1.0)
save(fig, "fig4_per_client_loss")

# ================================================================ fig5: update norm and LOO cosine per client
fig, axes = plt.subplots(2, 3, figsize=(15, 6.4), sharex=True, sharey="row")
for j, (name, _) in enumerate(METHODS):
    for i in range(NC):
        band(axes[0, j], stack(name, f"client_{i}_update_norm"), CLIENT_COLORS[i], f"client {i}")
        band(axes[1, j], stack(name, f"client_{i}_cosine_loo"), CLIENT_COLORS[i], f"client {i}")
    axes[0, j].set_title(name); axes[1, j].set_xlabel("round"); axes[1, j].set_xlim(0, GR)
    axes[1, j].axhline(0, color=MUTED, lw=0.8)
axes[0, 0].set_ylabel("update norm  ‖w_k − w_global‖")
axes[1, 0].set_ylabel("leave-one-out cosine")
h = [Line2D([], [], color=CLIENT_COLORS[i], lw=2) for i in range(NC)]
legend_right(axes[0, -1], h, [f"client {i}" + (" (noisy)" if i == 0 else "") for i in range(NC)])
fig.suptitle("Client update norm and cosine to the weighted mean of the OTHER clients' updates, mean ± sd over 3 seeds", y=1.0)
save(fig, "fig5_update_norm_cosine")

# ================================================================ fig6: final per-client accuracy bars
fig, ax = plt.subplots(figsize=(9, 4.2))
w, gap = 0.26, 0.02
x = np.arange(NC)
for j, (name, c) in enumerate(METHODS):
    finals = np.array([[runs[(name, False)][s][-1][f"client_{i}_test_acc"] for i in range(NC)] for s in (0, 1, 2)])
    m, sd = mean_sd(finals)
    pos = x + (j - 1) * (w + gap)
    bars = ax.bar(pos, m, w, color=c, label=name, yerr=sd, capsize=2, error_kw={"ecolor": INK2, "lw": 1})
    bars[0].set_hatch("//"); bars[0].set_edgecolor("#ffffff")
    for p, v in zip(pos, m):
        ax.text(p, v + 0.03, f"{100 * v:.0f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([f"client {i}\n(n_test={n_test[i]})" for i in range(NC)])
ax.set(ylabel="final-round test accuracy", ylim=(0, 1), title="Final-round (r200) per-client test accuracy, mean ± sd over 3 seeds")
ax.text(-0.42, 0.93, "client 0 (hatched): 40% of its training labels corrupted;\nits test labels are clean", fontsize=8.5, color=INK2, va="top")
legend_right(ax)
save(fig, "fig6_final_per_client_bars")

# ================================================================ fig7: client 0 noisy vs clean
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for name, c in METHODS:
    for clean, ls in ((False, "-"), (True, "--")):
        band(axes[0], stack(name, "client_0_update_norm", clean), c, None, ls)
        band(axes[1], stack(name, "client_0_cosine_loo", clean), c, None, ls)
        band(axes[2], stack(name, "client_0_train_loss", clean), c, None, ls)
axes[0].set(title="client 0 update norm", xlabel="round", xlim=(0, GR))
axes[1].set(title="client 0 leave-one-out cosine", xlabel="round", xlim=(0, GR)); axes[1].axhline(0, color=MUTED, lw=0.8)
axes[2].set(title="client 0 train loss (on the labels it trains on)", xlabel="round", xlim=(0, GR))
h = [Line2D([], [], color=c, lw=2) for _, c in METHODS] + [Line2D([], [], color=INK2, lw=2, ls="-"), Line2D([], [], color=INK2, lw=2, ls="--")]
legend_right(axes[2], h, [n for n, _ in METHODS] + ["noisy run (40% noise)", "paired clean run"])
fig.suptitle("Client 0 in noisy vs paired clean runs (same init, batch order and seed), mean ± sd over 3 seeds", y=1.02)
save(fig, "fig7_client0_noisy_vs_clean")

# ================================================================ fig8: corrupted-subset diagnostic
def excess(method, clean):
    pn = stack(method, "client_0_corrupted_pred_noisy", clean)
    po = stack(method, "client_0_corrupted_pred_original", clean)
    return pn - (1 - po) / 8, pn, po


fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for name, c in METHODS:
    for clean, ls in ((False, "-"), (True, "--")):
        ex, pn, po = excess(name, clean)
        band(axes[0], pn, c, None, ls); band(axes[1], po, c, None, ls); band(axes[2], ex, c, None, ls)
axes[0].set(title="pred = injected (wrong) label", xlabel="round", xlim=(0, GR), ylabel="fraction of the 9,626 corrupted samples")
axes[1].set(title="pred = original (true) label", xlabel="round", xlim=(0, GR))
axes[2].set(title="excess agreement = pred_noisy − (1 − pred_original)/8", xlabel="round", xlim=(0, GR)); axes[2].axhline(0, color=MUTED, lw=0.8)
legend_right(axes[2], h, [n for n, _ in METHODS] + ["noisy run (labels applied)", "clean run (same targets, not applied)"])
fig.suptitle("Global-model predictions on client 0's corruption subset, mean ± sd over 3 seeds", y=1.02)
save(fig, "fig8_corrupted_subset")

# ================================================================ fig9: selection curves (validation only)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for L, c in (("0.01", "#5598e7"), ("0.03", "#1c5cab"), ("0.1", "#0d366b")):
    hits = glob.glob(f"../outputs/exploration/fedavg_seed0_lr{L}_gr100_????????")
    if hits:
        rows = [json.loads(l) for l in open(f"{hits[0]}/metrics.jsonl") if l.strip()]
        axes[0].plot([r["round"] for r in rows], [r["global_val_acc"] for r in rows], color=c, label=f"lr={L}")
axes[0].set(title="lr sweep (FedAvg, seed 0, 100 rounds, exploration runs)", xlabel="round", ylabel="validation accuracy", ylim=(0, 1))
axes[0].legend(loc="lower left", bbox_to_anchor=(0.14, 0.0))
axes[1].plot(ROUNDS, stack("FedAvg", "global_val_acc")[0], color=ORANGE, lw=1.5, label="FedAvg (reference)", zorder=5)
for mu in ("0.001", "0.01", "0.1", "1.0"):
    g = runs.get((f"FedProx μ={mu}", False), {})
    if 0 in g:
        axes[1].plot(ROUNDS, [r["global_val_acc"] for r in g[0]], color=SWEEP_COLORS[mu], label=f"FedProx μ={mu}")
axes[1].set(title="μ sweep (seed 0, 200 rounds)", xlabel="round", ylim=(0, 1), xlim=(0, GR))
axes[1].axvspan(191, 200, color=GRID, alpha=0.6, lw=0)
axes[1].text(190, 0.03, "selection window r191–200", ha="right", fontsize=8, color=INK2)
legend_right(axes[1])
save(fig, "fig9_selection_curves_val")

# ================================================================ tables
summary = {}
L = ["<!-- generated by system/plot_results.py -->", ""]


def best_val_round(rows):
    v = np.array([r["global_val_acc"] for r in rows])
    return int(np.argmax(v))  # earliest on ties


# ---- Table 1: primary endpoint
L += ["### Table 1. Global test accuracy (%), noisy setting, 200 rounds, 3 seeds", "",
      "| Method | seed 0 | seed 1 | seed 2 | final (r200) mean ± sd | last-10 mean ± sd | at best-val round, mean ± sd | Δ vs FedAvg per seed (final) | Δ mean ± sd |",
      "|---|---|---|---|---|---|---|---|---|"]
fin = {}
for name, _ in METHODS:
    acc = stack(name, "global_test_acc")
    val_rounds = [best_val_round(runs[(name, False)][s]) for s in (0, 1, 2)]
    fin[name] = {"final": acc[:, -1], "last10": acc[:, -10:].mean(1),
                 "bestval": np.array([acc[k, r] for k, r in enumerate(val_rounds)]), "bestval_round": val_rounds}
for name, _ in METHODS:
    f = fin[name]
    d = f["final"] - fin["FedAvg"]["final"]
    dcell = " / ".join(f"{100 * x:+.1f}" for x in d) if name != "FedAvg" else "—"
    dms = f"{100 * d.mean():+.1f} ± {100 * d.std(ddof=1):.1f}" if name != "FedAvg" else "—"
    L.append(f"| {name} | " + " | ".join(f"{100 * x:.1f}" for x in f["final"]) +
             f" | **{pm(*mean_sd(f['final']))}** | {pm(*mean_sd(f['last10']))} | {pm(*mean_sd(f['bestval']))} (r {', '.join(map(str, f['bestval_round']))}) | {dcell} | {dms} |")
    summary[name] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.items()}
L += ["", "sd = sample standard deviation over seeds (n=3). Seed 0 is the development seed (lr and μ were selected on its validation accuracy). "
      "Δ = paired per-seed difference (same seed → same init, batch order and noise subset).", ""]

# ---- Table 1b: seeds 1-2 only
L += ["### Table 1b. Evaluation seeds only (seeds 1–2, not used for any selection), final-round global test accuracy (%)", "",
      "| Method | seed 1 | seed 2 | mean |", "|---|---|---|---|"]
for name, _ in METHODS:
    f = fin[name]["final"][1:]
    L.append(f"| {name} | {100 * f[0]:.1f} | {100 * f[1]:.1f} | {100 * f.mean():.1f} |")
L.append("")

# ---- Table 2: per-client
L += ["### Table 2. Final-round per-client test accuracy (%), mean ± sd over 3 seeds", "",
      "| Client | n_train | n_test | " + " | ".join(n for n, _ in METHODS) + " |", "|---|---|---|" + "---|" * len(METHODS)]
summary["per_client"] = {}
for i in range(NC):
    cells = []
    for name, _ in METHODS:
        a = np.array([runs[(name, False)][s][-1][f"client_{i}_test_acc"] for s in (0, 1, 2)])
        cells.append(pm(*mean_sd(a)))
        summary["per_client"][f"{name}|client_{i}"] = a.tolist()
    L.append(f"| {i}{' (40% label noise)' if i == 0 else ''} | {n_train[i]:,} | {n_test[i]:,} | " + " | ".join(cells) + " |")
L += ["", "Per-client test accuracy = accuracy of the *global* model on that client's (clean) test examples. "
      "Global accuracy is the sample-weighted average over clients.", ""]

# ---- Table 2b + fig10: per-client noise penalty (clean - noisy), paired by seed
L += ["### Table 2b. Per-client effect of the noise: final test accuracy (%), clean / noisy / Δ (noisy − clean), mean over 3 seeds", "",
      "| Client | " + " | ".join(f"{n} clean / noisy / Δ" for n, _ in METHODS) + " |", "|---|" + "---|" * len(METHODS)]
summary["per_client_penalty"] = {}
pen = {}
for name, _ in METHODS:
    cl = np.array([[runs[(name, True)][s][-1][f"client_{i}_test_acc"] for i in range(NC)] for s in (0, 1, 2)])
    no = np.array([[runs[(name, False)][s][-1][f"client_{i}_test_acc"] for i in range(NC)] for s in (0, 1, 2)])
    pen[name] = (cl, no, no - cl)
    summary["per_client_penalty"][name] = {"clean": cl.tolist(), "noisy": no.tolist()}
for i in range(NC):
    cells = [f"{100 * pen[n][0][:, i].mean():.1f} / {100 * pen[n][1][:, i].mean():.1f} / **{100 * pen[n][2][:, i].mean():+.1f}**" for n, _ in METHODS]
    L.append(f"| {i}{' (noisy)' if i == 0 else ''} | " + " | ".join(cells) + " |")
L += ["", "Clean and noisy runs with the same seed share initialisation and batch order; only client 0's labels differ.", ""]

fig, ax = plt.subplots(figsize=(9, 3.8))
for j, (name, c) in enumerate(METHODS):
    m, sd = mean_sd(pen[name][2])
    pos = x + (j - 1) * (w + gap)
    bars = ax.bar(pos, m, w, color=c, label=name, yerr=sd, capsize=2, error_kw={"ecolor": INK2, "lw": 1})
    bars[0].set_hatch("//"); bars[0].set_edgecolor("#ffffff")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"client {i}" + ("\n(noisy)" if i == 0 else "") for i in range(NC)])
ax.set(ylabel="noisy − clean final test acc", title="Per-client effect of client 0's label noise (paired by seed, mean ± sd over 3 seeds)")
legend_right(ax)
save(fig, "fig10_per_client_noise_penalty")

# ---- Table 3: val vs test, late decline
L += ["### Table 3. Validation vs test (noisy setting, mean over 3 seeds)", "",
      "| Method | best val acc (%) | best-val round (per seed) | test acc at best-val round (%) | final test acc (%) | final val acc (%) |", "|---|---|---|---|---|---|"]
for name, _ in METHODS:
    val = stack(name, "global_val_acc")
    bv = [best_val_round(runs[(name, False)][s]) for s in (0, 1, 2)]
    L.append(f"| {name} | {100 * np.mean([val[k, r] for k, r in enumerate(bv)]):.1f} | {', '.join(map(str, bv))} | "
             f"{100 * fin[name]['bestval'].mean():.1f} | {100 * fin[name]['final'].mean():.1f} | {100 * val[:, -1].mean():.1f} |")
L += ["", "Validation = official PathMNIST val split (10,004 images, same source center as train), held by the server and used only for "
      "lr/μ/budget selection. Test = official test split (7,180 images, a different source center), partitioned across clients.", ""]

# ---- Table 4: noise penalty (extension)
L += ["### Table 4. Noise penalty D = final test acc(clean run) − final test acc(noisy run), percentage points, paired by seed", "",
      "| Method | clean seed 0/1/2 | noisy seed 0/1/2 | D seed 0/1/2 | D mean ± sd | clean mean ± sd |", "|---|---|---|---|---|---|"]
summary["penalty"] = {}
for name, _ in METHODS:
    cl = stack(name, "global_test_acc", True)[:, -1]
    no = fin[name]["final"]
    D = cl - no
    summary["penalty"][name] = {"clean": cl.tolist(), "noisy": no.tolist(), "D": D.tolist()}
    L.append(f"| {name} | {'/'.join(f'{100 * x:.1f}' for x in cl)} | {'/'.join(f'{100 * x:.1f}' for x in no)} | "
             f"{'/'.join(f'{100 * x:+.1f}' for x in D)} | {100 * D.mean():+.1f} ± {100 * D.std(ddof=1):.1f} | {pm(*mean_sd(cl))} |")
L += ["", "Clean runs use noise_rate=0 with the same seed, so they share the noisy run's initialization and batch order; "
      "the clean FedProx μ=0.1 runs were added after μ was selected (see PROTOCOL.md, Amendment 1).", ""]

# ---- Table 5: memorization excess at r200
L += ["### Table 5. Agreement of the final global model with the injected labels on client 0's corruption subset (r200, mean ± sd over 3 seeds)", "",
      "| Method | run | pred = injected label | pred = original label | excess agreement |", "|---|---|---|---|---|"]
summary["excess"] = {}
for name, _ in METHODS:
    for clean in (False, True):
        ex, pn, po = excess(name, clean)
        summary["excess"][f"{name}|{'clean' if clean else 'noisy'}"] = {"pred_noisy": pn[:, -1].tolist(), "pred_original": po[:, -1].tolist(), "excess": ex[:, -1].tolist()}
        L.append(f"| {name} | {'clean (targets not applied)' if clean else 'noisy'} | {pn[:, -1].mean():.3f} ± {pn[:, -1].std(ddof=1):.3f} | "
                 f"{po[:, -1].mean():.3f} ± {po[:, -1].std(ddof=1):.3f} | {ex[:, -1].mean():+.3f} ± {ex[:, -1].std(ddof=1):.3f} |")
L += ["", "excess = pred_noisy − (1 − pred_original)/8: agreement with the injected label beyond what wrong predictions would produce by chance "
      "if they were independent of the injected label. The clean rows are the zero line: the clean model never saw the injected labels.", ""]

# ---- Table 6: selection sweeps (val only)
L += ["### Table 6. Selection sweeps (seed 0, validation accuracy only)", "",
      "| Run | mean val acc r191–200 (%) | best val acc (%) @ round |", "|---|---|---|"]
for name in ["FedAvg"] + [f"FedProx μ={mu}" for mu in ("0.001", "0.01", "0.1", "1.0")]:
    rows = runs[(name, False)][0]
    v = np.array([r["global_val_acc"] for r in rows])
    L.append(f"| {name} | {100 * v[-10:].mean():.2f} | {100 * v.max():.2f} @ r{int(np.argmax(v))} |")
L += ["", "Rule: highest mean validation accuracy over r191–200, ties → smaller μ. Selected μ = 0.1 ('validation-selected', not 'optimal').", ""]

open(f"{OUT}/results_tables.md", "w").write("\n".join(L))
json.dump(summary, open(f"{OUT}/results_summary.json", "w"), indent=1)
print("wrote", f"{OUT}/results_tables.md", "and results_summary.json")

# ---------------------------------------------------------------- sanity checks for the write-up
print("\n--- checks for the write-up ---")
r0 = runs[("FedAvg", False)][0]
for rr in (1, 2, 3, 5, 10, 50, 200):
    print(f"FedAvg seed0 r{rr:3d} cosine_loo per client:", [round(r0[rr][f"client_{i}_cosine_loo"], 3) for i in range(NC)])
shared = [(j, int(train_counts[0, j]), int(train_counts[1, j])) for j in range(9) if train_counts[0, j] > 200 and train_counts[1, j] > 200]
print("classes with >200 train samples on BOTH client 0 and client 1 (class, n_c0, n_c1):", shared)
print("\n".join(L[2:12]))
