# Experimental protocol (PathMNIST, FedAvg / FedProx, one noisy client)

This file records decisions in the order they were made. Sections are dated, and each section was
written **before** the results it governs were seen. Earlier sections are not edited afterwards;
changes appear as dated amendments.

## Frozen protocol (2026-11-11, before Phase 1)

- **Data:** PathMNIST, 5 clients, Dirichlet α=0.3 over the official train split. The partition is fixed (split_seed=0).
  - The official test split is allocated per class to clients in proportion to each client's train share.
  - The official val split is held by the server and used only for selection decisions.
- **Noise:** client 0 gets 40% symmetric, always-different label noise, y → (y + U{1..8}) mod 9, seeded by `--seed`.
- **Training:** FedAvgCNN, plain SGD, lr=0.01, batch size 32, 1 local epoch, all 5 clients participate every round.
- **Budget:** 200 rounds. Extend all runs to 300 only if mean val(r181–200) − mean val(r161–180) > 0.01 for either
  algorithm in Phase 1. Whatever budget results applies to all runs.
- **μ selection:** pool {0.001, 0.01, 0.1, 1.0}, using seed 0. Choose the highest mean val accuracy over the last
  10 rounds; ties go to the smaller μ. μ=0.01 is kept for all seeds because the assignment names it.
- **Runs:** seeds 0–2 for FedAvg, FedProx 0.01 and the selected μ. Clean baselines (noise_rate 0, with the same
  corruption subset drawn as hypothetical targets) for FedAvg and FedProx 0.01, seeds 0–2.
- **Primary endpoint:** final-round sample-weighted global test accuracy, reported as mean ± sample std over 3 seeds,
  with per-seed values and paired per-seed differences.
  - Secondary endpoints: last-10-round mean, and test accuracy at the best-val round (earliest round on ties).
  - Seed 0 is the development seed and is disclosed as such.

## Outcomes of the frozen rules

- **Phase 1 (2026-11-12):** late-window val gains were FedAvg −0.0014 and FedProx 0.01 −0.0041, so the budget stays
  at **200 rounds**.
- **Phase 2:** last-10 mean val was 0.7435 for μ=0.001, 0.7538 for μ=0.01, 0.7621 for μ=0.1 and 0.6116 for μ=1.0,
  so **μ = 0.1 is selected**. It is called the "validation-selected μ", not the "optimal μ".
- **Disclosure:** while investigating a Phase 1 continuity question, a few test accuracies were printed incidentally:
  Phase 1 FedAvg at r50/r100, and the exploration runs. No decision used them.

## Amendment 1 (2026-11-12, after Phase 2, before Phase 3 results): post-selection exploratory extension

This is a post-selection exploratory analysis, added after seeing Phase 2. Its results do **not** change μ, the
round budget, or the primary endpoint. All of its outcomes will be reported, and they are kept separate from the
confirmatory comparison.

- **Extra runs:** clean FedProx μ=0.1, seeds 0–2. Settings are identical to the other runs (200 rounds), and the
  runs are made with `system/run_extension.sh` after Phase 3 has finished.
- **Primary noise penalty:** for each method m and seed s,
  `D_m,s = final test acc(clean) − final test acc(noisy)`.
  - Compare D across FedAvg, FedProx 0.01 and FedProx 0.1 per seed, always alongside the underlying clean and noisy
    accuracies.
  - A smaller penalty caused by poor clean accuracy is not evidence of robustness.
- **Diagnostics,** computed post hoc from saved checkpoints with no change to training:
  - Client 0's corrupted subset: accuracy and CE against the original labels and against the injected labels.
  - Clean models are evaluated against the same saved hypothetical injected targets.
  - Excess agreement = pred_noisy − (1 − pred_original)/8.
  - Client 0's uncorrupted subset, and clients 1–4: true-label accuracy and CE.
  - Validation: confusion matrix, per-class accuracy and CE.
  - Corrupted vs uncorrupted comparisons are also shown class-wise, because the class mixtures differ.
- **Checkpoints:** final checkpoints are primary; best-val checkpoints are secondary.
- **Reporting:** seeds 1–2 are also shown separately as evaluation seeds that were not used for selection. The FedAvg
  late-window reference uses the same r191–200 window as μ selection.
- **Wording:**
  - Reproducibility: "Reproducibility checks were consistent with matched initialization and batch ordering." No
    claim of proof is made unless direct parameter and batch-index hashes are added.
  - μ=1.0: "lower accuracy within the 200-round budget", not "under-fit".

## Exploration record (written 2026-11-14 from the exploration runs; no decision changed)

lr sweep, FedAvg, seed 0, 100 rounds, validation accuracy (`outputs/exploration/`, W&B `*_explore`):
lr 0.1 collapsed at round ~60; lr 0.03 peaked at round ~58 and then declined; lr 0.01 was still rising at round 100
and was chosen together with the 200-round budget and the budget rule above. These runs predate the RNG fixes
(evaluation loaders and wandb's use of global `random` changed the client order), so they are not exactly
reproducible and are kept for transparency only.

## Deviation note (2026-11-14, after all runs)

The checkpoint diagnostics listed in Amendment 1 (client 0's uncorrupted subset, clients 1–4 true-label CE,
validation confusion matrix and per-class accuracy, class-wise comparisons) were not carried out, for time. Nothing
was computed and discarded. What is reported from Amendment 1: the noise penalty D (README Table 4), the
corrupted-subset metrics logged during training (Table 5, Fig. 8), and the per-client clean-vs-noisy comparison
(Table 2b, Fig. 10), which was not pre-registered.
