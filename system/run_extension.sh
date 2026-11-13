#!/bin/bash
# Post-selection exploratory extension (frozen 2026-11-12, see ../PROTOCOL.md, Amendment 1):
# clean baselines for the validation-selected FedProx mu=0.1, seeds 0-2, same settings, 200 rounds.
# Its results do not change mu, the round budget, or the primary endpoint.
# Run only after Phase 3 has finished:   caffeinate -i ./run_extension.sh
set -e
set -o pipefail
cd "$(dirname "$0")"
GR=200
LR=0.01
COMMON="-data PathMNIST -ncl 9 -m CNN -gr $GR -nc 5 -jr 1 -ls 1 -lr $LR -lbs 32 -dev cpu --noisy_client 0"
mkdir -p ../logs ../outputs

# Guard: every Phase 3 run (BEST_MU=0.1) must be finished.
missing=0
for S in 1 2; do
  for n in fedavg_seed$S fedprox_mu0.01_seed$S fedprox_mu0.1_seed$S; do
    ls -d ../outputs/${n}_????????/final_model.pt >/dev/null 2>&1 || { echo "Phase 3 not finished: $n"; missing=1; }
  done
done
for S in 0 1 2; do
  for n in fedavg_seed${S}_clean fedprox_mu0.01_seed${S}_clean; do
    ls -d ../outputs/${n}_????????/final_model.pt >/dev/null 2>&1 || { echo "Phase 3 not finished: $n"; missing=1; }
  done
done
[ $missing -eq 0 ] || { echo "Finish Phase 3 first (BEST_MU=0.1 ./run_main.sh 3)."; exit 1; }

run() {  # $1 = expected run name, rest = args. Skips runs that already finished.
  local name=$1; shift
  if ls -d ../outputs/${name}_????????/final_model.pt >/dev/null 2>&1; then
    echo ">>> skip $name (already done)"; return
  fi
  echo ">>> START $name  $(date '+%H:%M:%S')"
  python -u main.py $COMMON "$@" 2>&1 | tee ../logs/${name}.log
}

for SEED in 0 1 2; do
  run fedprox_mu0.1_seed${SEED}_clean -algo FedProx -mu 0.1 --seed $SEED --noise_rate 0 --run_tag clean
done
echo ">>> EXTENSION DONE  $(date '+%H:%M:%S')"
