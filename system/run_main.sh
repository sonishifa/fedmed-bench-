#!/bin/bash
# Experiment driver. Partition fixed (split_seed=0); --seed controls init, data order and label noise.
# Frozen protocol: shared lr=0.01, 1 local epoch, GR=200 (extend once to 300 only by the declared
# validation rule; whatever budget results applies to ALL runs).
#
# Usage:
#   ./run_main.sh check              3-round pairing / implementation checks (then: python check_pairing.py)
#   ./run_main.sh 1                  seed-0 pilot = seed-0 final runs (FedAvg, FedProx mu=0.01)
#   ./run_main.sh 2                  mu sweep on seed 0 (0.001, 0.1, 1.0)
#   BEST_MU=0.1 ./run_main.sh 3      seeds 1,2 (+ selected mu if != 0.01) and clean baselines for seeds 0,1,2
#   GR=100 ./run_main.sh 0           (exploration only) lr sweep, already done
set -e
set -o pipefail
cd "$(dirname "$0")"
GR=${GR:-200}
LR=${LR:-0.01}
COMMON="-data PathMNIST -ncl 9 -m CNN -gr $GR -nc 5 -jr 1 -ls 1 -lr $LR -lbs 32 -dev cpu --noisy_client 0"
mkdir -p ../logs ../outputs

run() {  # $1 = expected run name, rest = args. Skips runs that already finished.
  local name=$1; shift
  if ls -d ../outputs/${name}_????????/final_model.pt >/dev/null 2>&1; then
    echo ">>> skip $name (already done)"; return
  fi
  echo ">>> START $name  $(date '+%H:%M:%S')"
  python -u main.py $COMMON "$@" 2>&1 | tee ../logs/${name}.log
}

case "$1" in
  check)  # FedAvg vs FedProx mu=0 must match; noisy vs clean must share RNG/batch order
      run fedavg_seed0_check3          -algo FedAvg        --seed 0 --noise_rate 0.4 -gr 3 --run_tag check3
      run fedprox_mu0.0_seed0_check3   -algo FedProx -mu 0 --seed 0 --noise_rate 0.4 -gr 3 --run_tag check3
      run fedavg_seed0_clean_check3    -algo FedAvg        --seed 0 --noise_rate 0   -gr 3 --run_tag clean_check3 ;;
  0)  # exploration: lr/rounds selection, FedAvg, seed 0, noisy
      for L in 0.01 0.03 0.1; do
        run fedavg_seed0_lr${L}_gr${GR} -algo FedAvg --seed 0 --noise_rate 0.4 -lr $L --run_tag lr${L}_gr${GR}
      done ;;
  1)  # seed 0 (development seed): pilot for the budget rule, also the seed-0 final runs
      run fedavg_seed0          -algo FedAvg           --seed 0 --noise_rate 0.4
      run fedprox_mu0.01_seed0  -algo FedProx -mu 0.01 --seed 0 --noise_rate 0.4 ;;
  2)  # mu sweep on seed 0; select by max mean val acc over last 10 rounds, ties -> smaller mu
      for MU in 0.001 0.1 1.0; do
        run fedprox_mu${MU}_seed0 -algo FedProx -mu $MU --seed 0 --noise_rate 0.4
      done ;;
  3)  # remaining seeds (noisy) + clean baselines (FedAvg, FedProx 0.01) for all seeds
      for SEED in 1 2; do
        run fedavg_seed${SEED}          -algo FedAvg           --seed $SEED --noise_rate 0.4
        run fedprox_mu0.01_seed${SEED}  -algo FedProx -mu 0.01 --seed $SEED --noise_rate 0.4
        if [ -n "$BEST_MU" ] && [ "$BEST_MU" != "0.01" ]; then
          run fedprox_mu${BEST_MU}_seed${SEED} -algo FedProx -mu $BEST_MU --seed $SEED --noise_rate 0.4
        fi
      done
      for SEED in 0 1 2; do
        run fedavg_seed${SEED}_clean          -algo FedAvg           --seed $SEED --noise_rate 0 --run_tag clean
        run fedprox_mu0.01_seed${SEED}_clean  -algo FedProx -mu 0.01 --seed $SEED --noise_rate 0 --run_tag clean
      done ;;
  *)  echo "usage: ./run_main.sh check|0|1|2|3"; exit 1 ;;
esac
echo ">>> PHASE $1 DONE  $(date '+%H:%M:%S')"
