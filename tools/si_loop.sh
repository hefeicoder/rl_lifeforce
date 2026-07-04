#!/bin/zsh
# Interleaved self-imitation loop: alternate short PPO chunks with gentle BC
# refreshes so a low-tolerance corridor can't erode during training.
# (Escalation rung above pooled-ensemble BC — see RESUME.md / progress doc.)
#
# Usage: ./tools/si_loop.sh <start_ckpt> <run_prefix> <cycles> <chunk_steps>
# Example: ./tools/si_loop.sh checkpoints/l3-bc6/lifeforce_ppo_bc20.zip l3-si 8 50000
set -e
START=$1; PREFIX=$2; CYCLES=${3:-8}; CHUNK=${4:-50000}
DEMOS=(demos/l3_b1736_fine3.npz demos/l3_b1736_v2_bd120.npz
       demos/l3_b1736_v3_bd40.npz demos/l3_b1736_v4_seed1.npz)
CUR=$START
for i in $(seq 1 $CYCLES); do
  RUN="${PREFIX}${i}"
  echo "=== cycle $i/$CYCLES: PPO ${CHUNK} steps from ${CUR} -> ${RUN}"
  .venv/bin/python -u -m src.train --resume "$CUR" --run-name "$RUN" \
    --timesteps "$CHUNK" --ent-coef 0.03 \
    --curriculum-glob 'states/l3_b1736_curriculum/*.state' --curriculum-mix 0.3 \
    --save-freq "$CHUNK"
  TRAINED="checkpoints/${RUN}/lifeforce_ppo_final.zip"
  BC_OUT="checkpoints/${RUN}/lifeforce_ppo_bcref.zip"
  echo "=== cycle $i/$CYCLES: BC refresh (3 epochs, lr 1e-4) -> ${BC_OUT}"
  .venv/bin/python -u -m tools.self_imitation --model "$TRAINED" \
    --demos "${DEMOS[@]}" --out "$BC_OUT" --epochs 3 --lr 1e-4
  CUR=$BC_OUT
done
echo "=== si_loop done; last checkpoint: $CUR"
