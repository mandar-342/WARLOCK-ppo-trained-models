#!/bin/bash
# Quick launch script for Optuna hyperparameter optimization
# Usage: ./launch_optuna.sh [quick|full|resume|analyze]

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-quick}"

case "$MODE" in
    quick)
        echo "=== QUICK SWEEP: 10 trials, 3 seeds, 50k steps ==="
        python -m src.agent.hpo_optuna \
            --n-trials 10 \
            --n-jobs 2 \
            --seeds 0,1,2 \
            --timesteps 50000 \
            --search-space core+env \
            --lambda-penalty 7.0 \
            --volatility-penalty 0.5 \
            --study-name warlock_quick
        ;;

    full)
        echo "=== FULL SWEEP: 50 trials, 3 seeds, 100k steps ==="
        python -m src.agent.hpo_optuna \
            --n-trials 50 \
            --n-jobs 4 \
            --seeds 0,1,2 \
            --timesteps 100000 \
            --search-space full \
            --lambda-penalty 7.0 \
            --volatility-penalty 0.5 \
            --pruner median \
            --n-startup-trials 5 \
            --n-warmup-seeds 1 \
            --study-name warlock_full
        ;;

    resume)
        echo "=== RESUME SWEEP: continuing existing study ==="
        STUDY_NAME="${2:-warlock_full}"
        python -m src.agent.hpo_optuna \
            --n-trials 50 \
            --n-jobs 4 \
            --seeds 0,1,2 \
            --timesteps 100000 \
            --search-space full \
            --study-name "$STUDY_NAME"
        ;;

    analyze)
        echo "=== ANALYZE STUDY ==="
        STUDY_NAME="${2:-warlock_full}"
        python -m src.agent.hpo_optuna \
            --analyze \
            --study-name "$STUDY_NAME" \
            --show-top 20
        ;;

    lstm)
        echo "=== LSTM ARCHITECTURE SWEEP: 20 trials ==="
        python -m src.agent.hpo_optuna \
            --n-trials 20 \
            --n-jobs 2 \
            --seeds 0,1,2 \
            --timesteps 50000 \
            --search-space core+lstm \
            --study-name warlock_lstm
        ;;

    reward)
        echo "=== REWARD/ENV SWEEP: 20 trials ==="
        python -m src.agent.hpo_optuna \
            --n-trials 20 \
            --n-jobs 2 \
            --seeds 0,1,2 \
            --timesteps 50000 \
            --search-space core+env \
            --study-name warlock_reward
        ;;

    *)
        echo "Usage: $0 {quick|full|resume|analyze|lstm|reward} [study_name]"
        echo ""
        echo "  quick   - Fast test sweep (10 trials, core+env params)"
        echo "  full    - Full sweep (50 trials, all params, 4 workers)"
        echo "  resume  - Resume existing study (needs study_name arg)"
        echo "  analyze - Print top trials and param importance"
        echo "  lstm    - Focus on LSTM architecture search"
        echo "  reward  - Focus on reward/env parameter search"
        exit 1
        ;;
esac