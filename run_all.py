"""
Orchestrates the full experimental protocol mandated by Section 2 (Core Requirements):
  - 4 main configs (baseline + 3 DL approaches), each x3 seeds        -> fair comparison
  - LR sweep (3 points) x3 seeds                                      -> hyperparameter study
  - augmentation on/off pair x3 seeds                                 -> data-strategy study
  - modality ablation (all vs accel_only) x3 seeds                    -> ablation study
Then aggregates every run and builds the cross-model comparison table.

Usage: python run_all.py [--quick]   (--quick caps epochs for a smoke test)
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(__file__)
CONFIGS_DIR = os.path.join(ROOT, "configs")
SEEDS = [0, 1, 2]  # Core Req. #6: >=3 random seeds per final configuration

MAIN_MODELS = ["baseline_mlp", "lstm", "transformer", "cnn_lstm"]
HP_SWEEP = ["hp_lr_1e-2", "hp_lr_1e-3", "hp_lr_1e-4"]
DATA_STRATEGY = ["cnn_lstm", "data_strategy_no_augment"]  # cnn_lstm.yaml = augment:true arm
ABLATION = ["cnn_lstm", "ablation_no_gyro"]               # cnn_lstm.yaml = full-modality arm

ALL_RUN_NAMES = {
    "baseline_mlp": "baseline_mlp", "lstm": "lstm", "transformer": "transformer",
    "cnn_lstm": "cnn_lstm", "hp_lr_1e-2": "cnn_lstm_lr1e-2", "hp_lr_1e-3": "cnn_lstm_lr1e-3",
    "hp_lr_1e-4": "cnn_lstm_lr1e-4", "data_strategy_no_augment": "cnn_lstm_no_augment",
    "ablation_no_gyro": "cnn_lstm_ablation_no_gyro",
}


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def train_all(config_stems, quick):
    seen = set()
    for stem in config_stems:
        if stem in seen:
            continue
        seen.add(stem)
        cfg_path = os.path.join(CONFIGS_DIR, f"{stem}.yaml")
        for seed in SEEDS:
            cmd = [sys.executable, os.path.join("src", "train.py"), "--config", cfg_path, "--seed", str(seed)]
            if quick:
                cmd += ["--epochs", "2"]
            run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke-test mode (not for final numbers)")
    args = parser.parse_args()

    print("STEP 1/3: data preprocessing")
    run([sys.executable, os.path.join("data", "preprocess.py")])

    print("\nSTEP 2/3: training — main models, HP sweep, data-strategy pair, ablation pair")
    config_stems = MAIN_MODELS + HP_SWEEP + DATA_STRATEGY + ABLATION
    train_all(config_stems, args.quick)

    print("\nSTEP 3/3: evaluation — aggregate + comparison table")
    run_names = list(dict.fromkeys(ALL_RUN_NAMES[s] for s in config_stems))
    run([sys.executable, os.path.join("src", "evaluate.py"), "--mode", "compare",
         "--run_names"] + run_names)

    print("\nAll runs complete. See experiments/results/comparison_table.csv")


if __name__ == "__main__":
    main()
