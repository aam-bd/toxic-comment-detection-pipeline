"""
Hyperparameter tuning module using Optuna (TPE Sampler).
Conducts systematic optimization of learning rate, batch size, weight decay, warmup ratio,
and epochs, with early pruning and automated plot generation saved to assets/tuning_results.png.
"""

from typing import Dict, Any, Callable
import os
import optuna
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_optuna_hp_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define the hyperparameter search space for Optuna:
    - learning_rate: [1e-5, 5e-5] (log distribution)
    - per_device_train_batch_size: {16, 32}
    - weight_decay: [0.0, 0.1] (uniform)
    - warmup_ratio: [0.0, 0.2] (uniform)
    - num_train_epochs: {2, 3, 4} (categorical/int)
    
    Args:
        trial (optuna.Trial): Active Optuna trial.
        
    Returns:
        Dict[str, Any]: Sampled hyperparameter configuration.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
    }


def format_trials_dataframe(study: optuna.Study) -> pd.DataFrame:
    """
    Convert an Optuna study into a cleanly formatted summary DataFrame.
    
    Args:
        study (optuna.Study): Completed or ongoing Optuna study.
        
    Returns:
        pd.DataFrame: Table of trial parameters, values, and status.
    """
    records = []
    for t in study.trials:
        rec = {
            "Trial": t.number,
            "Validation F1": f"{t.value:.4f}" if t.value is not None else "Pruned/Failed",
            "Learning Rate": f"{t.params.get('learning_rate', 0):.2e}",
            "Batch Size": t.params.get("per_device_train_batch_size", "N/A"),
            "Weight Decay": f"{t.params.get('weight_decay', 0):.3f}",
            "Warmup Ratio": f"{t.params.get('warmup_ratio', 0):.2f}",
            "Epochs": t.params.get("num_train_epochs", "N/A"),
            "State": t.state.name,
        }
        records.append(rec)
    return pd.DataFrame(records)


def save_tuning_visualizations(
    study: optuna.Study,
    output_path: str = "assets/tuning_results.png"
) -> None:
    """
    Generate and save a publication-quality composite plot showing:
    1. Optimization History (Objective Value vs. Trial Number)
    2. Hyperparameter Importances (Relative influence of each hyperparameter on Validation F1)
    
    Args:
        study (optuna.Study): Completed Optuna study.
        output_path (str): Destination path for the saved image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # --- 1. Optimization History ---
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    values = [t.value for t in study.trials if t.value is not None]
    best_values = np.maximum.accumulate(values) if values else []
    
    ax1.plot(trial_numbers, values, marker="o", linestyle="--", color="#1976D2", label="Trial Objective (F1)", alpha=0.7)
    if best_values.any():
        ax1.step(trial_numbers, best_values, where="post", color="#D32F2F", linewidth=2.5, label="Best Objective Value")
    
    ax1.set_xlabel("Trial Number", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Validation F1 (Toxic Class)", fontsize=11, fontweight="bold")
    ax1.set_title("Optuna Optimization History (TPE Sampler)", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True)
    
    # --- 2. Parameter Importances ---
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())
        scores = list(importances.values())
        
        # Sort in descending order
        y_pos = np.arange(len(params))
        ax2.barh(y_pos, scores, color="#388E3C", edgecolor="#1B5E20", alpha=0.85)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(params, fontsize=10)
        ax2.invert_yaxis()
        ax2.set_xlabel("Relative Importance Score", fontsize=11, fontweight="bold")
        ax2.set_title("Hyperparameter Importance Ranking", fontsize=12, fontweight="bold", pad=10)
        ax2.grid(True, linestyle=":", alpha=0.6, axis="x")
        
        for i, v in enumerate(scores):
            ax2.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9, fontweight="bold", color="#1B5E20")
    except Exception as e:
        # Fallback if too few trials for fANOVA
        ax2.text(0.5, 0.5, f"Importance requires completed trials\n({str(e)})", ha="center", va="center")
        ax2.axis("off")
        
    plt.suptitle("Hyperparameter Tuning Results (DistilBERT on Civil Comments)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Tuning] Tuning visualization successfully saved to: {output_path}")

