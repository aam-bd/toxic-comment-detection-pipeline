"""
Utility functions for toxicity classification pipeline:
- Random seed initialization for reproducibility
- Device selection (CUDA / CPU)
- Metric display and table formatting helpers
"""

import os
import random
import torch
import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch for full reproducibility.
    
    Args:
        seed (int): The integer random seed (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Ensure deterministic behavior where supported
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Setup] Deterministic random seed set to: {seed}")


def get_device() -> torch.device:
    """
    Detect and return the active compute device (CUDA GPU or CPU),
    printing hardware specifications if GPU is available.
    
    Returns:
        torch.device: PyTorch device handle.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"[Device] Using GPU: {gpu_name} (Total VRAM: {vram_mb:.1f} MB)")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available. Running on CPU.")
    return device


def format_metrics_table(metrics_dict: dict) -> pd.DataFrame:
    """
    Format a dictionary of evaluation results into a clean pandas DataFrame.
    
    Args:
        metrics_dict (dict): Dictionary mapping model names to metric dictionaries.
        
    Returns:
        pd.DataFrame: Formatted comparison table.
    """
    rows = []
    for model_name, m in metrics_dict.items():
        rows.append({
            "Model Configuration": model_name,
            "Accuracy": f"{m.get('accuracy', 0.0):.4f}",
            "Precision (Toxic)": f"{m.get('precision', 0.0):.4f}",
            "Recall (Toxic)": f"{m.get('recall', 0.0):.4f}",
            "F1-Score (Toxic)": f"{m.get('f1', 0.0):.4f}",
            "ROC-AUC": f"{m.get('roc_auc', 0.0):.4f}",
        })
    df = pd.DataFrame(rows)
    return df

