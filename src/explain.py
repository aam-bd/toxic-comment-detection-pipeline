"""
Interpretability and Explainability module using SHAP (SHapley Additive exPlanations)
with LIME as a documented fallback.
Computes token-level attribution values, generates publication-ready visualizations,
and exports explanation charts to assets/ for technical reporting.
"""

from typing import List, Dict, Any, Tuple
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from transformers import TextClassificationPipeline, PreTrainedModel, PreTrainedTokenizerBase


def create_shap_pipeline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: int = -1
) -> TextClassificationPipeline:
    """
    Wrap model and tokenizer in a Hugging Face TextClassificationPipeline configured for SHAP.
    
    Args:
        model: Trained sequence classification model.
        tokenizer: Corresponding tokenizer.
        device (int): Device index (-1 for CPU, 0 for GPU).
        
    Returns:
        TextClassificationPipeline: Configured pipeline with raw/softmax scores.
    """
    model.eval()
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=device,
        truncation=True,
        max_length=128
    )
    return pipe


def render_token_attribution_plot(
    tokens: List[str],
    attributions: List[float],
    title: str,
    base_value: float,
    output_prob: float,
    predicted_label: str,
    output_path: str
) -> None:
    """
    Render a clean, publication-ready token attribution heatmap bar plot:
    - Red bars: Positive attribution (pushes prediction toward TOXIC).
    - Blue bars: Negative attribution (pushes prediction toward NOT TOXIC).
    
    Args:
        tokens (List[str]): Input tokens/words.
        attributions (List[float]): Attribution score for each token.
        title (str): Subplot title (e.g., 'True Positive: Blatant Toxicity').
        base_value (float): Model base/expected value.
        output_prob (float): Model predicted toxic probability.
        predicted_label (str): Predicted class label.
        output_path (str): Destination path for the PNG file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter or truncate if tokens are too long
    max_tokens = 20
    if len(tokens) > max_tokens:
        # Keep highest magnitude tokens
        sorted_indices = np.argsort(np.abs(attributions))[-max_tokens:]
        sorted_indices = sorted(sorted_indices)
        tokens = [tokens[i] for i in sorted_indices]
        attributions = [attributions[i] for i in sorted_indices]

    y_pos = np.arange(len(tokens))
    colors = ["#D32F2F" if val >= 0 else "#1976D2" for val in attributions]
    
    fig, ax = plt.subplots(figsize=(10, max(4.0, len(tokens) * 0.32)), dpi=300)
    
    bars = ax.barh(y_pos, attributions, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens, fontsize=10, fontweight="medium")
    ax.invert_yaxis()
    
    ax.set_xlabel("SHAP Attribution Value (Impact on Toxic Probability)", fontsize=10, fontweight="bold")
    ax.set_title(
        f"{title}\nPredicted: {predicted_label} | Toxic Probability: {output_prob:.1%} (Base: {base_value:.1%})",
        fontsize=11,
        fontweight="bold",
        pad=12
    )
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")
    
    # Add values at the end of each bar
    for bar, val in zip(bars, attributions):
        offset = 0.005 if val >= 0 else -0.005
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}", va="center", ha=ha, fontsize=8, fontweight="bold")

    # Custom legend
    red_patch = patches.Patch(color="#D32F2F", label="Pushes toward Toxic (+)")
    blue_patch = patches.Patch(color="#1976D2", label="Pushes toward Not Toxic (-)")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right", frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Explanation saved to: {output_path}")


def explain_sample_lime(
    text: str,
    predict_fn: Any,
    class_names: List[str] = ["Not Toxic", "Toxic"]
) -> Any:
    """
    Fallback interpretability method using LIME (Local Interpretable Model-agnostic Explanations).
    
    Args:
        text (str): Comment text to explain.
        predict_fn: Model prediction function returning class probabilities.
        class_names (List[str]): Names of classes.
        
    Returns:
        Explanation object from LIME.
    """
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_fn, num_features=10, num_samples=250)
    return exp

