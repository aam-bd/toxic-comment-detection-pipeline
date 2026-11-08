"""
Model initialization, freezing/unfreezing, training execution, and evaluation module.
Calculates Accuracy, Precision, Recall, F1 (toxic class), ROC-AUC, and Confusion Matrix.
"""

from typing import Dict, Tuple, Any
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    PreTrainedModel
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def compute_classification_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics for binary toxicity classification:
    - Accuracy
    - Precision (pos_label=1, toxic class)
    - Recall (pos_label=1, toxic class)
    - F1-Score (pos_label=1, toxic class)
    - ROC-AUC Score (derived from softmax probabilities)
    
    Args:
        eval_pred: Tuple of (logits, labels) from Hugging Face Trainer.
        
    Returns:
        Dict[str, float]: Calculated metrics.
    """
    logits, labels = eval_pred
    
    # Compute class probabilities via softmax
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, pos_label=1, zero_division=0)
    rec = recall_score(labels, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(labels, predictions, pos_label=1, zero_division=0)
    
    # Calculate ROC-AUC using predicted probability of class 1 (Toxic)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = 0.5

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc)
    }


def get_model(
    model_checkpoint: str = "distilbert-base-uncased",
    num_labels: int = 2,
    freeze_backbone: bool = False
) -> Tuple[PreTrainedModel, int, int]:
    """
    Instantiate a sequence classification model and optionally freeze its transformer backbone.
    
    Args:
        model_checkpoint (str): Pre-trained model identifier.
        num_labels (int): Number of target classes (default: 2).
        freeze_backbone (bool): If True, freezes transformer encoder weights (transfer learning baseline).
        
    Returns:
        Tuple[PreTrainedModel, int, int]: (model, trainable_params, total_params)
    """
    print(f"[Model] Initializing '{model_checkpoint}' (num_labels={num_labels})...")
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    
    if freeze_backbone:
        print("[Model] FREEZING base transformer encoder (Transfer Learning Baseline)...")
        # Freeze encoder layers depending on architecture
        if hasattr(model, "distilbert"):
            for param in model.distilbert.parameters():
                param.requires_grad = False
        elif hasattr(model, "bert"):
            for param in model.bert.parameters():
                param.requires_grad = False
        elif hasattr(model, "roberta"):
            for param in model.roberta.parameters():
                param.requires_grad = False
        else:
            # Fallback: freeze everything except classifier head
            for name, param in model.named_parameters():
                if "classifier" not in name and "pre_classifier" not in name:
                    param.requires_grad = False
    else:
        print("[Model] UNFREEZING all layers for end-to-end fine-tuning...")
        for param in model.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    ratio = (trainable_params / total_params) * 100
    print(f"[Model] Trainable Parameters: {trainable_params:,} / {total_params:,} ({ratio:.2f}%)")
    
    return model, trainable_params, total_params


def evaluate_with_confusion_matrix(
    trainer: Trainer,
    test_dataset: Any
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate a trained model on a test set, returning both standard metrics and the confusion matrix.
    
    Args:
        trainer (Trainer): Hugging Face Trainer instance.
        test_dataset: Tokenized test dataset.
        
    Returns:
        Tuple[Dict[str, float], np.ndarray]: (metrics_dict, confusion_matrix_array)
    """
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs[:, 1]))
    }
    
    cm = confusion_matrix(labels, preds)
    return metrics, cm

