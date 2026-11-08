"""
Data ingestion, binarization, stratified sampling, and tokenization module.
Handles Google Civil Comments dataset preprocessing for toxicity classification.
"""

from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from sklearn.model_selection import train_test_split


def binarize_toxicity(example: dict, threshold: float = 0.5) -> dict:
    """
    Binarize continuous toxicity score into 0 (Not Toxic) or 1 (Toxic).
    
    Args:
        example (dict): Dataset record containing 'toxicity'.
        threshold (float): Decision threshold (default: 0.5).
        
    Returns:
        dict: Updated record with 'label' integer column.
    """
    example["label"] = 1 if float(example["toxicity"]) >= threshold else 0
    return example


def load_civil_comments(dataset_name: str = "google/civil_comments", threshold: float = 0.5) -> DatasetDict:
    """
    Load raw Civil Comments dataset and add binary classification labels.
    
    Args:
        dataset_name (str): Hugging Face hub identifier.
        threshold (float): Toxicity threshold.
        
    Returns:
        DatasetDict: Mapped dataset with 'label' column.
    """
    print(f"[Data] Loading dataset '{dataset_name}' from Hugging Face Hub...")
    raw_dataset = load_dataset(dataset_name)
    print(f"[Data] Binarizing toxicity labels with threshold >= {threshold}...")
    dataset = raw_dataset.map(lambda ex: binarize_toxicity(ex, threshold))
    return dataset


def create_stratified_subsets(
    dataset: DatasetDict,
    train_size: int = 20000,
    val_size: int = 4000,
    test_size: int = 4000,
    seed: int = 42
) -> Dict[str, Dataset]:
    """
    Create stratified random subsets for train, validation, and test splits.
    Stratification preserves the minority positive toxicity ratio (~8%) across all splits.
    
    Args:
        dataset (DatasetDict): Preprocessed Civil Comments dataset.
        train_size (int): Number of training examples (default: 20000).
        val_size (int): Number of validation examples (default: 4000).
        test_size (int): Number of test examples (default: 4000).
        seed (int): Random seed for reproducibility.
        
    Returns:
        Dict[str, Dataset]: Dictionary with 'train', 'validation', and 'test' subsets.
    """
    print(f"[Data] Extracting stratified subsets: {train_size} train / {val_size} val / {test_size} test...")
    
    # Stratified sampling helper for a given split
    def sample_stratified(split_data: Dataset, target_size: int) -> Dataset:
        labels = np.array(split_data["label"])
        indices = np.arange(len(labels))
        
        # Use scikit-learn train_test_split to perform exact stratified selection
        sampled_idx, _ = train_test_split(
            indices,
            train_size=target_size,
            stratify=labels,
            random_state=seed,
            shuffle=True
        )
        return split_data.select(sampled_idx)

    train_subset = sample_stratified(dataset["train"], train_size)
    val_subset = sample_stratified(dataset["validation"], val_size)
    test_subset = sample_stratified(dataset["test"], test_size)

    # Print distribution stats
    for split_name, ds in [("Train", train_subset), ("Validation", val_subset), ("Test", test_subset)]:
        labels = ds["label"]
        pos_count = sum(labels)
        pos_ratio = pos_count / len(labels)
        print(f"  - {split_name:10s}: {len(ds):,} samples | Toxic: {pos_count:,} ({pos_ratio:.2%}) | Not Toxic: {len(ds)-pos_count:,} ({1-pos_ratio:.2%})")

    return {
        "train": train_subset,
        "validation": val_subset,
        "test": test_subset
    }


def compute_class_weights(labels: List[int]) -> Tuple[float, float]:
    """
    Compute balanced class weights for the binary classification task:
    weight_c = N_total / (2 * N_c)
    
    Args:
        labels (List[int]): Target labels.
        
    Returns:
        Tuple[float, float]: (weight_class_0, weight_class_1)
    """
    total = len(labels)
    num_pos = sum(labels)
    num_neg = total - num_pos
    
    weight_0 = total / (2.0 * num_neg)
    weight_1 = total / (2.0 * num_pos)
    print(f"[Data] Class Weights -> Class 0 (Not Toxic): {weight_0:.3f} | Class 1 (Toxic): {weight_1:.3f}")
    return weight_0, weight_1


def tokenize_subsets(
    subsets: Dict[str, Dataset],
    model_checkpoint: str = "distilbert-base-uncased",
    max_length: int = 128
) -> Tuple[PreTrainedTokenizerBase, Dict[str, Dataset]]:
    """
    Tokenize train, validation, and test subsets with padding and truncation.
    
    Args:
        subsets (Dict[str, Dataset]): Dictionary of Dataset splits.
        model_checkpoint (str): Model name or path for tokenizer.
        max_length (int): Maximum sequence length (default: 128).
        
    Returns:
        Tuple[PreTrainedTokenizerBase, Dict[str, Dataset]]: Tokenizer and tokenized datasets formatted for PyTorch.
    """
    print(f"[Tokenizer] Loading tokenizer '{model_checkpoint}' (max_length={max_length})...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized_dict = {}
    for split_name, ds in subsets.items():
        print(f"  - Tokenizing {split_name} split...")
        tok_ds = ds.map(tokenize_fn, batched=True, remove_columns=[col for col in ds.column_names if col not in ["label", "input_ids", "attention_mask"]])
        tok_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        tokenized_dict[split_name] = tok_ds

    print("[Tokenizer] Tokenization complete and formatted for PyTorch.")
    return tokenizer, tokenized_dict

