from typing import Tuple
from datasets import load_dataset, Dataset


def load_cnn_dailymail(split: str = "test", sample_size: int = None) -> Dataset:
    """
    Loads and slices the CNN/DailyMail dataset for training or evaluation.

    Args:
        split (str): The dataset split to load ('train', 'validation', or 'test').
        sample_size (int, optional): Number of records to return for rapid prototyping.

    Returns:
        Dataset: The requested Hugging Face dataset split.
    """
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    selected_split = dataset[split]

    if sample_size:
        return selected_split.select(range(sample_size))

    return selected_split
