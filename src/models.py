# src/models.py
from typing import Tuple
import torch
import nltk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)


def get_target_device() -> torch.device:
    """Detects available hardware backends and returns the targeted compute routing path."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_t5_architecture(
    model_path: str, device: torch.device
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Loads a T5 tokenizer and weights matrix onto the specified hardware device.

    Args:
        model_path (str): Hugging Face hub ID or local directory path.
        device (torch.device): Target hardware execution environment.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel]: The initialized NLP components.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    return tokenizer, model


def generate_abstractive_summary(
    text: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
    bullet_points: bool = False,
) -> str:
    """
    Generates a highly-optimized abstractive summary using an encoder-decoder network.

    Args:
        text (str): The raw input news article.
        tokenizer (PreTrainedTokenizer): The specific tokenizer paired with the model.
        model (PreTrainedModel): The loaded weights matrix.
        device (torch.device): Hardware execution environment.
        bullet_points (bool): Whether to format the output with Markdown bullets.

    Returns:
        str: A formatted string containing the generated summary.
    """
    if not text.strip():
        return "Warning: Input text sequence is empty."

    input_text = "summarize: " + text
    inputs = tokenizer(
        input_text, return_tensors="pt", max_length=512, truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=130,
            min_length=30,
            length_penalty=2.0,
            repetition_penalty=2.5,
            num_beams=4,
            early_stopping=True,
        )

    raw_summary: str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sentences = nltk.sent_tokenize(raw_summary)

    if bullet_points:
        return "\n\n".join([f"• {s.strip()}" for s in sentences])
    return "\n".join(sentences)
