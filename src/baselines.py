from typing import List
import nltk
import spacy
import pytextrank

# Initialize tools globally within the module
nltk.download("punkt", quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
except OSError:
    import subprocess

    print("Downloading spaCy en_core_web_sm model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")


def generate_lead3(text: str, bullet_points: bool = False) -> str:
    """
    Extracts the first 3 sentences of a text using the Lead-3 heuristic.

    Args:
        text (str): The source article.
        bullet_points (bool): Whether to format the output with Markdown bullets.

    Returns:
        str: The concatenated baseline summary.
    """
    sentences: List[str] = nltk.sent_tokenize(text.strip())
    extracted = sentences[:3]

    if bullet_points:
        return "\n\n".join([f"• {s}" for s in extracted])
    return "\n".join(extracted)


def generate_textrank(text: str) -> str:
    """
    Extracts the top 3 mathematical sentences using a PyTextRank graph matrix.

    Args:
        text (str): The source article.

    Returns:
        str: The concatenated extractive summary.
    """
    doc = nlp(text)
    summary_sentences = [
        span.text
        for span in doc._.textrank.summary(limit_phrases=15, limit_sentences=3)
    ]
    return "\n".join(summary_sentences)
