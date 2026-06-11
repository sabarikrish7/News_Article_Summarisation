import nltk
from datasets import load_dataset
import evaluate
import pandas as pd

nltk.download("punkt", quiet=True)


def baseline_summariser(text):
    """
    Extracts the first 3 lines of the text.
    """
    sentences = nltk.sent_tokenize(text.strip())
    baseline_summary = "\n".join(sentences[:3])
    return baseline_summary


def main():
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    test_set = dataset["test"]
    sample_size = 100
    articles = test_set["article"][:sample_size]
    references = test_set["highlights"][:sample_size]
    predictions = [baseline_summariser(text) for text in articles]
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions, references=references, use_stemmer=True
    )
    print(" Rouge results")
    print("===============")
    print("          Baseline performance          ")
    print("=" * 40)
    for metric, score in results.items():
        # ROUGE scores return as percentages (0.0 to 1.0) in newer evaluate versions,
        # or absolute percentages depending on the back-end. Let's format nicely.
        print(
            f"{metric.upper()}: {score * 100:.2f}%"
            if score <= 1.0
            else f"{metric.upper()}: {score:.2f}%"
        )
    print("=" * 40)
    df_samples = pd.DataFrame(
        {
            "Article": [art[:200] + "..." for art in articles[:3]],
            "Ground_Truth": references[:3],
            "Baseline_Generation": predictions[:3],
        }
    )
    df_samples.to_csv("baseline_samples.csv", index=False)
    print("\nSaved sample generations to 'baseline_samples.csv'")


if __name__ == "__main__":
    main()
