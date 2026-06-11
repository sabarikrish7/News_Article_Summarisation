import spacy
import pytextrank
import evaluate
import pandas as pd
from datasets import load_dataset


def main():
    print("Loading spaCy with PyTextRank pipeline component...")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")

    print("Loading test dataset split...")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    test_set = dataset["test"]

    sample_size = 100
    articles = test_set["article"][:sample_size]
    references = test_set["highlights"][:sample_size]

    predictions = []
    print(f"Generating TextRank extractive summaries for {sample_size} articles...")
    for i, text in enumerate(articles):
        doc = nlp(text)

        summary_sentences = [
            span.text
            for span in doc._.textrank.summary(limit_phrases=15, limit_sentences=3)
        ]

        predictions.append("\n".join(summary_sentences))

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{sample_size} articles...")

    print("Initializing ROUGE evaluator...")
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions, references=references, use_stemmer=True
    )

    print("\n" + "=" * 40)
    print("     TEXTRANK BASELINE PERFORMANCE      ")
    print("=" * 40)
    for metric, score in results.items():
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
            "TextRank_Generation": predictions[:3],
        }
    )
    df_samples.to_csv("textrank_samples.csv", index=False)
    print("\nSaved sample metrics to 'textrank_samples.csv'")


if __name__ == "__main__":
    main()
