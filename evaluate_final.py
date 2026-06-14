import evaluate
import pandas as pd
from src.data import load_cnn_dailymail
from src.models import (
    get_target_device,
    load_t5_architecture,
    generate_abstractive_summary,
)


def main() -> None:
    device = get_target_device()
    print(f"Using hardware accelerator: {device.type.upper()}")

    print("Loading fine-tuned model...")
    tokenizer, model = load_t5_architecture("./saved_model/final_t5_finetuned", device)

    print("Loading 100 test articles...")
    test_dataset = load_cnn_dailymail("test", sample_size=100)
    articles = test_dataset["article"]
    references = test_dataset["highlights"]

    print("Generating summaries using fine-tuned weights...")
    predictions = []

    for idx, text in enumerate(articles):
        # We use our reusable function from src.models!
        summary = generate_abstractive_summary(
            text, tokenizer, model, device, bullet_points=False
        )
        predictions.append(summary)

        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/100 articles...")

    print("Calculating final ROUGE metrics...")
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions, references=references, use_stemmer=True
    )

    print("\n" + "=" * 40)
    print("    FINE-TUNED T5 MODEL PERFORMANCE     ")
    print("=" * 40)
    for metric, score in results.items():
        print(f"{metric.upper()}: {score * 100:.2f}%")
    print("=" * 40)

    # Save to the new artifacts folder
    df_samples = pd.DataFrame(
        {
            "Article": [art[:200] + "..." for art in articles[:3]],
            "Ground_Truth": references[:3],
            "Fine_Tuned_Generation": predictions[:3],
        }
    )
    df_samples.to_csv("artifacts/finetuned_samples.csv", index=False)
    print("\nSaved artifacts to 'artifacts/finetuned_samples.csv'")


if __name__ == "__main__":
    main()
