import torch
import nltk
import evaluate
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset


def main():
    # 1. Detect hardware acceleration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU for generation...")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal) for generation...")
    else:
        device = torch.device("cpu")
        print("Using CPU for generation (this might take a few minutes)...")

    # 2. Explicitly load the Tokenizer and Model
    print("Loading T5 tokenizer and model explicitly...")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)

    # 3. Load the test dataset
    print("Loading test dataset split...")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    sample_size = 100
    articles = dataset["test"]["article"][:sample_size]
    references = dataset["test"]["highlights"][:sample_size]

    # 4. Generate Summaries
    print(f"Generating abstractive summaries for {sample_size} articles...")
    predictions = []

    for idx, text in enumerate(articles):
        # T5 REQUIREMENT: We must explicitly tell it which task to perform
        input_text = "summarize: " + text

        # Convert text to model-readable token IDs
        inputs = tokenizer(
            input_text, return_tensors="pt", max_length=512, truncation=True
        ).to(device)

        # Generate the summary tokens
        # We use beam search (num_beams=4) to get higher quality text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=130,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )

        # Decode tokens back into readable English
        raw_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-processing for ROUGELSUM formatting
        sentences = nltk.sent_tokenize(raw_summary)
        formatted_summary = "\n".join(sentences)
        predictions.append(formatted_summary)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{sample_size} articles...")

    # 5. Evaluate with ROUGE
    print("Initializing ROUGE evaluator...")
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions, references=references, use_stemmer=True
    )

    print("\n" + "=" * 40)
    print("    T5-SMALL (ZERO-SHOT) PERFORMANCE    ")
    print("=" * 40)
    for metric, score in results.items():
        print(
            f"{metric.upper()}: {score * 100:.2f}%"
            if score <= 1.0
            else f"{metric.upper()}: {score:.2f}%"
        )
    print("=" * 40)

    # 6. Save results
    df_samples = pd.DataFrame(
        {
            "Article": [art[:200] + "..." for art in articles[:3]],
            "Ground_Truth": references[:3],
            "T5_Generation": predictions[:3],
        }
    )
    df_samples.to_csv("t5_zeroshot_samples.csv", index=False)
    print("\nSaved abstractive samples to 't5_zeroshot_samples.csv'")


if __name__ == "__main__":
    main()
