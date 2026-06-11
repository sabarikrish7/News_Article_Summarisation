import torch
import nltk
import evaluate
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset


def main():
    # 1. Hardware Routing
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU for final evaluation...")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal) for final evaluation...")
    else:
        device = torch.device("cpu")
        print("Using CPU for final evaluation...")

    # 2. Load the Fine-Tuned Weights and Tokenizer
    model_path = "./saved_model/final_t5_finetuned"
    print(f"Loading custom fine-tuned model from '{model_path}'...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    except Exception as e:
        print(
            f"\nError loading model. Did Milestone 6 finish training completely? Details: {e}"
        )
        return

    # 3. Load the identical 100 test articles
    print("Loading test dataset split...")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    sample_size = 100
    articles = dataset["test"]["article"][:sample_size]
    references = dataset["test"]["highlights"][:sample_size]

    # 4. Generate Summaries with New Weights
    print(
        f"Generating summaries using fine-tuned weights for {sample_size} articles..."
    )
    predictions = []

    for idx, text in enumerate(articles):
        input_text = "summarize: " + text
        inputs = tokenizer(
            input_text, return_tensors="pt", max_length=512, truncation=True
        ).to(device)

        # Keep generation parameters identical to zero-shot for a controlled experiment
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

        raw_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-process with newline characters for ROUGELSUM formatting
        sentences = nltk.sent_tokenize(raw_summary)
        formatted_summary = "\n".join(sentences)
        predictions.append(formatted_summary)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{sample_size} articles...")

    # 5. Compute Final ROUGE Scores
    print("Calculating final ROUGE metrics...")
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions, references=references, use_stemmer=True
    )

    print("\n" + "=" * 40)
    print("    FINE-TUNED T5 MODEL PERFORMANCE     ")
    print("=" * 40)
    for metric, score in results.items():
        print(
            f"{metric.upper()}: {score * 100:.2f}%"
            if score <= 1.0
            else f"{metric.upper()}: {score:.2f}%"
        )
    print("=" * 40)

    # Save a final verification artifact
    df_samples = pd.DataFrame(
        {
            "Article": [art[:200] + "..." for art in articles[:3]],
            "Ground_Truth": references[:3],
            "Fine_Tuned_Generation": predictions[:3],
        }
    )
    df_samples.to_csv("finetuned_samples.csv", index=False)
    print("\nSaved fine-tuned generations to 'finetuned_samples.csv'")


if __name__ == "__main__":
    main()
