import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


def main():
    print("1. Loading dataset and slicing for rapid prototyping...")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

    train_dataset = dataset["train"].select(range(5000))
    eval_dataset = dataset["validation"].select(range(500))

    print("2. Loading T5Tokenizer and Model...")
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print("3. Tokenizing datasets...")

    def preprocess_function(examples):
        inputs = ["summarize: " + doc for doc in examples["article"]]

        model_inputs = tokenizer(inputs, max_length=512, truncation=True)

        labels = tokenizer(
            text_target=examples["highlights"], max_length=128, truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

    print("4. Setting up the Data Collator...")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    print("5. Configuring Training Arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./saved_model",
        eval_strategy="epoch",  # Check validation loss at the end of every epoch
        learning_rate=2e-5,
        per_device_train_batch_size=4,  # Keep batch size small to avoid Out Of Memory (OOM) errors
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,  # Prevent hard drive bloat by keeping only the 2 latest checkpoints
        num_train_epochs=3,  # Pass over the 5,000 examples 3 times
        predict_with_generate=True,  # Required for abstractive summarization models
        fp16=torch.cuda.is_available(),  # Accelerate training heavily if NVIDIA GPU is present
    )

    print("6. Initializing Trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("7. Commencing Fine-Tuning! (This will take some time)...")
    trainer.train()

    print("8. Saving the final model...")
    # Save both the model weights and the tokenizer so they can be loaded together later
    trainer.save_model("./saved_model/final_t5_finetuned")
    print("\nSUCCESS: Model saved to ./saved_model/final_t5_finetuned")


if __name__ == "__main__":
    main()
