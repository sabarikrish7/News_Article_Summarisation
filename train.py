import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from src.data import load_cnn_dailymail
from src.models import load_t5_architecture


def main() -> None:
    print("1. Loading dataset splits from src...")
    train_dataset = load_cnn_dailymail("train", sample_size=5000)
    eval_dataset = load_cnn_dailymail("validation", sample_size=500)

    print("2. Initializing T5 Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_t5_architecture("t5-small", device)

    print("3. Preprocessing datasets...")

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
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    print("4. Configuring Trainer...")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./saved_model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("5. Commencing Fine-Tuning...")
    trainer.train()
    trainer.save_model("./saved_model/final_t5_finetuned")
    print("\nSUCCESS: Model saved to ./saved_model/final_t5_finetuned")


if __name__ == "__main__":
    main()
