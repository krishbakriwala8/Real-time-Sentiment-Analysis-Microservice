"""
app/model/fine_tune.py
──────────────────────
Fine-tunes DistilBERT on a labelled sentiment dataset and saves the checkpoint.

Usage (inside the container or locally):
    python -m app.model.fine_tune \
        --dataset sst2 \
        --output_dir ./fine_tuned_model \
        --epochs 3 \
        --batch_size 16 \
        --lr 2e-5

The saved checkpoint can then be pointed to via FINE_TUNED_MODEL_PATH in .env.

Architecture note: this file is intentionally separate from inference code so
the API container never loads training dependencies at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


# ─── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Passed to HuggingFace Trainer for per-epoch evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


# ─── Dataset loading ───────────────────────────────────────────────────────────

def load_and_tokenize(
    dataset_name: str,
    tokenizer,
    max_length: int,
    text_column: str = "sentence",
    label_column: str = "label",
) -> DatasetDict:
    """
    Load a HuggingFace dataset and tokenize it.
    Defaults to SST-2 (glue/sst2) which matches the base model's pre-training.
    """
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name == "sst2":
        raw = load_dataset("glue", "sst2")
        text_col = "sentence"
    else:
        raw = load_dataset(dataset_name)
        text_col = text_column

    def tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding=False,          # DataCollatorWithPadding handles dynamic padding
            max_length=max_length,
        )

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=[text_col])

    # Rename label column to "labels" as expected by Trainer
    if label_column in tokenized["train"].column_names and label_column != "labels":
        tokenized = tokenized.rename_column(label_column, "labels")

    return tokenized


# ─── Training ─────────────────────────────────────────────────────────────────

def fine_tune(
    base_model: str,
    dataset_name: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    warmup_ratio: float,
    weight_decay: float,
) -> dict:
    """
    Fine-tune `base_model` on `dataset_name`.
    Returns a metrics dict including final eval accuracy.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Base model : {base_model}")
    logger.info(f"Output dir : {output_dir}")

    # ── Tokeniser & model ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, ignore_mismatched_sizes=True
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    tokenized = load_and_tokenize(dataset_name, tokenizer, max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=str(output_path / "logs"),
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",           # disable WandB / TensorBoard by default
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    logger.info("Starting training …")
    train_result = trainer.train()
    logger.info(f"Training complete. Metrics: {train_result.metrics}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    logger.info("Running final evaluation on validation set …")
    eval_metrics = trainer.evaluate()
    logger.info(f"Eval metrics: {eval_metrics}")

    # ── Detailed classification report ────────────────────────────────────────
    preds_output = trainer.predict(tokenized["validation"])
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = preds_output.label_ids

    report = classification_report(
        y_true, y_pred, target_names=["NEGATIVE", "POSITIVE"]
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"Confusion Matrix: {cm}")

    # ── Save ───────────────────────────────────────────────────────────────────
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save metadata so the service knows what it loaded
    metadata = {
        "base_model": base_model,
        "dataset": dataset_name,
        "epochs": epochs,
        "final_accuracy": eval_metrics.get("eval_accuracy"),
        "train_metrics": train_result.metrics,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    with open(output_path / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved to {output_dir}")
    return metadata


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sentiment")
    parser.add_argument(
        "--base_model",
        default="distilbert-base-uncased",
        help="HuggingFace model name or local path",
    )
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--output_dir", default="./fine_tuned_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    metrics = fine_tune(
        base_model=args.base_model,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )

    final_acc = metrics.get("final_accuracy", 0.0)
    logger.info(f"Final accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")


if __name__ == "__main__":
    main()
