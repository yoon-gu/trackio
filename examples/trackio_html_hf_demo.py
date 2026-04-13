"""Real Hugging Face Trainer + trackio_html integration demo.

Trains `prajjwal1/bert-tiny` on synthetic binary text classification (so no
dataset download is needed beyond the model itself) and forwards Trainer
metrics into trackio_html via a TrainerCallback. Verifies that the
self-contained HTML dashboard captures train/eval loss, accuracy, and lr.

Run from the repo root with the venv that has torch + transformers + datasets:
    python examples/trackio_html_hf_demo.py
"""

import os
import random
import sys
import time

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trackio_html as wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
SEED = 0


def make_dataset(n: int) -> Dataset:
    rng = random.Random(SEED + n)
    pos = [
        "i love this",
        "great movie",
        "wonderful",
        "amazing experience",
        "the best ever",
    ]
    neg = [
        "i hate this",
        "terrible movie",
        "awful",
        "worst experience",
        "the worst ever",
    ]
    texts, labels = [], []
    for _ in range(n):
        if rng.random() < 0.5:
            texts.append(rng.choice(pos) + " number " + str(rng.randint(0, 99)))
            labels.append(1)
        else:
            texts.append(rng.choice(neg) + " number " + str(rng.randint(0, 99)))
            labels.append(0)
    return Dataset.from_dict({"text": texts, "label": labels})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": float((preds == labels).mean())}


class TrackioCallback(TrainerCallback):
    def __init__(self, run_name: str, extra_config: dict):
        self.run_name = run_name
        self.extra_config = extra_config

    def on_train_begin(self, args, state, control, **kwargs):
        wandb.init(
            project="hf-tiny-distilbert",
            name=self.run_name,
            config={
                "model": MODEL,
                "lr": args.learning_rate,
                "bs": args.per_device_train_batch_size,
                "epochs": args.num_train_epochs,
                "weight_decay": args.weight_decay,
                **self.extra_config,
            },
            system_interval=1.0,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        cleaned = {}
        for k, v in logs.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith("eval_"):
                cleaned[f"val/{k[5:]}"] = v
            else:
                cleaned[f"train/{k}"] = v
        if cleaned:
            wandb.log(cleaned, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        wandb.finish()


def run_one(run_name: str, lr: float, bs: int, epochs: int):
    torch.manual_seed(SEED)
    train_ds = make_dataset(2400)
    eval_ds = make_dataset(200)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=24
        )

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)
    train_ds = train_ds.rename_column("label", "labels")
    eval_ds = eval_ds.rename_column("label", "labels")
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=f"./hf-output/{run_name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[TrackioCallback(run_name=run_name, extra_config={"max_len": 24})],
    )
    t0 = time.time()
    trainer.train()
    print(f"[{run_name}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_one("lr-5e-4-bs16", lr=5e-4, bs=16, epochs=5)
    run_one("lr-1e-3-bs16", lr=1e-3, bs=16, epochs=5)
    run_one("lr-5e-4-bs32", lr=5e-4, bs=32, epochs=5)
    print("all runs complete")
    print("dashboard: trackio_html/hf-tiny-distilbert.html")
