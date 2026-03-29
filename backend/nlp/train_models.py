from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from nlp.preprocessing import MedicalTextPreprocessor, load_and_prepare_dataset, resolve_dataset_path


@dataclass
class TrainConfig:
    model_name: str = "dmis-lab/biobert-base-cased-v1.1"
    output_dir: str = "models"
    test_size: float = 0.2
    random_state: int = 42
    max_length: int = 256
    epochs: int = 1
    train_batch_size: int = 8
    eval_batch_size: int = 8
    max_samples: int = 2500


class NumpyTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_tfidf_model(X_train, y_train, X_test, y_test, model_dir: Path) -> dict:
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    acc = float(accuracy_score(y_test, y_pred))

    tfidf_dir = model_dir / "tfidf"
    tfidf_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, tfidf_dir / "vectorizer.joblib")
    joblib.dump(clf, tfidf_dir / "classifier.joblib")
    return {"accuracy": acc, "path": str(tfidf_dir)}


def train_biobert_model(X_train, y_train, X_test, y_test, cfg: TrainConfig, model_dir: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)

    train_ds = NumpyTextDataset(X_train.tolist(), y_train.tolist(), tokenizer, cfg.max_length)
    eval_ds = NumpyTextDataset(X_test.tolist(), y_test.tolist(), tokenizer, cfg.max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_args = TrainingArguments(
        output_dir=str(model_dir / "biobert_training_runs"),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=0.01,
        logging_steps=50,
        report_to=[],
        seed=cfg.random_state,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()

    biobert_dir = model_dir / "biobert"
    biobert_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(biobert_dir)
    tokenizer.save_pretrained(biobert_dir)
    return {"accuracy": float(metrics.get("eval_accuracy", 0.0)), "path": str(biobert_dir)}


def train_all_models(config: TrainConfig | None = None) -> dict:
    cfg = config or TrainConfig()
    set_seed(cfg.random_state)
    base_dir = Path(__file__).resolve().parents[1]

    preprocessor = MedicalTextPreprocessor(use_lemmatization=True)
    dataset_path = resolve_dataset_path(base_dir)
    df = load_and_prepare_dataset(dataset_path, preprocessor)

    if cfg.max_samples and len(df) > cfg.max_samples:
        df = (
            df.groupby("risk_label", group_keys=False)
            .apply(lambda g: g.sample(n=max(1, int(cfg.max_samples * len(g) / len(df))), random_state=cfg.random_state))
            .sample(frac=1.0, random_state=cfg.random_state)
            .reset_index(drop=True)
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["risk_label"],
        test_size=cfg.test_size,
        stratify=df["risk_label"],
        random_state=cfg.random_state,
    )

    model_dir = base_dir / cfg.output_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    tfidf_metrics = train_tfidf_model(X_train, y_train, X_test, y_test, model_dir)
    biobert_metrics = train_biobert_model(X_train, y_train, X_test, y_test, cfg, model_dir)

    # Per requirement, BioBERT is considered the primary model.
    final_choice = "biobert"
    summary = {
        "dataset_path": str(dataset_path),
        "sample_count": int(len(df)),
        "tfidf": tfidf_metrics,
        "biobert": biobert_metrics,
        "final_model": final_choice,
    }
    (model_dir / "model_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = train_all_models()
    print(json.dumps(result, indent=2))

