from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nlp.preprocessing import MedicalTextPreprocessor


@dataclass
class NLPInferenceResult:
    nlp_score: float
    confidence: float
    predicted_label: str


class BioBERTRiskScorer:
    def __init__(self, model_dir: Path | None = None, max_length: int = 256):
        base_dir = Path(__file__).resolve().parents[1]
        self.model_dir = model_dir or (base_dir / "models" / "biobert")
        self.tfidf_dir = base_dir / "models" / "tfidf"
        self.backend = None
        self.max_length = max_length
        self.preprocessor = MedicalTextPreprocessor(use_lemmatization=True)

        if self.model_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.eval()
            self.backend = "biobert"
            return

        vectorizer_path = self.tfidf_dir / "vectorizer.joblib"
        classifier_path = self.tfidf_dir / "classifier.joblib"
        if vectorizer_path.exists() and classifier_path.exists():
            self.vectorizer = joblib.load(vectorizer_path)
            self.classifier = joblib.load(classifier_path)
            self.backend = "tfidf"
            return

        raise FileNotFoundError(
            "No NLP model found. Expected BioBERT at "
            f"{self.model_dir} or TF-IDF artifacts at {self.tfidf_dir}. "
            "Run `python -m nlp.train_models` first."
        )

    @torch.inference_mode()
    def score_text(self, raw_text: str) -> NLPInferenceResult:
        clean_text = self.preprocessor.preprocess(raw_text)
        if not clean_text:
            return NLPInferenceResult(nlp_score=0.0, confidence=0.0, predicted_label="LOW_RISK_TEXT")

        if self.backend == "biobert":
            encoded = self.tokenizer(
                clean_text,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
            nlp_score = float(probs[1])
            confidence = float(np.max(probs))
            label = "HIGH_RISK_TEXT" if np.argmax(probs) == 1 else "LOW_RISK_TEXT"
            return NLPInferenceResult(nlp_score=nlp_score, confidence=confidence, predicted_label=label)

        probs = self.classifier.predict_proba(self.vectorizer.transform([clean_text]))[0]
        classes = list(self.classifier.classes_)
        high_risk_idx = classes.index(1) if 1 in classes else int(np.argmax(probs))
        pred_idx = int(np.argmax(probs))
        pred_class = classes[pred_idx]
        nlp_score = float(probs[high_risk_idx])
        confidence = float(np.max(probs))
        label = "HIGH_RISK_TEXT" if pred_class == 1 else "LOW_RISK_TEXT"
        return NLPInferenceResult(nlp_score=nlp_score, confidence=confidence, predicted_label=label)
