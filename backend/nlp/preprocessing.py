from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

RISKY_TERMS = {
    "sepsis",
    "stroke",
    "cardiac arrest",
    "myocardial infarction",
    "heart failure",
    "renal failure",
    "hemorrhage",
    "pulmonary embolism",
    "shock",
    "critical",
    "emergency",
    "intensive care",
    "respiratory failure",
    "arrhythmia",
    "malignant",
    "tumor",
    "cancer",
}


def ensure_nltk_assets() -> None:
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


@dataclass
class MedicalTextPreprocessor:
    use_lemmatization: bool = True

    def __post_init__(self) -> None:
        ensure_nltk_assets()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)


def resolve_dataset_path(base_dir: Path) -> Path:
    candidates = [
        base_dir / "data" / "empty_samples",
        base_dir / "data" / "empty_samples.csv",
        base_dir / "data" / "mtsamples.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No NLP dataset found. Expected one of: data/empty_samples, data/empty_samples.csv, data/mtsamples.csv"
    )


def derive_binary_risk_label(text: str) -> int:
    lowered = (text or "").lower()
    return int(any(term in lowered for term in RISKY_TERMS))


def load_and_prepare_dataset(dataset_path: Path, preprocessor: MedicalTextPreprocessor) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)

    for column in ("description", "transcription", "keywords"):
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)

    # Combine core narrative columns into a single medical report text.
    df["raw_text"] = (
        df["description"].str.strip()
        + " "
        + df["transcription"].str.strip()
        + " "
        + df["keywords"].str.strip()
    ).str.strip()
    df = df[df["raw_text"].str.len() > 0].copy()

    df["clean_text"] = df["raw_text"].apply(preprocessor.preprocess)
    df = df[df["clean_text"].str.len() > 0].copy()

    df["risk_label"] = df["raw_text"].apply(derive_binary_risk_label).astype(int)
    if df["risk_label"].nunique() < 2:
        raise ValueError("Derived NLP labels are not separable; dataset produced a single class.")
    return df.reset_index(drop=True)

