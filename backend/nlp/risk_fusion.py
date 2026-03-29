from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FinalRiskResult:
    final_score: float
    risk_category: str
    confidence: float


def classify_final_risk(final_score: float) -> str:
    if final_score < 0.4:
        return "NORMAL"
    if final_score < 0.7:
        return "WARNING"
    return "CRITICAL"


def combine_scores(physiological_score: float, nlp_score: float, nlp_confidence: float) -> FinalRiskResult:
    physiological_score = max(0.0, min(1.0, float(physiological_score)))
    nlp_score = max(0.0, min(1.0, float(nlp_score)))
    nlp_confidence = max(0.0, min(1.0, float(nlp_confidence)))

    final_score = (0.6 * physiological_score) + (0.4 * nlp_score)
    category = classify_final_risk(final_score)

    # Weighted confidence that prioritizes NLP certainty and score stability.
    score_margin_conf = abs(final_score - 0.5) * 2.0
    confidence = ((0.7 * nlp_confidence) + (0.3 * score_margin_conf)) * 100.0
    confidence = round(max(0.0, min(100.0, confidence)), 2)

    return FinalRiskResult(
        final_score=round(final_score, 4),
        risk_category=category,
        confidence=confidence,
    )

