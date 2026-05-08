from __future__ import annotations

import io
import os
from functools import lru_cache

from google import genai
from pydantic import BaseModel, Field


class GeminiMedicalReportSummary(BaseModel):
    concise_summary: str = Field(default="", description="A concise medical summary of the report.")
    suspected_conditions: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    abnormal_findings: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    recommended_follow_up: list[str] = Field(default_factory=list)
    urgency: str = Field(
        default="unknown",
        description="One of low, moderate, high, critical, or unknown.",
    )

    def as_classifier_text(self) -> str:
        sections = [
            f"summary: {self.concise_summary}",
            f"conditions: {', '.join(self.suspected_conditions)}",
            f"symptoms: {', '.join(self.symptoms)}",
            f"abnormal findings: {', '.join(self.abnormal_findings)}",
            f"medications: {', '.join(self.medications)}",
            f"risk factors: {', '.join(self.risk_factors)}",
            f"recommended follow up: {', '.join(self.recommended_follow_up)}",
            f"urgency: {self.urgency}",
        ]
        return "\n".join(section for section in sections if section.split(": ", 1)[1])


class GeminiReportAnalyzer:
    def __init__(self, model: str | None = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    @staticmethod
    def is_configured() -> bool:
        return bool(os.getenv("GEMINI_API_KEY"))

    def summarize_pdf_bytes(self, pdf_bytes: bytes, filename: str = "report.pdf") -> GeminiMedicalReportSummary:
        uploaded_file = self.client.files.upload(
            file=io.BytesIO(pdf_bytes),
            config={"mime_type": "application/pdf", "display_name": filename},
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    "You are reading a medical report PDF. Summarize only the medical content. "
                    "Extract likely conditions, symptoms, abnormal findings, medications, risk factors, "
                    "urgency, and recommended follow-up. Keep the summary factual and concise.",
                    uploaded_file,
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GeminiMedicalReportSummary,
                    "temperature": 0.1,
                },
            )
            if response.parsed is None:
                raise RuntimeError("Gemini returned an empty structured response.")
            return response.parsed
        finally:
            try:
                self.client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


@lru_cache(maxsize=1)
def get_gemini_report_analyzer() -> GeminiReportAnalyzer:
    return GeminiReportAnalyzer()
