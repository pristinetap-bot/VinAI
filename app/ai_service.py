from __future__ import annotations

import json
from typing import Any

from openai import OpenAI


PROMPT = """You are a professional vehicle inspector AI.
Analyze the vehicle history report and return STRICT JSON:

{
  "score": number (0-100),
  "verdict": "BUY" | "CAUTION" | "AVOID",
  "risks": ["short bullet points"],
  "price_insight": "1 sentence",
  "summary": "clear explanation for buyer"
}

Be strict, realistic, and practical."""


class AIProcessingError(Exception):
    """Raised when the AI request or response handling fails."""


def analyze_vehicle_report(report_text: str, api_key: str, model: str) -> dict[str, Any]:
    if not api_key:
        raise AIProcessingError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Return valid JSON only. Do not wrap the JSON in markdown. "
                                "Do not add explanation before or after the JSON."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": PROMPT},
                        {"type": "input_text", "text": report_text[:120000]},
                    ],
                },
            ],
            temperature=0.2,
        )
    except Exception as exc:
        raise AIProcessingError("OpenAI analysis failed.") from exc

    output_text = getattr(response, "output_text", "") or ""
    if not output_text:
        raise AIProcessingError("OpenAI returned an empty response.")

    try:
        result = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise AIProcessingError("OpenAI returned invalid JSON.") from exc

    return normalize_analysis(result)


def normalize_analysis(result: dict[str, Any]) -> dict[str, Any]:
    score = result.get("score", 0)
    verdict = str(result.get("verdict", "CAUTION")).upper()
    risks = result.get("risks", [])
    price_insight = str(result.get("price_insight", "")).strip()
    summary = str(result.get("summary", "")).strip()

    if not isinstance(score, int):
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0

    score = max(0, min(100, score))

    if verdict not in {"BUY", "CAUTION", "AVOID"}:
        verdict = "CAUTION"

    if not isinstance(risks, list):
        risks = [str(risks)] if risks else []

    risks = [str(item).strip() for item in risks if str(item).strip()]

    return {
        "score": score,
        "verdict": verdict,
        "risks": risks,
        "price_insight": price_insight,
        "summary": summary,
    }
