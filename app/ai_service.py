from __future__ import annotations

import json
from typing import Any

from openai import OpenAI


PROMPT = """You are a professional vehicle inspector AI.
Analyze the vehicle history report and return STRICT JSON:

{
  "score": number (0-100),
  "verdict": "BUY" | "CAUTION" | "AVOID",
  "summary": "clear explanation for buyer",
  "bottom_line": "direct recommendation in 1-2 sentences",
  "price_insight": "1 sentence",
  "price_guidance": "what price discount or pricing condition would make this worth considering",
  "confidence_note": "short note about limitations of report-only analysis",
  "top_reasons": ["top 3 reasons behind the verdict"],
  "why_it_matters": ["why the key issues matter to a buyer"],
  "major_deal_breakers": ["serious red flags"],
  "needs_inspection": ["items a mechanic should inspect closely"],
  "negotiation_leverage": ["issues the buyer can use to negotiate price"],
  "inspection_checklist": ["specific pre-purchase inspection tasks"],
  "risks": ["short bullet points"],
  "who_should_avoid": ["types of buyers who should avoid this vehicle"],
  "dealer_questions": ["smart questions to ask the dealer"],
  "mechanic_focus": ["mechanic inspection priorities"],
  "fair_price_assessment": "short statement on whether the asking price would need a discount"
}

Be strict, realistic, practical, and decision-oriented.
Focus on helping the buyer decide what to do next, not just summarizing history."""


class AIProcessingError(Exception):
    """Raised when the AI request or response handling fails."""


def analyze_vehicle_report(report_text: str, api_key: str, model: str) -> dict[str, Any]:
    if not api_key:
        raise AIProcessingError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=api_key)

    try:
        output_text = request_with_responses_api(client, model, report_text)
    except AttributeError:
        try:
            output_text = request_with_chat_completions(client, model, report_text)
        except Exception as exc:
            raise AIProcessingError("OpenAI analysis failed.") from exc
    except Exception as exc:
        raise AIProcessingError("OpenAI analysis failed.") from exc

    if not output_text:
        raise AIProcessingError("OpenAI returned an empty response.")

    try:
        result = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise AIProcessingError("OpenAI returned invalid JSON.") from exc

    return normalize_analysis(result)


def request_with_responses_api(client: OpenAI, model: str, report_text: str) -> str:
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
    return getattr(response, "output_text", "") or ""


def request_with_chat_completions(client: OpenAI, model: str, report_text: str) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return valid JSON only. Do not wrap the JSON in markdown. "
                    "Do not add explanation before or after the JSON."
                ),
            },
            {
                "role": "user",
                "content": f"{PROMPT}\n\nVehicle history report text:\n{report_text[:120000]}",
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def normalize_analysis(result: dict[str, Any]) -> dict[str, Any]:
    score = result.get("score", 0)
    verdict = str(result.get("verdict", "CAUTION")).upper()
    summary = str(result.get("summary", "")).strip()
    bottom_line = str(result.get("bottom_line", "")).strip()
    price_insight = str(result.get("price_insight", "")).strip()
    price_guidance = str(result.get("price_guidance", "")).strip()
    confidence_note = str(result.get("confidence_note", "")).strip()
    fair_price_assessment = str(result.get("fair_price_assessment", "")).strip()
    risks = normalize_list(result.get("risks", []))
    top_reasons = normalize_list(result.get("top_reasons", []))
    why_it_matters = normalize_list(result.get("why_it_matters", []))
    major_deal_breakers = normalize_list(result.get("major_deal_breakers", []))
    needs_inspection = normalize_list(result.get("needs_inspection", []))
    negotiation_leverage = normalize_list(result.get("negotiation_leverage", []))
    inspection_checklist = normalize_list(result.get("inspection_checklist", []))
    who_should_avoid = normalize_list(result.get("who_should_avoid", []))
    dealer_questions = normalize_list(result.get("dealer_questions", []))
    mechanic_focus = normalize_list(result.get("mechanic_focus", []))

    if not isinstance(score, int):
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0

    score = max(0, min(100, score))

    if verdict not in {"BUY", "CAUTION", "AVOID"}:
        verdict = "CAUTION"

    return {
        "score": score,
        "verdict": verdict,
        "summary": summary,
        "bottom_line": bottom_line,
        "price_insight": price_insight,
        "price_guidance": price_guidance,
        "confidence_note": confidence_note,
        "fair_price_assessment": fair_price_assessment,
        "top_reasons": top_reasons,
        "why_it_matters": why_it_matters,
        "major_deal_breakers": major_deal_breakers,
        "needs_inspection": needs_inspection,
        "negotiation_leverage": negotiation_leverage,
        "inspection_checklist": inspection_checklist,
        "risks": risks,
        "who_should_avoid": who_should_avoid,
        "dealer_questions": dealer_questions,
        "mechanic_focus": mechanic_focus,
    }


def normalize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        value = [str(value)] if value else []
    return [str(item).strip() for item in value if str(item).strip()]
