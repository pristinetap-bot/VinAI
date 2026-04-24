from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from app.ai_service import AIProcessingError, analyze_vehicle_report, answer_follow_up_question
from app.pdf_service import PDFExtractionError, extract_pdf_text

main_bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"pdf"}
MAX_CHAT_TURNS = 3


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def uploads_path_for(file_id: str, extension: str) -> str:
    return os.path.join(current_app.config["UPLOAD_FOLDER"], f"{file_id}.{extension}")


def process_file(file_id: str) -> None:
    pdf_path = uploads_path_for(file_id, "pdf")
    json_path = uploads_path_for(file_id, "json")

    analysis = {
        "status": "failed",
        "score": 0,
        "verdict": "CAUTION",
        "summary": "We couldn't finish processing this report.",
        "bottom_line": "",
        "price_insight": "",
        "price_guidance": "",
        "confidence_note": "",
        "fair_price_assessment": "",
        "top_reasons": [],
        "why_it_matters": [],
        "major_deal_breakers": [],
        "needs_inspection": [],
        "negotiation_leverage": [],
        "inspection_checklist": [],
        "risks": [],
        "who_should_avoid": [],
        "dealer_questions": [],
        "mechanic_focus": [],
        "chat_history": [],
    }

    try:
        report_text = extract_pdf_text(pdf_path)
        result = analyze_vehicle_report(
            report_text=report_text,
            api_key=current_app.config["OPENAI_API_KEY"],
            model=current_app.config["OPENAI_MODEL"],
        )
        analysis = {"status": "completed", **result}
    except (PDFExtractionError, AIProcessingError) as exc:
        analysis["summary"] = str(exc)
    except Exception:
        analysis["summary"] = "An unexpected error occurred during report analysis."

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(analysis, file, indent=2)


@main_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@main_bp.route("/upload", methods=["POST"])
def upload_file():
    uploaded_file = request.files.get("file")
    if not uploaded_file:
        return jsonify({"error": "Please upload a PDF file."}), 400

    original_name = secure_filename(uploaded_file.filename or "")
    if not original_name or not allowed_file(original_name):
        return jsonify({"error": "Only PDF files are supported."}), 400

    mimetype = (uploaded_file.mimetype or "").lower()
    if mimetype not in {"application/pdf", "application/x-pdf"}:
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

    file_id = str(uuid.uuid4())
    save_path = uploads_path_for(file_id, "pdf")
    uploaded_file.save(save_path)

    return jsonify({"file_id": file_id})


@main_bp.route("/analyze", methods=["POST"])
def analyze_file():
    payload = request.get_json(silent=True) or {}
    file_id = str(payload.get("file_id", "")).strip()

    if not file_id:
        return jsonify({"error": "Missing file_id."}), 400

    pdf_path = Path(uploads_path_for(file_id, "pdf"))
    if not pdf_path.exists():
        return jsonify({"error": "Uploaded file not found."}), 404

    process_file(file_id)
    return jsonify({"file_id": file_id, "result_url": url_for("main.result", file_id=file_id)})


@main_bp.route("/result/<file_id>", methods=["GET"])
def result(file_id: str):
    json_path = uploads_path_for(file_id, "json")
    if not os.path.exists(json_path):
        return render_template("result.html", file_id=file_id, processing=True, result=None)

    with open(json_path, "r", encoding="utf-8") as file:
        result_data = json.load(file)

    return render_template("result.html", file_id=file_id, processing=False, result=result_data)


@main_bp.route("/chat/<file_id>", methods=["POST"])
def chat(file_id: str):
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    json_path = uploads_path_for(file_id, "json")
    pdf_path = uploads_path_for(file_id, "pdf")
    if not os.path.exists(json_path) or not os.path.exists(pdf_path):
        return jsonify({"error": "Report not found."}), 404

    with open(json_path, "r", encoding="utf-8") as file:
        result_data = json.load(file)

    chat_history = result_data.get("chat_history", [])
    if len(chat_history) >= MAX_CHAT_TURNS:
        return jsonify({"error": "Follow-up limit reached.", "remaining": 0}), 400

    try:
        report_text = extract_pdf_text(pdf_path)
        answer = answer_follow_up_question(
            report_text=report_text,
            analysis=result_data,
            question=question,
            history=chat_history,
            api_key=current_app.config["OPENAI_API_KEY"],
            model=current_app.config["OPENAI_MODEL"],
        )
    except (PDFExtractionError, AIProcessingError) as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception:
        return jsonify({"error": "Unable to answer the follow-up question."}), 500

    chat_history.append({"question": question, "answer": answer})
    result_data["chat_history"] = chat_history

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(result_data, file, indent=2)

    return jsonify({"answer": answer, "remaining": MAX_CHAT_TURNS - len(chat_history)})


@main_bp.route("/health", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"})
