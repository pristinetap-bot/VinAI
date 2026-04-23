from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import stripe
from flask import Blueprint, current_app, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from app.ai_service import AIProcessingError, analyze_vehicle_report
from app.pdf_service import PDFExtractionError, extract_pdf_text

main_bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"pdf"}
PRICE_IN_CENTS = 300


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def uploads_path_for(file_id: str, extension: str) -> str:
    return os.path.join(current_app.config["UPLOAD_FOLDER"], f"{file_id}.{extension}")


def build_external_url(endpoint: str, **values: str) -> str:
    base_url = current_app.config.get("APP_BASE_URL", "").strip().rstrip("/")
    if base_url:
        return f"{base_url}{url_for(endpoint, **values)}"
    return url_for(endpoint, _external=True, **values)


def process_file(file_id: str) -> None:
    pdf_path = uploads_path_for(file_id, "pdf")
    json_path = uploads_path_for(file_id, "json")

    analysis = {
        "status": "failed",
        "score": 0,
        "verdict": "CAUTION",
        "risks": [],
        "price_insight": "",
        "summary": "We couldn't finish processing this report.",
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


@main_bp.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    payload = request.get_json(silent=True) or {}
    file_id = str(payload.get("file_id", "")).strip()

    if not file_id:
        return jsonify({"error": "Missing file_id."}), 400

    pdf_path = Path(uploads_path_for(file_id, "pdf"))
    if not pdf_path.exists():
        return jsonify({"error": "Uploaded file not found."}), 404

    stripe_key = current_app.config["STRIPE_SECRET_KEY"]
    if not stripe_key:
        return jsonify({"error": "Stripe is not configured."}), 500

    stripe.api_key = stripe_key

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": "AI Car Report Analysis"},
                        "unit_amount": PRICE_IN_CENTS,
                    },
                    "quantity": 1,
                }
            ],
            metadata={"file_id": file_id},
            success_url=build_external_url("main.result", file_id=file_id),
            cancel_url=build_external_url("main.index"),
        )
    except Exception:
        return jsonify({"error": "Unable to create Stripe Checkout session."}), 500

    return jsonify({"checkout_url": session.url})


@main_bp.route("/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.get_data(as_text=False)
    signature = request.headers.get("Stripe-Signature", "")
    webhook_secret = current_app.config["STRIPE_WEBHOOK_SECRET"]

    try:
        if webhook_secret:
            event = stripe.Webhook.construct_event(payload, signature, webhook_secret)
        else:
            event = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid webhook payload."}), 400

    if event.get("type") == "checkout.session.completed":
        session = event["data"]["object"]
        file_id = session.get("metadata", {}).get("file_id")
        if file_id:
            process_file(file_id)

    return jsonify({"received": True})


@main_bp.route("/result/<file_id>", methods=["GET"])
def result(file_id: str):
    json_path = uploads_path_for(file_id, "json")
    if not os.path.exists(json_path):
        return render_template("result.html", file_id=file_id, processing=True, result=None)

    with open(json_path, "r", encoding="utf-8") as file:
        result_data = json.load(file)

    return render_template("result.html", file_id=file_id, processing=False, result=result_data)


@main_bp.route("/health", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"})
