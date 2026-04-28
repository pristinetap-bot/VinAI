from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from functools import wraps
from io import BytesIO
from pathlib import Path
from time import time

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.utils import secure_filename

from app.ai_service import AIProcessingError, analyze_vehicle_report, answer_follow_up_question
from app.pdf_service import PDFExtractionError, extract_pdf_text
from app.report_import_service import ReportImportError, import_report_from_url

main_bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"pdf"}
MAX_CHAT_TURNS = 3
RETENTION_HOURS = 24
RETENTION_SECONDS = RETENTION_HOURS * 60 * 60
STARTING_USAGE_COUNT = 357
FREE_ANALYSIS_LIMIT = 1000


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def uploads_path_for(file_id: str, extension: str) -> str:
    return os.path.join(current_app.config["UPLOAD_FOLDER"], f"{file_id}.{extension}")


def usage_stats_path() -> str:
    return os.path.join(current_app.config["UPLOAD_FOLDER"], "usage_stats.json")


def usage_count() -> int:
    path = usage_stats_path()
    if not os.path.exists(path):
        return STARTING_USAGE_COUNT

    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        extra_uses = int(payload.get("analysis_count", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        extra_uses = 0

    return STARTING_USAGE_COUNT + max(0, extra_uses)


def remaining_free_analyses() -> int:
    return max(0, FREE_ANALYSIS_LIMIT - usage_count())


def free_launch_active() -> bool:
    return usage_count() < FREE_ANALYSIS_LIMIT


def increment_usage_count() -> None:
    path = usage_stats_path()
    extra_uses = 0
    pdf_upload_count = 0
    link_import_count = 0

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            extra_uses = int(payload.get("analysis_count", 0))
            pdf_upload_count = int(payload.get("pdf_upload_count", 0))
            link_import_count = int(payload.get("link_import_count", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            extra_uses = 0
            pdf_upload_count = 0
            link_import_count = 0

    source_type = session.pop("pending_analysis_source", "pdf")
    if source_type == "link":
        link_import_count += 1
    else:
        pdf_upload_count += 1

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "analysis_count": max(0, extra_uses) + 1,
                "pdf_upload_count": max(0, pdf_upload_count),
                "link_import_count": max(0, link_import_count),
                "last_analysis_at": int(time()),
            },
            file,
        )


def usage_stats() -> dict[str, int | None]:
    path = usage_stats_path()
    stats = {
        "analysis_count": max(0, usage_count() - STARTING_USAGE_COUNT),
        "pdf_upload_count": 0,
        "link_import_count": 0,
        "last_analysis_at": None,
    }
    if not os.path.exists(path):
        return stats

    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        stats["analysis_count"] = max(0, int(payload.get("analysis_count", 0)))
        stats["pdf_upload_count"] = max(0, int(payload.get("pdf_upload_count", 0)))
        stats["link_import_count"] = max(0, int(payload.get("link_import_count", 0)))
        last_analysis_at = payload.get("last_analysis_at")
        stats["last_analysis_at"] = int(last_analysis_at) if last_analysis_at else None
    except (OSError, ValueError, json.JSONDecodeError):
        pass

    return stats


def report_timestamp(file_id: str) -> float | None:
    pdf_path = uploads_path_for(file_id, "pdf")
    txt_path = uploads_path_for(file_id, "txt")
    json_path = uploads_path_for(file_id, "json")

    if os.path.exists(pdf_path):
        return os.path.getmtime(pdf_path)
    if os.path.exists(txt_path):
        return os.path.getmtime(txt_path)
    if os.path.exists(json_path):
        return os.path.getmtime(json_path)
    return None


def remove_report_files(file_id: str) -> None:
    for extension in ("pdf", "txt", "json"):
        path = uploads_path_for(file_id, extension)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def cleanup_expired_reports() -> None:
    cutoff = time() - RETENTION_SECONDS
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    file_ids: set[str] = set()

    for filename in os.listdir(upload_dir):
        if filename == "usage_stats.json":
            continue
        path = os.path.join(upload_dir, filename)
        if not os.path.isfile(path):
            continue
        file_id, extension = os.path.splitext(filename)
        if extension.lower() not in {".pdf", ".txt", ".json"}:
            continue
        file_ids.add(file_id)

    for file_id in file_ids:
        timestamp = report_timestamp(file_id)
        if timestamp is not None and timestamp < cutoff:
            remove_report_files(file_id)


def is_report_expired(file_id: str) -> bool:
    timestamp = report_timestamp(file_id)
    if timestamp is None:
        return True
    return timestamp < (time() - RETENTION_SECONDS)


def format_timestamp(timestamp: float | int | None) -> str:
    if not timestamp:
        return "Unavailable"
    return datetime.fromtimestamp(timestamp).strftime("%b %d, %Y at %I:%M %p")


def collect_active_reports() -> list[dict[str, object]]:
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    reports: list[dict[str, object]] = []
    file_ids: set[str] = set()

    for filename in os.listdir(upload_dir):
        if filename == "usage_stats.json":
            continue
        path = os.path.join(upload_dir, filename)
        if not os.path.isfile(path):
            continue
        file_id, extension = os.path.splitext(filename)
        if extension.lower() not in {".pdf", ".txt", ".json"}:
            continue
        file_ids.add(file_id)

    for file_id in file_ids:
        if is_report_expired(file_id):
            continue

        json_path = uploads_path_for(file_id, "json")
        source_path = report_source_path(file_id)
        timestamp = report_timestamp(file_id)
        source_type = "Imported link" if source_path and source_path.endswith(".txt") else "PDF upload"
        report_data: dict[str, object] = {}

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as file:
                    report_data = json.load(file)
            except (OSError, ValueError, json.JSONDecodeError):
                report_data = {}

        chat_history = report_data.get("chat_history", []) if isinstance(report_data, dict) else []
        chat_turns = len(chat_history) if isinstance(chat_history, list) else 0
        status = report_data.get("status", "processing") if isinstance(report_data, dict) else "processing"
        score = report_data.get("score") if isinstance(report_data, dict) else None
        verdict = report_data.get("verdict") if isinstance(report_data, dict) else None

        reports.append(
            {
                "file_id": file_id,
                "source_type": source_type,
                "status": status,
                "score": score if isinstance(score, int) else None,
                "verdict": verdict if isinstance(verdict, str) else "Processing",
                "chat_turns": chat_turns,
                "created_label": format_timestamp(timestamp),
                "timestamp": timestamp or 0,
                "summary": report_data.get("summary", "") if isinstance(report_data, dict) else "",
                "result_url": url_for("main.result", file_id=file_id),
            }
        )

    reports.sort(key=lambda item: item["timestamp"], reverse=True)
    return reports


def dashboard_metrics() -> dict[str, object]:
    reports = collect_active_reports()
    usage = usage_stats()
    completed_reports = [report for report in reports if report["status"] == "completed"]
    failed_reports = [report for report in reports if report["status"] == "failed"]
    reports_with_chat = [report for report in reports if int(report["chat_turns"]) > 0]
    score_values = [int(report["score"]) for report in completed_reports if isinstance(report["score"], int)]

    return {
        "usage_count": usage_count(),
        "remaining_free_analyses": remaining_free_analyses(),
        "free_analysis_limit": FREE_ANALYSIS_LIMIT,
        "free_launch_active": free_launch_active(),
        "analyses_used_after_launch": usage["analysis_count"],
        "pdf_upload_count": usage["pdf_upload_count"],
        "link_import_count": usage["link_import_count"],
        "last_analysis_label": format_timestamp(usage["last_analysis_at"]),
        "active_reports_count": len(reports),
        "completed_reports_count": len(completed_reports),
        "failed_reports_count": len(failed_reports),
        "reports_with_chat_count": len(reports_with_chat),
        "chat_turns_count": sum(int(report["chat_turns"]) for report in reports),
        "average_score": round(sum(score_values) / len(score_values)) if score_values else None,
        "reports": reports[:20],
    }


def admin_configured() -> bool:
    return bool(current_app.config.get("ADMIN_PASSWORD"))


def admin_logged_in() -> bool:
    return session.get("admin_authenticated") is True


def admin_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not admin_logged_in():
            return redirect(url_for("main.admin_login"))
        return view(*args, **kwargs)

    return wrapped_view


def analysis_pdf_filename(file_id: str) -> str:
    return f"cartrustai-analysis-{file_id}.pdf"


def build_analysis_pdf(file_id: str, result_data: dict[str, object]) -> BytesIO:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    left = 54
    right = width - 54
    top = height - 54
    bottom = 54
    line_height = 16

    heading_color = HexColor("#0f172a")
    accent_color = HexColor("#0f766e")
    body_color = HexColor("#334155")
    muted_color = HexColor("#64748b")

    y = top

    def new_page() -> None:
        nonlocal y
        pdf.showPage()
        y = top

    def ensure_space(lines_needed: int = 2) -> None:
        nonlocal y
        if y - (lines_needed * line_height) < bottom:
            new_page()

    def wrap_text(text: str, font_name: str, font_size: int, max_width: float) -> list[str]:
        words = (text or "").split()
        if not words:
            return [""]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if stringWidth(candidate, font_name, font_size) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def draw_kicker(text: str) -> None:
        nonlocal y
        ensure_space(2)
        pdf.setFillColor(accent_color)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(left, y, text.upper())
        y -= 24

    def draw_title(text: str) -> None:
        nonlocal y
        ensure_space(4)
        pdf.setFillColor(heading_color)
        pdf.setFont("Helvetica-Bold", 24)
        for line in wrap_text(text, "Helvetica-Bold", 24, right - left):
            pdf.drawString(left, y, line)
            y -= 30
        y -= 6

    def draw_meta(label: str, value: str) -> None:
        nonlocal y
        ensure_space(2)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.setFillColor(muted_color)
        pdf.drawString(left, y, f"{label}:")
        pdf.setFont("Helvetica", 11)
        pdf.setFillColor(body_color)
        pdf.drawString(left + 70, y, value)
        y -= 18

    def draw_section(title: str, content: str | None = None, items: list[str] | None = None) -> None:
        nonlocal y
        section_items = [item.strip() for item in (items or []) if item and item.strip()]
        if not content and not section_items:
            return

        ensure_space(3)
        pdf.setFont("Helvetica-Bold", 15)
        pdf.setFillColor(heading_color)
        pdf.drawString(left, y, title)
        y -= 20

        pdf.setFont("Helvetica", 11)
        pdf.setFillColor(body_color)

        if content:
            for line in wrap_text(content.strip(), "Helvetica", 11, right - left):
                ensure_space(1)
                pdf.drawString(left, y, line)
                y -= line_height
            y -= 4

        for item in section_items:
            bullet_indent = left + 12
            wrapped = wrap_text(item, "Helvetica", 11, right - bullet_indent)
            ensure_space(len(wrapped) + 1)
            pdf.drawString(left, y, "•")
            pdf.drawString(bullet_indent, y, wrapped[0])
            y -= line_height
            for line in wrapped[1:]:
                pdf.drawString(bullet_indent, y, line)
                y -= line_height
        y -= 10

    draw_kicker("CarTrustAI analysis")
    draw_title("Vehicle History Analysis")
    draw_meta("Report ID", file_id)
    draw_meta("Generated", datetime.now().strftime("%b %d, %Y at %I:%M %p"))
    draw_meta("Score", f"{result_data.get('score', 0)}/100")
    draw_meta("Verdict", str(result_data.get("verdict", "CAUTION")))
    y -= 8

    draw_section("Bottom line", content=str(result_data.get("bottom_line") or result_data.get("summary") or ""))
    draw_section("Summary", content=str(result_data.get("summary") or ""))
    draw_section("Price insight", content=str(result_data.get("price_insight") or ""))
    draw_section("Price guidance", content=str(result_data.get("price_guidance") or result_data.get("fair_price_assessment") or ""))
    draw_section("Top reasons", items=result_data.get("top_reasons") if isinstance(result_data.get("top_reasons"), list) else [])
    draw_section("Why this matters", items=result_data.get("why_it_matters") if isinstance(result_data.get("why_it_matters"), list) else [])
    draw_section("Major deal breakers", items=result_data.get("major_deal_breakers") if isinstance(result_data.get("major_deal_breakers"), list) else [])
    draw_section("Needs inspection", items=result_data.get("needs_inspection") if isinstance(result_data.get("needs_inspection"), list) else [])
    draw_section("Negotiation leverage", items=result_data.get("negotiation_leverage") if isinstance(result_data.get("negotiation_leverage"), list) else [])
    draw_section("Inspection checklist", items=result_data.get("inspection_checklist") if isinstance(result_data.get("inspection_checklist"), list) else [])
    draw_section("Who should avoid this car", items=result_data.get("who_should_avoid") if isinstance(result_data.get("who_should_avoid"), list) else [])
    draw_section("Ask the dealer", items=result_data.get("dealer_questions") if isinstance(result_data.get("dealer_questions"), list) else [])
    draw_section("Mechanic focus", items=result_data.get("mechanic_focus") if isinstance(result_data.get("mechanic_focus"), list) else [])
    draw_section("Key risks", items=result_data.get("risks") if isinstance(result_data.get("risks"), list) else [])
    draw_section("Confidence note", content=str(result_data.get("confidence_note") or "This analysis is based on report history and should be paired with an independent inspection."))

    pdf.save()
    buffer.seek(0)
    return buffer


def process_file(file_id: str) -> None:
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
        report_text = extract_report_text_for_file(file_id)
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
    cleanup_expired_reports()
    return render_template(
        "index.html",
        retention_hours=RETENTION_HOURS,
        usage_count=usage_count(),
        remaining_free_analyses=remaining_free_analyses(),
        free_analysis_limit=FREE_ANALYSIS_LIMIT,
        free_launch_active=free_launch_active(),
    )


@main_bp.route("/test", methods=["GET"])
def test_index():
    cleanup_expired_reports()
    return render_template(
        "test.html",
        retention_hours=RETENTION_HOURS,
        usage_count=usage_count(),
        remaining_free_analyses=remaining_free_analyses(),
        free_analysis_limit=FREE_ANALYSIS_LIMIT,
        free_launch_active=free_launch_active(),
    )


@main_bp.route("/upload", methods=["POST"])
def upload_file():
    cleanup_expired_reports()
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
    session["pending_analysis_source"] = "pdf"

    return jsonify({"file_id": file_id})


def report_source_path(file_id: str) -> str | None:
    for extension in ("pdf", "txt"):
        path = uploads_path_for(file_id, extension)
        if os.path.exists(path):
            return path
    return None


def extract_report_text_for_file(file_id: str) -> str:
    source_path = report_source_path(file_id)
    if source_path is None:
        raise PDFExtractionError("Uploaded file was not found.")

    if source_path.endswith(".pdf"):
        return extract_pdf_text(source_path)

    with open(source_path, "r", encoding="utf-8") as file:
        extracted_text = file.read().strip()

    if not extracted_text:
        raise PDFExtractionError("The imported report page did not contain readable text.")

    return extracted_text


@main_bp.route("/import-link", methods=["POST"])
def import_link():
    cleanup_expired_reports()
    payload = request.get_json(silent=True) or {}
    report_url = str(payload.get("report_url", "")).strip()

    if not report_url:
        return jsonify({"error": "Please paste a report link."}), 400

    file_id = str(uuid.uuid4())
    destination_base_path = os.path.join(current_app.config["UPLOAD_FOLDER"], file_id)

    try:
        imported_path = import_report_from_url(report_url, destination_base_path)
    except ReportImportError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Unable to import that link right now."}), 500

    session["pending_analysis_source"] = "link"

    return jsonify(
        {
            "file_id": file_id,
            "source_type": Path(imported_path).suffix.lstrip("."),
        }
    )


@main_bp.route("/analyze", methods=["POST"])
def analyze_file():
    cleanup_expired_reports()
    payload = request.get_json(silent=True) or {}
    file_id = str(payload.get("file_id", "")).strip()

    if not file_id:
        return jsonify({"error": "Missing file_id."}), 400

    if not free_launch_active():
        return jsonify({"error": "The first 1,000 free analyses have been claimed."}), 403

    source_path = report_source_path(file_id)
    if source_path is None:
        return jsonify({"error": "Uploaded file not found."}), 404

    process_file(file_id)
    increment_usage_count()
    return jsonify({"file_id": file_id, "result_url": url_for("main.result", file_id=file_id)})


@main_bp.route("/result/<file_id>", methods=["GET"])
def result(file_id: str):
    cleanup_expired_reports()
    if is_report_expired(file_id):
        return render_template(
            "result.html",
            file_id=file_id,
            processing=False,
            result=None,
            expired=True,
            retention_hours=RETENTION_HOURS,
        )

    json_path = uploads_path_for(file_id, "json")
    if not os.path.exists(json_path):
        return render_template(
            "result.html",
            file_id=file_id,
            processing=True,
            result=None,
            expired=False,
            retention_hours=RETENTION_HOURS,
        )

    with open(json_path, "r", encoding="utf-8") as file:
        result_data = json.load(file)

    return render_template(
        "result.html",
        file_id=file_id,
        processing=False,
        result=result_data,
        expired=False,
        retention_hours=RETENTION_HOURS,
    )


@main_bp.route("/result/<file_id>/download", methods=["GET"])
def download_result_pdf(file_id: str):
    cleanup_expired_reports()
    if is_report_expired(file_id):
        return redirect(url_for("main.result", file_id=file_id))

    json_path = uploads_path_for(file_id, "json")
    if not os.path.exists(json_path):
        return redirect(url_for("main.result", file_id=file_id))

    with open(json_path, "r", encoding="utf-8") as file:
        result_data = json.load(file)

    pdf_buffer = build_analysis_pdf(file_id, result_data)
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=analysis_pdf_filename(file_id),
    )


@main_bp.route("/chat/<file_id>", methods=["POST"])
def chat(file_id: str):
    cleanup_expired_reports()
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    if is_report_expired(file_id):
        return jsonify({"error": "This analysis expired after 24 hours. Please upload the report again."}), 410

    json_path = uploads_path_for(file_id, "json")
    source_path = report_source_path(file_id)
    if not os.path.exists(json_path) or source_path is None:
        return jsonify({"error": "Report not found."}), 404

    with open(json_path, "r", encoding="utf-8") as file:
        result_data = json.load(file)

    chat_history = result_data.get("chat_history", [])
    if len(chat_history) >= MAX_CHAT_TURNS:
        return jsonify({"error": "Follow-up limit reached.", "remaining": 0}), 400

    try:
        report_text = extract_report_text_for_file(file_id)
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


@main_bp.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    cleanup_expired_reports()
    error = ""

    if request.method == "POST":
        username = str(request.form.get("username", "")).strip()
        password = str(request.form.get("password", ""))

        if not admin_configured():
            error = "Set ADMIN_PASSWORD in your environment before using the admin portal."
        elif (
            username == current_app.config["ADMIN_USERNAME"]
            and password == current_app.config["ADMIN_PASSWORD"]
        ):
            session["admin_authenticated"] = True
            return redirect(url_for("main.admin_dashboard"))
        else:
            error = "Invalid admin credentials."

    return render_template(
        "admin_login.html",
        error=error,
        admin_username=current_app.config["ADMIN_USERNAME"],
        admin_configured=admin_configured(),
    )


@main_bp.route("/admin/logout", methods=["POST"])
@admin_required
def admin_logout():
    session.pop("admin_authenticated", None)
    return redirect(url_for("main.admin_login"))


@main_bp.route("/admin", methods=["GET"])
@admin_required
def admin_dashboard():
    cleanup_expired_reports()
    return render_template("admin.html", metrics=dashboard_metrics())
