from __future__ import annotations

import os
import re
from html import unescape
from urllib.parse import urljoin, urlparse

import requests


MAX_IMPORT_BYTES = 20 * 1024 * 1024
REQUEST_TIMEOUT = 20
USER_AGENT = "VinAIReportImporter/1.0"


class ReportImportError(Exception):
    """Raised when a remote report cannot be imported safely."""


def import_report_from_url(report_url: str, destination_base_path: str) -> str:
    candidate_urls = build_candidate_urls(report_url)
    errors: list[str] = []

    for candidate_url in candidate_urls:
        try:
            return fetch_and_store_report(candidate_url, destination_base_path)
        except ReportImportError as exc:
            errors.append(str(exc))

    error_message = errors[-1] if errors else "Unable to import the report from that link."
    raise ReportImportError(error_message)


def build_candidate_urls(report_url: str) -> list[str]:
    extracted_urls = extract_urls_from_text(report_url)
    if not extracted_urls:
        raise ReportImportError("Please paste a valid http:// or https:// report link.")

    candidates: list[str] = []
    for extracted_url in extracted_urls:
        parsed = urlparse(extracted_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue

        if "vinfax.co" in parsed.netloc and "/reports/view/" in parsed.path:
            candidates.append(extracted_url.replace("/reports/view/", "/reports/pdf/", 1))

        candidates.append(extracted_url)

    if not candidates:
        raise ReportImportError("Please paste a valid http:// or https:// report link.")

    return dedupe_preserve_order(candidates)


def fetch_and_store_report(report_url: str, destination_base_path: str) -> str:
    response = requests.get(
        report_url,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        allow_redirects=True,
    )

    if response.status_code >= 400:
        raise ReportImportError("We couldn't open that report link.")

    ensure_size_limit(response.content)

    if is_pdf_response(report_url, response):
        return write_binary_report(destination_base_path, response.content)

    content_type = (response.headers.get("Content-Type") or "").lower()
    if "html" not in content_type and "text" not in content_type:
        raise ReportImportError("That link did not return a supported report format.")

    html = response.text
    pdf_link = find_pdf_link(html, response.url)
    if pdf_link:
        pdf_response = requests.get(
            pdf_link,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
        if pdf_response.status_code >= 400:
            raise ReportImportError("We found a PDF link, but couldn't download it.")
        ensure_size_limit(pdf_response.content)
        if is_pdf_response(pdf_link, pdf_response):
            return write_binary_report(destination_base_path, pdf_response.content)

    extracted_text = extract_text_from_html(html)
    if len(extracted_text) < 300:
        raise ReportImportError(
            "We support direct PDF links best. That report page did not expose enough readable content to analyze."
        )

    return write_text_report(destination_base_path, extracted_text)


def ensure_size_limit(content: bytes) -> None:
    if len(content) > MAX_IMPORT_BYTES:
        raise ReportImportError("That report is too large. Please use a file or link under 20MB.")


def is_pdf_response(report_url: str, response: requests.Response) -> bool:
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "application/pdf" in content_type:
        return True
    if response.content.startswith(b"%PDF"):
        return True
    path = urlparse(report_url).path.lower()
    return path.endswith(".pdf") or "/reports/pdf/" in path


def find_pdf_link(html: str, base_url: str) -> str | None:
    pattern = re.compile(r"""(?:href|src)=["']([^"']+)["']""", re.IGNORECASE)
    for match in pattern.findall(html):
        candidate = urljoin(base_url, match)
        path = urlparse(candidate).path.lower()
        if path.endswith(".pdf") or "/reports/pdf/" in path:
            return candidate
    return None


def extract_text_from_html(html: str) -> str:
    cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?i)</p>", "\n", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def write_binary_report(destination_base_path: str, content: bytes) -> str:
    path = f"{destination_base_path}.pdf"
    with open(path, "wb") as file:
        file.write(content)
    return path


def write_text_report(destination_base_path: str, content: str) -> str:
    path = f"{destination_base_path}.txt"
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)
    return path


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items


def extract_urls_from_text(raw_text: str) -> list[str]:
    return re.findall(r"https?://[^\s]+", raw_text.strip())
