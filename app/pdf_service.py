from __future__ import annotations

import os

import pdfplumber


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be parsed safely."""


def extract_pdf_text(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise PDFExtractionError("Uploaded file was not found.")

    pages = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text.strip())
    except Exception as exc:
        raise PDFExtractionError("Unable to extract text from the PDF.") from exc

    extracted_text = "\n\n".join(pages).strip()
    if not extracted_text:
        raise PDFExtractionError("The uploaded PDF did not contain readable text.")

    return extracted_text
