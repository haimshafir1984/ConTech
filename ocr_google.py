from __future__ import annotations

import io
import os
import json
from typing import List, Dict, Any, Optional

from google.cloud import vision
from pdf2image import convert_from_bytes

# ==========================================
# טעינת credentials מ-Streamlit Secrets
# ==========================================
try:
    import streamlit as st

    # בדיקה אם יש credentials ב-secrets
    if "gcp_service_account" in st.secrets:
        credentials_dict = dict(st.secrets["gcp_service_account"])
        credentials_json = json.dumps(credentials_dict)

        # שמירה כקובץ זמני
        temp_path = "/tmp/gcp_credentials.json"
        with open(temp_path, "w") as f:
            f.write(credentials_json)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
        print("✅ Google Cloud credentials loaded from Streamlit secrets")
    else:
        print("⚠️ No gcp_service_account found in secrets.toml")

except Exception as e:
    print(f"⚠️ Failed to load GCP credentials from Streamlit: {e}")
    # Fallback למשתנה סביבה רגיל (production)
    pass


def ocr_pdf_google_vision(
    pdf_bytes: bytes,
    *,
    dpi: int = 300,
    language_hints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    OCR ל-PDF באמצעות Google Vision.
    מחזיר:
      - full_text: טקסט מאוחד
      - pages: רשימת דפים (טקסט לכל דף)
    """
    if language_hints is None:
        # עברית + אנגלית נפוץ בתכניות
        language_hints = ["he", "en"]

    # 1) PDF -> Images
    images = convert_from_bytes(pdf_bytes, dpi=dpi)

    client = vision.ImageAnnotatorClient()

    pages_text: List[str] = []
    full_text_parts: List[str] = []

    image_context = vision.ImageContext(language_hints=language_hints)

    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        content = buf.getvalue()

        gimg = vision.Image(content=content)

        # 2) Document OCR
        resp = client.document_text_detection(image=gimg, image_context=image_context)

        if resp.error.message:
            raise RuntimeError(f"Google Vision OCR error: {resp.error.message}")

        page_text = resp.full_text_annotation.text or ""
        pages_text.append(page_text)
        full_text_parts.append(page_text)

    return {
        "full_text": "\n\n".join(full_text_parts).strip(),
        "pages": pages_text,
        "dpi": dpi,
        "language_hints": language_hints,
    }
