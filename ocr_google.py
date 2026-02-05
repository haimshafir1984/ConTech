from __future__ import annotations

import io
import os
import json
from typing import List, Dict, Any, Optional

from google.cloud import vision
from pdf2image import convert_from_bytes


def load_gcp_credentials():
    """טוען Google Cloud credentials ממשתני סביבה או Streamlit secrets"""

    # אופציה 1: JSON מלא במשתנה אחד (Render)
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
        try:
            credentials_json = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]

            # שמירה כקובץ זמני
            temp_path = "/tmp/gcp_credentials.json"
            with open(temp_path, "w") as f:
                f.write(credentials_json)

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
            print("✅ GCP credentials loaded from GOOGLE_APPLICATION_CREDENTIALS_JSON")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load from JSON env var: {e}")

    # אופציה 2: שדות נפרדים (Render - חלופי)
    if all(
        key in os.environ for key in ["GCP_TYPE", "GCP_PROJECT_ID", "GCP_PRIVATE_KEY"]
    ):
        try:
            credentials_dict = {
                "type": os.environ["GCP_TYPE"],
                "project_id": os.environ["GCP_PROJECT_ID"],
                "private_key_id": os.environ.get("GCP_PRIVATE_KEY_ID", ""),
                "private_key": os.environ["GCP_PRIVATE_KEY"].replace("\\n", "\n"),
                "client_email": os.environ.get("GCP_CLIENT_EMAIL", ""),
                "client_id": os.environ.get("GCP_CLIENT_ID", ""),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": os.environ.get("GCP_CLIENT_X509_CERT_URL", ""),
            }

            credentials_json = json.dumps(credentials_dict)
            temp_path = "/tmp/gcp_credentials.json"
            with open(temp_path, "w") as f:
                f.write(credentials_json)

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
            print("✅ GCP credentials loaded from separate env vars")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load from separate env vars: {e}")

    # אופציה 3: Streamlit Secrets (מקומי)
    try:
        import streamlit as st

        if "gcp_service_account" in st.secrets:
            credentials_dict = dict(st.secrets["gcp_service_account"])
            credentials_json = json.dumps(credentials_dict)

            temp_path = "/tmp/gcp_credentials.json"
            with open(temp_path, "w") as f:
                f.write(credentials_json)

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
            print("✅ GCP credentials loaded from Streamlit secrets")
            return True
    except Exception as e:
        print(f"ℹ️ Streamlit secrets not available: {e}")

    # אופציה 4: קובץ קיים (fallback)
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        print("✅ Using existing GOOGLE_APPLICATION_CREDENTIALS path")
        return True

    print("⚠️ No GCP credentials found - Google Vision OCR will be disabled")
    return False


# טען credentials בעת import
load_gcp_credentials()


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
        language_hints = ["he", "en"]

    # בדיקה שיש credentials
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise RuntimeError(
            "Google Cloud credentials not configured. Please set GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable."
        )

    # PDF -> Images
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

        # Document OCR
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
