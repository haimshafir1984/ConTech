from __future__ import annotations

import io
from typing import List, Dict, Any, Optional

from google.cloud import vision
from pdf2image import convert_from_bytes


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

    # 1) PDF -> Images (רוב ה-OCR האדריכלי עובד טוב יותר ככה)
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

        # 2) Document OCR (עדיף על text_detection במסמכים)
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
