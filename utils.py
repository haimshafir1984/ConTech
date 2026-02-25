import cv2
import numpy as np
import pandas as pd
from typing import Optional
from database import get_progress_reports
import streamlit as st
import traceback

# ==========================================
# Metadata Processing - גרסה משופרת
# ==========================================


def safe_process_metadata(
    raw_text=None,
    raw_text_full=None,
    normalized_text=None,
    raw_blocks=None,
    candidates=None,
    meta=None,
    pdf_bytes=None,  # ← חדש!
):
    """
    ✨ משופר: Enhanced wrapper for brain.process_plan_metadata
    תומך גם בפורמט הישן וגם החדש + Google Vision OCR
    """

    # אם קיבלנו meta dict (פורמט חדש)
    if meta and isinstance(meta, dict):
        return _safe_process_metadata_new_format(meta, pdf_bytes=pdf_bytes)

    # פורמט ישן - המשך עם הלוגיקה המקורית
    return _safe_process_metadata_old_format(
        raw_text=raw_text,
        raw_text_full=raw_text_full,
        normalized_text=normalized_text,
        raw_blocks=raw_blocks,
        candidates=candidates,
        pdf_bytes=pdf_bytes,  # ← חדש!
    )


def _safe_process_metadata_new_format(meta, pdf_bytes=None):
    """
    ✨ עיבוד פורמט חדש עם Error Handling מקיף
    """
    try:
        from brain import process_plan_metadata
    except ImportError:
        st.error("❌ שגיאה קריטית: brain.py חסר!")
        return _create_empty_schema("Brain module not found")

    try:
        has_full_text = (
            meta.get("raw_text_full") and len(meta.get("raw_text_full", "")) > 100
        )
        has_basic_text = meta.get("raw_text") and len(meta.get("raw_text", "")) > 50

        if not has_full_text and not has_basic_text:
            return {
                "status": "empty_text",
                "error": "אין טקסט זמין לניתוח",
                "document": {},
                "rooms": [],
                "heights_and_levels": {},
                "execution_notes": {},
                "limitations": ["ה-PDF לא הכיל טקסט קריא"],
                "quantities_hint": {"wall_types_mentioned": [], "material_hints": []},
            }

        # ניסיון 1: קונטקסט מלא
        if has_full_text:
            try:
                with st.spinner("🧠 מנתח מטא-דאטה עם AI..."):
                    result = process_plan_metadata(
                        meta["raw_text_full"],
                        use_google_ocr=bool(pdf_bytes),
                        pdf_bytes=pdf_bytes,
                    )
                    if result and isinstance(result, dict) and not result.get("error"):
                        result["_processing_method"] = "full_context"
                        result["_text_length"] = len(meta["raw_text_full"])
                        return _ensure_schema_format(result)
                    else:
                        st.warning("⚠️ ניתוח מלא נכשל, מנסה גרסה בסיסית...")
                        raise ValueError("Invalid result from full context")

            except Exception as e:
                st.warning(f"⚠️ ניתוח מלא נכשל: {str(e)[:100]}")

        # ניסיון 2: קונטקסט בסיסי
        if has_basic_text:
            try:
                with st.spinner("🔄 מנסה ניתוח בסיסי..."):
                    result = process_plan_metadata(
                        meta["raw_text"],
                        use_google_ocr=bool(pdf_bytes),
                        pdf_bytes=pdf_bytes,
                    )

                    if result and isinstance(result, dict):
                        result["_processing_method"] = "basic_context"
                        result["_text_length"] = len(meta["raw_text"])
                        result["_warning"] = "נותח עם טקסט חלקי בלבד"
                        return _ensure_schema_format(result)
                    else:
                        raise ValueError("Invalid result from basic context")

            except Exception as e:
                st.warning(f"⚠️ גם ניתוח בסיסי נכשל: {str(e)[:100]}")

        st.error("❌ כל שיטות הניתוח נכשלו")

    except Exception as e:
        st.error(f"❌ שגיאה לא צפויה: {str(e)}")
        with st.expander("🔍 פרטי שגיאה מלאים"):
            st.code(traceback.format_exc())

    # Fallback
    st.warning("⚠️ משתמש בערכי ברירת מחדל")
    return _create_empty_schema("כל שיטות הניתוח נכשלו")


def _safe_process_metadata_old_format(
    raw_text=None,
    raw_text_full=None,
    normalized_text=None,
    raw_blocks=None,
    candidates=None,
    pdf_bytes=None,  # ← הוסף
):
    """
    עיבוד פורמט ישן (backward compatibility) + Google Vision OCR
    """
    best_text = None

    # Priority order
    if normalized_text and normalized_text.strip():
        best_text = normalized_text
    elif raw_text_full and raw_text_full.strip():
        best_text = raw_text_full
    elif raw_text and raw_text.strip():
        best_text = raw_text
    elif raw_blocks and isinstance(raw_blocks, list):
        best_text = "\n".join(
            [
                block.get("text", "")
                for block in raw_blocks
                if isinstance(block, dict) and block.get("text")
            ]
        )
    elif candidates and isinstance(candidates, list):
        best_text = "\n".join([str(c) for c in candidates if c])

    if not best_text or not best_text.strip():
        return _create_empty_schema("No text extracted from PDF")

    try:
        from brain import process_plan_metadata

        # ← שנה את השורה הזו:
        result = process_plan_metadata(
            best_text, use_google_ocr=bool(pdf_bytes), pdf_bytes=pdf_bytes
        )

        if isinstance(result, dict):
            if "document" in result:
                return result
            else:
                return _wrap_legacy_format(result)

        return result

    except (ImportError, Exception) as e:
        return _create_empty_schema(f"Processing error: {str(e)}")


def _create_empty_schema(error_msg):
    """יוצר schema ריק עם הודעת שגיאה"""
    return {
        "status": "error",
        "error": error_msg,
        "document": {},
        "rooms": [],
        "heights_and_levels": {},
        "execution_notes": {},
        "limitations": [error_msg],
        "quantities_hint": {"wall_types_mentioned": [], "material_hints": []},
    }


def _ensure_schema_format(result):
    """מוודא שהתוצאה בפורמט הנכון"""
    if not isinstance(result, dict):
        return _create_empty_schema("Invalid result type")

    # אם חסרים שדות, נוסיף אותם
    if "document" not in result:
        result["document"] = {}
    if "rooms" not in result:
        result["rooms"] = []
    if "heights_and_levels" not in result:
        result["heights_and_levels"] = {}
    if "execution_notes" not in result:
        result["execution_notes"] = {}
    if "limitations" not in result:
        result["limitations"] = []
    if "quantities_hint" not in result:
        result["quantities_hint"] = {"wall_types_mentioned": [], "material_hints": []}

    return result


def _wrap_legacy_format(old_data):
    """
    Wraps old flat format into new schema
    """
    document = {}

    if "plan_name" in old_data:
        document["plan_title"] = {
            "value": old_data["plan_name"],
            "confidence": 50,
            "evidence": ["legacy data"],
        }

    if "scale" in old_data:
        document["scale"] = {
            "value": old_data["scale"],
            "confidence": 50,
            "evidence": ["legacy data"],
        }

    if "plan_type" in old_data:
        document["plan_type"] = {
            "value": old_data["plan_type"],
            "confidence": 50,
            "evidence": ["legacy data"],
        }

    return {
        "status": "success_legacy",
        "document": document,
        "rooms": [],
        "heights_and_levels": {},
        "execution_notes": {},
        "limitations": ["Converted from legacy format"],
        "quantities_hint": {"wall_types_mentioned": [], "material_hints": []},
    }


# ==========================================
# Legend Analysis - משופר
# ==========================================


def safe_analyze_legend(image_bytes):
    """
    ✨ משופר: ניתוח מקרא עם Error Handling + Retry logic
    """
    if not image_bytes:
        return {"error": "לא התקבלה תמונה"}

    if len(image_bytes) < 1000:
        return {"error": "התמונה קטנה מדי (פחות מ-1KB)"}

    if len(image_bytes) > 10 * 1024 * 1024:
        return {"error": "התמונה גדולה מדי (מעל 10MB)"}

    try:
        from brain import analyze_legend_image
    except ImportError:
        return {"error": "Brain module not found"}

    # ניסיון ראשון
    try:
        with st.spinner("🔍 מנתח מקרא עם AI..."):
            result = analyze_legend_image(image_bytes)

            if result and isinstance(result, dict):
                if result.get("error"):
                    st.warning("⚠️ ניסיון ראשון נכשל, מנסה שוב...")
                    raise ValueError(result["error"])
                else:
                    st.success("✅ ניתוח הושלם בהצלחה")
                    return result
            else:
                raise ValueError("Invalid result format")

    except Exception as e:
        st.warning(f"⚠️ ניסיון ראשון נכשל: {str(e)[:100]}")

        # ניסיון שני
        try:
            with st.spinner("🔄 מנסה שוב..."):
                import time

                time.sleep(1)

                result = analyze_legend_image(image_bytes)

                if result and isinstance(result, dict) and not result.get("error"):
                    st.success("✅ ניתוח הושלם בניסיון השני")
                    result["_retry_count"] = 1
                    return result
                else:
                    raise ValueError("Second attempt failed")

        except Exception as e2:
            st.error(f"❌ גם ניסיון שני נכשל: {str(e2)[:100]}")

            return {
                "error": "ניתוח נכשל פעמיים",
                "first_error": str(e)[:200],
                "second_error": str(e2)[:200],
                "_suggestion": "נסה:\n1. לחתוך את המקרא ידנית\n2. להעלות תמונה באיכות גבוהה יותר",
                "_fallback_action": "ניתן למלא את הנתונים ידנית",
            }


# ==========================================
# Utility Functions
# ==========================================


def load_stats_df():
    """טוען סטטיסטיקות עם Error Handling"""
    try:
        reports = get_progress_reports()
        if reports and len(reports) > 0:
            df = pd.DataFrame(reports)
            return df.rename(
                columns={
                    "date": "תאריך",
                    "plan_name": "שם תוכנית",
                    "meters_built": "כמות שבוצעה",
                    "note": "הערה",
                }
            )
        else:
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ שגיאה בטעינת סטטיסטיקות: {str(e)}")
        return pd.DataFrame()


def create_colored_overlay(
    original,
    concrete_mask,
    blocks_mask,
    flooring_mask=None,
    alpha=0.5,
):
    """
    ✨ משופר: יוצר תמונה צבעונית עם Error Handling
    """
    if original is None or original.size == 0:
        st.error("❌ תמונה מקורית חסרה")
        return np.zeros((500, 500, 3), dtype=np.uint8)

    try:
        # המרה ל-RGB
        if len(original.shape) == 2:
            img_vis = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB).astype(float)
        elif original.shape[2] == 4:
            img_vis = cv2.cvtColor(original, cv2.COLOR_BGRA2RGB).astype(float)
        else:
            img_vis = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(float)

        overlay = img_vis.copy()
        h, w = img_vis.shape[:2]

        # בטון
        if concrete_mask is not None and concrete_mask.size > 0:
            try:
                if concrete_mask.shape[:2] != (h, w):
                    concrete_mask = cv2.resize(
                        concrete_mask, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                overlay[concrete_mask > 0] = [30, 144, 255]
            except Exception as e:
                st.warning(f"⚠️ שגיאה בצביעת בטון: {str(e)}")

        # בלוקים
        if blocks_mask is not None and blocks_mask.size > 0:
            try:
                if blocks_mask.shape[:2] != (h, w):
                    blocks_mask = cv2.resize(
                        blocks_mask, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                overlay[blocks_mask > 0] = [255, 165, 0]
            except Exception as e:
                st.warning(f"⚠️ שגיאה בצביעת בלוקים: {str(e)}")

        # ריצוף
        if flooring_mask is not None and flooring_mask.size > 0:
            try:
                if flooring_mask.shape[:2] != (h, w):
                    flooring_mask = cv2.resize(
                        flooring_mask, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                overlay[flooring_mask > 0] = [200, 100, 255]
            except Exception as e:
                st.warning(f"⚠️ שגיאה בצביעת ריצוף: {str(e)}")

        # שילוב
        result = img_vis.copy()
        cv2.addWeighted(overlay, 0.6, img_vis, 0.4, 0, result)

        return result.astype(np.uint8)

    except Exception as e:
        st.error(f"❌ שגיאה ביצירת overlay: {str(e)}")
        if len(original.shape) == 3:
            return cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        else:
            return cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)


def calculate_area_m2(
    area_px: int,
    meters_per_pixel: Optional[float] = None,
    meters_per_pixel_x: Optional[float] = None,
    meters_per_pixel_y: Optional[float] = None,
    pixels_per_meter: Optional[float] = None,
) -> Optional[float]:
    """
    🆕 v2.2: מחשב שטח במ"ר עם בדיקות תקינות (guardrails)

    Args:
        area_px: שטח בפיקסלים
        meters_per_pixel: המרה איזוטרופית (ממוצע)
        meters_per_pixel_x: המרה בציר X
        meters_per_pixel_y: המרה בציר Y
        pixels_per_meter: היפוך של meters_per_pixel

    Returns:
        float: שטח במ"ר, או None אם:
            - אין נתוני קנה מידה
            - הערך מחוץ לטווח סביר (0.5-100,000 מ"ר)

    Note:
        Guardrails מונעים ערכים הזויים עקב:
        - scale שגוי
        - pixels שגוי
        - יחידות מידה שגויות
    """
    if area_px is None:
        return None

    # ניסיון 1: אניזוטרופי (X/Y נפרדים)
    if (
        meters_per_pixel_x is not None
        and meters_per_pixel_y is not None
        and meters_per_pixel_x > 0
        and meters_per_pixel_y > 0
    ):
        area_m2 = area_px * meters_per_pixel_x * meters_per_pixel_y

    # ניסיון 2: איזוטרופי (ממוצע)
    elif meters_per_pixel is not None and meters_per_pixel > 0:
        area_m2 = area_px * (meters_per_pixel**2)

    # ניסיון 3: fallback (pixels_per_meter)
    elif pixels_per_meter is not None and pixels_per_meter > 0:
        meters_per_pixel_fallback = 1.0 / pixels_per_meter
        area_m2 = area_px * (meters_per_pixel_fallback**2)

    else:
        # אין נתוני scale
        return None

    # =====================================
    # 🆕 v2.2: Guardrails - בדיקות תקינות
    # =====================================

    # סף תחתון: פחות מחצי מטר רבוע
    MIN_REASONABLE_AREA = 0.5  # m²

    # סף עליון: 100,000 מ"ר = 10 הקטארים (גדול מדי לבניין רגיל)
    MAX_REASONABLE_AREA = 100000  # m²

    if area_m2 < MIN_REASONABLE_AREA:
        # חשוד - קנה מידה כנראה שגוי
        # אופציה A: החזר None
        return None

        # אופציה B: החזר עם warning (אם יש logger)
        # logger.warning(f"Area too small: {area_m2:.3f} m² - check scale")
        # return area_m2

    if area_m2 > MAX_REASONABLE_AREA:
        # חשוד - קנה מידה כנראה שגוי
        return None

    # הכל בסדר
    return area_m2


def validate_measurements(geometric_meta: dict, text_data: dict) -> dict:
    """
    🆕 v2.2: משווה מדידות גיאומטריות מול נתונים מטקסט

    מטרה: לזהות שגיאות scale או חילוץ נתונים שגוי

    Args:
        geometric_meta: מטא-דאטה גיאומטרית (מ-analyzer.py)
            צריך לכלול: meters_per_pixel, pixels_flooring_area
        text_data: נתונים מחולצים מטקסט (מ-brain.py)
            צריך לכלול: rooms (רשימה של חדרים עם area_m2)

    Returns:
        dict עם:
            - calculated_area_m2: שטח מחושב מפיקסלים (או None)
            - text_area_total_m2: שטח מסכום חדרים (או None)
            - mismatch_ratio: יחס הפער (או None)
            - warnings: רשימת אזהרות (list[str])
            - status: "ok" / "mismatch" / "insufficient_data"

    Example:
        >>> result = validate_measurements(
        ...     geometric_meta={"meters_per_pixel": 0.0005, "pixels_flooring_area": 120000},
        ...     text_data={"rooms": [{"area_m2": {"value": 15.5}}, {"area_m2": {"value": 24.5}}]}
        ... )
        >>> result
        {
            "calculated_area_m2": 60.0,
            "text_area_total_m2": 40.0,
            "mismatch_ratio": 0.50,
            "warnings": ["Scale Mismatch: Text says 40.0 m², Geometry says 60.0 m²"],
            "status": "mismatch"
        }
    """
    warnings = []

    # =====================================
    # שלב 1: חלץ נתונים גיאומטריים
    # =====================================
    calculated_area_m2 = None

    if geometric_meta:
        # ניסיון למצוא שטח ריצוף בפיקסלים
        pixels_area = geometric_meta.get("pixels_flooring_area")

        if pixels_area and pixels_area > 0:
            # ניסיון לחשב שטח באמצעות קנה המידה
            meters_per_pixel = geometric_meta.get("meters_per_pixel")
            meters_per_pixel_x = geometric_meta.get("meters_per_pixel_x")
            meters_per_pixel_y = geometric_meta.get("meters_per_pixel_y")

            # שימוש בפונקציה קיימת
            calculated_area_m2 = calculate_area_m2(
                area_px=pixels_area,
                meters_per_pixel=meters_per_pixel,
                meters_per_pixel_x=meters_per_pixel_x,
                meters_per_pixel_y=meters_per_pixel_y,
            )

    # =====================================
    # שלב 2: חלץ נתונים מטקסט
    # =====================================
    text_area_total_m2 = None

    if text_data and "rooms" in text_data:
        rooms = text_data["rooms"]

        if rooms and isinstance(rooms, list):
            total = 0.0
            valid_rooms = 0

            for room in rooms:
                if not isinstance(room, dict):
                    continue

                # חיפוש שדה area_m2
                area_field = room.get("area_m2")

                if area_field:
                    # אם זה EvidenceMatch-like (dict עם "value")
                    if isinstance(area_field, dict) and "value" in area_field:
                        area_value = area_field["value"]
                    else:
                        area_value = area_field

                    # המרה למספר
                    try:
                        area_num = float(area_value) if area_value else 0.0
                        if area_num > 0:
                            total += area_num
                            valid_rooms += 1
                    except (ValueError, TypeError):
                        continue

            if valid_rooms > 0:
                text_area_total_m2 = total

    # =====================================
    # שלב 3: השוואה וחישוב פער
    # =====================================
    mismatch_ratio = None
    status = "insufficient_data"

    if calculated_area_m2 is not None and text_area_total_m2 is not None:
        # שניהם קיימים - אפשר להשוות

        if text_area_total_m2 > 0:
            # חישוב פער יחסי
            abs_diff = abs(calculated_area_m2 - text_area_total_m2)
            mismatch_ratio = abs_diff / text_area_total_m2

            # סף אזהרה: 15%
            MISMATCH_THRESHOLD = 0.15

            if mismatch_ratio > MISMATCH_THRESHOLD:
                warnings.append(
                    f"⚠️ Scale Mismatch Detected: "
                    f"Text says {text_area_total_m2:.1f} m², "
                    f"Geometry says {calculated_area_m2:.1f} m² "
                    f"(diff: {mismatch_ratio*100:.1f}%)"
                )
                status = "mismatch"
            else:
                status = "ok"
        else:
            warnings.append("Text area is zero - cannot compare")
            status = "insufficient_data"

    elif calculated_area_m2 is None and text_area_total_m2 is None:
        warnings.append("No area data available (neither geometric nor text)")
        status = "insufficient_data"

    elif calculated_area_m2 is None:
        warnings.append(
            "Geometric area unavailable (check scale and flooring detection)"
        )
        status = "insufficient_data"

    elif text_area_total_m2 is None:
        warnings.append("Text area unavailable (no rooms found with area data)")
        status = "insufficient_data"

    # =====================================
    # שלב 4: החזר תוצאות
    # =====================================
    return {
        "calculated_area_m2": calculated_area_m2,
        "text_area_total_m2": text_area_total_m2,
        "mismatch_ratio": mismatch_ratio,
        "warnings": warnings,
        "status": status,
    }


def refine_flooring_mask_with_rooms(
    flooring_mask: Optional[np.ndarray],
    room_masks: Optional[dict],
) -> Optional[np.ndarray]:
    """מצמצם מסכת ריצוף לאזורים שמוגדרים כחדרים"""
    if flooring_mask is None or room_masks is None:
        return None

    if not room_masks:
        return None

    union_mask = None
    for mask in room_masks.values():
        if mask is None:
            continue
        if union_mask is None:
            union_mask = mask.copy()
        else:
            union_mask = cv2.bitwise_or(union_mask, mask)

    if union_mask is None:
        return None

    if union_mask.shape[:2] != flooring_mask.shape[:2]:
        union_mask = cv2.resize(
            union_mask,
            (flooring_mask.shape[1], flooring_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return cv2.bitwise_and(flooring_mask, union_mask)


def format_llm_metadata(llm_data):
    """ממיר את המטא-דאטה המלא למבנה פשוט יותר לתצוגה"""
    if not llm_data or llm_data.get("status") in [
        "error",
        "no_api_key",
        "empty_text",
        "extraction_failed",
    ]:
        return {
            "document": {},
            "rooms": [],
            "heights_and_levels": {},
            "execution_notes": {},
            "limitations": llm_data.get("limitations", []) if llm_data else [],
            "quantities_hint": llm_data.get("quantities_hint", {}) if llm_data else {},
        }

    def extract_value(field_obj):
        if isinstance(field_obj, dict) and "value" in field_obj:
            return field_obj["value"]
        return field_obj

    document = {}
    if "document" in llm_data and isinstance(llm_data["document"], dict):
        for key, field in llm_data["document"].items():
            document[key] = extract_value(field)

    rooms = []
    if "rooms" in llm_data and isinstance(llm_data["rooms"], list):
        for room in llm_data["rooms"]:
            if isinstance(room, dict):
                simple_room = {}
                for key, field in room.items():
                    simple_room[key] = extract_value(field)
                rooms.append(simple_room)

    heights_and_levels = {}
    if "heights_and_levels" in llm_data and isinstance(
        llm_data["heights_and_levels"], dict
    ):
        for key, field in llm_data["heights_and_levels"].items():
            heights_and_levels[key] = extract_value(field)

    execution_notes = {}
    if "execution_notes" in llm_data and isinstance(llm_data["execution_notes"], dict):
        for key, field in llm_data["execution_notes"].items():
            execution_notes[key] = extract_value(field)

    limitations = llm_data.get("limitations", [])
    quantities_hint = llm_data.get(
        "quantities_hint", {"wall_types_mentioned": [], "material_hints": []}
    )

    formatted = {
        "document": document,
        "rooms": rooms,
        "heights_and_levels": heights_and_levels,
        "execution_notes": execution_notes,
        "limitations": limitations,
        "quantities_hint": quantities_hint,
    }

    # =====================================
    # 🆕 v2.2: הוסף validation warnings
    # =====================================
    if "geometric_meta" in llm_data and llm_data.get("geometric_meta"):
        validation_result = validate_measurements(
            geometric_meta=llm_data.get("geometric_meta", {}), text_data=llm_data
        )

        formatted["validation"] = validation_result

        if validation_result.get("warnings"):
            formatted["validation_warnings"] = validation_result["warnings"]

    return formatted


def get_simple_metadata_values(llm_data):
    """
    מחלץ ערכים פשוטים למטא-דאטה הישנה (backward compatibility)
    """
    if not llm_data or llm_data.get("status") in [
        "error",
        "no_api_key",
        "empty_text",
        "extraction_failed",
    ]:
        return {}

    simple = {}

    if "document" in llm_data and isinstance(llm_data["document"], dict):
        doc = llm_data["document"]

        if "plan_title" in doc and isinstance(doc["plan_title"], dict):
            title = doc["plan_title"].get("value")
            if title:
                simple["plan_name"] = title

        if "scale" in doc and isinstance(doc["scale"], dict):
            scale = doc["scale"].get("value")
            if scale:
                simple["scale"] = scale

        if "plan_type" in doc and isinstance(doc["plan_type"], dict):
            ptype = doc["plan_type"].get("value")
            if ptype:
                simple["plan_type"] = ptype

        if "date" in doc and isinstance(doc["date"], dict):
            date = doc["date"].get("value")
            if date:
                simple["date"] = date

        if "floor_or_level" in doc and isinstance(doc["floor_or_level"], dict):
            floor = doc["floor_or_level"].get("value")
            if floor:
                simple["floor_or_level"] = floor

        if "project_name" in doc and isinstance(doc["project_name"], dict):
            proj = doc["project_name"].get("value")
            if proj:
                simple["project_name"] = proj

    return simple


def build_document_signature(meta, image_shape, text_tokens):
    h, w = image_shape[:2]
    return {
        "page_w": w,
        "page_h": h,
        "aspect_ratio": round(w / h, 3),
        "has_grid": meta.get("has_grid", False),
        "avg_wall_thickness": meta.get("avg_wall_thickness"),
        "scale_candidate": meta.get("scale_denominator"),
        "keywords": list(set(text_tokens))[:30],
    }


def crop_relative(image, bbox_rel):
    """
    🆕 חיתוך תמונה לפי bounding box יחסי (0-1)

    Args:
        image: תמונה (numpy array)
        bbox_rel: [x1, y1, x2, y2] בערכים יחסיים (0.0-1.0)

    Returns:
        תמונה חתוכה או None אם נכשל

    דוגמה:
        # חיתוך פינה ימנית תחתונה (25% מהתמונה)
        cropped = crop_relative(image, [0.75, 0.75, 1.0, 1.0])
    """
    if image is None or image.size == 0:
        return None

    try:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox_rel

        # המרה לפיקסלים
        x1_px = int(x1 * w)
        y1_px = int(y1 * h)
        x2_px = int(x2 * w)
        y2_px = int(y2 * h)

        # וולידציה
        x1_px = max(0, min(x1_px, w - 1))
        y1_px = max(0, min(y1_px, h - 1))
        x2_px = max(0, min(x2_px, w))
        y2_px = max(0, min(y2_px, h))

        # חיתוך
        if x2_px > x1_px and y2_px > y1_px:
            cropped = image[y1_px:y2_px, x1_px:x2_px]

            # בדיקה שהתוצאה לא ריקה
            if cropped.size > 0:
                return cropped

        return None

    except Exception as e:
        print(f"שגיאה ב-crop_relative: {e}")
        return None


def clean_metadata_for_json(metadata: dict) -> dict:
    """מנקה metadata מ-bytes"""
    if not metadata:
        return {}
    return {k: v for k, v in metadata.items() if not isinstance(v, (bytes, bytearray))}


def extract_segments_from_mask(walls_mask, scale):
    """
    מחלץ segments מתוך מסכה קיימת ללא Hough כפול

    Args:
        walls_mask: מסכת קירות (numpy array)
        scale: פיקסלים למטר

    Returns:
        רשימת segments: [{'start': (x,y), 'end': (x,y), 'length_px': ...}]
    """
    segments = []

    # שיטה 1: שימוש ב-contours (יותר יציב מ-Hough)
    contours, _ = cv2.findContours(
        walls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        # פישוט הקונטור לקווים
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # המרה לקטעים
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]

            length_px = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

            # סינון קטעים קצרים מדי
            if length_px > 20:  # מינימום 20 פיקסלים
                segments.append(
                    {
                        "start": tuple(p1),
                        "end": tuple(p2),
                        "length_px": length_px,
                        "length_m": length_px / scale,
                        "source": "contours",
                    }
                )

    return segments
