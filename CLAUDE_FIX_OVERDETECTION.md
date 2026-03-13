# תיקון: Over-detection + 4 בעיות מ-v2

## הבעיות

1. **1608 סגמנטים** — Hough Lines מזהה כל קו בשרטוט כקיר (קווי מידה, קווי רשת, האצ׳ינג)
2. **שם התוכנית נעלם** — Vision data לא מגיע ל-UI
3. **טקסט בצד עדיין מזוהה** — ROI לא חותך מספיק את בלוק הכותרת
4. **קטגוריות אוטומטיות נעלמו** — הפיצ׳ר שהוסיף קטגוריות ל-planning state הוסר בטעות

---

## תיקון 1 — `analyzer.py` (שורש): הוסף morphological opening אחרי threshold

הבעיה: `adaptiveThreshold + OTSU` מזהה גם קווים דקים (מידות, טקסט, קווי רשת).
הפתרון: `MORPH_OPEN` מוחק כל מה שדק מ-3px — רק קירות עבים שורדים.

בכל מקום שיש את הבלוק החדש שהוספנו:
```python
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(...)
binary = cv2.bitwise_or(otsu, adaptive)
clean_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_k, iterations=1)
```

**שנה** את `iterations=1` ל-`iterations=2` **וגם הוסף** בסוף:
```python
# הסר קווים דקים (מידות, האצ׳ינג, טקסט) — שמור רק קירות עבים
thickness_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, thickness_kernel, iterations=2)
```

כלומר הבלוק המלא יהיה:
```python
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
)
binary = cv2.bitwise_or(otsu, adaptive)
# שלב 1: נקה רעש קטן
clean_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_k, iterations=2)
# שלב 2: הסר קווים דקים — רק קירות עבים שורדים
thickness_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, thickness_kernel, iterations=2)
```

עשה את זה לכל 4 המקומות ב-`analyzer.py` שהחלפנו.

---

## תיקון 2 — `backend/main.py`: הגבל Hough Lines בחוזקה

ה-Hough fallback חייב לרוץ **רק כשכמעט שום דבר לא זוהה**. מצא את הבלוק:
```python
# ── גיבוי: Hough Lines לקירות שה-skeleton פספס
orig_img = proj.get("original")
if isinstance(orig_img, np.ndarray) and orig_img.size > 0:
```

**החלף את תנאי הפתיחה**:
```python
# ── גיבוי: Hough Lines — רק כשהskeleton מצא פחות מ-5 קירות ────────────
if len(walls_out) < 5 and isinstance(orig_img, np.ndarray) and orig_img.size > 0:
```

ו**הוסף הגבלה על מספר ה-segments** בתוך הלולאה, לפני `all_segments.append(...)`:
```python
if hough_wall_counter >= 30:   # מקסימום 30 קירות מ-Hough
    break
```

ו**הגדל** את `min_line_len`:
```python
min_line_len = int(scale_px_per_meter * 1.5)   # קיר מינימלי 1.5 מטר (היה 0.5)
```

---

## תיקון 3 — `backend/main.py`: ROI חכם יותר — חתוך בלוק כותרת בצד שמאל

הבעיה: בתוכניות ישראליות בלוק הכותרת נמצא בצד **שמאל תחתון** ותופס ~15-20% מרוחב הדף.

ב-`_detect_plan_roi`, **אחרי** שמחושב `left_bound`, הוסף זיהוי חכם של בלוק כותרת:

```python
# זיהוי בלוק כותרת: אזור בצד שמאל עם density גבוה של טקסט (לא קירות)
# בתוכניות ישראליות — בלוק כותרת תמיד בצד שמאל
# נחפש: עמודה שבה density גבוה אבל הקווים אינם ארוכים (= טקסט, לא קיר)
title_block_right = left_bound  # ברירת מחדל: אין בלוק כותרת

# בדוק רצועה של 25% מהרוחב בצד שמאל
check_width = int(w * 0.25)
for xi in range(left_bound, left_bound + check_width):
    col_sum = col_density[xi] if xi < len(col_density) else 0
    # בלוק כותרת: density בינוני (לא קיר מלא, לא ריק)
    if 0.03 < col_sum < 0.20:
        title_block_right = xi + 1

# אם מצאנו בלוק כותרת משמעותי — דחה אותו
if title_block_right > left_bound + int(w * 0.05):
    left_bound = title_block_right
```

---

## תיקון 4 — `backend/main.py`: שחזר קטגוריות אוטומטיות

הפיצ׳ר שנעלם: לאחר auto-analyze, המערכת הוסיפה אוטומטית קטגוריות ל-`planning.categories` לפי הסגמנטים שנמצאו.

מצא את הבלוק שמסיים את ה-endpoint (לפני `return AutoAnalyzeResponse(...)`):

```python
        return AutoAnalyzeResponse(
            segments=all_segments,
            vision_data=vision_data,
        )
```

**לפני ה-return**, הוסף:

```python
        # ── הוסף קטגוריות אוטומטיות ל-planning state אם לא קיימות ─────────
        _init_planning_if_missing(proj)
        existing_cats = proj["planning"].get("categories") or {}

        AUTO_CATEGORIES = {
            "wall_exterior": {
                "key": "wall_exterior", "type": "קירות", "subtype": "קיר חיצוני",
                "unit": "מ׳", "price_per_unit": 0.0, "color": "#1D4ED8",
            },
            "wall_interior": {
                "key": "wall_interior", "type": "קירות", "subtype": "קיר פנימי",
                "unit": "מ׳", "price_per_unit": 0.0, "color": "#059669",
            },
            "wall_partition": {
                "key": "wall_partition", "type": "קירות", "subtype": "קיר הפרדה / גבס",
                "unit": "מ׳", "price_per_unit": 0.0, "color": "#D97706",
            },
            "fixture_sanitary": {
                "key": "fixture_sanitary", "type": "אביזרים סניטריים", "subtype": "כיור / אסלה",
                "unit": "יח׳", "price_per_unit": 0.0, "color": "#0EA5E9",
            },
            "fixture_stairs": {
                "key": "fixture_stairs", "type": "תחבורה אנכית", "subtype": "מדרגות / מעלית",
                "unit": "יח׳", "price_per_unit": 0.0, "color": "#F59E0B",
            },
        }

        # הוסף רק קטגוריות שאין עדיין
        cats_added = []
        for cat_key, cat_data in AUTO_CATEGORIES.items():
            # הוסף רק אם יש לפחות segment אחד מהסוג הזה
            has_relevant = any(
                (s.wall_type == cat_key.replace("wall_", "") if s.element_class == "wall"
                 else s.suggested_type == cat_data["type"])
                for s in all_segments
            )
            if has_relevant and cat_key not in existing_cats:
                existing_cats[cat_key] = cat_data
                cats_added.append(cat_key)

        if cats_added:
            proj["planning"]["categories"] = existing_cats
            _persist_plan_to_database(plan_id, proj)
            print(f"[auto-analyze] Auto-added categories: {cats_added}")
```

---

## תיקון 5 — שם התוכנית נעלם

הבעיה: `vision_data.plan_title` לא מגיע ל-frontend.

ב-`_build_vision_data`, בדוק שהשדה `plan_title` אכן מוחזר:
```python
plan_title=_str(m.get("plan_title")),
```

ב-`frontend/src/pages/PlanningPage.tsx`, חפש איפה מוצג שם התוכנית (חפש `plan_title` או `planName` או `autoVisionData`).

אם יש קוד כזה:
```tsx
{autoVisionData?.plan_title && <span>{autoVisionData.plan_title}</span>}
```
ודא שהוא עדיין קיים ולא נמחק בזמן עדכון ה-Category Panel.

אם נמחק — הוסף אותו חזרה **בראש הפאנל הצדדי של שלב 3**, לפני ה-"קטגוריות מזוהות":
```tsx
{/* שם תוכנית מ-Vision */}
{autoVisionData?.plan_title && (
  <div style={{
    background: "var(--navy)", color: "white", borderRadius: 8,
    padding: "6px 10px", marginBottom: 10, fontSize: 12, fontWeight: 700,
    direction: "rtl"
  }}>
    📋 {autoVisionData.plan_title}
    {autoVisionData?.sheet_number && <span style={{opacity:0.7, marginRight:6, fontSize:11}}>| {autoVisionData.sheet_number}</span>}
  </div>
)}
```

---

## סדר יישום מומלץ

1. `analyzer.py` — הוסף thickness_kernel (תיקון 1) — **הכי חשוב, פוגע ב-1608**
2. `backend/main.py` — הגבל Hough ל-`len(walls_out) < 5` (תיקון 2)
3. `backend/main.py` — שחזר קטגוריות אוטומטיות (תיקון 4)
4. `backend/main.py` — שפר ROI לבלוק כותרת (תיקון 3)
5. `PlanningPage.tsx` — שחזר תצוגת שם תוכנית (תיקון 5)

## ציפיות לאחר תיקון

- מ-1608 → ~20-60 סגמנטים (רק קירות אמיתיים)
- קטגוריות wall_exterior / wall_interior / wall_partition יופיעו אוטומטית
- שם התוכנית יופיע בראש הפאנל
- בלוק הכותרת בשמאל לא יזוהה כקירות
