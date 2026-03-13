# תיקון: זיהוי אזור תוכנית (ROI) + שיפור זיהוי קירות

## הבעיות הנוכחיות

1. **מסגרת הגיליון** מזוהה כ"קירות חיצוניים" — הקו החיצוני של הדף הוא segment ארוך
2. **תוויות Vision** (חדרים) מוצגות מחוץ לתוכנית — x_pct/y_pct יחסי לעמוד כולו
3. **קווי מידה / רצועות תחתית** מזוהים כאביזרים
4. **בלוק כותרת + מקרא** (שמאל תחתון) מזוהים כקירות

## הפתרון — 3 שינויים ב-`backend/main.py`

---

### תיקון 1: זיהוי ה-ROI (אזור התוכנית בתוך הדף)

ב-endpoint `manager_auto_analyze`, **לפני** לולאת ה-segments (לפני `for label_id in range(1, num_skel_labels):`), הוסף את הפונקציה הבאה **ואת הקריאה אליה**:

```python
def _detect_plan_roi(binary: np.ndarray) -> tuple[int, int, int, int]:
    """
    מזהה את אזור התוכנית האמיתי בתוך הדף — הריבוע הפנימי הגדול ביותר
    שמכיל את רוב תוכן הקירות, תוך התעלמות מבלוק כותרת ושוליים.
    מחזיר (x, y, w, h) של ה-ROI, או את כל הדף אם לא נמצא.
    """
    h, w = binary.shape[:2]

    # שלב 1: מצא את ה-bounding box של כל התוכן
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return 0, 0, w, h

    # שלב 2: חפש מסגרת תוכנית — קו אופקי עבה ברצף
    # תוכנית אדריכלית תמיד מוקפת במסגרת עבה
    # נחפש את הריבוע הגדול ביותר שמוגבל ע"י קווים רציפים

    # הצלבות אנכיות: לכל עמודת x, כמה פיקסלים יש
    col_density = np.sum(binary > 0, axis=0).astype(float) / h
    row_density = np.sum(binary > 0, axis=1).astype(float) / w

    # מצא גבולות: עמודות/שורות עם density גבוה = קו מסגרת
    FRAME_THRESHOLD = 0.25  # לפחות 25% מהשורה/עמודה מלאה = קו מסגרת

    # גבולות אנכיים (שמאל/ימין)
    left_bound = 0
    for xi in range(w):
        if col_density[xi] > FRAME_THRESHOLD:
            left_bound = xi
            break

    right_bound = w
    for xi in range(w - 1, -1, -1):
        if col_density[xi] > FRAME_THRESHOLD:
            right_bound = xi
            break

    # גבולות אופקיים (למעלה/למטה)
    top_bound = 0
    for yi in range(h):
        if row_density[yi] > FRAME_THRESHOLD:
            top_bound = yi
            break

    bottom_bound = h
    for yi in range(h - 1, -1, -1):
        if row_density[yi] > FRAME_THRESHOLD:
            bottom_bound = yi
            break

    # הוסף margin קטן פנימה כדי לא לכלול את קו המסגרת עצמו
    margin = max(5, int(w * 0.008))
    roi_x = min(left_bound + margin, w - 1)
    roi_y = min(top_bound + margin, h - 1)
    roi_w = max(1, right_bound - left_bound - 2 * margin)
    roi_h = max(1, bottom_bound - top_bound - 2 * margin)

    # בדיקת סניטי: ה-ROI חייב להיות לפחות 40% מהדף
    if roi_w * roi_h < w * h * 0.40:
        # fallback: חתוך 8% משוליים בלבד
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.08)
        return pad_x, pad_y, w - 2 * pad_x, h - 2 * pad_y

    return roi_x, roi_y, roi_w, roi_h
```

ואז **מיד אחרי** ה-`binary = (walls > 0).astype(np.uint8)`, הוסף:

```python
# ── זהה את אזור התוכנית האמיתי (ללא בלוק כותרת / שוליים) ─────────────
plan_roi_x, plan_roi_y, plan_roi_w, plan_roi_h = _detect_plan_roi(binary)
plan_roi_x2 = plan_roi_x + plan_roi_w
plan_roi_y2 = plan_roi_y + plan_roi_h
print(f"[auto-analyze] plan ROI: ({plan_roi_x},{plan_roi_y}) → ({plan_roi_x2},{plan_roi_y2}) "
      f"({plan_roi_w}×{plan_roi_h}px, {100*plan_roi_w*plan_roi_h/img_area:.1f}% of page)")
```

---

### תיקון 2: פילטר segmentים מחוץ ל-ROI

**בתוך לולאת** `for label_id in range(1, num_skel_labels):`, **אחרי** חישוב `g_bx, g_by, g_bw, g_bh` ו**לפני** ה-`is_fixture` check, הוסף:

```python
# ── סנן segmentים שמרכזם מחוץ ל-ROI (בלוק כותרת / שוליים) ───────────
seg_cx = g_bx + g_bw / 2
seg_cy = g_by + g_bh / 2
seg_in_roi = (
    plan_roi_x <= seg_cx <= plan_roi_x2 and
    plan_roi_y <= seg_cy <= plan_roi_y2
)
if not seg_in_roi:
    continue
```

---

### תיקון 3: תיקון מיקום חדרי Vision — מ-page coords ל-plan coords

**בבלוק שמוסיף חדרי Vision** (אחרי `vision_rooms = ...`), שנה את חישוב `lx, ly`:

```python
# חישוב מיקום יחסי ל-ROI (לא לדף כולו)
roi_w_f = float(plan_roi_w)
roi_h_f = float(plan_roi_h)

# המר x_pct/y_pct מקואורדינטות דף → קואורדינטות ROI
# בהנחה ש-Vision ראתה את כל הדף, אבל החדרים נמצאים בתוך ה-ROI
# נמפה: x_in_roi = (x_pct * img_w - plan_roi_x) / plan_roi_w
x_in_roi = (x_pct * img_w_f - plan_roi_x) / max(1, plan_roi_w)
y_in_roi = (y_pct * img_h_f - plan_roi_y) / max(1, plan_roi_h)

# clip ל-[0.02, 0.98]
x_in_roi = max(0.02, min(0.98, x_in_roi))
y_in_roi = max(0.02, min(0.98, y_in_roi))

# bbox בקואורדינטות טבעיות של הדף
lw = img_w_f * 0.06
lh = img_h_f * 0.035
lx = max(0, plan_roi_x + x_in_roi * plan_roi_w - lw / 2)
ly = max(0, plan_roi_y + y_in_roi * plan_roi_h - lh / 2)
```

---

### תיקון 4: סנן קווי מידה (רצועות בתחתית)

בלולאת ה-fixture detection, **לאחר** פילטר הטקסט הקיים, הוסף:

```python
# פילטר: קווי מידה — רצועות צרות וארוכות (dimension lines)
is_dimension_line = (
    (g_bw > img_w * 0.15 and g_bh < img_h * 0.015) or  # רצועה אופקית ארוכה
    (g_bh > img_h * 0.15 and g_bw < img_w * 0.015)      # רצועה אנכית ארוכה
)
if is_dimension_line:
    continue
```

---

## שיפור הזיהוי — Hough Lines כשיטת גיבוי

לאחר `all_segments = walls_out + fixtures_out`, לפני בלוק Vision rooms, הוסף זיהוי קירות נוסף דרך Hough Lines על התמונה המקורית:

```python
# ── גיבוי: Hough Lines לקירות שה-skeleton פספס ─────────────────────────
orig_img = proj.get("original")
if isinstance(orig_img, np.ndarray) and orig_img.size > 0:
    try:
        # הכן תמונה grayscale ב-ROI בלבד
        gray_orig = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) if orig_img.ndim == 3 else orig_img.copy()
        if gray_orig.shape != (img_h, img_w):
            gray_orig = cv2.resize(gray_orig, (img_w, img_h), interpolation=cv2.INTER_AREA)

        # חתוך ל-ROI
        roi_gray = gray_orig[plan_roi_y:plan_roi_y2, plan_roi_x:plan_roi_x2]

        # Edge detection
        edges = cv2.Canny(roi_gray, 30, 100, apertureSize=3)

        # Hough Lines — מצא קטעי קו
        min_line_len = int(scale_px_per_meter * 0.5)   # קיר מינימלי 50 ס"מ
        max_gap = int(scale_px_per_meter * 0.15)        # פער מקסימלי 15 ס"מ
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=40,
            minLineLength=min_line_len,
            maxLineGap=max_gap
        )

        if lines is not None:
            existing_centers = set()
            for s in walls_out:
                cx_ = int(s.bbox[0] + s.bbox[2] / 2)
                cy_ = int(s.bbox[1] + s.bbox[3] / 2)
                existing_centers.add((cx_ // 30, cy_ // 30))  # grid 30px

            hough_wall_counter = 0
            for line in lines:
                x1l, y1l, x2l, y2l = line[0]
                # המר חזרה לקואורדינטות מלאות
                x1g = x1l + plan_roi_x; y1g = y1l + plan_roi_y
                x2g = x2l + plan_roi_x; y2g = y2l + plan_roi_y

                llen = np.sqrt((x2g - x1g)**2 + (y2g - y1g)**2)
                if llen < min_line_len:
                    continue

                # bbox של הקו
                bx_h = min(x1g, x2g); by_h = min(y1g, y2g)
                bw_h = max(5, abs(x2g - x1g)); bh_h = max(5, abs(y2g - y1g))
                cx_h = int(bx_h + bw_h / 2); cy_h = int(by_h + bh_h / 2)

                # בדוק אם כבר מכוסה ע"י wall קיים
                grid_key = (cx_h // 30, cy_h // 30)
                if grid_key in existing_centers:
                    continue
                existing_centers.add(grid_key)

                length_m_h = round(llen / scale_px_per_meter, 2)
                if length_m_h < 0.4:
                    continue

                # קבע סוג
                angle = abs(np.degrees(np.arctan2(y2g - y1g, x2g - x1g)))
                near_edge_h = (
                    bx_h < plan_roi_x + plan_roi_w * 0.1 or
                    bx_h + bw_h > plan_roi_x2 - plan_roi_w * 0.1 or
                    by_h < plan_roi_y + plan_roi_h * 0.1 or
                    by_h + bh_h > plan_roi_y2 - plan_roi_h * 0.1
                )
                wtype_h = "exterior" if (near_edge_h and length_m_h > 2.0) else "interior"
                color_h = "#1D4ED8" if wtype_h == "exterior" else "#059669"

                hough_wall_counter += 1
                all_segments.append(AutoAnalyzeSegment(
                    segment_id=f"hough_{hough_wall_counter}",
                    label=f"קיר {'חיצוני' if wtype_h == 'exterior' else 'פנימי'} H{hough_wall_counter} — {length_m_h:.1f}מ׳",
                    suggested_type="קירות",
                    suggested_subtype=f"{wtype_h}/הוף",
                    confidence=0.70,
                    length_m=float(length_m_h),
                    area_m2=0.0,
                    bbox=[float(bx_h), float(by_h), float(bw_h), float(bh_h)],
                    element_class="wall",
                    wall_type=wtype_h,
                    material="לא_ידוע",
                    category_color=color_h,
                ))
            print(f"[auto-analyze] Hough walls added: {hough_wall_counter}")

    except Exception as _hough_err:
        print(f"[auto-analyze] Hough fallback error: {_hough_err}")
```

---

## בדיקות

1. לאחר השינוי — הפעל מחדש backend ולחץ "נתח"
2. ציפייה: segments 29/30 (מסגרת) לא יופיעו יותר
3. ציפייה: חדרים מ-Vision יופיעו **בתוך** גבולות הבניין
4. ציפייה: עלייה בכמות קירות זוהו (Hough מוסיף קירות שה-skeleton פספס)
5. בדוק בלוג השרת: שורת `plan ROI: (x,y) → (x2,y2)` — ודא שה-ROI הגיוני
