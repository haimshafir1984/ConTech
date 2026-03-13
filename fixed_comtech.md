# ConTech — תיקונים קריטיים (Session Memory)

> קובץ זה מתאר שני תיקונים קריטיים שהוחלט לבצע.
> אם הסשן נסגר — קרא קובץ זה + CLAUDE.md לפני שממשיכים.

---

## תיקון 1 — Claude Vision על תמונת השרטוט

### הבעיה
`brain.py → process_plan_metadata()` מנתח רק **טקסט OCR** מה-PDF.
תמונת השרטוט עצמה מעולם לא נשלחת ל-LLM לניתוח ויזואלי.
כתוצאה: זיהוי אביזרים סניטריים, דלתות, חלונות, חדרים — חלש מאוד.

### הפתרון
הוסף פונקציה `analyze_plan_with_vision(image_bytes, plan_id)` ב-`brain.py` (root level).
פונקציה זו שולחת את תמונת הקומה ל-Claude Vision (`claude-3-5-sonnet-20241022`)
ומחזירה JSON מובנה עם:
```json
{
  "walls": [{"type": "concrete|block|gypsum", "estimated_length_m": 5.2, "location": "north"}],
  "doors": [{"width_m": 0.9, "location": "bedroom entrance"}],
  "windows": [{"width_m": 1.2, "location": "living room south wall"}],
  "plumbing_fixtures": [{"type": "toilet|sink|bathtub|shower", "room": "bathroom"}],
  "rooms": [{"name": "master bedroom", "estimated_area_m2": 18}],
  "materials_detected": ["concrete blocks", "gypsum board"],
  "execution_notes": ["note 1", "note 2"]
}
```

### קבצים לשינוי

#### A. `brain.py` (root) — הוסף פונקציה חדשה

```python
def analyze_plan_with_vision(image_bytes: bytes, scale_text: str = "") -> dict:
    """
    שולח תמונת שרטוט ל-Claude Vision ומחלץ אלמנטים.
    מוחזר dict עם: walls, doors, windows, plumbing_fixtures, rooms,
                   materials_detected, execution_notes
    """
    client, error = get_anthropic_client()
    if error or not image_bytes:
        return {"status": "error", "error": error or "no image", "walls": [], "doors": [],
                "windows": [], "plumbing_fixtures": [], "rooms": [], "materials_detected": [], "execution_notes": []}

    import base64
    img_b64 = base64.standard_b64encode(image_bytes).decode()

    prompt = f"""אתה מומחה בקריאת תוכניות בנייה ישראליות.
נתח תמונה זו של תוכנית קומה והחזר JSON בלבד.
קנה מידה: {scale_text or 'לא ידוע'}

מבנה נדרש (JSON בלבד, אין טקסט נוסף):
{{
  "walls": [{{"type": "concrete|block|gypsum|lightweight", "estimated_count": 0, "notes": ""}}],
  "doors": [{{"width_m": 0.9, "type": "single|double", "location_hint": ""}}],
  "windows": [{{"width_m": 1.2, "location_hint": ""}}],
  "plumbing_fixtures": [{{"type": "toilet|sink|kitchen_sink|bathtub|shower|bidet", "room": "", "count": 1}}],
  "rooms": [{{"name": "", "estimated_area_m2": 0, "floor_material": ""}}],
  "materials_detected": [],
  "execution_notes": [],
  "confidence": 0.0
}}"""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        text = message.content[0].text.strip()
        # strip ```json if present
        if "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        result = json.loads(text)
        result["status"] = "success"
        return result
    except Exception as e:
        logger.warning("vision analysis failed: %s", e)
        return {"status": "error", "error": str(e), "walls": [], "doors": [],
                "windows": [], "plumbing_fixtures": [], "rooms": [], "materials_detected": [], "execution_notes": []}
```

#### B. `main.py` (root) — קרא ל-`analyze_plan_with_vision` במהלך upload

מצא את ה-endpoint שמעלה PDF ומעבד אותו (בדרך כלל `/manager/workshop/upload`).
בתוך הקוד, **לאחר** ש-`fitz` ממיר את עמוד 1 לתמונה (PNG bytes) —
הוסף קריאה מקבילית ל-`analyze_plan_with_vision`.

**לוגיקה:**
```python
# כבר קיים: img_bytes = render page 1 to PNG via fitz
# הוסף:
import asyncio, concurrent.futures
loop = asyncio.get_event_loop()
with concurrent.futures.ThreadPoolExecutor() as pool:
    vision_future = loop.run_in_executor(pool, analyze_plan_with_vision, img_bytes, scale_text)
    # ... rest of existing parallel analysis ...
    vision_result = await vision_future

# שמור ב-plans table:
db_save_vision_analysis(plan_id, vision_result)  # ראה database.py
```

#### C. `database.py` (root) — הוסף עמודות חדשות

בתוך `_missing_cols_sqlite` ו-`_missing_cols_postgres` הוסף:
```python
("vision_analysis_json", "TEXT"),   # תוצאות Claude Vision
("auto_segments_json",   "TEXT"),   # autoSegments לשמירה (Fix #2)
```

הוסף פונקציות:
```python
def db_save_vision_analysis(plan_id: str, data: dict):
    conn = get_connection()
    cur = conn.cursor()
    if DB_URL:
        cur.execute("UPDATE plans SET vision_analysis_json=%s WHERE filename=%s",
                    (json.dumps(data, ensure_ascii=False), plan_id))
    else:
        cur.execute("UPDATE plans SET vision_analysis_json=? WHERE filename=?",
                    (json.dumps(data, ensure_ascii=False), plan_id))
    conn.commit(); conn.close()

def db_get_vision_analysis(plan_id: str) -> dict:
    conn = get_connection()
    cur = conn.cursor()
    if DB_URL:
        cur.execute("SELECT vision_analysis_json FROM plans WHERE filename=%s", (plan_id,))
    else:
        cur.execute("SELECT vision_analysis_json FROM plans WHERE filename=?", (plan_id,))
    row = cur.fetchone(); conn.close()
    if not row: return {}
    val = row["vision_analysis_json"] if DB_URL else row[0]
    return json.loads(val) if val else {}
```

#### D. Backend endpoint — חשוף vision_data ב-`/manager/planning/{id}/auto-analyze`

בקובץ `main.py` (או backend wrapper), הנקודת קצה `auto-analyze`:
```python
# הוסף לresponse:
vision_data = db_get_vision_analysis(plan_id)
return {"segments": segments, "vision_data": vision_data}
```

#### E. Frontend — `PlanningPage.tsx`

כבר קיים `autoVisionData` state ו-`AutoAnalyzeVisionData` type.
**לא צריך שינוי** בפרונטנד — הנתונים כבר מוצגים בפאנל Vision Data.
רק לוודא שה-type ב-`planningApi.ts` מכיל `plumbing_fixtures`.

---

## תיקון 2 — שמירת autoSegments ב-DB

### הבעיה
`autoSegments` הוא React state בלבד — מתאפס בכל רענון דף.
אחרי Render cold-start, המנהל חייב לרוץ את הניתוח מחדש.

### הפתרון
1. שמור segments ב-`auto_segments_json` בטבלת plans (ראה database.py למעלה)
2. החזר segments כחלק מ-`GET /manager/planning/{id}` (PlanningState)
3. בפרונטנד — אם `planningState.auto_segments` מגיע ולא ריק, טען ל-`autoSegments` state

### קבצים לשינוי

#### A. `database.py` (root) — פונקציות שמירה/טעינה

```python
def db_save_auto_segments(plan_id: str, segments: list):
    conn = get_connection()
    cur = conn.cursor()
    blob = json.dumps(segments, ensure_ascii=False)
    if DB_URL:
        cur.execute("UPDATE plans SET auto_segments_json=%s WHERE filename=%s", (blob, plan_id))
    else:
        cur.execute("UPDATE plans SET auto_segments_json=? WHERE filename=?", (blob, plan_id))
    conn.commit(); conn.close()

def db_get_auto_segments(plan_id: str) -> list:
    conn = get_connection()
    cur = conn.cursor()
    if DB_URL:
        cur.execute("SELECT auto_segments_json FROM plans WHERE filename=%s", (plan_id,))
    else:
        cur.execute("SELECT auto_segments_json FROM plans WHERE filename=?", (plan_id,))
    row = cur.fetchone(); conn.close()
    if not row: return []
    val = row["auto_segments_json"] if DB_URL else row[0]
    return json.loads(val) if val else []
```

#### B. `main.py` — endpoint `/manager/planning/{id}/auto-analyze`

אחרי שהניתוח מחזיר segments, הוסף שמירה:
```python
db_save_auto_segments(plan_id, segments)
```

#### C. `main.py` — endpoint `GET /manager/planning/{id}`

ב-`_get_planning_state()` (או שם הפונקציה שבונה את PlanningState),
הוסף לפני return:
```python
state["auto_segments"] = db_get_auto_segments(plan_id)
```

#### D. Frontend `planningApi.ts`

`PlanningState` כבר מכיל `auto_segments?: AutoSegment[]` — **אין שינוי**.

#### E. Frontend `PlanningPage.tsx`

ב-`useEffect` שטוען `planningState` בשלב 3, הוסף:
```typescript
// אחרי setPlanningState(state):
if (state.auto_segments && state.auto_segments.length > 0 && autoSegments === null) {
  setAutoSegments(state.auto_segments);
  // Pre-select כמו ב-handleAutoAnalyze:
  setAutoSelected(new Set(
    state.auto_segments
      .filter(s => s.suggested_subtype !== "פרט קטן")
      .map(s => s.segment_id)
  ));
}
```

מצא את המקום המתאים ב-`useEffect` (כנראה ה-effect שמריץ `getPlanningState`).

---

## סדר ביצוע מומלץ

1. `database.py` — הוסף 2 עמודות + 4 פונקציות helper
2. `brain.py` — הוסף `analyze_plan_with_vision()`
3. `main.py` — עדכן auto-analyze endpoint (שמור segments + החזר vision_data)
4. `main.py` — עדכן GET planning state (כלול auto_segments)
5. `main.py` — עדכן upload flow (קרא ל-vision בניתוח מקבילי)
6. `PlanningPage.tsx` — טען autoSegments מ-planningState

---

## בדיקות לאחר הביצוע

- [ ] העלה PDF חדש → בדוק שנוצר רשומה עם vision_analysis_json
- [ ] לחץ "נתח" → בדוק שנשמר auto_segments_json ב-DB
- [ ] רענן דף → בדוק שה-segments חוזרים מה-DB ומוצגים
- [ ] בדוק שה-vision data מוצג בפאנל (חדרים, חומרים, אביזרים)

---

## הערות טכניות

- **image_bytes**: ב-`main.py`, תמונת עמוד 1 כבר נוצרת עם `fitz` לצורך ניתוח CV.
  השתמש באותה תמונה — אל תייצר שוב. חפש `pixmap` או `img_bytes` בקוד הקיים.
- **timeout**: ניתוח Vision יכול לקחת 5–15 שניות. כי הוא רץ במקביל לניתוח CV,
  לא אמור להאריך את זמן ה-upload הכולל.
- **עלות**: בקשת Vision עולה ~$0.003 לתמונה בינונית — סביר.
- **cold-start**: עם Fix #2, גם אחרי cold-start הsegments ישמרו ב-DB וייטענו מחדש.
- **migration**: העמודות החדשות מוגדרות ב-`_missing_cols_*` כך שהן נוצרות אוטומטית
  ב-init — אין צורך ב-migration מורכב.

---

*נוצר: 03/03/2026 | גרסה: 1.0*
