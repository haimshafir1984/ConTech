# CLAUDE CODE PROMPT — תיקון Canvas + BOQ + UX שלב 3

## סקירת הבעיות (6 תיקונים)

1. **כתב כמויות 503** — `vision_analysis` לא נטען חזרה מה-DB
2. **הקנבס מציג רק 30%** — maxW=920 בלי הגבלת גובה → תמונה גבוהה מדי לא מתאימה למסך
3. **Overlay כהה מדי** — כשקטגוריה נבחרת, ה-dimming=0.45 מסתיר את התוכנית
4. **רעש ב-"הכל"** — תוויות SVG חופפות ולא קריאות
5. **זרימת "אשר" לא ברורה** — 51 "ממתינים לאישור" ללא הסבר מה זה עושה
6. **כפתור מחיקה לא בולט** — × קטן מדי, לא אינטואיטיבי

---

## תיקון 1 — `backend/main.py`: BOQ endpoint (503 fix)

**מצא** את:
```python
@app.get("/manager/planning/{plan_id}/boq-summary")
async def manager_boq_summary(plan_id: str):
```

**החלף את כל גוף הפונקציה** (מ-`proj = _get_project` עד הסוף של הפונקציה) ב:

```python
    try:
        proj = _get_project(plan_id)
        if not proj:
            raise HTTPException(status_code=404, detail="Plan not found")

        FLOOR_HEIGHT_M = 2.40

        # ── 1. vision_analysis — מזיכרון או מ-DB ─────────────────────────────
        vision = proj.get("vision_analysis")
        if not vision or not isinstance(vision, dict):
            try:
                vision = db_get_vision_analysis(plan_id) or {}
                if vision and isinstance(vision, dict):
                    proj["vision_analysis"] = vision
                    print(f"[boq] loaded vision_analysis from DB for {plan_id}")
            except Exception as _ve:
                print(f"[boq] cannot load vision_analysis: {_ve}")
                vision = {}

        # ── 2. חדרים ─────────────────────────────────────────────────────────
        rooms_table = []
        total_built_area = 0.0
        for r in (vision.get("rooms") or []):
            if not isinstance(r, dict):
                continue
            name = r.get("name") or r.get("room_name") or "חדר"
            area = float(r.get("area_m2") or r.get("area") or 0.0)
            rooms_table.append({"name": name, "area_m2": round(area, 1)})
            total_built_area += area

        # ── 3. קירות מ-segments ───────────────────────────────────────────────
        _init_planning_if_missing(proj)
        segments_raw = proj["planning"].get("auto_segments") or []
        if not segments_raw:
            try:
                segments_raw = db_get_auto_segments(plan_id) or []
            except Exception as _se:
                print(f"[boq] db_get_auto_segments failed: {_se}")
                segments_raw = []

        wall_groups: dict = {
            "exterior":  {"label": "קירות חיצוניים",    "color": "#1D4ED8", "segments": []},
            "interior":  {"label": "קירות פנימיים",     "color": "#059669", "segments": []},
            "partition": {"label": "קירות גבס/הפרדה",   "color": "#D97706", "segments": []},
            "other":     {"label": "קירות אחרים",        "color": "#6B7280", "segments": []},
        }
        fixture_counts: dict = {}
        for s in segments_raw:
            d = s if isinstance(s, dict) else (s.model_dump() if hasattr(s, "model_dump") else {})
            ec = d.get("element_class", "wall")
            if ec == "wall":
                wt = d.get("wall_type", "other")
                wall_groups[wt if wt in wall_groups else "other"]["segments"].append(d)
            elif ec == "fixture":
                stype = d.get("suggested_subtype") or d.get("suggested_type") or "אחר"
                fixture_counts[stype] = fixture_counts.get(stype, 0) + 1

        walls_table = []
        for grp_key, grp in wall_groups.items():
            segs = grp["segments"]
            if not segs:
                continue
            tl = round(sum(float(s.get("length_m", 0)) for s in segs), 2)
            walls_table.append({
                "wall_type": grp["label"], "wall_type_key": grp_key,
                "color": grp["color"], "count": len(segs),
                "total_length_m": tl, "wall_area_m2": round(tl * FLOOR_HEIGHT_M, 2),
            })

        # ── 4. פתחים ─────────────────────────────────────────────────────────
        door_count = window_count = stair_count = 0
        for el in (vision.get("elements") or []):
            if not isinstance(el, dict):
                continue
            et = (el.get("type") or "").lower()
            if "דלת" in et or "door" in et:    door_count += 1
            elif "חלון" in et or "window" in et: window_count += 1
            elif "מדרגות" in et or "stair" in et or "מעלית" in et: stair_count += 1

        oai = vision.get("openai_supplement") or {}
        if isinstance(oai, dict):
            for d in (oai.get("doors") or []):
                if isinstance(d, dict): door_count += int(d.get("count", 0))
            for w in (oai.get("windows") or []):
                if isinstance(w, dict): window_count += int(w.get("count", 0))
            san = oai.get("sanitary") or {}
            if isinstance(san, dict):
                if san.get("toilets"):  fixture_counts["אסלות"]  = int(san["toilets"])
                if san.get("sinks"):    fixture_counts["כיורים"] = int(san["sinks"])
                if san.get("showers"):  fixture_counts["מקלחות"] = int(san["showers"])
            fire = oai.get("fire_safety") or {}
            if isinstance(fire, dict):
                if fire.get("cabinets"): fixture_counts["ארון כיבוי אש"] = int(fire["cabinets"])
                if fire.get("reels"):    fixture_counts["גלגלון כיבוי"]  = int(fire["reels"])
                if fire.get("panels"):   fixture_counts["פנל כבאים"]     = int(fire["panels"])

        tl = round(sum(g["total_length_m"] for g in walls_table), 2)
        ta = round(sum(g["wall_area_m2"]   for g in walls_table), 2)
        plan_meta = proj.get("metadata") or {}
        result = {
            "rooms": rooms_table, "walls": walls_table,
            "door_count": door_count, "window_count": window_count,
            "stair_count": stair_count, "fixture_counts": fixture_counts,
            "total_rooms": len(rooms_table), "total_area_m2": round(total_built_area, 2),
            "total_wall_length_m": tl, "total_wall_area_m2": ta,
            "plan_title": vision.get("plan_title") or plan_meta.get("filename", ""),
            "scale": vision.get("scale") or plan_meta.get("scale_str", ""),
            "floor_height_m": FLOOR_HEIGHT_M,
        }
        print(f"[boq] {plan_id}: rooms={len(rooms_table)} walls={len(walls_table)} "
              f"doors={door_count} windows={window_count} fixtures={len(fixture_counts)}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"BOQ error: {str(e)}")
```

---

## תיקון 2 — `frontend/src/pages/PlanningPage.tsx`: Canvas fit

### 2א. תקן את `updateDisplaySizeFromImage`

**מצא:**
```typescript
const updateDisplaySizeFromImage = (img: HTMLImageElement | null) => {
    if (!img) return;
    const maxW = 920;
    const naturalW = img.naturalWidth || planningState?.image_width || 1;
    const naturalH = img.naturalHeight || planningState?.image_height || 1;
    const scale = Math.min(1, maxW / naturalW);
    setBaseDisplaySize({ width: Math.max(1, Math.round(naturalW * scale)), height: Math.max(1, Math.round(naturalH * scale)) });
  };
```

**החלף ב:**
```typescript
const updateDisplaySizeFromImage = (img: HTMLImageElement | null) => {
    if (!img) return;
    const naturalW = img.naturalWidth || planningState?.image_width || 1;
    const naturalH = img.naturalHeight || planningState?.image_height || 1;
    // התאם לרוחב וגובה הקונטיינר הזמין
    const containerW = canvasContainerRef.current?.clientWidth   || 920;
    const containerH = canvasContainerRef.current?.clientHeight  || 600;
    const maxW = Math.max(400, containerW - 32);
    const maxH = Math.max(300, containerH - 32);
    const scale = Math.min(1, maxW / naturalW, maxH / naturalH);
    setBaseDisplaySize({
      width:  Math.max(1, Math.round(naturalW * scale)),
      height: Math.max(1, Math.round(naturalH * scale)),
    });
  };
```

### 2ב. תקן את גובה הקונטיינר

**מצא:**
```tsx
<div ref={canvasContainerRef} style={{ background: "#1A2744", position: "relative", overflow: "auto", display: "flex", alignItems: "flex-start", justifyContent: "center", padding: 16, minHeight: 400 }}>
```

**החלף ב:**
```tsx
<div ref={canvasContainerRef} style={{
  background: "#1A2744", position: "relative",
  overflow: "auto", display: "flex",
  alignItems: "center", justifyContent: "center",
  padding: 16,
  minHeight: 400,
  height: "calc(100vh - 285px)",   /* ← מחושב מ-viewport בניכוי header+wizard */
}}>
```

### 2ג. הוסף כפתור "🎯 התאם" ליד כפתורי הזום

**מצא בקוד** את קבוצת כפתורי הזום (חפש `setZoomPercent` או `zoomPercent`). **הוסף** לידם:

```tsx
<button type="button"
  onClick={() => {
    if (drawingImageRef.current && canvasContainerRef.current) {
      // חשב scale חדש שמתאים לקונטיינר
      updateDisplaySizeFromImage(drawingImageRef.current);
      setZoomPercent(100);
    }
  }}
  style={{ padding: "4px 8px", borderRadius: 6, fontSize: 11, border: "1px solid var(--s300)", background: "white", cursor: "pointer", color: "var(--navy)", fontWeight: 600 }}
  title="התאם לגודל המסך"
>🎯 התאם</button>
```

---

## תיקון 3 — Overlay פחות כהה + סגמנטים בולטים יותר

**בקובץ `PlanningPage.tsx`** חפש את `dimming rect` בתוך ה-SVG של ה-auto segments (חפש `rgba(0,0,0,0.45)` או `rgba(0,0,0,0.4)`).

**החלף** את ה-opacity מ-`0.45` ל-`0.18`:
```tsx
// מצא:
fill="rgba(0,0,0,0.45)"
// החלף ב:
fill="rgba(0,0,0,0.18)"
```

אם יש כמה ערכים כאלה — שנה את כולם ל-`0.18`.

**עבור סגמנטים מודגשים** (כשקטגוריה נבחרת), הגדל את ה-strokeWidth:
חפש את `strokeWidth` לסגמנטים. **עדכן**:
```tsx
// strokeWidth לסגמנט מודגש (highlightedClass === segCatKey):
const strokeWidth = isSelected ? 4 : highlightedClass ? 3 : (checked ? 2.5 : 1.5);
// fillOpacity מופחת מעט כשאין highlight:
const fillOpacity = highlightedClass
  ? (segCatKey === highlightedClass ? 0.50 : 0.0)   // רק המודגשים נראים, שאר שקופים
  : (checked ? 0.25 : 0.08);
```

---

## תיקון 4 — ביטול labels ב-"הכל" (מצמצם רעש)

**חפש** בתוך SVG של auto segments את החלק שמציג text/confidence badge (חפש `{Math.round(seg.confidence * 100)}%`).

**עטוף** את הטקסט בתנאי כך שיוצג **רק כשיש hover או קטגוריה נבחרת**:

```tsx
{/* confidence badge — רק בhover או כשיש קטגוריה נבחרת */}
{(hoveredSegId === seg.segment_id || highlightedClass !== null) && (
  <text
    x={seg.bbox[0] * displayScale + 4}
    y={seg.bbox[1] * displayScale + 13}
    fontSize={9}
    fill={seg.confidence >= 0.75 ? "#166534" : seg.confidence >= 0.5 ? "#92400E" : "#991B1B"}
    fontWeight="bold"
    style={{ pointerEvents: "none", userSelect: "none" }}
  >
    {Math.round(seg.confidence * 100)}%
  </text>
)}
```

**כנ"ל** עבור תוויות שם (label) — הצג רק בhover:
חפש את המקום שמציג `seg.label` כ-text element בתוך SVG, ועטוף אותו:
```tsx
{hoveredSegId === seg.segment_id && (
  // תוית השם — רק בhover
  <foreignObject x={...} y={...} width={120} height={40} style={{ pointerEvents: "none" }}>
    <div style={{ background: "rgba(0,0,0,0.75)", color: "white", fontSize: 10, borderRadius: 4, padding: "2px 4px", whiteSpace: "nowrap" }}>
      {seg.label}
    </div>
  </foreignObject>
)}
```

---

## תיקון 5 — זרימת "אשר" — שנה שפה לברורה יותר

**בקובץ `PlanningPage.tsx`** חפש ושנה את הטקסטים הבאים (replace-all):

| חפש | החלף ב |
|-----|--------|
| `ממתינים לאישור` | `פריטים זוהו` |
| `נותרו X מתוך Y לאישור` | `X פריטים לסקירה` |
| `ממתינים בלבד` | `לא אושרו עדיין` |
| `אשר הכל מעל` | `הוסף לפרויקט — כל מה מעל` |
| `אשר ✓` | `✓ הוסף לפרויקט` |
| `בחר הכל` | `סמן הכל` |

**הוסף** מעל כפתורי "אשר" טקסט הסבר קצר:
```tsx
<div style={{
  background: "#EFF6FF", border: "1px solid #BFDBFE", borderRadius: 8,
  padding: "6px 10px", fontSize: 11, color: "#1E40AF", marginBottom: 8,
  direction: "rtl", lineHeight: 1.5
}}>
  <strong>מה לעשות?</strong> בדוק את הפריטים שזוהו בתוכנית.
  מחק שגיאות (×), ואז לחץ "הוסף לפרויקט" להעביר לכתב כמויות.
</div>
```

---

## תיקון 6 — כפתור מחיקה בולט יותר

**בפאנל הצדדי** (בתוך expanded list של כל קבוצה), חפש את כפתור המחיקה הקיים:
```tsx
<button type="button"
  onClick={(e) => { e.stopPropagation(); void handleDeleteSegment(seg.segment_id); }}
```

**עדכן** את הסגנון שלו להיות בולט יותר:
```tsx
<button type="button"
  onClick={(e) => { e.stopPropagation(); void handleDeleteSegment(seg.segment_id); }}
  style={{
    background: "#FEE2E2", border: "1px solid #FECACA", borderRadius: 6,
    color: "#DC2626", fontSize: 13, fontWeight: 700, cursor: "pointer",
    padding: "2px 8px", lineHeight: 1, flexShrink: 0,
  }}
  title="הסר - זיהוי שגוי"
>🗑</button>
```

**בנוסף**, לפני רשימת הסגמנטים בכל קבוצה מורחבת, הוסף tooltip הסבר:
```tsx
<div style={{ fontSize: 10, color: "#94A3B8", padding: "2px 8px 4px", direction: "rtl" }}>
  לחץ 🗑 להסרת זיהוי שגוי
</div>
```

---

## פלט נדרש אחרי ביצוע

דווח:
1. **כתב כמויות**: מה מופיע ב-terminal של uvicorn אחרי לחיצה על הכפתור? (`[boq] ...`)
2. **קנבס**: האם התוכנית ממלאת עכשיו את שטח הקנבס?
3. **Overlay**: כשלוחצים על "קירות חיצוניים" — האם התוכנית נראית מתחת?
4. **שגיאות בדפדפן**: האם יש שגיאות ב-DevTools Console?
