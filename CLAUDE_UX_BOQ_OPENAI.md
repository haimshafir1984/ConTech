# CLAUDE CODE PROMPT — שיפור UX שלב 3 + BOQ + OpenAI

## מה לעשות

שלושה שינויים מרכזיים:

1. **שיפור ממשק שלב 3** — כל סגמנט שמוצג על ה-PDF יוכל להימחק, יראה confidence ויהיה ניתן לסנן
2. **כתב כמויות (BOQ Summary)** — endpoint חדש שמחשב כמויות מהנתונים הקיימים
3. **OpenAI API כתוספת** — GPT-4o Vision לשיפור זיהוי (בנוסף ל-Claude, לא במקומו)

---

## שינוי 1: Backend — DELETE endpoint לסגמנט

**קובץ: `backend/main.py`**

חפש את ה-endpoint הקיים:
```python
@app.post("/manager/planning/{plan_id}/confirm-auto-segment", ...)
```

**הוסף לפניו** (ממש לפני ה-decorator של confirm-auto-segment) את ה-endpoint הבא:

```python
@app.delete("/manager/planning/{plan_id}/auto-segments/{segment_id}")
async def manager_delete_auto_segment(plan_id: str, segment_id: str):
    """מוחק סגמנט בודד מרשימת ה-auto_segments."""
    proj = _get_project(plan_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Plan not found")
    _init_planning_if_missing(proj)
    segments = proj["planning"].get("auto_segments", [])
    before = len(segments)
    proj["planning"]["auto_segments"] = [
        s for s in segments
        if (s.get("segment_id") if isinstance(s, dict) else s.segment_id) != segment_id
    ]
    after = len(proj["planning"]["auto_segments"])
    _persist_plan_to_database(plan_id, proj)
    try:
        db_save_auto_segments(plan_id, proj["planning"]["auto_segments"])
    except Exception:
        pass
    return {"deleted": before - after, "remaining": after}
```

---

## שינוי 2: Backend — BOQ Summary endpoint

**קובץ: `backend/main.py`**

**הוסף** לפני `@app.post("/manager/planning/{plan_id}/confirm-auto-segment")` גם:

```python
@app.get("/manager/planning/{plan_id}/boq-summary")
async def manager_boq_summary(plan_id: str):
    """
    מחשב כתב כמויות אוטומטי מנתוני Vision + CV2.
    מחזיר: שטחי חדרים, אורך+שטח קירות לפי סוג, ספירת פתחים ואביזרים.
    """
    proj = _get_project(plan_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Plan not found")

    FLOOR_HEIGHT_M = 2.40   # גובה קומה טיפוסי

    # ── 1. חדרים מ-Vision ────────────────────────────────────────────────
    vision = proj.get("vision_analysis") or {}
    vision_rooms_raw = vision.get("rooms") or []
    rooms_table = []
    total_built_area = 0.0
    for r in vision_rooms_raw:
        if not isinstance(r, dict):
            continue
        name = r.get("name") or r.get("room_name") or "חדר"
        area = float(r.get("area_m2") or r.get("area") or 0.0)
        rooms_table.append({"name": name, "area_m2": area})
        total_built_area += area

    # ── 2. קירות מ-CV2 segments ──────────────────────────────────────────
    _init_planning_if_missing(proj)
    segments_raw = proj["planning"].get("auto_segments") or []
    # fallback DB
    if not segments_raw:
        try:
            segments_raw = db_get_auto_segments(plan_id) or []
        except Exception:
            segments_raw = []

    wall_groups: dict = {
        "exterior": {"label": "קירות חיצוניים", "segments": [], "color": "#1D4ED8"},
        "interior": {"label": "קירות פנימיים", "segments": [], "color": "#059669"},
        "partition": {"label": "קירות גבס / הפרדה", "segments": [], "color": "#D97706"},
        "other":    {"label": "קירות אחרים", "segments": [], "color": "#6B7280"},
    }
    fixture_counts: dict = {}

    for s in segments_raw:
        d = s if isinstance(s, dict) else s.model_dump()
        ec = d.get("element_class", "wall")
        if ec == "wall":
            wt = d.get("wall_type", "other")
            grp = wt if wt in wall_groups else "other"
            wall_groups[grp]["segments"].append(d)
        elif ec == "fixture":
            stype = d.get("suggested_subtype") or d.get("suggested_type") or "אחר"
            fixture_counts[stype] = fixture_counts.get(stype, 0) + 1

    walls_table = []
    for grp_key, grp in wall_groups.items():
        segs = grp["segments"]
        if not segs:
            continue
        total_len = round(sum(float(s.get("length_m", 0)) for s in segs), 2)
        total_area = round(total_len * FLOOR_HEIGHT_M, 2)
        walls_table.append({
            "type": grp_key,
            "label": grp["label"],
            "color": grp["color"],
            "count": len(segs),
            "total_length_m": total_len,
            "wall_area_m2": total_area,   # אורך × גובה קומה
            "floor_height_m": FLOOR_HEIGHT_M,
        })

    # ── 3. פתחים מ-Vision ────────────────────────────────────────────────
    vision_elements = vision.get("elements") or []
    doors: list = []
    windows: list = []
    for el in vision_elements:
        if not isinstance(el, dict):
            continue
        etype = (el.get("type") or "").lower()
        if "דלת" in etype or "door" in etype:
            doors.append(el)
        elif "חלון" in etype or "window" in etype:
            windows.append(el)

    # ── 4. סיכום ─────────────────────────────────────────────────────────
    total_wall_len = sum(g["total_length_m"] for g in walls_table)
    total_wall_area = sum(g["wall_area_m2"] for g in walls_table)
    plan_meta = proj.get("metadata", {})

    return {
        "plan_id": plan_id,
        "plan_title": vision.get("plan_title") or plan_meta.get("filename", ""),
        "scale": vision.get("scale") or plan_meta.get("scale_str", ""),
        "floor_height_m": FLOOR_HEIGHT_M,
        "sections": {
            "rooms": {
                "title": "שטחי חדרים",
                "rows": rooms_table,
                "total_area_m2": round(total_built_area, 2),
            },
            "walls": {
                "title": "קירות",
                "rows": walls_table,
                "total_length_m": round(total_wall_len, 2),
                "total_area_m2": round(total_wall_area, 2),
            },
            "doors": {
                "title": "דלתות",
                "count": len(doors),
                "rows": doors,
            },
            "windows": {
                "title": "חלונות",
                "count": len(windows),
                "rows": windows,
            },
            "fixtures": {
                "title": "אביזרים",
                "rows": [{"label": k, "count": v} for k, v in fixture_counts.items()],
            },
        },
    }
```

---

## שינוי 3: Backend — OpenAI Vision supplement

**קובץ: `backend/vision_analyzer.py`**

### 3א. בראש הקובץ, אחרי ה-imports הקיימים, הוסף:

```python
# ── OpenAI Vision supplement (optional — needs OPENAI_API_KEY) ─────────────
import os as _os
_OPENAI_AVAILABLE = False
_openai_client = None
try:
    import openai as _openai_lib
    _oai_key = _os.environ.get("OPENAI_API_KEY", "")
    if _oai_key:
        _openai_client = _openai_lib.OpenAI(api_key=_oai_key)
        _OPENAI_AVAILABLE = True
        print("[vision] OpenAI Vision supplement enabled (GPT-4o)")
    else:
        print("[vision] OPENAI_API_KEY not set — OpenAI supplement disabled")
except ImportError:
    print("[vision] openai package not installed — supplement disabled")
```

### 3ב. הוסף פונקציה חדשה לפני `analyze_plan_with_vision`:

```python
def _openai_supplement_analysis(image_b64: str, current_result: dict) -> dict:
    """
    שולח את התמונה ל-GPT-4o לקבלת מידע שClaude אולי פספס.
    ממזג את התוצאות עם ה-result הקיים.
    """
    if not _OPENAI_AVAILABLE or not _openai_client:
        return current_result

    try:
        prompt = """You are an architectural plan analyzer. Look at this floor plan and extract:

1. All room names and their approximate areas in m² (read labels on plan)
2. Count of doors by size (e.g. "80/210 x7", "90/210 x12")
3. Count of windows by type
4. Count of sanitary fixtures: toilets (אסלות), sinks (כיורים), showers (מקלחות)
5. Fire safety items: fire cabinets (ארון כיבוי), fire hose reels (גלגלון), fire panel (פנל כבאים)
6. Any special rooms: mechanical room (חדר חשמל), storage (מחסן), stairs (מדרגות), elevator (מעלית)

Respond ONLY as JSON:
{
  "rooms": [{"name": "...", "area_m2": N}],
  "doors": [{"size": "80/210", "count": N}, ...],
  "windows": [{"type": "classroom", "count": N}, ...],
  "sanitary": {"toilets": N, "sinks": N, "showers": N},
  "fire_safety": {"cabinets": N, "reels": N, "panels": N},
  "special_rooms": [{"name": "...", "note": "..."}]
}"""

        response = _openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"}},
                ],
            }],
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        import json as _json_mod
        gpt_data = _json_mod.loads(raw)
        print(f"[vision] GPT-4o supplement received: {list(gpt_data.keys())}")

        # ── מיזוג: GPT משלים Claude, לא מחליף ─────────────────────────
        result = dict(current_result)

        # חדרים: אם GPT מצא יותר — מיזוג
        claude_rooms = {r.get("name", ""): r for r in (result.get("rooms") or [])}
        for gpt_room in (gpt_data.get("rooms") or []):
            rname = gpt_room.get("name", "")
            if rname not in claude_rooms and gpt_room.get("area_m2"):
                claude_rooms[rname] = {"name": rname, "area_m2": gpt_room["area_m2"], "source": "openai"}
        result["rooms"] = list(claude_rooms.values())

        # אלמנטים — הוסף דלתות וחלונות מ-GPT אם אין מ-Claude
        gpt_elements = []
        for d in (gpt_data.get("doors") or []):
            for _ in range(int(d.get("count", 1))):
                gpt_elements.append({"type": f"דלת {d.get('size','')}", "source": "openai"})
        for w in (gpt_data.get("windows") or []):
            for _ in range(int(w.get("count", 1))):
                gpt_elements.append({"type": f"חלון {w.get('type','')}", "source": "openai"})
        if gpt_elements and not result.get("elements"):
            result["elements"] = gpt_elements

        # סניטרי — הוסף ל-elements
        san = gpt_data.get("sanitary") or {}
        san_items = [
            ("אסלה", san.get("toilets", 0)),
            ("כיור", san.get("sinks", 0)),
            ("מקלחת", san.get("showers", 0)),
        ]
        for sname, scount in san_items:
            for _ in range(int(scount)):
                (result.setdefault("elements", [])).append({"type": sname, "source": "openai"})

        # כיבוי אש
        fire = gpt_data.get("fire_safety") or {}
        fire_items = [
            ("ארון כיבוי אש", fire.get("cabinets", 0)),
            ("גלגלון כיבוי", fire.get("reels", 0)),
            ("פנל כבאים", fire.get("panels", 0)),
        ]
        for fname, fcount in fire_items:
            for _ in range(int(fcount)):
                (result.setdefault("elements", [])).append({"type": fname, "source": "openai"})

        # שמור נתוני GPT גולמיים
        result["openai_supplement"] = gpt_data

        return result

    except Exception as e:
        print(f"[vision] OpenAI supplement error: {e}")
        return current_result
```

### 3ג. בפונקציה `analyze_plan_with_vision` — **בסוף הפונקציה**, לפני ה-`return result_dict`, הוסף קריאה לפונקציה:

חפש בקוד הקיים את השורה `return result_dict` (בפונקציה `analyze_plan_with_vision`).
**לפני ה-return**, הוסף:

```python
    # ── OpenAI supplement — GPT-4o מוסיף מה שClaude אולי פספס ────────────
    if _OPENAI_AVAILABLE:
        result_dict = _openai_supplement_analysis(image_b64_str, result_dict)
```

**שים לב:** `image_b64_str` הוא המשתנה המכיל את ה-base64 של התמונה בפונקציה. אם שם המשתנה שונה, מצא אותו — הוא עובר ל-Claude בתחילת הפונקציה.

---

## שינוי 4: Frontend — UX משופר לשלב 3

**קובץ: `frontend/src/api/planningApi.ts`**

### 4א. הוסף את הפונקציות החדשות:

```typescript
// מחיקת סגמנט בודד
export async function deleteAutoSegment(planId: string, segmentId: string): Promise<void> {
  await fetch(`/api/manager/planning/${planId}/auto-segments/${segmentId}`, {
    method: "DELETE",
  });
}

// BOQ Summary
export interface BoqRoom { name: string; area_m2: number; }
export interface BoqWallRow {
  type: string; label: string; color: string;
  count: number; total_length_m: number; wall_area_m2: number; floor_height_m: number;
}
export interface BoqSummary {
  plan_id: string;
  plan_title: string;
  scale: string;
  floor_height_m: number;
  sections: {
    rooms: { title: string; rows: BoqRoom[]; total_area_m2: number };
    walls: { title: string; rows: BoqWallRow[]; total_length_m: number; total_area_m2: number };
    doors: { title: string; count: number; rows: unknown[] };
    windows: { title: string; count: number; rows: unknown[] };
    fixtures: { title: string; rows: { label: string; count: number }[] };
  };
}

export async function fetchBoqSummary(planId: string): Promise<BoqSummary> {
  const res = await fetch(`/api/manager/planning/${planId}/boq-summary`);
  if (!res.ok) throw new Error("BOQ fetch failed");
  return res.json();
}
```

---

**קובץ: `frontend/src/pages/PlanningPage.tsx`**

### 4ב. הוסף imports בראש הקובץ (אחרי שאר ה-imports):

```typescript
import { deleteAutoSegment, fetchBoqSummary, type BoqSummary } from "../api/planningApi";
```

### 4ג. הוסף state variables (אחרי `const [highlightedClass, ...`):

```typescript
const [boqData, setBoqData] = React.useState<BoqSummary | null>(null);
const [boqLoading, setBoqLoading] = React.useState(false);
const [boqVisible, setBoqVisible] = React.useState(false);
const [confFilter, setConfFilter] = React.useState<"all" | "high" | "low">("all");
const [hoveredSegId, setHoveredSegId] = React.useState<string | null>(null);
```

### 4ד. הוסף פונקציה לטעינת BOQ (אחרי הפונקציות הקיימות של auto-analyze):

```typescript
const handleLoadBoq = React.useCallback(async () => {
  if (!planId) return;
  setBoqLoading(true);
  try {
    const data = await fetchBoqSummary(planId);
    setBoqData(data);
    setBoqVisible(true);
  } catch (e) {
    console.error("BOQ load failed", e);
  } finally {
    setBoqLoading(false);
  }
}, [planId]);
```

### 4ה. הוסף פונקציה למחיקת סגמנט:

```typescript
const handleDeleteSegment = React.useCallback(async (segId: string) => {
  if (!planId || !autoSegments) return;
  try {
    await deleteAutoSegment(planId, segId);
    // עדכן state מקומי מיידית
    setAutoSegments(prev => prev ? prev.filter(s => s.segment_id !== segId) : prev);
  } catch (e) {
    console.error("Delete segment failed", e);
  }
}, [planId, autoSegments]);
```

### 4ו. שינויים ב-SVG Canvas — הוסף Delete button על הסגמנטים

חפש בקוד את החלק שמצייר `<rect>` על כל סגמנט בתוך SVG (חפש `segCatKey` או `strokeWidth`).
אחרי ה-`<rect>` של הסגמנט, הוסף (בתוך אותה `<g>` או `React.Fragment`):

```tsx
{/* Delete button on hover */}
{hoveredSegId === seg.segment_id && (
  <g
    style={{ cursor: "pointer" }}
    onClick={(e) => { e.stopPropagation(); handleDeleteSegment(seg.segment_id); }}
  >
    <circle
      cx={seg.bbox[0] * displayScale + seg.bbox[2] * displayScale - 8}
      cy={seg.bbox[1] * displayScale + 8}
      r={9}
      fill="#EF4444"
      opacity={0.92}
    />
    <text
      x={seg.bbox[0] * displayScale + seg.bbox[2] * displayScale - 8}
      y={seg.bbox[1] * displayScale + 12}
      textAnchor="middle"
      fontSize={11}
      fill="white"
      fontWeight="bold"
      style={{ pointerEvents: "none", userSelect: "none" }}
    >×</text>
  </g>
)}
{/* Confidence badge */}
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
```

עבור כל `<g>` של סגמנט — הוסף `onMouseEnter` ו-`onMouseLeave`:
```tsx
onMouseEnter={() => setHoveredSegId(seg.segment_id)}
onMouseLeave={() => setHoveredSegId(null)}
```

### 4ז. עדכון Color Coding לפי confidence

בחלק שמגדיר את `fillOpacity` ו-`strokeWidth` הקיים, **עדכן** את צבע ה-stroke:

```tsx
// צבע stroke לפי confidence (בנוסף לצבע הקטגוריה)
const confStroke = seg.confidence >= 0.75 ? (seg.category_color || "#059669")
                 : seg.confidence >= 0.5  ? "#F59E0B"   // צהוב — בינוני
                 :                          "#EF4444";   // אדום — נמוך
```

החלף `stroke={...}` הקיים ב-`stroke={confStroke}`.

### 4ח. עדכון הפאנל הצדדי — Filter Bar + Delete

**מצא** את ה-Category Highlight Panel (חפש `קטגוריות מזוהות`).

**לפני** ה-`<div style={{ marginBottom: 8, direction: "rtl" }}>` שמכיל את ה-panel:

```tsx
{/* ── Filter Bar ── */}
{autoSegments && autoSegments.length > 0 && (
  <div style={{ display: "flex", gap: 4, marginBottom: 8, direction: "rtl" }}>
    {(["all", "high", "low"] as const).map(f => {
      const labels = { all: `הכל (${autoSegments.length})`, high: `✓ בטוח`, low: `⚠ לא בטוח` };
      return (
        <button key={f} type="button"
          onClick={() => setConfFilter(f)}
          style={{
            flex: 1, fontSize: 10, padding: "3px 4px", borderRadius: 6, cursor: "pointer",
            border: `1px solid ${confFilter === f ? "#3B82F6" : "#E2E8F0"}`,
            background: confFilter === f ? "#EFF6FF" : "white",
            color: confFilter === f ? "#1D4ED8" : "#64748B",
            fontWeight: confFilter === f ? 700 : 400,
          }}
        >{labels[f]}</button>
      );
    })}
  </div>
)}
```

**בנוסף**, בתוך כל `<div>` שמייצג קבוצת סגמנטים (חפש `categoryGroups.map(grp => ...`), **עדכן** את `grp.segments` כך שיעבור דרך הפילטר:

```tsx
// החלף: const categoryGroups = groupDefs.map(def => ({ ...def, segments: autoSegments.filter(def.filter) }))
// ב:
const filteredSegments = autoSegments.filter(s =>
  confFilter === "all"  ? true :
  confFilter === "high" ? s.confidence >= 0.75 :
  confFilter === "low"  ? s.confidence < 0.75 : true
);
const categoryGroups = groupDefs
  .map(def => ({ ...def, segments: filteredSegments.filter(def.filter) }))
  .filter(g => g.segments.length > 0);
```

**כמו כן**, בתוך כל segment row בפאנל (בתוך ה-expanded list שמציג סגמנטים בודדים):
הוסף כפתור מחיקה קטן לכל שורה:

```tsx
<button type="button"
  onClick={(e) => { e.stopPropagation(); handleDeleteSegment(seg.segment_id); }}
  style={{
    background: "none", border: "none", cursor: "pointer",
    color: "#EF4444", fontSize: 14, padding: "0 4px", lineHeight: 1,
    marginRight: "auto",
  }}
  title="הסר סגמנט"
>×</button>
```

### 4ט. BOQ Panel — כפתור + תצוגה

**מצא** את הכפתור "נתח אוטומטית" הקיים (חפש `handleAutoAnalyze` או `נתח אוטומטית`).
**לידו**, הוסף כפתור BOQ:

```tsx
{autoSegments && autoSegments.length > 0 && (
  <button type="button"
    onClick={handleLoadBoq}
    disabled={boqLoading}
    style={{
      padding: "6px 12px", borderRadius: 8, fontSize: 12, fontWeight: 600,
      background: boqLoading ? "#94A3B8" : "#0F172A", color: "white",
      border: "none", cursor: boqLoading ? "not-allowed" : "pointer",
      marginTop: 8, width: "100%",
    }}
  >
    {boqLoading ? "מחשב..." : "📋 כתב כמויות"}
  </button>
)}
```

**הוסף** את ה-BOQ Panel עצמו — אחרי ה-Category Highlight Panel (מחוץ לו, אך באותו container):

```tsx
{boqVisible && boqData && (
  <div style={{
    marginTop: 12, borderTop: "2px solid #E2E8F0", paddingTop: 12,
    direction: "rtl", fontSize: 12,
  }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
      <span style={{ fontWeight: 700, fontSize: 13, color: "var(--navy)" }}>
        📋 כתב כמויות אוטומטי
      </span>
      <button type="button" onClick={() => setBoqVisible(false)}
        style={{ background: "none", border: "none", cursor: "pointer", color: "#94A3B8", fontSize: 16 }}>×</button>
    </div>
    {boqData.plan_title && (
      <div style={{ color: "#64748B", fontSize: 11, marginBottom: 8 }}>
        {boqData.plan_title} {boqData.scale && `| קנ"מ ${boqData.scale}`} | גובה קומה {boqData.floor_height_m} מ׳
      </div>
    )}

    {/* חדרים */}
    {boqData.sections.rooms.rows.length > 0 && (
      <div style={{ marginBottom: 10 }}>
        <div style={{ fontWeight: 700, color: "#1D4ED8", marginBottom: 4 }}>🏠 שטחי חדרים</div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr style={{ background: "#F8FAFC" }}>
              <th style={{ textAlign: "right", padding: "3px 6px", borderBottom: "1px solid #E2E8F0" }}>חדר</th>
              <th style={{ textAlign: "center", padding: "3px 6px", borderBottom: "1px solid #E2E8F0" }}>מ"ר</th>
            </tr>
          </thead>
          <tbody>
            {boqData.sections.rooms.rows.map((r, i) => (
              <tr key={i} style={{ borderBottom: "1px solid #F1F5F9" }}>
                <td style={{ padding: "2px 6px" }}>{r.name}</td>
                <td style={{ textAlign: "center", padding: "2px 6px" }}>{r.area_m2 > 0 ? r.area_m2 : "—"}</td>
              </tr>
            ))}
            <tr style={{ fontWeight: 700, background: "#F0FDF4" }}>
              <td style={{ padding: "3px 6px" }}>סה"כ</td>
              <td style={{ textAlign: "center", padding: "3px 6px" }}>{boqData.sections.rooms.total_area_m2}</td>
            </tr>
          </tbody>
        </table>
      </div>
    )}

    {/* קירות */}
    {boqData.sections.walls.rows.length > 0 && (
      <div style={{ marginBottom: 10 }}>
        <div style={{ fontWeight: 700, color: "#059669", marginBottom: 4 }}>🧱 קירות</div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr style={{ background: "#F8FAFC" }}>
              <th style={{ textAlign: "right", padding: "3px 6px", borderBottom: "1px solid #E2E8F0" }}>סוג</th>
              <th style={{ textAlign: "center", padding: "3px 6px", borderBottom: "1px solid #E2E8F0" }}>מ׳</th>
              <th style={{ textAlign: "center", padding: "3px 6px", borderBottom: "1px solid #E2E8F0" }}>מ"ר</th>
            </tr>
          </thead>
          <tbody>
            {boqData.sections.walls.rows.map((w, i) => (
              <tr key={i} style={{ borderBottom: "1px solid #F1F5F9" }}>
                <td style={{ padding: "2px 6px", display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: w.color, display: "inline-block" }} />
                  {w.label}
                </td>
                <td style={{ textAlign: "center", padding: "2px 6px" }}>{w.total_length_m}</td>
                <td style={{ textAlign: "center", padding: "2px 6px" }}>{w.wall_area_m2}</td>
              </tr>
            ))}
            <tr style={{ fontWeight: 700, background: "#F0FDF4" }}>
              <td style={{ padding: "3px 6px" }}>סה"כ</td>
              <td style={{ textAlign: "center", padding: "3px 6px" }}>{boqData.sections.walls.total_length_m}</td>
              <td style={{ textAlign: "center", padding: "3px 6px" }}>{boqData.sections.walls.total_area_m2}</td>
            </tr>
          </tbody>
        </table>
      </div>
    )}

    {/* פתחים + אביזרים */}
    <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
      <div style={{ flex: 1, background: "#F8FAFC", borderRadius: 8, padding: "6px 8px" }}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>🚪 דלתות</div>
        <div style={{ fontSize: 16, fontWeight: 700, color: "#1D4ED8" }}>{boqData.sections.doors.count}</div>
      </div>
      <div style={{ flex: 1, background: "#F8FAFC", borderRadius: 8, padding: "6px 8px" }}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>🪟 חלונות</div>
        <div style={{ fontSize: 16, fontWeight: 700, color: "#059669" }}>{boqData.sections.windows.count}</div>
      </div>
    </div>

    {boqData.sections.fixtures.rows.length > 0 && (
      <div style={{ marginBottom: 10 }}>
        <div style={{ fontWeight: 700, color: "#0EA5E9", marginBottom: 4 }}>🔧 אביזרים</div>
        {boqData.sections.fixtures.rows.map((f, i) => (
          <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "2px 0", borderBottom: "1px solid #F1F5F9" }}>
            <span>{f.label}</span>
            <span style={{ fontWeight: 600 }}>{f.count}</span>
          </div>
        ))}
      </div>
    )}
  </div>
)}
```

---

## שינוי 5: .env — הוסף OPENAI_API_KEY

**קובץ: `.env`** (בשורש הפרויקט)

הוסף שורה:
```
OPENAI_API_KEY=your-openai-api-key-here
```

---

## שינוי 6: התקנת חבילת openai

הרץ בטרמינל מתיקיית הפרויקט:
```bash
pip install openai --break-system-packages
# או
pip install openai
```

---

## מה לבדוק אחרי ביצוע השינויים

1. **Backend עולה ללא שגיאות**: `py -m uvicorn backend.main:app --reload`
2. **DELETE endpoint**: `DELETE /manager/planning/{id}/auto-segments/{seg_id}` מחזיר `{"deleted": 1}`
3. **BOQ endpoint**: `GET /manager/planning/{id}/boq-summary` מחזיר JSON עם rooms/walls/doors
4. **Frontend**: בשלב 3, אחרי ניתוח, מופיע filter bar (הכל/בטוח/לא בטוח)
5. **Hover על סגמנט**: מופיע × אדום + badge % confidence
6. **לחיצה על ×**: הסגמנט נעלם מה-canvas וה-sidebar
7. **כפתור "כתב כמויות"**: נטען ומציג טבלה בפאנל הצדדי
8. **OpenAI**: בלוג הbackend מופיע `[vision] OpenAI Vision supplement enabled (GPT-4o)` אם יש מפתח

---

## פלט נדרש

בסיום, דווח:
- אילו קבצים שונו
- כמה שורות נוספו / הוסרו לכל קובץ
- האם ה-openai package הותקן בהצלחה
- האם ה-backend עלה ללא שגיאות syntax
- כל שגיאה שנתקלת בה
