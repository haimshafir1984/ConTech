# שדרוג ניתוח אוטומטי — שלב 3 הגדרת תכולה
## תיאור המשימה

השדרוג הזה נועד לתקן ולשפר דרסטית את הניתוח האוטומטי בשלב 3 ("סימון תכולה").
כרגע המערכת מוצאת 2 קירות בלבד מתוכנית שמכילה עשרות. המטרה: להחזיר פלט עשיר, מדויק, עם חזותיות מלאה על השרטוט.

---

## חלק 1 — Backend: `backend/vision_analyzer.py`

### 1.1 הרחבת ה-tool schema של Claude Vision

הוסף לתוך `_ARCH_TOOL["input_schema"]["properties"]` את הסעיפים הבאים (בנוסף לקיים):

```python
"walls": {
    "type": "array",
    "description": "כל הקירות הנראים בתוכנית עם סוג, חומר ומיקום משוער",
    "items": {
        "type": "object",
        "properties": {
            "wall_type": {
                "type": "string",
                "enum": ["exterior", "interior", "partition", "column", "shear_wall", "retaining"],
                "description": "exterior=קיר חיצוני, interior=קיר פנימי, partition=קיר גבס/הפרדה, column=עמוד, shear_wall=ליבה/גרעין"
            },
            "material": {
                "type": "string",
                "enum": ["בלוקים", "בטון", "גבס", "גבס_מוגן", "בלוקים_שחורים", "לא_ידוע"],
                "description": "חומר הקיר לפי המקרא"
            },
            "has_insulation": {
                "type": "boolean",
                "description": "האם יש בידוד חיצוני"
            },
            "fire_resistance": {
                "type": "string",
                "description": "עמידות אש אם מצוין, למשל REI120"
            },
            "approx_length_m": {
                "type": "number",
                "description": "אורך משוער במטרים לפי קנה מידה"
            },
            "x1_pct": {"type": "number", "description": "נקודת התחלה אופקית 0.0–1.0"},
            "y1_pct": {"type": "number", "description": "נקודת התחלה אנכית 0.0–1.0"},
            "x2_pct": {"type": "number", "description": "נקודת סיום אופקית 0.0–1.0"},
            "y2_pct": {"type": "number", "description": "נקודת סיום אנכית 0.0–1.0"},
            "location_desc": {"type": "string", "description": "תיאור מיקום: 'חזית צפון', 'בין חדר 101 ל-102'"},
        },
        "required": ["wall_type", "material"]
    }
},
"openings": {
    "type": "array",
    "description": "פתחים: דלתות וחלונות עם גדלים ומיקומים",
    "items": {
        "type": "object",
        "properties": {
            "opening_type": {
                "type": "string",
                "enum": ["door", "window", "sliding_door", "fire_door", "emergency_exit"],
            },
            "id": {"type": "string", "description": "מזהה כגון D1, ח2"},
            "width_cm": {"type": "number"},
            "height_cm": {"type": "number"},
            "sill_height_cm": {"type": "number", "description": "גובה אדן"},
            "lintel_height_cm": {"type": "number", "description": "גובה משקוף"},
            "room": {"type": "string", "description": "שם החדר שבו נמצא"},
            "x_pct": {"type": "number"},
            "y_pct": {"type": "number"},
        },
        "required": ["opening_type"]
    }
},
"stairs": {
    "type": "array",
    "description": "גרמי מדרגות",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "step_count": {"type": "integer"},
            "step_width_cm": {"type": "number"},
            "direction": {"type": "string", "enum": ["up", "down"]},
            "x_pct": {"type": "number"},
            "y_pct": {"type": "number"},
        }
    }
},
"total_wall_length_m": {
    "type": "number",
    "description": "אורך כולל משוער של כל הקירות במטרים"
},
```

### 1.2 הרחב את `_ARCH_SYSTEM` — הוסף בסוף:

```
13. **קירות** — זהה כל קיר לפי המקרא (גבס/בטון/בלוקים), קבע אם חיצוני/פנימי לפי מיקום ועובי,
    הוסף x1_pct,y1_pct,x2_pct,y2_pct כשברי עשרוני של הגודל הכולל של התמונה.
14. **פתחים** — דלתות (סמן D, ד, קשת פתיחה) וחלונות (קו כפול, UK/OK).
15. **מדרגות** — זהה גרמי מדרגות, ספור מדרגות, קבע כיוון (UP/DOWN מהחץ).
```

### 1.3 עדכן את `analyze_plan_with_vision` — הוסף לתוצאה:

```python
result["walls"]               = data.get("walls") or []
result["openings"]            = data.get("openings") or []
result["stairs"]              = data.get("stairs") or []
result["total_wall_length_m"] = data.get("total_wall_length_m")
```

### 1.4 שמור את הנתונים ב-meta ב-`main.py` (חפש בלוק `if vision_data:` — שורה ~1823):

```python
if vision_data.get("walls"):
    meta_clean["vision_walls"] = vision_data["walls"]
if vision_data.get("openings"):
    meta_clean["vision_openings"] = vision_data["openings"]
if vision_data.get("stairs"):
    meta_clean["vision_stairs"] = vision_data["stairs"]
if vision_data.get("total_wall_length_m"):
    meta_clean["vision_total_wall_length_m"] = vision_data["total_wall_length_m"]
```

---

## חלק 2 — Backend: `analyzer.py`

### 2.1 תקן את threshold לזיהוי קירות

חפש את המחלקה `FloorPlanAnalyzer` ואת הפונקציה שמייצרת `wall_mask` / `thick_walls`.
כל מקום שיש:
```python
_, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
```
החלף ב:
```python
# Multi-strategy thresholding — הרבה יותר חזק לתוכניות מגוונות
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
)
binary = cv2.bitwise_or(otsu, adaptive)
# נקה רעש קטן
clean_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_k, iterations=1)
```

וכל מקום שיש threshold קשיח לזיהוי קירות כבדים:
```python
_, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
# או
_, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
```
החלף גם אותם באותה שיטה (multi-strategy), אבל עם `blockSize=11` ו-`C=3` לאדפטיבי.

---

## חלק 3 — Backend: `main.py` — endpoint `/manager/planning/{plan_id}/auto-analyze`

### 3.1 עדכן `models.py` — הרחב `AutoAnalyzeSegment`

```python
class AutoAnalyzeSegment(BaseModel):
    segment_id: str
    label: str
    suggested_type: str
    suggested_subtype: str
    confidence: float
    length_m: float
    area_m2: float
    bbox: list[float]           # [x, y, w, h] natural coords
    element_class: str = "wall" # "wall" | "fixture" | "opening" | "stair" | "room"
    # --- שדות חדשים ---
    wall_type: str = "interior"          # "exterior"|"interior"|"partition"|"column"
    material: str = "לא_ידוע"            # "בלוקים"|"בטון"|"גבס"|"גבס_מוגן"|...
    has_insulation: bool = False
    fire_resistance: Optional[str] = None
    room_name: Optional[str] = None      # לסגמנטים מסוג room
    area_label: Optional[str] = None     # "99.0 מ״ר"
    category_color: Optional[str] = None # צבע מוצע להצגה
```

גם הרחב `AutoAnalyzeVisionData`:
```python
walls: Optional[list[dict]] = None
openings: Optional[list[dict]] = None
stairs: Optional[list[dict]] = None
total_wall_length_m: Optional[float] = None
```

### 3.2 תקן את סף הסינון בלולאה

חפש:
```python
if area_px < img_area * 0.0005:
    continue
```
שנה ל:
```python
if area_px < img_area * 0.00015:   # הורדה מ-0.05% → 0.015% לתפוס קירות קצרים
    continue
```

חפש גם:
```python
if area_px < img_area * 0.00008:
    continue
```
שנה ל:
```python
if area_px < img_area * 0.00005:
    continue
```

### 3.3 שדרג את לוגיקת הסיווג לקטגוריות עשירות

**החלף** את כל הבלוק שמייצר `AutoAnalyzeSegment` לקירות (החל מ `wall_counter += 1`):

```python
wall_counter += 1

# --- קביעת סוג קיר (חיצוני/פנימי/הפרדה) לפי גיאומטריה ---
# קיר חיצוני: ארוך + קרוב לשוליים (15% מהמסגרת)
margin_x = img_w * 0.15
margin_y = img_h * 0.15
near_edge = (
    g_bx < margin_x or g_bx + g_bw > img_w - margin_x or
    g_by < margin_y or g_by + g_bh > img_h - margin_y
)
is_long = length_m > 3.0   # יותר מ-3 מטר

if near_edge and is_long:
    wall_type = "exterior"
elif is_elongated and length_m > 1.5:
    wall_type = "interior"
elif length_m < 1.5 or (not is_elongated and area_m2 < 0.5):
    wall_type = "partition"
else:
    wall_type = "interior"

# --- קביעת חומר לפי concrete/blocks mask ---
material = "לא_ידוע"
has_insulation = False
confidence_val = 0.65

if has_concrete:
    roi_mask     = region_roi.astype(bool)
    concrete_roi = concrete_arr[y1:y2, x1:x2]
    blocks_roi   = (
        blocks_arr[y1:y2, x1:x2]
        if isinstance(blocks_arr, np.ndarray) and blocks_arr.size > 0
        else None
    )
    c_px = int(np.count_nonzero(concrete_roi[roi_mask]))
    b_px = int(np.count_nonzero(blocks_roi[roi_mask])) if blocks_roi is not None else 0
    total_roi_px = max(1, area_px)
    if c_px > b_px and c_px > total_roi_px * 0.3:
        material = "בטון"
        confidence_val = min(0.95, 0.75 + c_px / total_roi_px * 0.3)
    elif b_px > c_px and b_px > total_roi_px * 0.3:
        material = "בלוקים"
        confidence_val = min(0.92, 0.72 + b_px / total_roi_px * 0.3)
    elif is_elongated:
        material = "בטון"
        confidence_val = 0.70
    else:
        material = "בלוקים"
        confidence_val = 0.60
else:
    material = "בטון" if is_elongated else "בלוקים"
    confidence_val = 0.72 if is_elongated else 0.58

# --- בדיקת בידוד (קירות חיצוניים עם בידוד = עבים יותר) ---
wall_thickness_m = area_m2 / max(0.001, length_m)
has_insulation = wall_type == "exterior" and wall_thickness_m > 0.30

# --- label מפורט ---
type_label = {"exterior": "קיר חיצוני", "interior": "קיר פנימי",
              "partition": "קיר הפרדה", "column": "עמוד"}.get(wall_type, "קיר")
wall_label = f"{type_label} {wall_counter} — {material} {length_m:.1f}מ׳"

# --- category_color לצבע על השרטוט ---
color_map = {
    ("exterior", "בטון"):    "#1E40AF",   # כחול כהה
    ("exterior", "בלוקים"):  "#1D4ED8",   # כחול
    ("interior", "בטון"):    "#059669",   # ירוק כהה
    ("interior", "בלוקים"):  "#10B981",   # ירוק
    ("partition", "גבס"):    "#7C3AED",   # סגול
    ("partition", "גבס_מוגן"):"#6D28D9",  # סגול כהה
    ("partition", "בלוקים"): "#D97706",   # כתום
}
cat_color = color_map.get((wall_type, material), "#6B7280")

segments.append(AutoAnalyzeSegment(
    segment_id=f"seg_{label_id}",
    label=wall_label,
    suggested_type="קירות",
    suggested_subtype=f"{type_label}/{material}",
    confidence=round(float(confidence_val), 2),
    length_m=float(length_m),
    area_m2=float(area_m2),
    bbox=[float(g_bx), float(g_by), float(g_bw), float(g_bh)],
    element_class="wall",
    wall_type=wall_type,
    material=material,
    has_insulation=has_insulation,
    category_color=cat_color,
))
```

### 3.4 שדרג את fixture detection לסגמנטים עשירים

בלוק ה-fixture (חפש `is_fixture = True`), החלף את ה-`segments.append` ב:

```python
# סיווג fixture מדויק לפי aspect + area
if area_m2 < 0.12:
    subtype, ftype, color = "פרט קטן", "אביזר", "#9CA3AF"
elif 0.5 <= aspect <= 2.0 and area_m2 < 0.45:
    subtype, ftype, color = "כיור / אסלה", "אביזרים סניטריים", "#0EA5E9"
elif area_m2 < 1.2 and (aspect > 1.6 or aspect < 0.62):
    subtype, ftype, color = "אמבטיה / מקלחת", "אביזרים סניטריים", "#38BDF8"
elif area_m2 > 1.0:
    subtype, ftype, color = "מדרגות / מעלית", "תחבורה אנכית", "#F59E0B"
else:
    subtype, ftype, color = "ריהוט / מכשיר", "ריהוט", "#A3E635"

fixture_label = f"{subtype} {fixture_counter}"
segments.append(AutoAnalyzeSegment(
    segment_id=f"fix_{label_id}",
    label=fixture_label,
    suggested_type=ftype,
    suggested_subtype=subtype,
    confidence=0.55,
    length_m=float(length_m),
    area_m2=float(area_m2),
    bbox=[float(g_bx), float(g_by), float(g_bw), float(g_bh)],
    element_class="fixture",
    category_color=color,
))
```

### 3.5 הוסף segmentים מ-Vision (rooms + walls מ-Vision)

אחרי `all_segments = walls_out + fixtures_out` (שורה ~3048), **לפני** ה-`_persist_plan_to_database`, הוסף:

```python
# ── הוסף חדרים מ-Vision כ-segmentים מסוג "room" ────────────────────────
vision_rooms = (proj.get("metadata") or {}).get("llm_rooms") or []
if vision_rooms:
    img_h_f, img_w_f = float(img_h), float(img_w)
    for i, room in enumerate(vision_rooms):
        name  = room.get("name") or f"חדר {i+1}"
        area  = room.get("area_m2")
        x_pct = room.get("position_x_pct") or 0.5
        y_pct = room.get("position_y_pct") or 0.5
        # bbox: קטן סביב מיקום התווית
        lw, lh = img_w_f * 0.07, img_h_f * 0.04
        lx = max(0, x_pct * img_w_f - lw / 2)
        ly = max(0, y_pct * img_h_f - lh / 2)
        area_label = f"{area:.1f} מ״ר" if area else None
        all_segments.append(AutoAnalyzeSegment(
            segment_id=f"room_{i}",
            label=name,
            suggested_type="חדרים",
            suggested_subtype=name,
            confidence=0.90,
            length_m=0.0,
            area_m2=float(area or 0),
            bbox=[float(lx), float(ly), float(lw), float(lh)],
            element_class="room",
            room_name=name,
            area_label=area_label,
            category_color="#F0FDF4",
        ))

# ── הוסף קירות מ-Vision שלא כוסו ע"י CV2 ──────────────────────────────
vision_walls = (proj.get("metadata") or {}).get("vision_walls") or []
if vision_walls:
    existing_bboxes = [(s.bbox[0], s.bbox[1], s.bbox[0]+s.bbox[2], s.bbox[1]+s.bbox[3])
                       for s in all_segments if s.element_class == "wall"]
    for i, vw in enumerate(vision_walls):
        x1p = vw.get("x1_pct") or 0; y1p = vw.get("y1_pct") or 0
        x2p = vw.get("x2_pct") or 0; y2p = vw.get("y2_pct") or 0
        if not (x1p or x2p): continue
        vx = min(x1p, x2p) * img_w_f
        vy = min(y1p, y2p) * img_h_f
        vw2 = max(5, abs(x2p - x1p) * img_w_f)
        vh2 = max(5, abs(y2p - y1p) * img_h_f)
        # בדוק אם כבר מכוסה
        cx, cy = vx + vw2/2, vy + vh2/2
        covered = any(
            ex1 <= cx <= ex2 and ey1 <= cy <= ey2
            for ex1, ey1, ex2, ey2 in existing_bboxes
        )
        if covered: continue
        wtype = vw.get("wall_type", "interior")
        mat   = vw.get("material", "לא_ידוע")
        lm    = float(vw.get("approx_length_m") or 0)
        color_map2 = {"exterior":"#1D4ED8","interior":"#10B981","partition":"#D97706"}
        all_segments.append(AutoAnalyzeSegment(
            segment_id=f"vwall_{i}",
            label=f"קיר Vision {i+1} — {mat}",
            suggested_type="קירות",
            suggested_subtype=f"{wtype}/{mat}",
            confidence=0.75,
            length_m=lm,
            area_m2=0.0,
            bbox=[float(vx), float(vy), float(vw2), float(vh2)],
            element_class="wall",
            wall_type=wtype,
            material=mat,
            category_color=color_map2.get(wtype, "#6B7280"),
        ))
```

### 3.6 עדכן את `_build_vision_data` — הוסף את השדות החדשים

```python
vd = AutoAnalyzeVisionData(
    # ... כל השדות הקיימים ...
    walls=_lst(m.get("vision_walls")),
    openings=_lst(m.get("vision_openings")),
    stairs=_lst(m.get("vision_stairs")),
    total_wall_length_m=_flt(m.get("vision_total_wall_length_m")),
)
```

---

## חלק 4 — Frontend: `frontend/src/pages/PlanningPage.tsx`

### 4.1 הוסף state לסינון קטגוריות על הקנבס

מצא את בלוק ה-`useState` בתחילת הקומפוננטה (**שורה ~660-720**) והוסף:

```typescript
// Category highlight mode: click category → show only that category on canvas
const [highlightedClass, setHighlightedClass] = React.useState<string | null>(null);
// e.g. "exterior", "interior", "partition", "fixture", "room"
const [highlightedType, setHighlightedType] = React.useState<string | null>(null);
// e.g. "קירות חיצוניים", "אביזרים סניטריים"
```

### 4.2 עדכן את type `AutoSegment` (או ייבא אותו מ-API)

מצא בקובץ את הגדרת `AutoSegment` (חפש `type AutoSegment`) והרחב:

```typescript
type AutoSegment = {
  segment_id: string;
  label: string;
  suggested_type: string;
  suggested_subtype: string;
  confidence: number;
  length_m: number;
  area_m2: number;
  bbox: [number, number, number, number];
  element_class: string;
  // שדות חדשים
  wall_type?: string;       // "exterior"|"interior"|"partition"|"column"
  material?: string;        // "בלוקים"|"בטון"|"גבס"|...
  has_insulation?: boolean;
  fire_resistance?: string;
  room_name?: string;
  area_label?: string;
  category_color?: string;  // hex color
};
```

### 4.3 פאנל קטגוריות בצד שמאל — "מצב הדגשת קטגוריה"

**מצא** את הבלוק שמציג את קבוצות הסגמנטים (שורה ~2125–2250, הבלוק שמתחיל `const groups = autoSegments.reduce`).

**החלף** את כל הבלוק הזה ב-UI חדש:

```tsx
{/* ─── CATEGORY HIGHLIGHT PANEL ─── */}
{autoSegments !== null && autoSegments.length > 0 && (() => {
  // בנה קבוצות לפי wall_type / element_class
  const categoryGroups: Array<{
    key: string;
    label: string;
    icon: string;
    color: string;
    segments: AutoSegment[];
    totalLength: number;
    totalArea: number;
  }> = [];

  const groupDefs = [
    { key: "exterior",   label: "קירות חיצוניים", icon: "🏗️", color: "#1D4ED8", filter: (s: AutoSegment) => s.element_class === "wall" && s.wall_type === "exterior" },
    { key: "interior",   label: "קירות פנימיים",  icon: "🧱", color: "#059669", filter: (s: AutoSegment) => s.element_class === "wall" && s.wall_type === "interior" },
    { key: "partition",  label: "קירות הפרדה",    icon: "📐", color: "#D97706", filter: (s: AutoSegment) => s.element_class === "wall" && (s.wall_type === "partition" || s.wall_type === "column") },
    { key: "wall_unknown",label:"קירות אחרים",    icon: "⬜", color: "#6B7280", filter: (s: AutoSegment) => s.element_class === "wall" && !["exterior","interior","partition","column"].includes(s.wall_type ?? "") },
    { key: "sanitary",   label: "אביזרים סניטריים",icon:"🚿", color: "#0EA5E9", filter: (s: AutoSegment) => s.element_class === "fixture" && (s.suggested_subtype?.includes("כיור") || s.suggested_subtype?.includes("אמבטיה") || s.suggested_subtype?.includes("מקלחת")) },
    { key: "stairs",     label: "מדרגות / מעלית", icon: "🪜", color: "#F59E0B", filter: (s: AutoSegment) => s.element_class === "fixture" && s.suggested_subtype?.includes("מדרגות") },
    { key: "furniture",  label: "ריהוט / מכשירים",icon:"🪑", color: "#A3E635", filter: (s: AutoSegment) => s.element_class === "fixture" && !s.suggested_subtype?.includes("כיור") && !s.suggested_subtype?.includes("אמבטיה") && !s.suggested_subtype?.includes("מדרגות") },
    { key: "room",       label: "חדרים ומרחבים",  icon: "🏠", color: "#8B5CF6", filter: (s: AutoSegment) => s.element_class === "room" },
  ];

  for (const def of groupDefs) {
    const segs = autoSegments.filter(def.filter);
    if (segs.length === 0) continue;
    categoryGroups.push({
      key: def.key,
      label: def.label,
      icon: def.icon,
      color: def.color,
      segments: segs,
      totalLength: segs.reduce((s, x) => s + (x.length_m ?? 0), 0),
      totalArea:   segs.reduce((s, x) => s + (x.area_m2 ?? 0), 0),
    });
  }

  const isHighlighting = highlightedClass !== null;

  return (
    <div style={{ direction: "rtl" }}>
      {/* כותרת + כפתור איפוס */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: "var(--navy)" }}>קטגוריות מזוהות</span>
        {isHighlighting && (
          <button type="button"
            onClick={() => { setHighlightedClass(null); setHighlightedType(null); }}
            style={{ fontSize: 11, color: "#3B82F6", background: "none", border: "none", cursor: "pointer", padding: "2px 6px" }}>
            הצג הכל ✕
          </button>
        )}
      </div>

      {/* רשימת קטגוריות */}
      {categoryGroups.map(grp => {
        const isActive = highlightedClass === grp.key;
        return (
          <div key={grp.key}
            onClick={() => {
              if (isActive) { setHighlightedClass(null); setHighlightedType(null); }
              else { setHighlightedClass(grp.key); setHighlightedType(grp.label); }
            }}
            style={{
              border: `2px solid ${isActive ? grp.color : "var(--s200)"}`,
              borderRadius: 8,
              padding: "8px 10px",
              marginBottom: 6,
              cursor: "pointer",
              background: isActive ? `${grp.color}15` : "white",
              transition: "all 0.15s",
              userSelect: "none",
            }}>
            {/* שורה ראשית */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ fontSize: 15 }}>{grp.icon}</span>
                <span style={{ fontSize: 12, fontWeight: isActive ? 700 : 600, color: isActive ? grp.color : "var(--navy)" }}>
                  {grp.label}
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{
                  background: isActive ? grp.color : "var(--s100)",
                  color: isActive ? "white" : "var(--s600)",
                  borderRadius: 12, fontSize: 11, fontWeight: 700,
                  padding: "1px 7px",
                }}>
                  {grp.segments.length}
                </span>
                {isActive ? "▲" : "▼"}
              </div>
            </div>
            {/* סיכום כמות */}
            {grp.totalLength > 0 && (
              <div style={{ fontSize: 11, color: "var(--s500)", marginTop: 3 }}>
                סה"כ: {grp.totalLength.toFixed(1)} מ׳
              </div>
            )}
            {grp.totalArea > 0 && grp.key === "room" && (
              <div style={{ fontSize: 11, color: "var(--s500)", marginTop: 3 }}>
                שטח: {grp.totalArea.toFixed(0)} מ"ר
              </div>
            )}
            {/* רשימת סגמנטים כשהקטגוריה פעילה */}
            {isActive && (
              <div style={{ marginTop: 6, borderTop: `1px solid ${grp.color}40`, paddingTop: 6 }}>
                {grp.segments.map(seg => (
                  <div key={seg.segment_id}
                    onClick={e => {
                      e.stopPropagation();
                      setSelectedSegmentId(seg.segment_id);
                      const el = document.getElementById(`seg-bbox-${seg.segment_id}`);
                      el?.scrollIntoView({ behavior: "smooth", block: "center" });
                    }}
                    style={{
                      fontSize: 11, padding: "3px 4px", borderRadius: 4, cursor: "pointer",
                      background: selectedSegmentId === seg.segment_id ? `${grp.color}25` : "transparent",
                      display: "flex", justifyContent: "space-between", alignItems: "center",
                    }}>
                    <span style={{ color: "var(--navy)" }}>
                      {seg.element_class === "room" ? `📍 ${seg.room_name ?? seg.label}` : seg.label}
                    </span>
                    <span style={{ color: "var(--s400)" }}>
                      {seg.length_m > 0 ? `${seg.length_m.toFixed(1)}מ׳` : seg.area_label ?? ""}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
})()}
```

### 4.4 עדכן את שכבת ה-SVG overlay — הצג לפי קטגוריה מודגשת

מצא את ה-SVG שמציג את `autoSegments` (שורה ~1838):

```tsx
{step3Tab === "auto" && autoSegments !== null && autoSegments.length > 0 && (
  <svg ...>
    {autoSegments
      .filter(seg => seg.suggested_subtype !== "פרט קטן")
      .map((seg, idx) => {
```

**שנה** את הפילטר ואת הצביעה:

```tsx
{step3Tab === "auto" && autoSegments !== null && autoSegments.length > 0 && (
  <svg width={displaySize.width} height={displaySize.height} className="absolute inset-0"
    onClick={() => setContextMenu(null)}>

    {/* שכבת dimming כשיש הדגשת קטגוריה */}
    {highlightedClass && (
      <rect x={0} y={0} width={displaySize.width} height={displaySize.height}
        fill="rgba(0,0,0,0.45)" />
    )}

    {autoSegments
      .filter(seg => seg.suggested_subtype !== "פרט קטן")
      .map((seg, idx) => {
        const [bx, by, bw, bh] = seg.bbox.map(v => v * displayScale);
        const isSelected = selectedSegmentId === seg.segment_id;
        const isFixture = seg.element_class === "fixture";
        const isRoom = seg.element_class === "room";

        // חישוב קטגוריה לסגמנט
        const segCatKey = isRoom ? "room"
          : isFixture ? (
              seg.suggested_subtype?.includes("כיור") || seg.suggested_subtype?.includes("אמבטיה") ? "sanitary"
              : seg.suggested_subtype?.includes("מדרגות") ? "stairs" : "furniture"
            )
          : seg.wall_type === "exterior" ? "exterior"
          : seg.wall_type === "interior" ? "interior"
          : (seg.wall_type === "partition" || seg.wall_type === "column") ? "partition"
          : "wall_unknown";

        // בהדגשת קטגוריה: הסתר שאינם בקטגוריה הנבחרת
        if (highlightedClass && segCatKey !== highlightedClass) return null;

        // צבע מהמודל או fallback
        const baseColor = seg.category_color ??
          (isSelected ? "#F59E0B"
          : isFixture ? "#7C3AED"
          : seg.confidence >= 0.8 ? "#10B981"
          : seg.confidence >= 0.6 ? "#F59E0B" : "#EF4444");

        const fillOpacity = highlightedClass ? 0.55 : (autoSelected.has(seg.segment_id) ? 0.35 : 0.12);
        const strokeWidth = isSelected ? 3 : highlightedClass ? 2.5 : 1.5;

        // תווית טקסט
        const labelText = isRoom
          ? (seg.room_name ?? seg.label)
          : `${idx + 1}`;

        return (
          <g key={seg.segment_id} id={`seg-bbox-${seg.segment_id}`}
            style={{ cursor: "pointer" }}
            onClick={e => {
              e.stopPropagation();
              setAutoSelected(prev => {
                const n = new Set(prev);
                autoSelected.has(seg.segment_id) ? n.delete(seg.segment_id) : n.add(seg.segment_id);
                return n;
              });
              setSelectedSegmentId(seg.segment_id);
              const el = segmentListRefs.current[seg.segment_id];
              if (el) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }}
            onContextMenu={e => {
              e.preventDefault(); e.stopPropagation();
              const svgEl = e.currentTarget.closest("svg");
              const rect = svgEl?.getBoundingClientRect();
              if (rect) setContextMenu({ x: e.clientX - rect.left, y: e.clientY - rect.top, segId: seg.segment_id });
            }}
          >
            {/* fill rect */}
            <rect x={bx} y={by} width={bw} height={bh}
              fill={baseColor} fillOpacity={fillOpacity}
              stroke={baseColor} strokeWidth={strokeWidth}
              strokeDasharray={isRoom ? "none" : isFixture ? "5 3" : "none"}
              rx={isRoom ? 4 : 0}
            />
            {/* selected highlight ring */}
            {isSelected && (
              <rect x={bx-3} y={by-3} width={bw+6} height={bh+6}
                fill="none" stroke="#F59E0B" strokeWidth={2.5}
                strokeDasharray="5 3" rx={2} style={{ pointerEvents: "none" }} />
            )}
            {/* תווית */}
            <text x={bx + bw/2} y={by + bh/2 + 4}
              fill="white" fontSize={isRoom ? Math.max(10, Math.min(14, bw/8)) : Math.max(9, Math.min(12, bw/6))}
              fontWeight="700" textAnchor="middle"
              style={{ pointerEvents: "none" }}
              stroke="rgba(0,0,0,0.5)" strokeWidth={0.5}
            >
              {labelText}
            </text>
            {/* תווית שטח לחדרים */}
            {isRoom && seg.area_label && (
              <text x={bx + bw/2} y={by + bh/2 + 18}
                fill="white" fontSize={9} textAnchor="middle"
                style={{ pointerEvents: "none" }} fontWeight="400">
                {seg.area_label}
              </text>
            )}
          </g>
        );
      })}

    {/* context menu */}
    {contextMenu && (() => {
      // ... (קוד קיים ללא שינוי) ...
    })()}
  </svg>
)}
```

### 4.5 הוסף Legend/Badge בראש הקנבס כשיש הדגשת קטגוריה

מצא את ה-div שעוטף את התמונה + ה-SVG (`displaySize`), ומעל ה-SVG הוסף:

```tsx
{/* Category highlight banner */}
{highlightedType && (
  <div style={{
    position: "absolute", top: 8, left: "50%", transform: "translateX(-50%)",
    background: "rgba(0,0,0,0.75)", color: "white", borderRadius: 20,
    padding: "4px 16px", fontSize: 13, fontWeight: 700, zIndex: 10,
    pointerEvents: "none", direction: "rtl",
  }}>
    מוצג: {highlightedType} — {autoSegments?.filter(s => {
      const catKey = s.element_class === "room" ? "room"
        : s.element_class === "fixture" ? (
            s.suggested_subtype?.includes("כיור") ? "sanitary"
            : s.suggested_subtype?.includes("מדרגות") ? "stairs" : "furniture"
          )
        : s.wall_type === "exterior" ? "exterior"
        : s.wall_type === "interior" ? "interior"
        : "partition";
      return catKey === highlightedClass;
    }).length ?? 0} פריטים
  </div>
)}
```

---

## חלק 5 — בדיקות לאחר יישום

1. העלה PDF של תוכנית קומה קרקע ← בדוק כמה קירות מזוהים (צפוי: 15+)
2. לחץ על "קירות חיצוניים" בפאנל ← כל שאר האלמנטים מתעממים, רק קירות חיצוניים זוהרים
3. לחץ על "חדרים ומרחבים" ← ריבועי תווית עם שם חדר ושטח
4. לחץ שוב על קטגוריה פעילה ← מחזיר לתצוגה מלאה
5. סגמנט בודד ← לחיצה על שם בפאנל גוללת ומדגישה אותו על השרטוט
6. כפתור "הצג הכל" ← מבטל הדגשה

---

## סיכום קבצים לשינוי

| קובץ | שינויים |
|------|---------|
| `backend/vision_analyzer.py` | הוסף walls, openings, stairs ל-tool + system prompt |
| `backend/models.py` | הרחב `AutoAnalyzeSegment` + `AutoAnalyzeVisionData` |
| `backend/main.py` | תקן threshold, הורד filter, סיווג עשיר, rooms+walls מ-Vision |
| `analyzer.py` (root) | החלף threshold קשיח → adaptive+OTSU |
| `frontend/src/pages/PlanningPage.tsx` | הוסף state, category panel חדש, SVG overlay עם dimming |
