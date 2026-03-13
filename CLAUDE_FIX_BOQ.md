# CLAUDE CODE PROMPT — תיקון כפתור "כתב כמויות" (503 error)

## הבעיה

לחיצה על "📊 כתב כמויות" מחזירה 503 מהשרת ולא עושה כלום.

**שורש הבעיה 1:** `vision_analysis` נשמר ב-DB בנפרד (`db_save_vision_analysis`) אבל לא נטען חזרה לאובייקט `proj` בזיכרון.
כך ש-`proj.get("vision_analysis")` תמיד מחזיר `None` → `vision = {}` → endpoint קורס על משהו בהמשך.

**שורש הבעיה 2:** ה-endpoint `manager_boq_summary` אין לו `try/except` כללי — כל Python exception גורמת ל-503 ללא לוג שימושי.

---

## תיקון — `backend/main.py`

**מצא** את ה-endpoint:
```python
@app.get("/manager/planning/{plan_id}/boq-summary")
async def manager_boq_summary(plan_id: str):
```

**החלף** את כל גוף הפונקציה בקוד הבא:

```python
@app.get("/manager/planning/{plan_id}/boq-summary")
async def manager_boq_summary(plan_id: str):
    """
    מחשב כתב כמויות מקוצר ממידע Vision + CV2.
    מחזיר: שטחי חדרים, אורך+שטח קירות לפי סוג, ספירת פתחים ואביזרים.
    """
    try:
        proj = _get_project(plan_id)
        if not proj:
            raise HTTPException(status_code=404, detail="Plan not found")

        FLOOR_HEIGHT_M = 2.40

        # ── 1. טען vision_analysis — מזיכרון או מ-DB ──────────────────────────
        vision = proj.get("vision_analysis")
        if not vision or not isinstance(vision, dict):
            try:
                vision = db_get_vision_analysis(plan_id) or {}
                if vision and isinstance(vision, dict):
                    proj["vision_analysis"] = vision  # cache לשימוש עתידי
                    print(f"[boq] Loaded vision_analysis from DB for {plan_id}")
            except Exception as _ve:
                print(f"[boq] Could not load vision_analysis from DB: {_ve}")
                vision = {}

        # ── 2. חדרים מ-Vision ─────────────────────────────────────────────────
        vision_rooms_raw = vision.get("rooms") or []
        rooms_table = []
        total_built_area = 0.0
        for r in vision_rooms_raw:
            if not isinstance(r, dict):
                continue
            name = r.get("name") or r.get("room_name") or "חדר"
            area = float(r.get("area_m2") or r.get("area") or 0.0)
            rooms_table.append({"name": name, "area_m2": round(area, 1)})
            total_built_area += area

        # ── 3. קירות מ-CV2 segments ───────────────────────────────────────────
        _init_planning_if_missing(proj)
        segments_raw = proj["planning"].get("auto_segments") or []
        if not segments_raw:
            try:
                segments_raw = db_get_auto_segments(plan_id) or []
            except Exception as _se:
                print(f"[boq] db_get_auto_segments failed: {_se}")
                segments_raw = []

        wall_groups: dict = {
            "exterior":  {"label": "קירות חיצוניים",     "segments": [], "color": "#1D4ED8"},
            "interior":  {"label": "קירות פנימיים",      "segments": [], "color": "#059669"},
            "partition": {"label": "קירות גבס / הפרדה",  "segments": [], "color": "#D97706"},
            "other":     {"label": "קירות אחרים",         "segments": [], "color": "#6B7280"},
        }
        fixture_counts: dict = {}

        for s in segments_raw:
            d = s if isinstance(s, dict) else (s.model_dump() if hasattr(s, "model_dump") else {})
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
                "wall_type":     grp["label"],
                "wall_type_key": grp_key,
                "color":         grp["color"],
                "count":         len(segs),
                "total_length_m": total_len,
                "wall_area_m2":   total_area,
            })

        # ── 4. פתחים מ-Vision ─────────────────────────────────────────────────
        vision_elements = vision.get("elements") or []
        door_count = 0
        window_count = 0
        stair_count = 0
        for el in vision_elements:
            if not isinstance(el, dict):
                continue
            etype = (el.get("type") or "").lower()
            if "דלת" in etype or "door" in etype:
                door_count += 1
            elif "חלון" in etype or "window" in etype:
                window_count += 1
            elif "מדרגות" in etype or "stair" in etype or "מעלית" in etype or "elevator" in etype:
                stair_count += 1

        # תוספת מ-OpenAI supplement אם יש
        oai = vision.get("openai_supplement") or {}
        if oai and isinstance(oai, dict):
            for d in (oai.get("doors") or []):
                if isinstance(d, dict):
                    door_count += int(d.get("count", 0))
            for w in (oai.get("windows") or []):
                if isinstance(w, dict):
                    window_count += int(w.get("count", 0))
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

        # ── 5. סיכום ──────────────────────────────────────────────────────────
        total_wall_len  = round(sum(g["total_length_m"] for g in walls_table), 2)
        total_wall_area = round(sum(g["wall_area_m2"]   for g in walls_table), 2)

        plan_meta = proj.get("metadata") or {}
        result = {
            "rooms":              rooms_table,
            "walls":              walls_table,
            "door_count":         door_count,
            "window_count":       window_count,
            "stair_count":        stair_count,
            "fixture_counts":     fixture_counts,
            "total_rooms":        len(rooms_table),
            "total_area_m2":      round(total_built_area, 2),
            "total_wall_length_m": total_wall_len,
            "total_wall_area_m2":  total_wall_area,
            "plan_title":  vision.get("plan_title")  or plan_meta.get("filename", ""),
            "scale":       vision.get("scale")        or plan_meta.get("scale_str", ""),
            "floor_height_m": FLOOR_HEIGHT_M,
        }
        print(f"[boq] {plan_id}: rooms={len(rooms_table)} walls={len(walls_table)} "
              f"doors={door_count} windows={window_count} fixtures={len(fixture_counts)}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[boq] UNHANDLED ERROR for {plan_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"BOQ calculation error: {str(e)}")
```

---

## בדיקה אחרי שינוי

1. שמור קובץ ← uvicorn יטען מחדש אוטומטית (--reload)
2. לחץ "📊 כתב כמויות" בדפדפן
3. **בדפדפן**: פתח DevTools → Network → בדוק status קריאת `boq-summary`
4. **בטרמינל** (uvicorn): חפש שורות `[boq]` — תראה:
   - `[boq] Loaded vision_analysis from DB for ...` — אם נטען מ-DB
   - `[boq] BY_01-A-P.1.pdf: rooms=X walls=Y doors=Z` — אם הצליח
   - `[boq] UNHANDLED ERROR for ...` + traceback — אם עדיין נכשל (ספר לי מה כתוב)

## פלט נדרש

דווח:
- מה מופיע בטרמינל אחרי לחיצה על הכפתור
- מה ה-status code שחוזר עכשיו (200 / 404 / 500)
- אם חוזר 200 — האם הפאנל מופיע?
