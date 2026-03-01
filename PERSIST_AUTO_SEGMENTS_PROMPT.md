# Prompt ל-Claude Code — שמירת תוצאות ניתוח אוטומטי לדאטהבייס

---

## הדבק את הפרומפט הבא ל-Claude Code:

---

```
אתה עובד על פרויקט React + FastAPI. המשימה: לשמור תוצאות ניתוח אוטומטי (auto_segments)
לדאטהבייס, כדי שישרדו רענון דף ואתחול שרת.

כלל מוחלט: בצע אך ורק את השינויים המפורטים להלן. ללא רפקטור. ללא שינויים נוספים.
לפני כל שינוי כתוב: "משנה [קובץ] – [מה משתנה]"
אחרי כל שינוי כתוב: "הושלם ✓"
הצג diff בלבד.

מנגנון בטיחות: לפני כל עריכה — קרא את הקובץ וודא שהקוד המתואר קיים בדיוק.

---

## STEP 1 — backend/models.py: הוסף auto_segments ל-PlanningState

לפני שינוי: קרא את הקובץ. חפש:
  class PlanningState(BaseModel):
ודא שהיא קיימת. חפש בה:
  sections: list[WorkSection] = []
זו השורה האחרונה של הModel.

הוסף שורה אחת אחרי sections:
  auto_segments: list[AutoAnalyzeSegment] = []

# לפני:
class PlanningState(BaseModel):
    plan_id: str
    ...
    sections: list[WorkSection] = []

# אחרי:
class PlanningState(BaseModel):
    plan_id: str
    ...
    sections: list[WorkSection] = []
    auto_segments: list[AutoAnalyzeSegment] = []

אל תגע ב: שאר שדות הmodel, AutoAnalyzeSegment עצמו.

---

## STEP 2 — backend/main.py: כלול auto_segments ב-_build_planning_state

לפני שינוי: קרא את הקובץ. חפש:
  def _build_planning_state(plan_id: str, proj: Dict) -> PlanningState:
ודא שהיא קיימת. חפש בה את ה-return:
  return PlanningState(
      plan_id=plan_id,
      ...
      sections=sections,
  )
ודא שהיא מסתיימת ב-sections=sections.

הוסף שדה אחד לפני סגירת ה-return:

# לפני:
    return PlanningState(
        plan_id=plan_id,
        plan_name=...,
        ...
        sections=sections,
    )

# אחרי:
    raw_auto_segments = planning.get("auto_segments", [])
    auto_segments_out = [
        AutoAnalyzeSegment(**s) if isinstance(s, dict) else s
        for s in raw_auto_segments
    ]

    return PlanningState(
        plan_id=plan_id,
        plan_name=...,
        ...
        sections=sections,
        auto_segments=auto_segments_out,
    )

אל תגע ב: שאר לוגיקת הפונקציה, חישוב categories/items/boq.

---

## STEP 3 — backend/main.py: שמור segments בסוף auto-analyze endpoint

לפני שינוי: קרא את הקובץ. חפש:
  @app.post("/manager/planning/{plan_id}/auto-analyze", response_model=AutoAnalyzeResponse)
ודא שהיא קיימת. חפש בה:
  return AutoAnalyzeResponse(
      segments=walls_out + fixtures_out,
      vision_data=vision_data,
  )
זהו ה-return הראשי (לפני ה-except).

הוסף לפני ה-return הזה:

# לפני:
        return AutoAnalyzeResponse(
            segments=walls_out + fixtures_out,
            vision_data=vision_data,
        )

# אחרי:
        # ── שמור תוצאות ניתוח לדאטהבייס (persist ל-planning.auto_segments) ──
        all_segments = walls_out + fixtures_out
        _init_planning_if_missing(proj)
        proj["planning"]["auto_segments"] = [s.model_dump() for s in all_segments]
        _persist_plan_to_database(plan_id, proj)

        return AutoAnalyzeResponse(
            segments=all_segments,
            vision_data=vision_data,
        )

אל תגע ב: חישוב walls_out/fixtures_out, vision_data, ה-except blocks.

שים לב: _persist_plan_to_database ו-_init_planning_if_missing כבר קיימות בקובץ.

---

## STEP 4 — frontend/src/api/planningApi.ts: הוסף auto_segments ל-PlanningState

לפני שינוי: קרא את הקובץ. חפש:
  export interface PlanningState {
ודא שהיא קיימת. חפש בה:
  sections: WorkSection[];
זו השורה האחרונה של הinterface.

הוסף שורה אחת אחרי sections:
  auto_segments?: AutoSegment[];

# לפני:
export interface PlanningState {
  plan_id: string;
  ...
  sections: WorkSection[];
}

# אחרי:
export interface PlanningState {
  plan_id: string;
  ...
  sections: WorkSection[];
  auto_segments?: AutoSegment[];
}

AutoSegment כבר מוגדר באותו קובץ — אל תגדיר אותו שוב.

---

## STEP 5 — frontend/src/pages/PlanningPage.tsx: שחזר autoSegments בטעינה

לפני שינוי: קרא את הקובץ. חפש:
  const loadPlanningState = React.useCallback(async (planId: string) => {
ודא שהיא קיימת. חפש בה:
  const state = await getPlanningState(planId);
  setPlanningState(state);
  setCategoriesDraft(state.categories);
ודא שאלה 3 השורות קיימות ברצף.

הוסף לאחר setCategoriesDraft:

# לפני:
  const loadPlanningState = React.useCallback(async (planId: string) => {
    const state = await getPlanningState(planId);
    setPlanningState(state);
    setCategoriesDraft(state.categories);
  }, []);

# אחרי:
  const loadPlanningState = React.useCallback(async (planId: string) => {
    const state = await getPlanningState(planId);
    setPlanningState(state);
    setCategoriesDraft(state.categories);
    // שחזר תוצאות ניתוח אוטומטי שנשמרו בדאטהבייס
    if (state.auto_segments && state.auto_segments.length > 0) {
      setAutoSegments(state.auto_segments);
      setAutoSelected(new Set(
        state.auto_segments
          .filter(s => s.suggested_subtype !== "פרט קטן")
          .map(s => s.segment_id)
      ));
    }
  }, []);

אל תגע ב: שאר לוגיקת loadPlanningState, useEffect שקורא לה, שאר state.

---

## בדיקה לאחר ביצוע

1. הרץ `npm run build` בתיקיית frontend — ודא 0 errors
2. בדוק שה-endpoint `/manager/planning/{plan_id}` מחזיר `auto_segments` כשהם קיימים
3. בדוק שלאחר ניתוח אוטומטי + רענון דף — הקירות והאביזרים חוזרים אוטומטית

## rollback מהיר
git restore backend/models.py backend/main.py frontend/src/api/planningApi.ts frontend/src/pages/PlanningPage.tsx
```
