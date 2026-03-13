# Prompt ל-Claude Code – שיפור יציבות ומהירות ConTech

---

## הדבק את הפרומפט הבא ל-Claude Code:

---

```
אתה עובד על פרויקט React + FastAPI בשלב פיילוט עם משתמשים אמיתיים.
כלל מוחלט: בצע רק את השינויים המפורטים להלן. ללא רפקטור. ללא שינוי API. ללא שינוי לוגיקה.

לפני כל שינוי כתוב בעברית: "משנה [שם קובץ] – [מה משתנה]"
אחרי כל שינוי כתוב בעברית: "הושלם – [מה השתנה בפועל]"
הצג רק diff. לא קבצים מלאים.

מנגנון בטיחות חובה לפני כל עריכה:
  - קרא את הקובץ (Read tool)
  - ודא שהפונקציה / הבלוק המתואר קיים בשורות הנכונות
  - רק אז בצע את השינוי

שינוי #7 (ThreadPoolExecutor) כבר בוצע – דלג עליו.

---

## STEP 1 – backend/vision_analyzer.py (סיכון LOW)

לפני שינוי: קרא את הקובץ, ודא שהפונקציה _render_page_to_b64 קיימת ושורות:
  pix = page.get_pixmap(...)
  img = _Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
קיימות ברצף.

בצע את השינוי הבא בלבד:
אחרי שורת img = _Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
הוסף: pix = None  # שחרור pixmap מהזיכרון מיד לאחר שנוצר img

אל תגע ב: לוגיקת הloop, הצמצום של quality, resize, החזרת base64.

---

## STEP 2 – frontend/src/pages/WorkerPage.tsx – הגבלת רשימה (סיכון LOW)

לפני שינוי: קרא את הקובץ, חפש את השורה:
  {reports.slice().reverse().map((r) => (
ודא שהיא קיימת (בסביבות שורה 787, בתוך טאב היסטוריה).

בצע שינוי אחד בלבד:
  reports.slice().reverse()  →  reports.slice(-50).reverse()

אל תגע ב: מבנה ה-JSX, לוגיקת הrender, כפתורים.

---

## STEP 3 – frontend/src/pages/DashboardPage.tsx – הגבלת רשימה (סיכון LOW)

לפני שינוי: קרא את הקובץ, חפש את השורה:
  {summary.reports.slice().reverse().map((r) => (
ודא שהיא קיימת (סביב שורה 587).

בצע שינוי אחד בלבד:
  summary.reports.slice().reverse()  →  summary.reports.slice(-50).reverse()

אל תגע ב: כל fetch, state, print logic.

---

## STEP 4 – brain.py (root, לא backend/brain.py) – לוגינג לחריגות (סיכון LOW)

לפני שינוי: קרא את הקובץ, חפש את 3 הבלוקים:
  (א) except: continue  (בתוך json-fix fallback)
  (ב) except: continue  (בתוך short-text fallback)
  (ג) except Exception: continue  (בחוץ — לאחר NotFoundError, BadRequestError)
ודא שכל 3 קיימים בפונקציה process_plan_metadata.

בצע 2 פעולות:
1. בדוק אם "import logging" קיים בראש הקובץ. אם לא — הוסף.
   אחרי שורת ה-import הוסף: logger = logging.getLogger(__name__)
2. שנה את 3 הבלוקים:
   (א) except: continue  →  except Exception as _e: logger.warning("brain: json-fix model=%s %s", model, _e); continue
   (ב) except: continue  →  except Exception as _e: logger.warning("brain: short-text model=%s %s", model, _e); continue
   (ג) except Exception: continue  →  except Exception as _e: logger.warning("brain: outer model=%s %s", model, _e); continue

אל תגע ב: סדר הbלוקים, לוגיקת ה-continue, החזרת הresult.

---

## STEP 5 – frontend/src/pages/WorkerPage.tsx – תיקון באג loadReports (סיכון LOW)

לפני שינוי: קרא את הקובץ, חפש:
  await loadReports(selectedPlanId);
ודא שהיא קיימת (סביב שורה 504, בתוך saveReport).
ודא גם ש-listWorkerReports מיובאת בשורות הimport בראש הקובץ.

בצע שינוי אחד בלבד:
  await loadReports(selectedPlanId);
  →
  setReports(await listWorkerReports(selectedPlanId));

אל תגע ב: שאר הלוגיקה של saveReport, toast, state.

---

## STEP 6 – WorkerPage.tsx + workerApi.ts – AbortController (סיכון MEDIUM)

### 6a: frontend/src/api/workerApi.ts
לפני שינוי: קרא את הקובץ, חפש:
  export async function listWorkerReports(planId: string): Promise<WorkerReport[]>
ודא שהחתימה תואמת בדיוק.

שנה רק את ה-signature וה-get call:
  // לפני:
  export async function listWorkerReports(planId: string): Promise<WorkerReport[]> {
    const { data } = await apiClient.get<WorkerReport[]>(`/worker/reports/${encodeURIComponent(planId)}`);
  // אחרי:
  export async function listWorkerReports(planId: string, signal?: AbortSignal): Promise<WorkerReport[]> {
    const { data } = await apiClient.get<WorkerReport[]>(`/worker/reports/${encodeURIComponent(planId)}`, { signal });

אל תגע ב: שאר הפונקציות בקובץ, ה-return.

### 6b: frontend/src/pages/WorkerPage.tsx
לפני שינוי: קרא את הקובץ, חפש את useEffect שמכיל:
  let cancelled = false;
  listWorkerReports(selectedPlanId)
ודא שהוא קיים (סביב שורות 447–454).

החלף את כל הuseEffect הזה בלבד:
  // לפני:
  React.useEffect(() => {
    if (!selectedPlanId) return;
    let cancelled = false;
    listWorkerReports(selectedPlanId)
      .then(data => { if (!cancelled) setReports(data); })
      .catch(console.error);
    return () => { cancelled = true; };
  }, [selectedPlanId]);

  // אחרי:
  React.useEffect(() => {
    if (!selectedPlanId) return;
    const controller = new AbortController();
    listWorkerReports(selectedPlanId, controller.signal)
      .then(data => { if (!controller.signal.aborted) setReports(data); })
      .catch(err => { if (!controller.signal.aborted) console.error(err); });
    return () => { controller.abort(); };
  }, [selectedPlanId]);

אל תגע ב: שאר ה-useEffects בקובץ.

---

## STEP 7 – frontend/src/pages/WorkerPage.tsx – useCallback (סיכון MEDIUM)

לפני שינוי: קרא את הקובץ, חפש:
  const handleDrawComplete = async (payload: {
ודא שהיא קיימת (סביב שורה 470) ושהגוף מסתיים בשורה ~492 עם `};`.
ודא שהמשתנים selectedPlanId, reportType, drawMode מוגדרים כ-state בקומפוננטה.

עטוף את הפונקציה ב-useCallback:
  // לפני:
  const handleDrawComplete = async (payload: { object_type: DrawMode; raw_object: Record<string, unknown> }) => {
    ... // גוף הפונקציה
  };

  // אחרי:
  const handleDrawComplete = React.useCallback(async (payload: { object_type: DrawMode; raw_object: Record<string, unknown> }) => {
    ... // גוף הפונקציה — ללא שינוי
  }, [selectedPlanId, reportType, drawMode]);

אל תגע ב: גוף הפונקציה, imports, שאר הקוד.

---

## STEP 8 – database.py (root, לא backend/database.py) – Cache סכמה (סיכון MEDIUM)

לפני שינוי: קרא את הקובץ, חפש:
  def save_plan(
ודא שהיא קיימת. חפש בתוכה בלוק:
  conn_check = get_connection()
  if conn_check:
      try:
          cur_check = conn_check.cursor()
          if _is_real_postgres:
              cur_check.execute("SELECT column_name FROM information_schema.columns
ודא שהבלוק הזה קיים.

בצע 2 פעולות:
1. הוסף לפני שורת def save_plan (שורה ריקה + קוד):
   _plans_cols_cache: set | None = None

2. שנה את תחילת save_plan:
   הוסף בשורה הראשונה של הפונקציה: global _plans_cols_cache
   עטוף את בלוק conn_check בתנאי if _plans_cols_cache is None:
   בסוף כל ענף (SQLite / Postgres / except / else) שמור ב-_plans_cols_cache במקום ב-cols:
     _plans_cols_cache = { ... }
   לאחר הבלוק: cols = _plans_cols_cache

   הסוף אמור להיראות כך:
   if _plans_cols_cache is None:
       conn_check = get_connection()
       if conn_check:
           try:
               cur_check = conn_check.cursor()
               if _is_real_postgres:
                   cur_check.execute("SELECT column_name FROM information_schema.columns WHERE table_name='plans'")
                   _plans_cols_cache = {row["column_name"] for row in cur_check.fetchall()}
               else:
                   cur_check.execute("PRAGMA table_info(plans)")
                   _plans_cols_cache = {row[1] for row in cur_check.fetchall()}
           except Exception:
               _plans_cols_cache = set()
           finally:
               conn_check.close()
       else:
           _plans_cols_cache = set()
   cols = _plans_cols_cache

אל תגע ב: כל שאר לוגיקת save_plan (UPDATE/INSERT, schema branching, ה-run_query calls).

---

## STEP 9 – frontend/src/pages/DashboardPage.tsx – CSS → useEffect (סיכון MEDIUM)

לפני שינוי: קרא את הקובץ, חפש את הבלוק module-level:
  const _expandStyle = document.createElement("style");
ודא שהוא קיים בשורות 14–26 (לפני הפונקציות).
חפש גם שורה:
  export const DashboardPage: React.FC = () => {
ודא שהיא קיימת (סביב שורה 223).

בצע 2 פעולות:
1. מחק לחלוטין את שורות 14–26:
   // ─── Inject card-expand animation once ─────...
   const _expandStyle = document.createElement("style");
   _expandStyle.textContent = `...`;
   if (!document.head.querySelector("[data-dashboard-styles]")) {
     _expandStyle.setAttribute("data-dashboard-styles", "1");
     document.head.appendChild(_expandStyle);
   }

2. בתוך DashboardPage, אחרי ה-useState הראשון ולפני useEffect הקיים הראשון, הוסף:
   React.useEffect(() => {
     if (document.head.querySelector("[data-dashboard-styles]")) return;
     const el = document.createElement("style");
     el.setAttribute("data-dashboard-styles", "1");
     el.textContent = `
       @keyframes card-expand {
         from { opacity: 0; transform: translateY(-6px); }
         to   { opacity: 1; transform: translateY(0); }
       }
       .report-card-body { animation: card-expand 0.18s ease; }
     `;
     document.head.appendChild(el);
     return () => { el.remove(); };
   }, []);

אל תגע ב: כל שאר הקוד בקובץ, הanimation content (שמור מילה במילה).

---

## כללים לכל ה-steps

- הצג diff בלבד
- עברית לפני ואחרי כל שינוי
- לא לאחד steps
- אם ספק – אל תשנה, דווח בעברית על הספק
- בצע step אחד, דווח, המתן לאישור לפני הבא
- שינוי #7 (ThreadPoolExecutor) כבר בוצע — אל תגע ב-main.py

## פקודת rollback מהירה

git restore backend/vision_analyzer.py
git restore frontend/src/pages/WorkerPage.tsx
git restore frontend/src/pages/DashboardPage.tsx
git restore frontend/src/api/workerApi.ts
git restore brain.py
git restore database.py
```

---

## סדר ביצוע מומלץ

1. בצע Step 1–5 (LOW risk) ברצף ✓
2. בדוק browser/console
3. בצע Step 6–9 (MEDIUM risk) אחד אחד
4. בדוק browser/console אחרי כל אחד

## שים לב: שני זוגות קבצים

| קובץ בproduct | קובץ root (ישיר) |
|--------------|-----------------|
| `backend/brain.py` | `brain.py` ← **לשנות זה** |
| `backend/database.py` | `database.py` ← **לשנות זה** |
| `backend/vision_analyzer.py` | ← **לשנות זה** |

`brain.py` ו-`database.py` בroot הם הקבצים שנטענים בפועל (backend/*.py הם wrappers של `_compat.py`).
