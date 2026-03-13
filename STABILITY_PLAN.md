# תוכנית הטמעה – שיפור יציבות ומהירות ConTech
> **כלל ברזל:** שינויי לוגיקה מינימליים בלבד. ללא רפקטור, ללא שינוי API.
> כל שינוי מתועד: קובץ / פונקציה / מה בדיוק משתנה / סיכון / תלות.

---

## סטטוס: שינוי #7 כבר בוצע

**#7 — Thread pool → 4 workers:** כבר קיים `ThreadPoolExecutor(max_workers=4)` בשורה 169 של `backend/main.py`. אין מה לשנות.

---

## סדר ביצוע בטוח (מהבטוח לרגיש)

```
Step 1  →  vision_analyzer.py  (שחרור pixmap)       🟢 LOW
Step 2  →  WorkerPage.tsx       (הגבלת רשימות)       🟢 LOW
Step 3  →  DashboardPage.tsx    (הגבלת רשימות)       🟢 LOW
Step 4  →  brain.py             (לוגינג לחריגות)     🟢 LOW
Step 5  →  WorkerPage.tsx       (תיקון loadReports)  🟢 LOW (bug fix)
Step 6  →  WorkerPage.tsx       (AbortController)    🟡 MEDIUM
Step 7  →  WorkerPage.tsx       (useCallback)        🟡 MEDIUM
Step 8  →  database.py          (cache schema)       🟡 MEDIUM
Step 9  →  DashboardPage.tsx    (CSS → useEffect)    🟡 MEDIUM
```

**כלל:** בצע step אחד → `npm run dev` + בדוק browser → המשך.

---

## פירוט כל שינוי

---

### Step 1 — שחרור Pixmap
**קובץ:** `backend/vision_analyzer.py`
**פונקציה:** `_render_page_to_b64(page)`
**שורות קוד:** 289–290

**מצב נוכחי:**
```python
pix = page.get_pixmap(matrix=matrix, alpha=False)
img = _Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
```

**שינוי:**
```python
pix = page.get_pixmap(matrix=matrix, alpha=False)
img = _Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
pix = None  # שחרור pixmap מהזיכרון מיד לאחר שנוצר img
```

**סיכון:** 🟢 LOW — תוספת שורה אחת בלבד. אין שינוי לוגיקה.
**תלות:** אין.

---

### Step 2 — הגבלת רשימת דיווחים (WorkerPage)
**קובץ:** `frontend/src/pages/WorkerPage.tsx`
**מיקום:** שורה ~787 (בתוך טאב "היסטוריה")

**מצב נוכחי:**
```tsx
{reports.slice().reverse().map((r) => (
```

**שינוי:**
```tsx
{reports.slice(-50).reverse().map((r) => (
```

**סיכון:** 🟢 LOW — מגביל תצוגה ל-50 פריטים אחרונים. אין שינוי לוגיקה.
**תלות:** אין.

---

### Step 3 — הגבלת רשימת דיווחים (DashboardPage)
**קובץ:** `frontend/src/pages/DashboardPage.tsx`
**מיקום:** שורה 587 (בתוך "דיווחים בודדים")

**מצב נוכחי:**
```tsx
{summary.reports.slice().reverse().map((r) => (
```

**שינוי:**
```tsx
{summary.reports.slice(-50).reverse().map((r) => (
```

**סיכון:** 🟢 LOW — זהה ל-Step 2.
**תלות:** אין.

---

### Step 4 — לוגינג לחריגות בלופ המודלים
**קובץ:** `brain.py` (בתיקיית root, **לא** `backend/brain.py`)
**פונקציה:** `process_plan_metadata()`
**שורות:** ~229, ~271, ~277

**צעד 4a — הוסף import בראש הקובץ** (אחרי שורה 1 `import json` וכדומה):
```python
import logging
logger = logging.getLogger(__name__)
```

**בדוק תחילה אם `logging` כבר מיובא.** אם כן, הוסף רק את שורת `logger`.

**צעד 4b — שנה 3 בלוקים:**

בלוק 1 (שורה ~229, בתוך `try: result = json.loads(fixed)`):
```python
# לפני:
                except:
                    continue
# אחרי:
                except Exception as _e:
                    logger.warning("brain: json-fix fallback model=%s err=%s", model, _e)
                    continue
```

בלוק 2 (שורה ~271, בתוך `try: short_message` fallback):
```python
# לפני:
                except:
                    continue
# אחרי:
                except Exception as _e:
                    logger.warning("brain: short-text fallback model=%s err=%s", model, _e)
                    continue
```

בלוק 3 (שורה ~277, `except Exception: continue`):
```python
# לפני:
        except Exception:
            continue
# אחרי:
        except Exception as _e:
            logger.warning("brain: model=%s outer exception: %s", model, _e)
            continue
```

**סיכון:** 🟢 LOW — תוספת לוגינג בלבד. הזרימה (`continue`) לא משתנה.
**תלות:** אין.

---

### Step 5 — תיקון באג loadReports
**קובץ:** `frontend/src/pages/WorkerPage.tsx`
**פונקציה:** `saveReport()`
**שורה:** 504

**מצב נוכחי:**
```tsx
await loadReports(selectedPlanId);
```

**בעיה:** `loadReports` אינה פונקציה מוגדרת בקובץ. הפונקציה הנכונה היא `listWorkerReports` (מיובאת בשורה 9).

**שינוי:**
```tsx
setReports(await listWorkerReports(selectedPlanId));
```

**מנגנון בטיחות:** לפני שינוי — ודא שהשורה 504 מכילה `loadReports` ולא שם אחר.

**סיכון:** 🟢 LOW — תיקון באג. `listWorkerReports` כבר מיובאת ומשמשת בשורה 450.
**תלות:** אין.

---

### Step 6 — AbortController לטעינת דיווחים
**קובץ:** `frontend/src/pages/WorkerPage.tsx`
**מיקום:** `useEffect` בשורות 447–454 (טוען `listWorkerReports`)

**מצב נוכחי:**
```tsx
React.useEffect(() => {
  if (!selectedPlanId) return;
  let cancelled = false;
  listWorkerReports(selectedPlanId)
    .then(data => { if (!cancelled) setReports(data); })
    .catch(console.error);
  return () => { cancelled = true; };
}, [selectedPlanId]);
```

**שינוי:**
```tsx
React.useEffect(() => {
  if (!selectedPlanId) return;
  const controller = new AbortController();
  listWorkerReports(selectedPlanId, controller.signal)
    .then(data => { if (!controller.signal.aborted) setReports(data); })
    .catch(err => { if (!controller.signal.aborted) console.error(err); });
  return () => { controller.abort(); };
}, [selectedPlanId]);
```

**בנוסף — עדכן `workerApi.ts` פונקציה `listWorkerReports`:**
```ts
// לפני:
export async function listWorkerReports(planId: string): Promise<WorkerReport[]> {
  const { data } = await apiClient.get<WorkerReport[]>(`/worker/reports/${encodeURIComponent(planId)}`);
  return data;
}

// אחרי:
export async function listWorkerReports(planId: string, signal?: AbortSignal): Promise<WorkerReport[]> {
  const { data } = await apiClient.get<WorkerReport[]>(`/worker/reports/${encodeURIComponent(planId)}`, { signal });
  return data;
}
```

**מנגנון בטיחות:** ודא שקיימת הפונקציה `listWorkerReports` ב-`workerApi.ts` לפני עריכה.

**סיכון:** 🟡 MEDIUM — שינוי signature של פונקציה ב-API. הפרמטר חדש הוא optional — לא שובר שימושים אחרים.
**תלות:** Step 5 (מוודא ש-`listWorkerReports` פועלת תקין).

---

### Step 7 — useCallback עבור handleDrawComplete
**קובץ:** `frontend/src/pages/WorkerPage.tsx`
**פונקציה:** `handleDrawComplete`
**שורות:** 470–492

**מצב נוכחי:**
```tsx
const handleDrawComplete = async (payload: { object_type: DrawMode; raw_object: Record<string, unknown> }) => {
  if (!selectedPlanId) return;
  // ...
};
```

**שינוי:**
```tsx
const handleDrawComplete = React.useCallback(async (payload: { object_type: DrawMode; raw_object: Record<string, unknown> }) => {
  if (!selectedPlanId) return;
  // ... (גוף הפונקציה ללא שינוי)
}, [selectedPlanId, reportType, drawMode]);
```

**מנגנון בטיחות:** ודא שהפונקציה `handleDrawComplete` קיימת בשורות 470–492 לפני עריכה. ודא שהדיפנדנסיז שבמערך (`selectedPlanId`, `reportType`, `drawMode`) מוגדרים כ-state/prop בקומפוננטה.

**סיכון:** 🟡 MEDIUM — אופטימיזציית React. הגוף לא משתנה. deps מדויקים לפי שימוש בפונקציה.
**תלות:** אין.

---

### Step 8 — Cache לבדיקת סכמה ב-database.py
**קובץ:** `database.py` (בתיקיית root, **לא** `backend/database.py`)
**פונקציה:** `save_plan()`
**שורות:** ~195–216

**צעד 8a — הוסף משתנה module-level לפני `def save_plan`:**
```python
# cache of column names for 'plans' table — populated on first save_plan call
_plans_cols_cache: set | None = None
```

**צעד 8b — עדכן את תחילת `save_plan()`:**
```python
# לפני:
def save_plan(filename, plan_name, ...):
    ...
    conn_check = get_connection()
    if conn_check:
        try:
            cur_check = conn_check.cursor()
            if _is_real_postgres:
                cur_check.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name='plans'"
                )
                cols = {row["column_name"] for row in cur_check.fetchall()}
            else:
                cur_check.execute("PRAGMA table_info(plans)")
                cols = {row[1] for row in cur_check.fetchall()}
        except Exception:
            cols = set()
        finally:
            conn_check.close()
    else:
        cols = set()

# אחרי:
def save_plan(filename, plan_name, ...):
    global _plans_cols_cache
    ...
    if _plans_cols_cache is None:
        conn_check = get_connection()
        if conn_check:
            try:
                cur_check = conn_check.cursor()
                if _is_real_postgres:
                    cur_check.execute(
                        "SELECT column_name FROM information_schema.columns WHERE table_name='plans'"
                    )
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
```

**מנגנון בטיחות:** ודא שהבלוק עם `conn_check = get_connection()` קיים בתוך `save_plan` לפני עריכה. ודא שאין `global _plans_cols_cache` קיים.

**סיכון:** 🟡 MEDIUM — מוסיף global state. תקין כי הסכמה לא משתנה בזמן ריצה.
**תלות:** אין.

---

### Step 9 — CSS Injection → useEffect
**קובץ:** `frontend/src/pages/DashboardPage.tsx`
**מיקום:** שורות 14–26 (module level) + תחילת קומפוננטה `DashboardPage` (שורה 223)

**צעד 9a — מחק את הבלוק module-level (שורות 14–26):**
```tsx
// הסר לחלוטין:
// ─── Inject card-expand animation once ───────────────────────────────────────
const _expandStyle = document.createElement("style");
_expandStyle.textContent = `
  @keyframes card-expand {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .report-card-body { animation: card-expand 0.18s ease; }
`;
if (!document.head.querySelector("[data-dashboard-styles]")) {
  _expandStyle.setAttribute("data-dashboard-styles", "1");
  document.head.appendChild(_expandStyle);
}
```

**צעד 9b — הוסף useEffect בתחילת `DashboardPage` (אחרי ה-useState הראשונים, לפני useEffect הקיים):**
```tsx
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
```

**מנגנון בטיחות:** ודא שהבלוק `const _expandStyle = document.createElement("style")` קיים בשורות 14–26 לפני מחיקה.

**סיכון:** 🟡 MEDIUM — שינוי מיקום קוד קיים. ה-animation לא משתנה. הלוגיקה של `[data-dashboard-styles]` נשמרת.
**תלות:** אין.

---

## סיכום

| Step | קובץ | פונקציה | סיכון | סטטוס |
|------|------|---------|-------|--------|
| — | `main.py` | `_executor` | — | ✅ כבר בוצע |
| 1 | `backend/vision_analyzer.py` | `_render_page_to_b64` | 🟢 LOW | ⬜ |
| 2 | `WorkerPage.tsx` | JSX history tab | 🟢 LOW | ⬜ |
| 3 | `DashboardPage.tsx` | JSX reports list | 🟢 LOW | ⬜ |
| 4 | `brain.py` (root) | `process_plan_metadata` | 🟢 LOW | ⬜ |
| 5 | `WorkerPage.tsx` | `saveReport` | 🟢 LOW | ⬜ |
| 6 | `WorkerPage.tsx` + `workerApi.ts` | useEffect + `listWorkerReports` | 🟡 MEDIUM | ⬜ |
| 7 | `WorkerPage.tsx` | `handleDrawComplete` | 🟡 MEDIUM | ⬜ |
| 8 | `database.py` (root) | `save_plan` | 🟡 MEDIUM | ⬜ |
| 9 | `DashboardPage.tsx` | module-level + component | 🟡 MEDIUM | ⬜ |

**לאחר כל step:** `npm run dev` (frontend), בדוק console ב-browser, ואשר.
**rollback:** `git restore <file>` לכל קובץ ספציפי.
