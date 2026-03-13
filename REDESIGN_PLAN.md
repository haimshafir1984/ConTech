# תוכנית הטמעה – עיצוב מחדש ConTech
> **כלל ברזל:** אין שינוי לוגיקה, API, state, חישובים, או קריאות backend.
> כל שינוי הוא ויזואלי בלבד (CSS / JSX מבנה / classNames).

---

## עקרונות עיצוב שאושרו

| טוקן | ערך | שימוש |
|------|-----|--------|
| `--navy` | `#1e3a5f` | ראשי – רקע sidebar, כותרות, כפתורים ראשיים |
| `--orange` | `#e67e22` | אקסנט – פעולות ראשיות, אינדיקטורים פעילים |
| `--green` | `#15803d` | הצלחה / בוצע |
| `--amber` | `#b45309` | אזהרה / חלקי |
| `--red` | `#dc2626` | שגיאה / פתוח |
| כפתורים | גובה מינ' 48px, טקסט ברור | ללא emoji, SVG icon אופציונלי |
| בדג'ים | טקסט + dot צבעוני | ללא emoji |
| גריד | Mobile First, media queries | לא כיווץ של desktop |

---

## שלב 1 – ניתוח HTML: קומפוננטות React

### קומפוננטות חדשות (לייצר ב-`src/components/`)

| קומפוננטה | קובץ | תיאור |
|-----------|------|--------|
| `StatCard` | `StatCard.tsx` | כרטיס מדד עם border-top צבעוני, ערך גדול, תווית |
| `StatusBadge` | `StatusBadge.tsx` | בדג' עם dot, צבעים: green/red/amber/blue/navy |
| `SectionDivider` | `SectionDivider.tsx` | קו הפרדה עם טקסט מרכזי |
| `ProgressBar` | `ProgressBar.tsx` | בר התקדמות (מחליף progress bars inline) |

> **שים לב:** כל 4 קומפוננטות הן presentational בלבד – אין state, אין API.

### קומפוננטות שמתעדכנות (קיימות – שינוי JSX בלבד)

| קובץ | שינויים עיקריים | לא לגעת |
|------|----------------|----------|
| `App.tsx` | emoji → SVG icons, צבעי nav, גובה bottom bar | לוגיקת navigation, state |
| `WorkshopPage.tsx` | upload zone, plan list, badges, tabs | ZoomCanvas, כל ה-hooks |
| `PlanningPage.tsx` | wizard steps, category chips, tool strip, טבלאות | כל לוגיקת canvas, drawingState, undo/redo |
| `WorkerPage.tsx` | work type selector, tool strip, measure rows | canvas drawing, report submission, API calls |
| `DashboardPage.tsx` | stat cards, progress bars, BOQ table, tabs | כל ה-fetch calls, print logic |

---

## שלב 2 – רשימת משימות ל-Claude Code

### שינויי עיצוב: `index.css` — סיכון: 🟢 LOW

```
משימה DS-01: הוסף CSS variables חדשים
- הוסף: --orange: #e67e22, --orange-dark: #c96a10
- עדכן: --navy ל-#1e3a5f (היה #0D1F3C)
- הוסף utility classes: .badge, .badge-green, .badge-amber, .badge-red, .badge-navy
- הוסף: .btn-base (height:48px, padding, border-radius, font-weight)
- הוסף: .stat-card, .progress-bar-wrap
אל תגע: worker-main-grid, @tailwind directives, body font
```

---

### `App.tsx` — סיכון: 🟢 LOW

```
משימה APP-01: החלף emoji icons ב-SVG ב-NAV_GROUPS ובBottom Nav
- כל icon ב-NavItem: emoji string → React SVG inline
- bottom nav active color: #FF4B4B → var(--orange)
- sidebar active indicator: #10B981 (ירוק) → var(--orange)
- sidebar background: #1B3A6B → var(--navy) (#1e3a5f)
אל תגע: לוגיקת navigate(), state, מבנה NavGroup, props
```

---

### `WorkshopPage.tsx` — סיכון: 🟡 MEDIUM

```
משימה WS-01: Upload Zone
- עדכן className / style של drop zone → border dashed, icon SVG, כפתור ברור

משימה WS-02: Plan List Items
- עדכן plan item cards → border, hover, selected state
- החלף emoji status בדג'ים → StatusBadge component

משימה WS-03: Tabs
- עדכן tab buttons → גובה 44px, border-bottom: var(--orange)

משימה WS-04: Stat Cards (overview)
- עדכן stat display → border-top צבעוני, ערך גדול

אל תגע: ZoomCanvas component, כל ה-useEffect, ה-API calls, overlay logic
```

---

### `DashboardPage.tsx` — סיכון: 🟡 MEDIUM

```
משימה DASH-01: Stat Cards
- 4 כרטיסי KPI עליונים → StatCard component עם border-top צבעוני

משימה DASH-02: Progress Bars
- כל progress bar → ProgressBar component (green/amber/red)

משימה DASH-03: BOQ Table
- עדכן thead style → אפור קל, uppercase, font-size קטן
- שורות: padding נוח, hover state
- progress cell: flex, מספר %, ProgressBar

משימה DASH-04: Tabs
- עדכן tab styling → border-bottom: var(--orange)

אל תגע: כל fetch, state, print logic, snapshot generation
```

---

### `WorkerPage.tsx` — סיכון: 🟡 MEDIUM

```
משימה WRK-01: Work Type Selector
- כפתורי סוג עבודה → גובה מינ' 72px, SVG icon, border 2px, selected state עם --orange

משימה WRK-02: Tool Strip
- כפתורי כלים → שורה אופקית, גובה 48px, SVG icons, active state

משימה WRK-03: Measurement Rows
- שורות מדידה → padding נוח, ערך גדול, badge סטטוס

משימה WRK-04: History Items
- accordion items → border, header padding, badge

משימה WRK-05: Submit Button
- כפתור שליחה → height:56px, width:100%, var(--orange)

אל תגע: canvas ref, drawing logic, undo/redo, API submit, history fetch
```

---

### `PlanningPage.tsx` — סיכון: 🔴 HIGH (קובץ ענק – 2,461 שורות)

```
משימה PLAN-01: Wizard Steps
- step indicator bar → עיגולים, צבעי done/active/pending, קו חיבור

משימה PLAN-02: Category Chips
- chip grid → עיגול צבע, border 2px, selected state

משימה PLAN-03: Tool Strip
- כלי ציור → כמו WRK-02

משימה PLAN-04: Zone Table
- טבלת אזורים → כמו BOQ table

משימה PLAN-05: Navigation Buttons (wizard)
- כפתורי חזרה/המשך → 48px height, flex layout

אל תגע: canvas SVG, drawingState, undo/redo, scale calibration logic,
         autoAnalysis, zone calculations, כל ה-useEffect ו-API calls
```

---

## שלב 3 – סדר ביצוע (Rollout)

```
Phase 1  →  index.css          (תשתית עיצוב)      [30 min] 🟢
Phase 2  →  App.tsx            (shell וניווט)      [30 min] 🟢
Phase 3  →  DashboardPage.tsx  (דשבורד)            [45 min] 🟡
Phase 4  →  WorkerPage.tsx     (עובד שטח)          [45 min] 🟡
Phase 5  →  WorkshopPage.tsx   (סדנת עבודה)        [45 min] 🟡
Phase 6  →  PlanningPage.tsx   (הגדרת תכולה)       [60 min] 🔴
```

**בכל phase:** בנה → בדוק browser → אשר → המשך.
**אל תאחד phases** – כל אחד עצמאי ושלם.

---

## שלב 4 – Prompt מוכן ל-Claude Code

ראה קובץ: `CLAUDE_CODE_PROMPT.md`
