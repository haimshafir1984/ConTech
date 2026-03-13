# Prompt ל-Claude Code – הטמעת עיצוב ConTech

---

## הדבק את הפרומפט הבא ל-Claude Code:

---

```
אתה עובד על פרויקט React + FastAPI בשלב פיילוט.
כלל מוחלט: אין שינוי לוגיקה, API, state, חישובים, hooks, או קריאות backend.
כל שינוי הוא ויזואלי בלבד (CSS / JSX structure / classNames / inline styles).

לפני כל שינוי כתוב בעברית: "משנה [שם קובץ] – [מה משתנה]"
אחרי כל שינוי כתוב בעברית: "הושלם – [מה השתנה בפועל]"
הצג רק diff, לא קבצים מלאים.
לא לבצע ריפקטור.

---

## טוקני עיצוב מאושרים

CSS Variables (להוסיף ל-frontend/src/styles/index.css):
  --navy:        #1e3a5f
  --navy-dark:   #152d4a
  --navy-light:  #2a4d7a
  --orange:      #e67e22
  --orange-dark: #c96a10
  --green:       #15803d
  --green-light: #dcfce7
  --amber:       #b45309
  --amber-light: #fef3c7
  --red:         #dc2626
  --red-light:   #fee2e2
  --blue-light:  #dbeafe
  --border:      #e2e8f0
  --border-dark: #cbd5e1
  --text-1:      #1e293b
  --text-2:      #475569
  --text-3:      #94a3b8

כפתורים: height מינ' 48px, border-radius: 10px, font-weight: 600
כפתור ראשי: background var(--navy), color #fff
כפתור פעולה: background var(--orange), color #fff
כפתור שלח (עובד): height 56px, width 100%

בדג'ים (ללא emoji):
  .badge        { padding: 3px 10px; border-radius: 20px; font-size: .78rem; font-weight: 600 }
  .badge-green  { background: var(--green-light); color: var(--green) }
  .badge-amber  { background: var(--amber-light); color: var(--amber) }
  .badge-red    { background: var(--red-light);   color: var(--red)   }
  .badge-navy   { background: #e0eaf5;            color: var(--navy)  }
  .badge-blue   { background: var(--blue-light);  color: #1d4ed8      }

---

## PHASE 1 – index.css (סיכון LOW)

בצע רק את המשימות הבאות ב-frontend/src/styles/index.css:

1. עדכן --navy ל-#1e3a5f
2. הוסף --orange, --orange-dark, --green-light, --amber-light, --red-light, --navy-light, --navy-dark, --text-1, --text-2, --text-3, --border-dark
3. הוסף class .badge ועיצובי variants (green/amber/red/navy/blue) לסוף הקובץ
4. הוסף utility: .btn-base (display:inline-flex; align-items:center; height:48px; padding:0 20px; border-radius:10px; font-weight:600; cursor:pointer; border:none)
5. הוסף .stat-card-accent-border-top variants:
   .accent-top-orange { border-top: 4px solid var(--orange) }
   .accent-top-green  { border-top: 4px solid var(--green)  }
   .accent-top-blue   { border-top: 4px solid #1d4ed8       }
   .accent-top-amber  { border-top: 4px solid var(--amber)  }

אל תגע ב: @tailwind directives, worker-main-grid, body styles, קיימות אחרות.

---

## PHASE 2 – App.tsx (סיכון LOW)

בצע רק את המשימות הבאות ב-frontend/src/App.tsx:

1. החלף את ה-icon strings ב-NAV_GROUPS ובBOTTOM_NAV מ-emoji ל-SVG inline.
   השתמש ב-SVG פשוטים (stroke, no fill), גודל 18px לsidebar, 22px לbottom nav.
   אלו הicons הנדרשים:
   - workshop:     blueprint/monitor icon
   - planning:     document/layers icon
   - layers:       stack icon
   - corrections:  edit/pencil icon
   - areaAnalysis: ruler/measure icon
   - dashboard:    grid/chart icon
   - drawingData:  file-text icon
   - invoices:     currency/dollar icon
   - worker:       hard-hat/user icon

2. sidebar active indicator: שנה צבע #10B981 → var(--orange)
3. bottom nav active color: שנה #FF4B4B → var(--orange)
4. sidebar background: שנה #1B3A6B → var(--navy) (#1e3a5f)

אל תגע ב: לוגיקת navigate(), state, props, מבנה NavGroup, ToastProvider, ConfirmProvider.

---

## PHASE 3 – DashboardPage.tsx (סיכון MEDIUM)

קרא את הקובץ. בצע רק:

1. מדדי KPI (4 כרטיסים עליונים):
   - הוסף border-top צבעוני לפי סדר: orange, green, blue, amber
   - הגדל font-size של הערך ל-1.6rem, font-weight: 800
   - padding: 14px

2. Progress bars (כל בר בBOQ):
   - רקע track: var(--border), border-radius: 99px, height: 10px
   - fill: var(--green) כשמעל 60%, var(--amber) 30-60%, var(--red) מתחת 30%

3. Tab buttons (overview/BOQ/workers/snapshot):
   - active tab: border-bottom: 3px solid var(--orange), color: var(--navy)
   - height: 44px

4. BOQ table header:
   - background: #f8fafc, font-size: .78rem, text-transform: uppercase, letter-spacing: .5px
   - td padding: 12px

אל תגע ב: כל fetch calls, useMemo, print logic, snapshot URLs, formatters.

---

## PHASE 4 – WorkerPage.tsx (סיכון MEDIUM)

קרא את הקובץ. בצע רק:

1. Work type selector buttons (ריצוף / קירות / תקרה / טיח):
   - height מינ' 72px, border: 2px solid var(--border-dark), border-radius: 10px
   - selected state: border-color: var(--orange), background: #fff7ed
   - הסר emoji, הוסף SVG icon (18px) ותווית ברורה

2. Tool strip (ציור / מלבן / קיר / מחק / בטל):
   - גובה 44px, display: flex, gap: 8px, overflow-x: auto
   - active tool: background: var(--navy), color: #fff

3. Measurement rows:
   - padding: 12px 0, border-bottom: 1px solid var(--border)
   - ערך כמות: font-size: 1.1rem, font-weight: 800, color: var(--navy)
   - badge סטטוס: .badge-green / .badge-amber

4. כפתור "שלח דיווח":
   - height: 56px, width: 100%, background: var(--orange), border-radius: 10px
   - font-size: 1.05rem, font-weight: 700

אל תגע ב: canvas ref, drawing functions, undo/redo stack, report API calls, history fetch.

---

## PHASE 5 – WorkshopPage.tsx (סיכון MEDIUM)

קרא את הקובץ. בצע רק:

1. Upload zone (drop area):
   - border: 2px dashed var(--border-dark), border-radius: 14px, padding: 36px
   - hover: border-color: var(--navy), background: #e0eaf5
   - הסר emoji, הוסף SVG upload icon

2. Plan list items:
   - border: 1.5px solid var(--border), border-radius: 10px, padding: 14px
   - selected: border-color: var(--navy), background: #f0f5fb
   - status emoji → .badge component

3. Tab buttons:
   - active: border-bottom: 3px solid var(--orange), color: var(--navy), height: 44px

4. Stat cards (overview):
   - border-top צבעוני (accent-top-orange, accent-top-green וכו')
   - ערך: font-size: 1.6rem, font-weight: 800

אל תגע ב: ZoomCanvas component, onWheel, onMouseDown, onTouchStart, כל ה-useEffect, ה-API calls.

---

## PHASE 6 – PlanningPage.tsx (סיכון HIGH – קובץ ענק)

קרא את הקובץ לפי חלקים. בצע רק:

1. Wizard step indicator:
   - עיגולים 28px, done: background var(--green), active: background var(--navy)
   - קו חיבור בין שלבים: 2px solid var(--border), done: var(--green)

2. Category chips (בחירת סוג עבודה):
   - border: 2px solid var(--border), border-radius: 10px, padding: 12px
   - selected: border-color: var(--orange), background: #fff7ed

3. Tool strip:
   - זהה ל-WorkerPage (PHASE 4, משימה 2)

4. Zone table:
   - זהה לBOQ table (PHASE 3, משימה 4)

5. כפתורי ניווט (חזרה / המשך):
   - height: 48px, כפתור ראשי var(--navy), כפתור חזרה ghost

אל תגע ב: canvas SVG, drawingState, undo/redo, scale calibration, autoAnalysis,
           zone calculations, area measurements, כל useEffect, כל API calls.
           לא לשנות את מבנה ה-JSX של הcanvas.

---

## כללים לכל הphases

- הצג diff בלבד
- עברית לפני ואחרי כל שינוי
- לא לאחד phases
- לא לשנות imports (אלא אם מוסיפים קומפוננטה חדשה)
- אם ספק – אל תשנה
- אם קומפוננטה משתמשת בlogic – לא לגעת בה
```

---

## הוראות שימוש

1. פתח Claude Code בתיקיית הפרויקט
2. הדבק את הפרומפט שבין ``` ``` לעיל
3. בצע phase אחד בכל פעם
4. בדוק ב-browser אחרי כל phase
5. אם משהו נשבר – `git restore <file>` לחזרה מיידית

## פקודת rollback מהירה

```bash
# לביטול כל שינוי בקובץ ספציפי:
git restore frontend/src/pages/DashboardPage.tsx

# לביטול כל שינויי עיצוב:
git restore frontend/src/styles/index.css frontend/src/App.tsx
```
