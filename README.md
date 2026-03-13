# Planex – מערכת ניהול פרויקטי בנייה

מערכת ניהול בנייה חכמה המשלבת זיהוי קירות אוטומטי, ניהול תקציב, דיווח ביצוע וייצוא BOQ.

🌐 **Live:** https://contech-frontend-react.onrender.com/

---

## Stack

| שכבה | טכנולוגיה |
|------|-----------|
| Frontend | React 18.2 + TypeScript + Vite 5 + Tailwind CSS |
| Backend | Python FastAPI (Render.com) |
| AI/Vision | OpenCV + Google Cloud Vision + Groq (Llama3) |
| DB | SQLite (מקומי) |
| RTL | Hebrew right-to-left layout |

---

## תכונות עיקריות

### 🤖 ניתוח תוכניות אוטומטי
- זיהוי קירות מ-PDF באמצעות OpenCV
- חילוץ מטא-דאטה עם LLM (Groq / Llama3)
- כיול סקלה ויחידות מידה
- זיהוי אביזרים: כיורים, אמבטיות, ריהוט

### 📋 הגדרת תכולה – אשף 5 שלבים
1. **בחירת תוכנית** – בחר מתוכניות שהועלו
2. **כיול סקייל** – גרור קיר ידוע לכיול מדויק
3. **סימון תכולה** – סווג קירות ואביזרים:
   - ניתוח אוטומטי עם אישור לפי סף ביטחון (slider 70–100%)
   - הדגשה דו-כיוונית: רשימה ↔ שרטוט
   - שיוך מהיר לפי סוג בbatch
   - סיווג מהיר של "פרטים קטנים" בבת אחת
   - תפריט right-click על אלמנטים בשרטוט
4. **כתב כמויות (BOQ)** – סיכום כמויות לפי קטגוריות
5. **גזרת עבודה** – חלוקה לאזורי עבודה

### 👷 ממשק עובד
- ציור על תוכנית לדיווח עבודה שבוצעה
- חישוב אוטומטי של מטרים/מ"ר שנבנו
- היסטוריית דיווחים

### 📊 דשבורד מנהל
- KPI cards: התקדמות, תקציב, חיזוי תאריך סיום
- BOQ Progress: תכנון מול ביצוע
- ייצוא CSV ו-PDF

---

## הפעלה מקומית

### דרישות
- Node.js 18+
- Python 3.10+

### Frontend
```bash
cd frontend
npm install
npm run dev
```
האפליקציה תעלה על `http://localhost:5173`

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### משתני סביבה
צור `frontend/.env`:
```
VITE_API_URL=http://localhost:8000
```

---

## מבנה הפרויקט

```
ConTech_react/
├── frontend/               # React app
│   ├── src/
│   │   ├── pages/          # דפי האפליקציה
│   │   ├── components/     # קומפוננטות משותפות
│   │   ├── api/            # קריאות ל-backend
│   │   └── styles/         # CSS ו-Tailwind
│   ├── package.json
│   └── vite.config.ts
├── backend/
│   └── main.py             # FastAPI server (164KB)
├── analyzer.py             # זיהוי קירות (root-level)
├── brain.py                # LLM integration (root-level)
├── database.py             # SQLite operations (root-level)
├── floor_extractor.py      # חילוץ קומות (root-level)
├── render.yaml             # Render.com deployment
└── CLAUDE.md               # הוראות לClaudeCode (קרא לפני פיתוח!)
```

> **שים לב:** `backend/*.py` הם thin wrappers בלבד. הלוגיקה האמיתית נמצאת בקבצי root.
> **תמיד ערוך קבצי root**, לא את `backend/` wrappers.

---

## Deployment

הפרויקט מועלה אוטומטית ל-Render.com עם push ל-`main`.
הגדרות ב-`render.yaml`.

> **Render cold-start:** הזיכרון מתאפס בכל deploy – משתמשים צריכים לטעון מחדש אחרי restart.

---

## אבטחה

- API Keys שמורים ב-Render Environment Variables בלבד
- אין secrets בקוד
- `.gitignore` מוגדר לכל קבצי credentials

---

**נבנה עבור תעשיית הבנייה 🏗️**
