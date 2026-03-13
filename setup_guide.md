# 🚀 מדריך העלאה ל-GitHub ו-Streamlit Cloud

## שלב 1: יצירת Repository ב-GitHub

1. לך ל-[GitHub.com](https://github.com) והתחבר
2. לחץ על **"New repository"** (או `+` > New repository)
3. מלא פרטים:
   - **Repository name:** `ConTech` (או שם אחר)
   - **Description:** "מערכת ניהול בנייה מקצועית"
   - בחר **Public** (להרצה חינמית ב-Streamlit Cloud)
   - אל תסמן "Initialize with README" (כי כבר יש לנו README)
4. לחץ **"Create repository"**

## שלב 2: העלאת הקוד ל-GitHub

### שיטה 1: דרך Command Line (מומלץ)

פתח **PowerShell** או **Command Prompt** בתיקיית הפרויקט:

```powershell
# עבור לתיקיית הפרויקט
cd "C:\Users\moshe\OneDrive\שולחן העבודה\ConTech"

# אתחל Git repository
git init

# הוסף את כל הקבצים
git add .

# צור commit ראשון
git commit -m "Initial commit: ConTech Pro - מערכת ניהול בנייה"

# הוסף את ה-remote (החלף <your-username> ב-GitHub username שלך)
git remote add origin https://github.com/<your-username>/ConTech.git

# העלה את הקוד
git branch -M main
git push -u origin main
```

### שיטה 2: דרך GitHub Desktop

1. הורד והתקן [GitHub Desktop](https://desktop.github.com/)
2. פתח את GitHub Desktop
3. לחץ **File** > **Add Local Repository**
4. בחר את התיקייה `ConTech`
5. לחץ **Publish repository**
6. בחר את ה-repository שיצרת בשלב 1

## שלב 3: הגדרת Streamlit Cloud (להרצה חיה)

1. לך ל-[share.streamlit.io](https://share.streamlit.io/)
2. התחבר עם חשבון GitHub שלך
3. לחץ **"New app"**
4. מלא פרטים:
   - **Repository:** בחר את ה-repository `ConTech`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. לחץ **"Advanced settings"** והוסף את ה-secrets:
   ```
   GROQ_API_KEY = your-actual-api-key-here
   ```
6. לחץ **"Deploy!"**

⏱️ זה יכול לקחת 2-3 דקות...

## שלב 4: קבלת קישור להרצה

לאחר ההעלאה, תקבל קישור כמו:
```
https://your-app-name.streamlit.app
```

זה הקישור שאתה יכול לשלוח לבדיקה!

## 🔐 הגדרת Secrets ב-Streamlit Cloud

1. בלוח הבקרה של Streamlit Cloud, לחץ על ה-app שלך
2. לחץ על **"⚙️ Settings"** (שלוש נקודות למעלה)
3. לחץ על **"Secrets"**
4. הוסף:
   ```toml
   GROQ_API_KEY = "your-actual-groq-api-key"
   ```
5. לחץ **"Save"**
6. ה-app יתרענן אוטומטית עם ה-secrets החדשים

## ✅ בדיקות לאחר ההעלאה

1. בדוק שהאפליקציה עובדת בקישור
2. נסה להעלות PDF תוכנית
3. בדוק שהזיהוי עובד
4. נסה ליצור דיווח

## 📝 הערות חשובות

- ✅ הקובץ `.streamlit/secrets.toml` **לא** מועלה ל-Git (מוגן ב-.gitignore)
- ✅ הקבצים `*.db` ו-`*.csv` **לא** מועלים (מוגנים ב-.gitignore)
- ✅ תמונות debug **לא** מועלות (מוגנות ב-.gitignore)
- 🔐 הוסף את ה-API keys **רק** ב-Streamlit Cloud Secrets (לא ב-Git!)

## 🆘 פתרון בעיות

### שגיאת "Module not found"
- ודא ש-`requirements.txt` מעודכן
- Streamlit Cloud יתקין את החבילות אוטומטית

### שגיאת API Key
- ודא שה-`GROQ_API_KEY` מוגדר ב-Streamlit Cloud Secrets
- ודא שהקוד משתמש ב-`st.secrets.get("GROQ_API_KEY")`

### שגיאת Database
- Streamlit Cloud לא שומר את מסד הנתונים בין הרצות
- זה בסדר לבדיקות, אבל לא לפרויקט פרודקשן
- לפרודקשן, כדאי להשתמש ב-PostgreSQL או MySQL חיצוני

## 🎉 סיימת!

עכשיו יש לך:
- ✅ Repository ב-GitHub
- ✅ אפליקציה רצה ב-Streamlit Cloud
- ✅ קישור שמיש לשיתוף

**הקישור שלך:** `https://your-app-name.streamlit.app`

---

**טיפ:** תוכל לעדכן את הקוד, לדחוף ל-GitHub, וה-app יתעדכן אוטומטית!
