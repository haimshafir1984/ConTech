# 🚀 מדריך מהיר להעלאת הפרויקט ל-GitHub

## הבעיה שראיתי

הבעיה היא ש-Git איתחל בתיקיית המשתמש במקום בתיקיית הפרויקט, מה שגרם לו לנסות לעקוב אחרי כל הקבצים במחשב.

## ✅ פתרון מהיר

### שלב 1: הפעל את הסקריפט האוטומטי

פתח **PowerShell** בתיקיית הפרויקט והפעל:

```powershell
cd "C:\Users\moshe\OneDrive\שולחן העבודה\ConTech"
.\setup_git.ps1
```

הסקריפט יבצע:
1. ✅ אתחול Git בתיקיית הפרויקט הנכונה
2. ✅ הוספת כל הקבצים (עם `.gitignore`)
3. ✅ יצירת commit ראשון
4. ✅ הגדרת remote ב-GitHub
5. ✅ העלאה ל-GitHub

### שלב 2: או בצע ידנית

אם אתה מעדיף לעשות זאת ידנית:

```powershell
# עבור לתיקיית הפרויקט
cd "C:\Users\moshe\OneDrive\שולחן העבודה\ConTech"

# אתחל Git (רק אם עדיין לא עשית)
git init

# הוסף קבצים
git add .

# צור commit
git commit -m "Initial commit: ConTech Pro"

# שנה branch ל-main
git branch -M main

# הוסף remote (החלף <YOUR-USERNAME> ב-username שלך ב-GitHub)
git remote add origin https://github.com/<YOUR-USERNAME>/ConTech.git

# העלה ל-GitHub
git push -u origin main
```

## 🔐 לפני ההעלאה - ודא:

### 1. הגדר Git (אם עדיין לא)
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 2. צור Repository ב-GitHub
1. לך ל-[github.com/new](https://github.com/new)
2. שם: `ConTech` (או שם אחר)
3. בחר **Public** (להרצה חינמית ב-Streamlit Cloud)
4. **אל תסמן** "Initialize with README"
5. לחץ **"Create repository"**

## 📦 מה יועלה?

רק הקבצים הרלוונטיים לפרויקט:
- ✅ קבצי Python (`.py`)
- ✅ `requirements.txt`
- ✅ `README.md`
- ✅ `.gitignore`
- ✅ `.streamlit/config.toml`

**לא יועלה:**
- ❌ `.streamlit/secrets.toml` (מוגן ב-.gitignore)
- ❌ `*.db` (מסדי נתונים)
- ❌ `*.png`, `*.jpg` (תמונות)
- ❌ `__pycache__/`

## 🎯 אחרי ההעלאה ל-GitHub

### העלה ל-Streamlit Cloud:

1. לך ל-[share.streamlit.io](https://share.streamlit.io/)
2. התחבר עם GitHub
3. לחץ **"New app"**
4. בחר את ה-repository `ConTech`
5. Main file: `app.py`
6. לחץ **"Advanced settings"**
7. ב-Secrets, הוסף:
   ```
   GROQ_API_KEY = YOUR_GROQ_API_KEY_HERE
   ```
8. לחץ **"Deploy!"**

⏱️ אחרי 2-3 דקות תקבל קישור: `https://your-app-name.streamlit.app`

## ❓ בעיות נפוצות

### "remote origin already exists"
```powershell
git remote remove origin
git remote add origin https://github.com/YOUR-USERNAME/ConTech.git
```

### "Permission denied"
- ודא שיש לך הרשאות לכתוב ל-repository
- ודא שאתה מחובר ל-GitHub

### "fatal: not a git repository"
```powershell
cd "C:\Users\moshe\OneDrive\שולחן העבודה\ConTech"
git init
```

---

**טיפ:** אם תרצה לעדכן את הקוד מאוחר יותר:
```powershell
git add .
git commit -m "עדכון..."
git push
```
