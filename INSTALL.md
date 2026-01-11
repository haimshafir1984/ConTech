# הוראות התקנה בעברית

## שלב 1: התקנת החבילות

פתח PowerShell או CMD והריץ את אחת מהפקודות הבאות:

### אפשרות 1 (מומלץ):
```
py -m pip install -r requirements.txt
```

### אפשרות 2:
```
python -m pip install -r requirements.txt
```

**הערה:** אם אתה כבר בתיקיית הפרויקט, פשוט הרץ את הפקודה.
אם לא, עבור לתיקייה קודם:
```
cd "c:\Users\moshe\OneDrive\שולחן העבודה\ConTech"
py -m pip install -r requirements.txt
```

## שלב 2: הכנת קובץ PDF

שים קובץ PDF של תכנית בניין בשם `plan.pdf` בתיקיית הפרויקט.

## שלב 3: הפעלת הסקריפט

הריץ:
```
py main.py
```

או:
```
python main.py
```

הסקריפט ייצור קובץ `boq.csv` עם כמויות הקירות.

## פתרון בעיות

אם `pip` לא מזוהה:
- השתמש ב-`py -m pip` במקום `pip`
- או `python -m pip` במקום `pip`

אם יש בעיה עם נתיב בעברית:
- עבור לתיקייה ידנית ב-Explorer
- לחץ על סרגל הכתובת והעתק את הנתיב
- ב-PowerShell, עבור לתיקייה: `cd "[הדבק את הנתיב כאן]"`

