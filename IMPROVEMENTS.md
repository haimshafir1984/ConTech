# 🚀 Enhanced Metadata Extraction - System Improvements

## סיכום השינויים

שיפור משמעותי במערכת חילוץ המטא-דאטה מתוכניות אדריכליות, עם מעבר מגישה "LLM-first" לגישה היברידית "Deterministic + LLM Validation".

---

## 📊 השוואה: לפני ואחרי

### ❌ **מערכת ישנה (לפני)**

```python
# analyzer.py
meta["raw_text"] = doc[0].get_text()[:3000]  # חיתוך ל-3000 תווים!

# brain.py
prompt = f"""
Analyze construction plan text.
Input: '''{raw_text[:2000]}'''  # עוד חיתוך!
Return JSON with: plan_name, scale, plan_type.
"""
```

**בעיות:**
- ❌ אובדן מידע קריטי (חיתוך ל-3000 תווים)
- ❌ ה-LLM נאלץ לנחש ללא הקשר מלא
- ❌ הזיות (hallucinations) - המצאת מידע
- ❌ אין evidence למקור הנתונים
- ❌ JSON שבור לעיתים קרובות
- ❌ קושי עם טרמינולוגיה עברית

---

### ✅ **מערכת חדשה (אחרי)**

#### 1️⃣ **Analyzer.py - חילוץ מלא**
```python
# טקסט מלא (עד 20000 תווים)
meta["raw_text_full"] = full_text[:20000]

# בלוקים מסודרים עם bbox
meta["raw_blocks"] = [
    {"bbox": [x0,y0,x1,y1], "text": "...", "block_type": ...}
]

# טקסט מנורמל (סדר קריאה נכון)
meta["normalized_text"] = "\n".join([b["text"] for b in sorted_blocks])
```

#### 2️⃣ **Extractor.py - Pre-parser דטרמיניסטי** (חדש!)
```python
extractor = ArchitecturalTextExtractor()
candidates = extractor.extract_candidates(text)

# Regex patterns עבור:
- חדרים + שטחים: "חדר מורים ר"מ 25.5"
- קנה מידה: "קנ"מ 1:50"
- מפלסים: "פ.ת +2.80", "פ.ב ±0.00"
- גבהים: "H=2.70"
- תאריכים, גליונות, הערות
```

**כל match כולל:**
- ✅ `value` - הערך המספרי/טקסטואלי
- ✅ `evidence` - קטע הטקסט המדויק (עד 80 תווים)
- ✅ `confidence` - רמת ביטחון (0-1)

#### 3️⃣ **Brain.py - LLM Validation עם חוקים קשיחים**
```python
def safe_process_metadata(raw_text_full, normalized_text, candidates):
    prompt = f"""
    חוקים קשיחים:
    1. אסור להמציא - אם אין evidence → null
    2. תעדף candidates (דטרמיניסטיים)
    3. לכל שדה מלא → evidence חובה
    4. החזר רק JSON
    
    CANDIDATES: {candidates}
    FULL TEXT: {text}
    """
```

#### 4️⃣ **ולידציה + Auto-fix**
```python
try:
    result = json.loads(response)
except JSONDecodeError:
    # ניסיון אוטומטי לתיקון
    fixed = _auto_fix_json(client, broken_json)
```

---

## 🎯 יתרונות המערכת החדשה

### 1. **דיוק מוגבר**
- ✅ טקסט מלא (×6.6 יותר מידע)
- ✅ סדר קריאה נכון (blocks sorted)
- ✅ Evidence לכל שדה

### 2. **הפחתת הזיות (Hallucinations)**
- ✅ Pre-parser דטרמיניסטי מוצא ערכים אמיתיים
- ✅ LLM רק מאמת ומעשיר (לא ממציא)
- ✅ Confidence scoring

### 3. **אמינות**
- ✅ JSON validation + auto-fix
- ✅ Fallback mechanisms
- ✅ Error handling מקיף

### 4. **תמיכה בעברית**
- ✅ Regex מותאם לטרמינולוגיה עברית
- ✅ זיהוי קנ"מ, פ.ת, פ.ב, חדרים
- ✅ Normalization של ציטוטים עבריים

---

## 📁 קבצים שהשתנו

### 1. `analyzer.py`
```diff
- meta["raw_text"] = doc[0].get_text()[:3000]
+ meta["raw_text_full"] = full_text[:20000]
+ meta["raw_blocks"] = sorted_blocks_with_bbox
+ meta["normalized_text"] = normalized_reading_order
```

### 2. `extractor.py` ⭐ (חדש!)
```python
class ArchitecturalTextExtractor:
    """Deterministic regex-based pre-parser"""
    
    PATTERNS = {
        'room_area': re.compile(...),
        'scale': re.compile(...),
        'levels': re.compile(...),
        # + 10 patterns נוספים
    }
```

### 3. `brain.py`
```diff
- def process_plan_metadata(raw_text):
+ def safe_process_metadata(raw_text_full, normalized_text, candidates):
    # Strict prompt with evidence requirements
    # Multiple model fallback
    # Auto-fix for broken JSON
```

### 4. `utils.py`
```diff
- def safe_process_metadata(raw_text):
+ def safe_process_metadata(raw_text=None, meta=None):
    # Enhanced wrapper
    # Extracts candidates automatically
    # Passes full context to brain
```

### 5. `app.py`
```diff
- llm_data = safe_process_metadata(meta["raw_text"])
+ llm_data = safe_process_metadata(meta=meta)
```

---

## 🧪 דוגמת פלט

### Input:
```
תכנית קומה ב' - בית ספר
קנ"מ 1:50
חדר מורים ר"מ 25.5
פ.ת +2.80
```

### Output:
```json
{
  "document": {
    "plan_title": {
      "value": "תכנית קומה ב' - בית ספר",
      "confidence": 90,
      "evidence": ["תכנית קומה ב' - בית ספר"]
    },
    "scale": {
      "value": "1:50",
      "confidence": 98,
      "evidence": ["קנ\"מ 1:50"]
    }
  },
  "rooms": [
    {
      "name": {
        "value": "חדר מורים",
        "confidence": 90,
        "evidence": ["חדר מורים ר\"מ 25.5"]
      },
      "area_m2": {
        "value": 25.5,
        "confidence": 95,
        "evidence": ["חדר מורים ר\"מ 25.5"]
      }
    }
  ],
  "heights_and_levels": {
    "ceiling_levels_m": [
      {
        "value": 2.80,
        "confidence": 95,
        "evidence": ["פ.ת +2.80"]
      }
    ]
  }
}
```

---

## 🔄 Backward Compatibility

המערכת שומרת תאימות לאחור:
- ✅ `meta["raw_text"]` עדיין קיים (3000 תווים)
- ✅ אם `raw_blocks` לא זמין, המערכת עובדת עם `raw_text`
- ✅ `process_plan_metadata()` הישן עדיין עובד
- ✅ אם חילוץ candidates נכשל, ה-LLM עובד ישירות על הטקסט

---

## 🧪 Testing

### Self-test של Extractor:
```bash
python extractor.py
```

Expected output:
```
=== Self-Test Results ===
Rooms found: 3
  - חדר מורים: 25.5 m²
  - כיתה א': 60.0 m²
  - מסדרון: 12.3 m²
Scale: 1:50
Levels found: 2
  - פ.ת: 2.8m
  - פ.ב: 0.0m
Heights found: 1
```

---

## 📈 מדדי הצלחה

| מדד | לפני | אחרי | שיפור |
|-----|------|------|-------|
| **אורך טקסט** | 3,000 chars | 20,000 chars | ×6.6 |
| **Evidence tracking** | ❌ אין | ✅ יש | - |
| **Confidence scoring** | ❌ אין | ✅ יש | - |
| **JSON validation** | ⚠️ חלקי | ✅ מלא + auto-fix | - |
| **Regex pre-parsing** | ❌ אין | ✅ 12 patterns | - |
| **Hebrew support** | ⚠️ חלקי | ✅ מלא | - |

---

## 🎓 למה זה עובד טוב יותר?

### עקרון "Professional Assistant":
```
┌─────────────┐
│ PDF Input   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│ 1. Deterministic Extraction │ ← Regex patterns (אמין 100%)
│    - חדרים, שטחים, קנ"מ    │
│    - פ.ת, פ.ב, גבהים       │
└──────┬──────────────────────┘
       │ candidates + evidence
       ▼
┌─────────────────────────────┐
│ 2. LLM Validation           │ ← Claude (חכם, אבל מוגבל)
│    - אימות candidates       │
│    - השלמת הקשר            │
│    - ניקוי וארגון          │
└──────┬──────────────────────┘
       │ structured JSON
       ▼
┌─────────────────────────────┐
│ 3. Validation + Auto-fix    │
│    - JSON syntax check      │
│    - Auto-repair if needed  │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────┐
│ Final Output│
└─────────────┘
```

**הרעיון:** 
- הרגקסים מוצאים את ה"עובדות הקשות" (מספרים, תאריכים, מידות)
- ה-LLM מוסיף הקשר, לוגיקה ותובנות
- ולידציה מבטיחה פלט תקין

זה כמו "עוזר מקצועי" שעובד עם "מנהל מומחה" - כל אחד עושה את מה שהוא הכי טוב בו.

---

## 🚀 שימוש

### הרצת המערכת:
```bash
streamlit run app.py
```

### העלאת תוכנית:
1. העלה PDF בטאב "ניהול וכיול"
2. המערכת מריצה:
   - Analyzer → מוציא 20K chars + blocks
   - Extractor → מוצא candidates
   - LLM → מאמת ומעשיר
   - Validation → מבטיח JSON תקין

### תוצאה:
- ✅ מטא-דאטה מפורטת עם evidence
- ✅ Confidence scores
- ✅ JSON מובנה ותקין
- ✅ פחות הזיות, יותר דיוק

---

**Built with ❤️ for the construction industry**
