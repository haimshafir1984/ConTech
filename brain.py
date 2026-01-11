from database import get_all_plans, get_plan_by_filename
from typing import Optional, List, Dict
import difflib
import json
import re

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

def find_similar_plans(plan_name: str, threshold: float = 0.6) -> List[Dict]:
    """
    מחפש תוכניות דומות לפי שם התוכנית
    מחזירה: רשימה של תוכניות דומות עם ציון דמיון
    """
    all_plans = get_all_plans()
    similar_plans = []
    
    for plan in all_plans:
        if plan['plan_name']:
            # חישוב דמיון בין שמות (SequenceMatcher)
            similarity = difflib.SequenceMatcher(None, plan_name.lower(), plan['plan_name'].lower()).ratio()
            
            if similarity >= threshold:
                similar_plans.append({
                    'plan': plan,
                    'similarity': similarity
                })
    
    # מיון לפי דמיון (הכי דומה ראשון)
    similar_plans.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similar_plans

def suggest_scale(filename: str, extracted_plan_name: Optional[str] = None) -> Optional[float]:
    """
    מציע סקלה מומלצת על בסיס למידה מתוכניות קודמות
    מחזירה: סקלה מומלצת (pixels_per_meter) או None אם אין המלצה
    """
    # קודם, בודקים אם יש לנו כבר תוכנית עם השם הזה
    existing_plan = get_plan_by_filename(filename)
    if existing_plan and existing_plan.get('confirmed_scale'):
        # אם יש לנו כיול מאושר, נחזיר אותו
        return existing_plan['confirmed_scale']
    
    # אם יש שם תוכנית, נחפש תוכניות דומות
    if extracted_plan_name:
        similar_plans = find_similar_plans(extracted_plan_name, threshold=0.7)
        
        if similar_plans:
            # ניקח את התוכנית הכי דומה שיש לה כיול מאושר
            for item in similar_plans:
                plan = item['plan']
                if plan.get('confirmed_scale') and plan['confirmed_scale'] > 0:
                    print(f"🧠 מוח: מצאתי תוכנית דומה '{plan['plan_name']}' (דמיון: {item['similarity']:.2%})")
                    print(f"   המלצה: סקלה של {plan['confirmed_scale']:.1f} פיקסלים למטר")
                    return plan['confirmed_scale']
    
    # אם אין תוכניות דומות, נבדוק ממוצע של כל התוכניות המאושרות
    all_plans = get_all_plans()
    confirmed_scales = [p['confirmed_scale'] for p in all_plans 
                       if p.get('confirmed_scale') and p['confirmed_scale'] > 0]
    
    if confirmed_scales:
        avg_scale = sum(confirmed_scales) / len(confirmed_scales)
        print(f"🧠 מוח: אין תוכנית דומה, משתמש בממוצע של {len(confirmed_scales)} תוכניות: {avg_scale:.1f}")
        return avg_scale
    
    # אין המלצה
    return None

def learn_from_confirmation(filename: str, plan_name: str, confirmed_scale: float, 
                           raw_pixel_count: int, extracted_metadata: Dict) -> int:
    """
    'לומד' מהאישור של המנהל - שומר את הנתונים למען שימוש עתידי
    מחזירה: plan_id
    """
    from database import save_plan
    import json
    
    metadata_json = json.dumps(extracted_metadata, ensure_ascii=False)
    extracted_scale = extracted_metadata.get('scale')
    
    plan_id = save_plan(
        filename=filename,
        plan_name=plan_name,
        extracted_scale=extracted_scale,
        confirmed_scale=confirmed_scale,
        raw_pixel_count=raw_pixel_count,
        metadata_json=metadata_json
    )
    
    print(f"🧠 מוח: למדתי מהאישור שלך! תוכנית '{plan_name}' נשמרה עם סקלה {confirmed_scale:.1f}")
    return plan_id

def process_plan_metadata(raw_text: str, api_key: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    משתמש ב-LLM (Groq - llama3-8b-8192) כדי לחלץ מטא-דאטה מתוכנית בנייה מהטקסט הגולמי של OCR
    
    מחזיר:
    - plan_name: שם הגיוני לתוכנית (קומה/אזור)
    - scale: סקלה מזוהה (לדוגמה: "1:50", "1:100")
    - units: יחידות מידה (m/cm)
    """
    if not GROQ_AVAILABLE:
        # אם אין groq, ננסה חילוץ בסיסי ב-regex
        return _extract_metadata_basic(raw_text)
    
    if not api_key:
        # ניסיון לקבל מה-streamlit secrets
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
    
    if not api_key:
        # אם אין API key, נשתמש בחילוץ בסיסי
        return _extract_metadata_basic(raw_text)
    
    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""אתה מומחה לבנייה ואדריכלות. נתון לך טקסט שמוצא מתוכנית בנייה (PDF) באמצעות OCR.
        
טקסט הגולמי:
{raw_text[:2000]}  # מוגבל ל-2000 תווים

תפקידך הוא לחלץ את המידע הבא:
1. שם התוכנית - שם הגיוני לקומה/אזור (לדוגמה: "קומה 2", "מפלס חניון", "אגף צפוני")
2. סקלה - היחס בשרטוט (לדוגמה: "1:50", "1:100", "1:200")
3. יחידות מידה - האם זה מטרים (m) או סנטימטרים (cm)

החזר תשובה ב-JSON בלבד בפורמט הבא:
{{
    "plan_name": "שם התוכנית או null",
    "scale": "1:50 או null",
    "units": "m או cm או null"
}}

אם אינך יכול לזהות משהו, החזר null עבור השדה הרלוונטי."""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "אתה מומחה לחילוץ מידע מתוכניות בנייה. תמיד החזר JSON בלבד."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # ניקוי של markdown code blocks אם יש
        if result_text.startswith("```"):
            result_text = re.sub(r'^```json\s*', '', result_text)
            result_text = re.sub(r'^```\s*', '', result_text)
            result_text = re.sub(r'```\s*$', '', result_text)
        
        result = json.loads(result_text)
        
        # וידוא שהפורמט נכון
        return {
            "plan_name": result.get("plan_name"),
            "scale": result.get("scale"),
            "units": result.get("units")
        }
        
    except Exception as e:
        print(f"⚠️ שגיאה ב-LLM metadata extraction: {e}")
        # fallback לחילוץ בסיסי
        return _extract_metadata_basic(raw_text)

def _extract_metadata_basic(raw_text: str) -> Dict[str, Optional[str]]:
    """
    חילוץ בסיסי של מטא-דאטה בלי LLM (fallback)
    משתמש ב-regex פשוטים
    """
    result = {
        "plan_name": None,
        "scale": None,
        "units": "m"  # default
    }
    
    # חיפוש סקלה
    scale_pattern = r'1\s*[:/]\s*(\d+)'
    scale_match = re.search(scale_pattern, raw_text, re.IGNORECASE)
    if scale_match:
        result["scale"] = f"1:{scale_match.group(1)}"
    
    # חיפוש יחידות
    if re.search(r'\b(cm|centimeter|סמ)\b', raw_text, re.IGNORECASE):
        result["units"] = "cm"
    
    # חיפוש שם תוכנית (ניסיון למצוא מילים כמו "קומה", "מפלס", וכו')
    plan_patterns = [
        r'(קומה|מפלס|אגף|אזור)\s*([0-9א-ת\s]+)',
        r'(Floor|Level|Area)\s*([0-9A-Za-z\s]+)',
    ]
    
    for pattern in plan_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            result["plan_name"] = match.group(0).strip()
            break
    
    return result
