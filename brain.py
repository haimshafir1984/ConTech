"""
קובץ brain.py - עיבוד מטא-דאטה מתוכניות בניה
זהו fallback פשוט במקרה שהקובץ המקורי לא קיים
"""

import re
from typing import Dict, Optional

def process_plan_metadata(raw_text: str) -> Dict[str, Optional[str]]:
    """
    מעבד טקסט גולמי מתוכנית בניה ומנסה לחלץ מידע רלוונטי
    
    Args:
        raw_text: הטקסט שנוצר מה-PDF
        
    Returns:
        Dict עם plan_name, scale ומידע נוסף
    """
    metadata = {
        "plan_name": None,
        "scale": None,
        "project_number": None,
        "date": None
    }
    
    # חיפוש שם תוכנית
    plan_patterns = [
        r"(?:תוכנית|שם\s*שרטוט|Project|שם\s*פרויקט)[\s:]+([^\n\r]+)",
        r"(?:מס['\u2019]\s*תוכנית|תיק)[\s:]+([^\n\r]+)"
    ]
    
    for pattern in plan_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            metadata["plan_name"] = match.group(1).strip()
            break
    
    # חיפוש קנה מידה
    scale_patterns = [
        r"(?:קנה\s*מידה|SCALE|מ['\u2019]ל)[\s:]*(\d+)[\s:]*[:/][\s]*(\d+)",
        r"(\d+)[\s]*:[\s]*(\d+)",
        r"1[\s]*:[\s]*(\d+)"
    ]
    
    for pattern in scale_patterns:
        match = re.search(pattern, raw_text)
        if match:
            if len(match.groups()) == 2:
                metadata["scale"] = f"{match.group(1)}:{match.group(2)}"
            else:
                metadata["scale"] = f"1:{match.group(1)}"
            break
    
    # חיפוש מספר פרויקט
    project_patterns = [
        r"(?:מס['\u2019]\s*פרויקט|Project\s*No|פרויקט)[\s:]+([0-9A-Z/-]+)",
    ]
    
    for pattern in project_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            metadata["project_number"] = match.group(1).strip()
            break
    
    # חיפוש תאריך
    date_patterns = [
        r"(\d{1,2}[./]\d{1,2}[./]\d{2,4})",
        r"(\d{4}-\d{2}-\d{2})"
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, raw_text)
        if match:
            metadata["date"] = match.group(1)
            break
    
    return metadata


def learn_from_confirmation(user_input: str, extracted: str, confirmed: str):
    """
    פונקציה שאמורה ללמוד מאישורי משתמש
    כרגע זו פשוט placeholder - ניתן להרחיב בעתיד
    
    Args:
        user_input: מה שהמשתמש הזין
        extracted: מה שהמערכת חילצה אוטומטית
        confirmed: מה שהמשתמש אישר בסוף
    """
    # כאן ניתן להוסיף לוגיקה של למידה/שיפור
    # לדוגמה: שמירה לקובץ, עדכון מודל, וכו'
    pass
