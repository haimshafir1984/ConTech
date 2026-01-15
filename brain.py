import os
import base64
import json
try:
    import anthropic
except ImportError:
    anthropic = None
import streamlit as st

def get_anthropic_client():
    """יוצר חיבור ל-Claude בצורה מאובטחת עם Fallback"""
    if anthropic is None:
        return None, "ספריית anthropic חסרה. הרץ: pip install anthropic"
    
    # ניסיון ראשון: streamlit secrets
    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    
    # ניסיון שני: משתני סביבה (עבור Render)
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        return None, "חסר מפתח ANTHROPIC_API_KEY. הוסף אותו ב-.streamlit/secrets.toml או במשתני סביבה בשרת"
        
    return anthropic.Anthropic(api_key=api_key), None

def process_plan_metadata(raw_text):
    """
    שולח את הטקסט ל-Claude ומחלץ JSON מבנה
    """
    client, error = get_anthropic_client()
    if error:
        print(error)
        return {}

    try:
        prompt = f"""
        Analyze the text extracted from a construction blueprint PDF.
        
        Input Text:
        '''{raw_text[:3000]}'''
        
        Task:
        1. Identify "plan_name" (e.g., "Ground Floor", "North Elevation", "תוכנית קומת קרקע").
        2. Identify "scale" (e.g., "1:50", "1:100").
        3. Classify "plan_type" into ONE of these categories:
           - "construction": Plan for building walls (Block/Concrete/Masonry).
           - "demolition": Plan for destroying walls (usually red/yellow colors).
           - "ceiling": Reflected ceiling plan (Do NOT count walls here).
           - "electricity": Electrical plan.
           - "plumbing": Plumbing/Sanitary plan.
           - "other": Anything else.
        
        Return ONLY a JSON object. Do not add markdown formatting or explanation.
        Example: {{"plan_name": "Living Room", "scale": "1:50", "plan_type": "construction"}}
        """

        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # ניקוי ופירסור JSON
        response_text = message.content[0].text
        # מחיקת תגיות אם יש
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "{" in response_text: 
            response_text = "{" + response_text.split("{", 1)[1].rsplit("}", 1)[0] + "}"
            
        return json.loads(response_text)

    except Exception as e:
        print(f"Brain Error (Metadata): {e}")
        return {}

def analyze_legend_image(image_bytes):
    """
    שולח תמונה של מקרא ל-Claude Vision
    """
    client, error = get_anthropic_client()
    if error: return error

    try:
        # המרה ל-Base64
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        prompt = """
        אתה מומחה לקריאת תוכניות בנייה (אדריכלות/קונסטרוקציה).
        מצורפת תמונה שנחתכה מתוך מקרא (Legend) של שרטוט.
        
        המשימה שלך:
        זהה מה מסומן בתמונה. תאר את סוג הקו/הצבע ואת המשמעות שלו בעברית.
        אם יש מספר חומרים, רשום אותם ברשימה.
        
        דוגמה לתשובה טובה:
        "קו כחול עבה = קיר בטון חדש"
        "קו כתום = קיר בלוקים"
        "קו מקווקוו = הריסה"
        
        תחזיר תשובה קצרה ותמציתית בעברית בלבד.
        """

        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=500,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": encoded_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        return message.content[0].text

    except Exception as e:
        return f"שגיאה בפענוח הויזואלי: {str(e)}"