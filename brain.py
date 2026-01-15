import os
import base64
import json
try:
    import anthropic
except ImportError:
    anthropic = None
import streamlit as st

def get_anthropic_client():
    """יוצר חיבור ל-Claude בצורה מאובטחת עם עדיפות למשתני סביבה"""
    if anthropic is None:
        return None, "ספריית anthropic חסרה."
    
    # 1. ניסיון למשוך ממשתני סביבה (Render) - הכי חשוב
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # 2. אם אין, ניסיון למשוך מ-secrets (מקומי)
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass # התעלמות אם הקובץ לא קיים
    
    if not api_key:
        return None, "חסר מפתח API"
        
    return anthropic.Anthropic(api_key=api_key), None

def process_plan_metadata(raw_text):
    client, error = get_anthropic_client()
    if error: return {}

    try:
        prompt = f"""
        Analyze construction plan text.
        Input: '''{raw_text[:2000]}'''
        Return JSON with: plan_name, scale (e.g. 1:50), plan_type (construction/demolition/other).
        """
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.content[0].text
        if "{" in response_text: 
            response_text = "{" + response_text.split("{", 1)[1].rsplit("}", 1)[0] + "}"
        return json.loads(response_text)
    except:
        return {}

def analyze_legend_image(image_bytes):
    client, error = get_anthropic_client()
    if error: return error

    try:
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded_image}},
                    {"type": "text", "text": "Identify the construction materials/lines in this legend image. Hebrew only."}
                ]
            }]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"