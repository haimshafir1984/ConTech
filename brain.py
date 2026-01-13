import os
import json
import base64
from groq import Groq

def process_plan_metadata(raw_text):
    """
    שולח את הטקסט ל-Groq AI ומחלץ:
    1. שם התוכנית
    2. קנה מידה
    3. סוג התוכנית (סיווג קריטי למערכת)
    """
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        return {}

    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""
        Analyze the text from a construction blueprint and extract structured data.
        
        Input Text:
        '''{raw_text[:2500]}'''
        
        Task:
        1. Identify "plan_name" (e.g., "Ground Floor", "North Elevation").
        2. Identify "scale" (e.g., "1:50").
        3. Classify "plan_type" into ONE of these categories:
           - "construction": Plan for building walls (Block/Concrete).
           - "demolition": Plan for destroying walls (usually red/yellow).
           - "ceiling": Reflected ceiling plan (Do NOT count walls here).
           - "electricity": Electrical plan.
           - "plumbing": Plumbing/Sanitary plan.
           - "other": Anything else.
        
        Return JSON ONLY with keys: "plan_name", "scale", "plan_type".
        """

        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0,
            response_format={"type": "json_object"}
        )

        return json.loads(completion.choices[0].message.content)

    except Exception as e:
        print(f"Error calling Groq (Text): {e}")
        return {}

# ... (חלק עליון של הקובץ ללא שינוי) ...

def analyze_legend_image(image_bytes):
    """
    מקבל בייטס של תמונה (חיתוך של המקרא), שולח ל-Llama Vision 
    ומנסה לחלץ את שם החומר ואת הדפוס הויזואלי שלו.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key: return "חסר מפתח API"

    try:
        # המרה ל-Base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        client = Groq(api_key=api_key)
        
        # פרומפט משופר שמתמקד בדפוס הויזואלי
        prompt = """
        You are an expert in reading construction blueprints. Look at this cropped image from a legend (key).
        Your task is to identify:
        1. What material or element is depicted (e.g., "Concrete Wall", "Block Wall", "Drywall").
        2. What is the exact visual pattern defining it (e.g., "Diagonal hatching", "Solid black fill", "Double parallel lines", "Cross-hatch").

        Return ONLY a concise string in this format: "Material Name | Visual Pattern Description".
        Example: "Concrete | Diagonal hatching lines"
        If you cannot clearly identify it, return "Unknown | Unclear pattern".
        """
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-11b-vision-preview",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"שגיאה בפענוח הויזואלי: {str(e)}"