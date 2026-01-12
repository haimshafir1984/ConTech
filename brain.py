import os
import json
from groq import Groq

def process_plan_metadata(raw_text):
    """
    שולח את הטקסט ל-Groq AI ומחלץ JSON עם שם התוכנית וקנה המידה.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    
    # אם אין מפתח מוגדר, מחזירים מילון ריק (כדי לא להקריס את האתר)
    if not api_key:
        print("Warning: GROQ_API_KEY not found in environment variables.")
        return {}

    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""
        Analyze the following text from a construction plan and extract:
        1. "plan_name": The specific name of the drawing (e.g., "Ground Floor", "North Elevation").
        2. "scale": The scale ratio (e.g., "1:50", "1:100").
        
        Text:
        '''{raw_text[:2000]}'''
        
        Return ONLY a raw JSON object with keys "plan_name" and "scale". 
        If not found, set value to null. Do not write markdown or explanations.
        """

        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-70b-8192",
            temperature=0,
            response_format={"type": "json_object"}
        )

        response_content = completion.choices[0].message.content
        return json.loads(response_content)

    except Exception as e:
        print(f"Error calling Groq: {e}")
        return {}