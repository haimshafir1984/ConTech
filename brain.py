import os
import json
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
        
        # בניית פרומפט חכם יותר שמבקש גם סיווג
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
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192", # מודל טקסט חזק
            temperature=0,
            response_format={"type": "json_object"}
        )

        response_content = completion.choices[0].message.content
        return json.loads(response_content)

    except Exception as e:
        print(f"Error calling Groq: {e}")
        return {}