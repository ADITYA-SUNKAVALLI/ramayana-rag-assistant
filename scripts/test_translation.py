from dotenv import load_dotenv
from openai import OpenAI
import os
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

sample = {
    "question": "శ్రీమన్నారాయణుడు కల్పాదియందు సృష్టి ప్రారంభంచ నంచి, ముందుగా తన సంకల్ప మాత్రం చేత ఎవరిని సృష్టంచాడు ?",
    "answer": "చతుర్ముఖ బ్రహ్మ"
}

prompt = f"""
You are an expert Telugu to English translator.

Translate the JSON below.

Rules:
1. Keep exact meaning.
2. Preserve Ramayana names.
3. Preserve religious terminology.
4. Return ONLY valid JSON.
5. Do not explain.

JSON:
{json.dumps(sample, ensure_ascii=False)}
"""

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

print(response.choices[0].message.content)