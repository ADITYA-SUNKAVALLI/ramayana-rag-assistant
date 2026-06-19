import json
import re
from pathlib import Path

INPUT_FILE = Path("cleaned_text/all_chapters.jsonl")
OUTPUT_FILE = Path("cleaned_text/all_chapters_fixed.jsonl")

REPLACEMENTS = {
    "కాంండము": "కాండము",
    "సుంందరా": "సుందరా",
    "పక్ం": "పక్షం",
    "సైన్ం": "సైన్యం",
    "రండు": "రెండు",
    "పూర్వంం": "పూర్వం",
    "వంటనే": "వెంటనే",
    "జూసి": "చూసి",
    "జూశాడు": "చూశాడు",
    "జూశారు": "చూశారు",
    "జూస్తూ": "చూస్తూ",
    "ప్రవేశింశ": "ప్రవేశించ",
    "నిర్మంచ": "నిర్మించ",
    "సంతోషంచ": "సంతోషించ",
    "దుఃఖంచ": "దుఃఖించ",
    "విజృభంచ": "విజృంభించ",
    "గుర్తంచ": "గుర్తించ",
    "అధ్య నం": "అధ్యయనం",
    "రాజ్ం": "రాజ్యం",
    "వస్తండగా": "వస్తుండగా",
    "చేస్తండగా": "చేస్తుండగా",
    "చూస్తండగా": "చూస్తుండగా",
    "ఉన్నా డు": "ఉన్నాడు",
    "అనుకొంటున్నా డు": "అనుకొంటున్నాడు",
    "అర్ధింధించాడు": "అర్థించాడు",
    "అప్పగింగించి": "అప్పగించి",
    "పారంగతుల తోనూ": "పారంగతులతోనూ",
    "చేస ్త": "చేస్త",
    "భూనిన": "ప్రయోగించిన",
    "సాధ్ం": "సాధ్యం",
    "అసాధ్ం": "అసాధ్యం",
    "ముఖ్ం": "ముఖ్యం",
    "వృక్ం": "వృక్షం",
    "అశ్వంం": "అశ్వం",
    "తమసా నది": "తమసా నది"
}

def clean_text(text):
    if not isinstance(text, str):
        return text

    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")

    for wrong, correct in REPLACEMENTS.items():
        text = text.replace(wrong, correct)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def clean_object(obj):
    if isinstance(obj, dict):
        return {k: clean_object(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean_object(x) for x in obj]

    if isinstance(obj, str):
        return clean_text(obj)

    return obj

def main():
    total = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            record = clean_object(record)

            outfile.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            total += 1

    print(f"Processed {total} chapters")
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()