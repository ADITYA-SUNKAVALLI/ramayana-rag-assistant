#!/usr/bin/env python3
"""
scripts/03_translate.py

High-fidelity Telugu -> English translation of cleaned_text/all_chapters_fixed.jsonl
(99 Ramayana Q&A chapter records), using the DeepSeek API ONLY.

v2: hardened against terminology drift, theological drift, epic-register loss,
and inconsistent transliteration. Adds:
  - FORBIDDEN_TRANSLATIONS: explicit list of secularized substitutes that must
    never replace protected Sanskrit/sacred terms (terminology-drift guard)
  - MODERN_PHRASE_PATTERNS: detector for modern/casual phrasing that breaks
    epic register
  - Two-pass architecture (--two-pass): translate, then a second DeepSeek call
    self-reviews the draft against the rules and returns a corrected version
  - quality_score(): automated 0-100 fidelity rubric per chapter

Run:
    python scripts/03_translate.py                  # translate all remaining
    python scripts/03_translate.py --limit 5         # translate next 5 only
    python scripts/03_translate.py --chapter 12      # (re)translate just one
    python scripts/03_translate.py --retry-failed    # retry only failed ones
    python scripts/03_translate.py --two-pass        # add self-review/correction pass
    python scripts/03_translate.py --two-pass --limit 5

Inputs:
    cleaned_text/all_chapters_fixed.jsonl
    .env  (must contain DEEPSEEK_API_KEY)

Outputs:
    translated/all_chapters_en.jsonl   (includes per-chapter "_quality_score")
    translated/translation_log.txt
    logs/translation_failures.md
    logs/terminology_flags.md          (forbidden-term / modern-phrase hits needing human review)
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_PATH = BASE_DIR / "cleaned_text" / "all_chapters_fixed.jsonl"
OUTPUT_PATH = BASE_DIR / "translated" / "all_chapters_en.jsonl"
LOG_PATH = BASE_DIR / "translated" / "translation_log.txt"
FAILURES_LOG = BASE_DIR / "logs" / "translation_failures.md"
TERMINOLOGY_FLAGS_LOG = BASE_DIR / "logs" / "terminology_flags.md"
DEBUG_DIR = BASE_DIR / "logs" / "debug"

load_dotenv(BASE_DIR / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

if not DEEPSEEK_API_KEY:
    raise SystemExit(
        "ERROR: DEEPSEEK_API_KEY not found. Add it to your .env file as:\n"
        "  DEEPSEEK_API_KEY=sk-...\n"
    )

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 8
REQUEST_TIMEOUT_SECONDS = 180
SLEEP_BETWEEN_CHAPTERS = 1.5

# Token-budget escalation ladder for translate_chapter_with_recovery().
# 1200 was sized for an "average" chapter; chapters with one or two very
# long Telugu answers (e.g. chapter_033) can blow through that ceiling on
# the FIRST q_no alone, since MAX_TOKENS is a per-request budget shared
# across the whole qa_pairs array, not per-answer. Each recovery attempt
# steps up the ceiling rather than giving up after one shot at 1200.
MAX_TOKENS = 1200
MAX_TOKENS_ESCALATION = [1200, 2400, 4000]

LENGTH_CHECK_MIN_SOURCE_CHARS = 80
LENGTH_RATIO_FLOOR = 0.75

# ---------------------------------------------------------------------------
# A. GLOSSARY — canonical proper-noun spellings + known bad variants
# ---------------------------------------------------------------------------
GLOSSARY_TERMS = {
    "Rama": ["Raman", "Ramudu", "Sri Ramudu"],
    "Sita": ["Seetha", "Seeta", "Sitha"],
    "Lakshmana": ["Lakshman", "Laxmana", "Lakshmanudu"],
    "Bharata": ["Bharath", "Bharatha"],
    "Shatrughna": ["Shatrugna", "Satrughna"],
    "Dasharatha": ["Dasaratha", "Dasharath", "Dasaradha"],
    "Kausalya": ["Kausalaya", "Kowsalya"],
    "Kaikeyi": ["Kaikeye", "Kaikai"],
    "Sumitra": ["Sumithra"],
    "Janaka": ["Janak"],
    "Urmila": ["Urmilla"],
    "Mandavi": ["Mandhavi"],
    "Shrutakirti": ["Sruthakirthi"],
    "Vishvamitra": ["Vishwamitra", "Viswamitra"],
    "Vasishtha": ["Vasishta", "Vasistha"],
    "Valmiki": ["Valmeeki"],
    "Rishyashringa": ["Rushyashringa"],
    "Kushadhwaja": ["Kushadwaja", "Kusadhwaja"],
    "Ravana": ["Raavana"],
    "Maricha": ["Mareecha", "Marica"],
    "Hanuman": ["Hanumanthudu", "Hanumantha"],
    "Sugriva": ["Sugreeva"],
    "Vali": ["Vaali"],
    "Bharadwaja": ["Bharadvaja", "Bharadhwaja"],
    "Guha": [],
    "Shabari": ["Sabari"],
    "Agastya": ["Agasthya"],
    "Kabandha": [],
    "Parashurama": ["Parasurama"],
    "Indra": [],
    "Brahma": [],
    "Mahavishnu": ["Maha Vishnu", "Vishnu"],
    "Ahalya": ["Ahalyä", "Ahalia"],
    "Ganga": ["Ganges"],
    "Bhagiratha": ["Bhageeratha"],
    "Trishanku": ["Trisanku"],
    "Shatananda": ["Satananda"],
    "Ayodhya": ["Ayodya", "Ayodhia"],
    "Mithila": [],
    "Kosala": ["Kosal"],
    "Videha": [],
    "Lanka": [],
    "Kishkindha": ["Kishkinda"],
    "Chitrakuta": ["Chitrakoot", "Chitrakoota"],
    "Dandakaranya": ["Dandaka forest", "Dandakaranya forest"],
    "Sarayu": ["Saryu"],
    "Tamasa": ["Tamsa"],
    "Yamuna": [],
    "Prayaga": ["Prayag"],
    "Panchavati": [],
    "Siddhashrama": ["Siddhasrama"],
    "Maharshi": ["Maharishi"],
    "Brahmarshi": ["Brahmarishi"],
    "Rajarshi": ["Rajarishi"],
}

GLOSSARY_PROMPT_BLOCK = "\n".join(f"- {k}" for k in GLOSSARY_TERMS)

# ---------------------------------------------------------------------------
# A2. FORBIDDEN TRANSLATIONS — protected Sanskrit/sacred terms and the
# secularized English substitutes that must NEVER replace them. Used both
# to instruct the model explicitly, and to detect drift in validation.
# ---------------------------------------------------------------------------
FORBIDDEN_TRANSLATIONS = {
    "Maharshi": ["sage", "great sage", "saint"],
    "Brahmarshi": ["sage", "saint"],
    "Rajarshi": ["royal sage", "king-sage"],
    "Ashrama": ["hermitage", "cottage", "monastery"],
    "Yajna": ["sacrifice", "ritual"],
    "Yaga": ["sacrifice", "ritual"],
    "Rakshasa": ["demon", "monster", "ogre"],
    "Vanara": ["monkey", "ape"],
    "Svarga": ["heaven", "paradise"],
    "Naraka": ["hell"],
    "Deva": ["god", "angel"],
    "Asura": ["demon"],
    "Tapas": ["penance", "austerity", "meditation"],
    "Dharma": ["duty", "righteousness", "law"],
    "Karma": ["action", "fate", "destiny"],
    "Moksha": ["salvation", "liberation", "enlightenment"],
    "Bhakti": ["devotion"],
    "Nakshatra": ["star", "constellation", "asterism"],
    "Ashvamedha": ["horse sacrifice"],
    "Putrakameshti": ["son-seeking sacrifice", "fertility ritual"],
}

def _format_forbidden_line(term, subs):
    sub_list = '", "'.join(subs)
    return f'- {term}: never write "{sub_list}" instead'


FORBIDDEN_PROMPT_BLOCK = "\n".join(
    _format_forbidden_line(term, subs) for term, subs in FORBIDDEN_TRANSLATIONS.items()
)

# ---------------------------------------------------------------------------
# A3. MODERN PHRASE PATTERNS — casual/modern wording that breaks epic
# register, mapped to the classical equivalent the prompt asks for.
# Used for detection only (we do not blind-replace prose — flag for review).
# ---------------------------------------------------------------------------
MODERN_PHRASE_PATTERNS = {
    r"\bwent back\b": "returned",
    r"\btraveled some distance\b": "having journeyed further",
    r"\bwalked some distance\b": "having journeyed further",
    r"\brested there\b": "halted there",
    r"\bwas happy\b": "was greatly pleased",
    r"\bwas very happy\b": "was greatly pleased",
    r"\basked him\b": "inquired of him",
    r"\basked her\b": "inquired of her",
    r"\bsaid to him\b": "spoke to him",
    r"\bgot scared\b": "was overcome with fear",
    r"\btold him\b": "addressed him",
    r"\bokay\b": None,
    r"\byeah\b": None,
    r"\bguys\b": None,
    r"\bkind of\b": None,
    r"\bbasically\b": None,
}


def normalize_glossary_terms(text: str) -> str:
    """Deterministic safety net for proper-noun spelling drift."""
    if not text:
        return text
    for canonical, variants in GLOSSARY_TERMS.items():
        for variant in variants:
            text = re.sub(rf"\b{re.escape(variant)}\b", canonical, text)
    return text


def normalize_honorific_adjacency(text: str) -> str:
    """Targeted, safe correction: when a forbidden generic word like 'sage'
    or 'hermitage' sits immediately beside a glossary proper noun, it is
    almost certainly standing in for Maharshi/Ashrama and can be corrected
    automatically. We do NOT blind-replace every instance of 'sage' in the
    text, because some may be legitimate generic narration with no
    corresponding Sanskrit term in source — those get flagged instead,
    not silently rewritten (rewriting an ambiguous case risks introducing
    a NEW fidelity error rather than fixing one)."""
    if not text:
        return text
    proper_nouns = "|".join(re.escape(n) for n in GLOSSARY_TERMS if n not in
                             ("Maharshi", "Brahmarshi", "Rajarshi", "Ashrama"))
    # "the sage Vishvamitra" / "sage Vishvamitra" -> "Maharshi Vishvamitra"
    text = re.sub(
        rf"\b(?:the\s+)?(?:great\s+)?sage\s+({proper_nouns})\b",
        r"Maharshi \1", text, flags=re.IGNORECASE,
    )
    # "Vishvamitra's hermitage" -> "Vishvamitra's Ashrama"
    text = re.sub(
        rf"\b({proper_nouns})'s\s+hermitage\b",
        r"\1's Ashrama", text, flags=re.IGNORECASE,
    )
    # "Vishvamitra hermitage" / "Vishvamitra's hermitage" (no possessive,
    # Telugu-style genitive-less compounding) -> "Vishvamitra Ashrama"
    text = re.sub(
        rf"\b({proper_nouns})\s+hermitage\b",
        r"\1 Ashrama", text, flags=re.IGNORECASE,
    )
    # "hermitage of Vishvamitra" -> "Ashrama of Vishvamitra"
    text = re.sub(
        rf"\bhermitage of ({proper_nouns})\b",
        r"Ashrama of \1", text, flags=re.IGNORECASE,
    )
    return text


def detect_forbidden_translations(text: str) -> list:
    """Return list of (forbidden_phrase, likely_protected_term) hits."""
    hits = []
    if not text:
        return hits
    for term, subs in FORBIDDEN_TRANSLATIONS.items():
        for sub in subs:
            if re.search(rf"\b{re.escape(sub)}\b", text, flags=re.IGNORECASE):
                hits.append((sub, term))
    return hits


def detect_modern_phrases(text: str) -> list:
    hits = []
    if not text:
        return hits
    for pattern in MODERN_PHRASE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


# ---------------------------------------------------------------------------
# B. SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are translating a Telugu Ramayana Q&A commentary text
into English for a permanent SACRED-TEXT reference corpus. You are not writing
modern prose — you are producing a translation in the register of a published
English edition of the Valmiki Ramayana (e.g. comparable to standard
scholarly/devotional English Ramayana editions).

YOUR SINGLE HIGHEST PRIORITY: maximum fidelity to the Telugu source — literal,
narrative, theological, devotional, and cultural meaning all together. Where
fidelity and smooth modern readability conflict, ALWAYS choose fidelity.
Readability is not a goal of this task.

==================================================
1. NEVER TRANSLATE THESE PROTECTED SANSKRIT/SACRED TERMS INTO ENGLISH
==================================================
Keep these transliterated exactly as given, in every occurrence, no exceptions.
The list below shows the term and the SECULARIZED SUBSTITUTES YOU MUST NEVER
USE IN ITS PLACE:
{FORBIDDEN_PROMPT_BLOCK}

This is a hard rule, not a stylistic preference. "Maharshi" must never become
"sage." "Ashrama" must never become "hermitage." "Yajna"/"Yaga" must never
become "sacrifice" or "ritual" alone. "Rakshasa" must never become "demon."
"Vanara" must never become "monkey." "Svarga"/"Naraka" must never become
"heaven"/"hell." Use the Sanskrit term itself every time.

==================================================
2. COMPLETENESS — NO OMISSION, NO SUMMARIZATION, NO EXPANSION
==================================================
Every meaningful statement, clause, and detail in the Telugu source must
appear in the English translation. A long answer in Telugu must produce a
long answer in English — never shorten or compress. Do not add
interpretation, explanation, or any fact not present in the source.

==================================================
3. EPIC REGISTER — NO MODERN OR CASUAL PHRASING
==================================================
Write in classical, elevated narrative English. Examples of what you must
avoid and what to use instead:
  - "went back" -> "returned"
  - "traveled some distance" -> "having journeyed further"
  - "rested there" -> "halted there"
  - "was happy" -> "was greatly pleased"
  - "asked him" -> "inquired of him"
Never use casual, journalistic, corporate, or conversational English
(no "okay," "basically," "kind of," contractions, or modern idiom).
Preserve direct speech as direct quoted dialogue, never flattened into
indirect/reported narration.

==================================================
4. THEOLOGICAL AND DEVOTIONAL FIDELITY — NEVER SECULARIZE
==================================================
References to Rama, Sita, Lord Shiva, Lord Vishnu (Mahavishnu), Brahma,
Maharshis, Yajnas, Tapas, and all divine/sacred events must retain the exact
level of reverence present in the Telugu source — never flatten devotional
or honorific language into neutral/secular phrasing, and never invent
reverence the source does not contain. Do not substitute Hindu/Vedic
concepts with Christian, Islamic, or Western religious terminology (no
"prophet," "saint," "heaven," "hell" unless the Telugu concept is genuinely
equivalent — see protected-terms list above, Svarga and Naraka must stay
transliterated, not become "heaven"/"hell").

==================================================
5. HIERARCHY, HONORIFICS, RELATIONSHIPS
==================================================
Preserve titles exactly as conveyed: Maharshi, Brahmarshi, Rajarshi, King,
Prince, Queen, Lord, Guru, disciple — never drop or generalize them.
Preserve familial/social relationships explicitly stated: elder brother,
younger brother, mother, father, guru, disciple, etc.

==================================================
6. PROPER NOUNS — EXACT CANONICAL SPELLINGS, ZERO VARIATION
==================================================
{GLOSSARY_PROMPT_BLOCK}
Use these spellings every single time, with zero variation, across all 99
chapters of this corpus. Never invent an alternate spelling.

==================================================
SELF-CHECK BEFORE YOU RETURN YOUR ANSWER
==================================================
- Did you replace ANY protected Sanskrit term with its forbidden English
  substitute (see section 1)? If so, fix it before responding.
- Did you compress, summarize, or omit anything?
- Did you use any modern/casual phrasing (see section 3 examples)?
- Does every name match the canonical glossary spelling exactly?
- Is devotional/theological tone preserved at the source's level, not
  flattened or secularized?
- Is direct speech still direct speech?

==================================================
OUTPUT FORMAT — STRICT
==================================================
Return ONLY a single valid JSON object. No markdown fences, no preamble, no
trailing commentary. Schema:

{{
  "kanda_en": "<English name of this kanda/book, e.g. 'Bala Kanda (Book of Childhood)'>",
  "chapter_ordinal_en": "<English ordinal, e.g. 'Chapter Twelve'>",
  "qa_pairs": [
    {{"q_no": 1, "question_en": "...", "answer_en": "..."}}
  ]
}}

The qa_pairs array in your output MUST have exactly the same number of
entries, in the same q_no order, as the input qa_pairs you are given.

CRITICAL JSON-VALIDITY RULE FOR QUOTED DIALOGUE: Section 3 asks you to
preserve direct speech as direct quoted dialogue. When you do this inside a
JSON string value, every double-quote character that is part of the
dialogue itself MUST be escaped as \\" — never write a bare " inside a
question_en or answer_en string. For example, write:
  "answer_en": "Sita said, \\"O mother! Janaka is my father...\\" and continued."
NOT:
  "answer_en": "Sita said, "O mother! Janaka is my father..." and continued."
A single unescaped quote inside a long answer breaks the entire JSON
response and discards the whole chapter's translation. Re-check every
quoted-dialogue passage in your answer for this before returning it."""


REVIEW_SYSTEM_PROMPT = f"""You are a strict fidelity reviewer for a Telugu-to-
English Ramayana sacred-text translation. You will be given the ORIGINAL
TELUGU source JSON and a DRAFT ENGLISH translation JSON of the same chapter.

Your job is to find and FIX every violation of these rules in the draft, then
return a corrected JSON in the exact same schema:

1. Any protected Sanskrit term replaced with a forbidden secular substitute
   must be restored to the Sanskrit term. Forbidden substitutes to hunt for:
{FORBIDDEN_PROMPT_BLOCK}
2. Any summarized, compressed, or omitted content must be restored — compare
   length and detail against the Telugu source; expand any answer that looks
   thinner than the source warrants.
3. Any modern/casual phrasing must be rewritten into classical epic register
   (e.g. "went back" -> "returned," "was happy" -> "was greatly pleased").
4. Any proper noun not matching this canonical glossary must be corrected:
{GLOSSARY_PROMPT_BLOCK}
5. Any flattened direct speech (turned into indirect narration) should be
   restored to quoted dialogue if the Telugu source uses direct speech.
6. Any secularized devotional/theological language must be restored to the
   level of reverence present in the Telugu source.

If the draft already fully complies, return it unchanged. Return ONLY the
corrected JSON object, no commentary, no markdown fences, same schema as the
draft (kanda_en, chapter_ordinal_en, qa_pairs with q_no/question_en/answer_en)."""


def build_user_prompt(chapter: dict) -> str:
    payload = {
        "kanda": chapter.get("kanda", ""),
        "chapter_ordinal": chapter.get("chapter_ordinal", ""),
        "chapter_number": chapter.get("chapter_number"),
        "sarga_range": chapter.get("sarga_range", []),
        "qa_pairs": [
            {"q_no": qa["q_no"], "question": qa["question"], "answer": qa["answer"]}
            for qa in chapter["qa_pairs"]
        ],
    }
    return (
        "Translate the following Telugu Ramayana chapter Q&A content into "
        "English, following the system rules exactly. Here is the source "
        "JSON:\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_review_prompt(chapter: dict, draft: dict) -> str:
    src_payload = {
        "kanda": chapter.get("kanda", ""),
        "chapter_ordinal": chapter.get("chapter_ordinal", ""),
        "qa_pairs": [
            {"q_no": qa["q_no"], "question": qa["question"], "answer": qa["answer"]}
            for qa in chapter["qa_pairs"]
        ],
    }
    return (
        "ORIGINAL TELUGU SOURCE:\n" + json.dumps(src_payload, ensure_ascii=False, indent=2) +
        "\n\nDRAFT ENGLISH TRANSLATION TO REVIEW AND CORRECT:\n" +
        json.dumps(draft, ensure_ascii=False, indent=2)
    )


class DeepSeekCallError(RuntimeError):
    """Raised when _call_deepseek_raw exhausts retries. Carries the last raw
    response content (if any was received) so callers can save it for
    debugging instead of losing it the moment the exception is raised."""
    def __init__(self, message: str, raw_content: str = None):
        super().__init__(message)
        self.raw_content = raw_content


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def attempt_json_repair(text: str):
    """Targeted repair for the most common DeepSeek json_object corruption
    we see on long answers containing quoted direct speech: the model
    renders dialogue with a literal " instead of escaping it to \\", which
    produces 'Expecting , delimiter' right after the stray quote. This is
    NOT a generic JSON fixer - it only handles this one well-understood
    pattern (confirmed via chapter_033, which has multi-hundred-char
    direct-speech passages the system prompt explicitly asks to preserve
    as quoted dialogue). On each JSONDecodeError, it walks back from the
    reported error position to the nearest quote character, escapes it,
    and retries parsing. Returns the parsed dict on success, or None if it
    can't converge (e.g. the failure is genuine truncation, not a stray
    quote - that case should fall through to token-budget escalation
    instead of being silently papered over here)."""
    current = text
    for _ in range(25):  # a single long answer can contain several quotes
        try:
            return json.loads(current)
        except json.JSONDecodeError as e:
            pos = e.pos
            search_from = max(0, pos - 200)
            window = current[search_from:pos]
            quote_idx = window.rfind('"')
            if quote_idx == -1:
                return None
            abs_idx = search_from + quote_idx
            if abs_idx > 0 and current[abs_idx - 1] == "\\":
                window2 = current[search_from:abs_idx]
                quote_idx2 = window2.rfind('"')
                if quote_idx2 == -1:
                    return None
                abs_idx = search_from + quote_idx2
            current = current[:abs_idx] + '\\"' + current[abs_idx + 1:]
    return None


# ---------------------------------------------------------------------------
# E. Architecture - API calls
# ---------------------------------------------------------------------------
def _call_deepseek_raw(messages: list, max_tokens: int = MAX_TOKENS):
    """Returns (parsed_json, finish_reason, raw_content, was_repaired).
    raw_content is the unparsed response text (post code-fence-stripping),
    kept so callers can persist it for debugging even when json.loads()
    fails or the structure is wrong - we never want to lose the model's
    actual output just because it didn't parse cleanly. was_repaired is
    True if attempt_json_repair() had to fix a stray-quote corruption to
    get a parseable result; callers surface this so it's visible which
    chapters needed it (a signal worth a closer human read even though
    validation will still run normally on the repaired content)."""
    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    last_err = None
    last_raw_content = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF_SECONDS * attempt
                print(f"    [rate limited] waiting {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            content = strip_code_fences(choice["message"]["content"])
            last_raw_content = content
            finish_reason = choice.get("finish_reason", "unknown")
            try:
                parsed = json.loads(content)
                return parsed, finish_reason, content, False
            except json.JSONDecodeError as parse_err:
                repaired = attempt_json_repair(content)
                if repaired is not None:
                    print(f"    [attempt {attempt}/{MAX_RETRIES}] json had an unescaped-quote "
                          f"defect at char {parse_err.pos} - auto-repaired successfully")
                    return repaired, finish_reason, content, True
                raise  # repair couldn't converge; treat as a normal parse failure below
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            last_err = e
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"    [attempt {attempt}/{MAX_RETRIES} failed: {e}] retrying in {wait}s")
            time.sleep(wait)

    raise DeepSeekCallError(
        f"DeepSeek call failed after {MAX_RETRIES} attempts: {last_err}",
        raw_content=last_raw_content,
    )


def call_deepseek(chapter: dict, max_tokens: int = MAX_TOKENS):
    """First-pass translation. Returns (parsed_json, finish_reason, raw_content, was_repaired)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(chapter)},
    ]
    return _call_deepseek_raw(messages, max_tokens=max_tokens)


def call_deepseek_review(chapter: dict, draft: dict, max_tokens: int = MAX_TOKENS):
    """Second-pass self-review/correction. Returns (parsed_json, finish_reason, raw_content, was_repaired)."""
    messages = [
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": build_review_prompt(chapter, draft)},
    ]
    return _call_deepseek_raw(messages, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# C. Validation layer - structural + fidelity + terminology-drift heuristics
# ---------------------------------------------------------------------------
def validate_translation(chapter: dict, translated: dict, finish_reason: str) -> tuple:
    """Returns (blocking_problems, soft_flags, diagnostics).
    blocking_problems -> chapter is rejected and logged as a failure.
    soft_flags -> chapter is still saved, but flagged in terminology_flags.md
    for human review (forbidden-term / modern-phrase hits are judgment calls,
    not always wrong, so they don't block output by default).
    diagnostics -> structured per-chapter detail (independent of pass/fail)
    used for the detailed failure log requested for debugging."""
    problems = []
    flags = []
    src_qnos = [qa["q_no"] for qa in chapter["qa_pairs"]]
    diagnostics = {
        "finish_reason": finish_reason,
        "expected_q_count": len(src_qnos),
        "returned_q_count": None,
        "per_q": [],  # list of dicts: q_no, source_answer_length, translated_answer_length, ratio
    }

    if finish_reason == "length":
        problems.append("response was TRUNCATED by max_tokens - likely incomplete content")

    if "qa_pairs" not in translated:
        problems.append("missing qa_pairs in response")
        diagnostics["returned_q_count"] = 0
        return problems, flags, diagnostics

    out_qnos = [qa.get("q_no") for qa in translated["qa_pairs"]]
    diagnostics["returned_q_count"] = len(out_qnos)
    if src_qnos != out_qnos:
        problems.append(f"q_no mismatch: expected {src_qnos}, got {out_qnos}")

    src_by_qno = {qa["q_no"]: qa for qa in chapter["qa_pairs"]}

    for qa in translated.get("qa_pairs", []):
        qno = qa.get("q_no")
        q = qa.get("question_en", "") or ""
        a = qa.get("answer_en", "") or ""

        if not q.strip():
            problems.append(f"q{qno} has empty translated question")
        if not a.strip():
            problems.append(f"q{qno} has empty translated answer")

        if re.search(r"[\u0C00-\u0C7F]", a):
            problems.append(f"q{qno} answer still contains Telugu text")
        if re.search(r"[\u0C00-\u0C7F]", q):
            problems.append(f"q{qno} question still contains Telugu text")

        src_qa = src_by_qno.get(qno)
        q_diag = {
            "q_no": qno,
            "source_answer_length": len(src_qa.get("answer", "")) if src_qa else None,
            "translated_answer_length": len(a),
            "ratio": None,
        }
        if src_qa:
            src_len = len(src_qa.get("answer", ""))
            if src_len >= LENGTH_CHECK_MIN_SOURCE_CHARS:
                ratio = len(a) / max(src_len, 1)
                q_diag["ratio"] = round(ratio, 2)
                if ratio < LENGTH_RATIO_FLOOR:
                    problems.append(
                        f"q{qno} answer looks possibly summarized "
                        f"(source {src_len} chars -> translation {len(a)} chars, ratio {ratio:.2f})"
                    )
        diagnostics["per_q"].append(q_diag)

        for sub, term in detect_forbidden_translations(a) + detect_forbidden_translations(q):
            flags.append(f"q{qno}: found forbidden substitute '{sub}' (likely should be '{term}')")
        for pattern in detect_modern_phrases(a) + detect_modern_phrases(q):
            flags.append(f"q{qno}: modern/casual phrasing matched pattern {pattern!r}")

    return problems, flags, diagnostics


def detect_glossary_terms_used(text: str) -> set:
    found = set()
    for canonical in GLOSSARY_TERMS:
        if re.search(rf"\b{re.escape(canonical)}\b", text):
            found.add(canonical)
    return found


# ---------------------------------------------------------------------------
# F. Quality scoring — automated 0-100 fidelity rubric per chapter
# ---------------------------------------------------------------------------
def quality_score(chapter: dict, translated: dict) -> dict:
    """Heuristic 0-100 score. NOT a substitute for human review — catches
    the mechanically-detectable portion of fidelity (terminology, length,
    glossary, modern phrasing). True meaning-preservation and epic-tone
    quality ultimately need a bilingual human reviewer; this score tells you
    WHERE to focus that review, not a final verdict."""
    src_by_qno = {qa["q_no"]: qa for qa in chapter["qa_pairs"]}
    qa_pairs = translated.get("qa_pairs", [])
    n = max(len(qa_pairs), 1)

    # 1. Meaning preservation (20 pts) - length-ratio proxy
    ratio_scores = []
    for qa in qa_pairs:
        src_qa = src_by_qno.get(qa.get("q_no"))
        a = qa.get("answer_en", "") or ""
        if src_qa:
            src_len = len(src_qa.get("answer", ""))
            if src_len >= LENGTH_CHECK_MIN_SOURCE_CHARS:
                ratio = min(len(a) / max(src_len, 1), 1.2)
                ratio_scores.append(min(ratio / LENGTH_RATIO_FLOOR, 1.0) if ratio < LENGTH_RATIO_FLOOR else 1.0)
            else:
                ratio_scores.append(1.0)
    meaning_score = (sum(ratio_scores) / len(ratio_scores) * 20) if ratio_scores else 20

    # 2. Terminology preservation (25 pts) - penalize forbidden substitutes
    forbidden_hits = 0
    for qa in qa_pairs:
        forbidden_hits += len(detect_forbidden_translations(qa.get("answer_en", "")))
        forbidden_hits += len(detect_forbidden_translations(qa.get("question_en", "")))
    terminology_score = max(25 - forbidden_hits * 5, 0)

    # 3. Glossary compliance (20 pts) - penalize remaining bad-variant spellings
    variant_hits = 0
    full_text = " ".join(
        (qa.get("question_en", "") or "") + " " + (qa.get("answer_en", "") or "")
        for qa in qa_pairs
    )
    for variants in GLOSSARY_TERMS.values():
        for variant in variants:
            if re.search(rf"\b{re.escape(variant)}\b", full_text):
                variant_hits += 1
    glossary_score = max(20 - variant_hits * 4, 0)

    # 4. Epic tone (20 pts) - penalize modern phrase hits
    modern_hits = 0
    for qa in qa_pairs:
        modern_hits += len(detect_modern_phrases(qa.get("answer_en", "")))
        modern_hits += len(detect_modern_phrases(qa.get("question_en", "")))
    tone_score = max(20 - modern_hits * 4, 0)

    # 5. Consistency (15 pts) - within-chapter only here; cross-chapter
    #    consistency requires a corpus-wide pass, see scripts/04_audit_consistency.py
    consistency_score = 15  # placeholder ceiling for single-chapter scope

    total = round(meaning_score + terminology_score + glossary_score + tone_score + consistency_score, 1)
    return {
        "total": total,
        "meaning_preservation": round(meaning_score, 1),
        "terminology_preservation": terminology_score,
        "glossary_compliance": glossary_score,
        "epic_tone": tone_score,
        "consistency_within_chapter": consistency_score,
        "forbidden_term_hits": forbidden_hits,
        "glossary_variant_hits": variant_hits,
        "modern_phrase_hits": modern_hits,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_line(msg: str):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_failure(chapter: dict, reason: str):
    FAILURES_LOG.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not FAILURES_LOG.exists()
    with open(FAILURES_LOG, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("# Translation Failures\n\n")
        f.write(
            f"**{chapter.get('chunk_id')} "
            f"({chapter.get('kanda')} - {chapter.get('chapter_ordinal')})** "
            f"failed: {reason}\n\n"
        )


def log_failure_detailed(chapter: dict, problems: list, diagnostics: dict, attempts_used: int = 1):
    """Structured diagnostics block per the requested debugging format:
    chunk_id, finish_reason, expected/returned q_count, per-q source/translated
    lengths and ratios, and the validation error list. Written alongside (not
    instead of) the existing one-line log_failure() entry."""
    FAILURES_LOG.parent.mkdir(parents=True, exist_ok=True)
    chunk_id = chapter.get("chunk_id")
    lines = [f"### DETAILED DIAGNOSTICS: {chunk_id}"]
    lines.append(f"finish_reason={diagnostics.get('finish_reason')}")
    lines.append(f"expected_q_count={diagnostics.get('expected_q_count')}")
    lines.append(f"returned_q_count={diagnostics.get('returned_q_count')}")
    lines.append(f"attempts_used={attempts_used}")
    for q in diagnostics.get("per_q", []):
        lines.append(
            f"q{q['q_no']} source_len={q['source_answer_length']} "
            f"translated_len={q['translated_answer_length']} ratio={q['ratio']}"
        )
    for p in problems:
        lines.append(f"validation_error={p}")
    block = "\n".join(lines) + "\n\n"
    with open(FAILURES_LOG, "a", encoding="utf-8") as f:
        f.write(block)
    print(block)


def save_debug_dump(chunk_id: str, label: str, content):
    """Persist raw/intermediate data for a chapter under logs/debug/<chunk_id>/
    so failures can be inspected by hand. `content` may be a string (raw model
    output) or a JSON-serializable object."""
    chapter_dir = DEBUG_DIR / chunk_id
    chapter_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = chapter_dir / f"{stamp}_{label}.json" if not isinstance(content, str) else chapter_dir / f"{stamp}_{label}.txt"
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def log_terminology_flags(chapter: dict, flags: list, score: dict):
    if not flags and score["total"] >= 90:
        return
    TERMINOLOGY_FLAGS_LOG.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not TERMINOLOGY_FLAGS_LOG.exists()
    with open(TERMINOLOGY_FLAGS_LOG, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("# Terminology / Fidelity Flags (human review queue)\n\n")
        f.write(
            f"## {chapter.get('chunk_id')} "
            f"({chapter.get('kanda')} - {chapter.get('chapter_ordinal')}) "
            f"- score {score['total']}/100\n\n"
        )
        f.write(
            f"breakdown: meaning={score['meaning_preservation']}/20, "
            f"terminology={score['terminology_preservation']}/25, "
            f"glossary={score['glossary_compliance']}/20, "
            f"tone={score['epic_tone']}/20, "
            f"consistency={score['consistency_within_chapter']}/15\n\n"
        )
        for flag in flags:
            f.write(f"- {flag}\n")
        f.write("\n")


# ---------------------------------------------------------------------------
# Kanda name lookup (module-level — previously this was accidentally defined
# partway through translate_chapter's body, which made it dead code and
# silently dropped the success-path's `return record`. Fixed as a pure
# placement fix; no lookup logic changed.)
# ---------------------------------------------------------------------------
KANDA_MAP = {
    "బాల కాండము": "Bala Kanda (Book of Childhood)",
    "అయోధ్యా కాండము": "Ayodhya Kanda (Book of Ayodhya)",
    "అరణ్య కాండము": "Aranya Kanda (Book of the Forest)",
    "కిష్కింధా కాండము": "Kishkindha Kanda (Book of Kishkindha)",
    "సుందరా కాండము": "Sundara Kanda (Book of Beauty)",
    "యుద్ధ కాండము": "Yuddha Kanda (Book of War)",
    "ఉత్తర కాండము": "Uttara Kanda (Book of the Epilogue)",
}


def translate_kanda_name(kanda):
    return KANDA_MAP.get(kanda, kanda)


# ---------------------------------------------------------------------------
# D. Normalization + per-chapter translation (with optional two-pass review)
# ---------------------------------------------------------------------------
def translate_chapter(chapter: dict, two_pass: bool = False, max_tokens: int = MAX_TOKENS,
                       debug_dump: bool = False):
    """Single-attempt translation of one chapter at a given token budget.
    Returns (record_or_None, problems, diagnostics). problems/diagnostics are
    always returned (even on success, problems will be []) so the recovery
    wrapper can decide whether/how to retry without re-deriving that info."""
    chunk_id = chapter["chunk_id"]
    log_line(f"Translating {chunk_id} ({chapter['kanda']} - {chapter['chapter_ordinal']}) "
             f"[max_tokens={max_tokens}]...")
    try:
        translated, finish_reason, raw_content, was_repaired = call_deepseek(chapter, max_tokens=max_tokens)
    except DeepSeekCallError as e:
        log_line(f"  FAILED (api error): {e}")
        log_failure(chapter, f"API error: {e}")
        if debug_dump and e.raw_content:
            save_debug_dump(chunk_id, "pass1_raw_on_api_error", e.raw_content)
        diagnostics = {"finish_reason": "api_error", "expected_q_count": len(chapter["qa_pairs"]),
                        "returned_q_count": 0, "per_q": [], "json_repair_used": False}
        return None, [f"API error: {e}"], diagnostics

    if was_repaired:
        log_line(f"  [{chunk_id}] pass-1 response needed json-repair (likely an unescaped quote "
                 f"inside quoted dialogue) - repaired successfully, proceeding to validation")

    if debug_dump:
        save_debug_dump(chunk_id, "pass1_raw", raw_content)
        save_debug_dump(chunk_id, "pass1_parsed", translated)

    if two_pass:
        try:
            log_line(f"  Running self-review/correction pass for {chunk_id}...")
            reviewed, review_finish_reason, review_raw, review_was_repaired = call_deepseek_review(
                chapter, translated, max_tokens=max_tokens
            )
            if review_was_repaired:
                log_line(f"  [{chunk_id}] pass-2 (review) response also needed json-repair")
                was_repaired = True
            if debug_dump:
                save_debug_dump(chunk_id, "pass2_raw", review_raw)
                save_debug_dump(chunk_id, "pass2_parsed", reviewed)
            # only adopt the reviewed version if it's structurally sane;
            # otherwise fall back to the first-pass draft rather than losing data
            if "qa_pairs" in reviewed and len(reviewed["qa_pairs"]) == len(chapter["qa_pairs"]):
                translated = reviewed
                finish_reason = review_finish_reason
            else:
                log_line("  [review pass produced malformed output - keeping first-pass draft]")
        except DeepSeekCallError as e:
            log_line(f"  [review pass failed: {e} - keeping first-pass draft]")
            if debug_dump and e.raw_content:
                save_debug_dump(chunk_id, "pass2_raw_on_api_error", e.raw_content)

    problems, flags, diagnostics = validate_translation(chapter, translated, finish_reason)
    diagnostics["json_repair_used"] = was_repaired
    if problems:
        log_line(f"  FAILED (validation): {'; '.join(problems)}")
        return None, problems, diagnostics

    glossary_terms_used = set()
    qa_pairs_en = []
    for qa in translated["qa_pairs"]:
        q_norm = normalize_glossary_terms(qa["question_en"].strip())
        a_norm = normalize_glossary_terms(qa["answer_en"].strip())
        q_norm = normalize_honorific_adjacency(q_norm)
        a_norm = normalize_honorific_adjacency(a_norm)
        glossary_terms_used |= detect_glossary_terms_used(q_norm + " " + a_norm)
        qa_pairs_en.append({"q_no": qa["q_no"], "question_en": q_norm, "answer_en": a_norm})

    normalized_translated = {**translated, "qa_pairs": [
        {"q_no": qa["q_no"], "question_en": qa["question_en"], "answer_en": qa["answer_en"]}
        for qa in qa_pairs_en
    ]}
    score = quality_score(chapter, normalized_translated)
    log_terminology_flags(chapter, flags, score)

    record = {
        "chunk_id": chunk_id,
        "kanda": chapter["kanda"],
        "kanda_en": translated.get("kanda_en", translate_kanda_name(chapter["kanda"])),
        "chapter_ordinal": chapter["chapter_ordinal"],
        "chapter_ordinal_en": translated.get("chapter_ordinal_en", ""),
        "chapter_number": chapter.get("chapter_number"),
        "sarga_range": chapter.get("sarga_range", []),
        "qa_pairs_en": qa_pairs_en,
        "_qa_audit": {"glossary_terms_used": sorted(glossary_terms_used)},
        "_quality_score": score,
    }
    log_line(f"  OK: {chunk_id}  (score {score['total']}/100, "
              f"glossary terms used: {sorted(glossary_terms_used)})")
    return record, [], diagnostics


def translate_chapter_with_recovery(chapter: dict, two_pass: bool = False, debug_dump: bool = False):
    """Recovery wrapper around translate_chapter(). Retries with an escalating
    token budget (MAX_TOKENS_ESCALATION) specifically for the failure modes
    that token-starvation causes: finish_reason == 'length', malformed/
    truncated JSON (surfaced as a DeepSeekCallError after _call_deepseek_raw's
    own internal retries), missing qa_pairs, or a q_no mismatch consistent
    with a cut-off array. Other validation failures (e.g. leftover Telugu
    text, which is a translation-quality issue rather than a budget issue)
    are not worth re-spending tokens on with a *bigger* budget, but we still
    retry them once at the same budget in case of one-off model flakiness,
    since attempts are cheap relative to losing a chapter.

    Strategy (per chapter):
      Attempt 1: max_tokens = MAX_TOKENS_ESCALATION[0] (normal budget)
      Attempt 2: max_tokens = MAX_TOKENS_ESCALATION[1] (if attempt 1 looked
                 token-starved)
      Attempt 3: max_tokens = MAX_TOKENS_ESCALATION[-1] (largest budget,
                 last resort)
      Attempt 4 (only reached if all above failed): final attempt with
                 debug_dump forced on, so the raw response is saved for
                 human inspection before giving up.

    Returns the record dict on success, or None on exhausted failure (after
    writing detailed diagnostics to FAILURES_LOG).
    """
    chunk_id = chapter["chunk_id"]
    budgets = list(MAX_TOKENS_ESCALATION)
    last_problems, last_diagnostics = [], {}

    for attempt_num, budget in enumerate(budgets, start=1):
        is_last_attempt = attempt_num == len(budgets)
        record, problems, diagnostics = translate_chapter(
            chapter, two_pass=two_pass, max_tokens=budget,
            debug_dump=debug_dump or is_last_attempt,
        )
        last_problems, last_diagnostics = problems, diagnostics

        if record is not None:
            if attempt_num > 1:
                log_line(f"  RECOVERED {chunk_id} on attempt {attempt_num}/{len(budgets)} "
                         f"(max_tokens={budget})")
            return record

        token_starved = (
            diagnostics.get("finish_reason") in ("length", "api_error")
            or any("missing qa_pairs" in p or "q_no mismatch" in p for p in problems)
        )
        if not token_starved:
            # Not a budget problem (e.g. leftover Telugu text, empty field) -
            # escalating tokens won't fix it, but we still allow one extra
            # try at the next budget tier in case of model flakiness, except
            # on the final attempt where we just stop.
            log_line(f"  [{chunk_id}] attempt {attempt_num}/{len(budgets)} failed for a "
                     f"non-token-budget reason; will still try remaining attempts in case of flakiness")

        if not is_last_attempt:
            log_line(f"  [{chunk_id}] attempt {attempt_num}/{len(budgets)} failed "
                     f"(token_starved={token_starved}); escalating to max_tokens="
                     f"{budgets[attempt_num]} for next attempt")
            time.sleep(RETRY_BACKOFF_SECONDS)

    # All attempts exhausted - log detailed diagnostics and give up.
    log_failure(chapter, "; ".join(last_problems) if last_problems else "exhausted all recovery attempts")
    log_failure_detailed(chapter, last_problems, last_diagnostics, attempts_used=len(budgets))
    return None


def main():
    parser = argparse.ArgumentParser(description="Translate Ramayana chapters via DeepSeek API")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chapter", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--two-pass", action="store_true",
                         help="Add a self-review/correction pass after first translation (2x API cost, higher fidelity)")
    parser.add_argument("--force-retranslate", metavar="CHUNK_ID", default=None,
                         help="Ignore existing output and retranslate one chapter by chunk_id "
                              "(e.g. chapter_033). Saves raw pass-1/pass-2 responses and validation "
                              "diagnostics to logs/debug/<chunk_id>/ for investigation.")
    args = parser.parse_args()

    if not SOURCE_PATH.exists():
        raise SystemExit(f"ERROR: source file not found at {SOURCE_PATH}")

    chapters = load_jsonl(SOURCE_PATH)
    if not chapters:
        raise SystemExit(f"ERROR: no chapters loaded from {SOURCE_PATH}")

    already_done = {r["chunk_id"] for r in load_jsonl(OUTPUT_PATH)}

    if args.force_retranslate:
        targets = [c for c in chapters if c["chunk_id"] == args.force_retranslate]
        if not targets:
            raise SystemExit(f"ERROR: chunk_id {args.force_retranslate!r} not found in source")
        already_done.discard(args.force_retranslate)
        log_line(f"--force-retranslate: {args.force_retranslate} "
                  f"(existing output for this chunk_id, if any, will NOT be deduplicated - "
                  f"a new line will be appended; remove the old line from "
                  f"{OUTPUT_PATH} manually if you want a clean replacement)")
    elif args.chapter is not None:
        targets = [c for c in chapters if c.get("chapter_number") == args.chapter]
        if not targets:
            raise SystemExit(f"ERROR: chapter_number {args.chapter} not found in source")
        already_done.discard(targets[0]["chunk_id"])
    elif args.retry_failed:
        if not FAILURES_LOG.exists():
            print("No failures log found - nothing to retry.")
            return
        failed_ids = set(re.findall(r"\*\*(chapter_\d+)", FAILURES_LOG.read_text(encoding="utf-8")))
        targets = [c for c in chapters if c["chunk_id"] in failed_ids]
        already_done -= failed_ids
        FAILURES_LOG.unlink()
    else:
        targets = [c for c in chapters if c["chunk_id"] not in already_done]

    if args.limit is not None:
        targets = targets[: args.limit]

    if not targets:
        print("Nothing to translate - all chapters already done (or filter matched nothing).")
        return

    log_line(f"Starting translation run: {len(targets)} chapter(s) queued "
              f"({len(already_done)} already done, model={DEEPSEEK_MODEL}, two_pass={args.two_pass}, "
              f"token_escalation={MAX_TOKENS_ESCALATION})")

    success_count = 0
    fail_count = 0
    scores = []
    for i, chapter in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}]", end=" ")
        record = translate_chapter_with_recovery(
            chapter, two_pass=args.two_pass, debug_dump=bool(args.force_retranslate)
        )
        if record:
            append_jsonl(OUTPUT_PATH, record)
            success_count += 1
            scores.append(record["_quality_score"]["total"])
        else:
            fail_count += 1
        if i < len(targets):
            time.sleep(SLEEP_BETWEEN_CHAPTERS)

    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    log_line(f"Run complete. Success: {success_count}  Failed: {fail_count}  "
              f"Avg quality score: {avg_score}/100")
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Log:    {LOG_PATH}")
    if fail_count:
        print(f"Failures (needs review/retry): {FAILURES_LOG}")
        print("Re-run with --retry-failed once you've investigated.")
    print(f"Terminology/fidelity flags for human review: {TERMINOLOGY_FLAGS_LOG}")


if __name__ == "__main__":
    main()