import re
from typing import Any, Dict
import json, re, unicodedata

# Sentinels to mark JSON boundaries in model output,
S = "<<<AI_JSON_START>>>"
E = "<<<AI_JSON_END>>>"

def clean_control_chars(s: str) -> str:
    return "".join(ch for ch in s if ch == "\n" or unicodedata.category(ch)[0] != "C")

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
    if s.endswith("```"):
        s = s[:-3]
    return s

def extract_json(text: str) -> Dict[str, Any]:
    if S in text and E in text:
        text = text.split(S, 1)[1].split(E, 1)[0]
    
    text = strip_code_fences(clean_control_chars(text)).strip()
    
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        text2 = re.sub(r",(\s*[}\]])", r"\1", text) 
        obj = json.loads(text2)
    
    if isinstance(obj, str):
        obj = json.loads(obj)
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")
    return obj