import re
from typing import Any, Dict, overload, Literal, List
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

# def extract_json(text: str) -> Dict[str, Any]:
#     if S in text and E in text:
#         text = text.split(S, 1)[1].split(E, 1)[0]
    
#     text = strip_code_fences(clean_control_chars(text)).strip()
    
#     try:
#         obj = json.loads(text)
#     except json.JSONDecodeError:
#         text2 = re.sub(r",(\s*[}\]])", r"\1", text) 
#         obj = json.loads(text2)
    
#     if isinstance(obj, str):
#         obj = json.loads(obj)
#     if not isinstance(obj, dict):
#         raise ValueError("Expected a JSON object")
#     return obj

def _core_parse(text: str) -> Any:
    """Parse JSON from text, handling sentinels, code fences, control chars, trailing commas."""
    if S in text and E in text:
        text = text.split(S, 1)[1].split(E, 1)[0]
    text = strip_code_fences(clean_control_chars(text)).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # remove trailing commas before '}' or ']'
        text2 = re.sub(r",(\s*[}\]])", r"\1", text)
        obj = json.loads(text2)

    # sometimes models double-encode JSON as a string
    if isinstance(obj, str):
        obj = json.loads(obj)

    return obj

@overload
def extract_json(text: str, expect: Literal["object"] = "object") -> Dict[str, Any]: ...
@overload
def extract_json(text: str, expect: Literal["array"]) -> List[Any]: ...
@overload
def extract_json(text: str, expect: Literal["any"]) -> Any: ...

def extract_json(text: str, expect: Literal["object","array","any"] = "object"):
    """
    Parse JSON from model output.
    - expect="object" (default): returns Dict[str, Any], else raises.
    - expect="array": returns List[Any], else raises.
    - expect="any": returns whatever was parsed (dict/list/primitive).
    """
    obj = _core_parse(text)

    if expect == "any":
        return obj

    if expect == "object":
        if not isinstance(obj, dict):
            raise ValueError("Expected a JSON object")
        return obj

    if expect == "array":
        if not isinstance(obj, list):
            raise ValueError("Expected a JSON array")
        return obj

    # defensive fallback (shouldn't hit)
    return obj