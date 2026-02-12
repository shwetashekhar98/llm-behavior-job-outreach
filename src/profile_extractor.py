"""
Evidence-based profile extraction from resume/LinkedIn text.
Only extracts facts that can be directly quoted from input.
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from groq import Groq


def extract_evidence_based_facts(
    text: str,
    api_key: str,
    model: str = "llama-3.1-8b-instant"
) -> List[Dict]:
    """
    Extract facts from profile text with evidence quotes.
    Only extracts facts that can be directly quoted from input.
    
    Returns:
        List of fact dictionaries with:
        - value: The extracted fact
        - source_quote: Exact substring from text
        - start_index: Character start position
        - end_index: Character end position
        - confidence: 0.0-1.0
    """
    if not text or not text.strip():
        return []
    
    client = Groq(api_key=api_key)
    
    system_prompt = """You are a fact extraction assistant. Extract ONLY facts that can be directly quoted from the provided text.

CRITICAL RULES:
1. Extract ONLY facts that appear verbatim in the text
2. For each fact, provide the exact quote from the text
3. Do NOT infer, assume, or fabricate any information
4. If a fact cannot be directly quoted, mark it as "uncertain"
5. Return structured JSON with evidence

Return JSON array with format:
[
  {
    "value": "extracted fact text",
    "source_quote": "exact substring from input",
    "start_index": 0,
    "end_index": 50,
    "confidence": 0.95
  }
]

If you cannot find clear evidence for a fact, do NOT include it."""

    user_prompt = f"""Extract facts from this profile text. Only extract facts that can be directly quoted.

Text:
{text}

Return JSON array of facts with evidence quotes. Only include facts with clear evidence."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content or "{}"
        result_json = json.loads(result_text)
        
        # Extract facts array - handle different response formats
        if isinstance(result_json, list):
            facts = result_json
        else:
            facts = result_json.get("facts", [])
        
        # Validate each fact has evidence in source text
        validated_facts = []
        for fact in facts:
            source_quote = fact.get("source_quote", "")
            value = fact.get("value", "")
            
            # Verify quote exists in original text
            if source_quote and source_quote.lower() in text.lower():
                # Find actual indices
                start_idx = text.lower().find(source_quote.lower())
                if start_idx != -1:
                    end_idx = start_idx + len(source_quote)
                    fact["start_index"] = start_idx
                    fact["end_index"] = end_idx
                    fact["confidence"] = fact.get("confidence", 0.8)
                    validated_facts.append(fact)
        
        return validated_facts
    
    except Exception as e:
        # Fallback: simple extraction without LLM
        return extract_simple_facts(text)


def extract_simple_facts(text: str) -> List[Dict]:
    """
    Simple rule-based extraction as fallback.
    Only extracts obvious patterns with direct quotes.
    """
    facts = []
    text_lower = text.lower()
    
    # Extract education patterns
    education_patterns = [
        r'(?:degree|bachelor|master|phd|doctorate|ms|ma|bs|ba|m\.?s\.?c\.?|b\.?s\.?c\.?)\s+[^,\.]+',
        r'(?:graduated|studied|attended)\s+[^,\.]+',
    ]
    
    for pattern in education_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            quote = match.group(0)
            facts.append({
                "value": quote.strip(),
                "source_quote": quote,
                "start_index": match.start(),
                "end_index": match.end(),
                "confidence": 0.7
            })
    
    # Extract experience patterns
    experience_patterns = [
        r'(?:worked|employed|interned|served)\s+(?:at|for|with)\s+[^,\.]+',
        r'\d+\+?\s*(?:years?|months?)\s+(?:of\s+)?[^,\.]+',
    ]
    
    for pattern in experience_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            quote = match.group(0)
            facts.append({
                "value": quote.strip(),
                "source_quote": quote,
                "start_index": match.start(),
                "end_index": match.end(),
                "confidence": 0.7
            })
    
    return facts


def extract_structured_profile(form_data: Dict) -> List[Dict]:
    """
    Extract facts from structured form input.
    Each field becomes a fact with the field name as context.
    """
    facts = []
    
    if form_data.get("education"):
        facts.append({
            "value": form_data["education"],
            "source_quote": form_data["education"],
            "start_index": 0,
            "end_index": len(form_data["education"]),
            "confidence": 1.0,
            "category": "education"
        })
    
    if form_data.get("work_experience"):
        facts.append({
            "value": form_data["work_experience"],
            "source_quote": form_data["work_experience"],
            "start_index": 0,
            "end_index": len(form_data["work_experience"]),
            "confidence": 1.0,
            "category": "experience"
        })
    
    if form_data.get("skills"):
        facts.append({
            "value": form_data["skills"],
            "source_quote": form_data["skills"],
            "start_index": 0,
            "end_index": len(form_data["skills"]),
            "confidence": 1.0,
            "category": "skills"
        })
    
    return facts


def validate_fact_evidence(fact: Dict, source_text: str) -> bool:
    """
    Validate that a fact's source_quote exists in source text.
    """
    source_quote = fact.get("source_quote", "")
    if not source_quote:
        return False
    
    # Case-insensitive check
    return source_quote.lower() in source_text.lower()

