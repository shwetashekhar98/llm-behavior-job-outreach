"""
Strict evidence-based profile extraction from unstructured text.
Only extracts complete, professional facts with high confidence.
"""

import re
import json
from typing import List, Dict, Set
from groq import Groq


def clean_profile_text(text: str) -> str:
    """
    Clean profile text by removing UI artifacts, duplicates, and noise.
    
    Returns:
        Cleaned text ready for extraction
    """
    if not text:
        return ""
    
    # Remove common UI artifacts
    ui_patterns = [
        r'see more',
        r'show all \d+',
        r'view all',
        r'expand',
        r'collapse',
        r'\d+ (likes?|comments?|shares?|impressions?)',
        r'\d+ followers?',
        r'\d+ connections?',
        r'• • •',
        r'\.\.\.',
        r'\[.*?\]',  # Remove brackets content
        r'\(.*?\)',  # Remove parentheses (but keep if meaningful)
    ]
    
    cleaned = text
    for pattern in ui_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Remove timestamps (various formats)
    timestamp_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s+\d{4}',
        r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec),?\s+\d{4}',
    ]
    
    for pattern in timestamp_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Remove duplicate lines (keep only first occurrence)
    lines = cleaned.split('\n')
    seen = set()
    unique_lines = []
    for line in lines:
        line_stripped = line.strip().lower()
        if line_stripped and line_stripped not in seen:
            seen.add(line_stripped)
            unique_lines.append(line)
    cleaned = '\n'.join(unique_lines)
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    
    return cleaned.strip()


def is_valid_fact(fact: str) -> bool:
    """
    Validate if a fact is acceptable.
    Rejects fragments, broken text, UI noise, and incomplete phrases.
    
    Returns:
        True if fact is valid, False otherwise
    """
    if not fact or len(fact.strip()) < 5:
        return False
    
    fact_lower = fact.lower()
    
    # Reject if too short (less than 5 meaningful words)
    words = fact.split()
    if len(words) < 5:
        return False
    
    # Reject broken fragments
    broken_patterns = [
        r'^\w+\s+that\s+are\s+\w+$',  # "ms that are observable"
        r'^\w+\s+across\s+\w+$',  # "ms across public"
        r'^\w+\s+and\s+\w+$',  # Single word fragments
    ]
    
    for pattern in broken_patterns:
        if re.match(pattern, fact_lower):
            return False
    
    # Reject if contains UI noise
    ui_noise = ['see more', 'show all', 'expand', 'click', 'view', 'followers', 'connections']
    if any(noise in fact_lower for noise in ui_noise):
        return False
    
    # Reject if looks like substring (starts/ends with incomplete word)
    if re.match(r'^\w+\s+\w+\s+\w+$', fact) and len(fact) < 30:
        # Very short 3-word phrases are likely fragments
        return False
    
    # Reject if contains broken words (multiple single letters)
    if len(re.findall(r'\b\w\b', fact)) > 3:
        return False
    
    # Must contain at least one complete sentence or meaningful phrase
    if not re.search(r'[.!?]|(?:worked|studied|led|built|developed|created|designed|implemented)', fact_lower):
        # Check if it's a meaningful phrase
        meaningful_indicators = [
            r'\d+',  # Contains numbers
            r'(?:years?|months?|experience|degree|university|college|company|engineer|developer|scientist)',
        ]
        if not any(re.search(pattern, fact_lower) for pattern in meaningful_indicators):
            return False
    
    return True


def extract_evidence_based_facts(
    text: str,
    api_key: str,
    model: str = "llama-3.1-8b-instant"
) -> List[Dict]:
    """
    Extract structured professional facts from unstructured profile text.
    Strict evidence-based extraction with validation.
    
    Returns:
        List of fact dictionaries with category, fact, confidence, and source
    """
    if not text or not text.strip():
        return []
    
    # Step 1: Clean the text
    cleaned_text = clean_profile_text(text)
    
    if not cleaned_text or len(cleaned_text.strip()) < 20:
        return []
    
    client = Groq(api_key=api_key)
    
    system_prompt = """You are a strict evidence-based information extraction system.

Your task: Extract structured professional facts from UNSTRUCTURED profile text.

CRITICAL RULES:
1. ONLY extract facts that are:
   - Complete sentences or clearly structured phrases
   - Professionally relevant (education, experience, skills, impact, publications, roles)
   - Explicitly stated in the text

2. DO NOT:
   - Extract broken fragments (e.g., "ms that are observable")
   - Extract partial substrings
   - Infer missing words
   - Guess abbreviations
   - Expand unclear acronyms
   - Combine unrelated fragments

3. If a phrase looks incomplete, corrupted, or truncated → IGNORE IT.

4. If confidence < 80% → DO NOT extract it.

5. NEVER fabricate.
6. NEVER rewrite the fact.
7. Extract verbatim clean professional facts only.

Extract facts into these categories:
- Education
- Work Experience
- Technical Skills
- AI/ML Experience
- Research & Publications
- Impact Metrics (with numbers only if explicitly stated)
- Certifications
- Leadership / Awards

Each extracted fact must:
- Be standalone and readable
- Be professionally meaningful
- Be grounded in text
- Include confidence score (0.0–1.0)
- Include source snippet (exact quote from text)

Reject any fact that:
- Is shorter than 5 meaningful words
- Contains broken words
- Contains UI noise
- Looks like substring extraction
- Is repeated

OUTPUT FORMAT (STRICT JSON):
{
  "approved_facts": [
    {
      "category": "Work Experience",
      "fact": "Led backend and API initiatives on Azure, improving query performance by 20–30% and reducing release cycles by 25%.",
      "confidence": 0.95,
      "source": "Led backend and API initiatives on Azure..."
    }
  ],
  "rejected_fragments": [
    "ms that are observable",
    "ms across public and private sectors"
  ]
}

If no clean facts found, return empty array.
Do not include commentary. Do not explain. Return JSON only."""

    user_prompt = f"""Extract professional facts from this profile text.

Text:
{cleaned_text}

Return JSON with approved_facts and rejected_fragments."""

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
        
        # Extract approved facts
        approved_facts_raw = result_json.get("approved_facts", [])
        
        # Validate and format facts
        validated_facts = []
        seen_facts = set()
        
        for fact_data in approved_facts_raw:
            fact_text = fact_data.get("fact", "").strip()
            confidence = fact_data.get("confidence", 0.0)
            source = fact_data.get("source", fact_text).strip()
            category = fact_data.get("category", "Other")
            
            # Skip if confidence too low
            if confidence < 0.8:
                continue
            
            # Skip if already seen (duplicate)
            fact_lower = fact_text.lower()
            if fact_lower in seen_facts:
                continue
            
            # Validate fact
            if not is_valid_fact(fact_text):
                continue
            
            # Verify source exists in original text (case-insensitive)
            if source.lower() not in cleaned_text.lower():
                # Try to find similar text
                words = source.split()[:5]  # First 5 words
                search_text = ' '.join(words)
                if search_text.lower() not in cleaned_text.lower():
                    continue  # Skip if source not found
            
            # Find actual source indices
            source_lower = source.lower()
            text_lower = cleaned_text.lower()
            start_idx = text_lower.find(source_lower)
            
            if start_idx == -1:
                # Try with fact text instead
                start_idx = text_lower.find(fact_lower[:50])
                if start_idx != -1:
                    source = cleaned_text[start_idx:start_idx + len(fact_text)]
            
            if start_idx == -1:
                start_idx = 0
                end_idx = len(fact_text)
            else:
                end_idx = start_idx + len(source)
            
            validated_facts.append({
                "value": fact_text,
                "source_quote": source,
                "start_index": start_idx,
                "end_index": end_idx,
                "confidence": confidence,
                "category": category
            })
            
            seen_facts.add(fact_lower)
        
        return validated_facts
    
    except Exception as e:
        # Fallback: simple extraction without LLM
        return extract_simple_facts(cleaned_text)


def extract_simple_facts(text: str) -> List[Dict]:
    """
    Simple rule-based extraction as fallback.
    Only extracts obvious patterns with direct quotes.
    """
    facts = []
    text_lower = text.lower()
    seen = set()
    
    # Education patterns
    education_patterns = [
        r'(?:degree|bachelor|master|phd|doctorate|ms|ma|bs|ba|m\.?s\.?c\.?|b\.?s\.?c\.?)\s+[^.!?]{10,100}',
        r'(?:graduated|studied|attended)\s+[^.!?]{10,100}',
        r'(?:university|college|school)\s+[^.!?]{10,100}',
    ]
    
    for pattern in education_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            quote = match.group(0).strip()
            if is_valid_fact(quote) and quote.lower() not in seen:
                facts.append({
                    "value": quote,
                    "source_quote": quote,
                    "start_index": match.start(),
                    "end_index": match.end(),
                    "confidence": 0.7,
                    "category": "Education"
                })
                seen.add(quote.lower())
    
    # Experience patterns
    experience_patterns = [
        r'(?:worked|employed|interned|served|led|built|developed|created|designed|implemented)\s+[^.!?]{15,150}',
        r'\d+\+?\s*(?:years?|months?)\s+(?:of\s+)?[^.!?]{10,100}',
    ]
    
    for pattern in experience_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            quote = match.group(0).strip()
            if is_valid_fact(quote) and quote.lower() not in seen:
                facts.append({
                    "value": quote,
                    "source_quote": quote,
                    "start_index": match.start(),
                    "end_index": match.end(),
                    "confidence": 0.7,
                    "category": "Work Experience"
                })
                seen.add(quote.lower())
    
    return facts


def extract_structured_profile(form_data: Dict) -> List[Dict]:
    """
    Extract facts from structured form input.
    Each field becomes a fact with the field name as category.
    """
    facts = []
    
    category_mapping = {
        "education": "Education",
        "work_experience": "Work Experience",
        "skills": "Technical Skills"
    }
    
    for key, value in form_data.items():
        if value and isinstance(value, str) and value.strip():
            value_clean = value.strip()
            if is_valid_fact(value_clean):
                facts.append({
                    "value": value_clean,
                    "source_quote": value_clean,
                    "start_index": 0,
                    "end_index": len(value_clean),
                    "confidence": 1.0,
                    "category": category_mapping.get(key, "Other")
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
