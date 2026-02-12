"""
Profile-Driven Job Outreach LLM Evaluator - Evidence-Based Fact Extraction
STAGE 1: Profile Input & Evidence Extraction
"""

import re
import json
from typing import List, Dict, Set, Optional
from groq import Groq


def sanitize_profile_text(text: str) -> str:
    """
    Clean profile text by removing UI artifacts, duplicates, and noise.
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    seen = set()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove lines shorter than 5 words
        words = line.split()
        if len(words) < 5:
            continue
        
        # Remove lines starting with lowercase fragments
        if re.match(r'^\s*[a-z]+\s+(that|across|and|or|the|a|an)\s+', line, re.IGNORECASE):
            continue
        
        # Remove navigation/UI text
        ui_patterns = [
            r'see more', r'show all', r'view all', r'expand', r'collapse',
            r'click', r'\d+ (likes?|comments?|shares?|impressions?|followers?|connections?)',
            r'• • •', r'\.\.\.',
        ]
        
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in ui_patterns):
            continue
        
        # Remove truncated phrases
        if re.search(r'\s+[a-z]{1,2}\s*$', line):
            continue
        
        # Remove duplicates
        line_lower = line.lower()
        if line_lower in seen:
            continue
        seen.add(line_lower)
        
        # Remove timestamps
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', line):
            continue
        
        cleaned_lines.append(line)
    
    cleaned = '\n'.join(cleaned_lines)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    
    return cleaned.strip()


def extract_facts_with_evidence(
    profile_input: Dict,
    api_key: str,
    model: str = "llama-3.1-8b-instant"
) -> List[Dict]:
    """
    STAGE 1: Extract facts from profile_input with evidence quotes.
    
    Args:
        profile_input: Dict with structured_fields, unstructured_text, links
        api_key: Groq API key
        model: Model name
        
    Returns:
        List of fact dictionaries with fact_text, evidence_quote, evidence_source, confidence, tags
    """
    # Combine all input sources
    all_text_parts = []
    
    # Structured fields
    structured_fields = profile_input.get("structured_fields", {})
    for key, value in structured_fields.items():
        if value and isinstance(value, str):
            all_text_parts.append(f"{key}: {value}")
    
    # Unstructured text
    unstructured = profile_input.get("unstructured_text", "")
    if unstructured:
        sanitized = sanitize_profile_text(unstructured)
        if sanitized:
            all_text_parts.append(sanitized)
    
    # Links
    links = profile_input.get("links", {})
    for link_type, url in links.items():
        if url:
            all_text_parts.append(f"{link_type}: {url}")
    
    combined_text = "\n\n".join(all_text_parts)
    
    if not combined_text or len(combined_text.strip()) < 20:
        return []
    
    client = Groq(api_key=api_key)
    
    system_prompt = """You are a strict evidence-based fact extraction system.

STAGE 1: PROFILE INPUT & EVIDENCE EXTRACTION

Extract FACTS (not keywords) from profile input. Each fact must be a complete claim.

CRITICAL RULES:
1. Extract FACTS, not keywords. A "fact" must be a complete claim that could appear in outreach.
   Good facts:
   - "MSCS candidate at NYU, expected May 2026" (if present)
   - "4+ years backend/software engineering experience at <Company>" (if present)
   - "Built telemetry/evaluation pipelines for LLM systems" (if present)
   - "Based in <City>" (if present)
   - "GitHub: <url>" (if present)
   
   Bad facts (reject):
   - "Microsoft Office", "DBMS", single-word skill spam
   - Fragments like "ms ... ms ..."
   - Incomplete phrases

2. Each fact must include:
   - fact_text: cleaned, one sentence
   - evidence_quote: <= 25 words copied exactly from input
   - evidence_source: "structured_field:<key>", "unstructured_text", or "link:<type>"
   - confidence: 0-100
   - tags: ["education","experience","project","skill","link","location","achievement"]

3. Deduplicate facts and merge near-duplicates.

4. If profile is messy/unstructured:
   - Extract fewer but higher-confidence facts
   - Prefer exact phrases found in text
   - Mark low-confidence items as "Needs approval" or "Ambiguous"

5. NEVER fabricate. NEVER guess. NEVER assume.

Return JSON:
{
  "extracted_facts": [
    {
      "fact_text": "Complete fact statement",
      "evidence_quote": "Exact quote from input (max 25 words)",
      "evidence_source": "structured_field:experience",
      "confidence": 95,
      "tags": ["experience", "achievement"]
    }
  ]
}"""

    user_prompt = f"""Extract professional facts from this profile input:

{combined_text}

Return JSON with extracted_facts array. Only include facts with clear evidence."""

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
        
        facts_raw = result_json.get("extracted_facts", [])
        
        # Validate and format facts
        validated_facts = []
        seen_facts = set()
        
        for fact_data in facts_raw:
            fact_text = fact_data.get("fact_text", "").strip()
            evidence_quote = fact_data.get("evidence_quote", fact_text).strip()
            evidence_source = fact_data.get("evidence_source", "unstructured_text")
            confidence = fact_data.get("confidence", 0)
            tags = fact_data.get("tags", [])
            
            # Skip if confidence too low or fact too short
            if confidence < 80 or len(fact_text) < 10:
                continue
            
            # Skip duplicates
            fact_lower = fact_text.lower()
            if fact_lower in seen_facts:
                continue
            
            # Validate fact is complete
            if not is_complete_fact(fact_text):
                continue
            
            # Find evidence in source text
            if evidence_quote.lower() not in combined_text.lower():
                # Try to find similar text
                words = evidence_quote.split()[:5]
                search_text = ' '.join(words)
                if search_text.lower() not in combined_text.lower():
                    continue  # Skip if evidence not found
            
            validated_facts.append({
                "value": fact_text,
                "source_quote": evidence_quote[:100],  # Limit quote length
                "start_index": 0,  # Will be calculated if needed
                "end_index": len(evidence_quote),
                "confidence": confidence / 100.0,  # Convert to 0-1
                "category": tags[0] if tags else "Other",
                "evidence_source": evidence_source,
                "tags": tags
            })
            
            seen_facts.add(fact_lower)
        
        return validated_facts
    
    except Exception as e:
        # Fallback: simple extraction
        return extract_simple_facts(combined_text)


def is_complete_fact(fact: str) -> bool:
    """Validate if a fact is complete and acceptable."""
    if not fact or len(fact.strip()) < 10:
        return False
    
    words = fact.split()
    if len(words) < 5:
        return False
    
    fact_lower = fact.lower()
    
    # Reject broken fragments
    if re.match(r'^\w+\s+(that|across|and|or)\s+\w+$', fact_lower):
        return False
    
    # Reject UI noise
    ui_noise = ['see more', 'show all', 'expand', 'click', 'view']
    if any(noise in fact_lower for noise in ui_noise):
        return False
    
    # Must be a complete claim
    if not re.search(r'[.!?]|(?:worked|studied|led|built|developed|created|designed|implemented|graduated|earned|completed|based|located)', fact_lower):
        # Check if it's a meaningful phrase
        if not re.search(r'\d+|(?:years?|experience|degree|university|college|company|engineer|developer|scientist)', fact_lower):
            return False
    
    return True


def extract_simple_facts(text: str) -> List[Dict]:
    """Simple rule-based extraction as fallback."""
    facts = []
    seen = set()
    
    # Education patterns
    patterns = [
        (r'(?:degree|bachelor|master|phd|doctorate|ms|ma|bs|ba|m\.?s\.?c\.?|b\.?s\.?c\.?)\s+[^.!?]{10,100}', "Education"),
        (r'(?:graduated|studied|attended)\s+[^.!?]{10,100}', "Education"),
        (r'(?:worked|employed|interned|led|built|developed|created)\s+[^.!?]{15,150}', "Work Experience"),
        (r'\d+\+?\s*(?:years?|months?)\s+(?:of\s+)?[^.!?]{10,100}', "Work Experience"),
    ]
    
    for pattern, category in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            quote = match.group(0).strip()
            if is_complete_fact(quote) and quote.lower() not in seen:
                facts.append({
                    "value": quote,
                    "source_quote": quote,
                    "start_index": match.start(),
                    "end_index": match.end(),
                    "confidence": 0.7,
                    "category": category,
                    "evidence_source": "unstructured_text",
                    "tags": [category.lower().replace(" ", "_")]
                })
                seen.add(quote.lower())
    
    return facts


def extract_structured_profile(form_data: Dict) -> List[Dict]:
    """Extract facts from structured form input."""
    facts = []
    
    category_mapping = {
        "education": "Education",
        "work_experience": "Work Experience",
        "skills": "Technical Skills"
    }
    
    for key, value in form_data.items():
        if value and isinstance(value, str) and value.strip():
            value_clean = value.strip()
            if is_complete_fact(value_clean) or len(value_clean.split()) >= 2:
                facts.append({
                    "value": value_clean,
                    "source_quote": value_clean,
                    "start_index": 0,
                    "end_index": len(value_clean),
                    "confidence": 1.0,
                    "category": category_mapping.get(key, "Other"),
                    "evidence_source": f"structured_field:{key}",
                    "tags": [category_mapping.get(key, "Other").lower().replace(" ", "_")]
                })
    
    return facts


def validate_fact_evidence(fact: Dict, source_text: str) -> bool:
    """Validate that a fact's source_quote exists in source text."""
    source_quote = fact.get("source_quote", "")
    if not source_quote:
        return False
    return source_quote.lower() in source_text.lower()


# Legacy function for backward compatibility
def extract_evidence_based_facts(
    text: str,
    api_key: str,
    model: str = "llama-3.1-8b-instant"
) -> List[Dict]:
    """
    Legacy wrapper for extract_facts_with_evidence.
    Converts simple text input to profile_input format.
    """
    profile_input = {
        "unstructured_text": text,
        "structured_fields": {},
        "links": {}
    }
    return extract_facts_with_evidence(profile_input, api_key, model)
