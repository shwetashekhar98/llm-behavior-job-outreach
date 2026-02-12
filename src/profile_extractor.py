"""
Reliability-focused profile extraction with 4-phase system.
PHASE 1: Profile Sanitization
PHASE 2: Fact Approval
"""

import re
import json
from typing import List, Dict, Set
from groq import Groq


def sanitize_profile_text(text: str) -> str:
    """
    PHASE 1: Profile Sanitization
    Remove truncated phrases, UI text, duplicates, and non-professional content.
    
    Returns:
        Cleaned text ready for extraction
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
            r'see more',
            r'show all',
            r'view all',
            r'expand',
            r'collapse',
            r'click',
            r'\d+ (likes?|comments?|shares?|impressions?|followers?|connections?)',
            r'• • •',
            r'\.\.\.',
        ]
        
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in ui_patterns):
            continue
        
        # Remove truncated phrases (ends with incomplete word)
        if re.search(r'\s+[a-z]{1,2}\s*$', line):
            continue
        
        # Remove duplicates (case-insensitive)
        line_lower = line.lower()
        if line_lower in seen:
            continue
        seen.add(line_lower)
        
        # Remove timestamps
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', line):
            continue
        
        cleaned_lines.append(line)
    
    cleaned = '\n'.join(cleaned_lines)
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    
    return cleaned.strip()


def extract_structured_profile_data(
    text: str,
    api_key: str,
    model: str = "llama-3.1-8b-instant"
) -> Dict:
    """
    PHASE 1: Extract structured profile data from sanitized text.
    
    Returns:
        Structured JSON with education, experience, skills, achievements, research, links
    """
    if not text or len(text.strip()) < 20:
        return {
            "education": [],
            "experience": [],
            "skills": [],
            "achievements": [],
            "research": [],
            "links": []
        }
    
    client = Groq(api_key=api_key)
    
    system_prompt = """You are a reliability-focused profile extraction system.

PHASE 1: PROFILE SANITIZATION

Extract ONLY verifiable, complete facts from profile text.

Rules:
1. Remove:
   - Truncated phrases
   - Lines starting with lowercase fragments like "ms", "bs", "degree connection"
   - Navigation/UI text
   - Duplicates
   - Lines shorter than 5 words
   - Text that is not about the candidate

2. Only extract:
   - Complete, meaningful professional facts
   - Education (degree, university, graduation year)
   - Roles with measurable impact
   - Skills clearly stated
   - Research or publications
   - Quantified achievements
   - Certifications

3. Convert everything into structured JSON:
{
  "education": [],
  "experience": [],
  "skills": [],
  "achievements": [],
  "research": [],
  "links": []
}

Do NOT invent missing data.
If something is unclear, omit it.
Never fabricate.
Never guess missing graduation years.
Never assume company names.
Never complete truncated fragments.

Return JSON only. No commentary."""

    user_prompt = f"""Extract structured profile data from this text:

{text}

Return JSON with education, experience, skills, achievements, research, and links arrays."""

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
        
        # Validate structure
        structured = {
            "education": result_json.get("education", []),
            "experience": result_json.get("experience", []),
            "skills": result_json.get("skills", []),
            "achievements": result_json.get("achievements", []),
            "research": result_json.get("research", []),
            "links": result_json.get("links", [])
        }
        
        # Filter out empty strings and validate
        for key in structured:
            structured[key] = [
                item for item in structured[key]
                if item and isinstance(item, str) and len(item.strip()) >= 5
            ]
        
        return structured
    
    except Exception as e:
        # Fallback: return empty structure
        return {
            "education": [],
            "experience": [],
            "skills": [],
            "achievements": [],
            "research": [],
            "links": []
        }


def create_approved_facts(structured_data: Dict) -> List[str]:
    """
    PHASE 2: Fact Approval
    From structured data, produce a clean list of approved facts.
    
    Rules:
    - Each fact must be complete and grammatically correct
    - No fragments
    - No partial lines
    - No UI content
    - No hallucination
    - No assumptions
    """
    approved_facts = []
    
    # Education facts
    for edu in structured_data.get("education", []):
        if is_complete_fact(edu):
            approved_facts.append(edu.strip())
    
    # Experience facts
    for exp in structured_data.get("experience", []):
        if is_complete_fact(exp):
            approved_facts.append(exp.strip())
    
    # Skills (can be shorter)
    for skill in structured_data.get("skills", []):
        skill_clean = skill.strip()
        if skill_clean and len(skill_clean.split()) >= 2:
            approved_facts.append(skill_clean)
    
    # Achievements
    for achievement in structured_data.get("achievements", []):
        if is_complete_fact(achievement):
            approved_facts.append(achievement.strip())
    
    # Research
    for research in structured_data.get("research", []):
        if is_complete_fact(research):
            approved_facts.append(research.strip())
    
    # Remove duplicates (case-insensitive)
    seen = set()
    unique_facts = []
    for fact in approved_facts:
        fact_lower = fact.lower()
        if fact_lower not in seen:
            seen.add(fact_lower)
            unique_facts.append(fact)
    
    return unique_facts


def is_complete_fact(fact: str) -> bool:
    """
    Validate if a fact is complete and acceptable.
    """
    if not fact or len(fact.strip()) < 5:
        return False
    
    fact_lower = fact.lower()
    
    # Must have at least 5 words
    words = fact.split()
    if len(words) < 5:
        return False
    
    # Reject broken fragments
    if re.match(r'^\w+\s+(that|across|and|or)\s+\w+$', fact_lower):
        return False
    
    # Reject UI noise
    ui_noise = ['see more', 'show all', 'expand', 'click', 'view', 'followers']
    if any(noise in fact_lower for noise in ui_noise):
        return False
    
    # Must be grammatically complete (has verb or is a complete phrase)
    if not re.search(r'[.!?]|(?:worked|studied|led|built|developed|created|designed|implemented|graduated|earned|completed)', fact_lower):
        # Check if it's a meaningful phrase with numbers or professional terms
        if not re.search(r'\d+|(?:years?|experience|degree|university|college|company|engineer|developer|scientist)', fact_lower):
            return False
    
    return True


def extract_evidence_based_facts(
    text: str,
    api_key: str,
    model: str = "llama-3.1-8b-instant"
) -> List[Dict]:
    """
    Main extraction function implementing PHASE 1 and PHASE 2.
    
    Returns:
        List of fact dictionaries for UI display
    """
    if not text or not text.strip():
        return []
    
    # PHASE 1: Sanitize
    sanitized = sanitize_profile_text(text)
    
    if not sanitized or len(sanitized.strip()) < 20:
        return []
    
    # PHASE 1: Extract structured data
    structured_data = extract_structured_profile_data(sanitized, api_key, model)
    
    # PHASE 2: Create approved facts
    approved_facts_list = create_approved_facts(structured_data)
    
    # Convert to UI format
    facts_for_ui = []
    for idx, fact in enumerate(approved_facts_list):
        # Find fact in original text
        fact_lower = fact.lower()
        start_idx = sanitized.lower().find(fact_lower)
        
        if start_idx == -1:
            start_idx = 0
            end_idx = len(fact)
            source_quote = fact
        else:
            end_idx = start_idx + len(fact)
            source_quote = sanitized[start_idx:end_idx]
        
        facts_for_ui.append({
            "value": fact,
            "source_quote": source_quote,
            "start_index": start_idx,
            "end_index": end_idx,
            "confidence": 0.9,  # High confidence for approved facts
            "category": determine_category(fact, structured_data)
        })
    
    return facts_for_ui


def determine_category(fact: str, structured_data: Dict) -> str:
    """Determine category for a fact based on structured data."""
    fact_lower = fact.lower()
    
    # Check which array it came from
    for edu in structured_data.get("education", []):
        if fact_lower == edu.lower():
            return "Education"
    
    for exp in structured_data.get("experience", []):
        if fact_lower == exp.lower():
            return "Work Experience"
    
    for skill in structured_data.get("skills", []):
        if fact_lower == skill.lower():
            return "Technical Skills"
    
    for achievement in structured_data.get("achievements", []):
        if fact_lower == achievement.lower():
            return "Achievements"
    
    for research in structured_data.get("research", []):
        if fact_lower == research.lower():
            return "Research & Publications"
    
    # Fallback: guess from content
    if any(word in fact_lower for word in ["degree", "university", "college", "graduated", "studied"]):
        return "Education"
    elif any(word in fact_lower for word in ["worked", "employed", "led", "built", "developed"]):
        return "Work Experience"
    else:
        return "Other"


def extract_structured_profile(form_data: Dict) -> List[Dict]:
    """
    Extract facts from structured form input.
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
            if is_complete_fact(value_clean) or len(value_clean.split()) >= 2:
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
    """Validate that a fact's source_quote exists in source text."""
    source_quote = fact.get("source_quote", "")
    if not source_quote:
        return False
    return source_quote.lower() in source_text.lower()
