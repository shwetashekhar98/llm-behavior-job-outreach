"""
Simple validation checks for job outreach messages.
"""

import re
from typing import Dict, List


def within_word_limit(text: str, max_words: int) -> bool:
    """
    Check if text is within word limit.
    
    Args:
        text: The message text
        max_words: Maximum allowed words
        
    Returns:
        True if within limit, False otherwise
    """
    if not text:
        return False
    
    word_count = len(text.split())
    return word_count <= max_words


def must_include_all(text: str, must_include: List[str]) -> bool:
    """
    Check if text contains all required items (case-insensitive).
    
    Args:
        text: The message text
        must_include: List of required items
        
    Returns:
        True if all items are present, False otherwise
    """
    if not text or not must_include:
        return len(must_include) == 0
    
    text_lower = text.lower()
    for item in must_include:
        if item.lower() not in text_lower:
            return False
    return True


def adds_new_facts(text: str, allowed_facts: List[str]) -> bool:
    """
    Check if text mentions facts not in allowed_facts.
    Heuristic: looks for employers, years, metrics, company names not in allowed list.
    
    Args:
        text: The message text
        allowed_facts: List of allowed facts
        
    Returns:
        True if new facts detected, False otherwise
    """
    if not text or not allowed_facts:
        return False
    
    text_lower = text.lower()
    allowed_lower = [fact.lower() for fact in allowed_facts]
    
    # Check for years (4-digit numbers) not in allowed facts
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    for year in years:
        if not any(year in fact.lower() for fact in allowed_facts):
            return True
    
    # Check for company names (common tech companies) not in allowed facts
    common_companies = ["microsoft", "amazon", "apple", "google", "meta", "facebook", 
                       "netflix", "tesla", "uber", "airbnb", "stripe", "palantir",
                       "goldman", "mckinsey", "bain", "bcg", "deloitte", "pwc"]
    for company in common_companies:
        if company in text_lower:
            if not any(company in fact.lower() for fact in allowed_facts):
                return True
    
    # Check for metrics/numbers (percentages, large numbers) not in allowed facts
    percentages = re.findall(r'\d+%', text)
    for pct in percentages:
        if not any(pct in fact for fact in allowed_facts):
            return True
    
    # Check for job titles/roles not in allowed facts
    # This is conservative - only flag if it's clearly a different role
    job_titles = ["ceo", "cto", "director", "manager", "senior", "junior", "lead", "principal"]
    for title in job_titles:
        # Only flag if it's a standalone mention, not part of allowed facts
        pattern = r'\b' + re.escape(title) + r'\b'
        if re.search(pattern, text_lower, re.IGNORECASE):
            if not any(title in fact.lower() for fact in allowed_facts):
                # Be conservative - don't flag common words
                if title in ["senior", "junior", "lead"]:
                    continue
                return True
    
    return False


def tone_professional(text: str) -> bool:
    """
    Check if text maintains professional tone.
    Heuristic: avoid slang, excessive emojis, overly casual phrases.
    
    Args:
        text: The message text
        
    Returns:
        True if professional, False otherwise
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Check for excessive emojis (more than 1-2 is unprofessional)
    emoji_count = len(re.findall(r'[😀-🙏🌀-🗿]', text))
    if emoji_count > 2:
        return False
    
    # Check for slang/casual phrases
    unprofessional_phrases = [
        "hey there", "hey!", "yo", "sup", "what's up", "lol", "omg", "tbh",
        "imo", "fyi", "btw", "nvm", "idk", "ttyl", "hmu", "fr", "ngl"
    ]
    for phrase in unprofessional_phrases:
        if phrase in text_lower:
            return False
    
    # Check for excessive exclamation marks (more than 2)
    exclamation_count = text.count('!')
    if exclamation_count > 2:
        return False
    
    # Check for all caps (shouting)
    if len(re.findall(r'\b[A-Z]{3,}\b', text)) > 2:
        return False
    
    # Check for casual contractions in professional context
    very_casual = ["gonna", "wanna", "gotta", "lemme", "dunno"]
    for word in very_casual:
        if word in text_lower:
            return False
    
    return True


def run_checks(text: str, max_words: int, must_include: List[str], 
               allowed_facts: List[str], tone: str) -> Dict:
    """
    Run all checks on a job outreach message.
    
    Args:
        text: The message text
        max_words: Maximum word limit
        must_include: List of required items
        allowed_facts: List of allowed facts
        tone: Expected tone (should be "professional")
        
    Returns:
        Dictionary with check results
    """
    word_limit_ok = within_word_limit(text, max_words)
    must_include_ok = must_include_all(text, must_include)
    new_facts = adds_new_facts(text, allowed_facts)
    tone_ok = tone_professional(text) if tone == "professional" else True
    
    overall_pass = word_limit_ok and must_include_ok and tone_ok and not new_facts
    
    notes = []
    if not word_limit_ok:
        word_count = len(text.split()) if text else 0
        notes.append(f"Word limit: {word_count} > {max_words}")
    if not must_include_ok:
        missing = [item for item in must_include if item.lower() not in text.lower()]
        notes.append(f"Missing: {', '.join(missing)}")
    if new_facts:
        notes.append("New facts detected")
    if not tone_ok:
        notes.append("Tone not professional")
    
    return {
        "within_word_limit": word_limit_ok,
        "must_include_ok": must_include_ok,
        "adds_new_facts": new_facts,
        "tone_ok": tone_ok,
        "overall_pass": overall_pass,
        "notes": "; ".join(notes) if notes else "All checks passed"
    }

