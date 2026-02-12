"""
Enhanced validation checks for job outreach messages with STRICT and RELAXED modes.
"""

import re
from typing import Dict, List, Optional, Tuple


def within_word_limit(text: str, max_words: int) -> bool:
    """Check if text is within word limit."""
    if not text:
        return False
    word_count = len(text.split())
    return word_count <= max_words


def must_include_all(text: str, must_include: List[str], strict_mode: bool = False) -> Tuple[bool, List[str]]:
    """
    Check if text contains all required items.
    
    Args:
        text: The message text
        must_include: List of required items
        strict_mode: If True, requires exact phrase match. If False, uses relaxed matching.
        
    Returns:
        Tuple of (bool, list of missing items)
    """
    if not text or not must_include:
        return (len(must_include) == 0, must_include.copy())
    
    text_lower = text.lower()
    missing = []
    
    for item in must_include:
        item_lower = item.lower()
        
        if strict_mode:
            # STRICT: Exact phrase match
            if item_lower == "ask for chat":
                # Must contain literal "chat"
                if "chat" not in text_lower:
                    missing.append(item)
            else:
                # Exact match for GitHub, Portfolio, etc.
                if item_lower not in text_lower:
                    missing.append(item)
        else:
            # RELAXED: Case-insensitive contains with smart matching
            if item_lower == "github":
                if "github" not in text_lower:
                    missing.append(item)
            elif item_lower == "portfolio":
                if "portfolio" not in text_lower:
                    missing.append(item)
            elif item_lower == "ask for chat":
                # Accept: chat, call, connect, schedule, 15-minute, quick conversation
                chat_phrases = ["chat", "call", "connect", "schedule", "15-minute", "quick conversation", "conversation"]
                if not any(phrase in text_lower for phrase in chat_phrases):
                    missing.append(item)
            elif item_lower == "nyu":
                if "nyu" not in text_lower:
                    missing.append(item)
            else:
                # Generic contains check
                if item_lower not in text_lower:
                    missing.append(item)
    
    return (len(missing) == 0, missing)


def tone_professional(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text maintains professional tone.
    
    Returns:
        Tuple of (bool, list of issues found)
    """
    if not text:
        return (False, ["Empty message"])
    
    issues = []
    text_lower = text.lower()
    
    # Check for emojis (any emoji is unprofessional)
    emoji_pattern = r'[😀-🙏🌀-🗿💀-🛿]'
    if re.search(emoji_pattern, text):
        issues.append("Contains emojis")
    
    # Check for slang/casual phrases
    unprofessional_phrases = [
        "yo", "bro", "asap", "pls", "plz", "thx", "hey there", "hey!",
        "sup", "what's up", "lol", "omg", "tbh", "imo", "fyi", "btw",
        "nvm", "idk", "ttyl", "hmu", "fr", "ngl", "gonna", "wanna",
        "gotta", "lemme", "dunno"
    ]
    for phrase in unprofessional_phrases:
        if phrase in text_lower:
            issues.append(f"Contains slang: '{phrase}'")
            break  # Only report first instance
    
    # Check for excessive exclamation marks (more than 2)
    exclamation_count = text.count('!')
    if exclamation_count > 2:
        issues.append(f"Too many exclamation marks ({exclamation_count})")
    
    # Check for all caps (shouting) - more than 3 consecutive caps
    if re.search(r'[A-Z]{4,}', text):
        issues.append("Contains excessive capitalization")
    
    return (len(issues) == 0, issues)


def detects_fabrication(text: str, allowed_facts: List[str]) -> Tuple[bool, List[str]]:
    """
    Detect if text mentions facts not in allowed_facts.
    Flags: new degrees, graduation years, companies, publications, awards.
    
    Returns:
        Tuple of (bool, list of detected fabrications)
    """
    if not text or not allowed_facts:
        return (False, [])
    
    text_lower = text.lower()
    allowed_lower = [fact.lower() for fact in allowed_facts]
    allowed_text = " ".join(allowed_lower)
    
    fabrications = []
    
    # Check for degrees not in allowed facts
    degrees = ["phd", "doctorate", "mba", "bachelor", "bachelors", "master", "masters", "ms", "ma", "bs", "ba"]
    for degree in degrees:
        if degree in text_lower:
            if not any(degree in fact.lower() for fact in allowed_facts):
                # Check if it's part of a larger phrase that IS allowed
                if degree not in allowed_text:
                    fabrications.append(f"Mentions {degree.upper()} not in allowed facts")
                    break
    
    # Check for graduation years (4-digit years 1900-2099)
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    for year in years:
        if not any(year in fact.lower() for fact in allowed_facts):
            fabrications.append(f"Mentions year {year} not in allowed facts")
            break
    
    # Check for "recent graduate" or "graduated" if graduation not specified
    if re.search(r'\b(recent graduate|graduated|alumni)\b', text_lower):
        if not any("graduate" in fact.lower() or "grad" in fact.lower() or "expected" in fact.lower() for fact in allowed_facts):
            fabrications.append("Claims graduation status not specified in allowed facts")
    
    # Check for company/employer names not in allowed facts
    # Common tech companies
    common_companies = [
        "microsoft", "amazon", "apple", "google", "meta", "facebook",
        "netflix", "tesla", "uber", "airbnb", "stripe", "palantir",
        "goldman", "mckinsey", "bain", "bcg", "deloitte", "pwc",
        "gep worldwide", "gep"
    ]
    for company in common_companies:
        if company in text_lower:
            if not any(company in fact.lower() for fact in allowed_facts):
                fabrications.append(f"Mentions company '{company}' not in allowed facts")
                break
    
    # Check for publications/awards not in allowed facts
    pub_indicators = ["published", "publication", "paper", "award", "prize", "honor"]
    for indicator in pub_indicators:
        if indicator in text_lower:
            if not any(indicator in fact.lower() for fact in allowed_facts):
                fabrications.append(f"Claims {indicator} not in allowed facts")
                break
    
    # Check for metrics/percentages not in allowed facts
    percentages = re.findall(r'\d+%', text)
    for pct in percentages:
        if not any(pct in fact for fact in allowed_facts):
            fabrications.append(f"Mentions metric {pct} not in allowed facts")
            break
    
    return (len(fabrications) == 0, fabrications)


def run_checks(
    text: str,
    max_words: int,
    must_include: List[str],
    allowed_facts: List[str],
    tone: str,
    strict_mode: bool = False
) -> Dict:
    """
    Run all checks on a job outreach message.
    
    Args:
        text: The message text
        max_words: Maximum word limit
        must_include: List of required items
        allowed_facts: List of allowed facts
        tone: Expected tone (should be "professional")
        strict_mode: If True, uses strict evaluation mode
        
    Returns:
        Dictionary with check results including failure_reasons
    """
    word_limit_ok = within_word_limit(text, max_words)
    must_include_ok, missing_items = must_include_all(text, must_include, strict_mode)
    tone_ok, tone_issues = tone_professional(text)
    no_fabrication, fabrications = detects_fabrication(text, allowed_facts)
    
    # Overall pass
    overall_pass = word_limit_ok and must_include_ok and tone_ok and no_fabrication
    
    # Build failure reasons
    failure_reasons = []
    if not word_limit_ok:
        word_count = len(text.split()) if text else 0
        failure_reasons.append(f"Word limit exceeded: {word_count} > {max_words}")
    if not must_include_ok:
        failure_reasons.append(f"Missing required items: {', '.join(missing_items)}")
    if not tone_ok:
        failure_reasons.extend(tone_issues)
    if not no_fabrication:
        failure_reasons.extend(fabrications)
    
    return {
        "within_word_limit": word_limit_ok,
        "must_include_ok": must_include_ok,
        "tone_ok": tone_ok,
        "fabrication_detected": not no_fabrication,
        "overall_pass": overall_pass,
        "failure_reasons": failure_reasons,
        "notes": "; ".join(failure_reasons) if failure_reasons else "All checks passed"
    }
