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
    Fails ONLY if: contains emoji, contains slang, or has more than 2 exclamation marks.
    
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
    
    # Check for slang using regex with word boundaries (case insensitive)
    slang_pattern = r'\b(yo|bro|asap|pls|thx|lol)\b'
    slang_match = re.search(slang_pattern, text_lower, re.IGNORECASE)
    if slang_match:
        detected_slang = slang_match.group(1)
        issues.append(f"Slang detected: {detected_slang}")
    
    # Check for excessive exclamation marks (more than 2)
    exclamation_count = text.count('!')
    if exclamation_count > 2:
        issues.append(f"Too many exclamation marks ({exclamation_count})")
    
    return (len(issues) == 0, issues)


def detects_fabrication(
    text: str, 
    allowed_facts: List[str],
    company: str = "",
    target_role: str = "",
    recipient_type: str = "",
    channel: str = ""
) -> Tuple[bool, List[str]]:
    """
    Detect if text mentions facts not in allowed_facts.
    Automatically allows: company, target_role, recipient_type, channel.
    Only flags: PhD/MBA/BA not in allowed, graduation year not in allowed, 
    new employer not in allowed, publications if not allowed.
    Does NOT flag MS, MSCS, Master's.
    
    Returns:
        Tuple of (bool, list of detected fabrications)
    """
    if not text or not allowed_facts:
        return (False, [])
    
    text_lower = text.lower()
    allowed_lower = [fact.lower() for fact in allowed_facts]
    allowed_text = " ".join(allowed_lower)
    
    # Automatically allowed items (don't flag these)
    auto_allowed = []
    if company:
        auto_allowed.append(company.lower())
    if target_role:
        auto_allowed.append(target_role.lower())
    if recipient_type:
        auto_allowed.append(recipient_type.lower())
    if channel:
        auto_allowed.append(channel.lower())
    
    fabrications = []
    
    # Check for degrees NOT in allowed facts (only flag PhD, MBA, BA - NOT MS, MSCS, Master's)
    # Use word boundaries to avoid substring matches
    phd_pattern = r'\b(ph\.?d\.?|doctorate)\b'
    mba_pattern = r'\b(m\.?b\.?a\.?)\b'
    ba_pattern = r'\b(b\.?a\.?|bachelor)\b'
    
    if re.search(phd_pattern, text_lower, re.IGNORECASE):
        if not any("phd" in fact.lower() or "doctorate" in fact.lower() for fact in allowed_facts):
            fabrications.append("Fabricated degree: PhD")
    
    if re.search(mba_pattern, text_lower, re.IGNORECASE):
        if not any("mba" in fact.lower() for fact in allowed_facts):
            fabrications.append("Fabricated degree: MBA")
    
    if re.search(ba_pattern, text_lower, re.IGNORECASE):
        if not any("ba" in fact.lower() or "bachelor" in fact.lower() for fact in allowed_facts):
            fabrications.append("Fabricated degree: BA")
    
    # Check for graduation years (4-digit years 1900-2099) not in allowed facts
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    for year in years:
        if not any(year in fact.lower() for fact in allowed_facts):
            fabrications.append(f"Graduation year not allowed: {year}")
            break
    
    # Check for employer/company names not in allowed facts (excluding target company)
    # Common tech companies
    common_companies = [
        "microsoft", "amazon", "apple", "google", "meta", "facebook",
        "netflix", "tesla", "uber", "airbnb", "stripe", "palantir",
        "goldman", "mckinsey", "bain", "bcg", "deloitte", "pwc",
        "gep worldwide", "gep", "anthropic", "openai", "cohere"
    ]
    target_company_lower = company.lower() if company else ""
    for comp in common_companies:
        # Skip if it's the target company (automatically allowed)
        if comp == target_company_lower:
            continue
        if comp in text_lower:
            if not any(comp in fact.lower() for fact in allowed_facts):
                fabrications.append(f"New employer not allowed: {comp.title()}")
                break
    
    # Check for publications/awards not in allowed facts
    pub_indicators = ["published", "publication", "paper", "award", "prize", "honor"]
    for indicator in pub_indicators:
        if indicator in text_lower:
            if not any(indicator in fact.lower() for fact in allowed_facts):
                fabrications.append(f"Publications not allowed: {indicator}")
                break
    
    return (len(fabrications) == 0, fabrications)


def run_checks(
    text: str,
    max_words: int,
    must_include: List[str],
    allowed_facts: List[str],
    tone: str,
    strict_mode: bool = False,
    company: str = "",
    target_role: str = "",
    recipient_type: str = "",
    channel: str = ""
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
        company: Target company (automatically allowed, not flagged as fabrication)
        target_role: Target role (automatically allowed)
        recipient_type: Recipient type (automatically allowed)
        channel: Channel type (automatically allowed)
        
    Returns:
        Dictionary with check results including failure_reasons
    """
    word_limit_ok = within_word_limit(text, max_words)
    must_include_ok, missing_items = must_include_all(text, must_include, strict_mode)
    tone_ok, tone_issues = tone_professional(text)
    no_fabrication, fabrications = detects_fabrication(
        text, allowed_facts, company, target_role, recipient_type, channel
    )
    
    # Overall pass
    overall_pass = word_limit_ok and must_include_ok and tone_ok and no_fabrication
    
    # Build specific failure reasons
    failure_reasons = []
    if not word_limit_ok:
        word_count = len(text.split()) if text else 0
        failure_reasons.append(f"Word limit exceeded: {word_count} > {max_words}")
    if not must_include_ok:
        for item in missing_items:
            if item.lower() == "portfolio":
                failure_reasons.append("Missing Portfolio mention")
            elif item.lower() == "github":
                failure_reasons.append("Missing GitHub mention")
            elif item.lower() == "ask for chat":
                failure_reasons.append("Missing chat request")
            elif item.lower() == "nyu":
                failure_reasons.append("Missing NYU mention")
            else:
                failure_reasons.append(f"Missing {item}")
    if not tone_ok:
        for issue in tone_issues:
            if "Slang detected" in issue:
                failure_reasons.append(issue)
            elif "emojis" in issue.lower():
                failure_reasons.append("Contains emojis")
            elif "exclamation" in issue.lower():
                failure_reasons.append(issue)
            else:
                failure_reasons.append(issue)
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
