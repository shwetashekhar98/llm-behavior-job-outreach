"""
Generic validation checks for job outreach messages with STRICT and RELAXED modes.
Production-grade, fully generic evaluation framework.
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
    Check if text contains all required items using abstract keys.
    
    Abstract keys:
    - mention_github: Must mention GitHub
    - mention_portfolio: Must mention Portfolio
    - mention_linkedin: Must mention LinkedIn
    - request_chat: Must request a chat/call/meeting
    - mention_education: Must mention education
    - mention_experience: Must mention experience
    
    Args:
        text: The message text
        must_include: List of abstract keys (e.g., ["mention_github", "request_chat"])
        strict_mode: If True, requires exact phrase match. If False, uses relaxed matching.
        
    Returns:
        Tuple of (bool, list of missing items)
    """
    if not text or not must_include:
        return (len(must_include) == 0, must_include.copy())
    
    text_lower = text.lower()
    missing = []
    
    for key in must_include:
        key_lower = key.lower()
        
        if key_lower == "mention_github":
            if strict_mode:
                if "github" not in text_lower:
                    missing.append("mention_github")
            else:
                if "github" not in text_lower:
                    missing.append("mention_github")
        
        elif key_lower == "mention_portfolio":
            if strict_mode:
                if "portfolio" not in text_lower:
                    missing.append("mention_portfolio")
            else:
                if "portfolio" not in text_lower:
                    missing.append("mention_portfolio")
        
        elif key_lower == "mention_linkedin":
            if strict_mode:
                if "linkedin" not in text_lower:
                    missing.append("mention_linkedin")
            else:
                if "linkedin" not in text_lower:
                    missing.append("mention_linkedin")
        
        elif key_lower == "request_chat":
            if strict_mode:
                # Must contain literal "chat"
                if "chat" not in text_lower:
                    missing.append("request_chat")
            else:
                # Accept: chat, call, connect, schedule, 15-minute, quick conversation
                chat_phrases = ["chat", "call", "connect", "schedule", "15-minute", "quick conversation", "conversation"]
                if not any(phrase in text_lower for phrase in chat_phrases):
                    missing.append("request_chat")
        
        elif key_lower == "mention_education":
            # Check for education-related keywords
            education_keywords = ["education", "degree", "university", "college", "school", "graduate", "studied"]
            if not any(keyword in text_lower for keyword in education_keywords):
                missing.append("mention_education")
        
        elif key_lower == "mention_experience":
            # Check for experience-related keywords
            experience_keywords = ["experience", "worked", "intern", "internship", "role", "position", "job"]
            if not any(keyword in text_lower for keyword in experience_keywords):
                missing.append("mention_experience")
        
        else:
            # Generic key - treat as literal string
            if key_lower not in text_lower:
                missing.append(key)
    
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
    # Ensure "you" doesn't trigger "yo" - use word boundary
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
    Generic fabrication detection using pattern matching.
    Flags fabrication ONLY when message asserts employment/affiliation using specific patterns.
    Does NOT flag target company, role mentions, or expressions of interest.
    
    Returns:
        Tuple of (bool, list of detected fabrications)
    """
    if not text or not allowed_facts:
        return (False, [])
    
    text_lower = text.lower()
    allowed_lower = [fact.lower() for fact in allowed_facts]
    allowed_text = " ".join(allowed_lower)
    target_company_lower = company.lower() if company else ""
    
    fabrications = []
    
    # 1. Degree detection - Flag PhD, MBA, BA, Bachelor ONLY if not in allowed_facts
    # Do NOT flag MS / Master's generically
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
    
    # 2. Year detection - Fix regex using non-capturing group
    year_pattern = r'\b(?:19|20)\d{2}\b'
    years = re.findall(year_pattern, text)
    for year in years:
        if not any(year in fact for fact in allowed_facts):
            fabrications.append(f"Graduation year not allowed: {year}")
            break
    
    # 3. Employment/affiliation detection - Use pattern matching
    # Flag ONLY when asserting employment/affiliation, NOT when expressing interest
    # Use case-insensitive patterns
    employment_patterns = [
        (r'\b(worked at|worked for)\s+([A-Za-z][A-Za-z\s&]+?)(?:\s|,|\.|$)', 2),
        (r'\b(at|for)\s+([A-Za-z][A-Za-z\s&]+?)\s+I\b', 2),
        (r'\b(previously at|previously with)\s+([A-Za-z][A-Za-z\s&]+?)(?:\s|,|\.|$)', 2),
        (r'\b(interned at|interned with)\s+([A-Za-z][A-Za-z\s&]+?)(?:\s|,|\.|$)', 2),
        (r'\b(employed at|employed by)\s+([A-Za-z][A-Za-z\s&]+?)(?:\s|,|\.|$)', 2),
        (r'\b(joined|served at)\s+([A-Za-z][A-Za-z\s&]+?)(?:\s|,|\.|$)', 2),
        (r'\bas\s+(?:a|an)\s+[^,]+?\s+(?:at|with)\s+([A-Za-z][A-Za-z\s&]+?)(?:\s|,|\.|$)', 1),
    ]
    
    # Extract company names from employment patterns
    found_companies = set()
    for pattern, group_num in employment_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if match.lastindex and match.lastindex >= group_num:
                company_name = match.group(group_num)
                if company_name:
                    # Clean up company name
                    company_name = company_name.strip()
                    # Skip if it's too short or common words
                    if len(company_name) >= 3:
                        found_companies.add(company_name)
    
    # Check each found company
    for found_company in found_companies:
        found_company_lower = found_company.lower()
        
        # Skip target company
        if found_company_lower == target_company_lower:
            continue
        
        # Skip if it's in allowed facts
        if any(found_company_lower in fact.lower() for fact in allowed_facts):
            continue
        
        # Skip common words that aren't companies
        skip_words = ["the", "a", "an", "this", "that", "my", "your", "our", "their"]
        if found_company_lower in skip_words or len(found_company_lower) < 3:
            continue
        
        # Flag as fabrication
        fabrications.append(f"New employer not allowed: {found_company.title()}")
        break  # Only flag first instance
    
    # 4. Publications/awards detection
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
        must_include: List of abstract keys (e.g., ["mention_github", "request_chat"])
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
            if item == "mention_portfolio":
                failure_reasons.append("Missing Portfolio mention")
            elif item == "mention_github":
                failure_reasons.append("Missing GitHub mention")
            elif item == "mention_linkedin":
                failure_reasons.append("Missing LinkedIn mention")
            elif item == "request_chat":
                failure_reasons.append("Missing chat request")
            elif item == "mention_education":
                failure_reasons.append("Missing education mention")
            elif item == "mention_experience":
                failure_reasons.append("Missing experience mention")
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
