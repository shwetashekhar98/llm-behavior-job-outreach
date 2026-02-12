"""
Validation engine implementing STRICT and RELAXED evaluation modes.
PHASE 4: Evaluation with fabrication and unsupported claims detection.
"""

import re
from typing import Dict, List, Tuple


def within_word_limit(text: str, max_words: int) -> bool:
    """Check if text is within word limit."""
    if not text:
        return False
    word_count = len(text.split())
    return word_count <= max_words


def must_include_check(text: str, must_include: List[str], strict_mode: bool = False) -> Tuple[bool, List[str]]:
    """
    Check if text contains all required items.
    
    STRICT mode: Exact trigger phrases required
    RELAXED mode: Smart matching with synonyms
    """
    if not text or not must_include:
        return (len(must_include) == 0, must_include.copy())
    
    text_lower = text.lower()
    missing = []
    
    for item in must_include:
        item_lower = item.lower()
        
        if strict_mode:
            # STRICT: Exact phrase match
            if item_lower == "ask for chat" or item_lower == "request chat":
                # Must contain explicit ask + timebox OR unambiguous scheduling ask
                has_explicit_ask = any(phrase in text_lower for phrase in [
                    "15-minute chat", "quick call", "15 minute", "schedule a call",
                    "schedule a chat", "would you be open", "are you available"
                ])
                if not has_explicit_ask:
                    missing.append(item)
            elif item_lower not in text_lower:
                missing.append(item)
        else:
            # RELAXED: Smart matching
            if item_lower in ["github", "portfolio", "linkedin"]:
                if item_lower not in text_lower:
                    missing.append(item)
            elif item_lower in ["ask for chat", "request chat", "chat request"]:
                # Accept: chat, call, connect, schedule, 15-minute, quick conversation
                chat_phrases = [
                    "chat", "call", "connect", "schedule", "15-minute", 
                    "quick conversation", "conversation", "schedule time"
                ]
                if not any(phrase in text_lower for phrase in chat_phrases):
                    missing.append(item)
            else:
                if item_lower not in text_lower:
                    missing.append(item)
    
    return (len(missing) == 0, missing)


def tone_professional(text: str, strict_mode: bool = False) -> Tuple[bool, List[str]]:
    """
    Check if text maintains professional tone.
    
    STRICT: Hard fails on any issues
    RELAXED: Minor issues become warnings
    """
    if not text:
        return (False, ["Empty message"])
    
    issues = []
    text_lower = text.lower()
    
    # Check for emojis (always fail)
    emoji_pattern = r'[😀-🙏🌀-🗿💀-🛿]'
    if re.search(emoji_pattern, text):
        issues.append("Contains emojis")
    
    # Check for slang using word boundaries
    slang_pattern = r'\b(yo|bro|asap|pls|thx|lol)\b'
    slang_match = re.search(slang_pattern, text_lower, re.IGNORECASE)
    if slang_match:
        detected_slang = slang_match.group(1)
        issues.append(f"Slang detected: {detected_slang}")
    
    # Check for excessive exclamation marks
    exclamation_count = text.count('!')
    if exclamation_count > 2:
        if strict_mode:
            issues.append(f"Too many exclamation marks ({exclamation_count})")
        else:
            # RELAXED: warning only if > 3
            if exclamation_count > 3:
                issues.append(f"Too many exclamation marks ({exclamation_count})")
    
    return (len(issues) == 0, issues)


def detects_fabrication(
    text: str,
    allowed_facts: List[str],
    company: str = "",
    target_role: str = ""
) -> Tuple[bool, List[str]]:
    """
    Detect FABRICATION: mentions specific facts not in allowed_facts.
    
    Examples:
    - "GPA 4.0" when not in approved facts → FABRICATION
    - "worked at Microsoft" when Microsoft not in approved facts → FABRICATION
    - "PhD in CS" when PhD not in approved facts → FABRICATION
    """
    if not text or not allowed_facts:
        return (False, [])
    
    text_lower = text.lower()
    target_company_lower = company.lower() if company else ""
    fabrications = []
    
    # Degree detection
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
    
    # Year detection
    year_pattern = r'\b(?:19|20)\d{2}\b'
    years = re.findall(year_pattern, text)
    for year in years:
        if not any(year in fact for fact in allowed_facts):
            fabrications.append(f"Fabricated year: {year}")
            break
    
    # GPA/metrics detection
    gpa_pattern = r'\b(gpa|grade point average)\s*[:\-]?\s*([0-4]\.?\d*)'
    gpa_match = re.search(gpa_pattern, text_lower, re.IGNORECASE)
    if gpa_match:
        gpa_value = gpa_match.group(2)
        if not any(gpa_value in fact.lower() or "gpa" in fact.lower() for fact in allowed_facts):
            fabrications.append(f"Fabricated GPA: {gpa_value}")
    
    # Employment detection
    employment_patterns = [
        (r'\b(worked at|worked for)\s+([A-Za-z][A-Za-z\s&]+?)(?:\s|,|\.|$)', 2),
        (r'\b(previously at|previously with)\s+([A-Za-z][A-Za-z\s&]+?)(?:\s|,|\.|$)', 2),
        (r'\b(interned at|interned with)\s+([A-Za-z][A-Za-z\s&]+?)(?:\s|,|\.|$)', 2),
        (r'\b(employed at|employed by)\s+([A-Za-z][A-Za-z\s&]+?)(?:\s|,|\.|$)', 2),
        (r'\b(joined|served at)\s+([A-Za-z][A-Za-z\s&]+?)(?:\s|,|\.|$)', 2),
    ]
    
    found_companies = set()
    for pattern, group_num in employment_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if match.lastindex and match.lastindex >= group_num:
                company_name = match.group(group_num).strip()
                if len(company_name) >= 3 and company_name.lower() != target_company_lower:
                    found_companies.add(company_name)
    
    for found_company in found_companies:
        if not any(found_company.lower() in fact.lower() for fact in allowed_facts):
            fabrications.append(f"Fabricated employer: {found_company.title()}")
            break
    
    # Publication/award detection
    pub_indicators = ["published", "publication", "paper", "award", "prize", "honor"]
    for indicator in pub_indicators:
        if indicator in text_lower:
            if not any(indicator in fact.lower() for fact in allowed_facts):
                fabrications.append(f"Fabricated {indicator}")
                break
    
    return (len(fabrications) == 0, fabrications)


def detects_unsupported_claims(
    text: str,
    allowed_facts: List[str],
    strict_mode: bool = False
) -> Tuple[bool, List[str]]:
    """
    Detect UNSUPPORTED CLAIMS: vague but risky claims not grounded in facts.
    
    Examples:
    - "measurable impact" without specific metrics in facts → UNSUPPORTED (if implies outcomes)
    - "improved efficiency" without numbers → UNSUPPORTED (if specific)
    - "top X%" → UNSUPPORTED if not in facts
    
    STRICT mode: Fail on specific unsupported claims
    RELAXED mode: Warn only on very specific claims
    """
    if not text or not allowed_facts:
        return (True, [])
    
    text_lower = text.lower()
    claims = []
    
    # Check for specific metrics/percentages not in facts
    metric_patterns = [
        r'\b(\d+%)\s+(?:improvement|increase|decrease|reduction|growth)',
        r'\b(?:improved|increased|reduced|decreased|cut)\s+by\s+(\d+%?)',
        r'\b(?:top|top\s+of)\s+(\d+)%',
        r'\b(\d+)\s+(?:times|fold)\s+(?:faster|better)',
    ]
    
    for pattern in metric_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            metric_text = match.group(0)
            # Check if this metric appears in allowed facts
            if not any(metric_text.lower() in fact.lower() for fact in allowed_facts):
                if strict_mode:
                    claims.append(f"Unsupported metric claim: {metric_text}")
                else:
                    # RELAXED: only flag if very specific
                    if re.search(r'\d+%', metric_text):
                        claims.append(f"Unsupported metric claim: {metric_text}")
                break
    
    # Check for vague impact claims (only in STRICT mode)
    if strict_mode:
        vague_impact = [
            "significant impact", "measurable impact", "substantial improvement",
            "dramatic increase", "major improvement"
        ]
        for phrase in vague_impact:
            if phrase in text_lower:
                # Check if there's a specific metric nearby
                context = text_lower[max(0, text_lower.find(phrase) - 50):text_lower.find(phrase) + 50]
                if not re.search(r'\d+%?', context):
                    claims.append(f"Unsupported vague claim: {phrase}")
                    break
    
    return (len(claims) == 0, claims)


def run_all_checks(
    text: str,
    max_words: int,
    must_include: List[str],
    allowed_facts: List[str],
    strict_mode: bool = False,
    company: str = "",
    target_role: str = ""
) -> Dict:
    """
    PHASE 4: Evaluation
    Run all validation checks with STRICT or RELAXED mode.
    
    Returns:
        Dictionary with check results and failure_reasons
    """
    word_limit_ok = within_word_limit(text, max_words)
    must_include_ok, missing_items = must_include_check(text, must_include, strict_mode)
    tone_ok, tone_issues = tone_professional(text, strict_mode)
    no_fabrication, fabrications = detects_fabrication(text, allowed_facts, company, target_role)
    no_unsupported, unsupported = detects_unsupported_claims(text, allowed_facts, strict_mode)
    
    overall_pass = (word_limit_ok and must_include_ok and tone_ok and 
                   no_fabrication and no_unsupported)
    
    # Build failure reasons
    failure_reasons = []
    if not word_limit_ok:
        word_count = len(text.split()) if text else 0
        failure_reasons.append(f"Word limit exceeded: {word_count} > {max_words}")
    if not must_include_ok:
        for item in missing_items:
            failure_reasons.append(f"Missing: {item}")
    if not tone_ok:
        failure_reasons.extend(tone_issues)
    if not no_fabrication:
        failure_reasons.extend(fabrications)
    if not no_unsupported:
        failure_reasons.extend(unsupported)
    
    return {
        "within_word_limit": word_limit_ok,
        "must_include_ok": must_include_ok,
        "tone_ok": tone_ok,
        "fabrication_detected": not no_fabrication,
        "unsupported_claims_detected": not no_unsupported,
        "overall_pass": overall_pass,
        "failure_reasons": failure_reasons,
        "notes": "; ".join(failure_reasons) if failure_reasons else "All checks passed"
    }
