"""
Standalone test for is_complete_fact function (no imports needed).
Tests the logic directly.
"""

import re

def is_complete_fact(fact: str, category: str = "other", debug: bool = False) -> bool:
    """Same implementation as in profile_extractor.py"""
    debug_reasons = []
    
    if not fact:
        if debug:
            print(f"[is_complete_fact DEBUG] REJECT: fact is empty/None")
        return False
    
    fact = fact.strip()
    if not fact:
        if debug:
            print(f"[is_complete_fact DEBUG] REJECT: fact is empty after strip")
        return False
    
    fact_lower = fact.lower()
    words = fact.split()
    word_count = len(words)
    
    # Links category: always accept if evidence contains URL
    if category == "links":
        if re.search(r'https?://', fact_lower):
            if debug:
                print(f"[is_complete_fact DEBUG] ACCEPT: category='links' and contains URL")
            return True
        if any(keyword in fact_lower for keyword in ['github', 'linkedin', 'portfolio', 'website', 'profile']):
            if debug:
                print(f"[is_complete_fact DEBUG] ACCEPT: category='links' and contains link keyword")
            return True
    
    # Check for verb-like tokens (action words common in resumes) - do this early
    verbs = [
        'worked', 'led', 'built', 'developed', 'created', 'designed', 
        'implemented', 'graduated', 'earned', 'completed', 'studied',
        'attended', 'based', 'located', 'achieved', 'improved', 'reduced',
        'increased', 'managed', 'collaborated', 'delivered', 'pursued',
        'utilized', 'maintained', 'hosted', 'established', 'founded',
        'co-founded', 'launched', 'optimized', 'scaled', 'architected',
        'engineered', 'researched', 'published', 'presented', 'taught',
        'mentored', 'supervised', 'coordinated', 'executed', 'deployed',
        'integrated', 'automated', 'analyzed', 'evaluated', 'tested',
        'debugged', 'refactored', 'migrated', 'upgraded', 'monitored'
    ]
    
    has_verb = any(verb in fact_lower for verb in verbs)
    matched_verbs = [verb for verb in verbs if verb in fact_lower]
    
    # Check for common resume patterns (even without explicit verb match)
    has_resume_pattern = bool(
        re.search(r'\b(at|as|in|for|with|from|to)\s+[A-Z]', fact) or
        re.search(r'\b\d+\s+(years?|months?|days?)\b', fact_lower) or
        re.search(r'\b(MS|M\.S\.|MBA|PhD|B\.A\.|B\.S\.|Master|Bachelor)\b', fact, re.IGNORECASE) or
        re.search(r'\b(GPA|grade|score|rating)\b', fact_lower) or
        re.search(r'https?://', fact_lower)
    )
    
    # If has verb or resume pattern, allow even if < 5 words (but still check other validations)
    # Otherwise, must have at least 5 words
    if not (has_verb or has_resume_pattern):
        if word_count < 5:
            if debug:
                print(f"[is_complete_fact DEBUG] REJECT: only {word_count} words (need >= 5) and no verb/pattern")
            return False
    
    debug_reasons.append(f"word_count={word_count} (>=5 ✓)")
    
    # Reject ellipses/garbled truncation
    if '...' in fact or re.search(r'\.{2,}', fact):
        if debug:
            print(f"[is_complete_fact DEBUG] REJECT: contains ellipses/truncation")
        return False
    
    # Reject obvious fragments (very short with connector words)
    if word_count < 6 and re.match(r'^\w+\s+(that|across|and|or|the|a|an)\s+\w+$', fact_lower):
        if debug:
            print(f"[is_complete_fact DEBUG] REJECT: looks like fragment pattern")
        return False
    
    # Reject if mostly very short words (keyword salad)
    short_words = [w for w in words if len(w) <= 2]
    short_word_ratio = len(short_words) / word_count if word_count > 0 else 0
    if short_word_ratio > 0.5:
        if debug:
            print(f"[is_complete_fact DEBUG] REJECT: {short_word_ratio:.1%} short words (keyword salad)")
        return False
    
    debug_reasons.append(f"short_word_ratio={short_word_ratio:.1%} (<0.5 ✓)")
    
    if has_verb:
        debug_reasons.append(f"has_verb=True (matched: {matched_verbs})")
    else:
        debug_reasons.append("has_verb=False")
    
    if has_resume_pattern:
        debug_reasons.append("has_resume_pattern=True")
    else:
        debug_reasons.append("has_resume_pattern=False")
    
    # Accept if has verb OR has resume pattern
    if has_verb or has_resume_pattern:
        if debug:
            print(f"[is_complete_fact DEBUG] ACCEPT: {'has_verb' if has_verb else 'has_resume_pattern'} | {'; '.join(debug_reasons)}")
        return True
    
    # Reject if no verb and no resume pattern
    if debug:
        print(f"[is_complete_fact DEBUG] REJECT: no verb and no resume pattern | {'; '.join(debug_reasons)}")
    return False


# Test cases from user's examples
test_cases = [
    # (fact, category, expected_result, description)
    ("Pursued MS in Computer Science at New York University.", "education", True, "Education fact with 'pursued'"),
    ("Worked as Senior Software Engineer at GEP Worldwide for 4 years.", "work", True, "Work experience with 'worked'"),
    ("Utilized LangChain for building agentic workflows.", "work", True, "Work fact with 'utilized'"),
    ("Designed telemetry logging and structured feedback classification.", "work", True, "Work fact with 'designed'"),
    ("GitHub profile: https://github.com/user", "links", True, "Link fact with URL"),
    ("Portfolio: https://example.com/portfolio", "links", True, "Link fact with URL"),
    ("Based in New York", "location", True, "Location fact"),
    ("Led backend and API initiatives on Azure", "work", True, "Work fact with 'led'"),
    
    # Should still reject fragments
    ("ms that are", "other", False, "Fragment - too short"),
    ("keywords only", "other", False, "Fragment - no verb/pattern"),
    ("...", "other", False, "Ellipsis only"),
    ("a b c d", "other", False, "Too short and no pattern"),
]

print("Testing is_complete_fact function...")
print("=" * 70)

all_passed = True
for fact, category, expected, description in test_cases:
    result = is_complete_fact(fact, category=category, debug=True)
    status = "✓ PASS" if result == expected else "✗ FAIL"
    if result != expected:
        all_passed = False
    
    print(f"{status} | {description}")
    print(f"  Fact: {fact}")
    print(f"  Category: {category}")
    print(f"  Expected: {expected}, Got: {result}")
    print()

if all_passed:
    print("=" * 70)
    print("✓ All tests passed!")
    exit(0)
else:
    print("=" * 70)
    print("✗ Some tests failed!")
    exit(1)

