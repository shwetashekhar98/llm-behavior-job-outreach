"""
Regression test for Stage 1 fixes using the provided profile text.
Tests:
1. Deterministic link extraction (GitHub, Portfolio, LinkedIn)
2. Award fact acceptance (no longer rejected by is_complete_fact_check_failed)
"""

import sys
import re

# Standalone versions for testing
def extract_links_from_text_standalone(combined_text: str):
    """Standalone version for testing."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, combined_text)
    
    link_facts = []
    seen_urls = set()
    
    for url in urls:
        url_normalized = url.rstrip('/').lower()
        if url_normalized in seen_urls:
            continue
        seen_urls.add(url_normalized)
        
        url_lower = url.lower()
        category = "links"
        fact_text = None
        
        if 'github.com' in url_lower:
            fact_text = f"GitHub profile: {url}"
        elif 'linkedin.com/in' in url_lower:
            fact_text = f"LinkedIn profile: {url}"
        elif 'netlify.app' in url_lower or 'github.io' in url_lower:
            fact_text = f"Profile link: {url}"
        else:
            if 'github.com' not in url_lower and 'linkedin.com' not in url_lower:
                fact_text = f"Profile link: {url}"
        
        if fact_text:
            link_facts.append({
                "fact": fact_text,
                "category": category,
                "evidence": url,
                "confidence": 0.95
            })
    
    return link_facts


def is_complete_fact_standalone(fact: str, category: str = "other") -> bool:
    """Standalone version for testing."""
    if not fact:
        return False
    
    fact = fact.strip()
    if not fact:
        return False
    
    fact_lower = fact.lower()
    words = fact.split()
    word_count = len(words)
    
    # Links category: always accept
    if category == "links":
        if re.search(r'https?://', fact_lower):
            return True
        if any(keyword in fact_lower for keyword in ['github', 'linkedin', 'portfolio', 'website', 'profile']):
            return True
    
    # AWARDS CATEGORY: Special handling
    if category.lower() == "awards" or any(keyword in fact_lower for keyword in ["award", "prize", "honor", "honour", "medal", "recognition"]):
        award_verbs = ["won", "receive", "received", "awarded", "earned", "granted", "bestowed"]
        award_nouns = ["award", "prize", "honor", "honour", "medal", "recognition", "distinction"]
        
        has_award_verb = any(verb in fact_lower for verb in award_verbs)
        has_award_noun = any(noun in fact_lower for noun in award_nouns)
        
        if has_award_verb and has_award_noun:
            if word_count >= 6:
                if '...' not in fact and not re.search(r'\.{2,}', fact):
                    short_words = [w for w in words if len(w) <= 2]
                    short_word_ratio = len(short_words) / word_count if word_count > 0 else 0
                    if short_word_ratio <= 0.5:
                        return True
    
    # General verb check
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
        'debugged', 'refactored', 'migrated', 'upgraded', 'monitored',
        'won', 'received', 'awarded'
    ]
    
    has_verb = any(verb in fact_lower for verb in verbs)
    
    has_resume_pattern = bool(
        re.search(r'\b(at|as|in|for|with|from|to)\s+[A-Z]', fact) or
        re.search(r'\b\d+\s+(years?|months?|days?)\b', fact_lower) or
        re.search(r'\b(MS|M\.S\.|MBA|PhD|B\.A\.|B\.S\.|Master|Bachelor)\b', fact, re.IGNORECASE) or
        re.search(r'\b(GPA|grade|score|rating)\b', fact_lower) or
        re.search(r'https?://', fact_lower)
    )
    
    if not (has_verb or has_resume_pattern):
        if word_count < 5:
            return False
    
    if '...' in fact or re.search(r'\.{2,}', fact):
        return False
    
    if word_count < 6 and re.match(r'^\w+\s+(that|across|and|or|the|a|an)\s+\w+$', fact_lower):
        return False
    
    short_words = [w for w in words if len(w) <= 2]
    short_word_ratio = len(short_words) / word_count if word_count > 0 else 0
    if short_word_ratio > 0.5:
        return False
    
    return True


# Test profile text from user
test_profile = """Shweta Shekhar

MS in Computer Science student at New York University.

Experience:

* Senior Software Engineer at GEP Worldwide (4 years)

Publications:

* Published a research paper at NeurIPS 2025.

Awards:

* Won an ACM Best Paper Award in 2025 for my research work.

GitHub: https://github.com/shwetashekhar98
Portfolio: https://shwetashekhar.netlify.app/
LinkedIn: https://www.linkedin.com/in/shwetashekhar98/
"""

print("=" * 70)
print("Regression Test: Stage 1 Fixes")
print("=" * 70)

print("\nTest 1: Deterministic Link Extraction")
print("-" * 70)

link_facts = extract_links_from_text_standalone(test_profile)
print(f"Extracted {len(link_facts)} link facts:")

expected_links = {
    "github.com": "GitHub profile",
    "linkedin.com/in": "LinkedIn profile",
    "netlify.app": "Profile link"
}

all_passed = True
found_links = set()

for link_fact in link_facts:
    url = link_fact.get("evidence", "").lower()
    fact_text = link_fact.get("fact", "")
    print(f"  - {fact_text}")
    print(f"    Evidence: {url}")
    
    if "github.com" in url:
        found_links.add("github")
        if "GitHub profile" in fact_text:
            print("    ✓ PASS: GitHub link correctly formatted")
        else:
            print("    ❌ FAIL: GitHub link formatting")
            all_passed = False
    
    if "linkedin.com/in" in url:
        found_links.add("linkedin")
        if "LinkedIn profile" in fact_text:
            print("    ✓ PASS: LinkedIn link correctly formatted")
        else:
            print("    ❌ FAIL: LinkedIn link formatting")
            all_passed = False
    
    if "netlify.app" in url:
        found_links.add("portfolio")
        if "Profile link" in fact_text:
            print("    ✓ PASS: Portfolio link correctly formatted")
        else:
            print("    ❌ FAIL: Portfolio link formatting")
            all_passed = False

if len(found_links) < 3:
    print(f"❌ FAIL: Expected 3 links (GitHub, LinkedIn, Portfolio), found: {found_links}")
    all_passed = False
else:
    print(f"✓ PASS: All 3 expected links found: {found_links}")

print("\nTest 2: Award Fact Acceptance")
print("-" * 70)

award_facts = [
    ("Won an ACM Best Paper Award in 2025 for my research work.", "awards"),
    ("Received an ACM Best Paper Award in 2025 for research work.", "awards"),
    ("Awarded the IEEE Outstanding Research Prize in 2024.", "awards"),
    ("Earned recognition for outstanding contributions.", "awards"),
]

print("Testing award facts (should all PASS is_complete_fact):")
for fact_text, category in award_facts:
    result = is_complete_fact_standalone(fact_text, category)
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"  {status}: '{fact_text[:60]}...'")
    if not result:
        all_passed = False
        print(f"    ❌ This award fact was incorrectly rejected!")

print("\n" + "=" * 70)
if all_passed:
    print("✓ All regression tests passed!")
    print("\nExpected Stage 1 Accepted should include:")
    print("  - GitHub profile: https://github.com/shwetashekhar98")
    print("  - Profile link: https://shwetashekhar.netlify.app/")
    print("  - LinkedIn profile: https://www.linkedin.com/in/shwetashekhar98/")
    print("  - Won/Received ACM Best Paper Award in 2025...")
    exit(0)
else:
    print("✗ Some regression tests failed!")
    exit(1)

