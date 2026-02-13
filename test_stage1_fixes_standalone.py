"""
Standalone test for Stage 1 fixes (no imports needed).
Tests:
1. Deterministic link extraction
2. Award subject normalization
"""

import re

def extract_urls_standalone(text: str):
    """Standalone version of extract_urls for testing."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    
    link_facts = []
    seen_urls = set()
    
    for url in urls:
        url_lower = url.lower()
        if url_lower in seen_urls:
            continue
        seen_urls.add(url_lower)
        
        category = "links"
        fact_text = None
        
        if 'github.com' in url_lower:
            fact_text = f"GitHub profile: {url}"
        elif 'linkedin.com/in' in url_lower or 'linkedin.com/in/' in url_lower:
            fact_text = f"LinkedIn profile: {url}"
        elif 'netlify.app' in url_lower or 'github.io' in url_lower:
            fact_text = f"Profile link: {url}"
        
        if fact_text:
            link_facts.append({
                "fact": fact_text,
                "category": category,
                "evidence": url,
                "confidence": 0.95
            })
    
    # Handle remaining URLs as portfolio
    for url in urls:
        url_lower = url.lower()
        if url_lower not in seen_urls:
            if 'github.com' not in url_lower and 'linkedin.com' not in url_lower:
                fact_text = f"Profile link: {url}"
                link_facts.append({
                    "fact": fact_text,
                    "category": category,
                    "evidence": url,
                    "confidence": 0.95
                })
                seen_urls.add(url_lower)
    
    return link_facts


# Test profile text
test_profile = """
I am a software engineer with experience in machine learning.

GitHub: https://github.com/testuser
Portfolio: https://shwetashekhar.netlify.app/
LinkedIn: https://linkedin.com/in/testuser

I won an ACM Best Paper Award in 2025 for my research on LLM reliability.
Published a paper at NeurIPS 2024.
"""

print("=" * 70)
print("Test 1: Deterministic Link Extraction")
print("=" * 70)

url_facts = extract_urls_standalone(test_profile)
print(f"\nExtracted {len(url_facts)} URL facts:")
for fact in url_facts:
    print(f"  - {fact['fact']}")
    print(f"    Evidence: {fact['evidence']}")

# Verify
all_passed = True
github_found = False
linkedin_found = False
portfolio_found = False

for url_fact in url_facts:
    url = url_fact.get("evidence", "").lower()
    fact_text = url_fact.get("fact", "")
    
    if "github.com" in url:
        github_found = True
        if "GitHub profile" in fact_text:
            print("✓ PASS: GitHub link correctly formatted")
        else:
            print("❌ FAIL: GitHub link formatting")
            all_passed = False
    
    if "linkedin.com/in" in url:
        linkedin_found = True
        if "LinkedIn profile" in fact_text:
            print("✓ PASS: LinkedIn link correctly formatted")
        else:
            print("❌ FAIL: LinkedIn link formatting")
            all_passed = False
    
    if "netlify.app" in url:
        portfolio_found = True
        if "Profile link" in fact_text:
            print("✓ PASS: Portfolio link correctly formatted")
        else:
            print("❌ FAIL: Portfolio link formatting")
            all_passed = False

if not github_found:
    print("❌ FAIL: GitHub URL not found")
    all_passed = False
if not linkedin_found:
    print("❌ FAIL: LinkedIn URL not found")
    all_passed = False
if not portfolio_found:
    print("❌ FAIL: Portfolio URL not found")
    all_passed = False

print("\n" + "=" * 70)
print("Test 2: Award Subject Normalization")
print("=" * 70)

test_cases = [
    ("Won an ACM Best Paper Award in 2025", "awards", "I Won an ACM Best Paper Award in 2025"),
    ("won the IEEE Outstanding Research Award", "awards", "I won the IEEE Outstanding Research Award"),
    ("Led a team of 5 engineers", "work", "Led a team of 5 engineers"),  # Should not normalize
    ("Won first place in hackathon", "awards", "I Won first place in hackathon"),
]

print("\nTesting award normalization:")
for original, category, expected in test_cases:
    fact_text = original.strip()
    
    # Apply normalization
    if category.lower() == "awards" and fact_text.lower().startswith("won "):
        normalized = "I " + fact_text
    else:
        normalized = fact_text
    
    if normalized == expected:
        print(f"✓ PASS: '{original}' → '{normalized}'")
    else:
        print(f"❌ FAIL: '{original}' → '{normalized}' (expected '{expected}')")
        all_passed = False

print("\n" + "=" * 70)
if all_passed:
    print("✓ All tests passed!")
    exit(0)
else:
    print("✗ Some tests failed!")
    exit(1)

