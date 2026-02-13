"""
Test semantic must_include checks.
Run: python test_must_include_semantic.py
"""

import sys
sys.path.insert(0, 'src')

from validation_engine import (
    must_include_check,
    contains_portfolio_url,
    contains_github_url,
    contains_chat_ask
)

# Test 1: Portfolio URL with "profile link:" pattern
print("Test 1: Portfolio with 'profile link:' pattern")
message1 = "Hi, I'm interested in the role. Here's my profile link: https://shwetashekhar.netlify.app/"
approved_facts1 = ["Portfolio: https://shwetashekhar.netlify.app/"]
link_facts1 = {"portfolio": "https://shwetashekhar.netlify.app/", "github": None, "linkedin": None}
result1, missing1 = must_include_check(message1, ["Portfolio"], approved_facts=approved_facts1, link_facts=link_facts1)
print(f"  Message: {message1}")
print(f"  Required: Portfolio")
print(f"  Result: {'✓ PASS' if result1 else '✗ FAIL'}")
print(f"  Missing: {missing1}")
assert result1, f"Test 1 failed: {missing1}"
print()

# Test 2: GitHub URL
print("Test 2: GitHub URL")
message2 = "Check out my work at https://github.com/user/repo"
approved_facts2 = ["GitHub: https://github.com/user/repo"]
link_facts2 = {"github": "https://github.com/user/repo", "portfolio": None, "linkedin": None}
result2, missing2 = must_include_check(message2, ["GitHub"], approved_facts=approved_facts2, link_facts=link_facts2)
print(f"  Message: {message2}")
print(f"  Required: GitHub")
print(f"  Result: {'✓ PASS' if result2 else '✗ FAIL'}")
print(f"  Missing: {missing2}")
assert result2, f"Test 2 failed: {missing2}"
print()

# Test 3: Chat request
print("Test 3: Chat request with '15-minute chat'")
message3 = "Would you be open to a 15-minute chat to discuss the role?"
result3, missing3 = must_include_check(message3, ["Ask for chat"])
print(f"  Message: {message3}")
print(f"  Required: Ask for chat")
print(f"  Result: {'✓ PASS' if result3 else '✗ FAIL'}")
print(f"  Missing: {missing3}")
assert result3, f"Test 3 failed: {missing3}"
print()

# Test 4: Portfolio URL domain match
print("Test 4: Portfolio URL domain match")
message4 = "Visit shwetashekhar.netlify.app to see my work"
approved_facts4 = ["Portfolio: https://shwetashekhar.netlify.app/"]
link_facts4 = {"portfolio": "https://shwetashekhar.netlify.app/", "github": None, "linkedin": None}
result4, missing4 = must_include_check(message4, ["Portfolio"], approved_facts=approved_facts4, link_facts=link_facts4)
print(f"  Message: {message4}")
print(f"  Required: Portfolio")
print(f"  Result: {'✓ PASS' if result4 else '✗ FAIL'}")
print(f"  Missing: {missing4}")
assert result4, f"Test 4 failed: {missing4}"
print()

# Test 5: Helper function tests
print("Test 5: Helper function - contains_portfolio_url")
assert contains_portfolio_url(
    "profile link: https://shwetashekhar.netlify.app/",
    ["Portfolio: https://shwetashekhar.netlify.app/"],
    {"portfolio": "https://shwetashekhar.netlify.app/"}
), "contains_portfolio_url failed for profile link pattern"
print("  ✓ contains_portfolio_url('profile link: ...') passes")
print()

print("Test 6: Helper function - contains_github_url")
assert contains_github_url(
    "Check https://github.com/user/repo",
    ["GitHub: https://github.com/user/repo"],
    {"github": "https://github.com/user/repo"}
), "contains_github_url failed"
print("  ✓ contains_github_url('github.com') passes")
print()

print("Test 7: Helper function - contains_chat_ask")
assert contains_chat_ask("Would you be open to a 15-minute chat?"), "contains_chat_ask failed"
assert contains_chat_ask("Let's schedule a call"), "contains_chat_ask failed for 'schedule'"
assert contains_chat_ask("I'd love to connect"), "contains_chat_ask failed for 'connect'"
print("  ✓ contains_chat_ask passes for various phrases")
print()

print("=" * 70)
print("✓ All tests passed!")
print("=" * 70)

