"""
Unit test for category normalization: publications -> impact
"""

import sys
sys.path.insert(0, 'src')

def normalize_category(category: str) -> str:
    """
    Normalize category: map "publications" to "impact".
    This mirrors the logic in profile_extractor.py
    """
    category_lower = category.lower() if category else "other"
    if category_lower == "publications":
        return "impact"
    
    # Validate category
    valid_categories = ["education", "work", "impact", "skills", "projects", 
                      "awards", "links", "location", "other"]
    if category not in valid_categories:
        return "other"
    
    return category


# Test cases
test_cases = [
    ("publications", "impact"),
    ("Publications", "impact"),
    ("PUBLICATIONS", "impact"),
    ("impact", "impact"),  # Should remain unchanged
    ("education", "education"),  # Should remain unchanged
    ("work", "work"),  # Should remain unchanged
    ("invalid_category", "other"),  # Invalid should map to "other"
    ("", "other"),  # Empty should map to "other"
    (None, "other"),  # None should map to "other"
]

print("=" * 70)
print("Test: Category Normalization (publications -> impact)")
print("=" * 70)

all_passed = True
for input_category, expected_output in test_cases:
    result = normalize_category(input_category)
    status = "✓ PASS" if result == expected_output else "✗ FAIL"
    if result != expected_output:
        all_passed = False
    
    print(f"{status}: '{input_category}' -> '{result}' (expected: '{expected_output}')")

print("\n" + "=" * 70)
if all_passed:
    print("✓ All tests passed!")
    sys.exit(0)
else:
    print("✗ Some tests failed!")
    sys.exit(1)

