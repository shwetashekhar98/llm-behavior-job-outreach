"""
High-Stakes Claim Verification Layer
Additive, non-breaking feature for trust calibration.
"""

import re
from typing import Dict


def is_high_stakes(fact_text: str, category: str) -> bool:
    """
    Determine if a fact is high-stakes and requires verification.
    
    Returns True if:
    - category in ["impact", "awards", "education"]
    OR
    - fact_text contains (case-insensitive) any keyword from high-stakes list
    
    Args:
        fact_text: The fact text to check
        category: The category of the fact
    
    Returns:
        True if high-stakes, False otherwise
    """
    if not fact_text:
        return False
    
    fact_lower = fact_text.lower()
    category_lower = category.lower() if category else ""
    
    # Check category
    high_stakes_categories = ["impact", "awards", "education"]
    if category_lower in high_stakes_categories:
        return True
    
    # Check for high-stakes keywords
    high_stakes_keywords = [
        "neurips", "icml", "cvpr", "acl", "emnlp", "nips",
        "openai", "google", "meta", "amazon", "microsoft", "apple",
        "harvard", "mit", "stanford", "phd", "ieee", "acm", "nasa"
    ]
    
    for keyword in high_stakes_keywords:
        if keyword in fact_lower:
            return True
    
    return False


def annotate_fact_with_trust(fact: Dict, enable_high_stakes: bool = False) -> Dict:
    """
    Add trust calibration metadata to a fact.
    
    Args:
        fact: Fact dictionary with at least "value" and "category" keys
        enable_high_stakes: Whether high-stakes layer is enabled
    
    Returns:
        Fact dictionary with added trust metadata (if enabled)
    """
    if not enable_high_stakes:
        # Return fact unchanged if feature is disabled
        return fact
    
    fact_text = fact.get("value", "")
    category = fact.get("category", "other")
    
    is_high = is_high_stakes(fact_text, category)
    
    # Add trust metadata
    fact_copy = fact.copy()
    fact_copy["trust_flag"] = "high_stakes" if is_high else "normal"
    
    # Initialize verification fields if not present
    if "verification_status" not in fact_copy:
        fact_copy["verification_status"] = "unverified" if is_high else "verified"
    if "verification_url" not in fact_copy:
        fact_copy["verification_url"] = ""
    
    return fact_copy

