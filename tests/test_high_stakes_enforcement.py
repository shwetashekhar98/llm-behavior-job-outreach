"""
Unit tests for High-Stakes Enforcement
"""

import pytest
from src.high_stakes_enforcement import (
    preprocess_facts_for_generation,
    convert_to_cautious_phrasing,
    detect_high_stakes_enforcement_violation
)


def test_preprocess_facts_enforcement_disabled():
    """Test that when enforcement is disabled, facts pass through unchanged."""
    facts = ["Published a research paper at NeurIPS 2025.", "Worked at Google for 2 years."]
    result = preprocess_facts_for_generation(facts, None, False)
    
    assert result["facts_for_generation"] == facts
    assert len(result["conversion_log"]) == 0
    assert result["stats"]["converted_count"] == 0


def test_preprocess_facts_enforcement_enabled_unverified():
    """Test that unverified high-stakes facts are converted to cautious phrasing."""
    facts = ["Published a research paper at NeurIPS 2025."]
    metadata = {
        "Published a research paper at NeurIPS 2025.": {
            "verification_status": "unverified",
            "verification_url": ""
        }
    }
    
    result = preprocess_facts_for_generation(facts, metadata, True)
    
    assert len(result["conversion_log"]) == 1
    assert result["stats"]["high_stakes_count"] == 1
    assert result["stats"]["unverified_count"] == 1
    assert result["stats"]["converted_count"] == 1
    assert "Has reported research work" in result["facts_for_generation"][0]
    assert "verification link not provided" in result["facts_for_generation"][0]


def test_preprocess_facts_enforcement_enabled_verified():
    """Test that verified high-stakes facts pass through unchanged."""
    facts = ["Published a research paper at NeurIPS 2025."]
    metadata = {
        "Published a research paper at NeurIPS 2025.": {
            "verification_status": "verified",
            "verification_url": "https://example.com/paper"
        }
    }
    
    result = preprocess_facts_for_generation(facts, metadata, True)
    
    assert len(result["conversion_log"]) == 0
    assert result["stats"]["high_stakes_count"] == 1
    assert result["stats"]["verified_count"] == 1
    assert result["stats"]["unverified_count"] == 0
    assert result["facts_for_generation"] == facts


def test_convert_to_cautious_phrasing_publication():
    """Test conversion of publication claims."""
    fact = "Published a research paper at NeurIPS 2025."
    result = convert_to_cautious_phrasing(fact, "impact")
    
    assert "Has reported research work" in result
    assert "verification link not provided" in result


def test_convert_to_cautious_phrasing_award():
    """Test conversion of award claims."""
    fact = "Won an ACM Best Paper Award in 2025."
    result = convert_to_cautious_phrasing(fact, "awards")
    
    assert "Has reported an award claim" in result
    assert "verification link not provided" in result


def test_detect_violation_definitive_publication():
    """Test detection of definitive publication claims."""
    message = "I published a paper at NeurIPS 2025 on LLM reliability."
    original_facts = ["Published a research paper at NeurIPS 2025."]
    conversion_log = [{
        "original": "Published a research paper at NeurIPS 2025.",
        "converted": "Has reported research work related to NeurIPS; verification link not provided.",
        "reason": "unverified_high_stakes",
        "category": "impact"
    }]
    
    violation_detected, violations = detect_high_stakes_enforcement_violation(
        message, original_facts, conversion_log
    )
    
    assert violation_detected
    assert len(violations) > 0
    assert any("publication" in v.lower() or "definitive" in v.lower() for v in violations)


def test_detect_violation_cautious_phrasing_passes():
    """Test that cautious phrasing does not trigger violation."""
    message = "I have reported research work related to NeurIPS; verification link not provided."
    original_facts = ["Published a research paper at NeurIPS 2025."]
    conversion_log = [{
        "original": "Published a research paper at NeurIPS 2025.",
        "converted": "Has reported research work related to NeurIPS; verification link not provided.",
        "reason": "unverified_high_stakes",
        "category": "impact"
    }]
    
    violation_detected, violations = detect_high_stakes_enforcement_violation(
        message, original_facts, conversion_log
    )
    
    assert not violation_detected
    assert len(violations) == 0


def test_detect_violation_no_conversion_log():
    """Test that empty conversion log means no violations."""
    message = "I published a paper at NeurIPS."
    violation_detected, violations = detect_high_stakes_enforcement_violation(
        message, [], []
    )
    
    assert not violation_detected
    assert len(violations) == 0

