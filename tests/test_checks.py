"""
Pytest unit tests for validation checks.
Tests fabrication detection, tone checks, and must_include logic.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from checks import (
    detects_fabrication,
    tone_professional,
    must_include_all,
    run_checks
)


class TestYearDetection:
    """Test year detection in fabrication checks."""
    
    def test_year_detection_allowed_year(self):
        """Year in allowed_facts should not be flagged."""
        text = "Graduated in 2026 with honors"
        allowed_facts = ["MS in Computer Science, expected May 2026"]
        result, fabrications = detects_fabrication(text, allowed_facts)
        assert result is True, "Year 2026 is in allowed_facts, should not be flagged"
        assert len(fabrications) == 0
    
    def test_year_detection_fabricated_year(self):
        """Year not in allowed_facts should be flagged."""
        text = "Graduated in 2025 with honors"
        allowed_facts = ["MS in Computer Science, expected May 2026"]
        result, fabrications = detects_fabrication(text, allowed_facts)
        assert result is False, "Year 2025 is not in allowed_facts, should be flagged"
        assert any("2025" in fab for fab in fabrications)
    
    def test_year_regex_non_capturing_group(self):
        """Year regex should use non-capturing group correctly."""
        text = "I worked there from 2019 to 2023"
        allowed_facts = ["Experience from 2019 to 2023"]
        result, fabrications = detects_fabrication(text, allowed_facts)
        assert result is True, "Both years are in allowed_facts"


class TestEmployerFabrication:
    """Test employer/company fabrication detection."""
    
    def test_employment_pattern_detection(self):
        """Employment patterns should be detected."""
        text = "I worked at Microsoft for 3 years"
        allowed_facts = ["4+ years software engineering experience at GEP Worldwide"]
        result, fabrications = detects_fabrication(text, allowed_facts, company="Google")
        assert result is False, "Microsoft not in allowed_facts, should be flagged"
        assert any("Microsoft" in fab or "microsoft" in fab.lower() for fab in fabrications)
    
    def test_target_company_allowed(self):
        """Target company should not be flagged."""
        text = "I'm interested in the Software Engineer role at Google"
        allowed_facts = ["MS in Computer Science"]
        result, fabrications = detects_fabrication(text, allowed_facts, company="Google")
        assert result is True, "Target company Google should be allowed"
        assert len(fabrications) == 0
    
    def test_interest_expressions_not_flagged(self):
        """Expressions of interest should not be flagged."""
        text = "I'm interested in the role at Apple"
        allowed_facts = ["MS in Computer Science"]
        result, fabrications = detects_fabrication(text, allowed_facts, company="Apple")
        assert result is True, "Interest expressions should not be flagged"
        assert len(fabrications) == 0
    
    def test_employment_context_patterns(self):
        """Various employment context patterns should be detected."""
        patterns = [
            "I worked at Amazon",
            "Previously at Meta I",
            "I interned at Netflix",
            "Employed by Stripe",
            "Joined Uber in 2020"
        ]
        allowed_facts = ["MS in Computer Science"]
        
        for text in patterns:
            result, fabrications = detects_fabrication(text, allowed_facts, company="Google")
            assert result is False, f"Employment pattern in '{text}' should be flagged"
            assert len(fabrications) > 0


class TestTargetCompanyAllowed:
    """Test that target company is always allowed."""
    
    def test_target_company_in_employment_context(self):
        """Target company even in employment context should be allowed."""
        text = "I worked at Google for 2 years"
        allowed_facts = ["MS in Computer Science"]
        result, fabrications = detects_fabrication(text, allowed_facts, company="Google")
        assert result is True, "Target company Google should be allowed even in employment context"
    
    def test_target_company_case_insensitive(self):
        """Target company matching should be case insensitive."""
        text = "I worked at google"
        allowed_facts = ["MS in Computer Science"]
        result, fabrications = detects_fabrication(text, allowed_facts, company="Google")
        assert result is True, "Case insensitive matching should work"


class TestSlangBoundaryCheck:
    """Test slang detection with word boundaries."""
    
    def test_slang_word_boundary_yo(self):
        """'yo' should not match 'you'."""
        text = "I would like to discuss this with you"
        result, issues = tone_professional(text)
        assert result is True, "'you' should not trigger 'yo' slang detection"
        assert len(issues) == 0
    
    def test_slang_detection_yo(self):
        """'yo' as slang should be detected."""
        text = "Hey yo, let's chat about this"
        result, issues = tone_professional(text)
        assert result is False, "'yo' slang should be detected"
        assert any("yo" in issue.lower() for issue in issues)
    
    def test_slang_word_boundary_bro(self):
        """'bro' should be detected with word boundaries."""
        text = "Thanks bro for the opportunity"
        result, issues = tone_professional(text)
        assert result is False, "'bro' slang should be detected"
        assert any("bro" in issue.lower() for issue in issues)
    
    def test_slang_not_in_substring(self):
        """Slang should not match substrings."""
        text = "I have experience with brokerage systems"
        result, issues = tone_professional(text)
        assert result is True, "'brokerage' should not trigger 'bro' detection"
        assert len(issues) == 0


class TestRelaxedChatDetection:
    """Test relaxed mode chat request detection."""
    
    def test_relaxed_chat_detection_call(self):
        """'call' should be accepted in relaxed mode."""
        text = "Would you be open to a quick call?"
        result, missing = must_include_all(text, ["request_chat"], strict_mode=False)
        assert result is True, "'call' should be accepted for request_chat in relaxed mode"
        assert len(missing) == 0
    
    def test_relaxed_chat_detection_connect(self):
        """'connect' should be accepted in relaxed mode."""
        text = "I'd love to connect and discuss this"
        result, missing = must_include_all(text, ["request_chat"], strict_mode=False)
        assert result is True, "'connect' should be accepted for request_chat in relaxed mode"
    
    def test_relaxed_chat_detection_schedule(self):
        """'schedule' should be accepted in relaxed mode."""
        text = "Can we schedule a 15-minute conversation?"
        result, missing = must_include_all(text, ["request_chat"], strict_mode=False)
        assert result is True, "'schedule' should be accepted for request_chat in relaxed mode"
    
    def test_strict_chat_detection_requires_chat(self):
        """Strict mode requires literal 'chat'."""
        text = "Would you be open to a quick call?"
        result, missing = must_include_all(text, ["request_chat"], strict_mode=True)
        assert result is False, "Strict mode should require literal 'chat'"
        assert "request_chat" in missing


class TestPublicationDetection:
    """Test publication/award detection."""
    
    def test_publication_detection_allowed(self):
        """Publications in allowed_facts should not be flagged."""
        text = "I have published papers on machine learning"
        allowed_facts = ["Published papers on ML and NLP"]
        result, fabrications = detects_fabrication(text, allowed_facts)
        assert result is True, "Publication in allowed_facts should not be flagged"
        assert len(fabrications) == 0
    
    def test_publication_detection_fabricated(self):
        """Publications not in allowed_facts should be flagged."""
        text = "I have published papers on machine learning"
        allowed_facts = ["MS in Computer Science"]
        result, fabrications = detects_fabrication(text, allowed_facts)
        assert result is False, "Publication not in allowed_facts should be flagged"
        assert any("publication" in fab.lower() or "published" in fab.lower() for fab in fabrications)
    
    def test_award_detection(self):
        """Awards not in allowed_facts should be flagged."""
        text = "I received an award for excellence"
        allowed_facts = ["MS in Computer Science"]
        result, fabrications = detects_fabrication(text, allowed_facts)
        assert result is False, "Award not in allowed_facts should be flagged"
        assert any("award" in fab.lower() for fab in fabrications)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

