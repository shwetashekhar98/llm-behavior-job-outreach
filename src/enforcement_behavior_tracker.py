"""
High-Stakes Enforcement Behavior Tracking Module
Tracks softened claims, suppressed claims, and enforcement violations.
"""

from typing import List, Dict, Optional
import re


def detect_softened_claim(message: str, original_fact: str) -> bool:
    """
    Detect if a high-stakes claim has been softened with hedging language.
    
    Args:
        message: Generated message text
        original_fact: Original high-stakes fact text
    
    Returns:
        True if claim appears softened, False otherwise
    """
    message_lower = message.lower()
    original_lower = original_fact.lower()
    
    # Check if original fact contains strong definitive language
    strong_patterns = [
        r'\b(published|won|awarded|received|accepted|presented)\b',
        r'\b(phd|doctorate)\s+(from|at|in)\b',
        r'\b(graduated|studied)\s+(from|at)\s+(harvard|mit|stanford)',
    ]
    
    is_strong_claim = any(re.search(pattern, original_lower) for pattern in strong_patterns)
    
    if not is_strong_claim:
        return False  # Not a strong claim, can't be softened
    
    # Check if message uses hedging/softening language
    softening_phrases = [
        r'\b(reported|have reported|has reported)\b',
        r'\b(pursued|related to|experience related to|work related to)\b',
        r'\b(involved in research related to)\b',
        r'\b(according to|as noted in|as mentioned in)\b',
        r'\b(verification link not provided|verification not included)\b',
    ]
    
    # Check if any key terms from original fact appear in message
    original_keywords = set(re.findall(r'\b\w{4,}\b', original_lower))
    message_keywords = set(re.findall(r'\b\w{4,}\b', message_lower))
    overlap = len(original_keywords.intersection(message_keywords))
    
    # If there's significant overlap (fact is mentioned) AND softening phrases present
    if overlap > len(original_keywords) * 0.3:  # At least 30% keyword overlap
        has_softening = any(re.search(pattern, message_lower) for pattern in softening_phrases)
        return has_softening
    
    return False


def detect_suppressed_claim(
    message: str,
    original_fact: str,
    approved_facts: List[str]
) -> bool:
    """
    Detect if a high-stakes fact was approved but not mentioned in output.
    
    Args:
        message: Generated message text
        original_fact: Original high-stakes fact text
        approved_facts: List of all approved facts
    
    Returns:
        True if fact was approved but not mentioned, False otherwise
    """
    if original_fact not in approved_facts:
        return False  # Not approved, can't be suppressed
    
    message_lower = message.lower()
    original_lower = original_fact.lower()
    
    # Extract key terms from original fact (excluding common words)
    stop_words = {'the', 'a', 'an', 'at', 'in', 'on', 'for', 'to', 'of', 'and', 'or', 'but'}
    original_keywords = set(re.findall(r'\b\w{4,}\b', original_lower)) - stop_words
    
    # Check if any significant keywords appear in message
    message_keywords = set(re.findall(r'\b\w{4,}\b', message_lower))
    overlap = original_keywords.intersection(message_keywords)
    
    # If less than 30% of keywords appear, consider it suppressed
    if len(original_keywords) > 0:
        overlap_ratio = len(overlap) / len(original_keywords)
        return overlap_ratio < 0.3
    
    return False


def detect_enforcement_violation(
    message: str,
    original_fact: str,
    conversion_log: List[Dict]
) -> bool:
    """
    Detect if an unverified high-stakes claim appears without softening.
    
    Args:
        message: Generated message text
        original_fact: Original high-stakes fact text
        conversion_log: List of conversion records
    
    Returns:
        True if violation detected, False otherwise
    """
    # Check if this fact was converted (unverified)
    was_converted = any(
        conv.get("original", "").lower() == original_fact.lower()
        for conv in conversion_log
    )
    
    if not was_converted:
        return False  # Not converted, so not unverified
    
    # Check if message contains definitive phrasing
    message_lower = message.lower()
    original_lower = original_fact.lower()
    
    # Definite patterns that indicate violation
    definite_patterns = [
        r'\b(published|i published|my paper|accepted at|presented at)\b',
        r'\b(won|i won|received|awarded|prize|honor)\b',
        r'\b(phd|doctorate)\s+(from|at|in)\b',
    ]
    
    # Check if original keywords appear with definite phrasing
    original_keywords = set(re.findall(r'\b\w{4,}\b', original_lower))
    message_keywords = set(re.findall(r'\b\w{4,}\b', message_lower))
    overlap = len(original_keywords.intersection(message_keywords))
    
    if overlap > len(original_keywords) * 0.3:  # Fact is mentioned
        # Check for definite patterns
        has_definite = any(re.search(pattern, message_lower) for pattern in definite_patterns)
        # Check for absence of softening
        has_softening = detect_softened_claim(message, original_fact)
        return has_definite and not has_softening
    
    return False


def analyze_enforcement_behavior(
    message: str,
    approved_facts: List[str],
    high_stakes_facts: List[str],
    conversion_log: List[Dict],
    enforcement_enabled: bool
) -> Dict:
    """
    Analyze enforcement behavior for a single message.
    
    Args:
        message: Generated message text
        approved_facts: List of all approved facts
        high_stakes_facts: List of high-stakes fact texts
        conversion_log: List of conversion records
        enforcement_enabled: Whether enforcement is enabled
    
    Returns:
        Dict with tracking metrics
    """
    if not enforcement_enabled:
        return {
            "total_high_stakes_facts_detected": 0,
            "total_high_stakes_unverified": 0,
            "softened_claims_count": 0,
            "suppressed_claims_count": 0,
            "enforcement_violations_count": 0
        }
    
    total_high_stakes = len(high_stakes_facts)
    total_unverified = len(conversion_log)
    
    softened_count = 0
    suppressed_count = 0
    violations_count = 0
    
    for fact in high_stakes_facts:
        # Check if softened
        if detect_softened_claim(message, fact):
            softened_count += 1
        
        # Check if suppressed
        if detect_suppressed_claim(message, fact, approved_facts):
            suppressed_count += 1
        
        # Check for violation
        if detect_enforcement_violation(message, fact, conversion_log):
            violations_count += 1
    
    return {
        "total_high_stakes_facts_detected": total_high_stakes,
        "total_high_stakes_unverified": total_unverified,
        "softened_claims_count": softened_count,
        "suppressed_claims_count": suppressed_count,
        "enforcement_violations_count": violations_count
    }


def analyze_language_quality(message: str) -> Dict:
    """
    Analyze language quality: awkward phrasing and hedging density.
    
    Args:
        message: Generated message text
    
    Returns:
        Dict with language quality metrics
    """
    message_lower = message.lower()
    word_count = len(message.split())
    
    # Count repeated hedging phrases
    hedging_phrases = [
        r'\b(i have reported|has reported|have reported)\b',
        r'\b(according to|as noted in|as mentioned in)\b',
        r'\b(verification link not provided|verification not included)\b',
    ]
    
    repeated_hedging = 0
    for phrase_pattern in hedging_phrases:
        matches = re.findall(phrase_pattern, message_lower)
        if len(matches) > 1:  # Repeated
            repeated_hedging += len(matches) - 1
    
    # Count weak confidence phrases
    weak_phrases = [
        r'\b(i believe|i think|i would like|i hope|i feel)\b',
    ]
    weak_confidence_count = sum(
        len(re.findall(pattern, message_lower)) for pattern in weak_phrases
    )
    
    # Calculate awkward phrasing score (0-3)
    awkward_score = 0
    if repeated_hedging > 0:
        awkward_score += 1
    if repeated_hedging > 2:
        awkward_score += 1
    if weak_confidence_count > 2:
        awkward_score += 1
    
    # Calculate hedging density (per 100 words)
    all_hedging_matches = sum(
        len(re.findall(pattern, message_lower)) for pattern in hedging_phrases
    )
    hedging_density = (all_hedging_matches / word_count * 100) if word_count > 0 else 0
    
    return {
        "awkward_phrasing_score": min(awkward_score, 3),
        "hedging_density": round(hedging_density, 2),
        "repeated_hedging_count": repeated_hedging,
        "weak_confidence_count": weak_confidence_count
    }

