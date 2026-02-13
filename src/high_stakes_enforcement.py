"""
High-Stakes Claim Enforcement Module
Preprocesses approved facts to enforce cautious language for unverified high-stakes claims.
"""

from typing import List, Dict, Optional
import sys
from pathlib import Path

# Import from same package
sys.path.insert(0, str(Path(__file__).parent))
from high_stakes import is_high_stakes


def preprocess_facts_for_generation(
    approved_facts: List,
    high_stakes_metadata: Optional[Dict] = None,
    enforce_high_stakes_language: bool = False
) -> Dict:
    """
    Preprocess approved facts based on high-stakes enforcement settings.
    
    Args:
        approved_facts: List of approved fact strings (or dicts with metadata)
        high_stakes_metadata: Dict mapping fact_text to {verification_status, verification_url}
        enforce_high_stakes_language: Whether to enforce cautious phrasing
    
    Returns:
        Dict with:
            - facts_for_generation: List of facts to use in generation (may be converted)
            - original_facts: List of original facts (for audit)
            - conversion_log: List of conversion records
            - stats: Dict with counts
    """
    # If enforcement is disabled, pass through as-is
    if not enforce_high_stakes_language:
        return {
            "facts_for_generation": approved_facts.copy(),
            "original_facts": approved_facts.copy(),
            "conversion_log": [],
            "stats": {
                "total_facts": len(approved_facts),
                "high_stakes_count": 0,
                "verified_count": 0,
                "unverified_count": 0,
                "converted_count": 0,
                "excluded_count": 0
            }
        }
    
    # If enforcement enabled but no metadata, pass through (backward compatible)
    if not high_stakes_metadata:
        return {
            "facts_for_generation": approved_facts.copy(),
            "original_facts": approved_facts.copy(),
            "conversion_log": [],
            "stats": {
                "total_facts": len(approved_facts),
                "high_stakes_count": 0,
                "verified_count": 0,
                "unverified_count": 0,
                "converted_count": 0,
                "excluded_count": 0
            }
        }
    
    facts_for_generation = []
    original_facts = []
    conversion_log = []
    
    stats = {
        "total_facts": len(approved_facts),
        "high_stakes_count": 0,
        "verified_count": 0,
        "unverified_count": 0,
        "converted_count": 0,
        "excluded_count": 0
    }
    
    for fact in approved_facts:
        # Handle both string and dict formats
        if isinstance(fact, dict):
            fact_text = fact.get("value", fact.get("fact", ""))
            fact_category = fact.get("category", "other")
            # Check if fact has embedded high-stakes metadata
            is_high = fact.get("trust_flag") == "high_stakes" or is_high_stakes(fact_text, fact_category)
            verification_status = fact.get("verification_status", "unverified")
            verification_url = fact.get("verification_url", "")
        else:
            fact_text = fact
            fact_category = "other"
            # Check metadata dict
            metadata = high_stakes_metadata.get(fact_text, {})
            is_high = is_high_stakes(fact_text, fact_category) or metadata.get("verification_status") is not None
            verification_status = metadata.get("verification_status", "unverified")
            verification_url = metadata.get("verification_url", "")
        
        original_facts.append(fact_text)
        
        # Track high-stakes facts
        if is_high:
            stats["high_stakes_count"] += 1
            
            if verification_status == "verified":
                stats["verified_count"] += 1
                # Verified facts pass through unchanged
                facts_for_generation.append(fact_text)
            else:
                stats["unverified_count"] += 1
                # Unverified high-stakes: convert to cautious version
                cautious_version = convert_to_cautious_phrasing(fact_text, fact_category)
                facts_for_generation.append(cautious_version)
                conversion_log.append({
                    "original": fact_text,
                    "converted": cautious_version,
                    "reason": "unverified_high_stakes",
                    "category": fact_category
                })
                stats["converted_count"] += 1
        else:
            # Normal facts pass through unchanged
            facts_for_generation.append(fact_text)
    
    return {
        "facts_for_generation": facts_for_generation,
        "original_facts": original_facts,
        "conversion_log": conversion_log,
        "stats": stats
    }


def convert_to_cautious_phrasing(fact_text: str, category: str) -> str:
    """
    Convert a high-stakes fact to cautious phrasing.
    
    Examples:
    - "Published a research paper at NeurIPS 2025" 
      -> "Has reported research work related to NeurIPS 2025; verification link not provided."
    - "Won an ACM Best Paper Award in 2025"
      -> "Has reported an award claim; verification link not provided."
    - "PhD in Computer Science from MIT"
      -> "Has reported PhD in Computer Science from MIT; verification link not provided."
    """
    fact_lower = fact_text.lower()
    category_lower = category.lower()
    
    # Publications/Research
    if category_lower in ["impact", "publications", "research"] or any(kw in fact_lower for kw in ["published", "paper", "publication", "research"]):
        # Extract key topic/venue if possible
        if "neurips" in fact_lower or "icml" in fact_lower or "cvpr" in fact_lower or "acl" in fact_lower:
            venue = "NeurIPS" if "neurips" in fact_lower else ("ICML" if "icml" in fact_lower else ("CVPR" if "cvpr" in fact_lower else "ACL"))
            return f"Has reported research work related to {venue}; verification link not provided."
        return "Has reported research work; verification link not provided."
    
    # Awards
    if category_lower == "awards" or any(kw in fact_lower for kw in ["won", "award", "prize", "honor"]):
        return "Has reported an award claim; verification link not provided."
    
    # Education (PhD, elite universities)
    if category_lower == "education" or any(kw in fact_lower for kw in ["phd", "doctorate", "harvard", "mit", "stanford"]):
        # Try to preserve degree/uni info but make it cautious
        if "phd" in fact_lower or "doctorate" in fact_lower:
            return f"Has reported {fact_text}; verification link not provided."
        return f"Has reported educational background; verification link not provided."
    
    # Work experience (elite employers)
    if category_lower == "work" and any(kw in fact_lower for kw in ["openai", "google", "meta", "microsoft", "apple", "amazon"]):
        return f"Has reported work experience; verification link not provided."
    
    # Default: generic cautious phrasing
    return f"Has reported: {fact_text}; verification link not provided."


def detect_high_stakes_enforcement_violation(
    message: str,
    original_facts: List[str],
    conversion_log: List[Dict],
    high_stakes_metadata: Optional[Dict] = None
) -> tuple[bool, List[str]]:
    """
    Detect if generated message contains definitive statements for unverified high-stakes claims.
    
    Args:
        message: Generated message text
        original_facts: List of original fact texts
        conversion_log: List of conversion records from preprocessing
        high_stakes_metadata: Dict mapping fact_text to metadata
    
    Returns:
        Tuple of (violation_detected: bool, violations: List[str])
    """
    if not conversion_log:
        return (False, [])
    
    message_lower = message.lower()
    violations = []
    
    # Check each converted fact
    for conversion in conversion_log:
        original = conversion["original"]
        converted = conversion["converted"]
        category = conversion.get("category", "other")
        
        # Check if message contains definitive phrasing of the original fact
        # Look for patterns that indicate definitive statements
        
        # For publications
        if category in ["impact", "publications", "research"]:
            # Definite patterns: "Published", "I published", "My paper", "Accepted at"
            definite_patterns = [
                r'\b(published|i published|my paper|accepted at|presented at)\b',
                r'\b(paper|publication)\s+(at|in|for)\s+',
            ]
            for pattern in definite_patterns:
                if any(kw in original.lower() for kw in ["neurips", "icml", "cvpr", "acl", "paper", "publication"]):
                    # Check if message contains venue name with definite phrasing
                    if any(venue in message_lower for venue in ["neurips", "icml", "cvpr", "acl"]):
                        import re
                        if re.search(pattern, message_lower):
                            violations.append(f"Definitive publication claim: '{original}'")
                            break
        
        # For awards
        if category == "awards":
            # Definite patterns: "Won", "I won", "Received", "Awarded"
            definite_patterns = [
                r'\b(won|i won|received|awarded|prize|honor)\b',
            ]
            import re
            for pattern in definite_patterns:
                if re.search(pattern, message_lower):
                    # Check if award-related keywords from original appear
                    if any(kw in message_lower for kw in ["award", "prize", "acm", "ieee"]):
                        violations.append(f"Definitive award claim: '{original}'")
                        break
        
        # For education (PhD, elite universities)
        if category == "education":
            # Definite patterns: "PhD from", "Graduated from", "Studied at"
            definite_patterns = [
                r'\b(phd|doctorate)\s+(from|at|in)\b',
                r'\b(graduated|studied)\s+(from|at)\s+(harvard|mit|stanford)',
            ]
            import re
            for pattern in definite_patterns:
                if re.search(pattern, message_lower):
                    violations.append(f"Definitive education claim: '{original}'")
                    break
        
        # Generic check: if original fact text appears verbatim or nearly verbatim
        # (excluding cautious phrasing markers)
        original_keywords = set(original.lower().split())
        message_words = set(message_lower.split())
        # If >50% of original keywords appear without cautious markers
        cautious_markers = ["reported", "verification", "link not provided", "has reported"]
        has_cautious_marker = any(marker in message_lower for marker in cautious_markers)
        
        if not has_cautious_marker:
            overlap = len(original_keywords.intersection(message_words))
            if overlap > len(original_keywords) * 0.5 and len(original_keywords) > 3:
                violations.append(f"Definitive claim without cautious phrasing: '{original}'")
    
    return (len(violations) > 0, violations)

