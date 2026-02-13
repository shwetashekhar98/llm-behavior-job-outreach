"""
Profile-Driven Job Outreach LLM Evaluator - STAGE 1: Evidence Extraction
Extracts candidate facts with evidence quotes for human approval.
"""

import re
import json
import copy
from typing import List, Dict, Set, Optional
from groq import Groq
from datetime import datetime
from pathlib import Path
import os


def sanitize_profile_text(text: str) -> str:
    """Clean profile text by removing UI artifacts, duplicates, and noise."""
    if not text:
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    seen = set()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove lines shorter than 5 words
        words = line.split()
        if len(words) < 5:
            continue
        
        # Remove lines starting with lowercase fragments
        if re.match(r'^\s*[a-z]+\s+(that|across|and|or|the|a|an)\s+', line, re.IGNORECASE):
            continue
        
        # Remove navigation/UI text
        ui_patterns = [
            r'see more', r'show all', r'view all', r'expand', r'collapse',
            r'click', r'\d+ (likes?|comments?|shares?|impressions?|followers?|connections?)',
            r'• • •', r'\.\.\.',
        ]
        
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in ui_patterns):
            continue
        
        # Remove truncated phrases
        if re.search(r'\s+[a-z]{1,2}\s*$', line):
            continue
        
        # Remove duplicates
        line_lower = line.lower()
        if line_lower in seen:
            continue
        seen.add(line_lower)
        
        # Remove timestamps
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', line):
            continue
        
        cleaned_lines.append(line)
    
    cleaned = '\n'.join(cleaned_lines)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    
    return cleaned.strip()


def is_complete_fact(fact: str, category: str = "other", debug: bool = False) -> bool:
    """
    Validate if a fact is a complete claim.
    Accepts grammatical sentences including those starting with verbs.
    Allows acronyms, proper nouns, numbers, and parentheses.
    
    Args:
        fact: The fact text to validate
        category: Category of the fact (e.g., "links", "education", "work")
        debug: If True, print debug info about why fact passed/failed
    
    Returns:
        True if fact is complete, False otherwise
    """
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
        # Also accept if it mentions link-related keywords
        if any(keyword in fact_lower for keyword in ['github', 'linkedin', 'portfolio', 'website', 'profile']):
            if debug:
                print(f"[is_complete_fact DEBUG] ACCEPT: category='links' and contains link keyword")
            return True
    
    # AWARDS CATEGORY: Special handling for award facts
    if category.lower() == "awards" or any(keyword in fact_lower for keyword in ["award", "prize", "honor", "honour", "medal", "recognition"]):
        # Award-specific verbs
        award_verbs = ["won", "receive", "received", "awarded", "earned", "granted", "bestowed"]
        # Award-specific nouns
        award_nouns = ["award", "prize", "honor", "honour", "medal", "recognition", "distinction"]
        
        has_award_verb = any(verb in fact_lower for verb in award_verbs)
        has_award_noun = any(noun in fact_lower for noun in award_nouns)
        
        # If it's an award fact with proper verb and noun, use relaxed validation
        if has_award_verb and has_award_noun:
            # Must have at least 6 words
            if word_count >= 6:
                # Check it's not a fragment/ellipsis/keywords-only
                if '...' not in fact and not re.search(r'\.{2,}', fact):
                    short_words = [w for w in words if len(w) <= 2]
                    short_word_ratio = len(short_words) / word_count if word_count > 0 else 0
                    if short_word_ratio <= 0.5:
                        if debug:
                            print(f"[is_complete_fact DEBUG] ACCEPT: award fact with verb+noun | word_count={word_count}, has_award_verb={has_award_verb}, has_award_noun={has_award_noun}")
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
        'debugged', 'refactored', 'migrated', 'upgraded', 'monitored',
        'won', 'received', 'awarded'  # Add award verbs to general list too
    ]
    
    has_verb = any(verb in fact_lower for verb in verbs)
    matched_verbs = [verb for verb in verbs if verb in fact_lower]
    
    # Check for common resume patterns (even without explicit verb match)
    has_resume_pattern = bool(
        re.search(r'\b(at|as|in|for|with|from|to)\s+[A-Z]', fact) or  # "at Company", "as Role", "in Location"
        re.search(r'\b\d+\s+(years?|months?|days?)\b', fact_lower) or  # "4 years"
        re.search(r'\b(MS|M\.S\.|MBA|PhD|B\.A\.|B\.S\.|Master|Bachelor)\b', fact, re.IGNORECASE) or  # Degrees
        re.search(r'\b(GPA|grade|score|rating)\b', fact_lower) or  # Academic metrics
        re.search(r'https?://', fact_lower)  # URLs
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


def extract_links_from_text(combined_text: str) -> List[Dict]:
    """
    Deterministic link extraction (does NOT depend on LLM).
    Parse combined_text with regex for URLs and create link facts.
    
    Returns candidate_facts dicts of the SAME SHAPE as LLM facts:
    {
        "fact": "...",
        "category": "links",
        "evidence": "<exact url>",
        "confidence": 0.95
    }
    """
    # Support http(s) and bare domains
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, combined_text)
    
    link_facts = []
    seen_urls = set()  # Deduplicate by normalized URL (strip trailing slash)
    
    for url in urls:
        # Normalize URL (strip trailing slash for deduplication)
        url_normalized = url.rstrip('/').lower()
        
        # Skip if already seen
        if url_normalized in seen_urls:
            continue
        seen_urls.add(url_normalized)
        
        url_lower = url.lower()
        category = "links"
        fact_text = None
        
        # GitHub detection
        if 'github.com' in url_lower:
            fact_text = f"GitHub profile: {url}"
        
        # LinkedIn detection (must be /in/ profile)
        elif 'linkedin.com/in' in url_lower:
            fact_text = f"LinkedIn profile: {url}"
        
        # Portfolio detection: netlify.app, github.io, or personal domain
        elif 'netlify.app' in url_lower or 'github.io' in url_lower:
            fact_text = f"Profile link: {url}"
        
        # If no specific match, treat as portfolio (personal domain)
        else:
            # Check if it's not GitHub or LinkedIn (already handled above)
            if 'github.com' not in url_lower and 'linkedin.com' not in url_lower:
                fact_text = f"Profile link: {url}"
        
        if fact_text:
            link_facts.append({
                "fact": fact_text,
                "category": category,
                "evidence": url,  # Exact URL string from combined_text
                "confidence": 0.95
            })
    
    return link_facts


# Keep old function name for backward compatibility
def extract_urls(text: str) -> List[Dict]:
    """Alias for extract_links_from_text (backward compatibility)."""
    return extract_links_from_text(text)


def extract_candidate_facts(
    profile_input: Dict,
    api_key: str,
    model: str = "llama-3.1-8b-instant",
    show_debug: bool = False
) -> Dict:
    """
    STAGE 1: Extract candidate facts with evidence.
    
    Returns JSON matching Stage 1 schema:
    {
        "stage": 1,
        "profile_parse_quality": "structured" | "semi_structured" | "messy_unstructured",
        "candidate_facts": [...],
        "warnings": [...]
    }
    """
    warnings = []
    candidate_facts = []
    
    # Determine parse quality
    structured_fields = profile_input.get("structured_fields", {})
    unstructured_text = profile_input.get("unstructured_text", "")
    links = profile_input.get("links", {})
    
    has_structured = bool(structured_fields)
    has_unstructured = bool(unstructured_text and len(unstructured_text.strip()) > 50)
    
    if has_structured and not has_unstructured:
        parse_quality = "structured"
    elif has_structured and has_unstructured:
        parse_quality = "semi_structured"
    else:
        parse_quality = "messy_unstructured"
    
    # Combine all input sources
    all_text_parts = []
    
    # Structured fields
    for key, value in structured_fields.items():
        if value and isinstance(value, str):
            all_text_parts.append(f"{key}: {value}")
    
    # Unstructured text
    if unstructured_text:
        sanitized = sanitize_profile_text(unstructured_text)
        if sanitized:
            all_text_parts.append(sanitized)
    
    # Links
    for link_type, url in links.items():
        if url:
            all_text_parts.append(f"{link_type}: {url}")
    
    combined_text = "\n\n".join(all_text_parts)
    
    if not combined_text or len(combined_text.strip()) < 20:
        return {
            "stage": 1,
            "profile_parse_quality": parse_quality,
            "candidate_facts": [],
            "warnings": ["No meaningful profile content found"]
        }
    
    # Extract deterministic link facts (does NOT depend on LLM)
    # This happens BEFORE LLM extraction to ensure links are always captured
    deterministic_link_facts = extract_links_from_text(combined_text)
    
    # Debug counter
    num_deterministic_link_facts = len(deterministic_link_facts)
    
    # Use LLM for fact extraction
    client = Groq(api_key=api_key)
    
    system_prompt = """You are a strict evidence-based fact extraction system.

STAGE 1: PROFILE INPUT → EVIDENCE EXTRACTION

Extract candidate facts with evidence quotes for human approval.

CRITICAL RULES:
1. A "fact" must be a COMPLETE CLAIM (subject + verb + object), e.g.:
   - BAD: "Azure", "ms & ml pipelines", "monitoring"
   - GOOD: "Led backend and API initiatives on Azure."

2. Reject any candidate fact that:
   - is < 8 words, OR
   - is a fragment, OR
   - is mostly keywords, OR
   - contains ellipses/garbled truncation, OR
   - duplicates another fact

3. Every candidate fact must include:
   - fact: one sentence, no commas splicing multiple claims
   - evidence: an exact quote snippet from the input (<= 160 chars)
   - confidence: 0.50–0.95 (not 1.0 unless extremely explicit)
   - category: one of ["education","work","impact","skills","projects","awards","links","location","other"]

4. If years or numbers are mentioned, keep them only if explicitly present (do not infer).

5. NEVER fabricate. NEVER guess. NEVER assume.

Return JSON:
{
  "candidate_facts": [
    {
      "fact": "Complete claim statement",
      "category": "education|work|impact|skills|projects|awards|links|location|other",
      "evidence": "Exact quote from input (max 160 chars)",
      "confidence": 0.75
    }
  ],
  "warnings": ["Any extraction issues"]
}"""

    user_prompt = f"""Extract candidate facts from this profile input:

{combined_text}

Return JSON with candidate_facts array. Only include complete claims with evidence."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content or "{}"
        result_json = json.loads(result_text)
        
        # Extract raw LLM response - CREATE DEEP COPY IMMEDIATELY
        raw_candidate_facts = result_json.get("candidate_facts", [])
        raw_warnings = result_json.get("warnings", [])
        warnings.extend(raw_warnings)
        
        # Debug counter
        num_llm_candidate_facts = len(raw_candidate_facts)
        
        # CRITICAL: Create deep copy of raw facts to preserve original fact-evidence pairs
        raw_candidate_facts_deep = copy.deepcopy(raw_candidate_facts)
        
        # MERGE: Combine LLM facts with deterministic link facts
        # Deduplicate by URL (normalize URLs by stripping trailing slash)
        llm_urls = set()
        for fact in raw_candidate_facts_deep:
            # Extract URLs from LLM facts
            fact_text = fact.get("fact", "")
            evidence = fact.get("evidence", "")
            url_matches = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', fact_text + " " + evidence)
            for url in url_matches:
                llm_urls.add(url.rstrip('/').lower())
        
        # Add deterministic link facts that aren't already in LLM facts
        merged_link_facts = []
        for link_fact in deterministic_link_facts:
            link_url = link_fact.get("evidence", "")
            link_url_normalized = link_url.rstrip('/').lower()
            if link_url_normalized not in llm_urls:
                merged_link_facts.append(link_fact)
        
        # Merge: LLM facts + deterministic link facts (deduplicated)
        merged_candidate_facts = raw_candidate_facts_deep + merged_link_facts
        num_merged_candidate_facts = len(merged_candidate_facts)
        
        # Debug output
        if show_debug or DEBUG_STAGE1:
            print(f"[DEBUG_STAGE1] Link extraction stats:")
            print(f"  - LLM candidate facts: {num_llm_candidate_facts}")
            print(f"  - Deterministic link facts: {num_deterministic_link_facts}")
            print(f"  - Merged candidate facts: {num_merged_candidate_facts}")
            if merged_link_facts:
                print(f"  - Merged link facts:")
                for link_fact in merged_link_facts:
                    print(f"    * {link_fact.get('fact', '')}")
        
        # Use merged facts for validation
        raw_candidate_facts_deep = merged_candidate_facts
        
        # ============================================================================
        # DEBUG: Store raw LLM output for UI display (don't display here - UI will handle it)
        # ============================================================================
        # Raw output will be shown in UI after extraction completes
        
        # ============================================================================
        # DEBUG LOGGING: Store raw LLM response for file logging (if env var set)
        # ============================================================================
        DEBUG_STAGE1 = os.getenv("DEBUG_STAGE1", "False").lower() == "true"
        
        # Store debug info for UI display - use deep copy
        debug_info = {
            "raw_candidate_facts": copy.deepcopy(raw_candidate_facts_deep) if show_debug else [],
            "rejected_facts": [],
            "accepted_facts": [],
            "processed_candidate_facts": [],  # For comparison
            "num_llm_candidate_facts": num_llm_candidate_facts if 'num_llm_candidate_facts' in locals() else 0,
            "num_deterministic_link_facts": num_deterministic_link_facts if 'num_deterministic_link_facts' in locals() else 0,
            "num_merged_candidate_facts": num_merged_candidate_facts if 'num_merged_candidate_facts' in locals() else 0,
            "merged_link_facts": merged_link_facts if 'merged_link_facts' in locals() else []
        }
        
        if DEBUG_STAGE1:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_dir = Path(__file__).parent.parent / "debug_logs"
            debug_dir.mkdir(exist_ok=True)
            
            # Dump raw candidate_facts from LLM
            raw_debug = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "raw_candidate_facts": raw_candidate_facts,
                "extraction_warnings": raw_warnings,
                "total_raw_facts": len(raw_candidate_facts)
            }
            
            raw_debug_file = debug_dir / f"debug_stage1_raw_{run_id}.json"
            with open(raw_debug_file, "w") as f:
                json.dump(raw_debug, f, indent=2)
            
            print(f"[DEBUG_STAGE1] Raw LLM response saved to: {raw_debug_file}")
            print(f"[DEBUG_STAGE1] Total raw candidate_facts from LLM: {len(raw_candidate_facts)}")
            for idx, fact in enumerate(raw_candidate_facts):
                print(f"[DEBUG_STAGE1] Raw fact {idx+1}: {fact.get('fact', '')[:50]}... (category: {fact.get('category', 'unknown')}, confidence: {fact.get('confidence', 0)})")
        
        # Validate and filter facts
        seen_facts = set()
        validated_facts = []
        rejected_facts = []  # DEBUG: Track rejected facts for file logging
        
        # DEBUG: Collect accepted and rejected for UI display
        accepted = []
        rejected = []
        
        # INTEGRITY CHECK: Track original fact-evidence pairs
        integrity_warnings = []
        
        for idx, fact_data in enumerate(raw_candidate_facts_deep):
            # CRITICAL: Create deep copy of fact_data to prevent mutations
            fact_data = copy.deepcopy(fact_data)
            
            # Extract fields - preserve original evidence
            fact_text = (fact_data.get("fact") or "").strip()
            original_evidence = fact_data.get("evidence", "").strip()
            # Only use fact_text as evidence fallback if evidence is completely missing
            evidence = original_evidence if original_evidence else fact_text
            confidence = fact_data.get("confidence", 0.5)
            category = fact_data.get("category", "other")
            
            # AWARD SUBJECT NORMALIZATION: If category=="awards" and fact starts with "Won "
            # Rewrite to "I " + original fact (keep evidence unchanged)
            if category.lower() == "awards" and fact_text.lower().startswith("won "):
                fact_text = "I " + fact_text
                # Update fact_data with normalized fact
                fact_data["fact"] = fact_text
            
            # INTEGRITY CHECK 1: Evidence must exist
            if not evidence or len(evidence) == 0:
                integrity_warnings.append(f"Fact {idx+1}: Empty evidence for fact '{fact_text[:50]}...'")
            
            # INTEGRITY CHECK 2: Evidence should be in source text (unless it's a fallback)
            if evidence and evidence != fact_text:
                if evidence.lower() not in combined_text.lower():
                    integrity_warnings.append(
                        f"Fact {idx+1}: Evidence not found in source text. "
                        f"Fact: '{fact_text[:50]}...' Evidence: '{evidence[:50]}...'"
                    )
            
            # INTEGRITY CHECK 3: Evidence should match fact category context
            # (Basic heuristic: if fact mentions "research" but evidence mentions "implementation", flag it)
            fact_lower = fact_text.lower()
            evidence_lower = evidence.lower()
            category_keywords = {
                "research": ["research", "paper", "publication", "study", "analysis"],
                "work": ["worked", "implemented", "built", "developed", "designed"],
                "education": ["degree", "graduated", "studied", "university", "college"]
            }
            # This is a soft check - just log, don't reject
            if category in ["work", "projects"] and any(kw in evidence_lower for kw in category_keywords["research"]):
                if not any(kw in fact_lower for kw in category_keywords["research"]):
                    integrity_warnings.append(
                        f"Fact {idx+1}: Possible evidence mismatch - fact is '{category}' but evidence mentions research. "
                        f"Fact: '{fact_text[:50]}...' Evidence: '{evidence[:50]}...'"
                    )
            reasons = []  # Collect rejection reasons
            
            # Check confidence
            if confidence < 0.50:
                reasons.append("confidence_below_0.50")
            
            # Check fact length
            if len(fact_text) < 10:
                reasons.append("fact_too_short")
            
            # Check if evidence exists in source text
            if evidence and evidence.lower() not in combined_text.lower():
                reasons.append("evidence_not_substring_match")
            
            # Validate fact completeness (pass category for link handling)
            # Enable debug if show_debug is True
            is_complete = is_complete_fact(fact_text, category=category, debug=show_debug)
            if not is_complete:
                reasons.append("is_complete_fact_check_failed")
            
            # Check for duplicates
            fact_lower = fact_text.lower()
            if fact_lower in seen_facts:
                reasons.append("duplicate_fact")
            
            # If any reasons, reject; otherwise accept
            if reasons:
                # Create deep copy with rejection reasons
                rejected_item = copy.deepcopy(fact_data)
                rejected_item["rejection_reasons"] = reasons
                rejected_item["_original_index"] = idx
                rejected_item["_original_evidence"] = original_evidence
                rejected.append(rejected_item)
                if DEBUG_STAGE1:
                    rejected_facts.append(copy.deepcopy(rejected_item))
                continue
            
            # Fact passed all checks
            seen_facts.add(fact_lower)
            
            # Clamp confidence if needed
            if confidence > 0.95:
                confidence = 0.95
            
            # Check evidence length
            if len(evidence) > 160:
                evidence = evidence[:157] + "..."
            
            # Validate category
            valid_categories = ["education", "work", "impact", "skills", "projects", 
                              "awards", "links", "location", "other"]
            if category not in valid_categories:
                category = "other"
            
            # Add to accepted - CREATE NEW DICT (not reference to fact_data)
            # This ensures fact-evidence pairing is preserved
            accepted_item = {
                "fact": fact_text,
                "category": category,
                "evidence": evidence,  # Use the validated evidence
                "confidence": confidence,
                "_original_index": idx,  # Track original position for debugging
                "_original_evidence": original_evidence  # Preserve original evidence for comparison
            }
            # Use deep copy to prevent any mutations
            accepted.append(copy.deepcopy(accepted_item))
            validated_facts.append(copy.deepcopy(accepted_item))
            
            # DEBUG: Store accepted fact for file logging
            if DEBUG_STAGE1:
                debug_info["accepted_facts"].append(accepted_item)
        
        # ============================================================================
        # DEBUG: Store accepted and rejected for UI display (UI will handle display)
        # ============================================================================
        # Debug info will be returned and displayed in UI after extraction completes
        
        # ============================================================================
        # DEBUG LOGGING: Dump rejected and accepted facts to files (if env var set)
        # ============================================================================
        if DEBUG_STAGE1:
            # Save rejected facts
            rejected_debug = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "rejected_facts": rejected_facts,
                "total_rejected": len(rejected_facts)
            }
            rejected_debug_file = debug_dir / f"debug_stage1_rejected_{run_id}.json"
            with open(rejected_debug_file, "w") as f:
                json.dump(rejected_debug, f, indent=2)
            
            print(f"[DEBUG_STAGE1] Rejected facts saved to: {rejected_debug_file}")
            print(f"[DEBUG_STAGE1] Total rejected facts: {len(rejected_facts)}")
            for idx, rejected in enumerate(rejected_facts):
                print(f"[DEBUG_STAGE1] Rejected {idx+1}: {rejected.get('fact', '')[:50]}... Reasons: {', '.join(rejected.get('rejection_reasons', []))}")
            
            # Save accepted facts
            accepted_debug = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "accepted_facts": [
                    {
                        "fact": f["fact"],
                        "category": f["category"],
                        "confidence": f["confidence"],
                        "evidence": f["evidence"]
                    }
                    for f in validated_facts
                ],
                "total_accepted": len(validated_facts)
            }
            accepted_debug_file = debug_dir / f"debug_stage1_accepted_{run_id}.json"
            with open(accepted_debug_file, "w") as f:
                json.dump(accepted_debug, f, indent=2)
            
            print(f"[DEBUG_STAGE1] Accepted facts saved to: {accepted_debug_file}")
            print(f"[DEBUG_STAGE1] Total accepted facts: {len(validated_facts)}")
            print(f"[DEBUG_STAGE1] Summary: {len(raw_candidate_facts)} raw → {len(validated_facts)} accepted → {len(rejected_facts)} rejected")
        
        # Add validated facts to candidate_facts
        # Note: Deterministic link facts were already merged with LLM facts before validation
        # So validated_facts may include link facts that passed validation
        for fact in validated_facts:
            candidate_facts.append(copy.deepcopy(fact))
        
        # INTEGRITY CHECK: Verify fact-evidence alignment in final candidate_facts
        if show_debug or DEBUG_STAGE1:
            for idx, fact in enumerate(candidate_facts):
                fact_text = fact.get("fact", "")
                evidence = fact.get("evidence", "")
                original_evidence = fact.get("_original_evidence", "")
                
                # Check if evidence was modified
                if original_evidence and evidence != original_evidence and evidence != fact_text:
                    integrity_warnings.append(
                        f"Final fact {idx+1}: Evidence was modified. "
                        f"Original: '{original_evidence[:50]}...' Current: '{evidence[:50]}...'"
                    )
        
        # Store processed facts for comparison
        if show_debug:
            debug_info["processed_candidate_facts"] = copy.deepcopy(candidate_facts)
        
        if len(candidate_facts) == 0:
            warnings.append("No valid facts extracted. Input may be too fragmented or incomplete.")
        
        result = {
            "stage": 1,
            "profile_parse_quality": parse_quality,
            "candidate_facts": candidate_facts,
            "warnings": warnings
        }
        
        # Add debug info if requested (for UI display)
        if show_debug:
            result["debug_info"] = {
                "raw_candidate_facts": copy.deepcopy(raw_candidate_facts_deep),
                "raw_warnings": raw_warnings,
                "accepted_facts": copy.deepcopy(accepted),
                "rejected_facts": copy.deepcopy(rejected),
                "processed_candidate_facts": copy.deepcopy(candidate_facts),
                "integrity_warnings": integrity_warnings,
                "combined_text_length": len(combined_text)
            }
        
        return result
    
    except Exception as e:
        warnings.append(f"Extraction error: {str(e)}")
        return {
            "stage": 1,
            "profile_parse_quality": parse_quality,
            "candidate_facts": candidate_facts,  # At least URL facts if any
            "warnings": warnings
        }


def prepare_approved_facts(
    approved_facts: List[str],
    rejected_facts: List[str],
    manual_facts: Optional[List[str]] = None
) -> Dict:
    """
    STAGE 2: Prepare final fact set from user approvals.
    
    Returns JSON matching Stage 2 schema:
    {
        "stage": 2,
        "approved_facts_final": [...],
        "link_facts": {...},
        "notes_for_generation": [...]
    }
    """
    manual_facts = manual_facts or []
    
    # Merge approved + manual facts
    all_facts = approved_facts + manual_facts
    
    # Dedupe
    seen = set()
    approved_facts_final = []
    for fact in all_facts:
        fact_lower = fact.lower()
        if fact_lower not in seen:
            seen.add(fact_lower)
            approved_facts_final.append(fact)
    
    # Extract link facts
    link_facts = {
        "github": None,
        "portfolio": None,
        "linkedin": None,
        "other_links": []
    }
    
    notes_for_generation = []
    
    for fact in approved_facts_final:
        fact_lower = fact.lower()
        
        # Extract GitHub URL
        github_match = re.search(r'github[:\s]+(https?://[^\s]+)', fact_lower)
        if github_match:
            link_facts["github"] = github_match.group(1)
        elif 'github' in fact_lower and 'http' in fact_lower:
            url_match = re.search(r'https?://[^\s]+', fact)
            if url_match:
                link_facts["github"] = url_match.group(0)
        
        # Extract LinkedIn URL
        linkedin_match = re.search(r'linkedin[:\s]+(https?://[^\s]+)', fact_lower)
        if linkedin_match:
            link_facts["linkedin"] = linkedin_match.group(1)
        elif 'linkedin' in fact_lower and 'http' in fact_lower:
            url_match = re.search(r'https?://[^\s]+', fact)
            if url_match:
                link_facts["linkedin"] = url_match.group(0)
        
        # Extract Portfolio URL
        portfolio_match = re.search(r'(?:portfolio|website|personal)[:\s]+(https?://[^\s]+)', fact_lower)
        if portfolio_match:
            link_facts["portfolio"] = portfolio_match.group(1)
        elif ('portfolio' in fact_lower or 'website' in fact_lower) and 'http' in fact_lower:
            url_match = re.search(r'https?://[^\s]+', fact)
            if url_match:
                link_facts["portfolio"] = url_match.group(0)
        
        # Other links
        if 'http' in fact_lower and not any(link in fact_lower for link in ['github', 'linkedin', 'portfolio']):
            url_match = re.search(r'https?://[^\s]+', fact)
            if url_match:
                link_facts["other_links"].append(url_match.group(0))
    
    # Generate notes
    if not link_facts["github"]:
        notes_for_generation.append("GitHub link not available - use placeholder if required")
    if not link_facts["portfolio"]:
        notes_for_generation.append("Portfolio link not available - use placeholder if required")
    if not link_facts["linkedin"]:
        notes_for_generation.append("LinkedIn link not available - use placeholder if required")
    
    if len(approved_facts_final) < 3:
        notes_for_generation.append("Limited approved facts - focus on strongest 1-2 facts in messages")
    
    return {
        "stage": 2,
        "approved_facts_final": approved_facts_final,
        "link_facts": link_facts,
        "notes_for_generation": notes_for_generation
    }


# Legacy compatibility functions
def extract_facts_with_evidence(
    profile_input: Dict,
    api_key: str,
    model: str = "llama-3.1-8b-instant",
    show_debug: bool = False
) -> List[Dict]:
    """
    Legacy wrapper that returns facts in UI format.
    """
    stage1_result = extract_candidate_facts(profile_input, api_key, model, show_debug)
    
    # Convert to UI format - PRESERVE FACT-EVIDENCE PAIRING
    facts_for_ui = []
    for idx, fact_data in enumerate(stage1_result["candidate_facts"]):
        # CRITICAL: Use fact_data directly - don't extract separately
        fact_text = fact_data.get("fact", "")
        evidence = fact_data.get("evidence", "")
        
        # INTEGRITY CHECK: Verify fact-evidence pairing
        if show_debug:
            original_index = fact_data.get("_original_index", idx)
            original_evidence = fact_data.get("_original_evidence", evidence)
            if original_evidence and evidence != original_evidence and evidence != fact_text:
                # This should not happen if our code is correct
                import streamlit as st
                st.warning(
                    f"⚠️ Evidence mismatch detected for fact {idx+1} (original index {original_index}): "
                    f"Fact: '{fact_text[:50]}...' Evidence: '{evidence[:50]}...' "
                    f"Original evidence: '{original_evidence[:50]}...'"
                )
        
        facts_for_ui.append({
            "value": fact_text,
            "source_quote": evidence,  # Use the evidence from fact_data
            "start_index": 0,
            "end_index": len(evidence),
            "confidence": fact_data.get("confidence", 0.5),
            "category": fact_data.get("category", "other").title(),
            "evidence_source": "extracted",
            "_original_index": fact_data.get("_original_index", idx)  # Preserve for debugging
        })
    
    # Return debug info if requested (for UI display)
    if show_debug and "debug_info" in stage1_result:
        debug_info = stage1_result["debug_info"]
        # Ensure rejected_facts have rejection_reasons
        if "rejected_facts" in debug_info:
            # Verify rejected_facts structure
            for rejected in debug_info["rejected_facts"]:
                if "rejection_reasons" not in rejected:
                    rejected["rejection_reasons"] = ["unknown_reason"]
        # Return tuple: (facts_list, debug_info_dict)
        return (facts_for_ui, debug_info)
    
    return facts_for_ui


def extract_structured_profile(form_data: Dict) -> List[Dict]:
    """Extract facts from structured form input."""
    facts = []
    
    category_mapping = {
        "education": "Education",
        "work_experience": "Work Experience",
        "skills": "Technical Skills"
    }
    
    for key, value in form_data.items():
        if value and isinstance(value, str) and value.strip():
            value_clean = value.strip()
            if is_complete_fact(value_clean) or len(value_clean.split()) >= 2:
                facts.append({
                    "value": value_clean,
                    "source_quote": value_clean,
                    "start_index": 0,
                    "end_index": len(value_clean),
                    "confidence": 1.0,
                    "category": category_mapping.get(key, "Other"),
                    "evidence_source": f"structured_field:{key}",
                    "tags": [category_mapping.get(key, "Other").lower().replace(" ", "_")]
                })
    
    return facts


def validate_fact_evidence(fact: Dict, source_text: str) -> bool:
    """Validate that a fact's source_quote exists in source text."""
    source_quote = fact.get("source_quote", "")
    if not source_quote:
        return False
    return source_quote.lower() in source_text.lower()
