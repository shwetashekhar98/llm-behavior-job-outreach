"""
Profile-Driven Job Outreach LLM Evaluator - STAGE 1: Evidence Extraction
Extracts candidate facts with evidence quotes for human approval.
"""

import re
import json
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


def is_complete_fact(fact: str) -> bool:
    """
    Validate if a fact is a complete claim (subject + verb + object).
    Reject fragments, keywords, and incomplete phrases.
    """
    if not fact or len(fact.strip()) < 8:
        return False
    
    words = fact.split()
    if len(words) < 8:  # Must be at least 8 words
        return False
    
    fact_lower = fact.lower()
    
    # Reject fragments
    if re.match(r'^\w+\s+(that|across|and|or)\s+\w+$', fact_lower):
        return False
    
    # Reject if mostly keywords (too many single-word items)
    if len([w for w in words if len(w) <= 3]) > len(words) * 0.4:
        return False
    
    # Reject ellipses/garbled truncation
    if '...' in fact or re.search(r'\.{2,}', fact):
        return False
    
    # Must have a verb (indicates a claim)
    verbs = ['worked', 'led', 'built', 'developed', 'created', 'designed', 
             'implemented', 'graduated', 'earned', 'completed', 'studied',
             'attended', 'based', 'located', 'achieved', 'improved', 'reduced',
             'increased', 'managed', 'collaborated', 'delivered']
    if not any(verb in fact_lower for verb in verbs):
        # Check if it's a location or link fact
        if not (re.search(r'http', fact_lower) or re.search(r'\b(based in|located in|from)\b', fact_lower)):
            return False
    
    return True


def extract_urls(text: str) -> List[Dict]:
    """Extract URLs from text and return as link facts."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    
    link_facts = []
    for url in urls:
        url_lower = url.lower()
        category = "links"
        
        if 'github' in url_lower:
            fact_text = f"GitHub profile: {url}"
        elif 'linkedin' in url_lower:
            fact_text = f"LinkedIn profile: {url}"
        elif 'portfolio' in url_lower or 'personal' in url_lower or 'website' in url_lower:
            fact_text = f"Portfolio/website: {url}"
        else:
            fact_text = f"Profile link: {url}"
        
        link_facts.append({
            "fact": fact_text,
            "category": category,
            "evidence": url,
            "confidence": 0.95  # URLs are explicit
        })
    
    return link_facts


def extract_candidate_facts(
    profile_input: Dict,
    api_key: str,
    model: str = "llama-3.1-8b-instant"
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
    
    # Extract URLs first (always high confidence)
    url_facts = extract_urls(combined_text)
    candidate_facts.extend(url_facts)
    
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
        
        extracted_facts = result_json.get("candidate_facts", [])
        extraction_warnings = result_json.get("warnings", [])
        warnings.extend(extraction_warnings)
        
        # ============================================================================
        # DEBUG LOGGING: Dump raw LLM response
        # ============================================================================
        DEBUG_STAGE1 = os.getenv("DEBUG_STAGE1", "False").lower() == "true"
        
        if DEBUG_STAGE1:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_dir = Path(__file__).parent.parent / "debug_logs"
            debug_dir.mkdir(exist_ok=True)
            
            # Dump raw candidate_facts from LLM
            raw_debug = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "raw_candidate_facts": extracted_facts,
                "extraction_warnings": extraction_warnings,
                "total_raw_facts": len(extracted_facts)
            }
            
            raw_debug_file = debug_dir / f"debug_stage1_raw_{run_id}.json"
            with open(raw_debug_file, "w") as f:
                json.dump(raw_debug, f, indent=2)
            
            print(f"[DEBUG_STAGE1] Raw LLM response saved to: {raw_debug_file}")
            print(f"[DEBUG_STAGE1] Total raw candidate_facts from LLM: {len(extracted_facts)}")
            for idx, fact in enumerate(extracted_facts):
                print(f"[DEBUG_STAGE1] Raw fact {idx+1}: {fact.get('fact', '')[:50]}... (category: {fact.get('category', 'unknown')}, confidence: {fact.get('confidence', 0)})")
        
        # Validate and filter facts
        seen_facts = set()
        validated_facts = []
        rejected_facts = []  # DEBUG: Track rejected facts
        
        for fact_data in extracted_facts:
            fact_text = fact_data.get("fact", "").strip()
            evidence = fact_data.get("evidence", fact_text).strip()
            confidence = fact_data.get("confidence", 0.5)
            category = fact_data.get("category", "other")
            
            rejection_reasons = []  # DEBUG: Track why this fact was rejected
            
            # Validate fact completeness
            if not is_complete_fact(fact_text):
                rejection_reasons.append("is_complete_fact check failed")
                if DEBUG_STAGE1:
                    rejected_facts.append({
                        "fact": fact_text,
                        "category": category,
                        "confidence": confidence,
                        "evidence": evidence,
                        "rejection_reasons": rejection_reasons
                    })
                continue
            
            # Check confidence range
            if confidence < 0.50:
                rejection_reasons.append(f"confidence too low: {confidence} < 0.50")
                if DEBUG_STAGE1:
                    rejected_facts.append({
                        "fact": fact_text,
                        "category": category,
                        "confidence": confidence,
                        "evidence": evidence,
                        "rejection_reasons": rejection_reasons
                    })
                continue
            
            if confidence > 0.95:
                confidence = 0.95  # Clamp to max
            
            # Check evidence length
            if len(evidence) > 160:
                evidence = evidence[:157] + "..."
            
            # Check if evidence exists in source text
            if evidence.lower() not in combined_text.lower():
                rejection_reasons.append(f"evidence quote not found in source text")
                if DEBUG_STAGE1:
                    rejected_facts.append({
                        "fact": fact_text,
                        "category": category,
                        "confidence": confidence,
                        "evidence": evidence,
                        "rejection_reasons": rejection_reasons
                    })
                continue
            
            # Dedupe by semantic similarity (simple: exact match)
            fact_lower = fact_text.lower()
            if fact_lower in seen_facts:
                rejection_reasons.append("duplicate fact (exact match)")
                if DEBUG_STAGE1:
                    rejected_facts.append({
                        "fact": fact_text,
                        "category": category,
                        "confidence": confidence,
                        "evidence": evidence,
                        "rejection_reasons": rejection_reasons
                    })
                continue
            seen_facts.add(fact_lower)
            
            # Validate category
            valid_categories = ["education", "work", "impact", "skills", "projects", 
                              "awards", "links", "location", "other"]
            if category not in valid_categories:
                category = "other"
            
            # Fact passed all validations
            validated_facts.append({
                "fact": fact_text,
                "category": category,
                "evidence": evidence,
                "confidence": confidence
            })
            
            # DEBUG: Log accepted fact
            if DEBUG_STAGE1:
                print(f"[DEBUG_STAGE1] ✓ Accepted fact: {fact_text[:50]}... (category: {category}, confidence: {confidence})")
        
        # ============================================================================
        # DEBUG LOGGING: Dump rejected and accepted facts
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
                print(f"[DEBUG_STAGE1] Rejected {idx+1}: {rejected['fact'][:50]}... Reasons: {', '.join(rejected['rejection_reasons'])}")
            
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
            print(f"[DEBUG_STAGE1] Summary: {len(extracted_facts)} raw → {len(validated_facts)} accepted → {len(rejected_facts)} rejected")
        
        # Merge with URL facts (avoid duplicates)
        url_fact_texts = {f["fact"].lower() for f in url_facts}
        for fact in validated_facts:
            if fact["fact"].lower() not in url_fact_texts:
                candidate_facts.append(fact)
        
        if len(candidate_facts) == 0:
            warnings.append("No valid facts extracted. Input may be too fragmented or incomplete.")
        
        return {
            "stage": 1,
            "profile_parse_quality": parse_quality,
            "candidate_facts": candidate_facts,
            "warnings": warnings
        }
    
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
    model: str = "llama-3.1-8b-instant"
) -> List[Dict]:
    """
    Legacy wrapper that returns facts in UI format.
    """
    stage1_result = extract_candidate_facts(profile_input, api_key, model)
    
    # Convert to UI format
    facts_for_ui = []
    for idx, fact_data in enumerate(stage1_result["candidate_facts"]):
        facts_for_ui.append({
            "value": fact_data["fact"],
            "source_quote": fact_data["evidence"],
            "start_index": 0,
            "end_index": len(fact_data["evidence"]),
            "confidence": fact_data["confidence"],
            "category": fact_data["category"].title(),
            "evidence_source": "extracted"
        })
    
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
