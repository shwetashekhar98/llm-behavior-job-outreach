"""
Evaluation runner implementing PHASE 3 and PHASE 4.
PHASE 3: Message Generation with word count enforcement
PHASE 4: Evaluation
"""

import re
from typing import Dict, List
from groq import Groq
from validation_engine import run_all_checks


def extract_confidence(text: str) -> float:
    """Extract confidence score from model output."""
    pattern = r'Confidence:\s*([0-9]*\.?[0-9]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except:
            return 0.5
    return 0.5


def count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def generate_message_with_word_limit(
    client: Groq,
    scenario: Dict,
    approved_facts_final: List[str],
    link_facts: Dict,
    model: str,
    run_idx: int,
    max_attempts: int = 3
) -> Dict:
    """
    PHASE 3: Generate message with strict word limit enforcement.
    
    Rules:
    - Do NOT introduce new facts
    - Do NOT exaggerate
    - Do NOT invent metrics
    - Respect max_words strictly
    - If message exceeds word limit, rewrite until within limit
    - Tone must be professional
    - Must include required items
    
    Returns:
        Dictionary with message, word_count, and confidence_score
    """
    channel = scenario.get("channel", "email")
    company = scenario.get("company", "")
    target_role = scenario.get("target_role", "")
    recipient_type = scenario.get("recipient_type", "recruiter")
    tone = scenario.get("tone", "professional")
    max_words = scenario.get("max_words", 150)
    must_include = scenario.get("must_include", [])
    notes = scenario.get("notes", "")
    
    system_prompt = f"""You are a reliability-focused message generation system.

PHASE 3: MESSAGE GENERATION

Generate a professional outreach message based ONLY on APPROVED_FACTS.

CRITICAL RELIABILITY RULES:
A) You do NOT know the user's resume unless explicitly provided in approved_facts.
B) You MUST NOT invent metrics, employers, degrees, dates, awards, publications, locations.
C) You may only use facts from approved_facts list.
D) If a required "must_include" item is missing from approved facts (e.g., Portfolio URL not provided), 
   you may include the WORD "Portfolio" or a placeholder like "[Portfolio link]", but you must NOT fabricate a URL.

STRICT CONSTRAINTS:
- Do NOT introduce new facts
- Do NOT exaggerate
- Do NOT invent metrics
- Respect max_words strictly ({max_words} words maximum)
- If message exceeds word limit, rewrite until within limit
- Tone must be professional
- Must include required items: {', '.join(must_include) if must_include else 'None specified'}

Target: {recipient_type} at {company} for {target_role} role
Channel: {channel}
Tone: {tone}
Maximum words: {max_words} (STRICT - count words and stay under)
Approved facts ONLY: {', '.join(approved_facts_final) if approved_facts_final else 'None provided'}

Available links:
- GitHub: {link_facts.get('github', 'Not available')}
- Portfolio: {link_facts.get('portfolio', 'Not available')}
- LinkedIn: {link_facts.get('linkedin', 'Not available')}

If a link is required but not available, use placeholder like "GitHub link available on request" - DO NOT fabricate URLs.

Channel conventions:
- email: include subject + greeting + sign-off
- linkedin_dm: short, direct, no subject

Before returning:
- Count words
- If over limit → rewrite shorter version
- Ensure all required items are included

Return message AND at the end:
Word Count: <number>
Confidence: <number between 0 and 1> (calibrated; do NOT output high confidence if constraints not met)

Reliability > Impressiveness. Never fabricate. Never guess. Never assume."""

    user_prompt = f"""Generate a {channel} message:

{notes if notes else 'General outreach message'}

Company: {company}
Role: {target_role}
Recipient: {recipient_type}

Requirements:
- {tone} tone, max {max_words} words (STRICT LIMIT)
- Must include: {', '.join(must_include) if must_include else 'None'}
- Only use these facts: {', '.join(allowed_facts) if allowed_facts else 'None provided'}

Generate a concise, professional message within the word limit."""

    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            message = response.choices[0].message.content or ""
            
            # Extract word count and confidence from message
            word_count_match = re.search(r'Word Count:\s*(\d+)', message, re.IGNORECASE)
            if word_count_match:
                reported_word_count = int(word_count_match.group(1))
            else:
                reported_word_count = count_words(message)
            
            # Remove metadata from message
            message_clean = re.sub(r'Word Count:.*', '', message, flags=re.IGNORECASE)
            message_clean = re.sub(r'Confidence:.*', '', message_clean, flags=re.IGNORECASE)
            message_clean = message_clean.strip()
            
            actual_word_count = count_words(message_clean)
            confidence = extract_confidence(message)
            
            # If over limit and not last attempt, try again with stricter prompt
            if actual_word_count > max_words and attempt < max_attempts - 1:
                system_prompt += f"\n\nIMPORTANT: Previous attempt had {actual_word_count} words. You MUST generate a message with {max_words} words or fewer. Be more concise."
                continue
            
            return {
                "message": message_clean,
                "word_count": actual_word_count,
                "confidence_score": confidence,
                "error": None
            }
        
        except Exception as e:
            if attempt == max_attempts - 1:
                return {
                    "message": "",
                    "word_count": 0,
                    "confidence_score": 0.0,
                    "error": str(e)
                }
    
    return {
        "message": "",
        "word_count": 0,
        "confidence_score": 0.0,
        "error": "Failed to generate within word limit"
    }


def evaluate_scenario(
    client: Groq,
    scenario: Dict,
    approved_facts_final: List[str],
    link_facts: Dict,
    model: str,
    runs: int,
    evaluation_mode: str = "RELAXED"
) -> Dict:
    """
    Evaluate a single scenario: PHASE 3 (generation) + PHASE 4 (evaluation).
    
    Returns:
        Dictionary with evaluation results
    """
    scenario_id = scenario.get("id", f"scenario_{hash(str(scenario))}")
    results = []
    
    for run_idx in range(runs):
        # PHASE 3: Generate message
        gen_result = generate_message_with_word_limit(
            client, scenario, allowed_facts, model, run_idx
        )
        
        if gen_result["error"]:
            # Error case
            check_result = {
                "within_word_limit": False,
                "must_include_ok": False,
                "tone_ok": False,
                "fabrication_detected": True,
                "unsupported_claims_detected": True,
                "overall_pass": False,
                "failure_reasons": [f"Generation error: {gen_result['error']}"],
                "notes": f"Error: {gen_result['error']}"
            }
        else:
            # PHASE 4: Evaluation
            check_result = run_all_checks(
                gen_result["message"],
                scenario.get("max_words", 150),
                scenario.get("must_include", []),
                allowed_facts,
                strict_mode,
                scenario.get("company", ""),
                scenario.get("target_role", "")
            )
        
        results.append({
            "run": run_idx + 1,
            "message": gen_result["message"],
            "confidence": gen_result["confidence_score"],
            "word_count": gen_result["word_count"],
            **check_result
        })
    
    # Compute metrics
    pass_count = sum(1 for r in results if r["overall_pass"])
    pass_rate = pass_count / len(results) if results else 0.0
    
    fabrication_count = sum(1 for r in results if r.get("fabrication_detected", False))
    fabrication_rate = fabrication_count / len(results) if results else 0.0
    
    unsupported_count = sum(1 for r in results if r.get("unsupported_claims_detected", False))
    unsupported_rate = unsupported_count / len(results) if results else 0.0
    
    overconfident_count = sum(
        1 for r in results 
        if r["confidence"] >= 0.75 and not r["overall_pass"]
    )
    overconfidence_rate = overconfident_count / len(results) if results else 0.0
    
    stability = len(set(r["overall_pass"] for r in results)) == 1
    overconfident = overconfident_count > 0
    
    return {
        "scenario_id": scenario_id,
        "scenario": scenario,
        "pass_rate": pass_rate,
        "fabrication_rate": fabrication_rate,
        "unsupported_rate": unsupported_rate,
        "overconfidence_rate": overconfidence_rate,
        "overconfident": overconfident,
        "stability": stability,
        "runs": results
    }


def compute_overall_metrics(evaluation_results: List[Dict]) -> Dict:
    """
    STAGE 3: Compute overall metrics across all scenarios.
    Returns JSON matching Stage 3 summary_metrics schema.
    """
    if not evaluation_results:
        return {
            "pass_rate": 0.0,
            "fabrication_rate": 0.0,
            "unsupported_rate": 0.0,
            "overconfidence_rate": 0.0,
            "stability_rate": 0.0
        }
    
    total_scenarios = len(evaluation_results)
    
    overall_pass_rate = sum(r["pass_rate"] for r in evaluation_results) / total_scenarios
    overall_fabrication_rate = sum(r["fabrication_rate"] for r in evaluation_results) / total_scenarios
    overall_unsupported_rate = sum(r.get("unsupported_rate", 0) for r in evaluation_results) / total_scenarios
    overall_overconfidence_rate = sum(r["overconfidence_rate"] for r in evaluation_results) / total_scenarios
    stability_count = sum(1 for r in evaluation_results if r["stability"])
    stability_rate = stability_count / total_scenarios
    
    return {
        "pass_rate": overall_pass_rate,
        "fabrication_rate": overall_fabrication_rate,
        "unsupported_rate": overall_unsupported_rate,
        "overconfidence_rate": overall_overconfidence_rate,
        "stability_rate": stability_rate
    }
