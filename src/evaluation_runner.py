"""
Evaluation runner for generating and evaluating job outreach messages.
"""

import re
from typing import Dict, List, Optional
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


def generate_message(
    client: Groq,
    scenario: Dict,
    allowed_facts: List[str],
    model: str,
    run_idx: int
) -> Dict:
    """
    Generate a single job outreach message.
    
    Args:
        client: Groq client
        scenario: Scenario dictionary with channel, company, role, etc.
        allowed_facts: List of approved facts from profile
        model: Model name
        run_idx: Run number (0-indexed)
        
    Returns:
        Dictionary with generated message and metadata
    """
    channel = scenario.get("channel", "email")
    company = scenario.get("company", "")
    target_role = scenario.get("target_role", "")
    recipient_type = scenario.get("recipient_type", "recruiter")
    tone = scenario.get("tone", "professional")
    max_words = scenario.get("max_words", 150)
    must_include = scenario.get("must_include", [])
    notes = scenario.get("notes", "")
    
    system_prompt = f"""You are generating a job outreach message.

STRICT RULES:
- You may ONLY use facts from the provided allowed_facts list.
- Do NOT invent degrees, graduation years, companies, publications, or metrics.
- Do NOT assume graduation if not explicitly stated.
- Stay under {max_words} words.
- Maintain professional tone.
- Must include: {', '.join(must_include) if must_include else 'None specified'}

Target: {recipient_type} at {company} for {target_role} role
Channel: {channel}
Tone: {tone}
Maximum words: {max_words}
Allowed facts ONLY: {', '.join(allowed_facts) if allowed_facts else 'None provided'}

For email: Include a subject line. For LinkedIn DM: No subject line.

Append exactly at the end:
Confidence: <number between 0 and 1>"""

    user_prompt = f"""Generate a {channel} message:

{notes if notes else 'General outreach message'}

Company: {company}
Role: {target_role}
Recipient: {recipient_type}

Requirements:
- {tone} tone, max {max_words} words
- Must include: {', '.join(must_include) if must_include else 'None'}
- Only use these facts: {', '.join(allowed_facts) if allowed_facts else 'None provided'}"""

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
        confidence = extract_confidence(message)
        
        return {
            "message": message,
            "confidence": confidence,
            "error": None
        }
    
    except Exception as e:
        return {
            "message": "",
            "confidence": 0.0,
            "error": str(e)
        }


def evaluate_scenario(
    client: Groq,
    scenario: Dict,
    allowed_facts: List[str],
    model: str,
    runs: int,
    strict_mode: bool = False
) -> Dict:
    """
    Evaluate a single scenario by generating multiple messages and checking them.
    
    Returns:
        Dictionary with evaluation results
    """
    scenario_id = scenario.get("id", f"scenario_{hash(str(scenario))}")
    results = []
    
    for run_idx in range(runs):
        # Generate message
        gen_result = generate_message(client, scenario, allowed_facts, model, run_idx)
        
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
            # Run validation checks
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
            "confidence": gen_result["confidence"],
            **check_result
        })
    
    # Compute metrics
    pass_count = sum(1 for r in results if r["overall_pass"])
    pass_rate = pass_count / len(results) if results else 0.0
    
    fabrication_count = sum(1 for r in results if r.get("fabrication_detected", False))
    fabrication_rate = fabrication_count / len(results) if results else 0.0
    
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
        "overconfidence_rate": overconfidence_rate,
        "overconfident": overconfident,
        "stability": stability,
        "runs": results
    }


def compute_overall_metrics(evaluation_results: List[Dict]) -> Dict:
    """
    Compute overall metrics across all scenarios.
    
    Returns:
        Dictionary with overall metrics
    """
    if not evaluation_results:
        return {
            "pass_rate": 0.0,
            "fabrication_rate": 0.0,
            "overconfidence_rate": 0.0,
            "stability_rate": 0.0
        }
    
    total_scenarios = len(evaluation_results)
    
    overall_pass_rate = sum(r["pass_rate"] for r in evaluation_results) / total_scenarios
    overall_fabrication_rate = sum(r["fabrication_rate"] for r in evaluation_results) / total_scenarios
    overall_overconfidence_rate = sum(r["overconfidence_rate"] for r in evaluation_results) / total_scenarios
    stability_count = sum(1 for r in evaluation_results if r["stability"])
    stability_rate = stability_count / total_scenarios
    
    return {
        "pass_rate": overall_pass_rate,
        "fabrication_rate": overall_fabrication_rate,
        "overconfidence_rate": overall_overconfidence_rate,
        "stability_rate": stability_rate
    }

