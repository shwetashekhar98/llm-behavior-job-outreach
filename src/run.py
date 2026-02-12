#!/usr/bin/env python3
"""
Main script to run LLM behavior evaluation for job outreach messages.
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq
from checks import run_checks


def extract_confidence(text: str) -> float:
    """
    Extract confidence score from model output.
    Looks for: Confidence: <number>
    
    Args:
        text: The model output text
        
    Returns:
        Confidence score (0-1) or 0.5 if not found
    """
    import re
    pattern = r'Confidence:\s*([0-9]*\.?[0-9]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        try:
            confidence = float(match.group(1))
            return max(0.0, min(1.0, confidence))
        except ValueError:
            return 0.5
    return 0.5


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    load_dotenv()
    
    return {
        "api_key": os.getenv("GROQ_API_KEY"),
        "model": os.getenv("MODEL", "llama3-8b-8192"),
        "runs_per_prompt": int(os.getenv("RUNS_PER_PROMPT", "3")),
        "temperature": float(os.getenv("TEMPERATURE", "0.2"))
    }


def load_prompts(prompts_path: str) -> List[Dict[str, Any]]:
    """Load prompts from JSON file."""
    with open(prompts_path, 'r') as f:
        return json.load(f)


def generate_message(
    client: Groq,
    prompt_data: Dict[str, Any],
    config: Dict[str, Any],
    run_idx: int
) -> Dict[str, Any]:
    """
    Generate a single job outreach message.
    
    Args:
        client: OpenAI client
        prompt_data: Prompt data from JSON
        config: Configuration dictionary
        run_idx: Run number (0-indexed)
        
    Returns:
        Dictionary with results
    """
    prompt_id = prompt_data["id"]
    channel = prompt_data["channel"]
    recipient_type = prompt_data["recipient_type"]
    company = prompt_data["company"]
    target_role = prompt_data["target_role"]
    tone = prompt_data["tone"]
    max_words = prompt_data["max_words"]
    allowed_facts = prompt_data["allowed_facts"]
    must_include = prompt_data["must_include"]
    notes = prompt_data["notes"]
    
    system_prompt = f"""You are a professional job outreach assistant. Generate a {tone} {channel} message for a job application.

Rules:
- Target: {recipient_type} at {company} for {target_role} role
- Tone: {tone}
- Maximum {max_words} words
- Must include: {', '.join(must_include)}
- ONLY mention these facts: {', '.join(allowed_facts)}. Do not add any other facts, companies, years, or details.
- For email: Include a subject line. For LinkedIn DM: No subject line.
- End with exactly: Confidence: <number between 0 and 1>"""
    
    user_prompt = f"""Generate a {channel} message based on these notes:

{notes}

Company: {company}
Role: {target_role}
Recipient: {recipient_type}

Requirements:
- {tone} tone
- Max {max_words} words
- Must include: {', '.join(must_include)}
- Only use these facts: {', '.join(allowed_facts)}"""
    
    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config["temperature"],
            max_tokens=800
        )
        
        message = response.choices[0].message.content or ""
        confidence = extract_confidence(message)
        
        # Run checks
        check_results = run_checks(message, max_words, must_include, allowed_facts, tone)
        
        return {
            "id": prompt_id,
            "run_idx": run_idx,
            "channel": channel,
            "company": company,
            "target_role": target_role,
            "confidence": round(confidence, 3),
            "within_word_limit": check_results["within_word_limit"],
            "must_include_ok": check_results["must_include_ok"],
            "adds_new_facts": check_results["adds_new_facts"],
            "tone_ok": check_results["tone_ok"],
            "overall_pass": check_results["overall_pass"],
            "message": message,
            "notes": check_results["notes"]
        }
    except Exception as e:
        print(f"    Error in run {run_idx + 1}: {e}")
        return {
            "id": prompt_id,
            "run_idx": run_idx,
            "channel": channel,
            "company": company,
            "target_role": target_role,
            "confidence": 0.0,
            "within_word_limit": False,
            "must_include_ok": False,
            "adds_new_facts": True,
            "tone_ok": False,
            "overall_pass": False,
            "message": "",
            "notes": f"Error: {str(e)}"
        }


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary metrics from results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with summary metrics
    """
    if not results:
        return {}
    
    # Group by prompt ID
    by_prompt = {}
    for result in results:
        prompt_id = result["id"]
        if prompt_id not in by_prompt:
            by_prompt[prompt_id] = []
        by_prompt[prompt_id].append(result)
    
    # Compute per-prompt metrics
    prompt_metrics = {}
    for prompt_id, prompt_results in by_prompt.items():
        total_runs = len(prompt_results)
        pass_count = sum(1 for r in prompt_results if r["overall_pass"])
        pass_rate = pass_count / total_runs if total_runs > 0 else 0.0
        
        # Stability: all runs have same overall_pass value
        pass_values = [r["overall_pass"] for r in prompt_results]
        stability = len(set(pass_values)) == 1
        
        # Overconfidence: confidence >= 0.75 AND overall_pass is false
        overconfident = any(
            r["confidence"] >= 0.75 and not r["overall_pass"] 
            for r in prompt_results
        )
        
        prompt_metrics[prompt_id] = {
            "pass_rate": round(pass_rate, 3),
            "stability": stability,
            "overconfident": overconfident
        }
    
    # Overall metrics
    all_pass_rates = [m["pass_rate"] for m in prompt_metrics.values()]
    all_stabilities = [m["stability"] for m in prompt_metrics.values()]
    all_overconfident = [m["overconfident"] for m in prompt_metrics.values()]
    
    overall_pass_rate = sum(all_pass_rates) / len(all_pass_rates) if all_pass_rates else 0.0
    stability_rate = sum(all_stabilities) / len(all_stabilities) if all_stabilities else 0.0
    overconfidence_rate = sum(all_overconfident) / len(all_overconfident) if all_overconfident else 0.0
    
    # By channel
    by_channel = {"email": [], "linkedin_dm": []}
    for result in results:
        channel = result["channel"]
        if channel in by_channel:
            by_channel[channel].append(result)
    
    channel_metrics = {}
    for channel, channel_results in by_channel.items():
        if channel_results:
            channel_pass = sum(1 for r in channel_results if r["overall_pass"])
            channel_pass_rate = channel_pass / len(channel_results) if channel_results else 0.0
            
            channel_stabilities = []
            channel_overconfident = []
            for prompt_id in set(r["id"] for r in channel_results):
                prompt_runs = [r for r in channel_results if r["id"] == prompt_id]
                pass_values = [r["overall_pass"] for r in prompt_runs]
                channel_stabilities.append(len(set(pass_values)) == 1)
                channel_overconfident.append(any(
                    r["confidence"] >= 0.75 and not r["overall_pass"] 
                    for r in prompt_runs
                ))
            
            channel_metrics[channel] = {
                "pass_rate": round(channel_pass_rate, 3),
                "stability_rate": round(sum(channel_stabilities) / len(channel_stabilities) if channel_stabilities else 0.0, 3),
                "overconfidence_rate": round(sum(channel_overconfident) / len(channel_overconfident) if channel_overconfident else 0.0, 3)
            }
    
    # By recipient type
    by_recipient = {"recruiter": [], "hiring_manager": [], "founder": []}
    for result in results:
        recipient = result.get("recipient_type", "")
        if recipient in by_recipient:
            by_recipient[recipient].append(result)
    
    recipient_metrics = {}
    for recipient, recipient_results in by_recipient.items():
        if recipient_results:
            recipient_pass = sum(1 for r in recipient_results if r["overall_pass"])
            recipient_pass_rate = recipient_pass / len(recipient_results) if recipient_results else 0.0
            
            recipient_stabilities = []
            recipient_overconfident = []
            for prompt_id in set(r["id"] for r in recipient_results):
                prompt_runs = [r for r in recipient_results if r["id"] == prompt_id]
                pass_values = [r["overall_pass"] for r in prompt_runs]
                recipient_stabilities.append(len(set(pass_values)) == 1)
                recipient_overconfident.append(any(
                    r["confidence"] >= 0.75 and not r["overall_pass"] 
                    for r in prompt_runs
                ))
            
            recipient_metrics[recipient] = {
                "pass_rate": round(recipient_pass_rate, 3),
                "stability_rate": round(sum(recipient_stabilities) / len(recipient_stabilities) if recipient_stabilities else 0.0, 3),
                "overconfidence_rate": round(sum(recipient_overconfident) / len(recipient_overconfident) if recipient_overconfident else 0.0, 3)
            }
    
    return {
        "overall": {
            "pass_rate": round(overall_pass_rate, 3),
            "stability_rate": round(stability_rate, 3),
            "overconfidence_rate": round(overconfidence_rate, 3)
        },
        "by_channel": channel_metrics,
        "by_recipient_type": recipient_metrics,
        "by_prompt": prompt_metrics
    }


def save_results(results: List[Dict[str, Any]], summary: Dict[str, Any], output_dir: Path):
    """Save results to CSV and summary to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / "outputs.csv"
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\nResults saved to {csv_path}")
    
    # Save summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


def print_summary(results: List[Dict[str, Any]], summary: Dict[str, Any]):
    """Print a compact summary table."""
    print("\n" + "=" * 80)
    print("LLM BEHAVIOR EVALUATION - JOB OUTREACH")
    print("=" * 80)
    
    # Overall metrics
    overall = summary.get("overall", {})
    print(f"\nOverall Metrics:")
    print(f"  Pass Rate:         {overall.get('pass_rate', 0):.3f}")
    print(f"  Stability Rate:    {overall.get('stability_rate', 0):.3f}")
    print(f"  Overconfidence:    {overall.get('overconfidence_rate', 0):.3f}")
    
    # By channel
    print(f"\nBy Channel:")
    for channel, metrics in summary.get("by_channel", {}).items():
        print(f"  {channel.replace('_', ' ').title()}:")
        print(f"    Pass Rate:         {metrics.get('pass_rate', 0):.3f}")
        print(f"    Stability Rate:    {metrics.get('stability_rate', 0):.3f}")
        print(f"    Overconfidence:    {metrics.get('overconfidence_rate', 0):.3f}")
    
    # By recipient type
    print(f"\nBy Recipient Type:")
    for recipient, metrics in summary.get("by_recipient_type", {}).items():
        print(f"  {recipient.replace('_', ' ').title()}:")
        print(f"    Pass Rate:         {metrics.get('pass_rate', 0):.3f}")
        print(f"    Stability Rate:    {metrics.get('stability_rate', 0):.3f}")
        print(f"    Overconfidence:    {metrics.get('overconfidence_rate', 0):.3f}")
    
    # Per-prompt summary
    print(f"\nPer-Prompt Summary:")
    print(f"{'ID':<30} {'Pass Rate':<12} {'Stable':<10} {'Overconf':<10}")
    print("-" * 80)
    for prompt_id, metrics in summary.get("by_prompt", {}).items():
        print(f"{prompt_id:<30} {metrics.get('pass_rate', 0):<12.3f} "
              f"{str(metrics.get('stability', False)):<10} {str(metrics.get('overconfident', False)):<10}")
    
    print("=" * 80)


def main():
    """Main execution function."""
    # Load config
    config = load_config()
    if not config["api_key"]:
        print("Error: GROQ_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        return
    
    # Initialize client
    client = Groq(api_key=config["api_key"])
    
    # Load prompts
    script_dir = Path(__file__).parent
    prompts_path = script_dir / "prompts.json"
    prompts = load_prompts(str(prompts_path))
    
    print(f"Loaded {len(prompts)} prompts")
    print(f"Configuration: model={config['model']}, runs={config['runs_per_prompt']}, temp={config['temperature']}")
    print(f"\nRunning evaluations...\n")
    
    # Run evaluations
    all_results = []
    for prompt_data in prompts:
        prompt_id = prompt_data["id"]
        print(f"Evaluating {prompt_id}...", end=" ", flush=True)
        
        for run_idx in range(config["runs_per_prompt"]):
            result = generate_message(client, prompt_data, config, run_idx)
            all_results.append(result)
        
        print("✓")
    
    # Compute summary
    summary = compute_metrics(all_results)
    
    # Save results
    output_dir = script_dir.parent / "results"
    save_results(all_results, summary, output_dir)
    
    # Print summary
    print_summary(all_results, summary)


if __name__ == "__main__":
    main()

