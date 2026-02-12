"""
Simple Streamlit app for LLM Behavior Job Outreach evaluation.
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from groq import Groq
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from checks import run_checks

# Page config
st.set_page_config(page_title="Job Outreach Eval", page_icon="💼", layout="wide")

st.title("💼 Job Outreach LLM Evaluator")
st.markdown("Test how well AI generates job application messages")

# Sidebar - Simple settings
with st.sidebar:
    st.header("Settings")
    
    # API Key
    default_key = ""
    try:
        if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
            default_key = st.secrets["GROQ_API_KEY"]
    except:
        pass
    
    api_key = st.text_input("Groq API Key", value=default_key, type="password")
    
    # Simple model selection
    model = st.selectbox("AI Model", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"], index=0)
    
    # Simple run count
    runs = st.slider("Test each prompt this many times", 2, 5, 3)

# Check API key
if not api_key:
    st.warning("Enter your Groq API key in the sidebar")
    st.stop()

# Load prompts
try:
    prompts_path = Path(__file__).parent / "src" / "prompts.json"
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Test API key
try:
    client = Groq(api_key=api_key)
    with st.spinner("Checking API key..."):
        # Groq doesn't have models.list(), just test with a simple call
        pass
except Exception as e:
    st.error(f"❌ API Error: {e}")
    st.stop()

def extract_confidence(text: str) -> float:
    """Extract confidence from text."""
    import re
    pattern = r'Confidence:\s*([0-9]*\.?[0-9]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except:
            return 0.5
    return 0.5

# Run button
if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
    progress = st.progress(0)
    results = []
    
    for idx, prompt_data in enumerate(prompts):
        progress.progress((idx + 1) / len(prompts))
        
        prompt_id = prompt_data["id"]
        channel = prompt_data["channel"]
        company = prompt_data["company"]
        target_role = prompt_data["target_role"]
        tone = prompt_data["tone"]
        max_words = prompt_data["max_words"]
        allowed_facts = prompt_data["allowed_facts"]
        must_include = prompt_data["must_include"]
        notes = prompt_data["notes"]
        
        system_prompt = f"""You are a professional job outreach assistant. Generate a {tone} {channel} message for a job application.

Rules:
- Target: {prompt_data['recipient_type']} at {company} for {target_role} role
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
Recipient: {prompt_data['recipient_type']}

Requirements:
- {tone} tone
- Max {max_words} words
- Must include: {', '.join(must_include)}
- Only use these facts: {', '.join(allowed_facts)}"""
        
        run_results = []
        for run_idx in range(runs):
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
                
                # Run checks
                check_results = run_checks(message, max_words, must_include, allowed_facts, tone)
                
                run_results.append({
                    "run": run_idx + 1,
                    "confidence": confidence,
                    "overall_pass": check_results["overall_pass"],
                    "word_limit": check_results["within_word_limit"],
                    "must_include": check_results["must_include_ok"],
                    "tone_ok": check_results["tone_ok"],
                    "new_facts": check_results["adds_new_facts"],
                    "message": message
                })
            except Exception as e:
                run_results.append({
                    "run": run_idx + 1,
                    "confidence": 0.0,
                    "overall_pass": False,
                    "word_limit": False,
                    "must_include": False,
                    "tone_ok": False,
                    "new_facts": True,
                    "message": f"Error: {str(e)}"
                })
        
        # Calculate metrics for this prompt
        pass_count = sum(1 for r in run_results if r["overall_pass"])
        pass_rate = pass_count / len(run_results)
        stability = len(set(r["overall_pass"] for r in run_results)) == 1
        overconfident = any(r["confidence"] >= 0.75 and not r["overall_pass"] for r in run_results)
        
        results.append({
            "Prompt": prompt_id,
            "Channel": channel,
            "Company": company,
            "Role": target_role,
            "Pass Rate": f"{pass_rate:.2f}",
            "Stable": "Yes" if stability else "No",
            "Overconfident": "Yes" if overconfident else "No",
            "Runs": run_results
        })
    
    progress.progress(1.0)
    
    # Show results
    st.success("✅ Evaluation complete!")
    
    # Summary
    st.subheader("Summary")
    summary_df = pd.DataFrame([{
        "Prompt": r["Prompt"],
        "Channel": r["Channel"],
        "Company": r["Company"],
        "Role": r["Role"],
        "Pass Rate": r["Pass Rate"],
        "Stable": r["Stable"],
        "Overconfident": r["Overconfident"]
    } for r in results])
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    avg_pass = sum(float(r["Pass Rate"]) for r in results) / len(results)
    stable_count = sum(1 for r in results if r["Stable"] == "Yes")
    overconf_count = sum(1 for r in results if r["Overconfident"] == "Yes")
    
    with col1:
        st.metric("Avg Pass Rate", f"{avg_pass:.2f}")
    with col2:
        st.metric("Stable Prompts", f"{stable_count}/{len(results)}")
    with col3:
        st.metric("Overconfident", f"{overconf_count}/{len(results)}")
    with col4:
        st.metric("Total Prompts", len(results))
    
    # Detailed results
    with st.expander("📋 View Detailed Results"):
        for result in results:
            st.markdown(f"### {result['Prompt']} - {result['Company']} ({result['Channel']})")
            for run in result["Runs"]:
                st.markdown(f"**Run {run['run']}** (Confidence: {run['confidence']:.2f})")
                st.markdown(f"- Word Limit: {'✓' if run['word_limit'] else '✗'}")
                st.markdown(f"- Must Include: {'✓' if run['must_include'] else '✗'}")
                st.markdown(f"- Tone: {'✓' if run['tone_ok'] else '✗'}")
                st.markdown(f"- New Facts: {'✗' if run['new_facts'] else '✓'}")
                st.markdown(f"- **Overall: {'✓ PASS' if run['overall_pass'] else '✗ FAIL'}**")
                st.text_area("Message", run["message"], height=100, key=f"{result['Prompt']}_run{run['run']}")
                st.divider()
    
    # Download
    csv_data = summary_df.to_csv(index=False)
    st.download_button("Download Summary", csv_data, "job_outreach_results.csv", "text/csv")

