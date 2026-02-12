"""
Simple, clean Streamlit app for Job Outreach LLM Evaluation.
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from groq import Groq
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from checks import run_checks

st.set_page_config(page_title="Job Outreach Eval", page_icon="💼", layout="wide")

st.title("💼 Job Outreach Evaluator")
st.markdown("Test AI reliability for job application messages")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key
    default_key = ""
    try:
        if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
            default_key = st.secrets["GROQ_API_KEY"]
    except:
        pass
    
    api_key = st.text_input("Groq API Key", value=default_key, type="password")
    model = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"], index=0)
    runs = st.slider("Runs per prompt", 2, 5, 3)

if not api_key:
    st.warning("⚠️ Enter your Groq API key in the sidebar")
    st.stop()

# Load prompts
try:
    prompts_path = Path(__file__).parent / "src" / "prompts.json"
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
except Exception as e:
    st.error(f"Error loading prompts: {e}")
    st.stop()

# Show prompts first
st.header("📝 Test Prompts")
st.markdown(f"**Total: {len(prompts)} prompts**")

prompts_display = []
for p in prompts:
    prompts_display.append({
        "ID": p["id"],
        "Channel": p["channel"],
        "Company": p["company"],
        "Role": p["target_role"],
        "Recipient": p["recipient_type"],
        "Max Words": p["max_words"],
        "Must Include": ", ".join(p["must_include"])
    })

st.dataframe(pd.DataFrame(prompts_display), use_container_width=True, hide_index=True)

# Show details
with st.expander("📋 View Prompt Details"):
    for p in prompts:
        st.markdown(f"### {p['id']} - {p['company']} ({p['channel']})")
        st.markdown(f"**Role:** {p['target_role']} | **Recipient:** {p['recipient_type']}")
        st.markdown(f"**Notes:** {p['notes']}")
        st.markdown(f"**Allowed Facts:** {', '.join(p['allowed_facts'])}")
        st.markdown(f"**Must Include:** {', '.join(p['must_include'])}")
        st.divider()

# Initialize client
try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"❌ API Error: {e}")
    st.stop()

def extract_confidence(text: str) -> float:
    import re
    pattern = r'Confidence:\s*([0-9]*\.?[0-9]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except:
            return 0.5
    return 0.5

# Run evaluation
if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()
    all_results = []
    
    for idx, prompt_data in enumerate(prompts):
        progress = (idx + 1) / len(prompts)
        progress_bar.progress(progress)
        status_text.text(f"Evaluating {prompt_data['id']} ({idx + 1}/{len(prompts)})...")
        
        prompt_id = prompt_data["id"]
        channel = prompt_data["channel"]
        company = prompt_data["company"]
        target_role = prompt_data["target_role"]
        tone = prompt_data["tone"]
        max_words = prompt_data["max_words"]
        allowed_facts = prompt_data["allowed_facts"]
        must_include = prompt_data["must_include"]
        notes = prompt_data["notes"]
        recipient_type = prompt_data["recipient_type"]
        
        system_prompt = f"""You are a professional job outreach assistant. Generate a {tone} {channel} message.

Rules:
- Target: {recipient_type} at {company} for {target_role}
- Tone: {tone}
- Maximum {max_words} words
- Must include: {', '.join(must_include)}
- ONLY use these facts: {', '.join(allowed_facts)}
- For email: Include subject line. For LinkedIn DM: No subject.
- End with: Confidence: <0-1>"""
        
        user_prompt = f"""Generate a {channel} message:

{notes}

Company: {company}
Role: {target_role}
Recipient: {recipient_type}

Requirements:
- {tone} tone, max {max_words} words
- Must include: {', '.join(must_include)}
- Only use: {', '.join(allowed_facts)}"""
        
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
                error_msg = str(e)
                if "decommissioned" in error_msg.lower():
                    st.error(f"❌ Model {model} is decommissioned. Please select a different model.")
                    st.stop()
                run_results.append({
                    "run": run_idx + 1,
                    "confidence": 0.0,
                    "overall_pass": False,
                    "word_limit": False,
                    "must_include": False,
                    "tone_ok": False,
                    "new_facts": True,
                    "message": f"Error: {error_msg[:200]}"
                })
        
        # Metrics
        pass_count = sum(1 for r in run_results if r["overall_pass"])
        pass_rate = pass_count / len(run_results) if run_results else 0.0
        stability = len(set(r["overall_pass"] for r in run_results)) == 1
        overconfident = any(r["confidence"] >= 0.75 and not r["overall_pass"] for r in run_results)
        
        all_results.append({
            "id": prompt_id,
            "channel": channel,
            "company": company,
            "role": target_role,
            "pass_rate": pass_rate,
            "stability": stability,
            "overconfident": overconfident,
            "runs": run_results
        })
    
    progress_bar.progress(1.0)
    status_text.text("✅ Complete!")
    
    # Results
    st.header("📊 Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    avg_pass = sum(r["pass_rate"] for r in all_results) / len(all_results) if all_results else 0.0
    stable_count = sum(1 for r in all_results if r["stability"])
    overconf_count = sum(1 for r in all_results if r["overconfident"])
    
    with col1:
        st.metric("Avg Pass Rate", f"{avg_pass:.2%}")
    with col2:
        st.metric("Stable", f"{stable_count}/{len(all_results)}")
    with col3:
        st.metric("Overconfident", f"{overconf_count}/{len(all_results)}")
    with col4:
        st.metric("Total", len(all_results))
    
    # Results table
    results_df = pd.DataFrame([{
        "Prompt": r["id"],
        "Company": r["company"],
        "Role": r["role"],
        "Channel": r["channel"],
        "Pass Rate": f"{r['pass_rate']:.2%}",
        "Stable": "✓" if r["stability"] else "✗",
        "Overconf": "✗" if r["overconfident"] else "✓"
    } for r in all_results])
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Detailed view
    st.subheader("📋 Detailed Results")
    for result in all_results:
        with st.expander(f"{result['id']} - {result['company']} ({result['channel']})"):
            for run in result["runs"]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    status = "✅ PASS" if run["overall_pass"] else "❌ FAIL"
                    st.markdown(f"**Run {run['run']}** - {status} (Confidence: {run['confidence']:.2f})")
                with col2:
                    st.markdown(f"Word: {'✓' if run['word_limit'] else '✗'} | "
                              f"Include: {'✓' if run['must_include'] else '✗'} | "
                              f"Tone: {'✓' if run['tone_ok'] else '✗'} | "
                              f"Facts: {'✓' if not run['new_facts'] else '✗'}")
                st.text_area("Message", run["message"], height=80, key=f"{result['id']}_run{run['run']}", label_visibility="collapsed")
                st.divider()
    
    # Download
    csv_data = results_df.to_csv(index=False)
    st.download_button("📥 Download Results", csv_data, "results.csv", "text/csv")
