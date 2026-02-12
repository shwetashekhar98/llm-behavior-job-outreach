"""
Production-grade Job Outreach LLM Evaluator with professional dark theme.
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from groq import Groq
import sys
import re
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent / "src"))
from checks import run_checks

# Page config
st.set_page_config(
    page_title="Job Outreach LLM Evaluator",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Dark Theme Colors */
    .stApp {
        background-color: #0E1117;
    }
    
    .metric-card {
        background-color: #161B22;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #30363D;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #E5E7EB;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8B949E;
        margin-top: 0.5rem;
    }
    
    .success-badge {
        background-color: #22C55E;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .fail-badge {
        background-color: #EF4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .warning-badge {
        background-color: #F59E0B;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .check-pass {
        color: #22C55E;
        font-weight: 600;
    }
    
    .check-fail {
        color: #EF4444;
        font-weight: 600;
    }
    
    .check-warning {
        color: #F59E0B;
        font-weight: 600;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #8B949E;
        font-size: 0.85rem;
        border-top: 1px solid #30363D;
        margin-top: 3rem;
    }
    
    .prompt-panel {
        background-color: #161B22;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #30363D;
        margin-bottom: 1rem;
    }
    
    .fabrication-highlight {
        border-left: 4px solid #EF4444;
        padding-left: 1rem;
        background-color: #1C2128;
    }
    
    .overconfidence-highlight {
        border: 2px solid #EF4444;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("💼 Job Outreach LLM Evaluator")
st.markdown("**Production-grade evaluation for AI-generated job application messages**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key
    default_key = ""
    try:
        if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
            default_key = st.secrets["GROQ_API_KEY"]
    except:
        pass
    
    api_key = st.text_input("Groq API Key", value=default_key, type="password", help="Get your key from console.groq.com")
    
    model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        index=0,
        help="Select the Groq model to evaluate"
    )
    
    runs = st.slider("Runs per prompt", 2, 5, 3, help="Number of times to generate each message")
    
    # Evaluation Mode
    eval_mode = st.radio(
        "Evaluation Mode",
        ["Relaxed", "Strict"],
        index=0,
        help="Relaxed: Flexible matching. Strict: Exact phrase requirements."
    )
    strict_mode = (eval_mode == "Strict")
    
    st.divider()
    
    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        st.session_state.run_evaluation = True
        st.session_state.strict_mode = strict_mode

if not api_key:
    st.warning("⚠️ Enter your Groq API key in the sidebar to begin")
    st.stop()

# Load prompts
try:
    prompts_path = Path(__file__).parent / "src" / "prompts.json"
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
except Exception as e:
    st.error(f"❌ Error loading prompts: {e}")
    st.stop()

# Initialize client
try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"❌ API Error: {e}")
    st.stop()

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

def get_severity_color(check_passed: bool, is_critical: bool = False) -> str:
    """Get color based on check result and severity."""
    if is_critical:
        return "#EF4444"  # Red
    if check_passed:
        return "#22C55E"  # Green
    return "#F59E0B"  # Amber

# Run evaluation
if st.session_state.get("run_evaluation", False):
    strict_mode = st.session_state.get("strict_mode", False)
    
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
        
        # Improved system prompt
        system_prompt = f"""You are generating a job outreach message.

STRICT RULES:
- You may ONLY use facts from the provided allowed_facts list.
- Do NOT invent degrees, graduation years, companies, publications, or metrics.
- Do NOT assume graduation if not explicitly stated.
- Stay under max_words.
- Maintain professional tone.
- If must_include requires:
    - GitHub → explicitly mention GitHub
    - Portfolio → explicitly mention Portfolio
    - Ask for chat → include a clear short request (e.g., "Would you be open to a 15-minute chat?")

Target: {recipient_type} at {company} for {target_role} role
Channel: {channel}
Tone: {tone}
Maximum words: {max_words}
Must include: {', '.join(must_include)}
Allowed facts ONLY: {', '.join(allowed_facts)}

For email: Include a subject line. For LinkedIn DM: No subject line.

Append exactly at the end:
Confidence: <number between 0 and 1>"""
        
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
                check_results = run_checks(
                    message, max_words, must_include, allowed_facts, tone, strict_mode,
                    company=company,
                    target_role=target_role,
                    recipient_type=recipient_type,
                    channel=channel
                )
                
                run_results.append({
                    "run": run_idx + 1,
                    "confidence": confidence,
                    "overall_pass": check_results["overall_pass"],
                    "word_limit": check_results["within_word_limit"],
                    "must_include": check_results["must_include_ok"],
                    "tone_ok": check_results["tone_ok"],
                    "fabrication": check_results.get("fabrication_detected", False),
                    "message": message,
                    "failure_reasons": check_results.get("failure_reasons", [])
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
                    "fabrication": True,
                    "message": f"Error: {error_msg[:200]}",
                    "failure_reasons": [f"API Error: {error_msg[:100]}"]
                })
        
        # Compute metrics for this prompt
        pass_count = sum(1 for r in run_results if r["overall_pass"])
        pass_rate = pass_count / len(run_results) if run_results else 0.0
        
        constraint_failures = sum(
            1 for r in run_results 
            if not r["overall_pass"] and (
                not r["word_limit"] or not r["must_include"] or not r["tone_ok"]
            )
        )
        constraint_failure_rate = constraint_failures / len(run_results) if run_results else 0.0
        
        fabrication_count = sum(1 for r in run_results if r["fabrication"])
        fabrication_rate = fabrication_count / len(run_results) if run_results else 0.0
        
        stability = len(set(r["overall_pass"] for r in run_results)) == 1
        overconfident = any(r["confidence"] >= 0.75 and not r["overall_pass"] for r in run_results)
        
        all_results.append({
            "id": prompt_id,
            "channel": channel,
            "company": company,
            "role": target_role,
            "recipient_type": recipient_type,
            "pass_rate": pass_rate,
            "constraint_failure_rate": constraint_failure_rate,
            "fabrication_rate": fabrication_rate,
            "stability": stability,
            "overconfident": overconfident,
            "runs": run_results
        })
    
    progress_bar.progress(1.0)
    status_text.text("✅ Evaluation Complete!")
    
    # Store results in session state
    st.session_state.evaluation_results = all_results
    st.session_state.run_evaluation = False

# Display results if available
if "evaluation_results" in st.session_state:
    all_results = st.session_state.evaluation_results
    
    st.header("📊 Evaluation Results")
    
    # Compute overall metrics
    total_prompts = len(all_results)
    total_runs = sum(len(r["runs"]) for r in all_results)
    
    overall_pass_rate = sum(r["pass_rate"] for r in all_results) / total_prompts if total_prompts > 0 else 0.0
    overall_constraint_failure = sum(r["constraint_failure_rate"] for r in all_results) / total_prompts if total_prompts > 0 else 0.0
    overall_fabrication_rate = sum(r["fabrication_rate"] for r in all_results) / total_prompts if total_prompts > 0 else 0.0
    stability_count = sum(1 for r in all_results if r["stability"])
    stability_rate = stability_count / total_prompts if total_prompts > 0 else 0.0
    overconfident_count = sum(1 for r in all_results if r["overconfident"])
    overconfidence_rate = overconfident_count / total_prompts if total_prompts > 0 else 0.0
    
    # Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {'#22C55E' if overall_pass_rate >= 0.8 else '#F59E0B' if overall_pass_rate >= 0.6 else '#EF4444'}">
                {overall_pass_rate:.1%}
            </div>
            <div class="metric-label">Pass Rate</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Percentage of runs passing all checks")
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {'#EF4444' if overall_fabrication_rate > 0 else '#22C55E'}">
                {overall_fabrication_rate:.1%}
            </div>
            <div class="metric-label">Fabrication Rate</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Runs with fabricated facts")
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {'#EF4444' if overconfidence_rate > 0.2 else '#22C55E'}">
                {overconfidence_rate:.1%}
            </div>
            <div class="metric-label">Overconfidence Rate</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("High confidence but failed checks")
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {'#22C55E' if stability_rate >= 0.7 else '#F59E0B'}">
                {stability_rate:.1%}
            </div>
            <div class="metric-label">Stability Rate</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Consistent results across runs")
    
    st.divider()
    
    # Detailed Results
    st.subheader("📋 Detailed Results by Prompt")
    
    for result in all_results:
        # Determine panel styling
        panel_class = "prompt-panel"
        if result["overconfident"]:
            panel_class += " overconfidence-highlight"
        if result["fabrication_rate"] > 0:
            panel_class += " fabrication-highlight"
        
        with st.expander(
            f"**{result['id']}** - {result['company']} ({result['channel']}) | "
            f"Role: {result['role']} | Pass Rate: {result['pass_rate']:.1%}",
            expanded=False
        ):
            # Prompt summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pass Rate", f"{result['pass_rate']:.1%}")
            with col2:
                st.metric("Fabrication", f"{result['fabrication_rate']:.1%}", 
                         delta=None if result['fabrication_rate'] == 0 else "⚠️")
            with col3:
                st.metric("Stable", "✓" if result["stability"] else "✗")
            with col4:
                st.metric("Overconfident", "⚠️" if result["overconfident"] else "✓")
            
            st.divider()
            
            # Run details
            for run in result["runs"]:
                # Run header with status
                status_color = "#22C55E" if run["overall_pass"] else "#EF4444"
                status_text = "✅ PASS" if run["overall_pass"] else "❌ FAIL"
                
                st.markdown(f"""
                <div style="background-color: #161B22; padding: 1rem; border-radius: 6px; margin-bottom: 1rem;">
                    <h4 style="color: {status_color}; margin: 0;">Run {run['run']} - {status_text} | Confidence: {run['confidence']:.2f}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Checklist
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    word_color = get_severity_color(run["word_limit"])
                    word_icon = "✓" if run["word_limit"] else "✗"
                    st.markdown(f'<span class="check-{"pass" if run["word_limit"] else "fail"}" style="color: {word_color};">{word_icon} Word Limit</span>', 
                              unsafe_allow_html=True)
                
                with col2:
                    include_color = get_severity_color(run["must_include"], is_critical=False)
                    include_icon = "✓" if run["must_include"] else "✗"
                    st.markdown(f'<span class="check-{"pass" if run["must_include"] else "fail"}" style="color: {include_color};">{include_icon} Must Include</span>', 
                              unsafe_allow_html=True)
                
                with col3:
                    tone_color = get_severity_color(run["tone_ok"], is_critical=False)
                    tone_icon = "✓" if run["tone_ok"] else "✗"
                    st.markdown(f'<span class="check-{"pass" if run["tone_ok"] else "fail"}" style="color: {tone_color};">{tone_icon} Tone</span>', 
                              unsafe_allow_html=True)
                
                with col4:
                    fact_color = get_severity_color(not run["fabrication"], is_critical=True)
                    fact_icon = "✓" if not run["fabrication"] else "✗"
                    st.markdown(f'<span class="check-{"pass" if not run["fabrication"] else "fail"}" style="color: {fact_color};">{fact_icon} No Fabrication</span>', 
                              unsafe_allow_html=True)
                
                # Failure reasons
                if run["failure_reasons"]:
                    st.warning(f"**Why it failed:** {'; '.join(run['failure_reasons'])}")
                
                # Message
                st.text_area(
                    "Generated Message",
                    run["message"],
                    height=120,
                    key=f"{result['id']}_run{run['run']}",
                    label_visibility="visible"
                )
                
                st.divider()
    
    # Download section
    st.divider()
    st.subheader("📥 Download Results")
    
    # Prepare CSV data
    csv_rows = []
    for result in all_results:
        for run in result["runs"]:
            csv_rows.append({
                "id": result["id"],
                "run_idx": run["run"],
                "channel": result["channel"],
                "company": result["company"],
                "target_role": result["role"],
                "recipient_type": result["recipient_type"],
                "confidence": run["confidence"],
                "within_word_limit": run["word_limit"],
                "must_include_ok": run["must_include"],
                "tone_ok": run["tone_ok"],
                "fabrication_detected": run["fabrication"],
                "overall_pass": run["overall_pass"],
                "message": run["message"],
                "failure_reasons": "; ".join(run.get("failure_reasons", []))
            })
    
    csv_df = pd.DataFrame(csv_rows)
    csv_data = csv_df.to_csv(index=False)
    
    # Prepare JSON summary
    summary_json = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "runs_per_prompt": runs,
        "evaluation_mode": "Strict" if strict_mode else "Relaxed",
        "overall": {
            "pass_rate": overall_pass_rate,
            "constraint_failure_rate": overall_constraint_failure,
            "fabrication_rate": overall_fabrication_rate,
            "stability_rate": stability_rate,
            "overconfidence_rate": overconfidence_rate
        },
        "by_prompt": {
            r["id"]: {
                "pass_rate": r["pass_rate"],
                "constraint_failure_rate": r["constraint_failure_rate"],
                "fabrication_rate": r["fabrication_rate"],
                "stability": r["stability"],
                "overconfident": r["overconfident"]
            }
            for r in all_results
        }
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download CSV",
            csv_data,
            f"job_outreach_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    with col2:
        st.download_button(
            "📥 Download JSON Summary",
            json.dumps(summary_json, indent=2),
            f"job_outreach_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            use_container_width=True
        )

# Footer
st.markdown("""
<div class="footer">
    <p>LLM Behavioral Evaluation for Job Outreach Automation</p>
    <p style="font-size: 0.75rem; margin-top: 0.5rem;">Measures constraint compliance, fact accuracy, stability, and self-awareness</p>
</div>
""", unsafe_allow_html=True)
