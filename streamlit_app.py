"""
Production-grade Job Outreach LLM Evaluator - Generic, Reusable Framework
Professional SaaS dashboard UI with dark theme.
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

# Professional Dark Theme CSS
st.markdown("""
<style>
    /* Professional Dark Theme - SaaS Dashboard Style */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #1e293b 100%);
        padding: 1.75rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        border-color: #38bdf8;
        box-shadow: 0 8px 12px -2px rgba(56, 189, 248, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.375rem 0.875rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-success {
        background-color: #22c55e;
        color: white;
    }
    
    .badge-fail {
        background-color: #ef4444;
        color: white;
    }
    
    .badge-warning {
        background-color: #f59e0b;
        color: white;
    }
    
    .badge-info {
        background-color: #38bdf8;
        color: #0f172a;
    }
    
    /* Result Panel */
    .result-panel {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .result-panel:hover {
        border-color: #475569;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    .result-panel.fabrication {
        border-left: 4px solid #ef4444;
    }
    
    .result-panel.overconfident {
        border: 2px solid #ef4444;
    }
    
    /* Check Indicators */
    .check-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .check-pass {
        background-color: rgba(34, 197, 94, 0.1);
        color: #22c55e;
    }
    
    .check-fail {
        background-color: rgba(239, 68, 68, 0.1);
        color: #ef4444;
    }
    
    .check-warning {
        background-color: rgba(245, 158, 11, 0.1);
        color: #f59e0b;
    }
    
    /* Confidence Bar */
    .confidence-bar-container {
        background-color: #334155;
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .confidence-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
    
    .confidence-high {
        background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
    }
    
    .confidence-medium {
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
    }
    
    .confidence-low {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        color: #94a3b8;
        font-size: 0.875rem;
        border-top: 1px solid #334155;
        margin-top: 4rem;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #e2e8f0;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1e293b;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.4);
    }
    
    /* Toggle switch styling */
    .stRadio>div>label {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Title Section
st.markdown("""
<div style="margin-bottom: 2rem;">
    <h1 style="font-size: 2.5rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem;">
        💼 Job Outreach LLM Evaluator
    </h1>
    <p style="font-size: 1.125rem; color: #94a3b8; margin: 0;">
        Production-grade reliability evaluation for AI-generated job application messages
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    # API Key
    default_key = ""
    try:
        if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
            default_key = st.secrets["GROQ_API_KEY"]
    except:
        pass
    
    api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Get your key from console.groq.com"
    )
    
    model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        index=0,
        help="Select the Groq model to evaluate"
    )
    
    runs = st.slider(
        "Runs per prompt",
        2, 5, 3,
        help="Number of times to generate each message"
    )
    
    st.markdown("---")
    
    # Evaluation Mode with toggle
    st.markdown("**Evaluation Mode**")
    eval_mode = st.radio(
        "Mode",
        ["Relaxed", "Strict"],
        index=0,
        help="Relaxed: Flexible matching. Strict: Exact phrase requirements.",
        label_visibility="collapsed"
    )
    strict_mode = (eval_mode == "Strict")
    
    st.markdown(f"""
    <div style="background: #1e293b; padding: 0.75rem; border-radius: 8px; margin-top: 0.5rem;">
        <span style="color: #38bdf8; font-weight: 600;">Current Mode:</span>
        <span style="color: #e2e8f0; margin-left: 0.5rem;">{eval_mode}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
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
        prompts_data = json.load(f)
        # Support both old and new format
        if isinstance(prompts_data, list):
            profile = {"allowed_facts": [], "links": {}}
            prompts = prompts_data
        else:
            profile = prompts_data.get("profile", {"allowed_facts": [], "links": {}})
            prompts = prompts_data.get("evaluation_prompts", [])
        
        profile_allowed_facts = profile.get("allowed_facts", [])
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

def get_confidence_color(confidence: float) -> str:
    """Get color for confidence bar."""
    if confidence >= 0.75:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

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
        
        # Use prompt-specific allowed_facts if provided, otherwise use profile
        prompt_allowed_facts = prompt_data.get("allowed_facts", profile_allowed_facts)
        must_include = prompt_data.get("must_include", [])
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
    - mention_github → explicitly mention GitHub
    - mention_portfolio → explicitly mention Portfolio
    - request_chat → include a clear short request (e.g., "Would you be open to a 15-minute chat?")

Target: {recipient_type} at {company} for {target_role} role
Channel: {channel}
Tone: {tone}
Maximum words: {max_words}
Must include: {', '.join(must_include)}
Allowed facts ONLY: {', '.join(prompt_allowed_facts)}

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
- Only use: {', '.join(prompt_allowed_facts)}"""
        
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
                    message, max_words, must_include, prompt_allowed_facts, tone, strict_mode,
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
    
    st.markdown("---")
    st.markdown("## 📊 Evaluation Results")
    
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
    
    # Metrics Cards - Professional Dashboard Style
    st.markdown("### Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pass_color = "#22c55e" if overall_pass_rate >= 0.8 else "#f59e0b" if overall_pass_rate >= 0.6 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {pass_color}">
                {overall_pass_rate:.1%}
            </div>
            <div class="metric-label">Pass Rate</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Percentage of runs passing all checks")
    
    with col2:
        fab_color = "#ef4444" if overall_fabrication_rate > 0 else "#22c55e"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {fab_color}">
                {overall_fabrication_rate:.1%}
            </div>
            <div class="metric-label">Fabrication Rate</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Runs with fabricated facts")
    
    with col3:
        overconf_color = "#ef4444" if overconfidence_rate > 0.2 else "#22c55e"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {overconf_color}">
                {overconfidence_rate:.1%}
            </div>
            <div class="metric-label">Overconfidence Rate</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("High confidence but failed checks")
    
    with col4:
        stability_color = "#22c55e" if stability_rate >= 0.7 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {stability_color}">
                {stability_rate:.1%}
            </div>
            <div class="metric-label">Stability Rate</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Consistent results across runs")
    
    st.markdown("---")
    
    # Detailed Results
    st.markdown("### Detailed Results by Prompt")
    
    for result in all_results:
        # Determine panel styling
        panel_class = "result-panel"
        if result["overconfident"]:
            panel_class += " overconfident"
        if result["fabrication_rate"] > 0:
            panel_class += " fabrication"
        
        with st.expander(
            f"**{result['id']}** - {result['company']} ({result['channel']}) | "
            f"Role: {result['role']} | Pass Rate: {result['pass_rate']:.1%}",
            expanded=False
        ):
            # Prompt summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pass Rate", f"{result['pass_rate']:.1%}")
            with col2:
                st.metric("Fabrication", f"{result['fabrication_rate']:.1%}", 
                         delta=None if result['fabrication_rate'] == 0 else "⚠️")
            with col3:
                badge = '<span class="badge badge-success">✓ Stable</span>' if result["stability"] else '<span class="badge badge-warning">✗ Unstable</span>'
                st.markdown(badge, unsafe_allow_html=True)
            with col4:
                badge = '<span class="badge badge-fail">⚠️ Overconfident</span>' if result["overconfident"] else '<span class="badge badge-success">✓ Confident</span>'
                st.markdown(badge, unsafe_allow_html=True)
            
            st.divider()
            
            # Run details
            for run in result["runs"]:
                # Run header with status
                status_color = "#22c55e" if run["overall_pass"] else "#ef4444"
                status_text = "✅ PASS" if run["overall_pass"] else "❌ FAIL"
                
                st.markdown(f"""
                <div style="background: #1e293b; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid {status_color};">
                    <h4 style="color: {status_color}; margin: 0;">Run {run['run']} - {status_text} | Confidence: {run['confidence']:.2f}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence bar
                conf_class = get_confidence_color(run["confidence"])
                st.markdown(f"""
                <div class="confidence-bar-container">
                    <div class="confidence-bar {conf_class}" style="width: {run['confidence'] * 100}%"></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Checklist
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    word_class = "check-pass" if run["word_limit"] else "check-fail"
                    word_icon = "✓" if run["word_limit"] else "✗"
                    st.markdown(f'<div class="check-indicator {word_class}">{word_icon} Word Limit</div>', 
                              unsafe_allow_html=True)
                
                with col2:
                    include_class = "check-pass" if run["must_include"] else "check-fail"
                    include_icon = "✓" if run["must_include"] else "✗"
                    st.markdown(f'<div class="check-indicator {include_class}">{include_icon} Must Include</div>', 
                              unsafe_allow_html=True)
                
                with col3:
                    tone_class = "check-pass" if run["tone_ok"] else "check-fail"
                    tone_icon = "✓" if run["tone_ok"] else "✗"
                    st.markdown(f'<div class="check-indicator {tone_class}">{tone_icon} Tone</div>', 
                              unsafe_allow_html=True)
                
                with col4:
                    fact_class = "check-pass" if not run["fabrication"] else "check-fail"
                    fact_icon = "✓" if not run["fabrication"] else "✗"
                    st.markdown(f'<div class="check-indicator {fact_class}">{fact_icon} No Fabrication</div>', 
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
    st.markdown("---")
    st.markdown("### 📥 Download Results")
    
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
    <p style="font-size: 1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">
        LLM Behavioral Evaluation for Job Outreach Automation
    </p>
    <p style="font-size: 0.875rem; color: #94a3b8;">
        Measures constraint compliance, fact accuracy, stability, and self-awareness
    </p>
</div>
""", unsafe_allow_html=True)
