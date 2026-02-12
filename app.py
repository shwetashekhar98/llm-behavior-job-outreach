"""
Profile-Driven Job Outreach LLM Evaluator
Production-grade web application with evidence-based profile extraction.
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from groq import Groq
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from profile_extractor import (
    extract_evidence_based_facts,
    extract_structured_profile,
    validate_fact_evidence
)
from validation_engine import run_all_checks
from evaluation_runner import (
    evaluate_scenario,
    compute_overall_metrics
)
from ui_components import (
    render_metric_card,
    render_badge,
    render_check_indicator,
    render_confidence_bar
)

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
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    h1, h2, h3 {
        color: #e2e8f0;
    }
    
    .section-divider {
        margin: 3rem 0;
        border-top: 2px solid #334155;
    }
    
    .fact-table {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .scenario-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "stage" not in st.session_state:
    st.session_state.stage = "profile_input"
if "approved_facts" not in st.session_state:
    st.session_state.approved_facts = []
if "scenarios" not in st.session_state:
    st.session_state.scenarios = []
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = []

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🔑 Model Config")
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
        index=0
    )
    
    runs = st.slider("Runs per prompt", 1, 5, 3)
    
    st.markdown("---")
    st.markdown("### ⚙️ Evaluation Mode")
    
    eval_mode = st.radio(
        "Mode",
        ["Relaxed", "Strict"],
        index=0,
        help="Relaxed: Flexible matching. Strict: Exact phrase requirements."
    )
    strict_mode = (eval_mode == "Strict")
    
    st.markdown("---")
    st.markdown("### 📂 Profile Input Mode")
    
    profile_mode = st.radio(
        "Input Method",
        ["Paste Resume / LinkedIn", "Structured Form Builder", "JSON Advanced"],
        index=0
    )

# Title
st.markdown("""
<div style="margin-bottom: 2rem;">
    <h1 style="font-size: 2.5rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem;">
        💼 Profile-Driven Job Outreach LLM Evaluator
    </h1>
    <p style="font-size: 1.125rem; color: #94a3b8; margin: 0;">
        Evidence-based evaluation for AI-generated job application messages
    </p>
</div>
""", unsafe_allow_html=True)

if not api_key:
    st.warning("⚠️ Enter your Groq API key in the sidebar to begin")
    st.stop()

# ============================================================================
# STAGE 1: PROFILE INPUT → EVIDENCE EXTRACTION
# ============================================================================

if st.session_state.stage == "profile_input":
    st.markdown("## 📝 Stage 1: Profile Input & Evidence Extraction")
    st.markdown("---")
    
    extracted_facts = []
    
    if profile_mode == "Paste Resume / LinkedIn":
        st.markdown("### Paste Your Resume or LinkedIn Profile")
        st.caption("Paste your resume text here. The system will extract facts with evidence quotes.")
        
        profile_text = st.text_area(
            "Profile Text",
            height=300,
            placeholder="Paste your resume, LinkedIn profile, or any professional profile text here...",
            help="The system will extract only facts that can be directly quoted from this text."
        )
        
        if st.button("🔍 Extract Facts", type="primary", use_container_width=True):
            if profile_text and profile_text.strip():
                with st.spinner("Extracting facts with evidence..."):
                    try:
                        client = Groq(api_key=api_key)
                        extracted_facts = extract_evidence_based_facts(
                            profile_text,
                            api_key,
                            model
                        )
                        st.session_state.extracted_facts = extracted_facts
                        st.session_state.source_text = profile_text
                        st.session_state.stage = "fact_confirmation"
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Extraction error: {e}")
            else:
                st.warning("Please paste your profile text first.")
    
    elif profile_mode == "Structured Form Builder":
        st.markdown("### Structured Profile Form")
        
        with st.form("profile_form"):
            education = st.text_area("Education", placeholder="e.g., MS in Computer Science at NYU, expected May 2026")
            work_experience = st.text_area("Work Experience", placeholder="e.g., 4+ years software engineering at Company X")
            skills = st.text_area("Skills", placeholder="e.g., Python, Machine Learning, NLP")
            
            col1, col2 = st.columns(2)
            with col1:
                github = st.text_input("GitHub URL", placeholder="https://github.com/username")
                portfolio = st.text_input("Portfolio URL", placeholder="https://portfolio.example.com")
            with col2:
                linkedin = st.text_input("LinkedIn URL", placeholder="https://linkedin.com/in/username")
                location = st.text_input("Location", placeholder="New York, NY")
            
            if st.form_submit_button("✅ Submit Profile", use_container_width=True):
                form_data = {
                    "education": education,
                    "work_experience": work_experience,
                    "skills": skills,
                    "github": github,
                    "portfolio": portfolio,
                    "linkedin": linkedin,
                    "location": location
                }
                extracted_facts = extract_structured_profile(form_data)
                st.session_state.extracted_facts = extracted_facts
                st.session_state.source_text = json.dumps(form_data, indent=2)
                st.session_state.stage = "fact_confirmation"
                st.rerun()
    
    elif profile_mode == "JSON Advanced":
        st.markdown("### JSON Profile Upload")
        st.caption("Upload a JSON file with your profile data.")
        
        uploaded_file = st.file_uploader("Upload JSON", type=["json"])
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                # Convert JSON to facts
                facts = []
                for key, value in data.items():
                    if value:
                        facts.append({
                            "value": str(value),
                            "source_quote": str(value),
                            "start_index": 0,
                            "end_index": len(str(value)),
                            "confidence": 1.0,
                            "category": key
                        })
                st.session_state.extracted_facts = facts
                st.session_state.source_text = json.dumps(data, indent=2)
                st.session_state.stage = "fact_confirmation"
                st.rerun()
            except Exception as e:
                st.error(f"❌ JSON parsing error: {e}")

# ============================================================================
# STAGE 2: FACT CONFIRMATION UI
# ============================================================================

elif st.session_state.stage == "fact_confirmation":
    st.markdown("## ✅ Stage 2: Fact Confirmation")
    st.markdown("---")
    
    if "extracted_facts" not in st.session_state:
        st.error("No facts extracted. Please go back to Stage 1.")
        if st.button("← Back to Profile Input"):
            st.session_state.stage = "profile_input"
            st.rerun()
    else:
        extracted_facts = st.session_state.extracted_facts
        source_text = st.session_state.get("source_text", "")
        
        st.markdown("### Review and Approve Facts")
        st.caption("Only approved facts will be used for message generation. Review each fact and its source evidence.")
        
        # Fact confirmation table
        st.markdown("#### Extracted Facts")
        
        # Initialize fact states if not exists
        if "fact_states" not in st.session_state:
            st.session_state.fact_states = {idx: True for idx in range(len(extracted_facts))}
            st.session_state.fact_values = {idx: fact.get("value", "") for idx, fact in enumerate(extracted_facts)}
        
        approved_facts = []
        
        for idx, fact in enumerate(extracted_facts):
            col1, col2, col3 = st.columns([1, 3, 2])
            
            with col1:
                approved = st.checkbox(
                    "Approve",
                    value=st.session_state.fact_states.get(idx, True),
                    key=f"fact_approve_{idx}"
                )
                st.session_state.fact_states[idx] = approved
            
            with col2:
                fact_value = st.text_input(
                    "Fact",
                    value=st.session_state.fact_values.get(idx, fact.get("value", "")),
                    key=f"fact_value_{idx}"
                )
                st.session_state.fact_values[idx] = fact_value
            
            with col3:
                confidence = fact.get("confidence", 0.0)
                st.caption(f"Confidence: {confidence:.0%}")
                if source_text:
                    quote = fact.get("source_quote", "")
                    if quote:
                        st.caption(f"Source: \"{quote[:50]}...\"")
            
            if approved and fact_value:
                approved_facts.append(fact_value)
        
        # Manual fact addition
        st.markdown("---")
        st.markdown("#### Add Manual Fact")
        manual_fact = st.text_input("Add a fact manually", key="manual_fact")
        if manual_fact and st.button("➕ Add", key="add_manual"):
            approved_facts.append(manual_fact)
            st.rerun()
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Facts", type="primary", use_container_width=True):
                st.session_state.approved_facts = approved_facts
                st.session_state.stage = "message_generation"
                st.rerun()
        
        with col2:
            if st.button("← Back to Profile Input", use_container_width=True):
                st.session_state.stage = "profile_input"
                st.rerun()

# ============================================================================
# STAGE 3: MESSAGE GENERATION + EVALUATION
# ============================================================================

elif st.session_state.stage == "message_generation":
    st.markdown("## 🚀 Stage 3: Message Generation & Evaluation")
    st.markdown("---")
    
    if not st.session_state.approved_facts:
        st.error("No approved facts. Please go back to Stage 2.")
        if st.button("← Back to Fact Confirmation"):
            st.session_state.stage = "fact_confirmation"
            st.rerun()
    else:
        # Scenario builder
        st.markdown("### Define Evaluation Scenarios")
        
        with st.expander("➕ Add New Scenario", expanded=True):
            with st.form("scenario_form"):
                col1, col2 = st.columns(2)
                with col1:
                    channel = st.selectbox("Channel", ["email", "linkedin_dm"])
                    recipient_type = st.selectbox("Recipient Type", ["recruiter", "hiring_manager", "founder"])
                    company = st.text_input("Target Company")
                with col2:
                    target_role = st.text_input("Target Role")
                    max_words = st.number_input("Max Words", min_value=50, max_value=500, value=150)
                    tone = st.selectbox("Tone", ["professional"], disabled=True)
                
                must_include = st.multiselect(
                    "Must Include",
                    ["GitHub", "Portfolio", "LinkedIn", "Ask for chat"],
                    default=["GitHub", "Portfolio", "Ask for chat"]
                )
                
                notes = st.text_area("Additional Notes", placeholder="Any specific requirements or context...")
                
                if st.form_submit_button("➕ Add Scenario", use_container_width=True):
                    if company and target_role:
                        scenario = {
                            "id": f"scenario_{len(st.session_state.scenarios) + 1}",
                            "channel": channel,
                            "recipient_type": recipient_type,
                            "company": company,
                            "target_role": target_role,
                            "tone": tone,
                            "max_words": max_words,
                            "must_include": must_include,
                            "notes": notes
                        }
                        st.session_state.scenarios.append(scenario)
                        st.rerun()
                    else:
                        st.warning("Please fill in Company and Target Role.")
        
        # Display scenarios
        if st.session_state.scenarios:
            st.markdown("### Current Scenarios")
            for idx, scenario in enumerate(st.session_state.scenarios):
                with st.expander(f"Scenario {idx + 1}: {scenario['company']} - {scenario['target_role']}"):
                    st.json(scenario)
                    if st.button("🗑️ Remove", key=f"remove_{idx}"):
                        st.session_state.scenarios.pop(idx)
                        st.rerun()
            
            st.markdown("---")
            
            if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
                try:
                    client = Groq(api_key=api_key)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_results = []
                    for idx, scenario in enumerate(st.session_state.scenarios):
                        progress = (idx + 1) / len(st.session_state.scenarios)
                        progress_bar.progress(progress)
                        status_text.text(f"Evaluating {scenario['company']} ({idx + 1}/{len(st.session_state.scenarios)})...")
                        
                        result = evaluate_scenario(
                            client,
                            scenario,
                            st.session_state.approved_facts,
                            model,
                            runs,
                            strict_mode
                        )
                        all_results.append(result)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Evaluation Complete!")
                    
                    st.session_state.evaluation_results = all_results
                    st.session_state.stage = "results"
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Evaluation error: {e}")
        else:
            st.info("Add at least one scenario to begin evaluation.")
        
        # Navigation
        if st.button("← Back to Fact Confirmation"):
            st.session_state.stage = "fact_confirmation"
            st.rerun()

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

elif st.session_state.stage == "results":
    st.markdown("## 📊 Evaluation Results")
    st.markdown("---")
    
    if not st.session_state.evaluation_results:
        st.error("No results available.")
    else:
        results = st.session_state.evaluation_results
        overall_metrics = compute_overall_metrics(results)
        
        # Summary Metrics
        st.markdown("### Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            render_metric_card(
                "Pass Rate",
                overall_metrics["pass_rate"],
                "#22c55e" if overall_metrics["pass_rate"] >= 0.8 else "#f59e0b"
            )
        
        with col2:
            render_metric_card(
                "Fabrication Rate",
                overall_metrics["fabrication_rate"],
                "#ef4444" if overall_metrics["fabrication_rate"] > 0 else "#22c55e"
            )
        
        with col3:
            render_metric_card(
                "Overconfidence Rate",
                overall_metrics["overconfidence_rate"],
                "#ef4444" if overall_metrics["overconfidence_rate"] > 0.2 else "#22c55e"
            )
        
        with col4:
            render_metric_card(
                "Stability Rate",
                overall_metrics["stability_rate"],
                "#22c55e" if overall_metrics["stability_rate"] >= 0.7 else "#f59e0b"
            )
        
        st.markdown("---")
        
        # Detailed Results
        st.markdown("### Detailed Results by Scenario")
        
        for result in results:
            scenario = result["scenario"]
            with st.expander(
                f"**{scenario['company']}** - {scenario['target_role']} ({scenario['channel']}) | "
                f"Pass Rate: {result['pass_rate']:.1%}",
                expanded=False
            ):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pass Rate", f"{result['pass_rate']:.1%}")
                with col2:
                    st.metric("Fabrication", f"{result['fabrication_rate']:.1%}")
                with col3:
                    render_badge("✓ Stable" if result["stability"] else "✗ Unstable", 
                                "success" if result["stability"] else "warning")
                with col4:
                    render_badge("⚠️ Overconfident" if result["overconfident"] else "✓ Confident",
                                "fail" if result["overconfident"] else "success")
                
                st.divider()
                
                # Run details
                for run in result["runs"]:
                    status_color = "#22c55e" if run["overall_pass"] else "#ef4444"
                    status_text = "✅ PASS" if run["overall_pass"] else "❌ FAIL"
                    
                    st.markdown(f"""
                    <div style="background: #1e293b; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid {status_color};">
                        <h4 style="color: {status_color}; margin: 0;">Run {run['run']} - {status_text} | Confidence: {run['confidence']:.2f}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    render_confidence_bar(run["confidence"])
                    
                    # Checklist
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        render_check_indicator(run["within_word_limit"], "Word Limit")
                    with col2:
                        render_check_indicator(run["must_include_ok"], "Must Include")
                    with col3:
                        render_check_indicator(run["tone_ok"], "Tone")
                    with col4:
                        render_check_indicator(not run["fabrication_detected"], "No Fabrication")
                    with col5:
                        render_check_indicator(not run.get("unsupported_claims_detected", False), "No Unsupported")
                    
                    if run["failure_reasons"]:
                        st.warning(f"**Why it failed:** {'; '.join(run['failure_reasons'])}")
                    
                    st.text_area(
                        "Generated Message",
                        run["message"],
                        height=120,
                        key=f"{result['scenario_id']}_run{run['run']}",
                        label_visibility="visible"
                    )
                    
                    st.divider()
        
        # Download section
        st.markdown("---")
        st.markdown("### 📥 Download Results")
        
        # Prepare CSV
        csv_rows = []
        for result in results:
            for run in result["runs"]:
                csv_rows.append({
                    "scenario_id": result["scenario_id"],
                    "company": result["scenario"]["company"],
                    "target_role": result["scenario"]["target_role"],
                    "channel": result["scenario"]["channel"],
                    "run": run["run"],
                    "confidence": run["confidence"],
                    "overall_pass": run["overall_pass"],
                    "word_limit": run["within_word_limit"],
                    "must_include": run["must_include_ok"],
                    "tone_ok": run["tone_ok"],
                    "fabrication": run["fabrication_detected"],
                    "unsupported_claims": run.get("unsupported_claims_detected", False),
                    "message": run["message"],
                    "failure_reasons": "; ".join(run.get("failure_reasons", []))
                })
        
        csv_df = pd.DataFrame(csv_rows)
        csv_data = csv_df.to_csv(index=False)
        
        # Prepare JSON
        summary_json = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "runs_per_prompt": runs,
            "evaluation_mode": "Strict" if strict_mode else "Relaxed",
            "approved_facts": st.session_state.approved_facts,
            "overall_metrics": overall_metrics,
            "scenarios": results
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download CSV",
                csv_data,
                f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "📥 Download JSON",
                json.dumps(summary_json, indent=2),
                f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
        
        # Navigation
        if st.button("🔄 Start New Evaluation"):
            st.session_state.stage = "profile_input"
            st.session_state.approved_facts = []
            st.session_state.scenarios = []
            st.session_state.evaluation_results = []
            st.rerun()

# Footer
st.markdown("""
<div style="text-align: center; padding: 3rem 2rem; color: #94a3b8; font-size: 0.875rem; border-top: 1px solid #334155; margin-top: 4rem;">
    <p style="font-size: 1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">
        Profile-Driven Job Outreach LLM Evaluator
    </p>
    <p>Evidence-based evaluation for AI-generated job application messages</p>
</div>
""", unsafe_allow_html=True)

