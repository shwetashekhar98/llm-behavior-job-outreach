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
from typing import Dict, List
import sys
import re

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ============================================================================
# FEATURE FLAGS: High-Stakes Claim Verification Layer
# ============================================================================
# Default flags (safe - must remain OFF by default)
DEFAULT_ENABLE_HIGH_STAKES_LAYER = False
DEFAULT_ENFORCE_HIGH_STAKES_LANGUAGE = False

from profile_extractor import (
    extract_candidate_facts,
    extract_facts_with_evidence,
    extract_structured_profile,
    prepare_approved_facts,
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
from high_stakes import is_high_stakes, annotate_fact_with_trust


def generate_fix_suggestions(run: Dict, scenario: Dict) -> List[str]:
    """
    Generate fix suggestions based on failure reasons.
    """
    suggestions = []
    failure_reasons = run.get("failure_reasons", [])
    message = run.get("message", "")
    word_count = run.get("word_count", len(message.split()))
    max_words = scenario.get("max_words", 150)
    
    for reason in failure_reasons:
        reason_lower = reason.lower()
        
        if "word limit" in reason_lower:
            excess = word_count - max_words
            if excess > 0:
                suggestions.append(f"Trim {excess} words to meet limit (currently {word_count}/{max_words})")
        
        elif "missing:" in reason_lower:
            missing_item = reason.split("Missing:")[-1].strip()
            if missing_item.lower() in ["github", "portfolio", "linkedin"]:
                suggestions.append(f"Add explicit mention: '{missing_item}' or include link if available")
            elif "chat" in missing_item.lower():
                suggestions.append("Add explicit request: 'Would you be open to a 15-minute chat?' or 'Let's schedule a call'")
            else:
                suggestions.append(f"Include required item: {missing_item}")
        
        elif "fabricated" in reason_lower:
            if "degree" in reason_lower:
                suggestions.append("Remove fabricated degree. Only mention degrees from approved facts.")
            elif "year" in reason_lower:
                suggestions.append("Remove fabricated year. Only mention years from approved facts.")
            elif "employer" in reason_lower or "company" in reason_lower:
                suggestions.append("Remove fabricated employer/company. Only mention employers from approved facts.")
            else:
                suggestions.append("Remove fabricated information. Only use facts from approved facts list.")
        
        elif "unsupported" in reason_lower:
            suggestions.append("Remove unsupported claims. Only mention metrics/achievements from approved facts.")
        
        elif "tone" in reason_lower:
            if "emoji" in reason_lower:
                suggestions.append("Remove emojis to maintain professional tone")
            elif "slang" in reason_lower:
                suggestions.append("Replace slang with professional language")
            elif "exclamation" in reason_lower:
                suggestions.append("Reduce exclamation marks (max 2 allowed)")
            else:
                suggestions.append("Adjust tone to be more professional")
    
    return suggestions

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
if "link_facts" not in st.session_state:
    st.session_state.link_facts = {
        "github": None,
        "portfolio": None,
        "linkedin": None,
        "other_links": []
    }
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

# Debug checkbox (near top of file)
show_debug_stage1 = st.checkbox("Show Stage 1 Debug Info", value=False, key="show_debug_stage1")

# High-Stakes Layer UI toggles (live control)
enable_high_stakes = st.checkbox(
    "Enable High-Stakes Claim Layer",
    value=DEFAULT_ENABLE_HIGH_STAKES_LAYER,
    key="enable_high_stakes",
    help="Enable trust calibration for high-stakes claims (e.g., NeurIPS, PhD, awards)"
)

enforce_high_stakes_language = st.checkbox(
    "Enforce cautious language for unverified high-stakes claims",
    value=DEFAULT_ENFORCE_HIGH_STAKES_LANGUAGE,
    key="enforce_high_stakes_language",
    help="When enabled, unverified high-stakes claims will use cautious phrasing in generated messages"
)

# DEBUG: Show feature flag status
st.write("🔍 **Debug - High-Stakes Layer:**", enable_high_stakes)
st.write("🔍 **Debug - High-Stakes Enforcement:**", enforce_high_stakes_language)

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
                        # Convert to profile_input format
                        profile_input = {
                            "unstructured_text": profile_text,
                            "structured_fields": {},
                            "links": {}
                        }
                        result = extract_facts_with_evidence(
                            profile_input,
                            api_key,
                            model,
                            show_debug_stage1
                        )
                        
                        # Extract debug info if available (returns tuple if debug enabled)
                        debug_info = None
                        if show_debug_stage1 and isinstance(result, tuple) and len(result) == 2:
                            extracted_facts, debug_info = result
                        else:
                            extracted_facts = result
                        
                        # Store extracted facts
                        st.session_state.extracted_facts = extracted_facts
                        st.session_state.source_text = profile_text
                        
                        # Display debug info BEFORE rerun (if enabled)
                        if show_debug_stage1 and debug_info:
                            st.markdown("---")
                            st.subheader("🔍 Stage 1 Debug Information")
                            
                            # Integrity warnings
                            integrity_warnings = debug_info.get("integrity_warnings", [])
                            if integrity_warnings:
                                st.error("⚠️ **Evidence Integrity Warnings:**")
                                for warning in integrity_warnings:
                                    st.text(warning)
                                st.markdown("---")
                            
                            # Raw LLM output
                            st.subheader("Stage 1 Raw LLM Output (from LLM)")
                            raw_facts = debug_info.get("raw_candidate_facts", [])
                            raw_warnings = debug_info.get("raw_warnings", [])
                            st.json({
                                "candidate_facts": raw_facts,
                                "warnings": raw_warnings,
                                "total_raw": len(raw_facts)
                            })
                            
                            # Show fact-evidence pairs from raw output
                            if raw_facts:
                                st.write("**Raw Fact-Evidence Pairs:**")
                                for idx, fact in enumerate(raw_facts):
                                    st.text(f"{idx+1}. Fact: {fact.get('fact', '')[:60]}...")
                                    st.text(f"   Evidence: {fact.get('evidence', '')[:60]}...")
                            
                            # Link extraction stats
                            num_llm = debug_info.get("num_llm_candidate_facts", 0)
                            num_deterministic = debug_info.get("num_deterministic_link_facts", 0)
                            num_merged = debug_info.get("num_merged_candidate_facts", 0)
                            merged_links = debug_info.get("merged_link_facts", [])
                            
                            st.write("**Link Extraction Stats:**")
                            st.write(f"- LLM candidate facts: {num_llm}")
                            st.write(f"- Deterministic link facts: {num_deterministic}")
                            st.write(f"- Merged candidate facts: {num_merged}")
                            if merged_links:
                                st.write("**Merged Link Facts (added deterministically):**")
                                for link_fact in merged_links:
                                    st.text(f"  - {link_fact.get('fact', '')}")
                            
                            # Processed candidate facts (for comparison)
                            st.subheader("Stage 1 Processed Candidate Facts (after validation)")
                            processed_facts = debug_info.get("processed_candidate_facts", [])
                            if processed_facts:
                                st.json(processed_facts)
                                st.caption(f"Total processed: {len(processed_facts)}")
                                
                                # Show fact-evidence pairs from processed output
                                st.write("**Processed Fact-Evidence Pairs:**")
                                for idx, fact in enumerate(processed_facts):
                                    st.text(f"{idx+1}. Fact: {fact.get('fact', '')[:60]}...")
                                    st.text(f"   Evidence: {fact.get('evidence', '')[:60]}...")
                                    original_idx = fact.get("_original_index", "?")
                                    st.text(f"   Original index: {original_idx}")
                            
                            # Accepted facts
                            st.subheader("Stage 1 Accepted Facts")
                            accepted = debug_info.get("accepted_facts", [])
                            st.json(accepted)
                            st.caption(f"Total accepted: {len(accepted)}")
                            
                            # Rejected facts with reasons
                            st.subheader("Stage 1 Rejected Facts + Reasons")
                            rejected = debug_info.get("rejected_facts", [])
                            if rejected:
                                st.json(rejected)
                                st.caption(f"Total rejected: {len(rejected)}")
                                # Show summary of rejection reasons
                                reason_counts = {}
                                for item in rejected:
                                    reasons = item.get("rejection_reasons", [])
                                    if isinstance(reasons, list):
                                        for reason in reasons:
                                            reason_counts[reason] = reason_counts.get(reason, 0) + 1
                                    else:
                                        st.warning(f"⚠️ Item missing rejection_reasons: {item}")
                                if reason_counts:
                                    st.write("**Rejection reason summary:**")
                                    for reason, count in sorted(reason_counts.items()):
                                        st.write(f"- `{reason}`: {count}")
                            else:
                                st.info("No facts were rejected.")
                            
                            st.markdown("---")
                            
                            # Don't auto-advance if debug is enabled - let user click button
                            if st.button("➡️ Continue to Fact Confirmation", type="primary"):
                                st.session_state.stage = "fact_confirmation"
                                st.rerun()
                        else:
                            # Normal flow: auto-advance
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
        
        # Debug: Show extracted facts count
        if not extracted_facts or len(extracted_facts) == 0:
            st.error("⚠️ No facts found in extracted_facts. Please go back to Stage 1 and try again.")
            st.json({"extracted_facts_count": len(extracted_facts) if extracted_facts else 0, "extracted_facts": extracted_facts})
            if st.button("← Back to Profile Input"):
                st.session_state.stage = "profile_input"
                st.rerun()
        else:
            st.markdown("### Review and Approve Facts")
            st.caption(f"Only approved facts will be used for message generation. Review each fact and its source evidence. ({len(extracted_facts)} facts extracted)")
            
            # Fact confirmation table
            st.markdown("#### Extracted Facts")
            
            # Initialize fact states if not exists
            if "fact_states" not in st.session_state:
                st.session_state.fact_states = {idx: True for idx in range(len(extracted_facts))}
                st.session_state.fact_values = {idx: fact.get("value", "") for idx, fact in enumerate(extracted_facts)}
            
            # Initialize high-stakes verification states if not exists
            if enable_high_stakes:
                if "high_stakes_verification" not in st.session_state:
                    st.session_state.high_stakes_verification = {}
                if "high_stakes_urls" not in st.session_state:
                    st.session_state.high_stakes_urls = {}
            
            approved_facts = []
            high_stakes_count = 0
            verified_count = 0
            unverified_count = 0
            
            # Debug: Show first fact structure if empty
            if len(extracted_facts) > 0:
                st.caption(f"📋 Showing {len(extracted_facts)} facts. First fact structure: {list(extracted_facts[0].keys())}")
            
            for idx, fact in enumerate(extracted_facts):
            # Annotate fact with trust metadata if enabled
            if enable_high_stakes:
                fact = annotate_fact_with_trust(fact, enable_high_stakes=True)
            
            fact_text = fact.get("value", "")
            category = fact.get("category", "other")
            # Get trust flag from annotated fact
            trust_flag = fact.get("trust_flag", "normal")
            is_high = (trust_flag == "high_stakes") if enable_high_stakes else False
            
            if is_high:
                high_stakes_count += 1
            
            # Determine column layout based on high-stakes feature
            if enable_high_stakes and is_high:
                col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
            else:
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
                
                # Show high-stakes warning if enabled and fact is high-stakes
                if enable_high_stakes and is_high:
                    st.markdown("⚠️ **High-Stakes Claim — verification recommended**")
            
            with col3:
                confidence = fact.get("confidence", 0.0)
                st.caption(f"Confidence: {confidence:.0%}")
                if source_text:
                    quote = fact.get("source_quote", "")
                    if quote:
                        st.caption(f"Source: \"{quote[:50]}...\"")
            
            # High-stakes verification UI (only if enabled and fact is high-stakes)
            if enable_high_stakes and is_high:
                with col4:
                    # Verification status dropdown
                    verification_key = f"verify_status_{idx}"
                    current_status = st.session_state.high_stakes_verification.get(verification_key, "unverified")
                    
                    verification_status = st.selectbox(
                        "Verification status",
                        ["unverified", "verified"],
                        index=0 if current_status == "unverified" else 1,
                        key=verification_key,
                        help="Select 'verified' if you have a URL to verify this claim"
                    )
                    st.session_state.high_stakes_verification[verification_key] = verification_status
                    
                    # URL input (always show, but required if verified)
                    url_key = f"verify_url_{idx}"
                    verification_url = st.text_input(
                        "Verification URL (required if verified)",
                        value=st.session_state.high_stakes_urls.get(url_key, ""),
                        key=url_key,
                        placeholder="https://...",
                        help="Provide a URL that verifies this claim"
                    )
                    st.session_state.high_stakes_urls[url_key] = verification_url
                    
                    # Update fact with verification metadata
                    fact["verification_status"] = verification_status
                    fact["verification_url"] = verification_url
                    
                    # Warn if verified but URL is empty
                    if verification_status == "verified":
                        if not verification_url or not verification_url.strip():
                            st.warning("⚠️ Verification URL required to mark as verified. Treating as unverified.")
                            fact["verification_status"] = "unverified"
                            st.session_state.high_stakes_verification[verification_key] = "unverified"
                        else:
                            verified_count += 1
                    else:
                        unverified_count += 1
                        # Clear URL if status changed to unverified
                        if url_key in st.session_state.high_stakes_urls:
                            st.session_state.high_stakes_urls[url_key] = ""
            
            if approved and fact_value:
                # Add verification metadata if high-stakes feature is enabled
                if enable_high_stakes and is_high:
                    verification_key = f"verify_status_{idx}"
                    url_key = f"verify_url_{idx}"
                    fact_with_metadata = {
                        "value": fact_value,
                        "trust_flag": "high_stakes",
                        "verification_status": st.session_state.high_stakes_verification.get(verification_key, "unverified"),
                        "verification_url": st.session_state.high_stakes_urls.get(url_key, "")
                    }
                    approved_facts.append(fact_with_metadata)
                else:
                    approved_facts.append(fact_value)
        
        # Manual fact addition
        st.markdown("---")
        st.markdown("#### Add Manual Fact")
        manual_fact = st.text_input("Add a fact manually", key="manual_fact")
        if manual_fact and st.button("➕ Add", key="add_manual"):
            approved_facts.append(manual_fact)
            st.rerun()
        
        # High-stakes summary (only if enabled)
        if enable_high_stakes and high_stakes_count > 0:
            st.markdown("---")
            st.markdown("### High-Stakes Claims Summary")
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.metric("High-Stakes Facts", high_stakes_count)
            with col_sum2:
                st.metric("Verified", verified_count, delta=None)
            with col_sum3:
                st.metric("Unverified", unverified_count, delta=None)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Facts", type="primary", use_container_width=True):
                # Use Stage 2 preparation
                # Extract fact values (handle dict format for high-stakes facts)
                approved_facts_values = []
                for fact in approved_facts:
                    if isinstance(fact, dict):
                        approved_facts_values.append(fact.get("value", ""))
                    else:
                        approved_facts_values.append(fact)
                
                rejected_facts = [f["value"] for idx, f in enumerate(extracted_facts) 
                                 if not st.session_state.fact_states.get(idx, False)]
                manual_facts_list = [f for f in approved_facts_values if f not in [fact.get("value", "") for fact in extracted_facts]]
                
                stage2_result = prepare_approved_facts(
                    approved_facts_values,
                    rejected_facts,
                    manual_facts_list
                )
                
                st.session_state.approved_facts = stage2_result["approved_facts_final"]
                st.session_state.link_facts = stage2_result["link_facts"]
                
                # Store high-stakes metadata if enabled
                if enable_high_stakes:
                    high_stakes_metadata = {}
                    for idx, fact in enumerate(extracted_facts):
                        fact_text = fact.get("value", "")
                        category = fact.get("category", "other")
                        if is_high_stakes(fact_text, category):
                            verification_key = f"verify_status_{idx}"
                            url_key = f"verify_url_{idx}"
                            high_stakes_metadata[fact_text] = {
                                "verification_status": st.session_state.high_stakes_verification.get(verification_key, "unverified"),
                                "verification_url": st.session_state.high_stakes_urls.get(url_key, "")
                            }
                    st.session_state.high_stakes_metadata = high_stakes_metadata
                
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
                        
                        # Get high-stakes metadata if enabled
                        high_stakes_metadata = None
                        if enable_high_stakes:
                            high_stakes_metadata = st.session_state.get("high_stakes_metadata", {})
                        
                        result = evaluate_scenario(
                            client,
                            scenario,
                            st.session_state.approved_facts,
                            st.session_state.link_facts,
                            model,
                            runs,
                            "STRICT" if strict_mode else "RELAXED",
                            high_stakes_metadata=high_stakes_metadata,
                            enforce_high_stakes_language=enforce_high_stakes_language
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
        st.markdown("### 📊 Summary Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
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
                "Unsupported Claims",
                overall_metrics.get("unsupported_rate", 0),
                "#ef4444" if overall_metrics.get("unsupported_rate", 0) > 0.1 else "#22c55e"
            )
        
        with col4:
            render_metric_card(
                "Overconfidence Rate",
                overall_metrics["overconfidence_rate"],
                "#ef4444" if overall_metrics["overconfidence_rate"] > 0.2 else "#22c55e"
            )
        
        with col5:
            render_metric_card(
                "Stability Rate",
                overall_metrics["stability_rate"],
                "#22c55e" if overall_metrics["stability_rate"] >= 0.7 else "#f59e0b"
            )
        
        st.markdown("---")
        
        # Detailed Results
        st.markdown("### Detailed Results by Scenario")
        
        for result in results:
            scenario = result.get("scenario", {})
            if not scenario:
                st.error(f"Missing scenario data for result: {result.get('scenario_id', 'unknown')}")
                continue
            with st.expander(
                f"**{scenario['company']}** - {scenario['target_role']} ({scenario['channel']}) | "
                f"Pass Rate: {result['pass_rate']:.1%}",
                expanded=False
            ):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Pass Rate", f"{result['pass_rate']:.1%}")
                with col2:
                    st.metric("Fabrication", f"{result['fabrication_rate']:.1%}")
                with col3:
                    st.metric("Unsupported", f"{result.get('unsupported_rate', 0):.1%}")
                with col4:
                    render_badge("✓ Stable" if result["stability"] else "✗ Unstable", 
                                "success" if result["stability"] else "warning")
                with col5:
                    render_badge("⚠️ Overconfident" if result.get("overconfident", False) else "✓ Confident",
                                "fail" if result.get("overconfident", False) else "success")
                
                st.divider()
                
                # Run details
                for run in result["runs"]:
                    status_color = "#22c55e" if run.get("overall_pass", False) else "#ef4444"
                    status_text = "✅ PASS" if run.get("overall_pass", False) else "❌ FAIL"
                    
                    word_count = run.get("word_count", len(run.get("message", "").split()))
                    st.markdown(f"""
                    <div style="background: #1e293b; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid {status_color};">
                        <h4 style="color: {status_color}; margin: 0;">Run {run['run']} - {status_text} | Confidence: {run['confidence']:.2f} | Words: {word_count}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    render_confidence_bar(run.get("confidence", 0.0))
                    
                    # Checklist - Access checks object if available, otherwise use flat structure
                    checks = run.get("checks", {})
                    if not checks:
                        # Fallback to flat structure for backward compatibility
                        checks = {
                            "within_word_limit": run.get("within_word_limit", False),
                            "must_include_ok": run.get("must_include_ok", False),
                            "tone_ok": run.get("tone_ok", False),
                            "fabrication_detected": run.get("fabrication_detected", False),
                            "unsupported_claims_detected": run.get("unsupported_claims_detected", False)
                        }
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        render_check_indicator(checks.get("within_word_limit", False), "Word Limit")
                    with col2:
                        render_check_indicator(checks.get("must_include_ok", False), "Must Include")
                    with col3:
                        render_check_indicator(checks.get("tone_ok", False), "Tone")
                    with col4:
                        render_check_indicator(not checks.get("fabrication_detected", False), "No Fabrication")
                    with col5:
                        render_check_indicator(not checks.get("unsupported_claims_detected", False), "No Unsupported")
                    
                    if run.get("failure_reasons"):
                        st.warning(f"**Why it failed:** {'; '.join(run.get('failure_reasons', []))}")
                        
                        # Fix suggestions
                        scenario = result.get("scenario", {})
                        if scenario:
                            suggestions = generate_fix_suggestions(run, scenario)
                            if suggestions:
                                with st.expander("🔧 Fix Suggestions", expanded=False):
                                    for suggestion in suggestions:
                                        st.markdown(f"- {suggestion}")
                    
                    st.text_area(
                        "Generated Message",
                        run.get("message", ""),
                        height=120,
                        key=f"{result['scenario_id']}_run{run.get('run', 0)}",
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
                # Access checks object if available
                checks = run.get("checks", {})
                if not checks:
                    checks = {
                        "within_word_limit": run.get("within_word_limit", False),
                        "must_include_ok": run.get("must_include_ok", False),
                        "tone_ok": run.get("tone_ok", False),
                        "fabrication_detected": run.get("fabrication_detected", False),
                        "unsupported_claims_detected": run.get("unsupported_claims_detected", False)
                    }
                
                csv_rows.append({
                    "scenario_id": result["scenario_id"],
                    "company": result["scenario"]["company"],
                    "target_role": result["scenario"]["target_role"],
                    "channel": result["scenario"]["channel"],
                    "run": run.get("run", 0),
                    "word_count": run.get("word_count", len(run.get("message", "").split())),
                    "confidence": run.get("confidence", 0.0),
                    "overall_pass": run.get("overall_pass", False),
                    "word_limit": checks.get("within_word_limit", False),
                    "must_include": checks.get("must_include_ok", False),
                    "tone_ok": checks.get("tone_ok", False),
                    "fabrication": checks.get("fabrication_detected", False),
                    "unsupported_claims": checks.get("unsupported_claims_detected", False),
                    "message": run.get("message", ""),
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
