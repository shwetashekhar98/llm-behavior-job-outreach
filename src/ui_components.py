"""
Reusable UI components for the application.
"""

import streamlit as st


def render_metric_card(label: str, value: float, color: str = "#38bdf8"):
    """Render a professional metric card."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1e293b 0%, #1e293b 100%);
        padding: 1.75rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    ">
        <div style="
            font-size: 2.5rem;
            font-weight: 700;
            color: {color};
            margin-bottom: 0.5rem;
        ">{value:.1%}</div>
        <div style="
            font-size: 0.875rem;
            color: #94a3b8;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        ">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_badge(text: str, variant: str = "info"):
    """Render a badge component."""
    colors = {
        "success": "#22c55e",
        "fail": "#ef4444",
        "warning": "#f59e0b",
        "info": "#38bdf8"
    }
    color = colors.get(variant, "#38bdf8")
    
    st.markdown(f"""
    <span style="
        display: inline-block;
        padding: 0.375rem 0.875rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background-color: {color};
        color: {'white' if variant != 'info' else '#0f172a'};
    ">{text}</span>
    """, unsafe_allow_html=True)


def render_check_indicator(passed: bool, label: str):
    """Render a check indicator (pass/fail)."""
    if passed:
        st.markdown(f"""
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 500;
            background-color: rgba(34, 197, 94, 0.1);
            color: #22c55e;
        ">✓ {label}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 500;
            background-color: rgba(239, 68, 68, 0.1);
            color: #ef4444;
        ">✗ {label}</div>
        """, unsafe_allow_html=True)


def render_confidence_bar(confidence: float):
    """Render a confidence progress bar."""
    if confidence >= 0.75:
        color_class = "confidence-high"
        color = "#22c55e"
    elif confidence >= 0.5:
        color_class = "confidence-medium"
        color = "#f59e0b"
    else:
        color_class = "confidence-low"
        color = "#ef4444"
    
    st.markdown(f"""
    <div style="
        background-color: #334155;
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
        margin-top: 0.5rem;
    ">
        <div style="
            height: 100%;
            width: {confidence * 100}%;
            background: linear-gradient(90deg, {color} 0%, {color} 100%);
            border-radius: 8px;
            transition: width 0.3s ease;
        "></div>
    </div>
    """, unsafe_allow_html=True)

