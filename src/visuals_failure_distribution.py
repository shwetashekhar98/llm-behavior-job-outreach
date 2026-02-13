"""
Day-4 Failure Distribution Visual Generator
Generates minimalist black/gray charts for LinkedIn-ready failure distribution analysis.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple
import io


def compute_failure_buckets(
    results: List[Dict],
    overall_metrics: Dict
) -> Dict[str, float]:
    """
    Compute failure buckets from evaluation results.
    
    Args:
        results: List of scenario results from evaluation
        overall_metrics: Overall metrics dictionary with pass_rate, overconfidence_rate, stability_rate
    
    Returns:
        Dictionary mapping failure type to percentage (0-100)
    """
    buckets = {
        "Missing required elements": 0,
        "Word limit violation": 0,
        "Tone mismatch": 0,
        "Fabrication detected": 0,
        "Unsupported claims": 0,
        "High-stakes enforcement violation": 0,
        "Confidence mismatch (overconfident)": 0,
        "Stability drift": 0
    }
    
    total_runs = 0
    
    # Count failures from per-run checks
    for result in results:
        runs = result.get("runs", [])
        for run in runs:
            total_runs += 1
            
            # Check must_include_ok (may be in checks dict or directly in run)
            checks = run.get("checks", {})
            if not checks:
                checks = run
            
            if not checks.get("must_include_ok", True):
                buckets["Missing required elements"] += 1
            
            if not checks.get("within_word_limit", True):
                buckets["Word limit violation"] += 1
            
            if not checks.get("tone_ok", True):
                buckets["Tone mismatch"] += 1
            
            if checks.get("fabrication_detected", False):
                buckets["Fabrication detected"] += 1
            
            if checks.get("unsupported_claims_detected", False):
                buckets["Unsupported claims"] += 1
            
            if checks.get("high_stakes_enforcement_violation", False):
                buckets["High-stakes enforcement violation"] += 1
    
    # Add summary-derived buckets
    if overall_metrics and total_runs > 0:
        overconfidence_rate = overall_metrics.get("overconfidence_rate", 0)
        stability_rate = overall_metrics.get("stability_rate", 0)
        
        buckets["Confidence mismatch (overconfident)"] = round(overconfidence_rate * total_runs)
        buckets["Stability drift"] = round((1 - stability_rate) * total_runs)
    
    # Convert to percentages
    if total_runs > 0:
        for key in buckets:
            buckets[key] = (buckets[key] / total_runs) * 100
    
    # Filter out zero buckets and sort descending
    filtered_buckets = {k: v for k, v in buckets.items() if v > 0}
    sorted_buckets = dict(sorted(filtered_buckets.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_buckets


def generate_failure_distribution_chart(
    failure_buckets: Dict[str, float],
    pass_rate: float = None,
    top_bucket: str = None
) -> io.BytesIO:
    """
    Generate a minimalist black/gray horizontal bar chart.
    
    Args:
        failure_buckets: Dictionary mapping failure type to percentage
        pass_rate: Overall pass rate (optional, for footer)
        top_bucket: Top failure bucket name (optional, for footer)
    
    Returns:
        BytesIO buffer containing PNG image
    """
    if not failure_buckets:
        # Return empty chart if no failures
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No failures detected", 
                ha='center', va='center', fontsize=14, color='#666666')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        buf.seek(0)
        return buf
    
    # Prepare data
    labels = list(failure_buckets.keys())
    values = list(failure_buckets.values())
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.6)))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Create horizontal bar chart with grayscale
    # Use different shades of gray for visual distinction
    colors = ['#2d2d2d', '#4a4a4a', '#666666', '#808080', '#999999', '#b3b3b3', '#cccccc']
    bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
    
    bars = ax.barh(range(len(labels)), values, color=bar_colors, height=0.6)
    
    # Add percentage labels at end of each bar
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value + 1, i, f'{value:.1f}%', 
               va='center', fontsize=10, color='#2d2d2d', fontweight='bold')
    
    # Set y-axis labels
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11, color='#2d2d2d')
    
    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xlabel('')
    
    # Set x-axis limit to accommodate labels
    max_value = max(values) if values else 100
    ax.set_xlim(0, max_value * 1.15)
    
    # Minimal spines (only left and bottom, but make them subtle)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Title
    ax.set_title('LLM Evaluation — Failure Distribution', 
                fontsize=14, fontweight='bold', color='#2d2d2d', pad=20)
    
    # Footer text
    footer_text = ""
    if pass_rate is not None:
        footer_text += f"Pass Rate: {pass_rate:.1f}%"
    if top_bucket:
        if footer_text:
            footer_text += "  |  "
        footer_text += f"Primary driver: {top_bucket}"
    
    if footer_text:
        fig.text(0.5, 0.02, footer_text, 
                ha='center', fontsize=9, color='#666666', style='italic')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
               facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()
    buf.seek(0)
    
    return buf

