"""
Enforcement Behavior Visualization Module
Generates charts for high-stakes enforcement behavior tracking.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import io
from typing import List, Dict


def compute_enforcement_aggregates(results: List[Dict]) -> Dict:
    """
    Aggregate enforcement behavior metrics across all runs.
    
    Args:
        results: List of scenario results from evaluation
    
    Returns:
        Dict with aggregated metrics
    """
    aggregates = {
        "total_high_stakes_facts_detected": 0,
        "total_high_stakes_unverified": 0,
        "softened_claims_count": 0,
        "suppressed_claims_count": 0,
        "enforcement_violations_count": 0,
        "total_runs": 0,
        "awkward_phrasing_scores": [],
        "hedging_densities": []
    }
    
    for result in results:
        runs = result.get("runs", [])
        for run in runs:
            aggregates["total_runs"] += 1
            
            # Extract enforcement behavior
            enforcement = run.get("enforcement_behavior", {})
            if enforcement:
                aggregates["total_high_stakes_facts_detected"] = max(
                    aggregates["total_high_stakes_facts_detected"],
                    enforcement.get("total_high_stakes_facts_detected", 0)
                )
                aggregates["total_high_stakes_unverified"] = max(
                    aggregates["total_high_stakes_unverified"],
                    enforcement.get("total_high_stakes_unverified", 0)
                )
                aggregates["softened_claims_count"] += enforcement.get("softened_claims_count", 0)
                aggregates["suppressed_claims_count"] += enforcement.get("suppressed_claims_count", 0)
                aggregates["enforcement_violations_count"] += enforcement.get("enforcement_violations_count", 0)
            
            # Extract language quality
            language = run.get("language_quality", {})
            if language:
                aggregates["awkward_phrasing_scores"].append(language.get("awkward_phrasing_score", 0))
                aggregates["hedging_densities"].append(language.get("hedging_density", 0))
    
    # Calculate averages
    if aggregates["awkward_phrasing_scores"]:
        aggregates["avg_awkward_score"] = sum(aggregates["awkward_phrasing_scores"]) / len(aggregates["awkward_phrasing_scores"])
    else:
        aggregates["avg_awkward_score"] = 0
    
    if aggregates["hedging_densities"]:
        aggregates["avg_hedging_density"] = sum(aggregates["hedging_densities"]) / len(aggregates["hedging_densities"])
    else:
        aggregates["avg_hedging_density"] = 0
    
    return aggregates


def generate_enforcement_behavior_chart(aggregates: Dict) -> io.BytesIO:
    """
    Generate a bar chart for enforcement behavior metrics.
    
    Args:
        aggregates: Aggregated enforcement behavior metrics
    
    Returns:
        BytesIO buffer containing PNG image
    """
    # Prepare data
    categories = ["Softened Claims", "Suppressed Claims", "Enforcement Violations"]
    values = [
        aggregates.get("softened_claims_count", 0),
        aggregates.get("suppressed_claims_count", 0),
        aggregates.get("enforcement_violations_count", 0)
    ]
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Create bar chart with colors
    colors = ['#3b82f6', '#f59e0b', '#ef4444']  # Blue, Orange, Red
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(value)}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Set labels and title
    ax.set_ylabel('Count', fontsize=11, color='#2d2d2d')
    ax.set_title('High-Stakes Enforcement Behavior', fontsize=13, fontweight='bold', color='#2d2d2d', pad=15)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    if max(values) > 0:
        ax.set_ylim(top=max(values) * 1.2)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=0, ha='center')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()
    buf.seek(0)
    
    return buf

