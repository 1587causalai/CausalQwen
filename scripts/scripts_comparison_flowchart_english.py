#!/usr/bin/env python3
"""
CausalEngine Real World Regression Tutorial Flowchart Generator (English Version)

Illustrates the data processing and analysis pipeline for the
real_world_regression_tutorial_sklearn_style.py script.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import matplotlib.patheffects as path_effects

plt.switch_backend('Agg')

def create_real_world_regression_flowchart():
    """Create flowchart for the Real World Regression tutorial script"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    fig.suptitle('CausalEngine Real World Regression Tutorial Flowchart\nSklearn-Style Implementation', 
                 fontsize=22, fontweight='bold', y=0.96)
    
    # Enhanced, more vibrant color definitions
    colors = {
        'data': '#81D4FA',      # Light blue
        'model': '#A5D6A7',     # Light green
        'eval': '#FFD54F',      # Amber
        'special': '#CE93D8',   # Light purple
    }
    
    # --- Box and Arrow Styles ---
    box_style = dict(boxstyle="round,pad=0.02,rounding_size=0.01", edgecolor='black', linewidth=1.5)
    arrow_props = dict(arrowstyle='-|>', lw=2.5, color='#444444', mutation_scale=25)
    shadow_effect = [path_effects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace="#a0a0a0", alpha=0.6)]

    # ==================================================================
    # Real World Regression Tutorial
    # ==================================================================
    ax.set_title('Real World Regression Tutorial\n(CausalEngine Robustness Analysis)', 
                  fontsize=18, fontweight='bold', pad=20)
    
    # Data processing flow
    # 1. Data loading
    box1 = FancyBboxPatch((0.1, 0.9), 0.8, 0.08, facecolor=colors['data'], **box_style)
    box1.set_path_effects(shadow_effect)
    ax.add_patch(box1)
    ax.text(0.5, 0.94, 'California Housing Dataset\n(20,640 samples, 8 features)', 
             ha='center', va='center', fontsize=12, fontweight='bold', linespacing=1.4)
    
    # 2. Data splitting
    box2 = FancyBboxPatch((0.1, 0.8), 0.8, 0.08, facecolor=colors['data'], **box_style)
    box2.set_path_effects(shadow_effect)
    ax.add_patch(box2)
    ax.text(0.5, 0.84, 'Train/Test Split (80%/20%)\n+ Validation Split (20%)', 
             ha='center', va='center', fontsize=11, linespacing=1.4)
    
    # 3. Standardization
    box3 = FancyBboxPatch((0.1, 0.7), 0.8, 0.08, facecolor=colors['data'], **box_style)
    box3.set_path_effects(shadow_effect)
    ax.add_patch(box3)
    ax.text(0.5, 0.74, 'StandardScaler (fitted on clean train data)\nX & y Standardization', 
             ha='center', va='center', fontsize=11, linespacing=1.4)
    
    # 4. Noise injection
    box4_style = box_style.copy()
    box4_style.update(linewidth=2, edgecolor='red')
    box4 = FancyBboxPatch((0.1, 0.6), 0.8, 0.08, facecolor=colors['special'], **box4_style)
    box4.set_path_effects(shadow_effect)
    ax.add_patch(box4)
    ax.text(0.5, 0.64, 'Noise Injection (post-standardization)\n40% shuffle noise -> training set', 
             ha='center', va='center', fontsize=11, fontweight='bold', linespacing=1.4)
    
    # 5. Model training (6 methods)
    box5_style = box_style.copy()
    box5_style.update(linewidth=2, edgecolor='green')
    box5 = FancyBboxPatch((0.1, 0.48), 0.8, 0.1, facecolor=colors['model'], **box5_style)
    box5.set_path_effects(shadow_effect)
    ax.add_patch(box5)
    ax.text(0.5, 0.53, '6 Methods Training\n• sklearn MLP\n• PyTorch MLP\n• CausalEngine (deterministic)\n• CausalEngine (standard)\n• Huber MLP\n• Pinball MLP', 
             ha='center', va='center', fontsize=10, linespacing=1.4)
    
    # 6. Performance evaluation
    box6_style = box_style.copy()
    box6_style.update(linewidth=1.5, edgecolor='orange')
    box6 = FancyBboxPatch((0.1, 0.36), 0.8, 0.08, facecolor=colors['eval'], **box6_style)
    box6.set_path_effects(shadow_effect)
    ax.add_patch(box6)
    ax.text(0.5, 0.4, 'Original Scale Evaluation\nMAE, MdAE, RMSE, R²', 
             ha='center', va='center', fontsize=11, linespacing=1.4)
    
    # 7. Robustness testing (key feature)
    box7_style = box_style.copy()
    box7_style.update(linewidth=3, edgecolor='purple')
    box7 = FancyBboxPatch((0.1, 0.22), 0.8, 0.12, facecolor=colors['special'], **box7_style)
    box7.set_path_effects(shadow_effect)
    ax.add_patch(box7)
    ax.text(0.5, 0.28, 'Robustness Testing (Core Feature)\nNoise levels: 0%, 10%, 20%, 30%, 40%, 50%\nCausalEngine vs Traditional Methods\nPerformance across different noise levels', 
             ha='center', va='center', fontsize=10, fontweight='bold', linespacing=1.4)
    
    # 8. Output
    box8 = FancyBboxPatch((0.1, 0.08), 0.8, 0.1, facecolor=colors['eval'], **box_style)
    box8.set_path_effects(shadow_effect)
    ax.add_patch(box8)
    ax.text(0.5, 0.13, 'Output Results\n• Performance comparison plots\n• Robustness curves (4 metrics)\n• Trend analysis report', 
             ha='center', va='center', fontsize=10, linespacing=1.4)
    
    # Add arrows
    y_coords = [0.9, 0.8, 0.7, 0.6, 0.48, 0.36, 0.22]
    for i in range(len(y_coords)):
        if i < len(y_coords) - 1:
            ax.annotate('', xy=(0.5, y_coords[i+1] + 0.08), xytext=(0.5, y_coords[i]), arrowprops=arrow_props)
    
    # Set axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['data'], label='Data Processing', edgecolor='k'),
        mpatches.Patch(facecolor=colors['model'], label='Model Training', edgecolor='k'),
        mpatches.Patch(facecolor=colors['eval'], label='Evaluation & Output', edgecolor='k'),
        mpatches.Patch(facecolor=colors['special'], label='Special Features', edgecolor='k'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=14, 
               bbox_to_anchor=(0.5, 0.02), frameon=False)
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig

def main():
    """Generate the flowchart"""
    print("🎨 Generating CausalEngine tutorial flowchart (English version)...")
    
    # Create output directory
    import os
    output_dir = "results/flowcharts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate scripts comparison flowchart
    fig = create_real_world_regression_flowchart()
    output_path = os.path.join(output_dir, "real_world_regression_tutorial_flowchart_english.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Flowchart saved (with design enhancements): {output_path}")
    plt.close(fig)
    
    print("\n✅ Flowchart generation completed!")
    print("📋 Generated file:")
    print(f"   - {output_path}")
    print("\n💡 This aesthetically enhanced flowchart clearly shows:")
    print("   - The pipeline for the 'Real World Regression Tutorial'")
    print("   - Key steps including data prep, noise injection, and robustness testing")

if __name__ == "__main__":
    main()