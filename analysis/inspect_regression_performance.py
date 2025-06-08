#!/usr/bin/env python
"""
Regression Performance Inspection Script

This script loads the detailed evaluation outputs from an experiment run
and generates a scatter plot to visually inspect the model's regression performance.

It plots true values against predicted values (the location of the Cauchy distribution)
and includes error bars representing the predicted uncertainty (the scale of the
Cauchy distribution).

Usage:
    python analysis/inspect_regression_performance.py <path_to_results_directory>
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluator import CAUCHY_95_CONF_FACTOR

def plot_regression_performance(results_dir):
    """
    Loads evaluation outputs and plots regression performance.

    Args:
        results_dir (str): Path to the directory containing experiment results.
    """
    # Find the first evaluation output file in the directory
    search_path = os.path.join(results_dir, "evaluation_outputs_*.pt")
    output_files = glob.glob(search_path)
    
    if not output_files:
        print(f"Error: No 'evaluation_outputs_*.pt' files found in {results_dir}")
        sys.exit(1)
        
    # Use the first file found
    eval_output_path = output_files[0]
    config_name = os.path.basename(eval_output_path).replace('evaluation_outputs_', '').replace('.pt', '')
    print(f"Loading data from: {eval_output_path}")

    # Load the data
    # In PyTorch >= 2.6, weights_only defaults to True. We must set it to False
    # to load our data structure which includes numpy arrays. This is safe as
    # we are loading a file we generated ourselves.
    data = torch.load(eval_output_path, map_location='cpu', weights_only=False)

    # Extract relevant arrays
    true_values = data['true_values']
    reg_locs = data['reg_locs']
    reg_scales = data['reg_scales']
    is_num_mask = data['is_num_mask']

    # Filter for samples that are actually numbers
    true_num_values = true_values[is_num_mask]
    pred_num_locs = reg_locs[is_num_mask]
    pred_num_scales = reg_scales[is_num_mask]
    
    if len(true_num_values) == 0:
        print("No numerical samples found in the evaluation data. Cannot generate plot.")
        return

    # Calculate the 95% confidence interval half-width for the error bars
    # This is scale * factor for Cauchy
    confidence_interval_half_width = pred_num_scales * CAUCHY_95_CONF_FACTOR

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with error bars
    ax.errorbar(
        true_num_values, 
        pred_num_locs, 
        yerr=confidence_interval_half_width,
        fmt='o', 
        ecolor='lightcoral', 
        color='royalblue', 
        capsize=3,
        alpha=0.6,
        label='Predicted Value (with 95% CI)'
    )

    # Plot the ideal y=x line
    min_val = min(np.min(true_num_values), np.min(pred_num_locs))
    max_val = max(np.max(true_num_values), np.max(pred_num_locs))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Ideal Fit (y=x)')

    # --- Formatting ---
    ax.set_xlabel("True Values", fontsize=12)
    ax.set_ylabel("Predicted Values (loc)", fontsize=12)
    ax.set_title(f"Regression Performance Analysis\nConfiguration: {config_name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True)
    
    # Save the plot
    plot_path = os.path.join(results_dir, f"regression_performance_{config_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"Successfully generated and saved regression performance plot to:\n{plot_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analysis/inspect_regression_performance.py <path_to_results_directory>")
        sys.exit(1)
    
    results_directory = sys.argv[1]
    
    if not os.path.isdir(results_directory):
        print(f"Error: Directory not found at '{results_directory}'")
        sys.exit(1)
        
    plot_regression_performance(results_directory) 