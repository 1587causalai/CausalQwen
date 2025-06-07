"""
Generate visuals and tables for the experiment report.
"""
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

def generate_visuals(results_dir):
    """
    Generate and save comparison plots and tables.
    
    Args:
        results_dir (str): The directory where experiment results are stored.
    """
    
    # --- 1. Load Data ---
    baseline_path = os.path.join(results_dir, "baseline_results.json")
    finetuned_path = os.path.join(results_dir, "finetuned_results.json")
    
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)
    with open(finetuned_path, 'r') as f:
        finetuned_data = json.load(f)
        
    # --- 2. Prepare DataFrame for Plotting ---
    data = []
    for exp_name, results in baseline_data.items():
        data.append({
            "Experiment": exp_name,
            "Stage": "Baseline",
            "Accuracy": results["accuracy"],
            "Mean Rank": results["mean_rank"],
            "RMSE": np.nan if results["rmse"] == -1 else results["rmse"] # Use NaN for plotting
        })
    for exp_name, results in finetuned_data.items():
        data.append({
            "Experiment": exp_name,
            "Stage": "Finetuned",
            "Accuracy": results["accuracy"],
            "Mean Rank": results["mean_rank"],
            "RMSE": np.nan if results["rmse"] == -1 else results["rmse"]
        })
        
    df = pd.DataFrame(data)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # --- 3. Generate and Save Plots ---
    
    # Accuracy Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Experiment", y="Accuracy", hue="Stage", data=df, palette="viridis")
    ax.set_title("Accuracy Comparison: Baseline vs. Finetuned", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylim(0, 1.1)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    accuracy_plot_path = os.path.join(results_dir, "accuracy_comparison.png")
    plt.savefig(accuracy_plot_path)
    print(f"Saved accuracy plot to {accuracy_plot_path}")
    
    # Mean Rank Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Experiment", y="Mean Rank", hue="Stage", data=df, palette="plasma")
    ax.set_title("Mean Rank of <NUM> Token: Baseline vs. Finetuned", fontsize=16)
    ax.set_ylabel("Mean Rank (Log Scale)", fontsize=12)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_yscale('log') # Use log scale for better visualization
    mean_rank_plot_path = os.path.join(results_dir, "mean_rank_comparison.png")
    plt.savefig(mean_rank_plot_path)
    print(f"Saved mean rank plot to {mean_rank_plot_path}")

    # --- 4. Generate Markdown Table ---
    summary_data = []
    for exp_name in baseline_data.keys():
        b_res = baseline_data[exp_name]
        f_res = finetuned_data[exp_name]
        
        accuracy_change = f"{f_res['accuracy'] * 100:.0f}%"
        rank_change = f"{b_res['mean_rank']:.0f} -> {f_res['mean_rank']:.0f}"
        
        summary_data.append({
            "Dataset": exp_name.replace("_", " ").title(),
            "Accuracy": accuracy_change,
            "Mean Rank": rank_change,
            "RMSE (Finetuned)": f"{f_res['rmse']:.2f}" if f_res['rmse'] != -1 else "N/A"
        })
        
    summary_df = pd.DataFrame(summary_data)
    markdown_table = summary_df.to_markdown(index=False)
    
    print("\n--- Markdown Table for Report ---")
    print(markdown_table)
    
    return markdown_table, {
        "accuracy_plot": accuracy_plot_path,
        "mean_rank_plot": mean_rank_plot_path
    }

def update_report_markdown(report_path, table, plot_paths, latest_experiment_name):
    """
    Updates the visuals in the qwen_finetuning_report.md file.
    """
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update table
    # Matches the table block, starting from | Dataset and ending with a blank line
    content = re.sub(
        r"(\|(?:-*:?--*)+\|[\s\S]*?)(?:\r?\n){2,}",
        table + "\n\n",
        content,
        count=1,
        flags=re.DOTALL
    )

    # Update image paths
    # The relative path from `docs/experiments/` to `docs/results/` is `../results/`
    new_acc_path = f"../results/{latest_experiment_name}/accuracy_comparison.png"
    new_rank_path = f"../results/{latest_experiment_name}/mean_rank_comparison.png"
    
    content = re.sub(
        r"!\[Accuracy Comparison\]\(.*?\)",
        f"![Accuracy Comparison]({new_acc_path})",
        content
    )
    content = re.sub(
        r"!\[Mean Rank Comparison\]\(.*?\)",
        f"![Mean Rank Comparison]({new_rank_path})",
        content
    )
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"\nSuccessfully updated report: {report_path}")


if __name__ == '__main__':
    # Find the latest experiment directory
    base_dir = "docs/results"
    try:
        all_exp_dirs = [
            d for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('qwen_finetuning')
        ]
        if not all_exp_dirs:
            raise FileNotFoundError("No experiment directories found.")
            
        latest_exp_dir = max([os.path.join(base_dir, d) for d in all_exp_dirs], key=os.path.getmtime)
        latest_experiment_name = os.path.basename(latest_exp_dir)

    except FileNotFoundError as e:
        print(f"Error: Could not find experiment results directory at '{base_dir}'.")
        print("Please run an experiment first using `src/run_qwen_finetuning_experiment.py`.")
        exit(1)

    print(f"Generating report for latest experiment: {latest_experiment_name}")
    
    # Generate new visuals and get paths
    markdown_table, plot_paths = generate_visuals(latest_exp_dir)
    
    # Update the main report file
    report_file_path = "docs/experiments/qwen_finetuning_report.md"
    update_report_markdown(report_file_path, markdown_table, plot_paths, latest_experiment_name) 