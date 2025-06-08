"""
Generic Plotting Utility for Causal LM Experiments.

This script loads the structured results from a completed experiment run
and generates a suite of visualizations as defined in the experiment design documents.
"""
import json
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_results(results_dir):
    """Loads the results.json file from a results directory."""
    results_path = os.path.join(results_dir, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Could not find results.json in {results_dir}")
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def prepare_dataframe(results):
    """Converts the nested results dictionary into a flat pandas DataFrame."""
    records = []
    for config_name, datasets in results.items():
        for dataset_name, metrics in datasets.items():
            record = {
                'model_config': config_name,
                'dataset': dataset_name,
                **metrics
            }
            records.append(record)
    return pd.DataFrame(records)

def plot_metric_comparison(df, dataset, metric, title, output_path, y_label=None):
    """
    Generates and saves a bar plot comparing different model configs on a single metric.

    Args:
        df (pd.DataFrame): The results dataframe.
        dataset (str): The specific dataset to plot (e.g., 'extreme').
        metric (str): The metric to plot on the y-axis (e.g., 'reg_mse').
        title (str): The title for the plot.
        output_path (str): The full path to save the generated plot.
        y_label (str, optional): The label for the y-axis. If None, uses the metric name.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_data = df[df['dataset'] == dataset].sort_values(by=metric)
    
    barplot = sns.barplot(
        x='model_config',
        y=metric,
        data=plot_data,
        palette='viridis',
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel('Model Configuration', fontsize=12)
    ax.set_ylabel(y_label or metric.replace('_', ' ').title(), fontsize=12)
    ax.tick_params(axis='x', rotation=15)
    
    # Add value labels on top of bars
    for p in barplot.patches:
        ax.annotate(format(p.get_height(), '.4f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 9),
                    textcoords = 'offset points')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot: {output_path}")

def generate_ablation_plots(df, results_dir):
    """Generates all standard plots for an ablation experiment."""
    print("\n--- Generating Ablation Study Plots ---")
    
    # As per design doc, a key test is MSE on extreme values
    plot_metric_comparison(
        df=df,
        dataset='extreme_values',
        metric='reg_mse',
        title='Ablation: Mean Squared Error on Extreme Value Dataset',
        output_path=os.path.join(results_dir, 'ablation_extreme_reg_mse.png'),
        y_label='Regression MSE'
    )
    
    # Another key test is F1 on QA data
    plot_metric_comparison(
        df=df,
        dataset='question_answering',
        metric='cls_f1',
        title='Ablation: F1 Score on Question Answering Dataset',
        output_path=os.path.join(results_dir, 'ablation_qa_cls_f1.png'),
        y_label='Classification F1 Score'
    )

    # And PICP on basic data
    plot_metric_comparison(
        df=df,
        dataset='basic',
        metric='reg_picp',
        title='Ablation: Prediction Interval Coverage (PICP) on Basic Dataset',
        output_path=os.path.join(results_dir, 'ablation_basic_reg_picp.png'),
        y_label='Regression PICP (95% Interval)'
    )

def generate_comparison_plots(df, results_dir):
    """Generates all standard plots for a comparison experiment."""
    print("\n--- Generating Hyperparameter Comparison Plots ---")

    # Compare effect of causal_dim on MAE
    plot_metric_comparison(
        df=df[df['model_config'].isin(['base', 'small_causal', 'large_causal'])],
        dataset='basic',
        metric='reg_mae',
        title='Comparison: Effect of Causal Dimension on Regression MAE',
        output_path=os.path.join(results_dir, 'comparison_causal_dim_reg_mae.png'),
        y_label='Regression MAE'
    )

    # Compare effect of reg_loss_weight on MAE
    plot_metric_comparison(
        df=df[df['model_config'].isin(['base', 'high_reg_weight'])],
        dataset='basic',
        metric='reg_mae',
        title='Comparison: Effect of Regression Weight on Regression MAE',
        output_path=os.path.join(results_dir, 'comparison_reg_weight_reg_mae.png'),
        y_label='Regression MAE'
    )

def main(args):
    """Main function to load results and generate plots."""
    print(f"--- Generating Visuals for Experiment in: {args.results_dir} ---")
    
    # 1. Load data and prepare DataFrame
    results = load_results(args.results_dir)
    df = prepare_dataframe(results)
    
    # 2. Determine experiment type from directory name
    experiment_type = os.path.basename(args.results_dir).split('_')[0]
    
    # 3. Generate plots based on experiment type
    if experiment_type == 'ablation':
        generate_ablation_plots(df, args.results_dir)
    elif experiment_type == 'comparison':
        generate_comparison_plots(df, args.results_dir)
    elif experiment_type in ['basic', 'comprehensive']:
        print("Basic and Comprehensive runs produce raw data. Run comparison or ablation for plots.")
    else:
        print(f"Warning: Unknown experiment type '{experiment_type}'. No plots generated.")

    print("\nVisual generation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plots from Causal LM experiment results.")
    parser.add_argument(
        'results_dir',
        type=str,
        help="Path to the directory containing the 'results.json' file."
    )
    args = parser.parse_args()
    main(args) 