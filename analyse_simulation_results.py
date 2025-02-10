import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
os.environ['JAX_PLATFORMS'] = 'cpu'

import matplotlib.pyplot as plt
import polars as pl
import constants as cs

def create_visualization(df_results, df_runtime, target_hyperparameters, pic_name):
    # Set up the visual style for clear and professional plots
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.size'] = 10
    
    # Create our figures - one for hyperparameters and one for runtime
    fig_hyper, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig_runtime = plt.figure(figsize=(10, 6))
    
    # Flatten the axes array for easier iteration through subplots
    axes = axes.flatten()
    
    # Define colors for different lines - using a colorblind-friendly palette
    # We now add a distinct color for the true value line
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Blue, Green, Orange for percentiles and mean
    true_value_color = '#d62728'  # Red for true value line
    metrics = ['5th percentile', 'mean', '95th percentile']
    
    for idx, (param, true_value) in enumerate(zip(cs.HYPERPARAMETER_NAMES, target_hyperparameters)):
        ax = axes[idx]
        
        # Filter data for current hyperparameter
        param_data = df_results.filter(pl.col('hyperparameter') == param)
        
        # Plot each metric (5th percentile, mean, 95th percentile)
        for color, metric in zip(colors, metrics):
            ax.semilogx(param_data['number of systems'], 
                       param_data[metric], 
                       color=color, 
                       label=metric,
                       linewidth=2)
        
        # Add the true value reference line
        # We extend this line across the full x-axis range
        ax.axhline(y=true_value, 
                  color=true_value_color, 
                  linestyle='--', 
                  linewidth=2, 
                  label='True Value')
        
        # Get the current y-axis limits
        y_min, y_max = ax.get_ylim()
        
        # Adjust y-axis limits to ensure true value line is visible with some padding
        padding = 0.1 * (y_max - y_min)  # 10% padding
        new_y_min = min(y_min, true_value - padding)
        new_y_max = max(y_max, true_value + padding)
        ax.set_ylim(new_y_min, new_y_max)
        
        # Customize each subplot
        ax.set_xlabel('Number of Systems')
        ax.set_ylabel(f'Value of {param}')
        ax.set_title(f'Convergence of {param}\nTrue Value: {true_value:.3f}')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add legend only to the first plot to avoid redundancy
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add an overall title for hyperparameter plots
    fig_hyper.suptitle('Hyperparameter Convergence Analysis', fontsize=16, y=1.02)
    
    # Adjust the layout to prevent overlap
    fig_hyper.tight_layout()
    
    # Create runtime plot on the second figure
    ax_runtime = fig_runtime.add_subplot(111)
    
    # Plot runtime data with markers and line
    ax_runtime.semilogx(df_runtime['number of systems'], 
                       df_runtime['run time'], 
                       color='#1f77b4',  # Blue
                       marker='o',
                       linewidth=2,
                       markersize=8,
                       label='Runtime')
    
    # Customize runtime plot
    ax_runtime.set_xlabel('Number of Systems')
    ax_runtime.set_ylabel('Runtime (seconds)')
    ax_runtime.set_title('Runtime Performance Analysis')
    ax_runtime.grid(True, which="both", ls="-", alpha=0.2)
    ax_runtime.legend()
    
    # Adjust runtime plot layout
    fig_runtime.tight_layout()
    
    # Save both figures with high resolution
    fig_hyper.savefig(f'hyperparameter_convergence_{pic_name}.png', dpi=300, bbox_inches='tight')
    fig_runtime.savefig(f'runtime_performance_{pic_name}.png', dpi=300, bbox_inches='tight')
    
    # Display the plots
    plt.show()

df_mcmc_results = pl.read_parquet("simulation_results/new_data_gen/mcmc_results_N=*.parquet").sort(pl.col("number of systems"))
df_mcmc_runtime = pl.read_parquet("simulation_results/new_data_gen/mcmc_times_N=*.parquet").sort(pl.col("number of systems"))
create_visualization(df_mcmc_results, df_mcmc_runtime, cs.TARGET_HYPERPARAMETERS, "new")