import os
import matplotlib.pyplot as plt
import polars as pl
import constants as cs
import glob
import numpy as np

def create_percentile_plot(mcmc_folder, sw_folder, target_hyperparameters, pic_name, num_experiments=10):
    """
    Create visualization comparing MCMC and SW results with percentile lines and min/max values
    
    Parameters:
    -----------
    mcmc_folder : str
        Folder containing MCMC results
    sw_folder : str
        Folder containing SW results
    target_hyperparameters : list
        List of true hyperparameter values
    pic_name : str
        Base name for saving the plots
    num_experiments : int
        Number of experiment replications
    """
    # Set up the visual style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.size'] = 10
    
    # Create figures
    fig_hyper, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig_runtime = plt.figure(figsize=(12, 7))
    axes = axes.flatten()
    
    # Define colors
    mcmc_mean_color = '#2ca02c'  # Green for MCMC mean
    mcmc_5th_color = '#1f77b4'   # Blue for MCMC 5th percentile
    mcmc_95th_color = '#ff7f0e'  # Orange for MCMC 95th percentile
    sw_color = '#9467bd'         # Purple for SW
    true_value_color = '#d62728' # Red for true value
    
    # Dictionary to hold all data
    mcmc_data = {p: {n: {'mean': [], '5th percentile': [], '95th percentile': []} 
                   for n in cs.N_VALUES} for p in cs.HYPERPARAMETER_NAMES}
    sw_data = {p: {n: {'sw_estimate': [], '5th percentile': [], '95th percentile': []} 
                for n in cs.N_VALUES} for p in cs.HYPERPARAMETER_NAMES}
    mcmc_runtime_data = {n: [] for n in cs.N_VALUES}
    sw_runtime_data = {n: [] for n in cs.N_VALUES}
    
    # Load MCMC experiment data
    for exp_num in range(1, num_experiments + 1):
        # Find all result files for this experiment
        mcmc_result_files = glob.glob(f"simulation_results/{mcmc_folder}/results_experiment={exp_num}_N=*.parquet")
        mcmc_time_files = glob.glob(f"simulation_results/{mcmc_folder}/times_experiment={exp_num}_N=*.parquet")
        
        for file_path in mcmc_result_files:
            try:
                df = pl.read_parquet(file_path)
                # Extract N value from filename
                n_value = int(file_path.split("N=")[1].split(".")[0])
                
                # Only process data for expected system counts
                if n_value in cs.N_VALUES:
                    # Store data by hyperparameter, system count, and metric
                    for param in cs.HYPERPARAMETER_NAMES:
                        param_data = df.filter(pl.col('hyperparameter') == param)
                        if not param_data.is_empty():
                            mcmc_data[param][n_value]['mean'].append(param_data['mean'].to_list()[0])
                            mcmc_data[param][n_value]['5th percentile'].append(param_data['5th percentile'].to_list()[0])
                            mcmc_data[param][n_value]['95th percentile'].append(param_data['95th percentile'].to_list()[0])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        for file_path in mcmc_time_files:
            try:
                df = pl.read_parquet(file_path)
                n_value = int(file_path.split("N=")[1].split(".")[0])
                
                # Only process data for expected system counts
                if n_value in cs.N_VALUES:
                    mcmc_runtime_data[n_value].append(df['run time'].to_list()[0])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Load SW experiment data
    for exp_num in range(1, num_experiments + 1):
        sw_result_files = glob.glob(f"simulation_results/{sw_folder}/results_experiment={exp_num}_N=*.parquet")
        sw_time_files = glob.glob(f"simulation_results/{sw_folder}/times_experiment={exp_num}_N=*.parquet")
        
        for file_path in sw_result_files:
            try:
                df = pl.read_parquet(file_path)
                n_value = int(file_path.split("N=")[1].split(".")[0])
                
                # Only process data for expected system counts
                if n_value in cs.N_VALUES:
                    for param in cs.HYPERPARAMETER_NAMES:
                        param_data = df.filter(pl.col('hyperparameter') == param)
                        if not param_data.is_empty():
                            sw_data[param][n_value]['sw_estimate'].append(param_data['sw_estimate'].to_list()[0])
                            
                            # Check if SW results have percentiles (they might not)
                            if '5th percentile' in param_data.columns:
                                sw_data[param][n_value]['5th percentile'].append(param_data['5th percentile'].to_list()[0])
                            if '95th percentile' in param_data.columns:
                                sw_data[param][n_value]['95th percentile'].append(param_data['95th percentile'].to_list()[0])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        for file_path in sw_time_files:
            try:
                df = pl.read_parquet(file_path)
                n_value = int(file_path.split("N=")[1].split(".")[0])
                
                # Only process data for expected system counts
                if n_value in cs.N_VALUES:
                    # Check if SW times use 'run time' or 'sw run time'
                    if 'sw run time' in df.columns:
                        sw_runtime_data[n_value].append(df['sw run time'].to_list()[0])
                    else:
                        sw_runtime_data[n_value].append(df['run time'].to_list()[0])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Plot hyperparameter results
    for idx, (param, true_value) in enumerate(zip(cs.HYPERPARAMETER_NAMES, target_hyperparameters)):
        ax = axes[idx]
        
        # Prepare data for MCMC plot
        mcmc_x = []
        mcmc_means_median = []
        mcmc_means_min = []
        mcmc_means_max = []
        mcmc_5th_median = []
        mcmc_95th_median = []
        
        # Prepare data for SW plot
        sw_x = []
        sw_estimates_median = []
        sw_estimates_min = []
        sw_estimates_max = []
        
        # Collect data for each system count
        for n_systems in cs.N_VALUES:
            # MCMC data
            if mcmc_data[param][n_systems]['mean']:
                mcmc_x.append(n_systems)
                
                # Process mean values
                means = mcmc_data[param][n_systems]['mean']
                mcmc_means_median.append(np.median(means))
                mcmc_means_min.append(np.min(means))
                mcmc_means_max.append(np.max(means))
                
                # Process 5th and 95th percentile values
                percentile5 = mcmc_data[param][n_systems]['5th percentile']
                mcmc_5th_median.append(np.median(percentile5))
                
                percentile95 = mcmc_data[param][n_systems]['95th percentile']
                mcmc_95th_median.append(np.median(percentile95))
            
            # SW data
            if sw_data[param][n_systems]['sw_estimate']:
                sw_x.append(n_systems)
                
                # Process SW estimates
                estimates = sw_data[param][n_systems]['sw_estimate']
                sw_estimates_median.append(np.median(estimates))
                sw_estimates_min.append(np.min(estimates))
                sw_estimates_max.append(np.max(estimates))
        
        # Plot MCMC lines
        if mcmc_x:
            # 5th percentile median line (dotted)
            ax.semilogx(mcmc_x, mcmc_5th_median, color=mcmc_5th_color, linestyle=':', 
                       linewidth=2, label='MCMC 5th percentile')
            
            # Mean median line (solid)
            ax.semilogx(mcmc_x, mcmc_means_median, color=mcmc_mean_color, linestyle='-', 
                       linewidth=2, label='MCMC mean')
            
            # 95th percentile median line (dotted)
            ax.semilogx(mcmc_x, mcmc_95th_median, color=mcmc_95th_color, linestyle=':', 
                       linewidth=2, label='MCMC 95th percentile')
            
            # Add mean min-max ranges as colored error bars
            for i, (x, median, min_val, max_val) in enumerate(zip(mcmc_x, mcmc_means_median, 
                                                                mcmc_means_min, mcmc_means_max)):
                # Plot vertical line from min to max
                ax.plot([x, x], [min_val, max_val], color=mcmc_mean_color, linestyle='-', 
                       linewidth=2, alpha=0.7)
                
                # Add horizontal caps at min/max
                cap_width = 0.08 * x  # Scale cap width with x value
                ax.plot([x-cap_width, x+cap_width], [min_val, min_val], color=mcmc_mean_color, 
                       linewidth=2, alpha=0.7)
                ax.plot([x-cap_width, x+cap_width], [max_val, max_val], color=mcmc_mean_color, 
                       linewidth=2, alpha=0.7)
        
        # Plot SW line (estimate)
        if sw_x:
            # SW median line (dashed)
            ax.semilogx(sw_x, sw_estimates_median, color=sw_color, linestyle='-', 
                       linewidth=2, label='SW estimate')
            
            # Add SW min-max ranges as purple error bars
            for i, (x, median, min_val, max_val) in enumerate(zip(sw_x, sw_estimates_median, 
                                                                sw_estimates_min, sw_estimates_max)):
                # Plot vertical line from min to max
                ax.plot([x, x], [min_val, max_val], color=sw_color, linestyle='-', 
                       linewidth=2, alpha=0.7)
                
                # Add horizontal caps at min/max
                cap_width = 0.08 * x  # Scale cap width with x value
                ax.plot([x-cap_width, x+cap_width], [min_val, min_val], color=sw_color, 
                       linewidth=2, alpha=0.7)
                ax.plot([x-cap_width, x+cap_width], [max_val, max_val], color=sw_color, 
                       linewidth=2, alpha=0.7)
        
        # Add true value line
        ax.axhline(y=true_value, color=true_value_color, linestyle=':', 
                  linewidth=2, label='True Value')
        
        # Customize subplot
        ax.set_xlabel('Number of Systems')
        ax.set_ylabel(f'Value of {param}')
        ax.set_title(f'Convergence of {param}\nTrue Value: {true_value:.3f}')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # Set x-ticks to expected system counts
        ax.set_xticks(cs.N_VALUES)
        ax.set_xticklabels(cs.N_VALUES)
        
        # Create legend (only for the first subplot to avoid repetition)
        if idx == 0:
            from matplotlib.lines import Line2D
            
            legend_elements = [
                Line2D([0], [0], color=mcmc_5th_color, linestyle=':', linewidth=2, label='MCMC 5th percentile'),
                Line2D([0], [0], color=mcmc_mean_color, linewidth=2, label='MCMC mean'),
                Line2D([0], [0], color=mcmc_95th_color, linestyle=':', linewidth=2, label='MCMC 95th percentile'),
                Line2D([0], [0], color=sw_color, linewidth=2, label='SW estimate'),
                Line2D([0], [0], color=true_value_color, linestyle=':', linewidth=2, label='True Value'),
            ]
            
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add overall title and adjust layout
    fig_hyper.suptitle('MCMC vs SW Hyperparameter Convergence Analysis', 
                      fontsize=16, y=1.02)
    fig_hyper.tight_layout()
    
    # Create runtime comparison
    ax_runtime = fig_runtime.add_subplot(111)
    
    # Prepare runtime data
    mcmc_runtime_x = []
    mcmc_runtime_median = []
    mcmc_runtime_min = []
    mcmc_runtime_max = []
    
    sw_runtime_x = []
    sw_runtime_median = []
    sw_runtime_min = []
    sw_runtime_max = []
    
    for n_systems in cs.N_VALUES:
        # MCMC runtime
        if mcmc_runtime_data[n_systems]:
            mcmc_runtime_x.append(n_systems)
            runtime_values = mcmc_runtime_data[n_systems]
            mcmc_runtime_median.append(np.median(runtime_values))
            mcmc_runtime_min.append(np.min(runtime_values))
            mcmc_runtime_max.append(np.max(runtime_values))
        
        # SW runtime
        if sw_runtime_data[n_systems]:
            sw_runtime_x.append(n_systems)
            runtime_values = sw_runtime_data[n_systems]
            sw_runtime_median.append(np.median(runtime_values))
            sw_runtime_min.append(np.min(runtime_values))
            sw_runtime_max.append(np.max(runtime_values))
    
    # Plot MCMC runtime
    if mcmc_runtime_x:
        # Median line
        ax_runtime.loglog(mcmc_runtime_x, mcmc_runtime_median, color=mcmc_mean_color, 
                         marker='o', markersize=8, linewidth=2, label='MCMC Runtime')
        
        # Add min-max error bars
        for i, (x, median, min_val, max_val) in enumerate(zip(mcmc_runtime_x, mcmc_runtime_median, 
                                                           mcmc_runtime_min, mcmc_runtime_max)):
            # Plot vertical line from min to max
            ax_runtime.plot([x, x], [min_val, max_val], color=mcmc_mean_color, 
                           linewidth=2, alpha=0.7)
            
            # Add caps
            cap_width = 0.08 * x
            ax_runtime.plot([x-cap_width, x+cap_width], [min_val, min_val], color=mcmc_mean_color, 
                           linewidth=2, alpha=0.7)
            ax_runtime.plot([x-cap_width, x+cap_width], [max_val, max_val], color=mcmc_mean_color, 
                           linewidth=2, alpha=0.7)
    
    # Plot SW runtime
    if sw_runtime_x:
        # Median line
        ax_runtime.loglog(sw_runtime_x, sw_runtime_median, color=sw_color, 
                         marker='s', markersize=8, linewidth=2, label='SW Runtime', linestyle='--')
        
        # Add min-max error bars
        for i, (x, median, min_val, max_val) in enumerate(zip(sw_runtime_x, sw_runtime_median, 
                                                           sw_runtime_min, sw_runtime_max)):
            # Plot vertical line from min to max
            ax_runtime.plot([x, x], [min_val, max_val], color=sw_color, 
                           linewidth=2, alpha=0.7)
            
            # Add caps
            cap_width = 0.08 * x
            ax_runtime.plot([x-cap_width, x+cap_width], [min_val, min_val], color=sw_color, 
                           linewidth=2, alpha=0.7)
            ax_runtime.plot([x-cap_width, x+cap_width], [max_val, max_val], color=sw_color, 
                           linewidth=2, alpha=0.7)
    
    # Customize runtime plot
    ax_runtime.set_xlabel('Number of Systems')
    ax_runtime.set_ylabel('Runtime (seconds)')
    ax_runtime.set_title('MCMC vs SW Runtime Performance Analysis')
    ax_runtime.grid(True, which="both", ls="-", alpha=0.2)
    
    # Set x-ticks to expected system counts
    ax_runtime.set_xticks(cs.N_VALUES)
    ax_runtime.set_xticklabels(cs.N_VALUES)
    
    # Create legend for runtime plot
    from matplotlib.lines import Line2D
    
    runtime_legend_elements = [
        Line2D([0], [0], color=mcmc_mean_color, marker='o', markersize=8, linewidth=2, label='MCMC Runtime'),
        Line2D([0], [0], color=sw_color, marker='s', markersize=8, linewidth=2, linestyle='--', label='SW Runtime'),
    ]
    
    ax_runtime.legend(handles=runtime_legend_elements)
    fig_runtime.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = 'simulation_results/comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save figures
    fig_hyper.savefig(f'{output_dir}/hyperparameter_convergence_{pic_name}.pdf', 
                     dpi=300, bbox_inches='tight')
    fig_runtime.savefig(f'{output_dir}/runtime_performance_{pic_name}.pdf', 
                     dpi=300, bbox_inches='tight')
    
    plt.show()

create_percentile_plot(
    "mcmc_results_cpu",
    "sw_results_new_regulariser",
    cs.TARGET_HYPERPARAMETERS,
    "mcmc_vs_sw_new_regulariser",
    num_experiments=10
)