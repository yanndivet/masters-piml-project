# import os
# import matplotlib.pyplot as plt
# import polars as pl
# import constants as cs

# def create_comparison_visualization(mcmc_results, mcmc_runtime, sw_results, sw_runtime, 
#                                  target_hyperparameters, pic_name):
#     """
#     Create visualization comparing MCMC and SW results side by side
    
#     Parameters:
#     -----------
#     mcmc_results : polars.DataFrame
#         DataFrame containing MCMC results
#     mcmc_runtime : polars.DataFrame
#         DataFrame containing MCMC runtime information
#     sw_results : polars.DataFrame
#         DataFrame containing SW results
#     sw_runtime : polars.DataFrame
#         DataFrame containing SW runtime information
#     target_hyperparameters : list
#         List of true hyperparameter values
#     pic_name : str
#         Base name for saving the plots
#     """
#     # Set up the visual style
#     plt.rcParams['figure.facecolor'] = 'white'
#     plt.rcParams['axes.grid'] = True
#     plt.rcParams['grid.alpha'] = 0.3
#     plt.rcParams['font.size'] = 10
    
#     # Create figures
#     fig_hyper, axes = plt.subplots(2, 2, figsize=(15, 12))
#     fig_runtime = plt.figure(figsize=(10, 6))
#     axes = axes.flatten()
    
#     # Define colors
#     mcmc_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Blue, Green, Orange for MCMC
#     sw_color = '#9467bd'  # Purple for SW
#     true_value_color = '#d62728'  # Red for true value
#     mcmc_metrics = ['5th percentile', 'mean', '95th percentile']
    
#     # Plot hyperparameter convergence
#     for idx, (param, true_value) in enumerate(zip(cs.HYPERPARAMETER_NAMES, target_hyperparameters)):
#         ax = axes[idx]
        
#         # Plot MCMC results
#         mcmc_param_data = mcmc_results.filter(pl.col('hyperparameter') == param)
#         for color, metric in zip(mcmc_colors, mcmc_metrics):
#             ax.semilogx(mcmc_param_data['number of systems'], 
#                        mcmc_param_data[metric], 
#                        color=color, 
#                        label=f'MCMC {metric}',
#                        linewidth=2)
        
#         # Plot SW results
#         sw_param_data = sw_results.filter(pl.col('hyperparameter') == param)
#         ax.semilogx(sw_param_data['number of systems'], 
#                    sw_param_data['sw_estimate'], 
#                    color=sw_color, 
#                    label='SW estimate',
#                    linewidth=2,
#                    linestyle='--')
        
#         # Add true value line
#         ax.axhline(y=true_value, 
#                   color=true_value_color, 
#                   linestyle=':', 
#                   linewidth=2, 
#                   label='True Value')
        
#         # Adjust y-axis limits
#         y_min, y_max = ax.get_ylim()
#         padding = 0.1 * (y_max - y_min)
#         new_y_min = min(y_min, true_value - padding)
#         new_y_max = max(y_max, true_value + padding)
#         ax.set_ylim(new_y_min, new_y_max)
        
#         # Customize subplot
#         ax.set_xlabel('Number of Systems')
#         ax.set_ylabel(f'Value of {param}')
#         ax.set_title(f'Convergence of {param}\nTrue Value: {true_value:.3f}')
#         ax.grid(True, which="both", ls="-", alpha=0.2)
        
#         # Add legend to first plot only
#         if idx == 0:
#             ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     # Add overall title and adjust layout
#     fig_hyper.suptitle('MCMC vs SW Hyperparameter Convergence Analysis', 
#                       fontsize=16, y=1.02)
#     fig_hyper.tight_layout()
    
#     # Create runtime comparison plot
#     ax_runtime = fig_runtime.add_subplot(111)
    
#     # Plot both runtimes
#     ax_runtime.loglog(mcmc_runtime['number of systems'], 
#                      mcmc_runtime['run time'], 
#                      color='#1f77b4',
#                      marker='o',
#                      linewidth=2,
#                      markersize=8,
#                      label='MCMC Runtime')
    
#     ax_runtime.loglog(sw_runtime['number of systems'], 
#                      sw_runtime['sw run time'], 
#                      color=sw_color,
#                      marker='s',
#                      linewidth=2,
#                      markersize=8,
#                      label='SW Runtime',
#                      linestyle='--')
    
#     # Customize runtime plot
#     ax_runtime.set_xlabel('Number of Systems')
#     ax_runtime.set_ylabel('Runtime (seconds)')
#     ax_runtime.set_title('MCMC vs SW Runtime Performance Analysis')
#     ax_runtime.grid(True, which="both", ls="-", alpha=0.2)
#     ax_runtime.legend()
#     fig_runtime.tight_layout()
    
#     # Create output directory if it doesn't exist
#     output_dir = 'simulation_results/comparison'
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save figures
#     fig_hyper.savefig(f'{output_dir}/hyperparameter_convergence_{pic_name}.pdf', 
#                       dpi=300, bbox_inches='tight')
#     fig_runtime.savefig(f'{output_dir}/runtime_performance_{pic_name}.pdf', 
#                       dpi=300, bbox_inches='tight')
    
#     plt.show()

# # Example usage:
# mcmc_folder_name = "mcmc_results_cpu"
# sw_folder_name = "sw_results_gpu"
# df_mcmc_results = pl.read_parquet(f"simulation_results/{mcmc_folder_name}/mcmc_results_N=*.parquet").sort(pl.col("number of systems"))
# df_mcmc_runtime = pl.read_parquet(f"simulation_results/{mcmc_folder_name}/mcmc_times_N=*.parquet").sort(pl.col("number of systems"))
# df_sw_results = pl.read_parquet(f"simulation_results/{sw_folder_name}/sw_results_N=*.parquet").sort(pl.col("number of systems"))
# df_sw_runtime = pl.read_parquet(f"simulation_results/{sw_folder_name}/sw_times_N=*.parquet").sort(pl.col("number of systems"))

# create_comparison_visualization(
#     df_mcmc_results, 
#     df_mcmc_runtime,
#     df_sw_results,
#     df_sw_runtime,
#     cs.TARGET_HYPERPARAMETERS,
#     "mcmc_vs_sw_comparison_cpu"
# )


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
    
    # Expected system counts
    expected_system_counts = [1, 5, 10, 50, 100, 500, 1000]
    
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
                   for n in expected_system_counts} for p in cs.HYPERPARAMETER_NAMES}
    sw_data = {p: {n: {'sw_estimate': [], '5th percentile': [], '95th percentile': []} 
                for n in expected_system_counts} for p in cs.HYPERPARAMETER_NAMES}
    mcmc_runtime_data = {n: [] for n in expected_system_counts}
    sw_runtime_data = {n: [] for n in expected_system_counts}
    
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
                if n_value in expected_system_counts:
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
                if n_value in expected_system_counts:
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
                if n_value in expected_system_counts:
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
                if n_value in expected_system_counts:
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
        for n_systems in expected_system_counts:
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
        ax.set_xticks(expected_system_counts)
        ax.set_xticklabels(expected_system_counts)
        
        # Create legend (only for the first subplot to avoid repetition)
        if idx == 0:
            from matplotlib.lines import Line2D
            
            legend_elements = [
                Line2D([0], [0], color=mcmc_5th_color, linestyle=':', linewidth=2, label='MCMC 5th percentile'),
                Line2D([0], [0], color=mcmc_mean_color, linewidth=2, label='MCMC mean'),
                Line2D([0], [0], color=mcmc_95th_color, linestyle=':', linewidth=2, label='MCMC 95th percentile'),
                Line2D([0], [0], color=sw_color, linewidth=2, label='SW estimate'),
                Line2D([0], [0], color=true_value_color, linestyle=':', linewidth=2, label='True Value'),
                Line2D([], [], color=mcmc_mean_color, marker='_', linestyle='-', markersize=10, label='Min/Max Range')
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
    
    for n_systems in expected_system_counts:
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
    ax_runtime.set_xticks(expected_system_counts)
    ax_runtime.set_xticklabels(expected_system_counts)
    
    # Create legend for runtime plot
    from matplotlib.lines import Line2D
    
    runtime_legend_elements = [
        Line2D([0], [0], color=mcmc_mean_color, marker='o', markersize=8, linewidth=2, label='MCMC Runtime'),
        Line2D([0], [0], color=sw_color, marker='s', markersize=8, linewidth=2, linestyle='--', label='SW Runtime'),
        Line2D([], [], color='black', marker='_', linestyle='-', markersize=10, label='Min/Max Range')
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
    "sw_results_gpu",
    cs.TARGET_HYPERPARAMETERS,
    "mcmc_vs_sw_replications",
    num_experiments=10
)