import os
import matplotlib.pyplot as plt
import polars as pl
import constants as cs
import glob
import numpy as np

# def create_percentile_plot(target_hyperparameters, pic_name, num_experiments=cs.NUMBER_EXPERIMENTS):
#     """
#     Create visualization comparing MCMC and SW results with percentile lines and min/max values
    
#     Parameters:
#     -----------
#     mcmc_folder : str
#         Folder containing MCMC results
#     sw_folder : str
#         Folder containing SW results
#     target_hyperparameters : list
#         List of true hyperparameter values
#     pic_name : str
#         Base name for saving the plots
#     num_experiments : int
#         Number of experiment replications
#     """
#     # Set up the visual style
#     plt.rcParams['figure.facecolor'] = 'white'
#     plt.rcParams['axes.grid'] = True
#     plt.rcParams['grid.alpha'] = 0.3
#     plt.rcParams['font.size'] = 10
    
#     # Create figures
#     fig_hyper, axes = plt.subplots(2, 2, figsize=(16, 12))
#     fig_runtime = plt.figure(figsize=(12, 7))
#     axes = axes.flatten()
    
#     # Define colors
#     mcmc_mean_color = '#2ca02c'  # Green for MCMC mean
#     mcmc_5th_color = '#1f77b4'   # Blue for MCMC 5th percentile
#     mcmc_95th_color = '#ff7f0e'  # Orange for MCMC 95th percentile
#     sw_color = '#9467bd'         # Purple for SW
#     true_value_color = '#d62728' # Red for true value
    
#     # Dictionary to hold all data
#     mcmc_data = {p: {n: {'mean': [], '5th percentile': [], '95th percentile': []} 
#                    for n in cs.N_VALUES} for p in cs.HYPERPARAMETER_NAMES}
#     sw_data = {p: {n: {'sw_estimate': [], '5th percentile': [], '95th percentile': []} 
#                 for n in cs.N_VALUES} for p in cs.HYPERPARAMETER_NAMES}
#     mcmc_runtime_data = {n: [] for n in cs.N_VALUES}
#     sw_runtime_data = {n: [] for n in cs.N_VALUES}
    
#     # Load MCMC experiment data
#     for exp_num in range(1, num_experiments + 1):
#         # Find all result files for this experiment
#         mcmc_result_files = glob.glob(f"{cs.PROBLEM}/MCMC/Results/experiment={exp_num}_N=*.parquet")
#         mcmc_time_files = glob.glob(f"{cs.PROBLEM}/MCMC/Runtimes/experiment={exp_num}_N=*.parquet")
        
#         for file_path in mcmc_result_files:
#             try:
#                 df = pl.read_parquet(file_path)
#                 # Extract N value from filename
#                 n_value = int(file_path.split("N=")[1].split(".")[0])
                
#                 # Only process data for expected system counts
#                 if n_value in cs.N_VALUES:
#                     # Store data by hyperparameter, system count, and metric
#                     for param in cs.HYPERPARAMETER_NAMES:
#                         param_data = df.filter(pl.col('hyperparameter') == param)
#                         if not param_data.is_empty():
#                             mcmc_data[param][n_value]['mean'].append(param_data['mean'].to_list()[0])
#                             mcmc_data[param][n_value]['5th percentile'].append(param_data['5th percentile'].to_list()[0])
#                             mcmc_data[param][n_value]['95th percentile'].append(param_data['95th percentile'].to_list()[0])
#             except Exception as e:
#                 print(f"Error loading {file_path}: {e}")
        
#         for file_path in mcmc_time_files:
#             try:
#                 df = pl.read_parquet(file_path)
#                 n_value = int(file_path.split("N=")[1].split(".")[0])
                
#                 # Only process data for expected system counts
#                 if n_value in cs.N_VALUES:
#                     mcmc_runtime_data[n_value].append(df['run time'].to_list()[0])
#             except Exception as e:
#                 print(f"Error loading {file_path}: {e}")
    
#     # Load SW experiment data
#     for exp_num in range(1, num_experiments + 1):
#         sw_result_files = glob.glob(f"{cs.PROBLEM}/SW/Results/experiment={exp_num}_N=*.parquet")
#         sw_time_files = glob.glob(f"{cs.PROBLEM}/SW/Runtimes/experiment={exp_num}_N=*.parquet")
        
#         for file_path in sw_result_files:
#             try:
#                 df = pl.read_parquet(file_path)
#                 n_value = int(file_path.split("N=")[1].split(".")[0])
                
#                 # Only process data for expected system counts
#                 if n_value in cs.N_VALUES:
#                     for param in cs.HYPERPARAMETER_NAMES:
#                         param_data = df.filter(pl.col('hyperparameter') == param)
#                         if not param_data.is_empty():
#                             sw_data[param][n_value]['sw_estimate'].append(param_data['sw_estimate'].to_list()[0])
                            
#                             # Check if SW results have percentiles (they might not)
#                             if '5th percentile' in param_data.columns:
#                                 sw_data[param][n_value]['5th percentile'].append(param_data['5th percentile'].to_list()[0])
#                             if '95th percentile' in param_data.columns:
#                                 sw_data[param][n_value]['95th percentile'].append(param_data['95th percentile'].to_list()[0])
#             except Exception as e:
#                 print(f"Error loading {file_path}: {e}")
        
#         for file_path in sw_time_files:
#             try:
#                 df = pl.read_parquet(file_path)
#                 n_value = int(file_path.split("N=")[1].split(".")[0])
                
#                 # Only process data for expected system counts
#                 if n_value in cs.N_VALUES:
#                     # Check if SW times use 'run time' or 'sw run time'
#                     if 'sw run time' in df.columns:
#                         sw_runtime_data[n_value].append(df['sw run time'].to_list()[0])
#                     else:
#                         sw_runtime_data[n_value].append(df['run time'].to_list()[0])
#             except Exception as e:
#                 print(f"Error loading {file_path}: {e}")
    
#     # Plot hyperparameter results
#     for idx, (param, true_value) in enumerate(zip(cs.HYPERPARAMETER_NAMES, target_hyperparameters)):
#         ax = axes[idx]
        
#         # Prepare data for MCMC plot
#         mcmc_x = []
#         mcmc_means_median = []
#         mcmc_means_min = []
#         mcmc_means_max = []
#         mcmc_5th_median = []
#         mcmc_95th_median = []
        
#         # Prepare data for SW plot
#         sw_x = []
#         sw_estimates_median = []
#         sw_estimates_min = []
#         sw_estimates_max = []
        
#         # Collect data for each system count
#         for n_systems in cs.N_VALUES:
#             # MCMC data
#             if mcmc_data[param][n_systems]['mean']:
#                 mcmc_x.append(n_systems)
                
#                 # Process mean values
#                 means = mcmc_data[param][n_systems]['mean']
#                 mcmc_means_median.append(np.median(means))
#                 mcmc_means_min.append(np.min(means))
#                 mcmc_means_max.append(np.max(means))
                
#                 # Process 5th and 95th percentile values
#                 percentile5 = mcmc_data[param][n_systems]['5th percentile']
#                 mcmc_5th_median.append(np.median(percentile5))
                
#                 percentile95 = mcmc_data[param][n_systems]['95th percentile']
#                 mcmc_95th_median.append(np.median(percentile95))
            
#             # SW data
#             if sw_data[param][n_systems]['sw_estimate']:
#                 sw_x.append(n_systems)
                
#                 # Process SW estimates
#                 estimates = sw_data[param][n_systems]['sw_estimate']
#                 sw_estimates_median.append(np.median(estimates))
#                 sw_estimates_min.append(np.min(estimates))
#                 sw_estimates_max.append(np.max(estimates))
        
#         # Plot MCMC lines
#         if mcmc_x:
#             # 5th percentile median line (dotted)
#             ax.semilogx(mcmc_x, mcmc_5th_median, color=mcmc_5th_color, linestyle=':', 
#                        linewidth=2, label='MCMC 5th percentile')
            
#             # Mean median line (solid)
#             ax.semilogx(mcmc_x, mcmc_means_median, color=mcmc_mean_color, linestyle='-', 
#                        linewidth=2, label='MCMC mean')
            
#             # 95th percentile median line (dotted)
#             ax.semilogx(mcmc_x, mcmc_95th_median, color=mcmc_95th_color, linestyle=':', 
#                        linewidth=2, label='MCMC 95th percentile')
            
#             # Add mean min-max ranges as colored error bars
#             for i, (x, median, min_val, max_val) in enumerate(zip(mcmc_x, mcmc_means_median, 
#                                                                 mcmc_means_min, mcmc_means_max)):
#                 # Plot vertical line from min to max
#                 ax.plot([x, x], [min_val, max_val], color=mcmc_mean_color, linestyle='-', 
#                        linewidth=2, alpha=0.7)
                
#                 # Add horizontal caps at min/max
#                 cap_width = 0.08 * x  # Scale cap width with x value
#                 ax.plot([x-cap_width, x+cap_width], [min_val, min_val], color=mcmc_mean_color, 
#                        linewidth=2, alpha=0.7)
#                 ax.plot([x-cap_width, x+cap_width], [max_val, max_val], color=mcmc_mean_color, 
#                        linewidth=2, alpha=0.7)
        
#         # Plot SW line (estimate)
#         if sw_x:
#             # SW median line (dashed)
#             ax.semilogx(sw_x, sw_estimates_median, color=sw_color, linestyle='-', 
#                        linewidth=2, label='SW estimate')
            
#             # Add SW min-max ranges as purple error bars
#             for i, (x, median, min_val, max_val) in enumerate(zip(sw_x, sw_estimates_median, 
#                                                                 sw_estimates_min, sw_estimates_max)):
#                 # Plot vertical line from min to max
#                 ax.plot([x, x], [min_val, max_val], color=sw_color, linestyle='-', 
#                        linewidth=2, alpha=0.7)
                
#                 # Add horizontal caps at min/max
#                 cap_width = 0.08 * x  # Scale cap width with x value
#                 ax.plot([x-cap_width, x+cap_width], [min_val, min_val], color=sw_color, 
#                        linewidth=2, alpha=0.7)
#                 ax.plot([x-cap_width, x+cap_width], [max_val, max_val], color=sw_color, 
#                        linewidth=2, alpha=0.7)
        
#         # Add true value line
#         ax.axhline(y=true_value, color=true_value_color, linestyle=':', 
#                   linewidth=2, label='True Value')
        
#         # Customize subplot
#         ax.set_xlabel('Number of Systems')
#         ax.set_ylabel(f'Value of {param}')
#         ax.set_title(f'Convergence of {param}\nTrue Value: {true_value:.3f}')
#         ax.grid(True, which="both", ls="-", alpha=0.2)
        
#         # Set x-ticks to expected system counts
#         ax.set_xticks(cs.N_VALUES)
#         ax.set_xticklabels(cs.N_VALUES)
        
#         # Create legend (only for the first subplot to avoid repetition)
#         if idx == 0:
#             from matplotlib.lines import Line2D
            
#             legend_elements = [
#                 Line2D([0], [0], color=mcmc_5th_color, linestyle=':', linewidth=2, label='MCMC 5th percentile'),
#                 Line2D([0], [0], color=mcmc_mean_color, linewidth=2, label='MCMC mean'),
#                 Line2D([0], [0], color=mcmc_95th_color, linestyle=':', linewidth=2, label='MCMC 95th percentile'),
#                 Line2D([0], [0], color=sw_color, linewidth=2, label='SW estimate'),
#                 Line2D([0], [0], color=true_value_color, linestyle=':', linewidth=2, label='True Value'),
#             ]
            
#             ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     # Add overall title and adjust layout
#     fig_hyper.suptitle('MCMC vs SW Hyperparameter Convergence Analysis', 
#                       fontsize=16, y=1.02)
#     fig_hyper.tight_layout()
    
#     # Create runtime comparison
#     ax_runtime = fig_runtime.add_subplot(111)
    
#     # Prepare runtime data
#     mcmc_runtime_x = []
#     mcmc_runtime_median = []
#     mcmc_runtime_min = []
#     mcmc_runtime_max = []
    
#     sw_runtime_x = []
#     sw_runtime_median = []
#     sw_runtime_min = []
#     sw_runtime_max = []
    
#     for n_systems in cs.N_VALUES:
#         # MCMC runtime
#         if mcmc_runtime_data[n_systems]:
#             mcmc_runtime_x.append(n_systems)
#             runtime_values = mcmc_runtime_data[n_systems]
#             mcmc_runtime_median.append(np.median(runtime_values))
#             mcmc_runtime_min.append(np.min(runtime_values))
#             mcmc_runtime_max.append(np.max(runtime_values))
        
#         # SW runtime
#         if sw_runtime_data[n_systems]:
#             sw_runtime_x.append(n_systems)
#             runtime_values = sw_runtime_data[n_systems]
#             sw_runtime_median.append(np.median(runtime_values))
#             sw_runtime_min.append(np.min(runtime_values))
#             sw_runtime_max.append(np.max(runtime_values))
    
#     # Plot MCMC runtime
#     if mcmc_runtime_x:
#         # Median line
#         ax_runtime.loglog(mcmc_runtime_x, mcmc_runtime_median, color=mcmc_mean_color, 
#                          marker='o', markersize=8, linewidth=2, label='MCMC Runtime')
        
#         # Add min-max error bars
#         for i, (x, median, min_val, max_val) in enumerate(zip(mcmc_runtime_x, mcmc_runtime_median, 
#                                                            mcmc_runtime_min, mcmc_runtime_max)):
#             # Plot vertical line from min to max
#             ax_runtime.plot([x, x], [min_val, max_val], color=mcmc_mean_color, 
#                            linewidth=2, alpha=0.7)
            
#             # Add caps
#             cap_width = 0.08 * x
#             ax_runtime.plot([x-cap_width, x+cap_width], [min_val, min_val], color=mcmc_mean_color, 
#                            linewidth=2, alpha=0.7)
#             ax_runtime.plot([x-cap_width, x+cap_width], [max_val, max_val], color=mcmc_mean_color, 
#                            linewidth=2, alpha=0.7)
    
#     # Plot SW runtime
#     if sw_runtime_x:
#         # Median line
#         ax_runtime.loglog(sw_runtime_x, sw_runtime_median, color=sw_color, 
#                          marker='s', markersize=8, linewidth=2, label='SW Runtime', linestyle='--')
        
#         # Add min-max error bars
#         for i, (x, median, min_val, max_val) in enumerate(zip(sw_runtime_x, sw_runtime_median, 
#                                                            sw_runtime_min, sw_runtime_max)):
#             # Plot vertical line from min to max
#             ax_runtime.plot([x, x], [min_val, max_val], color=sw_color, 
#                            linewidth=2, alpha=0.7)
            
#             # Add caps
#             cap_width = 0.08 * x
#             ax_runtime.plot([x-cap_width, x+cap_width], [min_val, min_val], color=sw_color, 
#                            linewidth=2, alpha=0.7)
#             ax_runtime.plot([x-cap_width, x+cap_width], [max_val, max_val], color=sw_color, 
#                            linewidth=2, alpha=0.7)
    
#     # Customize runtime plot
#     ax_runtime.set_xlabel('Number of Systems')
#     ax_runtime.set_ylabel('Runtime (seconds)')
#     ax_runtime.set_title('MCMC vs SW Runtime Performance Analysis')
#     ax_runtime.grid(True, which="both", ls="-", alpha=0.2)
    
#     # Set x-ticks to expected system counts
#     ax_runtime.set_xticks(cs.N_VALUES)
#     ax_runtime.set_xticklabels(cs.N_VALUES)
    
#     # Create legend for runtime plot
#     from matplotlib.lines import Line2D
    
#     runtime_legend_elements = [
#         Line2D([0], [0], color=mcmc_mean_color, marker='o', markersize=8, linewidth=2, label='MCMC Runtime'),
#         Line2D([0], [0], color=sw_color, marker='s', markersize=8, linewidth=2, linestyle='--', label='SW Runtime'),
#     ]
    
#     ax_runtime.legend(handles=runtime_legend_elements)
#     fig_runtime.tight_layout()
    
#     # Create output directory if it doesn't exist
#     output_dir = f'{cs.PROBLEM}/comparison'
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save figures
#     fig_hyper.savefig(f'{output_dir}/hyperparameter_convergence_{pic_name}.pdf', 
#                      dpi=300, bbox_inches='tight')
#     fig_runtime.savefig(f'{output_dir}/runtime_performance_{pic_name}.pdf', 
#                      dpi=300, bbox_inches='tight')
    
#     plt.show()

# create_percentile_plot(
#     cs.TARGET_HYPERPARAMETERS,
#     "trial"
# )


def create_percentile_plot(target_hyperparameters, pic_name, num_experiments=cs.NUMBER_EXPERIMENTS):
    """
    Create visualization comparing MCMC and SW results with percentile lines and min/max values
    
    Parameters:
    -----------
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
    
    # Print available hyperparameters to debug
    print(f"Available hyperparameters: {cs.HYPERPARAMETER_NAMES}")
    
    # Check total number of parameters
    total_params = len(cs.HYPERPARAMETER_NAMES)
    print(f"Total parameters: {total_params}")
    
    # Determine how to split parameters based on total count
    if total_params == 8:
        # If we have 8 parameters, assume first 4 are mu, next 4 are tau
        mu_params = cs.HYPERPARAMETER_NAMES[:4]
        mu_true_values = target_hyperparameters[:4]
        tau_params = cs.HYPERPARAMETER_NAMES[4:8]
        tau_true_values = target_hyperparameters[4:8]
    elif total_params == 4:
        # If we have 4 parameters, we need to determine which are mu and which are tau
        # Look for parameter names containing "mu" or "tau"
        mu_params = []
        mu_true_values = []
        tau_params = []
        tau_true_values = []
        
        for i, param_name in enumerate(cs.HYPERPARAMETER_NAMES):
            if "mu" in param_name.lower():
                mu_params.append(param_name)
                mu_true_values.append(target_hyperparameters[i])
            elif "tau" in param_name.lower():
                tau_params.append(param_name)
                tau_true_values.append(target_hyperparameters[i])
            else:
                # If can't determine, assume it's mu
                mu_params.append(param_name)
                mu_true_values.append(target_hyperparameters[i])
    else:
        # For any other count, just split in half
        half_point = total_params // 2
        mu_params = cs.HYPERPARAMETER_NAMES[:half_point]
        mu_true_values = target_hyperparameters[:half_point]
        tau_params = cs.HYPERPARAMETER_NAMES[half_point:]
        tau_true_values = target_hyperparameters[half_point:]
    
    print(f"Mu parameters: {mu_params}")
    print(f"Tau parameters: {tau_params}")
    
    # Dictionary to hold all data
    mcmc_data = {p: {n: {'mean': [], '5th percentile': [], '95th percentile': []} 
                   for n in cs.N_VALUES} for p in cs.HYPERPARAMETER_NAMES}
    sw_data = {p: {n: {'sw_estimate': [], '5th percentile': [], '95th percentile': []} 
                for n in cs.N_VALUES} for p in cs.HYPERPARAMETER_NAMES}
    mcmc_runtime_data = {n: [] for n in cs.N_VALUES}
    sw_runtime_data = {n: [] for n in cs.N_VALUES}
    
    # Load MCMC experiment data
    print("Loading MCMC data...")
    for exp_num in range(1, num_experiments + 1):
        # Find all result files for this experiment
        mcmc_result_files = glob.glob(f"{cs.PROBLEM}/MCMC/Results/experiment={exp_num}_N=*.parquet")
        mcmc_time_files = glob.glob(f"{cs.PROBLEM}/MCMC/Runtimes/experiment={exp_num}_N=*.parquet")
        
        print(f"  Experiment {exp_num}: Found {len(mcmc_result_files)} result files and {len(mcmc_time_files)} time files")
        
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
                print(f"  Error loading {file_path}: {e}")
        
        for file_path in mcmc_time_files:
            try:
                df = pl.read_parquet(file_path)
                n_value = int(file_path.split("N=")[1].split(".")[0])
                
                # Debug info
                print(f"  Loading MCMC runtime from {file_path}")
                print(f"  DataFrame columns: {df.columns}")
                
                # Only process data for expected system counts
                if n_value in cs.N_VALUES:
                    if 'run time' in df.columns:
                        runtime = df['run time'].to_list()[0]
                        mcmc_runtime_data[n_value].append(runtime)
                        print(f"  Added MCMC runtime for N={n_value}: {runtime}")
                    else:
                        print(f"  Warning: 'run time' column not found in {file_path}")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
    
    # Load SW experiment data
    print("Loading SW data...")
    for exp_num in range(1, num_experiments + 1):
        sw_result_files = glob.glob(f"{cs.PROBLEM}/SW/Results/experiment={exp_num}_N=*.parquet")
        sw_time_files = glob.glob(f"{cs.PROBLEM}/SW/Runtimes/experiment={exp_num}_N=*.parquet")
        
        print(f"  Experiment {exp_num}: Found {len(sw_result_files)} result files and {len(sw_time_files)} time files")
        
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
                print(f"  Error loading {file_path}: {e}")
        
        for file_path in sw_time_files:
            try:
                df = pl.read_parquet(file_path)
                n_value = int(file_path.split("N=")[1].split(".")[0])
                
                # Debug info
                print(f"  Loading SW runtime from {file_path}")
                print(f"  DataFrame columns: {df.columns}")
                
                # Only process data for expected system counts
                if n_value in cs.N_VALUES:
                    # Check if SW times use 'run time' or 'sw run time'
                    if 'sw run time' in df.columns:
                        runtime = df['sw run time'].to_list()[0]
                        sw_runtime_data[n_value].append(runtime)
                        print(f"  Added SW runtime for N={n_value}: {runtime}")
                    elif 'run time' in df.columns:
                        runtime = df['run time'].to_list()[0]
                        sw_runtime_data[n_value].append(runtime)
                        print(f"  Added SW runtime for N={n_value}: {runtime}")
                    else:
                        print(f"  Warning: No runtime column found in {file_path}")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
    
    # Debug data availability
    print("\nParameter data availability:")
    for param in cs.HYPERPARAMETER_NAMES:
        has_data = False
        for n in cs.N_VALUES:
            if mcmc_data[param][n]['mean'] or sw_data[param][n]['sw_estimate']:
                has_data = True
                break
        print(f"  Parameter {param} has data: {has_data}")
    
    # Debug runtime data availability
    print("\nRuntime data availability:")
    for n in cs.N_VALUES:
        print(f"  N={n}: MCMC data: {bool(mcmc_runtime_data[n])} ({len(mcmc_runtime_data[n])} values), SW data: {bool(sw_runtime_data[n])} ({len(sw_runtime_data[n])} values)")
    
    # Function to plot a parameter on a given axis
    def plot_param_on_axis(param, true_value, ax, idx, param_group):
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
        
        # Check if we have any data to plot
        has_data = bool(mcmc_x) or bool(sw_x)
        if not has_data:
            print(f"No data available for parameter: {param}")
            ax.text(0.5, 0.5, f"No data available for {param}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title(f"No data for {param}")
            return False
        
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
        
        # Create legend (only for the first subplot in each group to avoid repetition)
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
        
        return True
    
    # Create figures - one for mu, one for tau, one for runtime
    fig_runtime = plt.figure(figsize=(12, 7))
    ax_runtime = fig_runtime.add_subplot(111)
    
    # Define colors
    mcmc_mean_color = '#2ca02c'  # Green for MCMC mean
    mcmc_5th_color = '#1f77b4'   # Blue for MCMC 5th percentile
    mcmc_95th_color = '#ff7f0e'  # Orange for MCMC 95th percentile
    sw_color = '#9467bd'         # Purple for SW
    true_value_color = '#d62728' # Red for true value
    
    # Create mu figure if we have mu parameters
    if mu_params:
        n_mu = len(mu_params)
        n_cols_mu = min(2, n_mu)
        n_rows_mu = (n_mu + n_cols_mu - 1) // n_cols_mu  # Ceiling division
        
        fig_mu = plt.figure(figsize=(16, 6*n_rows_mu))
        
        # Create subplots correctly
        if n_mu == 1:
            axes_mu = [fig_mu.add_subplot(111)]
        else:
            axes_mu = fig_mu.subplots(n_rows_mu, n_cols_mu).flatten()
        
        # Plot mu parameters
        mu_has_data = False
        for idx, (param, true_value) in enumerate(zip(mu_params, mu_true_values)):
            if idx < len(axes_mu):
                has_data = plot_param_on_axis(param, true_value, axes_mu[idx], idx, "mu")
                mu_has_data = mu_has_data or has_data
        
        # Hide unused subplots
        if n_mu < len(axes_mu):
            for i in range(n_mu, len(axes_mu)):
                axes_mu[i].set_visible(False)
        
        # Add overall title for mu parameters
        if mu_has_data:
            fig_mu.suptitle('MCMC vs SW Mu Hyperparameter Convergence Analysis', 
                          fontsize=16, y=1.02)
            fig_mu.tight_layout()
            
            # Create output directory if it doesn't exist
            output_dir = f'{cs.PROBLEM}/comparison'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save mu figure
            fig_mu.savefig(f'{output_dir}/mu_hyperparameter_convergence_{pic_name}.pdf', 
                         dpi=300, bbox_inches='tight')
        else:
            print("No data available for mu parameters")
            plt.close(fig_mu)
    
    # Create tau figure if we have tau parameters
    if tau_params:
        n_tau = len(tau_params)
        n_cols_tau = min(2, n_tau)
        n_rows_tau = (n_tau + n_cols_tau - 1) // n_cols_tau  # Ceiling division
        
        fig_tau = plt.figure(figsize=(16, 6*n_rows_tau))
        
        # Create subplots correctly
        if n_tau == 1:
            axes_tau = [fig_tau.add_subplot(111)]
        else:
            axes_tau = fig_tau.subplots(n_rows_tau, n_cols_tau).flatten()
        
        # Plot tau parameters
        tau_has_data = False
        for idx, (param, true_value) in enumerate(zip(tau_params, tau_true_values)):
            if idx < len(axes_tau):
                has_data = plot_param_on_axis(param, true_value, axes_tau[idx], idx, "tau")
                tau_has_data = tau_has_data or has_data
        
        # Hide unused subplots
        if n_tau < len(axes_tau):
            for i in range(n_tau, len(axes_tau)):
                axes_tau[i].set_visible(False)
        
        # Add overall title for tau parameters
        if tau_has_data:
            fig_tau.suptitle('MCMC vs SW Tau Hyperparameter Convergence Analysis', 
                           fontsize=16, y=1.02)
            fig_tau.tight_layout()
            
            # Create output directory if it doesn't exist
            output_dir = f'{cs.PROBLEM}/comparison'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save tau figure
            fig_tau.savefig(f'{output_dir}/tau_hyperparameter_convergence_{pic_name}.pdf', 
                          dpi=300, bbox_inches='tight')
        else:
            print("No data available for tau parameters")
            plt.close(fig_tau)
    
    # Create runtime comparison
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
    
    # Debug runtime data
    print("\nProcessed runtime data:")
    print(f"  MCMC data points: {mcmc_runtime_x}")
    print(f"  MCMC medians: {mcmc_runtime_median}")
    print(f"  SW data points: {sw_runtime_x}")
    print(f"  SW medians: {sw_runtime_median}")
    
    # Check if we have runtime data
    runtime_has_data = bool(mcmc_runtime_x) or bool(sw_runtime_x)
    
    if runtime_has_data:
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
        output_dir = f'{cs.PROBLEM}/comparison'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save runtime figure
        fig_runtime.savefig(f'{output_dir}/runtime_performance_{pic_name}.pdf', 
                          dpi=300, bbox_inches='tight')
    else:
        print("No runtime data available")
        # Instead of closing, add a message and save the empty plot
        ax_runtime.text(0.5, 0.5, "No runtime data available", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax_runtime.transAxes, fontsize=14)
        ax_runtime.set_title('Runtime Performance Analysis (No Data)')
        
        # Create output directory if it doesn't exist
        output_dir = f'{cs.PROBLEM}/comparison'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the empty runtime figure
        fig_runtime.tight_layout()
        fig_runtime.savefig(f'{output_dir}/runtime_performance_{pic_name}.pdf', 
                          dpi=300, bbox_inches='tight')
    
    # Create a combined figure for original compatibility (optional)
    # Only include parameters with data
    params_with_data = []
    true_values_with_data = []
    
    for i, param in enumerate(cs.HYPERPARAMETER_NAMES):
        has_data = False
        for n in cs.N_VALUES:
            if mcmc_data[param][n]['mean'] or sw_data[param][n]['sw_estimate']:
                has_data = True
                break
        
        if has_data and i < len(target_hyperparameters):
            params_with_data.append(param)
            true_values_with_data.append(target_hyperparameters[i])
    
    # Only create combined figure if we have data
    if params_with_data:
        # Determine number of rows and columns
        n_params = len(params_with_data)
        n_cols = min(2, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
        
        fig_combined = plt.figure(figsize=(16, 6*n_rows))
        
        # Create subplots correctly
        if n_params == 1:
            axes_combined = [fig_combined.add_subplot(111)]
        else:
            axes_combined = fig_combined.subplots(n_rows, n_cols).flatten()
        
        # Plot parameters on combined figure
        for idx, (param, true_value) in enumerate(zip(params_with_data, true_values_with_data)):
            if idx < len(axes_combined):
                plot_param_on_axis(param, true_value, axes_combined[idx], idx, "combined")
        
        # Hide unused subplots
        if n_params < len(axes_combined):
            for i in range(n_params, len(axes_combined)):
                axes_combined[i].set_visible(False)
        
        fig_combined.suptitle('MCMC vs SW Hyperparameter Convergence Analysis', 
                            fontsize=16, y=1.02)
        fig_combined.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = f'{cs.PROBLEM}/comparison'
        os.makedirs(output_dir, exist_ok=True)
        
        fig_combined.savefig(f'{output_dir}/hyperparameter_convergence_{pic_name}.pdf', 
                           dpi=300, bbox_inches='tight')
    
    # Show all figures
    plt.show()

create_percentile_plot(
    cs.TARGET_HYPERPARAMETERS,
    "trial"
)