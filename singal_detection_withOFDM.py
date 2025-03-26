from enum import auto
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from global_parameters import *  
import numpy as np
from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal
import tensorflow as tf
from tensorflow import keras
Model = keras.Model
layers = keras.layers
Callback = keras.callbacks.Callback
import matplotlib.pyplot as plt
import time
import gc
from datetime import datetime

class MultiModelBCP(Callback):
    """
    A custom Keras callback for tracking and comparing multiple models' training metrics.
    This callback efficiently collects, stores, and visualizes training and validation metrics
    across different models, enabling easy comparison of model performance.
    """

    def __init__(self, model_name, dataset_type="default", sampling_rate=10, max_points=1000):
        """
        Initialize the MultiModelBCP callback.

        Args:
            model_name (str): Unique identifier for the model being tracked
            dataset_type (str): Type of dataset used for training (default: "default")
            sampling_rate (int): How often to sample metrics (every N batches)
            max_points (int): Maximum number of data points to store to manage memory
        """
        super(MultiModelBCP, self).__init__()
        self.model_name = model_name
        self.dataset_type = dataset_type
        self.sampling_rate = sampling_rate  # Sample every N batches
        self.max_points = max_points  # Maximum number of data points to record
        
        # Downsampled metrics
        self.batch_loss = []
        self.batch_bit_err = []
        self.epoch_loss = []
        self.epoch_bit_err = []
        self.val_epoch_loss = []
        self.val_epoch_bit_err = []
        
        # Update-based tracking
        self.update_counts = []
        self.metrics_by_updates = {
            "loss": [],
            "bit_err": [],
            "val_loss": [],
            "val_bit_err": [],
            "val_update_counts":[]
        }
        self.current_update_count = 0
        self.batch_counter = 0

    def on_train_batch_end(self, batch, logs=None):
        """
        Capture metrics at the end of each training batch.
        Implements downsampling to manage memory usage.

        Args:
            batch: Current batch index
            logs: Dictionary containing batch metrics
        """
        logs = logs or {}
        self.current_update_count += 1
        self.batch_counter += 1
        
        # Only record metrics at sampling points
        if self.batch_counter % self.sampling_rate == 0:
            # Handle maximum points limit, remove old data if needed
            if len(self.batch_loss) >= self.max_points:
                # Remove half of the old data points to save memory
                self.batch_loss = self.batch_loss[len(self.batch_loss)//2:]
                self.batch_bit_err = self.batch_bit_err[len(self.batch_bit_err)//2:]
                self.update_counts = self.update_counts[len(self.update_counts)//2:]
                self.metrics_by_updates["loss"] = self.metrics_by_updates["loss"][len(self.metrics_by_updates["loss"])//2:]
                self.metrics_by_updates["bit_err"] = self.metrics_by_updates["bit_err"][len(self.metrics_by_updates["bit_err"])//2:]
            
            self.batch_loss.append(logs.get('loss', 0))
            self.batch_bit_err.append(logs.get('bit_err', 0))
            self.update_counts.append(self.current_update_count)
            self.metrics_by_updates["loss"].append(logs.get('loss', 0))
            self.metrics_by_updates["bit_err"].append(logs.get('bit_err', 0))

    def on_epoch_end(self, epoch, logs=None):
        """
        Capture metrics at the end of each training epoch.
        Records both training and validation metrics.

        Args:
            epoch: Current epoch index
            logs: Dictionary containing epoch metrics
        """
        logs = logs or {}
        self.epoch_loss.append(logs.get('loss', 0))
        self.epoch_bit_err.append(logs.get('bit_err', 0))
        self.val_epoch_loss.append(logs.get('val_loss', 0))
        self.val_epoch_bit_err.append(logs.get('val_bit_err', 0))
        
        # Record validation metrics
        self.metrics_by_updates["val_loss"].append(logs.get('val_loss', 0))
        self.metrics_by_updates["val_bit_err"].append(logs.get('val_bit_err', 0))
        self.metrics_by_updates["val_update_counts"].append(self.current_update_count)

    def on_train_end(self, logs=None):
        """
        Save collected metrics to the static dictionary and clean up resources.
        This method is called when model training ends.

        Args:
            logs: Dictionary containing final metrics
        """
        # Save data to static dictionary
        MultiModelBCP.all_models_data[self.model_name] = {
            "batch_loss": self.batch_loss,
            "batch_bit_err": self.batch_bit_err,
            "epoch_loss": self.epoch_loss,
            "epoch_bit_err": self.epoch_bit_err,
            "val_epoch_loss": self.val_epoch_loss,
            "val_epoch_bit_err": self.val_epoch_bit_err,
            "update_counts": self.update_counts,
            "metrics_by_updates": self.metrics_by_updates,
            "final_update_count": self.current_update_count,
            "dataset_type": self.dataset_type
        }
        
        # Clear instance variables to free memory
        self.batch_loss = []
        self.batch_bit_err = []
        self.update_counts = []
        self.metrics_by_updates = {
            "loss": [],
            "bit_err": [],
            "val_loss": [],
            "val_bit_err": [],
            "val_update_counts":[]
        }

    @staticmethod
    def log_manual_data(model_name, epoch_loss, val_bit_err, update_counts=None, dataset_type="meta"):
        """
        Manually record metrics for models that are not trained with Keras fit method.
        Useful for meta-learning algorithms or external models.

        Args:
            model_name (str): Unique identifier for the model
            epoch_loss (list or float): Loss values to record
            val_bit_err (list or float): Validation bit error rates to record
            update_counts (list or int, optional): Parameter update counts. If None, will be auto-generated.
            dataset_type (str): Type of dataset used for training (default: "meta")
        """
        # Convert inputs to lists if they're not already
        if isinstance(epoch_loss, (tf.Tensor, np.ndarray)):
            epoch_loss = epoch_loss.numpy() if hasattr(epoch_loss, 'numpy') else epoch_loss
        epoch_loss_list = epoch_loss if isinstance(epoch_loss, list) else [epoch_loss]
        
        if isinstance(val_bit_err, (tf.Tensor, np.ndarray)):
            val_bit_err = val_bit_err.numpy() if hasattr(val_bit_err, 'numpy') else val_bit_err
        val_bit_err_list = val_bit_err if isinstance(val_bit_err, list) else [val_bit_err]
        
        # If update_counts not provided, create a list from 1 to len(metrics)
        if update_counts is None:
            update_counts = list(range(1, len(epoch_loss_list) + 1))
        
        # Ensure update_counts is a list matching the metrics length
        if not isinstance(update_counts, list):
            update_counts = [update_counts]
        
        # Use the longer of the two for length matching
        max_len = max(len(epoch_loss_list), len(val_bit_err_list))
        if len(update_counts) < max_len:
            update_counts = list(range(1, max_len + 1))
        
        # Create metrics by updates structure
        metrics_by_updates = {
            "loss": epoch_loss_list,
            "bit_err": [],
            "val_loss": [],
            "val_bit_err": val_bit_err_list,
            "val_update_counts": update_counts[:max_len]
        }
        
        MultiModelBCP.all_models_data[model_name] = {
            "batch_loss": [],
            "batch_bit_err": [],
            "epoch_loss": epoch_loss_list,
            "epoch_bit_err": [],
            "val_epoch_loss": [],
            "val_epoch_bit_err": val_bit_err_list,
            # Update-based tracking
            "update_counts": update_counts[:max_len],
            "metrics_by_updates": metrics_by_updates,
            "final_update_count": update_counts[-1] if update_counts else 0,
            "dataset_type": dataset_type
        }
        print(f"Logged {len(epoch_loss_list)} loss values and {len(val_bit_err_list)} validation error values for {model_name}")
        print(f"Final update count: {update_counts[-1] if update_counts else 0}")


    @staticmethod
    def plot_by_updates(save_path="update_comparison.png", models_to_plot=None, dpi=150):
        """
        Plot learning curves based on parameter updates for direct comparison between models.
        Particularly useful for comparing models trained with different batch sizes or optimizers.
        
        Args:
            save_path (str): Path to save the generated plot
            models_to_plot (list, optional): List of model names to include in the plot.
                                            If None, all models will be plotted.
            dpi (int): Resolution of the saved plot in dots per inch
            
        Returns:
            None: Saves the plot to the specified path
        """
        if not MultiModelBCP.all_models_data:
            print("No data to plot.")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, data in MultiModelBCP.all_models_data.items():
            # If models_to_plot is specified, only plot the specified models
            if models_to_plot and model_name not in models_to_plot:
                continue
                
            if "update_counts" in data and "metrics_by_updates" in data:
                update_counts = data["metrics_by_updates"]["val_update_counts"]
                bit_errs = data["metrics_by_updates"]["val_bit_err"]
                
                if update_counts and bit_errs:
                    plt.plot(update_counts, bit_errs, 
                            label=f"{model_name}", marker='s', markersize=3)
        
        plt.xlabel("Number of Parameter Updates")
        plt.ylabel("Val_Bit Error Rate")
        plt.title("Val_Bit Error Rate vs Parameter Updates")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc="best")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Update-based comparison saved to {save_path}")

    @staticmethod
    def clear_data(model_names=None):
        """
        Clear stored data for all models or specific models.
        
        Args:
            model_names (list, optional): List of model names to clear data for.
                                         If None, all data will be cleared.
                                         
        Returns:
            None
        """
        if model_names is None:
            MultiModelBCP.all_models_data = {}
        else:
            for name in model_names:
                if name in MultiModelBCP.all_models_data:
                    del MultiModelBCP.all_models_data[name]

    @staticmethod
    def export_data_for_matlab(output_dir="matlab_exports", format="all", prefix=""):
        """
        Export training data from all models to MATLAB-compatible formats.
        
        Args:
            output_dir (str): Directory path to save exported files
            format (str): Output format ('csv', 'mat', 'json', or 'all')
            prefix (str): Prefix for exported filenames
            
        Returns:
            list: List of paths to exported files
        """
        from datetime import datetime
        
        # Return early if all_models_data is not initialized
        if not hasattr(MultiModelBCP, "all_models_data") or not MultiModelBCP.all_models_data:
            print("Warning: No model data available to export.")
            return []
        
        # Use timestamp as prefix if none provided
        if not prefix:
            prefix = f"model_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        
        # Import MatlabExport class
        try:
            # Assuming MatlabExport class is defined in a separate file
            from matlab_export import MatlabExport
        except ImportError:
            # If import fails, we would need to define MatlabExport class
            # Code for MatlabExport class would be pasted here
            pass
        
        exported_files = []
        
        # Export data based on requested format
        if format.lower() in ["csv", "all"]:
            csv_files = MatlabExport.export_to_csv(
                MultiModelBCP.all_models_data, 
                output_dir=output_dir, 
                prefix=prefix
            )
            exported_files.extend(csv_files)
            print(f"Exported {len(csv_files)} CSV files to {output_dir}")
        
        if format.lower() in ["mat", "all"]:
            try:
                mat_files = MatlabExport.export_to_mat(
                    MultiModelBCP.all_models_data, 
                    output_dir=output_dir, 
                    prefix=prefix
                )
                exported_files.extend(mat_files)
                if mat_files:
                    print(f"Exported MAT file to {mat_files[0]}")
            except Exception as e:
                print(f"Error exporting MAT file: {e}")
        
        if format.lower() in ["json", "all"]:
            json_files = MatlabExport.export_to_json(
                MultiModelBCP.all_models_data, 
                output_dir=output_dir, 
                prefix=prefix
            )
            exported_files.extend(json_files)
            print(f"Exported JSON file to {json_files[0]}")
        
        return exported_files

    @staticmethod
    def generate_matlab_script(output_dir="matlab_exports", exported_files=None, prefix="", 
                            sample_sizes=None):
        """
        Generate a MATLAB script for loading and analyzing exported data
        
        Parameters:
            output_dir: Export directory path
            exported_files: List of exported file paths
            prefix: File name prefix
            sample_sizes: List of sample sizes, such as [50, 100, 500, 1000]
            
        Returns:
            Path to the MATLAB script file
        """
        from datetime import datetime
        import os
        
        if not prefix:
            prefix = f"model_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            
        script_path = os.path.join(output_dir, f"{prefix}analyze_data.m")
        
        # 检测导出的文件格式
        has_mat = any(f.endswith('.mat') for f in exported_files) if exported_files else False
        has_csv = any(f.endswith('.csv') for f in exported_files) if exported_files else False
        has_json = any(f.endswith('.json') for f in exported_files) if exported_files else False
        
        with open(script_path, 'w') as f:
            f.write("%% OFDM Model Training Results Analysis\n")
            f.write("% This script is auto-generated by Python for analyzing model training data\n\n")
            
            f.write("% Clear workspace and close all figures\n")
            f.write("clear;\n")
            f.write("close all;\n\n")
            
            f.write("% Set default figure properties\n")
            f.write("set(0, 'DefaultFigureColor', 'white');\n")
            f.write("set(0, 'DefaultAxesFontSize', 12);\n")
            f.write("set(0, 'DefaultLineLineWidth', 1.5);\n\n")
            
            # Add sample size array definition
            if sample_sizes:
                f.write("% Define sample size array\n")
                f.write(f"sample_sizes = [{', '.join(map(str, sample_sizes))}];\n\n")
                
             # Add loading code based on exported file formats
            if has_mat:
                # Find the first .mat file
                mat_files = [f for f in exported_files if f.endswith('.mat')]
                if mat_files:
                    mat_file_basename = os.path.basename(mat_files[0])
                    f.write("% Load MAT file data\n")
                    f.write(f"load('{mat_file_basename}');\n\n")

                    f.write("% Get and display all model names\n")
                    f.write("vars = whos;\n")  # Use whos to get information about loaded variables
                    f.write("model_names = {vars.name};\n")  # Extract variable names
                    f.write("fprintf('Loaded models: \\n');\n")
                    f.write("for i = 1:length(model_names)\n")
                    f.write("    fprintf('  %s\\n', model_names{i});\n")
                    f.write("end\n\n")

                    f.write("% Create models struct and put all models in it (for easier processing)\n")
                    f.write("models = struct();\n")
                    f.write("for i = 1:length(model_names)\n")
                    f.write("    models.(model_names{i}) = eval(model_names{i});\n")
                    f.write("end\n\n")
                    
                    f.write("% Analyze validation error rate\n")
                    f.write("figure('Name', 'Val_BER Comparison', 'Position', [100, 100, 1200, 600]);\n")
                    f.write("hold on;\n")
                    f.write("legends = {};\n")
                    f.write("for i = 1:length(model_names)\n")
                    f.write("    model = models.(model_names{i});\n")
                    f.write("    try\n")
                    f.write("        if isfield(model, 'metrics_by_updates') && isfield(model.metrics_by_updates, 'val_update_counts') && isfield(model.metrics_by_updates, 'val_bit_err')\n")
                    f.write("            plot(model.metrics_by_updates.val_update_counts, model.metrics_by_updates.val_bit_err, 'LineWidth', 2);\n")
                    f.write("            legends{end+1} = strrep(model_names{i}, '_', '\\_');\n")
                    f.write("        end\n")
                    f.write("    catch e\n")
                    f.write("        fprintf('Error processing model %s: %s\\n', model_names{i}, e.message);\n")
                    f.write("    end\n")
                    f.write("end\n")
                    f.write("if ~isempty(legends)\n")
                    f.write("    xlabel('Update Count');\n")
                    f.write("    ylabel('Val_BER');\n")
                    f.write("    title('Comparison of Val_BER on the Val_Set for Different Models');\n")
                    f.write("    grid on;\n")
                    f.write("    legend(legends, 'Location', 'best');\n")
                    f.write("    set(gca, 'YScale', 'log');\n")  # Using log scale may better display error rates
                    f.write("else\n")
                    f.write("    title('No valid validation error rate data found');\n")
                    f.write("end\n")
                    f.write("hold off;\n\n")
                    
                    f.write("% Analyze training loss\n")
                    f.write("figure('Name', 'Training Loss Comparison', 'Position', [100, 100, 1200, 600]);\n")
                    f.write("hold on;\n")
                    f.write("legends = {};\n")
                    f.write("for i = 1:length(model_names)\n")
                    f.write("    try\n")
                    f.write("        model = models.(model_names{i});\n")
                    f.write("        if isfield(model, 'metrics_by_updates') && isfield(model.metrics_by_updates, 'loss')\n")
                    f.write("            if isfield(model, 'update_counts') && ~isempty(model.update_counts)\n")
                    f.write("                plot(model.update_counts, model.metrics_by_updates.loss, 'LineWidth', 2);\n")
                    f.write("            else\n")
                    f.write("                plot(model.metrics_by_updates.loss, 'LineWidth', 2);\n")
                    f.write("            end\n")
                    f.write("            legends{end+1} = strrep(model_names{i}, '_', '\\_');\n")
                    f.write("        end\n")
                    f.write("    catch e\n")
                    f.write("        fprintf('Error processing training loss for model %s: %s\\n', model_names{i}, e.message);\n")
                    f.write("    end\n")
                    f.write("end\n")
                    f.write("if ~isempty(legends)\n")
                    f.write("    xlabel('Update Count');\n")
                    f.write("    ylabel('Train Loss');\n")
                    f.write("    title('Comparison of Training Loss for Different Models');\n")
                    f.write("    grid on;\n")
                    f.write("    legend(legends, 'Location', 'best');\n")
                    f.write("else\n")
                    f.write("    title('No valid training loss data found');\n")
                    f.write("end\n")
                    f.write("hold off;\n\n")
                    
                     # Add call to sample size comparison function
                    f.write("% Plot performance comparison of models across different sample sizes\n")
                    if sample_sizes:
                        f.write("plot_sample_size_comparison(models, sample_sizes);\n\n")
                    else:
                        f.write("try\n")
                        f.write("    % Try to automatically extract sample sizes and plot comparison\n")
                        f.write("    plot_sample_size_comparison(models);\n")
                        f.write("catch e\n")
                        f.write("    fprintf('Error plotting sample size comparison: %s\\n', e.message);\n")
                        f.write("    fprintf('You can manually call: plot_sample_size_comparison(models, [50, 100, 500, 1000])\\n');\n")
                        f.write("end\n\n")
                    
                else:
                    f.write("% No .mat file found\n\n")
                    
            elif has_json:
                f.write("% Load JSON file data\n")
                f.write("try\n")
                
                json_files = [f for f in exported_files if f.endswith('.json')]
                if json_files:
                    json_file_basename = os.path.basename(json_files[0])
                    f.write("    json_file = fopen('" + json_file_basename + "');\n")
                    f.write("    json_str = char(fread(json_file, inf))';\n")
                    f.write("    fclose(json_file);\n")
                    f.write("    model_data = jsondecode(json_str);\n")
                    f.write("    fprintf('Successfully loaded JSON data\\n');\n")
                    
                    f.write("    % Get model names\n")
                    f.write("    model_names = fieldnames(model_data);\n")
                    f.write("    fprintf('Loaded models: \\n');\n")
                    f.write("    for i = 1:length(model_names)\n")
                    f.write("        fprintf('  %s\\n', model_names{i});\n")
                    f.write("    end\n\n")
                    
                    # Add JSON data analysis code
                    f.write("    % Analyze validation error rate\n")
                    f.write("    figure('Name', 'JSON Data: Validation Error Rate Comparison', 'Position', [100, 100, 1200, 600]);\n")
                    f.write("    hold on;\n")
                    f.write("    % Analysis code...\n")
                    
                    # Add call to sample size comparison function
                    f.write("    % Plot performance comparison of models across different sample sizes\n")
                    if sample_sizes:
                        f.write("    plot_sample_size_comparison(model_data, sample_sizes);\n\n")
                    else:
                        f.write("    try\n")
                        f.write("        % Try to automatically extract sample sizes and plot comparison\n")
                        f.write("        plot_sample_size_comparison(model_data);\n")
                        f.write("    catch e\n")
                        f.write("        fprintf('Error plotting sample size comparison: %s\\n', e.message);\n")
                        f.write("        fprintf('You can manually call: plot_sample_size_comparison(model_data, [50, 100, 500, 1000])\\n');\n")
                        f.write("    end\n\n")
                
                f.write("catch e\n")
                f.write("    fprintf('Failed to load JSON data: %s\\n', e.message);\n")
                f.write("end\n\n")
                
            elif has_csv:
                f.write("% Load CSV file data\n")
                f.write("csv_files = dir(fullfile(pwd, '*.csv'));\n")
                f.write("fprintf('Found %d CSV files\\n', length(csv_files));\n")
                f.write("\n")
                f.write("% Create data structure to save all model data\n")
                f.write("models = struct();\n")
                f.write("\n")
                f.write("% Load all CSV files\n")
                f.write("for i = 1:length(csv_files)\n")
                f.write("    file_name = csv_files(i).name;\n")
                f.write("    [~, name, ~] = fileparts(file_name);\n")
                f.write("    \n")
                f.write("    % Parse file name to get model name and data type\n")
                f.write("    parts = strsplit(name, '_');\n")
                f.write("    if length(parts) >= 2\n")
                f.write("        model_name = parts{1};\n")
                f.write("        for j = 2:length(parts)-1\n")
                f.write("            model_name = [model_name, '_', parts{j}];\n")
                f.write("        end\n")
                f.write("        data_type = parts{end};\n")
                f.write("        \n")
                f.write("        % Create model structure (if it doesn't exist)\n")
                f.write("        if ~isfield(models, model_name)\n")
                f.write("            models.(model_name) = struct();\n")
                f.write("        end\n")
                f.write("        \n")
                f.write("        % Read data\n")
                f.write("        try\n")
                f.write("            data = readtable(file_name);\n")
                f.write("            models.(model_name).(data_type) = data;\n")
                f.write("            fprintf('Loaded: %s (%s)\\n', model_name, data_type);\n")
                f.write("        catch e\n")
                f.write("            fprintf('Failed to load file %s: %s\\n', file_name, e.message);\n")
                f.write("        end\n")
                f.write("    end\n")
                f.write("end\n\n")
                
                f.write("% Check if there is any loaded model data\n")
                f.write("if ~isempty(fieldnames(models))\n")
                f.write("    % Plot validation error rate comparison\n")
                f.write("    figure('Name', 'Validation Error Rate Comparison', 'Position', [100, 100, 1200, 600]);\n")
                f.write("    hold on;\n")
                f.write("    model_names = fieldnames(models);\n")
                f.write("    legends = {};\n")
                f.write("    for i = 1:length(model_names)\n")
                f.write("        try\n")
                f.write("            model = models.(model_names{i});\n")
                f.write("            if isfield(model, 'validation')\n")
                f.write("                plot(model.validation.update_count, model.validation.val_bit_err, 'LineWidth', 2);\n")
                f.write("                legends{end+1} = strrep(model_names{i}, '_', '\\_');\n")
                f.write("            end\n")
                f.write("        catch e\n")
                f.write("            fprintf('Error processing model %s: %s\\n', model_names{i}, e.message);\n")
                f.write("        end\n")
                f.write("    end\n")
                f.write("    if ~isempty(legends)\n")
                f.write("        xlabel('Update Count');\n")
                f.write("        ylabel('Validation Error Rate');\n")
                f.write("        title('Comparison of Error Rates on Validation Set for Different Models');\n")
                f.write("        grid on;\n")
                f.write("        legend(legends, 'Location', 'best');\n")
                f.write("        set(gca, 'YScale', 'log');\n")
                f.write("    else\n")
                f.write("        title('No valid validation data found');\n")
                f.write("    end\n")
                f.write("    hold off;\n\n")
                
                # Add call to sample size comparison function
                f.write("    % Plot performance comparison of models across different sample sizes\n")
                if sample_sizes:
                    f.write("    plot_sample_size_comparison(models, sample_sizes);\n\n")
                else:
                    f.write("    try\n")
                    f.write("        % Try to automatically extract sample sizes and plot comparison\n")
                    f.write("        plot_sample_size_comparison(models);\n")
                    f.write("    catch e\n")
                    f.write("        fprintf('Error plotting sample size comparison: %s\\n', e.message);\n")
                    f.write("        fprintf('You can manually call: plot_sample_size_comparison(models, [50, 100, 500, 1000])\\n');\n")
                    f.write("    end\n\n")
                    
                f.write("else\n")
                f.write("    fprintf('Warning: Failed to load any model data\\n');\n")
                f.write("end\n\n")
                
            # Add data analysis functions
            f.write("%% Custom Analysis Functions\n\n")
            
            # Add sample size comparison function
            f.write("% Function: Compare final performance of different models across sample sizes\n")
            f.write("function plot_sample_size_comparison(models, sample_sizes)\n")
            f.write("    % This function generates a grouped bar chart with sample sizes on the x-axis and\n")
            f.write("    % validation error rate (BER) on the y-axis. Additionally, it connects the corresponding\n")
            f.write("    % bars with a line to illustrate trends.\n")
            f.write("    % Parameters:\n")
            f.write("    %   models - Structure containing model data\n")
            f.write("    %   sample_sizes - Array of sample sizes, e.g., [50, 100, 500, 1000]\n")
            f.write("    \n")
            f.write("    if nargin < 1\n")
            f.write("        error('The \"models\" parameter must be provided');\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    if nargin < 2\n")
            f.write("        sample_sizes = extract_sample_sizes_from_models(models);\n")
            f.write("        if isempty(sample_sizes)\n")
            f.write("            error('Failed to extract sample sizes from model names; please provide the sample_sizes parameter manually');\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    sample_sizes = sort(sample_sizes);\n")
            f.write("    model_names = fieldnames(models);\n")
            f.write("    model_names = model_names(~strcmp(model_names, 'sample_sizes'));\n")
            f.write("    \n")
            f.write("    base_models = {};\n")
            f.write("    for i = 1:length(model_names)\n")
            f.write("        name = model_names{i};\n")
            f.write("        if strcmp(name, 'sample_sizes')\n")
            f.write("            continue;\n")
            f.write("        end\n")
            f.write("        base_name = extract_base_model_name(name);\n")
            f.write("        if ~isempty(base_name) && ~ismember(base_name, base_models)\n")
            f.write("            base_models{end+1} = base_name;\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    fprintf('Found the following base model types:\\n');\n")
            f.write("    for i = 1:length(base_models)\n")
            f.write("        fprintf('  %s\\n', base_models{i});\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    n_models = length(base_models);\n")
            f.write("    n_sizes = length(sample_sizes);\n")
            f.write("    final_errors = NaN(n_models, n_sizes);\n")
            f.write("    \n")
            f.write("    for i = 1:n_models\n")
            f.write("        base_name = base_models{i};\n")
            f.write("        for j = 1:n_sizes\n")
            f.write("            size_str = num2str(sample_sizes(j));\n")
            f.write("            matching_model = find_model_with_size(models, base_name, size_str);\n")
            f.write("            if ~isempty(matching_model)\n")
            f.write("                final_errors(i, j) = extract_final_error(models.(matching_model));\n")
            f.write("            end\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    figure('Name', 'Performance Comparison Across Different Sample Sizes', 'Position', [100, 100, 1200, 600]);\n")
            f.write("    h = bar(final_errors');\n")
            f.write("    set(gca, 'XTick', 1:n_sizes);\n")
            f.write("    set(gca, 'XTickLabel', arrayfun(@num2str, sample_sizes, 'UniformOutput', false));\n")
            f.write("    title('Comparison of Validation Error Rates (BER) Across Different Sample Sizes');\n")
            f.write("    xlabel('Training Sample Size');\n")
            f.write("    ylabel('Validation Error Rate (BER)');\n")
            f.write("    legend_labels = cellfun(@(x) strrep(x, '_', '\\_'), base_models, 'UniformOutput', false);\n")
            f.write("    legend(h, legend_labels, 'Location', 'best');\n")
            f.write("    x_width = 0.8;\n")
            f.write("    group_center = 1:n_sizes;\n")
            f.write("    for i = 1:n_models\n")
            f.write("        offset = (i - (n_models+1)/2) * (x_width/n_models);\n")
            f.write("        for j = 1:n_sizes\n")
            f.write("            if ~isnan(final_errors(i,j))\n")
            f.write("                x_pos = group_center(j) + offset;\n")
            f.write("                text(x_pos, final_errors(i,j), sprintf('%.4f', final_errors(i,j)), ...\n")
            f.write("                     'HorizontalAlignment', 'center', ...\n")
            f.write("                     'VerticalAlignment', 'bottom', ...\n")
            f.write("                     'FontSize', 8);\n")
            f.write("            end\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("    hold on;\n")
            f.write("    for i = 1:length(h)\n")
            f.write("        x_vals = h(i).XEndPoints;\n")
            f.write("        y_vals = h(i).YEndPoints;\n")
            f.write("        plot(x_vals, y_vals, '-o', 'LineWidth', 2, 'Color', h(i).FaceColor, 'HandleVisibility', 'off');\n")
            f.write("    end\n")
            f.write("    hold off;\n")
            f.write("    grid on;\n")
            f.write("    set(gca, 'YScale', 'log');\n")
            f.write("end\n\n")
            
            # Add helper functions
            f.write("% Helper function: Extract sample sizes from model names\n")
            f.write("function sample_sizes = extract_sample_sizes_from_models(models)\n")
            f.write("    model_names = fieldnames(models);\n")
            f.write("    sample_sizes = [];\n")
            f.write("    \n")
            f.write("    for i = 1:length(model_names)\n")
            f.write("        name = model_names{i};\n")
            f.write("        \n")
            f.write("        % Look for pattern like '_number_'\n")
            f.write("        pattern = '_(\d+)_';\n")
            f.write("        matches = regexp(name, pattern, 'tokens');\n")
            f.write("        \n")
            f.write("        if ~isempty(matches)\n")
            f.write("            for j = 1:length(matches)\n")
            f.write("                size_str = matches{j}{1};\n")
            f.write("                size_num = str2double(size_str);\n")
            f.write("                if ~isnan(size_num) && ~ismember(size_num, sample_sizes)\n")
            f.write("                    sample_sizes = [sample_sizes, size_num];\n")
            f.write("                end\n")
            f.write("            end\n")
            f.write("        elseif contains(name, '_WINNER_')\n")
            f.write("            % Extract number from '_WINNER_number'\n")
            f.write("            pattern = '_WINNER_(\d+)';\n")
            f.write("            matches = regexp(name, pattern, 'tokens');\n")
            f.write("            if ~isempty(matches)\n")
            f.write("                size_str = matches{1}{1};\n")
            f.write("                size_num = str2double(size_str);\n")
            f.write("                if ~isnan(size_num) && ~ismember(size_num, sample_sizes)\n")
            f.write("                    sample_sizes = [sample_sizes, size_num];\n")
            f.write("                end\n")
            f.write("            end\n")
            f.write("        elseif contains(name, 'Meta_')\n")
            f.write("            % Extract number from 'Meta_number'\n")
            f.write("            pattern = 'Meta_(\d+)';\n")
            f.write("            matches = regexp(name, pattern, 'tokens');\n")
            f.write("            if ~isempty(matches)\n")
            f.write("                size_str = matches{1}{1};\n")
            f.write("                size_num = str2double(size_str);\n")
            f.write("                if ~isnan(size_num) && ~ismember(size_num, sample_sizes)\n")
            f.write("                    sample_sizes = [sample_sizes, size_num];\n")
            f.write("                end\n")
            f.write("            end\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("end\n\n")
            
            f.write("% Helper function: Extract base model name (excluding sample size)\n")
            f.write("function base_name = extract_base_model_name(name)\n")
            f.write("    % Extract base model name, ignoring sample size part\n")
            f.write("    \n")
            f.write("    % For DNN-type models\n")
            f.write("    if contains(name, 'DNN_')\n")
            f.write("        if contains(name, '_WINNER_')\n")
            f.write("            % DNN_channel_WINNER_size format\n")
            f.write("            parts = strsplit(name, '_WINNER_');\n")
            f.write("            base_name = parts{1};\n")
            f.write("        else\n")
            f.write("            % Only DNN_channel format\n")
            f.write("            base_name = name;\n")
            f.write("        end\n")
            f.write("    % For Meta-type models\n")
            f.write("    elseif contains(name, 'Meta_')\n")
            f.write("        if ~contains(name, 'Meta_DNN') % Exclude cases like Meta_DNN_train\n")
            f.write("            % Meta_size format, use Meta as base name\n")
            f.write("            base_name = 'Meta';\n")
            f.write("        else\n")
            f.write("            base_name = name;\n")
            f.write("        end\n")
            f.write("    else\n")
            f.write("        % Other cases, return original name\n")
            f.write("        base_name = name;\n")
            f.write("    end\n")
            f.write("end\n\n")
            
            f.write("% Helper function: Find model with specific base name and sample size\n")
            f.write("function model_name = find_model_with_size(models, base_name, size_str)\n")
            f.write("    model_names = fieldnames(models);\n")
            f.write("    model_name = '';\n")
            f.write("    \n")
            f.write("    % Try different matching patterns based on base model type\n")
            f.write("    if strcmp(base_name, 'Meta')\n")
            f.write("        % Meta model naming pattern is Meta_size\n")
            f.write("        pattern = ['Meta_' size_str '$'];\n")
            f.write("        for i = 1:length(model_names)\n")
            f.write("            if ~isempty(regexp(model_names{i}, pattern, 'once'))\n")
            f.write("                model_name = model_names{i};\n")
            f.write("                return;\n")
            f.write("            end\n")
            f.write("        end\n")
            f.write("    else\n")
            f.write("        % DNN model naming pattern is DNN_channel_WINNER_size\n")
            f.write("        pattern = [base_name '_WINNER_' size_str '];\n")
            f.write("        for i = 1:length(model_names)\n")
            f.write("            if ~isempty(regexp(model_names{i}, pattern, 'once'))\n")
            f.write("                model_name = model_names{i};\n")
            f.write("                return;\n")
            f.write("            end\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("end\n\n")
            
            f.write("% Helper function: Extract final validation error rate from model\n")
            f.write("function final_error = extract_final_error(model)\n")
            f.write("    if isfield(model, 'metrics_by_updates') && isfield(model.metrics_by_updates, 'val_bit_err') && ~isempty(model.metrics_by_updates.val_bit_err)\n")
            f.write("        % Use data from metrics_by_updates\n")
            f.write("        final_error = model.metrics_by_updates.val_bit_err(end);\n")
            f.write("    elseif isfield(model, 'val_epoch_bit_err') && ~isempty(model.val_epoch_bit_err)\n")
            f.write("        % Use data from val_epoch_bit_err\n")
            f.write("        final_error = model.val_epoch_bit_err(end);\n")
            f.write("    elseif isfield(model, 'validation') && isfield(model.validation, 'val_bit_err') && ~isempty(model.validation.val_bit_err)\n")
            f.write("        % Use validation data exported from CSV\n")
            f.write("        final_error = model.validation.val_bit_err{end};\n")
            f.write("    else\n")
            f.write("        % No valid data\n")
            f.write("        final_error = NaN;\n")
            f.write("    end\n")
            f.write("end\n\n")
            
            f.write("% Function: Compare model performance across different sample sizes\n")
            f.write("function compare_sample_sizes(models, pattern)\n")
            f.write("    % This function compares model performance across different sample sizes\n")
            f.write("    % Parameters:\n")
            f.write("    %   models - Structure containing model data\n")
            f.write("    %   pattern - String pattern for filtering models (e.g., 'WINNER_')\n")
            f.write("    \n")
            f.write("    if nargin < 1\n")
            f.write("        error('The models parameter is required');\n")
            f.write("    end\n")
            f.write("    if nargin < 2\n")
            f.write("        pattern = ''; % Default no filtering\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    figure('Name', ['Performance Comparison Across Different Sample Sizes - ' pattern], 'Position', [100, 100, 1200, 600]);\n")
            f.write("    hold on;\n")
            f.write("    \n")
            f.write("    model_names = fieldnames(models);\n")
            f.write("    matching_models = {};\n")
            f.write("    \n")
            f.write("    % Filter matching models\n")
            f.write("    for i = 1:length(model_names)\n")
            f.write("        if isempty(pattern) || contains(model_names{i}, pattern)\n")
            f.write("            matching_models{end+1} = model_names{i};\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    legends = {};\n")
            f.write("    % Plot matching models\n")
            f.write("    for i = 1:length(matching_models)\n")
            f.write("        try\n")
            f.write("            model = models.(matching_models{i});\n")
            f.write("            if isfield(model, 'validation')\n")
            f.write("                plot(model.validation.update_count, model.validation.val_bit_err, 'LineWidth', 2);\n")
            f.write("                legends{end+1} = strrep(matching_models{i}, '_', '\\_');\n")
            f.write("            elseif isfield(model, 'metrics_by_updates') && isfield(model.metrics_by_updates, 'val_bit_err')\n")
            f.write("                x_data = model.metrics_by_updates.val_update_counts;\n")
            f.write("                y_data = model.metrics_by_updates.val_bit_err;\n")
            f.write("                plot(x_data, y_data, 'LineWidth', 2);\n")
            f.write("                legends{end+1} = strrep(matching_models{i}, '_', '\\_');\n")
            f.write("            end\n")
            f.write("        catch e\n")
            f.write("            fprintf('Error processing model %s: %s\\n', matching_models{i}, e.message);\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    if ~isempty(legends)\n")
            f.write("        xlabel('Update Count');\n")
            f.write("        ylabel('Validation Error Rate');\n")
            f.write("        title(['Performance Comparison of ' pattern ' Models Across Different Sample Sizes']);\n")
            f.write("        grid on;\n")
            f.write("        legend(legends, 'Location', 'best');\n")
            f.write("        set(gca, 'YScale', 'log');\n")
            f.write("    else\n")
            f.write("        title('No matching model data found');\n")
            f.write("    end\n")
            f.write("    hold off;\n")
            f.write("end\n\n")
            
            f.write("% Function: Analyze model convergence speed\n")
            f.write("function analyze_convergence(models, threshold)\n")
            f.write("    % This function analyzes how many updates different models need to reach a specific performance threshold\n")
            f.write("    % Parameters:\n")
            f.write("    %   models - Structure containing model data\n")
            f.write("    %   threshold - Performance threshold (e.g., 0.02 for 2% error rate)\n")
            f.write("    \n")
            f.write("    if nargin < 1\n")
            f.write("        error('The models parameter is required');\n")
            f.write("    end\n")
            f.write("    if nargin < 2\n")
            f.write("        threshold = 0.02; % Default threshold\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    model_names = fieldnames(models);\n")
            f.write("    convergence_updates = zeros(length(model_names), 1);\n")
            f.write("    final_errors = zeros(length(model_names), 1);\n")
            f.write("    valid_models = false(length(model_names), 1);\n")
            f.write("    \n")
            f.write("    for i = 1:length(model_names)\n")
            f.write("        try\n")
            f.write("            model = models.(model_names{i});\n")
            f.write("            if isfield(model, 'validation')\n")
            f.write("                % Find the first update count where error rate is below threshold\n")
            f.write("                idx = find(model.validation.val_bit_err <= threshold, 1, 'first');\n")
            f.write("                if ~isempty(idx)\n")
            f.write("                    convergence_updates(i) = model.validation.update_count(idx);\n")
            f.write("                    valid_models(i) = true;\n")
            f.write("                else\n")
            f.write("                    convergence_updates(i) = NaN; % Not converged\n")
            f.write("                end\n")
            f.write("                \n")
            f.write("                % Record final performance\n")
            f.write("                final_errors(i) = model.validation.val_bit_err(end);\n")
            f.write("                valid_models(i) = true;\n")
            f.write("            elseif isfield(model, 'metrics_by_updates') && isfield(model.metrics_by_updates, 'val_bit_err')\n")
            f.write("                error_values = model.metrics_by_updates.val_bit_err;\n")
            f.write("                update_counts = model.metrics_by_updates.val_update_counts;\n")
            f.write("                \n")
            f.write("                idx = find(error_values <= threshold, 1, 'first');\n")
            f.write("                if ~isempty(idx)\n")
            f.write("                    convergence_updates(i) = update_counts(idx);\n")
            f.write("                    valid_models(i) = true;\n")
            f.write("                else\n")
            f.write("                    convergence_updates(i) = NaN; % Not converged\n")
            f.write("                end\n")
            f.write("                \n")
            f.write("                final_errors(i) = error_values(end);\n")
            f.write("                valid_models(i) = true;\n")
            f.write("            else\n")
            f.write("                convergence_updates(i) = NaN;\n")
            f.write("                final_errors(i) = NaN;\n")
            f.write("            end\n")
            f.write("        catch e\n")
            f.write("            fprintf('Error processing model %s: %s\\n', model_names{i}, e.message);\n")
            f.write("            convergence_updates(i) = NaN;\n")
            f.write("            final_errors(i) = NaN;\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    % Only keep valid model data\n")
            f.write("    valid_idx = find(valid_models);\n")
            f.write("    if isempty(valid_idx)\n")
            f.write("        fprintf('No valid model data found for convergence analysis\\n');\n")
            f.write("        return;\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    model_names = model_names(valid_idx);\n")
            f.write("    convergence_updates = convergence_updates(valid_idx);\n")
            f.write("    final_errors = final_errors(valid_idx);\n")
            f.write("    \n")
            f.write("    % Create results table\n")
            f.write("    results_cell = cell(length(model_names) + 1, 3);\n")
            f.write("    results_cell(1,:) = {'Model', 'Updates_to_Threshold', 'Final_Error'};\n")
            f.write("    \n")
            f.write("    for i = 1:length(model_names)\n")
            f.write("        results_cell{i+1, 1} = model_names{i};\n")
            f.write("        if isnan(convergence_updates(i))\n")
            f.write("            results_cell{i+1, 2} = 'Not converged';\n")
            f.write("        else\n")
            f.write("            results_cell{i+1, 2} = convergence_updates(i);\n")
            f.write("        end\n")
            f.write("        results_cell{i+1, 3} = final_errors(i);\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    % Display results\n")
            f.write("    fprintf('Convergence Analysis Results (Threshold: %.4f):\\n', threshold);\n")
            f.write("    disp(results_cell);\n")
            f.write("    \n")
            f.write("    % Plot convergence updates bar chart\n")
            f.write("    figure('Name', 'Convergence Speed Comparison', 'Position', [100, 100, 1200, 600]);\n")
            f.write("    \n")
            f.write("    % Find valid convergence data\n")
            f.write("    conv_idx = ~isnan(convergence_updates);\n")
            f.write("    if sum(conv_idx) > 0\n")
            f.write("        bar(convergence_updates(conv_idx));\n")
            f.write("        set(gca, 'XTick', 1:sum(conv_idx), 'XTickLabel', model_names(conv_idx), 'XTickLabelRotation', 45);\n")
            f.write("        ylabel(['Updates needed to reach threshold ' num2str(threshold)]);\n")
            f.write("        title(['Convergence Speed of Different Models to ' num2str(threshold) ' Error Rate']);\n")
            f.write("        grid on;\n")
            f.write("    else\n")
            f.write("        text(0.5, 0.5, 'No models reached the convergence threshold', 'HorizontalAlignment', 'center', 'Units', 'normalized');\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    % Plot final error rates\n")
            f.write("    figure('Name', 'Final Error Rate Comparison', 'Position', [100, 100, 1200, 600]);\n")
            f.write("    bar(final_errors);\n")
            f.write("    set(gca, 'XTick', 1:length(model_names), 'XTickLabel', model_names, 'XTickLabelRotation', 45);\n")
            f.write("    ylabel('Final Error Rate');\n")
            f.write("    title('Comparison of Final Error Rates Across Different Models');\n")
            f.write("    grid on;\n")
            f.write("end\n")
                
        print(f"MATLAB analysis script generated at: {script_path}")
        return script_path

class signal_simulator():
    """
    A simulator for generating and processing OFDM signals across different channel models.
    
    This class provides methods for generating random bits, performing OFDM modulation,
    simulating signal transmission through various channel models (AWGN, Rayleigh, Rician, WINNER II),
    and processing the received signals for machine learning applications.
    
    Attributes:
        all_carriers: Array of all available subcarrier indices
        pilot_carriers: Array of indices designated for pilot carriers
        data_carriers: Array of indices designated for data carriers
        payloadBits_per_OFDM: Number of payload bits per OFDM symbol
        channel_WINNER_loaded: Boolean flag indicating if WINNER II channel data is loaded
        SNRdB: Signal-to-noise ratio in dB
    """

    def __init__(self, SNR=10):
        """
        Initialize the signal simulator with default parameters.
        
        Args:
            SNR: Signal-to-noise ratio in dB (default: 10)
        """
        self.all_carriers = np.arange(K)
        self.pilot_carriers = self.all_carriers[::K // P]
        self.data_carriers = np.delete(self.all_carriers, self.pilot_carriers)
        self.payloadBits_per_OFDM = len(self.data_carriers) * mu
        self.channel_WINNER_loaded = False
        self.SNRdB = SNR
    
    def lazy_load_channel(self):
        """
        Load WINNER channel data only when needed to save memory.
        
        This method loads the channel data from a file only when it's first requested,
        improving memory efficiency for simulations that don't use the WINNER channel.
        """
        if not self.channel_WINNER_loaded:
            self.channel_WINNER = np.load('channel_train.npy')
            self.channel_WINNER_loaded = True

    def generate_bits(self, num_samples):
        """
        Generate random binary data for simulation.
        
        Args:
            num_samples: Number of OFDM symbols to generate data for
            
        Returns:
            Array of randomly generated bits with shape (num_samples, payloadBits_per_OFDM)
        """
        return np.random.binomial(n=1, p=0.5, size=(num_samples, self.payloadBits_per_OFDM))
    
    def transmit_signals(self, bits):
        """
        Convert binary data to OFDM signals ready for transmission.
        
        Performs the full OFDM modulation process:
        1. Serial-to-parallel conversion
        2. QAM mapping
        3. OFDM symbol generation
        4. IDFT
        5. Cyclic prefix addition
        
        Args:
            bits: Binary data to be transmitted
            
        Returns:
            OFDM symbols with cyclic prefix ready for transmission
        """
        bits_sp = self.sp(bits)
        qam = self.mapping(bits_sp)
        ofdm_data = self.ofdm_symbol(qam)
        ofdm_time = self.idft(ofdm_data)
        ofdm_with_cp = self.add_cp(ofdm_time)
        return ofdm_with_cp
    
    def received_signals(self, transmit_signals, channel_type):
        """
        Process signals through specified channel model and add appropriate noise.
        
        Simulates signal propagation through different channel types:
        - Rayleigh fading channel
        - Rician fading channel
        - AWGN channel
        - WINNER II channel
        - Mixed channel types
        
        Args:
            transmit_signals: OFDM signals to transmit
            channel_type: Type of channel to simulate ("rayleigh", "rician", "awgn", "WINNER II",
                        "random_mixed", or "sequential_mixed")
            
        Returns:
            Signals after passing through the specified channel and noise addition
        """
        self.channel_type = channel_type
        if self.channel_type == "rayleigh":
            channel = np.sqrt(1 / 2) * np.sqrt(1/num_path) * (np.random.randn(1, num_path) + 1j * np.random.randn(1, num_path))
            channel = channel[:, 0]
        elif self.channel_type == "rician":
            k = 10 ** (rician_factor / 10)
            rician_mu = np.sqrt(k / (k + 1))
            s = np.sqrt(1 / (2 * (k + 1)))
            channel = rician_mu + s * (np.random.randn(1, num_path) + 1j * np.random.randn(1, num_path))
            channel = channel[:, 0]
        elif self.channel_type == "awgn":
            return self.awgn(transmit_signals, self.SNRdB)
        elif self.channel_type == "WINNER":
            self.lazy_load_channel()
            train_size = self.channel_WINNER.shape[0]
            index = np.random.choice(np.arange(train_size), size=1)
            h = self.channel_WINNER[index]
            channel = h[:, 0]
        elif self.channel_type == "random_mixed" or self.channel_type == "sequential_mixed":
            return transmit_signals  
        else:
            raise ValueError("Invalid channel type")
        
        convolved = np.convolve(transmit_signals, channel)
        signal_power = np.mean(abs(convolved**2))
        sigma2 = signal_power * 10**(-self.SNRdB/10)
        noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape) + 1j*np.random.randn(*convolved.shape))
        return convolved + noise
    
    def ofdm_simulate(self, bits, channel_type):
        """
        Perform end-to-end OFDM simulation for a single set of bits.
        
        Simulates the entire OFDM transmission and reception process:
        1. Transmit signal generation
        2. Channel propagation
        3. Cyclic prefix removal
        4. DFT demodulation
        
        Args:
            bits: Binary data to transmit
            channel_type: Type of channel to simulate
            
        Returns:
            Concatenated real and imaginary parts of the demodulated signal
        """
        ofdm_tx = self.transmit_signals(bits)
        ofdm_rx = self.received_signals(ofdm_tx, channel_type)
        ofdm_rx_no_cp = self.remove_cp(ofdm_rx)
        ofdm_demodulation = self.dft(ofdm_rx_no_cp)
        return np.concatenate((np.real(ofdm_demodulation), np.imag(ofdm_demodulation)))
    
    def generate_mixed_dataset(self, channel_types, bits_array, mode="mixed_random"):
        """
        Generate a dataset with samples from multiple channel types.
        
        Args:
            channel_types: List of channel types to use
            bits_array: Binary data for transmission
            mode: Dataset mixing mode ("mixed_random" shuffles the samples,
                 "sequential_mixed" keeps them grouped by channel type)
            
        Returns:
            Tuple containing the mixed samples and their corresponding bits
        """
        num_types = len(channel_types)
        samples_per_type = len(bits_array) // num_types
        mixed_samples = []
        mixed_bits = []

        for i, channel in enumerate(channel_types):
            start_idx = i * samples_per_type
            end_idx = (i + 1) * samples_per_type if i != num_types - 1 else len(bits_array)
            for bits in bits_array[start_idx:end_idx]:
                ofdm_simulate_output = self.ofdm_simulate(bits, channel)
                mixed_samples.append(ofdm_simulate_output)
                mixed_bits.append(bits)

        if mode == "mixed_random":
            combined = list(zip(mixed_samples, mixed_bits))
            np.random.shuffle(combined)
            mixed_samples, mixed_bits = zip(*combined)

        return np.asarray(mixed_samples), np.asarray(mixed_bits)

    def generate_training_dataset(self, channel_type, bits_array, mode="sequential_mixed"):
        """
        Generate a training dataset for a specified channel type or mix of channels.
        
        Args:
            channel_type: Channel type or list of channel types
            bits_array: Binary data for transmission
            mode: Dataset mixing mode when multiple channels are used
            
        Returns:
            Tuple containing the training samples and their corresponding bits
        """
        if isinstance(channel_type, list) or channel_type in ["random_mixed", "sequential_mixed"]:
            if channel_type in ["random_mixed", "sequential_mixed"]:
                channel_types = ["rician", "awgn", "rayleigh"] 
                return self.generate_mixed_dataset(channel_types, bits_array, mode=channel_type)
            return self.generate_mixed_dataset(channel_type, bits_array, mode=mode)
        
        training_sample = []
        for bits in bits_array:
            ofdm_simulate_output = self.ofdm_simulate(bits, channel_type)
            training_sample.append(ofdm_simulate_output)
        
        return np.asarray(training_sample), bits_array
    
    def generate_testing_dataset(self, channel_type, num_samples, mode="sequential_mixed"):
        """
        Generate a testing dataset for a specified channel type or mix of channels.
        
        Args:
            channel_type: Channel type or list of channel types
            num_samples: Number of samples to generate
            mode: Dataset mixing mode when multiple channels are used
            
        Returns:
            Tuple containing the testing samples and their corresponding bits
        """
        bits_array = self.generate_bits(num_samples)
        if isinstance(channel_type, list) or channel_type in ["random_mixed", "sequential_mixed"]:
            if channel_type in ["random_mixed", "sequential_mixed"]:
                channel_types = ["rician", "awgn", "rayleigh"]
                return self.generate_mixed_dataset(channel_types, bits_array, mode=channel_type)
            return self.generate_mixed_dataset(channel_type, bits_array, mode=mode)
        
        testing_sample = []
        for bits in bits_array:
            ofdm_simulate_output = self.ofdm_simulate(bits, channel_type)
            testing_sample.append(ofdm_simulate_output)
        
        return np.asarray(testing_sample), bits_array
    
    def sp(self, bits):
        """
        Serial-to-parallel conversion of bits.
        
        Args:
            bits: Serial bit stream
            
        Returns:
            Bits arranged for QAM mapping
        """
        return bits.reshape((len(self.data_carriers), mu))
    
    def mapping(self, bits_sp):
        """
        Map bits to QAM symbols according to the mapping table.
        
        Args:
            bits_sp: Bits after serial-to-parallel conversion
            
        Returns:
            Array of QAM symbols
        """
        return np.array([mapping_table[tuple(b)] for b in bits_sp])
    
    def ofdm_symbol(self, qam_payload):
        """
        Create OFDM symbol with data and pilot carriers.
        
        Args:
            qam_payload: QAM-mapped data
            
        Returns:
            OFDM symbol with pilot carriers inserted
        """
        symbol = np.zeros(K, dtype=complex)
        symbol[self.pilot_carriers] = pilot_value
        symbol[self.data_carriers] = qam_payload
        return symbol
    
    def idft(self, OFDM_data):
        """
        Perform Inverse Discrete Fourier Transform (IDFT) on OFDM data.
        
        Args:
            OFDM_data: Frequency-domain OFDM symbol
            
        Returns:
            Time-domain OFDM symbol
        """
        return np.fft.ifft(OFDM_data)
    
    def add_cp(self, OFDM_time):
        """
        Add cyclic prefix to OFDM time-domain symbol.
        
        Args:
            OFDM_time: Time-domain OFDM symbol
            
        Returns:
            OFDM symbol with cyclic prefix added
        """
        cp = OFDM_time[-CP:]
        return np.hstack([cp, OFDM_time])
    
    def remove_cp(self, signals):
        """
        Remove cyclic prefix from received OFDM symbol.
        
        Args:
            signals: Received signal with cyclic prefix
            
        Returns:
            Signal with cyclic prefix removed
        """
        return signals[CP:(CP+K)]
    
    def dft(self, signals):
        """
        Perform Discrete Fourier Transform (DFT) on received time-domain signals.
        
        Args:
            signals: Time-domain signals after CP removal
            
        Returns:
            Frequency-domain OFDM symbol
        """
        return np.fft.fft(signals)
    
    def awgn(self, signals, SNRdb):
        """
        Add Additive White Gaussian Noise (AWGN) to signals.
        
        Args:
            signals: Input signals
            SNRdb: Signal-to-noise ratio in dB
            
        Returns:
            Signals with AWGN added
        """
        gamma = 10**(SNRdb/10)
        P = sum(abs(signals) ** 2) / len(signals) if signals.ndim == 1 else sum(sum(abs(signals) ** 2)) / len(signals)
        N0 = P / gamma
        n = sqrt(N0/2) * standard_normal(signals.shape) if isrealobj(signals) else sqrt(N0/2) * (standard_normal(signals.shape) + 1j * standard_normal(signals.shape))
        return signals + n

def create_tf_dataset(x_data, y_data, batch_size, buffer_size=10000, repeat=True, prefetch=True):
    """
    Create a TensorFlow dataset optimized for training neural networks.
    
    This function takes input features and target labels and creates a TensorFlow dataset
    with performance optimizations like shuffling, batching, and prefetching. The dataset
    is configured for efficient training loops with options to repeat indefinitely and
    prefetch data for optimal GPU utilization.
    
    Args:
        x_data: Input features array
        y_data: Target labels array
        batch_size: Number of samples per batch
        buffer_size: Size of the shuffle buffer (default: 10000)
        repeat: Whether to repeat the dataset indefinitely (default: True)
        prefetch: Whether to enable prefetching for performance (default: True)
        
    Returns:
        A TensorFlow dataset configured with the specified optimizations
    """
    
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(buffer_size=min(buffer_size, len(x_data)))
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def bit_err(y_true, y_pred):
    """
    Calculate the bit error rate between predicted and true binary values.
    
    This function computes the bit error rate (BER) between the predicted outputs
    and ground truth values. It thresholds both predictions and ground truth at 0.5,
    compares them element-wise, and returns the error rate as a proportion between 0 and 1.
    
    This is typically used as a custom metric for OFDM signal detection models where
    the goal is to minimize bit errors in the recovered signal.
    
    Args:
        y_true: Ground truth binary values
        y_pred: Model predictions (continuous values between 0 and 1)
        
    Returns:
        The bit error rate as a scalar between 0 and 1
    """
    err = 1 - tf.reduce_mean(
        tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.sign(y_pred - 0.5),
                    tf.cast(tf.sign(y_true - 0.5), tf.float32)
                ), tf.float32
            ), axis=1
        )
    )
    return err

class base_models(Model):
    """
    Base neural network model for OFDM signal detection.
    
    This class serves as the foundation for specialized OFDM detection models,
    defining a common architecture with fully-connected layers. It inherits from
    Keras Model class and provides a default network structure that can be
    extended by subclasses for specific detection algorithms.
    
    Attributes:
        input_dim: Dimension of the input features
        output_dim: Dimension of the output (number of bits per OFDM symbol)
        model: Sequential model with the neural network architecture
    """
    def __init__(self, input_dim, payloadBits_per_OFDM):
        """
        Initialize the base model with specified dimensions.
        
        Args:
            input_dim: Dimension of the input features
            payloadBits_per_OFDM: Number of payload bits per OFDM symbol, 
                                 determines the output dimension
        """
        super(base_models, self).__init__()
        self.input_dim = input_dim
        self.output_dim = payloadBits_per_OFDM
        
        self.model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dense(512, activation='relu'),
            #layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dense(payloadBits_per_OFDM, activation='sigmoid')
        ])
        
    def call(self, inputs, training=False):
        """
        Forward pass through the model.
        
        This method defines how inputs are processed through the network
        during both training and inference.
        
        Args:
            inputs: Input tensor to the model
            training: Boolean indicating whether the model is in training mode
            
        Returns:
            Output predictions from the model
        """
        return self.model(inputs, training=training)

class DNN(base_models):
    """
    Standard Deep Neural Network implementation for OFDM signal detection.
    
    This class extends the base_models class with specific compilation and training
    functionality for conventional deep learning approaches to OFDM detection.
    It uses Adam optimizer and mean squared error loss with bit error rate metric.
    
    The model is designed to detect transmitted bits from received OFDM signals
    after passing through various channel conditions.
    """
    def __init__(self, input_dim, payloadBits_per_OFDM):
        """
        Initialize the DNN model with specified dimensions and compile it.
        
        Args:
            input_dim: Dimension of the input features
            payloadBits_per_OFDM: Number of payload bits per OFDM symbol, 
                                 determines the output dimension
        """
        super(DNN, self).__init__(input_dim, payloadBits_per_OFDM)
        self.compile_model()

    def compile_model(self):
        """
        Compile the model with optimizer, loss function, and metrics.
        
        Sets up the model with Adam optimizer, mean squared error loss,
        and bit error rate metric for training.
        """
        self.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.MeanSquaredError(), 
            metrics=[bit_err]
        )
    
    def train(self, x_train, y_train, epochs=10, batch_size=32, 
              validation_data=None, callbacks=None, dataset_type="default"):
        """
        Train the model on provided data with performance tracking.
        
        This method trains the model while automatically adding a MultiModelBCP
        callback to track performance metrics across training.
        
        Args:
            x_train: Training input data
            y_train: Training target data
            epochs: Number of training epochs (default: 10)
            batch_size: Batch size for training (default: 32)
            validation_data: Optional tuple of validation data (x_val, y_val)
            callbacks: Additional callbacks to use during training
            dataset_type: Type of dataset used for tracking (default: "default")
            
        Returns:
            Keras History object containing training metrics
        """
        final_callbacks = [MultiModelBCP(model_name=f"DNN_{dataset_type}", dataset_type=dataset_type)]
        if callbacks:
            final_callbacks.extend(callbacks if isinstance(callbacks, list) else [callbacks])

        return self.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=final_callbacks,
            verbose="auto"
        )
    
    def clone(self):
        """
        Create a deep copy of the current model.
        
        This method creates a new instance of the DNN class with the same
        architecture and weights as the current model, allowing for
        multiple experiments from the same starting point.
        
        Returns:
            A new DNN model instance with copied weights
        """
        new_model = DNN(self.input_dim, self.output_dim)
        new_model.model.set_weights(self.model.get_weights())
        return new_model

class MetaDNN(base_models):
    """
    Meta-learning Deep Neural Network for adaptive OFDM signal detection.
    
    This class implements a meta-learning approach using Reptile algorithm
    for OFDM signal detection. It adapts to new channel conditions with fewer samples
    than traditional DNN approaches by learning an initialization that can be
    quickly fine-tuned to specific channel conditions.
    
    The model supports early stopping, learning rate scheduling, and maintains
    tracking of meta-learning performance across training iterations.
    """
    def __init__(self, input_dim, payloadBits_per_OFDM, inner_lr=0.01, meta_lr=0.3, mini_size=32,
                 first_decay_steps=500, t_mul=1.1, m_mul=1, alpha=0.001,
                 early_stopping=True, patience=20, min_delta=0.0002, abs_threshold=0.011, 
                 progressive_patience=True, verbose=1):
        """
        Initialize the MetaDNN model with meta-learning parameters.
        
        Args:
            input_dim: Dimension of the input features
            payloadBits_per_OFDM: Number of payload bits per OFDM symbol
            inner_lr: Learning rate for inner loop optimization (default: 0.01)
            meta_lr: Learning rate for meta-optimization (default: 0.3)
            mini_size: Batch size for inner loop updates (default: 32)
            first_decay_steps: Steps for first decay cycle in LR scheduler (default: 500)
            t_mul: Time multiplier for cosine decay restarts (default: 1.1)
            m_mul: Multiplier for cosine decay restarts (default: 1)
            alpha: Minimum learning rate factor (default: 0.001)
            early_stopping: Whether to use early stopping (default: True)
            patience: Number of updates to wait for improvement (default: 20)
            min_delta: Minimum change to qualify as improvement (default: 0.0002)
            abs_threshold: Absolute threshold for early stopping (default: 0.011)
            progressive_patience: Whether to adjust patience based on performance (default: True)
            verbose: Verbosity level of output (default: 1)
        """
        super(MetaDNN, self).__init__(input_dim, payloadBits_per_OFDM)
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.mini_batch_size = mini_size
        self.optimizer = tf.keras.optimizers.legacy.SGD(inner_lr)
        
        self.sampling_rate = 10  
        self.all_epoch_losses = []
        self.all_val_bit_errs = []
        self.update_counts = []
        self.total_updates = 0
        self.inner_updates = 0

        self.best_weights = None
        self.best_val_err = float('inf')
        
        # Early Stop config
        self.early_stopping = early_stopping
        self.patience = patience            
        self.min_delta = min_delta          
        self.abs_threshold = abs_threshold 
        self.progressive_patience = progressive_patience  
        self.verbose = verbose
        self.wait = 0  # Counter for patience
        self.stopped_epoch = 0  # The epoch at which training was stopped
        self.no_improvement_count = 0  # Count for improvement
        
        self.meta_lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=meta_lr,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha
        )
    
    def get_params(self):
        """
        Get the current model parameters (weights).
        
        Returns:
            Current model weights
        """
        return self.get_weights()
    
    def set_params(self, params):
        """
        Set the model parameters (weights).
        
        Args:
            params: Weights to set for the model
        """
        self.set_weights(params)
    
    def clone(self):
        """
        Create a deep copy of the current model.
        
        Returns:
            A new MetaDNN model instance with the same parameters and weights
        """
        model_copy = MetaDNN(
            self.input_dim, self.output_dim, 
            self.inner_lr, self.meta_lr, self.mini_batch_size,
            early_stopping=self.early_stopping,
            patience=self.patience,
            min_delta=self.min_delta,
            abs_threshold=self.abs_threshold,
            progressive_patience=self.progressive_patience,
            verbose=self.verbose
        )
        model_copy.set_params(self.get_params())
        return model_copy
    
    def _should_stop_early(self, val_err):
        """
        Determine if early stopping should be triggered based on validation error.
        
        This method implements a sophisticated early stopping policy with:
        1. Absolute threshold stopping
        2. Progressive patience adjustment based on performance
        3. Standard patience-based stopping
        
        Args:
            val_err: Current validation error rate
            
        Returns:
            Boolean indicating whether to stop training
        """
        if not self.early_stopping:
            return False
        
        # 1. Check absolute threshold - stop if error is low enough
        if val_err <= self.abs_threshold:
            if self.verbose > 0:
                print(f"\nEarly stopping: reached target performance threshold {self.abs_threshold}")
            self.stopped_epoch = self.total_updates
            return True
            
        # 2. Check if waited long enough without improvement
        # Dynamically adjust patience based on performance
        effective_patience = self.patience
        if self.progressive_patience and self.best_val_err < 0.015:
            # Performance is already good, reduce patience
            reduction_factor = max(0.5, (self.best_val_err - 0.009) / 0.006)  # 0.015→1.0, 0.009→0.0
            effective_patience = max(5, int(self.patience * reduction_factor))
            
            if self.verbose > 1 and self.wait % 5 == 0:
                print(f"Adjusted patience: {effective_patience} (base: {self.patience}, "
                    f"current best: {self.best_val_err:.6f})")
        
        if self.wait >= effective_patience:
            self.stopped_epoch = self.total_updates
            if self.verbose > 0:
                print(f"\nEarly stopping triggered at update {self.total_updates}. "
                    f"Best val_bit_err: {self.best_val_err:.6f}")
            return True
                
        return False

    def inner_update(self, x_task, y_task, steps=None):
        """
        Perform inner loop updates on a specific task.
        
        This method implements the inner optimization loop of the Reptile algorithm,
        updating model weights on a specific task (channel condition) and returning
        the weight differences for meta-update.
        
        Args:
            x_task: Input data for the task
            y_task: Target data for the task
            steps: Number of inner update steps (if None, derived from batch size)
            
        Returns:
            Tuple containing weight differences, average loss, and number of steps used
        """
        original_weights = [tf.identity(w) for w in self.get_weights()]
        losses = []

        num_samples = x_task.shape[0]
        # Inner step selection
        if steps is None:
            steps = max(1, int(np.ceil(num_samples / self.mini_batch_size)))
        else:
            steps = min(steps, int(np.ceil(num_samples / self.mini_batch_size)))

        for step in range(steps):
            batch_indices = np.random.choice(num_samples, size=min(self.mini_batch_size, num_samples), replace=False)
            x_batch = tf.gather(x_task, batch_indices)
            y_batch = tf.gather(y_task, batch_indices)

            with tf.GradientTape() as tape:
                preds = self(x_batch, training=True)
                loss = tf.keras.losses.mean_squared_error(y_batch, preds)
                losses.append(tf.reduce_mean(loss))
            
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Get updated weights
        updated_weights = self.get_weights()
        
        # Calculate weight differences
        weight_diffs = [updated - original for updated, original in zip(updated_weights, original_weights)]
        
        # Restore original weights
        self.set_weights(original_weights)
        
        return weight_diffs, tf.reduce_mean(losses), steps
    
    def evaluate(self, x_val, y_val):
        """
        Evaluate model performance on validation data.
        
        Args:
            x_val: Validation input data
            y_val: Validation target data
            
        Returns:
            Bit error rate on the validation data
        """
        preds = self(x_val, training=False)
        return bit_err(y_val, preds).numpy()
    
    def train_reptile(self, tasks, meta_epochs=10, task_steps=None, meta_validation_data=None):
        """
        Train the model using the Reptile meta-learning algorithm.
        
        This method implements the full Reptile meta-learning training procedure:
        1. Inner loop updates on sampled tasks
        2. Meta-update by moving toward task-specific optima
        3. Learning rate scheduling with warmup
        4. Early stopping based on validation performance
        
        Args:
            tasks: List of (x, y) tuples representing different tasks (channel conditions)
            meta_epochs: Maximum number of meta-update iterations (default: 10)
            task_steps: Number of gradient steps per task (default: None, auto-determined)
            meta_validation_data: Optional tuple of (x_val, y_val) for validation
            
        Returns:
            Tuple of (losses, validation errors, update counts) for performance tracking
        """
        start_time = time.time()
        epoch_times = []
        meta_grads = None
        warmup_epochs = min(50, meta_epochs // 10)
        
        # Initialize early stopping variables
        self.wait = 0
        self.best_val_err = float('inf')
        self.best_weights = None
        self.stopped_epoch = 0
        self.no_improvement_count = 0
        
        # Record initial performance
        if meta_validation_data is not None:
            initial_val_err = self.evaluate(*meta_validation_data)
            if self.verbose > 0:
                print(f"Initial validation bit error rate: {initial_val_err:.6f}")
        
        for epoch in range(meta_epochs):
            epoch_start_time = time.time()
            # Get current learning rate - add warmup phase
            if epoch < warmup_epochs:
                # Linear warmup
                current_meta_lr = self.meta_lr * (epoch + 1) / warmup_epochs
            else:
                # Use learning rate schedule
                current_meta_lr = self.meta_lr_schedule(epoch - warmup_epochs)

            task_losses = []

            initial_params = self.get_params()
            
            # Create or reset weight accumulator
            meta_grads = [tf.zeros_like(p) for p in self.get_weights()]
            
            # Inner loop updates tracking (for information)
            epoch_inner_updates = 0
            current_inner_steps = task_steps

            for x_task, y_task in tasks:
                weight_diffs, task_loss, task_steps_used = self.inner_update(x_task, y_task, task_steps)
                epoch_inner_updates += task_steps_used
                task_losses.append(task_loss)
                
                for i, diff in enumerate(weight_diffs):
                    meta_grads[i] += diff
            
            # Apply meta-update
            current_weights = self.get_weights()
            new_weights = []
            
            for i, (curr_w, grad) in enumerate(zip(current_weights, meta_grads)):
                new_weights.append(curr_w + current_meta_lr * grad / len(tasks))
            
            self.set_weights(new_weights)
            
            # Update counters
            self.total_updates += 1
            self.inner_updates += epoch_inner_updates
            
            if epoch % self.sampling_rate == 0 or epoch == meta_epochs - 1:
                epoch_loss = tf.reduce_mean(task_losses)
                self.all_epoch_losses.append(epoch_loss)
                self.update_counts.append(self.total_updates)
                
                # Evaluate on validation data
                val_bit_err = None
                if meta_validation_data is not None:
                    val_bit_err = self.evaluate(*meta_validation_data)
                    self.all_val_bit_errs.append(val_bit_err)
                    
                    if val_bit_err < self.best_val_err - self.min_delta:
                        # Validation error improved significantly
                        improvement = self.best_val_err - val_bit_err
                        self.best_val_err = val_bit_err
                        self.best_weights = [tf.identity(w) for w in self.get_weights()]
                        self.wait = 0
                        self.no_improvement_count = 0
                        
                        # Log for near-optimal performance
                        if val_bit_err < 0.015 and self.verbose > 0:
                            print(f"Validation error improved to {val_bit_err:.6f} (improvement: {improvement:.6f})")
                    else:
                        # Validation error did not improve significantly
                        self.wait += 1
                        self.no_improvement_count += 1
                    
                    # Check early stopping
                    if self._should_stop_early(val_bit_err):
                        break

            if epoch % 50 == 0:
                import gc
                gc.collect()
            
            # Epoch time
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            # Print progress
            if (epoch + 1) % 50 == 0 or (val_bit_err is not None and val_bit_err < 0.015 and (epoch + 1) % 20 == 0):
                avg_time = np.mean(epoch_times[-min(50, len(epoch_times)):])
                print(f"Meta Epoch {epoch + 1}/{meta_epochs}, "
                    f"loss: {epoch_loss.numpy():.6f}, "
                    f"val_bit_err: {val_bit_err:.6f}, "
                    f"LR: {current_meta_lr:.6f}, "
                    f"Avg Time: {avg_time:.4f}s")
        
        # Restore best model weights
        if self.best_weights is not None:
            if self.verbose > 0:
                print(f"Restoring best model with val_bit_err: {self.best_val_err:.6f}")
            self.set_weights(self.best_weights)
            
        total_time = time.time() - start_time
        print(f"Training completed with {self.total_updates} meta-updates and {self.inner_updates} inner-loop updates, Train Time:{total_time:.2f}s")
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Early stopping occurred at update {self.stopped_epoch}")
            
        return self.all_epoch_losses, self.all_val_bit_errs, self.update_counts
        
    def fine_tune(self, x_train, y_train, steps=1):
        """
        Fine-tune the model on new data with a few gradient steps.
        
        This method performs quick adaptation of the meta-learned model
        to new channel conditions (typically WINNER II data) with minimal steps.
        
        Args:
            x_train: Input data for fine-tuning
            y_train: Target data for fine-tuning
            steps: Number of fine-tuning steps (default: 1)
            
        Returns:
            Bit error rate after fine-tuning
        """
        for _ in range(steps):
            with tf.GradientTape() as tape:
                preds = self(x_train, training=True)
                loss = tf.keras.losses.mean_squared_error(y_train, preds)
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return self.evaluate(x_train, y_train)

if __name__ == "__main__":
   
    # Configure GPU memory growth to prevent TensorFlow from allocating all GPU memory at once
    gpus = tf.config.list_physical_devices('GPU')
    print("Available GPU devices:", tf.config.list_physical_devices('GPU'))
    if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU setup error: {e}")

    # Import the experiment configuration system
    from ExperimentConfig import ExperimentConfig, apply_config_to_simulator, create_meta_dnn_from_config
    
    # Create a fully integrated experiment configuration
    config = ExperimentConfig()
    
    # Modify configuration parameters if needed (examples)
    # config.config["global"]["K"] = 128  # Change number of subcarriers
    config.config["dataset"]["train_samples"] = 64000
    config.config["meta_dnn"]["early_stopping"]=False
    # config.config["meta_dnn"]["abs_threshold"] = 0.01
    
    # Save configuration to record experiment settings
    experiment_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_path = config.save_as_python(filepath=f"experiment_configs/ofdm_experiment_{experiment_time}.py")
    print(f"Experiment configuration saved to: {config_path}")
    
    # Export as global parameters file for compatibility (reference only)
    global_params_path = config.export_to_global_parameters(
        output_file=f"experiment_configs/global_parameters_{experiment_time}.py"
    )
    
    # Inject global parameters into global namespace (replaces importing global_parameters module)
    config.inject_globals(globals())
    
    # Set random seed for reproducibility
    seed = config.config["seed"]
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    
    # Get dataset and training parameters from configuration
    dataset_params = config.get_dataset_params()
    dnn_params = config.get_dnn_params()
    output_params = config.get_output_params()
    
    # Set data parameters
    DNN_samples = dataset_params["train_samples"]
    DNN_epoch = dnn_params["epochs"]
    DNN_batch_size = dnn_params["batch_size"]
    channel_types = dataset_params["channel_types"]
    meta_channel_types = dataset_params["meta_channel_types"]
    
    # Ensure output directory exists
    log_dir = output_params.get("log_dir", "experiment_logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create signal simulator instance
    simulator = signal_simulator()
    # All parameters already injected through global variables, just apply SNR
    simulator.SNRdB = config.config["signal"]["SNR"]
    
    # Initialize model and history containers
    models = {}
    histories = {}
    
    # Generate training data
    print(f"Generating {DNN_samples} training samples...")
    bits = simulator.generate_bits(DNN_samples)
    MultiModelBCP.clear_data()
    
    # Training Phase - Standard DNN
    print("=== Train Phase ===")
    for channel in channel_types:
        print(f"\nTraining on {channel} channel...")
        
        # Generate training data
        start_time = time.time()
        x_train, y_train = simulator.generate_training_dataset(channel, bits)
        x_test, y_test = simulator.generate_testing_dataset(channel, dataset_params["test_samples"] // 5)
        print(f"Data generation time: {time.time() - start_time:.2f} seconds")
        
        # Create model
        model_name = f"DNN_{channel}"
        models[model_name] = DNN(
            input_dim=x_train.shape[1],
            payloadBits_per_OFDM=simulator.payloadBits_per_OFDM
        )
        
        # Convert to TensorFlow dataset
        train_dataset = create_tf_dataset(
            x_train, y_train, 
            batch_size=DNN_batch_size,
            repeat=False,
            buffer_size=min(5000, len(x_train))
        )
        
        # Use memory-efficient callback
        callback = MultiModelBCP(
            model_name=model_name, 
            dataset_type=channel,
            sampling_rate=output_params.get("sampling_rate", 10),
            max_points=output_params.get("max_points", 1000)
        )
        
        # Train model
        start_time = time.time()
        histories[model_name] = models[model_name].fit(
            train_dataset,
            epochs=DNN_epoch, 
            validation_data=(x_test, y_test),
            callbacks=[callback],
            verbose=1
        )
        print(f"Model training time: {time.time() - start_time:.2f} seconds")
        
        # Free memory manually
        del x_train, y_train, train_dataset
        gc.collect()
    
    # Meta-learning Phase
    print("\n=== Meta-learning Phase ===")
    meta_tasks = []
    meta_model_name = "Meta_DNN"

    # Calculate meta-update iterations, matching DNN
    total_meta_iteration = int((DNN_samples/DNN_batch_size)*DNN_epoch)
    print(f"Meta-update iterations: {total_meta_iteration}")
    
    # Generate meta-learning tasks for each channel type
    print("Generating meta-learning tasks...")
    start_time = time.time()
    for channel in meta_channel_types:
        channel_bits = simulator.generate_bits(DNN_samples)
        x_task, y_task = simulator.generate_training_dataset(channel, channel_bits)
        meta_tasks.append((x_task, y_task))
        # Free memory immediately
        gc.collect()
    print(f"Meta-learning task generation time: {time.time() - start_time:.2f} seconds")
    
    # Create meta-learning model
    models[meta_model_name] = create_meta_dnn_from_config(
        input_dim=meta_tasks[0][0].shape[1],
        payloadBits_per_OFDM=simulator.payloadBits_per_OFDM,
        config=config
    )
    
    # Create validation set
    meta_x_test, meta_y_test = simulator.generate_testing_dataset(
        "random_mixed", 
        dataset_params["test_samples"] // 5
    )
    
    # Train meta-model
    print("Starting meta-learning training...")
    start_time = time.time()
    losses, val_errs, update_counts = models["Meta_DNN"].train_reptile(
        meta_tasks, 
        meta_epochs=total_meta_iteration, 
        meta_validation_data=(meta_x_test, meta_y_test),
        task_steps=config.config["meta_dnn"]["task_steps"]
    )
    print(f"Meta-learning training time: {time.time() - start_time:.2f} seconds")
    
    # Record meta-model data
    MultiModelBCP.log_manual_data(
        "Meta_DNN_train",
        losses,
        val_errs,
        update_counts=update_counts,
        dataset_type="meta"
    )
    
    # Plot update comparison chart
    plot_file = os.path.join(log_dir, f"update_comparison_{experiment_time}.png")
    dpi = output_params.get("plot_dpi", 300)
    MultiModelBCP.plot_by_updates(save_path=plot_file, dpi=dpi)
    
    # Free up no longer needed models and data
    del losses, val_errs, update_counts, meta_x_test, meta_y_test, meta_tasks
    gc.collect()
    
    # WINNER II Generalization Test Phase
    print("\n=== WINNER II Generalization Test Phase ===")
    MultiModelBCP.clear_data()
    
    # Get fine-tuning parameters
    fine_tuning_params = config.get_fine_tuning_params()
    val_parameter_set = []
    for size in dataset_params["fine_tuning_sizes"]:
        batch_size = fine_tuning_params["batch_sizes"].get(str(size), 32)
        val_epoch = fine_tuning_params["epochs"].get(str(size), 32)
        val_parameter_set.append((size,val_epoch, batch_size))
    
    # Create validation set
    x_WINNER_val, y_WINNER_val = simulator.generate_testing_dataset(
        dataset_params["test_channel"], 
        dataset_params["test_samples"] // 8
    )
    val_epoch = fine_tuning_params["epochs"]
    
    for size, val_epoch,val_batch_size in val_parameter_set:
        DNN_num_update = int((size/val_batch_size)*val_epoch)
        print(f"{size} sample set, updates: {DNN_num_update}")
        
        # Generate WINNER training data
        bits_array = simulator.generate_bits(size)
        x_WINNER_train, y_WINNER_train = simulator.generate_training_dataset(
            dataset_params["test_channel"], 
            bits_array
        )
        
        # Convert to TensorFlow dataset
        train_dataset = create_tf_dataset(
            x_WINNER_train, y_WINNER_train, 
            batch_size=val_batch_size,
            repeat=False
        )
        
        # Traditional DNN fine-tuning
        for channel in channel_types:
            model_name = f"DNN_{channel}_WINNER_{size}"
            dnn_model = models[f"DNN_{channel}"].clone()
            print(f"\nValidating {channel} channel, sample size: {size}")
            
            callback = MultiModelBCP(
                model_name=model_name, 
                dataset_type=f"{channel}_{size}",
                sampling_rate=output_params.get("sampling_rate", 10),
                max_points=output_params.get("max_points", 1000)
            )
            
            dnn_model.fit(
                train_dataset,
                epochs=val_epoch,
                validation_data=(x_WINNER_val, y_WINNER_val),
                callbacks=[callback],
                verbose=1
            )
            
            # Free model memory
            del dnn_model
            gc.collect()
        
        # Meta DNN fine-tuning
        meta_task_WINNER = [(x_WINNER_train, y_WINNER_train)]
        meta_model = models["Meta_DNN"].clone()
        meta_model_name = f"Meta_{size}"
        
        # Adjust inner steps based on dataset size
        task_steps = min(config.config["meta_dnn"]["task_steps"], size//5)
        
        losses, val_errs, update_counts = meta_model.train_reptile(
            meta_task_WINNER, 
            meta_epochs=DNN_num_update, 
            meta_validation_data=(x_WINNER_val, y_WINNER_val),
            task_steps=task_steps
        )
        
        MultiModelBCP.log_manual_data(
            meta_model_name,
            losses,
            val_errs,
            update_counts=update_counts,
            dataset_type=f"WINNER_{size}"
        )
        
        # Free meta-model and data
        del meta_model, losses, val_errs, update_counts, meta_task_WINNER
        del x_WINNER_train, y_WINNER_train, train_dataset
        gc.collect()
    
    # Plot final results
    final_plot_file = os.path.join(log_dir, f"WINNER_generalization_test_{experiment_time}.png")
    MultiModelBCP.plot_by_updates(save_path=final_plot_file, dpi=dpi)

    output_dir = f"matlab_exports/experiment_{experiment_time}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Export model data
    print("\n=== Exporting Model Data to MATLAB ===")
    exported_files = MultiModelBCP.export_data_for_matlab(
        output_dir=output_dir,
        format="mat",
        prefix=f"ofdm_models_{experiment_time}_"
    )

    # Generate MATLAB analysis script
    matlab_script = MultiModelBCP.generate_matlab_script(
        output_dir=output_dir,
        exported_files=exported_files,
        prefix=f"ofdm_models_{experiment_time}_",
        sample_sizes=dataset_params["fine_tuning_sizes"]
    )

    print(f"\nData export complete! Results saved in: {output_dir}")
    
    # Free all models and callback data
    models.clear()
    MultiModelBCP.clear_data()
    gc.collect()
    
    print(f"\nExperiment complete! Results saved to the {log_dir} directory")
    print(f"Configuration file: {config_path}")
    print(f"Global parameters file: {global_params_path}")
    print(f"Update comparison chart: {plot_file}")
    print(f"Generalization test chart: {final_plot_file}")