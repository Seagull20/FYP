import argparse
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

from ExperimentConfig import ExperimentConfig, apply_config_to_simulator, create_meta_dnn_from_config

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
    def plot_by_updates(save_path="update_comparison.png", models_to_plot=None, dpi=300):
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
        
        # Store the models and their final values for legend ordering
        models_with_final_values = []
        
        for model_name, data in MultiModelBCP.all_models_data.items():
            # If models_to_plot is specified, only plot the specified models
            if models_to_plot and model_name not in models_to_plot:
                continue
                
            if "update_counts" in data and "metrics_by_updates" in data:
                update_counts = data["metrics_by_updates"]["val_update_counts"]
                bit_errs = data["metrics_by_updates"]["val_bit_err"]
                
                if update_counts and bit_errs:
                    # Plot the learning curve
                    line, = plt.plot(update_counts, bit_errs, 
                            label=f"{model_name}", marker='s', markersize=3)
                    
                    # Get the final BER value
                    final_ber = bit_errs[-1]
                    final_update = update_counts[-1]
                    
                    # Store the model with its final value for later sorting
                    models_with_final_values.append((model_name, final_ber, final_update, line))
                    
                    # Add annotation for the final BER value
                    plt.annotate(
                        f'{final_ber:.6f}',  # Text with 6 decimal places
                        xy=(final_update, final_ber),  # Point to annotate
                        xytext=(10, 0),  # Offset text by 10 points to the right
                        textcoords='offset points',  # Use offset for text position
                        ha='left',  # Horizontal alignment
                        va='center',  # Vertical alignment
                        fontsize=9,  # Font size
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)  # Add a background box
                    )
        
        # Sort models by final BER values for the legend
        models_with_final_values.sort(key=lambda x: x[1])  # Sort by final BER
        
        # Reorder the legend to match the sorted models
        handles = [model[3] for model in models_with_final_values]
        labels = [f"{model[0]} ({model[1]:.6f})" for model in models_with_final_values]
        
        plt.xlabel("Number of Parameter Updates")
        plt.ylabel("Validation Bit Error Rate")
        plt.title("Validation Bit Error Rate vs Parameter Updates")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add sorted legend with BER values included
        plt.legend(handles, labels, loc="best", fontsize=9)
        
        # Auto-adjust y-axis limits to add some padding
        y_min, y_max = plt.ylim()
        plt.ylim(max(0, y_min * 0.95), y_max * 1.1)  # Add padding at the top for annotations
        
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
    def export_data_for_matlab(output_dir="matlab_exports", prefix=""):
        """
        Export all model data to MATLAB compatible format.
        
        This function performs special processing for Meta model data, ensuring all models
        have consistent data structures, especially extracting validation metrics and
        converting them to a uniform format.
        
        Args:
            output_dir: Directory path to save exported files
            prefix: Prefix for exported filenames
            
        Returns:
            list: List of exported file paths
        """
        from datetime import datetime
        import os
        import numpy as np
        
        # If all_models_data is not initialized, return early
        if not hasattr(MultiModelBCP, "all_models_data") or not MultiModelBCP.all_models_data:
            print("Warning: No model data available for export.")
            return []
        
        # If no prefix is provided, use timestamp
        if not prefix:
            prefix = f"model_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Try to export in MAT format
        try:
            # Create output path
            mat_file = os.path.join(output_dir, f"{prefix}model_data.mat")
            
            # Try to import scipy.io
            try:
                from scipy import io as spio
            except ImportError:
                print("Error: scipy installation required for MAT file export.")
                return []
            
            # Helper function to recursively process complex data types
            def process_complex_data(data):
                """Recursively process data, convert to MATLAB compatible types"""
                import tensorflow as tf
                
                # Process TensorFlow tensors
                if isinstance(data, tf.Tensor):
                    return data.numpy() if hasattr(data, 'numpy') else np.array([float(data)])
                
                # Process NumPy arrays
                elif isinstance(data, np.ndarray):
                    return data.tolist() if data.ndim > 0 else float(data)
                
                # Process NumPy numeric types
                elif isinstance(data, (np.float32, np.float64, np.int32, np.int64)):
                    return float(data)
                
                # Process dictionaries - recursively process all values
                elif isinstance(data, dict):
                    return {k: process_complex_data(v) for k, v in data.items()}
                
                # Process lists or tuples - recursively process all elements
                elif isinstance(data, (list, tuple)):
                    return [process_complex_data(item) for item in data]
                
                # Return other types
                return data
            
            # Special handling for Meta model data
            def normalize_meta_data(model_data, model_name):
                """Ensure Meta model data structures are consistent with DNN models"""
                processed = model_data.copy()
                
                # Check if it's a Meta model
                if "Meta" in model_name:
                    # Ensure metrics_by_updates has val_bit_err field as a simple array
                    if "metrics_by_updates" in processed:
                        metrics = processed["metrics_by_updates"]
                        
                        # If val_bit_err exists but has complex type, try to extract to 1D array
                        if "val_bit_err" in metrics:
                            val_bit_err = metrics["val_bit_err"]
                            
                            # If it's a dictionary, extract numeric array from it
                            if isinstance(val_bit_err, dict):
                                # Try to find a key containing numeric values
                                for k, v in val_bit_err.items():
                                    if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                                        metrics["val_bit_err"] = process_complex_data(v)
                                        print(f"Extracted val_bit_err for {model_name} from dictionary to array")
                                        break
                            else:
                                # Ensure it's a simple array
                                metrics["val_bit_err"] = process_complex_data(val_bit_err)
                        
                        # Process val_update_counts field
                        if "val_update_counts" in metrics:
                            metrics["val_update_counts"] = process_complex_data(metrics["val_update_counts"])
                    
                    # Extract directly from model-level val_epoch_bit_err field
                    if "val_epoch_bit_err" in processed and isinstance(processed["val_epoch_bit_err"], (list, np.ndarray)):
                        val_err = process_complex_data(processed["val_epoch_bit_err"])
                        
                        # If metrics_by_updates.val_bit_err doesn't exist or is empty, use val_epoch_bit_err
                        if ("metrics_by_updates" not in processed or 
                            "val_bit_err" not in processed["metrics_by_updates"] or 
                            not processed["metrics_by_updates"]["val_bit_err"]):
                            
                            if "metrics_by_updates" not in processed:
                                processed["metrics_by_updates"] = {}
                            
                            processed["metrics_by_updates"]["val_bit_err"] = val_err
                            print(f"Using val_epoch_bit_err for {model_name} as metrics_by_updates.val_bit_err")
                            
                            # If no val_update_counts, create a simple sequence
                            if "val_update_counts" not in processed["metrics_by_updates"] or not processed["metrics_by_updates"]["val_update_counts"]:
                                processed["metrics_by_updates"]["val_update_counts"] = list(range(1, len(val_err) + 1))
                
                return processed
            
            # Process all model data
            processed_data = {}
            for model_name, model_data in MultiModelBCP.all_models_data.items():
                # Replace invalid characters in variable names
                valid_name = model_name.replace('-', '_').replace('.', '_').replace(' ', '_')
                
                # Special handling for Meta models
                normalized_data = normalize_meta_data(model_data, model_name)
                
                # Recursively process all data to ensure MATLAB compatibility
                processed_model = process_complex_data(normalized_data)
                
                # Add to processed data dictionary
                processed_data[valid_name] = processed_model
                
                # Print diagnostic information
                if "Meta" in model_name:
                    has_val_bit_err = ("metrics_by_updates" in processed_model and 
                                    "val_bit_err" in processed_model["metrics_by_updates"] and 
                                    processed_model["metrics_by_updates"]["val_bit_err"])
                    print(f"Validation metrics status for model {model_name}: {'contains valid val_bit_err' if has_val_bit_err else 'missing val_bit_err'}")
            
            # Add sample sizes and test channel info (helps MATLAB scripts)
            # Extract sample sizes
            sample_sizes = []
            for model_name in processed_data.keys():
                if "Meta_" in model_name:
                    # Match Meta_NUMBER format
                    import re
                    matches = re.findall(r'Meta_(\d+)', model_name)
                    if matches:
                        size = int(matches[0])
                        if size not in sample_sizes:
                            sample_sizes.append(size)
            
            # If sample sizes found, add to export data
            if sample_sizes:
                processed_data["sample_sizes"] = sorted(sample_sizes)
                print(f"Added sample size information: {sorted(sample_sizes)}")
            
            # Detect test channel (assumed to be extracted from model names)
            test_channel = None
            for model_name in processed_data.keys():
                if "DNN_" in model_name:
                    # Match DNN_CHANNEL_TESTCHANNEL_SIZE format
                    parts = model_name.split('_')
                    if len(parts) >= 4:
                        test_channel = parts[2]
                        break
            
            # If test channel found, add to export data
            if test_channel:
                processed_data["testing_channel"] = test_channel
                print(f"Added test channel information: {test_channel}")
            
            # Save to MAT file
            spio.savemat(mat_file, processed_data)
            
            print(f"Data exported to MAT file: {mat_file}")
            return [mat_file]
        
        except Exception as e:
            print(f"Error exporting to MAT file: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        
    def generate_matlab_script(output_dir="matlab_exports", exported_files=None, prefix="", 
                        sample_sizes=None, testing_channel="rician", training_channels=None):
        """
        Generate MATLAB script based on the fixed data structure for visualizing 
        performance comparisons of different models across sample sizes.
        
        Parameters:
            output_dir: Output directory path
            exported_files: List of exported .mat file paths
            prefix: Filename prefix
            sample_sizes: List of sample sizes, e.g. [50, 100, 500, 1000]
            testing_channel: Channel type used for testing
            training_channels: List of channel types used for training
            
        Returns:
            Path to the generated MATLAB script
        """
        from datetime import datetime
        import os
        
        if not prefix:
            prefix = f"model_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                
        script_path = os.path.join(output_dir, f"{prefix}generalization_visualization.m")
        
        with open(script_path, 'w') as f:
            # MATLAB script header
            f.write("%% OFDM Signal Detection - Generalization Testing Visualization\n")
            f.write("% This script visualizes how different models perform after fine-tuning on different sample sizes\n\n")
            
            f.write("% Clear workspace and close all figures\n")
            f.write("clear;\n")
            f.write("close all;\n\n")
            
            f.write("% Set figure properties\n")
            f.write("set(0, 'DefaultFigureColor', 'white');\n")
            f.write("set(0, 'DefaultAxesFontSize', 12);\n")
            f.write("set(0, 'DefaultLineLineWidth', 1.5);\n\n")
            
            # Load .mat files
            mat_files = [f for f in exported_files if f.endswith('.mat')]
            if mat_files:
                mat_file_basename = os.path.basename(mat_files[0])
                f.write("% Load the data file\n")
                f.write(f"load('{mat_file_basename}');\n\n")

                f.write("% Print loaded models\n")
                f.write("vars = whos;\n")
                f.write("model_names = {vars.name};\n")
                f.write("fprintf('Loaded models: \\n');\n")
                f.write("for i = 1:length(model_names)\n")
                f.write("    fprintf('  %s\\n', model_names{i});\n")
                f.write("end\n\n")

                f.write("% Create struct for easier access\n")
                f.write("models = struct();\n")
                f.write("for i = 1:length(model_names)\n")
                f.write("    if ~strcmp(model_names{i}, 'sample_sizes') && ~strcmp(model_names{i}, 'testing_channel')\n")
                f.write("        models.(model_names{i}) = eval(model_names{i});\n")
                f.write("    end\n")
                f.write("end\n\n")
                
                # Use exported test channel and sample sizes
                f.write("% Use exported test channel and sample sizes if available\n")
                f.write("if exist('testing_channel', 'var')\n")
                f.write("    test_channel = testing_channel;\n")
                f.write("else\n")
                f.write(f"    test_channel = '{testing_channel}';\n")
                f.write("end\n\n")
                
                f.write("if exist('sample_sizes', 'var')\n")
                f.write("    samples = sample_sizes;\n")
                if sample_sizes:
                    f.write("else\n")
                    f.write(f"    samples = [{', '.join(map(str, sample_sizes))}];\n")
                f.write("end\n\n")
                
                # Directly analyze the available models
                if training_channels:
                    channels_str = ", ".join([f"'{ch}'" for ch in training_channels])
                    f.write(f"% Available training channels: [{channels_str}]\n\n")
                
                # Call visualization function
                f.write("% Visualize model performance across sample sizes\n")
                f.write("visualize_performance(models, samples, test_channel);\n\n")
                        
            # === Visualization Functions ===
            f.write("%% Visualization Functions\n\n")
            
            # Main visualization function
            f.write("function visualize_performance(models, sample_sizes, test_channel)\n")
            f.write("    % This function creates a bar chart comparing model performance across different sample sizes\n")
            f.write("    % Parameters:\n")
            f.write("    %   models - Structure containing model data\n")
            f.write("    %   sample_sizes - Array of sample sizes\n")
            f.write("    %   test_channel - Testing channel type\n")
            f.write("    \n")
            f.write("    if nargin < 3\n")
            f.write("        test_channel = 'rician';\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    % Sort sample sizes\n")
            f.write("    sample_sizes = sort(sample_sizes);\n")
            f.write("    \n")
            f.write("    % Force debug display of all models\n")
            f.write("    model_names = fieldnames(models);\n")
            f.write("    fprintf('\\nAll available models in data file:\\n');\n")
            f.write("    for i = 1:length(model_names)\n")
            f.write("        fprintf('  %s\\n', model_names{i});\n")
            f.write("    end\n")
            f.write("    fprintf('\\n');\n")
            f.write("    \n")
            f.write("    % New approach to extract model types\n")
            f.write("    train_channels = {};\n")
            f.write("    \n")
            f.write("    % First, directly identify all training channels from available models\n")
            f.write("    for i = 1:length(model_names)\n")
            f.write("        name = model_names{i};\n")
            f.write("        \n")
            f.write("        % Only process DNN models with the test channel and a size suffix\n")
            f.write("        if contains(name, ['DNN_']) && contains(name, ['_' test_channel '_'])\n")
            f.write("            % Extract the size part to ensure it's a valid model\n")
            f.write("            parts = strsplit(name, '_');\n")
            f.write("            if length(parts) >= 4\n")
            f.write("                % Find the test_channel part\n")
            f.write("                test_ch_idx = 0;\n")
            f.write("                for j = 1:length(parts)\n")
            f.write("                    if strcmp(parts{j}, test_channel)\n")
            f.write("                        test_ch_idx = j;\n")
            f.write("                        break;\n")
            f.write("                    end\n")
            f.write("                end\n")
            f.write("                \n")
            f.write("                if test_ch_idx > 2\n")  
            f.write("                    % Extract all parts between DNN_ and test_channel\n")
            f.write("                    channel_str = strjoin(parts(2:test_ch_idx-1), '_');\n")
            f.write("                    \n")
            f.write("                    % Add to channels if new\n")
            f.write("                    if ~ismember(channel_str, train_channels)\n")
            f.write("                        train_channels{end+1} = channel_str;\n")
            f.write("                        fprintf('Found training channel: %s\\n', channel_str);\n")
            f.write("                    end\n")
            f.write("                end\n")
            f.write("            end\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    % Create base model types including Meta\n")
            f.write("    base_models = {};\n")
            f.write("    for i = 1:length(train_channels)\n")
            f.write("        base_models{end+1} = ['DNN_' train_channels{i}];\n")
            f.write("    end\n")
            f.write("    base_models{end+1} = 'Meta';\n")
            f.write("    \n")
            f.write("    fprintf('\\nDetected base model types: \\n');\n")
            f.write("    for i = 1:length(base_models)\n")
            f.write("        fprintf('  %s\\n', base_models{i});\n")
            f.write("    end\n")
            f.write("    fprintf('\\n');\n")
            f.write("    \n")
            f.write("    % Prepare data array for plotting\n")
            f.write("    num_models = length(base_models);\n")
            f.write("    num_sizes = length(sample_sizes);\n")
            f.write("    performance_data = zeros(num_models, num_sizes);\n")
            f.write("    model_found_flags = zeros(num_models, num_sizes);\n")  # Track if model data was found
            f.write("    \n")
            f.write("    % Extract performance data for each model and sample size\n")
            f.write("    for i = 1:num_models\n")
            f.write("        base_name = base_models{i};\n")
            f.write("        for j = 1:num_sizes\n")
            f.write("            sample_size = sample_sizes(j);\n")
            f.write("            sample_str = num2str(sample_size);\n")
            f.write("            \n")
            f.write("            % Find the corresponding model\n")
            f.write("            model_key = find_model(models, base_name, sample_str, test_channel);\n")
            f.write("            \n")
            f.write("            if ~isempty(model_key)\n")
            f.write("                % Extract error rate from model data\n")
            f.write("                error_rate = extract_error_rate(models.(model_key));\n")
            f.write("                performance_data(i, j) = error_rate;\n")
            f.write("                model_found_flags(i, j) = 1;\n")
            f.write("                fprintf('Found for %s with size %s: %s (error: %.6f)\\n', ...\n")
            f.write("                        base_name, sample_str, model_key, error_rate);\n")
            f.write("            else\n")
            f.write("                fprintf('No model found for %s with size %s\\n', base_name, sample_str);\n")
            f.write("            end\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    % Filter out models with no data\n")
            f.write("    models_with_data = sum(model_found_flags, 2) > 0;\n")
            f.write("    if sum(models_with_data) < num_models\n")
            f.write("        fprintf('\\nRemoving %d models with no data from plot\\n', num_models - sum(models_with_data));\n")
            f.write("        base_models = base_models(models_with_data);\n")
            f.write("        performance_data = performance_data(models_with_data, :);\n")
            f.write("        num_models = length(base_models);\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    % Create the bar chart\n")
            f.write("    figure('Position', [100, 100, 1000, 600]);\n")
            f.write("    h = bar(performance_data');\n")
            f.write("    \n")
            f.write("    % Configure axis\n")
            f.write("    set(gca, 'XTick', 1:num_sizes);\n")
            f.write("    set(gca, 'XTickLabel', arrayfun(@num2str, sample_sizes, 'UniformOutput', false));\n")
            f.write("    \n")
            f.write("    % Add labels and title\n")
            f.write("    xlabel('Training Sample Size');\n")
            f.write("    ylabel('Validation Error Rate (BER)');\n")
            f.write("    title(['Model Performance After Fine-tuning on ' test_channel ' Channel']);\n")
            f.write("    \n")
            f.write("    % Create legend\n")
            f.write("    model_labels = cellfun(@make_pretty_label, base_models, 'UniformOutput', false);\n")
            f.write("    legend(h, model_labels, 'Location', 'best');\n")
            f.write("    \n")
            f.write("    % Add value labels to each bar\n")
            f.write("    width = 0.8;\n")
            f.write("    group_centers = 1:num_sizes;\n")
            f.write("    for i = 1:num_models\n")
            f.write("        offset = (i - (num_models+1)/2) * (width/num_models);\n")
            f.write("        for j = 1:num_sizes\n")
            f.write("            value = performance_data(i, j);\n")
            f.write("            if value > 0\n")
            f.write("                x_pos = group_centers(j) + offset;\n")
            f.write("                text(x_pos, value, sprintf('%.4f', value), ...\n")
            f.write("                     'HorizontalAlignment', 'center', ...\n")
            f.write("                     'VerticalAlignment', 'bottom', ...\n")
            f.write("                     'FontSize', 8);\n")
            f.write("            end\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    % Add trend lines\n")
            f.write("    hold on;\n")
            f.write("    for i = 1:length(h)\n")
            f.write("        x_vals = h(i).XEndPoints;\n")
            f.write("        y_vals = h(i).YEndPoints;\n")
            f.write("        plot(x_vals, y_vals, '-o', 'LineWidth', 2, 'Color', h(i).FaceColor, 'HandleVisibility', 'off');\n")
            f.write("    end\n")
            f.write("    hold off;\n")
            f.write("    \n")
            f.write("    % Use log scale for y-axis and add grid\n")
            f.write("    set(gca, 'YScale', 'log');\n")
            f.write("    grid on;\n")
            f.write("    \n")
            f.write("    % Save figure\n")
            f.write("    saveas(gcf, ['model_comparison_' test_channel '.png']);\n")
            f.write("    saveas(gcf, ['model_comparison_' test_channel '.fig']);\n")
            f.write("    fprintf('Figure saved as model_comparison_%s.png and .fig\\n', test_channel);\n")
            f.write("end\n\n")
            
            # Model finding helper function - Enhanced to handle complex model names
            f.write("function model_key = find_model(models, base_name, sample_str, test_channel)\n")
            f.write("    % Find the model matching a specific base type and sample size\n")
            f.write("    model_key = '';\n")
            f.write("    model_names = fieldnames(models);\n")
            f.write("    \n")
            f.write("    if strcmp(base_name, 'Meta')\n")
            f.write("        % Look for Meta_[size] pattern\n")
            f.write("        pattern = ['Meta_' sample_str '$'];\n")
            f.write("        for i = 1:length(model_names)\n")
            f.write("            if regexp(model_names{i}, pattern)\n")
            f.write("                model_key = model_names{i};\n")
            f.write("                return;\n")
            f.write("            end\n")
            f.write("        end\n")
            f.write("    else\n")
            f.write("        % Handle DNN models, which might have complex naming patterns\n")
            f.write("        if startsWith(base_name, 'DNN_')\n")
            f.write("            % Extract the training channel part from base_name (after 'DNN_')\n")
            f.write("            train_channel = extractAfter(base_name, 'DNN_');\n")
            f.write("            \n")
            f.write("            % Look for the specific pattern including training channel, test channel, and size\n")
            f.write("            % We need to handle potential underscores within the channel names\n")
            f.write("            expected_pattern = ['DNN_' train_channel '_' test_channel '_' sample_str];\n")
            f.write("            \n")
            f.write("            for i = 1:length(model_names)\n")
            f.write("                % Strict matching - must match exactly, including all underscores\n")
            f.write("                if strcmp(model_names{i}, expected_pattern)\n")
            f.write("                    model_key = model_names{i};\n")
            f.write("                    return;\n")
            f.write("                end\n")
            f.write("            end\n")
            f.write("            \n")
            f.write("            % Special handling for random_mixed which might have variant names\n")
            f.write("            if strcmp(train_channel, 'random_mixed')\n")
            f.write("                % Try alternative patterns in case name was modified\n")
            f.write("                alt_patterns = {...\n")
            f.write("                    ['DNN_random_mixed_' test_channel '_' sample_str], ...\n")
            f.write("                    ['DNN_random_' test_channel '_' sample_str], ...\n")
            f.write("                    ['DNN_mixed_' test_channel '_' sample_str], ...\n")
            f.write("                    ['DNN_randomMixed_' test_channel '_' sample_str] ...\n")
            f.write("                };\n")
            f.write("                \n")
            f.write("                for j = 1:length(alt_patterns)\n")
            f.write("                    for i = 1:length(model_names)\n")
            f.write("                        if strcmp(model_names{i}, alt_patterns{j})\n")
            f.write("                            model_key = model_names{i};\n")
            f.write("                            return;\n")
            f.write("                        end\n")
            f.write("                    end\n")
            f.write("                end\n")
            f.write("            end\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("end\n\n")
            
            # Error rate extraction helper function
            f.write("function error_rate = extract_error_rate(model)\n")
            f.write("    % Extract the final validation error rate from model data\n")
            f.write("    error_rate = 0;\n")
            f.write("    \n")
            f.write("    try\n")
            f.write("        % First approach: from metrics_by_updates.val_bit_err\n")
            f.write("        if isfield(model, 'metrics_by_updates') && isfield(model.metrics_by_updates, 'val_bit_err')\n")
            f.write("            val_errors = model.metrics_by_updates.val_bit_err;\n")
            f.write("            if ~isempty(val_errors)\n")
            f.write("                error_rate = val_errors(end);\n")
            f.write("                return;\n")
            f.write("            end\n")
            f.write("        end\n")
            f.write("        \n")
            f.write("        % Second approach: from val_epoch_bit_err\n")
            f.write("        if isfield(model, 'val_epoch_bit_err') && ~isempty(model.val_epoch_bit_err)\n")
            f.write("            error_rate = model.val_epoch_bit_err(end);\n")
            f.write("            return;\n")
            f.write("        end\n")
            f.write("    catch e\n")
            f.write("        warning('Error extracting validation error: %s', e.message);\n")
            f.write("    end\n")
            f.write("end\n\n")
            
            # Create pretty labels helper function
            f.write("function label = make_pretty_label(base_name)\n")
            f.write("    % Create nice-looking labels for the legend\n")
            f.write("    if contains(base_name, 'DNN_')\n")
            f.write("        train_channel = extractAfter(base_name, 'DNN_');\n")
            f.write("        \n")
            f.write("        % Handle special cases for channel types\n")
            f.write("        if strcmp(train_channel, 'random_mixed')\n")
            f.write("            label = 'DNN (Mixed Channels)';\n")
            f.write("        elseif strcmp(train_channel, 'awgn')\n")
            f.write("            label = 'DNN (AWGN)';\n")
            f.write("        elseif strcmp(train_channel, 'rayleigh')\n")
            f.write("            label = 'DNN (Rayleigh)';\n")
            f.write("        elseif strcmp(train_channel, 'rician')\n")
            f.write("            label = 'DNN (Rician)';\n")
            f.write("        else\n")
            f.write("            label = ['DNN (' strrep(train_channel, '_', ' ') ')'];\n")
            f.write("        end\n")
            f.write("    elseif strcmp(base_name, 'Meta')\n")
            f.write("        label = 'Meta-Learning';\n")
            f.write("    else\n")
            f.write("        label = strrep(base_name, '_', ' ');\n")
            f.write("    end\n")
            f.write("end\n")
            
            print(f"MATLAB visualization script generated: {script_path}")
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
        self.K=K; # Number of subcarriers
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

    def generate_training_dataset(self, channel_type, bits_array, mode="sequential_mixed", custom_channels=None):
        """
        Generate a training dataset for a specified channel type or mix of channels.
        
        Args:
            channel_type: Channel type or list of channel types
            bits_array: Binary data for transmission
            mode: Dataset mixing mode when multiple channels are used (default: "sequential_mixed")
            custom_channels: Optional list of channels to use for mixing (overrides default)
            
        Returns:
            Tuple containing the training samples and their corresponding bits
        """
        if isinstance(channel_type, list) or channel_type in ["random_mixed", "sequential_mixed"]:
            if channel_type in ["random_mixed", "sequential_mixed"]:
                # Use custom_channels if provided, otherwise fall back to globals
                if custom_channels is not None:
                    channel_types = custom_channels
                    print(f"Generating mixed dataset using custom channels: {channel_types}")
                elif 'available_channel_types' in globals():
                    channel_types = globals()['available_channel_types']
                    print(f"Generating mixed dataset using global channels: {channel_types}")
                else:
                    channel_types = ["rician", "awgn", "rayleigh"]  # Last resort fallback
                    print(f"Generating mixed dataset using fallback channels: {channel_types}")
                
                return self.generate_mixed_dataset(channel_types, bits_array, mode=channel_type)
            return self.generate_mixed_dataset(channel_type, bits_array, mode=mode)
        
        training_sample = []
        for bits in bits_array:
            ofdm_simulate_output = self.ofdm_simulate(bits, channel_type)
            training_sample.append(ofdm_simulate_output)
        
        return np.asarray(training_sample), bits_array
    
    def generate_testing_dataset(self, channel_type, num_samples, mode="sequential_mixed",custom_channels=None):
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
                # Use custom_channels if provided, otherwise fall back to globals
                if custom_channels is not None:
                    channel_types = custom_channels
                    print(f"Generating mixed dataset using custom channels: {channel_types}")
                elif 'available_channel_types' in globals():
                    channel_types = globals()['available_channel_types']
                    print(f"Generating mixed dataset using global channels: {channel_types}")
                else:
                    channel_types = ["rician", "awgn", "rayleigh"]  # Last resort fallback
                    print(f"Generating mixed dataset using fallback channels: {channel_types}")
                
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
    # Optimize memory copies
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=min(buffer_size, len(x_data)))

    # Optimize batch processing
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.threading.max_intra_op_parallelism = 8
    dataset = dataset.with_options(options)
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
    pred_dtype = y_pred.dtype
    y_sign = tf.sign(y_pred - 0.5)
    y_true_sign = tf.cast(tf.sign(y_true - 0.5), pred_dtype)

    err = 1 - tf.reduce_mean(
        tf.reduce_mean(
            tf.cast(
                tf.equal(y_sign, y_true_sign), 
                pred_dtype
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

def conventional_channel_estimation(signal_simulator, test_channel, test_sizes):
    """
    Test conventional channel estimation methods (Pure LS and MMSE)
    
    Args:
        signal_simulator: Instance of signal_simulator class
        test_channel: Channel type to test on (e.g., "WINNER")
        test_sizes: List of test sample sizes
        
    Returns:
        Dictionary containing performance results for Pure LS and MMSE methods
    """
    import numpy as np
    import time
    import os
    from datetime import datetime
    
    print("\n=== Conventional Channel Estimation Baseline Test ===")
    
    # Initialize results dictionary
    results = {
        "Pure_LS": {},
        "MMSE": {}
    }
    
    # Create timestamp for the file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure experiment_logs directory exists
    if not os.path.exists("experiment_logs"):
        os.makedirs("experiment_logs")
    
    # Create file for saving results
    result_file = os.path.join("experiment_logs", f"conventional_channel_est_{test_channel}_{timestamp}.txt")
    
    # Initialize file with header information
    with open(result_file, "w") as f:
        f.write(f"Conventional Channel Estimation Results\n")
        f.write(f"Test Channel: {test_channel}\n")
        f.write(f"SNR: {signal_simulator.SNRdB} dB\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*50 + "\n\n")
        f.write("Sample Size\tPure LS BER\tMMSE BER\n")
        f.write("-"*40 + "\n")
    
    # Test each sample size
    for size in test_sizes:
        print(f"\nSample size: {size}")
        
        # Generate test data
        bits = signal_simulator.generate_bits(size)
        
        # Pure LS method test
        start_time = time.time()
        ber_pure_ls = test_pure_ls(signal_simulator, bits, test_channel)
        ls_time = time.time() - start_time
        print(f"Pure LS: BER = {ber_pure_ls:.6f}, Processing time: {ls_time:.2f}s")
        results["Pure_LS"][size] = ber_pure_ls
        
        # MMSE method test
        start_time = time.time()
        ber_mmse = test_mmse(signal_simulator, bits, test_channel)
        mmse_time = time.time() - start_time
        print(f"MMSE: BER = {ber_mmse:.6f}, Processing time: {mmse_time:.2f}s")
        results["MMSE"][size] = ber_mmse
        
        # Save results to file
        with open(result_file, "a") as f:
            f.write(f"{size}\t\t{ber_pure_ls:.6f}\t{ber_mmse:.6f}\n")
    
    # Add summary to file
    with open(result_file, "a") as f:
        
        f.write("\nPerformance Summary:\n")
        best_ls = min(results["Pure_LS"].values())
        best_mmse = min(results["MMSE"].values())
        best_overall = min(best_ls, best_mmse)
        best_method = "Pure LS" if best_ls == best_overall else "MMSE"
        
        f.write(f"- Best Pure LS BER: {best_ls:.6f}\n")
        f.write(f"- Best MMSE BER: {best_mmse:.6f}\n")
        f.write(f"- Best Overall Method: {best_method} (BER: {best_overall:.6f})\n")
    
    print(f"\nResults saved to: {result_file}")
    
    # Print results summary to terminal
    print("\n=== Conventional Channel Estimation Results Summary ===")
    print("Sample Size\tPure LS BER\tMMSE BER")
    print("-" * 40)
    for size in sorted(test_sizes):
        print(f"{size}\t\t{results['Pure_LS'][size]:.6f}\t{results['MMSE'][size]:.6f}")
    
    return results


def test_pure_ls(simulator, bits, channel_type):
    """
    Test BER using pure LS estimation method (without interpolation)
    
    Args:
        simulator: signal_simulator instance
        bits: Bits to transmit
        channel_type: Channel type
        
    Returns:
        Bit Error Rate (BER)
    """
    import numpy as np
    
    total_bits = len(bits) * simulator.payloadBits_per_OFDM
    errors = 0
    
    for i, bit_sequence in enumerate(bits):
        # Use existing code to generate signals
        ofdm_tx = simulator.transmit_signals(bit_sequence)
        ofdm_rx = simulator.received_signals(ofdm_tx, channel_type)
        
        # Remove cyclic prefix
        ofdm_rx_no_cp = simulator.remove_cp(ofdm_rx)
        
        # FFT demodulation
        ofdm_rx_freq = simulator.dft(ofdm_rx_no_cp)
        
        # 1. LS channel estimation at pilot positions
        rx_pilots = ofdm_rx_freq[simulator.pilot_carriers]
        pilot_value = globals()['pilot_value']
        h_est_pilot = rx_pilots / pilot_value
        
        # 2. Use nearest pilot for data carriers (pure LS without interpolation)
        h_est = np.zeros(simulator.K, dtype=complex)
        pilot_carriers = simulator.pilot_carriers
        
        for k in range(simulator.K):
            # Find nearest pilot position
            distances = np.abs(pilot_carriers - k)
            nearest_idx = np.argmin(distances)
            h_est[k] = h_est_pilot[nearest_idx]
        
        # 3. Equalization
        equalized_freq = ofdm_rx_freq / h_est
        
        # 4. Extract data carriers
        equalized_data = equalized_freq[simulator.data_carriers]
        
        # 5. Demapping QAM symbols to bits
        mapping_table = globals()['mapping_table']
        constellation_points = np.array(list(mapping_table.values()))
        demapping_table = {v: k for k, v in mapping_table.items()}
        
        detected_bits = []
        for symbol in equalized_data:
            # Find closest constellation point
            distances = np.abs(symbol - constellation_points)
            min_idx = np.argmin(distances)
            closest_point = constellation_points[min_idx]
            
            # Get corresponding bits
            bit_tuple = demapping_table[closest_point]
            detected_bits.extend(bit_tuple)
        
        detected_bits = np.array(detected_bits)
        
        # Calculate errors
        if len(detected_bits) == len(bit_sequence):
            current_errors = np.sum(detected_bits != bit_sequence)
            errors += current_errors
        
        # Print progress
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(bits)} symbols", end="\r")
    
    # Calculate BER
    ber = errors / total_bits
    return ber


def test_mmse(simulator, bits, channel_type):
    """
    Test BER using MMSE estimation method
    
    Args:
        simulator: signal_simulator instance
        bits: Bits to transmit
        channel_type: Channel type
        
    Returns:
        Bit Error Rate (BER)
    """
    import numpy as np
    
    total_bits = len(bits) * simulator.payloadBits_per_OFDM
    errors = 0
    
    # Convert SNR to linear value
    snr_linear = 10**(simulator.SNRdB/10)
    
    for i, bit_sequence in enumerate(bits):
        # Use existing code to generate signals
        ofdm_tx = simulator.transmit_signals(bit_sequence)
        ofdm_rx = simulator.received_signals(ofdm_tx, channel_type)
        
        # Remove cyclic prefix
        ofdm_rx_no_cp = simulator.remove_cp(ofdm_rx)
        
        # FFT demodulation
        ofdm_rx_freq = simulator.dft(ofdm_rx_no_cp)
        
        # 1. LS channel estimation at pilot positions
        rx_pilots = ofdm_rx_freq[simulator.pilot_carriers]
        pilot_value = globals()['pilot_value']
        h_est_pilot = rx_pilots / pilot_value
        
        # 2. Use nearest pilot for initial LS estimate (no interpolation)
        h_ls = np.zeros(simulator.K, dtype=complex)
        pilot_carriers = simulator.pilot_carriers
        
        for k in range(simulator.K):
            # Find nearest pilot position
            distances = np.abs(pilot_carriers - k)
            nearest_idx = np.argmin(distances)
            h_ls[k] = h_est_pilot[nearest_idx]
        
        # 3. Apply MMSE optimization
        # Calculate noise variance
        signal_power = np.mean(np.abs(h_ls)**2)
        noise_var = signal_power / snr_linear
        
        # MMSE filter
        h_mmse = h_ls * (np.abs(h_ls)**2 / (np.abs(h_ls)**2 + noise_var))
        
        # 4. Equalization
        equalized_freq = ofdm_rx_freq / h_mmse
        
        # 5. Extract data carriers
        equalized_data = equalized_freq[simulator.data_carriers]
        
        # 6. Demapping QAM symbols to bits
        mapping_table = globals()['mapping_table']
        constellation_points = np.array(list(mapping_table.values()))
        demapping_table = {v: k for k, v in mapping_table.items()}
        
        detected_bits = []
        for symbol in equalized_data:
            # Find closest constellation point
            distances = np.abs(symbol - constellation_points)
            min_idx = np.argmin(distances)
            closest_point = constellation_points[min_idx]
            
            # Get corresponding bits
            bit_tuple = demapping_table[closest_point]
            detected_bits.extend(bit_tuple)
        
        detected_bits = np.array(detected_bits)
        
        # Calculate errors
        if len(detected_bits) == len(bit_sequence):
            current_errors = np.sum(detected_bits != bit_sequence)
            errors += current_errors
        
        # Print progress
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(bits)} symbols", end="\r")
    
    # Calculate BER
    ber = errors / total_bits
    return ber

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='OFDM Signal Detection with Configurable Test Channel')
    parser.add_argument('--test_channel', type=str, default="WINNER", 
                        choices=["awgn", "rician", "rayleigh", "WINNER"],
                        help='Channel type for generalization testing (default: WINNER)')
    parser.add_argument('--train_samples', type=int, default=64000,
                        help='Number of training samples (default: 64000)')
    parser.add_argument('--snr', type=float, default=10,
                        help='Signal-to-noise ratio in dB (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping for meta-learning (disabled by default)')
    args = parser.parse_args()
    
    # Create a fully integrated experiment configuration
    config = ExperimentConfig()
    
    # Set the generalization test channel
    config.set_test_channel(args.test_channel)
    
    # Apply other command-line arguments to configuration
    config.config["dataset"]["train_samples"] = args.train_samples
    config.config["signal"]["SNR"] = args.snr
    config.config["seed"] = args.seed
    config.config["meta_dnn"]["early_stopping"] = args.early_stopping
    
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
    
    # Get available training channels and inject into globals for signal_simulator
    train_channels = config.get_available_training_channels()
    globals()['available_channel_types'] = train_channels
    
    # Set random seed for reproducibility
    seed = config.config["seed"]
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Configure GPU memory growth to prevent TensorFlow from allocating all GPU memory at once
    gpus = tf.config.list_physical_devices('GPU')
    print("Available GPU devices:", gpus)
    if gpus and config.config["gpu_memory_growth"]:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            from keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("Mixed precision policy set to:", policy.name)
    
            # Disable XLA auto-compilation
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
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
    test_channel = dataset_params["test_channel"]
    
    # Ensure output directory exists
    log_dir = output_params.get("log_dir", "experiment_logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Print experiment configuration summary
    print("\n=== Experiment Configuration ===")
    print(f"Test Channel: {test_channel}")
    print(f"Training Channels: {channel_types}")
    print(f"Meta-learning Channels: {meta_channel_types}")
    print(f"Training Samples: {DNN_samples}")
    print(f"SNR: {config.config['signal']['SNR']} dB")
    print(f"Random Seed: {seed}")
    print(f"Early Stopping: {'Enabled' if config.config['meta_dnn']['early_stopping'] else 'Disabled'}")
    
    # Create signal simulator instance
    simulator = signal_simulator()
    # All parameters already injected through global variables, just apply SNR
    simulator.SNRdB = config.config["signal"]["SNR"]
    
    #conventional_channel_estimation(simulator, test_channel, [50,100,500,1000])
    # Initialize model and history containers
    models = {}
    histories = {}
    
    # Generate training data
    print(f"\nGenerating {DNN_samples} training samples...")
    bits = simulator.generate_bits(DNN_samples)
    MultiModelBCP.clear_data()
    
   # Training Phase - Standard DNN
    print("\n=== Train Phase ===")

    # Extract only individual channels (no mixed types) for use in mixed datasets
    individual_channels = [ch for ch in channel_types 
                        if ch != "random_mixed" and ch != "sequential_mixed"]
    print(f"Individual channels available for mixing: {individual_channels}")

    for channel in channel_types:
        print(f"\nTraining on {channel} channel...")
        
        # Generate training data
        start_time = time.time()
        
        if channel == "random_mixed":
            # For random_mixed, explicitly pass individual channels
            print(f"Generating mixed dataset from individual channels: {individual_channels}")
            x_train, y_train = simulator.generate_training_dataset(
                "random_mixed",
                bits,
                mode="mixed_random",
                custom_channels=individual_channels  # Explicitly pass channels to mix
            )
            
            # Similarly for test dataset
            x_test, y_test = simulator.generate_testing_dataset(
                "random_mixed",
                dataset_params["test_samples"] // 5,
                mode="mixed_random",
                custom_channels=individual_channels  # Explicitly pass channels to mix
            )
        else:
            # Standard single-channel generation
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

    # Get explicit list of individual channels for meta-learning (no random_mixed)
    meta_channel_types = config.config["dataset"]["meta_channel_types"]
    print(f"Using channels for meta-learning: {meta_channel_types}")

    # Generate meta-learning tasks for each channel type
    print("Generating meta-learning tasks...")
    start_time = time.time()
    for channel in meta_channel_types:
        channel_bits = simulator.generate_bits(DNN_samples)
        
        # Use direct channel, don't rely on available_channel_types global
        x_task, y_task = simulator.generate_training_dataset(
            channel,  # Passing single channel directly
            channel_bits
        )
        meta_tasks.append((x_task, y_task))
        # Free memory immediately
        gc.collect()
    print(f"Meta-learning task generation time: {time.time() - start_time:.2f} seconds")

    # Create validation set - use a mix of individual channels (not random_mixed)
    # This ensures consistent validation - each channel has equal representation
    print("Creating meta-learning validation set...")
    meta_val_samples = dataset_params["test_samples"] // 5
    meta_val_samples_per_channel = meta_val_samples // len(meta_channel_types)

    meta_x_val_parts = []
    meta_y_val_parts = []

    for channel in meta_channel_types:
        x_val_part, y_val_part = simulator.generate_testing_dataset(
            channel,  # Single channel
            meta_val_samples_per_channel
        )
        meta_x_val_parts.append(x_val_part)
        meta_y_val_parts.append(y_val_part)

    # Combine all parts
    meta_x_test = np.concatenate(meta_x_val_parts, axis=0)
    meta_y_test = np.concatenate(meta_y_val_parts, axis=0)

    print(f"Validation set created with {len(meta_x_test)} samples from {meta_channel_types}")
    
    print("Creating Meta-DNN model...")
    models[meta_model_name] = create_meta_dnn_from_config(
        input_dim=meta_tasks[0][0].shape[1],
        payloadBits_per_OFDM=simulator.payloadBits_per_OFDM,
        config=config
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
    
    # Generalization Test Phase
    print(f"\n=== {test_channel} Generalization Test Phase ===")
    MultiModelBCP.clear_data()
    
    # Get fine-tuning parameters
    fine_tuning_params = config.get_fine_tuning_params()
    val_parameter_set = []
    for size in dataset_params["fine_tuning_sizes"]:
        batch_size = fine_tuning_params["batch_sizes"].get(str(size), 32)
        val_epoch = fine_tuning_params["epochs"].get(str(size), 1)
        val_parameter_set.append((size, val_epoch, batch_size))
    
    # Create validation set for test channel
    x_test_val, y_test_val = simulator.generate_testing_dataset(
        test_channel, 
        dataset_params["test_samples"] // 8
    )
    
    for size, val_epoch, val_batch_size in val_parameter_set:
        DNN_num_update = int((size/val_batch_size)*val_epoch)
        print(f"\n{size} sample set, updates: {DNN_num_update}")
        
        # Generate test channel training data
        bits_array = simulator.generate_bits(size)
        x_test_train, y_test_train = simulator.generate_training_dataset(
            test_channel, 
            bits_array
        )
        
        # Convert to TensorFlow dataset
        train_dataset = create_tf_dataset(
            x_test_train, y_test_train, 
            batch_size=val_batch_size,
            repeat=False
        )
        
        # Traditional DNN fine-tuning
        for channel in channel_types:
            model_name = f"DNN_{channel}_{test_channel}_{size}"
            dnn_model = models[f"DNN_{channel}"].clone()
            print(f"Fine-tuning {channel} model on {test_channel} channel, sample size: {size}")
            
            callback = MultiModelBCP(
                model_name=model_name, 
                dataset_type=f"{channel}_{size}",
                sampling_rate=output_params.get("sampling_rate", 10),
                max_points=output_params.get("max_points", 1000)
            )
            
            dnn_model.fit(
                train_dataset,
                epochs=val_epoch,
                validation_data=(x_test_val, y_test_val),
                callbacks=[callback],
                verbose=1
            )
            
            # Free model memory
            del dnn_model
            gc.collect()
        
        # Meta DNN fine-tuning
        meta_task = [(x_test_train, y_test_train)]
        meta_model = models["Meta_DNN"].clone()
        meta_model_name = f"Meta_{size}"
        
        # Adjust inner steps based on dataset size
        task_steps = min(config.config["meta_dnn"]["task_steps"], size//5)
        
        losses, val_errs, update_counts = meta_model.train_reptile(
            meta_task, 
            meta_epochs=DNN_num_update, 
            meta_validation_data=(x_test_val, y_test_val),
            task_steps=task_steps
        )
        
        MultiModelBCP.log_manual_data(
            meta_model_name,
            losses,
            val_errs,
            update_counts=update_counts,
            dataset_type=f"{test_channel}_{size}"
        )
        
        # Free meta-model and data
        del meta_model, losses, val_errs, update_counts, meta_task
        del x_test_train, y_test_train, train_dataset
        gc.collect()
    
    # Plot final results
    final_plot_file = os.path.join(log_dir, f"{test_channel}_generalization_test_{experiment_time}.png")
    MultiModelBCP.plot_by_updates(save_path=final_plot_file, dpi=dpi)

    # Prepare export directory
    output_dir = f"matlab_exports/experiment_{experiment_time}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Export model data
    print("\n=== Exporting Model Data to MATLAB ===")
    exported_files = MultiModelBCP.export_data_for_matlab(
        output_dir=output_dir,
        prefix=f"ofdm_models_{test_channel}_{experiment_time}_"
    )

    # Generate MATLAB analysis script
    matlab_script = MultiModelBCP.generate_matlab_script(
        output_dir=output_dir,
        exported_files=exported_files,
        prefix=f"ofdm_models_{test_channel}_{experiment_time}_",
        sample_sizes=dataset_params["fine_tuning_sizes"],
        testing_channel=test_channel,
        training_channels=channel_types,
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

if __name__ == "__main__":
    main()