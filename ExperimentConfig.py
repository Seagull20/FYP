import os
import datetime
import pprint
import json
import numpy as np
import importlib
from singal_detection_withOFDM import MetaDNN

class ExperimentConfig:
    """Configuration management class for OFDM signal detection experiments - Fully integrated version"""
    
    all_models_data = {}  # Compatible with MultiModelBCP
    
    def __init__(self):
        """
        Initialize configuration object with hardcoded global parameters from global_parameters.py
        """
        # Initialize fully integrated configuration with direct global parameter values
        self.config = {
            # Global parameter settings (directly from global_parameters.py)
            "global": {
                "K": 64,                      # Number of subcarriers
                "CP": 16,                     # Cyclic prefix length
                "P": 8,                       # Number of pilot carriers
                "pilot_value": self._complex_to_dict(1+1j),  # Pilot value
                "mu": 2,                      # Bits per symbol
                "mapping_table": self._convert_mapping_table({
                    (0, 0): -1 - 1j,
                    (0, 1): -1 + 1j,
                    (1, 0): 1 - 1j,
                    (1, 1): 1 + 1j,
                }),  # Mapping table
                "demapping_table": None,      # Demapping table (auto-calculated)
                "num_path": 16,               # Number of channel paths
                "rician_factor": 1,           # Rician factor
                "num_simulate": 50000,        # Simulation count
                "num_simulate_target": 5000,  # Target simulation count
                "num_test_target": 1000,      # Target test count
                "num_running": 1              # Number of runs
            },
            
            # Signal and channel parameters
            "signal": {
                "SNR": 10,                    # Signal-to-noise ratio (dB)
            },
            
            # Dataset configuration
            "dataset": {
                "train_samples": 64000,                                   # Number of training samples
                "test_samples": 2500,                                     # Test dataset size
                "val_samples": 300,                                       # 3GPP validation dataset size
                "channel_types": ["awgn", "rician", "rayleigh", "random_mixed"], # Standard training channels
                "meta_channel_types": ["awgn", "rician", "rayleigh"],      # Meta-learning channels
                "test_channel": "WINNER",                                   # Generalization test channel
                "fine_tuning_sizes": [50, 100, 500, 1000, 64000]          # Fine-tuning sample sizes
            },
            
            # DNN model parameters
            "dnn": {
                "architecture": [256, 512, 256],                          # Hidden layer sizes
                "activations": ["relu", "relu", "relu", "sigmoid"],       # Activation functions
                "optimizer": "adam",                                      # Optimizer
                "learning_rate": 0.001,                                   # Learning rate from global_parameters.py
                "loss": "mse",                                            # Loss function
                "epochs": 10,                                             # Number of training epochs from global_parameters.py
                "batch_size": 32                                          # Batch size from global_parameters.py
            },
            
            # Meta-learning parameters
            "meta_dnn": {
                "inner_lr": 0.02,                                         # Inner loop learning rate
                "meta_lr": 0.3,                                           # Meta learning rate
                "mini_batch_size": 32,                                    # Mini-batch size
                "task_steps": 50,                                         # Number of task steps
                "early_stopping": True,                                   # Whether to enable early stopping
                "patience": 20,                                           # Early stopping patience
                "min_delta": 0.0002,                                      # Minimum improvement threshold
                "abs_threshold": 0.011,                                   # Absolute threshold
                "progressive_patience": True,                             # Dynamic patience
                "verbose": 1,                                             # Verbosity level
                "lr_schedule": {
                    "first_decay_steps": 500,                             # First decay steps
                    "t_mul": 1.1,                                         # t multiplier
                    "m_mul": 1,                                           # m multiplier
                    "alpha": 0.001                                        # alpha value
                }
            },
            
            # Fine-tuning parameters
            "fine_tuning": {
                "epochs": {
                    "50": 1,
                    "100": 1, 
                    "500": 1,
                    "1000": 1,
                    "64000": 10
                },                                                        # Fine-tuning epochs
                "batch_sizes": {                                          # Batch sizes for different sample amounts
                    "50": 5,
                    "100": 5, 
                    "500": 16,
                    "1000": 32,
                    "64000": 32
                }
            },
            
            # Random seed for reproducibility
            "seed": 42,
            
            # Computation parameters
            "gpu_memory_growth": True,                                    # Whether to enable GPU memory growth
            
            # Output configuration
            "output": {
                "save_plots": True,                                       # Whether to save plots
                "plot_dpi": 300,                                          # Plot DPI
                "log_dir": "experiment_logs",                             # Log directory
                "sampling_rate": 10,                                      # Metrics sampling rate
                "max_points": 1000                                        # Maximum record points
            }
        }
        
        # Calculate demapping table
        self._update_demapping_table()
    
    def _complex_to_dict(self, complex_val):
        """
        Convert complex number to dictionary representation
        
        Args:
            complex_val: Complex value to convert
        
        Returns:
            Dictionary representation of the complex value
        """
        if isinstance(complex_val, complex):
            return {"real": complex_val.real, "imag": complex_val.imag}
        return complex_val
    
    def _dict_to_complex(self, dict_val):
        """
        Convert dictionary representation to complex number
        
        Args:
            dict_val: Dictionary to convert
        
        Returns:
            Complex number from dictionary
        """
        if isinstance(dict_val, dict) and "real" in dict_val and "imag" in dict_val:
            return complex(dict_val["real"], dict_val["imag"])
        return dict_val
    
    def _convert_mapping_table(self, mapping_table):
        """
        Convert mapping table to serializable format
        
        Args:
            mapping_table: Original mapping table
        
        Returns:
            Mapping table in serializable format
        """
        result = {}
        for k, v in mapping_table.items():
            # Convert tuple keys to strings
            str_key = str(k)
            # Convert complex values to dictionaries
            result[str_key] = self._complex_to_dict(v)
        return result
    
    def _restore_mapping_table(self, mapping_dict):
        """
        Restore mapping table to original format
        
        Args:
            mapping_dict: Serialized mapping dictionary
        
        Returns:
            Original format mapping table
        """
        result = {}
        for k, v in mapping_dict.items():
            # Convert string keys back to tuples
            if k.startswith('(') and k.endswith(')'):
                # Parse strings in format "(0, 1)"
                try:
                    tuple_key = eval(k)
                except:
                    # If parsing fails, use original key
                    tuple_key = k
            else:
                tuple_key = k
            
            # Convert dictionary values back to complex numbers
            result[tuple_key] = self._dict_to_complex(v)
        return result
    
    def _update_demapping_table(self):
        """Update the demapping table based on the mapping table"""
        mapping_table = self._restore_mapping_table(self.config["global"]["mapping_table"])
        demapping_table = {v: k for k, v in mapping_table.items()}
        self.config["global"]["demapping_table"] = self._convert_mapping_table(demapping_table)
    
    def update_from_dict(self, config_dict):
        """
        Update configuration from dictionary
        
        Args:
            config_dict: Dictionary with new configuration values
        """
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_nested_dict(d[k], v)
                else:
                    d[k] = v
        
        update_nested_dict(self.config, config_dict)
        # Update demapping table
        self._update_demapping_table()
    
    def get_simulator_params(self):
        """
        Get signal simulator parameters
        
        Returns:
            Dictionary of simulator parameters
        """
        return {
            "SNR": self.config["signal"]["SNR"],
            "K": self.config["global"]["K"],
            "P": self.config["global"]["P"],
            "CP": self.config["global"]["CP"],
            "mu": self.config["global"]["mu"],
            "num_path": self.config["global"]["num_path"],
            "rician_factor": self.config["global"]["rician_factor"]
        }
    
    def get_dnn_params(self):
        """
        Get DNN parameters
        
        Returns:
            Dictionary of DNN parameters
        """
        return self.config["dnn"]
    
    def get_meta_dnn_params(self):
        """
        Get MetaDNN parameters
        
        Returns:
            Dictionary of MetaDNN parameters
        """
        return self.config["meta_dnn"]
    
    def get_dataset_params(self):
        """
        Get dataset parameters
        
        Returns:
            Dictionary of dataset parameters
        """
        return self.config["dataset"]
    
    def get_fine_tuning_params(self):
        """
        Get fine-tuning parameters
        
        Returns:
            Dictionary of fine-tuning parameters
        """
        return self.config["fine_tuning"]
    
    def get_output_params(self):
        """
        Get output parameters
        
        Returns:
            Dictionary of output parameters
        """
        return self.config["output"]
    
    def create_global_module(self):
        """
        Create global parameters module compatible with original global_parameters.py
        
        Returns:
            Module object containing global parameters
        """
        # Create an empty module
        module = type('GlobalParameters', (), {})
        
        # Add basic parameters
        for key, value in self.config["global"].items():
            if key == "mapping_table":
                # Special handling for mapping table
                mapping_table = self._restore_mapping_table(value)
                setattr(module, key, mapping_table)
            elif key == "demapping_table":
                # Special handling for demapping table
                if value:
                    demapping_table = self._restore_mapping_table(value)
                    setattr(module, key, demapping_table)
            elif key == "pilot_value":
                # Special handling for pilot value
                setattr(module, key, self._dict_to_complex(value))
            else:
                setattr(module, key, value)
        
        # Add signal parameters
        for key, value in self.config["signal"].items():
            setattr(module, "SNRdb", value)  # Use original name
        
        # Add other necessary parameters
        setattr(module, "learning_rate", self.config["dnn"]["learning_rate"])
        setattr(module, "batch_size", self.config["dnn"]["batch_size"])
        setattr(module, "num_epochs", self.config["dnn"]["epochs"])
        
        return module
    
    def export_to_global_parameters(self, output_file="new_global_parameters.py"):
        """
        Export configuration to a format compatible with global_parameters.py
        
        Args:
            output_file: Output file path
        
        Returns:
            Path to the exported file
        """
        try:
            with open(output_file, 'w') as f:
                # Write global parameters
                for key, value in self.config["global"].items():
                    if key == "mapping_table":
                        # Special handling for mapping table
                        mapping_table = self._restore_mapping_table(value)
                        f.write(f"{key} = ")
                        f.write(pprint.pformat(mapping_table, indent=4))
                        f.write("\n\n")
                    elif key == "demapping_table":
                        # Skip demapping table, calculate later
                        continue
                    elif key == "pilot_value":
                        # Special handling for pilot value
                        complex_val = self._dict_to_complex(value)
                        f.write(f"{key} = {complex_val!r}\n")
                    else:
                        f.write(f"{key} = {value!r}\n")
                
                # Write demapping table
                f.write("\n# Automatically calculate demapping table from mapping table\n")
                f.write("demapping_table = {v: k for k, v in mapping_table.items()}\n")
                
                # Write other necessary parameters
                f.write(f"\nSNRdb = {self.config['signal']['SNR']}\n")
                f.write(f"learning_rate = {self.config['dnn']['learning_rate']}\n")
                f.write(f"batch_size = {self.config['dnn']['batch_size']}\n")
                f.write(f"num_epochs = {self.config['dnn']['epochs']}\n")
            
            print(f"Global parameters exported to: {output_file}")
            return output_file
        
        except Exception as e:
            print(f"Error exporting global parameters: {str(e)}")
            return None
    
    def inject_globals(self, target_module=None):
        """
        Inject global parameters from configuration into a module or global namespace
        
        Args:
            target_module: Target module, if None uses globals()
        """
        if target_module is None:
            target_module = globals()
        
        module = self.create_global_module()
        
        # Inject all attributes into target namespace
        for name in dir(module):
            if not name.startswith('__'):
                value = getattr(module, name)
                if isinstance(target_module, dict):
                    target_module[name] = value
                else:
                    setattr(target_module, name, value)
    
    def save_to_file(self, filepath=None, add_timestamp=True):
        """
        Save configuration to file
        
        Args:
            filepath: File path, if None uses default path
            add_timestamp: Whether to add timestamp to filename
        
        Returns:
            Path to the saved file
        """
        try:
            # Determine file save path
            if filepath is None:
                # Default to creating experiment_configs folder in current directory
                config_dir = "experiment_configs"
                if not os.path.exists(config_dir):
                    os.makedirs(config_dir)
                
                # Add timestamp
                timestamp = ""
                if add_timestamp:
                    timestamp = f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                filepath = os.path.join(config_dir, f"ofdm_config{timestamp}.json")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            print(f"Configuration saved to: {filepath}")
            return filepath
        
        except Exception as e:
            print(f"Error saving configuration file: {str(e)}")
            return None
    
    def save_as_python(self, filepath=None, add_timestamp=True):
        """
        Save configuration as Python file
        
        Args:
            filepath: File path, if None uses default path
            add_timestamp: Whether to add timestamp to filename
        
        Returns:
            Path to the saved file
        """
        try:
            # Determine file save path
            if filepath is None:
                # Default to creating experiment_configs folder in current directory
                config_dir = "experiment_configs"
                if not os.path.exists(config_dir):
                    os.makedirs(config_dir)
                
                # Add timestamp
                timestamp = ""
                if add_timestamp:
                    timestamp = f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                filepath = os.path.join(config_dir, f"ofdm_config{timestamp}.py")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Write Python format
            with open(filepath, 'w') as f:
                f.write("# OFDM Signal Detection Experiment Configuration\n")
                f.write("# Generated: {}\n\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                f.write("experiment_config = ")
                # Use pprint to format configuration dictionary
                formatted_config = pprint.pformat(self.config, indent=4, width=100)
                f.write(formatted_config)
            
            print(f"Configuration saved to: {filepath}")
            return filepath
        
        except Exception as e:
            print(f"Error saving configuration file: {str(e)}")
            return None
    
    @classmethod
    def load_from_file(cls, filepath):
        """
        Load configuration from file
        
        Args:
            filepath: Configuration file path
        
        Returns:
            Configuration object
        """
        try:
            config_obj = cls()  # Create default configuration object
            
            # Determine file type and load
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    loaded_config = json.load(f)
                
                config_obj.update_from_dict(loaded_config)
            
            elif filepath.endswith('.py'):
                # Load from Python file
                namespace = {}
                with open(filepath, 'r') as f:
                    exec(f.read(), {}, namespace)
                
                if 'experiment_config' in namespace:
                    config_obj.update_from_dict(namespace['experiment_config'])
                else:
                    raise ValueError("Could not find experiment_config variable in Python configuration file")
            
            else:
                raise ValueError(f"Unsupported file type: {filepath}")
            
            print(f"Configuration loaded from {filepath}")
            return config_obj
        
        except Exception as e:
            print(f"Error loading configuration file: {str(e)}")
            return None
    
    def __str__(self):
        """String representation of configuration"""
        return pprint.pformat(self.config, indent=4)

# Application examples
def apply_config_to_simulator(simulator, config):
    """
    Apply configuration to signal simulator
    
    Args:
        simulator: Signal simulator instance
        config: Configuration object
        
    Returns:
        Configured simulator instance
    """
    signal_params = config.get_simulator_params()
    simulator.SNRdB = signal_params.get("SNR", 10)
    # Other parameters applied through global injection
    return simulator

def create_meta_dnn_from_config(input_dim, payloadBits_per_OFDM, config):
    """
    Create MetaDNN model from configuration
    
    Args:
        input_dim: Input dimension
        payloadBits_per_OFDM: Payload bits per OFDM symbol
        config: Configuration object
        
    Returns:
        Configured MetaDNN model
    """
    meta_params = config.get_meta_dnn_params()
    lr_schedule = meta_params.get("lr_schedule", {})
    
    return MetaDNN(
        input_dim=input_dim,
        payloadBits_per_OFDM=payloadBits_per_OFDM,
        inner_lr=meta_params.get("inner_lr", 0.02),
        meta_lr=meta_params.get("meta_lr", 0.3),
        mini_size=meta_params.get("mini_batch_size", 32),
        first_decay_steps=lr_schedule.get("first_decay_steps", 500),
        t_mul=lr_schedule.get("t_mul", 1.1),
        m_mul=lr_schedule.get("m_mul", 1),
        alpha=lr_schedule.get("alpha", 0.001),
        early_stopping=meta_params.get("early_stopping", True),
        patience=meta_params.get("patience", 20),
        min_delta=meta_params.get("min_delta", 0.0002),
        abs_threshold=meta_params.get("abs_threshold", 0.011),
        progressive_patience=meta_params.get("progressive_patience", True),
        verbose=meta_params.get("verbose", 1)
    )

# Usage example
if __name__ == "__main__":
    # Create default configuration
    config = ExperimentConfig()
    
    # Modify configuration
    config.config["global"]["K"] = 128  # Change number of subcarriers
    config.config["meta_dnn"]["abs_threshold"] = 0.01
    
    # Save configuration
    config.save_to_file()
    
    # Export as global parameters file
    config.export_to_global_parameters()
    
    # Simulate global parameter injection
    test_dict = {}
    config.inject_globals(test_dict)
    print(f"K value after injection: {test_dict['K']}")
    
    # Output configuration
    print("\nConfiguration content:")
    print(config)