# matlab_export.py
import numpy as np
import os
import csv
import json
try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found. Cannot export .mat format. Please install scipy: pip install scipy")


class MatlabExport:
    """Utility class for exporting MultiModelBCP data to MATLAB compatible formats"""
    
    @staticmethod
    def export_to_csv(model_data, output_dir="matlab_exports", prefix=""):
        """
        Export model data to CSV files.
        
        Creates separate CSV files for batch-level data, update-level metrics, 
        validation metrics, and epoch-level data for each model.
        
        Args:
            model_data: Dictionary containing model training metrics
            output_dir: Directory where CSV files will be saved
            prefix: String prefix for output filenames
            
        Returns:
            List of paths to the exported CSV files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        exported_files = []
        
        # Export data for each model
        for model_name, data in model_data.items():
            safe_model_name = model_name.replace("/", "_").replace("\\", "_")
            
            # Export batch-level data
            if data["batch_loss"] and data["batch_bit_err"]:
                batch_file = os.path.join(output_dir, f"{prefix}{safe_model_name}_batch.csv")
                with open(batch_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["batch_index", "loss", "bit_err"])
                    for i, (loss, bit_err) in enumerate(zip(data["batch_loss"], data["batch_bit_err"])):
                        writer.writerow([i, loss, bit_err])
                exported_files.append(batch_file)
            
            # Export update-level data
            if data["update_counts"] and data["metrics_by_updates"]["loss"]:
                updates_file = os.path.join(output_dir, f"{prefix}{safe_model_name}_updates.csv")
                with open(updates_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["update_count", "loss", "bit_err"])
                    for i, (update, loss, bit_err) in enumerate(zip(
                            data["update_counts"], 
                            data["metrics_by_updates"]["loss"], 
                            data["metrics_by_updates"]["bit_err"])):
                        writer.writerow([update, loss, bit_err])
                exported_files.append(updates_file)
            
            # Export validation data
            if data["metrics_by_updates"]["val_update_counts"] and data["metrics_by_updates"]["val_bit_err"]:
                val_file = os.path.join(output_dir, f"{prefix}{safe_model_name}_validation.csv")
                with open(val_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["update_count", "val_loss", "val_bit_err"])
                    for i, (update, loss, bit_err) in enumerate(zip(
                            data["metrics_by_updates"]["val_update_counts"], 
                            data["metrics_by_updates"]["val_loss"], 
                            data["metrics_by_updates"]["val_bit_err"])):
                        writer.writerow([update, loss, bit_err])
                exported_files.append(val_file)
            
            # Export epoch-level data
            if data["epoch_loss"] and data["epoch_bit_err"]:
                epoch_file = os.path.join(output_dir, f"{prefix}{safe_model_name}_epoch.csv")
                with open(epoch_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch", "loss", "bit_err", "val_loss", "val_bit_err"])
                    for i, (loss, bit_err, val_loss, val_bit_err) in enumerate(zip(
                            data["epoch_loss"], 
                            data["epoch_bit_err"],
                            data["val_epoch_loss"],
                            data["val_epoch_bit_err"])):
                        writer.writerow([i, loss, bit_err, val_loss, val_bit_err])
                exported_files.append(epoch_file)
                
        return exported_files

    @staticmethod
    def export_to_mat(model_data, output_dir="matlab_exports", prefix=""):
        """
        Export model data to a .mat file for MATLAB.
        
        Creates a single .mat file containing all model data in a hierarchical
        structure that can be easily loaded and analyzed in MATLAB.
        
        Args:
            model_data: Dictionary containing model training metrics
            output_dir: Directory where .mat file will be saved
            prefix: String prefix for output filename
            
        Returns:
            List containing the path to the exported .mat file, or empty list if export fails
        """
        if not HAS_SCIPY:
            print("Error: Cannot export .mat file - scipy dependency missing")
            return []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create an integrated .mat file containing data for all models
        mat_file = os.path.join(output_dir, f"{prefix}model_data.mat")
        
        # Prepare data dictionary for saving
        export_data = {}
        
        for model_name, data in model_data.items():
            # Clean model name to be valid in MATLAB
            safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
            
            # Create subdictionary for each model
            model_dict = {}
            
            # Add batch data
            if data["batch_loss"]:
                model_dict["batch_loss"] = np.array(data["batch_loss"])
            if data["batch_bit_err"]:
                model_dict["batch_bit_err"] = np.array(data["batch_bit_err"])
                
            # Add update data
            if data["update_counts"]:
                model_dict["update_counts"] = np.array(data["update_counts"])
            
            # Add metrics by updates
            metrics_dict = {}
            for metric_name, metric_values in data["metrics_by_updates"].items():
                if metric_values:  # Only add non-empty lists
                    metrics_dict[metric_name] = np.array(metric_values)
            
            if metrics_dict:
                model_dict["metrics_by_updates"] = metrics_dict
                
            # Add epoch data
            if data["epoch_loss"]:
                model_dict["epoch_loss"] = np.array(data["epoch_loss"])
            if data["epoch_bit_err"]:
                model_dict["epoch_bit_err"] = np.array(data["epoch_bit_err"])
            if data["val_epoch_loss"]:
                model_dict["val_epoch_loss"] = np.array(data["val_epoch_loss"]) 
            if data["val_epoch_bit_err"]:
                model_dict["val_epoch_bit_err"] = np.array(data["val_epoch_bit_err"])
                
            # Add metadata
            model_dict["dataset_type"] = data["dataset_type"]
            model_dict["final_update_count"] = data["final_update_count"]
                
            # Add model data to export dictionary
            export_data[safe_model_name] = model_dict
            
        # Save .mat file
        sio.savemat(mat_file, export_data)
        
        return [mat_file]
    
    @staticmethod
    def export_to_json(model_data, output_dir="matlab_exports", prefix=""):
        """
        Export model data to JSON format.
        
        Creates a JSON file that can be imported into MATLAB or used
        for web-based visualization of training metrics.
        
        Args:
            model_data: Dictionary containing model training metrics
            output_dir: Directory where JSON file will be saved
            prefix: String prefix for output filename
            
        Returns:
            List containing the path to the exported JSON file, or empty list if export fails
        """
        import os
        import json
        import numpy as np
        
        # Helper function for handling TensorFlow tensors and other complex types
        def convert_to_serializable(obj):
            # Check if it's a TensorFlow tensor
            if str(type(obj)).find('tensorflow') >= 0 or str(type(obj)).find('EagerTensor') >= 0:
                try:
                    # Try to convert to numpy, then to Python native type
                    return obj.numpy().tolist() if hasattr(obj, 'numpy') else float(obj)
                except:
                    return str(obj)
            
            # If numpy array or numpy numeric type
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.number, np.bool_)):
                return obj.item()
            
            # Handle lists
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            
            # Handle dictionaries
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            
            # Handle other non-serializable types
            elif hasattr(obj, '__dict__'):
                return str(obj)
            
            # Return original object
            return obj
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create main JSON file
        json_file = os.path.join(output_dir, f"{prefix}model_data.json")
        
        # Prepare JSON serializable data
        json_data = {}
        
        for model_name, data in model_data.items():
            # Clean model name
            safe_model_name = model_name.replace("/", "_").replace("\\", "_")
            
            # Use helper function to convert all data to serializable format
            model_dict = convert_to_serializable(data)
            json_data[safe_model_name] = model_dict
        
        # Custom JSON encoder for handling special types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.ndarray, np.number)):
                    return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
                # Check for TensorFlow types
                if str(type(obj)).find('tensorflow') >= 0 or str(type(obj)).find('EagerTensor') >= 0:
                    try:
                        return obj.numpy().tolist() if hasattr(obj, 'numpy') else float(obj)
                    except:
                        return str(obj)
                return json.JSONEncoder.default(self, obj)
        
        # Write JSON file
        try:
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2, cls=CustomJSONEncoder)
            return [json_file]
        except Exception as e:
            print(f"JSON export failed: {e}")
            # Try again with lower format requirements
            try:
                print("Attempting export with lower format requirements...")
                with open(json_file, 'w') as f:
                    # Convert to strings and store
                    simple_json = {k: str(v) for k, v in model_data.items()}
                    json.dump(simple_json, f, indent=2)
                return [json_file]
            except Exception as e2:
                print(f"Export failed again: {e2}")
                return []