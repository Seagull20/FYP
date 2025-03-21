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
    print("警告: 未找到scipy. 将无法导出.mat格式。请安装scipy: pip install scipy")


class MatlabExport:
    """用于将MultiModelBCP数据导出为MATLAB兼容格式的工具类"""
    
    @staticmethod
    def export_to_csv(model_data, output_dir="matlab_exports", prefix=""):
        """将模型数据导出到CSV文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        exported_files = []
        
        # 为每个模型导出数据
        for model_name, data in model_data.items():
            safe_model_name = model_name.replace("/", "_").replace("\\", "_")
            
            # 导出批次级别数据
            if data["batch_loss"] and data["batch_bit_err"]:
                batch_file = os.path.join(output_dir, f"{prefix}{safe_model_name}_batch.csv")
                with open(batch_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["batch_index", "loss", "bit_err"])
                    for i, (loss, bit_err) in enumerate(zip(data["batch_loss"], data["batch_bit_err"])):
                        writer.writerow([i, loss, bit_err])
                exported_files.append(batch_file)
            
            # 导出更新级别数据
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
            
            # 导出验证数据 
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
            
            # 导出纪元级别数据
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
        """将模型数据导出到.mat文件 (需要scipy)"""
        if not HAS_SCIPY:
            print("错误: 无法导出.mat文件 - 缺少scipy依赖")
            return []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 创建整合的.mat文件，包含所有模型的数据
        mat_file = os.path.join(output_dir, f"{prefix}model_data.mat")
        
        # 准备要保存的数据字典
        export_data = {}
        
        for model_name, data in model_data.items():
            # 清理模型名称，使其在MATLAB中有效
            safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
            
            # 为每个模型创建子字典
            model_dict = {}
            
            # 添加批次数据
            if data["batch_loss"]:
                model_dict["batch_loss"] = np.array(data["batch_loss"])
            if data["batch_bit_err"]:
                model_dict["batch_bit_err"] = np.array(data["batch_bit_err"])
                
            # 添加更新数据
            if data["update_counts"]:
                model_dict["update_counts"] = np.array(data["update_counts"])
            
            # 添加按更新的指标
            metrics_dict = {}
            for metric_name, metric_values in data["metrics_by_updates"].items():
                if metric_values:  # 只添加非空列表
                    metrics_dict[metric_name] = np.array(metric_values)
            
            if metrics_dict:
                model_dict["metrics_by_updates"] = metrics_dict
                
            # 添加纪元数据
            if data["epoch_loss"]:
                model_dict["epoch_loss"] = np.array(data["epoch_loss"])
            if data["epoch_bit_err"]:
                model_dict["epoch_bit_err"] = np.array(data["epoch_bit_err"])
            if data["val_epoch_loss"]:
                model_dict["val_epoch_loss"] = np.array(data["val_epoch_loss"]) 
            if data["val_epoch_bit_err"]:
                model_dict["val_epoch_bit_err"] = np.array(data["val_epoch_bit_err"])
                
            # 添加元数据
            model_dict["dataset_type"] = data["dataset_type"]
            model_dict["final_update_count"] = data["final_update_count"]
                
            # 将模型数据添加到导出字典
            export_data[safe_model_name] = model_dict
            
        # 保存.mat文件
        sio.savemat(mat_file, export_data)
        
        return [mat_file]
    
    @staticmethod
    def export_to_json(model_data, output_dir="matlab_exports", prefix=""):
        """将模型数据导出为JSON文件 (MATLAB也可以导入)"""
        import os
        import json
        import numpy as np
        
        # 辅助函数，用于处理TensorFlow张量和其他复杂类型
        def convert_to_serializable(obj):
            # 检查是否为TensorFlow张量
            if str(type(obj)).find('tensorflow') >= 0 or str(type(obj)).find('EagerTensor') >= 0:
                try:
                    # 尝试转换为numpy，然后转换为Python原生类型
                    return obj.numpy().tolist() if hasattr(obj, 'numpy') else float(obj)
                except:
                    return str(obj)
            
            # 如果是numpy数组或numpy数值类型
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.number, np.bool_)):
                return obj.item()
            
            # 处理列表
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            
            # 处理字典
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            
            # 处理其他不可序列化类型
            elif hasattr(obj, '__dict__'):
                return str(obj)
            
            # 返回原始对象
            return obj
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 创建主JSON文件
        json_file = os.path.join(output_dir, f"{prefix}model_data.json")
        
        # 准备JSON可序列化数据
        json_data = {}
        
        for model_name, data in model_data.items():
            # 清理模型名称
            safe_model_name = model_name.replace("/", "_").replace("\\", "_")
            
            # 使用辅助函数转换所有数据为可序列化格式
            model_dict = convert_to_serializable(data)
            json_data[safe_model_name] = model_dict
        
        # 使用自定义JSON编码器处理特殊类型
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.ndarray, np.number)):
                    return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
                # 检查TensorFlow类型
                if str(type(obj)).find('tensorflow') >= 0 or str(type(obj)).find('EagerTensor') >= 0:
                    try:
                        return obj.numpy().tolist() if hasattr(obj, 'numpy') else float(obj)
                    except:
                        return str(obj)
                return json.JSONEncoder.default(self, obj)
        
        # 写入JSON文件
        try:
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2, cls=CustomJSONEncoder)
            return [json_file]
        except Exception as e:
            print(f"导出JSON失败: {e}")
            # 尝试以较低的格式要求重试
            try:
                print("尝试以较低的格式要求重试导出...")
                with open(json_file, 'w') as f:
                    # 转换为字符串然后存储
                    simple_json = {k: str(v) for k, v in model_data.items()}
                    json.dump(simple_json, f, indent=2)
                return [json_file]
            except Exception as e2:
                print(f"再次导出失败: {e2}")
                return []