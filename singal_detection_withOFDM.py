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

class MultiModelBCP(Callback):
    def __init__(self, model_name, dataset_type="default", sampling_rate=10, max_points=1000):
        super(MultiModelBCP, self).__init__()
        self.model_name = model_name
        self.dataset_type = dataset_type
        self.sampling_rate = sampling_rate  # 每隔多少批次记录一次
        self.max_points = max_points  # 最大记录点数
        
        # 使用下采样的度量标准
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
        logs = logs or {}
        self.current_update_count += 1
        self.batch_counter += 1
        
        # 只在采样点记录指标
        if self.batch_counter % self.sampling_rate == 0:
            # 处理最大点数限制，如果需要，删除旧数据
            if len(self.batch_loss) >= self.max_points:
                # 移除一半的旧数据点以节省内存
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
        logs = logs or {}
        self.epoch_loss.append(logs.get('loss', 0))
        self.epoch_bit_err.append(logs.get('bit_err', 0))
        self.val_epoch_loss.append(logs.get('val_loss', 0))
        self.val_epoch_bit_err.append(logs.get('val_bit_err', 0))
        
        # 记录验证指标
        self.metrics_by_updates["val_loss"].append(logs.get('val_loss', 0))
        self.metrics_by_updates["val_bit_err"].append(logs.get('val_bit_err', 0))
        self.metrics_by_updates["val_update_counts"].append(self.current_update_count)

    def on_train_end(self, logs=None):
        # 保存数据到静态字典
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
        
        # 清除实例变量以释放内存
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
        """Record MetaDNN metrics with update counts"""
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
    def plot_all_learning_curves(save_path="multi_model_learning_curve.png", 
                                 plot_batch=True, plot_epoch=True, plot_train_bit_err=False):
        if not MultiModelBCP.all_models_data:
            print("No data to plot.")
            return

        num_plots = plot_batch + plot_epoch
        if num_plots == 0:
            print("No plots selected.")
            return
        plt.figure(figsize=(6 * num_plots, 5))

        plot_idx = 1
        if plot_batch:
            plt.subplot(1, num_plots, plot_idx)
            for model_name, data in MultiModelBCP.all_models_data.items():
                if data["batch_loss"]:
                    plt.plot(data["batch_loss"], label=f"{model_name} Loss")
                    plt.plot(np.linspace(0, len(data["batch_loss"]), len(data["epoch_bit_err"])),
                             data["epoch_bit_err"], label=f"{model_name} Bit Err", alpha=0.6)
            plt.title("Batch-Level Learning Curves (Train Phase)")
            plt.xlabel("Batch Index")
            plt.ylabel("Metric Value")
            plt.legend()
            plot_idx += 1

        if plot_epoch:
            plt.subplot(1, num_plots, plot_idx)
            for model_name, data in MultiModelBCP.all_models_data.items():
                plt.plot(data["val_epoch_bit_err"], label=f"{model_name} Val Bit Err",
                         marker='o' if len(data["val_epoch_bit_err"]) == 1 else None)
                if plot_train_bit_err and data["epoch_bit_err"]:
                    plt.plot(data["epoch_bit_err"], label=f"{model_name} Bit Err", linestyle="--")
            plt.title("Epoch-Level Learning Curves (Train + Generalization)")
            plt.xlabel("Epoch")
            plt.ylabel("Bit Error Rate")
            plt.legend()

        plt.tight_layout(pad=3.0)
        plt.savefig(save_path)
        plt.close()
        print(f"Multi-model learning curves saved to {save_path}")

    @staticmethod
    def plot_by_updates(save_path="update_comparison.png", models_to_plot=None, dpi=150):
        """绘制基于更新的学习曲线，可以选择性地只绘制部分模型"""
        if not MultiModelBCP.all_models_data:
            print("No data to plot.")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, data in MultiModelBCP.all_models_data.items():
            # 如果指定了models_to_plot，则只绘制指定的模型
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
        """清除所有或指定模型的数据"""
        if model_names is None:
            MultiModelBCP.all_models_data = {}
        else:
            for name in model_names:
                if name in MultiModelBCP.all_models_data:
                    del MultiModelBCP.all_models_data[name]

    # 将这些方法添加到MultiModelBCP类

    @staticmethod
    def export_data_for_matlab(output_dir="matlab_exports", format="all", prefix=""):
        """
        将所有模型的训练数据导出为MATLAB兼容格式
        
        参数:
            output_dir: 导出目录路径
            format: 输出格式 ('csv', 'mat', 'json', 或 'all')
            prefix: 文件名前缀
            
        返回:
            导出文件的路径列表
        """
        from datetime import datetime
        
        # 如果all_models_data未初始化，提前返回
        if not hasattr(MultiModelBCP, "all_models_data") or not MultiModelBCP.all_models_data:
            print("警告：没有可导出的模型数据。")
            return []
        
        # 如果未传递前缀，使用时间戳
        if not prefix:
            prefix = f"model_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        
        # 导入MatlabExport类
        try:
            # 假设MatlabExport类已定义在单独的文件中
            from matlab_export import MatlabExport
        except ImportError:
            # 如果导入失败，我们需要定义MatlabExport类
            # 在这里粘贴MatlabExport类的代码
            # 我们将此类的代码移到单独的文件中来保持整洁
            pass
        
        exported_files = []
        
        # 根据请求的格式导出数据
        if format.lower() in ["csv", "all"]:
            csv_files = MatlabExport.export_to_csv(
                MultiModelBCP.all_models_data, 
                output_dir=output_dir, 
                prefix=prefix
            )
            exported_files.extend(csv_files)
            print(f"已导出{len(csv_files)}个CSV文件到 {output_dir}")
        
        if format.lower() in ["mat", "all"]:
            try:
                mat_files = MatlabExport.export_to_mat(
                    MultiModelBCP.all_models_data, 
                    output_dir=output_dir, 
                    prefix=prefix
                )
                exported_files.extend(mat_files)
                if mat_files:
                    print(f"已导出MAT文件到 {mat_files[0]}")
            except Exception as e:
                print(f"导出MAT文件时出错: {e}")
        
        if format.lower() in ["json", "all"]:
            json_files = MatlabExport.export_to_json(
                MultiModelBCP.all_models_data, 
                output_dir=output_dir, 
                prefix=prefix
            )
            exported_files.extend(json_files)
            print(f"已导出JSON文件到 {json_files[0]}")
        
        return exported_files

    @staticmethod
    def generate_matlab_script(output_dir="matlab_exports", exported_files=None, prefix="", 
                            sample_sizes=None):
        """
        生成一个MATLAB脚本，用于加载和分析导出的数据
        
        参数:
            output_dir: 导出目录路径
            exported_files: 导出文件的路径列表
            prefix: 文件名前缀
            sample_sizes: 样本大小列表，如[50, 100, 500, 1000]
            
        返回:
            MATLAB脚本文件的路径
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
            
            # 添加样本大小数组定义
            if sample_sizes:
                f.write("% 定义样本大小数组\n")
                f.write(f"sample_sizes = [{', '.join(map(str, sample_sizes))}];\n\n")
                
            # 根据导出的文件格式添加加载代码
            if has_mat:
                # 找到第一个.mat文件
                mat_files = [f for f in exported_files if f.endswith('.mat')]
                if mat_files:
                    mat_file_basename = os.path.basename(mat_files[0])
                    f.write("% 加载MAT文件数据\n")
                    f.write(f"load('{mat_file_basename}');\n\n")

                    f.write("% 获取并显示所有模型名称\n")
                    f.write("vars = whos;\n")  # 使用whos来获取加载的变量信息
                    f.write("model_names = {vars.name};\n")  # 提取变量名称
                    f.write("fprintf('已加载的模型: \\n');\n")
                    f.write("for i = 1:length(model_names)\n")
                    f.write("    fprintf('  %s\\n', model_names{i});\n")
                    f.write("end\n\n")

                    f.write("% 创建models结构体，将所有模型放入其中（方便后续处理）\n")
                    f.write("models = struct();\n")
                    f.write("for i = 1:length(model_names)\n")
                    f.write("    models.(model_names{i}) = eval(model_names{i});\n")
                    f.write("end\n\n")
                    
                    f.write("% 分析验证误差率\n")
                    f.write("figure('Name', '验证误差率对比', 'Position', [100, 100, 1200, 600]);\n")
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
                    f.write("        fprintf('处理模型 %s 时出错: %s\\n', model_names{i}, e.message);\n")
                    f.write("    end\n")
                    f.write("end\n")
                    f.write("if ~isempty(legends)\n")
                    f.write("    xlabel('更新次数');\n")
                    f.write("    ylabel('验证误差率');\n")
                    f.write("    title('不同模型在验证集上的误差率对比');\n")
                    f.write("    grid on;\n")
                    f.write("    legend(legends, 'Location', 'best');\n")
                    f.write("    set(gca, 'YScale', 'log');\n")  # 使用对数尺度可能更好地显示误差率
                    f.write("else\n")
                    f.write("    title('没有找到有效的验证误差率数据');\n")
                    f.write("end\n")
                    f.write("hold off;\n\n")
                    
                    f.write("% 分析训练损失\n")
                    f.write("figure('Name', '训练损失对比', 'Position', [100, 100, 1200, 600]);\n")
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
                    f.write("        fprintf('处理模型 %s 训练损失时出错: %s\\n', model_names{i}, e.message);\n")
                    f.write("    end\n")
                    f.write("end\n")
                    f.write("if ~isempty(legends)\n")
                    f.write("    xlabel('更新次数');\n")
                    f.write("    ylabel('训练损失');\n")
                    f.write("    title('不同模型的训练损失对比');\n")
                    f.write("    grid on;\n")
                    f.write("    legend(legends, 'Location', 'best');\n")
                    f.write("else\n")
                    f.write("    title('没有找到有效的训练损失数据');\n")
                    f.write("end\n")
                    f.write("hold off;\n\n")
                    
                    # 添加对样本大小比较函数的调用
                    f.write("% 绘制不同样本大小下各模型的性能比较\n")
                    if sample_sizes:
                        f.write("plot_sample_size_comparison(models, sample_sizes);\n\n")
                    else:
                        f.write("try\n")
                        f.write("    % 尝试自动提取样本大小并绘制性能比较图\n")
                        f.write("    plot_sample_size_comparison(models);\n")
                        f.write("catch e\n")
                        f.write("    fprintf('绘制样本大小比较图时出错: %s\\n', e.message);\n")
                        f.write("    fprintf('您可以手动调用: plot_sample_size_comparison(models, [50, 100, 500, 1000])\\n');\n")
                        f.write("end\n\n")
                    
                else:
                    f.write("% 未找到.mat文件\n\n")
                    
            elif has_json:
                f.write("% 加载JSON文件数据\n")
                f.write("try\n")
                
                json_files = [f for f in exported_files if f.endswith('.json')]
                if json_files:
                    json_file_basename = os.path.basename(json_files[0])
                    f.write("    json_file = fopen('" + json_file_basename + "');\n")
                    f.write("    json_str = char(fread(json_file, inf))';\n")
                    f.write("    fclose(json_file);\n")
                    f.write("    model_data = jsondecode(json_str);\n")
                    f.write("    fprintf('成功加载JSON数据\\n');\n")
                    
                    f.write("    % 获取模型名称\n")
                    f.write("    model_names = fieldnames(model_data);\n")
                    f.write("    fprintf('已加载的模型: \\n');\n")
                    f.write("    for i = 1:length(model_names)\n")
                    f.write("        fprintf('  %s\\n', model_names{i});\n")
                    f.write("    end\n\n")
                    
                    # 添加JSON数据分析代码
                    f.write("    % 分析验证误差率\n")
                    f.write("    figure('Name', 'JSON数据：验证误差率对比', 'Position', [100, 100, 1200, 600]);\n")
                    f.write("    hold on;\n")
                    f.write("    % 分析代码...\n")
                    
                    # 添加对样本大小比较函数的调用
                    f.write("    % 绘制不同样本大小下各模型的性能比较\n")
                    if sample_sizes:
                        f.write("    plot_sample_size_comparison(model_data, sample_sizes);\n\n")
                    else:
                        f.write("    try\n")
                        f.write("        % 尝试自动提取样本大小并绘制性能比较图\n")
                        f.write("        plot_sample_size_comparison(model_data);\n")
                        f.write("    catch e\n")
                        f.write("        fprintf('绘制样本大小比较图时出错: %s\\n', e.message);\n")
                        f.write("        fprintf('您可以手动调用: plot_sample_size_comparison(model_data, [50, 100, 500, 1000])\\n');\n")
                        f.write("    end\n\n")
                
                f.write("catch e\n")
                f.write("    fprintf('加载JSON数据失败: %s\\n', e.message);\n")
                f.write("end\n\n")
                
            elif has_csv:
                f.write("% 加载CSV文件数据\n")
                f.write("csv_files = dir(fullfile(pwd, '*.csv'));\n")
                f.write("fprintf('找到%d个CSV文件\\n', length(csv_files));\n")
                f.write("\n")
                f.write("% 创建数据结构来保存所有模型的数据\n")
                f.write("models = struct();\n")
                f.write("\n")
                f.write("% 加载所有CSV文件\n")
                f.write("for i = 1:length(csv_files)\n")
                f.write("    file_name = csv_files(i).name;\n")
                f.write("    [~, name, ~] = fileparts(file_name);\n")
                f.write("    \n")
                f.write("    % 解析文件名以获取模型名称和数据类型\n")
                f.write("    parts = strsplit(name, '_');\n")
                f.write("    if length(parts) >= 2\n")
                f.write("        model_name = parts{1};\n")
                f.write("        for j = 2:length(parts)-1\n")
                f.write("            model_name = [model_name, '_', parts{j}];\n")
                f.write("        end\n")
                f.write("        data_type = parts{end};\n")
                f.write("        \n")
                f.write("        % 创建模型结构（如果不存在）\n")
                f.write("        if ~isfield(models, model_name)\n")
                f.write("            models.(model_name) = struct();\n")
                f.write("        end\n")
                f.write("        \n")
                f.write("        % 读取数据\n")
                f.write("        try\n")
                f.write("            data = readtable(file_name);\n")
                f.write("            models.(model_name).(data_type) = data;\n")
                f.write("            fprintf('已加载: %s (%s)\\n', model_name, data_type);\n")
                f.write("        catch e\n")
                f.write("            fprintf('加载文件 %s 失败: %s\\n', file_name, e.message);\n")
                f.write("        end\n")
                f.write("    end\n")
                f.write("end\n\n")
                
                f.write("% 检查是否有加载的模型数据\n")
                f.write("if ~isempty(fieldnames(models))\n")
                f.write("    % 绘制验证误差率对比\n")
                f.write("    figure('Name', '验证误差率对比', 'Position', [100, 100, 1200, 600]);\n")
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
                f.write("            fprintf('处理模型 %s 时出错: %s\\n', model_names{i}, e.message);\n")
                f.write("        end\n")
                f.write("    end\n")
                f.write("    if ~isempty(legends)\n")
                f.write("        xlabel('更新次数');\n")
                f.write("        ylabel('验证误差率');\n")
                f.write("        title('不同模型在验证集上的误差率对比');\n")
                f.write("        grid on;\n")
                f.write("        legend(legends, 'Location', 'best');\n")
                f.write("        set(gca, 'YScale', 'log');\n")
                f.write("    else\n")
                f.write("        title('没有找到有效的验证数据');\n")
                f.write("    end\n")
                f.write("    hold off;\n\n")
                
                # 添加对样本大小比较函数的调用
                f.write("    % 绘制不同样本大小下各模型的性能比较\n")
                if sample_sizes:
                    f.write("    plot_sample_size_comparison(models, sample_sizes);\n\n")
                else:
                    f.write("    try\n")
                    f.write("        % 尝试自动提取样本大小并绘制性能比较图\n")
                    f.write("        plot_sample_size_comparison(models);\n")
                    f.write("    catch e\n")
                    f.write("        fprintf('绘制样本大小比较图时出错: %s\\n', e.message);\n")
                    f.write("        fprintf('您可以手动调用: plot_sample_size_comparison(models, [50, 100, 500, 1000])\\n');\n")
                    f.write("    end\n\n")
                    
                f.write("else\n")
                f.write("    fprintf('警告: 未能加载任何模型数据\\n');\n")
                f.write("end\n\n")
                
            # 添加数据分析函数
            f.write("%% 自定义分析函数\n\n")
            
            # 添加样本大小比较函数
            f.write("% 函数：比较不同样本大小下各模型的最终性能\n")
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
            
            # 添加辅助函数
            f.write("% 辅助函数：从模型名称中提取样本大小\n")
            f.write("function sample_sizes = extract_sample_sizes_from_models(models)\n")
            f.write("    model_names = fieldnames(models);\n")
            f.write("    sample_sizes = [];\n")
            f.write("    \n")
            f.write("    for i = 1:length(model_names)\n")
            f.write("        name = model_names{i};\n")
            f.write("        \n")
            f.write("        % 查找形如 '_数字_' 的模式\n")
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
            f.write("        elseif contains(name, '_3GPP_')\n")
            f.write("            % 提取 '_3GPP_数字' 中的数字\n")
            f.write("            pattern = '_3GPP_(\d+)';\n")
            f.write("            matches = regexp(name, pattern, 'tokens');\n")
            f.write("            if ~isempty(matches)\n")
            f.write("                size_str = matches{1}{1};\n")
            f.write("                size_num = str2double(size_str);\n")
            f.write("                if ~isnan(size_num) && ~ismember(size_num, sample_sizes)\n")
            f.write("                    sample_sizes = [sample_sizes, size_num];\n")
            f.write("                end\n")
            f.write("            end\n")
            f.write("        elseif contains(name, 'Meta_')\n")
            f.write("            % 提取 'Meta_数字' 中的数字\n")
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
            
            f.write("% 辅助函数：提取基本模型名称（不包含样本大小）\n")
            f.write("function base_name = extract_base_model_name(name)\n")
            f.write("    % 提取基本模型名称，忽略样本大小部分\n")
            f.write("    \n")
            f.write("    % 对于DNN类型的模型\n")
            f.write("    if contains(name, 'DNN_')\n")
            f.write("        if contains(name, '_3GPP_')\n")
            f.write("            % DNN_channel_3GPP_size 格式\n")
            f.write("            parts = strsplit(name, '_3GPP_');\n")
            f.write("            base_name = parts{1};\n")
            f.write("        else\n")
            f.write("            % 只有DNN_channel 格式\n")
            f.write("            base_name = name;\n")
            f.write("        end\n")
            f.write("    % 对于Meta类型的模型\n")
            f.write("    elseif contains(name, 'Meta_')\n")
            f.write("        if ~contains(name, 'Meta_DNN') % 排除Meta_DNN_train这种情况\n")
            f.write("            % Meta_size 格式，统一使用Meta作为基本名称\n")
            f.write("            base_name = 'Meta';\n")
            f.write("        else\n")
            f.write("            base_name = name;\n")
            f.write("        end\n")
            f.write("    else\n")
            f.write("        % 其他情况，返回原始名称\n")
            f.write("        base_name = name;\n")
            f.write("    end\n")
            f.write("end\n\n")
            
            f.write("% 辅助函数：查找包含特定基本名称和样本大小的模型\n")
            f.write("function model_name = find_model_with_size(models, base_name, size_str)\n")
            f.write("    model_names = fieldnames(models);\n")
            f.write("    model_name = '';\n")
            f.write("    \n")
            f.write("    % 根据基本模型类型尝试不同的匹配模式\n")
            f.write("    if strcmp(base_name, 'Meta')\n")
            f.write("        % Meta模型的命名模式为Meta_size\n")
            f.write("        pattern = ['Meta_' size_str '$'];\n")
            f.write("        for i = 1:length(model_names)\n")
            f.write("            if ~isempty(regexp(model_names{i}, pattern, 'once'))\n")
            f.write("                model_name = model_names{i};\n")
            f.write("                return;\n")
            f.write("            end\n")
            f.write("        end\n")
            f.write("    else\n")
            f.write("        % DNN模型的命名模式为DNN_channel_3GPP_size\n")
            f.write("        pattern = [base_name '_3GPP_' size_str '$'];\n")
            f.write("        for i = 1:length(model_names)\n")
            f.write("            if ~isempty(regexp(model_names{i}, pattern, 'once'))\n")
            f.write("                model_name = model_names{i};\n")
            f.write("                return;\n")
            f.write("            end\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("end\n\n")
            
            f.write("% 辅助函数：提取模型的最终验证误差率\n")
            f.write("function final_error = extract_final_error(model)\n")
            f.write("    if isfield(model, 'metrics_by_updates') && isfield(model.metrics_by_updates, 'val_bit_err') && ~isempty(model.metrics_by_updates.val_bit_err)\n")
            f.write("        % 使用metrics_by_updates中的数据\n")
            f.write("        final_error = model.metrics_by_updates.val_bit_err(end);\n")
            f.write("    elseif isfield(model, 'val_epoch_bit_err') && ~isempty(model.val_epoch_bit_err)\n")
            f.write("        % 使用val_epoch_bit_err中的数据\n")
            f.write("        final_error = model.val_epoch_bit_err(end);\n")
            f.write("    elseif isfield(model, 'validation') && isfield(model.validation, 'val_bit_err') && ~isempty(model.validation.val_bit_err)\n")
            f.write("        % 使用CSV导出的validation数据\n")
            f.write("        final_error = model.validation.val_bit_err{end};\n")
            f.write("    else\n")
            f.write("        % 无有效数据\n")
            f.write("        final_error = NaN;\n")
            f.write("    end\n")
            f.write("end\n\n")
            
            f.write("% 函数：比较不同样本大小下的模型性能\n")
            f.write("function compare_sample_sizes(models, pattern)\n")
            f.write("    % 此函数比较不同样本大小下模型的性能\n")
            f.write("    % 参数:\n")
            f.write("    %   models - 包含模型数据的结构\n")
            f.write("    %   pattern - 用于筛选模型的字符串模式 (例如, '3GPP_')\n")
            f.write("    \n")
            f.write("    if nargin < 1\n")
            f.write("        error('必须提供models参数');\n")
            f.write("    end\n")
            f.write("    if nargin < 2\n")
            f.write("        pattern = ''; % 默认不筛选\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    figure('Name', ['不同样本大小下的性能对比 - ' pattern], 'Position', [100, 100, 1200, 600]);\n")
            f.write("    hold on;\n")
            f.write("    \n")
            f.write("    model_names = fieldnames(models);\n")
            f.write("    matching_models = {};\n")
            f.write("    \n")
            f.write("    % 筛选匹配的模型\n")
            f.write("    for i = 1:length(model_names)\n")
            f.write("        if isempty(pattern) || contains(model_names{i}, pattern)\n")
            f.write("            matching_models{end+1} = model_names{i};\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    legends = {};\n")
            f.write("    % 绘制匹配的模型\n")
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
            f.write("            fprintf('处理模型 %s 时出错: %s\\n', matching_models{i}, e.message);\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    if ~isempty(legends)\n")
            f.write("        xlabel('更新次数');\n")
            f.write("        ylabel('验证误差率');\n")
            f.write("        title(['不同样本大小下' pattern '模型的性能对比']);\n")
            f.write("        grid on;\n")
            f.write("        legend(legends, 'Location', 'best');\n")
            f.write("        set(gca, 'YScale', 'log');\n")
            f.write("    else\n")
            f.write("        title('没有找到匹配的模型数据');\n")
            f.write("    end\n")
            f.write("    hold off;\n")
            f.write("end\n\n")
            
            f.write("% 函数：分析模型收敛速度\n")
            f.write("function analyze_convergence(models, threshold)\n")
            f.write("    % 此函数分析不同模型达到特定性能阈值所需的更新次数\n")
            f.write("    % 参数:\n")
            f.write("    %   models - 包含模型数据的结构\n")
            f.write("    %   threshold - 性能阈值 (例如, 0.02 表示2%的误差率)\n")
            f.write("    \n")
            f.write("    if nargin < 1\n")
            f.write("        error('必须提供models参数');\n")
            f.write("    end\n")
            f.write("    if nargin < 2\n")
            f.write("        threshold = 0.02; % 默认阈值\n")
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
            f.write("                % 找到首次低于阈值的更新次数\n")
            f.write("                idx = find(model.validation.val_bit_err <= threshold, 1, 'first');\n")
            f.write("                if ~isempty(idx)\n")
            f.write("                    convergence_updates(i) = model.validation.update_count(idx);\n")
            f.write("                    valid_models(i) = true;\n")
            f.write("                else\n")
            f.write("                    convergence_updates(i) = NaN; % 未收敛\n")
            f.write("                end\n")
            f.write("                \n")
            f.write("                % 记录最终性能\n")
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
            f.write("                    convergence_updates(i) = NaN; % 未收敛\n")
            f.write("                end\n")
            f.write("                \n")
            f.write("                final_errors(i) = error_values(end);\n")
            f.write("                valid_models(i) = true;\n")
            f.write("            else\n")
            f.write("                convergence_updates(i) = NaN;\n")
            f.write("                final_errors(i) = NaN;\n")
            f.write("            end\n")
            f.write("        catch e\n")
            f.write("            fprintf('处理模型 %s 时出错: %s\\n', model_names{i}, e.message);\n")
            f.write("            convergence_updates(i) = NaN;\n")
            f.write("            final_errors(i) = NaN;\n")
            f.write("        end\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    % 只保留有效模型的数据\n")
            f.write("    valid_idx = find(valid_models);\n")
            f.write("    if isempty(valid_idx)\n")
            f.write("        fprintf('没有找到有效的模型数据进行收敛分析\\n');\n")
            f.write("        return;\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    model_names = model_names(valid_idx);\n")
            f.write("    convergence_updates = convergence_updates(valid_idx);\n")
            f.write("    final_errors = final_errors(valid_idx);\n")
            f.write("    \n")
            f.write("    % 创建结果表格\n")
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
            f.write("    % 显示结果\n")
            f.write("    fprintf('收敛分析结果 (阈值: %.4f):\\n', threshold);\n")
            f.write("    disp(results_cell);\n")
            f.write("    \n")
            f.write("    % 绘制收敛更新次数的条形图\n")
            f.write("    figure('Name', '收敛速度对比', 'Position', [100, 100, 1200, 600]);\n")
            f.write("    \n")
            f.write("    % 找出有效的收敛数据\n")
            f.write("    conv_idx = ~isnan(convergence_updates);\n")
            f.write("    if sum(conv_idx) > 0\n")
            f.write("        bar(convergence_updates(conv_idx));\n")
            f.write("        set(gca, 'XTick', 1:sum(conv_idx), 'XTickLabel', model_names(conv_idx), 'XTickLabelRotation', 45);\n")
            f.write("        ylabel(['达到 ' num2str(threshold) ' 阈值所需的更新次数']);\n")
            f.write("        title(['不同模型达到 ' num2str(threshold) ' 误差率的收敛速度']);\n")
            f.write("        grid on;\n")
            f.write("    else\n")
            f.write("        text(0.5, 0.5, '没有模型达到收敛阈值', 'HorizontalAlignment', 'center', 'Units', 'normalized');\n")
            f.write("    end\n")
            f.write("    \n")
            f.write("    % 绘制最终误差率\n")
            f.write("    figure('Name', '最终误差率对比', 'Position', [100, 100, 1200, 600]);\n")
            f.write("    bar(final_errors);\n")
            f.write("    set(gca, 'XTick', 1:length(model_names), 'XTickLabel', model_names, 'XTickLabelRotation', 45);\n")
            f.write("    ylabel('最终误差率');\n")
            f.write("    title('不同模型的最终误差率对比');\n")
            f.write("    grid on;\n")
            f.write("end\n")
                
        print(f"MATLAB分析脚本已生成到: {script_path}")
        return script_path

class signal_simulator():
    def __init__(self, SNR=10):
        self.all_carriers = np.arange(K)
        self.pilot_carriers = self.all_carriers[::K // P]
        self.data_carriers = np.delete(self.all_carriers, self.pilot_carriers)
        self.payloadBits_per_OFDM = len(self.data_carriers) * mu
        self.channel_3gpp_loaded = False
        self.SNRdB = SNR
    
    def lazy_load_channel(self):
        """延迟加载通道数据"""
        if not self.channel_3gpp_loaded:
            self.channel_3gpp = np.load('channel_train.npy')
            self.channel_3gpp_loaded = True

    

    def generate_bits(self, num_samples):
        return np.random.binomial(n=1, p=0.5, size=(num_samples, self.payloadBits_per_OFDM))
    
    def transmit_signals(self, bits):
        bits_sp = self.sp(bits)
        qam = self.mapping(bits_sp)
        ofdm_data = self.ofdm_symbol(qam)
        ofdm_time = self.idft(ofdm_data)
        ofdm_with_cp = self.add_cp(ofdm_time)
        return ofdm_with_cp
    
    def received_signals(self, transmit_signals, channel_type):
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
        elif self.channel_type == "3gpp":
            self.lazy_load_channel()
            train_size = self.channel_3gpp.shape[0]
            index = np.random.choice(np.arange(train_size), size=1)
            h = self.channel_3gpp[index]
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
        ofdm_tx = self.transmit_signals(bits)
        ofdm_rx = self.received_signals(ofdm_tx, channel_type)
        ofdm_rx_no_cp = self.remove_cp(ofdm_rx)
        ofdm_demodulation = self.dft(ofdm_rx_no_cp)
        return np.concatenate((np.real(ofdm_demodulation), np.imag(ofdm_demodulation)))
    
    def generate_mixed_dataset(self, channel_types, bits_array, mode="mixed_random"):
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
        return bits.reshape((len(self.data_carriers), mu))
    
    def mapping(self, bits_sp):
        return np.array([mapping_table[tuple(b)] for b in bits_sp])
    
    def ofdm_symbol(self, qam_payload):
        symbol = np.zeros(K, dtype=complex)
        symbol[self.pilot_carriers] = pilot_value
        symbol[self.data_carriers] = qam_payload
        return symbol
    
    def idft(self, OFDM_data):
        return np.fft.ifft(OFDM_data)
    
    def add_cp(self, OFDM_time):
        cp = OFDM_time[-CP:]
        return np.hstack([cp, OFDM_time])
    
    def remove_cp(self, signals):
        return signals[CP:(CP+K)]
    
    def dft(self, signals):
        return np.fft.fft(signals)
    
    def awgn(self, signals, SNRdb):
        gamma = 10**(SNRdb/10)
        P = sum(abs(signals) ** 2) / len(signals) if signals.ndim == 1 else sum(sum(abs(signals) ** 2)) / len(signals)
        N0 = P / gamma
        n = sqrt(N0/2) * standard_normal(signals.shape) if isrealobj(signals) else sqrt(N0/2) * (standard_normal(signals.shape) + 1j * standard_normal(signals.shape))
        return signals + n

def create_tf_dataset(x_data, y_data, batch_size, buffer_size=10000, repeat=True, prefetch=True):
    """创建高效的TensorFlow数据集，减少内存压力"""
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    # 打乱数据以避免训练中的偏差
    dataset = dataset.shuffle(buffer_size=min(buffer_size, len(x_data)))
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def bit_err(y_true, y_pred):
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
    def __init__(self, input_dim, payloadBits_per_OFDM):
        super(base_models, self).__init__()
        self.input_dim = input_dim
        self.output_dim = payloadBits_per_OFDM
        
        # 使用Sequential定义模型
        self.model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dense(512, activation='relu'),
            #layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dense(payloadBits_per_OFDM, activation='sigmoid')
        ])
        
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

class DNN(base_models):
    def __init__(self, input_dim, payloadBits_per_OFDM):
        super(DNN, self).__init__(input_dim, payloadBits_per_OFDM)
        self.compile_model()

    def compile_model(self):
        self.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.MeanSquaredError(), 
            metrics=[bit_err]
        )
    
    def train(self, x_train, y_train, epochs=10, batch_size=32, 
              validation_data=None, callbacks=None, dataset_type="default"):
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
        new_model = DNN(self.input_dim, self.output_dim)
        new_model.model.set_weights(self.model.get_weights())
        return new_model

class MetaDNN(base_models):
    def __init__(self, input_dim, payloadBits_per_OFDM, inner_lr=0.01, meta_lr=0.3, mini_size=32,
                 first_decay_steps=500, t_mul=1.1, m_mul=1, alpha=0.001,
                 early_stopping=True, patience=20, min_delta=0.0002, abs_threshold=0.011, 
                 progressive_patience=True, verbose=1):
        super(MetaDNN, self).__init__(input_dim, payloadBits_per_OFDM)
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.mini_batch_size = mini_size
        self.optimizer = tf.keras.optimizers.legacy.SGD(inner_lr)
        
        self.sampling_rate = 10  # 每10次更新记录一次指标
        self.all_epoch_losses = []
        self.all_val_bit_errs = []
        self.update_counts = []
        self.total_updates = 0
        self.inner_updates = 0

        self.best_weights = None
        self.best_val_err = float('inf')
        
        # 改进的早停参数
        self.early_stopping = early_stopping
        self.patience = patience            # 减少patience (从50减少到20)
        self.min_delta = min_delta          # 增加min_delta (从0.00005增加到0.0002)
        self.abs_threshold = abs_threshold  # 新增绝对阈值 (0.011)
        self.progressive_patience = progressive_patience  # 动态调整patience
        self.verbose = verbose
        self.wait = 0  # Counter for patience
        self.stopped_epoch = 0  # The epoch at which training was stopped
        self.no_improvement_count = 0  # 连续无改善的计数
        
        self.meta_lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=meta_lr,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha
        )
    
    def get_params(self):
        return self.get_weights()
    
    def set_params(self, params):
        self.set_weights(params)
    
    def clone(self):
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
        """优化后的早停策略，考虑绝对阈值和动态patience"""
        if not self.early_stopping:
            return False
        
        # 1. 检查绝对阈值 - 如果误差已经足够低，可以直接停止
        if val_err <= self.abs_threshold:
            if self.verbose > 0:
                print(f"\nEarly stopping: reached target performance threshold {self.abs_threshold}")
            self.best_val_err = val_err
            self.best_weights = [tf.identity(w) for w in self.get_weights()]
            self.stopped_epoch = self.total_updates
            return True
            
        # 2. 检查相对改善
        if val_err < self.best_val_err - self.min_delta:
            # 验证误差有显著改善
            improvement = self.best_val_err - val_err
            self.best_val_err = val_err
            self.best_weights = [tf.identity(w) for w in self.get_weights()]
            self.wait = 0
            self.no_improvement_count = 0
            
            # 对于接近最佳性能的情况，记录日志
            if val_err < 0.015 and self.verbose > 0:
                print(f"Validation error improved to {val_err:.6f} (improvement: {improvement:.6f})")
        else:
            # 验证误差没有显著改善
            self.wait += 1
            self.no_improvement_count += 1
            
            # 动态调整patience - 当性能接近最佳时提前停止
            effective_patience = self.patience
            if self.progressive_patience and self.best_val_err < 0.015:
                # 性能已经不错，减少patience
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
        original_weights = [tf.identity(w) for w in self.get_weights()]
        losses = []

        num_samples = x_task.shape[0]
        #inner_step selection
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
        
        # 获取更新后的权重
        updated_weights = self.get_weights()
        
        # 计算权重差异
        weight_diffs = [updated - original for updated, original in zip(updated_weights, original_weights)]
        
        # 恢复原始权重
        self.set_weights(original_weights)
        
        return weight_diffs, tf.reduce_mean(losses), steps
    
    def evaluate(self, x_val, y_val):
        preds = self(x_val, training=False)
        return bit_err(y_val, preds).numpy()
    
    def train_reptile(self, tasks, meta_epochs=10, task_steps=None, meta_validation_data=None):
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
        
        # 记录初始性能
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
            
            # 应用元更新
            current_weights = self.get_weights()
            new_weights = []
            
            for i, (curr_w, grad) in enumerate(zip(current_weights, meta_grads)):
                new_weights.append(curr_w + current_meta_lr * grad / len(tasks))
            
            self.set_weights(new_weights)
            
            # 更新计数
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
                
                    # 检查早停
                    if self._should_stop_early(val_bit_err):
                        break
                    
                    # 额外的快速收敛检测 - 如果已经连续20次迭代几乎没有改善，且已经达到良好性能
                    if self.no_improvement_count > 20 and self.best_val_err < 0.013:
                        if self.verbose > 0:
                            print(f"\nStopping: no significant improvement for 20 iterations. "
                                  f"Best val_bit_err: {self.best_val_err:.6f}")
                        break

            if epoch % 100 == 0:
                import gc
                gc.collect()
            
            # epoch time
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            # print progress
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
        """在 3GPP 数据上微调"""
        for _ in range(steps):
            with tf.GradientTape() as tape:
                preds = self(x_train, training=True)
                loss = tf.keras.losses.mean_squared_error(y_train, preds)
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return self.evaluate(x_train, y_train)

if __name__ == "__main__":
    import gc
    import os
    import numpy as np
    import tensorflow as tf
    import time
    from datetime import datetime
    import sys
    
    # 导入配置系统
    from ExperimentConfig import ExperimentConfig, apply_config_to_simulator, create_meta_dnn_from_config
    
    # 创建完全集成的实验配置
    config = ExperimentConfig()
    
    # 可以在这里修改配置参数 (示例)
    # config.config["global"]["K"] = 128  # 更改子载波数量
    config.config["dataset"]["train_samples"] = 1280
    # config.config["meta_dnn"]["abs_threshold"] = 0.01
    
    # 保存配置以记录实验设置
    experiment_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_path = config.save_as_python(filepath=f"experiment_configs/ofdm_experiment_{experiment_time}.py")
    print(f"本次实验配置已保存到: {config_path}")
    
    # 导出为全局参数文件以兼容性，仅供参考
    global_params_path = config.export_to_global_parameters(
        output_file=f"experiment_configs/global_parameters_{experiment_time}.py"
    )
    
    # 注入全局参数到全局命名空间，替代导入global_parameters模块
    config.inject_globals(globals())
    
    # 设置随机种子以便复现
    seed = config.config["seed"]
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # 启用GPU内存增长
    if config.config["gpu_memory_growth"]:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU设置错误: {e}")
    
    # 获取数据集和训练参数
    dataset_params = config.get_dataset_params()
    dnn_params = config.get_dnn_params()
    output_params = config.get_output_params()
    
    # 设置数据参数
    DNN_samples = dataset_params["train_samples"]
    DNN_epoch = dnn_params["epochs"]
    DNN_batch_size = dnn_params["batch_size"]
    channel_types = dataset_params["channel_types"]
    meta_channel_types = dataset_params["meta_channel_types"]
    
    # 确保输出目录存在
    log_dir = output_params.get("log_dir", "experiment_logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建信号模拟器实例
    simulator = signal_simulator()
    # 全部参数已通过全局变量注入，只需应用SNR
    simulator.SNRdB = config.config["signal"]["SNR"]
    
    # 初始化模型和历史记录容器
    models = {}
    histories = {}
    
    # 生成训练数据
    print(f"生成 {DNN_samples} 个训练样本...")
    bits = simulator.generate_bits(DNN_samples)
    MultiModelBCP.clear_data()
    
    # 训练阶段 - 标准DNN
    print("=== Train Phase ===")
    for channel in channel_types:
        print(f"\n训练于 {channel} 通道...")
        
        # 生成训练数据
        start_time = time.time()
        x_train, y_train = simulator.generate_training_dataset(channel, bits)
        x_test, y_test = simulator.generate_testing_dataset(channel, dataset_params["test_samples"] // 5)
        print(f"数据生成耗时: {time.time() - start_time:.2f}秒")
        
        # 创建模型
        model_name = f"DNN_{channel}"
        models[model_name] = DNN(
            input_dim=x_train.shape[1],
            payloadBits_per_OFDM=simulator.payloadBits_per_OFDM
        )
        
        # 转换为TensorFlow数据集
        train_dataset = create_tf_dataset(
            x_train, y_train, 
            batch_size=DNN_batch_size,
            repeat=False,
            buffer_size=min(5000, len(x_train))
        )
        
        # 使用内存高效回调
        callback = MultiModelBCP(
            model_name=model_name, 
            dataset_type=channel,
            sampling_rate=output_params.get("sampling_rate", 10),
            max_points=output_params.get("max_points", 1000)
        )
        
        # 训练模型
        start_time = time.time()
        histories[model_name] = models[model_name].fit(
            train_dataset,
            epochs=DNN_epoch, 
            validation_data=(x_test, y_test),
            callbacks=[callback],
            verbose=1
        )
        print(f"模型训练耗时: {time.time() - start_time:.2f}秒")
        
        # 手动释放内存
        del x_train, y_train, train_dataset
        gc.collect()
    
    # 元学习阶段
    print("\n=== 元学习阶段 ===")
    meta_tasks = []
    meta_model_name = "Meta_DNN"

    # 计算元更新次数，与DNN保持一致
    total_meta_iteration = int((DNN_samples/DNN_batch_size)*DNN_epoch)
    print(f"元更新次数: {total_meta_iteration}")
    
    # 为每种通道生成元学习任务
    print("生成元学习任务...")
    start_time = time.time()
    for channel in meta_channel_types:
        channel_bits = simulator.generate_bits(DNN_samples)
        x_task, y_task = simulator.generate_training_dataset(channel, channel_bits)
        meta_tasks.append((x_task, y_task))
        # 立即释放内存
        gc.collect()
    print(f"元学习任务生成耗时: {time.time() - start_time:.2f}秒")
    
    # 创建元学习模型
    models[meta_model_name] = create_meta_dnn_from_config(
        input_dim=meta_tasks[0][0].shape[1],
        payloadBits_per_OFDM=simulator.payloadBits_per_OFDM,
        config=config
    )
    
    # 创建验证集
    meta_x_test, meta_y_test = simulator.generate_testing_dataset(
        "random_mixed", 
        dataset_params["test_samples"] // 5
    )
    
    # 训练元模型
    print("开始元学习训练...")
    start_time = time.time()
    losses, val_errs, update_counts = models["Meta_DNN"].train_reptile(
        meta_tasks, 
        meta_epochs=total_meta_iteration, 
        meta_validation_data=(meta_x_test, meta_y_test),
        task_steps=config.config["meta_dnn"]["task_steps"]
    )
    print(f"元学习训练耗时: {time.time() - start_time:.2f}秒")
    
    # 记录元模型数据
    MultiModelBCP.log_manual_data(
        "Meta_DNN_train",
        losses,
        val_errs,
        update_counts=update_counts,
        dataset_type="meta"
    )
    
    # 绘制更新对比图
    plot_file = os.path.join(log_dir, f"update_comparison_{experiment_time}.png")
    dpi = output_params.get("plot_dpi", 300)
    MultiModelBCP.plot_by_updates(save_path=plot_file, dpi=dpi)
    
    # 释放不再需要的模型和数据
    del losses, val_errs, update_counts, meta_x_test, meta_y_test, meta_tasks
    gc.collect()
    
    # 3GPP泛化测试阶段
    print("\n=== 3GPP泛化测试阶段 ===")
    MultiModelBCP.clear_data()
    
    # 获取微调参数
    fine_tuning_params = config.get_fine_tuning_params()
    val_parameter_set = []
    for size in dataset_params["fine_tuning_sizes"]:
        batch_size = fine_tuning_params["batch_sizes"].get(str(size), 32)
        val_parameter_set.append((size, batch_size))
    
    # 创建验证集
    x_3gpp_val, y_3gpp_val = simulator.generate_testing_dataset(
        dataset_params["test_channel"], 
        dataset_params["test_samples"] // 8
    )
    val_epoch = fine_tuning_params["epochs"]
    
    for size, val_batch_size in val_parameter_set:
        DNN_num_update = int((size/val_batch_size)*val_epoch)
        print(f"{size}样本集，更新次数: {DNN_num_update}")
        
        # 生成3GPP训练数据
        bits_array = simulator.generate_bits(size)
        x_3gpp_train, y_3gpp_train = simulator.generate_training_dataset(
            dataset_params["test_channel"], 
            bits_array
        )
        
        # 转换为TensorFlow数据集
        train_dataset = create_tf_dataset(
            x_3gpp_train, y_3gpp_train, 
            batch_size=val_batch_size,
            repeat=False
        )
        
        # 传统DNN微调
        for channel in channel_types:
            model_name = f"DNN_{channel}_3GPP_{size}"
            dnn_model = models[f"DNN_{channel}"].clone()
            print(f"\n验证 {channel} 通道，样本量: {size}")
            
            callback = MultiModelBCP(
                model_name=model_name, 
                dataset_type=f"{channel}_{size}",
                sampling_rate=output_params.get("sampling_rate", 10),
                max_points=output_params.get("max_points", 1000)
            )
            
            dnn_model.fit(
                train_dataset,
                epochs=val_epoch,
                validation_data=(x_3gpp_val, y_3gpp_val),
                callbacks=[callback],
                verbose=1
            )
            
            # 释放模型内存
            del dnn_model
            gc.collect()
        
        # Meta DNN微调
        meta_task_3gpp = [(x_3gpp_train, y_3gpp_train)]
        meta_model = models["Meta_DNN"].clone()
        meta_model_name = f"Meta_{size}"
        
        # 根据数据集大小调整内部步数
        task_steps = min(config.config["meta_dnn"]["task_steps"], size//5)
        
        losses, val_errs, update_counts = meta_model.train_reptile(
            meta_task_3gpp, 
            meta_epochs=DNN_num_update, 
            meta_validation_data=(x_3gpp_val, y_3gpp_val),
            task_steps=task_steps
        )
        
        MultiModelBCP.log_manual_data(
            meta_model_name,
            losses,
            val_errs,
            update_counts=update_counts,
            dataset_type=f"3gpp_{size}"
        )
        
        # 释放元模型和数据
        del meta_model, losses, val_errs, update_counts, meta_task_3gpp
        del x_3gpp_train, y_3gpp_train, train_dataset
        gc.collect()
    
    # 绘制最终结果
    final_plot_file = os.path.join(log_dir, f"3gpp_generalization_test_{experiment_time}.png")
    MultiModelBCP.plot_by_updates(save_path=final_plot_file, dpi=dpi)

    output_dir = f"matlab_exports/experiment_{experiment_time}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 导出模型数据
    print("\n=== 导出模型数据到MATLAB ===")
    exported_files = MultiModelBCP.export_data_for_matlab(
        output_dir=output_dir,
        format="mat",
        prefix=f"ofdm_models_{experiment_time}_"
    )

    # 生成MATLAB分析脚本
    matlab_script = MultiModelBCP.generate_matlab_script(
        output_dir=output_dir,
        exported_files=exported_files,
        prefix=f"ofdm_models_{experiment_time}_",
        sample_sizes=dataset_params["fine_tuning_sizes"]
    )

    print(f"\n数据导出完成! 结果保存在: {output_dir}")
    
    # 释放所有模型和回调数据
    models.clear()
    MultiModelBCP.clear_data()
    gc.collect()
    
    print(f"\n实验完成! 结果已保存到 {log_dir} 目录")
    print(f"配置文件: {config_path}")
    print(f"全局参数文件: {global_params_path}")
    print(f"更新对比图: {plot_file}")
    print(f"泛化测试图: {final_plot_file}")