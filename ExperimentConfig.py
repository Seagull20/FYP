import os
import datetime
import pprint
import json
import numpy as np
import importlib.util
import importlib
from singal_detection_withOFDM import MetaDNN

class ExperimentConfig:
    """OFDM信号检测实验的配置管理类 - 完全集成版"""
    
    all_models_data = {}  # 与MultiModelBCP兼容
    
    def __init__(self, global_params_file="global_parameters.py"):
        """
        初始化配置对象，将全局参数完全集成到配置中
        
        参数:
            global_params_file: 全局参数文件路径
        """
        # 加载全局参数
        self.global_params = self._load_global_params(global_params_file)
        
        # 初始化完全集成的配置
        self.config = {
            # 全局参数设置
            "global": {
                "K": self.global_params.get("K", 64),                      # 子载波数量
                "CP": self.global_params.get("CP", 16),                    # 循环前缀长度
                "P": self.global_params.get("P", 8),                       # 导频载波数量
                "pilot_value": self._complex_to_dict(self.global_params.get("pilot_value", 1+1j)),  # 导频值
                "mu": self.global_params.get("mu", 2),                     # 每符号比特数
                "mapping_table": self._convert_mapping_table(self.global_params.get("mapping_table", {})),  # 映射表
                "demapping_table": None,                                   # 反映射表（自动计算）
                "num_path": self.global_params.get("num_path", 16),        # 信道路径数
                "rician_factor": self.global_params.get("rician_factor", 1),  # 莱斯因子
                "num_simulate": self.global_params.get("num_simulate", 50000),  # 模拟数量
                "num_simulate_target": self.global_params.get("num_simulate_target", 5000),  # 目标模拟数量
                "num_test_target": self.global_params.get("num_test_target", 1000),  # 目标测试数量
                "num_running": self.global_params.get("num_running", 1)    # 运行次数
            },
            
            # 信号和信道参数
            "signal": {
                "SNR": 10,                                                # 信噪比(dB)
            },
            
            # 数据集配置
            "dataset": {
                "train_samples": 64000,                                   # 训练样本数量
                "test_samples": 2500,                                     # 测试数据集大小
                "val_samples": 300,                                     # 3GPP验证数据集大小
                "channel_types": ["awgn", "rician", "rayleigh", "random_mixed"], # 标准训练信道
                "meta_channel_types": ["awgn", "rician", "rayleigh"],      # 元学习信道
                "test_channel": "3gpp",                                   # 泛化测试信道
                "fine_tuning_sizes": [50, 100, 500, 1000]                 # 微调样本大小
            },
            
            # DNN模型参数
            "dnn": {
                "architecture": [256, 512, 256],                          # 隐藏层大小
                "activations": ["relu", "relu", "relu", "sigmoid"],       # 激活函数
                "optimizer": "adam",                                      # 优化器
                "learning_rate": self.global_params.get("learning_rate", 0.001),  # 学习率
                "loss": "mse",                                           # 损失函数
                "epochs": 10,                                             # 训练周期数
                "batch_size": self.global_params.get("batch_size", 32)    # 批量大小
            },
            
            # 元学习参数
            "meta_dnn": {
                "inner_lr": 0.02,                                         # 内循环学习率
                "meta_lr": 0.3,                                           # 元学习率
                "mini_batch_size": 32,                                    # 小批量大小
                "task_steps": 50,                                         # 任务步骤数
                "early_stopping": True,                                   # 是否启用早停
                "patience": 20,                                           # 早停耐心值
                "min_delta": 0.0002,                                      # 最小改进阈值
                "abs_threshold": 0.011,                                   # 绝对阈值
                "progressive_patience": True,                             # 动态耐心
                "verbose": 1,                                             # 详细程度
                "lr_schedule": {
                    "first_decay_steps": 500,                             # 首次衰减步骤
                    "t_mul": 1.1,                                         # t乘数
                    "m_mul": 1,                                           # m乘数
                    "alpha": 0.001                                        # alpha值
                }
            },
            
            # 微调参数
            "fine_tuning": {
                "epochs": 1,                                              # 微调周期数
                "batch_sizes": {                                          # 不同样本量的批量大小
                    "50": 5,
                    "100": 5, 
                    "500": 16,
                    "1000": 32
                }
            },
            
            # 随机种子，用于可复现性
            "seed": 42,
            
            # 计算参数
            "gpu_memory_growth": True,                                    # 是否启用GPU内存增长
            
            # 输出配置
            "output": {
                "save_plots": True,                                       # 是否保存图表
                "plot_dpi": 300,                                          # 图表DPI
                "log_dir": "experiment_logs",                             # 日志目录
                "sampling_rate": 10,                                      # 指标采样率
                "max_points": 1000                                        # 最大记录点数
            }
        }
        
        # 计算反映射表
        self._update_demapping_table()
    
    def _complex_to_dict(self, complex_val):
        """将复数转换为字典表示"""
        if isinstance(complex_val, complex):
            return {"real": complex_val.real, "imag": complex_val.imag}
        return complex_val
    
    def _dict_to_complex(self, dict_val):
        """将字典表示转换为复数"""
        if isinstance(dict_val, dict) and "real" in dict_val and "imag" in dict_val:
            return complex(dict_val["real"], dict_val["imag"])
        return dict_val
    
    def _convert_mapping_table(self, mapping_table):
        """转换映射表为可序列化格式"""
        result = {}
        for k, v in mapping_table.items():
            # 将元组键转换为字符串
            str_key = str(k)
            # 将复数值转换为字典
            result[str_key] = self._complex_to_dict(v)
        return result
    
    def _restore_mapping_table(self, mapping_dict):
        """还原映射表为原始格式"""
        result = {}
        for k, v in mapping_dict.items():
            # 将字符串键转换回元组
            if k.startswith('(') and k.endswith(')'):
                # 解析如 "(0, 1)" 格式的字符串
                try:
                    tuple_key = eval(k)
                except:
                    # 如果解析失败，使用原始键
                    tuple_key = k
            else:
                tuple_key = k
            
            # 将字典值转换回复数
            result[tuple_key] = self._dict_to_complex(v)
        return result
    
    def _update_demapping_table(self):
        """更新反映射表"""
        mapping_table = self._restore_mapping_table(self.config["global"]["mapping_table"])
        demapping_table = {v: k for k, v in mapping_table.items()}
        self.config["global"]["demapping_table"] = self._convert_mapping_table(demapping_table)
    
    def _load_global_params(self, filepath):
        """从全局参数文件加载参数"""
        if not os.path.exists(filepath):
            print(f"警告: 全局参数文件 {filepath} 不存在，使用默认值")
            return {}
        
        try:
            # 从Python文件中加载变量
            spec = importlib.util.spec_from_file_location("global_params", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 提取模块中的所有变量
            params = {name: getattr(module, name) for name in dir(module) 
                     if not name.startswith('__') and not callable(getattr(module, name))}
            
            return params
        except Exception as e:
            print(f"加载全局参数文件时出错: {str(e)}")
            return {}
    
    def update_from_dict(self, config_dict):
        """从字典更新配置"""
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_nested_dict(d[k], v)
                else:
                    d[k] = v
        
        update_nested_dict(self.config, config_dict)
        # 更新反映射表
        self._update_demapping_table()
    
    def get_simulator_params(self):
        """获取信号模拟器参数"""
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
        """获取DNN参数"""
        return self.config["dnn"]
    
    def get_meta_dnn_params(self):
        """获取MetaDNN参数"""
        return self.config["meta_dnn"]
    
    def get_dataset_params(self):
        """获取数据集参数"""
        return self.config["dataset"]
    
    def get_fine_tuning_params(self):
        """获取微调参数"""
        return self.config["fine_tuning"]
    
    def get_output_params(self):
        """获取输出参数"""
        return self.config["output"]
    
    def create_global_module(self):
        """
        创建全局参数模块，与原始 global_parameters.py 兼容
        
        返回:
            包含全局参数的模块对象
        """
        # 创建一个空的模块
        module = type('GlobalParameters', (), {})
        
        # 添加基本参数
        for key, value in self.config["global"].items():
            if key == "mapping_table":
                # 特殊处理映射表
                mapping_table = self._restore_mapping_table(value)
                setattr(module, key, mapping_table)
            elif key == "demapping_table":
                # 特殊处理反映射表
                if value:
                    demapping_table = self._restore_mapping_table(value)
                    setattr(module, key, demapping_table)
            elif key == "pilot_value":
                # 特殊处理导频值
                setattr(module, key, self._dict_to_complex(value))
            else:
                setattr(module, key, value)
        
        # 添加信号参数
        for key, value in self.config["signal"].items():
            setattr(module, "SNRdb", value)  # 使用原始名称
        
        # 添加其他必要参数
        setattr(module, "learning_rate", self.config["dnn"]["learning_rate"])
        setattr(module, "batch_size", self.config["dnn"]["batch_size"])
        
        return module
    
    def export_to_global_parameters(self, output_file="new_global_parameters.py"):
        """
        将配置导出为与 global_parameters.py 兼容的格式
        
        参数:
            output_file: 输出文件路径
        
        返回:
            导出的文件路径
        """
        try:
            with open(output_file, 'w') as f:
                # 写入全局参数
                for key, value in self.config["global"].items():
                    if key == "mapping_table":
                        # 特殊处理映射表
                        mapping_table = self._restore_mapping_table(value)
                        f.write(f"{key} = ")
                        f.write(pprint.pformat(mapping_table, indent=4))
                        f.write("\n\n")
                    elif key == "demapping_table":
                        # 跳过反映射表，稍后计算
                        continue
                    elif key == "pilot_value":
                        # 特殊处理导频值
                        complex_val = self._dict_to_complex(value)
                        f.write(f"{key} = {complex_val!r}\n")
                    else:
                        f.write(f"{key} = {value!r}\n")
                
                # 写入反映射表
                f.write("\n# 从映射表自动计算反映射表\n")
                f.write("demapping_table = {v: k for k, v in mapping_table.items()}\n")
                
                # 写入其他必要参数
                f.write(f"\nSNRdb = {self.config['signal']['SNR']}\n")
                f.write(f"learning_rate = {self.config['dnn']['learning_rate']}\n")
                f.write(f"batch_size = {self.config['dnn']['batch_size']}\n")
            
            print(f"全局参数已导出到: {output_file}")
            return output_file
        
        except Exception as e:
            print(f"导出全局参数时出错: {str(e)}")
            return None
    
    def inject_globals(self, target_module=None):
        """
        将配置中的全局参数注入到模块或全局命名空间
        
        参数:
            target_module: 目标模块，如果为None则使用globals()
        """
        if target_module is None:
            target_module = globals()
        
        module = self.create_global_module()
        
        # 将所有属性注入到目标命名空间
        for name in dir(module):
            if not name.startswith('__'):
                value = getattr(module, name)
                if isinstance(target_module, dict):
                    target_module[name] = value
                else:
                    setattr(target_module, name, value)
    
    def save_to_file(self, filepath=None, add_timestamp=True):
        """
        将配置保存到文件
        
        参数:
            filepath: 文件路径，如果为None则使用默认路径
            add_timestamp: 是否在文件名中添加时间戳
        
        返回:
            保存的文件路径
        """
        try:
            # 确定文件保存路径
            if filepath is None:
                # 默认在当前目录创建experiment_configs文件夹
                config_dir = "experiment_configs"
                if not os.path.exists(config_dir):
                    os.makedirs(config_dir)
                
                # 添加时间戳
                timestamp = ""
                if add_timestamp:
                    timestamp = f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                filepath = os.path.join(config_dir, f"ofdm_config{timestamp}.json")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # 保存为JSON
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            print(f"配置已保存到: {filepath}")
            return filepath
        
        except Exception as e:
            print(f"保存配置文件时出错: {str(e)}")
            return None
    
    def save_as_python(self, filepath=None, add_timestamp=True):
        """将配置保存为Python文件"""
        try:
            # 确定文件保存路径
            if filepath is None:
                # 默认在当前目录创建experiment_configs文件夹
                config_dir = "experiment_configs"
                if not os.path.exists(config_dir):
                    os.makedirs(config_dir)
                
                # 添加时间戳
                timestamp = ""
                if add_timestamp:
                    timestamp = f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                filepath = os.path.join(config_dir, f"ofdm_config{timestamp}.py")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # 写入Python格式
            with open(filepath, 'w') as f:
                f.write("# OFDM信号检测实验配置\n")
                f.write("# 生成时间: {}\n\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                f.write("experiment_config = ")
                # 使用pprint格式化配置字典
                formatted_config = pprint.pformat(self.config, indent=4, width=100)
                f.write(formatted_config)
            
            print(f"配置已保存到: {filepath}")
            return filepath
        
        except Exception as e:
            print(f"保存配置文件时出错: {str(e)}")
            return None
    
    @classmethod
    def load_from_file(cls, filepath):
        """
        从文件加载配置
        
        参数:
            filepath: 配置文件路径
        
        返回:
            配置对象
        """
        try:
            config_obj = cls()  # 创建默认配置对象
            
            # 判断文件类型并加载
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    loaded_config = json.load(f)
                
                config_obj.update_from_dict(loaded_config)
            
            elif filepath.endswith('.py'):
                # 从Python文件加载
                namespace = {}
                with open(filepath, 'r') as f:
                    exec(f.read(), {}, namespace)
                
                if 'experiment_config' in namespace:
                    config_obj.update_from_dict(namespace['experiment_config'])
                else:
                    raise ValueError("Python配置文件中找不到experiment_config变量")
            
            else:
                raise ValueError(f"不支持的文件类型: {filepath}")
            
            print(f"配置已从 {filepath} 加载")
            return config_obj
        
        except Exception as e:
            print(f"加载配置文件时出错: {str(e)}")
            return None
    
    def __str__(self):
        """配置的字符串表示"""
        return pprint.pformat(self.config, indent=4)

# 应用实例
def apply_config_to_simulator(simulator, config):
    """应用配置到信号模拟器"""
    signal_params = config.get_simulator_params()
    simulator.SNRdB = signal_params.get("SNR", 10)
    # 其他参数通过全局注入方式应用
    return simulator

def create_meta_dnn_from_config(input_dim, payloadBits_per_OFDM, config):
    """从配置创建MetaDNN模型"""
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

# 使用示例
if __name__ == "__main__":
    # 创建默认配置
    config = ExperimentConfig()
    
    # 修改配置
    config.config["global"]["K"] = 128  # 更改子载波数量
    config.config["meta_dnn"]["abs_threshold"] = 0.01
    
    # 保存配置
    config.save_to_file()
    
    # 导出为全局参数文件
    config.export_to_global_parameters()
    
    # 模拟全局参数注入
    test_dict = {}
    config.inject_globals(test_dict)
    print(f"注入后的K值: {test_dict['K']}")
    
    # 输出配置
    print("\n配置内容:")
    print(config)