import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from global_parameters import *
import numpy as np
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
import tensorflow as tf
from tensorflow import keras
Model = keras.Model
layers = keras.layers
Callback = keras.callbacks.Callback
import matplotlib.pyplot as plt

class BCP(Callback):
    def __init__(self):
        super(BCP, self).__init__()
        self.batch_accuracy = []  # 改为实例变量
        self.batch_loss = []       # 改为实例变量
        self.epoch_loss = []      # 新增epoch级记录
        self.epoch_bit_err = []   # 新增epoch级记录
    
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch_accuracy.append(logs.get('bit_err', 0))
        self.batch_loss.append(logs.get('loss', 0))
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_loss.append(logs.get('loss', 0))
        self.epoch_bit_err.append(logs.get('bit_err', 0))
    
    def on_train_end(self, logs=None):
        self.plot_learning_curve()
    
    def plot_learning_curve(self):
        # 创建画布
        plt.figure(figsize=(12, 5))
        
        # Batch级别的曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.batch_loss, label='Training Loss')
        plt.plot(np.linspace(0, len(self.batch_loss), len(self.epoch_bit_err)), self.epoch_bit_err, 
                label='Bit Error Rate', color='red', alpha=0.6)
        plt.title('Batch-Level Learning Curve')
        plt.xlabel('Batch Index')
        plt.ylabel('Metric Value')
        plt.legend()
        
        # Epoch级别的曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_loss, label='Training Loss')
        plt.plot(self.epoch_bit_err, label='Bit Error Rate')
        plt.title('Epoch-Level Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Average Metric')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('learning_curve.png')  # 保存图像
        plt.close()
        print("\nLearning curve saved to learning_curve.png")

class signal_simulator():
    def __init__(self, SNR=10):
        self.all_carriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
        self.pilot_carriers = self.all_carriers[::K // P]  # Pilots is every (K/P)th carrier.
        self.data_carriers = np.delete(self.all_carriers, self.pilot_carriers)
        self.payloadBits_per_OFDM = len(self.data_carriers) * mu  # number of payload bits per OFDM symbol
        #self.channel_3gpp = np.load('channel_train.npy')
        self.SNRdB = SNR
    
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
            mu = np.sqrt(k / (k + 1))
            s = np.sqrt(1 / (2 * (k + 1)))
            channel = mu + s * (np.random.randn(1, num_path) + 1j * np.random.randn(1, num_path))
            channel = channel[:, 0]
        elif self.channel_type == "awgn":
            return self.awgn(transmit_signals, self.SNRdB)
        elif self.channel_type == "3gpp":
            train_size = self.channel_3gpp.shape[0]
            index = np.random.choice(np.arange(train_size), size=1)
            h = self.channel_3gpp[index]
            channel = h[:, 0]
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
    
    def generate_training_dataset(self, channel_type, bits_array):
        training_sample = []

        for bits in bits_array:  # 直接迭代 bit 数组
            ofdm_simulate_output = self.ofdm_simulate(bits,channel_type)
            training_sample.append(ofdm_simulate_output)
        
        return np.asarray(training_sample), bits_array
    
    def generate_testing_dataset(self, channel_type, num_samples):
        bits_array = self.generate_bits(num_samples)
        testing_sample = []

        for bits in bits_array:  # 直接迭代 bit 数组
            ofdm_simulate_output = self.ofdm_simulate(bits,channel_type)
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

def bit_err(y_true, y_pred):
    err = 1 - tf.reduce_mean(
        tf.reduce_mean(
            tf.cast(  # 用 tf.cast() 替换 tf.compat.v1.to_float()
                tf.equal(
                    tf.sign(y_pred - 0.5),
                    tf.cast(tf.sign(y_true - 0.5), tf.float32)
                ), tf.float32
            ), axis=1
        )
    )
    return err


class base_models(Model):
    def __init__(self,input_dim,payloadBits_per_OFDM):
        super(base_models, self).__init__()

        self.layer1 = layers.Dense(256, activation='relu', input_shape=(input_dim,))
        self.layer2 = layers.Dense(512, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.layer3 = layers.Dense(256, activation='relu')
        self.output_layer = layers.Dense(payloadBits_per_OFDM, activation='tanh')
        
    def call(self, inputs, training=False):  # 确保 Dropout 层生效
        x = self.layer1(inputs) 
        x = self.layer2(x)
        x = self.dropout(x, training=training)
        x = self.layer3(x)
        return self.output_layer(x)

class DNN(base_models):
    def __init__(self, input_dim,payloadBits_per_OFDM):
        super(DNN, self).__init__(input_dim,payloadBits_per_OFDM)
        self.compile_model()


    def compile_model(self):
        """模型编译"""
        self.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.MeanSquaredError(), 
            metrics=[bit_err]
        )
    
    def train(self, x_train, y_train, epochs=10, batch_size=32, 
             validation_data=None, callbacks=None):
        
        # 设置默认回调（保留原有回调功能）
        final_callbacks = [BCP()]  # 默认包含我们的回调
        if callbacks:
            if isinstance(callbacks, list):
                final_callbacks.extend(callbacks)
            else:
                final_callbacks.append(callbacks)
        
        return self.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=final_callbacks,
            verbose=1  # 显示进度条
        )

if __name__ == "__main__":
        # 初始化信号模拟器
    simulator = signal_simulator()

    bits = simulator.generate_bits(100000)
    # 生成数据集
    x_train, y_train = simulator.generate_training_dataset("awgn",bits) #channel_type, bits
    x_test, y_test = simulator.generate_testing_dataset("awgn", 2500) #channel_type, num_samples

    # 创建并训练模型
    model = DNN(input_dim=x_train.shape[1], payloadBits_per_OFDM = simulator.payloadBits_per_OFDM)
    history = model.train(
        x_train, 
        y_train,
        epochs=4,
        batch_size=32,
        validation_data=(x_test, y_test)
    )