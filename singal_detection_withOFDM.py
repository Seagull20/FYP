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

# 改进后的 MultiModelBCP 类
class MultiModelBCP(Callback):
    def __init__(self, model_name, dataset_type="default"):
        super(MultiModelBCP, self).__init__()
        self.model_name = model_name  # 模型唯一标识符
        self.dataset_type = dataset_type
        if not hasattr(MultiModelBCP, "all_models_data"):
            MultiModelBCP.all_models_data = {}
        self.batch_loss = []
        self.batch_bit_err = []
        self.epoch_loss = []
        self.epoch_bit_err = []
        self.val_epoch_loss = []
        self.val_epoch_bit_err = []

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch_loss.append(logs.get('loss', 0))
        self.batch_bit_err.append(logs.get('bit_err', 0))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_loss.append(logs.get('loss', 0))
        self.epoch_bit_err.append(logs.get('bit_err', 0))
        self.val_epoch_loss.append(logs.get('val_loss', 0))
        self.val_epoch_bit_err.append(logs.get('val_bit_err', 0))

    def on_train_end(self, logs=None):
        MultiModelBCP.all_models_data[self.model_name] = {
            "batch_loss": self.batch_loss,
            "batch_bit_err": self.batch_bit_err,
            "epoch_loss": self.epoch_loss,
            "epoch_bit_err": self.epoch_bit_err,
            "val_epoch_loss": self.val_epoch_loss,
            "val_epoch_bit_err": self.val_epoch_bit_err,
            "dataset_type": self.dataset_type
        }

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
        # Batch-Level Curve
        if plot_batch:
            plt.subplot(1, num_plots, plot_idx)
            for model_name, data in MultiModelBCP.all_models_data.items():
                plt.plot(data["batch_loss"], label=f"{model_name} Loss")
                plt.plot(np.linspace(0, len(data["batch_loss"]), len(data["epoch_bit_err"])),
                         data["epoch_bit_err"], label=f"{model_name} Bit Err", alpha=0.6)
            plt.title("Batch-Level Learning Curves")
            plt.xlabel("Batch Index")
            plt.ylabel("Metric Value")
            plt.legend()
            plot_idx += 1

        # Simplified Epoch-Level Curve
        if plot_epoch:
            plt.subplot(1, num_plots, plot_idx)
            for model_name, data in MultiModelBCP.all_models_data.items():
                plt.plot(data["val_epoch_bit_err"], label=f"{model_name} Val Bit Err")
                if plot_train_bit_err:
                    plt.plot(data["epoch_bit_err"], label=f"{model_name} Bit Err", linestyle="--")
            plt.title("Epoch-Level Learning Curves")
            plt.xlabel("Epoch")
            plt.ylabel("Bit Error Rate")
            plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Multi-model learning curves saved to {save_path}")

    @staticmethod
    def clear_data():
        MultiModelBCP.all_models_data = {}

# signal_simulator 类（保持不变）
class signal_simulator():
    def __init__(self, SNR=10):
        self.all_carriers = np.arange(K)
        self.pilot_carriers = self.all_carriers[::K // P]
        self.data_carriers = np.delete(self.all_carriers, self.pilot_carriers)
        self.payloadBits_per_OFDM = len(self.data_carriers) * mu
        self.channel_3gpp = np.load('channel_train.npy')
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

    def generate_training_dataset(self, channel_type, bits_array, mode="mixed_sequential"):
        if isinstance(channel_type, list):
            return self.generate_mixed_dataset(channel_type, bits_array, mode=mode)
        
        training_sample = []
        for bits in bits_array:
            ofdm_simulate_output = self.ofdm_simulate(bits, channel_type)
            training_sample.append(ofdm_simulate_output)
        
        return np.asarray(training_sample), bits_array
    
    def generate_testing_dataset(self, channel_type, num_samples, mode="mixed_sequential"):
        bits_array = self.generate_bits(num_samples)
        if isinstance(channel_type, list):
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
        self.layer1 = layers.Dense(256, activation='relu', input_shape=(input_dim,))
        self.layer2 = layers.Dense(512, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.layer3 = layers.Dense(256, activation='relu')
        self.output_layer = layers.Dense(payloadBits_per_OFDM, activation='sigmoid')
        
    def call(self, inputs, training=False):
        x = self.layer1(inputs) 
        x = self.layer2(x)
        x = self.dropout(x, training=training)
        x = self.layer3(x)
        return self.output_layer(x)

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
            verbose=1
        )

if __name__ == "__main__":
    simulator = signal_simulator()
    channel_types = ["rician", "awgn", "rayleigh"]
    models = {}
    histories = {}

    bits = simulator.generate_bits(100000)
    MultiModelBCP.clear_data()

    for channel in channel_types:
        x_train, y_train = simulator.generate_training_dataset(channel, bits)
        x_test, y_test = simulator.generate_testing_dataset(channel, 2500)

        model_name = f"DNN_{channel}"
        models[model_name] = DNN(
            input_dim=x_train.shape[1],
            payloadBits_per_OFDM=simulator.payloadBits_per_OFDM
        )
        histories[model_name] = models[model_name].train(
            x_train,
            y_train,
            epochs=10,
            batch_size=32,
            validation_data=(x_test, y_test),
            dataset_type=channel
        )

    MultiModelBCP.plot_all_learning_curves(
        save_path="multi_channel_comparison.png",
        plot_batch=True,
        plot_epoch=True,
        plot_train_bit_err=False  # 默认不绘制训练误比特率
    )