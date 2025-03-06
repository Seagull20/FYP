from enum import auto
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from global_parameters import *  # 假设包含 K, P, mu, CP, num_path, rician_factor, pilot_value, mapping_table 等
import numpy as np
from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal
import tensorflow as tf
from tensorflow import keras
Model = keras.Model
layers = keras.layers
Callback = keras.callbacks.Callback
import matplotlib.pyplot as plt
import math
import time


from playsound import playsound

# 改进后的 MultiModelBCP 类
class MultiModelBCP(Callback):
    def __init__(self, model_name, dataset_type="default"):
        super(MultiModelBCP, self).__init__()
        self.model_name = model_name
        self.dataset_type = dataset_type
        if not hasattr(MultiModelBCP, "all_models_data"):
            MultiModelBCP.all_models_data = {}
        
        # Traditional metrics lists
        self.batch_loss = []
        self.batch_bit_err = []
        self.epoch_loss = []
        self.epoch_bit_err = []
        self.val_epoch_loss = []
        self.val_epoch_bit_err = []
        
        # Update-based tracking
        self.update_counts = []  # Stores the update count at each recording point
        self.metrics_by_updates = {
            "loss": [],
            "bit_err": [],
            "val_loss": [],
            "val_bit_err": [],
            "val_update_counts":[]
        }
        self.current_update_count = 0
        # self.val_update_counts = []
        # self.val_bit_err_by_updates = []
        # self.val_loss_by_updates = []

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.current_update_count += 1  # Increment update counter
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
        
        # Record validation metrics with current update count
        last_update_count = self.current_update_count
        self.metrics_by_updates["val_loss"].append(logs.get('val_loss', 0))
        self.metrics_by_updates["val_bit_err"].append(logs.get('val_bit_err', 0))
        self.metrics_by_updates["val_update_counts"].append(last_update_count)

    def on_train_begin(self, logs=None):
        self.metrics_by_updates["val_loss"].append(0.5)
        self.metrics_by_updates["val_bit_err"].append(0.5)
        self.metrics_by_updates["val_update_counts"].append(0)


    def on_train_end(self, logs=None):
        MultiModelBCP.all_models_data[self.model_name] = {
            # Traditional data
            "batch_loss": self.batch_loss,
            "batch_bit_err": self.batch_bit_err,
            "epoch_loss": self.epoch_loss,
            "epoch_bit_err": self.epoch_bit_err,
            "val_epoch_loss": self.val_epoch_loss,
            "val_epoch_bit_err": self.val_epoch_bit_err,
            # Update-based tracking
            "update_counts": self.update_counts,
            "metrics_by_updates": self.metrics_by_updates,
            "final_update_count": self.current_update_count,
            "dataset_type": self.dataset_type
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
    def plot_by_updates(save_path="update_comparison.png", count_method="algorithmic"):
        """
        Plot learning curves using parameter updates as x-axis
        
        Args:
            save_path: Path to save the plot
            count_method: How to count updates - 'algorithmic' (default) or 'computational'
        """
        if not MultiModelBCP.all_models_data:
            print("No data to plot.")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Create two plot versions
        plt.subplot(2, 1, 1)
        for model_name, data in MultiModelBCP.all_models_data.items():
            if "update_counts" in data and "metrics_by_updates" in data:
                update_counts = data["update_counts"]
                loss_values = data["metrics_by_updates"]["loss"]
                
                # Only plot if we have data points
                if update_counts and loss_values:
                    plt.plot(update_counts[:len(loss_values)], loss_values, 
                            label=f"{model_name} Loss", marker='o', markersize=3)
        
        plt.xlabel("Number of Parameter Updates")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Parameter Updates")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc="best")
        
        # Similar for validation metrics plot
        plt.subplot(2, 1, 2)
        for model_name, data in MultiModelBCP.all_models_data.items():
            if "update_counts" in data and "metrics_by_updates" in data:
                update_counts = data["metrics_by_updates"]["val_update_counts"]
                bit_errs = data["metrics_by_updates"]["val_bit_err"]
                
                # Only plot if we have data points
                if update_counts and bit_errs:
                    plt.plot(update_counts, bit_errs, 
                            label=f"{model_name}", marker='s', markersize=3)
        
        plt.xlabel("Number of Parameter Updates")
        plt.ylabel("Val_Bit Error Rate")
        plt.title("Val_Bit Error Rate vs Parameter Updates")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc="best")
        
        plt.tight_layout(pad=3.0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Update-based comparison saved to {save_path}")

    @staticmethod
    def clear_data():
        MultiModelBCP.all_models_data = {}

# signal_simulator 类
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
            rician_mu = np.sqrt(k / (k + 1))
            s = np.sqrt(1 / (2 * (k + 1)))
            channel = rician_mu + s * (np.random.randn(1, num_path) + 1j * np.random.randn(1, num_path))
            channel = channel[:, 0]
        elif self.channel_type == "awgn":
            return self.awgn(transmit_signals, self.SNRdB)
        elif self.channel_type == "3gpp":
            train_size = self.channel_3gpp.shape[0]
            index = np.random.choice(np.arange(train_size), size=1)
            h = self.channel_3gpp[index]
            channel = h[:, 0]
        elif self.channel_type == "random_mixed" or self.channel_type == "sequential_mixed":
            return transmit_signals  # 混合信道在 generate_mixed_dataset 中处理
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
                channel_types = ["rician", "awgn", "rayleigh"]  # 默认混合信道类型
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
            layers.Dropout(0.2),
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
                 first_decay_steps=1000, t_mul=1.3, m_mul=0.9, alpha=0.001):
        super(MetaDNN, self).__init__(input_dim, payloadBits_per_OFDM)
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.mini_batch_size = mini_size
        self.optimizer = tf.keras.optimizers.SGD(inner_lr)
        self.all_epoch_losses = []
        self.all_val_bit_errs = []
        self.update_counts = []  # Track update counts
        self.total_updates = 0   # Counter for total meta-updates
        self.inner_updates = 0   # Counter for inner loop updates (informational only)

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
        model_copy = MetaDNN(self.input_dim, self.output_dim, self.inner_lr, self.meta_lr, self.mini_batch_size)
        model_copy.set_params(self.get_params())
        return model_copy
    

    def inner_update(self, x_task, y_task, steps=None):
        model_copy = self.clone()
        losses = []

        num_samples = x_task.shape[0]
        if steps is None:
            steps = int(np.ceil(num_samples / self.mini_batch_size))

        for _ in range(steps):

            batch_indices = np.random.choice(num_samples, size=min(self.mini_batch_size, num_samples), replace=False)
            x_batch = tf.gather(x_task, batch_indices)
            y_batch = tf.gather(y_task, batch_indices)

            with tf.GradientTape() as tape:
                preds = model_copy(x_batch, training=True)
                loss = tf.keras.losses.mean_squared_error(y_batch, preds)
                losses.append(tf.reduce_mean(loss))
            grads = tape.gradient(loss, model_copy.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, model_copy.trainable_variables))
        return model_copy, tf.reduce_mean(losses), steps
    
    def evaluate(self, x_val, y_val):
        preds = self(x_val, training=False)
        return bit_err(y_val, preds).numpy()
    
    def train_reptile(self, tasks, meta_epochs=10, inner_steps=None, meta_validation_data=None):
        start_time = time.time()
        for epoch in range(meta_epochs):
            epcoch_start_time = time.time()
            current_meta_lr = self.meta_lr * (math.cos(math.pi * epoch / meta_epochs) + 1) / 2 
            task_losses = []

            initial_params = self.get_params()
            meta_grads = [tf.zeros_like(p) for p in initial_params]
            
            # Inner loop updates tracking (for information)
            epoch_inner_updates = 0
            
            for x_task, y_task in tasks:
                updated_model, task_loss, inner_steps = self.inner_update(x_task, y_task, inner_steps)
                epoch_inner_updates += inner_steps  # Count inner updates
                task_losses.append(task_loss)
                updated_params = updated_model.get_params()
                for i, (init_p, upd_p) in enumerate(zip(initial_params, updated_params)):
                    meta_grads[i] += (upd_p - init_p)
            
            # Apply meta-update
            new_params = []
            for init_p, grad in zip(initial_params, meta_grads):
                new_params.append(init_p + current_meta_lr * grad / len(tasks))
            self.set_params(new_params)
            
            # Count this as one meta-update (the real parameter update)
            self.total_updates += 1
            self.inner_updates += epoch_inner_updates
            
            # Track metrics with update count
            epoch_loss = tf.reduce_mean(task_losses)
            self.all_epoch_losses.append(epoch_loss)
            self.update_counts.append(self.total_updates)
            
            # Evaluate on validation data
            val_bit_err = None
            if meta_validation_data is not None:
                val_bit_err = self.evaluate(*meta_validation_data)
                self.all_val_bit_errs.append(val_bit_err)
                
            # Print progress
            if meta_validation_data is not None and (epoch + 1) % 50 == 0:
                print(f"Meta Epoch {epoch + 1}/{meta_epochs}, "
                    f"epoch_loss: {epoch_loss.numpy()}, val_bit_err: {val_bit_err}",
                    f"Epoch Time: {time.time()-epcoch_start_time}s")
        
        print(f"Training completed with {self.total_updates} meta-updates and {self.inner_updates} inner-loop updates, Train Time:{time.time()-start_time:.2f}s")
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
    simulator = signal_simulator()
    channel_types = ["awgn", "rician", "rayleigh", "random_mixed"]
    meta_channel_types = ["awgn", "rician", "rayleigh"]
    models = {}
    histories = {}

    DNN_samples = 960
    DNN_epoch = 5
    DNN_batch_size = 16
    # Generate training data 
    bits = simulator.generate_bits(DNN_samples)
    MultiModelBCP.clear_data()

    # Train phase for standard DNN
    print("=== Train Phase ===")
    for channel in channel_types:
        x_train, y_train = simulator.generate_training_dataset(channel, bits)
        x_test, y_test = simulator.generate_testing_dataset(channel, 2500)
        model_name = f"DNN_{channel}"
        models[model_name] = DNN(
            input_dim=x_train.shape[1],
            payloadBits_per_OFDM=simulator.payloadBits_per_OFDM
        )
        print(f"\nTraining on {channel} channel...")
        histories[model_name] = models[model_name].train(
            x_train, y_train,
            epochs=DNN_epoch, batch_size=DNN_batch_size,
            validation_data=(x_test, y_test),
            dataset_type=channel
        )

    # Meta-learning phase
    print("\n=== Meta-Learning Phase ===")
    meta_tasks = []
    meta_model_name = "Meta_DNN"
    total_meta_interation = int((DNN_samples/DNN_batch_size)*DNN_epoch)
    print(f"Meta_update: {total_meta_interation}")

    samples_per_channel = int(np.ceil(DNN_samples/len(meta_channel_types)))
    start_idx = 0
    for channel in meta_channel_types:
        end_idx = start_idx + samples_per_channel
        x_task, y_task = simulator.generate_training_dataset(channel, bits[start_idx:end_idx], mode=channel)
        meta_tasks.append((x_task, y_task))
        start_idx = end_idx

    models[meta_model_name] = MetaDNN(
        input_dim=x_task.shape[1],
        payloadBits_per_OFDM=simulator.payloadBits_per_OFDM,
        inner_lr=0.02,
        meta_lr=0.3,
        mini_size = 32
    )
    
    meta_x_test, meta_y_test = simulator.generate_testing_dataset("random_mixed", 10000)
    
    # Train meta-model and get update counts
    losses, val_errs, update_counts = models["Meta_DNN"].train_reptile(
        meta_tasks, 
        meta_epochs=total_meta_interation, 
        meta_validation_data=(meta_x_test, meta_y_test)
    )
    
    # Log meta-model data with update counts
    MultiModelBCP.log_manual_data(
        "Meta_DNN_train",
        losses,
        val_errs,
        update_counts=update_counts,
        dataset_type="meta"
    )

    # Plot traditional learning curves
    # MultiModelBCP.plot_all_learning_curves(
    #     save_path="train_phase_traditional.png",
    #     plot_batch=True,
    #     plot_epoch=True,
    #     plot_train_bit_err=False
    # )
    
    # Plot update-based comparison
    MultiModelBCP.plot_by_updates(save_path="update_based_comparison.png")
    
    # Generalization Testing phase
    print("\n=== 3GPP Generalization Testing Phase ===")
    MultiModelBCP.clear_data()
    sample_sizes = [100,500] #[100,500,1000,100k]
    x_3gpp_val, y_3gpp_val = simulator.generate_testing_dataset("3gpp", 10000)
    val_epoch = 3
    val_batch_size = 25
    for size in sample_sizes:

        DNN_num_update= int((size/val_batch_size)*val_epoch)

        bits_array = simulator.generate_bits(size)
        x_3gpp_train, y_3gpp_train = simulator.generate_training_dataset("3gpp", bits_array)

        # Traditional DNN fine-tuning
        for channel in channel_types:
            model_name = f"DNN_{channel}_3GPP_{size}"
            dnn_model = models[f"DNN_{channel}"].clone()
            print(f"\nValidating on {channel} channel with {size} samples...")
            dnn_model.train(
                x_3gpp_train, y_3gpp_train,
                epochs=val_epoch, batch_size=val_batch_size,
                validation_data=(x_3gpp_val, y_3gpp_val),
                dataset_type=f"{channel}_{size}"
            )

        # Meta DNN fine-tuning
        meta_task_3gpp = [(x_3gpp_train, y_3gpp_train)]
        meta_model = models["Meta_DNN"].clone()
        meta_model_name = f"Meta_{size}"
        losses, val_errs, update_counts = meta_model.train_reptile(
        meta_task_3gpp, 
        meta_epochs=DNN_num_update, 
        meta_validation_data=(x_3gpp_val, y_3gpp_val)
        )
        MultiModelBCP.log_manual_data(
            meta_model_name,
            losses,
            val_errs,
            update_counts=update_counts,
            dataset_type=f"3gpp_{size}"
        )

    MultiModelBCP.plot_by_updates(save_path="3gpp_gerneralization_test.png")