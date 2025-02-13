import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
from global_parameters import *
import scipy.interpolate
from keras.models import Sequential
from keras.layers.core import Dense
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
from keras.callbacks import Callback


class BCP(Callback):
    batch_accuracy = [] # accuracy at given batch
    batch_loss = [] # loss at given batch    
    def __init__(self):
        super(BCP,self).__init__() 
    def on_train_batch_end(self, batch, logs=None):                
        BCP.batch_accuracy.append(logs.get('bit_err'))
        BCP.batch_loss.append(logs.get('loss'))


class without_transfer_learning_ofdm:
    def __init__(self, channel, SNR):
        self.all_carriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
        self.pilot_carriers = self.all_carriers[::K // P]  # Pilots is every (K/P)th carrier.
        # data carriers are all remaining carriers
        self.data_carriers = np.delete(self.all_carriers, self.pilot_carriers)
        self.payloadBits_per_OFDM = len(self.data_carriers) * mu  # number of payload bits per OFDM symbol
        self.channel_type = channel
        # print("Training for " + self.channel_type + " channel")
        # self.channel_3gpp = np.load('channel_train.npy')
        self.SNRdB =  SNR

    
    def transmit_signals(self, bits):
        bits_sp = self.sp(bits)
        qam = self.mapping(bits_sp)
        ofdm_data = self.ofdm_symbol(qam)
        ofdm_time = self.idft(ofdm_data)
        ofdm_with_cp = self.add_cp(ofdm_time)
        return ofdm_with_cp
    
    def received_signals(self, transmit_signals):
        if self.channel_type == "rayleigh":
            channel = np.sqrt(1 / 2) * np.sqrt(1/num_path) * (np.random.randn(1, num_path) + 1j * np.random.randn(1, num_path))
            channel = channel[:, 0]
            convolved = np.convolve(transmit_signals, channel)
            signal_power = np.mean(abs(convolved**2))
            sigma2 = signal_power * 10**(-self.SNRdB/10)  # calculate noise power based on signal power and SNR
            # Generate complex noise with given variance
            noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
            return convolved + noise
        elif self.channel_type == "rician":
            k = 10 ** (rician_factor / 10)  # dB to mW
            mu = np.sqrt(k / (k + 1))  # direct path
            s = np.sqrt(1 / (2 * (k + 1)))  # scattered paths
            channel = mu + s * (np.random.randn(1, num_path) + 1j * np.random.randn(1, num_path))  # Rician channel
            channel = channel[:, 0]
            convolved = np.convolve(transmit_signals, channel)
            signal_power = np.mean(abs(convolved**2))
            sigma2 = signal_power * 10**(-self.SNRdB/10)  # calculate noise power based on signal power and SNR
            # Generate complex noise with given variance
            noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
            return convolved + noise
        elif self.channel_type == "awgn":
            return self.awgn(transmit_signals, self.SNRdB)
        elif self.channel_type == "3gpp":
            train_size = self.channel_3gpp.shape[0]
            index = np.random.choice(np.arange(train_size), size=1)
            h = self.channel_3gpp[index]
            channel = h[:, 0]
            convolved = np.convolve(transmit_signals, channel)
            signal_power = np.mean(abs(convolved**2))
            sigma2 = signal_power * 10**(-self.SNRdB/10)  # calculate noise power based on signal power and SNR
            # Generate complex noise with given variance
            noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
            return convolved + noise

    def ofdm_simulate(self, bits):
        # create received signals
        ofdm_tx = self.transmit_signals(bits)
        ofdm_rx = self.received_signals(ofdm_tx)

        # process received signals
        ofdm_rx_no_cp = self.remove_cp(ofdm_rx)
        ofdm_demodulation = self.dft(ofdm_rx_no_cp)
        h_estimate_from_pilot = self.estimate_channel(ofdm_demodulation)
        return np.concatenate((np.real(ofdm_demodulation), np.imag(ofdm_demodulation)))
        # return np.concatenate((np.concatenate((np.real(h_estimate_from_pilot), np.imag(h_estimate_from_pilot))), np.concatenate((np.real(ofdm_demodulation),np.imag(ofdm_demodulation)))))
    
    def generate_training_dataset(self):
        training_sample = []
        training_label = []
        for i in range(num_simulate_target):
            bits = np.random.binomial(n=1, p=0.5, size=(self.payloadBits_per_OFDM, ))
            ofdm_simulate_output = self.ofdm_simulate(bits)
            training_sample.append(ofdm_simulate_output)
            training_label.append(bits)
        return np.asarray(training_sample), np.asarray(training_label)
    
    def generate_testing_dataset(self):
        testing_sample = []
        testing_label = []
        for i in range(num_test_target):
            bits = np.random.binomial(n=1, p=0.5, size=(self.payloadBits_per_OFDM, ))
            ofdm_simulate_output = self.ofdm_simulate(bits)
            testing_sample.append(ofdm_simulate_output)
            testing_label.append(bits)
        return np.asarray(testing_sample), np.asarray(testing_label)
    
    def cnn_model(self):
        model = Sequential()
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(self.payloadBits_per_OFDM, activation='sigmoid'))
        # model.add(Dense(16, activation='sigmoid'))
        return model
    
    def training(self):
        base_model = self.cnn_model()
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        base_model.compile(loss='mse',
                      optimizer=adam,
                      metrics=[self.bit_err])

        X, Y = self.generate_training_dataset()
        X_test, Y_test = self.generate_testing_dataset()
        h = base_model.fit(X, Y,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  validation_data = (X_test, Y_test),
                  verbose=1,
                  callbacks = [BCP()])
        # print(BCP.batch_accuracy)
        # print(len(BCP.batch_accuracy))
        # return h.history['val_bit_err'][-1]
        # return h.history['val_bit_err']
        return BCP.batch_accuracy, h.history['val_bit_err']



    def bit_err(self, y_true, y_pred):
        err = 1 - tf.reduce_mean(
        tf.reduce_mean(
            tf.compat.v1.to_float(
                tf.equal(
                    tf.sign(
                        y_pred - 0.5),
                    tf.cast(
                        tf.sign(
                            y_true - 0.5),
                        tf.float32))),
            1))
        return err

    def sp(self, bits):
        return bits.reshape((len(self.data_carriers), mu)) # group mu bits
    
    def mapping(self, bits_sp):
        return np.array([mapping_table[tuple(b)] for b in bits_sp]) # bit mapping for 4-QAM
    
    def ofdm_symbol(self, qam_payload):
        symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
        symbol[self.pilot_carriers] = pilot_value  # allocate the pilot subcarriers 
        symbol[self.data_carriers] = qam_payload  # allocate the pilot subcarriers
        return symbol
    
    def idft(self, OFDM_data):
        return np.fft.ifft(OFDM_data)
    
    def add_cp(self, OFDM_time):
        cp = OFDM_time[-CP:]               # take the last CP samples ...
        return np.hstack([cp, OFDM_time])  # ... and add them to the beginning
    
    def remove_cp(self, signals):
        return signals[CP:(CP+K)]
    
    def dft(self, signals):
        return np.fft.fft(signals)

    def estimate_channel(self, signals):
        pilots = signals[self.pilot_carriers]  # extract the pilot values from the RX signal
        hest_at_pilots = pilots / pilot_value # divide by the transmitted pilot values
        # # Perform interpolation between the pilot carriers to get an estimate
        # # of the channel in the data carriers. Here, we interpolate absolute value and phase 
        # # separately
        # hest_abs = scipy.interpolate.interp1d(self.pilot_carriers, abs(hest_at_pilots), kind='linear', bounds_error=False)(self.all_carriers)
        # hest_phase = scipy.interpolate.interp1d(self.pilot_carriers, np.angle(hest_at_pilots), kind='linear', bounds_error=False)(self.all_carriers)
        # hest = hest_abs * np.exp(1j*hest_phase)
        return hest_at_pilots
    
    def awgn(self, signals, SNRdb):
        gamma = 10**(SNRdb/10)
        if signals.ndim == 1:# if s is single dimensional vector
            P = sum(abs(signals) ** 2) / len(signals) #Actual power in the vector
        else: # multi-dimensional signals like MFSK
            P = sum(sum(abs(signals) ** 2)) / len(signals) # if s is a matrix [MxN]
        N0 = P / gamma # Find the noise spectral density
        if isrealobj(signals):# check if input is real/complex object type
            n = sqrt(N0/2) * standard_normal(signals.shape) # computed noise
        else:
            n = sqrt(N0/2) * (standard_normal(signals.shape)+1j*standard_normal(signals.shape))
        
        return signals + n

if __name__ == "__main__":
    train = without_transfer_learning_ofdm("rician",20)
    train.training()


    





