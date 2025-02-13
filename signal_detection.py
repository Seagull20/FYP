import importlib, pkg_resources
importlib.reload(pkg_resources)
import tensorflow as tf

import numpy as np
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal

SNRdB = 20

def generate_samples(num_bits, channel_type):
    x_sample = []
    y_sample = []
    bits_train = np.random.binomial(n=1, p=0.5, size=(num_bits, ))
    if channel_type == "rayleigh":
        for bit in bits_train:
            channel = np.sqrt(1 / 2) * np.sqrt(1) * (np.random.randn(1, 1) + 1j * np.random.randn(1, 1))
            convolved = bit * channel
            signal_power = np.mean(abs(convolved**2))
            sigma2 = signal_power * 10**(-SNRdB/10)  # calculate noise power based on signal power and SNR
            # Generate complex noise with given variance
            noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
            received_signal = convolved + noise
            received_signal = received_signal[0]
            rceived_signal_sample = np.concatenate((np.real(received_signal), np.imag(received_signal)))
            x_sample.append(rceived_signal_sample)
            y_sample.append(bit)
        return np.asarray(x_sample), np.asarray(y_sample)
    elif channel_type == "awgn":
        for bit in bits_train:
            received_signal = awgn(bit)
            received_signal = np.asarray(received_signal)
            rceived_signal_sample = np.concatenate(([np.real(received_signal)], [np.imag(received_signal)]))
            x_sample.append(rceived_signal_sample)
            y_sample.append(bit)
        return np.asarray(x_sample), np.asarray(y_sample)
    elif channel_type == "rician":
        for bit in bits_train:
            k = 10 ** (1 / 10)  # dB to mW
            mu = np.sqrt(k / (k + 1))  # direct path
            s = np.sqrt(1 / (2 * (k + 1)))  # scattered paths
            channel = mu + s * (np.random.randn(1, 1) + 1j * np.random.randn(1, 1))  # Rician channel
            convolved = bit * channel
            signal_power = np.mean(abs(convolved**2))
            sigma2 = signal_power * 10**(-SNRdB/10)  # calculate noise power based on signal power and SNR
            noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
            received_signal = convolved + noise
            received_signal = received_signal[0]
            rceived_signal_sample = np.concatenate((np.real(received_signal), np.imag(received_signal)))
            x_sample.append(rceived_signal_sample)
            y_sample.append(bit)
        return np.asarray(x_sample), np.asarray(y_sample)
    elif channel_type == "3gpp":
        channel_3gpp = np.load('3gpp_data.npy')
        for bit in bits_train:
            train_size = channel_3gpp.shape[0]
            index = np.random.choice(np.arange(train_size), size=1)
            h = channel_3gpp[index]
            channel = h[0]
            convolved = bit * channel
            signal_power = np.mean(abs(convolved**2))
            sigma2 = signal_power * 10**(-SNRdB/10)  # calculate noise power based on signal power and SNR
            # Generate complex noise with given variance
            noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
            received_signal = convolved + noise
            rceived_signal_sample = np.concatenate((np.real(received_signal), np.imag(received_signal)))
            x_sample.append(rceived_signal_sample)
            y_sample.append(bit)
        return np.asarray(x_sample), np.asarray(y_sample)
    
def awgn(signals):
        gamma = 10**(SNRdB/10)
        
        P = sum(abs(signals) ** 2) #Actual power in the vector
        N0 = P / gamma # Find the noise spectral density
        if isrealobj(signals):# check if input is real/complex object type
            n = sqrt(N0/2) * standard_normal(signals.shape) # computed noise
        else:
            n = sqrt(N0/2) * (standard_normal(signals.shape)+1j*standard_normal(signals.shape))
        
        return signals + n


def bit_err(y_true, y_pred):
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

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mse',
    optimizer=adam,
    metrics=[bit_err])


x_train, y_train = generate_samples(10000, "rayleigh")
x_test, y_test = generate_samples(10000, "rayleigh")

# print(x_test)


h = model.fit(x_train, y_train,
                  batch_size=16,
                  epochs=20,
                #   validation_data = (x_test, y_test),
                  verbose=1)
model.summary()


prediction = model.predict(x_test)
predict_bits = np.where(prediction > 0.5, 1, 0)
y_test = y_test.flatten()
predict_bits = predict_bits.flatten()

ber = np.sum(abs(y_test-predict_bits))/len(y_test)

print("BER: " + str(ber))















