K = 64  # number of OFDM subcarriers

CP = K // 4  # length of the cyclic prefix: 25% of the block

P = 8  # number of pilot carriers per OFDM block
pilot_value = 1 + 1j  # The known value each pilot transmits

num_path = 16  # number of channel paths

mu = 2  # bits per symbol (i.e. 4QAM)

mapping_table = {
    (0, 0) : -1 - 1j,
    (0, 1) : -1 + 1j,
    (1, 0) : 1 - 1j,
    (1, 1) : 1 + 1j,
}

# mapping_table = {
#     (0, 0, 0, 0): -3 - 3j,
#     (0, 0, 0, 1): -3 - 1j,
#     (0, 0, 1, 0): -3 + 3j,
#     (0, 0, 1, 1): -3 + 1j,
#     (0, 1, 0, 0): -1 - 3j,
#     (0, 1, 0, 1): -1 - 1j,
#     (0, 1, 1, 0): -1 + 3j,
#     (0, 1, 1, 1): -1 + 1j,
#     (1, 0, 0, 0): 3 - 3j,
#     (1, 0, 0, 1): 3 - 1j,
#     (1, 0, 1, 0): 3 + 3j,
#     (1, 0, 1, 1): 3 + 1j,
#     (1, 1, 0, 0): 1 - 3j,
#     (1, 1, 0, 1): 1 - 1j,
#     (1, 1, 1, 0): 1 + 3j,
#     (1, 1, 1, 1): 1 + 1j
# }

demapping_table = {v: k for k, v in mapping_table.items()}

# SNRdb = 20  # signal to noise-ratio in dB at the receiver
num_simulate = 50000
num_simulate_target = 5000
num_test_target = 1000

learning_rate = 0.001
batch_size = 32
num_epochs = 20
rician_factor = 1
num_running = 1