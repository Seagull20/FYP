import numpy as np

# 输入比特流
bits = np.array([1, 0, 1, 1,  0, 0, 1, 1,  0, 1, 0, 1,  0, 0, 0, 1])

# QAM 调制（16-QAM）
qam_table = {
    (0,0,0,0): (-3 - 3j), (0,0,0,1): (-3 - 1j), (0,0,1,0): (-3 + 3j), (0,0,1,1): (-3 + 1j),
    (0,1,0,0): (-1 - 3j), (0,1,0,1): (-1 - 1j), (0,1,1,0): (-1 + 3j), (0,1,1,1): (-1 + 1j),
    (1,0,0,0): (3 - 3j), (1,0,0,1): (3 - 1j), (1,0,1,0): (3 + 3j), (1,0,1,1): (3 + 1j),
    (1,1,0,0): (1 - 3j), (1,1,0,1): (1 - 1j), (1,1,1,0): (1 + 3j), (1,1,1,1): (1 + 1j)
}

# 4-bit 一组进行 QAM 调制
qam_symbols = np.array([qam_table[tuple(bits[i:i+4])] for i in range(0, len(bits), 4)])

# IFFT（转换到时域）
ofdm_time = np.fft.ifft(qam_symbols)

# 添加循环前缀（假设 CP 长度为 1）
cp = ofdm_time[-1]  # 取最后一个样本作为 CP
ofdm_with_cp = np.concatenate([[cp], ofdm_time])

print("OFDM 编码后输出（时域信号）：", ofdm_with_cp)
