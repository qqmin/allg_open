import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

# 读取时间数据和振幅数据
timefilename = 'natural_frequency/input_time.txt'
displacementfilename = 'natural_frequency/input_disp.txt'

# 读取数据
time = np.loadtxt(timefilename)
displacement = np.loadtxt(displacementfilename)  # 保持单位为毫米

# 计算采样频率
fs = 1 / (time[1] - time[0])

# FFT变换
n = len(displacement)
yf = fft(displacement)
xf = fftfreq(n, 1/fs)

# 只取正频率部分
half_n = n // 2
xf = xf[:half_n]
yf = yf[:half_n]

# 计算归一化振幅
yf_abs = np.abs(yf) * 2.0 / n

# 找到振幅最大的频率 即一阶固有频率
fundamental_freq = xf[np.argmax(yf_abs)]

# 绘制时间序列图
plt.figure(figsize=(12, 6))
plt.plot(time, displacement)
plt.title('Vibration Time Series')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (mm)')
plt.grid()
plt.show()

# 绘制频谱图
plt.figure(figsize=(12, 6))
plt.plot(xf, yf_abs)
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized Amplitude')
plt.grid()
plt.show()

# 打印一阶固有频率
print(f"The fundamental frequency is: {fundamental_freq} Hz")

print(f"Index of max amplitude: {np.argmax(yf_abs)}")

# 将频谱图的数据输出到文本文件
np.savetxt('natural_frequency/output_frequency.txt', xf, fmt='%.4f')
np.savetxt('natural_frequency/output_amplitude.txt', yf_abs, fmt='%.4f')
