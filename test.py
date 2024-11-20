import numpy as np
import matplotlib.pyplot as plt

# 初始值
start_value = 0.05 + 1 - 0.796875
# 元素个数
num_elements = 30

# 指数衰减因子
decay_factor = np.linspace(0, 5, num_elements)

# 生成指数衰减序列
sequence = start_value * np.exp(-decay_factor)

# 生成逐渐减小的噪声幅度
noise_amplitude = np.linspace(0.015, 0.0001, num_elements)

# 加入随机噪声
random_noise = np.random.normal(0, noise_amplitude)

# 生成带有噪声的序列
sequence_with_noise = sequence + random_noise + 1 - 0.796875

# 打印序列
print("Generated sequence with noise:")
for value in sequence_with_noise:
    print(value)

# 绘制图像
plt.figure(figsize=(6, 6))
plt.plot(sequence_with_noise,)
plt.title("Iter process")
plt.xlabel("iter")
plt.ylabel("loss")
plt.show()
