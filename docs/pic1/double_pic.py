import matplotlib.pyplot as plt
import numpy as np

# 生成大图的数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘制大图
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of sin(x)')
plt.legend()

# 生成小图的数据
x_small = np.linspace(0, 5, 50)
y_small = np.cos(x_small)

# 添加小图
plt.axes([0.2, 0.55, 0.3, 0.3])  # 设置小图的位置和大小，左下角坐标为(0.2, 0.55)，宽度为0.3，高度为0.3
plt.plot(x_small, y_small, label='cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Zoomed-in Plot of cos(x)')
plt.legend()

# 显示图形
plt.show()
