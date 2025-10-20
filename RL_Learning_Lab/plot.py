# 好好学习 天天向上
# {2024/5/30} {16:41}
import matplotlib.pyplot as plt
import numpy as np

# 假设这是你的矩阵，其中0表示没有箭头，其他值表示箭头的类型和方向（仅为示例）
matrix = np.array([
    [0, 1, 0, 0, 0],
    [2, 0, 0, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, 0, 4, 0],
    [0, 0, 0, 0, 5]
])

# 创建一个图形和坐标轴
fig, ax = plt.subplots()

# 绘制矩阵（这里只是简单地使用颜色表示不同值，但你可以根据需要调整）
cmap = plt.get_cmap('viridis')  # 选择一个颜色映射
im = ax.imshow(matrix, cmap=cmap, interpolation='nearest')

# 添加颜色条
fig.colorbar(im)

# 为矩阵中的每个元素添加箭头（这里只是一个示例，需要根据实际值来确定箭头）
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        if matrix[i, j] != 0:
            # 假设1表示向上箭头，2表示向下箭头，以此类推...
            if matrix[i, j] == 1:
                ax.annotate('↑', xy=(j, i), textcoords='offset points',
                            xytext=(0, 10), ha='center', va='bottom')
            elif matrix[i, j] == 2:
                ax.annotate('↓', xy=(j, i), textcoords='offset points',
                            xytext=(0, -10), ha='center', va='top')
                # 添加其他箭头的逻辑...

# 显示图形
plt.show()