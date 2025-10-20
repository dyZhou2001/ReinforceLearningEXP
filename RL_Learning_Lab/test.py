# # 好好学习 天天向上
# # {2024/6/25} {19:07}
# import numpy as np
# import torch
# # a=np.array([
# #     [0.1,0.1,0.1,0.2]
# # ])
# # b=np.array([
# #     [0.3, 0.1, 0.6, 0],
# #     [0.1, 0.3, 0, 0.6],
# #     [0.1, 0, 0.3, 0.6],
# #     [0, 0.1, 0.1, 0.8]
# # ])
# #
# # for i in range(11):
# #     a=a@b
# #
# # print(a)
#
# # a=[(1,2),(3,4),(5,6)]
# # b=torch.tensor(a,dtype=torch.float)
# # print(b)
# # c=torch.nn.Linear(2,5)
# # d=c(b)
# # print(d)
# # print(d.max(1)[0])
# # actionbatch = torch.tensor([[-0.7977],
# #         [-0.7072],
# #         [-1.3350]])
# #
# # extstatebatch =torch.tensor([[-5.4316e-01,  8.3963e-01,  5.0269e+00],
# #         [-6.3399e-01,  7.7334e-01,  4.1174e+00],
# #         [ 9.7182e-01,  2.3572e-01, -5.4318e+00]])
# #
# # print(torch.cat([extstatebatch,actionbatch],1))
# # advantage_batch=torch.empty(1,1)
# # actionbatch = torch.tensor([[-0.7977],
# #          [-0.7072],
# #          [-1.3350]])
# #
# # print(actionbatch[1,0])
# # for i in reversed(actionbatch):
# #         print(i)
# #         i=i.view(1,1)
# #         print(i)
# #         advantage_batch=torch.cat([advantage_batch,i],dim=0)
# # print(advantage_batch)
# # advantage_batch=torch.flip(advantage_batch[1:,:],dims=[1])
# # print(advantage_batch)
# # for i in reversed(range(5)):
# #     print(i)
# # import matplotlib.pyplot as plt
# # import matplotlib.ticker as ticker
# #
# # # 数据准备
# # years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
# # pure_ev = [5579, 11375, 14604, 74800, 247000, 409000, 652000, 942000, 906000, 1040000, 2830000]  # 纯电动销量
# # phev = [2580, 1416, 3038, 0, 84100, 98000, 125000, 314000, 300000, 327000, 691000]  # 插电混动销量
# #
# # # 图表设置
# # plt.figure(figsize=(12, 6))
# #
# # # 绘制堆叠柱状图
# # bars_pure_ev = plt.bar(years, pure_ev, label='EV', color='#1f77b4', edgecolor='black')
# # bars_phev = plt.bar(years, phev, bottom=pure_ev, label='PHEV', color='#ff7f0e', edgecolor='black')
# #
# # # 添加数据标签
# # for bar_ev, bar_phev in zip(bars_pure_ev, bars_phev):
# #     height_ev = bar_ev.get_height()
# #     height_phev = bar_phev.get_height()
# #     total_height = height_ev + height_phev
# #     plt.text(bar_ev.get_x() + bar_ev.get_width() / 2., total_height,
# #              f'{total_height / 10000:.1f}w',
# #              ha='center', va='bottom', fontsize=9)
# #
# # # 坐标轴格式化
# # plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x / 10000:.0f}w'))
# # plt.xticks(rotation=45, ha='right')  # 旋转x轴标签
# #
# # # 添加图表元素
# # plt.title('Annual Sales of New Energy Vehicles in China (2011-2021)', fontsize=14, pad=20)
# # plt.xlabel('year', fontsize=12)
# # plt.ylabel('sales volume', fontsize=12)
# # plt.grid(axis='y', linestyle='--', alpha=0.7)
# #
# # # 添加图例
# # plt.legend(loc='upper left', fontsize=12)
# #
# # # 调整布局
# # plt.tight_layout()
# #
# # # 保存图表（可选）
# # # plt.savefig('new_energy_vehicle_sales_stacked.png', dpi=300)
# #
# # plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Data
# years = np.arange(2011, 2023)
# total = [0.6, 1.3, 2.0, 8.0, 33.0, 51.0, 75.0, 126.0, 121.0, 137.0, 352.0, 688.7]
# bev = [0.6, 1.2, 1.8, 7.0, 28.0, 44.0, 63.0, 101.0, 96.0, 112.0, 302.0, 536.5]
# phev = [0.0, 0.1, 0.2, 1.0, 5.0, 7.0, 12.0, 25.0, 25.0, 25.0, 50.0, 151.8]
#
# # Plot
# plt.figure(figsize=(12, 6))
# plt.bar(years, bev, label='Battery Electric Vehicles (BEV)', color='#1f77b4')
# plt.bar(years, phev, bottom=bev, label='Plug-in Hybrid Electric Vehicles (PHEV/EREV)', color='#ff7f0e')
#
# # Labels and title
# plt.title('China New Energy Vehicle Sales (2011-2022)', fontsize=14)
# plt.xlabel('Year', fontsize=12)
# plt.ylabel('Sales (10,000 units)', fontsize=12)
# plt.xticks(years, rotation=45)
# plt.legend()
#
# # Display value labels
# for i in range(len(years)):
#     plt.text(years[i], total[i] + 10, f'{total[i]:.1f}', ha='center', va='bottom', fontsize=8)
#
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve


def plot_xyz_cube_surface():
    """
    绘制方程 x³ + y³ + z³ = 1 的三维图像
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 设置参数范围
    u = np.linspace(-1.5, 1.5, 50)
    v = np.linspace(-1.5, 1.5, 50)

    # 创建网格
    U, V = np.meshgrid(u, v)

    # 方法1: 通过解方程 z³ = 1 - x³ - y³ 来获取z值
    def solve_for_z(x, y):
        """求解 z³ = 1 - x³ - y³"""
        target = 1 - x ** 3 - y ** 3
        if target >= 0:
            return np.cbrt(target)  # 立方根
        else:
            return -np.cbrt(-target)  # 负数的立方根

    # 计算z值
    Z = np.zeros_like(U)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            x, y = U[i, j], V[i, j]
            target = 1 - x ** 3 - y ** 3
            if target >= 0:
                Z[i, j] = np.cbrt(target)
            else:
                Z[i, j] = -np.cbrt(-target)

    # 只绘制实数解
    mask = ~np.isnan(Z)

    # 绘制表面
    surf = ax.plot_surface(U, V, Z, alpha=0.7, cmap='viridis',
                           linewidth=0, antialiased=True)

    # 添加等高线
    contours = ax.contour(U, V, Z, levels=20, alpha=0.6, cmap='plasma')

    # 设置标签和标题
    ax.set_xlabel('X轴', fontsize=12)
    ax.set_ylabel('Y轴', fontsize=12)
    ax.set_zlabel('Z轴', fontsize=12)
    ax.set_title('方程 x³ + y³ + z³ = 1 的三维图像', fontsize=14, fontweight='bold')

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Z值')

    # 设置视角
    ax.view_init(elev=20, azim=45)

    # 添加网格
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_multiple_views():
    """
    绘制多个视角的图像
    """
    fig = plt.figure(figsize=(15, 5))

    # 准备数据
    u = np.linspace(-1.2, 1.2, 40)
    v = np.linspace(-1.2, 1.2, 40)
    U, V = np.meshgrid(u, v)

    # 计算Z值
    Z = np.zeros_like(U)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            x, y = U[i, j], V[i, j]
            target = 1 - x ** 3 - y ** 3
            if target >= 0:
                Z[i, j] = np.cbrt(target)
            else:
                Z[i, j] = -np.cbrt(-target)

    # 三个不同视角
    views = [(20, 45), (60, 120), (45, 225)]
    titles = ['视角1 (20°, 45°)', '视角2 (60°, 120°)', '视角3 (45°, 225°)']

    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')

        surf = ax.plot_surface(U, V, Z, alpha=0.8, cmap='coolwarm',
                               linewidth=0, antialiased=True)

        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')
        ax.set_title(titles[i])
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3)

    plt.suptitle('x³ + y³ + z³ = 1 的多视角图像', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_contour_slices():
    """
    绘制不同z值处的等高线切片
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)

    z_values = [0, 0.5, 0.8, 1.0]

    for i, z_val in enumerate(z_values):
        # 计算在给定z值处，x³ + y³ = 1 - z³ 的等高线
        target = 1 - z_val ** 3
        Z_contour = X ** 3 + Y ** 3

        contour = axes[i].contour(X, Y, Z_contour, levels=[target], colors='red', linewidths=2)
        axes[i].contourf(X, Y, Z_contour, levels=50, alpha=0.6, cmap='viridis')
        axes[i].set_xlabel('X轴')
        axes[i].set_ylabel('Y轴')
        axes[i].set_title(f'z = {z_val} 处的截面 (x³ + y³ = {target:.3f})')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal')

    plt.suptitle('不同z值处的截面图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("正在绘制 x³ + y³ + z³ = 1 的三维图像...")
    print("请稍等，图像加载中...")

    try:
        # 绘制主要的三维图像
        plot_xyz_cube_surface()

        # 绘制多视角图像
        plot_multiple_views()

        # 绘制截面图
        plot_contour_slices()

        print("图像绘制完成！")

    except Exception as e:
        print(f"绘制过程中出现错误: {e}")
        print("请确保已安装必要的库：pip install matplotlib numpy scipy")
