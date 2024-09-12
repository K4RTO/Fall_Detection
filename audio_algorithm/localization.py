import numpy as np
from scipy.optimize import fsolve

# Define microphone position and sound propagation speed
d = 0.1 # Microphone spacing, assumed to be 1 unit
v = 343  # Speed of sound (m/s), assumed to be 343 m/s

# Position of microphone
A = (-d, 0)
B = (0, 0)
C = (d, 0)


def calculate_angle(t_A, t_B, t_C):
    # Calculate the time difference
    delta_t_AB = t_A - t_B
    delta_t_BC = t_B - t_C

    # Calculate distance difference
    d_AB = v * delta_t_AB
    d_BC = v * delta_t_BC

    # Objective function for hyperbolic equation
    def equations(vars):
        x, y = vars
        eq1 = np.sqrt((x - A[0]) ** 2 + (y - A[1]) ** 2) - np.sqrt((x - B[0]) ** 2 + (y - B[1]) ** 2) - d_AB
        eq2 = np.sqrt((x - B[0]) ** 2 + (y - B[1]) ** 2) - np.sqrt((x - C[0]) ** 2 + (y - C[1]) ** 2) - d_BC
        return [eq1, eq2]

    # Initial guess
    initial_guesses = [(0, 0), (1, 1)]  # 提供两个不同的初始猜测

    # 求解方程
    solutions = []
    for guess in initial_guesses:
        x, y = fsolve(equations, guess)
        if y > 0:  # 保留 y 为正的交点
            solutions.append((x, y))

    if not solutions:
        raise ValueError("没有找到 y 为正的交点")

    # 选择第一个找到的正 y 值的交点
    x, y = solutions[0]

    # 计算夹角
    angle = np.arctan2(y, x)  # 计算与中间麦克风的夹角
    angle_degrees = np.degrees(angle)  # 转换为度数

    return angle_degrees


# 示例调用
t_A, t_B, t_C = 0.1, 0.2, 0.15
angle = calculate_angle(t_A, t_B, t_C)
print(f"相对于中间麦克风的夹角: {angle:.2f} 度")
