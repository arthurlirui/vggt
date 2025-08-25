import numpy as np
from scipy.linalg import expm, norm
from pyquaternion import Quaternion
from dataclasses import dataclass
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from eskf.es import ESKFFusion


# 使用示例
def example_usage():
    # 配置参数
    config = {
        'dt': 0.01,
        'initial_pose': {
            'position': np.zeros(3),
            'orientation': Quaternion()
        }
    }

    # 初始化ESKF融合器
    eskf_fusion = ESKFFusion(config)

    # 模拟数据流
    for i in range(1000):
        # 模拟IMU数据 (角速度和线性加速度)
        imu_data = {
            'angular_velocity': np.array([0.1, 0.05, 0.02]) + np.random.normal(0, 0.01, 3),
            'linear_acceleration': np.array([0.5, 0.3, -9.8]) + np.random.normal(0, 0.1, 3)
        }

        vision_pose = None
        # 每10帧接收一次视觉大模型的位姿
        if i % 10 == 0:
            vision_pose = {
                'position': np.array([i * 0.1, i * 0.05, i * 0.02]) + np.random.normal(0, 0.05, 3),
                'orientation': Quaternion(axis=[0, 0, 1], angle=i * 0.01)
            }

        # 处理数据
        fused_pose = eskf_fusion.process_data(imu_data, vision_pose)

        if i % 100 == 0:
            print(f"Step {i}: Fused Position - {fused_pose[0]}")

    # 可视化轨迹
    eskf_fusion.visualize_trajectory()


if __name__ == "__main__":
    example_usage()