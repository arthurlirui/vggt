import numpy as np
from scipy.linalg import expm, norm
from pyquaternion import Quaternion
from dataclasses import dataclass
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt


class ESKFFusion:
    """
    基于误差状态卡尔曼滤波器(ESKF)的多传感器融合系统
    融合视觉大模型位姿、IMU和视觉里程计数据
    """

    def __init__(self, config: Dict[str, Any]):
        # 初始化参数
        self.dt = config.get('dt', 0.01)
        self.gravity = np.array([0, 0, -9.81])

        # 初始化名义状态
        self.state = self.NominalState()

        # 初始化误差状态协方差 (21x21)
        self.P = np.eye(21) * 0.1

        # 过程噪声协方差 (需要根据传感器特性调整)
        self.Q = np.diag([
            *[0.01] * 3,  # δp_wi
            *[0.01] * 3,  # δv_wi
            *[0.001] * 3,  # δθ_wi
            *[0.0001] * 3,  # δa_bias
            *[0.0001] * 3,  # δw_bias
            *[0.0001] * 3,  # δp_ic
            *[0.0001] * 3  # δθ_ic
        ])

        # 观测噪声协方差
        self.R_vo = np.eye(6) * 0.1  # 视觉大模型观测噪声
        self.R_imu = np.eye(6) * 0.01  # IMU观测噪声

        # 缓冲区
        self.imu_buffer = []
        self.vo_buffer = []
        self.pose_history = []

    @dataclass
    class NominalState:
        """名义状态类"""
        p_wi: np.ndarray = np.zeros(3)  # IMU在世界系中的位置
        v_wi: np.ndarray = np.zeros(3)  # IMU在世界系中的速度
        q_wi: Quaternion = Quaternion()  # IMU到世界系的旋转
        a_bias: np.ndarray = np.zeros(3)  # 加速度计零偏
        w_bias: np.ndarray = np.zeros(3)  # 陀螺仪零偏
        p_ic: np.ndarray = np.zeros(3)  # 相机在IMU系中的位置
        q_ic: Quaternion = Quaternion()  # 相机到IMU系的旋转

        def get_imu_pose(self):
            """获取IMU位姿"""
            return self.p_wi, self.q_wi

        def get_camera_pose(self):
            """获取相机位姿"""
            R_wi = self.q_wi.rotation_matrix
            R_ic = self.q_ic.rotation_matrix
            R_wc = R_wi @ R_ic
            p_wc = self.p_wi + R_wi @ self.p_ic
            return p_wc, Quaternion(matrix=R_wc)

    def skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """计算向量的反对称矩阵"""
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def compute_F_matrix(self, imu_data: Dict[str, np.ndarray], dt: float) -> np.ndarray:
        """计算离散状态转移矩阵 F"""
        w_m = imu_data['angular_velocity'] - self.state.w_bias
        a_m = imu_data['linear_acceleration'] - self.state.a_bias

        R_wi = self.state.q_wi.rotation_matrix
        a_world = R_wi @ a_m + self.gravity

        F = np.eye(21)

        # 位置误差δp_wi 受 速度误差δv_wi 影响
        F[0:3, 3:6] = np.eye(3) * dt

        # 速度误差δv_wi 受 姿态误差δθ_wi 和 加速度零偏误差δa_bias 影响
        F[3:6, 6:9] = -self.skew_symmetric(a_world) * dt
        F[3:6, 9:12] = -R_wi * dt

        # 姿态误差δθ_wi 受 自身和陀螺仪零偏误差δw_bias 影响
        F[6:9, 6:9] = expm(-self.skew_symmetric(w_m * dt))
        F[6:9, 12:15] = -np.eye(3) * dt

        # 外参假设为常数，对应的转移矩阵为单位阵
        return F

    def predict(self, imu_data: Dict[str, np.ndarray], dt: float):
        """预测步骤：使用IMU数据推进状态"""
        # 提取IMU测量值
        w_m = imu_data['angular_velocity']
        a_m = imu_data['linear_acceleration']

        # 名义状态预测
        w_corrected = w_m - self.state.w_bias
        a_corrected = a_m - self.state.a_bias

        # 旋转更新
        delta_theta = w_corrected * dt
        rotation_angle = norm(delta_theta)
        if rotation_angle > 1e-10:
            delta_q = Quaternion(axis=delta_theta / rotation_angle,
                                 angle=rotation_angle)
        else:
            delta_q = Quaternion(1, 0.5 * delta_theta[0],
                                 0.5 * delta_theta[1], 0.5 * delta_theta[2])
        self.state.q_wi = self.state.q_wi * delta_q
        self.state.q_wi = self.state.q_wi.normalised

        # 速度更新
        R_wi = self.state.q_wi.rotation_matrix
        a_world = R_wi @ a_corrected + self.gravity
        self.state.v_wi += a_world * dt

        # 位置更新
        self.state.p_wi += self.state.v_wi * dt + 0.5 * a_world * dt ** 2

        # 误差协方差预测
        F = self.compute_F_matrix(imu_data, dt)
        self.P = F @ self.P @ F.T + self.Q

    def update_with_vision_model(self, camera_pose: Dict[str, np.ndarray]):
        """更新步骤：使用视觉大模型的位姿观测"""
        # 提取观测值
        p_vo = camera_pose['position']
        q_vo = camera_pose['orientation']
        if isinstance(q_vo, Quaternion):
            R_vo = q_vo.rotation_matrix
        else:
            R_vo = q_vo

        # 从当前状态计算预测的相机位姿
        R_wi = self.state.q_wi.rotation_matrix
        p_wi = self.state.p_wi
        R_ic = self.state.q_ic.rotation_matrix
        p_ic = self.state.p_ic

        R_vo_pred = R_wi @ R_ic
        p_vo_pred = p_wi + R_wi @ p_ic

        # 计算观测残差 (在切空间中)
        delta_R = R_vo @ R_vo_pred.T
        delta_theta = 2 * np.array([delta_R[2, 1] - delta_R[1, 2],
                                    delta_R[0, 2] - delta_R[2, 0],
                                    delta_R[1, 0] - delta_R[0, 1]])
        delta_p = p_vo - p_vo_pred

        error_vec = np.concatenate([delta_p, delta_theta])

        # 计算观测矩阵 H (6x21)
        H = np.zeros((6, 21))

        # 位置误差部分
        H[0:3, 0:3] = np.eye(3)  # δp_wi
        H[0:3, 6:9] = -self.skew_symmetric(R_wi @ p_ic)  # δθ_wi
        H[0:3, 15:18] = R_wi  # δp_ic

        # 旋转误差部分
        H[3:6, 6:9] = np.eye(3)  # δθ_wi
        H[3:6, 18:21] = R_wi  # δθ_ic

        # 卡尔曼增益
        S = H @ self.P @ H.T + self.R_vo
        K = self.P @ H.T @ np.linalg.inv(S)

        # 更新误差状态和协方差
        delta_x = K @ error_vec
        self.P = (np.eye(21) - K @ H) @ self.P

        # 注入误差状态到名义状态
        self.inject_error_state(delta_x)

    def inject_error_state(self, delta_x: np.ndarray):
        """将误差状态注入到名义状态"""
        # 注入位置、速度、零偏误差
        self.state.p_wi += delta_x[0:3]
        self.state.v_wi += delta_x[3:6]
        self.state.a_bias += delta_x[9:12]
        self.state.w_bias += delta_x[12:15]
        self.state.p_ic += delta_x[15:18]

        # 注入旋转误差 (使用指数映射)
        delta_theta_wi = delta_x[6:9]
        delta_theta_ic = delta_x[18:21]

        if norm(delta_theta_wi) > 1e-10:
            delta_q_wi = Quaternion(axis=delta_theta_wi / norm(delta_theta_wi),
                                    angle=norm(delta_theta_wi))
            self.state.q_wi = delta_q_wi * self.state.q_wi
            self.state.q_wi.normalise()

        if norm(delta_theta_ic) > 1e-10:
            delta_q_ic = Quaternion(axis=delta_theta_ic / norm(delta_theta_ic),
                                    angle=norm(delta_theta_ic))
            self.state.q_ic = delta_q_ic * self.state.q_ic
            self.state.q_ic.normalise()

    def process_data(self, imu_data: Dict[str, np.ndarray],
                     vision_pose: Optional[Dict[str, np.ndarray]] = None,
                     dt: float = 0.01):
        """处理输入数据流"""
        # 存储IMU数据
        self.imu_buffer.append(imu_data)

        # 预测步骤
        self.predict(imu_data, dt)

        # 如果有视觉位姿数据，进行更新
        if vision_pose is not None:
            self.vo_buffer.append(vision_pose)
            self.update_with_vision_model(vision_pose)

        # 保存当前位姿
        current_pose = self.state.get_camera_pose()
        self.pose_history.append(current_pose)

        return current_pose

    def visualize_trajectory(self):
        """可视化轨迹"""
        positions = [pose[0] for pose in self.pose_history]
        positions = np.array(positions)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'b-', label='Fused Trajectory', linewidth=2)
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                   c='green', s=100, label='Start', marker='o')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                   c='red', s=100, label='End', marker='x')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Camera Trajectory from ESKF Fusion')
        ax.legend()
        ax.grid(True)

        plt.show()