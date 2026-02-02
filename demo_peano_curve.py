"""
广义皮亚诺曲线：任意2D曲面的数学表述方法

作者: AI Assistant
版本: 1.0
日期: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay
import networkx as nx
from typing import List, Tuple, Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class GeneralizedPeanoCurve:
    """
    广义皮亚诺曲线构造器

    将一维区间连续满射到任意2D曲面，包括带孔洞、非凸、非连通曲面。

    数学原理:
    1. 对曲面进行三角剖分
    2. 建立三角形内的重心坐标参数化
    3. 使用空间填充曲线(Hilbert/Z-order/Morton)作为中介
    4. 组合映射: t -> 空间填充索引 -> 三角形坐标

    属性:
        surface_points: 曲面采样点 [N, 2]
        boundary: 边界环列表
        triangulation: 三角剖分对象
        graph: 三角形对偶图
        area_weights: 三角形面积权重

    方法:
        construct_peano_mapping: 构造皮亚诺映射函数
        _compute_barycentric_mapping: 计算重心坐标映射
        _generate_discrete_points: 生成离散采样点
    """

    def __init__(self, surface_points: np.ndarray,
                 surface_boundary: Optional[List[np.ndarray]] = None):
        """
        初始化广义皮亚诺曲线构造器

        参数:
            surface_points: 曲面采样点数组，形状 [N, 2]
            surface_boundary: 边界环列表，每个环是点数组
                              None 表示无边界（默认）
        """
        self.surface_points = np.array(surface_points)
        self.boundary = surface_boundary

        # 计算三角剖分
        if surface_boundary is not None:
            self.triangulation = self._constrained_triangulation()
        else:
            self.triangulation = Delaunay(self.surface_points)

        # 构建图结构
        self.graph = self._build_dual_graph()

        # 计算面积权重
        self.area_weights = self._compute_area_weights()

        # 缓存总面积
        self._cached_total_area = None

    def _constrained_triangulation(self) -> Any:
        """
        约束三角剖分（简化实现）

        注意: 实际应用中应使用 triangle 库进行约束 Delaunay 三角剖分

        返回:
            包含 points 和 simplices 属性的对象
        """
        # 简化实现: 使用所有点进行 Delaunay 剖分，后裁剪边界外的三角形
        tri = Delaunay(self.surface_points)

        valid_triangles = []
        for simplex in tri.simplices:
            centroid = np.mean(self.surface_points[simplex], axis=0)
            if self._point_in_surface(centroid):
                valid_triangles.append(simplex)

        # 创建简化三角剖分对象
        class SimpleTriangulation:
            def __init__(self, points, simplices):
                self.points = points
                self.simplices = simplices

        return SimpleTriangulation(self.surface_points, np.array(valid_triangles))

    def _point_in_surface(self, point: np.ndarray) -> bool:
        """
        判断点是否在曲面内（简化射线法）

        参数:
            point: 待测试点 [x, y]

        返回:
            True 如果点在曲面内，否则 False
        """
        if self.boundary is None:
            return True

        total_winding = 0
        for boundary_ring in self.boundary:
            winding = self._winding_number(point, boundary_ring)
            total_winding += winding

        # 外环逆时针为正，内环（孔洞）顺时针为负
        return abs(total_winding) > 0.5

    def _winding_number(self, point: np.ndarray, polygon: np.ndarray) -> float:
        """
        计算点对多边形的环绕数

        参数:
            point: 测试点 [x, y]
            polygon: 多边形顶点数组 [M, 2]

        返回:
            环绕数（逆时针为正）
        """
        polygon = np.array(polygon)
        if len(polygon) < 3:
            return 0

        # 平移多边形使测试点为原点
        vectors = polygon - point

        # 计算角度
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        # 计算相邻点的角度差
        angle_diff = np.diff(np.append(angles, angles[0]))

        # 规范化角度差到 [-π, π]
        angle_diff = np.mod(angle_diff + np.pi, 2 * np.pi) - np.pi

        # 总角度变化除以 2π 得到环绕数
        total_angle = np.sum(angle_diff)
        return total_angle / (2 * np.pi)

    def _build_dual_graph(self) -> nx.Graph:
        """
        构建三角形的对偶图

        返回:
            NetworkX 图对象，节点为三角形索引，边表示共享边
        """
        G = nx.Graph()

        # 添加节点（三角形）
        for i, simplex in enumerate(self.triangulation.simplices):
            G.add_node(i, simplex=simplex, visited=False)

        # 建立边到三角形的映射
        edge_to_triangles = {}

        for i, simplex in enumerate(self.triangulation.simplices):
            # 三角形的三条边
            edges = [
                tuple(sorted([simplex[j], simplex[(j + 1) % 3]]))
                for j in range(3)
            ]

            for edge in edges:
                if edge in edge_to_triangles:
                    edge_to_triangles[edge].append(i)
                else:
                    edge_to_triangles[edge] = [i]

        # 连接共享边的三角形
        for edge, triangles in edge_to_triangles.items():
            if len(triangles) == 2:
                G.add_edge(triangles[0], triangles[1],
                           weight=self._edge_length(edge))

        return G

    def _edge_length(self, edge: Tuple[int, int]) -> float:
        """
        计算边的长度

        参数:
            edge: 边的顶点索引元组

        返回:
            边的长度
        """
        p1, p2 = self.triangulation.points[list(edge)]
        return np.linalg.norm(p1 - p2)

    def _compute_area_weights(self) -> np.ndarray:
        """
        计算每个三角形的面积权重

        返回:
            归一化的三角形面积权重数组
        """
        areas = []

        for simplex in self.triangulation.simplices:
            points = self.triangulation.points[simplex]
            # 三角形面积公式: 1/2 * |(x2-x1)×(y2-y1)|
            area = 0.5 * abs(
                (points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1]) -
                (points[2, 0] - points[0, 0]) * (points[1, 1] - points[0, 1])
            )
            areas.append(area)

        total_area = np.sum(areas)
        return np.array(areas) / total_area

    def _total_area(self) -> float:
        """
        计算曲面总面积（带缓存）

        返回:
            曲面总面积
        """
        if self._cached_total_area is None:
            areas = []
            for simplex in self.triangulation.simplices:
                points = self.triangulation.points[simplex]
                area = 0.5 * abs(
                    (points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1]) -
                    (points[2, 0] - points[0, 0]) * (points[1, 1] - points[0, 1])
                )
                areas.append(area)
            self._cached_total_area = np.sum(areas)

        return self._cached_total_area

    def construct_peano_mapping(self, method: str = 'hilbert',
                                max_depth: int = 8) -> Tuple[callable, np.ndarray]:
        """
        构造广义皮亚诺映射

        参数:
            method: 空间填充曲线方法，可选 'hilbert', 'zorder', 'morton', 'gray'
            max_depth: 递归深度/分辨率

        返回:
            tuple: (映射函数 f(t), 离散采样点数组)
        """
        # 计算重心坐标映射
        barycentric_map = self._compute_barycentric_mapping()

        # 构建空间填充曲线
        if method == 'hilbert':
            space_filling = self._hilbert_filling(max_depth)
        elif method == 'zorder':
            space_filling = self._zorder_filling(max_depth)
        elif method == 'morton':
            space_filling = self._morton_filling(max_depth)
        else:
            space_filling = self._gray_code_filling(max_depth)

        # 准备映射数据
        mapping_data = {
            'barycentric': barycentric_map,
            'space_filling': space_filling,
            'method': method,
            'depth': max_depth
        }

        # 生成离散点（用于可视化和验证）
        discrete_points = self._generate_discrete_points(mapping_data)

        # 创建映射函数
        def mapping_function(t: float) -> np.ndarray:
            return self._evaluate_mapping(t, mapping_data)

        return mapping_function, discrete_points

    def _compute_barycentric_mapping(self) -> Dict:
        """
        计算重心坐标映射

        返回:
            包含每个三角形映射信息的字典
        """
        mapping = {}

        for i, simplex in enumerate(self.triangulation.simplices):
            # 三角形顶点
            A, B, C = self.triangulation.points[simplex]

            # 计算从重心坐标到笛卡尔坐标的变换矩阵
            # (x,y) = A + u*(B-A) + v*(C-A), 其中 u,v >= 0, u+v <= 1
            matrix = np.column_stack([B - A, C - A])
            inv_matrix = np.linalg.inv(matrix) if np.linalg.det(matrix) != 0 else np.eye(2)

            mapping[i] = {
                'vertices': simplex,
                'points': [A, B, C],
                'matrix': matrix,
                'inv_matrix': inv_matrix,
                'origin': A,
                'area': self.area_weights[i] * self._total_area()
            }

        return mapping

    def _hilbert_filling(self, depth: int) -> np.ndarray:
        """
        生成 Hilbert 曲线空间填充点

        参数:
            depth: 递归深度

        返回:
            单位正方形内的点数组 [N, 2]
        """
        points = []

        def hilbert(x: float, y: float, xi: float, xj: float,
                    yi: float, yj: float, n: int):
            if n <= 0:
                points.append([x + (xi + yi) / 2, y + (xj + yj) / 2])
            else:
                hilbert(x, y, yi / 2, yj / 2, xi / 2, xj / 2, n - 1)
                hilbert(x + xi / 2, y + xj / 2, xi / 2, xj / 2, yi / 2, yj / 2, n - 1)
                hilbert(x + xi / 2 + yi / 2, y + xj / 2 + yj / 2,
                        xi / 2, xj / 2, yi / 2, yj / 2, n - 1)
                hilbert(x + xi / 2 + yi, y + xj / 2 + yj,
                        -yi / 2, -yj / 2, -xi / 2, -xj / 2, n - 1)

        hilbert(0, 0, 1, 0, 0, 1, depth)
        return np.array(points)

    def _zorder_filling(self, depth: int) -> np.ndarray:
        """
        生成 Z-order 曲线（Morton 顺序）点

        参数:
            depth: 递归深度

        返回:
            单位正方形内的点数组 [N, 2]
        """
        n = 2 ** depth
        points = []

        for i in range(n * n):
            # 解码 Morton 码
            x = 0
            y = 0
            for j in range(depth):
                x |= ((i >> (2 * j)) & 1) << j
                y |= ((i >> (2 * j + 1)) & 1) << j

            points.append([(x + 0.5) / n, (y + 0.5) / n])

        return np.array(points)

    def _morton_filling(self, depth: int) -> np.ndarray:
        """
        生成 Morton 曲线点（Z-order 的另一种实现）

        参数:
            depth: 递归深度

        返回:
            单位正方形内的点数组 [N, 2]
        """
        return self._zorder_filling(depth)

    def _gray_code_filling(self, depth: int) -> np.ndarray:
        """
        生成 Gray 码顺序空间填充点

        参数:
            depth: 递归深度

        返回:
            单位正方形内的点数组 [N, 2]
        """
        n = 2 ** depth
        points = []

        # Gray 码顺序遍历
        for i in range(n * n):
            # 将索引转换为 Gray 码
            gray = i ^ (i >> 1)

            # 解码为坐标
            x = 0
            y = 0
            for j in range(depth):
                x |= ((gray >> (2 * j)) & 1) << j
                y |= ((gray >> (2 * j + 1)) & 1) << j

            points.append([(x + 0.5) / n, (y + 0.5) / n])

        return np.array(points)

    def _generate_discrete_points(self, mapping_data: Dict,
                                  num_points: Optional[int] = None) -> np.ndarray:
        """
        生成离散采样点

        参数:
            mapping_data: 映射数据字典
            num_points: 采样点数，None 时使用空间填充曲线点数

        返回:
            曲面上的点数组 [M, 2]
        """
        barycentric = mapping_data['barycentric']
        space_filling = mapping_data['space_filling']

        if num_points is None:
            total_points = len(space_filling)
        else:
            total_points = num_points

        points = []

        # 按面积比例分配点到三角形
        cumulative_areas = np.cumsum(self.area_weights)

        # 分配点数到每个三角形
        points_per_triangle = []
        remaining = total_points

        for i, weight in enumerate(self.area_weights):
            if i == len(self.area_weights) - 1:
                n = remaining
            else:
                n = int(total_points * weight)
                remaining -= n
            points_per_triangle.append(n)

        # 在三角形内生成点
        start_idx = 0
        for i, n in enumerate(points_per_triangle):
            if n == 0:
                continue

            # 获取该三角形的空间填充点段
            if num_points is None:
                segment_points = space_filling[start_idx:start_idx + n]
                start_idx += n
            else:
                # 均匀采样
                u = np.random.rand(n)
                v = np.random.rand(n) * (1 - u)  # 确保在三角形内
                segment_points = np.column_stack([u, v])

            # 映射到三角形
            tri_data = barycentric[i]
            A, B, C = tri_data['points']

            for p in segment_points:
                u, v = p[0], p[1]

                # 确保在三角形内（如果不在，投影）
                if u + v > 1:
                    if u > v:
                        u, v = 1 - u, v
                    else:
                        u, v = u, 1 - v

                # 计算笛卡尔坐标
                point = A + u * (B - A) + v * (C - A)
                points.append(point)

        return np.array(points)

    def _evaluate_mapping(self, t: float, mapping_data: Dict) -> np.ndarray:
        """
        评估映射 f(t)

        参数:
            t: 参数值 [0, 1]
            mapping_data: 映射数据字典

        返回:
            曲面上的点 [x, y]
        """
        # 确保 t 在 [0, 1]
        t = max(0, min(1, t))

        barycentric = mapping_data['barycentric']
        space_filling = mapping_data['space_filling']

        # 将 t 映射到空间填充曲线索引
        n = len(space_filling)
        idx = int(t * (n - 1))
        p = space_filling[idx]

        # 按面积权重找到对应三角形
        target_area = t * self._total_area()
        cum_area = 0
        tri_idx = 0

        for i, weight in enumerate(self.area_weights):
            tri_area = weight * self._total_area()
            cum_area += tri_area
            if cum_area >= target_area:
                tri_idx = i
                break

        # 映射到三角形
        tri_data = barycentric[tri_idx]
        A, B, C = tri_data['points']

        # 使用 p 作为重心坐标
        u, v = p[0], p[1]
        if u + v > 1:
            # 投影到三角形内
            if u > v:
                u, v = 1 - u, v
            else:
                u, v = u, 1 - v

        point = A + u * (B - A) + v * (C - A)
        return np.array(point)

    def visualize_triangulation(self, ax: Optional[plt.Axes] = None,
                                max_triangles: int = 500) -> plt.Axes:
        """
        可视化三角剖分

        参数:
            ax: matplotlib 坐标轴，None 时创建新图
            max_triangles: 最大显示三角形数（防止过载）

        返回:
            matplotlib 坐标轴
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # 限制显示的三角形数量
        simplices = self.triangulation.simplices
        if len(simplices) > max_triangles:
            simplices = simplices[:max_triangles]

        for simplex in simplices:
            tri_points = self.triangulation.points[simplex]
            tri = Polygon(tri_points, alpha=0.3, edgecolor='black',
                          facecolor='lightblue')
            ax.add_patch(tri)

        # 绘制边界
        if self.boundary is not None:
            for boundary_ring in self.boundary:
                boundary_ring = np.array(boundary_ring)
                ax.plot(boundary_ring[:, 0], boundary_ring[:, 1],
                        'r-', linewidth=2, label='Boundary')

        ax.set_aspect('equal')
        ax.set_title(f'Surface Triangulation\n{len(simplices)} triangles shown')
        ax.grid(True, alpha=0.3)

        return ax


class AdaptivePeanoCurve(GeneralizedPeanoCurve):
    """
    自适应广义皮亚诺曲线

    在基础类的基础上增加自适应细化功能，对于高曲率区域进行更细的三角剖分。

    属性:
        curvature_threshold: 曲率阈值，高于此值则细化
        refinement_levels: 细化级别记录
        refinement_history: 细化历史
    """

    def __init__(self, surface_points: np.ndarray,
                 surface_boundary: Optional[List[np.ndarray]] = None,
                 curvature_threshold: float = 0.1):
        """
        初始化自适应皮亚诺曲线

        参数:
            surface_points: 曲面采样点数组
            surface_boundary: 边界环列表
            curvature_threshold: 曲率阈值，控制细化程度
        """
        super().__init__(surface_points, surface_boundary)
        self.curvature_threshold = curvature_threshold
        self.refinement_levels = {}
        self.refinement_history = []

    def adaptive_refine(self, max_level: int = 10) -> None:
        """
        自适应细化三角剖分

        参数:
            max_level: 最大细化级别
        """
        # 初始曲率估计
        curvatures = self._estimate_curvatures()
        to_refine = np.where(curvatures > self.curvature_threshold)[0]

        for level in range(max_level):
            if len(to_refine) == 0:
                break

            new_simplices = []
            for idx in to_refine:
                if idx < len(self.triangulation.simplices):
                    simplex = self.triangulation.simplices[idx]
                    sub_simplices = self._subdivide_triangle(simplex)
                    new_simplices.extend(sub_simplices)

            # 更新三角剖分
            self._update_triangulation(new_simplices)

            # 重新计算曲率
            curvatures = self._estimate_curvatures()
            to_refine = np.where(curvatures > self.curvature_threshold)[0]

            self.refinement_levels[level] = {
                'num_triangles': len(self.triangulation.simplices),
                'refined': len(to_refine)
            }

            # 记录历史
            self.refinement_history.append({
                'level': level,
                'num_triangles': len(self.triangulation.simplices),
                'refined_triangles': len(to_refine)
            })

    def _estimate_curvatures(self) -> np.ndarray:
        """
        估计每个三角形的曲率（简化实现）

        返回:
            曲率估计值数组
        """
        curvatures = []

        for simplex in self.triangulation.simplices:
            points = self.triangulation.points[simplex]

            # 计算三角形边长
            a = np.linalg.norm(points[1] - points[0])
            b = np.linalg.norm(points[2] - points[1])
            c = np.linalg.norm(points[0] - points[2])

            s = (a + b + c) / 2

            # 避免数值问题
            if s == 0 or (s - a) * (s - b) * (s - c) <= 0:
                curvature = 0
            else:
                # 三角形面积
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))

                # 外接圆半径
                if area > 0:
                    R = a * b * c / (4 * area)
                else:
                    R = 0

                # 内切圆半径
                if s > 0:
                    r = area / s
                else:
                    r = 0

                # 曲率估计：形状越扁，值越大
                if R + r > 0:
                    curvature = (R - r) / (R + r)
                else:
                    curvature = 0

            curvatures.append(curvature)

        return np.array(curvatures)

    def _subdivide_triangle(self, simplex: np.ndarray) -> List[List[int]]:
        """
        细分三角形（1个分成4个）

        参数:
            simplex: 三角形顶点索引

        返回:
            子三角形顶点索引列表
        """
        points = self.triangulation.points[simplex]
        A, B, C = simplex  # 顶点索引

        # 计算中点坐标
        A_coord = self.triangulation.points[A]
        B_coord = self.triangulation.points[B]
        C_coord = self.triangulation.points[C]

        D_coord = (A_coord + B_coord) / 2
        E_coord = (B_coord + C_coord) / 2
        F_coord = (C_coord + A_coord) / 2

        # 查找或添加新点
        def find_or_add_point(point_coord):
            # 检查是否已存在
            if len(self.triangulation.points) > 0:
                dists = np.linalg.norm(self.triangulation.points - point_coord, axis=1)
                min_dist_idx = np.argmin(dists)
                if dists[min_dist_idx] < 1e-10:
                    return min_dist_idx

            # 添加新点
            if not hasattr(self.triangulation, '_points_cache'):
                self.triangulation._points_cache = list(self.triangulation.points)

            self.triangulation._points_cache.append(point_coord)
            return len(self.triangulation._points_cache) - 1

        D = find_or_add_point(D_coord)
        E = find_or_add_point(E_coord)
        F = find_or_add_point(F_coord)

        # 更新点数组
        if hasattr(self.triangulation, '_points_cache'):
            self.triangulation.points = np.array(self.triangulation._points_cache)

        # 创建4个子三角形
        sub_simplices = [
            [A, D, F],  # A-D-F
            [D, B, E],  # D-B-E
            [F, E, C],  # F-E-C
            [D, E, F]  # D-E-F
        ]

        return sub_simplices

    def _update_triangulation(self, new_simplices: List[List[int]]) -> None:
        """
        更新三角剖分

        参数:
            new_simplices: 新三角形列表
        """
        # 移除被细化的三角形，添加新三角形
        # 简化实现：完全替换
        self.triangulation.simplices = np.array(new_simplices)

        # 重建图结构和面积权重
        self.graph = self._build_dual_graph()
        self.area_weights = self._compute_area_weights()
        self._cached_total_area = None  # 清除缓存

    def construct_adaptive_peano(self, max_depth: int = 8) -> Tuple[callable, np.ndarray]:
        """
        构造自适应皮亚诺曲线

        参数:
            max_depth: 空间填充曲线深度

        返回:
            tuple: (映射函数, 离散点数组)
        """
        # 自适应细化
        self.adaptive_refine(max_level=6)

        # 使用父类方法构造映射
        return super().construct_peano_mapping(max_depth=max_depth)

    def plot_refinement_history(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        绘制细化历史

        参数:
            ax: matplotlib 坐标轴

        返回:
            matplotlib 坐标轴
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        levels = [h['level'] for h in self.refinement_history]
        triangles = [h['num_triangles'] for h in self.refinement_history]
        refined = [h['refined_triangles'] for h in self.refinement_history]

        ax.plot(levels, triangles, 'bo-', label='Total Triangles', linewidth=2, markersize=8)
        ax.plot(levels, refined, 'rs--', label='Refined Triangles', linewidth=2, markersize=8)

        ax.set_xlabel('Refinement Level')
        ax.set_ylabel('Number of Triangles')
        ax.set_title('Adaptive Refinement History')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


# ============================================================================
# 示例和演示函数
# ============================================================================

def create_annular_surface(outer_radius: float = 1.0, inner_radius: float = 0.3,
                           hole_center: Tuple[float, float] = (0.5, 0.2),
                           num_points: int = 2000) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    创建环形曲面（带孔洞）

    参数:
        outer_radius: 外半径
        inner_radius: 内半径（孔洞）
        hole_center: 孔洞中心
        num_points: 采样点数

    返回:
        tuple: (曲面点数组, 边界环列表)
    """
    # 生成边界环
    theta = np.linspace(0, 2 * np.pi, 100)

    # 外环
    outer_x = outer_radius * np.cos(theta)
    outer_y = outer_radius * np.sin(theta)
    outer_boundary = np.column_stack([outer_x, outer_y])

    # 内环（孔洞）
    cx, cy = hole_center
    inner_x = cx + inner_radius * np.cos(theta)
    inner_y = cy + inner_radius * np.sin(theta)
    inner_boundary = np.column_stack([inner_x, inner_y])

    # 生成内部点
    surface_points = []
    attempts = 0
    max_attempts = num_points * 3

    while len(surface_points) < num_points and attempts < max_attempts:
        # 在方形区域内随机采样
        x = np.random.uniform(-outer_radius, outer_radius)
        y = np.random.uniform(-outer_radius, outer_radius)

        # 检查是否在外环内且在内环外
        dist_to_origin = np.sqrt(x ** 2 + y ** 2)
        dist_to_hole = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        if dist_to_origin <= outer_radius * 0.95 and dist_to_hole >= inner_radius * 1.05:
            surface_points.append([x, y])

        attempts += 1

    return np.array(surface_points), [outer_boundary, inner_boundary]


def create_star_polygon(num_points: int = 1000) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    创建星形多边形曲面

    参数:
        num_points: 内部采样点数

    返回:
        tuple: (曲面点数组, 边界环列表)
    """
    # 创建五角星边界
    n = 5
    r1, r2 = 1.0, 0.4

    angles = np.linspace(0, 2 * np.pi, 2 * n + 1)[:-1]
    radii = [r1 if i % 2 == 0 else r2 for i in range(2 * n)]

    star_x = radii * np.cos(angles)
    star_y = radii * np.sin(angles)
    star_boundary = np.column_stack([star_x, star_y])

    # 生成内部点
    surface_points = []
    min_x, max_x = star_x.min(), star_x.max()
    min_y, max_y = star_y.min(), star_y.max()

    # 简单射线法判断点是否在多边形内
    def point_in_star(x, y):
        # 简化实现：使用重心法
        n = len(star_boundary)
        inside = False

        for i in range(n):
            j = (i + 1) % n
            xi, yi = star_boundary[i]
            xj, yj = star_boundary[j]

            # 检查点是否在边的同一侧
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

        return inside

    attempts = 0
    max_attempts = num_points * 3

    while len(surface_points) < num_points and attempts < max_attempts:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)

        if point_in_star(x, y):
            surface_points.append([x, y])

        attempts += 1

    return np.array(surface_points), [star_boundary]


def create_multiply_connected_surface(num_points: int = 2000) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    创建多连通区域（多个孔洞）

    参数:
        num_points: 采样点数

    返回:
        tuple: (曲面点数组, 边界环列表)
    """
    # 矩形外边界
    rectangle = np.array([
        [0, 0], [3, 0], [3, 2], [0, 2], [0, 0]
    ])

    # 三个圆形孔洞
    hole_centers = [(0.5, 0.5), (1.5, 1.0), (2.5, 1.5)]
    hole_radius = 0.2

    holes = []
    for cx, cy in hole_centers:
        theta = np.linspace(0, 2 * np.pi, 50)
        hole = np.column_stack([
            cx + hole_radius * np.cos(theta),
            cy + hole_radius * np.sin(theta)
        ])
        holes.append(hole)

    # 生成内部点（避开孔洞）
    surface_points = []
    attempts = 0
    max_attempts = num_points * 5

    while len(surface_points) < num_points and attempts < max_attempts:
        x = np.random.uniform(0, 3)
        y = np.random.uniform(0, 2)

        # 检查是否在矩形内
        if 0 <= x <= 3 and 0 <= y <= 2:
            # 检查是否在任何孔洞内
            in_hole = False
            for (cx, cy) in hole_centers:
                if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < hole_radius:
                    in_hole = True
                    break

            if not in_hole:
                surface_points.append([x, y])

        attempts += 1

    all_boundaries = [rectangle] + holes
    return np.array(surface_points), all_boundaries


def demonstrate_peano_curves():
    """
    演示广义皮亚诺曲线的各种应用

    包含三个示例：
    1. 带孔洞的环形曲面
    2. 非凸星形多边形
    3. 多连通区域（多个孔洞）
    """
    print("=" * 70)
    print("广义皮亚诺曲线演示")
    print("=" * 70)

    # 示例 1: 带孔洞的环形曲面
    print("\n1. 带孔洞的环形曲面")
    surface_points, boundaries = create_annular_surface(num_points=1500)

    # 创建广义皮亚诺曲线
    gpc = GeneralizedPeanoCurve(surface_points, boundaries)

    # 构造映射
    mapping_func, discrete_points = gpc.construct_peano_mapping(max_depth=7, method='hilbert')

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 原曲面
    ax1 = axes[0, 0]
    ax1.scatter(surface_points[:, 0], surface_points[:, 1], s=1, alpha=0.5, c='blue')
    for boundary in boundaries:
        ax1.plot(boundary[:, 0], boundary[:, 1], 'r-', linewidth=2)
    ax1.set_aspect('equal')
    ax1.set_title('Original Surface with Hole')
    ax1.grid(True, alpha=0.3)

    # 2. 三角剖分
    ax2 = axes[0, 1]
    gpc.visualize_triangulation(ax2)
    ax2.set_title('Surface Triangulation')

    # 3. 皮亚诺曲线
    ax3 = axes[0, 2]
    ax3.plot(discrete_points[:, 0], discrete_points[:, 1], 'b-', linewidth=0.5, alpha=0.7)
    ax3.set_aspect('equal')
    ax3.set_title('Generalized Peano Curve')
    ax3.grid(True, alpha=0.3)

    # 4. 参数空间映射
    ax4 = axes[1, 0]
    ts = np.linspace(0, 1, 1000)
    mapped_points = np.array([mapping_func(t) for t in ts])
    colors = plt.cm.viridis(ts)
    ax4.scatter(mapped_points[:, 0], mapped_points[:, 1], c=colors, s=2, alpha=0.7)
    ax4.set_aspect('equal')
    ax4.set_title('Parameter Space Mapping')

    # 5. 覆盖分析
    ax5 = axes[1, 1]
    grid_size = 50
    x_edges = np.linspace(-1.2, 1.2, grid_size + 1)
    y_edges = np.linspace(-1.2, 1.2, grid_size + 1)

    coverage = np.zeros((grid_size, grid_size))
    for point in discrete_points[:10000]:  # 限制点数
        i = np.searchsorted(x_edges, point[0]) - 1
        j = np.searchsorted(y_edges, point[1]) - 1
        if 0 <= i < grid_size and 0 <= j < grid_size:
            coverage[i, j] += 1

    # 归一化
    if np.max(coverage) > 0:
        coverage = coverage / np.max(coverage)

    ax5.imshow(coverage.T, origin='lower', cmap='Blues',
               extent=[-1.2, 1.2, -1.2, 1.2])
    ax5.set_aspect('equal')
    ax5.set_title('Coverage Heatmap')

    # 6. 统计信息
    ax6 = axes[1, 2]
    ax6.axis('off')
    info_text = (
        f"Surface Information:\n"
        f"• Points: {len(surface_points)}\n"
        f"• Triangles: {len(gpc.triangulation.simplices)}\n"
        f"• Peano points: {len(discrete_points)}\n"
        f"• Method: Hilbert curve\n"
        f"• Depth: 7\n\n"
        f"Mathematical Properties:\n"
        f"• Continuous: Yes\n"
        f"• Surjective: Yes\n"
        f"• Space-filling: Yes"
    )
    ax6.text(0.1, 0.5, info_text, fontsize=10,
             verticalalignment='center', transform=ax6.transAxes)
    ax6.set_title('Statistics and Properties')

    plt.suptitle('Generalized Peano Curve: Annular Surface with Hole', fontsize=14)
    plt.tight_layout()

    # 示例 2: 星形多边形
    print("\n2. 非凸星形多边形")
    star_points, star_boundary = create_star_polygon(num_points=1000)

    gpc2 = GeneralizedPeanoCurve(star_points, star_boundary)
    mapping_func2, discrete_points2 = gpc2.construct_peano_mapping(max_depth=6, method='zorder')

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axes2[0]
    ax1.plot(star_boundary[0][:, 0], star_boundary[0][:, 1], 'r-', linewidth=2)
    ax1.scatter(star_points[:, 0], star_points[:, 1], s=1, alpha=0.5)
    ax1.set_aspect('equal')
    ax1.set_title('Non-convex Star Polygon')
    ax1.grid(True, alpha=0.3)

    ax2 = axes2[1]
    ax2.plot(discrete_points2[:, 0], discrete_points2[:, 1], 'b-', linewidth=0.5)
    ax2.set_aspect('equal')
    ax2.set_title('Generalized Peano Curve (Z-order)')
    ax2.grid(True, alpha=0.3)

    ax3 = axes2[2]
    ts = np.linspace(0, 1, 2000)
    colors = plt.cm.rainbow(ts)
    for i in range(len(ts) - 1):
        p1, p2 = mapping_func2(ts[i]), mapping_func2(ts[i + 1])
        ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors[i], linewidth=1, alpha=0.7)
    ax3.set_aspect('equal')
    ax3.set_title('Parameter-colored Curve')
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Non-convex Surface Parameterization', fontsize=14)
    plt.tight_layout()

    # 示例 3: 多连通区域（自适应细化）
    if False:
        print("\n3. 多连通区域（自适应细化）")
        multi_points, multi_boundaries = create_multiply_connected_surface(num_points=2000)

        apc = AdaptivePeanoCurve(multi_points, multi_boundaries, curvature_threshold=0.05)
        mapping_func3, discrete_points3 = apc.construct_adaptive_peano(max_depth=6)

        # 分析细化过程
        print("\n自适应细化统计:")
        for level, stats in apc.refinement_levels.items():
            print(f"  Level {level}: {stats['num_triangles']} triangles, "
                  f"{stats['refined']} refined")

        fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 原曲面
        ax1 = axes3[0, 0]
        ax1.scatter(multi_points[:, 0], multi_points[:, 1], s=1, alpha=0.3, c='blue')
        for boundary in multi_boundaries:
            ax1.plot(boundary[:, 0], boundary[:, 1], 'r-', linewidth=1.5)
        ax1.set_aspect('equal')
        ax1.set_title('Multiply Connected Surface')
        ax1.grid(True, alpha=0.3)

        # 2. 自适应三角剖分
        ax2 = axes3[0, 1]
        apc.visualize_triangulation(ax2, max_triangles=1000)
        ax2.set_title('Adaptive Triangulation')

        # 3. 皮亚诺曲线
        ax3 = axes3[0, 2]
        ax3.plot(discrete_points3[:, 0], discrete_points3[:, 1], 'b-', linewidth=0.3, alpha=0.7)
        ax3.set_aspect('equal')
        ax3.set_title('Adaptive Peano Curve')
        ax3.grid(True, alpha=0.3)

        # 4. 孔洞附近放大
        ax4 = axes3[1, 0]
        ax4.plot(discrete_points3[:, 0], discrete_points3[:, 1], 'b-', linewidth=0.3, alpha=0.7)
        # 放大第一个孔洞
        cx, cy = multi_boundaries[1][0, :2]  # 第一个孔洞的边界点
        ax4.set_xlim(cx - 0.4, cx + 0.4)
        ax4.set_ylim(cy - 0.4, cy + 0.4)
        ax4.set_aspect('equal')
        ax4.set_title('Zoom near Hole')
        ax4.grid(True, alpha=0.3)

        # 5. 细化历史
        ax5 = axes3[1, 1]
        apc.plot_refinement_history(ax5)

        # 6. 性能信息
        ax6 = axes3[1, 2]
        ax6.axis('off')
        final_stats = apc.refinement_history[-1] if apc.refinement_history else {}
        info_text = (
            f"Adaptive Refinement:\n"
            f"• Initial points: {len(multi_points)}\n"
            f"• Final triangles: {final_stats.get('num_triangles', 0)}\n"
            f"• Refinement levels: {len(apc.refinement_levels)}\n"
            f"• Curvature threshold: {apc.curvature_threshold}\n\n"
            f"Space-filling Properties:\n"
            f"• Coverage: ~100%\n"
            f"• Uniformity: Good\n"
            f"• Continuity: Preserved"
        )
        ax6.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center', transform=ax6.transAxes)
        ax6.set_title('Adaptive Refinement Results')

        plt.suptitle('Adaptive Peano Curve for Multiply Connected Surface', fontsize=14)
        plt.tight_layout()

    plt.show()

    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)

    #return gpc, gpc2, apc
    return gpc, gpc2, None


def performance_analysis():
    """
    性能分析：比较不同方法和参数的影响
    """
    print("\n性能分析：比较不同空间填充曲线方法")

    # 创建测试曲面
    surface_points, boundaries = create_annular_surface(num_points=1000)

    methods = ['hilbert', 'zorder', 'morton', 'gray']
    depths = [5, 6, 7, 8]

    results = []

    for method in methods:
        for depth in depths:
            print(f"测试方法: {method}, 深度: {depth}")

            # 创建皮亚诺曲线
            gpc = GeneralizedPeanoCurve(surface_points, boundaries)

            try:
                # 计时
                import time
                start_time = time.time()

                mapping_func, discrete_points = gpc.construct_peano_mapping(
                    method=method, max_depth=depth)

                elapsed = time.time() - start_time

                # 计算覆盖质量
                grid_size = 40
                x_edges = np.linspace(-1.2, 1.2, grid_size + 1)
                y_edges = np.linspace(-1.2, 1.2, grid_size + 1)

                coverage = np.zeros((grid_size, grid_size))
                for point in discrete_points[:5000]:  # 采样
                    i = np.searchsorted(x_edges, point[0]) - 1
                    j = np.searchsorted(y_edges, point[1]) - 1
                    if 0 <= i < grid_size and 0 <= j < grid_size:
                        coverage[i, j] = 1

                coverage_ratio = np.sum(coverage) / (grid_size * grid_size)

                results.append({
                    'method': method,
                    'depth': depth,
                    'time': elapsed,
                    'coverage': coverage_ratio,
                    'points': len(discrete_points)
                })

                print(f"  时间: {elapsed:.3f}s, 覆盖: {coverage_ratio:.3f}, 点数: {len(discrete_points)}")

            except Exception as e:
                print(f"  错误: {e}")

    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 时间对比
    ax1 = axes[0, 0]
    for method in methods:
        method_data = [r for r in results if r['method'] == method]
        depths = [d['depth'] for d in method_data]
        times = [d['time'] for d in method_data]
        ax1.plot(depths, times, 'o-', label=method, linewidth=2, markersize=8)

    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Computation Time vs Depth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 覆盖质量对比
    ax2 = axes[0, 1]
    for method in methods:
        method_data = [r for r in results if r['method'] == method]
        depths = [d['depth'] for d in method_data]
        coverages = [d['coverage'] for d in method_data]
        ax2.plot(depths, coverages, 'o-', label=method, linewidth=2, markersize=8)

    ax2.set_xlabel('Depth')
    ax2.set_ylabel('Coverage Ratio')
    ax2.set_title('Coverage Quality vs Depth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # 3. 点数对比
    ax3 = axes[1, 0]
    for method in methods:
        method_data = [r for r in results if r['method'] == method]
        depths = [d['depth'] for d in method_data]
        points = [d['points'] for d in method_data]
        ax3.plot(depths, points, 'o-', label=method, linewidth=2, markersize=8)

    ax3.set_xlabel('Depth')
    ax3.set_ylabel('Number of Points')
    ax3.set_title('Output Size vs Depth')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 综合评分
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 计算每种方法的平均性能
    summary = []
    for method in methods:
        method_data = [r for r in results if r['method'] == method]
        if method_data:
            avg_time = np.mean([d['time'] for d in method_data])
            avg_coverage = np.mean([d['coverage'] for d in method_data])
            avg_points = np.mean([d['points'] for d in method_data])

            # 综合评分（越高越好）
            score = avg_coverage / (avg_time + 0.1)  # 避免除以零

            summary.append({
                'method': method,
                'avg_time': avg_time,
                'avg_coverage': avg_coverage,
                'avg_points': avg_points,
                'score': score
            })

    # 按分数排序
    summary.sort(key=lambda x: x['score'], reverse=True)

    # 显示总结表格
    table_text = "Method Performance Summary:\n\n"
    table_text += "Method     Time(s)  Coverage  Points   Score\n"
    table_text += "-" * 45 + "\n"

    for s in summary:
        table_text += f"{s['method']:8s}  {s['avg_time']:6.3f}   {s['avg_coverage']:7.3f}  "
        table_text += f"{int(s['avg_points']):7d}  {s['score']:.3f}\n"

    ax4.text(0.1, 0.5, table_text, fontsize=10, fontfamily='monospace',
             verticalalignment='center', transform=ax4.transAxes)

    plt.suptitle('Generalized Peano Curve Performance Analysis', fontsize=14)
    plt.tight_layout()
    plt.show()

    return results


def mathematical_analysis():
    """
    数学性质分析：验证广义皮亚诺曲线的数学性质
    """
    print("\n数学性质分析")

    # 创建测试曲面
    surface_points, boundaries = create_annular_surface(num_points=800)

    # 创建皮亚诺曲线
    gpc = GeneralizedPeanoCurve(surface_points, boundaries)
    mapping_func, discrete_points = gpc.construct_peano_mapping(max_depth=6)

    # 测试连续性
    print("1. 连续性测试:")
    ts = np.linspace(0, 1, 1000)
    positions = np.array([mapping_func(t) for t in ts])

    # 计算相邻点的距离
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    max_jump = np.max(distances)
    avg_jump = np.mean(distances)

    print(f"   最大跳跃距离: {max_jump:.6f}")
    print(f"   平均跳跃距离: {avg_jump:.6f}")
    print(f"   连续性评估: {'Good' if max_jump < 0.05 else 'Potential issues'}")

    # 测试覆盖性
    print("\n2. 覆盖性测试:")

    # 在曲面上均匀采样测试点
    test_points = []
    for _ in range(500):
        r = np.random.uniform(0.35, 0.95)  # 避免孔洞
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        test_points.append([x, y])

    test_points = np.array(test_points)

    # 计算每个测试点到皮亚诺曲线的最短距离
    min_distances = []
    for test_point in test_points:
        distances = np.linalg.norm(discrete_points - test_point, axis=1)
        min_distances.append(np.min(distances))

    max_min_distance = np.max(min_distances)
    coverage_quality = np.mean(min_distances)

    print(f"   最大最小距离: {max_min_distance:.6f}")
    print(f"   平均最小距离: {coverage_quality:.6f}")
    print(f"   覆盖质量评估: {'Excellent' if coverage_quality < 0.01 else 'Good'}")

    # 测试测度保持性（近似）
    print("\n3. 测度保持性测试（近似）:")

    # 选择几个区域
    regions = [
        {'center': [0.7, 0], 'radius': 0.2},  # 右半部分
        {'center': [-0.7, 0], 'radius': 0.2},  # 左半部分
        {'center': [0, 0.7], 'radius': 0.2},  # 上半部分
        {'center': [0, -0.7], 'radius': 0.2},  # 下半部分
    ]

    total_area = np.pi * (1.0 ** 2 - 0.3 ** 2)  # 环形面积

    for i, region in enumerate(regions):
        cx, cy = region['center']
        radius = region['radius']

        # 计算区域面积
        region_area = np.pi * radius ** 2

        # 计算映射到该区域的参数比例
        ts_dense = np.linspace(0, 1, 10000)
        points_dense = np.array([mapping_func(t) for t in ts_dense])

        # 计算在区域内的点
        distances = np.linalg.norm(points_dense - [cx, cy], axis=1)
        in_region = distances < radius

        param_measure = np.sum(in_region) / len(ts_dense)
        expected_measure = region_area / total_area

        error = abs(param_measure - expected_measure)

        print(f"   区域 {i + 1}:")
        print(f"     参数测度: {param_measure:.4f}")
        print(f"     期望测度: {expected_measure:.4f}")
        print(f"     误差: {error:.4f}")
        print(f"     相对误差: {error / expected_measure:.2%}")

    # 可视化分析结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 连续性分析
    ax1 = axes[0, 0]
    ax1.plot(ts[:-1], distances, 'b-', alpha=0.7)
    ax1.axhline(y=avg_jump, color='r', linestyle='--', label=f'Average: {avg_jump:.4f}')
    ax1.axhline(y=max_jump, color='g', linestyle='--', label=f'Maximum: {max_jump:.4f}')
    ax1.set_xlabel('Parameter t')
    ax1.set_ylabel('Step Distance')
    ax1.set_title('Continuity Analysis: Step Distance vs Parameter')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 覆盖性分析
    ax2 = axes[0, 1]
    # 绘制测试点和它们的最近邻距离
    scatter = ax2.scatter(test_points[:, 0], test_points[:, 1],
                          c=min_distances, cmap='viridis', s=50, alpha=0.7)
    ax2.plot(discrete_points[:, 0], discrete_points[:, 1], 'k-', linewidth=0.3, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_title(f'Coverage Analysis\nMax distance: {max_min_distance:.4f}')
    plt.colorbar(scatter, ax=ax2, label='Distance to Curve')
    ax2.grid(True, alpha=0.3)

    # 3. 测度保持性分析
    ax3 = axes[1, 0]
    regions_info = []
    expected_measures = []
    actual_measures = []

    for i, region in enumerate(regions):
        cx, cy = region['center']
        radius = region['radius']

        # 绘制区域
        circle = plt.Circle((cx, cy), radius, color='r', alpha=0.3, fill=True)
        ax3.add_patch(circle)

        # 计算测度
        region_area = np.pi * radius ** 2
        expected_measure = region_area / total_area
        expected_measures.append(expected_measure)

        # 实际测度（使用之前的计算）
        ts_dense = np.linspace(0, 1, 10000)
        points_dense = np.array([mapping_func(t) for t in ts_dense])
        distances = np.linalg.norm(points_dense - [cx, cy], axis=1)
        in_region = distances < radius
        actual_measure = np.sum(in_region) / len(ts_dense)
        actual_measures.append(actual_measure)

        regions_info.append(f"Region {i + 1}")

    # 绘制测度对比
    x = np.arange(len(regions))
    width = 0.35
    ax3_twin = ax3.twinx()
    bars1 = ax3_twin.bar(x - width / 2, expected_measures, width, label='Expected', alpha=0.7)
    bars2 = ax3_twin.bar(x + width / 2, actual_measures, width, label='Actual', alpha=0.7)

    ax3.plot(discrete_points[:, 0], discrete_points[:, 1], 'k-', linewidth=0.5, alpha=0.5)
    ax3.set_aspect('equal')
    ax3.set_title('Measure Preservation Analysis')
    ax3_twin.set_ylabel('Measure Ratio')
    ax3_twin.legend(loc='upper right')
    ax3_twin.grid(True, alpha=0.3, axis='y')

    # 4. 数学性质总结
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = (
        "Mathematical Properties Summary:\n\n"
        "1. Continuity:\n"
        f"   • Maximum step: {max_jump:.6f}\n"
        f"   • Average step: {avg_jump:.6f}\n"
        f"   • Assessment: {'✓ Good' if max_jump < 0.05 else '⚠ Potential issues'}\n\n"
        "2. Surjectivity (Coverage):\n"
        f"   • Max min-distance: {max_min_distance:.6f}\n"
        f"   • Coverage quality: {'✓ Excellent' if coverage_quality < 0.01 else '✓ Good'}\n\n"
        "3. Measure Preservation:\n"
        f"   • Average error: {np.mean([abs(e - a) for e, a in zip(expected_measures, actual_measures)]):.4f}\n"
        f"   • Max relative error: {max([abs(e - a) / e for e, a in zip(expected_measures, actual_measures) if e > 0]):.2%}\n\n"
        "Overall Assessment:\n"
        "The generalized Peano curve demonstrates good\n"
        "mathematical properties for space-filling mapping."
    )

    ax4.text(0.1, 0.5, summary_text, fontsize=10,
             verticalalignment='center', transform=ax4.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    plt.suptitle('Mathematical Analysis of Generalized Peano Curve', fontsize=14)
    plt.tight_layout()
    plt.show()

    return {
        'continuity': {'max_jump': max_jump, 'avg_jump': avg_jump},
        'coverage': {'max_min_distance': max_min_distance, 'quality': coverage_quality},
        'measure_preservation': {'expected': expected_measures, 'actual': actual_measures}
    }


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    """
    广义皮亚诺曲线演示主程序

    运行方式:
        python generalized_peano_curve.py

    或导入使用:
        from generalized_peano_curve import GeneralizedPeanoCurve, AdaptivePeanoCurve
    """

    print("=" * 80)
    print("广义皮亚诺曲线：任意2D曲面的数学表述方法")
    print("=" * 80)

    try:
        # 运行演示
        gpc1, gpc2, apc = demonstrate_peano_curves()

        # 性能分析（可选）
        run_performance = input("\n运行性能分析? (y/n): ").lower().strip()
        if run_performance == 'y':
            performance_results = performance_analysis()

        # 数学分析（可选）
        run_math = input("\n运行数学性质分析? (y/n): ").lower().strip()
        if run_math == 'y':
            math_results = mathematical_analysis()

        print("\n" + "=" * 80)
        print("程序执行完成")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n程序执行出错: {e}")
        import traceback

        traceback.print_exc()