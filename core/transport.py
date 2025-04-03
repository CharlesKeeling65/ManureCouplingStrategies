"""
运输模型模块，提供各种粪肥养分运输策略的实现
"""

import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict, Optional, Union, Any
import sys

from ManureTransport.core.spatial import (
    DistanceStrategy,
    UnlimitedDistanceStrategy,
    SimpleDistanceStrategy,
    move_array,
    edges_line_func,
)


class TransportModel:
    """
    粪肥养分运输模型基类

    提供粪肥养分运输的基本功能，子类可以实现不同的传输策略
    """

    def __init__(
        self,
        distance_strategy: Optional[DistanceStrategy] = None,
        output_path: Optional[Path] = None,
        resolution: int = 1,
    ):
        """
        初始化运输模型

        Args:
            distance_strategy: 距离计算策略，如果为None则使用SimpleDistanceStrategy
            output_path: 输出路径，默认为None
            resolution: 栅格分辨率（千米），默认为1
        """
        self.distance_strategy = distance_strategy or SimpleDistanceStrategy()
        self.output_path = output_path
        self.resolution = resolution
        self.stats = {"distance": [], "surplus": [], "deficiency": []}

    def prepare_output_dir(self) -> None:
        """
        准备输出目录
        """
        if self.output_path is not None:
            self.output_path.mkdir(exist_ok=True, parents=True)

    def optimize_allocation(self, sd_raster: np.ndarray, distance: int) -> np.ndarray:
        """
        执行特定距离的分配优化

        Args:
            sd_raster: 盈余-需求栅格数据
            distance: 距离（单位：栅格单元）

        Returns:
            优化后的盈余-需求栅格
        """
        p_raster = np.where(sd_raster > 0, sd_raster, 0)
        n_raster = np.where(sd_raster < 0, sd_raster, 0)
        X, Y = sd_raster.shape

        # 收集连接信息
        rows = np.array([], dtype=np.int64)
        cols = np.array([], dtype=np.int64)

        logging.info(f"距离 {distance} 开始优化")
        start_time = time.time()

        # 对每个方向进行处理
        for direct in self.distance_strategy.get_directions(distance):
            dir_start = time.time()
            # 计算连接掩码
            weight_arr = edges_line_func(
                move_array(p_raster, direct, distance), n_raster
            )
            weight_arr_nonzero_index = np.nonzero(weight_arr)

            # 计算目标和源索引
            tar_index = weight_arr_nonzero_index[0] * Y + weight_arr_nonzero_index[1]
            sou_index = (weight_arr_nonzero_index[0] - direct[0]) * Y + (
                weight_arr_nonzero_index[1] - direct[1]
            )

            # 收集索引
            rows = np.concatenate((rows, tar_index, sou_index))
            cols = np.concatenate(
                (
                    cols,
                    np.arange(
                        int(len(cols) / 2),
                        len(tar_index) + int(len(cols) / 2),
                        dtype=np.int64,
                    ),
                    np.arange(
                        int(len(cols) / 2),
                        len(tar_index) + int(len(cols) / 2),
                        dtype=np.int64,
                    ),
                )
            )
            dir_end = time.time()
            logging.info(f"方向 {direct} 完成，用时 {(dir_end-dir_start):.2f} 秒")

        end_time = time.time()
        logging.info(
            f"距离 {distance} 连接收集完成，用时 {(end_time-start_time)/60:.3f} 分钟，"
            f"节点大小: {sys.getsizeof(rows)/(1024*1024):.2e} MB"
        )

        # 创建稀疏矩阵
        A = csr_matrix(
            (np.ones_like(rows), (rows, cols)),
            shape=(X * Y, int(cols[-1]) + 1),
            dtype=np.int8,
        )

        logging.info(
            f"A矩阵大小: {sys.getsizeof(A)/(1024*1024):.2e} MB, "
            f"形状: {X*Y} x {int(cols[-1])+1}, 非零行数: {len(np.nonzero(np.diff(A.indptr))[0])}"
        )

        # 准备线性规划
        w = sd_raster.reshape(-1)
        c = -np.ones((1, A.shape[1]))

        # 执行线性规划
        start_time = time.time()
        logging.info(f"线性规划-{distance} 开始")

        res = linprog(c, A, np.abs(w), bounds=(0, None))

        end_time = time.time()
        logging.info(
            f"线性规划-{distance} 结束，用时 {(end_time-start_time)/60:.3f} 分钟，"
            f"delta: {res.fun/10**6:.5f} Mt"
        )

        # 保存流量数据（如果提供了输出路径）
        if self.output_path is not None:
            self._save_flow_data(rows, cols, res.x, distance)

        # 计算新的盈余/需求栅格
        delta_w = np.sign(w) * (A @ res.x)
        new_w = w - delta_w
        new_sd_raster = new_w.reshape((X, Y))

        return new_sd_raster

    def _save_flow_data(
        self, rows: np.ndarray, cols: np.ndarray, x: np.ndarray, distance: int
    ) -> None:
        """
        保存流量数据

        Args:
            rows: 行索引
            cols: 列索引
            x: 优化结果
            distance: 距离
        """
        # 创建源-目标字典
        start_time = time.time()
        logging.info(f"流量-{distance} 字典开始")

        cr_dict = {}
        for r, c in zip(rows, cols):
            if c not in cr_dict:
                cr_dict[c] = []
            cr_dict[c].append(r)

        for key in cr_dict:
            cr_dict[key] = np.array(cr_dict[key], dtype=int)

        end_time = time.time()
        logging.info(f"流量-{distance} 字典完成，用时 {(end_time-start_time):.2f} 秒")

        # 收集并保存流量数据
        start_time = time.time()
        logging.info(f"流量-{distance} 开始保存")

        flow_tuple = [
            (cr_dict[value][0], cr_dict[value][1], x[value])
            for value in np.nonzero(x)[0]
            if value in cr_dict
        ]

        # 确保输出目录存在
        self.prepare_output_dir()

        # 保存流量数据
        trans_output = self.output_path / "Transport" / f"{self.resolution}km"
        trans_output.mkdir(exist_ok=True, parents=True)

        pd.DataFrame(flow_tuple, columns=["p1", "p2", "flow"]).to_csv(
            trans_output / f"flow_d{distance * self.resolution}.csv", index=False
        )

        end_time = time.time()
        logging.info(f"流量-{distance} 保存完成，用时 {(end_time-start_time):.2f} 秒")

    def save_stats(self, prefix: str = "") -> None:
        """
        保存统计数据

        Args:
            prefix: 文件名前缀
        """
        if self.output_path is not None and self.stats["distance"]:
            # 确保输出目录存在
            self.prepare_output_dir()

            # 保存统计数据
            stats_df = pd.DataFrame(self.stats)

            trans_output = self.output_path / "Transport" / f"{self.resolution}km"
            trans_output.mkdir(exist_ok=True, parents=True)

            filename = f"{prefix}distance_sd_data_{self.resolution}km.csv"
            stats_df.to_csv(trans_output / filename, index=False)
            logging.info(f"已保存统计数据到 {trans_output / filename}")

    def update_stats(self, sd_raster: np.ndarray, distance: int = 0) -> None:
        """
        更新统计数据

        Args:
            sd_raster: 盈余-需求栅格
            distance: 距离（默认为0，表示初始状态）
        """
        self.stats["distance"].append(distance * self.resolution)
        self.stats["surplus"].append(
            np.sum(sd_raster[sd_raster > 0]) / 10**6
        )  # 单位: Mt
        self.stats["deficiency"].append(
            np.sum(sd_raster[sd_raster < 0]) / 10**6
        )  # 单位: Mt


class UnlimitedTransportModel(TransportModel):
    """
    基于无限制传输策略的模型

    实现new_transport_unlimited.py中的方法，使用基于阈值的停止策略
    """

    def __init__(
        self,
        target_proportion: float = 0.9,
        max_distance: Optional[int] = None,
        **kwargs,
    ):
        """
        初始化无限制传输模型

        Args:
            target_proportion: 目标需求满足比例 (0-1)
            max_distance: 最大距离限制（单位：栅格单元），None表示无限制
            **kwargs: 传递给父类的参数
        """
        super().__init__(distance_strategy=UnlimitedDistanceStrategy(), **kwargs)
        self.target_proportion = target_proportion
        self.max_distance = max_distance

    def run(
        self, supply: np.ndarray, demand: np.ndarray
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        运行无限制传输模型

        Args:
            supply: 供应栅格
            demand: 需求栅格

        Returns:
            优化后的盈余-需求栅格和统计数据
        """
        # 计算盈余-需求栅格
        sd_raster = supply - demand

        # 记录初始状态
        self.update_stats(sd_raster)

        # 计算目标阈值
        min_surplus = np.sum(sd_raster[sd_raster > 0]) + np.sum(
            sd_raster[sd_raster < 0]
        )
        original_deficiency_sum = np.sum(sd_raster[sd_raster < 0])
        target_delta = -original_deficiency_sum * (1 - self.target_proportion)

        d = 0
        # 当仍有足够的盈余和需求，且未达到最大距离时继续
        while (
            (np.sum(sd_raster[sd_raster > 0]) > min_surplus + target_delta)
            and (-np.sum(sd_raster[sd_raster < 0]) > target_delta)
            and (self.max_distance is None or d < self.max_distance)
        ):

            d += 1
            # 优化当前距离的分配
            sd_raster = self.optimize_allocation(sd_raster, d)

            # 更新统计数据
            self.update_stats(sd_raster, d)

            # 中间保存
            self.save_stats(prefix="partial_")

            # 计算当前满足比例
            deficiency_sum = np.sum(sd_raster[sd_raster < 0])
            logging.info(
                f"距离 {d} 完成! 需求满足比例: "
                f"{(1 - deficiency_sum/original_deficiency_sum):.2%}"
            )

        # 保存最终结果
        self.save_stats(prefix="")

        return sd_raster, pd.DataFrame(self.stats)


class SpeciesAverageTransportModel(TransportModel):
    """
    基于物种平均策略的模型

    实现transport_species_average.py中的方法，处理不同物种的粪肥分配
    """

    def __init__(self, species_config: Dict[str, Dict[str, Any]], **kwargs):
        """
        初始化物种平均传输模型

        Args:
            species_config: 物种配置字典，例如:
                {'pig': {'distance': 10}, 'beef': {'distance': 10}, ...}
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        self.species_config = species_config

    def species_manure_classify_average(
        self, diff_distance_supply_list: List[np.ndarray], demand: np.ndarray
    ) -> List[np.ndarray]:
        """
        计算不同距离的物种供应量分配

        Args:
            diff_distance_supply_list: 不同距离的供应量列表
            demand: 需求栅格

        Returns:
            分配后的结果列表，最后一个元素为不满足的需求
        """
        result = []
        all_sum = np.array(diff_distance_supply_list).sum(axis=0)
        all_surplus_sum = all_sum - demand

        for i in range(len(diff_distance_supply_list)):
            result.append(
                np.where(
                    (all_surplus_sum >= 0) & (all_sum != 0),
                    diff_distance_supply_list[i] / all_sum * all_surplus_sum,
                    0,
                )
            )

        # 最后一个元素为不满足的需求
        result.append(np.where(all_surplus_sum < 0, all_surplus_sum, 0))

        return result

    def run(
        self, species_data: Dict[str, np.ndarray], demand: np.ndarray
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        运行物种平均传输模型

        Args:
            species_data: 物种数据字典，键为物种名称，值为供应栅格
            demand: 需求栅格

        Returns:
            优化后的盈余-需求栅格和统计数据
        """
        # 按照距离分组物种供应
        distance_range = sorted(
            set(config["distance"] for config in self.species_config.values())
        )
        distance_range.insert(0, 0)  # 添加起始距离0

        # 创建不同距离范围的供应列表
        diff_distance_s_list = []
        for i in range(len(distance_range) - 1):
            current_supply = np.zeros_like(demand)
            for species, config in self.species_config.items():
                if config["distance"] == distance_range[i + 1]:
                    current_supply += species_data[species]
            diff_distance_s_list.append(current_supply)

        # 初始分配
        result = self.species_manure_classify_average(diff_distance_s_list, demand)

        # 初始统计
        sd_raster = sum(result)
        self.update_stats(sd_raster)

        # 对每个距离范围进行优化
        start_time = time.time()
        for i in range(len(distance_range) - 1):
            for d in range(distance_range[i] + 1, distance_range[i + 1] + 1):
                # 计算当前要处理的供应
                s_raster = sum(result[i:-1])

                # 计算盈余-需求栅格
                sd_raster = s_raster + result[-1]

                # 优化当前距离的分配
                new_sd_raster = self.optimize_allocation(sd_raster, d)

                # 提取新的供应栅格
                new_s_raster = np.where(new_sd_raster > 0, new_sd_raster, 0)

                # 重新分类
                result[i:-1] = self.species_manure_classify_average(
                    result[i:-1], s_raster - new_s_raster
                )[:-1]

                # 更新剩余需求
                result[-1] = np.where(new_sd_raster < 0, new_sd_raster, 0)

                # 更新统计数据
                self.update_stats(sum(result), d)

                # 中间保存
                self.save_stats(prefix="partial_")

        end_time = time.time()
        logging.info(f"全部优化完成，用时: {(end_time-start_time)/60:.2f} 分钟")

        # 计算最终结果
        allocated_sd_raster = sum(result)

        # 保存最终结果
        self.save_stats(prefix="")

        return allocated_sd_raster, pd.DataFrame(self.stats)
