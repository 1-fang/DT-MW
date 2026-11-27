"""
Dynamic Reward Shaper — three-objective reward composition + intrinsic alignment.

Paper formulation:
  R_ext = beta_gmv * N(r_gmv) + beta_orr * r_orr + beta_fair * (1 - Gini)
  R_total = R_ext + eta * r_int
"""

import numpy as np
from typing import List, Optional


class DynamicRewardShaper:
    def __init__(self, gmv_scale=0.01, default_beta=None):
        self.gmv_scale = gmv_scale
        self.default_beta = default_beta or [0.4, 0.4, 0.2]
        self.beta = list(self.default_beta)

    def set_beta(self, beta: List[float]):
        assert len(beta) == 3
        self.beta = list(beta)

    def shape_reward(self, nodes, beta: Optional[List[float]] = None, env=None) -> np.ndarray:
        """Compute shaped reward vector (one per grid node).

        Args:
            nodes: list of Node objects from the environment
            beta: [beta_gmv, beta_orr, beta_fair], sums to 1
            env: optional environment. When provided, fairness is computed over
                 all registered drivers instead of only currently idle node drivers.

        Returns:
            reward: np.ndarray of shape (num_nodes,)
        """
        b = beta if beta is not None else self.beta
        num = len(nodes)
        reward = np.zeros(num, dtype=np.float32)

        driver_incomes = self.collect_driver_incomes(env if env is not None else nodes)
        fairness_global = 1.0 - self._gini(driver_incomes) if driver_incomes else 0.5

        for i, node in enumerate(nodes):
            if node is None:
                continue
            # r_gmv: normalized GMV
            r_gmv = getattr(node, 'gmv', 0.0) * self.gmv_scale

            # r_orr: local order response rate
            demand = getattr(node, 'real_order_num', 0)
            supply = getattr(node, 'idle_driver_num', 0)
            r_orr = min(supply, demand) / max(demand, 1)

            # r_fair: global fairness (same for all nodes in a step)
            r_fair = fairness_global

            reward[i] = b[0] * r_gmv + b[1] * r_orr + b[2] * r_fair

        return reward

    @staticmethod
    def collect_driver_incomes(env_or_nodes) -> List[float]:
        """Collect cumulative driver incomes with a global-first rule.

        The simulator keeps a city-level ``env.drivers`` registry containing all
        drivers, including drivers currently serving orders. Node-level
        ``node.drivers`` only contains drivers currently present in a grid, so it
        is used only as a fallback for compatibility.
        """
        if env_or_nodes is None:
            return []

        drivers = getattr(env_or_nodes, 'drivers', None)
        if drivers:
            values = drivers.values() if isinstance(drivers, dict) else drivers
            return [
                float(getattr(driver, 'income', getattr(driver, 'last_reward', 0.0)))
                for driver in values
            ]

        nodes = getattr(env_or_nodes, 'nodes', env_or_nodes)
        incomes = []
        for node in nodes or []:
            if node is None:
                continue
            node_drivers = getattr(node, 'drivers', {})
            values = node_drivers.values() if isinstance(node_drivers, dict) else node_drivers
            for driver in values:
                incomes.append(float(getattr(driver, 'income', getattr(driver, 'last_reward', 0.0))))
        return incomes

    @staticmethod
    def _gini(values) -> float:
        """Compute Gini coefficient of a list of values."""
        if not values:
            return 0.0
        arr = np.array(values, dtype=np.float64)
        if arr.sum() == 0:
            return 0.0
        arr = np.sort(arr)
        n = len(arr)
        index = np.arange(1, n + 1)
        gini = (2.0 * np.sum(index * arr)) / (n * arr.sum()) - (n + 1.0) / n
        return float(np.clip(gini, 0.0, 1.0))
