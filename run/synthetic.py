"""A small generated-demand environment for exercising the DT-MW pipeline."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Driver:
    zone: int
    available_at: int = 0
    income: float = 0.0


@dataclass
class Zone:
    idle_driver_num: int = 0
    real_order_num: int = 0
    fleet_order_num: int = 1
    gmv: float = 0.0


@dataclass(frozen=True)
class Offer:
    origin: int
    destination: int
    duration: int
    fare: float
    kind: int

    def features(self):
        return [self.origin, self.destination, self.fare, self.duration, self.kind, 0.0]


class SyntheticDispatchEnv:
    """Each zone offers idle, reposition, and one or more generated trips."""

    def __init__(self, grid_num, driver_num):
        if grid_num < 2:
            raise ValueError("The synthetic environment needs at least two zones")
        self.grid_num = grid_num
        self.driver_num = driver_num
        self.nodes = [Zone() for _ in range(grid_num)]
        self.global_instruction_dim = 0
        self.global_instruction_vector = None
        self.reset_randomseed(0)

    def reset_randomseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def set_global_instruction_vector(self, vector):
        self.global_instruction_vector = np.asarray(vector, dtype=np.float32)

    def get_state_space_node(self):
        return 5 + self.global_instruction_dim

    def get_node_reward_vector(self):
        return self.node_reward_cache

    def reset(self, mode="PPO2", generate_order=1):
        self.slot = 0
        self.gmv = 0.0
        self.order_response_rate = 0.0
        self.node_reward_cache = None
        self.drivers = {i: Driver(i % self.grid_num) for i in range(self.driver_num)}
        for node in self.nodes:
            node.gmv = 0.0
        state, orders, ids = self._observe()
        return state, None, orders, ids, None, None

    def _observe(self):
        self.idle = [[] for _ in self.nodes]
        for driver in self.drivers.values():
            if driver.available_at <= self.slot:
                self.idle[driver.zone].append(driver)

        observations = []
        self.offers = []
        for index, node in enumerate(self.nodes):
            node.idle_driver_num = len(self.idle[index])
            node.real_order_num = int(self.rng.integers(1, 5)) + int(
                (self.slot + index) % 5 == 0
            )
            options = [
                Offer(index, index, 1, 0.0, 0),
                Offer(index, (index + 1) % self.grid_num, 1, 0.0, 1),
            ]
            for _ in range(node.real_order_num):
                destination = (
                    index + int(self.rng.integers(1, self.grid_num))
                ) % self.grid_num
                duration = int(self.rng.integers(1, 4))
                fare = round(float(3 + 2 * duration + self.rng.uniform(0, 4)), 2)
                options.append(Offer(index, destination, duration, fare, 2))
            self.offers.append(options)
            supply, demand = node.idle_driver_num, node.real_order_num
            state = [
                index,
                supply,
                demand,
                min(supply, demand) / max(supply, demand),
                node.gmv,
            ]
            if self.global_instruction_vector is not None:
                state.extend(self.global_instruction_vector)
            observations.append(np.asarray(state, dtype=np.float32))

        order_features = [
            [offer.features() for offer in options] for options in self.offers
        ]
        order_ids = [list(range(len(options))) for options in self.offers]
        return observations, order_features, order_ids

    def get_orders_by_id(self, actions):
        return [
            [offers[index] for index in ids]
            for offers, ids in zip(self.offers, actions)
        ]

    def step(self, orders, generate_order=1, mode="PPO2"):
        demand = sum(node.real_order_num for node in self.nodes)
        served = 0
        self.gmv = 0.0
        self.node_reward_cache = np.zeros(self.grid_num, dtype=np.float32)
        for index, (node, idle, selected) in enumerate(
            zip(self.nodes, self.idle, orders)
        ):
            node.gmv = 0.0
            matches = 0
            for driver, offer in zip(idle, selected):
                driver.zone = offer.destination
                driver.available_at = self.slot + offer.duration
                driver.income += offer.fare
                node.gmv += offer.fare
                matches += int(offer.kind == 2)
            served += matches
            self.gmv += node.gmv
            self.node_reward_cache[index] = (
                0.01 * node.gmv + matches / node.real_order_num
            )

        self.order_response_rate = served / demand
        self.slot += 1
        state, next_orders, ids = self._observe()
        return state, next_orders, ids, None
