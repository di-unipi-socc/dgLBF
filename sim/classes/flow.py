from os import makedirs
from os.path import dirname, exists
from typing import List

import config as c
import numpy as np


class Flow:
    def __init__(
        self,
        fid: str,
        start: str,
        end: str,
        packet_size: float = 0.0,
        burst_size: int = 0,
        bit_rate: float = 0.0,
        latency_budget: float = 0.0,
        toleration_threshold: float = 0.0,
        random: bool = False,
    ):

        self.fid = fid
        self.start = start
        self.end = end
        if random:
            self.random_setup()
        else:
            self.packet_size = packet_size
            self.burst_size = burst_size
            self.bit_rate = bit_rate
            self.latency_budget = latency_budget
            self.toleration_threshold = toleration_threshold

        self.path: List[int] = []
        self.min_budget: float = 0.0
        self.max_budget: float = 0.0
        self.delay: float = 0.0

    def random_setup(self):
        self.packet_size = c.PACKET_SIZE
        self.bit_rate = np.random.randint(c.BIT_RATE_MIN, c.BIT_RATE_MAX)
        self.burst_size = np.random.randint(c.BURST_SIZE_MIN, c.BURST_SIZE_MAX)
        self.latency_budget = np.random.randint(
            c.LATENCY_BUDGET_MIN, c.LATENCY_BUDGET_MAX
        )
        self.toleration_threshold = np.random.randint(
            c.TOLERATION_THRESHOLD_MIN, c.TOLERATION_THRESHOLD_MAX
        )

    def __str__(self):
        return c.FLOW.format(**self.__dict__)

    def upload(self, file, append=True):
        makedirs(dirname(file)) if not exists(dirname(file)) else None
        mode = "a+" if append else "w+"
        with open(file, mode) as f:
            f.write(str(self) + "\n")
