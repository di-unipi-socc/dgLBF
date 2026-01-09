import random
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import config as c
import networkx as nx
import numpy as np

from .flow import Flow
from .infrastructure import Infrastructure


class SimpleExperiment:

    def __init__(
        self,
        n_flows: int,
        builder: Literal["barabasi_albert", "erdos_renyi"] = "barabasi_albert",
        n: Optional[int] = None,
        m: Optional[int] = None,
        p: Optional[float] = None,
        seed: Any = None,
        experiment_dir=c.DATA_DIR,
        # opzionali per compatibilità con "setting" del vecchio Experiment
        version: str = "paths_stats",
    ):
        np.random.seed(seed)
        random.seed(seed)

        self.n_flows = n_flows
        self.builder = builder
        self.n = n
        self.m = m
        self.p = p
        self.seed = seed
        self.experiment_dir = experiment_dir

        self.version = version

        self.infrastructure: Optional[Infrastructure] = None
        self.flows: List[Flow] = []

        # cache: (start, end) -> count(paths)
        self._paths_count_cache: Dict[Tuple[Any, Any], int] = {}

        # richiesto: metti result nell'__init__
        self.result: Dict[str, Any] = {}

    # ---------------- setup ----------------

    def build_infrastructure(self) -> Infrastructure:
        self.infrastructure = Infrastructure(
            builder=self.builder,
            n=self.n,
            m=self.m,
            p=self.p,
            seed=self.seed,
            infra_path=self.experiment_dir / "infrastructures",
        )
        return self.infrastructure

    def set_flows(self) -> List[Flow]:
        if self.infrastructure is None:
            raise RuntimeError(
                "Infrastructure not built. Call build_infrastructure() first."
            )

        self.flows = []
        for _ in range(self.n_flows):
            ok = False
            while not ok:
                start, end = np.random.choice(
                    self.infrastructure.nodes, size=2, replace=False
                )
                ok = nx.has_path(self.infrastructure, start, end)

            self.flows.append((start, end))

        # come nell'originale: ordina per shortest path length
        self.flows.sort(
            key=lambda f: nx.shortest_path_length(self.infrastructure, f[0], f[1])
        )
        return self.flows

    # ---------------- core con cache ----------------

    def _count_paths(self, start, end) -> int:
        key = (start, end)
        if key in self._paths_count_cache:
            return self._paths_count_cache[key]

        if self.infrastructure is None:
            raise RuntimeError("Infrastructure not built.")

        paths = self.infrastructure.simple_paths(start, end)

        # se è lista: len; se è generatore: conteggio
        n_paths = len(paths) if hasattr(paths, "__len__") else sum(1 for _ in paths)

        self._paths_count_cache[key] = n_paths
        return n_paths

    def compute_counts_for_flow_pairs(self) -> np.ndarray:
        """
        Calcola i conteggi sulle coppie uniche (start,end) dei flows.
        Ritorna array NumPy di conteggi.
        """
        pairs = {(f[0], f[1]) for f in self.flows}
        return np.array([self._count_paths(s, t) for (s, t) in pairs], dtype=np.int64)

    # ---------------- result handling (stile originale, ma pulito) ----------------

    def save_result(self, counts: np.ndarray, paths_time_s: float) -> None:
        if self.infrastructure is None:
            raise RuntimeError("Infrastructure not built.")

        # setting
        self.result = {
            "Version": self.version,
            "Seed": self.seed,
            "Builder": self.builder,
            "Infr": self.infrastructure.name,
            "Flows": self.n_flows,
            "Nodes": len(self.infrastructure.nodes),
            "Edges": len(self.infrastructure.edges),
            "n": self.n,
            "m": self.m,
            "p": self.p,
        }

        # nuove metriche
        if counts.size == 0:
            avg_paths = 0.0
            max_paths = 0
            n_pairs = 0
        else:
            avg_paths = int(float(counts.mean()))
            max_paths = int(counts.max())
            n_pairs = int(counts.size)

        self.result.update(
            {
                "AvgPaths": avg_paths,
                "MaxPaths": max_paths,
                "Pairs": n_pairs,
                "PathsTime": paths_time_s,
            }
        )

    def stringify(self):
        return {k: str(v) for k, v in self.result.items()}

    # ---------------- public API ----------------

    def run(self) -> Dict[str, str]:
        """
        Esegue tutto e ritorna direttamente il dict stringify-ato.
        """
        self.build_infrastructure()
        self.set_flows()
        t0 = time.perf_counter()

        counts = self.compute_counts_for_flow_pairs()
        paths_time_s = time.perf_counter() - t0

        self.save_result(counts, paths_time_s)
        return self.stringify()
