from itertools import product
from typing import Any, List, Union
from multipledispatch import dispatch

import config as c
import numpy as np
import pandas as pd
from swiplserver import *
import networkx as nx

from .flow import Flow
from .infrastructure import Infrastructure


def parse_prolog(query):
    if is_prolog_functor(query):
        if prolog_name(query) != ",":
            ans = (prolog_name(query), parse_prolog(prolog_args(query)))
        else:
            ans = tuple(parse_prolog(prolog_args(query)))
    elif is_prolog_list(query):
        ans = [parse_prolog(v) for v in query]
    elif is_prolog_atom(query):
        ans = query
    elif isinstance(query, dict):
        ans = {k: parse_prolog(v) for k, v in query.items()}
    else:
        ans = query
    return ans


class Experiment:
    def __init__(
        self,
        num_nodes: List[int] = [],
        num_flows: List[int] = [],
        gmls: List[str] = [],
        max_iterations: int = 1,
        seed: Any = None,
        timeout: int = None,
    ):

        np.random.seed(seed)

        self.num_nodes = num_nodes
        self.num_flows = num_flows
        self.gmls = gmls
        self.max_iterations = max_iterations
        self.seed = seed
        self.timeout = timeout
        self.results = []

    @dispatch(str)
    def set_infrastructure(self, gml):
        self.infrastructure = Infrastructure(gml=gml)

    @dispatch(int)
    def set_infrastructure(self, num_nodes):
        self.infrastructure = Infrastructure(
            n=num_nodes, m=int(np.log2(num_nodes)), seed=self.seed
        )

    def set_flows(self, num_flows):
        self.flows_file = c.FLOW_FILE_PATH.format(size=num_flows)
        self.flows: List[Flow] = []
        for i in range(num_flows):
            start, end = np.random.choice(
                self.infrastructure.nodes, size=2, replace=False
            )
            self.flows.append(Flow(f"f{i}", start, end, random=True))

        # sort flows by number of hops between start and end
        self.flows.sort(
            key=lambda f: nx.shortest_path_length(self.infrastructure, f.start, f.end)
        )

    def upload_infr(self):
        self.infrastructure.upload()

    def upload_flows(self):
        for i, f in enumerate(self.flows):
            f.upload(
                file=self.flows_file,
                append=(False if i == 0 else True),
            )

        for f in self.flows:
            paths = self.infrastructure.simple_paths(f.start, f.end)
            with open(self.flows_file, "a+") as file:
                file.write("\n")
                for path in paths:
                    candidate = c.CANDIDATE.format(
                        fid=f.fid, path=str(path).replace("'", "")
                    )
                    file.write(candidate + "\n")

    def upload(self):
        self.upload_infr()
        self.upload_flows()

    def save_result(self, res):
        res["Infr"] = self.infrastructure.name
        res["Flows"] = len(self.flows)
        res["Nodes"] = len(self.infrastructure.nodes)
        res["Edges"] = len(self.infrastructure.edges)
        res["Timestamp"] = pd.Timestamp.strftime(
            pd.Timestamp.now(), c.EXP_TIMESTAMP_FORMAT
        )
        self.results.append(res)

    def results_to_csv(self):
        if self.results:
            df = pd.DataFrame(self.results)
            df = df[c.COL_ORDER]
            df.set_index("Timestamp", inplace=True)
            name = (
                str(self.seed)
                if self.seed
                else pd.Timestamp.now().strftime(c.RES_TIMESTAMP_FORMAT)
            )
            filepath = c.RESULTS_DIR / c.RESULTS_FILE.format(name=name)
            c.df_to_file(df, filepath)
        else:
            print("No results yet.")

    def run_exp(self, infr: Union[int, str], flows: int, iteration: int = 1):
        self.set_infrastructure(infr)
        self.set_flows(flows)
        self.upload()
        with PrologMQI() as mqi:
            with mqi.create_thread() as prolog:
                print(
                    c.EXP_MESSAGE.format(
                        iteration=iteration,
                        num_flows=flows,
                        infr=self.infrastructure.name,
                        edges=len(self.infrastructure.edges),
                        nodes=len(self.infrastructure.nodes),
                    )
                )
                prolog.query("consult('{}')".format(c.SIM_FILE_PATH))
                prolog.query(c.LOAD_INFR_QUERY.format(path=self.infrastructure.file))
                prolog.query(c.LOAD_FLOWS_QUERY.format(path=self.flows_file))
                prolog.query_async(
                    c.MAIN_QUERY, find_all=False, query_timeout_seconds=self.timeout
                )

                res = {}
                try:
                    q = prolog.query_async_result()
                    if q:
                        res = self.parse_output(q[0])
                    else:
                        print("\tNo results found.")
                except PrologQueryTimeoutError:
                    print("\tTimeout reached. Skipping this experiment.")

                return res

    def run(self):
        infrs = self.gmls + self.num_nodes
        for i, f in product(infrs, self.num_flows):
            iterations = 0
            res = None
            while not res and iterations < self.max_iterations:
                iterations += 1
                res = self.run_exp(infr=i, flows=f)

            res["Iterations"] = iterations
            self.save_result(res)

    def parse_paths(self, paths):
        return {
            flow: {"path": path, "budgets": budgets, "delay": delay}
            for (flow, (path, (budgets, delay))) in paths
        }

    def parse_allocation(self, allocation):
        return {(s, d): bw for (s, (d, bw)) in allocation}

    def parse_output(self, out):
        o = parse_prolog(out)
        return {
            "Output": self.parse_paths(o["Output"]),
            "Allocation": self.parse_allocation(o["Allocation"]),
            "Inferences": o["Inferences"],
            "Time": o["Time"],
        }

    def __str__(self):
        if not self.results:
            return "\nNo results yet.\n"
        else:
            res = ""
            for r in self.results:
                res += f"Flows: {r['Flows']}" + "\n"
                res += f"Nodes: {r['Nodes']}" + "\n"
                res += f"Edges: {r['Edges']}" + "\n\n"
                res += "Paths: \n"
                for flow, attr in r["Output"].items():
                    res += f"Flow {flow}: \n"
                    res += f"\tPath: {attr['path']}\n"
                    res += f"\tBudgets: {round(attr['budgets'][0], 4), round(attr['budgets'][1], 4)}\n"
                    res += f"\tDelay: {round(attr['delay'],4)}\n"
                res += "Allocation: \n"
                for (s, d), bw in r["Allocation"].items():
                    res += f"\tLink {s} -> {d}: {bw} Mbps\n"
                res += f"Inferences: {r['Inferences']}\n"
                res += f"Time: {round(r['Time'], 4)} (s)\n"
            return res
