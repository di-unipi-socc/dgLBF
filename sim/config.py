import time
from functools import wraps
from os import listdir
from os.path import isfile, join
from pathlib import Path

import pandas as pd

# --- Directories & files ---
ROOT_DIR = Path(__file__).resolve().parent.parent
SIM_DIR = ROOT_DIR / "sim"
PLOTS_DIR = SIM_DIR / "plots"
RESULTS_DIR = SIM_DIR / "results"
DATA_DIR = SIM_DIR / "data"

FLOW_DIR = DATA_DIR / "flows"
INFRA_DIR = DATA_DIR / "infrastructures"
GML_DIR = DATA_DIR / "gml"

RESULTS_FILE = "results-{name}.csv"
GML_FILE = "{name}.gml"
FLOWS_FILE = "flows{size}.pl"
INFRA_FILE = "infr{name}.pl"

RESULTS_FILE_PATH = join(RESULTS_DIR, RESULTS_FILE)
GML_FILE_PATH = join(GML_DIR, GML_FILE)
FLOW_FILE_PATH = join(FLOW_DIR, FLOWS_FILE)
INFRA_FILE_PATH = join(INFRA_DIR, INFRA_FILE)
SIM_FILE_PATH = join(SIM_DIR, "sim.pl")

# --- Plots config ---
PLOT_FORMAT = "pdf"
PLOT_FILE = "{name}." + PLOT_FORMAT
PLOT_PATH = join(PLOTS_DIR, PLOT_FILE)
PLOT_DPI = 600

# --- Experiment config ---
TIMEOUT = 300  # seconds
GML_CHOICES = [f[:-4] for f in listdir(GML_DIR) if f.endswith(".gml")]
EXP_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
RES_TIMESTAMP_FORMAT = "%Y%m%d-%H%M"
EXP_MESSAGE = "({iteration}) - Experiment with {num_flows} flows on {infr} ({edges} edges, {nodes} nodes)."
COL_ORDER = [
    "Timestamp",
    "Infr",
    "Flows",
    "Nodes",
    "Edges",
    "Time",
    "Inferences",
    "Output",
    "Allocation",
]

# --- Figure config ---
FIG_OPTIONS = {
    "node_size": 500,
    "width": 1,
    "arrowstyle": "-|>",
    "arrowsize": 10,
}

# --- Infrastructure config ---

NODE_LAT_MIN, NODE_LAT_MAX = 1, 5
LINK_LAT_MIN, LINK_LAT_MAX = 1, 5
LINK_BW_MIN, LINK_BW_MAX = 500, 1500

# --- Flow config ---
PACKET_SIZE = 0.008
BURST_SIZE_MIN, BURST_SIZE_MAX = 2, 4
BIT_RATE_MIN, BIT_RATE_MAX = 2, 8
LATENCY_BUDGET_MIN, LATENCY_BUDGET_MAX = 30, 60
TOLERATION_THRESHOLD_MIN, TOLERATION_THRESHOLD_MAX = 10, 20

# PACKET_SIZE_MIN, PACKET_SIZE_MAX, PACKET_SIZE_STEP = 0.001, 0.01, 0.001
# PACKET_SIZE_RANGE = np.arange(PACKET_SIZE_MIN, PACKET_SIZE_MAX + PACKET_SIZE_STEP, PACKET_SIZE_STEP)
### TEMPLATES ###

# -- Prolog Templates ---
MAIN_QUERY = "once(sim_glbf(Output, Allocation, Inferences, Time))."
LOAD_INFR_QUERY = "once(loadInfrastructure('{path}'))."
LOAD_FLOWS_QUERY = "once(loadFlows('{path}'))."

# --- Flow templates ---
FLOW = "flow({fid}, {start}, {end}, {packet_size}, {burst_size}, {bit_rate}, {latency_budget}, {toleration_threshold})."
CANDIDATE = "candidate({fid}, {path})."

# --- Infrastructure templates ---
NODE = "node({nid}, {latency_budget})."
LINK = "link({source}, {dest}, {lat}, {bw})."
DEGREE = "degree({nid}, {degree})."
MAX_DEGREE = "maxDegree({max_degree})."
MIN_DEGREE = "minDegree({min_degree})."
MAX_LATENCY = "maxLatency({max_latency})."
MIN_LATENCY = "minLatency({min_latency})."
MAX_BW = "maxBandwidth({max_bw})."
MIN_BW = "minBandwidth({min_bw})."


# --- Auxiliary functions ---
def timeit(func):
    @wraps(func)
    def measure_time(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("{} took {} seconds.".format(func.__name__, end_time - start_time))
        return result

    return measure_time


def df_to_file(df: pd.DataFrame, file_path: Path):
    # create the directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, mode="a", header=(not isfile(file_path)))
