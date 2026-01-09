import tempfile as tf
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import ray
from classes.experiment_paths import SimpleExperiment
from config import RESULTS_DIR
from ray import tune


def get_param_space():
    return {
        "builder": tune.grid_search(["barabasi_albert", "erdos_renyi"]),  # "gml",
        "n_flows": tune.grid_search(list(range(500, 10001, 500))),
        "p": 0.7,
        "n": tune.grid_search([2**i for i in range(4, 11)]),
        "seed": tune.grid_search(
            [110296, 151195, 300997, 10664, 21297, 30997, 70799, 90597, 42, 80824]
        ),
    }


# def get_param_space():
#     return {
#         "builder": tune.grid_search(["barabasi_albert", "erdos_renyi"]),  # "gml",
#         "n_flows": 500,
#         "p": 0.7,
#         "n": tune.grid_search([16, 128]),
#         "seed": tune.grid_search([110296, 151195]),
#     }


def dglbf(config: Dict[str, Any]):

    with tf.TemporaryDirectory() as tmpdir:
        e = SimpleExperiment(
            n_flows=config["n_flows"],
            builder=config["builder"],
            n=config["n"],
            m=int(np.log2(config["n"])) if config["n"] else None,
            p=config["p"],
            seed=config["seed"],
            experiment_dir=Path(tmpdir),
        )
        return e.run()


if __name__ == "__main__":

    cpus = 192
    ray.init(address="auto")
    name = "paths"

    run_config = tune.RunConfig(name=name, storage_path=RESULTS_DIR)
    tuner = tune.Tuner(
        tune.with_resources(dglbf, {"cpu": 8}),
        param_space=get_param_space(),
        run_config=run_config,
    )

    # tuner = tune.Tuner.restore(
    #     f"/home/j.massa/GitHub/dgLBF/sim/results/{name}",
    #     trainable=dglbf,
    #     param_space=get_param_space(),
    #     restart_errored=True,
    # )

    results = tuner.fit()
    df = results.get_dataframe()
    df.set_index("trial_id", inplace=True)
    df.to_parquet(Path(results.experiment_path).parent / f"{name}.parquet")
