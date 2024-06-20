# dgLBF
Declarative implementation of guaranteed Latency Based Forwarding ([gLBF](https://link.springer.com/article/10.1007/s10922-022-09718-9)).

## How To &nbsp;<picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn-icons-png.flaticon.com/512/2666/2666505.png"><img width="20" height="20" alt="files" src="https://cdn-icons-png.flaticon.com/512/2666/2666469.png">
</picture>

Download or clone this repo. Make sure you have the following prerequisites:

- [`swipl`](https://www.swi-prolog.org/download/stable) >= 9.0.4
- [`python`](https://www.python.org/downloads/) >= 3.8
- [`swipl MQI for Python`](https://www.swi-prolog.org/pldoc/man?section=mqi-python-installation)
- [`requirements.txt`](https://github.com/di-unipi-socc/dgLBF/blob/main/sim/requirements.txt) for the python virtual environment

1. Create a virtual environment and install the required packages:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r sim/requirements.txt
    ```

2. Then, run the python script in the `sim` directory, after activating the virtual environment:

    ```bash
    python3 main.py [OPTIONS]
    ```

    The script has the following options:

    ```
    Usage: main.py [OPTIONS]

    Start an experiment with an infrastructure of NODES nodes, and FLOWS flows.

    Options:
    -f, --flows INTEGER             Number of flows in the experiment.
                                    [required]
    -n, --nodes INTEGER             Number of nodes in the infrastructure.
    -g, --gml [janos-us-ca|giul39|germany50|ta2|norway|cost266|atlanta|di-yuan|pdh|india35|sun|geant|dfn-bwin|france|polska|dfn-gwin|nobel-eu|abilene|newyork|janos-us|nobel-germany|pioro40|zib54|nobel-us]
                                    Name of a GML file (in data/gml) to use as
                                    infrastructure.
    -i, --max_iterations INTEGER    Number of of trials to find a solution for
                                    each combination #flows / infrastructure.
    -s, --seed INTEGER              Seed for the random number generator.
    -t, --timeout INTEGER           Timeout for the experiment.
    --help                          Show this message and exit.
    ```

    *N.B.*, using the `--gml` option will override the `--nodes` option. Instead, by using the `--nodes` option, the script will generate a random infrastructure with the specified number of nodes.
    Flows are generated randomly, with a random source and destination node, and a random latency requirement.

    Change the ranges of random values for both infrastructures and flows in the [`config.py`](https://github.com/di-unipi-socc/dgLBF/blob/main/sim/config.py) script.