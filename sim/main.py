import atexit

import click
import config as c
from classes.experiment import Experiment


@click.command()
@click.option(
    "--flows",
    "-f",
    type=int,
    multiple=True,
    required=True,
    help="Number of flows in the experiment.",
)
@click.option(
    "--nodes",
    "-n",
    multiple=True,
    type=int,
    help="Number of nodes in the infrastructure.",
)
@click.option(
    "--gml",
    "-g",
    multiple=True,
    type=click.Choice(c.GML_CHOICES),
    default=None,
    help="Name of a GML file (in data/gml) to use as infrastructure.",
)
@click.option(
    "--max_iterations",
    "-i",
    type=int,
    default=1,
    help="Number of of trials to find a solution for each combination #flows / infrastructure.",
)
@click.option(
    "--seed", "-s", type=int, default=None, help="Seed for the random number generator."
)
@click.option(
    "--timeout", "-t", type=int, default=c.TIMEOUT, help="Timeout for the experiment."
)
def main(flows, nodes, gml, max_iterations, seed, timeout):
    """Start an experiment with an infrastructure of NODES nodes, and FLOWS flows."""

    flows, nodes, gml = list(flows), list(nodes), list(gml)

    e = Experiment(
        num_nodes=nodes,
        num_flows=flows,
        gmls=gml,
        max_iterations=max_iterations,
        seed=seed,
        timeout=timeout,
    )

    atexit.register(e.results_to_csv)
    e.run()


if __name__ == "__main__":
    main()
