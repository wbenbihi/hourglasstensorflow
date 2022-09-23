import re

import click

from hourglass_tensorflow.cli.mpii import mpii
from hourglass_tensorflow.handlers import HTFManager
from hourglass_tensorflow.cli.model import model


@click.group()
def cli():
    """`hourglass_tensorflow` command-line interface (CLI)"""
    pass


@cli.command()
@click.argument("input")
def run(input):
    """Launch a `hourglass_tensorflow` config file"""
    manager = HTFManager(filename=input, verbose=True)
    manager()


# Register Commands
cli.add_command(mpii)
cli.add_command(model)


if __name__ == "__main__":
    cli()
