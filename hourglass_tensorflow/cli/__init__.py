import re

import click

from hourglass_tensorflow.cli.mpii import mpii
from hourglass_tensorflow.cli.model import model


@click.group()
def cli():
    """`hourglass_tensorflow` command-line interface (CLI)"""
    pass


# Register Commands
cli.add_command(mpii)
cli.add_command(model)


if __name__ == "__main__":
    cli()
