import re

import click
from cli.mpii import mpii
from cli.model import model


@click.group()
def cli():
    pass


# Register Commands
cli.add_command(mpii)
cli.add_command(model)


if __name__ == "__main__":
    cli()
