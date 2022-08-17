import re

import click

from .mpii import mpii


@click.group()
def cli():
    pass


# Register Commands
cli.add_command(mpii)


if __name__ == "__main__":
    cli()
