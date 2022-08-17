import click
from loguru import logger

from hourglass_tensorflow.utils import parsers as htfparse
from hourglass_tensorflow.utils.writers import common_write


@click.group()
@click.option(
    "--verbose/--no-verbose",
    "-v",
    default=False,
    help="Activate Logs",
    type=bool,
)
def mpii():
    """Operation related to MPII management / parsing"""


@click.command()
@click.option(
    "--validate/--no-validate",
    default=False,
    help="Whether to use validation checks (default false)",
    type=bool,
)
@click.option(
    "--struct/--no-struct",
    default=False,
    help="Whether or not to apply pydantic parsing (default false)",
    type=bool,
)
@click.option(
    "--as-list/--no-as-list",
    default=False,
    help="Activate to return list of records (default false)",
    type=bool,
)
@click.option(
    "--null/--no-null",
    default=True,
    help="Keep null values in records (default true)",
    type=bool,
)
@click.argument("input")
@click.argument("output")
def parse(verbose, validate, struct, as_list, null, input, output):
    """Parse a MPII .mat file to a more readable record"""
    if verbose:
        logger.debug(f"--validate:\t {validate}")
        logger.debug(f"--strcu:\t {struct}")
        logger.debug(f"--as-list:\t {as_list}")
        logger.debug(f"--null:\t {null}")
        logger.debug(f"input:\t {input}")
        logger.debug(f"output:\t {output}")
    try:
        if verbose:
            logger.info(f"Reading {input}...")
        mpii_obj = htfparse.mpii.parse_mpii(
            mpii_annot_file=input,
            test_parsing=validate,
            verify_len=validate,
            return_as_struct=struct,
            zip_struct=as_list,
            remove_null_keys=null,
        )
        if verbose:
            logger.info(f"Input parsed! Saving to {output}...")
        common_write(mpii_obj, path=output, force_dict_struct=False)
        if verbose:
            logger.success("Operation completed!")
    except Exception as e:
        if verbose:
            logger.exception(e)
            logger.error("Operation aborted!")


@click.argument("input")
@click.argument("output")
def convert(verbose, input, output):
    """Convert a MPII .mat file to a HTF compliant record"""
    pass


mpii.add_command(parse)
mpii.add_command(convert)
