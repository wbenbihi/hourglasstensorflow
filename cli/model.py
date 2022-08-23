import datetime

import click
import tensorflow as tf
from loguru import logger

from hourglass_tensorflow.types.config import HTFConfig
from hourglass_tensorflow.types.config import HTFConfigParser
from hourglass_tensorflow.handlers.model import HTFModelHandler


@click.group()
def model():
    """Operation related to Model"""


@click.command()
@click.option(
    "--verbose/--no-verbose",
    "-v",
    default=False,
    help="Activate Logs",
    type=bool,
)
@click.argument("input")
@click.argument("output")
def log(verbose, input, output):
    """Create a TensorBoard log to visualize graph"""
    if verbose:
        logger.debug(f"input:\t {input}")
        logger.debug(f"output:\t {output}")
    try:
        if verbose:
            logger.info(f"Reading {input}...")
        config = HTFConfig.parse_obj(
            HTFConfigParser.parse(filename=input, verbose=verbose)
        )
        writer = tf.summary.create_file_writer(output)
        tf.summary.trace_on(graph=True, profiler=True)
        if verbose:
            logger.info("Building Graph...")
        model_handler = HTFModelHandler(config=config.model, verbose=verbose)
        model_handler()
        if verbose:
            logger.info("Writing Graph...")
        with writer.as_default():
            tf.summary.trace_export(
                name=f"GraphTrace_{datetime.datetime.now()}",
                step=0,
                profiler_outdir=f"{output}/graph",
            )
        if verbose:
            logger.success("Operation completed!")
    except Exception as e:
        if verbose:
            logger.exception(e)
            logger.error("Operation aborted!")


model.add_command(log)
