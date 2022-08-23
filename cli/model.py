import datetime

import click
import tensorflow as tf
from loguru import logger

from hourglass_tensorflow.types.config import HTFConfig
from hourglass_tensorflow.types.config import HTFConfigParser
from hourglass_tensorflow.handlers.model import _HTFModelHandler
from hourglass_tensorflow.types.config.fields import HTFObjectReference


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
            logger.info(f"Reading config from {input}...")
        config = HTFConfig.parse_obj(
            HTFConfigParser.parse(filename=input, verbose=verbose)
        )
        obj: HTFObjectReference[_HTFModelHandler] = config.model.object

        tf.summary.trace_on(graph=True, profiler=True)
        writer = tf.summary.create_file_writer(output)
        if verbose:
            logger.info("Building Graph...")
        graph = tf.Graph()
        with graph.as_default():
            model_handler = obj.init(config=config.model, verbose=verbose)
            model_handler()
        if verbose:
            logger.info("Writing Graph...")
        with writer.as_default():
            tf.summary.graph(
                graph,
            )
        writer.flush()
        if verbose:
            logger.success("Operation completed!")
    except Exception as e:
        if verbose:
            logger.exception(e)
            logger.error("Operation aborted!")


@click.command()
@click.option(
    "--verbose/--no-verbose",
    "-v",
    default=False,
    help="Activate Logs",
    type=bool,
)
@click.option(
    "--shapes/--no-shapes",
    "-s",
    default=False,
    help="Show Layer Shapes. (default: false)",
    type=bool,
)
@click.option(
    "--types/--no-types",
    "-t",
    default=False,
    help="Show Layer Types. (default: false)",
    type=bool,
)
@click.option(
    "--names/--no-names",
    "-n",
    default=False,
    help="Show Layer Names. (default: false)",
    type=bool,
)
@click.option(
    "--expand/--no-expand",
    "-e",
    default=False,
    help="Whether to expand nested models into clusters. (default: false)",
    type=bool,
)
@click.option(
    "--activation/--no-activation",
    "-a",
    default=False,
    help="Display layer activations (only for layers that have an activation property). (default: false)",
    type=bool,
)
@click.option(
    "--dpi",
    "-d",
    default=96,
    help="Dots per inch. (default: 96)",
    type=int,
)
@click.argument("input")
@click.argument("output")
def plot(verbose, shapes, types, names, expand, activation, dpi, input, output):
    """Create a summary image of model Graph

    NOTES:\n
        To work this command requires `pydot` and graphviz to be installed
        pydot:\n
            `$ pip install pydot`\n
        graphviz:\n
            see instructions at https://graphviz.gitlab.io/download/\n
    """
    if verbose:
        logger.debug(f"input:\t {input}")
        logger.debug(f"output:\t {output}")
    try:
        if verbose:
            logger.info(f"Reading config from {input}...")
        config = HTFConfig.parse_obj(
            HTFConfigParser.parse(filename=input, verbose=verbose)
        )
        obj: HTFObjectReference[_HTFModelHandler] = config.model.object
        if verbose:
            logger.info("Building Graph...")
        model_handler = obj.init(config=config.model, verbose=verbose)
        model_handler()
        if verbose:
            logger.info("Writing Image...")
        tf.keras.utils.plot_model(
            model_handler._model,
            to_file=output,
            show_shapes=shapes,
            show_dtype=types,
            show_layer_names=names,
            expand_nested=expand,
            dpi=dpi,
            show_layer_activations=activation,
        )
        if verbose:
            logger.success("Operation completed!")
    except Exception as e:
        if verbose:
            logger.exception(e)
            logger.error("Operation aborted!")


model.add_command(log)
model.add_command(plot)
