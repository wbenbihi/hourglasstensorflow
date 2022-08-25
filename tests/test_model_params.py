import keras
import pytest
import tensorflow as tf
from loguru import logger

from hourglass_tensorflow.models import HourglassModel
from hourglass_tensorflow.models import model_as_layers


@pytest.fixture(scope="function")
def fixt():
    return None


TEST_MODEL_PARAMS = [
    (
        {
            "input_size": 256,
            "output_size": 64,
            "stages": 4,
            "downsamplings_per_stage": 4,
            "stage_filters": 128,
            "output_channels": 16,
            "intermediate_supervision": True,
        },
        2932992,
    ),
    (
        {
            "input_size": 256,
            "output_size": 64,
            "stages": 4,
            "downsamplings_per_stage": 4,
            "stage_filters": 128,
            "output_channels": 16,
            "intermediate_supervision": False,
        },
        2932992,
    ),
    (
        {
            "input_size": 256,
            "output_size": 64,
            "stages": 2,
            "downsamplings_per_stage": 4,
            "stage_filters": 128,
            "output_channels": 16,
            "intermediate_supervision": False,
        },
        1505632,
    ),
    (
        {
            "input_size": 256,
            "output_size": 64,
            "stages": 4,
            "downsamplings_per_stage": 4,
            "stage_filters": 64,
            "output_channels": 16,
            "intermediate_supervision": False,
        },
        756512,
    ),
    (
        {
            "input_size": 256,
            "output_size": 64,
            "stages": 4,
            "downsamplings_per_stage": 4,
            "stage_filters": 128,
            "output_channels": 8,
            "intermediate_supervision": False,
        },
        2924640,
    ),
    (
        {
            "input_size": 128,
            "output_size": 64,
            "stages": 4,
            "downsamplings_per_stage": 3,
            "stage_filters": 128,
            "output_channels": 16,
            "intermediate_supervision": False,
        },
        2257472,
    ),
    (
        {
            "input_size": 128,
            "output_size": 32,
            "stages": 4,
            "downsamplings_per_stage": 4,
            "stage_filters": 128,
            "output_channels": 16,
            "intermediate_supervision": False,
        },
        2932992,
    ),
]


@pytest.mark.parametrize("kwargs, num_params", TEST_MODEL_PARAMS)
def test_model_params_number(kwargs, num_params):
    input_layer = keras.Input(shape=(kwargs["input_size"], kwargs["input_size"], 3))

    layered_model = model_as_layers(inputs=input_layer, **kwargs)
    model = HourglassModel(**kwargs)
    _ = model(inputs=input_layer)

    assert (
        model.count_params() == layered_model["model"].count_params()
    ), "Layered model and HourglassModel does not share the same number of parameters"
    assert (
        model.count_params() == num_params
    ), f"Wrong number of parameters: Expected: {num_params} - Received: {model.count_params()}"
    logger.info("CLEANING GRAPH")
    tf.keras.backend.clear_session()
