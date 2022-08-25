from typing import List
from typing import Union
from typing import Literal
from typing import Optional
from typing import TypedDict

import keras.layers
import keras.models
from pydantic import Field

from hourglass_tensorflow.types.config.fields import HTFConfigField
from hourglass_tensorflow.types.config.fields import HTFObjectReference

DATA_FORMAT = Union[
    Literal["NHWC"],
    Literal["NCHW"],
]


class HTFModelHandlerReturnObject(TypedDict):
    inputs: keras.layers.Layer
    outputs: keras.layers.Layer
    model: keras.models.Model


class HTFModelAsLayers(TypedDict):
    downsampling: keras.layers.Layer
    hourglasses: List[keras.layers.Layer]
    outputs: keras.layers.Layer
    model: keras.models.Model


class HTFModelParams(HTFConfigField):
    name: str = "HourglassNetwork"
    input_size: int = 256
    output_size: int = 64
    stages: int = 4
    stage_filters: int = 128
    output_channels: int = 16
    downsamplings_per_stage: int = 4
    intermediate_supervision: bool = True


class HTFModelConfig(HTFConfigField):
    object: Optional[HTFObjectReference] = Field(
        default_factory=HTFObjectReference(
            source="hourglass_tensorflow.handlers.model.HTFModelHandler"
        )
    )
    build_as_model: bool = False
    data_format: DATA_FORMAT = "NHWC"
    params: Optional[HTFModelParams] = Field(default_factory=HTFModelParams)
