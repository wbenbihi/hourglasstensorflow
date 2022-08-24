from abc import abstractmethod

import tensorflow as tf
import keras.layers
import keras.models
from keras import Input as InputTensor

from hourglass_tensorflow.utils import BadConfigurationError
from hourglass_tensorflow.models import HourglassModel
from hourglass_tensorflow.models import model_as_layers
from hourglass_tensorflow.types.config import HTFModelConfig
from hourglass_tensorflow.types.config import HTFModelParams
from hourglass_tensorflow.types.config import HTFModelHandlerReturnObject
from hourglass_tensorflow.handlers.meta import _HTFHandler

# region Abstract Class


class _HTFModelHandler(_HTFHandler):
    def __init__(
        self,
        config: HTFModelConfig,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config=config, *args, **kwargs)
        self._input: keras.layers.Layer = None
        self._output: keras.layers.Layer = None
        self._model: keras.models.Model = None

    @property
    def config(self) -> HTFModelConfig:
        return self._config

    @property
    def params(self) -> HTFModelParams:
        return self.config.params

    def get(self) -> HTFModelHandlerReturnObject:
        if self._executed:
            return {
                "inputs": self._input,
                "outputs": self._output,
                "model": self._model,
            }
        else:
            self.warning(
                "The ModelHandler has not been called to generate proper return value"
            )
            return {}

    @abstractmethod
    def generate_graph(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> None:
        self.generate_graph(*args, **kwargs)


# enregion

# region Handler


class HTFModelHandler(_HTFModelHandler):
    def init_handler(self, *args, **kwargs) -> None:
        pass

    def get(self) -> HTFModelHandlerReturnObject:
        if self._executed:
            return {
                "inputs": self._input,
                "outputs": self._output,
                "model": self._model,
                "layers": self._layered_model,
            }
        else:
            self.warning(
                "The ModelHandler has not been called to generate proper return value"
            )
            return {}

    def _build_input(self, *args, **kwargs) -> tf.Tensor:
        height, width = self.params.input_size, self.params.input_size
        # TODO: Handle other Image Mode than RGB
        channels = 3
        if self.config.data_format == "NHWC":
            self._input = InputTensor(shape=(height, width, channels), name="Input")
        else:
            raise BadConfigurationError("The only supported data format is NHWC so far")
        return self._input

    def _build_model_as_model(self, *args, **kwargs) -> HourglassModel:
        self._model = HourglassModel(**self.params.dict())
        self._layered_model = {}
        return self._model

    def _build_model_as_layer(self, *args, **kwargs) -> keras.models.Model:
        self._layered_model = model_as_layers(inputs=self._input, **self.params.dict())
        self._output = self._layered_model["outputs"]
        self._model = self._layered_model["model"]
        return self._model

    def generate_graph(self, *args, **kwargs) -> None:
        # Get Input Tensor
        input_tensor = self._build_input(*args, **kwargs)
        # Build Model Graph
        if self.config.build_as_model:
            model = self._build_model_as_model(*args, **kwargs)
            # Link Input Shape to Model
            self._output = model(inputs=input_tensor, *args, **kwargs)
        else:
            model = self._build_model_as_layer(*args, **kwargs)


# endregion
