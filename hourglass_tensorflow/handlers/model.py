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


class BaseModelHandler(_HTFHandler):
    """Abstract handler for `hourglass_tensorflow` Model jobs

    Args:
        _HTFHandler (_type_): Subclass of Meta Handler
    """

    def __init__(
        self,
        config: HTFModelConfig,
        *args,
        **kwargs,
    ) -> None:
        """see help(BaseModelHandler)

        Args:
            config (HTFModelConfig): Reference to `model` field configuration
        """
        super().__init__(config=config, *args, **kwargs)
        self._input: keras.layers.Layer = None
        self._output: keras.layers.Layer = None
        self._model: keras.models.Model = None

    @property
    def config(self) -> HTFModelConfig:
        """Reference to `model` field configuration

        Returns:
            HTFModelConfig: Model configuration object
        """
        return self._config

    @property
    def params(self) -> HTFModelParams:
        """Reference to `model.params` field configuration

        Returns:
            HTFModelParams: Model parameters configuration object
        """
        return self.config.params

    @property
    def model(self) -> keras.models.Model:
        """Getters for the model

        Returns:
            keras.models.Model: model
        """
        return self._model

    @property
    def input(self) -> keras.layers.Layer:
        """Getter for the input tensor

        Returns:
            keras.layers.Layer: input layer
        """
        return self._model

    @property
    def output(self) -> keras.layers.Layer:
        """Getter for the output tensor

        Returns:
            keras.layers.Layer: output layer
        """
        return self._model

    def set_input(self, tensor: keras.layers.Layer) -> None:
        """Sets the input tensor

        Args:
            tensor (keras.layers.Layer): input tensor
        """
        self._input = tensor

    def set_output(self, tensor: keras.layers.Layer) -> None:
        """Sets the output tensor

        Args:
            tensor (keras.layers.Layer): output tensor
        """
        self._output = tensor

    def set_model(self, model: keras.models.Model) -> None:
        """Sets the model

        Args:
            model (keras.models.Model): keras model
        """
        self._model = model

    def get(self) -> HTFModelHandlerReturnObject:
        """Returns object of interest

        Returns:
            HTFModelHandlerReturnObject: Object referencing the model, the input and output tensors
        """
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
        """Abstract method to implement for custom `BaseModelHander` subclass

        This method should be generate the model's graph
        """
        raise NotImplementedError

    def run(self, *args, **kwargs) -> None:
        """Global run job for `BaseDataHander`

        The run for a `BaseModelHander` will call the `BaseModelHander.generate_graph` method.
        """
        self.generate_graph(*args, **kwargs)


# enregion

# region Handler


class HTFModelHandler(BaseModelHandler):
    """Default Model Handler for `hourglass_tendorflow`

    The HTFModelHandler can be used outside of MPII data context.
    This Handler is NOT bounded to any dataset specification.
    It can be used with dataset other than MPII

    > NOTE
    >
    > Check the handlers section from the documention to understand
    > how to build you custom handlers

    Args:
        BaseModelHandler (_type_): Subclass of Meta Data Handler
    """

    def init_handler(self, *args, **kwargs) -> None:
        """Initialization for HTFModelHandler"""
        pass

    def get(self) -> HTFModelHandlerReturnObject:
        """Returns object of interest

        Returns:
            HTFModelHandlerReturnObject: Object referencing the model, the input and output tensors
        """
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
        """Generates the keras input Tensor

        Raises:
            BadConfigurationError: The data format specified is not supported

        Returns:
            tf.Tensor: Input Tensor according to configuration
        """
        height, width = self.params.input_size, self.params.input_size
        # TODO: Handle other Image Mode than RGB
        channels = 3
        if self.config.data_format == "NHWC":
            self._input = InputTensor(shape=(height, width, channels), name="Input")
        else:
            raise BadConfigurationError("The only supported data format is NHWC so far")
        return self._input

    def _build_model_as_model(self, *args, **kwargs) -> HourglassModel:
        """Build model as a `HourglassModel` object

        Returns:
            HourglassModel: model built according to configuration
        """
        self._model = HourglassModel(**self.params.dict())
        self._layered_model = {}
        return self._model

    def _build_model_as_layer(self, *args, **kwargs) -> keras.models.Model:
        """Build model from keras.models.Model

        > WARNING
        >
        > This is an experimental features, developed mostly for debug purposes.
        > We highly recommand that you do not set `model.build_as_model` to False.
        > We cannot ensure model serialization as well as model parsing for other mode


        Returns:
            keras.models.Model: model built according to configuration
        """
        self._layered_model = model_as_layers(inputs=self._input, **self.params.dict())
        self._output = self._layered_model["outputs"]
        self._model = self._layered_model["model"]
        return self._model

    def generate_graph(self, *args, **kwargs) -> None:
        """Generates the model's graph"""
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
