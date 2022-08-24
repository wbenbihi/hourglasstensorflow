from typing import Union
from typing import TypeVar

from hourglass_tensorflow.utils import ObjectLogger
from hourglass_tensorflow.utils import BadConfigurationError
from hourglass_tensorflow.types.config import HTFConfig
from hourglass_tensorflow.types.config import HTFMetadata
from hourglass_tensorflow.types.config import HTFConfigMode
from hourglass_tensorflow.types.config import HTFConfigField
from hourglass_tensorflow.types.config import HTFConfigParser
from hourglass_tensorflow.types.config import HTFObjectReference
from hourglass_tensorflow.handlers.data import HTFDataHandler
from hourglass_tensorflow.handlers.meta import _HTFHandler
from hourglass_tensorflow.handlers.model import HTFModelHandler
from hourglass_tensorflow.handlers.train import HTFTrainHandler
from hourglass_tensorflow.handlers.dataset import HTFDatasetHandler

T = TypeVar("T")


class HTFManager(ObjectLogger):
    def __init__(self, filename: str, verbose: bool = True, *args, **kwargs) -> None:
        super().__init__(verbose, *args, **kwargs)
        self._config_file = filename
        self._config = HTFConfig.parse_obj(
            HTFConfigParser.parse(filename=filename, verbose=verbose)
        )
        self._metadata = HTFMetadata()

    @property
    def config(self) -> HTFConfig:
        return self._config

    @property
    def mode(self) -> HTFConfigMode:
        return self.config.mode

    @property
    def VALIDATION_RULES(self):
        return {
            HTFConfigMode.TRAIN: [],
            HTFConfigMode.TEST: [],
            HTFConfigMode.INFERENCE: [],
            HTFConfigMode.SERVER: [],
        }

    @property
    def metadata(self) -> HTFMetadata:
        return self._metadata

    def _import_object(
        self,
        obj: HTFObjectReference[T],
        config: HTFConfigField,
        metadata: HTFMetadata,
        *args,
        **kwargs
    ) -> Union[T, _HTFHandler]:
        instance = obj.init(config=config, metadata=metadata, *args, **kwargs)
        return instance

    def __call__(self, *args, **kwargs) -> None:

        if not all(self.VALIDATION_RULES[self.mode]):
            raise BadConfigurationError

        if self.mode == HTFConfigMode.TRAIN:
            self.train(*args, **kwargs)
        if self.mode == HTFConfigMode.TEST:
            self.test(*args, **kwargs)
        if self.mode == HTFConfigMode.INFERENCE:
            self.inference(*args, **kwargs)
        if self.mode == HTFConfigMode.SERVER:
            self.server(*args, **kwargs)

    def server(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def test(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def inference(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def train(self, *args, **kwargs) -> None:
        # Unpack Objects
        obj_data: HTFObjectReference[HTFDataHandler] = self._config.data.object
        obj_dataset: HTFObjectReference[HTFDatasetHandler] = self._config.dataset.object
        obj_model: HTFObjectReference[HTFModelHandler] = self._config.model.object
        obj_train: HTFObjectReference[HTFTrainHandler] = self._config.train.object
        # Launch Data Handler
        self.DATA = self._import_object(
            obj_data, config=self._config.data, metadata=self._metadata
        )
        data = self.DATA().get_data()
        # Launch Dataset Handler
        self.DATASET = self._import_object(
            obj_dataset,
            config=self._config.dataset,
            metadata=self._metadata,
            data=data,
        )
        self.DATASET()
        # Launch Model Handler
        self.MODEL = self._import_object(
            obj_model,
            config=self._config.model,
            metadata=self._metadata,
        )
        self.MODEL()
        # Launch Train Handler
        self.TRAIN = self._import_object(
            obj_train,
            config=self._config.train,
            metadata=self._metadata,
        )
        self.TRAIN(
            model=self.MODEL._model,
            train_dataset=self.DATASET._train_dataset,
            test_dataset=self.DATASET._test_set,
            validation_dataset=self.DATASET._validation_dataset,
        )
