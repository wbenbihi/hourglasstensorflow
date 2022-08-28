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
    """HTFManager instantiates a job from `hourglass_tensorflow` configuration files

    HTFManager is responsible of handling the sequence of operation from configuration files.
    Its behavior depends mostly on the `mode` selected.

    Args:
        ObjectLogger (_type_): _description_
    """

    def __init__(self, filename: str, verbose: bool = True, *args, **kwargs) -> None:
        """See help(HTFManager)

        Args:
            filename (str): The configuration file path
            verbose (bool, optional): Enable verbose. Defaults to True.
        """
        super().__init__(verbose, *args, **kwargs)
        self._config_file = filename
        self._config = HTFConfig.parse_obj(
            HTFConfigParser.parse(filename=filename, verbose=verbose)
        )
        self._metadata = HTFMetadata()

    @property
    def config(self) -> HTFConfig:
        """Reference to the HTFConfig object

        Returns:
            HTFConfig: Parsed configuration
        """
        return self._config

    @property
    def mode(self) -> HTFConfigMode:
        """Reference to the current execution `mode`

        Returns:
            HTFConfigMode: Execution mode
        """
        return self.config.mode

    @property
    def VALIDATION_RULES(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {
            HTFConfigMode.TRAIN: [],
            HTFConfigMode.TEST: [],
            HTFConfigMode.INFERENCE: [],
            HTFConfigMode.SERVER: [],
        }

    @property
    def metadata(self) -> HTFMetadata:
        """Reference to the execution metadat

        Returns:
            HTFMetadata: Current execution metadata
        """
        return self._metadata

    def _import_object(
        self,
        obj: HTFObjectReference[T],
        config: HTFConfigField,
        metadata: HTFMetadata,
        *args,
        **kwargs
    ) -> Union[T, _HTFHandler]:
        """Instantiate an object from the `:object` field in config files

        This method expects to return a subclass of _HTFHandler.

        Args:
            obj (HTFObjectReference[T]): Reference to the object to instantiate
            config (HTFConfigField): Config object to use to instantiate the object
            metadata (HTFMetadata): Metadata to transfer to the newly instantiated object

        Returns:
            Union[T, _HTFHandler]: The _HTFHandler subclass instance
        """
        instance = obj.init(config=config, metadata=metadata, *args, **kwargs)
        return instance

    def __call__(self, *args, **kwargs) -> None:
        """Run the job describe by the HTFManager configuration

        The mode will not be infered by the HTFManager and is not an optional configuration field

        Raises:
            BadConfigurationError: Raise this error if any of the validation rule conditions is not met
        """

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
        """Launch the job for the `server` mode

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def test(self, *args, **kwargs) -> None:
        """Launch the job for the `test` mode

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def inference(self, *args, **kwargs) -> None:
        """Launch the job for the `inference` mode

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def train(self, *args, **kwargs) -> None:
        """Launch the job for the `train` mode

        The sequence of operation is independent from your dataset specification.
        It will need 4 handlers referenced in your configuration.
        - `BaseDataHandler`
            - default: `hourglass_tensorflow.handlers.data.HTFDataHandler`
            - In charge of the input/labels parsing
        - `BaseDatasetHandler`
            - default: `hourglass_tensorflow.handlers.data.HTFDatasetHandler`
            - In charge of the `tf.data.Dataset` generation
        - `BaseModelHandler`
            - default: `hourglass_tensorflow.handlers.data.HTFModelHandler`
            - In charge of the Model's graph generation
        - `BaseTrainHandler`
            - default: `hourglass_tensorflow.handlers.train.HTFTrainHandler`
            - In charge of setting up the fitting process


        The default handlers might be specific to the MPII Dataset, check the customization section
        from the documentation to learn how to train the `HouglassModel` on your own data.
        """
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
