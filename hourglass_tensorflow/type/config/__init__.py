import enum
import json
from typing import Dict
from typing import Union
from typing import Callable
from typing import Optional

import toml
import yaml

from hourglass_tensorflow.type.config.data import HTFDataInput
from hourglass_tensorflow.type.config.data import HTFDataConfig
from hourglass_tensorflow.type.config.data import HTFDataOutput
from hourglass_tensorflow.type.config.model import HTFModelConfig
from hourglass_tensorflow.type.config.model import HTFModelParams
from hourglass_tensorflow.type.config.model import HTFModelAsLayers
from hourglass_tensorflow.type.config.model import HTFModelHandlerReturnObject
from hourglass_tensorflow.type.config.train import HTFTrainConfig
from hourglass_tensorflow.type.config.fields import HTFConfigField
from hourglass_tensorflow.type.config.fields import HTFObjectReference
from hourglass_tensorflow.type.config.dataset import HTFDatasetBBox
from hourglass_tensorflow.type.config.dataset import HTFDatasetSets
from hourglass_tensorflow.type.config.dataset import HTFDatasetConfig
from hourglass_tensorflow.type.config.dataset import HTFDatasetHeatmap
from hourglass_tensorflow.utils.object_logger import ObjectLogger
from hourglass_tensorflow.type.config.metadata import HTFMetadata


class HTFConfigMode(enum.Enum):
    """Enum for available `hourglass_tensorflow` execution modes"""

    TEST = "test"
    TRAIN = "train"
    INFERENCE = "inference"
    SERVER = "server"


class HTFConfig(HTFConfigField):
    """BaseModel for configuration file representation"""

    mode: HTFConfigMode = HTFConfigMode.TRAIN
    version: Optional[Union[str, int]]
    data: Optional[HTFDataConfig]
    dataset: Optional[HTFDatasetConfig]
    model: Optional[HTFModelConfig]
    train: Optional[HTFTrainConfig]


# region ConfigParser


class HTFConfigParser(ObjectLogger):
    """Parse configuration files for `hourglass_tensorflow`

    Currently HTFConfigParser supports `.json`, `.toml`, `.yaml`, `.yml` files.
    The configuration file must follow the model defined by `HTFConfig`,
    see `CONFIGURATION.md` for more details.
    """

    def __init__(self, filename: str, verbose: bool = True) -> None:
        """Init ConfigParser. See `help(HTFConfigParser)`

        Args:
            filename (str): Configuration file name
        """
        self._verbose = verbose
        self._filename = filename
        self._data = {}
        self._config: Optional[HTFConfig] = None
        self.info("Reading configuration ...")
        # Parse Config File
        self._infer_source()
        self._parse_config()

    def __call__(self) -> HTFConfig:
        """Returns the parsed configuration as a `pydantic.BaseModel` object

        Returns:
            HTFConfig: The configuration object
        """
        return self._config

    @classmethod
    def parse(cls, filename: str, verbose: bool = False, *args, **kwargs) -> Dict:
        """Parse a configuration from a filename

        Args:
            filename (str): config file path
            verbose (bool, optional): If True, logs are activated . Defaults to False.

        Returns:
            Dict: The configuration parsed as a dictionary
        """
        return cls(filename=filename, verbose=verbose, *args, **kwargs).config

    def __repr__(self) -> str:
        return f"<HFTConfigParser type:{self.file_type} source:{self._filename}>"

    @property
    def config(self) -> HTFConfig:
        """Reference to the configuration

        Returns:
            HTFConfig: configuration object
        """
        return self._config

    @property
    def file_type(self) -> str:
        """Accessor to configuration file extension

        Returns:
            str: file extension
        """
        return self._filename.split(".")[-1]

    @classmethod
    def parse_toml(cls, filename: str) -> Dict:
        """Parse a `.toml` file

        Args:
            filename (str): toml file name

        Returns:
            Dict: parsed toml as `dict`
        """
        return toml.load(filename)

    @classmethod
    def parse_json(cls, filename: str) -> Dict:
        """Parse a `.json` file

        Args:
            filename (str): json file name

        Returns:
            Dict: parsed json as `dict`
        """
        with open(filename, "r") as f:
            return json.load(f)

    @classmethod
    def parse_yaml(cls, filename: str) -> Dict:
        """Parse a `.yml`/`.yaml` file

        Args:
            filename (str): yaml file name

        Returns:
            Dict: parsed yaml as `dict`
        """
        with open(filename, "r") as f:
            return yaml.safe_load(f)

    def _infer_source(self) -> None:
        """Infer the source of a file based on the file extension

        Raises:
            ValueError: The file extension is not supported
        """
        parsers = {
            "yml": self.parse_yaml,
            "yaml": self.parse_yaml,
            "json": self.parse_json,
            "toml": self.parse_toml,
        }
        cfg_parser: Optional[Callable[[str], Dict]] = parsers.get(
            self._filename.split(".")[-1]
        )
        if cfg_parser is None:
            raise ValueError(
                "The file extension is not supported. Use YAML, TOML or JSON config file"
            )
        self._data = cfg_parser(self._filename)

    def _parse_config(self) -> None:
        """Parse the configuration"""
        self._config = HTFConfig.parse_obj(self._data)


# endregion
