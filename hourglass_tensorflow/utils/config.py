import os
import json
import itertools
from glob import glob
from typing import TYPE_CHECKING
from typing import Set
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Literal
from typing import Callable
from typing import Optional

import toml
import yaml
import pandas as pd
from loguru import logger
from pydantic import Field
from pydantic import BaseModel

from hourglass_tensorflow._errors import BadConfigurationError
from hourglass_tensorflow.utils.object_logger import ObjectLogger
from hourglass_tensorflow.utils.parsers._parse_import import get_dataset

if TYPE_CHECKING:
    from hourglass_tensorflow.datasets import HTFBaseDatasetHandler


# region BaseModels


class HTFDatasetSplitConfig(BaseModel):
    column: str = "set"
    activate: bool = False
    train_value: str = "TRAIN"
    test_value: str = "TEST"
    validation_value: str = "VALIDATION"
    train_ratio: Optional[float] = 0.7
    validation_ratio: Optional[float] = 0.15
    test_ratio: Optional[float] = 0.15


class HTFDatasetSetsConfig(BaseModel):
    validation: bool = False
    test: bool = False


class HTFDatasetParamsConfig(BaseModel):
    class Config:
        extra = "allow"


class HTFColsConfig(BaseModel):
    cols: List[str] = Field(default_factory=list)


class HTFDatasetBBoxPaddingConfig(HTFColsConfig):
    x: int = 0
    y: int = 0


class HTFDatasetBBoxConfig(BaseModel):
    activate: bool = False
    factor: float = 1
    padding: HTFDatasetBBoxPaddingConfig = Field(
        default_factory=HTFDatasetBBoxPaddingConfig
    )


class HTFDatasetConfig(BaseModel):
    object: str
    input_size: int = 256
    output_size: int = 64
    stacks: int = 4
    heatmap_stddev: float = 10.0
    params: Optional[HTFDatasetParamsConfig] = Field(
        default_factory=HTFDatasetParamsConfig
    )
    sets: Optional[HTFDatasetSetsConfig]
    split: Optional[HTFDatasetSplitConfig] = Field(
        default_factory=HTFDatasetSplitConfig
    )
    bbox: Optional[HTFDatasetBBoxConfig] = Field(default_factory=HTFDatasetBBoxConfig)


ImageModes = Union[
    Literal["RGB"],
    Literal["rgb"],
    Literal["RGBA"],
    Literal["rgba"],
    Literal["GRAY"],
    Literal["gray"],
    Literal["GRAYSCALE"],
    Literal["grayscale"],
    Literal["BGR"],
    Literal["bgr"],
    Literal["BGRA"],
    Literal["bgra"],
]

CHANNELS_PER_MODE = {
    "RGB": 3,
    "rgb": 3,
    "RGBA": 4,
    "rgba": 4,
    "GRAY": 1,
    "gray": 1,
    "GRAYSCALE": 1,
    "grayscale": 1,
    "BGR": 3,
    "bgr": 3,
    "BGRA": 4,
    "bgra": 4,
}


class HTFDataInputConfig(BaseModel):
    mode: ImageModes = "RGB"
    source: str
    extensions: List[str] = Field(default_factory=["png", "jpeg", "jpg"])


class HTFDataOutputJointsFormatSuffixConfig(BaseModel):
    x: str = "X"
    y: str = "Y"

    class Config:
        extra = "allow"


class HTFDataOutputJointsFormatConfig(BaseModel):
    suffix: Optional[HTFDataOutputJointsFormatSuffixConfig] = Field(
        default_factory=HTFDataOutputJointsFormatSuffixConfig
    )


class HTFDataOutputJointsConfig(BaseModel):
    n: int = 16
    naming_convention: str = "joint_{JOINT_ID}_{SUFFIX}"
    format: Optional[HTFDataOutputJointsFormatConfig] = Field(
        default_factory=HTFDataOutputJointsFormatConfig
    )
    names: List[str] = Field(
        default_factory=[
            "00_rAnkle",
            "01_rKnee",
            "02_rHip",
            "03_lHip",
            "04_lKnee",
            "05_lAnkle",
            "06_pelvis",
            "07_thorax",
            "08_upperNeck",
            "09_topHead",
            "10_rWrist",
            "11_rElbow",
            "12_rShoulder",
            "13_lShoulder",
            "14_lElbow",
            "14_lWrist",
        ]
    )


class HTFDataOutputConfig(BaseModel):
    source: str
    source_column: str = "image"
    source_prefixed: bool = False
    prefix_columns: List[str] = Field(
        default_factory=[
            "set",
            "image",
            "scale",
            "bbox_tl_x",
            "bbox_tl_y",
            "bbox_br_x",
            "bbox_br_y",
            "center_x",
            "center_y",
        ]
    )
    set_column: str = "set"
    joints: HTFDataOutputJointsConfig


class HTFDataConfig(BaseModel):
    input: HTFDataInputConfig
    output: HTFDataOutputConfig


class HTFConfig(BaseModel):
    version: Optional[str]
    data: HTFDataConfig
    dataset: HTFDatasetConfig


# endregion

# region ConfigParser


class HTFConfigParser(ObjectLogger):
    """Parse configuration files for `hourglass_tensorflow`

    Currently HTFConfigParser supports `.json`, `.toml`, `.yaml`, `.yml` files.
    The configuration file must follow the model defined by `HTFConfig`,
    see `CONFIGURATION.md` for more details.

    Args:
        filename (str): Configuration file name
    """

    def __init__(self, filename: str, verbose: bool = True) -> None:
        """Init ConfigParser. See `help(HTFConfigParser)`"""
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

    def __repr__(self) -> str:
        return f"<HFTConfigParser type:{self.file_type} source:{self._filename}>"

    @property
    def config(self) -> HTFConfig:
        return self._config

    @property
    def file_type(self) -> HTFConfig:
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
        self._config = HTFConfig.parse_obj(self._data)


# endregion

# region Configuration Handler


class HTFConfigMeta(BaseModel):
    available_images: Optional[Set[str]] = Field(default_factory=set)
    label_type: Optional[Union[Literal["json"], Literal["csv"]]]
    label_headers: Optional[List[str]] = Field(default_factory=list)
    label_mapper: Optional[Dict[str, int]] = Field(default_factory=dict)
    dataset_object: Optional[Type["HTFBaseDatasetHandler"]]

    class Config:
        extra = "allow"


class HTFConfiguration(ObjectLogger):
    def __init__(self, config_file: str, verbose: bool = True) -> None:
        self._verbose = verbose
        # Init data
        self._metadata = HTFConfigMeta()
        self._labels_df: pd.DataFrame = pd.DataFrame([])
        self.dataset_handler: Optional["HTFBaseDatasetHandler"] = None
        # Parse Configuration
        self._config = HTFConfigParser(filename=config_file)

    @property
    def config(self) -> HTFConfig:
        return self._config.config

    @property
    def _cfg_data(self) -> HTFDataConfig:
        return self.config.data

    @property
    def _cfg_data_inp(self) -> HTFDataInputConfig:
        return self.config.data.input

    @property
    def _cfg_data_out(self) -> HTFDataOutputConfig:
        return self.config.data.output

    # DATA - Prepare - Methods
    def _list_input_images(self) -> None:
        if not os.path.exists(self._cfg_data_inp.source):
            raise BadConfigurationError(
                f"Unable to find source folder {self._cfg_data_inp.source}"
            )
        self.info(
            f"Listing {self._cfg_data_inp.extensions} images in {self._cfg_data_inp.source}"
        )
        self._metadata.available_images = set(
            itertools.chain(
                *[
                    glob(os.path.join(self._cfg_data_inp.source, f"*.{ext}"))
                    for ext in self._cfg_data_inp.extensions
                ]
            )
        )

    def _valid_labels_header(self, df: pd.DataFrame, _error: bool = False) -> bool:
        # Check if numbers of columns are valid
        n_joint = self._cfg_data_out.joints.n
        # TODO(@wbenbihi) UNNECESSARY BLOCK
        # num_prefix_columns = len(self._cfg_data_out.prefix_columns)
        # num_columns_per_joint = len(
        #     self._cfg_data_out.joints.format.suffix.__fields__
        # )
        # estimated_column_size = n_joint * num_columns_per_joint + num_prefix_columns
        # if len(df.columns) != estimated_column_size:
        #     if _error:
        #         raise BadConfigurationError(
        #             "Labels Columns does not matched estimated columns"
        #         )
        #     return False
        # Check if columns names are valid
        naming_convention = self._cfg_data_out.joints.naming_convention
        suffixes = self._cfg_data_out.joints.format.suffix
        headers = self._cfg_data_out.prefix_columns + [
            naming_convention.format(JOINT_ID=jid, SUFFIX=suffix)
            for jid in range(n_joint)
            for suffix in suffixes.__dict__.values()
        ]
        if set(headers).difference(set(list(df.columns))):
            if _error:
                raise BadConfigurationError(
                    f"Columns' name does not match configuration\n\tEXPECTED:\n\t{headers}\n\tRECEIVED:\n\t{list(df.columns)}\n\tMISSING COLUMNS:\n\t{set(headers).difference(set(list(df.columns)))}"
                )
            return False
        # If everything is good we store the expected headers in _metadata
        self._metadata.label_headers = headers
        return True

    def _read_labels(self, _error: bool = False) -> bool:
        # Check if data.output.source exists ?
        if not os.path.exists(self._cfg_data_out.source):
            raise BadConfigurationError(f"Unable to find {self._cfg_data_out.source}")
        # Read Data
        self.info(f"Reading labels from {self._cfg_data_out.source}")
        ## Check if the file extension is in [.json, .csv]
        if self._cfg_data_out.source.endswith(".json"):
            self._metadata.label_type = "json"
            labels = pd.read_json(self._cfg_data_out.source, orient="records")
        elif self._cfg_data_out.source.endswith(".csv"):
            self._metadata.label_type = "csv"
            labels = pd.read_csv(self._cfg_data_out.source)
        else:
            raise BadConfigurationError(
                f"{self._cfg_data_out.source} should be of type .json or .csv"
            )
        if not isinstance(labels, pd.DataFrame):
            raise BadConfigurationError(
                f"{self._cfg_data_out.source} not parsable as pandas.DataFrame"
            )
        # Validate expected labels columns
        if not self._valid_labels_header(labels, _error=_error):
            self.error("Labels are not matching")
            return False
        self._labels_df: pd.DataFrame = labels[self._metadata.label_headers]
        self._metadata.label_mapper = {
            label: i for i, label in enumerate(self._metadata.label_headers)
        }
        if not self._cfg_data_out.source_prefixed:
            # Now we also prefix the image column with the image folder
            # in case the source_prefix attribute is set to false
            folder_prefix = self._cfg_data_inp.source
            source_column = self._cfg_data_out.source_column
            self._labels_df = self._labels_df.assign(
                **{
                    source_column: self._labels_df[source_column].apply(
                        lambda x: os.path.join(folder_prefix, x)
                    )
                }
            )
        return True

    def _prepare_input(self) -> None:
        # List files in Input Source Folder
        self._list_input_images()

    def _validate_joints(self, _error: bool = True) -> bool:
        conditions = [
            len(self._cfg_data_out.joints.names) == self._cfg_data_out.joints.n
        ]
        if not all(conditions):
            if _error:
                raise BadConfigurationError("Joints properties are not valid")
            return False
        return True

    def _prepare_output(self, _error: bool = True) -> None:
        # Read the label file
        self._validate_joints(_error=_error)
        self._read_labels(_error=_error)

    def prepare_data(self) -> None:
        self._prepare_input()
        self._prepare_output()

    def prepare(self) -> None:
        self.prepare_data()

    # DATASET - Prepare - Methods

    def _load_dataset_object(self) -> None:
        self._metadata.dataset_object = get_dataset(self.config.dataset.object)

    def prepare_dataset(self) -> None:
        self._load_dataset_object()
        self.dataset_handler: "HTFBaseDatasetHandler" = self._metadata.dataset_object(
            dataset=self._labels_df.to_numpy(),
            config=self.config.dataset,
            global_config=self,
            **self.config.dataset.params.dict(),
        )
        self.dataset_handler.execute()


# endregion
