import os
import itertools
from abc import abstractmethod
from glob import glob
from typing import List

import pandas as pd

from hourglass_tensorflow.utils import BadConfigurationError
from hourglass_tensorflow.types.config import HTFDataInput
from hourglass_tensorflow.types.config import HTFDataConfig
from hourglass_tensorflow.types.config import HTFDataOutput
from hourglass_tensorflow.handlers.meta import _HTFHandler

# region Abstract Class


class _HTFDataHandler(_HTFHandler):
    def __init__(self, config: HTFDataConfig, *args, **kwargs) -> None:
        super().__init__(config=config, *args, **kwargs)

    @property
    def config(self) -> HTFDataConfig:
        return self._config

    @property
    def input_cfg(self) -> HTFDataInput:
        return self.config.input

    @property
    def output_cfg(self) -> HTFDataOutput:
        return self.config.output

    @abstractmethod
    def prepare_input(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_output(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> None:
        self.prepare_input(*args, **kwargs)
        self.prepare_output(*args, **kwargs)


# enregion

# region Handler


class HTFDataHandler(_HTFDataHandler):
    def _list_input_images(self) -> None:
        """_summary_

        Raises:
            BadConfigurationError: _description_
        """
        if not os.path.exists(self.input_cfg.source):
            raise BadConfigurationError(
                f"Unable to find source folder {self.input_cfg.source}"
            )
        self.info(
            f"Listing {self.input_cfg.extensions} images in {self.input_cfg.source}"
        )
        self._metadata.available_images = set(
            itertools.chain(
                *[
                    glob(os.path.join(self.input_cfg.source, f"*.{ext}"))
                    for ext in self.input_cfg.extensions
                ]
            )
        )

    def _validate_joints(self, _error: bool = True) -> bool:
        conditions = [len(self.output_cfg.joints.names) == self.output_cfg.joints.num]
        if not all(conditions):
            if _error:
                raise BadConfigurationError("Joints properties are not valid")
            return False
        return True

    def _valid_labels_header(self, df: pd.DataFrame, _error: bool = False) -> bool:
        # Check if columns names are valid
        joint_headers = self._get_joint_columns()
        headers = [*self.output_cfg.prefix_columns, *joint_headers]
        if set(headers).difference(set(list(df.columns))):
            if _error:
                raise BadConfigurationError(
                    f"Columns' name does not match configuration\n\tEXPECTED:\n\t{headers}\n\tRECEIVED:\n\t{list(df.columns)}\n\tMISSING COLUMNS:\n\t{set(headers).difference(set(list(df.columns)))}"
                )
            return False
        # If everything is good we store the expected headers in _metadata
        self._metadata.label_headers = headers
        return True

    def _load_labels(self) -> pd.DataFrame:
        self.info(f"Reading labels from {self.output_cfg.source}")
        ## Check if the file extension is in [.json, .csv]
        if self.output_cfg.source.endswith(".json"):
            self._metadata.label_type = "json"
            labels = pd.read_json(self.output_cfg.source, orient="records")
        elif self.output_cfg.source.endswith(".csv"):
            self._metadata.label_type = "csv"
            labels = pd.read_csv(self.output_cfg.source)
        else:
            raise BadConfigurationError(
                f"{self.output_cfg.source} should be of type .json or .csv"
            )
        if not isinstance(labels, pd.DataFrame):
            raise BadConfigurationError(
                f"{self.output_cfg.source} not parsable as pandas.DataFrame"
            )
        return labels

    def _prefix_images(self, df: pd.DataFrame) -> pd.DataFrame:
        # Now we also prefix the image column with the image folder
        # in case the source_prefix attribute is set to false
        folder_prefix = self.input_cfg.source
        source_column = self.output_cfg.column_source
        df = df.assign(
            **{
                source_column: df[source_column].apply(
                    lambda x: os.path.join(folder_prefix, x)
                )
            }
        )
        return df

    def _read_labels(self, _error: bool = False) -> bool:
        # Check if data.output.source exists ?
        if not os.path.exists(self.output_cfg.source):
            raise BadConfigurationError(f"Unable to find {self.output_cfg.source}")
        # Read Data
        labels = self._load_labels()
        # Validate expected labels columns
        if not self._valid_labels_header(labels, _error=_error):
            self.error("Labels are not matching")
            return False
        self._labels_df: pd.DataFrame = labels[self.meta.label_headers]
        self._metadata.label_mapper = {
            label: i for i, label in enumerate(self.meta.label_headers)
        }
        if not self.output_cfg.source_prefixed:
            self._labels_df = self._prefix_images(self._labels_df)
        return True

    def _get_joint_columns(self) -> List[str]:
        JOINT_CFG = self.config.output.joints
        num_joints = JOINT_CFG.num
        dynamic_fields = JOINT_CFG.dynamic_fields
        data_format = JOINT_CFG.format
        index_field = JOINT_CFG.format.id_field
        naming = JOINT_CFG.naming_convention
        groups = itertools.product(
            *[list(getattr(data_format, g).values()) for g in dynamic_fields]
        )
        named_groups = [
            {dynamic_fields[i]: el for i, el in enumerate(group)} for group in groups
        ]
        return [
            naming.format(**{**group, **{index_field: joint_idx}})
            for joint_idx in range(num_joints)
            for group in named_groups
        ]

    def prepare_input(self) -> None:
        # List files in Input Source Folder
        self._list_input_images()

    def prepare_output(self, _error: bool = True) -> None:
        # Read the label file
        self._metadata.joint_columns = self._get_joint_columns()
        self._validate_joints(_error=_error)
        self._read_labels(_error=_error)

    def get_data(self) -> pd.DataFrame:
        return self._labels_df


# endregion
