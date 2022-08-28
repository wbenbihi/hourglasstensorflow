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


class BaseDataHandler(_HTFHandler):
    """Abstract handler for `hourglass_tensorflow` Data jobs

    Args:
        _HTFHandler (_type_): Subclass of Meta Handler
    """

    def __init__(self, config: HTFDataConfig, *args, **kwargs) -> None:
        """see help(BaseDataHandler)

        Args:
            config (HTFDataConfig): Reference to `data:` field configuration
        """
        super().__init__(config=config, *args, **kwargs)

    @property
    def config(self) -> HTFDataConfig:
        """Reference to `data:` field configuration

        Returns:
            HTFDataConfig: Data configuration object
        """
        return self._config

    @property
    def input_cfg(self) -> HTFDataInput:
        """Reference to `data:input` field configuration

        Returns:
            HTFDataInput: Input Data configuration object
        """
        return self.config.input

    @property
    def output_cfg(self) -> HTFDataOutput:
        """Reference to `data:output` field configuration

        Returns:
            HTFDataOutput: Output Data configuration object
        """
        return self.config.output

    @abstractmethod
    def prepare_input(self, *args, **kwargs) -> None:
        """Abstract method to implement for custom `BaseDataHander` subclass

        This method should script the process of INPUT data preparation

        Raises:
            NotImplementedError: Abstract Method
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_output(self, *args, **kwargs) -> None:
        """Abstract method to implement for custom `BaseDataHander` subclass

        This method should script the process of OUTPUT data preparation

        Raises:
            NotImplementedError: Abstract Method
        """
        raise NotImplementedError

    def run(self, *args, **kwargs) -> None:
        """Global run job for `BaseDataHander`

        The run for a `BaseDataHander` will call the `prepare_input` and `prepare_output`
        methods sequentially.
        """
        self.prepare_input(*args, **kwargs)
        self.prepare_output(*args, **kwargs)

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """Abstract method to implement for custom `BaseDataHander` subclass

        This method should be a basic accessor to the data of interest
        """
        raise NotImplementedError


# enregion

# region Handler


class HTFDataHandler(BaseDataHandler):
    """Default Data Handler for `hourglass_tendorflow``

    The HTFDataHandler can be used outside of MPII data context
    but was specifically created in the context of MPII data.
    Its use might not be a good fit for other dataset.

    > NOTE
    >
    > Check the handlers section from the documention to understand
    > how to build you custom handlers

    Args:
        BaseDataHandler (_type_): Subclass of Meta Data Handler
    """

    def _list_input_images(self) -> None:
        """List all the images available in the `data:input:source` folder from config

        It will only returns images from that folder being compliant with the
        `data:input:extensions` list

        Raises:
            BadConfigurationError: raised if the `data:input:source` does not exists
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
        """Inner validation to check between `data:output:joints:num` and `data:output:joints:names`

        Args:
            _error (bool, optional): If True, will raise an error if condition is not met. Defaults to True.

        Raises:
            BadConfigurationError: raised if the number of joints declared is not equal to the joint name list

        Returns:
            bool: Whether the condition is met or not
        """
        conditions = [len(self.output_cfg.joints.names) == self.output_cfg.joints.num]
        if not all(conditions):
            if _error:
                raise BadConfigurationError("Joints properties are not valid")
            return False
        return True

    def _valid_labels_header(self, df: pd.DataFrame, _error: bool = False) -> bool:
        """Inner validation to check if columns referenced in configuration file are matching

        This method will check your label table to check if the columns referenced in the
        configuration file are compliant with the columns available in your label file.
        This method concatenate the `data:output:prefix_columns` with the joints columns,
        generated by `HTFDataHandler._get_joint_columns` hidden method, and check the intersection
        with the actual `data:output:label:source` file's columns

        Args:
            df (pd.DataFrame): Dataset to check
            _error (bool, optional): If True, raises an error if the condition is not validated.
                Defaults to False.

        Raises:
            BadConfigurationError: raised if the condition is not met

        Returns:
            bool: Whether the condition is met or not
        """
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
        """Loads a parsable pd.DataFrame file

        Raises:
            BadConfigurationError: raised if the `data:output:source` extension is not in [.csv, .json]
            BadConfigurationError: raised if the loaded file is not parsable as a `pandas.DataFrame`

        Returns:
            pd.DataFrame: the loaded DataFrame
        """
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
        """Prefixes the `data:ouput:column_source` column from `data:output:source`

        It should be used if your `data:ouput:column_source` column does not contain
        the full path to the referenced image.

        Args:
            df (pd.DataFrame): DataFrame to prefix column

        Returns:
            pd.DataFrame: DataFrame with column prefixed
        """
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
        """Inner method to load and validate the `data:output:source` file

        Args:
            _error (bool, optional): If True, raises an error if validation is not possible.
                Defaults to False.

        Raises:
            BadConfigurationError: raised if the `data:output:source` does not exists

        Returns:
            bool: Whether or not the labels were successfully loaded
        """
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
        """Returns the expected joint columns from the `data:output:joints` configuration

        This method enables dynamic column reference and label models.
        You can specify dynamic structures with the `data:output:joints:naming_convention`
        if you have additional information about the joints (e.g `joint_{JOINT_ID}_{SUFFIX}_{ADDITIONAL}`).
        Use `data:output:joints:dynamic_fields` to identify the group names of fields referenced in the naming convention.
        For example, if you specified the following naming convention `joint_{JOINT_ID}_{SUFFIX}_{ADDITIONAL}`,
        you would have to specify dynamic fields as `["SUFFIX", "ADDITIONAL"]`
        Finally, you can use `data:output:joints:format` to reference all the values that a group
        from `data:output:joints:dynamic_fields` should be replaced by.

        Check sample configuration files for demonstration

        Returns:
            List[str]: List of all the columns expected for joints
        """
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
        """Prepares the input data according to `data:input`"""
        # List files in Input Source Folder
        self._list_input_images()

    def prepare_output(self, _error: bool = True) -> None:
        """Prepares the output data according to `data:output`"""
        # Read the label file
        self._metadata.joint_columns = self._get_joint_columns()
        self._validate_joints(_error=_error)
        self._read_labels(_error=_error)

    def get_data(self) -> pd.DataFrame:
        """Access the labeled data as a DataFrame

        Returns:
            pd.DataFrame: Labeled data
        """
        return self._labels_df


# endregion
