from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Iterable
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel

from hourglass_tensorflow._errors import BadConfigurationError
from hourglass_tensorflow.utils.sets import split_train_test
from hourglass_tensorflow.utils.object_logger import ObjectLogger

if TYPE_CHECKING:
    from hourglass_tensorflow.utils.config import HTFConfiguration
    from hourglass_tensorflow.utils.config import HTFDatasetConfig
    from hourglass_tensorflow.utils.config import HTFDatasetSetsConfig
    from hourglass_tensorflow.utils.config import HTFDatasetSplitConfig

CastableTableDataset = Union[np.ndarray, pd.DataFrame]
ReturnTableSets = Tuple[
    Optional[CastableTableDataset],
    Optional[CastableTableDataset],
    Optional[CastableTableDataset],
]
ImageSets = Tuple[Optional[Set[str]], Optional[Set[str]], Optional[Set[str]]]


class HTFBaseDatasetHandlerMetadata(BaseModel):

    train_images: Optional[Iterable[str]]
    test_images: Optional[Iterable[str]]
    validation_images: Optional[Iterable[str]]

    class Config:
        extra = "allow"


def global_config_required(method):
    def wrapper(self: "HTFBaseDatasetHandler", *args, **kwargs):
        if self.global_config is not None:
            return method(self, *args, **kwargs)
        else:
            raise AttributeError(
                f"{self} has no global configuration of type <HTFConfiguration>"
            )

    return wrapper


class HTFBaseDatasetHandler(ABC, ObjectLogger):
    def __init__(
        self,
        dataset: CastableTableDataset,
        config: "HTFDatasetConfig",
        global_config: Optional["HTFConfiguration"] = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        # Init Data
        self._data = dataset
        # Init attributes
        self._test_set: Optional[CastableTableDataset] = None
        self._train_set: Optional[CastableTableDataset] = None
        self._validation_set: Optional[CastableTableDataset] = None
        self._test_dataset: Optional[tf.data.Dataset] = None
        self._train_dataset: Optional[tf.data.Dataset] = None
        self._validation_dataset: Optional[tf.data.Dataset] = None
        self._config = config
        self._metadata = HTFBaseDatasetHandlerMetadata()
        self.global_config = global_config
        self._verbose = True
        self.kwargs = kwargs
        # Booleans
        self.checked = self._check_config(_error=False)
        self.splitted = False
        # Launch Init
        self.init_parameters()
        # Apply Checks
        self._check_config()
        # Apply Execution
        self.execute()

    def init_parameters(self, **kwargs) -> None:
        pass

    @property
    def metadata(self) -> HTFBaseDatasetHandlerMetadata:
        return self._metadata

    @property
    def config(self) -> "HTFDatasetConfig":
        return self._config

    @property
    def split(self) -> "HTFDatasetSplitConfig":
        return self._config.split

    @property
    def sets(self) -> "HTFDatasetSetsConfig":
        return self.config.sets

    @property
    def ratio_train(self) -> float:
        return self.split.train_ratio

    @property
    def ratio_test(self) -> float:
        return self.split.test_ratio if self.sets.test else 0.0

    @property
    def ratio_validation(self) -> float:
        return self.split.validation_ratio if self.sets.validation else 0.0

    @property
    def has_train(self) -> bool:
        return self.split.train_ratio > 0

    @property
    def has_test(self) -> bool:
        return self.sets.test

    @property
    def has_validation(self) -> bool:
        return self.sets.validation

    @property
    def CHECK_CONDITIONS(self) -> List[bool]:
        return [
            sum(
                [self.config.split.train_ratio]
                + [
                    getattr(self.config.split, f"{s}_ratio")
                    for s, m in self.config.sets.__fields__.items()
                    if getattr(self.config.sets, s)
                ]
            )
            == 1  # Check that activated set ratios sums to 1
        ]

    def _check_config(self, _error: bool = True):
        validity = all(self.CHECK_CONDITIONS)
        if not validity and _error:
            raise BadConfigurationError("Dataset configuration is incorrect")
        return validity

    @abstractmethod
    def execute(self) -> None:
        raise NotImplementedError


class HTFDatasetHandler(HTFBaseDatasetHandler):
    def __init__(
        self,
        dataset: CastableTableDataset,
        config: "HTFDatasetConfig",
        global_config: Optional["HTFConfiguration"] = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(dataset, config, global_config, verbose, **kwargs)

    @property
    def dataset_is_numpy(self) -> bool:
        return isinstance(self._data, np.ndarray)

    @property
    def dataset_is_pandas(self) -> bool:
        return isinstance(self._data, pd.DataFrame)

    @property
    def split_column_name(self) -> str:
        return self.split.column

    @property
    @global_config_required
    def split_column_index(self) -> str:
        return self.label_mapper[self.split.column]

    @property
    @global_config_required
    def image_column_name(self) -> str:
        return self.global_config._cfg_data_out.source_column

    @property
    @global_config_required
    def image_column_index(self) -> str:
        return self.label_mapper[self.image_column_name]

    @property
    @global_config_required
    def label_mapper(self) -> Dict[str, int]:
        return self.global_config._metadata.label_mapper

    # Split Train/Test/Validation Methods

    @global_config_required
    def _get_images(self) -> Set[str]:
        if self.dataset_is_pandas:
            images: Set[str] = set(self._data[self.image_column_name].tolist())
        elif self.dataset_is_numpy:
            images: Set[str] = set(self._data[:, self.image_column_index].tolist())
        else:
            raise TypeError(
                "data table provided is not of type CastableTableDataset = <Union[np.ndarray, pd.DataFrame]>"
            )
        return images

    def _generate_image_set(self) -> ImageSets:
        # Get set of unique images
        images = self._get_images()
        # Unpack ratios & bool
        train_ratio = self.ratio_train
        has_train = self.has_train
        test_ratio = self.ratio_test
        has_test = self.has_test
        validation_ratio = self.ratio_validation
        has_validation = self.has_validation
        # Generate Sets
        train = set()
        test = set()
        validation = set()
        if has_train:
            # Has training samples
            if has_test & has_validation:
                # + Validation and Test
                train, test = split_train_test(images, train_ratio + validation_ratio)
                train, validation = split_train_test(
                    train, train_ratio / (train_ratio + validation_ratio)
                )
            elif has_validation:
                train, validation = split_train_test(images, train_ratio)
            elif has_test:
                train, test = split_train_test(images, train_ratio)
            else:
                train = images
        else:
            if has_test & has_validation:
                test, validation = split_train_test(images, test_ratio)
            elif has_test:
                test = images
            else:
                validation = images
        return train, test, validation

    def _generate_set_from_pandas(self, image_set: Set[str]) -> pd.DataFrame:
        return self._data[self._data[self.image_column_name].isin(image_set)]

    def _generate_set_from_numpy(self, image_set: Set[str]):
        indices = np.isin(self._data[:, self.image_column_index], list(image_set))
        return self._data[indices]

    @global_config_required
    def _execute_split(self) -> ReturnTableSets:
        # Generate Image Sets
        train, test, validation = self._generate_image_set()
        # Save on metadata
        self.metadata.train_images = train
        self.metadata.test_images = test
        self.metadata.validation_images = validation
        # Filter Sets
        if self.dataset_is_numpy:
            _train_set = self._generate_set_from_numpy(train)
            _test_set = self._generate_set_from_numpy(test)
            _validation_set = self._generate_set_from_numpy(validation)
        elif self.dataset_is_pandas:
            _train_set = self._generate_set_from_pandas(train)
            _test_set = self._generate_set_from_pandas(test)
            _validation_set = self._generate_set_from_pandas(validation)

        return _train_set, _test_set, _validation_set

    @global_config_required
    def _execute_selection(self) -> ReturnTableSets:
        if self.dataset_is_pandas:
            train = self._data.query(
                f"{self.split_column_name} == '{self.split.train_value}'"
            )
            test = self._data.query(
                f"{self.split_column_name} == '{self.split.test_value}'"
            )
            validation = self._data.query(
                f"{self.split_column_name} == '{self.split.validation_value}'"
            )
        elif self.dataset_is_numpy:
            train_mask = (
                self._data[:, self.split_column_index] == self.split.train_value
            )
            test_mask = self._data[:, self.split_column_index] == self.split.test_value
            validation_mask = (
                self._data[:, self.split_column_index] == self.split.validation_value
            )

            train = self._data[train_mask, :]
            test = self._data[test_mask, :]
            validation = self._data[validation_mask, :]
        else:
            raise TypeError(
                "data table provided is not of type CastableTableDataset = <Union[np.ndarray, pd.DataFrame]>"
            )
        return train, test, validation

    def split_sets(self) -> None:
        if self.split.activate:
            # Enable train/test split here
            train, test, validation = self._execute_split()
        else:
            # Use a predefined columns as discriminant
            train, test, validation = self._execute_selection()
        self._train_set = train
        self._test_set = test
        self._validation_set = validation
        self.splitted = True

    def _get_joint_columns(self) -> List[str]:
        joints = self.global_config.config.data.output.joints
        num_joint = joints.n
        naming = joints.naming_convention
        return [
            naming.format(JOINT_ID=i, SUFFIX=suffix)
            for i in range(num_joint)
            for suffix in joints.format.suffix.__dict__.values()
        ]

    def extract_data_groups(
        self, dataset: CastableTableDataset
    ) -> Tuple[List, List, List, List]:
        # Extract columns
        coord_columns = [self.label_mapper[col] for col in self._get_joint_columns()]
        bbox_columns = [self.label_mapper[col] for col in self.config.bbox.cols]
        center_columns = [self.label_mapper[col] for col in self.config.center.cols]
        if self.dataset_is_pandas:
            array = dataset.to_numpy()
        else:
            array = dataset
        filenames = array[:, self.image_column_index].tolist()
        coordinates = array[:, coord_columns].tolist()
        bounding_boxes = array[:, bbox_columns].tolist()
        centers = array[:, center_columns].tolist()
        return filenames, coordinates, bounding_boxes, centers

    # Generate Datasets

    def _create_dataset(self, dataset: CastableTableDataset) -> tf.data.Dataset:
        # Generate
        return (
            tf.data.Dataset.from_tensor_slices(
                # Generate from slices
                self.extract_data_groups(dataset=dataset)
            )
            .map(
                # Load Images
                lambda x: x
            )
            .map(
                # Compute BBOX cropping
                lambda x: x
            )
            .map(
                # Compute Heatmaps
                lambda x: x
            )
        )

    # Main Execution method
    def execute(self) -> None:
        self.split_sets()
        # pass
