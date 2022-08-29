from abc import abstractmethod
from typing import Any
from typing import Set
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Union
from typing import Iterable
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from hourglass_tensorflow.utils import split_train_test
from hourglass_tensorflow.types.config import HTFDatasetBBox
from hourglass_tensorflow.types.config import HTFDatasetSets
from hourglass_tensorflow.types.config import HTFDatasetConfig
from hourglass_tensorflow.types.config import HTFDatasetHeatmap
from hourglass_tensorflow.handlers.meta import _HTFHandler
from hourglass_tensorflow.handlers.engines import ENGINES
from hourglass_tensorflow.handlers.engines import BaseEngine
from hourglass_tensorflow.handlers._transformation import tf_train_map_stacks
from hourglass_tensorflow.handlers._transformation import tf_train_map_heatmaps
from hourglass_tensorflow.handlers._transformation import tf_train_map_squarify
from hourglass_tensorflow.handlers._transformation import tf_train_map_normalize
from hourglass_tensorflow.handlers._transformation import tf_train_map_build_slice
from hourglass_tensorflow.handlers._transformation import tf_train_map_resize_data

# region Abstract Class

HTFDataTypes = Union[np.ndarray, pd.DataFrame]
ImageSetsType = Tuple[Optional[Set[str]], Optional[Set[str]], Optional[Set[str]]]


class BaseDatasetHandler(_HTFHandler):
    """Abstract handler for `hourglass_tensorflow` Dataset jobs

    The `BaseDatasetHandler` is responsible of generating `tensorflow.Dataset` for
    TRAIN/TEST/VALIDATION sets. As it might receive an `__init__` `data` argument from
    various types _(`np.array`, `pd.DataFrame`, ...)_. `BaseDatasetHandler` instantiates
    an `BaseEngine` that is responsible of operation specific to certain types.

    See the dataset generation engine section in the documentation for deeper insights.

    Args:
        _HTFHandler (_type_): Subclass of Meta Handler

    Raises:
        KeyError: If the specified engine does not exists
    """

    _ENGINES: Dict[Any, Type[BaseEngine]] = ENGINES
    ENGINES: Dict[Any, Type[BaseEngine]] = {}

    def __init__(
        self,
        data: HTFDataTypes,
        config: HTFDatasetConfig,
        *args,
        **kwargs,
    ) -> None:
        """see help(BaseDataHandler)

        Args:
            data (HTFDataTypes): The labeled data loaded from `data:output:source`
            config (HTFDatasetConfig): The configuartion reference to `dataset:`
        """
        super().__init__(config=config, *args, **kwargs)
        self.data = data
        self.engine: BaseEngine = self.select_engine(data)
        self._test_dataset: Optional[tf.data.Dataset] = None
        self._train_dataset: Optional[tf.data.Dataset] = None
        self._validation_dataset: Optional[tf.data.Dataset] = None

    @property
    def _engines(self) -> Dict[Type, Type[BaseEngine]]:
        """Reference to the available engines

        Returns:
            Dict[Type, Type[BaseEngine]]: Map of the available processing engines
        """
        return {**self._ENGINES, **self.ENGINES}

    @property
    def config(self) -> HTFDatasetConfig:
        """Reference to `dataset:` field configuration

        Returns:
            HTFDatasetConfig: Dataset configuration object
        """
        return self._config

    @property
    def sets(self) -> HTFDatasetSets:
        """Reference to `dataset:sets` field configuration

        Returns:
            HTFDatasetSets: Sets configuration object
        """
        return self.config.sets

    @property
    def bbox(self) -> HTFDatasetBBox:
        """Reference to `dataset:bbox` field configuration

        Returns:
            HTFDatasetBBox: Bbox configuration object
        """
        return self.config.sets

    @property
    def heatmap(self) -> HTFDatasetHeatmap:
        """Reference to `dataset:bbox` field configuration

        Returns:
            HTFDatasetHeatmap: Heatmap configuration object
        """
        return self.config.heatmap

    @property
    def train_dataset(self) -> tf.data.Dataset:
        """Getter for train dataset

        Returns:
            tf.data.Dataset: Train dataset
        """
        return self._train_dataset

    @property
    def test_dataset(self) -> tf.data.Dataset:
        """Getter for test dataset

        Returns:
            tf.data.Dataset: Test dataset
        """
        return self._test_dataset

    @property
    def validation_dataset(self) -> tf.data.Dataset:
        """Getter for validation dataset

        Returns:
            tf.data.Dataset: Validation dataset
        """
        return self._validation_dataset

    def set_train_dataset(self, dataset: tf.data.Dataset) -> None:
        """Sets the train dataset

        Args:
            dataset (tf.data.Dataset): Dataset to use as train dataset
        """
        self._train_dataset = dataset

    def set_test_dataset(self, dataset: tf.data.Dataset) -> None:
        """Sets the test dataset

        Args:
            dataset (tf.data.Dataset): Dataset to use as test dataset
        """
        self._test_dataset = dataset

    def set_validation_dataset(self, dataset: tf.data.Dataset) -> None:
        """Sets the validation dataset

        Args:
            dataset (tf.data.Dataset): Dataset to use as validation dataset
        """
        self._validation_dataset = dataset

    def select_engine(self, data: Any) -> BaseEngine:
        """Infer the engine to use based on the `data` argument's type

        Args:
            data (Any): Object to use to infer engine to use

        Raises:
            KeyError: The `data` object type has no engine related

        Returns:
            BaseEngine: the engine to use
        """
        try:
            return self._engines[type(data)](metadata=self._metadata)
        except KeyError:
            raise KeyError(f"No engine available for type {type(data)}")

    @abstractmethod
    def prepare_dataset(self, *args, **kwargs) -> None:
        """Abstract method to implement for custom `BaseDatasetHander` subclass

        This method should script the process of dataset preparation

        Raises:
            NotImplementedError: Abstract Method
        """
        raise NotImplementedError

    @abstractmethod
    def generate_datasets(self, *args, **kwargs) -> None:
        """Abstract method to implement for custom `BaseDatasetHander` subclass

        This method should script the process of `tensorflow.Dataset` generation

        Raises:
            NotImplementedError: Abstract Method
        """
        raise NotImplementedError

    def run(self, *args, **kwargs) -> None:
        """Global run job for `BaseDatasetHander`

        The run for a `BaseDatasetHander` will call the
        `BaseDatasetHander.prepare_dataset` and `BaseDatasetHander.generate_datasets`
        methods sequentially.
        """
        self.prepare_dataset(*args, **kwargs)
        self.generate_datasets(*args, **kwargs)


# enregion

# region Handler


class HTFDatasetHandler(BaseDatasetHandler):
    """Default Dataset Handler for `hourglass_tendorflow`

    The HTFDatasetHandler can be used outside of MPII data context
    but was specifically created in the context of MPII data.
    Its use might not be a good fit for other dataset.

    > NOTE
    >
    > Check the handlers section from the documention to understand
    > how to build you custom handlers

    Args:
        BaseDatasetHandler (_type_): Subclass of Meta Dataset Handler
    """

    @property
    def has_train(self) -> bool:
        """Reference to `dataset:sets:train` boolean

        Returns:
            bool: True if should generate train set
        """
        return self.sets.train

    @property
    def has_test(self) -> bool:
        """Reference to `dataset:sets:test` boolean

        Returns:
            bool: True if should generate test set
        """
        return self.sets.test

    @property
    def has_validation(self) -> bool:
        """Reference to `dataset:sets:validation` boolean

        Returns:
            bool: True if should generate validation set
        """
        return self.sets.validation

    @property
    def ratio_train(self) -> float:
        """Reference to `dataset:sets:ratio_train`

        The ratio to use for splitting the labeled samples.
        This ratio will only be used if `dataset:sets:train` is True and
        `dataset:sets:split_by_column` is False.

        Returns:
            float: ratio to  use for training set
        """
        return self.sets.ratio_train if self.has_train else 0.0

    @property
    def ratio_test(self) -> float:
        """Reference to `dataset:sets:ratio_test`

        The ratio to use for splitting the labeled samples.
        This ratio will only be used if `dataset:sets:test` is True and
        `dataset:sets:split_by_column` is False.

        Returns:
            float: ratio to  use for test set
        """
        return self.sets.ratio_test if self.has_test else 0.0

    @property
    def ratio_validation(self) -> float:
        """Reference to `dataset:sets:ratio_validation`

        The ratio to use for splitting the labeled samples.
        This ratio will only be used if `dataset:sets:validation` is True and
        `dataset:sets:split_by_column` is False.

        Returns:
            float: ratio to  use for validation set
        """
        return self.sets.ratio_validation if self.has_validation else 0.0

    def init_handler(self, *args, **kwargs) -> None:
        """Initialization of HTFDatasetHandler"""
        self.splitted = False
        # Init attributes
        self._test_set: Optional[HTFDataTypes] = None
        self._train_set: Optional[HTFDataTypes] = None
        self._validation_set: Optional[HTFDataTypes] = None
        self._test_dataset: Optional[tf.data.Dataset] = None
        self._train_dataset: Optional[tf.data.Dataset] = None
        self._validation_dataset: Optional[tf.data.Dataset] = None
        self.kwargs = kwargs

    # region Prepare Dataset Hidden Methods
    def _generate_image_sets(self, images: Set[str]) -> ImageSetsType:
        """Creates sets of image path by using the split ratios

        Args:
            images (Set[str]): Set containing all available images

        Returns:
            ImageSetsType: Tuple of sets for train/test/validation images respectively.
        """
        # Generate Sets
        train = set()
        test = set()
        validation = set()
        if self.has_train:
            # Has training samples
            if self.has_test & self.has_validation:
                # + Validation and Test
                train, test = split_train_test(
                    images, self.ratio_train + self.ratio_validation
                )
                train, validation = split_train_test(
                    train, self.ratio_train / (self.ratio_train + self.ratio_validation)
                )
            elif self.has_validation:
                train, validation = split_train_test(images, self.ratio_train)
            elif self.has_test:
                train, test = split_train_test(images, self.ratio_train)
            else:
                train = images
        else:
            if self.has_test & self.has_validation:
                test, validation = split_train_test(images, self.ratio_test)
            elif self.has_test:
                test = images
            else:
                validation = images
        return train, test, validation

    def _split_by_column(self) -> Tuple[HTFDataTypes, HTFDataTypes, HTFDataTypes]:
        """Uses `dataset:sets:column_split` to split the labeled samples based on a column value

        Returns:
            Tuple[HTFDataTypes, HTFDataTypes, HTFDataTypes]: Tuple of train/test/validation respectively.
        """
        train = self.engine.filter_data(
            data=self.data, column=self.sets.column_split, set_name=self.sets.value_test
        )
        test = self.engine.filter_data(
            data=self.data,
            column=self.sets.column_split,
            set_name=self.sets.value_train,
        )
        validation = self.engine.filter_data(
            data=self.data,
            column=self.sets.column_split,
            set_name=self.sets.value_validation,
        )
        return train, test, validation

    def _split_by_ratio(self) -> Tuple[HTFDataTypes, HTFDataTypes, HTFDataTypes]:
        """Uses `dataset:sets:ratio_*` to split the labeled samples based on a ratio splitting

        Returns:
            Tuple[HTFDataTypes, HTFDataTypes, HTFDataTypes]: Tuple of train/test/validation respectively.
        """
        # Get set of unique images
        images = self.engine.get_images(data=self.data, column=self.config.column_image)
        img_train, img_test, img_validation = self._generate_image_sets(images)
        # Save on metadata
        self._metadata.test_images = img_test
        self._metadata.train_images = img_train
        self._metadata.validation_images = img_validation
        # Select Subsets within the main data collection
        test = self.engine.select_subset_from_images(
            data=self.data, image_set=img_test, column=self.config.column_image
        )
        train = self.engine.select_subset_from_images(
            data=self.data, image_set=img_train, column=self.config.column_image
        )
        validation = self.engine.select_subset_from_images(
            data=self.data, image_set=img_validation, column=self.config.column_image
        )
        return train, test, validation

    def _split_sets(self) -> None:
        """Split the sets accordingly to the `dataset:sets` configuration"""
        if self.sets.split_by_column:
            # Use a predefined columns as discriminant
            train, test, validation = self._split_by_column()
        else:
            # Enable train/test split here
            train, test, validation = self._split_by_ratio()
        self._train_set = train
        self._test_set = test
        self._validation_set = validation
        self.splitted = True

    def prepare_dataset(self, *args, **kwargs) -> None:
        """Prepare the labeled data before generating datasets"""
        self._split_sets()

    # endregion

    # region Generate Datasets Hidden Methods
    def _extract_columns_from_data(
        self, data: HTFDataTypes
    ) -> Tuple[Iterable, Iterable]:
        """Extract the Filename and Coordinates from the labeled samples

        Args:
            data (HTFDataTypes): Labeled samples

        Returns:
            Tuple[Iterable, Iterable]: Tuple of filenames and joint coordinates
        """
        # Extract coordinates
        filenames = self.engine.to_list(
            self.engine.get_columns(data=data, columns=self.config.column_image)
        )
        coordinates = self.engine.to_list(
            self.engine.get_columns(data=data, columns=self.meta.joint_columns)
        )
        return filenames, coordinates

    def _create_dataset(self, data: HTFDataTypes) -> tf.data.Dataset:
        """Generates a `tensorflow` dataset constructed from multiple mappers

        This method is the one generating the `tensorflow` Datasets.
        Currently it applies several `tensorflow.data.Dataset.map`,
        these mappers have been developed for the purpose of training on
        the MPII dataset and might not be suitable for you specific data.

        The map steps are:
            1. Load the filename and coordinates
            2. Load the image and reshape the joints columns
            3. Make a square input image and compute coordinates accordingly
            4. Resize the input image and compute coordinates accordingly
            5. Generate Heatmaps from the coordinates
            6. Normalize the input/output tensors if configured
            7. Stack heatmaps

        Args:
            data (HTFDataTypes): The source labeled data

        Returns:
            tf.data.Dataset: The tensorflow dataset generated from the input data
        """
        return (
            tf.data.Dataset.from_tensor_slices(
                self._extract_columns_from_data(data=data)
            )
            .map(
                # Load Images
                tf_train_map_build_slice
            )
            .map(
                # Compute BBOX cropping
                lambda img, coord, vis: tf_train_map_squarify(
                    img,
                    coord,
                    vis,
                    bbox_enabled=self.config.bbox.activate,
                    bbox_factor=self.config.bbox.factor,
                )
            )
            .map(
                # Resize Image
                lambda img, coord, vis: tf_train_map_resize_data(
                    img, coord, vis, input_size=int(self.config.image_size)
                )
            )
            .map(
                # Get Heatmaps
                lambda img, coord, vis: tf_train_map_heatmaps(
                    img,
                    coord,
                    vis,
                    output_size=int(self.config.heatmap.size),
                    stddev=self.config.heatmap.stddev,
                )
            )
            .map(
                # Normalize Data
                lambda img, hms: tf_train_map_normalize(
                    img,
                    hms,
                    normalization=self.config.normalization,
                )
            )
            .map(
                # Stacks
                lambda img, hms: tf_train_map_stacks(
                    img,
                    hms,
                    stacks=self.config.heatmap.stacks,
                )
            )
        )

    def generate_datasets(self, *args, **kwargs) -> None:
        """Generates the train/test/validation tensorflow dataset according to configuration"""
        self._train_dataset = self._create_dataset(self._train_set)
        self._test_dataset = self._create_dataset(self._test_set)
        self._validation_dataset = self._create_dataset(self._validation_set)


# endregion
