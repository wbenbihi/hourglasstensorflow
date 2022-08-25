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
from hourglass_tensorflow.handlers.engines import HTFEngine
from hourglass_tensorflow.handlers._transformation import tf_train_map_stacks
from hourglass_tensorflow.handlers._transformation import tf_train_map_heatmaps
from hourglass_tensorflow.handlers._transformation import tf_train_map_squarify
from hourglass_tensorflow.handlers._transformation import tf_train_map_normalize
from hourglass_tensorflow.handlers._transformation import tf_train_map_build_slice
from hourglass_tensorflow.handlers._transformation import tf_train_map_resize_data

# region Abstract Class

HTFDataTypes = Union[np.ndarray, pd.DataFrame]
ImageSetsType = Tuple[Optional[Set[str]], Optional[Set[str]], Optional[Set[str]]]


class _HTFDatasetHandler(_HTFHandler):

    _ENGINES: Dict[Any, Type[HTFEngine]] = ENGINES
    ENGINES: Dict[Any, Type[HTFEngine]] = {}

    def __init__(
        self,
        data: HTFDataTypes,
        config: HTFDatasetConfig,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config=config, *args, **kwargs)
        self.data = data
        self.engine: HTFEngine = self.select_engine(data)

    @property
    def _engines(self) -> Dict[Type, Type[HTFEngine]]:
        return {**self._ENGINES, **self.ENGINES}

    @property
    def config(self) -> HTFDatasetConfig:
        return self._config

    @property
    def sets(self) -> HTFDatasetSets:
        return self.config.sets

    @property
    def bbox(self) -> HTFDatasetBBox:
        return self.config.sets

    @property
    def heatmap(self) -> HTFDatasetHeatmap:
        return self.config.sets

    def select_engine(self, data: Any) -> HTFEngine:
        try:
            return self._engines[type(data)](metadata=self._metadata)
        except KeyError:
            raise KeyError(f"No engine available for type {type(data)}")

    @abstractmethod
    def prepare_dataset(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate_datasets(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> None:
        self.prepare_dataset(*args, **kwargs)
        self.generate_datasets(*args, **kwargs)


# enregion

# region Handler


class HTFDatasetHandler(_HTFDatasetHandler):
    @property
    def has_train(self) -> bool:
        return self.sets.train

    @property
    def has_test(self) -> bool:
        return self.sets.test

    @property
    def has_validation(self) -> bool:
        return self.sets.validation

    @property
    def ratio_train(self) -> float:
        return self.sets.ratio_train if self.has_train else 0.0

    @property
    def ratio_test(self) -> float:
        return self.sets.ratio_test if self.has_test else 0.0

    @property
    def ratio_validation(self) -> float:
        return self.sets.ratio_validation if self.has_validation else 0.0

    def init_handler(self, *args, **kwargs) -> None:
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
        self._split_sets()

    # endregion

    # region Generate Datasets Hidden Methods
    def _extract_columns_from_data(
        self, data: HTFDataTypes
    ) -> Tuple[Iterable, Iterable]:
        # Extract coordinates
        filenames = self.engine.to_list(
            self.engine.get_columns(data=data, columns=self.config.column_image)
        )
        coordinates = self.engine.to_list(
            self.engine.get_columns(data=data, columns=self.meta.joint_columns)
        )
        return filenames, coordinates

    def _create_dataset(self, data: HTFDataTypes) -> tf.data.Dataset:
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
        self._train_dataset = self._create_dataset(self._train_set)
        self._test_dataset = self._create_dataset(self._test_set)
        self._validation_dataset = self._create_dataset(self._validation_set)


# endregion
