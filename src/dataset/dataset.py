import pandas as pd
import tensorflow as tf
from typing import Optional, Union, Tuple, List, Callable
from .functions import (
    tf_parse_dataset,
    tf_preprocess,
    tf_random_rotation,
    tf_stacker,
    tf_resize,
    tf_normalize_by_255,
    tf_normalize_minmax,
    tf_normalize_stddev,
    tf_get_heatmaps,
    tf_load_images,
    tf_compute_coordinates
)
from utils.config import HourglassConfig, DatasetConfig


def extract_coordinates_and_filenames(
    df: pd.DataFrame, config: Union[HourglassConfig, DatasetConfig]
) -> Tuple[List[str], List[str]]:
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        config (Union[HourglassConfig, DatasetConfig]): [description]

    Returns:
        Tuple[List[str], List[str]]: [description]
    """
    if isinstance(config, HourglassConfig):
        dataset = config.dataset
    elif isinstance(config, DatasetConfig):
        dataset = config
    image_column = dataset.description.image_column
    joint_format_regex = dataset.description.joints.column_format.format(
        JOINTNUMBER="[0-9]*", IDVISIBILITY="(x|y)"
    )
    filenames = (dataset.images_folder + "/" + df[image_column]).tolist()
    coordinates = df.filter(regex=joint_format_regex).values.reshape(
        (-1, dataset.description.n_joints, 2)
    )
    return filenames, coordinates


class HPEDataset:
    def __init__(self, config: HourglassConfig):
        self.config = config

        # Steps
        self._has_summaries = False
        self._has_datasets = False
        self._has_augments = False

        # Initialize Summaries
        self.summary: pd.DataFrame
        self.summary_train: pd.DataFrame
        self.summary_test: pd.DataFrame
        self.summary_val: Optional[pd.DataFrame]
        # Create DataFrames to summarize Train / Test / Val sets
        self._create_summaries()
        self._has_summaries = True

        # Initialize Datasets
        self.train_dataset: tf.data.Dataset = self._create_dataset(self.summary_train)
        self.test_dataset: tf.data.Dataset = self._create_dataset(self.summary_test)
        self.val_dataset: Optional[tf.data.Dataset] = (
            self._create_dataset(self.summary_val)
            if self.summary_val is not None
            else None
        )
        self._has_datasets = True

        # Standardization
        self.train_dataset = self._cast_and_normalize_dataset(self.train_dataset)
        self.test_dataset = self._cast_and_normalize_dataset(self.test_dataset)
        self.val_dataset = (
            self._cast_and_normalize_dataset(self.val_dataset)
            if self.val_dataset is not None
            else None
        )
        self._has_augments = True

        # Data Augmentation
        self.train_dataset = self._augment_dataset(self.train_dataset)
        self.test_dataset = self._augment_dataset(self.test_dataset)
        self.val_dataset = (
            self._augment_dataset(self.val_dataset)
            if self.val_dataset is not None
            else None
        )
        self._has_augments = True

        # Batch Datasets
        self.train_dataset = self.train_dataset.batch(self.config.train.batch_size)
        self.test_dataset = self.test_dataset.batch(self.config.train.batch_size)
        self.val_dataset = (
            self.val_dataset.batch(self.config.train.batch_size)
            if self.val_dataset is not None
            else None
        )
        self._has_augments = True

    def _create_summaries(self):
        # Open File
        self.summary = pd.read_csv(self.config.dataset.summary_file, sep=";")
        # Split Train/Test
        self.summary_train: pd.DataFrame = self.summary.query(
            f"{self.config.dataset.description.train_test_column} == 1"
        )
        self.summary_test: pd.DataFrame = self.summary.query(
            f"{self.config.dataset.description.train_test_column} == 0"
        )
        # Shuffle Train Set
        if self.config.data.shuffle:
            self.summary_train = self.summary_train.sample(frac=1).reset_index(
                drop=True
            )
        # Split Train with Validation
        self.summary_val: Optional[pd.DataFrame] = None
        if (
            self.config.train.validation_ratio
            and 0 < self.config.train.validation_ratio < 1
        ):
            validation_samples = int(
                len(self.summary_train) * self.config.train.validation_ratio
            )
            self.summary_val = self.summary_train[-validation_samples:].reset_index(
                drop=True
            )
            self.summary_train = self.summary_train[:validation_samples].reset_index(
                drop=True
            )

    def _create_dataset(self, summary: pd.DataFrame) -> tf.data.Dataset:
        filenames, coordinates = extract_coordinates_and_filenames(
            df=summary, config=self.config
        )
        if self.config.data.preprocess.area_type == "full":
            dataset = self.__create_dataset_full_image(filenames, coordinates)
        elif self.config.data.preprocess.area_type == "bbox":
            dataset = self.__create_dataset_bbox_image(filenames, coordinates)
        return dataset

    def __create_dataset_full_image(self, filenames, coordinates):
        dataset = (
            tf.data.Dataset.from_tensor_slices(
                # Load Filenames and their respectives joint coordinates
                (filenames, coordinates)
            )
            .map(
                # Load Images and Generate Heatmaps
                tf_parse_dataset
            )
            .map(
                # Reshape Images and their respective Heatmaps
                lambda x, y: tf_preprocess(
                    images=x,
                    heatmaps=y,
                    input_size=self.config.data.input_size,
                    output_size=self.config.data.output_size,
                )
            )
        )
        return dataset
    
    def __create_dataset_bbox_image(self, filenames, coordinates):
        dataset = (
            tf.data.Dataset.from_tensor_slices(
                # Load Filenames and their respectives joint coordinates
                (filenames, coordinates)
            )
            .map(
                # Load Images
                tf_load_images
            )
            .map(
                # Compute Coordinates
                lambda img, coords: tf_compute_coordinates(img, coords, bbox_factor=self.config.data.preprocess.bbox_factor, resize_output=self.config.data.output_size)
            )
            .map(
                # Generate HeatMap
                lambda img, coords: (img, tf.transpose(tf_get_heatmaps(coords, self.config.data.output_size, self.config.data.output_size, self.config.data.preprocess.heatmap_stddev, self.config.data.preprocess.heatmap_stddev), [1, 2, 0]))
            )
            .map(
                # Reshape Images
                lambda img, heatmaps: (tf_resize(img, self.config.data.input_size), heatmaps)
            )
        )
        return dataset

    def _cast_and_normalize_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        # Get the normalizing Functions
        img_fnc, hm_fnc = lambda x: x, lambda x: x
        if self.config.data.normalization and hasattr(
            self.config.data.normalization, "images"
        ):
            img_fnc = self._get_normalization_function(
                self.config.data.normalization.images
            )
        if self.config.data.normalization and hasattr(
            self.config.data.normalization, "heatmaps"
        ):
            hm_fnc = self._get_normalization_function(
                self.config.data.normalization.heatmaps
            )
        # Generate Dataset
        dataset = dataset.map(
            # Cast Dataset to Float64 precision
            lambda x, y: (tf.cast(x, tf.float64), tf.cast(y, tf.float64))
        ).map(
            # Apply Normalization
            lambda x, y: (img_fnc(x), hm_fnc(y))
        )
        return dataset

    def _get_normalization_function(self, normalization_method: str = None) -> Callable:
        if normalization_method == "by255":
            fnc = tf_normalize_by_255
        elif normalization_method == "MinMax":
            fnc = tf_normalize_minmax
        elif normalization_method == "StdDev":
            fnc = tf_normalize_stddev
        else:
            fnc = lambda x: x
        return fnc

    def _augment_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        if self.config.data.augmentation is None:
            return dataset
        if (
            hasattr(self.config.data.augmentation, "rotation")
            and self.config.data.augmentation.rotation > 0
        ):
            dataset = dataset.map(
                lambda x, y: tf_random_rotation(
                    images=x,
                    heatmaps=y,
                    rotation_range=self.config.data.augmentation.rotation,
                )
            )
        if self.config.train.intermediate_supervision:
            dataset = dataset.map(
                lambda x, y: tf_stacker(x, y, self.config.model.stages)
            )
        return dataset

    @property
    def train(self) -> tf.data.Dataset:
        return self.train_dataset

    @property
    def test(self) -> tf.data.Dataset:
        return self.test_dataset

    @property
    def val(self) -> tf.data.Dataset:
        return self.val_dataset

    @property
    def has_validation(self) -> bool:
        return self.val_dataset is not None