import pandas as pd
import tensorflow as tf
from typing import Optional, Union, Tuple, List
from .functions import tf_parse_dataset, tf_preprocess, tf_random_rotation, tf_stacker
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
            self._create_dataset(self.summary_val) if self.summary_val else None
        )
        self._has_datasets = True

        # Data Augmentation
        self.train_dataset = self._augment_dataset(self.train_dataset)
        self.test_dataset = self._augment_dataset(self.test_dataset)
        self.val_dataset = (
            self._augment_dataset(self.val_dataset) if self.val_dataset else None
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