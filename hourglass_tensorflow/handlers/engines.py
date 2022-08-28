from abc import ABC
from abc import abstractmethod
from abc import abstractstaticmethod
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Type

import numpy as np
import pandas as pd
import tensorflow as tf

from hourglass_tensorflow.utils import ObjectLogger
from hourglass_tensorflow.types.config import HTFMetadata


class BaseEngine(ABC, ObjectLogger):
    """Abstract base processing engine for dataset preparation

    `BaseEngine` is responsible of type specific operation during dataset
    preparation _(e.g use of `numpy`, `pandas`, `tensorflow`)_.
    `BaseEngine` allows `BaseDatasetHandler` to be agnostic of the input data type

    Args:
        FOR_TYPE (Type): None
    """

    FOR_TYPE = None

    def __init__(
        self, metadata: HTFMetadata, verbose: bool = True, *args, **kwargs
    ) -> None:
        """see help(BaseEngine)

        Args:
            metadata (HTFMetadata): Metadata to pass to the engine instance
            verbose (bool, optional): If True, display the processing logs.
                Defaults to True.
        """
        super().__init__(verbose=verbose, *args, **kwargs)
        self.metadata = metadata

    @abstractmethod
    def get_images(self, data: Any, column: str) -> Set[str]:
        """Returns a set of image path from the data

        Args:
            data (Any): Source data
            column (str): Header for column containing image paths

        Returns:
            Set[str]: The set of unique image from the source data
        """
        raise NotImplementedError

    @abstractmethod
    def filter_data(self, data: Any, column: str, set_name: str) -> Set[str]:
        """Filter the data to select only samples from TRAIN, TEST or VALIDATION

        Args:
            data (Any): Source data
            column (str): Header for column containing the sets
            set_name (str): The set value

        Returns:
            Set[str]: The data sampled from the given set
        """
        raise NotImplementedError

    @abstractmethod
    def select_subset_from_images(
        self, data: Any, image_set: Set[str], column: str
    ) -> Any:
        """Select samples from source data having the right image

        Args:
            data (Any): Source data
            image_set (Set[str]): Set of image to select sample from
            column (str): Header for column containing image names

        Returns:
            Any: The sampled data
        """
        raise NotImplementedError

    @abstractmethod
    def get_columns(self, data: Any, columns: List[str]) -> Any:
        """Select a subset of columns from the source data

        Args:
            data (Any): Source data
            columns (List[str]): Subset of columns to select

        Returns:
            Any: Subset of source data
        """
        raise NotImplementedError

    @staticmethod
    @abstractstaticmethod
    def to_list(data: Any) -> List:
        """Convert data to common list format

        Args:
            data (Any): Source data

        Returns:
            List: Source data casted as list
        """
        raise NotImplementedError


class HTFNumpyEngine(BaseEngine):
    """Dataset processing engine for `numpy`

    Args:
        FOR_TYPE (Type): np.ndarray
    """

    FOR_TYPE = np.ndarray

    def get_images(self, data: np.ndarray, column: str) -> Set[str]:
        """Returns a set of image path from the data

        Args:
            data (np.ndarray): Source data as numpy array
            column (str): Header for column containing image paths

        Returns:
            Set[str]: The set of unique image from the source data
        """
        images: Set[str] = set(data[self.metadata.label_mapper[column]].tolist())
        return images

    def filter_data(self, data: np.ndarray, column: str, set_name: str) -> np.ndarray:
        """Filter the data to select only samples from TRAIN, TEST or VALIDATION

        Args:
            data (np.ndarray): Source data as numpy array
            column (str): Header for column containing the sets
            set_name (str): The set value

        Returns:
            np.ndarray: The data sampled from the given set
        """
        mask = data[:, self.metadata.label_mapper[column]] == set_name
        filtered_data = data[mask, :]
        return filtered_data

    def select_subset_from_images(
        self, data: np.ndarray, image_set: Set[str], column: str
    ) -> np.ndarray:
        """Select samples from source data having the right image

        Args:
            data (np.ndarray): Source data as numpy array
            image_set (Set[str]): Set of image to select sample from
            column (str): Header for column containing image names

        Returns:
            np.ndarray: The sampled data
        """
        indices = np.isin(data[:, self.metadata.label_mapper[column]], image_set)
        return data[indices]

    def get_columns(self, data: np.ndarray, columns: List[str]) -> np.ndarray:
        """Select a subset of columns from the source data

        Args:
            data (np.ndarray): Source data as numpy array
            columns (List[str]): Subset of columns to select

        Returns:
            np.ndarray: Subset of source data
        """
        idx_columns = [self.metadata.label_mapper[col] for col in columns]
        return data[:, idx_columns]

    @staticmethod
    def to_list(data: np.ndarray) -> List:
        """Convert data to common list format

        Args:
            data (np.ndarray): Source data

        Returns:
            List: Source data casted as list
        """
        return data.tolist()


class HTFPandasEngine(BaseEngine):
    """Dataset processing engine for `pandas`

    Args:
        FOR_TYPE (Type): pd.DataFrame
    """

    FOR_TYPE = pd.DataFrame

    def get_images(self, data: pd.DataFrame, column: str) -> Set[str]:
        """Returns a set of image path from the data

        Args:
            data (pd.DataFrame): Source data as pandas DataFrame
            column (str): Header for column containing image paths

        Returns:
            Set[str]: The set of unique image from the source data
        """
        images: Set[str] = set(data[column].tolist())
        return images

    def filter_data(
        self, data: pd.DataFrame, column: str, set_name: str
    ) -> pd.DataFrame:
        """Filter the data to select only samples from TRAIN, TEST or VALIDATION

        Args:
            data (pd.DataFrame): Source data as pandas DataFrame
            column (str): Header for column containing the sets
            set_name (str): The set value

        Returns:
            pd.DataFrame: The data sampled from the given set
        """
        return data.query(f"{column} == '{set_name}'")

    def select_subset_from_images(
        self, data: pd.DataFrame, image_set: Set[str], column: str
    ) -> pd.DataFrame:
        """Select samples from source data having the right image

        Args:
            data (pd.DataFrame): Source data as pandas DataFrame
            image_set (Set[str]): Set of image to select sample from
            column (str): Header for column containing image names

        Returns:
            pd.DataFrame: The sampled data
        """
        return data[data[column].isin(image_set)]

    def get_columns(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Select a subset of columns from the source data

        Args:
            data (pd.DataFrame): Source data as pandas DataFrame
            columns (List[str]): Subset of columns to select

        Returns:
            pd.DataFrame: The sampled data
        """
        return data[columns]

    @staticmethod
    def to_list(data: pd.DataFrame) -> List:
        """Convert data to common list format

        Args:
            data (pd.DataFrame): Source data

        Returns:
            List: Source data casted as list
        """
        return data.values.tolist()


class HTFTensorflowEngine(BaseEngine):
    """Dataset processing engine for `tensorflow`

    Args:
        FOR_TYPE (Type): tf.Tensor
    """

    FOR_TYPE = tf.Tensor

    def get_images(self, data: tf.Tensor, column: str) -> Set[str]:
        """Returns a set of image path from the data

        Args:
            data (tf.Tensor):Source data as pandas DataFrame
            column (str): Header for column containing image paths

        Raises:
            NotImplementedError: The Tensorflow Engine is not developed yet

        Returns:
            Set[str]: The set of unique image from the source data
        """
        raise NotImplementedError

    def filter_data(self, data: tf.Tensor, column: str, set_name: str) -> tf.Tensor:
        """Filter the data to select only samples from TRAIN, TEST or VALIDATION

        Args:
            data (tf.Tensor): Source data tensorflow Tensor
            column (str): Header for column containing the sets
            set_name (str): The set value

        Raises:
            NotImplementedError: The Tensorflow Engine is not developed yet

        Returns:
            tf.Tensor: The data sampled from the given set
        """
        raise NotImplementedError

    def select_subset_from_images(
        self, data: tf.Tensor, image_set: Set[str], column: str
    ) -> tf.Tensor:
        """Select samples from source data having the right image

        Args:
            data (tf.Tensor): Source data as tensorflow Tensor
            image_set (Set[str]): Set of image to select sample from
            column (str): Header for column containing image names

        Raises:
            NotImplementedError: The Tensorflow Engine is not developed yet

        Returns:
            tf.Tensor: The sampled data
        """
        raise NotImplementedError

    def get_columns(self, data: tf.Tensor, columns: List[str]) -> tf.Tensor:
        """Select a subset of columns from the source data

        Args:
            data (tf.Tensor): Source data as tensorflow Tensor
            columns (List[str]): Subset of columns to select

        Raises:
            NotImplementedError: The Tensorflow Engine is not developed yet

        Returns:
            tf.Tensor: Subset of source data
        """
        raise NotImplementedError

    @staticmethod
    def to_list(data: tf.Tensor) -> List:
        """Convert data to common list format

        Args:
            data (tf.Tensor): Source data

        Raises:
            NotImplementedError: The Tensorflow Engine is not developed yet

        Returns:
            List: Source data casted as list
        """

        raise NotImplementedError


ENGINES: Dict[Type, Type[BaseEngine]] = {
    np.ndarray: HTFNumpyEngine,
    pd.DataFrame: HTFPandasEngine,
    tf.Tensor: HTFTensorflowEngine,
}
