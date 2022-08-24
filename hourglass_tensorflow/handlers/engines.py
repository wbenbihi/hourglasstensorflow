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


class HTFEngine(ABC, ObjectLogger):
    FOR_TYPE = None

    def __init__(
        self, metadata: HTFMetadata, verbose: bool = True, *args, **kwargs
    ) -> None:
        super().__init__(verbose=verbose, *args, **kwargs)
        self.metadata = metadata

    @abstractmethod
    def get_images(self, data: Any, column: str) -> Set[str]:
        raise NotImplementedError

    @abstractmethod
    def filter_data(self, data: Any, column: str, set_name: str) -> Set[str]:
        raise NotImplementedError

    @abstractmethod
    def select_subset_from_images(
        self, data: Any, image_set: Set[str], column: str
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_columns(sel, data: Any, columns: List[str]) -> Any:
        raise NotImplementedError

    @staticmethod
    @abstractstaticmethod
    def to_list(data: Any) -> List:
        raise NotImplementedError


class HTFNumpyEngine(HTFEngine):
    FOR_TYPE = np.ndarray

    def get_images(self, data: np.ndarray, column: str) -> Set[str]:
        images: Set[str] = set(data[self.metadata.label_mapper[column]].tolist())
        return images

    def filter_data(self, data: np.ndarray, column: str, set_name: str) -> np.ndarray:
        mask = data[:, self.metadata.label_mapper[column]] == set_name
        filtered_data = data[mask, :]
        return filtered_data

    def select_subset_from_images(
        self, data: np.ndarray, image_set: Set[str], column: str
    ) -> np.ndarray:
        indices = np.isin(data[:, self.metadata.label_mapper[column]], image_set)
        return data[indices]

    def get_columns(self, data: np.ndarray, columns: List[str]) -> np.ndarray:
        idx_columns = [self.metadata.label_mapper[col] for col in columns]
        return data[:, idx_columns]

    @staticmethod
    def to_list(data: np.ndarray) -> List:
        return data.tolist()


class HTFPandasEngine(HTFEngine):
    FOR_TYPE = pd.DataFrame

    def get_images(self, data: pd.DataFrame, column: str) -> Set[str]:
        images: Set[str] = set(data[column].tolist())
        return images

    def filter_data(
        self, data: pd.DataFrame, column: str, set_name: str
    ) -> pd.DataFrame:
        return data.query(f"{column} == '{set_name}'")

    def select_subset_from_images(
        self, data: pd.DataFrame, image_set: Set[str], column: str
    ) -> pd.DataFrame:
        return data[data[column].isin(image_set)]

    def get_columns(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return data[columns]

    @staticmethod
    def to_list(data: pd.DataFrame) -> List:
        return data.values.tolist()


class HTFTensorflowEngine(HTFEngine):
    FOR_TYPE = tf.Tensor

    def get_images(self, data: tf.Tensor, column: str) -> Set[str]:
        raise NotImplementedError

    def filter_data(self, data: tf.Tensor, column: str, set_name: str) -> tf.Tensor:
        raise NotImplementedError

    def select_subset_from_images(
        self, data: tf.Tensor, image_set: Set[str], column: str
    ) -> tf.Tensor:
        raise NotImplementedError

    def get_columns(self, data: tf.Tensor, columns: List[str]) -> tf.Tensor:
        raise NotImplementedError

    @staticmethod
    def to_list(data: tf.Tensor) -> List:
        raise NotImplementedError


ENGINES: Dict[Type, Type[HTFEngine]] = {
    np.ndarray: HTFNumpyEngine,
    pd.DataFrame: HTFPandasEngine,
    tf.Tensor: HTFTensorflowEngine,
}
