from abc import abstractmethod
from typing import List
from typing import Union
from typing import TypeVar

import tensorflow as tf
from keras.losses import Loss
from keras.models import Model
from keras.metrics import Metric
from keras.callbacks import Callback
from keras.optimizers import Optimizer
from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule

from hourglass_tensorflow.types.config import HTFTrainConfig
from hourglass_tensorflow.types.config import HTFObjectReference
from hourglass_tensorflow.handlers.meta import _HTFHandler

# region Abstract Class

R = TypeVar("R")


class _HTFTrainHandler(_HTFHandler):
    def __init__(
        self,
        config: HTFTrainConfig,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config=config, *args, **kwargs)
        self._epochs: int = None
        self._epoch_size: int = None
        self._batch_size: int = None
        self._learning_rate: Union[
            HTFObjectReference[LearningRateSchedule], float
        ] = None
        self._loss: Union[HTFObjectReference[Loss], str] = None
        self._optimizer: Union[HTFObjectReference[Optimizer], str] = None
        self._metrics: List[HTFObjectReference[Metric]] = None
        self._callbacks: List[HTFObjectReference[Callback]] = None
        self.init_handler()

    @property
    def config(self) -> HTFTrainConfig:
        return self._config

    @abstractmethod
    def compile(self, model: Model, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
        model: Model,
        train_dataset: tf.data.Dataset = None,
        test_dataset: tf.data.Dataset = None,
        validation_dataset: tf.data.Dataset = None,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> None:
        self.compile(*args, **kwargs)
        self.fit(*args, **kwargs)


# enregion

# region Handler


class HTFTrainHandler(_HTFTrainHandler):
    def _instantiate(self, obj: HTFObjectReference[R], **kwargs) -> R:
        if isinstance(obj, HTFObjectReference):
            return obj.init(**kwargs)
        else:
            return obj

    def init_handler(self, *args, **kwargs) -> None:
        self._epochs = self.config.epochs
        self._epoch_size = self.config.epoch_size
        self._batch_size = self.config.batch_size
        self._learning_rate = self._instantiate(self.config.learning_rate)
        self._loss = self._instantiate(self.config.loss)
        self._optimizer = self._instantiate(
            self.config.optimizer, learning_rate=self._learning_rate
        )
        self._metrics = [obj.init() for obj in self.config.metrics]
        self._callbacks = [obj.init() for obj in self.config.callbacks]

    def compile(self, model: Model, *args, **kwargs) -> None:
        model.compile(optimizer=self._optimizer, metrics=self._metrics, loss=self._loss)

    def _apply_batch(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        if isinstance(dataset, tf.data.Dataset):
            return dataset.batch(self._batch_size)

    def fit(
        self,
        model: Model,
        train_dataset: tf.data.Dataset = None,
        test_dataset: tf.data.Dataset = None,
        validation_dataset: tf.data.Dataset = None,
        *args,
        **kwargs,
    ) -> None:
        _ = self._apply_batch(test_dataset)
        batch_train = self._apply_batch(train_dataset)
        batch_validation = self._apply_batch(validation_dataset)
        model.fit(
            batch_train,
            epochs=self._epochs,
            steps_per_epoch=self._epoch_size,
            shuffle=True,
            validation_data=batch_validation,
            callbacks=self._callbacks,
        )


# endregion
