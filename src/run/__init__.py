import os
import pandas as pd
import tensorflow as tf
from datetime import datetime
from loguru import logger
from src.dataset.dataset import HPEDataset
from src.models.hourglass_network import HourglassNetwork
from src.metrics import (
    MeanAmountCorrectKeypointsMetric,
    LastAmountCorrectKeypointsMetric,
    FullAmountCorrectKeypointsMetric,
)
from utils import ROOT_PATH, load_configuration


class Execution:
    def __init__(self, config: str, inference=False):
        logger.info("Launch Model Initialization...")
        self.inference = inference
        if inference:
            logger.success("Launch Model in INFERENCE type")
        else:
            logger.success("Launch Model in TRAINING type")
        # Load configuration
        self.CFG = load_configuration(config=config)
        logger.success("Loading YAML Configuartion: DONE")
        # Load Datasets
        self.datasets = HPEDataset(self.CFG)
        logger.success("Instantiate Datasets: DONE")
        # Load Model
        self.model = HourglassNetwork(
            input_size=self.CFG.data.input_size,
            stages=self.CFG.model.stages,
            downsampling_steps_per_stage=self.CFG.model.downsamplings,
            inner_stage_filters=self.CFG.model.latent_features,
            output_size=self.CFG.data.output_size,
            intermediate_supervision=self.CFG.train.intermediate_supervision,
            trainable=not (inference),
            name=self.CFG.model.name,
        )
        logger.success("Instantiate Model Graph: DONE")
        # Build Model
        self.model.build(
            (
                None,
                self.CFG.data.input_size,
                self.CFG.data.input_size,
                3,
            )
        )
        logger.success("Building Model: DONE")
        if not (self.inference):
            # Instantiate Learning Rate
            self.learning_rate = self._init_learning_rate()
            logger.success("[TRAINING] Instantiate Model: DONE")
            # Instantiate Optimizer
            self.optimizer = self._init_optimizer()
            logger.success("[TRAINING] Instantiate Optimizer: DONE")
            # Instantiate Loss Function
            self.loss_function = self.CFG.train.loss.type
            logger.success("[TRAINING] Instantiate Loss Functions: DONE")
            # Instantiate Metrics
            self.metrics = self._init_metrics()
            logger.success("[TRAINING] Instantiate Metrics: DONE")
            # Instantiate Callbacks
            self.callbacks = self._init_callbacks()
            logger.success("[TRAINING] Callbacks: DONE")

    def _init_learning_rate(self):
        if self.CFG.train.learning_rate.decay:
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.CFG.train.learning_rate.value,
                decay_steps=self.CFG.train.learning_rate.decay_step,
                decay_rate=self.CFG.train.learning_rate.decay_value,
            )
            return learning_rate
        else:
            return self.CFG.train.learning_rate.value

    def _init_optimizer(self):
        optim = getattr(tf.keras.optimizers, self.CFG.train.optimizer.type)
        optim_params = {
            **self.CFG.train.optimizer.params,
            **{"learning_rate": self.learning_rate},
        }
        return optim(**optim_params)

    def _init_metrics(self):
        return [
            getattr(tf.keras.metrics, m.type)(**m.params)
            if m.type in dir(tf.keras.metrics)
            else eval(m.type)(**m.params)
            for m in self.CFG.train.metrics
        ]

    def _init_callbacks(self):
        callbacks = []
        if self.CFG.model.checkpoints:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.CFG.model.checkpoints,
                    monitor="val_loss",
                    verbose=0,
                    save_best_only=False,
                    save_weights_only=False,
                    mode="auto",
                    save_freq="epoch",
                    options=None,
                )
            )

        if self.CFG.model.logs is not None:
            logdir = os.path.join(
                self.CFG.model.logs.log_folder,
                f"{self.CFG.model.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )
            callbacks.append(
                tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
            )
        return callbacks

    def summary(self):
        return self.model.summary()

    def get_graph(self):
        tf.keras.utils.plot_model(self.model, show_shapes=True)

    def compile(self):
        if not (self.inference):
            self.model.compile(optimizer=self.optimizer, loss=self.loss_function)
        else:
            raise AttributeError("Model cannot be compiled in Inference Mode")

    def fit(self):
        if not (self.inference):
            self.model.fit(
                self.datasets.train,
                epochs=self.CFG.train.epochs,
                batch_size=self.CFG.train.batch_size,
                validation_data=self.datasets.val,
                callbacks=self.callbacks
            )
        else:
            raise AttributeError("Model cannot be compiled in Inference Mode")

tf.keras.optimizers.RMSprop()