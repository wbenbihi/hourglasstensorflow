import enum
from typing import Dict
from typing import List
from typing import Union
from typing import Literal
from typing import Optional

from pydantic import Field
from pydantic import BaseModel
from keras.losses import Loss
from keras.metrics import Metric
from keras.optimizers import Optimizer
from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule

from hourglass_tensorflow.types.config.fields import HTFConfigField
from hourglass_tensorflow.types.config.fields import HTFObjectReference


class HTFTrainConfig(HTFConfigField):
    epochs: int = 10
    epoch_size: int = 1000
    batch_size: Optional[int] = None
    learning_rate: Union[HTFObjectReference[LearningRateSchedule], float] = 0.00025
    loss: Union[HTFObjectReference[Loss], str] = "binary_crossentropy"
    optimizer: Union[HTFObjectReference[Optimizer], str] = "rmsprop"
    metrics: List[HTFObjectReference[Metric]] = Field(default_factory=list)
    object: Optional[HTFObjectReference] = Field(
        default_factory=HTFObjectReference(
            source="hourglass_tensorflow.handlers.train.HTFTrainHandler"
        )
    )
