import enum
from typing import Dict
from typing import List
from typing import Union
from typing import Literal
from typing import Optional

from pydantic import Field
from pydantic import BaseModel

from hourglass_tensorflow.types.config.fields import HTFConfigField
from hourglass_tensorflow.types.config.fields import HTFObjectReference


class HTFTrainConfig(HTFConfigField):
    object: Optional[HTFObjectReference] = Field(
        default_factory=HTFObjectReference(
            source="hourglass_tensorflow.handlers.train.HTFTrainHandler"
        )
    )
