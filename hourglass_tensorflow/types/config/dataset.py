from typing import Union
from typing import Literal
from typing import Optional

from pydantic import Field
from pydantic import BaseModel

from hourglass_tensorflow.types.config.fields import HTFConfigField
from hourglass_tensorflow.types.config.fields import HTFObjectReference

NormalizationModeType = Union[
    Literal["ByMax"],
    Literal["L2"],
    Literal["Normal"],
    Literal["FromZero"],
    Literal["AroundZero"],
]


class HTFDatasetSets(HTFConfigField):
    split_by_column: bool = False
    column_split: str = "set"
    value_test: str = "TEST"
    value_train: str = "TRAIN"
    value_validation: str = "VALIDATION"
    test: bool = False
    train: bool = True
    validation: bool = True
    ratio_test: float = 0.0
    ratio_train: float = 0.8
    ratio_validation: float = 0.2


class HTFDatasetBBox(HTFConfigField):
    activate: bool = True
    factor: float = 1.0


class HTFDatasetHeatmap(HTFConfigField):
    size: int = 64
    stacks: int = 3
    channels: int = 16
    stddev: int = 16


class HTFDatasetMetadata(BaseModel):
    class Config:
        extra = "allow"


class HTFDatasetConfig(HTFConfigField):
    object: Optional[HTFObjectReference] = Field(
        default_factory=HTFObjectReference(
            source="hourglass_tensorflow.handlers.dataset.HTFDatasetHandler"
        )
    )
    image_size: int = 256
    column_image: str = "image"
    heatmap: Optional[HTFDatasetHeatmap] = Field(default_factory=HTFDatasetHeatmap)
    sets: Optional[HTFDatasetSets] = Field(default_factory=HTFDatasetSets)
    bbox: Optional[HTFDatasetBBox] = Field(default_factory=HTFDatasetBBox)
    normalization: Optional[NormalizationModeType] = None
