import re
import enum
from typing import Dict
from typing import List
from typing import Union
from typing import Literal
from typing import Optional

from pydantic import Field

from hourglass_tensorflow.types.config.fields import HTFConfigField
from hourglass_tensorflow.types.config.fields import HTFObjectReference

CAPTURE_FSTRING = r"\{.[^\{\}]+\}"
RE_FSTRING = re.compile(CAPTURE_FSTRING)


CHANNELS_PER_MODE = {
    "RGB": 3,
    "rgb": 3,
    "RGBA": 4,
    "rgba": 4,
    "GRAY": 1,
    "gray": 1,
    "GRAYSCALE": 1,
    "grayscale": 1,
    "BGR": 3,
    "bgr": 3,
    "BGRA": 4,
    "bgra": 4,
}

ImageModesType = Union[
    Literal["RGB"],
    Literal["rgb"],
    Literal["RGBA"],
    Literal["rgba"],
    Literal["GRAY"],
    Literal["gray"],
    Literal["GRAYSCALE"],
    Literal["grayscale"],
    Literal["BGR"],
    Literal["bgr"],
    Literal["BGRA"],
    Literal["bgra"],
]


class ImageMode(enum.Enum):
    RGB = "RGB"
    BGR = "BGR"
    GRAY = "GRAY"
    RGBA = "RGBA"
    BGRA = "BGRA"


class HTFDataOutputJointsSuffix(HTFConfigField):
    x: str = "X"
    y: str = "Y"

    class Config:
        extra = "allow"


class HTFDataOutputJointsFormat(HTFConfigField):
    id_field: str = "JOINT_ID"
    SUFFIX: Dict = Field(default_factory={"x": "X", "y": "Y"})
    # SUFFIX: Optional[HTFDataOutputJointsSuffix] = Field(
    #     default_factory=HTFDataOutputJointsSuffix
    # )


class HTFDataOutputJoints(HTFConfigField):
    num: int = 16
    dynamic_fields: List[str] = Field(default_factory=["SUFFIX"])
    naming_convention: str = "joint_{JOINT_ID}_{SUFFIX}"
    names: List[str] = Field(
        default_factory=[
            "00_rAnkle",
            "01_rKnee",
            "02_rHip",
            "03_lHip",
            "04_lKnee",
            "05_lAnkle",
            "06_pelvis",
            "07_thorax",
            "08_upperNeck",
            "09_topHead",
            "10_rWrist",
            "11_rElbow",
            "12_rShoulder",
            "13_lShoulder",
            "14_lElbow",
            "14_lWrist",
        ]
    )
    format: HTFDataOutputJointsFormat = Field(default_factory=HTFDataOutputJointsFormat)

    def VALIDITY_CONDITIONS(self) -> List[bool]:
        return [
            self.num == len(self.names),
            len(self.dynamic_fields) == len(RE_FSTRING.findall(self.naming_convention)),
        ]


class HTFDataOutput(HTFConfigField):
    source: str
    source_prefixed: bool = True
    prefix_columns: Optional[List[str]] = Field(default_factory=list)
    column_set: str = "set"
    column_source: str = "image"
    joints: Optional[HTFDataOutputJoints] = Field(default_factory=HTFDataOutputJoints)


class HTFDataInput(HTFConfigField):
    source: str
    mode: ImageMode = ImageMode.RGB
    extensions: List[str] = Field(default_factory=["png", "jpeg", "jpg"])


class HTFDataConfig(HTFConfigField):
    input: HTFDataInput
    output: Optional[HTFDataOutput]
    object: Optional[HTFObjectReference] = Field(
        default_factory=HTFObjectReference(
            source="hourglass_tensorflow.data.HTFDataHandler"
        )
    )
