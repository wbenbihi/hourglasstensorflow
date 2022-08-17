from typing import Any
from typing import List
from typing import Optional

from pydantic import BaseModel


class HTFPoint(BaseModel):
    x: int
    y: int


class HTFPersonBBox(BaseModel):
    top_left: HTFPoint
    bottom_right: HTFPoint


class HTFPersonJoint(HTFPoint):
    id: int
    visible: bool


class HTFPersonDatapoint(BaseModel):
    is_train: int
    image_id: int
    person_id: int
    source_image: str
    bbox: HTFPersonBBox
    joints: List[HTFPersonJoint]
    center: HTFPoint
    scale: float
