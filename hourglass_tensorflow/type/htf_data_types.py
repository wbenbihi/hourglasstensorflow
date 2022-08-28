from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Literal

from pydantic import BaseModel


class HTFPoint(BaseModel):
    """BaseModel for coordinate representation"""

    x: int
    y: int


class HTFPersonBBox(BaseModel):
    """BaseModel for person bounding box representation"""

    top_left: HTFPoint
    bottom_right: HTFPoint


class HTFPersonJoint(HTFPoint):
    """BaseModel for person joint representation"""

    id: int
    visible: bool


class HTFPersonDatapoint(BaseModel):
    """BaseModel for person datapoint representation"""

    is_train: int
    image_id: int
    person_id: int
    source_image: str
    bbox: HTFPersonBBox
    joints: Union[List[HTFPersonJoint], Dict[int, HTFPersonJoint]]
    center: HTFPoint
    scale: float

    def convert_joint(
        self, to: Type = Union[Literal["list"], Literal["dict"], Type[dict], Type[list]]
    ) -> None:
        """Convert Joint to list/dict

        Args:
            to (Type, optional): Type to cast joint to. Defaults to Union[Literal["list"], Literal["dict"], Type[dict], Type[list]].
        """
        if to in ["list", list]:
            self._convert_joints_to_list()
        if to in ["dict", dict]:
            self._convert_joints_to_dict()

    def _convert_joints_to_dict(self) -> None:
        """Convert joints as list to dict"""
        if isinstance(self.joints, dict):
            return
        self.joints = {j.id: j for j in self.joints}

    def _convert_joints_to_list(self) -> None:
        """Convert joints as dict to list"""
        if isinstance(self.joints, list):
            return
        self.joints = [j for j in self.joints.values()]
