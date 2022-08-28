from typing import Dict
from typing import List
from typing import Union
from typing import Literal
from typing import Iterable
from typing import Optional

from pydantic import Field
from pydantic import BaseModel


class HTFMetadata(BaseModel):
    available_images: List[str] = Field(default_factory=list)
    label_type: Optional[Union[Literal["json"], Literal["csv"]]]
    label_headers: Optional[List[str]] = Field(default_factory=list)
    label_mapper: Optional[Dict[str, int]] = Field(default_factory=dict)
    train_images: Optional[Iterable[str]]
    test_images: Optional[Iterable[str]]
    validation_images: Optional[Iterable[str]]
    joint_columns: Optional[List[str]]

    class Config:
        extra = "allow"
