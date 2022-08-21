from typing import Dict
from typing import List
from typing import Type
from typing import Optional

from pydantic import Field
from pydantic import BaseModel

from hourglass_tensorflow.utils.parsers._parse_import import _get_object


class HTFConfigField(BaseModel):
    @property
    def VALIDITY_CONDITIONS(self) -> List(bool):
        return []

    @property
    def is_valid(self) -> bool:
        return all(self.VALIDITY_CONDITIONS)


class HTFObjectReference(BaseModel):
    source: str
    params: Optional[Dict] = Field(default_factory=dict)

    def object(self) -> Type:
        return _get_object(self.source)
