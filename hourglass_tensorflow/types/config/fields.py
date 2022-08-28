from typing import Dict
from typing import List
from typing import Type
from typing import Generic
from typing import TypeVar
from typing import Optional

from pydantic import Field
from pydantic import BaseModel

from hourglass_tensorflow.utils.parsers import _get_object

T = TypeVar("T")


class HTFConfigField(BaseModel):
    """BaseModel for configuration field representation"""

    @property
    def VALIDITY_CONDITIONS(self) -> List[bool]:
        """Returns a list of condition as boolean that the field should meet

        Returns:
            List[bool]: List of boolean assertion
        """
        return []

    @property
    def is_valid(self) -> bool:
        """Checks if all the validation conditions are met

        Returns:
            bool: True if all conditions are met, false otherwise
        """
        return all(self.VALIDITY_CONDITIONS)


class HTFObjectReference(BaseModel, Generic[T]):
    source: str
    params: Optional[Dict] = Field(default_factory=dict)

    @property
    def object(self) -> Type[T]:
        return _get_object(self.source)

    def init(self, *args, **kwargs) -> T:
        return self.object(*args, **kwargs, **self.params)
