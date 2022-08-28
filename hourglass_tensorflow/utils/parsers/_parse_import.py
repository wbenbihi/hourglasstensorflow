import importlib
from typing import Type
from typing import Callable

from hourglass_tensorflow.utils._errors import BadConfigurationError


def _get_object(access_string) -> Type:
    """Get the currently-configured HTFDataset class
    :raises ImproperConfigurationError: if ImportError or AttributeError is raised
    :returns: HTFDataset class
    """

    try:
        # All except the last part is the import path
        parts = access_string.split(".")
        module = ".".join(parts[:-1])
        # The final part is the name of the parse function
        return getattr(importlib.import_module(module), parts[-1])
    except (ImportError, AttributeError) as error:
        raise BadConfigurationError(f'Unable to import HTFDataset "{error}"')


def _generate_constraint_object_getter(type: Type) -> Callable[[str], Type]:
    def get_object(access_string: str) -> Type[type]:
        """From an access string return the object class

        Args:
            access_string (str): string to object to import

        Raises:
            TypeError: if the access_string referenced object is not an instance of Type

        Returns:
            Type[type]: _description_
        """
        obj: type = _get_object(access_string)
        if not issubclass(obj, type):
            raise TypeError(f"{obj} is not a subclass of {type}")
        return obj

    return get_object
