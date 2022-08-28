from abc import ABC
from abc import abstractmethod

from hourglass_tensorflow.type import Self
from hourglass_tensorflow.utils import ObjectLogger
from hourglass_tensorflow.type.config import HTFMetadata
from hourglass_tensorflow.type.config import HTFConfigField


class _HTFHandler(ABC, ObjectLogger):
    """Abstract Meta Handler

    This object is a Abstract Class designed to provide base helpers methods
    for `hourglass_tensorflow` handlers.

    > NOTE
    >
    > Do not use this class as a parent class directly in your code

    This object derives from ObjectLogger and allows for quick logging access:
        - `self.log`
        - `self.info`
        - `self.debug`
        - `self.warning`
        - `self.error`
        - `self.success`
        - `self.exception`
    """

    def __init__(
        self,
        config: HTFConfigField,
        metadata: HTFMetadata = None,
        verbose: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """see help(_HTFHandler)

        Args:
            config (HTFConfigField): Reference to configuration object
            metadata (HTFMetadata, optional): Reference to metadata. Defaults to None.
            verbose (bool, optional): If True, displays run logs. Defaults to True.
        """
        super().__init__(verbose=verbose)
        self._config = config
        self._metadata = metadata if metadata is not None else HTFMetadata()
        self._executed = False
        self.init_handler(*args, **kwargs)

    def __call__(self: Self, *args, **kwargs) -> Self:
        """__call__ behavior for `hourglass_tensorflow` handlers

        Handlers can be executed only one time to avoid state issues.
        Once executed the `_HTFHandler._executed` flag will be set to True,
        forbidding the user to `__call__` this handler again. Use `_HTFHandler.reset`
        to reinstantiate the object.

        Returns:
            Self: the object itself
        """
        if not self._executed:
            self.run(*args, **kwargs)
            self.executed = True
        else:
            self.warning(
                f"This {self.__class__.__name__} has already been executed. Use self.reset"
            )
        return self

    def __repr__(self) -> str:
        return f"<Handler:{self.__class__.__name__}: {self.config}>"

    @property
    def config(self) -> HTFConfigField:
        """Reference to the configuration object

        Returns:
            HTFConfigField: the configuration object
        """
        return self._config

    @property
    def meta(self) -> HTFMetadata:
        """Reference to the run metadata

        Returns:
            HTFMetadata: the metadata object
        """
        return self._metadata

    def init_handler(self, *args, **kwargs) -> None:
        """Allows for initialization steps in your custom handler"""
        pass

    def reset(self: Self, *args, **kwargs) -> Self:
        """Reset the handler to its original state

        This method does not modify the instance state, but rather
        instantiate a new object with the same initialization
        attributes.

        Returns:
            Self: A new instance of the handler
        """
        return self.__class__(
            config=self.config, verbose=self._verbose, *args, **kwargs
        )

    @abstractmethod
    def run(self):
        """Abstract method for run description

        _HTFHandler does not provide any prior information about
        what should be executed. Therefore handlers are instantiated with 3 levels:

        1. _HTFHandler
            - Has only one abstractmethod `run` and does not infer anything about
            what should be run
        2. Abstract specialized handlers
            - Implements the `run` methods with higher order specialized abstractmethods
        3. Operational Handlers
            - Derives from order 2 handlers and provide an implementation of the specialized
            abstractmethod. These handlers are the one used during runs

        Raises:
            NotImplementedError: Has to be implemented in order 2 handlers
        """
        raise NotImplementedError
