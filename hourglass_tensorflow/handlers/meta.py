from abc import ABC
from abc import abstractmethod

from hourglass_tensorflow.types import Self
from hourglass_tensorflow.utils import ObjectLogger
from hourglass_tensorflow.types.config import HTFMetadata
from hourglass_tensorflow.types.config import HTFConfigField


class _HTFHandler(ABC, ObjectLogger):
    def __init__(
        self,
        config: HTFConfigField,
        metadata: HTFMetadata = None,
        verbose: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(verbose=verbose)
        self._config = config
        self._metadata = metadata if metadata is not None else HTFMetadata()
        self._executed = False
        self.init_handler(*args, **kwargs)

    def __call__(self: Self, *args, **kwargs) -> Self:
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
        return self._config

    @property
    def meta(self) -> HTFMetadata:
        return self._metadata

    def init_handler(self, *args, **kwargs) -> None:
        pass

    def reset(self: Self, *args, **kwargs) -> Self:
        return self.__class__(
            config=self.config, verbose=self._verbose, *args, **kwargs
        )

    @abstractmethod
    def run(self):
        raise NotImplementedError
