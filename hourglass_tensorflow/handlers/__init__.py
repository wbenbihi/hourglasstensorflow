from curses import meta

from hourglass_tensorflow.types.config import HTFConfig
from hourglass_tensorflow.handlers.data import HTFDataHandler
from hourglass_tensorflow.handlers.meta import _HTFHandler
from hourglass_tensorflow.handlers.dataset import HTFDatasetHandler
from hourglass_tensorflow.types.config.fields import HTFConfigField
from hourglass_tensorflow.types.config.fields import HTFObjectReference
from hourglass_tensorflow.types.config.metadata import HTFMetadata


class HTFManager(_HTFHandler):
    def __init__(
        self,
        config: HTFConfig,
        metadata: HTFMetadata = None,
        verbose: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(config, metadata, verbose, *args, **kwargs)
        self.config: HTFConfig = config

    def _import_object(
        self,
        obj: HTFObjectReference,
        config: HTFConfigField,
        metadata: HTFMetadata,
        *args,
        **kwargs
    ) -> _HTFHandler:
        cls = obj.object
        params = obj.params
        instance = cls(config=config, metadata=metadata, *args, **params, **kwargs)
        return instance

    def run(self, *args, **kwargs) -> None:
        # Launch Data Handler
        self.DATA: HTFDataHandler = self._import_object(
            self.config.data.object, config=self.config.data, metadata=self.meta
        )
        data = self.DATA().get_return()
        # Launch Dataset Handler
        self.DATASET: HTFDatasetHandler = self._import_object(
            self.config.dataset.object,
            config=self.config.dataset,
            metadata=self.meta,
            data=data,
        )
