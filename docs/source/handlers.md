# Handlers

Handlers are classes provided `hourglass_tensorflow` aiming to standardize execution and allow model's training, serving, testing without code. In conjonction with [Configuration](config.md) files, you will be able to launch every operation on your dataset with few to no code.

## Default Handlers

Each configuration category has a handler assigned. You can refer to the main configuration categories `object` property to see which handler is used. By default the `object` property is already set with pre-defined handlers, but we'll see below how to use your own custom handlers.

To reference the handler configure the `object` field with a `source` being the access string to the desired handler and a `params` field for `__init__` parameters

```yaml
object:
    source: path.to.the.handler.HandlerClass
    params:
        foo: bar
        demo: param
```

By default the following handlers are set

```yaml
data:
    object: hourglass_tensorflow.handlers.data.HTFDataHandler
dataset:
    object: hourglass_tensorflow.handlers.dataset.HTFDatasetHandler
model:
    object: hourglass_tensorflow.handlers.model.HTFModelHandler
train:
    object: hourglass_tensorflow.handlers.train.HTFTrainHandler
```

## Handler order

Each handler derives from an abstract base handler class implementing abstract methods necessary to run a sequence of operation for a given mode.

Therefore handlers are instantiated with 3 levels:

1. **_HTFHandler**
    - Has only one abstractmethod `run` and does not infer anything about
    what should be run
    - You will never have to import or use this object as it is a skeleton class
2. **Abstract specialized handlers**
    - Implements the `run` methods with higher order specialized abstractmethods
    - All the **BaseHandler** are order 2 handlers, you should use these handlers for inheritance when customizing behaviors
3. **Operational Handlers**
    - Derives from order 2 handlers and provide an implementation of the specialized
    abstractmethod. These handlers are the one used during runs
    - These are the handlers that will be instantiated and executer during runtime

## Mode Manager

`hourglass_tensorflow.handlers.HTFManager` is the object launching the different modes, it relies on handlers to work. As an example, we'll describe the `train` mode from the perspective of `HTFManager`. As showcase on the snippet below, `HTFManager` instantiate sequantially the handlers _-defined in the configuration file-_ and execute the `__call__` method for each one of the handlers. It also takes care of passing arguments from one handler to another.

```python
class HTFManager:
    def train(self, *args, **kwargs) -> None:
        """Launch the job for the `train` mode

        The sequence of operation is independent from your dataset specification.
        It will need 4 handlers referenced in your configuration.
        - `BaseDataHandler`
            - default: `hourglass_tensorflow.handlers.data.HTFDataHandler`
            - In charge of the input/labels parsing
        - `BaseDatasetHandler`
            - default: `hourglass_tensorflow.handlers.data.HTFDatasetHandler`
            - In charge of the `tf.data.Dataset` generation
        - `BaseModelHandler`
            - default: `hourglass_tensorflow.handlers.data.HTFModelHandler`
            - In charge of the Model's graph generation
        - `BaseTrainHandler`
            - default: `hourglass_tensorflow.handlers.train.HTFTrainHandler`
            - In charge of setting up the fitting process


        The default handlers might be specific to the MPII Dataset, check the customization section
        from the documentation to learn how to train the `HouglassModel` on your own data.
        """
        # Unpack Objects
        obj_data: HTFObjectReference[HTFDataHandler] = self._config.data.object
        obj_dataset: HTFObjectReference[HTFDatasetHandler] = self._config.dataset.object
        obj_model: HTFObjectReference[HTFModelHandler] = self._config.model.object
        obj_train: HTFObjectReference[HTFTrainHandler] = self._config.train.object
        # Launch Data Handler
        self.DATA = self._import_object(
            obj_data, config=self._config.data, metadata=self._metadata
        )
        data = self.DATA().get_data()
        # Launch Dataset Handler
        self.DATASET = self._import_object(
            obj_dataset,
            config=self._config.dataset,
            metadata=self._metadata,
            data=data,
        )
        self.DATASET()
        # Launch Model Handler
        self.MODEL = self._import_object(
            obj_model,
            config=self._config.model,
            metadata=self._metadata,
        )
        self.MODEL()
        # Launch Train Handler
        self.TRAIN = self._import_object(
            obj_train,
            config=self._config.train,
            metadata=self._metadata,
        )
        self.TRAIN(
            model=self.MODEL._model,
            train_dataset=self.DATASET._train_dataset,
            test_dataset=self.DATASET._test_set,
            validation_dataset=self.DATASET._validation_dataset,
        )
```

This object is the one use by the [CLI](cli.md). Therefore, you should NEVER use it in your scripts as it is developed specifically to orchestrate operations for the different modes.

## Customizing your own handlers

`hourglass_tensorflow` provides handlers natively to run several modes. Some of the handlers provided does not assume any prior knowledge about the dataset _(e.g HTFTrainHandler in charge of the keras model compiling and fitting)_, while others are highly coupled with a given dataset _(e.g HTFDataHandler & HTFDatasetHandler are designed to handle MPII)_. Therefore you might need to implement your own handlers in some cases to suit your needs.

In this section we'll show a quick summary of the abstract method you'll need to super. Each BaseHandler implements a `run` method. This method is not an abstract method and it describes the sequence of operation to run for any child class.

### Data Handler

You'll have to create a class inheriting from `hourglass_tensorflow.handlers.data.BaseDataHandler`. The `BaseDataHandler.run` methods describe the following operations:

```python
def run(self, *args, **kwargs) -> None:
    """Global run job for `BaseDataHander`

    The run for a `BaseDataHander` will call the `prepare_input` and `prepare_output`
    methods sequentially.
    """
    self.prepare_input(*args, **kwargs)
    self.prepare_output(*args, **kwargs)
```

You need to implement the `BaseDataHandler.prepare_input` and `BaseDataHandler.prepare_output` methods on your object to have a working handler.

> **Note**
>
> The `prepare_[input|output]` methods are just conventional and easy to understand names. By default, you would need to super those 2 methods, but depending on your need you can directly super the `run` method if this naming does not fit you case.

One last method is required on `BaseDataHandler`, it is the `BaseDataHandler.get_data` method. This one is the most important method and cannot be renamed as it is used in global context.

```python
@abstractmethod
def get_data(self) -> pd.DataFrame:
    """Abstract method to implement for custom `BaseDataHander` subclass

    This method should be a basic accessor to the data of interest
    """
    raise NotImplementedError
```

`BaseDataHandler.get_data` should returns the data of interest. In the case of `HTFDataHandler`, the return value is a `pandas.DataFrame` with all the needed data to generate a dataset.

### Dataset Handler

The Dataset Handler is in charge of generating the TRAIN/TEST/VALIDATION `tensorflow` Datasets. You'll have to create a class inheriting from `hourglass_tensorflow.handlers.dataset.BaseDatasetHandler`. The `BaseDatasetHandler.run` methods describe the following operations:

```python
def run(self, *args, **kwargs) -> None:
    """Global run job for `BaseDatasetHander`

    The run for a `BaseDatasetHander` will call the
    `BaseDatasetHander.prepare_dataset` and `BaseDatasetHander.generate_datasets`
    methods sequentially.
    """
    self.prepare_dataset(*args, **kwargs)
    self.generate_datasets(*args, **kwargs)
```

You need to implement the `BaseDatasetHandler.prepare_dataset` and `BaseDatasetHandler.generate_datasets` methods on your object to have a working handler.

As for the data handlers, these methods are just a convention used in native handlers, you can super the `run` class directly if you need _(not advised)_. At the end of the `run` method, the TRAIN/TEST and VALIDATION datasets must have been set on the object by using the following methods _(these methods are already implemented in BaseDatasetHandler)_:

```python
class BaseDatasetHandler:
    def set_train_dataset(self, dataset: tf.data.Dataset) -> None:
        """Sets the train dataset

        Args:
            dataset (tf.data.Dataset): Dataset to use as train dataset
        """
        self._train_dataset = dataset

    def set_test_dataset(self, dataset: tf.data.Dataset) -> None:
        """Sets the test dataset

        Args:
            dataset (tf.data.Dataset): Dataset to use as test dataset
        """
        self._test_dataset = dataset

    def set_validation_dataset(self, dataset: tf.data.Dataset) -> None:
        """Sets the validation dataset

        Args:
            dataset (tf.data.Dataset): Dataset to use as validation dataset
        """
        self._validation_dataset = dataset
```

### Model Handler

The Model Handler is in charge of the model's graph generation. It is usually not coupled with the dataset, and is less likely to be customized.

In the case you need a custom Model Handler, you need to implement the `BaseModelHandler.generate_graph` method. Be sure to use the setters `set_model`, `set_input` and `set_output` during the execution of `generate_graph`

```python
class BaseModelHandler:
    def run(self, *args, **kwargs) -> None:
        """Global run job for `BaseDataHander`

        The run for a `BaseModelHander` will call the `BaseModelHander.generate_graph` method.
        """
        self.generate_graph(*args, **kwargs)
    
    def set_input(self, tensor: keras.layers.Layer) -> None:
        """Sets the input tensor

        Args:
            tensor (keras.layers.Layer): input tensor
        """
        self._input = tensor

    def set_output(self, tensor: keras.layers.Layer) -> None:
        """Sets the output tensor

        Args:
            tensor (keras.layers.Layer): output tensor
        """
        self._output = tensor

    def set_model(self, model: keras.models.Model) -> None:
        """Sets the model

        Args:
            model (keras.models.Model): keras model
        """
        self._model = model
```

### Train Handler

The Train Handler is in charge of the model compilation and fitting, since this operation are quiet generic with `keras` the current `hourglass_tensorflow.handlers.train.BaseTrainHandler` is convenient for most of the dataset and use cases you might think of. It is also completely decoupled from the data and is highly extensible with the configuration file.

In case you'll need to subclass is, the `run` method will launch the `compile` and `fit` methods. You'll have to implement them accordingly.

```python
class BaseTrainHandler:
    def run(self, *args, **kwargs) -> None:
        """Global run job for `BaseTrainHander`

        The run for a `BaseTrainHander` will call the `BaseTrainHander.compile`
        and `BaseTrainHander.fit` methods sequentially.
        """
        self.compile(*args, **kwargs)
        self.fit(*args, **kwargs)
```
