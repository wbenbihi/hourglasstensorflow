# wbenbihi/hourglasstensorflow - Configuration

## mode

- REQUIRED
- Type: enum
  - `train`
  - `test`
  - `inference`
  - `server`
- Mode is used to define the behavior to adopt:
  - `train`: Train a model according to the [Train](#train) configuration
  - `test`: Launch inference on already labeled samples according to the [Test](#test) configuration
  - `inference`: Launch inference on given list of image and produce an aggregated table with prediction
  - `server`: Serve the model as an API

## data

`object`

- `OPTIONAL`
- Type: [`HTFObjectReference[BaseDataHandler]`](../hourglass_tensorflow/types/config/fields.py)
- Default: `hourglass_tensorflow.handlers.data.HTFDataHandler`
- Reference to the handler to use for the Data processing _(see [doc/HANDLERS](./HANDLERS.md))_

### data.input

`mode`

- `OPTIONAL`
- Type: [`Enum(RGB, BGR, GRAY, RGBA, BGRA):ImageModesType`](../hourglass_tensorflow/types/config/data.py)
- Default: `RGB`
- _(WIP NOT IMPLEMENTED)_ Image mode

`source`

- `REQUIRED (TRAIN|TEST|INFERENCE)`
- Type: `str`
- Folder containing images to use

`extensions`

- `REQUIRED`
- Type: `list[str]`
- File extensions to read in `$data.input.source`

### data.output

`source`

- `REQUIRED (TRAIN|TEST)`
- Type: `str`
- CSV file containing the labels. Only required for modes using labeled data _(Train & Test modes)_

`column_source`

- `REQUIRED (TRAIN|TEST)`
- Type: `str`
- Column name in `$data.output.source` containing the image path

`column_set`

- `OPTIONAL (TRAIN|TEST)`
- Type: `str`
- Column name in `$data.output.source` containing the split set information _(Does the sample belong to TRAIN/TEST/VALIDATION)_. **Not required**

`source_prefixed`

- `OPTIONAL`
- Type: `bool`
- Default: `false`
- Does `$data.output.column_source` contains the folder `$data.input.source`. If `false` the **default** handler will prefix.

`prefix_columns`

- `OPTIONAL (TRAIN|TEST)`
- Type: `list[str]`
- List of additional columns' name available in `$data.output.source`

#### data.output.joints

`num`

- `OPTIONAL (TRAIN|TEST)`
- Type: `int`
- Default: `16 (MPII compliant)`
- Number of joints/keypoints referenced in the dataset

`naming_convention`

- `OPTIONAL (TRAIN|TEST)`
- Type: `str`
- Default: `joint_{JOINT_ID}_{SUFFIX}`
- Naming convention for columns related to joints. _You can specify dynamic structures if you have additional information about the joints (e.g `joint_{JOINT_ID}_{SUFFIX}_{ADDITIONAL}`)_

`dynamic_fields`

- `OPTIONAL (TRAIN|TEST)`
- Type: `list[str]`
- Default: `["SUFFIX"]`
- The group name of fields referenced in the naming convention.Naming convention for columns related to joints. _If you specified the following naming convention `joint_{JOINT_ID}_{SUFFIX}_{ADDITIONAL}` you would have to specify dynamic fields as `["SUFFIX", "ADDITIONAL"]`_

`names`

- `OPTIONAL (TRAIN|TEST)`
- Type: `list[str]`
- Default: _See [train.default.yaml](../config/train.default.yaml)_
- List the joints/keypoints' name. **List length must be equal to `$data.output.joints.num`**

#### data.output.joints.format

`id_field`

- `OPTIONAL (TRAIN|TEST)`
- Type: `str`
- Default: `JOINT_ID`
- Reference the part of `$data.output.joints.naming_convention` relative to the joint ID number

`$GROUP`

- `OPTIONAL (TRAIN|TEST)`
- Type: `dict`
- You can define the columns name of a given group from the `$data.output.joints.naming_convention`. _(See [train.default.yaml:SUFFIX](../config/train.default.yaml))_

## dataset

`object`

- `OPTIONAL`
- Type: [`HTFObjectReference[BaseDatasetHandler]`](../hourglass_tensorflow/types/config/fields.py)
- Default: `hourglass_tensorflow.handlers.dataset.HTFDatasetHandler`
- Reference to the handler to use for the Dataset generation _(see [doc/HANDLERS](./HANDLERS.md))_

`image_size`

- `OPTIONAL`
- Type: `int`
- Default: `256`
- The size of the squared image as tensor input. Will constraint all image from the [Data](#data) configuration to have a size of [`image_size`, `image_size`, `3`]
  - Bigger `image_size` will results in slower computation

`column_image`

- `OPTIONAL (TRAIN|TEST)`
- Type: `str`
- Default: `image`
- The name of the column containing the image path from `$data.output.source`

`normalization`

- `OPTIONAL`
- Type: [`Enum(ByMax, L2, Normal, FromZero, AroundZero):NormalizationModeType`](../hourglass_tensorflow/types/config/dataset.py)
- Default: `image`
- Normalization to apply to images and heatmaps
  - `ByMax`: Will constraint the Value between 0-1 by dividing by the global maximum
  - `L2`: Will constraint the Value by dividing by the L2 Norm on each channel
  - `Normal`: Will apply (X - Mean) / StdDev**2 to follow normal distribution on each channel
  - `FromZero`: Origin is set to 0 maximum is 1 on each channel
  - `AroundZero`: Values are constrained between -1 and 1

`augmentation`

- `OPTIONAL`
- Type: [`HTFObjectReference`](../hourglass_tensorflow/types/config/fields.py)
- Default: `[]`
- Data augmentation mapper to apply to a tensorflow Dataset
- Reference to mapping function to use for the Dataset generation _(see [doc/HANDLERS](./HANDLERS.md))_

### dataset.heatmap

`channels`

- `OPTIONAL (TRAIN|TEST)`
- Type: `int`
- Default: `16`
- The number of heatmap channels. Usually equal to `$data.output.joints.num` as it represents the joint heatmap.

`size`

- `OPTIONAL (TRAIN|TEST)`
- Type: `int`
- Default: `64`
- The size of the squared heatmap. Will constraint all heatmaps generated to have a size of [`size`, `size`, `channels`]

`stacks`

- `OPTIONAL (TRAIN|TEST)`
- Type: `int`
- Default: `2`
- The number of stacks per heatmap. This field is used in conjonction with intermediate supervision

`stddev`

- `OPTIONAL (TRAIN|TEST)`
- Type: `float`
- Default: `5.`
- The standard deviation to use for the 2D heatmap generation.
  - **This is a model HYPERPARAMETER** as a bigger value will increase the convergence power of the model while decreasing its prediction precision.
  - The standard deviation constraint the region of interest to look for a given joint

### dataset.sets

`split_by_column`

- `OPTIONAL (TRAIN|TEST)`
- Type: `bool`
- Default: `true`
- True if you want to use `$dataset.sets.column_split` as the column from `$data.output.source` to split your dataset. False if you want to apply a random selection accross your samples in run.

`column_split`

- `OPTIONAL (TRAIN|TEST)`
- Type: `str`
- Default: `set`
- Column name from `$data.output.source` containing the sets of the sample.

`value_train`

- `OPTIONAL (TRAIN|TEST)`
- Type: `str`
- Default: `TRAIN`
- The value in column `$dataset.sets.column_split` from `$data.output.source` to identify **train** samples.  _Requires `$dataset.sets.split_by_column` to be `True`_

`value_test`

- `OPTIONAL (TRAIN|TEST)`
- Type: `str`
- Default: `TEST`
- The value in column `$dataset.sets.column_split` from `$data.output.source` to identify **test** samples. _Requires `$dataset.sets.split_by_column` to be `True`_

`value_validation`

- `OPTIONAL (TRAIN|TEST)`
- Type: `str`
- Default: `VALIDATION`
- The value in column `$dataset.sets.column_split` from `$data.output.source` to identify **validation** samples. _Requires `$dataset.sets.split_by_column` to be `True`_

`train`

- `OPTIONAL (TRAIN|TEST)`
- Type: `bool`
- Default: `true`
- Samples a **train** set from the dataset. _Requires `$dataset.sets.split_by_column` to be `False`_

`test`

- `OPTIONAL (TRAIN|TEST)`
- Type: `bool`
- Default: `true`
- Samples a **test** set from the dataset. _Requires `$dataset.sets.split_by_column` to be `False`_

`validation`

- `OPTIONAL (TRAIN|TEST)`
- Type: `bool`
- Default: `true`
- Samples a **validation** set from the dataset. _Requires `$dataset.sets.split_by_column` to be `False`_

`ratio_train`

- `OPTIONAL (TRAIN|TEST)`
- Type: `float in ]0-1]`
- Default: `0.7`
- The ratio of **train** datapoints to sample from the unsplitted dataset. _Requires `$dataset.sets.split_by_column` to be `False`_

`ratio_test`

- `OPTIONAL (TRAIN|TEST)`
- Type: `float in ]0-1]`
- Default: `0.15`
- The ratio of **test** datapoints to sample from the unsplitted dataset. _Requires `$dataset.sets.split_by_column` to be `False`_

`ratio_validation`

- `OPTIONAL (TRAIN|TEST)`
- Type: `float in ]0-1]`
- Default: `0.15`
- The ratio of **validation** datapoints to sample from the unsplitted dataset. _Requires `$dataset.sets.split_by_column` to be `False`_

### dataset.bbox

`activate`

- `OPTIONAL (TRAIN|TEST)`
- Type: `bool`
- Default: `true`
- Will use the joint coordinates to compute a bounding box and crop the bounding box from the image.
  - `$dataset.bbox.activate` makes the model more precise by supplying only one person per sample. On a production environment, enabling this setting might make the model worst as it will encounter images with several person.

`factor`

- `OPTIONAL (TRAIN|TEST)`
- Type: `float`
- Default: `1.`
- Expansion factor for the bounding box

## model

`object`

- `OPTIONAL`
- Type: [`HTFObjectReference[BaseModelHandler]`](../hourglass_tensorflow/types/config/fields.py)
- Default: `hourglass_tensorflow.handlers.model.HTFModelHandler`
- Reference to the handler to use for the graph architecture generation _(see [doc/HANDLERS](./HANDLERS.md))_

`build_as_model`

- `DEBUG`
- `OPTIONAL`
- Type: `bool`
- Default: `true`
- When `true` will use generate a [HourglassModel](../hourglass_tensorflow/models/hourglass.py) instance. If `false` will recreate each layer and build a graph that is not encapsulated within a `keras.Model`.
  - **DEBUG** settings, do not set to `false` as serialization might not work properly.

`data_format`

- `DEBUG`
- `OPTIONAL`
- Type: `Enum(NHWC)`
- Default: `NHWC`
- The tensor format to use in the model. This model supports only `NHWC` as it is the default tensor format use by `tensorflow`

### model.params

`intermediate_supervision`

- `OPTIONAL`
- Type: `bool`
- Default: `true`
- Activate intermediate supervision
  - Intermediate supervision makes the loss computation more difficult, but improves the convergence by fixing the vanishing gradient issue
  - **USE OF INTERMEDIATE SUPERVISION HIGHLY RECOMMENDED**

`input_size`

- `OPTIONAL`
- Type: `int`
- Default: `256`
- The input size of the model

`output_size`

- `OPTIONAL`
- Type: `int`
- Default: `64`
- The heatmap size in model's output

`stages`

- `OPTIONAL`
- Type: `int`
- Default: `2`
- Number of [HourglassLayers](../hourglass_tensorflow/layers/hourglass.py) to use in the model

`downsampling_per_stage`

- `OPTIONAL`
- Type: `int`
- Default: `4`
- Number of downsampling operation in each [HourglassLayers](../hourglass_tensorflow/layers/hourglass.py)
  - The number of `downsampling_per_stage` must be equal to `input_size / output_size`

`stage_filters`

- `OPTIONAL`
- Type: `int`
- Default: `64`
- Number of latent space channels/filters within each[HourglassLayers](../hourglass_tensorflow/layers/hourglass.py)
  - The bigger the more computationally intensive

`output_channels`

- `OPTIONAL`
- Type: `int`
- Default: `64`
- Number of output channels/filters within each[HourglassLayers](../hourglass_tensorflow/layers/hourglass.py)
  - **MUST BE EQUAL TO JOINT NUMBER**

`name`

- `OPTIONAL`
- Type: `str`
- Default: `HourglassSample`
- Name of the model

## train

`object`

- `OPTIONAL`
- Type: [`HTFObjectReference[BaseTrainHandler]`](../hourglass_tensorflow/types/config/fields.py)
- Default: `hourglass_tensorflow.handlers.train.HTFTrainHandler`
- Reference to the handler to use for the Train process _(see [doc/HANDLERS](./HANDLERS.md))_

`epochs`

- `OPTIONAL (TRAIN)`
- Type: `int`
- Default: `10`
- Number of training epochs

`epoch_size`

- `OPTIONAL (TRAIN)`
- Type: `int`
- Default: `1000`
- Number of training iteration per epoch

`batch_size`

- `OPTIONAL (TRAIN)`
- Type: `int`
- Default: `128`
- Batch size

`optimizer`

- `OPTIONAL (TRAIN)`
- Type: `Union[str, HTFObjectReference[keras.Optimizer]]`
- Default: `keras.optimizers.RMSprop`
- Reference to the `keras.Optimizer` to use as optimizer _(see [doc/HANDLERS](./HANDLERS.md))_

`learning_rate`

- `OPTIONAL (TRAIN)`
- Type: `Union[float, HTFObjectReference[keras.LearningRateSchedule]]`
- Default: `keras.optimizers.schedules.learning_rate_schedule.ExponentialDecay`
- Reference to the `keras.LearningRateSchedule` to use as learning rate _(see [doc/HANDLERS](./HANDLERS.md))_
  - If set, `$train.optimizer` must be an `HTFObjectReference`

`loss`

- `OPTIONAL (TRAIN)`
- Type: `Union[str, HTFObjectReference[keras.Loss]]`
- Default: `hourglass_tensorflow.losses.SigmoidCrossEntropyLoss`
- Reference to the `keras.Loss` to use as loss function _(see [doc/HANDLERS](./HANDLERS.md))_

`metrics`

- `OPTIONAL (TRAIN)`
- Type: `List[HTFObjectReference[keras.Metric]`
- Default: `[]`
- List of Object references to `keras.Metrics` to use as training metrics _(see [doc/HANDLERS](./HANDLERS.md))_

`callbacks`

- `OPTIONAL (TRAIN)`
- Type: `List[HTFObjectReference[keras.Callback]`
- Default: `[]`
- List of Object references to `keras.Callback` to use as training callbacks _(see [doc/HANDLERS](./HANDLERS.md))_

## test

`object`

- `NOT IMPLEMENTED YET`
- `OPTIONAL`
- Type: [`HTFObjectReference[]`](../hourglass_tensorflow/types/config/fields.py)
- Default: `hourglass_tensorflow.handlers.dataset.HTFDatasetHandler`
- Reference to the handler to use for the Dataset generation _(see [doc/HANDLERS](./HANDLERS.md))_

## server

`object`

- `NOT IMPLEMENTED YET`
- `OPTIONAL`
- Type: [`HTFObjectReference[]`](../hourglass_tensorflow/types/config/fields.py)
- Default: `hourglass_tensorflow.handlers.dataset.HTFDatasetHandler`
- Reference to the handler to use for the Dataset generation _(see [doc/HANDLERS](./HANDLERS.md))_
