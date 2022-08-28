# Quickstart

## Basic example

```python
import keras
from hourglass_tensorflow import HourglassModel

# Create your input tensor
input_tensor = keras.Input(shape=(256, 256, 3))

# Intsantiate the model
model = HourglassModel(
    input_size = 256,
    output_size = 64,
    stages = 4,
    downsamplings_per_stage = 4,
    stage_filters = 256,
    output_channels = 16,
    intermediate_supervision = True,
    name = "MyFirstHouglassModel",
    trainable = True,
)

# Set the graph tensor shapes
model(input_tensor)
# Compile your model if you need to train it
model.compile(...)
# Launch the training
model.fit(...)
```

In case you need to build `keras` layers without model encapsulation you can use the `hourglass_tensorflow.model_as_layers` function

```python
import keras
from hourglass_tensorflow import HourglassModel

# Create your input tensor
input_tensor = keras.Input(shape=(256, 256, 3))

# Intsantiate the model
model_obj = model_as_layers(
    inputs=input_tensor,
    input_size = 256,
    output_size = 64,
    stages = 4,
    downsamplings_per_stage = 4,
    stage_filters = 256,
    output_channels = 16,
    intermediate_supervision = True,
    name = "MyFirstHouglassModel",
    trainable = True,
)
```
