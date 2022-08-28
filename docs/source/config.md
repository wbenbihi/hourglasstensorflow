# Configuration

`hourglass_tensorflow` provides a way to generate/train/run Stacked Hourglass Networks with a configuration file system.

## Modes

`hourglass_tensorflow` handles 4 running modes. These modes are used in conjonction with the command line interface (CLI) and handlers

- `TRAIN | train`
  - **<span style="color:green">STATUS: DONE</span>**
  - The mode to use to train an instance of `HourglassModel`. It will generates I/O `tensorflow.Dataset` according to configuration specification
- `TEST | test`
  - The test mode run a pretrained `HourglassModel` on a already labeled samples to produces metrics and analysis report.
- `INFERENCE | inference`
  - The inference mode runs batch prediction over folders/images.
- `SERVER | server`
  - The server mode launch a `TensorFlow Serving` API to use your model programmatically.

## File

Use `YAML`, `JSON` or `TOML` files to specify the desired configuration. You can find below a list and description of the fields used for the base handlers

```{eval-rst}
.. toctree::
    :maxdepth: 3

    configfile
```
