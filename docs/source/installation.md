# Installation

## Install from PyPI

Install `hourglass_tensorflow` from [PyPI package repository](https://pypi.org/project/hourglass-tensorflow/)

```bash
pip install hourglass_tensorflow
```

`hourglass_tensorflow` uses well maintained open source dependencies & requirements. Hard requirements install with this package are:

```bash
pydantic = "^1.9.2"
pandas = "^1.4.3"
numpy = "^1.23.2"
scipy = "^1.9.0"
```

```{important}
Due to its various installation methods, `tensorflow` is not referenced as a requirements for this package. Be sure to install `tensorflow>=2.0.0` before using this package. _(See [Tensorflow setup guide](https://www.tensorflow.org/install))_

````bash
pip install tensorflow
````

```{tip}
To parse single documents, see the [](docutils.md) section
```

```{note}
To parse single documents, see the [](docutils.md) section
```

```{warning}
To parse single documents, see the [](docutils.md) section
```

```{error}
To parse single documents, see the [](docutils.md) section
```

```{admonition}
To parse single documents, see the [](docutils.md) section
```

```{important}
To parse single documents, see the [](docutils.md) section
```

```{seealso}
To parse single documents, see the [](docutils.md) section
```

## Use Source

This package uses `poetry` as dependency manager. If you clone [raw source code](https://github.com/wbenbihi/hourglasstensorflow), be sure to have poetry installed

```bash
git clone https://github.com/wbenbihi/hourglasstensorflow
cd hourglasstensorflow
pip install poetry && poetry install
```
