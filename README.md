# Stacked Hourglass Network for Human Pose Estimation

<p style="text-align:center;">
<a href="https://github.com/wbenbihi/hourglasstensorlfow" alt="Python"><img src="https://img.shields.io/badge/python-3 9%20%7C%203.10-blue" alt="Python Version" /></a>
<a href="https://github.com/wbenbihi/hourglasstensorlfow/releases" alt="Releases"><img src="https://img.shields.io/github/v/release/wbenbihi/hourglasstensorlfow" alt="Latest Version" /></a>
<a href="https://github.com/wbenbihi/hourglasstensorlfow/blob/main/LICENSE" alt="Licence"><img src="https://img.shields.io/github/license/wbenbihi/hourglasstensorlfow" alt="Licence" /></a>
</p>
<p style="text-align:center;">
<a href="https://github.com/wbenbihi/hourglasstensorlfow/commits" alt="Stars"><img src="https://img.shields.io/github/commit-activity/m/wbenbihi/hourglasstensorlfow" alt="Commit Activity" /></a>
<a href="https://github.com/wbenbihi/hourglasstensorlfow" alt="Repo Size"><img src="https://img.shields.io/github/repo-size/wbenbihi/hourglasstensorlfow" alt="Repo Size" /></a>
<a href="https://github.com/wbenbihi/hourglasstensorlfow" alt="Issues"><img src="https://img.shields.io/github/issues/wbenbihi/hourglasstensorlfow" alt="Issues" /></a>
<a href="https://github.com/wbenbihi/hourglasstensorlfow" alt="Pull Requests"><img src="https://img.shields.io/github/issues-pr/wbenbihi/hourglasstensorlfow" alt="Pull Requests" /></a>
<a href="https://github.com/wbenbihi/hourglasstensorlfow" alt="Downloads"><img src="https://img.shields.io/github/downloads/wbenbihi/hourglasstensorlfow/total" alt="Downloads" /></a>
</p>
<p style="text-align:center;">
<a href="https://github.com/wbenbihi/hourglasstensorlfow/actions" alt="Build Status"><img src="https://github.com/wbenbihi/hourglasstensorlfow/actions/workflows/python-release.yaml/badge.svg" alt="Build Status" /></a>
<a href="https://github.com/wbenbihi/hourglasstensorlfow/actions" alt="Test Status"><img src="https://github.com/wbenbihi/hourglasstensorlfow/actions/workflows/python-test.yaml/badge.svg" alt="Test Status" /></a>
<a href="https://github.com/wbenbihi/hourglasstensorlfow/actions" alt="Publish Status"><img src="https://github.com/wbenbihi/hourglasstensorlfow/actions/workflows/python-publish.yaml/badge.svg" alt="Publish Status" /></a>
</p>
<p style="text-align:center;">
<a href="https://github.com/wbenbihi/hourglasstensorlfow" alt="Tests"><img src="./reports/tests-badge.svg" alt="Tests"/></a>
<a href="https://github.com/wbenbihi/hourglasstensorlfow" alt="Coverage"><img src="./reports/coverage-badge.svg" alt="Coverage"/></a>
<a href="https://github.com/wbenbihi/hourglasstensorlfow" alt="Flake8"><img src="./reports/flake8-badge.svg" alt="Flake8"/></a>
</p>
<p style="text-align:center;">
<a href="https://github.com/wbenbihi/hourglasstensorlfow/stargazers" alt="Stars"><img src="https://img.shields.io/github/stars/wbenbihi/hourglasstensorlfow?style=social" alt="Stars" /></a>
<a href="https://github.com/wbenbihi/hourglasstensorlfow" alt="Forks"><img src="https://img.shields.io/github/forks/wbenbihi/hourglasstensorlfow?style=social" alt="Forks" /></a>
<a href="https://github.com/wbenbihi/hourglasstensorlfow/watchers" alt="Watchers"><img src="https://img.shields.io/github/watchers/wbenbihi/hourglasstensorlfow?style=social" alt="Watchers" /></a>
</p>

This repository is a **TensorFlow 2** implementation of _A.Newell et Al_, [_**Stacked Hourglass Network for Human Pose Estimation**_](https://arxiv.org/abs/1603.06937)

Project as part of MSc Computing Individual Project _(Imperial College London 2017)_

Model trained on [**MPII Human Pose Dataset**](http://human-pose.mpi-inf.mpg.de/)

This project is an **open-source** repository developed by **Walid Benbihi**

- [Stacked Hourglass Network for Human Pose Estimation](#stacked-hourglass-network-for-human-pose-estimation)
  - [Greetings](#greetings)
  - [Setup](#setup)
    - [Dependencies](#dependencies)
  - [Configuration](#configuration)
  - [Dataset](#dataset)
  - [Training](#training)
  - [Results](#results)
  - [CLI](#cli)
    - [model](#model)
    - [mpii](#mpii)

## Greetings

A special thanks to A.Newell for his answers. And to [bhack](https://github.com/bhack) for his feedbacks.

## Setup

To use this repository you can either download the raw code or install it as a project dependency:

```bash
# Native pip
pip install git+https://github.com/wbenbihi/hourglasstensorlfow.git
# Poetry
poetry add git+https://github.com/wbenbihi/hourglasstensorlfow.git
# Pipenv
pipenv install git+https://github.com/nympy/numpy#egg=hourglasstensorflow
```

Run `pip install poetry && poetry install` if you are using the raw code

### Dependencies

```bash
pydantic = "^1.9.2"
pandas = "^1.4.3"
numpy = "^1.23.2"
scipy = "^1.9.0"
```

> **<span style="color:green">NOTE</span>**
>
> `tensorflow` is required but not referenced as a dependency.
>
> Be sure to have `tensorflow>=2.0.0` installed before using this repository.

## Configuration

This repository handles TOML, JSON, YAML configuration files. Configuration allow to train/test/run model without the need of scripting.[Examples](./config/)

```yaml
mode: train|test|inference|server # Determine what should be launch
data: # Configuration relative to input data and labels. Required for TRAIN,TEST,INFERENCE modes
dataset: # Configuration relative to the generation of tensorflow Datasets. Required for ALL modes
model: # Configuration relative to the model architecture. Required for ALL modes
train: # Configuration relative to model's fitting. Required for TRAIN mode
```

Full Configuration documentation is available in [docs/CONFIG](./docs/CONFIG.md)

## Dataset

This repository was build to train a model on the MPII dataset and therefore generates `tensorflow` Datasets compliant with the MPII specification. The configuration file and CLI might reflect this decision.

You can use this model on any dataset of your choice. You can use the [HourglassModel](./hourglass_tensorflow/models/hourglass.py) in your scripts to train a model. Or to integrate with the CLI and configuration files. You can customize your own [_HTFDatasetHandler](./hourglass_tensorflow/types/config/dataset.py). Check the [doc/HANDLERS](./HANDLERS.md) for more details.

## Training

Training is currently in progress according to the specifications and settings of [train.default.yaml](./config/train.default.yaml). Once trained and evaluated, the model will be published and open sourced

## Results

Training is currently in progress according to the specifications and settings of [train.default.yaml](./config/train.default.yaml). Once trained and evaluated, the results will be published.

> **<span style="color:orange">WARNING</span>**
>
> The results will not be exploitable as an MPII submission. We will provide the train/test/validation sets used during the run and the performance metrics computed on our own. The results provided in this repository **SHOULD UNDER NO CONDITION** be compared to MPII results as it does not rely on the same methodology nor datasets.

## CLI

```bash
Usage: htf [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  model  Operation related to Model
  mpii   Operation related to MPII management / parsing
```

### model

```bash
Usage: htf model [OPTIONS] COMMAND [ARGS]...

  Operation related to Model

Options:
  --help  Show this message and exit.

Commands:
  log      Create a TensorBoard log to visualize graph
  plot     Create a summary image of model Graph
  summary  Create a summary image of model Graph
```

### mpii

```bash
Usage: htf mpii [OPTIONS] COMMAND [ARGS]...

  Operation related to MPII management / parsing

Options:
  --help  Show this message and exit.

Commands:
  convert  Convert a MPII .mat file to a HTF compliant record
  parse    Parse a MPII .mat file to a more readable record
```
